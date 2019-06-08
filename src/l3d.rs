use crate::util::{DSend, DSendMut};
use rayon::prelude::*;

pub unsafe fn dgemm(
    _transa: bool,
    _transb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    const MC: usize = 256;
    const KC: usize = 128;
    const NB: usize = 1024;

    let mut packed_a = Vec::with_capacity(MC * KC);
    let mut packed_b = Vec::with_capacity(KC * NB);

    for j in (0..n).step_by(NB) {
        let jb = std::cmp::min(n - j, NB);
        let mut beta_scale = beta;
        for p in (0..k).step_by(KC) {
            let pb = std::cmp::min(k - p, KC);
            for i in (0..m).step_by(MC) {
                let ib = std::cmp::min(m - i, MC);
                inner_kernel(
                    ib,
                    jb,
                    pb,
                    alpha,
                    a.add(i + p * lda),
                    lda,
                    b.add(p + j * ldb),
                    ldb,
                    beta_scale,
                    c.add(i + j * ldc),
                    ldc,
                    packed_a.as_mut_ptr(),
                    packed_b.as_mut_ptr(),
                    i == 0,
                );
            }
            beta_scale = 1.0;
        }
    }

    unsafe fn inner_kernel(
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        b: *const f64,
        ldb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
        packed_a: *mut f64,
        packed_b: *mut f64,
        first_time: bool,
    ) {
        let n_left = n % 4;
        let n_main = n - n_left;
        let m_left = m % 8;
        let m_main = m - m_left;

        let a = DSend(a);
        let b = DSend(b);
        let c = DSendMut(c);
        let packed_a = DSendMut(packed_a);
        let packed_b = DSendMut(packed_b);

        if first_time {
            (0..n_main)
                .step_by(4)
                .collect::<Vec<_>>()
                .par_iter()
                .for_each(move |&j| {
                    pack_b(k, b.0.add(j * ldb), ldb, packed_b.0.add(j * k));
                });
        }

        (0..m_main)
            .step_by(8)
            .collect::<Vec<_>>()
            .par_iter()
            .for_each(move |&i| {
                crate::d_pack_a(k, alpha, a.0.add(i), lda, packed_a.0.add(i * k));
            });

        (0..n_main)
            .step_by(4)
            .collect::<Vec<_>>()
            .par_iter()
            .for_each(move |&j| {
                for i in (0..m_main).step_by(8) {
                    crate::dgemm_8x4_packed(
                        k,
                        packed_a.0.add(i * k),
                        packed_b.0.add(j * k),
                        beta,
                        c.0.add(i + j * ldc),
                        ldc,
                    );
                }

                for i in m_main..m {
                    add_dot_1x4(
                        k,
                        alpha,
                        a.0.add(i),
                        lda,
                        b.0.add(j * ldb),
                        ldb,
                        beta,
                        c.0.add(i + j * ldc),
                        ldc,
                    );
                }
            });

        for j in n_main..n {
            for i in 0..m {
                add_dot_1x1(
                    k,
                    alpha,
                    a.0.add(i),
                    lda,
                    b.0.add(j * ldb),
                    beta,
                    c.0.add(i + j * ldc),
                );
            }
        }
    }

    unsafe fn pack_b(k: usize, b: *const f64, ldb: usize, mut packed_b: *mut f64) {
        let mut bptr0 = b;
        let mut bptr1 = b.add(ldb);
        let mut bptr2 = b.add(ldb * 2);
        let mut bptr3 = b.add(ldb * 3);

        for _ in 0..k {
            *packed_b = *bptr0;
            *packed_b.add(1) = *bptr1;
            *packed_b.add(2) = *bptr2;
            *packed_b.add(3) = *bptr3;

            packed_b = packed_b.add(4);
            bptr0 = bptr0.add(1);
            bptr1 = bptr1.add(1);
            bptr2 = bptr2.add(1);
            bptr3 = bptr3.add(1);
        }
    }

    unsafe fn add_dot_1x1(
        k: usize,
        alpha: f64,
        mut a: *const f64,
        lda: usize,
        mut b: *const f64,
        beta: f64,
        c: *mut f64,
    ) {
        let mut c0reg = *c * beta;
        for _ in 0..k {
            c0reg += *a * *b * alpha;

            a = a.add(lda);
            b = b.add(1);
        }
        *c = c0reg;
    }

    unsafe fn add_dot_1x4(
        k: usize,
        alpha: f64,
        mut a: *const f64,
        lda: usize,
        b: *const f64,
        ldb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    ) {
        let mut bptr0 = b;
        let mut bptr1 = b.add(ldb);
        let mut bptr2 = b.add(ldb * 2);
        let mut bptr3 = b.add(ldb * 3);

        let cptr0 = c;
        let cptr1 = c.add(ldc);
        let cptr2 = c.add(2 * ldc);
        let cptr3 = c.add(3 * ldc);

        let mut c0_reg = *cptr0 * beta;
        let mut c1_reg = *cptr1 * beta;
        let mut c2_reg = *cptr2 * beta;
        let mut c3_reg = *cptr3 * beta;

        for _ in 0..k {
            let a0_reg = *a * alpha;
            let bp0reg = *bptr0;
            let bp1reg = *bptr1;
            let bp2reg = *bptr2;
            let bp3reg = *bptr3;

            c0_reg += a0_reg * bp0reg;
            c1_reg += a0_reg * bp1reg;
            c2_reg += a0_reg * bp2reg;
            c3_reg += a0_reg * bp3reg;

            a = a.add(lda);
            bptr0 = bptr0.add(1);
            bptr1 = bptr1.add(1);
            bptr2 = bptr2.add(1);
            bptr3 = bptr3.add(1);
        }

        *cptr0 = c0_reg;
        *cptr1 = c1_reg;
        *cptr2 = c2_reg;
        *cptr3 = c3_reg;
    }
}
