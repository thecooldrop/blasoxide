use crate::{DSend, DSendMut};
use core::arch::x86_64::*;
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

    let mut packed_a = vec![0.0; MC * KC];
    let mut packed_b = vec![0.0; KC * NB];

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
        let m_left = m % 4;
        let m_main = m - m_left;

        let a = DSend(a);
        let b = DSend(b);
        let c = DSendMut(c);
        let packed_a = DSendMut(packed_a);
        let packed_b = DSendMut(packed_b);

        (0..n_main)
            .step_by(4)
            .collect::<Vec<_>>()
            .par_iter()
            .for_each(move |&j| {
                if first_time {
                    pack_b(k, b.0.add(j * ldb), ldb, packed_b.0.add(j * k));
                }
                if j == 0 {
                    for i in (0..m_main).step_by(4) {
                        pack_a(k, alpha, a.0.add(i), lda, packed_a.0.add(i * k));
                    }
                }
            });

        (0..n_main)
            .step_by(4)
            .collect::<Vec<_>>()
            .par_iter()
            .for_each(move |&j| {
                for i in (0..m_main).step_by(4) {
                    add_dot_4x4_packed(
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
            for i in (0..m_main).step_by(4) {
                add_dot_4x1(
                    k,
                    alpha,
                    a.0.add(i),
                    lda,
                    b.0.add(j * ldb),
                    beta,
                    c.0.add(i + j * ldc),
                );
            }
            for i in m_main..m {
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

    unsafe fn pack_a(k: usize, alpha: f64, mut a: *const f32, lda: usize, mut packed_a: *mut f64) {
        let alphav = _mm256_broadcast_ss(&alpha);

        for _ in 0..k {
            _mm256_storeu_pd(packed_a, _mm256_mul_pd(alphav, _mm256_loadu_pd(a)));

            a = a.add(lda);
            packed_a = packed_a.add(8);
        }
    }

    unsafe fn add_dot_4x4_packed(
        k: usize,
        mut a: *const f64,
        mut b: *const f64,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    ) {
        let cptr0 = &mut *c;
        let cptr1 = &mut *c.add(ldc);
        let cptr2 = &mut *c.add(2 * ldc);
        let cptr3 = &mut *c.add(3 * ldc);

        let betav = _mm256_broadcast_sd(&beta);
        let mut c0_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr0));
        let mut c1_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr1));
        let mut c2_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr2));
        let mut c3_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr3));

        for _ in 0..k {
            let a0_reg_v = _mm256_loadu_pd(a);
            let bp0reg = _mm256_broadcast_sd(&*b);
            let bp1reg = _mm256_broadcast_sd(&*b.add(1));
            let bp2reg = _mm256_broadcast_sd(&*b.add(2));
            let bp3reg = _mm256_broadcast_sd(&*b.add(3));

            c0_reg_v = _mm256_fmadd_pd(a0_reg_v, bp0reg, c0_reg_v);
            c1_reg_v = _mm256_fmadd_pd(a0_reg_v, bp1reg, c1_reg_v);
            c2_reg_v = _mm256_fmadd_pd(a0_reg_v, bp2reg, c2_reg_v);
            c3_reg_v = _mm256_fmadd_pd(a0_reg_v, bp3reg, c3_reg_v);

            a = a.add(8);
            b = b.add(4);
        }

        _mm256_storeu_pd(cptr0, c0_reg_v);
        _mm256_storeu_pd(cptr1, c1_reg_v);
        _mm256_storeu_pd(cptr2, c2_reg_v);
        _mm256_storeu_pd(cptr3, c3_reg_v);
    }

    unsafe fn add_dot_1x1(
        k: usize,
        alpha: f32,
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

    unsafe fn add_dot_4x1(
        k: usize,
        alpha: f64,
        mut a: *const f64,
        lda: usize,
        mut b: *const f64,
        beta: f64,
        c: *mut f64,
    ) {
        let cptr0 = &mut *c;

        let betav = _mm256_broadcast_sd(&beta);
        let mut c0_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr0));
        let alphav = _mm256_broadcast_sd(&alpha);

        for _ in 0..k {
            let a0_reg_v = _mm256_mul_pd(alphav, _mm256_loadu_pd(a));
            let b0_reg_v = _mm256_broadcast_sd(&*b);

            c0_reg_v = _mm256_fmadd_pd(a0_reg_v, b0_reg_v, c0_reg_v);

            a = a.add(lda);
            b = b.add(1);
        }

        _mm256_storeu_pd(cptr0, c0_reg_v);
    }
}
