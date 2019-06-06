use crate::{SSend, SSendMut};
use core::arch::x86_64::*;
use rayon::prelude::*;

pub unsafe fn sgemm(
    _transa: bool,
    _transb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    const MC: usize = 512;
    const KC: usize = 256;
    const NB: usize = 2048;

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
        alpha: f32,
        a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        beta: f32,
        c: *mut f32,
        ldc: usize,
        packed_a: *mut f32,
        packed_b: *mut f32,
        first_time: bool,
    ) {
        let n_left = n % 4;
        let n_main = n - n_left;
        let m_left = m % 16;
        let m_main = m - m_left;

        let a = SSend(a);
        let b = SSend(b);
        let c = SSendMut(c);
        let packed_a = SSendMut(packed_a);
        let packed_b = SSendMut(packed_b);

        (0..n_main)
            .step_by(4)
            .collect::<Vec<_>>()
            .par_iter()
            .for_each(move |&j| {
                if first_time {
                    pack_b(k, b.0.add(j * ldb), ldb, packed_b.0.add(j * k));
                }
                if j == 0 {
                    for i in (0..m_main).step_by(8) {
                        pack_a(k, alpha, a.0.add(i), lda, packed_a.0.add(i * k));
                    }
                }
            });

        (0..n_main)
            .step_by(4)
            .collect::<Vec<_>>()
            .par_iter()
            .for_each(move |&j| {
                for i in (0..m_main).step_by(16) {
                    add_dot_16x4_packed(
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
            for i in (0..m_main).step_by(16) {
                add_dot_16x1(
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

    unsafe fn pack_b(k: usize, b: *const f32, ldb: usize, mut packed_b: *mut f32) {
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

    unsafe fn pack_a(k: usize, alpha: f32, mut a: *const f32, lda: usize, mut packed_a: *mut f32) {
        let alphav = _mm256_broadcast_ss(&alpha);

        for _ in 0..k {
            _mm256_storeu_ps(packed_a, _mm256_mul_ps(alphav, _mm256_loadu_ps(a)));

            a = a.add(lda);
            packed_a = packed_a.add(8);
        }
    }

    unsafe fn add_dot_16x4_packed(
        k: usize,
        mut a: *const f32,
        mut b: *const f32,
        beta: f32,
        c: *mut f32,
        ldc: usize,
    ) {
        let cptr0 = c;
        let cptr1 = c.add(ldc);
        let cptr2 = c.add(2 * ldc);
        let cptr3 = c.add(3 * ldc);

        let betav = _mm256_broadcast_ss(&beta);
        let mut c0_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(cptr0));
        let mut c1_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(cptr1));
        let mut c2_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(cptr2));
        let mut c3_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(cptr3));
        let mut c01_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(cptr0.add(8)));
        let mut c11_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(cptr1.add(8)));
        let mut c21_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(cptr2.add(8)));
        let mut c31_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(cptr3.add(8)));

        for _ in 0..k {
            let a0_reg_v = _mm256_loadu_ps(a);
            let a1_reg_v = _mm256_loadu_ps(a.add(8));

            let bp0reg = _mm256_broadcast_ss(&*b);
            let bp1reg = _mm256_broadcast_ss(&*b.add(1));
            let bp2reg = _mm256_broadcast_ss(&*b.add(2));
            let bp3reg = _mm256_broadcast_ss(&*b.add(3));

            c0_reg_v = _mm256_fmadd_ps(a0_reg_v, bp0reg, c0_reg_v);
            c1_reg_v = _mm256_fmadd_ps(a0_reg_v, bp1reg, c1_reg_v);
            c2_reg_v = _mm256_fmadd_ps(a0_reg_v, bp2reg, c2_reg_v);
            c3_reg_v = _mm256_fmadd_ps(a0_reg_v, bp3reg, c3_reg_v);
            c01_reg_v = _mm256_fmadd_ps(a1_reg_v, bp0reg, c01_reg_v);
            c11_reg_v = _mm256_fmadd_ps(a1_reg_v, bp1reg, c11_reg_v);
            c21_reg_v = _mm256_fmadd_ps(a1_reg_v, bp2reg, c21_reg_v);
            c31_reg_v = _mm256_fmadd_ps(a1_reg_v, bp3reg, c31_reg_v);

            a = a.add(16);
            b = b.add(4);
        }

        _mm256_storeu_ps(cptr0, c0_reg_v);
        _mm256_storeu_ps(cptr1, c1_reg_v);
        _mm256_storeu_ps(cptr2, c2_reg_v);
        _mm256_storeu_ps(cptr3, c3_reg_v);
        _mm256_storeu_ps(cptr0.add(8), c01_reg_v);
        _mm256_storeu_ps(cptr1.add(8), c11_reg_v);
        _mm256_storeu_ps(cptr2.add(8), c21_reg_v);
        _mm256_storeu_ps(cptr3.add(8), c31_reg_v);
    }

    unsafe fn add_dot_1x1(
        k: usize,
        alpha: f32,
        mut a: *const f32,
        lda: usize,
        mut b: *const f32,
        beta: f32,
        c: *mut f32,
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
        alpha: f32,
        mut a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        beta: f32,
        c: *mut f32,
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

    unsafe fn add_dot_16x1(
        k: usize,
        alpha: f32,
        mut a: *const f32,
        lda: usize,
        mut b: *const f32,
        beta: f32,
        c: *mut f32,
    ) {
        let betav = _mm256_broadcast_ss(&beta);
        let mut c0_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(c));
        let mut c1_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(c.add(8)));
        let alphav = _mm256_broadcast_ss(&alpha);

        for _ in 0..k {
            let a0_reg_v = _mm256_mul_ps(alphav, _mm256_loadu_ps(a));
            let a1_reg_v = _mm256_mul_ps(alphav, _mm256_loadu_ps(a.add(8)));
            let b0_reg_v = _mm256_broadcast_ss(&*b);

            c0_reg_v = _mm256_fmadd_ps(a0_reg_v, b0_reg_v, c0_reg_v);
            c1_reg_v = _mm256_fmadd_ps(a1_reg_v, b0_reg_v, c1_reg_v);

            a = a.add(lda);
            b = b.add(1);
        }

        _mm256_storeu_ps(c, c0_reg_v);
        _mm256_storeu_ps(c.add(8), c1_reg_v);
    }
}
