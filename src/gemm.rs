// https://github.com/flame/how-to-optimize-gemm

use core::arch::x86_64::*;

pub unsafe fn sgemm(
    _transa: bool,
    _transb: bool,
    m: usize,
    n: usize,
    k: usize,
    _alpha: f32,
    mut a: *const f32,
    lda: isize,
    mut b: *const f32,
    ldb: isize,
    _beta: f32,
    mut c: *mut f32,
    ldc: isize,
) {
    const MC: usize = 256;
    const KC: usize = 128;
    const UNROLL: usize = 4;

    for _ in 0..k / KC {
        for _ in 0..m / MC {
            inner_kernel(MC, n, KC, a, lda, b, ldb, c, ldc);

            b = b.add(MC);
            c = c.add(MC);
        }

        a = a.offset(KC as isize * ldb);
        b = b.add(KC);
    }

    unsafe fn inner_kernel(
        m: usize,
        n: usize,
        k: usize,
        mut a: *const f32,
        lda: isize,
        mut b: *const f32,
        ldb: isize,
        mut c: *mut f32,
        ldc: isize,
    ) {
        for _ in 0..n / UNROLL {
            for _ in 0..m / 8 {
                add_dot_4x8(k, a, lda, b, ldb, c, ldc);
                
                a = a.add(8);
                c = c.add(8);
            }

            b = b.offset(UNROLL as isize * ldb);
            c = c.offset(UNROLL as isize * ldc);
        }
    }

    unsafe fn add_dot_4x8(
        k: usize,
        mut a: *const f32,
        lda: isize,
        b: *const f32,
        ldb: isize,
        c: *mut f32,
        ldc: isize,
    ) {
        let mut bptr0 = b;
        let mut bptr1 = b.offset(ldb);
        let mut bptr2 = b.offset(2 * ldb);
        let mut bptr3 = b.offset(3 * ldb);

        let mut c0_reg_v = _mm256_setzero_ps();
        let mut c1_reg_v = _mm256_setzero_ps();
        let mut c2_reg_v = _mm256_setzero_ps();
        let mut c3_reg_v = _mm256_setzero_ps();

        for _ in 0..k {
            let a0_reg_v = _mm256_loadu_ps(a);
            let bp0reg = _mm256_broadcast_ss(&*bptr0);
            let bp1reg = _mm256_broadcast_ss(&*bptr1);
            let bp2reg = _mm256_broadcast_ss(&*bptr2);
            let bp3reg = _mm256_broadcast_ss(&*bptr3);

            c0_reg_v = _mm256_fmadd_ps(a0_reg_v, bp0reg, c0_reg_v);
            c1_reg_v = _mm256_fmadd_ps(a0_reg_v, bp1reg, c1_reg_v);
            c2_reg_v = _mm256_fmadd_ps(a0_reg_v, bp2reg, c2_reg_v);
            c3_reg_v = _mm256_fmadd_ps(a0_reg_v, bp3reg, c3_reg_v);

            a = a.offset(lda);
            bptr0 = bptr0.offset(1);
            bptr1 = bptr1.offset(1);
            bptr2 = bptr2.offset(1);
            bptr3 = bptr3.offset(1);
        }

        let cptr0 = &mut *c;
        let cptr1 = &mut *c.offset(ldc);
        let cptr2 = &mut *c.offset(2 * ldc);
        let cptr3 = &mut *c.offset(3 * ldc);

        _mm256_storeu_ps(cptr0, _mm256_add_ps(_mm256_loadu_ps(cptr0), c0_reg_v));
        _mm256_storeu_ps(cptr1, _mm256_add_ps(_mm256_loadu_ps(cptr1), c1_reg_v));
        _mm256_storeu_ps(cptr2, _mm256_add_ps(_mm256_loadu_ps(cptr2), c2_reg_v));
        _mm256_storeu_ps(cptr3, _mm256_add_ps(_mm256_loadu_ps(cptr3), c3_reg_v));
    }
}
