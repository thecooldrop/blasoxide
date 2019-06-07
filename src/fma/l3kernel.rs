use core::arch::x86_64::*;

#[target_feature(enable = "fma")]
pub unsafe fn sgemm_16x4_packed(
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

#[target_feature(enable = "fma")]
pub unsafe fn s_pack_a(k: usize, alpha: f32, mut a: *const f32, lda: usize, mut packed_a: *mut f32) {
    let alphav = _mm256_broadcast_ss(&alpha);

    for _ in 0..k {
        _mm256_storeu_ps(packed_a, _mm256_mul_ps(alphav, _mm256_loadu_ps(a)));
        _mm256_storeu_ps(
            packed_a.add(8),
            _mm256_mul_ps(alphav, _mm256_loadu_ps(a.add(8))),
        );

        a = a.add(lda);
        packed_a = packed_a.add(16);
    }
}

#[target_feature(enable = "fma")]
pub unsafe fn dgemm_8x4_packed(
    k: usize,
    mut a: *const f64,
    mut b: *const f64,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    let cptr0 = c;
    let cptr1 = c.add(ldc);
    let cptr2 = c.add(2 * ldc);
    let cptr3 = c.add(3 * ldc);

    let betav = _mm256_broadcast_sd(&beta);
    let mut c0_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr0));
    let mut c1_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr1));
    let mut c2_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr2));
    let mut c3_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr3));
    let mut c01_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr0.add(4)));
    let mut c11_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr1.add(4)));
    let mut c21_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr2.add(4)));
    let mut c31_reg_v = _mm256_mul_pd(betav, _mm256_loadu_pd(cptr3.add(4)));

    for _ in 0..k {
        let a0_reg_v = _mm256_loadu_pd(a);
        let a1_reg_v = _mm256_loadu_pd(a.add(4));

        let bp0reg = _mm256_broadcast_sd(&*b);
        let bp1reg = _mm256_broadcast_sd(&*b.add(1));
        let bp2reg = _mm256_broadcast_sd(&*b.add(2));
        let bp3reg = _mm256_broadcast_sd(&*b.add(3));

        c0_reg_v = _mm256_fmadd_pd(a0_reg_v, bp0reg, c0_reg_v);
        c1_reg_v = _mm256_fmadd_pd(a0_reg_v, bp1reg, c1_reg_v);
        c2_reg_v = _mm256_fmadd_pd(a0_reg_v, bp2reg, c2_reg_v);
        c3_reg_v = _mm256_fmadd_pd(a0_reg_v, bp3reg, c3_reg_v);
        c01_reg_v = _mm256_fmadd_pd(a1_reg_v, bp0reg, c01_reg_v);
        c11_reg_v = _mm256_fmadd_pd(a1_reg_v, bp1reg, c11_reg_v);
        c21_reg_v = _mm256_fmadd_pd(a1_reg_v, bp2reg, c21_reg_v);
        c31_reg_v = _mm256_fmadd_pd(a1_reg_v, bp3reg, c31_reg_v);

        a = a.add(8);
        b = b.add(4);
    }

    _mm256_storeu_pd(cptr0, c0_reg_v);
    _mm256_storeu_pd(cptr1, c1_reg_v);
    _mm256_storeu_pd(cptr2, c2_reg_v);
    _mm256_storeu_pd(cptr3, c3_reg_v);
    _mm256_storeu_pd(cptr0.add(4), c01_reg_v);
    _mm256_storeu_pd(cptr1.add(4), c11_reg_v);
    _mm256_storeu_pd(cptr2.add(4), c21_reg_v);
    _mm256_storeu_pd(cptr3.add(4), c31_reg_v);
}

#[target_feature(enable = "fma")]
pub unsafe fn d_pack_a(k: usize, alpha: f64, mut a: *const f64, lda: usize, mut packed_a: *mut f64) {
    let alphav = _mm256_broadcast_sd(&alpha);

    for _ in 0..k {
        _mm256_storeu_pd(packed_a, _mm256_mul_pd(alphav, _mm256_loadu_pd(a)));
        _mm256_storeu_pd(
            packed_a.add(4),
            _mm256_mul_pd(alphav, _mm256_loadu_pd(a.add(4))),
        );

        a = a.add(lda);
        packed_a = packed_a.add(8);
    }
}