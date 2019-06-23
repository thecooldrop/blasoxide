use super::fma::fmadd_ps;
use super::intrinsics::*;

pub(crate) unsafe fn sgemm_ukr_16x4(
    k: usize,
    alpha: f32,
    pa: *const f32,
    pb: *const f32,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    let mut mt00 = _mm256_setzero_ps();
    let mut mt01 = _mm256_setzero_ps();
    let mut mt02 = _mm256_setzero_ps();
    let mut mt03 = _mm256_setzero_ps();
    let mut mt10 = _mm256_setzero_ps();
    let mut mt11 = _mm256_setzero_ps();
    let mut mt12 = _mm256_setzero_ps();
    let mut mt13 = _mm256_setzero_ps();

    let mut pa = pa;
    let mut pb = pb;

    for _ in 0..k {
        let a0 = _mm256_load_ps(pa);
        let a1 = _mm256_load_ps(pa.add(8));

        let b0 = _mm256_broadcast_ss(&*pb);
        let b1 = _mm256_broadcast_ss(&*pb.add(1));
        let b2 = _mm256_broadcast_ss(&*pb.add(2));
        let b3 = _mm256_broadcast_ss(&*pb.add(3));

        mt00 = fmadd_ps(a0, b0, mt00);
        mt01 = fmadd_ps(a0, b1, mt01);
        mt02 = fmadd_ps(a0, b2, mt02);
        mt03 = fmadd_ps(a0, b3, mt03);
        mt10 = fmadd_ps(a1, b0, mt10);
        mt11 = fmadd_ps(a1, b1, mt11);
        mt12 = fmadd_ps(a1, b2, mt12);
        mt13 = fmadd_ps(a1, b3, mt13);

        pa = pa.add(16);
        pb = pb.add(4);
    }

    let alpha = _mm256_broadcast_ss(&alpha);

    mt00 = _mm256_mul_ps(alpha, mt00);
    mt01 = _mm256_mul_ps(alpha, mt01);
    mt02 = _mm256_mul_ps(alpha, mt02);
    mt03 = _mm256_mul_ps(alpha, mt03);
    mt10 = _mm256_mul_ps(alpha, mt10);
    mt11 = _mm256_mul_ps(alpha, mt11);
    mt12 = _mm256_mul_ps(alpha, mt12);
    mt13 = _mm256_mul_ps(alpha, mt13);

    let ccol0 = c;
    let ccol1 = c.add(ldc);
    let ccol2 = c.add(ldc * 2);
    let ccol3 = c.add(ldc * 3);

    if beta != 0.0 {
        let beta = _mm256_broadcast_ss(&beta);

        mt00 = fmadd_ps(beta, _mm256_load_ps(ccol0), mt00);
        mt01 = fmadd_ps(beta, _mm256_load_ps(ccol1), mt01);
        mt02 = fmadd_ps(beta, _mm256_load_ps(ccol2), mt02);
        mt03 = fmadd_ps(beta, _mm256_load_ps(ccol3), mt03);
        mt10 = fmadd_ps(beta, _mm256_load_ps(ccol0.add(8)), mt10);
        mt11 = fmadd_ps(beta, _mm256_load_ps(ccol1.add(8)), mt11);
        mt12 = fmadd_ps(beta, _mm256_load_ps(ccol2.add(8)), mt12);
        mt13 = fmadd_ps(beta, _mm256_load_ps(ccol3.add(8)), mt13);
    }

    _mm256_store_ps(ccol0, mt00);
    _mm256_store_ps(ccol1, mt01);
    _mm256_store_ps(ccol2, mt02);
    _mm256_store_ps(ccol3, mt03);
    _mm256_store_ps(ccol0.add(8), mt10);
    _mm256_store_ps(ccol1.add(8), mt11);
    _mm256_store_ps(ccol2.add(8), mt12);
    _mm256_store_ps(ccol3.add(8), mt13);
}

pub(crate) unsafe fn sgemm_sup_16x1(
    k: usize,
    alpha: f32,
    pa: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
) {
    let mut mt0 = _mm256_setzero_ps();
    let mut mt1 = _mm256_setzero_ps();

    let mut pa = pa;
    let mut b = b;

    for _ in 0..k {
        let a0 = _mm256_load_ps(pa);
        let a1 = _mm256_load_ps(pa.add(8));

        let b0 = _mm256_broadcast_ss(&*b);

        mt0 = fmadd_ps(a0, b0, mt0);
        mt1 = fmadd_ps(a1, b0, mt1);

        pa = pa.add(16);
        b = b.add(1);
    }

    let alpha = _mm256_broadcast_ss(&alpha);

    mt0 = _mm256_mul_ps(alpha, mt0);
    mt1 = _mm256_mul_ps(alpha, mt1);

    if beta != 0.0 {
        let beta = _mm256_broadcast_ss(&beta);

        mt0 = fmadd_ps(beta, _mm256_load_ps(c), mt0);
        mt1 = fmadd_ps(beta, _mm256_load_ps(c.add(8)), mt1);
    }

    _mm256_store_ps(c, mt0);
    _mm256_store_ps(c.add(8), mt1);
}

pub(crate) unsafe fn sgemm_pa_16x(k: usize, a: *const f32, lda: usize, pa: *mut f32) {
    let mut a = a;
    let mut pa = pa;

    for _ in 0..k {
        _mm256_store_ps(pa, _mm256_load_ps(a));
        _mm256_store_ps(pa.add(8), _mm256_load_ps(a.add(8)));

        pa = pa.add(16);
        a = a.add(lda);
    }
}
