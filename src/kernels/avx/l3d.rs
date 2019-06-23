use super::fma::fmadd_pd;
use super::intrinsics::*;

pub(crate) unsafe fn dgemm_ukr_8x4(
    k: usize,
    alpha: f64,
    pa: *const f64,
    pb: *const f64,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    let mut mt00 = _mm256_setzero_pd();
    let mut mt01 = _mm256_setzero_pd();
    let mut mt02 = _mm256_setzero_pd();
    let mut mt03 = _mm256_setzero_pd();
    let mut mt10 = _mm256_setzero_pd();
    let mut mt11 = _mm256_setzero_pd();
    let mut mt12 = _mm256_setzero_pd();
    let mut mt13 = _mm256_setzero_pd();

    let mut pa = pa;
    let mut pb = pb;

    for _ in 0..k {
        let a0 = _mm256_load_pd(pa);
        let a1 = _mm256_load_pd(pa.add(4));

        let b0 = _mm256_broadcast_sd(&*pb);
        let b1 = _mm256_broadcast_sd(&*pb.add(1));
        let b2 = _mm256_broadcast_sd(&*pb.add(2));
        let b3 = _mm256_broadcast_sd(&*pb.add(3));

        mt00 = fmadd_pd(a0, b0, mt00);
        mt01 = fmadd_pd(a0, b1, mt01);
        mt02 = fmadd_pd(a0, b2, mt02);
        mt03 = fmadd_pd(a0, b3, mt03);
        mt10 = fmadd_pd(a1, b0, mt10);
        mt11 = fmadd_pd(a1, b1, mt11);
        mt12 = fmadd_pd(a1, b2, mt12);
        mt13 = fmadd_pd(a1, b3, mt13);

        pa = pa.add(8);
        pb = pb.add(4);
    }

    let alpha = _mm256_broadcast_sd(&alpha);

    mt00 = _mm256_mul_pd(alpha, mt00);
    mt01 = _mm256_mul_pd(alpha, mt01);
    mt02 = _mm256_mul_pd(alpha, mt02);
    mt03 = _mm256_mul_pd(alpha, mt03);
    mt10 = _mm256_mul_pd(alpha, mt10);
    mt11 = _mm256_mul_pd(alpha, mt11);
    mt12 = _mm256_mul_pd(alpha, mt12);
    mt13 = _mm256_mul_pd(alpha, mt13);

    let ccol0 = c;
    let ccol1 = c.add(ldc);
    let ccol2 = c.add(ldc * 2);
    let ccol3 = c.add(ldc * 3);

    if beta != 0.0 {
        let beta = _mm256_broadcast_sd(&beta);

        mt00 = fmadd_pd(beta, _mm256_load_pd(ccol0), mt00);
        mt01 = fmadd_pd(beta, _mm256_load_pd(ccol1), mt01);
        mt02 = fmadd_pd(beta, _mm256_load_pd(ccol2), mt02);
        mt03 = fmadd_pd(beta, _mm256_load_pd(ccol3), mt03);
        mt10 = fmadd_pd(beta, _mm256_load_pd(ccol0.add(4)), mt10);
        mt11 = fmadd_pd(beta, _mm256_load_pd(ccol1.add(4)), mt11);
        mt12 = fmadd_pd(beta, _mm256_load_pd(ccol2.add(4)), mt12);
        mt13 = fmadd_pd(beta, _mm256_load_pd(ccol3.add(4)), mt13);
    }

    _mm256_store_pd(ccol0, mt00);
    _mm256_store_pd(ccol1, mt01);
    _mm256_store_pd(ccol2, mt02);
    _mm256_store_pd(ccol3, mt03);
    _mm256_store_pd(ccol0.add(4), mt10);
    _mm256_store_pd(ccol1.add(4), mt11);
    _mm256_store_pd(ccol2.add(4), mt12);
    _mm256_store_pd(ccol3.add(4), mt13);
}

pub(crate) unsafe fn dgemm_sup_8x1(
    k: usize,
    alpha: f64,
    pa: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
) {
    let mut mt0 = _mm256_setzero_pd();
    let mut mt1 = _mm256_setzero_pd();

    let mut pa = pa;
    let mut b = b;

    for _ in 0..k {
        let a0 = _mm256_load_pd(pa);
        let a1 = _mm256_load_pd(pa.add(4));

        let b0 = _mm256_broadcast_sd(&*b);

        mt0 = fmadd_pd(a0, b0, mt0);
        mt1 = fmadd_pd(a1, b0, mt1);

        pa = pa.add(8);
        b = b.add(1);
    }

    let alpha = _mm256_broadcast_sd(&alpha);

    mt0 = _mm256_mul_pd(alpha, mt0);
    mt1 = _mm256_mul_pd(alpha, mt1);

    if beta != 0.0 {
        let beta = _mm256_broadcast_sd(&beta);

        mt0 = fmadd_pd(beta, _mm256_load_pd(c), mt0);
        mt1 = fmadd_pd(beta, _mm256_load_pd(c.add(4)), mt1);
    }

    _mm256_store_pd(c, mt0);
    _mm256_store_pd(c.add(4), mt1);
}

pub(crate) unsafe fn dgemm_pa_8x(k: usize, a: *const f64, lda: usize, pa: *mut f64) {
    let mut a = a;
    let mut pa = pa;

    for _ in 0..k {
        _mm256_store_pd(pa, _mm256_load_pd(a));
        _mm256_store_pd(pa.add(4), _mm256_load_pd(a.add(4)));

        pa = pa.add(8);
        a = a.add(lda);
    }
}
