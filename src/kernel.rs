use core::arch::x86_64::*;

pub unsafe fn sadot_8x4(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    let mut bptr0 = b;
    let mut bptr1 = b.add(ldb);
    let mut bptr2 = b.add(ldb * 2);
    let mut bptr3 = b.add(ldb * 3);

    let mut creg0 = _mm256_setzero_ps();
    let mut creg1 = _mm256_setzero_ps();
    let mut creg2 = _mm256_setzero_ps();
    let mut creg3 = _mm256_setzero_ps();

    let mut aptr = a;

    for _ in 0..k {
        let areg = _mm256_loadu_ps(aptr);

        creg0 = _mm256_fmadd_ps(areg, _mm256_broadcast_ss(&*bptr0), creg0);
        creg1 = _mm256_fmadd_ps(areg, _mm256_broadcast_ss(&*bptr1), creg1);
        creg2 = _mm256_fmadd_ps(areg, _mm256_broadcast_ss(&*bptr2), creg2);
        creg3 = _mm256_fmadd_ps(areg, _mm256_broadcast_ss(&*bptr3), creg3);

        aptr = aptr.add(lda);
        bptr0 = bptr0.add(1);
        bptr1 = bptr1.add(1);
        bptr2 = bptr2.add(1);
        bptr3 = bptr3.add(1);
    }

    _mm256_storeu_ps(c, creg0);
    _mm256_storeu_ps(c.add(ldc), creg1);
    _mm256_storeu_ps(c.add(ldc * 2), creg2);
    _mm256_storeu_ps(c.add(ldc * 3), creg3);
}

pub unsafe fn sadot_8x1(k: usize, a: *const f32, lda: usize, b: *const f32, c: *mut f32) {
    let mut bptr = b;

    let mut creg = _mm256_setzero_ps();

    let mut aptr = a;

    for _ in 0..k {
        creg = _mm256_fmadd_ps(_mm256_loadu_ps(aptr), _mm256_broadcast_ss(&*bptr), creg);

        aptr = aptr.add(lda);
        bptr = bptr.add(1);
    }

    _mm256_storeu_ps(c, creg);
}

pub unsafe fn sadot_1x4(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    let mut bptr0 = b;
    let mut bptr1 = b.add(ldb);
    let mut bptr2 = b.add(ldb * 2);
    let mut bptr3 = b.add(ldb * 3);

    let mut creg0 = 0.0;
    let mut creg1 = 0.0;
    let mut creg2 = 0.0;
    let mut creg3 = 0.0;

    let mut aptr = a;

    for _ in 0..k {
        let areg = *a;

        creg0 += areg * *bptr0;
        creg1 += areg * *bptr1;
        creg2 += areg * *bptr2;
        creg3 += areg * *bptr3;

        aptr = aptr.add(lda);
        bptr0 = bptr0.add(1);
        bptr1 = bptr1.add(1);
        bptr2 = bptr2.add(1);
        bptr3 = bptr3.add(1);
    }

    *c = creg0;
    *c.add(ldc) = creg1;
    *c.add(ldc * 2) = creg2;
    *c.add(ldc * 3) = creg3;
}

pub unsafe fn sadot_1x1(k: usize, a: *const f32, lda: usize, b: *const f32, c: *mut f32) {
    let mut creg = 0.0;

    let mut aptr = a;

    let mut bptr = b;

    for _ in 0..k {
        creg += *aptr * *bptr;

        aptr = aptr.add(lda);
        bptr = bptr.add(1);
    }

    *c = creg;
}
