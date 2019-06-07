pub unsafe fn sgemm_16x4_packed(
    k: usize,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::sgemm_16x4_packed(k, a, b, beta, c, ldc);
            return;
        }
    }

    let cptr0 = c;
    let cptr1 = c.add(ldc);
    let cptr2 = c.add(ldc * 2);
    let cptr3 = c.add(ldc * 3);

    for i in 0..16 {
        let mut creg0 = *cptr0.add(i) * beta;
        let mut creg1 = *cptr1.add(i) * beta;
        let mut creg2 = *cptr2.add(i) * beta;
        let mut creg3 = *cptr3.add(i) * beta;
        
        let mut bptr = b;

        for p in 0..k {
            let areg = *a.add(i + p * 16);

            let breg0 = *bptr;
            let breg1 = *bptr.add(1);
            let breg2 = *bptr.add(2);
            let breg3 = *bptr.add(3);

            creg0 += breg0 * areg;
            creg1 += breg1 * areg;
            creg2 += breg2 * areg;
            creg3 += breg3 * areg;

            bptr = bptr.add(4);
        }

        *cptr0.add(i) = creg0;
        *cptr1.add(i) = creg1;
        *cptr2.add(i) = creg2;
        *cptr3.add(i) = creg3;
    }
}

pub unsafe fn dgemm_8x4_packed(
    k: usize,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::dgemm_8x4_packed(k, a, b, beta, c, ldc);
            return;
        }
    }

    let cptr0 = c;
    let cptr1 = c.add(ldc);
    let cptr2 = c.add(ldc * 2);
    let cptr3 = c.add(ldc * 3);

    for i in 0..8 {
        let mut creg0 = *cptr0.add(i) * beta;
        let mut creg1 = *cptr1.add(i) * beta;
        let mut creg2 = *cptr2.add(i) * beta;
        let mut creg3 = *cptr3.add(i) * beta;
        
        let mut bptr = b;

        for p in 0..k {
            let areg = *a.add(i + p * 8);

            let breg0 = *bptr;
            let breg1 = *bptr.add(1);
            let breg2 = *bptr.add(2);
            let breg3 = *bptr.add(3);

            creg0 += breg0 * areg;
            creg1 += breg1 * areg;
            creg2 += breg2 * areg;
            creg3 += breg3 * areg;

            bptr = bptr.add(4);
        }

        *cptr0.add(i) = creg0;
        *cptr1.add(i) = creg1;
        *cptr2.add(i) = creg2;
        *cptr3.add(i) = creg3;
    }
}
