pub unsafe fn sgemv(
    _trans: bool,
    m: usize,
    n: usize,
    alpha: f32,
    mut a: *const f32,
    lda: usize,
    mut x: *const f32,
    incx: usize,
    beta: f32,
    y: *mut f32,
    incy: usize,
) {
    let mut beta_scale = beta;

    if incy == 1 {
        for _ in 0..n / 4 {
            crate::scombine_4(m, alpha, a, lda, x, incx, beta_scale, y);

            x = x.add(4 * incx);
            a = a.add(4 * lda);
            beta_scale = 1.0;
        }

        for _ in 0..n % 4 {
            crate::scombine_1(m, alpha, a, x, beta_scale, y);

            x = x.add(incx);
            a = a.add(lda);
            beta_scale = 1.0;
        }
    } else {
        for _ in 0..n / 4 {
            let mut aptr0 = a;
            let mut aptr1 = a.add(lda);
            let mut aptr2 = a.add(lda * 2);
            let mut aptr3 = a.add(lda * 3);

            let xreg0 = *x * alpha;
            let xreg1 = *x.add(incx) * alpha;
            let xreg2 = *x.add(2 * incx) * alpha;
            let xreg3 = *x.add(3 * incx) * alpha;

            let mut yptr = y;

            for _ in 0..m {
                *yptr = beta_scale * *yptr +
                    *aptr0 * xreg0 +
                    *aptr1 * xreg1 +
                    *aptr2 * xreg2 +
                    *aptr3 * xreg3;
                yptr = yptr.add(incy);
                aptr0 = aptr0.add(1);
                aptr1 = aptr1.add(1);
                aptr2 = aptr2.add(1);
                aptr3 = aptr3.add(1);
            }

            x = x.add(4 * incx);
            a = a.add(4 * lda);
            beta_scale = 1.0;
        }

        for _ in 0..n % 4 {
            let mut aptr = a;

            let xreg = *x * alpha;

            let mut yptr = y;

            for _ in 0..m {
                *yptr = beta_scale * *yptr + *aptr * xreg;

                yptr = yptr.add(incy);
                aptr = aptr.add(1);
            }

            x = x.add(incx);
            a = a.add(lda);
            beta_scale = 1.0;
        }
    }
}
