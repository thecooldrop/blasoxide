pub unsafe fn dgemv(
    _trans: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    x: *const f64,
    incx: usize,
    beta: f64,
    y: *mut f64,
    incy: usize,
) {
    const MC: usize = 256;
    const NC: usize = 128;

    let mut beta_scale = beta;

    for j in (0..n).step_by(NC) {
        let jb = std::cmp::min(n - j, NC);
        for i in (0..m).step_by(MC) {
            let ib = std::cmp::min(m - i, MC);
            inner_kernel(
                ib,
                jb,
                alpha,
                a.add(i + j * lda),
                lda,
                x.add(j * incx),
                incx,
                beta_scale,
                y.add(i * incy),
                incy,
            );
        }
        beta_scale = 1.0;
    }

    unsafe fn inner_kernel(
        m: usize,
        n: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        x: *const f64,
        incx: usize,
        beta: f64,
        y: *mut f64,
        incy: usize,
    ) {
        if incy == 1 {
            let m_left = m % 4;
            let m_main = m - m_left;

            let alphav = _mm256_broadcast_sd(&alpha);
            let mut beta_scale = beta;

            for j in 0..n {
                let betav = _mm256_broadcast_sd(&beta_scale);
                let xbase = x.add(j * incx);
                let xreg = _mm256_broadcast_sd(&*xbase);
                for i in (0..m_main).step_by(4) {
                    let areg = _mm256_mul_pd(alphav, _mm256_loadu_pd(a.add(i + j * lda)));
                    let ybase = y.add(i * incy);
                    _mm256_storeu_pd(
                        ybase,
                        _mm256_fmadd_pd(areg, xreg, _mm256_mul_pd(betav, _mm256_loadu_pd(ybase))),
                    );
                }
                for i in m_main..m {
                    let areg = *a.add(i + j * lda) * alpha;
                    let ybase = y.add(i * incy);
                    let xbase = x.add(j * incx);
                    *ybase = beta_scale * *ybase + areg * *xbase;
                }

                beta_scale = 1.0;
            }
        } else {
            let mut beta_scale = beta;
            for j in 0..n {
                let xreg = *x.add(j * incx);
                for i in 0..m {
                    let areg = alpha * *a.add(i + j * lda);
                    let ybase = y.add(i * incy);
                    *ybase = *ybase * beta_scale + xreg * areg;
                }

                beta_scale = 1.0;
            }
        }
    }
}
