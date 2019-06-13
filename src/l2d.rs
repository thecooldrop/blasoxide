pub unsafe fn dgemv(
    trans: bool,
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
    const MC: usize = 2048;

    if trans {
        crate::dscal(m, beta, y, incy);

        let m_b: usize = MC / incx;

        for i in (0..m).step_by(m_b) {
            let ib = std::cmp::min(m - i, m_b);
            inner_kernel_trans(ib, n, alpha, a.add(i), lda, x.add(i * incx), incx, y, incy);
        }
        return;
    }

    let m_b = MC / incy;

    for i in (0..m).step_by(m_b) {
        let ib = std::cmp::min(m - i, m_b);
        inner_kernel(
            ib,
            n,
            alpha,
            a.add(i),
            lda,
            x,
            incx,
            beta,
            y.add(i * incy),
            incy,
        );
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
        crate::dscal(m, beta, y, incy);
        for j in 0..n {
            crate::daxpy(m, alpha * *x.add(j * incx), a.add(j * lda), 1, y, incy);
        }
    }

    unsafe fn inner_kernel_trans(
        m: usize,
        n: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        x: *const f64,
        incx: usize,
        y: *mut f64,
        incy: usize,
    ) {
        for j in 0..n {
            *y.add(j * incy) += alpha * crate::ddot(m, a.add(j * lda), 1, x, incx);
        }
    }
}

pub unsafe fn dtrmv(
    upper: bool,
    trans: bool,
    diag: bool,
    n: usize,
    a: *const f64,
    lda: usize,
    x: *mut f64,
    incx: usize,
) {
    if trans {
        if upper {
            if diag {
                for j in (1..n).rev() {
                    *x.add(j * incx) += crate::ddot(j, a.add(j * lda), 1, x, incx);
                }
            } else {
                for j in (0..n).rev() {
                    *x.add(j * incx) = crate::ddot(j + 1, a.add(j * lda), 1, x, incx);
                }
            }
        } else if diag {
            for j in 0..n - 1 {
                *x.add(j * incx) += crate::ddot(
                    n - (j + 1),
                    a.add((j + 1) + j * lda),
                    1,
                    x.add((j + 1) * incx),
                    incx,
                );
            }
        } else {
            for j in 0..n {
                *x.add(j * incx) = crate::ddot(n - j, a.add(j + j * lda), 1, x.add(j * incx), incx);
            }
        }
    } else if upper {
        if diag {
            for j in 1..n {
                crate::daxpy(j, *x.add(j * incx), a.add(j * lda), 1, x, incx);
            }
        } else {
            for j in 0..n {
                let scal = *x.add(j * incx);
                *x.add(j * incx) = 0.;
                crate::daxpy(j + 1, scal, a.add(j * lda), 1, x, incx);
            }
        }
    } else if diag {
        for j in (0..n - 1).rev() {
            crate::daxpy(
                n - (j + 1),
                *x.add(j * incx),
                a.add((j + 1) + j * lda),
                1,
                x.add((j + 1) * incx),
                incx,
            );
        }
    } else {
        for j in (0..n).rev() {
            let scal = *x.add(j * incx);
            *x.add(j * incx) = 0.;
            crate::daxpy(n - j, scal, a.add(j + j * lda), 1, x.add(j * incx), incx);
        }
    }
}

pub unsafe fn dsymv(
    upper: bool,
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
    crate::dscal(n, beta, y, incy);
    if upper {
        for j in 0..n {
            crate::daxpy(j + 1, alpha * *x.add(j * incx), a.add(j * lda), 1, y, incy);
            *y.add(j * incy) += alpha * crate::ddot(j, a.add(j * lda), 1, x, incx);
        }
    } else {
        for j in 0..n {
            crate::daxpy(
                n - j,
                alpha * *x.add(j * incx),
                a.add(j + j * lda),
                1,
                y.add(j * incy),
                incy,
            );
            *y.add(j * incy) += alpha
                * crate::ddot(
                    n - (j + 1),
                    a.add((j + 1) + j * lda),
                    1,
                    x.add((j + 1) * incx),
                    incx,
                );
        }
    }
}

pub unsafe fn dsyr(
    upper: bool,
    n: usize,
    alpha: f64,
    x: *const f64,
    incx: usize,
    a: *mut f64,
    lda: usize,
) {
    if upper {
        for j in 0..n {
            crate::daxpy(j + 1, alpha * *x.add(j * incx), x, incx, a.add(j * lda), 1);
        }
    } else {
        for j in 0..n {
            crate::daxpy(
                n - j,
                alpha * *x.add(j * incx),
                x.add(j * incx),
                incx,
                a.add(j + j * lda),
                1,
            );
        }
    }
}

pub unsafe fn dsyr2(
    upper: bool,
    n: usize,
    alpha: f64,
    x: *const f64,
    incx: usize,
    y: *const f64,
    incy: usize,
    a: *mut f64,
    lda: usize,
) {
    if upper {
        for j in 0..n {
            crate::daxpy(j + 1, alpha * *x.add(j * incx), y, incy, a.add(j * lda), 1);
            crate::daxpy(j + 1, alpha * *y.add(j * incy), x, incx, a.add(j * lda), 1);
        }
    } else {
        for j in 0..n {
            crate::daxpy(
                n - j,
                alpha * *x.add(j * incx),
                y.add(j * incy),
                incy,
                a.add(j + j * lda),
                1,
            );
            crate::daxpy(
                n - j,
                alpha * *y.add(j * incy),
                x.add(j * incx),
                incx,
                a.add(j + j * lda),
                1,
            );
        }
    }
}

pub unsafe fn dger(
    m: usize,
    n: usize,
    alpha: f64,
    x: *const f64,
    incx: usize,
    y: *const f64,
    incy: usize,
    a: *mut f64,
    lda: usize,
) {
    for j in 0..n {
        crate::daxpy(m, alpha * *y.add(j * incy), x, incx, a.add(j * lda), 1);
    }
}
