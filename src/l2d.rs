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
                for j in 1..n {
                    *x.add(j * incx) += crate::ddot(j - 1, a.add(j * lda), 1, x, incx);
                }
            } else {
                for j in 0..n {
                    *x.add(j * incx) = crate::ddot(j, a.add(j * lda), 1, x, incx);
                }
            }
        } else if diag {
            for j in (0..n - 1).rev() {
                *x.add(j * incx) += crate::ddot(
                    n - (j + 1),
                    a.add((j + 1) + j * lda),
                    1,
                    x.add((j + 1) * incx),
                    incx,
                );
            }
        } else {
            for j in (0..n).rev() {
                *x.add(j * incx) = crate::ddot(n - j, a.add(j + j * lda), 1, x.add(j * incx), incx);
            }
        }
    } else if upper {
        if diag {
            for j in 1..n {
                crate::daxpy(j - 1, *x.add(j * incx) - 1.0, a.add(j * lda), 1, x, incx);
            }
        } else {
            for j in 0..n {
                crate::daxpy(j, *x.add(j * incx) - 1.0, a.add(j * lda), 1, x, incx);
            }
        }
    } else if diag {
        for j in (0..n - 1).rev() {
            crate::daxpy(
                n - (j + 1),
                *x.add(j * incx) - 1.0,
                a.add((j + 1) + j * lda),
                1,
                x.add((j + 1) * incx),
                incx,
            );
        }
    } else {
        for j in (0..n).rev() {
            crate::daxpy(
                n - j,
                *x.add(j * incx) - 1.0,
                a.add(j + j * lda),
                1,
                x.add(j * incx),
                incx,
            );
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
            crate::daxpy(j, alpha * *x.add(j * incx), a.add(j * lda), 1, y, incy);
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
            *y.add(j * incy) +=
                alpha * crate::ddot(n - j, a.add(j + j * lda), 1, x.add(j * incx), incx);
        }
    }
}
