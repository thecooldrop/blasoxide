pub unsafe fn sgemv(
    trans: bool,
    m: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    x: *const f32,
    incx: usize,
    beta: f32,
    y: *mut f32,
    incy: usize,
) {
    const MC: usize = 4096;

    if trans {
        crate::sscal(m, beta, y, incy);

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
        alpha: f32,
        a: *const f32,
        lda: usize,
        x: *const f32,
        incx: usize,
        beta: f32,
        y: *mut f32,
        incy: usize,
    ) {
        crate::sscal(m, beta, y, incy);
        for j in 0..n {
            crate::saxpy(m, alpha * *x.add(j * incx), a.add(j * lda), 1, y, incy);
        }
    }

    unsafe fn inner_kernel_trans(
        m: usize,
        n: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        x: *const f32,
        incx: usize,
        y: *mut f32,
        incy: usize,
    ) {
        for j in 0..n {
            *y.add(j * incy) += alpha * crate::sdot(m, a.add(j * lda), 1, x, incx);
        }
    }
}

pub unsafe fn strmv(
    upper: bool,
    trans: bool,
    diag: bool,
    n: usize,
    a: *const f32,
    lda: usize,
    x: *mut f32,
    incx: usize,
) {
    if trans {
        if upper {
            if diag {
                for j in (1..n).rev() {
                    *x.add(j * incx) += crate::sdot(j, a.add(j * lda), 1, x, incx);
                }
            } else {
                for j in (0..n).rev() {
                    *x.add(j * incx) = crate::sdot(j + 1, a.add(j * lda), 1, x, incx);
                }
            }
        } else if diag {
            for j in 0..n - 1 {
                *x.add(j * incx) += crate::sdot(
                    n - (j + 1),
                    a.add((j + 1) + j * lda),
                    1,
                    x.add((j + 1) * incx),
                    incx,
                );
            }
        } else {
            for j in 0..n {
                *x.add(j * incx) = crate::sdot(n - j, a.add(j + j * lda), 1, x.add(j * incx), incx);
            }
        }
    } else if upper {
        if diag {
            for j in 1..n {
                crate::saxpy(j, *x.add(j * incx), a.add(j * lda), 1, x, incx);
            }
        } else {
            for j in 0..n {
                let scal = *x.add(j * incx);
                *x.add(j * incx) = 0.;
                crate::saxpy(j + 1, scal, a.add(j * lda), 1, x, incx);
            }
        }
    } else if diag {
        for j in (0..n - 1).rev() {
            crate::saxpy(
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
            crate::saxpy(n - j, scal, a.add(j + j * lda), 1, x.add(j * incx), incx);
        }
    }
}

pub unsafe fn ssymv(
    upper: bool,
    n: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    x: *const f32,
    incx: usize,
    beta: f32,
    y: *mut f32,
    incy: usize,
) {
    crate::sscal(n, beta, y, incy);
    if upper {
        for j in 0..n {
            crate::saxpy(j + 1, alpha * *x.add(j * incx), a.add(j * lda), 1, y, incy);
            *y.add(j * incy) += alpha * crate::sdot(j, a.add(j * lda), 1, x, incx);
        }
    } else {
        for j in 0..n {
            crate::saxpy(
                n - j,
                alpha * *x.add(j * incx),
                a.add(j + j * lda),
                1,
                y.add(j * incy),
                incy,
            );
            *y.add(j * incy) += alpha
                * crate::sdot(
                    n - (j + 1),
                    a.add((j + 1) + j * lda),
                    1,
                    x.add((j + 1) * incx),
                    incx,
                );
        }
    }
}

pub unsafe fn ssyr(
    upper: bool,
    n: usize,
    alpha: f32,
    x: *const f32,
    incx: usize,
    a: *mut f32,
    lda: usize,
) {
    if upper {
        for j in 0..n {
            crate::saxpy(j + 1, alpha * *x.add(j * incx), x, incx, a.add(j * lda), 1);
        }
    } else {
        for j in 0..n {
            crate::saxpy(
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

pub unsafe fn ssyr2(
    upper: bool,
    n: usize,
    alpha: f32,
    x: *const f32,
    incx: usize,
    y: *const f32,
    incy: usize,
    a: *mut f32,
    lda: usize,
) {
    if upper {
        for j in 0..n {
            crate::saxpy(j + 1, alpha * *x.add(j * incx), y, incy, a.add(j * lda), 1);
            crate::saxpy(j + 1, alpha * *y.add(j * incy), x, incx, a.add(j * lda), 1);
        }
    } else {
        for j in 0..n {
            crate::saxpy(
                n - j,
                alpha * *x.add(j * incx),
                y.add(j * incy),
                incy,
                a.add(j + j * lda),
                1,
            );
            crate::saxpy(
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

pub unsafe fn sger(
    m: usize,
    n: usize,
    alpha: f32,
    x: *const f32,
    incx: usize,
    y: *const f32,
    incy: usize,
    a: *mut f32,
    lda: usize,
) {
    for j in 0..n {
        crate::saxpy(m, alpha * *y.add(j * incy), x, incx, a.add(j * lda), 1);
    }
}

pub unsafe fn strsv(
    upper: bool,
    _trans: bool,
    _diag: bool,
    n: usize,
    a: *const f32,
    lda: usize,
    x: *mut f32,
    incx: usize,
) {
    if upper {
        for j in (0..n).rev() {
            let beta = *x.add(j * incx);
            *x.add(j * incx) = (beta
                - crate::sdot(
                    n - (j + 1),
                    x.add((j + 1) * incx),
                    incx,
                    a.add(j + 1 + (j + 1) * lda),
                    1,
                ))
                / *a.add(j + j * lda);
        }
    }
}
