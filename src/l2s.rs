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
                for j in 1..n {
                    *x.add(j * incx) += crate::sdot(j - 1, a.add(j * lda), 1, x, incx);
                }
            } else {
                for j in 0..n {
                    *x.add(j * incx) = crate::sdot(j, a.add(j * lda), 1, x, incx);
                }
            }
        } else if diag {
            for j in (0..n - 1).rev() {
                *x.add(j * incx) += crate::sdot(
                    n - (j + 1),
                    a.add((j + 1) + j * lda),
                    1,
                    x.add((j + 1) * incx),
                    incx,
                );
            }
        } else {
            for j in (0..n).rev() {
                *x.add(j * incx) = crate::sdot(n - j, a.add(j + j * lda), 1, x.add(j * incx), incx);
            }
        }
    } else if upper {
        if diag {
            for j in 1..n {
                crate::saxpy(j - 1, *x.add(j * incx) - 1.0, a.add(j * lda), 1, x, incx);
            }
        } else {
            for j in 0..n {
                crate::saxpy(j, *x.add(j * incx) - 1.0, a.add(j * lda), 1, x, incx);
            }
        }
    } else if diag {
        for j in (0..n - 1).rev() {
            crate::saxpy(
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
            crate::saxpy(
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
            crate::saxpy(j, alpha * *x.add(j * incx), a.add(j * lda), 1, y, incy);
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
            *y.add(j * incy) +=
                alpha * crate::sdot(n - j, a.add(j + j * lda), 1, x.add(j * incx), incx);
        }
    }
}
