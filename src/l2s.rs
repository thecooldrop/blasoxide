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
