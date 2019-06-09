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
    const MC: usize = 2048;

    for i in (0..m).step_by(MC) {
        let ib = std::cmp::min(m-i, MC);
        inner_kernel(ib, n, alpha, a.add(i), lda, x, incx, beta, y.add(i * incy), incy);
    }

    unsafe fn inner_kernel(m: usize, n: usize, alpha: f64, a: *const f64, lda: usize, x: *const f64, incx: usize, beta: f64, y: *mut f64, incy: usize) {
        crate::dscal(m, beta, y, incy);
        for j in 0..n {
            crate::daxpy(m, alpha * *x.add(j * incx), a.add(j * lda), 1, y, incy);
        }
    }
}
