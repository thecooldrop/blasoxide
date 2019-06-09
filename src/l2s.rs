pub unsafe fn sgemv(
    _trans: bool,
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

    for i in (0..m).step_by(MC) {
        let ib = std::cmp::min(m-i, MC);
        inner_kernel(ib, n, alpha, a.add(i), lda, x, incx, beta, y.add(i * incy), incy);
    }

    unsafe fn inner_kernel(m: usize, n: usize, alpha: f32, a: *const f32, lda: usize, x: *const f32, incx: usize, beta: f32, y: *mut f32, incy: usize) {
        crate::sscal(m, beta, y, incy);
        for j in 0..n {
            crate::saxpy(m, alpha * *x.add(j * incx), a.add(j * lda), 1, y, incy);
        }
    }
}
