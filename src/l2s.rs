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
    if incy == 1 {
        for _ in 0..n / 4 {
            crate::scombine_4(m, alpha, a, lda, x, incx, beta, y);

            x = x.add(4 * incx);
            a = a.add(4 * lda);
        }

        for _ in 0..n % 4 {
            crate::scombine_1(m, alpha, a, x, beta, y);

            x = x.add(incx);
            a = a.add(lda);
        }
    } else {
        for _ in 0..n / 4 {}

        for _ in 0..n % 4 {}
    }
}
