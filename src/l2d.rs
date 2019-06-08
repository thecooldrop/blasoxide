pub unsafe fn dgemv(
    _trans: bool,
    m: usize,
    n: usize,
    alpha: f64,
    mut a: *const f64,
    lda: usize,
    mut x: *const f64,
    incx: usize,
    beta: f64,
    y: *mut f64,
    incy: usize,
) {
    if incy == 1 {
        for _ in 0..n / 4 {
            crate::dcombine_4(m, alpha, a, lda, x, incx, beta, y);

            x = x.add(4 * incx);
            a = a.add(4 * lda);
        }

        for _ in 0..n % 4 {
            crate::dcombine_1(m, alpha, a, x, beta, y);

            x = x.add(incx);
            a = a.add(lda);
        }
    } else {

    }
}
