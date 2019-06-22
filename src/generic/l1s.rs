pub unsafe fn srot(
    n: usize,
    mut x: *mut f32,
    incx: usize,
    mut y: *mut f32,
    incy: usize,
    c: f32,
    s: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::srot(n, x, incx, y, incy, c, s);
            return;
        }
    }

    for _ in 0..n {
        let xi = *x;
        let yi = *y;

        *x = c * xi + s * yi;
        *y = c * yi - s * xi;

        x = x.add(incx);
        y = y.add(incy);
    }
}

pub unsafe fn sswap(n: usize, mut x: *mut f32, incx: usize, mut y: *mut f32, incy: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::sswap(n, x, incx, y, incy);
            return;
        }
    }

    for _ in 0..n {
        let xi = *x;
        let yi = *y;

        *x = yi;
        *y = xi;

        x = x.add(incx);
        y = y.add(incy);
    }
}

pub unsafe fn sscal(n: usize, a: f32, mut x: *mut f32, incx: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::sscal(n, a, x, incx);
            return;
        }
    }

    for _ in 0..n {
        *x *= a;
        x = x.add(incx);
    }
}

pub unsafe fn scopy(n: usize, mut x: *const f32, incx: usize, mut y: *mut f32, incy: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::scopy(n, x, incx, y, incy);
            return;
        }
    }

    for _ in 0..n {
        *y = *x;
        x = x.add(incx);
        y = y.add(incy);
    }
}

pub unsafe fn saxpy(
    n: usize,
    a: f32,
    mut x: *const f32,
    incx: usize,
    mut y: *mut f32,
    incy: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::saxpy(n, a, x, incx, y, incy);
            return;
        }
    }
    for _ in 0..n {
        *y += a * *x;
        x = x.add(incx);
        y = y.add(incy);
    }
}

pub unsafe fn sdot(
    n: usize,
    mut x: *const f32,
    incx: usize,
    mut y: *const f32,
    incy: usize,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            return crate::fma::sdot(n, x, incx, y, incy);
        }
    }

    let mut acc = 0.0;
    for _ in 0..n {
        acc += *x * *y;
        x = x.add(incx);
        y = y.add(incy);
    }
    acc
}

pub unsafe fn snrm2(n: usize, mut x: *const f32, incx: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            return crate::fma::snrm2(n, x, incx);
        }
    }

    let mut acc = 0.0;
    for _ in 0..n {
        let xi = *x;
        acc += xi * xi;
        x = x.add(incx);
    }
    acc.sqrt()
}

pub unsafe fn sasum(n: usize, mut x: *const f32, incx: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            return crate::fma::sasum(n, x, incx);
        }
    }

    let mut acc = 0.0;
    for _ in 0..n {
        acc += (*x).abs();
        x = x.add(incx);
    }
    acc
}
