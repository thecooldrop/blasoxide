pub fn drotg(a: f64, b: f64) -> (f64, f64, f64, f64) {
    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0, 1.0, 0.0);
    }
    let h = a.hypot(b);
    let r = if a.abs() > b.abs() {
        h.copysign(a)
    } else {
        h.copysign(b)
    };
    let c = a / r;
    let s = b / r;
    let z = if a.abs() > b.abs() {
        s
    } else if c != 0.0 {
        1.0 / c
    } else {
        1.0
    };
    (r, z, c, s)
}

pub unsafe fn drot(
    n: usize,
    mut x: *mut f64,
    incx: usize,
    mut y: *mut f64,
    incy: usize,
    c: f64,
    s: f64,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::drot(n, x, incx, y, incy, c, s);
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

pub unsafe fn dswap(n: usize, mut x: *mut f64, incx: usize, mut y: *mut f64, incy: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::dswap(n, x, incx, y, incy);
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

pub unsafe fn dscal(n: usize, a: f64, mut x: *mut f64, incx: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::dscal(n, a, x, incx);
            return;
        }
    }

    for _ in 0..n {
        *x *= a;
        x = x.add(incx);
    }
}

pub unsafe fn dcopy(n: usize, mut x: *const f64, incx: usize, mut y: *mut f64, incy: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::dcopy(n, x, incx, y, incy);
            return;
        }
    }

    for _ in 0..n {
        *y = *x;
        x = x.add(incx);
        y = y.add(incy);
    }
}

pub unsafe fn daxpy(
    n: usize,
    a: f64,
    mut x: *const f64,
    incx: usize,
    mut y: *mut f64,
    incy: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::daxpy(n, a, x, incx, y, incy);
            return;
        }
    }
    for _ in 0..n {
        *y += a * *x;
        x = x.add(incx);
        y = y.add(incy);
    }
}

pub unsafe fn ddot(
    n: usize,
    mut x: *const f64,
    incx: usize,
    mut y: *const f64,
    incy: usize,
) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            return crate::fma::ddot(n, x, incx, y, incy);
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

pub unsafe fn dnrm2(n: usize, mut x: *const f64, incx: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            return crate::fma::dnrm2(n, x, incx);
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

pub unsafe fn dasum(n: usize, mut x: *const f64, incx: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            return crate::fma::dasum(n, x, incx);
        }
    }

    let mut acc = 0.0;
    for _ in 0..n {
        acc += (*x).abs();
        x = x.add(incx);
    }
    acc
}

pub unsafe fn idamax(n: usize, mut x: *const f64, incx: usize) -> usize {
    let mut max = 0.0;
    let mut imax = 0;
    for i in 0..n {
        let xi = (*x).abs();
        if xi > max {
            max = xi;
            imax = i;
        }
        x = x.add(incx);
    }
    imax
}
