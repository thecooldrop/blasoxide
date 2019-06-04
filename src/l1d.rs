use crate::common::{hadd_pd, DABS_MASK};
use core::arch::x86_64::*;

const STEP: usize = 4 * 4;

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
    if incx == 1 && incy == 1 {
        let cv = _mm256_broadcast_sd(&c);
        let sv = _mm256_broadcast_sd(&s);

        for _ in 0..n / STEP {
            unroll4!({
                let xv = _mm256_loadu_pd(x);
                let yv = _mm256_loadu_pd(y);

                _mm256_storeu_pd(x, _mm256_fmadd_pd(cv, xv, _mm256_mul_pd(sv, yv)));
                _mm256_storeu_pd(y, _mm256_fmsub_pd(cv, yv, _mm256_mul_pd(sv, xv)));

                x = x.add(8);
                y = y.add(8);
            });
        }
        for _ in 0..n % STEP {
            let xi = *x;
            let yi = *y;

            *x = c * xi + s * yi;
            *y = c * yi - s * xi;

            x = x.add(1);
            y = y.add(1);
        }
    } else {
        for _ in 0..n {
            let xi = *x;
            let yi = *y;

            *x = c * xi + s * yi;
            *y = c * yi - s * xi;

            x = x.add(incx);
            y = y.add(incy);
        }
    }
}

pub unsafe fn dswap(n: usize, mut x: *mut f64, incx: usize, mut y: *mut f64, incy: usize) {
    if incx == 1 && incy == 1 {
        for _ in 0..n / STEP {
            unroll4!({
                let xv = _mm256_loadu_pd(x);
                let yv = _mm256_loadu_pd(y);
                _mm256_storeu_pd(x, yv);
                _mm256_storeu_pd(y, xv);
                x = x.add(8);
                y = y.add(8);
            });
        }
        for _ in 0..n % STEP {
            let xi = *x;
            let yi = *y;

            *x = yi;
            *y = xi;

            x = x.add(1);
            y = y.add(1);
        }
    } else {
        for _ in 0..n {
            let xi = *x;
            let yi = *y;

            *x = yi;
            *y = xi;

            x = x.add(incx);
            y = y.add(incy);
        }
    }
}

pub unsafe fn dscal(n: usize, a: f64, mut x: *mut f64, incx: usize) {
    if incx == 1 {
        let av = _mm256_broadcast_sd(&a);
        for _ in 0..n / STEP {
            unroll4!({
                _mm256_storeu_pd(x, _mm256_mul_pd(av, _mm256_loadu_pd(x)));
                x = x.add(8);
            });
        }
        for _ in 0..n % STEP {
            *x *= a;
            x = x.add(1);
        }
    } else {
        for _ in 0..n {
            *x *= a;
            x = x.add(incx);
        }
    }
}

pub unsafe fn dcopy(n: usize, mut x: *const f64, incx: usize, mut y: *mut f64, incy: usize) {
    if incx == 1 && incy == 1 {
        for _ in 0..n / STEP {
            unroll4!({
                _mm256_storeu_pd(y, _mm256_loadu_pd(x));
                x = x.add(8);
                y = y.add(8);
            });
        }
        for _ in 0..n % STEP {
            *y = *x;
            x = x.add(1);
            y = y.add(1);
        }
    } else {
        for _ in 0..n {
            *y = *x;
            x = x.add(incx);
            y = y.add(incy);
        }
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
    if incx == 1 && incy == 1 {
        let av = _mm256_broadcast_sd(&a);
        for _ in 0..n / STEP {
            unroll4!({
                _mm256_storeu_pd(
                    y,
                    _mm256_fmadd_pd(av, _mm256_loadu_pd(x), _mm256_loadu_pd(y)),
                );
                x = x.add(8);
                y = y.add(8);
            });
        }
        for _ in 0..n % STEP {
            *y += a * *x;
            x = x.add(1);
            y = y.add(1);
        }
    } else {
        for _ in 0..n {
            *y += a * *x;
            x = x.add(incx);
            y = y.add(incy);
        }
    }
}

pub unsafe fn ddot(
    n: usize,
    mut x: *const f64,
    incx: usize,
    mut y: *const f64,
    incy: usize,
) -> f64 {
    if incx == 1 && incy == 1 {
        let mut acc = _mm256_setzero_pd();
        for _ in 0..n / STEP {
            unroll4!({
                acc = _mm256_fmadd_pd(_mm256_loadu_pd(x), _mm256_loadu_pd(y), acc);
                x = x.add(8);
                y = y.add(8);
            });
        }
        let mut acc = hadd_pd(acc);
        for _ in 0..n % STEP {
            acc += *x * *y;
            x = x.add(1);
            y = y.add(1);
        }
        acc
    } else {
        let mut acc = 0.0;
        for _ in 0..n {
            acc += *x * *y;
            x = x.add(incx);
            y = y.add(incy);
        }
        acc
    }
}

pub unsafe fn dnrm2(n: usize, mut x: *const f64, incx: usize) -> f64 {
    if incx == 1 {
        let mut acc = _mm256_setzero_pd();
        for _ in 0..n / STEP {
            unroll4!({
                let xv = _mm256_loadu_pd(x);
                acc = _mm256_fmadd_pd(xv, xv, acc);
                x = x.add(8);
            });
        }
        let mut acc = hadd_pd(acc);
        for _ in 0..n % STEP {
            let xi = *x;
            acc += xi * xi;
            x = x.add(1);
        }
        acc.sqrt()
    } else {
        let mut acc = 0.0;
        for _ in 0..n {
            let xi = *x;
            acc += xi * xi;
            x = x.add(incx);
        }
        acc.sqrt()
    }
}

pub unsafe fn dasum(n: usize, mut x: *const f64, incx: usize) -> f64 {
    if incx == 1 {
        let mut acc = _mm256_setzero_pd();
        let mask = _mm256_broadcast_sd(&*(&DABS_MASK as *const u64 as *const f64));
        for _ in 0..n / STEP {
            unroll4!({
                acc = _mm256_add_pd(_mm256_and_pd(mask, _mm256_loadu_pd(x)), acc);
                x = x.add(8);
            });
        }
        let mut acc = hadd_pd(acc);
        for _ in 0..n % STEP {
            acc += (*x).abs();
            x = x.add(1);
        }
        acc
    } else {
        let mut acc = 0.0;
        for _ in 0..n {
            acc += (*x).abs();
            x = x.add(incx);
        }
        acc
    }
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
