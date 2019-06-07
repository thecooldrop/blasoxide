use super::common::{hadd_ps, SABS_MASK};
use core::arch::x86_64::*;

const STEP: usize = 8 * 4;

pub fn srotg(a: f32, b: f32) -> (f32, f32, f32, f32) {
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

#[target_feature(enable = "fma")]
pub unsafe fn srot(
    n: usize,
    mut x: *mut f32,
    incx: usize,
    mut y: *mut f32,
    incy: usize,
    c: f32,
    s: f32,
) {
    if incx == 1 && incy == 1 {
        let cv = _mm256_broadcast_ss(&c);
        let sv = _mm256_broadcast_ss(&s);

        for _ in 0..n / STEP {
            unroll4!({
                let xv = _mm256_loadu_ps(x);
                let yv = _mm256_loadu_ps(y);

                _mm256_storeu_ps(x, _mm256_fmadd_ps(cv, xv, _mm256_mul_ps(sv, yv)));
                _mm256_storeu_ps(y, _mm256_fmsub_ps(cv, yv, _mm256_mul_ps(sv, xv)));

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

#[target_feature(enable = "fma")]
pub unsafe fn sswap(n: usize, mut x: *mut f32, incx: usize, mut y: *mut f32, incy: usize) {
    if incx == 1 && incy == 1 {
        for _ in 0..n / STEP {
            unroll4!({
                let xv = _mm256_loadu_ps(x);
                let yv = _mm256_loadu_ps(y);
                _mm256_storeu_ps(x, yv);
                _mm256_storeu_ps(y, xv);
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

#[target_feature(enable = "fma")]
pub unsafe fn sscal(n: usize, a: f32, mut x: *mut f32, incx: usize) {
    if incx == 1 {
        let av = _mm256_broadcast_ss(&a);
        for _ in 0..n / STEP {
            unroll4!({
                _mm256_storeu_ps(x, _mm256_mul_ps(av, _mm256_loadu_ps(x)));
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

#[target_feature(enable = "fma")]
pub unsafe fn scopy(n: usize, mut x: *const f32, incx: usize, mut y: *mut f32, incy: usize) {
    if incx == 1 && incy == 1 {
        for _ in 0..n / STEP {
            unroll4!({
                _mm256_storeu_ps(y, _mm256_loadu_ps(x));
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

#[target_feature(enable = "fma")]
pub unsafe fn saxpy(
    n: usize,
    a: f32,
    mut x: *const f32,
    incx: usize,
    mut y: *mut f32,
    incy: usize,
) {
    if incx == 1 && incy == 1 {
        let av = _mm256_broadcast_ss(&a);
        for _ in 0..n / STEP {
            unroll4!({
                _mm256_storeu_ps(
                    y,
                    _mm256_fmadd_ps(av, _mm256_loadu_ps(x), _mm256_loadu_ps(y)),
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

#[target_feature(enable = "fma")]
pub unsafe fn sdot(
    n: usize,
    mut x: *const f32,
    incx: usize,
    mut y: *const f32,
    incy: usize,
) -> f32 {
    if incx == 1 && incy == 1 {
        let mut acc = _mm256_setzero_ps();
        for _ in 0..n / STEP {
            unroll4!({
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y), acc);
                x = x.add(8);
                y = y.add(8);
            });
        }
        let mut acc = hadd_ps(acc);
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

pub unsafe fn sdsdot(
    n: usize,
    b: f32,
    mut x: *const f32,
    incx: usize,
    mut y: *const f32,
    incy: usize,
) -> f32 {
    let mut acc: f64 = f64::from(b);
    for _ in 0..n {
        acc += f64::from(*x) * f64::from(*y);
        x = x.add(incx);
        y = y.add(incy);
    }
    acc as f32
}

#[target_feature(enable = "fma")]
pub unsafe fn snrm2(n: usize, mut x: *const f32, incx: usize) -> f32 {
    if incx == 1 {
        let mut acc = _mm256_setzero_ps();
        for _ in 0..n / STEP {
            unroll4!({
                let xv = _mm256_loadu_ps(x);
                acc = _mm256_fmadd_ps(xv, xv, acc);
                x = x.add(8);
            });
        }
        let mut acc = hadd_ps(acc);
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

#[target_feature(enable = "fma")]
pub unsafe fn sasum(n: usize, mut x: *const f32, incx: usize) -> f32 {
    if incx == 1 {
        let mut acc = _mm256_setzero_ps();
        let mask = _mm256_broadcast_ss(&*(&SABS_MASK as *const u32 as *const f32));
        for _ in 0..n / STEP {
            unroll4!({
                acc = _mm256_add_ps(_mm256_and_ps(mask, _mm256_loadu_ps(x)), acc);
                x = x.add(8);
            });
        }
        let mut acc = hadd_ps(acc);
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

pub unsafe fn isamax(n: usize, mut x: *const f32, incx: usize) -> usize {
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
