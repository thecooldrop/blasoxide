#![allow(clippy::many_single_char_names)]

use core::arch::x86_64::*;

macro_rules! unroll4 {
    ($e:expr) => {{
        $e;
        $e;
        $e;
        $e;
    }};
}
const STEP: usize = 8 * 4;

pub fn srotg(a: f32, b: f32) -> (f32, f32, f32, f32) {
    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0, 1.0, 0.0);
    }
    let h = a.hypot(b);
    let r = if a.abs() > b.abs() {
        h * a.signum()
    } else {
        h * b.signum()
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

pub unsafe fn srot(
    n: usize,
    mut x: *mut f32,
    incx: isize,
    mut y: *mut f32,
    incy: isize,
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

                x = x.offset(8);
                y = y.offset(8);
            });
        }
        for _ in 0..n % STEP {
            let xi = *x;
            let yi = *y;

            *x = c * xi + s * yi;
            *y = c * yi - s * xi;

            x = x.offset(1);
            y = y.offset(1);
        }
    } else {
        for _ in 0..n {
            let xi = *x;
            let yi = *y;

            *x = c * xi + s * yi;
            *y = c * yi - s * xi;

            x = x.offset(incx);
            y = y.offset(incy);
        }
    }
}
