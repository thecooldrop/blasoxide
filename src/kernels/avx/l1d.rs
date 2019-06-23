use super::fma::{fmadd_pd, fmsub_pd};
use super::hsum::hsum_pd;
use super::intrinsics::*;

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
        let c0 = _mm256_broadcast_sd(&c);
        let s0 = _mm256_broadcast_sd(&s);

        for _ in 0..n / 16 {
            let x0 = _mm256_load_pd(x);
            let y0 = _mm256_load_pd(y);
            let x1 = _mm256_load_pd(x.add(4));
            let y1 = _mm256_load_pd(y.add(4));
            let x2 = _mm256_load_pd(x.add(8));
            let y2 = _mm256_load_pd(y.add(8));
            let x3 = _mm256_load_pd(x.add(12));
            let y3 = _mm256_load_pd(y.add(12));

            _mm256_store_pd(x, fmadd_pd(c0, x0, _mm256_mul_pd(s0, y0)));
            _mm256_store_pd(y, fmsub_pd(c0, y0, _mm256_mul_pd(s0, x0)));
            _mm256_store_pd(x.add(4), fmadd_pd(c0, x1, _mm256_mul_pd(s0, y1)));
            _mm256_store_pd(y.add(4), fmsub_pd(c0, y1, _mm256_mul_pd(s0, x1)));
            _mm256_store_pd(x.add(8), fmadd_pd(c0, x2, _mm256_mul_pd(s0, y2)));
            _mm256_store_pd(y.add(8), fmsub_pd(c0, y2, _mm256_mul_pd(s0, x2)));
            _mm256_store_pd(x.add(12), fmadd_pd(c0, x3, _mm256_mul_pd(s0, y3)));
            _mm256_store_pd(y.add(12), fmsub_pd(c0, y3, _mm256_mul_pd(s0, x3)));

            x = x.add(16);
            y = y.add(16);
        }

        for _ in 0..n % 16 {
            let x0 = *x;
            let y0 = *y;

            *x = c * x0 + s * y0;
            *y = c * y0 - s * x0;

            x = x.add(1);
            y = y.add(1);
        }
    } else {
        for _ in 0..n {
            let x0 = *x;
            let y0 = *y;

            *x = c * x0 + s * y0;
            *y = c * y0 - s * x0;

            x = x.add(incx);
            y = y.add(incy);
        }
    }
}

pub unsafe fn dswap(n: usize, mut x: *mut f64, incx: usize, mut y: *mut f64, incy: usize) {
    if incx == 1 && incy == 1 {
        for _ in 0..n / 16 {
            let x0 = _mm256_load_pd(x);
            let y0 = _mm256_load_pd(y);
            let x1 = _mm256_load_pd(x.add(4));
            let y1 = _mm256_load_pd(y.add(4));
            let x2 = _mm256_load_pd(x.add(8));
            let y2 = _mm256_load_pd(y.add(8));
            let x3 = _mm256_load_pd(x.add(12));
            let y3 = _mm256_load_pd(y.add(12));

            _mm256_store_pd(x, y0);
            _mm256_store_pd(y, x0);
            _mm256_store_pd(x.add(4), y1);
            _mm256_store_pd(y.add(4), x1);
            _mm256_store_pd(x.add(8), y2);
            _mm256_store_pd(y.add(8), x2);
            _mm256_store_pd(x.add(12), y3);
            _mm256_store_pd(y.add(12), x3);

            x = x.add(16);
            y = y.add(16);
        }

        for _ in 0..n % 16 {
            let x0 = *x;

            *x = *y;
            *y = x0;

            x = x.add(1);
            y = y.add(1);
        }
    } else {
        for _ in 0..n {
            let x0 = *x;

            *x = *y;
            *y = x0;

            x = x.add(incx);
            y = y.add(incy);
        }
    }
}

pub unsafe fn dscal(n: usize, a: f64, mut x: *mut f64, incx: usize) {
    if incx == 1 {
        let a0 = _mm256_broadcast_sd(&a);
        for _ in 0..n / 32 {
            let mut x0 = _mm256_load_pd(x);
            let mut x1 = _mm256_load_pd(x.add(4));
            let mut x2 = _mm256_load_pd(x.add(8));
            let mut x3 = _mm256_load_pd(x.add(12));
            let mut x4 = _mm256_load_pd(x.add(16));
            let mut x5 = _mm256_load_pd(x.add(20));
            let mut x6 = _mm256_load_pd(x.add(24));
            let mut x7 = _mm256_load_pd(x.add(28));

            x0 = _mm256_mul_pd(a0, x0);
            x1 = _mm256_mul_pd(a0, x1);
            x2 = _mm256_mul_pd(a0, x2);
            x3 = _mm256_mul_pd(a0, x3);
            x4 = _mm256_mul_pd(a0, x4);
            x5 = _mm256_mul_pd(a0, x5);
            x6 = _mm256_mul_pd(a0, x6);
            x7 = _mm256_mul_pd(a0, x7);

            _mm256_store_pd(x, x0);
            _mm256_store_pd(x.add(4), x1);
            _mm256_store_pd(x.add(8), x2);
            _mm256_store_pd(x.add(12), x3);
            _mm256_store_pd(x.add(16), x4);
            _mm256_store_pd(x.add(20), x5);
            _mm256_store_pd(x.add(24), x6);
            _mm256_store_pd(x.add(28), x7);

            x = x.add(32);
        }
        for _ in 0..n % 32 {
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
        for _ in 0..n / 32 {
            let x0 = _mm256_load_pd(x);
            let x1 = _mm256_load_pd(x.add(4));
            let x2 = _mm256_load_pd(x.add(8));
            let x3 = _mm256_load_pd(x.add(12));
            let x4 = _mm256_load_pd(x.add(16));
            let x5 = _mm256_load_pd(x.add(20));
            let x6 = _mm256_load_pd(x.add(24));
            let x7 = _mm256_load_pd(x.add(28));

            _mm256_store_pd(y, x0);
            _mm256_store_pd(y.add(4), x1);
            _mm256_store_pd(y.add(8), x2);
            _mm256_store_pd(y.add(12), x3);
            _mm256_store_pd(y.add(16), x4);
            _mm256_store_pd(y.add(20), x5);
            _mm256_store_pd(y.add(24), x6);
            _mm256_store_pd(y.add(28), x7);

            x = x.add(32);
            y = y.add(32);
        }
        for _ in 0..n % 32 {
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
        let a0 = _mm256_broadcast_sd(&a);
        for _ in 0..n / 16 {
            let x0 = _mm256_load_pd(x);
            let y0 = _mm256_load_pd(y);
            let x1 = _mm256_load_pd(x.add(4));
            let y1 = _mm256_load_pd(y.add(4));
            let x2 = _mm256_load_pd(x.add(8));
            let y2 = _mm256_load_pd(y.add(8));
            let x3 = _mm256_load_pd(x.add(12));
            let y3 = _mm256_load_pd(y.add(12));

            _mm256_store_pd(y, fmadd_pd(a0, x0, y0));
            _mm256_store_pd(y.add(4), fmadd_pd(a0, x1, y1));
            _mm256_store_pd(y.add(8), fmadd_pd(a0, x2, y2));
            _mm256_store_pd(y.add(12), fmadd_pd(a0, x3, y3));

            x = x.add(16);
            y = y.add(16);
        }
        for _ in 0..n % 16 {
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
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut acc2 = _mm256_setzero_pd();
        let mut acc3 = _mm256_setzero_pd();
        for _ in 0..n / 16 {
            let x0 = _mm256_load_pd(x);
            let y0 = _mm256_load_pd(y);
            let x1 = _mm256_load_pd(x.add(4));
            let y1 = _mm256_load_pd(y.add(4));
            let x2 = _mm256_load_pd(x.add(8));
            let y2 = _mm256_load_pd(y.add(8));
            let x3 = _mm256_load_pd(x.add(12));
            let y3 = _mm256_load_pd(y.add(12));

            acc0 = fmadd_pd(x0, y0, acc0);
            acc1 = fmadd_pd(x1, y1, acc1);
            acc2 = fmadd_pd(x2, y2, acc2);
            acc3 = fmadd_pd(x3, y3, acc3);

            x = x.add(16);
            y = y.add(16);
        }
        acc0 = _mm256_add_pd(acc0, acc1);
        acc2 = _mm256_add_pd(acc2, acc3);
        acc0 = _mm256_add_pd(acc0, acc2);

        let mut acc = hsum_pd(acc0);
        for _ in 0..n % 16 {
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
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut acc2 = _mm256_setzero_pd();
        let mut acc3 = _mm256_setzero_pd();
        for _ in 0..n / 16 {
            let x0 = _mm256_load_pd(x);
            let x1 = _mm256_load_pd(x.add(4));
            let x2 = _mm256_load_pd(x.add(8));
            let x3 = _mm256_load_pd(x.add(12));

            acc0 = fmadd_pd(x0, x0, acc0);
            acc1 = fmadd_pd(x1, x1, acc1);
            acc2 = fmadd_pd(x2, x2, acc2);
            acc3 = fmadd_pd(x3, x3, acc3);

            x = x.add(16);
        }
        acc0 = _mm256_add_pd(acc0, acc1);
        acc2 = _mm256_add_pd(acc2, acc3);
        acc0 = _mm256_add_pd(acc0, acc2);

        let mut acc = hsum_pd(acc0);
        for _ in 0..n % 16 {
            let x0 = *x;
            acc += x0 * x0;
            x = x.add(1);
        }
        acc.sqrt()
    } else {
        let mut acc = 0.0;
        for _ in 0..n {
            let x0 = *x;
            acc += x0 * x0;
            x = x.add(incx);
        }
        acc.sqrt()
    }
}

pub unsafe fn dasum(n: usize, mut x: *const f64, incx: usize) -> f64 {
    if incx == 1 {
        let mask = _mm256_broadcast_sd(&f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));

        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut acc2 = _mm256_setzero_pd();
        let mut acc3 = _mm256_setzero_pd();
        let mut acc4 = _mm256_setzero_pd();
        let mut acc5 = _mm256_setzero_pd();
        let mut acc6 = _mm256_setzero_pd();
        let mut acc7 = _mm256_setzero_pd();
        for _ in 0..n / 32 {
            let mut x0 = _mm256_load_pd(x);
            let mut x1 = _mm256_load_pd(x.add(4));
            let mut x2 = _mm256_load_pd(x.add(8));
            let mut x3 = _mm256_load_pd(x.add(12));
            let mut x4 = _mm256_load_pd(x.add(16));
            let mut x5 = _mm256_load_pd(x.add(20));
            let mut x6 = _mm256_load_pd(x.add(24));
            let mut x7 = _mm256_load_pd(x.add(28));

            x0 = _mm256_and_pd(mask, x0);
            x1 = _mm256_and_pd(mask, x1);
            x2 = _mm256_and_pd(mask, x2);
            x3 = _mm256_and_pd(mask, x3);
            x4 = _mm256_and_pd(mask, x4);
            x5 = _mm256_and_pd(mask, x5);
            x6 = _mm256_and_pd(mask, x6);
            x7 = _mm256_and_pd(mask, x7);

            acc0 = _mm256_add_pd(acc0, x0);
            acc1 = _mm256_add_pd(acc1, x1);
            acc2 = _mm256_add_pd(acc2, x2);
            acc3 = _mm256_add_pd(acc3, x3);
            acc4 = _mm256_add_pd(acc4, x4);
            acc5 = _mm256_add_pd(acc5, x5);
            acc6 = _mm256_add_pd(acc6, x6);
            acc7 = _mm256_add_pd(acc7, x7);

            x = x.add(32);
        }
        acc0 = _mm256_add_pd(acc0, acc1);
        acc2 = _mm256_add_pd(acc2, acc3);
        acc4 = _mm256_add_pd(acc4, acc5);
        acc6 = _mm256_add_pd(acc6, acc7);

        acc0 = _mm256_add_pd(acc0, acc2);
        acc4 = _mm256_add_pd(acc4, acc6);

        acc0 = _mm256_add_pd(acc0, acc4);

        let mut acc = hsum_pd(acc0);
        for _ in 0..n % 32 {
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
