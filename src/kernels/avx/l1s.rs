use super::fma::{fmadd_ps, fmsub_ps};
use super::hsum::hsum_ps;
use super::intrinsics::*;

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
        let c0 = _mm256_broadcast_ss(&c);
        let s0 = _mm256_broadcast_ss(&s);

        for _ in 0..n / 32 {
            let x0 = _mm256_load_ps(x);
            let y0 = _mm256_load_ps(y);
            let x1 = _mm256_load_ps(x.add(8));
            let y1 = _mm256_load_ps(y.add(8));
            let x2 = _mm256_load_ps(x.add(16));
            let y2 = _mm256_load_ps(y.add(16));
            let x3 = _mm256_load_ps(x.add(24));
            let y3 = _mm256_load_ps(y.add(24));

            _mm256_store_ps(x, fmadd_ps(c0, x0, _mm256_mul_ps(s0, y0)));
            _mm256_store_ps(y, fmsub_ps(c0, y0, _mm256_mul_ps(s0, x0)));
            _mm256_store_ps(x.add(8), fmadd_ps(c0, x1, _mm256_mul_ps(s0, y1)));
            _mm256_store_ps(y.add(8), fmsub_ps(c0, y1, _mm256_mul_ps(s0, x1)));
            _mm256_store_ps(x.add(16), fmadd_ps(c0, x2, _mm256_mul_ps(s0, y2)));
            _mm256_store_ps(y.add(16), fmsub_ps(c0, y2, _mm256_mul_ps(s0, x2)));
            _mm256_store_ps(x.add(24), fmadd_ps(c0, x3, _mm256_mul_ps(s0, y3)));
            _mm256_store_ps(y.add(24), fmsub_ps(c0, y3, _mm256_mul_ps(s0, x3)));

            x = x.add(32);
            y = y.add(32);
        }

        for _ in 0..n % 32 {
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

pub unsafe fn sswap(n: usize, mut x: *mut f32, incx: usize, mut y: *mut f32, incy: usize) {
    if incx == 1 && incy == 1 {
        for _ in 0..n / 32 {
            let x0 = _mm256_load_ps(x);
            let y0 = _mm256_load_ps(y);
            let x1 = _mm256_load_ps(x.add(8));
            let y1 = _mm256_load_ps(y.add(8));
            let x2 = _mm256_load_ps(x.add(16));
            let y2 = _mm256_load_ps(y.add(16));
            let x3 = _mm256_load_ps(x.add(24));
            let y3 = _mm256_load_ps(y.add(24));

            _mm256_store_ps(x, y0);
            _mm256_store_ps(y, x0);
            _mm256_store_ps(x.add(8), y1);
            _mm256_store_ps(y.add(8), x1);
            _mm256_store_ps(x.add(16), y2);
            _mm256_store_ps(y.add(16), x2);
            _mm256_store_ps(x.add(24), y3);
            _mm256_store_ps(y.add(24), x3);

            x = x.add(32);
            y = y.add(32);
        }

        for _ in 0..n % 32 {
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

pub unsafe fn sscal(n: usize, a: f32, mut x: *mut f32, incx: usize) {
    if incx == 1 {
        let a0 = _mm256_broadcast_ss(&a);
        for _ in 0..n / 64 {
            let mut x0 = _mm256_load_ps(x);
            let mut x1 = _mm256_load_ps(x.add(8));
            let mut x2 = _mm256_load_ps(x.add(16));
            let mut x3 = _mm256_load_ps(x.add(24));
            let mut x4 = _mm256_load_ps(x.add(32));
            let mut x5 = _mm256_load_ps(x.add(40));
            let mut x6 = _mm256_load_ps(x.add(48));
            let mut x7 = _mm256_load_ps(x.add(56));

            x0 = _mm256_mul_ps(a0, x0);
            x1 = _mm256_mul_ps(a0, x1);
            x2 = _mm256_mul_ps(a0, x2);
            x3 = _mm256_mul_ps(a0, x3);
            x4 = _mm256_mul_ps(a0, x4);
            x5 = _mm256_mul_ps(a0, x5);
            x6 = _mm256_mul_ps(a0, x6);
            x7 = _mm256_mul_ps(a0, x7);

            _mm256_store_ps(x, x0);
            _mm256_store_ps(x.add(8), x1);
            _mm256_store_ps(x.add(16), x2);
            _mm256_store_ps(x.add(24), x3);
            _mm256_store_ps(x.add(32), x4);
            _mm256_store_ps(x.add(40), x5);
            _mm256_store_ps(x.add(48), x6);
            _mm256_store_ps(x.add(56), x7);

            x = x.add(64);
        }
        for _ in 0..n % 64 {
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

pub unsafe fn scopy(n: usize, mut x: *const f32, incx: usize, mut y: *mut f32, incy: usize) {
    if incx == 1 && incy == 1 {
        for _ in 0..n / 64 {
            let x0 = _mm256_load_ps(x);
            let x1 = _mm256_load_ps(x.add(8));
            let x2 = _mm256_load_ps(x.add(16));
            let x3 = _mm256_load_ps(x.add(24));
            let x4 = _mm256_load_ps(x.add(32));
            let x5 = _mm256_load_ps(x.add(40));
            let x6 = _mm256_load_ps(x.add(48));
            let x7 = _mm256_load_ps(x.add(56));

            _mm256_store_ps(y, x0);
            _mm256_store_ps(y.add(8), x1);
            _mm256_store_ps(y.add(16), x2);
            _mm256_store_ps(y.add(24), x3);
            _mm256_store_ps(y.add(32), x4);
            _mm256_store_ps(y.add(40), x5);
            _mm256_store_ps(y.add(48), x6);
            _mm256_store_ps(y.add(56), x7);

            x = x.add(64);
            y = y.add(64);
        }
        for _ in 0..n % 64 {
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

pub unsafe fn saxpy(
    n: usize,
    a: f32,
    mut x: *const f32,
    incx: usize,
    mut y: *mut f32,
    incy: usize,
) {
    if incx == 1 && incy == 1 {
        let a0 = _mm256_broadcast_ss(&a);
        for _ in 0..n / 32 {
            let x0 = _mm256_load_ps(x);
            let y0 = _mm256_load_ps(y);
            let x1 = _mm256_load_ps(x.add(8));
            let y1 = _mm256_load_ps(y.add(8));
            let x2 = _mm256_load_ps(x.add(16));
            let y2 = _mm256_load_ps(y.add(16));
            let x3 = _mm256_load_ps(x.add(24));
            let y3 = _mm256_load_ps(y.add(24));

            _mm256_store_ps(y, fmadd_ps(a0, x0, y0));
            _mm256_store_ps(y.add(8), fmadd_ps(a0, x1, y1));
            _mm256_store_ps(y.add(16), fmadd_ps(a0, x2, y2));
            _mm256_store_ps(y.add(24), fmadd_ps(a0, x3, y3));

            x = x.add(32);
            y = y.add(32);
        }
        for _ in 0..n % 32 {
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

pub unsafe fn sdot(
    n: usize,
    mut x: *const f32,
    incx: usize,
    mut y: *const f32,
    incy: usize,
) -> f32 {
    if incx == 1 && incy == 1 {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for _ in 0..n / 32 {
            let x0 = _mm256_load_ps(x);
            let y0 = _mm256_load_ps(y);
            let x1 = _mm256_load_ps(x.add(8));
            let y1 = _mm256_load_ps(y.add(8));
            let x2 = _mm256_load_ps(x.add(16));
            let y2 = _mm256_load_ps(y.add(16));
            let x3 = _mm256_load_ps(x.add(24));
            let y3 = _mm256_load_ps(y.add(24));

            acc0 = fmadd_ps(x0, y0, acc0);
            acc1 = fmadd_ps(x1, y1, acc1);
            acc2 = fmadd_ps(x2, y2, acc2);
            acc3 = fmadd_ps(x3, y3, acc3);

            x = x.add(32);
            y = y.add(32);
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        let mut acc = hsum_ps(acc0);
        for _ in 0..n % 32 {
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

pub unsafe fn snrm2(n: usize, mut x: *const f32, incx: usize) -> f32 {
    if incx == 1 {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for _ in 0..n / 32 {
            let x0 = _mm256_load_ps(x);
            let x1 = _mm256_load_ps(x.add(8));
            let x2 = _mm256_load_ps(x.add(16));
            let x3 = _mm256_load_ps(x.add(24));

            acc0 = fmadd_ps(x0, x0, acc0);
            acc1 = fmadd_ps(x1, x1, acc1);
            acc2 = fmadd_ps(x2, x2, acc2);
            acc3 = fmadd_ps(x3, x3, acc3);

            x = x.add(32);
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        let mut acc = hsum_ps(acc0);
        for _ in 0..n % 32 {
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

pub unsafe fn sasum(n: usize, mut x: *const f32, incx: usize) -> f32 {
    if incx == 1 {
        let mask = _mm256_broadcast_ss(&f32::from_bits(0x7FFF_FFFF));

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut acc4 = _mm256_setzero_ps();
        let mut acc5 = _mm256_setzero_ps();
        let mut acc6 = _mm256_setzero_ps();
        let mut acc7 = _mm256_setzero_ps();
        for _ in 0..n / 64 {
            let mut x0 = _mm256_load_ps(x);
            let mut x1 = _mm256_load_ps(x.add(8));
            let mut x2 = _mm256_load_ps(x.add(16));
            let mut x3 = _mm256_load_ps(x.add(24));
            let mut x4 = _mm256_load_ps(x.add(32));
            let mut x5 = _mm256_load_ps(x.add(40));
            let mut x6 = _mm256_load_ps(x.add(48));
            let mut x7 = _mm256_load_ps(x.add(56));

            x0 = _mm256_and_ps(mask, x0);
            x1 = _mm256_and_ps(mask, x1);
            x2 = _mm256_and_ps(mask, x2);
            x3 = _mm256_and_ps(mask, x3);
            x4 = _mm256_and_ps(mask, x4);
            x5 = _mm256_and_ps(mask, x5);
            x6 = _mm256_and_ps(mask, x6);
            x7 = _mm256_and_ps(mask, x7);

            acc0 = _mm256_add_ps(acc0, x0);
            acc1 = _mm256_add_ps(acc1, x1);
            acc2 = _mm256_add_ps(acc2, x2);
            acc3 = _mm256_add_ps(acc3, x3);
            acc4 = _mm256_add_ps(acc4, x4);
            acc5 = _mm256_add_ps(acc5, x5);
            acc6 = _mm256_add_ps(acc6, x6);
            acc7 = _mm256_add_ps(acc7, x7);

            x = x.add(64);
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc4 = _mm256_add_ps(acc4, acc5);
        acc6 = _mm256_add_ps(acc6, acc7);

        acc0 = _mm256_add_ps(acc0, acc2);
        acc4 = _mm256_add_ps(acc4, acc6);

        acc0 = _mm256_add_ps(acc0, acc4);

        let mut acc = hsum_ps(acc0);
        for _ in 0..n % 64 {
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
