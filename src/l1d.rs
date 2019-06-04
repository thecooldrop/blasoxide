use crate::common::{hadd_pd, DABS_MASK};
use core::arch::x86_64::*;

const STEP: usize = 4 * 4;

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
