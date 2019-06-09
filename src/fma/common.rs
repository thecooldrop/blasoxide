use core::arch::x86_64::*;

macro_rules! unroll4 {
    ($e:expr) => {{
        $e;
        $e;
        $e;
        $e;
    }};
}

#[inline]
#[target_feature(enable = "fma")]
pub unsafe fn hadd_ps(v: __m256) -> f32 {
    let qhigh = _mm256_extractf128_ps(v, 1);
    let qlow = _mm256_castps256_ps128(v);
    let qsum = _mm_add_ps(qhigh, qlow);
    let dhigh = _mm_movehl_ps(qsum, qsum);
    let dlow = qsum;
    let dsum = _mm_add_ps(dhigh, dlow);
    let high = _mm_shuffle_ps(dsum, dsum, 1);
    let low = dsum;
    _mm_cvtss_f32(_mm_add_ss(high, low))
}

pub static SABS_MASK: u32 = 0x7FFF_FFFF;

#[inline]
#[target_feature(enable = "fma")]
pub unsafe fn hadd_pd(v: __m256d) -> f64 {
    let vhigh = _mm256_extractf128_pd(v, 1);
    let vlow = _mm256_castpd256_pd128(v);
    let vsum = _mm_add_pd(vlow, vhigh);
    let h64 = _mm_unpackhi_pd(vsum, vsum);
    _mm_cvtsd_f64(_mm_add_sd(vsum, h64))
}

pub static DABS_MASK: u64 = 0x7FFF_FFFF_FFFF_FFFF;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadd_ps() {
        unsafe {
            let a = [1., 2., 3., 4., 5., 6., 7., 8.];
            assert_eq!(hadd_ps(_mm256_loadu_ps(a.as_ptr())), 36.);
        }
    }

    #[test]
    fn test_hadd_pd() {
        unsafe {
            let a = [1., 2., 3., 4.];
            assert_eq!(hadd_pd(_mm256_loadu_pd(a.as_ptr())), 10.);
        }
    }
}
