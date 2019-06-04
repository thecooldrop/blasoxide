use core::arch::x86_64::*;

macro_rules! unroll4 {
    ($e:expr) => {{
        $e;
        $e;
        $e;
        $e;
    }};
}

#[inline(always)]
pub unsafe fn hadd_ps(mut v: __m256) -> f32 {
    v = _mm256_hadd_ps(v, v);
    v = _mm256_hadd_ps(v, v);
    let vhigh = _mm256_extractf128_ps(v, 1);
    let vlow = _mm256_castps256_ps128(v);
    let vsum = _mm_add_ps(vhigh, vlow);
	_mm_cvtss_f32(vsum)
}

pub static SABS_MASK: u32 = 0x7FFF_FFFF;

#[inline(always)]
pub unsafe fn hadd_pd(v: __m256d) -> f64 {
	let vhigh = _mm256_extractf128_pd(v, 1);
	let vlow = _mm256_castpd256_pd128(v);
	let vsum = _mm_add_pd(vlow, vhigh);
	let h64 = _mm_unpackhi_pd(vsum, vsum);
	_mm_cvtsd_f64(_mm_add_sd(vsum, h64))
}

pub static DABS_MASK: u64 = 0x7FFF_FFFF_FFFF_FFFF;