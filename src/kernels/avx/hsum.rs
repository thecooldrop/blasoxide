use super::intrinsics::*;

#[inline(always)]
pub unsafe fn hsum_ps(v: __m256) -> f32 {
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

#[inline(always)]
pub unsafe fn hsum_pd(v: __m256d) -> f64 {
    let vhigh = _mm256_extractf128_pd(v, 1);
    let vlow = _mm256_castpd256_pd128(v);
    let vsum = _mm_add_pd(vlow, vhigh);
    let h64 = _mm_unpackhi_pd(vsum, vsum);
    _mm_cvtsd_f64(_mm_add_sd(vsum, h64))
}
