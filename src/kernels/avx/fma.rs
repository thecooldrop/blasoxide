use super::intrinsics::*;

#[cfg(target_feature = "fma")]
#[inline(always)]
pub unsafe fn fmadd_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_fmadd_ps(a, b, c)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub unsafe fn fmadd_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_add_ps(_mm256_mul_ps(a, b), c)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub unsafe fn fmsub_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_fmsub_ps(a, b, c)
}

#[cfg(not(target_feature = "fma"))]
pub fn fmsub_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_sub_ps(_mm256_mul_ps(a, b), c)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub unsafe fn fmadd_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    _mm256_fmadd_pd(a, b, c)
}

#[cfg(not(target_feature = "fma"))]
#[inline(always)]
pub unsafe fn fmadd_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    _mm256_add_pd(_mm256_mul_pd(a, b), c)
}

#[cfg(target_feature = "fma")]
#[inline(always)]
pub unsafe fn fmsub_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    _mm256_fmsub_pd(a, b, c)
}

#[cfg(not(target_feature = "fma"))]
pub unsafe fn fmsub_pd(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    _mm256_sub_pd(_mm256_mul_pd(a, b), c)
}
