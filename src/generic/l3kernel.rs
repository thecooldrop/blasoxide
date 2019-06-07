pub unsafe fn sgemm_16x4_packed(
    k: usize,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::sgemm_16x4_packed(k, a, b, beta, c, ldc);
            return;
        }
    }
}

pub unsafe fn dgemm_8x4_packed(
    k: usize,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            crate::fma::dgemm_8x4_packed(k, a, b, beta, c, ldc);
            return;
        }
    }
}
