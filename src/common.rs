use core::arch::x86_64::*;

macro_rules! unroll4 {
    ($e:expr) => {{
        $e;
        $e;
        $e;
        $e;
    }};
}

pub unsafe fn hadd_ps(mut v: __m256) -> f32 {
    v = _mm256_hadd_ps(v, v);
    v = _mm256_hadd_ps(v, v);
    let v = std::mem::transmute::<__m256, [f32; 8]>(v);
    v[0] + v[4]
}