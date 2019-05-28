use core::x86_64::*;
use rayon::prelude::*;

pub unsafe fn sgemv(
    _trans: bool,
    m: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    x: *const f32,
    incx: usize,
    beta: f32,
    y: *mut f32,
    incy: usize,
) {
    const MC: usize = 512;
    const NC: usize = 256;

    let mut packed_a = vec![0.0; MC * NC];

    let mut beta_scale = beta;

    for j in (0..n).step_by(NC) {
        let jb = std::cmp::min(n - j, NC);
        for i in (0..m).step_by(MC) {
            let ib = std::cmp::min(m - i, MC);
            inner_kernel(
                ib,
                jb,
                alpha,
                a.add(i + j * lda),
                lda,
                x.add(j * incx),
                incx,
                beta_scale,
                y.add(i * incy),
                incy,
                packed_a.as_mut_ptr(),
            );
        }
        beta_scale = 1.0;
    }

    unsafe fn inner_kernel(
        m: usize,
        n: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        x: *const f32,
        incx: usize,
        beta: f32,
        y: *mut f32,
        incy: usize,
    ) {

    }

    unsafe fn pack_a(k: usize, alpha: f32, mut a: *const f32, lda: usize, mut packed_a: *mut f32) {
        let alphav = _mm256_broadcast_ss(&alpha);

        for _ in 0..k {
            _mm256_storeu_ps(packed_a, _mm256_mul_ps(alphav, _mm256_loadu_ps(a)));

            a = a.add(lda);
            packed_a = packed_a.add(8);
        }
    }
}
