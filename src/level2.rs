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
    incx: isize,
    beta: f32,
    y: *mut f32,
    incy: isize,
) {
    const MC: usize = 512;
    const NC: usize = 256;

    let mut packed_a = vec![0.0; MC * NC];

    
}
