#![deny(warnings)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::float_cmp)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::new_without_default)]

mod aligned_alloc;

mod context;

pub use context::Context;

#[macro_use]
mod util;

#[cfg(target_arch = "x86_64")]
mod fma;

#[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
pub use fma::*;

#[cfg(not(all(target_arch = "x86_64", target_feature = "fma")))]
mod generic;
#[cfg(not(all(target_arch = "x86_64", target_feature = "fma")))]
pub use generic::*;

mod l2s;
pub use l2s::*;

mod l2d;
pub use l2d::*;

mod l3s;
pub use l3s::*;

mod l3d;
pub use l3d::*;

/// Returns coefficients of 2x2 rotation matrix, such that
/// when it is multiplied with a 2x1 vector consisting of coefficients
/// given as arguments to function results in a 2x1 vector which has
/// the length of input vector as first coefficient and zero as second coefficient
/// More information can be found in Wikipedia under article Givens rotation.
pub fn drotg(a: f64, b: f64) -> (f64, f64, f64, f64) {
    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0, 1.0, 0.0);
    }
    let h = a.hypot(b);
    let r = if a.abs() > b.abs() {
        h.copysign(a)
    } else {
        h.copysign(b)
    };
    let c = a / r;
    let s = b / r;
    let z = if a.abs() > b.abs() {
        s
    } else if c != 0.0 {
        1.0 / c
    } else {
        1.0
    };
    (r, z, c, s)
}

pub unsafe fn idamax(n: usize, mut x: *const f64, incx: usize) -> usize {
    let mut max = 0.0;
    let mut imax = 0;
    for i in 0..n {
        let xi = (*x).abs();
        if xi > max {
            max = xi;
            imax = i;
        }
        x = x.add(incx);
    }
    imax
}

/// Returns coefficients of 2x2 rotation matrix, such that
/// when it is multiplied with a 2x1 vector consisting of coefficients
/// given as arguments to function results in a 2x1 vector which has
/// the length of input vector as first coefficient and zero as second coefficient
/// More information can be found in Wikipedia under article Givens rotation.
pub fn srotg(a: f32, b: f32) -> (f32, f32, f32, f32) {
    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0, 1.0, 0.0);
    }
    let h = a.hypot(b);
    let r = if a.abs() > b.abs() {
        h.copysign(a)
    } else {
        h.copysign(b)
    };
    let c = a / r;
    let s = b / r;
    let z = if a.abs() > b.abs() {
        s
    } else if c != 0.0 {
        1.0 / c
    } else {
        1.0
    };
    (r, z, c, s)
}

pub unsafe fn sdsdot(
    n: usize,
    b: f32,
    mut x: *const f32,
    incx: usize,
    mut y: *const f32,
    incy: usize,
) -> f32 {
    let mut acc: f64 = f64::from(b);
    for _ in 0..n {
        acc += f64::from(*x) * f64::from(*y);
        x = x.add(incx);
        y = y.add(incy);
    }
    acc as f32
}

pub unsafe fn isamax(n: usize, mut x: *const f32, incx: usize) -> usize {
    let mut max = 0.0;
    let mut imax = 0;
    for i in 0..n {
        let xi = (*x).abs();
        if xi > max {
            max = xi;
            imax = i;
        }
        x = x.add(incx);
    }
    imax
}
