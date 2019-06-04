#![deny(warnings)]

use blasoxide::*;

const SIZES: [usize; 5] = [251, 501, 1001, 2001, 4001];
const STRIDES: [(usize, usize); 6] = [(1, 1), (1, 7), (7, 1), (7, 11), (11, 7), (11, 11)];

#[test]
fn test_drotg() {
    assert_eq!(drotg(0., 0.), (0., 0., 1., 0.));
    assert_eq!(drotg(-4., 3.), (-5., -0.6, 0.8, -0.6));
    assert_eq!(drotg(4., 3.), (5., 0.6, 0.8, 0.6));
    assert_eq!(drotg(-4., -3.), (-5., 0.6, 0.8, 0.6));
    assert_eq!(drotg(4., -3.), (5., -0.6, 0.8, -0.6));
    assert_eq!(drotg(-3., 4.), (5., 1. / -0.6, -0.6, 0.8));
    assert_eq!(drotg(3., 4.), (5., 1. / 0.6, 0.6, 0.8));
}

fn drot_driver(n: usize, incx: usize, incy: usize, c: f64, s: f64) {
    let mut x = vec![2.; n * incx];
    let mut y = vec![3.; n * incy];

    unsafe {
        drot(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, c, s);
    }

    for (i, xi) in x.into_iter().enumerate() {
        if i % incx == 0 {
            assert_eq!(xi, 2. * c + 3. * s);
        } else {
            assert_eq!(xi, 2.);
        }
    }

    for (i, yi) in y.into_iter().enumerate() {
        if i % incy == 0 {
            assert_eq!(yi, 3. * c - 2. * s);
        } else {
            assert_eq!(yi, 3.);
        }
    }
}

#[test]
fn test_drot() {
    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            drot_driver(n, incx, incy, 5., 4.);
        }
    }
}

fn dswap_driver(n: usize, incx: usize, incy: usize) {
    let mut x = vec![2.; n * incx];
    let mut y = vec![3.; n * incy];

    unsafe {
        dswap(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy);
    }

    for (i, xi) in x.into_iter().enumerate() {
        if i % incx == 0 {
            assert_eq!(xi, 3.);
        } else {
            assert_eq!(xi, 2.);
        }
    }

    for (i, yi) in y.into_iter().enumerate() {
        if i % incy == 0 {
            assert_eq!(yi, 2.);
        } else {
            assert_eq!(yi, 3.);
        }
    }
}

#[test]
fn test_dswap() {
    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            dswap_driver(n, incx, incy);
        }
    }
}

fn dscal_driver(n: usize, incx: usize) {
    let mut x = vec![2.; n * incx];

    unsafe {
        dscal(n, 7., x.as_mut_ptr(), incx);
    }

    for (i, xi) in x.into_iter().enumerate() {
        if i % incx == 0 {
            assert_eq!(xi, 14.);
        } else {
            assert_eq!(xi, 2.);
        }
    }
}

#[test]
fn test_dscal() {
    for &n in SIZES.iter() {
        for &(incx, _) in STRIDES.iter() {
            dscal_driver(n, incx);
        }
    }
}

fn dcopy_driver(n: usize, incx: usize, incy: usize) {
    let x = vec![2.; n * incx];
    let mut y = vec![3.; n * incy];

    unsafe {
        dcopy(n, x.as_ptr(), incx, y.as_mut_ptr(), incy);
    }

    for (i, yi) in y.into_iter().enumerate() {
        if i % incy == 0 {
            assert_eq!(yi, 2.);
        } else {
            assert_eq!(yi, 3.);
        }
    }
}

#[test]
fn test_dcopy() {
    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            dcopy_driver(n, incx, incy);
        }
    }
}

fn daxpy_driver(n: usize, incx: usize, incy: usize) {
    let x = vec![2.; n * incx];
    let mut y = vec![3.; n * incy];

    unsafe {
        daxpy(n, 7., x.as_ptr(), incx, y.as_mut_ptr(), incy);
    }

    for (i, yi) in y.into_iter().enumerate() {
        if i % incy == 0 {
            assert_eq!(yi, 17.);
        } else {
            assert_eq!(yi, 3.);
        }
    }
}

#[test]
fn test_daxpy() {
    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            daxpy_driver(n, incx, incy);
        }
    }
}

fn ddot_driver(n: usize, incx: usize, incy: usize) {
    let x = vec![2.; n * incx];
    let y = vec![3.; n * incy];

    unsafe {
        assert_eq!(ddot(n, x.as_ptr(), incx, y.as_ptr(), incy), n as f64 * 6.);
    }
}

#[test]
fn test_ddot() {
    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            ddot_driver(n, incx, incy);
        }
    }
}

fn dnrm2_driver(n: usize, incx: usize) {
    let x = vec![2.; n * incx];

    unsafe {
        assert_eq!(dnrm2(n, x.as_ptr(), incx), (4. * n as f64).sqrt());
    }
}

#[test]
fn test_dnrm2() {
    for &n in SIZES.iter() {
        for &(incx, _) in STRIDES.iter() {
            dnrm2_driver(n, incx);
        }
    }
}

fn dasum_driver(n: usize, incx: usize) {
    let x = vec![-2.; n * incx];

    unsafe {
        assert_eq!(dasum(n, x.as_ptr(), incx), 2. * n as f64);
    }
}

#[test]
fn test_dasum() {
    for &n in SIZES.iter() {
        for &(incx, _) in STRIDES.iter() {
            dasum_driver(n, incx);
        }
    }
}

fn idamax_driver(n: usize, incx: usize) {
    let mut x = vec![-2.; n * incx];

    x[incx] = 2.;

    unsafe {
        assert_eq!(idamax(n, x.as_ptr(), incx), 0);
    }

    x[incx * 2] = -3.;

    unsafe {
        assert_eq!(idamax(n, x.as_ptr(), incx), 2);
    }
}

#[test]
fn test_idamax() {
    for &n in SIZES.iter() {
        for &(incx, _) in STRIDES.iter() {
            idamax_driver(n, incx);
        }
    }
}
