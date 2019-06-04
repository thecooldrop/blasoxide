#![deny(warnings)]

use blasoxide::*;

const SIZES: [usize; 5] = [251, 501, 1001, 2001, 4001];
const STRIDES: [(usize, usize); 6] = [(1, 1), (1, 7), (7, 1), (7, 11), (11, 7), (11, 11)];

#[test]
fn test_srotg() {
    assert_eq!(srotg(0., 0.), (0., 0., 1., 0.));
    assert_eq!(srotg(-4., 3.), (-5., -0.6, 0.8, -0.6));
    assert_eq!(srotg(4., 3.), (5., 0.6, 0.8, 0.6));
    assert_eq!(srotg(-4., -3.), (-5., 0.6, 0.8, 0.6));
    assert_eq!(srotg(4., -3.), (5., -0.6, 0.8, -0.6));
    assert_eq!(srotg(-3., 4.), (5., 1. / -0.6, -0.6, 0.8));
    assert_eq!(srotg(3., 4.), (5., 1. / 0.6, 0.6, 0.8));
}

fn srot_driver(n: usize, incx: usize, incy: usize, c: f32, s: f32) {
    let mut x = vec![2.; n * incx];
    let mut y = vec![3.; n * incy];

    unsafe {
        srot(n, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, c, s);
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
fn test_srot() {
    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            srot_driver(n, incx, incy, 5., 4.);
        }
    }
}

fn sswap_driver(n: usize, incx: usize, incy: usize) {
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
fn test_sswap() {
    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            sswap_driver(n, incx, incy);
        }
    }
}

fn sscal_driver(n: usize, incx: usize) {
    let mut x = vec![2.; n * incx];

    unsafe {
        sscal(n, 7., x.as_mut_ptr(), incx);
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
fn test_sscal() {
    for &n in SIZES.iter() {
        for &(incx, _) in STRIDES.iter() {
            sscal_driver(n, incx);
        }
    }
}

fn scopy_driver(n: usize, incx: usize, incy: usize) {
    let x = vec![2.; n * incx];
    let mut y = vec![3.; n * incy];

    unsafe {
        scopy(n, x.as_ptr(), incx, y.as_mut_ptr(), incy);
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
            scopy_driver(n, incx, incy);
        }
    }
}

fn saxpy_driver(n: usize, incx: usize, incy: usize) {
    let x = vec![2.; n * incx];
    let mut y = vec![3.; n * incy];

    unsafe {
        saxpy(n, 7., x.as_ptr(), incx, y.as_mut_ptr(), incy);
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
fn test_saxpy() {
    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            saxpy_driver(n, incx, incy);
        }
    }
}

fn sdot_driver(n: usize, incx: usize, incy: usize) {
    let x = vec![2.; n * incx];
    let y = vec![3.; n * incy];

    unsafe {
        assert_eq!(sdot(n, x.as_ptr(), incx, y.as_ptr(), incy), n as f32 * 6.);
    }
}

#[test]
fn test_sdot() {
    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            sdot_driver(n, incx, incy);
        }
    }
}

fn snrm2_driver(n: usize, incx: usize) {
    let x = vec![2.; n * incx];

    unsafe {
        assert_eq!(snrm2(n, x.as_ptr(), incx), (4. * n as f32).sqrt());
    }
}

#[test]
fn test_snrm2() {
    for &n in SIZES.iter() {
        for &(incx, _) in STRIDES.iter() {
            snrm2_driver(n, incx);
        }
    }
}

fn sasum_driver(n: usize, incx: usize) {
    let x = vec![-2.; n * incx];

    unsafe {
        assert_eq!(sasum(n, x.as_ptr(), incx), 2. * n as f32);
    }
}

#[test]
fn test_sasum() {
    for &n in SIZES.iter() {
        for &(incx, _) in STRIDES.iter() {
            sasum_driver(n, incx);
        }
    }
}

fn isamax_driver(n: usize, incx: usize) {
    let mut x = vec![-2.; n * incx];

    x[incx] = 2.;

    unsafe {
        assert_eq!(isamax(n, x.as_ptr(), incx), 0);
    }

    x[incx * 2] = -3.;

    unsafe {
        assert_eq!(isamax(n, x.as_ptr(), incx), 2);
    }
}

#[test]
fn test_isamax() {
    for &n in SIZES.iter() {
        for &(incx, _) in STRIDES.iter() {
            isamax_driver(n, incx);
        }
    }
}
