#![deny(warnings)]

use blasoxide::*;

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

fn test_drot(n: usize, incx: usize, incy: usize, c: f64, s: f64) {
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

fn test_drot_mul(n: usize) {
    test_drot(n, 1, 1, 4., 5.);
    test_drot(n, 1, 2, 4., 5.);
    test_drot(n, 2, 1, 4., 5.);
    test_drot(n, 7, 1, 4., 5.);
    test_drot(n, 1, 7, 4., 5.);
    test_drot(n, 9, 7, 4., 5.);
    test_drot(n, 7, 9, 4., 5.);
    test_drot(n, 7, 7, 4., 5.);
}

#[test]
fn test_drot_250() {
    test_drot_mul(250);
}

#[test]
fn test_drot_500() {
    test_drot_mul(500);
}

#[test]
fn test_drot_1000() {
    test_drot_mul(1000);
}

#[test]
fn test_drot_2000() {
    test_drot_mul(2000);
}

#[test]
fn test_drot_4000() {
    test_drot_mul(4000);
}
