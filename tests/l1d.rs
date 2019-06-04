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
