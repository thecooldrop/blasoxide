#![deny(warnings)]

use blasoxide::*;

const SIZES: [usize; 5] = [251, 501, 1001, 2001, 4001];
const STRIDES: [(usize, usize); 6] = [(1, 1), (1, 7), (7, 1), (7, 11), (11, 7), (11, 11)];

fn dgemv_driver(m: usize, n: usize, lda: usize, incx: usize, incy: usize) {
    assert!(m <= lda);

    let x = vec![2.; n * incx];
    let a = vec![3.; n * lda];
    let mut y = vec![5.; m * incy];

    unsafe {
        dgemv(
            false,
            m,
            n,
            7.,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            11.,
            y.as_mut_ptr(),
            incy,
        );
    }

    for (i, yi) in y.into_iter().enumerate() {
        if i % incy == 0 {
            assert_eq!(yi, 5. * 11. + 7. * 3. * 2. * n as f64);
        } else {
            assert_eq!(yi, 5.);
        }
    }
}

fn dgemv_driver_trans(m: usize, n: usize, lda: usize, incx: usize, incy: usize) {
    assert!(m <= lda);

    let x = vec![2.; n * incx];
    let mut a = vec![3.; n * lda];
    let mut y = vec![5.; m * incy];

    for j in 0..n {
        for i in 0..m {
            a[i + j * lda] = j as f64;
        }
    }

    unsafe {
        dgemv(
            true,
            m,
            n,
            7.,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            11.,
            y.as_mut_ptr(),
            incy,
        );
    }

    for (i, yi) in y.into_iter().enumerate() {
        if i % incy == 0 {
            let index = i / incy;
            let expected = 5. * 11. + (index * m) as f64 * 2. * 7.;
            let diff = (expected - yi).abs();
            assert!(
                diff < (expected + yi) / 2.0 / 1.0e+8,
                "expected={};yi={}",
                expected,
                yi
            );
        } else {
            assert_eq!(yi, 5.);
        }
    }
}

#[test]
fn test_dgemv() {
    let lda = *SIZES.last().unwrap();

    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            dgemv_driver(n, n, lda, incx, incy);
            dgemv_driver_trans(n, n, lda, incx, incy);
        }
    }
}
