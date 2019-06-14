#![deny(warnings)]

use blasoxide::*;

const SIZES: [usize; 5] = [251, 501, 1001, 2001, 4001];
const STRIDES: [(usize, usize); 6] = [(1, 1), (1, 7), (7, 1), (7, 11), (11, 7), (11, 11)];

fn sgemv_driver(m: usize, n: usize, lda: usize, incx: usize, incy: usize) {
    assert!(m <= lda);

    let x = vec![2.; n * incx];
    let a = vec![3.; n * lda];
    let mut y = vec![5.; m * incy];

    unsafe {
        sgemv(
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
            assert_eq!(yi, 5. * 11. + 7. * 3. * 2. * n as f32);
        } else {
            assert_eq!(yi, 5.);
        }
    }
}

fn sgemv_driver_trans(m: usize, n: usize, lda: usize, incx: usize, incy: usize) {
    assert!(m <= lda);

    let x = vec![2.; n * incx];
    let mut a = vec![3.; n * lda];
    let mut y = vec![5.; m * incy];

    for j in 0..n {
        for i in 0..m {
            a[i + j * lda] = j as f32;
        }
    }

    unsafe {
        sgemv(
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
            let expected = 5. * 11. + (index * m) as f32 * 2. * 7.;
            let diff = (expected - yi).abs();
            assert!(
                diff < (expected + yi) / 2.0 / 1.0e+6,
                "expected={};yi={}",
                expected,
                yi
            );
        } else {
            assert_eq!(yi, 5.);
        }
    }
}

fn ssymv_driver(upper: bool, n: usize, lda: usize, incx: usize, incy: usize) {
    assert!(n <= lda);

    let x = vec![2.; n * incx];
    let mut a = vec![3.; n * lda];
    let mut na = vec![3.; n * lda];
    let mut y = vec![5.; n * incy];
    let mut ny = vec![5.; n * incy];

    let alpha = 7.;
    let beta = 11.;

    if upper {
        for j in 0..n {
            for i in 0..j + 1 {
                a[i + j * lda] = j as f32;
                na[i + j * lda] = j as f32;
                na[j + i * lda] = j as f32;
            }
        }
    } else {
        for j in 0..n {
            for i in j..n {
                a[i + j * lda] = j as f32;
                na[i + j * lda] = j as f32;
                na[j + i * lda] = j as f32;
            }
        }
    }

    unsafe {
        ssymv(
            upper,
            n,
            alpha,
            a.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            beta,
            y.as_mut_ptr(),
            incy,
        );
    }

    unsafe {
        sgemv(
            false,
            n,
            n,
            alpha,
            na.as_ptr(),
            lda,
            x.as_ptr(),
            incx,
            beta,
            ny.as_mut_ptr(),
            incy,
        );
    }

    for (&nyi, &yi) in ny.iter().zip(y.iter()) {
        let expected = nyi;
        let diff = (expected - yi).abs();
        assert!(
            diff < (expected + yi) / 2.0 / 1.0e+4,
            "expected={};yi={}",
            expected,
            yi
        );
    }
}

fn strmv_driver(upper: bool, trans: bool, diag: bool, n: usize, lda: usize, incx: usize) {
    assert!(n <= lda);

    let mut x = vec![2.; n * incx];
    let mut a = vec![3.; n * lda];
    let mut na = vec![0.; n * lda];
    let nx = vec![2.; n * incx];
    let mut ny = vec![2.; n * incx];

    if upper {
        for j in 0..n {
            for i in 0..j + 1 {
                a[i + j * lda] = j as f32;
                na[i + j * lda] = j as f32;
            }
        }
    } else {
        for j in 0..n {
            for i in j..n {
                a[i + j * lda] = j as f32;
                na[i + j * lda] = j as f32;
            }
        }
    }

    if diag {
        for j in 0..n {
            na[j + j * lda] = 1.;
        }
    }

    unsafe {
        strmv(upper, trans, diag, n, a.as_ptr(), lda, x.as_mut_ptr(), incx);
    }

    unsafe {
        sgemv(
            trans,
            n,
            n,
            1.,
            na.as_ptr(),
            lda,
            nx.as_ptr(),
            incx,
            0.,
            ny.as_mut_ptr(),
            incx,
        );
    }

    for (i, (&nyi, &xi)) in ny.iter().zip(x.iter()).enumerate() {
        if i % incx == 0 {
            let expected = nyi;
            let diff = (expected - xi).abs();
            assert!(
                diff == 0.0 || diff < (expected + xi) / 2.0 / 1.0e+5,
                "expected={};xi={};upper={};trans={};diag={};index={}",
                expected,
                xi,
                upper,
                trans,
                diag,
                i / incx,
            );
        } else {
            assert_eq!(xi, 2.);
        }
    }
}

fn ssyr_driver(upper: bool, n: usize, incx: usize, lda: usize) {
    let mut a = vec![31.; n * lda];
    let mut x = vec![3.; n * incx];

    for i in 0..n {
        x[incx * i] = i as f32;
    }

    if upper {
        for j in 0..n {
            for i in 0..j+1 {
                a[i + j * lda] = j as f32;
            }
        }
    } else {
        for j in 0..n {
            for i in j..n {
                a[i + j * lda] = j as f32;
            }
        }
    }

    unsafe {
        ssyr(upper, n, 7., x.as_ptr(), incx, a.as_mut_ptr(), lda);
    }

    if upper {
        for j in 0..n {
            for i in 0..j+1 {
                let expected = j as f32 + x[j * incx] * x[i * incx] * 7.;
                let got = a[i + j * lda];
                let diff = (expected - got).abs();
                assert!(
                    diff == 0.0 || diff < (expected + got) / 2.0 / 1.0e+6,
                );
            }
        }
    } else {
        for j in 0..n {
            for i in j..n {
                let expected = j as f32 + x[j * incx] * x[i * incx] * 7.;
                let got = a[i + j * lda];
                let diff = (expected - got).abs();
                assert!(
                    diff == 0.0 || diff < (expected + got) / 2.0 / 1.0e+6,
                );
            }
        }
    }
}

#[test]
fn test_ssyr() {
    let lda = *SIZES.last().unwrap();

    for &n in SIZES.iter() {
        for &(incx, _incy) in STRIDES.iter() {
            ssyr_driver(false, n, incx, lda);
            ssyr_driver(true, n, incx, lda);
        }
    }
}

#[test]
fn test_sgemv() {
    let lda = *SIZES.last().unwrap();

    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            sgemv_driver(n, n, lda, incx, incy);
            sgemv_driver_trans(n, n, lda, incx, incy);
        }
    }
}

#[test]
fn test_ssymv_sgemv_compare() {
    let lda = *SIZES.last().unwrap();

    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            ssymv_driver(true, n, lda, incx, incy);
            ssymv_driver(false, n, lda, incx, incy);
        }
    }
}

#[test]
fn test_strmv_sgemv_compare() {
    let lda = *SIZES.last().unwrap();

    for &n in SIZES.iter() {
        for &(incx, _incy) in STRIDES.iter() {
            for &trans in [true, false].iter() {
                for &diag in [true, false].iter() {
                    for &upper in [true, false].iter() {
                        strmv_driver(upper, trans, diag, n, lda, incx);
                    }
                }
            }
        }
    }
}
