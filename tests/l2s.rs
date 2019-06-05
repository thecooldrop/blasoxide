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

#[test]
fn test_sgemv() {
    let lda = *SIZES.last().unwrap();

    for &n in SIZES.iter() {
        for &(incx, incy) in STRIDES.iter() {
            sgemv_driver(n, n, lda, incx, incy);
        }
    }
}
