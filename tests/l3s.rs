#![deny(warnings)]

use blasoxide::*;

const SIZES: [(usize, usize, usize); 5] = [(251, 251, 251), (251, 551, 375), (251, 3001, 2001), (1025, 1025, 1025), (551, 251, 1001)];

fn sgemm_driver(m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize) {
    let a = vec![2.; k * lda];
    let b = vec![3.; n * ldb];
    let mut c = vec![5.; n * ldc];

    unsafe {
        sgemm(false, false, m, n, k, 7., a.as_ptr(), lda, b.as_ptr(), ldb, 11., c.as_mut_ptr(), ldc);
    }

    for j in 0..n {
        for i in 0..m {
            assert_eq!(c[i + j * ldc], 55. + k as f32 * 6. * 7.);
        }
        for i in m..ldc {
            assert_eq!(c[i + j * ldc], 5.);
        }
    }
}

#[test]
fn test_sgemm() {
    let ld = 251;
    for &(m, n, k) in SIZES.iter() {
        sgemm_driver(m, n, k, ld + m, ld + k, ld + m);
    }
}
