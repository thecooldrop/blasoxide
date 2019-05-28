#![deny(warnings)]
#![feature(test)]

#[cfg(not(all(
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "fma"
)))]
compile_error!(
    "Benchmark needs SIMD features to be enabled. Maybe set RUSTFLAGS=\"-C target-cpu=native\""
);

extern crate test;

use test::{black_box, Bencher};

#[bench]
fn bench_srot(bencher: &mut Bencher) {
    const LEN: usize = 1000;

    let mut x = vec![1.0; LEN];
    let mut y = vec![-1.0; LEN];

    bencher.iter(|| unsafe {
        blasoxide::srot(LEN, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, 3.0, -5.0);
    });
}

#[bench]
fn bench_sswap(bencher: &mut Bencher) {
    const LEN: usize = 1000;

    let mut x = vec![1.0; LEN];
    let mut y = vec![-1.0; LEN];

    bencher.iter(|| unsafe {
        blasoxide::sswap(LEN, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1);
    });
}

#[bench]
fn bench_sscal(bencher: &mut Bencher) {
    const LEN: usize = 1000;

    let mut x = vec![1.0; LEN];

    bencher.iter(|| unsafe {
        blasoxide::sscal(LEN, 3.2, x.as_mut_ptr(), 1);
    });
}

#[bench]
fn bench_scopy(bencher: &mut Bencher) {
    const LEN: usize = 1000;

    let x = vec![1.0; LEN];
    let mut y = vec![0.0; LEN];

    bencher.iter(|| unsafe {
        blasoxide::scopy(LEN, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    });
}

#[bench]
fn bench_saxpy(bencher: &mut Bencher) {
    const LEN: usize = 1000;

    let x = vec![1.0; LEN];
    let mut y = vec![0.0; LEN];

    bencher.iter(|| unsafe {
        blasoxide::saxpy(LEN, 3.3, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    });
}

#[bench]
fn bench_sdot(bencher: &mut Bencher) {
    const LEN: usize = 1000;

    let x = vec![1.0; LEN];
    let y = vec![0.3; LEN];

    bencher.iter(|| unsafe {
        black_box(blasoxide::sdot(LEN, x.as_ptr(), 1, y.as_ptr(), 1));
    });
}

#[bench]
fn bench_snrm2(bencher: &mut Bencher) {
    const LEN: usize = 1000;

    let x = vec![1.0; LEN];

    bencher.iter(|| unsafe {
        black_box(blasoxide::snrm2(LEN, x.as_ptr(), 1));
    });
}

#[bench]
fn bench_sasum(bencher: &mut Bencher) {
    const LEN: usize = 1000;

    let x = vec![1.0; LEN];

    bencher.iter(|| unsafe {
        black_box(blasoxide::sasum(LEN, x.as_ptr(), 1));
    });
}
