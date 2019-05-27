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

use test::Bencher;

#[bench]
fn bench_sgemm_1000(bencher: &mut Bencher) {
    const LEN: usize = 1000;
    let a = vec![1.0; LEN * LEN];
    let b = vec![1.0; LEN * LEN];
    let mut c = vec![0.0; LEN * LEN];

    bencher.iter(|| unsafe {
        blasoxide::sgemm(
            false,
            false,
            LEN,
            LEN,
            LEN,
            1.0,
            a.as_ptr(),
            LEN,
            b.as_ptr(),
            LEN,
            1.0,
            c.as_mut_ptr(),
            LEN,
        );
    });
}

#[bench]
fn bench_sgemm_500(bencher: &mut Bencher) {
    const LEN: usize = 500;
    let a = vec![1.0; LEN * LEN];
    let b = vec![1.0; LEN * LEN];
    let mut c = vec![0.0; LEN * LEN];

    bencher.iter(|| unsafe {
        blasoxide::sgemm(
            false,
            false,
            LEN,
            LEN,
            LEN,
            1.0,
            a.as_ptr(),
            LEN,
            b.as_ptr(),
            LEN,
            1.0,
            c.as_mut_ptr(),
            LEN,
        );
    });
}

#[bench]
fn bench_sgemm_250(bencher: &mut Bencher) {
    const LEN: usize = 250;
    let a = vec![1.0; LEN * LEN];
    let b = vec![1.0; LEN * LEN];
    let mut c = vec![0.0; LEN * LEN];

    bencher.iter(|| unsafe {
        blasoxide::sgemm(
            false,
            false,
            LEN,
            LEN,
            LEN,
            1.0,
            a.as_ptr(),
            LEN,
            b.as_ptr(),
            LEN,
            1.0,
            c.as_mut_ptr(),
            LEN,
        );
    });
}
