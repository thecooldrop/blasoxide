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

use test::Bencher;

#[bench]
fn bench_sgemv_1000(bencher: &mut Bencher) {
    const LEN: usize = 1000;
    let a = vec![0.5; LEN * LEN];
    let x = vec![0.5; LEN];
    let mut y = vec![0.0; LEN];
    bencher.iter(|| unsafe {
        blasoxide::sgemv(
            false,
            LEN,
            LEN,
            1.0,
            a.as_ptr(),
            LEN,
            x.as_ptr(),
            1,
            1.0,
            y.as_mut_ptr(),
            1,
        );
    });
}
