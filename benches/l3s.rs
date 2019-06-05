#![feature(test)]
#![deny(warnings)]

extern crate test;

use blasoxide::*;

use test::Bencher;

fn sgemm_driver(m: usize, n: usize, k: usize, bencher: &mut Bencher) {
    let a = vec![2.; m * k];
    let b = vec![3.; k * n];
    let mut c = vec![5.; m * n];

    bencher.iter(|| unsafe {
        sgemm(
            false,
            false,
            m,
            n,
            k,
            7.,
            a.as_ptr(),
            m,
            b.as_ptr(),
            k,
            11.,
            c.as_mut_ptr(),
            m,
        );
    });
}

#[bench]
fn bench_sgemm(bencher: &mut Bencher) {
    sgemm_driver(1024, 1024, 1024, bencher);
}
