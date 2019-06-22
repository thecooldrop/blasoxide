#![feature(test)]
#![deny(warnings)]

extern crate test;

use blasoxide::*;

use test::Bencher;

fn sgemm_driver(m: usize, n: usize, k: usize, bencher: &mut Bencher) {
    let a = vec![2.; m * k];
    let b = vec![3.; k * n];
    let mut c = vec![5.; m * n];

    let context = Context::new();

    bencher.iter(|| unsafe {
        sgemm(
            &context,
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
fn bench_sgemm_250(bencher: &mut Bencher) {
    const LEN: usize = 250;
    sgemm_driver(LEN, LEN, LEN, bencher);
}

#[bench]
fn bench_sgemm_500(bencher: &mut Bencher) {
    const LEN: usize = 500;
    sgemm_driver(LEN, LEN, LEN, bencher);
}

#[bench]
fn bench_sgemm_1000(bencher: &mut Bencher) {
    const LEN: usize = 1000;
    sgemm_driver(LEN, LEN, LEN, bencher);
}

#[bench]
fn bench_sgemm_2000(bencher: &mut Bencher) {
    const LEN: usize = 2000;
    sgemm_driver(LEN, LEN, LEN, bencher);
}
