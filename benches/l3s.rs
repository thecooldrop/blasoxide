#![feature(test)]
#![deny(warnings)]

extern crate test;

use blasoxide::*;

use test::Bencher;

fn sgemm_driver(m: usize, n: usize, k: usize, bencher: &mut Bencher) {
    let a = aligned_alloc::Alloc::new(m * k * std::mem::size_of::<f32>());
    let b = aligned_alloc::Alloc::new(n * k * std::mem::size_of::<f32>());
    let c = aligned_alloc::Alloc::new(m * n * std::mem::size_of::<f32>());

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
            a.ptr() as *const f32,
            m,
            b.ptr() as *const f32,
            k,
            11.,
            c.ptr() as *mut f32,
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
