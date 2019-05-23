#[macro_use]
extern crate criterion;

use criterion::black_box;
use criterion::Criterion;

use blasoxide::sgemm;

fn criterion_benchmark(cri: &mut Criterion) {
    const LEN: usize = 1024;
    let a = vec![1.0; LEN * LEN];
    let b = vec![1.0; LEN * LEN];
    let mut c = vec![0.0; LEN * LEN];

    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();

    cri.bench_function("sgemm", move |bencher| {
        bencher.iter(|| unsafe {
            black_box(sgemm(
                false,
                false,
                LEN,
                LEN,
                LEN,
                1.0,
                ap,
                LEN,
                bp,
                LEN,
                1.0,
                cp,
                LEN,
            ));
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
