use crate::context::Context;
use crate::kernels::{sgemm_pa, sgemm_pb, sgemm_sup0, sgemm_sup1, sgemm_ukr};
use crate::kernels::{SMR as MR, SNR as NR};
use crate::send::{SSend, SSendMut};

pub unsafe fn sgemm(
    context: &Context,
    _transa: bool,
    _transb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    let mc = context.smc();
    let kc = context.skc();
    let nc = context.snc();

    let pa = context.spa();
    let pb = context.spb();

    for j in (0..n).step_by(nc) {
        let js = std::cmp::min(n - j, nc);
        let mut beta_scale = beta;
        for p in (0..k).step_by(kc) {
            let ps = std::cmp::min(k - p, kc);
            for i in (0..m).step_by(mc) {
                let is = std::cmp::min(m - i, mc);
                sgemm_macrokernel(
                    context,
                    is,
                    js,
                    ps,
                    alpha,
                    a.add(i + p * lda),
                    lda,
                    b.add(p + j * ldb),
                    ldb,
                    beta_scale,
                    c.add(i + j * ldc),
                    ldc,
                    pa,
                    pb,
                    i == 0,
                );
            }
            beta_scale = 1.0;
        }
    }
}

unsafe fn sgemm_macrokernel(
    context: &Context,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
    pa: *mut f32,
    pb: *mut f32,
    first_time: bool,
) {
    let n_left = n % NR;
    let n_main = n - n_left;
    let m_left = m % MR;
    let m_main = m - m_left;

    let a = SSend(a);
    let b = SSend(b);
    let c = SSendMut(c);
    let pa = SSendMut(pa);
    let pb = SSendMut(pb);

    if first_time {
        context.execute(0, n_main, NR, move |j| {
            sgemm_pb(k, b.0.add(j * ldb), ldb, pb.0.add(j * k));
        });
    }

    context.execute(0, m_main, MR, move |i| {
        sgemm_pa(k, a.0.add(i), lda, pa.0.add(i * k));
    });

    context.execute(0, n_main, NR, move |j| {
        for i in (0..m_main).step_by(MR) {
            sgemm_ukr(
                k,
                alpha,
                pa.0.add(i * k),
                pb.0.add(j * k),
                beta,
                c.0.add(i + j * ldc),
                ldc,
            );
        }

        for i in m_main..m {
            sgemm_sup1(
                k,
                alpha,
                a.0.add(i),
                lda,
                pb.0.add(j * k),
                beta,
                c.0.add(i + j * ldc),
                ldc,
            );
        }
    });

    context.execute(n_main, n, 1, move |j| {
        for i in (0..m_main).step_by(MR) {
            sgemm_sup0(
                k,
                alpha,
                pa.0.add(i * k),
                b.0.add(j * ldb),
                beta,
                c.0.add(i + j * ldc),
            );
        }

        for i in m_main..m {
            let mut elem = 0.0;

            for p in 0..k {
                elem += *a.0.add(i + p * lda) * *b.0.add(p + j * ldb);
            }

            elem *= alpha;

            if beta != 0.0 {
                elem += beta * *c.0.add(i + j * ldc);
            }

            *c.0.add(i + j * ldc) = elem;
        }
    });
}
