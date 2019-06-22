use crate::util::{SSend, SSendMut};
use crate::context::Context;

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
                inner_kernel(
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

    unsafe fn inner_kernel(
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
        packed_a: *mut f32,
        packed_b: *mut f32,
        first_time: bool,
    ) {
        let n_left = n % 4;
        let n_main = n - n_left;
        let m_left = m % 16;
        let m_main = m - m_left;

        let a = SSend(a);
        let b = SSend(b);
        let c = SSendMut(c);
        let packed_a = SSendMut(packed_a);
        let packed_b = SSendMut(packed_b);

        if first_time {
            context.execute(0, n_main, 4, move |j| {
                pack_b(k, b.0.add(j * ldb), ldb, packed_b.0.add(j * k));
            });
        }

        context.execute(0, m_main, 16, move |i| {
            crate::s_pack_a(k, alpha, a.0.add(i), lda, packed_a.0.add(i * k));
        });

        context.execute(0, n_main, 4, move |j| {
                for i in (0..m_main).step_by(16) {
                    crate::sgemm_16x4_packed(
                        k,
                        packed_a.0.add(i * k),
                        packed_b.0.add(j * k),
                        beta,
                        c.0.add(i + j * ldc),
                        ldc,
                    );
                }

                for i in m_main..m {
                    add_dot_1x4(
                        k,
                        alpha,
                        a.0.add(i),
                        lda,
                        packed_b.0.add(j * k),
                        beta,
                        c.0.add(i + j * ldc),
                        ldc,
                    );
                }
        });

        context.execute(n_main, n, 1, move |j| {
            for i in 0..m {
                add_dot_1x1(
                    k,
                    alpha,
                    a.0.add(i),
                    lda,
                    b.0.add(j * ldb),
                    beta,
                    c.0.add(i + j * ldc),
                );
            }
        });
    }

    unsafe fn pack_b(k: usize, b: *const f32, ldb: usize, mut packed_b: *mut f32) {
        let mut bptr0 = b;
        let mut bptr1 = b.add(ldb);
        let mut bptr2 = b.add(ldb * 2);
        let mut bptr3 = b.add(ldb * 3);

        for _ in 0..k {
            *packed_b = *bptr0;
            *packed_b.add(1) = *bptr1;
            *packed_b.add(2) = *bptr2;
            *packed_b.add(3) = *bptr3;

            packed_b = packed_b.add(4);
            bptr0 = bptr0.add(1);
            bptr1 = bptr1.add(1);
            bptr2 = bptr2.add(1);
            bptr3 = bptr3.add(1);
        }
    }

    unsafe fn add_dot_1x1(
        k: usize,
        alpha: f32,
        mut a: *const f32,
        lda: usize,
        mut b: *const f32,
        beta: f32,
        c: *mut f32,
    ) {
        let mut c0reg = *c * beta;
        for _ in 0..k {
            c0reg += *a * *b * alpha;

            a = a.add(lda);
            b = b.add(1);
        }
        *c = c0reg;
    }

    unsafe fn add_dot_1x4(
        k: usize,
        alpha: f32,
        mut a: *const f32,
        lda: usize,
        mut pb: *const f32,
        beta: f32,
        c: *mut f32,
        ldc: usize,
    ) {

        let cptr0 = c;
        let cptr1 = c.add(ldc);
        let cptr2 = c.add(2 * ldc);
        let cptr3 = c.add(3 * ldc);

        let mut c0_reg = *cptr0 * beta;
        let mut c1_reg = *cptr1 * beta;
        let mut c2_reg = *cptr2 * beta;
        let mut c3_reg = *cptr3 * beta;

        for _ in 0..k {
            let a0_reg = *a * alpha;
            let bp0reg = *pb;
            let bp1reg = *pb.add(1);
            let bp2reg = *pb.add(2);
            let bp3reg = *pb.add(3);

            c0_reg += a0_reg * bp0reg;
            c1_reg += a0_reg * bp1reg;
            c2_reg += a0_reg * bp2reg;
            c3_reg += a0_reg * bp3reg;

            a = a.add(lda);
            pb = pb.add(4);
        }

        *cptr0 = c0_reg;
        *cptr1 = c1_reg;
        *cptr2 = c2_reg;
        *cptr3 = c3_reg;
    }
}
