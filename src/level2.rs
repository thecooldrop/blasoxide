use crate::{SSend, SSendMut};
use core::arch::x86_64::*;
use rayon::prelude::*;

pub unsafe fn sgemv(
    _trans: bool,
    m: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    x: *const f32,
    incx: usize,
    beta: f32,
    y: *mut f32,
    incy: usize,
) {
    const MC: usize = 512;
    const NC: usize = 256;

    let mut beta_scale = beta;

    for j in (0..n).step_by(NC) {
        let jb = std::cmp::min(n - j, NC);
        for i in (0..m).step_by(MC) {
            let ib = std::cmp::min(m - i, MC);
            inner_kernel(
                ib,
                jb,
                alpha,
                a.add(i + j * lda),
                lda,
                x.add(j * incx),
                incx,
                beta_scale,
                y.add(i * incy),
                incy,
            );
        }
        beta_scale = 1.0;
    }

    unsafe fn inner_kernel(
        m: usize,
        n: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        x: *const f32,
        incx: usize,
        beta: f32,
        y: *mut f32,
        incy: usize,
    ) {
        let n_left = n % 4;
        let n_main = n - n_left;
        let m_left = m % 8;
        let m_main = m - m_left;

        let a = SSend(a);
        let x = SSend(x);
        let y = SSendMut(y);

        (0..n_main)
            .step_by(4)
            .collect::<Vec<_>>()
            .par_iter()
            .for_each(move |&j| {
                for i in (0..m_main).step_by(8) {
                    add_dot_8x4(
                        n,
                        alpha,
                        a.0.add(i + j * lda),
                        lda,
                        x.0.add(j * incx),
                        incx,
                        beta,
                        y.0.add(i * incy),
                        incy,
                    );
                }
            });
    }

    unsafe fn add_dot_8x4(
        k: usize,
        alpha: f32,
        a: *const f32,
        lda: usize,
        x: *const f32,
        incx: usize,
        beta: f32,
        y: *mut f32,
        incy: usize,
    ) {
        let mut xptr0 = x;
        let mut xptr1 = x.add(incx);
        let mut xptr2 = x.add(incx * 2);
        let mut xptr3 = x.add(incx * 3);

        let yptr0 = &mut *y;
        let yptr1 = &mut *y.add(incy);
        let yptr2 = &mut *y.add(2 * incy);
        let yptr3 = &mut *y.add(3 * incy);

        let betav = _mm256_broadcast_ss(&beta);
        let mut y0_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(yptr0));
        let mut y1_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(yptr1));
        let mut y2_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(yptr2));
        let mut y3_reg_v = _mm256_mul_ps(betav, _mm256_loadu_ps(yptr3));

        let mut aptr = a;
        let xstep = incx * 4;
        let alphav = _mm256_broadcast_ss(&alpha);

        for _ in 0..k {
            let areg = _mm256_mul_ps(alphav, _mm256_loadu_ps(aptr));

            let xp0reg = _mm256_broadcast_ss(&*xptr0);
            let xp1reg = _mm256_broadcast_ss(&*xptr1);
            let xp2reg = _mm256_broadcast_ss(&*xptr2);
            let xp3reg = _mm256_broadcast_ss(&*xptr3);

            y0_reg_v = _mm256_fmadd_ps(areg, xp0reg, y0_reg_v);
            y1_reg_v = _mm256_fmadd_ps(areg, xp1reg, y1_reg_v);
            y2_reg_v = _mm256_fmadd_ps(areg, xp2reg, y2_reg_v);
            y3_reg_v = _mm256_fmadd_ps(areg, xp3reg, y3_reg_v);

            aptr = aptr.add(lda);
            xptr0 = xptr0.add(xstep);
            xptr1 = xptr1.add(xstep);
            xptr2 = xptr2.add(xstep);
            xptr3 = xptr3.add(xstep);
        }

        _mm256_storeu_ps(yptr0, y0_reg_v);
        _mm256_storeu_ps(yptr1, y1_reg_v);
        _mm256_storeu_ps(yptr2, y2_reg_v);
        _mm256_storeu_ps(yptr3, y3_reg_v);
    }
}
