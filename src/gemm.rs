// https://github.com/flame/how-to-optimize-gemm

use core::arch::x86_64::*;
use std::sync;

pub unsafe fn sgemm(
    _transa: bool,
    _transb: bool,
    m: usize,
    n: usize,
    k: usize,
    _alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    _beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    #[derive(Clone, Copy)]
    struct MatMut(*mut f32);

    unsafe impl Send for MatMut {}

    #[derive(Clone, Copy)]
    struct Mat(*const f32);

    unsafe impl Send for Mat {}

    const MC: usize = 512;
    const KC: usize = 256;
    const UNROLL: usize = 4;

    let a = Mat(a);

    let b = Mat(b);

    let c = MatMut(c);

    let mut recvs = Vec::new();

    for p in (0..k).step_by(KC) {
        let pb = std::cmp::min(k - p, KC);
        let (tx, rx) = sync::mpsc::channel();
        recvs.push(rx);
        rayon::spawn(move || {
            let mut packed_a = vec![0.0; MC * KC];
            for i in (0..m).step_by(MC) {
                let ib = std::cmp::min(m - i, MC);
                inner_kernel(
                    ib,
                    n,
                    pb,
                    a.0.add(i + p * lda),
                    lda,
                    b.0.add(p),
                    ldb,
                    c.0.add(i),
                    ldc,
                    packed_a.as_mut_ptr(),
                );
            }
            tx.send(()).unwrap();
        });
    }

    for rx in recvs {
        rx.recv().unwrap();
    }

    unsafe fn inner_kernel(
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        c: *mut f32,
        ldc: usize,
        packed_a: *mut f32,
    ) {
        for j in (0..n).step_by(UNROLL) {
            if n - j < UNROLL {
                for ji in j..n {
                    for ii in 0..m {
                        let mut cval = 0.0;
                        for p in 0..k {
                            cval += *a.add(ii + p * lda) * *b.add(p + ji * ldb);
                        }
                        *c.add(ii + ji * ldc) += cval;
                    }
                }

                break;
            }
            for i in (0..m).step_by(8) {
                if m - i < 8 {
                    for ii in i..m {
                        let mut cval0 = 0.0;
                        let mut cval1 = 0.0;
                        let mut cval2 = 0.0;
                        let mut cval3 = 0.0;
                        for p in 0..k {
                            cval0 += *a.add(ii + p * lda) * *b.add(p + j * ldb);
                            cval1 += *a.add(ii + p * lda) * *b.add(p + (j+1) * ldb);
                            cval2 += *a.add(ii + p * lda) * *b.add(p + (j+2) * ldb);
                            cval3 += *a.add(ii + p * lda) * *b.add(p + (j+3) * ldb);
                        }
                        *c.add(ii + j * ldc) += cval0;
                        *c.add(ii + (j+1) * ldc) += cval1;
                        *c.add(ii + (j+2) * ldc) += cval2;
                        *c.add(ii + (j+3) * ldc) += cval3;
                    }

                    break;
                }
                if j == 0 {
                    pack_a(k, a.add(i), lda, packed_a.add(i * k));
                }
                add_dot_4x8(
                    k,
                    packed_a.add(i * k),
                    8,
                    b.add(j * ldb),
                    ldb,
                    c.add(i + j * ldc),
                    ldc,
                );
            }
        }
    }

    unsafe fn pack_a(k: usize, mut a: *const f32, lda: usize, mut a_to: *mut f32) {
        for _ in 0..k {
            _mm256_storeu_ps(a_to, _mm256_loadu_ps(a));

            a = a.add(lda);
            a_to = a_to.add(8);
        }
    }

    unsafe fn add_dot_4x8(
        k: usize,
        mut a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        c: *mut f32,
        ldc: usize,
    ) {
        let mut bptr0 = b;
        let mut bptr1 = b.add(ldb);
        let mut bptr2 = b.add(2 * ldb);
        let mut bptr3 = b.add(3 * ldb);

        let mut c0_reg_v = _mm256_setzero_ps();
        let mut c1_reg_v = _mm256_setzero_ps();
        let mut c2_reg_v = _mm256_setzero_ps();
        let mut c3_reg_v = _mm256_setzero_ps();

        for _ in 0..k {
            let a0_reg_v = _mm256_loadu_ps(a);
            let bp0reg = _mm256_broadcast_ss(&*bptr0);
            let bp1reg = _mm256_broadcast_ss(&*bptr1);
            let bp2reg = _mm256_broadcast_ss(&*bptr2);
            let bp3reg = _mm256_broadcast_ss(&*bptr3);

            c0_reg_v = _mm256_fmadd_ps(a0_reg_v, bp0reg, c0_reg_v);
            c1_reg_v = _mm256_fmadd_ps(a0_reg_v, bp1reg, c1_reg_v);
            c2_reg_v = _mm256_fmadd_ps(a0_reg_v, bp2reg, c2_reg_v);
            c3_reg_v = _mm256_fmadd_ps(a0_reg_v, bp3reg, c3_reg_v);

            a = a.add(lda);
            bptr0 = bptr0.add(1);
            bptr1 = bptr1.add(1);
            bptr2 = bptr2.add(1);
            bptr3 = bptr3.add(1);
        }

        let cptr0 = &mut *c;
        let cptr1 = &mut *c.add(ldc);
        let cptr2 = &mut *c.add(2 * ldc);
        let cptr3 = &mut *c.add(3 * ldc);

        _mm256_storeu_ps(cptr0, _mm256_add_ps(_mm256_loadu_ps(cptr0), c0_reg_v));
        _mm256_storeu_ps(cptr1, _mm256_add_ps(_mm256_loadu_ps(cptr1), c1_reg_v));
        _mm256_storeu_ps(cptr2, _mm256_add_ps(_mm256_loadu_ps(cptr2), c2_reg_v));
        _mm256_storeu_ps(cptr3, _mm256_add_ps(_mm256_loadu_ps(cptr3), c3_reg_v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgemm() {
        const LEN: usize = 201;
        let mut a = vec![1.0; LEN * LEN];
        let mut b = vec![1.0; LEN * LEN];
        let mut c = vec![0.0; LEN * LEN];
        let mut cref = vec![0.0; LEN * LEN];

        for i in 0..LEN {
            for j in 0..LEN {
                a[i + j * LEN] = i as f32;
                b[i + j * LEN] = j as f32 + i as f32;
            }
        }

        unsafe {
            sgemm(
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
        }

        unsafe {
            sgemm_ref(
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
                cref.as_mut_ptr(),
                LEN,
            );
        }

        for i in 0..LEN {
            for j in 0..LEN {
                assert_eq!(c[i + j * LEN], cref[i + j * LEN]);
            }
        }
    }

    unsafe fn sgemm_ref(
        _transa: bool,
        _transb: bool,
        m: usize,
        n: usize,
        k: usize,
        _alpha: f32,
        a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        _beta: f32,
        c: *mut f32,
        ldc: usize,
    ) {
        for j in 0..n {
            for i in 0..m {
                let mut ci = *c.add(i + j * ldc);
                for p in 0..k {
                    ci += *a.add(i + p * lda) * *b.add(p + j * ldb);
                }
                *c.add(i + j * ldc) = ci;
            }
        }
    }
}
