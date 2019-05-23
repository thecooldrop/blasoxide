// https://github.com/flame/how-to-optimize-gemm

use core::arch::x86_64::*;

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
    const MC: usize = 256;
    const KC: usize = 128;
    const UNROLL: usize = 4;

    let mut packed_a = vec![0.0; MC * KC];

    for p in (0..k).step_by(KC) {
        let pb = std::cmp::min(k - p, KC);
        for i in (0..m).step_by(MC) {
            let ib = std::cmp::min(m - i, MC);
            inner_kernel(
                ib,
                n,
                pb,
                a.add(i + p * lda),
                lda,
                b.add(p),
                ldb,
                c.add(i),
                ldc,
                packed_a.as_mut_ptr(),
            );
        }
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
            for i in (0..m).step_by(8) {
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
        const LEN: usize = 1024;
        let a = vec![1.0; LEN * LEN];
        let b = vec![1.0; LEN * LEN];
        let mut c = vec![0.0; LEN * LEN];

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

        for i in 0..LEN {
            for j in 0..LEN {
                assert_eq!(c[i + j * LEN], LEN as f32);
            }
        }
    }
}
