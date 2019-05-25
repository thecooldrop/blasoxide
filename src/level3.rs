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
    const MC: usize = 512;
    const KC: usize = 256;
    const NB: usize = 1000;

    let mut packed_a = vec![0.0; MC * KC];
    let mut packed_b = vec![0.0; KC * NB];

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
                packed_b.as_mut_ptr(),
                i == 0,
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
        packed_b: *mut f32,
        first_time: bool,
    ) {
        let n_left = n % 4;
        let n_main = n - n_left;
        let m_left = m % 8;
        let m_main = m - m_left;

        for j in (0..n_main).step_by(4) {
            if first_time {
                pack_b(k, b.add(j * ldb), ldb, packed_b.add(j * k));
            }
            for i in (0..m_main).step_by(8) {
                if j == 0 {
                    pack_a(k, a.add(i), lda, packed_a.add(i * k));
                }

                add_dot_8x4_packed(
                    k,
                    packed_a.add(i * k),
                    packed_b.add(j * k),
                    c.add(i + j * ldc),
                    ldc,
                );
            }

            for i in m_main..m {
                add_dot_1x4(
                    k,
                    a.add(i),
                    lda,
                    b.add(j * ldb),
                    ldb,
                    c.add(i + j * ldc),
                    ldc,
                );
            }
        }

        for j in n_main..n {
            for i in (0..m_main).step_by(8) {
                add_dot_8x1(k, a.add(i), lda, b.add(j * ldb), c.add(i + j * ldc));
            }
            for i in m_main..m {
                add_dot_1x1(k, a.add(i), lda, b.add(j * ldb), c.add(i + j * ldc));
            }
        }
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

    unsafe fn pack_a(k: usize, mut a: *const f32, lda: usize, mut packed_a: *mut f32) {
        for _ in 0..k {
            _mm256_storeu_ps(packed_a, _mm256_loadu_ps(a));

            a = a.add(lda);
            packed_a = packed_a.add(8);
        }
    }

    unsafe fn add_dot_8x4_packed(
        k: usize,
        mut a: *const f32,
        mut b: *const f32,
        c: *mut f32,
        ldc: usize,
    ) {
        let mut c0_reg_v = _mm256_setzero_ps();
        let mut c1_reg_v = _mm256_setzero_ps();
        let mut c2_reg_v = _mm256_setzero_ps();
        let mut c3_reg_v = _mm256_setzero_ps();

        for _ in 0..k {
            let a0_reg_v = _mm256_loadu_ps(a);
            let bp0reg = _mm256_broadcast_ss(&*b);
            let bp1reg = _mm256_broadcast_ss(&*b.add(1));
            let bp2reg = _mm256_broadcast_ss(&*b.add(2));
            let bp3reg = _mm256_broadcast_ss(&*b.add(3));

            c0_reg_v = _mm256_fmadd_ps(a0_reg_v, bp0reg, c0_reg_v);
            c1_reg_v = _mm256_fmadd_ps(a0_reg_v, bp1reg, c1_reg_v);
            c2_reg_v = _mm256_fmadd_ps(a0_reg_v, bp2reg, c2_reg_v);
            c3_reg_v = _mm256_fmadd_ps(a0_reg_v, bp3reg, c3_reg_v);

            a = a.add(8);
            b = b.add(4);
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

    unsafe fn add_dot_1x1(k: usize, mut a: *const f32, lda: usize, mut b: *const f32, c: *mut f32) {
        let mut c0reg = 0.0;
        for _ in 0..k {
            c0reg += *a * *b;

            a = a.add(lda);
            b = b.add(1);
        }
        *c += c0reg;
    }

    unsafe fn add_dot_1x4(
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
        let mut bptr2 = b.add(ldb * 2);
        let mut bptr3 = b.add(ldb * 3);

        let mut c0_reg = 0.0;
        let mut c1_reg = 0.0;
        let mut c2_reg = 0.0;
        let mut c3_reg = 0.0;

        for _ in 0..k {
            let a0_reg = *a;
            let bp0reg = *bptr0;
            let bp1reg = *bptr1;
            let bp2reg = *bptr2;
            let bp3reg = *bptr3;

            c0_reg += a0_reg * bp0reg;
            c1_reg += a0_reg * bp1reg;
            c2_reg += a0_reg * bp2reg;
            c3_reg += a0_reg * bp3reg;

            a = a.add(lda);
            bptr0 = bptr0.add(1);
            bptr1 = bptr1.add(1);
            bptr2 = bptr2.add(1);
            bptr3 = bptr3.add(1);
        }

        *c += c0_reg;
        *c.add(ldc) += c1_reg;
        *c.add(2 * ldc) += c2_reg;
        *c.add(3 * ldc) += c3_reg;
    }

    unsafe fn add_dot_8x1(k: usize, mut a: *const f32, lda: usize, mut b: *const f32, c: *mut f32) {
        let mut c0_reg_v = _mm256_setzero_ps();

        for _ in 0..k {
            let a0_reg_v = _mm256_loadu_ps(a);
            let b0_reg_v = _mm256_broadcast_ss(&*b);

            c0_reg_v = _mm256_fmadd_ps(a0_reg_v, b0_reg_v, c0_reg_v);

            a = a.add(lda);
            b = b.add(1);
        }

        let cptr0 = &mut *c;

        _mm256_storeu_ps(cptr0, _mm256_add_ps(_mm256_loadu_ps(cptr0), c0_reg_v));
    }
}
