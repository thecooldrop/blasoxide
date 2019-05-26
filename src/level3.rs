use core::arch::x86_64::*;

use crate::kernel;

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

    let mut packed_a = vec![0.0; MC * KC];
    let mut packed_b = vec![0.0; KC * n];

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

                kernel::sadot_8x4_packed(
                    k,
                    packed_a.add(i * k),
                    packed_b.add(j * k),
                    c.add(i + j * ldc),
                    ldc,
                );
            }

            for i in m_main..m {
                kernel::sadot_1x4(
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
                kernel::sadot_8x1(k, a.add(i), lda, b.add(j * ldb), c.add(i + j * ldc));
            }
            for i in m_main..m {
                kernel::sadot_1x1(k, a.add(i), lda, b.add(j * ldb), c.add(i + j * ldc));
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
}
