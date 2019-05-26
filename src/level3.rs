use crate::kernel;
use std::sync::mpsc;

#[derive(Clone, Copy)]
struct Mat(*const f32);

unsafe impl Send for Mat {}

pub unsafe fn sgemm(
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
    const MC: usize = 512;
    const KC: usize = 256;

    let a = Mat(a);
    let b = Mat(b);

    let (tx, rx) = mpsc::channel();

    for p in (0..k).step_by(KC) {
        let pb = std::cmp::min(k - p, KC);
        let tx = tx.clone();
        rayon::spawn(move || {
            for i in (0..m).step_by(MC) {
                let ib = std::cmp::min(m - i, MC);
                let tx = tx.clone();
                rayon::spawn(move || {                    
                    let mut tgt_c = vec![0.0; MC * n];

                    inner_kernel(
                        ib,
                        n,
                        pb,
                        a.0.add(i + p * lda),
                        lda,
                        b.0.add(p),
                        ldb,
                        tgt_c.as_mut_ptr(),
                        ib,
                    );

                    tx.send((ib, i, tgt_c)).unwrap();
                });
            }
        });
    }

    std::mem::drop(tx);

    for (ib, i, tgt_c) in rx.iter() {
        kernel::sunpackc(ib, n, alpha, tgt_c.as_ptr(), beta, c.add(i), ldc);
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
    ) {
        let n_left = n % 4;
        let n_main = n - n_left;
        let m_left = m % 8;
        let m_main = m - m_left;

        let mut packed_a = vec![0.0; MC * KC];

        for j in (0..n_main).step_by(4) {
            for i in (0..m_main).step_by(8) {
                if j == 0 {
                    kernel::spacka(k, a.add(i), lda, packed_a.as_mut_ptr().add(i * k));
                }

                kernel::sadot_8x4_packed(
                    k,
                    packed_a.as_ptr().add(i * k),
                    b.add(j * ldb),
                    ldb,
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
}
