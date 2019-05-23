// https://github.com/flame/how-to-optimize-gemm

pub unsafe fn sgemm(
    _transa: bool,
    _transb: bool,
    m: usize,
    n: usize,
    k: usize,
    _alpha: f32,
    mut a: *const f32,
    lda: isize,
    mut b: *const f32,
    ldb: isize,
    _beta: f32,
    mut c: *mut f32,
    ldc: isize,
) {
    const MC: usize = 256;
    const KC: usize = 128;
    const UNROLL: usize = 4;

    for _ in 0..k / KC {
        a = a.offset(KC as isize * ldb);
        b = b.add(KC);
        for _ in 0..m / MC {
            b = b.add(MC);
            c = c.add(MC);
            inner_kernel(MC, n, KC, a, lda, b, ldb, c, ldc);
        }
    }

    unsafe fn inner_kernel(
        m: usize,
        n: usize,
        k: usize,
        mut a: *const f32,
        lda: isize,
        mut b: *const f32,
        ldb: isize,
        mut c: *mut f32,
        ldc: isize,
    ) {
        for _ in 0..n / UNROLL {
            b = b.offset(UNROLL as isize * ldb);
            c = c.offset(UNROLL as isize * ldc);
            for _ in 0..m / 8 {
                a = a.add(8);
                c = c.add(8);

                add_dot_4x8(k, a, lda, b, ldb, c, ldc);
            }
        }
    }

    unsafe fn add_dot_4x8(
        k: usize,
        a: *const f32,
        lda: isize,
        b: *const f32,
        ldb: isize,
        c: *const f32,
        ldc: isize,
    ) {
        let mut bptr0 = b;
        let mut bptr1 = b.offset(ldb);
        let mut bptr2 = b.offset(2 * ldb);
        let mut bptr3 = b.offset(3 * ldb);

    }
}
