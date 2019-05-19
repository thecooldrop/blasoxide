macro_rules! unroll4 {
    ($e:expr) => {{
        $e;
        $e;
        $e;
        $e
    }};
}

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
    const BLOCKSIZE: usize = 32;
    const UNROLL: usize = 4;
    const VEC_LEN: usize = 8;

    for j in (0..n).step_by(BLOCKSIZE) {
        for i in (0..m).step_by(BLOCKSIZE) {
            for p in (0..k).step_by(BLOCKSIZE) {
                do_block(i, j, p, a, lda, b, ldb, c, ldc);
            }
        }
    }

    unsafe fn do_block(
        si: usize,
        sj: usize,
        sp: usize,
        a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        c: *mut f32,
        ldc: usize,
    ) {
        for i in (0..si + BLOCKSIZE).step_by(VEC_LEN * UNROLL) {
            for j in 0..sj + BLOCKSIZE {
                for p in 0..sp + BLOCKSIZE {}
            }
        }
    }
}
