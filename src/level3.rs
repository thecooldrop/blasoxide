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

    let m_left = m % MC;
    let m_main = m - m_left;
    let k_left = k % KC;
    let k_main = k - k_left;

    for p in (0..k_main).step_by(KC) {
        for i in (0..m_main).step_by(MC) {
            inner_kernel(
                MC,
                n,
                KC,
                a.add(i + p * lda),
                lda,
                b.add(p),
                ldb,
                c.add(i),
                ldc,
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
    ) {
        
    }

    unsafe fn add_dot_4x8(
        k: usize,
        a: *const f32,
        lda: usize,
        b: *const f32,
        ldb: usize,
        c: *mut f32,
        ldc: usize,
    ) {
        
    }
}
