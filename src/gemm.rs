pub unsafe fn sgemm(
    transa: bool,
    transb: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *const f32,
    ldc: usize,
) {
    
}
