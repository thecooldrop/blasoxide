pub unsafe fn sgemm(
    transa: bool,
    transb: bool,
    m: isize,
    n: isize,
    k: isize,
    alpha: f32,
    a: *const f32,
    lda: isize,
    b: *const f32,
    ldb: isize,
    beta: f32,
    c: *mut f32,
    ldc: isize,
) {
    if !transa && !transb {
        for j in 0..n {
            for i in 0..m {
                for p in 0..k {
                    let cp = c.offset(i + j * ldc);
                    *cp = beta * *cp + alpha * *a.offset(i + p * lda) * *b.offset(p + j * ldb);
                }
            }
        }
    }   
}
