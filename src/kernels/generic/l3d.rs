pub(crate) unsafe fn dgemm_sup_1x4(
    k: usize,
    alpha: f64,
    a: *const f64,
    lda: usize,
    pb: *const f64,
    beta: f64,
    c: *mut f64,
    ldc: usize,
) {
    let mut c0 = 0.0;
    let mut c1 = 0.0;
    let mut c2 = 0.0;
    let mut c3 = 0.0;

    let mut a = a;
    let mut pb = pb;

    for _ in 0..k {
        let a0 = *a;

        c0 += *pb * a0;
        c1 += *pb.add(1) * a0;
        c2 += *pb.add(2) * a0;
        c3 += *pb.add(3) * a0;

        a = a.add(lda);
        pb = pb.add(4);
    }

    c0 *= alpha;
    c1 *= alpha;
    c2 *= alpha;
    c3 *= alpha;

    let ccol0 = c;
    let ccol1 = c.add(ldc);
    let ccol2 = c.add(ldc * 2);
    let ccol3 = c.add(ldc * 3);

    if beta != 0.0 {
        c0 *= beta * *ccol0;
        c1 *= beta * *ccol1;
        c2 *= beta * *ccol2;
        c3 *= beta * *ccol3;
    }

    *ccol0 = c0;
    *ccol1 = c1;
    *ccol2 = c2;
    *ccol3 = c3;
}

pub(crate) unsafe fn dgemm_pb_x4(k: usize, b: *const f64, ldb: usize, pb: *mut f64) {
    let mut bcol0 = b;
    let mut bcol1 = b.add(ldb);
    let mut bcol2 = b.add(ldb * 2);
    let mut bcol3 = b.add(ldb * 3);

    let mut pb = pb;

    for _ in 0..k {
        *pb = *bcol0;
        *pb.add(1) = *bcol1;
        *pb.add(2) = *bcol2;
        *pb.add(3) = *bcol3;

        bcol0 = bcol0.add(1);
        bcol1 = bcol1.add(1);
        bcol2 = bcol2.add(1);
        bcol3 = bcol3.add(1);
        pb = pb.add(4);
    }
}
