#![deny(warnings)]

#[test]
fn test_sgemv() {
    const LEN: usize = 1131;
    let mut a = vec![0.5; LEN * LEN];
    let mut x = vec![0.5; LEN];
    let mut y = vec![0.0; LEN];
    let mut yref = vec![0.0; LEN];

    for i in 0..LEN {
        for j in 0..LEN {
            a[i + j * LEN] = i as f32;
            x[j] = j as f32 + i as f32;
        }
    }

    unsafe {
        blasoxide::sgemv(
            false,
            LEN,
            LEN,
            0.5,
            a.as_ptr(),
            LEN,
            x.as_ptr(),
            1,
            -0.5,
            y.as_mut_ptr(),
            1,
        );
    }

    unsafe {
        sgemv_ref(
            false,
            LEN,
            LEN,
            0.5,
            a.as_ptr(),
            LEN,
            x.as_ptr(),
            1,
            -0.5,
            yref.as_mut_ptr(),
            1,
        );
    }

    for i in 0..LEN {
        let (a, b) = (y[i], yref[i]);
        assert!((a - b).abs() < 10.0, "a!=b, a={}, b={}", a, b);
    }
}

unsafe fn sgemv_ref(
    _trans: bool,
    m: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    x: *const f32,
    incx: usize,
    beta: f32,
    y: *mut f32,
    incy: usize,
) {
    for i in 0..n {
        let mut yreg = *y.add(i * incy) * beta;
        for j in 0..m {
            yreg += alpha * *a.add(i + j * lda) * *x.add(j * incx);
        }
        *y.add(i * incy) = yreg;
    }
}
