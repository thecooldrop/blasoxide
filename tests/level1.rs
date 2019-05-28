#[test]
fn test_srotg() {
    assert_eq!(blasoxide::srotg(0.0, 0.0), (0.0, 0.0, 1.0, 0.0));
    assert_eq!(blasoxide::srotg(0.0, -1.0), (-1.0, 1.0, 0.0, 1.0));
}

#[test]
fn test_srot() {
    const LEN: usize = 1000;

    let mut x = vec![1.0; LEN];
    let mut y = vec![-3.0; LEN];

    unsafe {
        blasoxide::srot(LEN, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, 2.0, 1.0);
    }

    for i in 0..LEN {
        assert_eq!(x[i], 1.0 * 2.0 - 3.0 * 1.0);
        assert_eq!(y[i], -3.0 * 2.0 - 1.0 * 1.0);
    }
}

#[test]
fn test_srot_with_stride() {
    const LEN: usize = 1000;
    const STRIDE: usize = 10;

    let mut x = vec![1.0; LEN];
    let mut y = vec![-3.0; LEN];

    unsafe {
        blasoxide::srot(
            LEN / STRIDE,
            x.as_mut_ptr(),
            STRIDE,
            y.as_mut_ptr(),
            STRIDE,
            2.0,
            1.0,
        );
    }

    for i in 0..LEN {
        if i % STRIDE == 0 {
            assert_eq!(x[i], 1.0 * 2.0 - 3.0 * 1.0);
            assert_eq!(y[i], -3.0 * 2.0 - 1.0 * 1.0);
        } else {
            assert_eq!(x[i], 1.0);
            assert_eq!(y[i], -3.0);
        }
    }
}

#[test]
fn test_sswap() {
    const LEN: usize = 1000;

    let mut x = vec![1.0; LEN];
    let mut y = vec![-3.0; LEN];

    unsafe {
        blasoxide::sswap(LEN, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1);
    }

    for i in 0..LEN {
        assert_eq!(x[i], -3.0);
        assert_eq!(y[i], 1.0);
    }
}

#[test]
fn test_sswap_with_stride() {
    const LEN: usize = 1000;
    const STRIDE: usize = 5;

    let mut x = vec![1.0; LEN];
    let mut y = vec![-3.0; LEN];

    unsafe {
        blasoxide::sswap(LEN / STRIDE, x.as_mut_ptr(), STRIDE, y.as_mut_ptr(), STRIDE);
    }

    for i in 0..LEN {
        if i % STRIDE == 0 {
            assert_eq!(x[i], -3.0);
            assert_eq!(y[i], 1.0);
        } else {
            assert_eq!(x[i], 1.0);
            assert_eq!(y[i], -3.0);
        }
    }
}

#[test]
fn test_sscal() {
    const LEN: usize = 1000;

    let mut x = vec![-3.0; LEN];

    unsafe {
        blasoxide::sscal(LEN, 5.0, x.as_mut_ptr(), 1);
    }

    for i in 0..LEN {
        assert_eq!(x[i], -3.0 * 5.0);
    }
}

#[test]
fn test_sscal_with_stride() {
    const LEN: usize = 1000;
    const STRIDE: usize = 20;

    let mut x = vec![-3.0; LEN];

    unsafe {
        blasoxide::sscal(LEN / STRIDE, 5.0, x.as_mut_ptr(), STRIDE);
    }

    for i in 0..LEN {
        if i % STRIDE == 0 {
            assert_eq!(x[i], -3.0 * 5.0);
        } else {
            assert_eq!(x[i], -3.0);
        }
    }
}

#[test]
fn test_scopy() {
    const LEN: usize = 1000;

    let x = vec![1.0; LEN];
    let mut y = vec![-3.0; LEN];

    unsafe {
        blasoxide::scopy(LEN, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }

    for i in 0..LEN {
        assert_eq!(y[i], 1.0);
    }
}

#[test]
fn test_scopy_with_stride() {
    const LEN: usize = 1000;
    const STRIDE: usize = 9;

    let x = vec![1.0; LEN];
    let mut y = vec![-3.0; LEN];

    unsafe {
        blasoxide::scopy(LEN / STRIDE, x.as_ptr(), STRIDE, y.as_mut_ptr(), STRIDE);
    }

    for i in 0..LEN - STRIDE {
        if i % STRIDE == 0 {
            assert_eq!(y[i], 1.0);
        } else {
            assert_eq!(y[i], -3.0);
        }
    }
}

#[test]
fn test_saxpy() {
    const LEN: usize = 1000;

    let x = vec![1.0; LEN];
    let mut y = vec![-3.0; LEN];

    unsafe {
        blasoxide::saxpy(LEN, 5.0, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }

    for i in 0..LEN {
        assert_eq!(y[i], 2.0);
    }
}

#[test]
fn test_saxpy_with_stride() {
    const LEN: usize = 1000;
    const STRIDE: usize = 5;

    let x = vec![1.0; LEN];
    let mut y = vec![-3.0; LEN];

    unsafe {
        blasoxide::saxpy(
            LEN / STRIDE,
            5.0,
            x.as_ptr(),
            STRIDE,
            y.as_mut_ptr(),
            STRIDE,
        );
    }

    for i in 0..LEN {
        if i % STRIDE == 0 {
            assert_eq!(y[i], 2.0);
        } else {
            assert_eq!(y[i], -3.0);
        }
    }
}

#[test]
fn test_sdot() {
    const LEN: usize = 1000;

    let x = vec![1.0; LEN];
    let y = vec![-3.0; LEN];

    let res = unsafe { blasoxide::sdot(LEN, x.as_ptr(), 1, y.as_ptr(), 1) };

    assert_eq!(res, LEN as f32 * 1.0 * -3.0);
}

#[test]
fn test_sdot_with_stride() {
    const LEN: usize = 1000;
    const STRIDE: usize = 5;

    let x = vec![1.0; LEN];
    let y = vec![-3.0; LEN];

    let res = unsafe { blasoxide::sdot(LEN / STRIDE, x.as_ptr(), STRIDE, y.as_ptr(), STRIDE) };

    assert_eq!(res, (LEN / STRIDE) as f32 * 1.0 * -3.0);
}

#[test]
fn test_sdsdot() {
    const LEN: usize = 1000;

    let x = vec![1.0; LEN];
    let y = vec![-3.0; LEN];

    let res = unsafe { blasoxide::sdsdot(LEN, 7.0, x.as_ptr(), 1, y.as_ptr(), 1) };

    assert_eq!(res, LEN as f32 * 1.0 * -3.0 + 7.0);
}

#[test]
fn test_sdsdot_with_stride() {
    const LEN: usize = 1000;
    const STRIDE: usize = 5;

    let x = vec![1.0; LEN];
    let y = vec![-3.0; LEN];

    let res =
        unsafe { blasoxide::sdsdot(LEN / STRIDE, 7.0, x.as_ptr(), STRIDE, y.as_ptr(), STRIDE) };

    assert_eq!(res, (LEN / STRIDE) as f32 * 1.0 * -3.0 + 7.0);
}

#[test]
fn test_snrm2() {
    const LEN: usize = 100;

    let x = vec![2.0; LEN];

    let res = unsafe { blasoxide::snrm2(LEN, x.as_ptr(), 1) };

    assert_eq!(res, 20.0);
}

#[test]
fn test_snrm2_with_stride() {
    const LEN: usize = 100;
    const STRIDE: usize = 4;

    let x = vec![2.0; LEN];

    let res = unsafe { blasoxide::snrm2(LEN / STRIDE, x.as_ptr(), STRIDE) };

    assert_eq!(res, 10.0);
}

#[test]
fn test_sasum() {
    const LEN: usize = 1000;

    let x = vec![-1.5; LEN];

    let res = unsafe { blasoxide::sasum(LEN, x.as_ptr(), 1) };

    assert_eq!(res, LEN as f32 * 1.5);
}

#[test]
fn test_sasum_with_stride() {
    const LEN: usize = 1000;
    const STRIDE: usize = 2;

    let x = vec![-1.5; LEN];

    let res = unsafe { blasoxide::sasum(LEN / STRIDE, x.as_ptr(), STRIDE) };

    assert_eq!(res, LEN as f32 * 1.5 / STRIDE as f32);
}

#[test]
fn test_isamax() {
    const LEN: usize = 1000;

    let mut x = vec![-1.0; LEN];

    x[15] = 30.0;

    x[29] = 45.0;

    x[31] = -12.0;

    let res = unsafe { blasoxide::isamax(LEN, x.as_ptr(), 1) };

    assert_eq!(res, 29);
}

#[test]
fn test_isamax_with_stride() {
    const LEN: usize = 1000;
    const STRIDE: usize = 5;

    let mut x = vec![-1.0; LEN];

    x[15] = 30.0;

    x[30] = 45.0;

    x[60] = -12.0;

    x[4] = 99.0;
    x[97] = 99.0;

    let res = unsafe { blasoxide::isamax(LEN / STRIDE, x.as_ptr(), STRIDE) };

    assert_eq!(res, 6);
}
