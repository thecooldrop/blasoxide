#[allow(clippy::many_single_char_names)]

pub fn srotg(a: f32, b: f32) -> (f32, f32, f32, f32) {
    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0, 1.0, 0.0);
    }
    let h = a.hypot(b);
    let r = if a.abs() > b.abs() {
        h * a.signum()
    } else {
        h * b.signum()
    };
    let c = a / r;
    let s = b / r;
    let z = if a.abs() > b.abs() {
        s
    } else if c != 0.0 {
        1.0 / c
    } else {
        1.0
    };
    (r, z, c, s)
}
