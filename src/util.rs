#[derive(Clone, Copy)]
pub struct SSend(pub *const f32);

unsafe impl Send for SSend {}
unsafe impl Sync for SSend {}

#[derive(Clone, Copy)]
pub struct SSendMut(pub *mut f32);

unsafe impl Send for SSendMut {}
unsafe impl Sync for SSendMut {}

#[derive(Clone, Copy)]
pub struct DSend(pub *const f64);

unsafe impl Send for DSend {}
unsafe impl Sync for DSend {}

#[derive(Clone, Copy)]
pub struct DSendMut(pub *mut f64);

unsafe impl Send for DSendMut {}
unsafe impl Sync for DSendMut {}

pub const SABS_MASK: u32 = 0x7FFF_FFFF;
pub const DABS_MASK: u64 = 0x7FFF_FFFF_FFFF_FFFF;

macro_rules! unroll4 {
    ($e:expr) => {{
        $e;
        $e;
        $e;
        $e;
    }};
}
