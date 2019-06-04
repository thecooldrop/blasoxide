#![deny(warnings)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::float_cmp)]

mod l1s;
pub use l1s::*;

mod l2s;
pub use l2s::*;

mod l3s;
pub use l3s::*;

#[derive(Clone, Copy)]
struct SSend(*const f32);

unsafe impl Send for SSend {}
unsafe impl Sync for SSend {}

#[derive(Clone, Copy)]
struct SSendMut(*mut f32);

unsafe impl Send for SSendMut {}
unsafe impl Sync for SSendMut {}
