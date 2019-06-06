#![deny(warnings)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::float_cmp)]

#[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
compile_error!("blasoxide needs avx and fma to compile, maybe set RUSTFLAGS=-C target-cpu=native");

#[macro_use]
mod common;

mod l1s;
pub use l1s::*;

mod l1d;
pub use l1d::*;

mod l2s;
pub use l2s::*;

mod l2d;
pub use l2d::*;

mod l3s;
pub use l3s::*;

mod l3d;
pub use l3d::*;

#[derive(Clone, Copy)]
struct SSend(*const f32);

unsafe impl Send for SSend {}
unsafe impl Sync for SSend {}

#[derive(Clone, Copy)]
struct SSendMut(*mut f32);

unsafe impl Send for SSendMut {}
unsafe impl Sync for SSendMut {}

#[derive(Clone, Copy)]
struct DSend(*const f64);

unsafe impl Send for DSend {}
unsafe impl Sync for DSend {}

#[derive(Clone, Copy)]
struct DSendMut(*mut f64);

unsafe impl Send for DSendMut {}
unsafe impl Sync for DSendMut {}
