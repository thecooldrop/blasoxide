#![deny(warnings)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::float_cmp)]

#[macro_use]
mod util;

#[cfg(target_arch = "x86_64")]
mod fma;

mod generic;
pub use generic::*;

mod l2s;
pub use l2s::*;

mod l2d;
pub use l2d::*;

mod l3s;
pub use l3s::*;

mod l3d;
pub use l3d::*;
