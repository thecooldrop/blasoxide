#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]

#[macro_use]
mod common;

mod level1;
pub use level1::*;

mod gemm;
pub use gemm::*;
