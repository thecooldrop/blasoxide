#![allow(clippy::many_single_char_names)]

#[macro_use]
mod common;

mod level1;
pub use level1::*;

mod gemm;
pub use gemm::*;
