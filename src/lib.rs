#[macro_use]
extern crate cfg_if;
extern crate rayon;

mod util;

#[cfg(target_arch = "x86_64")]
mod fma;

cfg_if! {
    if #[cfg(all(target_arch = "x86_64", target_feature = "fma"))] {
        pub use fma::*;
    } else {
        mod generic;
        pub use generic::*;
    }
}

mod l2s;
pub use l2s::*;

mod l2d;
pub use l2d::*;

mod l3s;
pub use l3s::*;

mod l3d;
pub use l3d::*;
