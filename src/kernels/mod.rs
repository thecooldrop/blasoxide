#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
mod avx;

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub use avx::{l1d::*, l1s::*};

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub(crate) use avx::l3s::{
    sgemm_pa_16x as sgemm_pa, sgemm_sup_16x1 as sgemm_sup0, sgemm_ukr_16x4 as sgemm_ukr,
};

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub(crate) use avx::l3d::{
    dgemm_pa_8x as dgemm_pa, dgemm_sup_8x1 as dgemm_sup0, dgemm_ukr_8x4 as dgemm_ukr,
};

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub(crate) use generic::l3s::{sgemm_pb_x4 as sgemm_pb, sgemm_sup_1x4 as sgemm_sup1};

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub(crate) use generic::l3d::{dgemm_pb_x4 as dgemm_pb, dgemm_sup_1x4 as dgemm_sup1};

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub(crate) const SMR: usize = 16;

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub(crate) const SNR: usize = 4;

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub(crate) const DMR: usize = 8;

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "x86"),
    target_feature = "avx"
))]
pub(crate) const DNR: usize = 4;

mod generic;
