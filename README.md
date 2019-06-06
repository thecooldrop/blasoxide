# blasoxide

[![crates.io](https://meritbadge.herokuapp.com/blasoxide)](https://crates.io/crates/blasoxide)
[![Released API docs](https://docs.rs/blasoxide/badge.svg)](https://docs.rs/blasoxide)

BLAS implementation in rust

## Features

- level 1 double and single precision operations
- sgemv, dgemv
- sgemm, dgemm
- all functions are optimized for cpus with avx and fma
- performance is very close to openblas

 ## Building
 
 Enable avx and fma by doing `export RUSTFLAGS="-C target-cpu=native"`
 
 ## Testing
 
 `cargo test --release`
 
 ## Benchmarking
 
 `cargo +nightly bench`
