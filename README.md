# blasoxide

[![crates.io](https://meritbadge.herokuapp.com/blasoxide)](https://crates.io/crates/blasoxide)
[![Released API docs](https://docs.rs/blasoxide/badge.svg)](https://docs.rs/blasoxide)

BLAS implementation in rust

### Architecture

Only Level1 functions and micro kernels are optimized with platform specific code.

Level3 functions are parallelized with rayon.

### Supported CPUs
These cpus have optimized implementations for them

- x86_64 cpus with fma support
