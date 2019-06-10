# blasoxide

[![crates.io](https://meritbadge.herokuapp.com/blasoxide)](https://crates.io/crates/blasoxide)
[![Released API docs](https://docs.rs/blasoxide/badge.svg)](https://docs.rs/blasoxide)
[![Build Status](https://travis-ci.org/oezgurmakkurt/blasoxide.svg?branch=master)](https://travis-ci.org/oezgurmakkurt/blasoxide)

BLAS implementation in rust

### Architecture

Only Level1 functions and micro kernels are optimized with platform specific code.

Level3 functions are parallelized with rayon.

### Supported CPUs
These cpus have optimized implementations for them

- x86_64 cpus with fma support

### Contributing
Anyone can contribute anything as they see fit. Just don't forget to run `cargo clippy` and `cargo fmt` before commiting
