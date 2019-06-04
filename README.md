# blasoxide

[![crates.io](https://meritbadge.herokuapp.com/blasoxide)](https://crates.io/crates/blasoxide)
[![Released API docs](https://docs.rs/blasoxide/badge.svg)](https://docs.rs/blasoxide)

BLAS implementation in rust

## Limitations
- Only latest stable rustc is supported
- Only cpus with avx and fma are supported

## Features
- Performance should be same as OpenBLAS, BLIS, MKL
- Level 3 functions use rayon for multithreading
- Compiles fast, just cargo build, no configuration needed other than RUSTFLAGS=-C target-cpu=native
- Zero dependencies except rayon, which is included because most programs will depend on it anyway

## Usage
All code is in unsafe rust, but it is very easy to wrap in safe rust. I am planning to write a simple safe wrapper to this with Matrix, Vector types.

## Contributing
Anyone can contribute to improve performance, add tests, add benchmarks, add documentation. Even running benchmarks and tests and reporting problems is much appreciated
