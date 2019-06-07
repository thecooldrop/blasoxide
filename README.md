# blasoxide

[![crates.io](https://meritbadge.herokuapp.com/blasoxide)](https://crates.io/crates/blasoxide)
[![Released API docs](https://docs.rs/blasoxide/badge.svg)](https://docs.rs/blasoxide)

BLAS implementation in rust

### Architecture

Only Level1 functions and micro kernels are optimized with platform specific code.

Optimizations are split into submodules and used statically if appropriate `target_feature`s are present at compile time.

If there are no `target_feature`s at compile time, generic code is compiled, generic code checks optimization support at runtime
and calls best possible optimization level.

Level3 functions are parallelized with rayon.

### Supported CPUs
These cpus have optimized implementations for them

- x86_64 cpus with fma support
