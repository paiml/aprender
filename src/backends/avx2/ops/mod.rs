//! AVX2 SIMD operation implementations.
//!
//! This module contains the actual SIMD implementations, isolated from the
//! trait dispatch in the parent module.

pub(super) mod arithmetic;
pub(super) mod reductions;
