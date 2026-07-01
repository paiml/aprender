//! Dequant-coupled APR v2 tests (issue #2231).
//!
//! These tests exercise the *dequantizing* `get_tensor_as_f32` accessor (the
//! `AprV2DequantExt` extension re-attached in `aprender-core`), the f16-scaled
//! APR-Q4 `dequantize_q4` helper, and the core-only `test_factory` pygmy
//! builders. They were SEVERED from the sovereign `apr-format` leaf (which can
//! only test pure container I/O) and live here next to the extension.
//!
//! This module reproduces the original flat `v2::tests` symbol scope that the
//! moved files reach via `super::*` / `super::super::*`: the leaf v2 namespace
//! plus the dequant accessor + helper.

#![cfg(test)]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp,
    unused_imports
)]

// Re-create the flat-scope surface the moved test files expect from `super::*`
// (the original `v2::tests` scope: the leaf v2 namespace + the formerly
// flat-scope crc32 / f16 helpers + the dequant accessor & helper).
pub(crate) use crate::format::dequant_ext::{dequantize_q4, AprV2DequantExt};
pub(crate) use crate::format::v2::*;
pub(crate) use apr_format::crc32;
pub(crate) use apr_format::f16::{f16_to_f32, f32_to_f16};

// The dequant-coupled test sub-chain (mirrors the original `v2::tests`
// declarations, now rooted here instead of in the leaf).
#[path = "tests_shard_provenance.rs"]
mod tests_shard_provenance;

#[path = "tests_pygmy_tensors.rs"]
mod tests_pygmy_tensors;

#[path = "tests_q6k_ref_reader.rs"]
mod tests_q6k_ref_reader;
