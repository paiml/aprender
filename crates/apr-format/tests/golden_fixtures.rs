#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp
)]
//! Golden-fixture regression tests for the sovereign `apr-format` leaf (#2231).
//!
//! `golden_v1.apr` / `golden_v2.apr` were captured by the temporary harness in
//! `aprender-core` (`format/golden_capture_tmp.rs`) while the format code still
//! lived in core — i.e. they are the byte-identity oracle produced by the
//! pre-extraction code. These tests prove the EXTRACTED leaf reads the SAME bytes.
//!
//! Stage 1 scope: the v1 (`APRN`) slice is moved, so the leaf reads `golden_v1.apr`.
//! The v2 (`APR\0`) reader lands in Stage 2; here we only assert the v2 fixture
//! exists and carries the `APR\0` magic so the oracle is pinned for Stage 2.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct GoldenModel {
    name: String,
    weights: Vec<f32>,
    bias: f32,
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

#[test]
fn golden_v1_loads_in_leaf() {
    let path = fixtures().join("golden_v1.apr");
    let model: GoldenModel =
        apr_format::load(&path, apr_format::ModelType::LinearRegression).expect("load golden v1");
    assert_eq!(model.name, "golden_v1");
    assert_eq!(
        model.weights,
        vec![1.0, 2.0, 0.5, -0.5, 4.0, -2.0, 0.25, 8.0]
    );
    assert_eq!(model.bias, 0.125);
}

#[test]
fn golden_v1_crc_and_header_are_consistent() {
    // The leaf's crc32 must validate the core-produced trailer (CRC-integrity).
    let bytes = std::fs::read(fixtures().join("golden_v1.apr")).expect("read v1");
    assert!(bytes.len() > apr_format::HEADER_SIZE + 4);
    let stored = u32::from_le_bytes([
        bytes[bytes.len() - 4],
        bytes[bytes.len() - 3],
        bytes[bytes.len() - 2],
        bytes[bytes.len() - 1],
    ]);
    let computed = apr_format::crc32(&bytes[..bytes.len() - 4]);
    assert_eq!(
        stored, computed,
        "leaf crc32 must match the core-written trailer"
    );

    let header = apr_format::Header::from_bytes(&bytes[..apr_format::HEADER_SIZE]).expect("hdr");
    assert_eq!(header.magic, apr_format::MAGIC);
    assert_eq!(header.quality_score, 85);
}

#[test]
fn golden_v2_fixture_is_pinned_for_stage2() {
    // v2 reader is Stage 2; assert the fixture exists with the APR\0 magic so the
    // byte-identity oracle is captured now (while format is still in core).
    let bytes = std::fs::read(fixtures().join("golden_v2.apr")).expect("read v2");
    assert_eq!(&bytes[0..4], &[0x41, 0x50, 0x52, 0x00], "APR\\0 magic");
    assert_eq!(bytes.len(), 1092, "v2 golden size pinned");
}
