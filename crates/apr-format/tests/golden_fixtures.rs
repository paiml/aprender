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
//! Stage 2 scope: the v1 (`APRN`) container AND the v2 (`APR\0`) container both
//! live in the leaf now, so the leaf reads `golden_v1.apr` AND `golden_v2.apr`
//! and round-trips their F32 tensors against the captured oracle bytes.

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
fn golden_v2_loads_in_leaf() {
    // Stage 2: the leaf's v2 reader parses the pre-extraction golden_v2.apr bytes
    // and round-trips its F32 tensors against the captured oracle values.
    let bytes = std::fs::read(fixtures().join("golden_v2.apr")).expect("read v2");
    assert_eq!(&bytes[0..4], &[0x41, 0x50, 0x52, 0x00], "APR\\0 magic");
    assert_eq!(bytes.len(), 1092, "v2 golden size pinned");

    let reader = apr_format::v2::AprV2Reader::from_bytes(&bytes).expect("leaf parses golden v2");
    assert!(reader.header().verify_checksum(), "v2 header CRC valid");
    assert_eq!(reader.metadata().model_type, "linear_regression");
    assert_eq!(reader.metadata().name.as_deref(), Some("golden-v2"));

    let mut names = reader.tensor_names();
    names.sort_unstable();
    assert_eq!(names, vec!["bias", "weights"]);

    // F32 tensors read back exactly (no dequant needed — leaf's get_f32_tensor).
    assert_eq!(reader.get_f32_tensor("bias").expect("bias"), vec![0.125]);
    assert_eq!(
        reader.get_f32_tensor("weights").expect("weights"),
        vec![1.0, 2.0, 0.5, -0.5, 4.0, -2.0, 0.25, 8.0]
    );
}

#[test]
fn golden_v2_f32_writer_is_byte_identical() {
    // v2 byte-identity (F32 scope, issue #2231): re-writing the same F32 tensors
    // + pinned metadata via the leaf's AprV2Writer reproduces golden_v2.apr
    // byte-for-byte. F32 payload is unaffected by the f16 IEEE-RNE write change.
    use apr_format::v2::{AprV2Metadata, AprV2Writer};

    let golden = std::fs::read(fixtures().join("golden_v2.apr")).expect("read v2");

    let mut metadata = AprV2Metadata::new("linear_regression");
    metadata.name = Some("golden-v2".to_string());
    metadata.version = Some("0.0.0-golden".to_string());
    metadata.created_at = Some("1700000000".to_string());
    metadata.param_count = 8;

    let mut writer = AprV2Writer::new(metadata);
    // Index is sorted by name on write; insertion order does not matter.
    writer.add_f32_tensor(
        "weights",
        vec![8],
        &[1.0, 2.0, 0.5, -0.5, 4.0, -2.0, 0.25, 8.0],
    );
    writer.add_f32_tensor("bias", vec![1], &[0.125]);
    let produced = writer.write().expect("v2 write");

    assert_eq!(
        produced.len(),
        golden.len(),
        "v2 F32 byte length drifted from golden_v2.apr"
    );
    assert_eq!(
        produced, golden,
        "leaf AprV2Writer F32 output is NOT byte-identical to golden_v2.apr"
    );
}
