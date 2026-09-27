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

/// The v2 F32 metadata + tensors both writer goldens encode.
fn golden_v2_writer_output() -> Vec<u8> {
    use apr_format::v2::{AprV2Metadata, AprV2Writer};

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
    writer.write().expect("v2 write")
}

#[test]
fn golden_v2_f32_writer_is_byte_identical() {
    // v2 byte-identity (F32 scope, issue #2231): re-writing the same F32 tensors
    // + pinned metadata via the leaf's AprV2Writer reproduces the writer golden
    // byte-for-byte. F32 payload is unaffected by the f16 IEEE-RNE write change.
    //
    // The writer golden is `golden_v2_skip_none.apr`, NOT `golden_v2.apr` (#4190).
    // #2254 (625ef84ec, 2026-07-02) made AprV2Metadata skip serializing `None`
    // fields (except the three C-APR-PROVENANCE keys, pinned to explicit null by
    // FALSIFY-SHIP-022) to fix realizar's duplicate-field poison. That shrank this
    // file from 1092 to 516 bytes, one day after `golden_v2.apr` was captured, and
    // this test failed from then on unseen: the target was not on ci.yml's
    // `--test` line. `golden_v2.apr` stays as the pre-#2254 READ-compat oracle
    // (`golden_v2_loads_in_leaf`), and the new golden is not taken on trust:
    // `golden_v2_skip_none_is_golden_v2_minus_nulls` derives it from the old one.
    let golden = std::fs::read(fixtures().join("golden_v2_skip_none.apr")).expect("read v2");
    let produced = golden_v2_writer_output();

    assert_eq!(
        produced.len(),
        golden.len(),
        "v2 F32 byte length drifted from golden_v2_skip_none.apr"
    );
    assert_eq!(
        produced, golden,
        "leaf AprV2Writer F32 output is NOT byte-identical to golden_v2_skip_none.apr"
    );
}

/// `(metadata JSON, tensor index, tensor data without the footer CRC)` of a v2
/// file, sliced by the offsets its own header records.
fn v2_sections(bytes: &[u8]) -> (serde_json::Value, &[u8], &[u8]) {
    let reader = apr_format::v2::AprV2Reader::from_bytes(bytes).expect("parse v2");
    let h = reader.header();
    assert!(h.verify_checksum(), "v2 header CRC valid");
    let footer = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().expect("4 bytes"));
    assert_eq!(
        footer,
        apr_format::crc32(&bytes[..bytes.len() - 4]),
        "v2 footer CRC valid"
    );
    let meta_start = h.metadata_offset as usize;
    let meta = &bytes[meta_start..meta_start + h.metadata_size as usize];
    let index = &bytes[h.tensor_index_offset as usize..h.data_offset as usize];
    let data = &bytes[h.data_offset as usize..bytes.len() - 4];
    (
        serde_json::from_slice(meta).expect("metadata is JSON"),
        index,
        data,
    )
}

#[test]
fn golden_v2_skip_none_is_golden_v2_minus_nulls() {
    // The regeneration's written reason, checked (#4190). The post-#2254 golden is
    // the pre-#2254 golden with exactly one change: `null` metadata keys are gone,
    // except the three C-APR-PROVENANCE keys that must stay explicit nulls. The
    // tensor index and tensor data are byte-identical; only the offsets, sizes and
    // CRCs that follow from a shorter metadata block move. Any other difference
    // (a changed value, a dropped provenance key, different tensor bytes) is a
    // writer regression, and this test fails.
    const PROVENANCE: [&str; 3] = ["license", "data_source", "data_license"];
    let old = std::fs::read(fixtures().join("golden_v2.apr")).expect("read v2");
    let new = std::fs::read(fixtures().join("golden_v2_skip_none.apr")).expect("read v2");
    let (old_meta, old_index, old_data) = v2_sections(&old);
    let (new_meta, new_index, new_data) = v2_sections(&new);

    let mut expected = old_meta.as_object().expect("metadata object").clone();
    expected.retain(|k, v| !v.is_null() || PROVENANCE.contains(&k.as_str()));
    assert_eq!(
        serde_json::Value::Object(expected),
        new_meta,
        "new metadata must be the old metadata minus its non-provenance nulls"
    );
    for key in PROVENANCE {
        assert!(
            new_meta[key].is_null(),
            "{key} must serialize as explicit null"
        );
    }
    assert!(
        old_meta.as_object().expect("object").len() > new_meta.as_object().expect("object").len(),
        "the old golden really does carry the nulls #2254 drops"
    );
    assert_eq!(old_index, new_index, "tensor index bytes unchanged");
    assert_eq!(old_data, new_data, "tensor data bytes unchanged");

    // Header fields other than offsets / sizes / checksum are unchanged.
    let ro = apr_format::v2::AprV2Reader::from_bytes(&old).expect("parse old");
    let rn = apr_format::v2::AprV2Reader::from_bytes(&new).expect("parse new");
    let (ho, hn) = (ro.header(), rn.header());
    assert_eq!(
        (ho.magic, ho.version, ho.tensor_count, ho.metadata_offset),
        (hn.magic, hn.version, hn.tensor_count, hn.metadata_offset)
    );
    assert_eq!(ho.flags, hn.flags);
}
