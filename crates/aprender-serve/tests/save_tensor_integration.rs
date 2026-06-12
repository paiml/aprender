//! Integration tests for `apr trace --save-tensor` byte format and layout.
//!
//! Contract: [`contracts/apr-cli-trace-save-tensor-v1.yaml`] v1.0.0 (PROPOSED).
//!
//! Exercises the public API exposed by:
//! - `realizar::inference_trace::save_tensor` (#1133 byte format primitives)
//! - `realizar::inference_trace::save_tensor_paths` (#1135 directory layout)
//!
//! These integration tests are the *external* mirror of the unit tests in
//! the same modules. They run against the public API exactly as a future
//! `apr trace --save-tensor` CLI implementation will, and as `apr diff
//! --values` will when reading the produced files. Catching regressions at
//! the public-API surface (in addition to the unit tests' internal-state
//! assertions) prevents the writer/reader pair from drifting silently.
//!
//! ## Discharge map
//!
//! | Falsifier | Discharge level | Test |
//! |-----------|-----------------|------|
//! | FALSIFY-APR-TRACE-SAVE-002 (determinism) | partial | `falsify_apr_trace_save_002_byte_determinism_two_writes` |
//! | FALSIFY-APR-TRACE-SAVE-004 (header self-describing) | partial | `falsify_apr_trace_save_004_header_format_via_public_api` |
//! | FALSIFY-APR-TRACE-SAVE-005 (multi-stage in one run) | partial | `falsify_apr_trace_save_005_three_stages_one_layer_independent_files` |
//!
//! "Partial" because full discharge requires the live CLI implementation
//! that calls these helpers from inside the forward pass.

use std::path::Path;

use realizar::inference_trace::save_tensor::{
    self, parse_header, HEADER_SIZE, MAGIC, WHOLE_MODEL_LAYER,
};
use realizar::inference_trace::save_tensor_paths::{ensure_layer_dir, output_path};

/// Helper: write a tensor file using the merged-as-of-#1133 public API.
///
/// Mirrors the future CLI flow:
///   1. ensure_layer_dir (paths)
///   2. compute output_path (paths)
///   3. open File at path
///   4. save_tensor::write_tensor_file (bytes)
///   5. flush
fn write_via_public_api(
    dir: &Path,
    layer: u32,
    stage_name: &str,
    values: &[f32],
) -> std::path::PathBuf {
    use std::io::Write;
    ensure_layer_dir(dir, layer).expect("ensure_layer_dir must succeed");
    let path = output_path(dir, layer, stage_name);
    let file = std::fs::File::create(&path).expect("create file");
    let mut writer = std::io::BufWriter::new(file);
    save_tensor::write_tensor_file(&mut writer, layer, values).expect("write_tensor_file");
    writer.flush().expect("flush");
    path
}

#[test]
fn falsify_apr_trace_save_002_byte_determinism_two_writes() {
    // Per FALSIFY-APR-TRACE-SAVE-002: two writes with the same inputs MUST
    // produce byte-identical files. This is structural (no random padding,
    // f32 LE is fixed, magic + dim_product are pure functions of inputs).
    let tmp = tempfile::tempdir().expect("tempdir");
    let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.0625).collect();

    let path_a = write_via_public_api(&tmp.path().join("run_a"), 3, "ffn_gate", &values);
    let path_b = write_via_public_api(&tmp.path().join("run_b"), 3, "ffn_gate", &values);

    let bytes_a = std::fs::read(&path_a).expect("read A");
    let bytes_b = std::fs::read(&path_b).expect("read B");
    assert_eq!(
        bytes_a, bytes_b,
        "FALSIFIED APR-TRACE-SAVE-002: identical inputs produced different bytes"
    );
}

#[test]
fn falsify_apr_trace_save_004_header_format_via_public_api() {
    // Per FALSIFY-APR-TRACE-SAVE-004: the 12-byte header must be self-
    // describing. Verify by reading raw file bytes (NOT via the convenience
    // reader) and feeding them to parse_header.
    let tmp = tempfile::tempdir().expect("tempdir");
    let values: Vec<f32> = vec![1.5_f32, -2.5, 0.0, 100.0];

    let path = write_via_public_api(tmp.path(), 7, "attention", &values);
    let bytes = std::fs::read(&path).expect("read");

    assert!(
        bytes.len() >= HEADER_SIZE,
        "file must contain at least the 12-byte header"
    );
    assert_eq!(&bytes[0..4], &MAGIC, "first 4 bytes must be the APRT magic");

    let header = parse_header(&bytes[..HEADER_SIZE]).expect("parse_header");
    assert_eq!(header.layer, 7);
    assert_eq!(header.dim_product as usize, values.len());
    assert!(!header.is_whole_model());
    assert_eq!(header.total_file_size(), HEADER_SIZE + values.len() * 4);
    assert_eq!(
        bytes.len(),
        header.total_file_size(),
        "actual file size must match header.total_file_size()"
    );

    // Verify body bytes decode element-wise as f32 LE.
    for (i, &expected) in values.iter().enumerate() {
        let off = HEADER_SIZE + i * 4;
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[off..off + 4]);
        assert_eq!(
            f32::from_le_bytes(buf),
            expected,
            "body element {i} must round-trip"
        );
    }
}

#[test]
fn falsify_apr_trace_save_005_three_stages_one_layer_independent_files() {
    // Per FALSIFY-APR-TRACE-SAVE-005: a multi-stage capture run must
    // produce 3 distinct files per layer when stages = embedding, ffn_gate,
    // ffn_swigl. Verify at the filesystem level via the public API.
    let tmp = tempfile::tempdir().expect("tempdir");
    let layer = 0;

    write_via_public_api(tmp.path(), layer, "embedding", &[1.0_f32, 2.0]);
    write_via_public_api(tmp.path(), layer, "ffn_gate", &[3.0_f32, 4.0]);
    write_via_public_api(tmp.path(), layer, "ffn_swigl", &[5.0_f32, 6.0]);

    let layer_dir = tmp.path().join("layer-0");
    let entries: std::collections::HashSet<String> = std::fs::read_dir(&layer_dir)
        .expect("read_dir")
        .map(|e| e.expect("entry").file_name().into_string().expect("utf8"))
        .collect();

    assert!(entries.contains("embedding.bin"));
    assert!(entries.contains("ffn_gate.bin"));
    assert!(entries.contains("ffn_swigl.bin"));
    assert_eq!(
        entries.len(),
        3,
        "exactly 3 files in layer-0/ for this 3-stage run"
    );

    // Each file must have its own correct dim_product (not, e.g., the last
    // write's dim_product applied to all three).
    for stage in ["embedding", "ffn_gate", "ffn_swigl"] {
        let bytes = std::fs::read(layer_dir.join(format!("{stage}.bin"))).expect("read");
        let header = parse_header(&bytes[..HEADER_SIZE]).expect("parse");
        assert_eq!(
            header.dim_product, 2,
            "{stage} dim_product must equal its own values.len() (= 2)"
        );
        assert_eq!(header.layer, 0);
    }
}

#[test]
fn whole_model_stages_dont_collide_with_per_layer_zero() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // Whole-model file at <dir>/lm_head.bin
    let whole = write_via_public_api(tmp.path(), WHOLE_MODEL_LAYER, "lm_head", &[1.0_f32]);
    // Per-layer file at <dir>/layer-0/lm_head.bin
    let per_layer = write_via_public_api(tmp.path(), 0, "lm_head", &[2.0_f32, 3.0]);

    assert_ne!(whole, per_layer);
    assert!(whole.is_file() && per_layer.is_file());

    // Read both back and verify each preserved its own values.
    let whole_bytes = std::fs::read(&whole).expect("read whole");
    let pl_bytes = std::fs::read(&per_layer).expect("read per-layer");

    let whole_header = parse_header(&whole_bytes[..HEADER_SIZE]).expect("parse whole");
    let pl_header = parse_header(&pl_bytes[..HEADER_SIZE]).expect("parse per-layer");

    assert!(whole_header.is_whole_model());
    assert!(!pl_header.is_whole_model());
    assert_eq!(whole_header.dim_product, 1);
    assert_eq!(pl_header.dim_product, 2);
}

#[test]
fn parse_header_on_truncated_file_errors_via_public_api() {
    // Defense against a future bug where the file is opened but truncated
    // (filesystem corruption, partial write, etc.). parse_header must error
    // cleanly rather than panic or silently zero-fill.
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("corrupted.bin");
    // Write only 8 bytes — less than the 12-byte header.
    std::fs::write(&path, [b'A', b'P', b'R', b'T', 0, 0, 0, 0]).expect("write partial");

    let bytes = std::fs::read(&path).expect("read");
    let result = parse_header(&bytes);
    assert!(
        result.is_err(),
        "parse_header on 8-byte file must error, not panic"
    );
}
