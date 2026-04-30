//! Composition helpers for `apr trace --save-tensor`.
//!
//! Contract: [`contracts/apr-cli-trace-save-tensor-v1.yaml`] v1.0.0 (PROPOSED).
//!
//! This module provides one-call wrappers that combine the byte-format
//! primitives in [`super::save_tensor`] with the directory-layout helpers in
//! [`super::save_tensor_paths`]. The future `apr trace --save-tensor` CLI
//! implementation can call these directly, one per (layer, stage) tuple,
//! without managing file handles or paths separately.
//!
//! ## Why a thin wrapper layer
//!
//! Lower-level modules expose `Read`/`Write` trait objects (good for in-memory
//! tests, sockets, compression streams), while the typical usage is path-
//! based. Keeping the path-based ergonomic API on its own module:
//!
//! 1. Lets the CLI layer call one function per stage (no boilerplate).
//! 2. Lets `apr diff --values` call [`read_stage_file`] symmetrically.
//! 3. Pins the writer ↔ reader ↔ layout invariants in tests so the future
//!    CLI-wiring PR cannot drift the on-disk encoding.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use super::save_tensor::{read_tensor_file, write_tensor_file, ReadError, TensorHeader};
use super::save_tensor_paths::{ensure_layer_dir, output_path};

/// Errors that can arise from a one-shot save-tensor write.
#[derive(Debug, thiserror::Error)]
pub enum WriteStageError {
    /// I/O failure (open, write, flush).
    #[error("save-tensor write I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Write a single save-tensor file to `<dir>/[layer-<N>/]<stage_name>.bin`.
///
/// One-call wrapper:
/// 1. `ensure_layer_dir(dir, layer)` — creates the parent directory chain.
/// 2. `output_path(dir, layer, stage_name)` — computes the file path.
/// 3. Opens a `BufWriter<File>` at that path (truncating any prior content).
/// 4. `write_tensor_file(writer, layer, values)` — writes header + body.
///
/// Returns the resolved file path on success so callers can log it or pass
/// it to `apr diff --values`.
///
/// # Errors
///
/// Returns [`WriteStageError::Io`] for `mkdir`, `open`, `write`, or `flush`
/// failures.
///
/// # Panics
///
/// Panics if `values.len() > u32::MAX` — same constraint as
/// [`super::save_tensor::write_tensor_file`].
pub fn write_stage_file(
    dir: &Path,
    layer: u32,
    stage_name: &str,
    values: &[f32],
) -> Result<PathBuf, WriteStageError> {
    ensure_layer_dir(dir, layer)?;
    let path = output_path(dir, layer, stage_name);
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);
    write_tensor_file(&mut writer, layer, values)?;
    // BufWriter::drop flushes lazily — make the write durable before returning
    // so the path is safe to hand to a downstream reader (e.g. apr diff).
    use std::io::Write as _;
    writer.flush()?;
    Ok(path)
}

/// Read a save-tensor file from `path`.
///
/// One-call wrapper around [`super::save_tensor::read_tensor_file`] that
/// opens the file behind a [`BufReader`].
///
/// # Errors
///
/// Returns [`ReadError`] for I/O, header-parse, or body-length failures.
pub fn read_stage_file(path: &Path) -> Result<(TensorHeader, Vec<f32>), ReadError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    read_tensor_file(&mut reader)
}

#[cfg(test)]
mod tests {
    use super::super::save_tensor::WHOLE_MODEL_LAYER;
    use super::*;

    #[test]
    fn write_stage_file_per_layer_roundtrip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let values: Vec<f32> = vec![0.5, -1.25, 3.14, -0.0, 0.0, 7.7];

        let path =
            write_stage_file(tmp.path(), 3, "ffn_gate", &values).expect("write must succeed");
        assert_eq!(path, tmp.path().join("layer-3/ffn_gate.bin"));
        assert!(path.is_file());

        let (header, read_values) = read_stage_file(&path).expect("read must succeed");
        assert_eq!(header.layer, 3);
        assert_eq!(header.dim_product as usize, values.len());
        assert_eq!(read_values, values);
    }

    #[test]
    fn write_stage_file_whole_model_roundtrip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();

        let path = write_stage_file(tmp.path(), WHOLE_MODEL_LAYER, "lm_head", &values)
            .expect("write must succeed");
        assert_eq!(path, tmp.path().join("lm_head.bin"));
        assert!(!path.parent().unwrap().to_str().unwrap().contains("layer-"));

        let (header, read_values) = read_stage_file(&path).expect("read must succeed");
        assert!(header.is_whole_model());
        assert_eq!(read_values, values);
    }

    #[test]
    fn write_stage_file_creates_missing_parent() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let nested = tmp.path().join("deep/run-0042");
        // nested does NOT exist yet; write_stage_file must create it.
        assert!(!nested.exists());

        let path = write_stage_file(&nested, 0, "embedding", &[1.0_f32, 2.0, 3.0])
            .expect("write must create parents");
        assert!(path.is_file());
        assert!(nested.join("layer-0").is_dir());
    }

    #[test]
    fn write_stage_file_truncates_existing() {
        let tmp = tempfile::tempdir().expect("tempdir");

        // First write: 4 elements.
        let path = write_stage_file(tmp.path(), 0, "ffn_gate", &[1.0_f32, 2.0, 3.0, 4.0])
            .expect("first write");
        let (h1, v1) = read_stage_file(&path).expect("read first");
        assert_eq!(h1.dim_product, 4);
        assert_eq!(v1.len(), 4);

        // Second write to same path: 2 elements. Must overwrite, not append.
        let _ = write_stage_file(tmp.path(), 0, "ffn_gate", &[9.0_f32, 8.0]).expect("second write");
        let (h2, v2) = read_stage_file(&path).expect("read second");
        assert_eq!(h2.dim_product, 2);
        assert_eq!(v2, vec![9.0_f32, 8.0]);
    }

    #[test]
    fn write_stage_file_zero_length_tensor() {
        // Per byte-format invariant, an empty tensor must still produce a
        // valid 12-byte header file.
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_stage_file(tmp.path(), 0, "embedding", &[]).expect("empty write");
        let (header, values) = read_stage_file(&path).expect("empty read");
        assert_eq!(header.dim_product, 0);
        assert!(values.is_empty());
        let metadata = std::fs::metadata(&path).expect("stat");
        assert_eq!(metadata.len(), 12, "empty tensor file is exactly 12 bytes");
    }

    #[test]
    fn write_stage_file_preserves_nan_inf() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let values: Vec<f32> = vec![f32::NAN, 1.0, f32::INFINITY, f32::NEG_INFINITY, -0.0];
        let path = write_stage_file(tmp.path(), 0, "ffn_silu", &values).expect("write");
        let (_, read_values) = read_stage_file(&path).expect("read");

        assert!(read_values[0].is_nan(), "NaN preserved");
        assert_eq!(read_values[1], 1.0);
        assert!(read_values[2].is_infinite() && read_values[2].is_sign_positive());
        assert!(read_values[3].is_infinite() && read_values[3].is_sign_negative());
        // -0.0 round-trips bit-identically (sign bit preserved).
        assert_eq!(read_values[4].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn write_stage_file_header_has_expected_magic_and_layer() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_stage_file(tmp.path(), 7, "attention", &[1.0_f32]).expect("write");
        let bytes = std::fs::read(&path).expect("read raw bytes");

        // First 4 bytes: magic
        assert_eq!(&bytes[0..4], b"APRT");
        // Bytes 4-7: layer = 7 LE
        assert_eq!(&bytes[4..8], &7u32.to_le_bytes());
        // Bytes 8-11: dim_product = 1 LE
        assert_eq!(&bytes[8..12], &1u32.to_le_bytes());
        // Bytes 12-15: f32 LE 1.0
        assert_eq!(&bytes[12..16], &1.0_f32.to_le_bytes());
    }

    #[test]
    fn read_stage_file_propagates_missing_path() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let result = read_stage_file(&tmp.path().join("does_not_exist.bin"));
        assert!(result.is_err(), "missing file must error");
    }

    #[test]
    fn write_stage_file_returns_resolved_path_for_logging() {
        // Per docs: caller can log the returned path or pass to apr diff.
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_stage_file(tmp.path(), 5, "ffn_swigl", &[1.0_f32]).expect("write");
        // Must be the same path we'd compute via output_path() — invariant
        // that downstream tooling (apr diff --values) relies on.
        assert_eq!(path, output_path(tmp.path(), 5, "ffn_swigl"));
    }

    #[test]
    fn write_then_read_three_stages_in_one_layer() {
        // Mirrors the FALSIFY-APR-TRACE-SAVE-005 scenario: three stages in one
        // run produce three distinct files under the same layer-N directory.
        let tmp = tempfile::tempdir().expect("tempdir");
        let layer = 0;

        write_stage_file(tmp.path(), layer, "embedding", &[1.0_f32]).expect("embedding");
        write_stage_file(tmp.path(), layer, "ffn_gate", &[2.0_f32]).expect("ffn_gate");
        write_stage_file(tmp.path(), layer, "ffn_swigl", &[3.0_f32]).expect("ffn_swigl");

        let layer_dir = tmp.path().join("layer-0");
        assert!(layer_dir.join("embedding.bin").is_file());
        assert!(layer_dir.join("ffn_gate.bin").is_file());
        assert!(layer_dir.join("ffn_swigl.bin").is_file());

        // Read each back and verify values.
        let (_, e) = read_stage_file(&layer_dir.join("embedding.bin")).expect("read e");
        let (_, g) = read_stage_file(&layer_dir.join("ffn_gate.bin")).expect("read g");
        let (_, s) = read_stage_file(&layer_dir.join("ffn_swigl.bin")).expect("read s");
        assert_eq!(e, vec![1.0_f32]);
        assert_eq!(g, vec![2.0_f32]);
        assert_eq!(s, vec![3.0_f32]);
    }
}
