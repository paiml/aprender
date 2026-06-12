//! Output path helpers for `apr trace --save-tensor`.
//!
//! Contract: [`contracts/apr-cli-trace-save-tensor-v1.yaml`] v1.0.0 (PROPOSED).
//!
//! ## Layout (per contract `cli_signature` invariants)
//!
//! Per-layer stages are written under `<DIR>/layer-<N>/<STAGE>.bin`; whole-
//! model stages (those marked with the [`super::save_tensor::WHOLE_MODEL_LAYER`]
//! sentinel — `final_norm`, `lm_head`) are written directly under
//! `<DIR>/<STAGE>.bin` with no `layer-<N>` segment.
//!
//! This module is the layout primitive that the future `apr trace
//! --save-tensor` CLI implementation calls before invoking the byte writer
//! in [`super::save_tensor`]. Keeping the path-building rules pure (no I/O,
//! no model state) lets `apr diff --values` use the same helper to *find*
//! files, so the writer and reader cannot drift apart.

use std::path::{Path, PathBuf};

use super::save_tensor::WHOLE_MODEL_LAYER;

/// Build the output path for a single save-tensor file.
///
/// Pure function — no I/O, does not touch the filesystem.
///
/// - `dir`: the `--output` directory passed to `apr trace --save-tensor`.
/// - `layer`: per-layer stage's decoder layer index, OR
///   [`WHOLE_MODEL_LAYER`] for whole-model stages.
/// - `stage_name`: the stage's canonical name from the contract
///   `cli_signature` enumeration (e.g. `"ffn_gate"`, `"final_norm"`). The
///   `.bin` extension is appended here.
///
/// # Examples
///
/// ```
/// # use realizar::inference_trace::save_tensor_paths::output_path;
/// # use realizar::inference_trace::save_tensor::WHOLE_MODEL_LAYER;
/// # use std::path::PathBuf;
/// // Per-layer stage:
/// assert_eq!(
///     output_path(&PathBuf::from("trace_out"), 3, "ffn_gate"),
///     PathBuf::from("trace_out/layer-3/ffn_gate.bin"),
/// );
/// // Whole-model stage:
/// assert_eq!(
///     output_path(&PathBuf::from("trace_out"), WHOLE_MODEL_LAYER, "lm_head"),
///     PathBuf::from("trace_out/lm_head.bin"),
/// );
/// ```
#[must_use]
pub fn output_path(dir: &Path, layer: u32, stage_name: &str) -> PathBuf {
    let file_name = format!("{stage_name}.bin");
    if layer == WHOLE_MODEL_LAYER {
        dir.join(file_name)
    } else {
        dir.join(format!("layer-{layer}")).join(file_name)
    }
}

/// Create the output parent directory for a save-tensor file if it does not
/// already exist.
///
/// Per-layer: ensures `<dir>/layer-<N>` exists.
/// Whole-model: ensures `<dir>` exists.
///
/// `--output DIR is created if missing; contents are NOT auto-cleaned`
/// per the contract. This helper performs only the create step.
///
/// # Errors
///
/// Forwards any [`std::io::Error`] from [`std::fs::create_dir_all`].
pub fn ensure_layer_dir(dir: &Path, layer: u32) -> std::io::Result<()> {
    if layer == WHOLE_MODEL_LAYER {
        std::fs::create_dir_all(dir)
    } else {
        std::fs::create_dir_all(dir.join(format!("layer-{layer}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_path_per_layer_layer_zero() {
        let p = output_path(Path::new("trace_out"), 0, "embedding");
        assert_eq!(p, PathBuf::from("trace_out/layer-0/embedding.bin"));
    }

    #[test]
    fn output_path_per_layer_arbitrary() {
        let p = output_path(Path::new("trace_out"), 3, "ffn_gate");
        assert_eq!(p, PathBuf::from("trace_out/layer-3/ffn_gate.bin"));
    }

    #[test]
    fn output_path_whole_model_no_layer_segment() {
        let p = output_path(Path::new("trace_out"), WHOLE_MODEL_LAYER, "lm_head");
        assert_eq!(p, PathBuf::from("trace_out/lm_head.bin"));
        let p = output_path(Path::new("trace_out"), WHOLE_MODEL_LAYER, "final_norm");
        assert_eq!(p, PathBuf::from("trace_out/final_norm.bin"));
    }

    #[test]
    fn output_path_absolute_dir() {
        let p = output_path(Path::new("/tmp/trace"), 5, "attention");
        assert_eq!(p, PathBuf::from("/tmp/trace/layer-5/attention.bin"));
    }

    #[test]
    fn output_path_layer_max_minus_one_is_per_layer() {
        // u32::MAX is the WHOLE_MODEL_LAYER sentinel; the largest valid
        // per-layer index is u32::MAX - 1.
        let edge = u32::MAX - 1;
        let p = output_path(Path::new("d"), edge, "ffn_silu");
        assert_eq!(p, PathBuf::from(format!("d/layer-{edge}/ffn_silu.bin")));
    }

    #[test]
    fn output_path_nested_relative_dir() {
        let p = output_path(Path::new("./out/run-0042"), 12, "qkv_matmul");
        assert_eq!(p, PathBuf::from("./out/run-0042/layer-12/qkv_matmul.bin"));
    }

    #[test]
    fn output_path_appends_bin_extension() {
        let p = output_path(Path::new("d"), 0, "ffn_swigl");
        assert_eq!(
            p.extension().and_then(|e| e.to_str()),
            Some("bin"),
            "all save-tensor files must use .bin extension"
        );
    }

    #[test]
    fn output_path_preserves_stage_name_verbatim() {
        // No case-folding, no aliasing — `output_path` is a thin formatter.
        // Caller (CLI layer) is expected to canonicalise via SaveTensorStage.
        let p = output_path(Path::new("d"), 0, "FFN_GATE");
        assert_eq!(
            p,
            PathBuf::from("d/layer-0/FFN_GATE.bin"),
            "no implicit case-fold; canonicalisation is caller's responsibility"
        );
    }

    #[test]
    fn ensure_layer_dir_creates_per_layer_dir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        ensure_layer_dir(tmp.path(), 7).expect("create per-layer");
        assert!(tmp.path().join("layer-7").is_dir());
    }

    #[test]
    fn ensure_layer_dir_creates_whole_model_dir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let nested = tmp.path().join("nested/output");
        ensure_layer_dir(&nested, WHOLE_MODEL_LAYER).expect("create whole-model");
        assert!(nested.is_dir());
        // No layer-* subdir for whole-model.
        let dir_entries: Vec<_> = std::fs::read_dir(&nested)
            .expect("read_dir")
            .collect::<Result<_, _>>()
            .expect("entries");
        assert!(dir_entries.is_empty());
    }

    #[test]
    fn ensure_layer_dir_is_idempotent() {
        let tmp = tempfile::tempdir().expect("tempdir");
        ensure_layer_dir(tmp.path(), 0).expect("first create");
        ensure_layer_dir(tmp.path(), 0).expect("second create must not fail");
        assert!(tmp.path().join("layer-0").is_dir());
    }

    #[test]
    fn ensure_layer_dir_creates_nested_parents() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let deep = tmp.path().join("a/b/c");
        ensure_layer_dir(&deep, 4).expect("create nested");
        assert!(deep.join("layer-4").is_dir());
    }

    #[test]
    fn ensure_layer_dir_no_collision_between_per_layer_and_whole_model() {
        // Whole-model output_path uses `<dir>/<stage>.bin`; per-layer uses
        // `<dir>/layer-<N>/<stage>.bin`. Verify the layouts cannot collide
        // by checking that the per-layer `layer-` prefix is never produced
        // for the whole-model case at any layer value (besides the sentinel
        // itself, which is handled).
        let p_whole = output_path(Path::new("d"), WHOLE_MODEL_LAYER, "lm_head");
        let p_per_layer = output_path(Path::new("d"), 0, "lm_head");
        assert_ne!(p_whole, p_per_layer);
        assert!(!p_whole.to_str().unwrap().contains("layer-"));
        assert!(p_per_layer.to_str().unwrap().contains("layer-0"));
    }
}
