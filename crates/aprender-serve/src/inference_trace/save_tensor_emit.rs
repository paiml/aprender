//! `maybe_save_stage` — DRY helper for `forward_traced_with_save_tensor`.
//!
//! Centralizes the boilerplate that the wrapper currently inlines twice
//! (Embedding step 1, LmHead step 2). When the SHIP-007 PR-C-real
//! step-3+ surgery threads `Option<&SaveTensorPlan>` through
//! `AprTransformer::forward_traced` itself, every per-layer stage will
//! call this single function instead of repeating the
//! `should_save → ensure_layer_dir → File::create → write_tensor_file →
//! flush` chain at each capture point.
//!
//! Contract: [`contracts/apr-cli-trace-save-tensor-v1.yaml`] —
//! `apr_diff_values_compat`, `byte_format`, `determinism` invariants
//! all flow through this single function so any future change to the
//! file layout has exactly one place to land.

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::inference_trace::save_tensor::{write_tensor_file, WHOLE_MODEL_LAYER};
use crate::inference_trace::save_tensor_paths::ensure_layer_dir;
use crate::inference_trace::save_tensor_plan::SaveTensorPlan;
use crate::inference_trace::save_tensor_stage::SaveTensorStage;

/// Save the F32 buffer for `(stage, layer)` to disk **iff** the supplied
/// plan selects it. Otherwise this is a cheap no-op.
///
/// - When `plan` is `None`: returns immediately. Used at every capture
///   point so a forward pass without `--save-tensor` pays only the
///   `Option` discriminant cost.
/// - When `plan` is `Some` but the stage/layer is not selected: also a
///   no-op (the plan's own `should_save` filtering does the work).
/// - When the stage IS selected: ensures the parent directory, opens
///   `<output_dir>[/layer-N]/<stage>.bin`, writes the 12-byte APRT header
///   followed by the f32 LE body, and flushes the BufWriter.
///
/// For per-layer stages, the file's header layer field equals the call
/// argument `layer`. For whole-model stages (Final
/// Norm, LmHead) the header carries [`WHOLE_MODEL_LAYER`] and the file
/// lands directly under `output_dir/` with no `layer-N/` subdirectory —
/// matching the `byte_format` and `cli_signature` invariants in the
/// `apr-cli-trace-save-tensor-v1` contract.
///
/// # Errors
///
/// Returns the underlying [`std::io::Error`] from any failing
/// step (mkdir, create, write, flush). Callers in `forward_traced`
/// typically wrap this in `RealizarError::IoError`.
pub fn maybe_save_stage(
    plan: Option<&SaveTensorPlan>,
    stage: SaveTensorStage,
    layer: u32,
    values: &[f32],
) -> std::io::Result<()> {
    let Some(plan) = plan else {
        return Ok(());
    };
    if !plan.should_save(stage, layer) {
        return Ok(());
    }
    write_stage_file(&plan.output_dir, stage, layer, values)
}

/// Lower-level helper that performs the unconditional write — the body
/// of `maybe_save_stage` after the gating checks. Exposed separately so
/// tests and any callers that have already done their own `should_save`
/// filtering can avoid the second indirection.
///
/// # Errors
///
/// Forwards [`std::io::Error`] from [`std::fs::create_dir_all`],
/// [`std::fs::File::create`], or the underlying buffered write/flush.
pub fn write_stage_file(
    output_dir: &Path,
    stage: SaveTensorStage,
    layer: u32,
    values: &[f32],
) -> std::io::Result<()> {
    // Per-layer stages live at <root>/layer-N/<stage>.bin; whole-model
    // stages live at <root>/<stage>.bin (no per-layer subdir).
    let dir_layer = if stage.is_per_layer() {
        layer
    } else {
        WHOLE_MODEL_LAYER
    };
    ensure_layer_dir(output_dir, dir_layer)?;

    // The plan owns path construction so the file layout stays in
    // sync with `apr diff --stage` (PR-D, #1413).
    let path = output_path_for(output_dir, stage, layer);
    let file = std::fs::File::create(&path)?;
    let mut writer = BufWriter::new(file);
    // Header layer is identical to dir_layer (per-layer index, or sentinel).
    write_tensor_file(&mut writer, dir_layer, values)?;
    writer.flush()?;
    Ok(())
}

/// Compute the on-disk path for `(stage, layer)` without going through
/// the plan. Mirrors `SaveTensorPlan::stage_path` but takes a raw
/// `output_dir`, so callers in tests and the future
/// `forward_traced_inner` plumbing don't need a full plan to construct
/// the same path.
fn output_path_for(output_dir: &Path, stage: SaveTensorStage, layer: u32) -> std::path::PathBuf {
    use crate::inference_trace::save_tensor_paths::output_path;

    let header_layer = if stage.is_per_layer() {
        layer
    } else {
        WHOLE_MODEL_LAYER
    };
    output_path(output_dir, header_layer, stage.canonical_name())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn read_back(path: &Path) -> Vec<u8> {
        std::fs::read(path).expect("file exists")
    }

    fn parse_header_bytes(bytes: &[u8]) -> (u32, u32) {
        assert!(bytes.len() >= 12, "header is 12 bytes");
        assert_eq!(&bytes[0..4], b"APRT", "magic must be APRT");
        let layer = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let dim = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        (layer, dim)
    }

    #[test]
    fn maybe_save_with_none_plan_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let values = vec![1.0_f32, 2.0, 3.0];
        let r = maybe_save_stage(None, SaveTensorStage::Embedding, 0, &values);
        assert!(r.is_ok());
        // No file should have been created.
        let entries: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().collect();
        assert!(entries.is_empty(), "None plan must not create any files");
    }

    #[test]
    fn maybe_save_with_unselected_stage_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = SaveTensorPlan::from_cli("embedding", "0..1", tmp.path().to_path_buf()).unwrap();
        // Plan selects Embedding only; calling for FfnGate must be a no-op.
        let values = vec![9.9_f32];
        let r = maybe_save_stage(Some(&plan), SaveTensorStage::FfnGate, 0, &values);
        assert!(r.is_ok());
        assert!(!tmp.path().join("layer-0").join("ffn_gate.bin").exists());
    }

    #[test]
    fn maybe_save_per_layer_writes_to_layer_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = SaveTensorPlan::from_cli("embedding", "0..1", tmp.path().to_path_buf()).unwrap();
        let values = vec![1.5_f32, 2.5, 3.5, -4.0];
        maybe_save_stage(Some(&plan), SaveTensorStage::Embedding, 0, &values).unwrap();

        let path = tmp.path().join("layer-0").join("embedding.bin");
        assert!(
            path.exists(),
            "per-layer stage lands in layer-N/<stage>.bin"
        );
        let bytes = read_back(&path);
        let (layer, dim) = parse_header_bytes(&bytes);
        assert_eq!(layer, 0);
        assert_eq!(dim, values.len() as u32);
    }

    #[test]
    fn maybe_save_whole_model_writes_to_root_with_sentinel() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = SaveTensorPlan::from_cli("lm_head", "0..1", tmp.path().to_path_buf()).unwrap();
        let values = vec![0.1_f32, 0.2, 0.3];
        // Caller passes an arbitrary "layer" arg; whole-model stages
        // ignore it and always use the sentinel.
        maybe_save_stage(Some(&plan), SaveTensorStage::LmHead, 42, &values).unwrap();

        let path = tmp.path().join("lm_head.bin");
        assert!(
            path.exists(),
            "whole-model stage lands at <root>/<stage>.bin"
        );
        // Per-layer subdir must NOT exist for whole-model stages.
        assert!(!tmp.path().join("layer-42").exists());
        let bytes = read_back(&path);
        let (layer, dim) = parse_header_bytes(&bytes);
        assert_eq!(
            layer, WHOLE_MODEL_LAYER,
            "whole-model header carries the 0xFFFFFFFF sentinel regardless of caller's `layer` arg"
        );
        assert_eq!(dim, values.len() as u32);
    }

    #[test]
    fn maybe_save_layer_filter_excludes_out_of_range() {
        let tmp = tempfile::tempdir().unwrap();
        // Plan selects layers 0..2; a write at layer 5 must be a no-op.
        let plan = SaveTensorPlan::from_cli("ffn_gate", "0..2", tmp.path().to_path_buf()).unwrap();
        let values = vec![1.0_f32];
        maybe_save_stage(Some(&plan), SaveTensorStage::FfnGate, 5, &values).unwrap();
        assert!(!tmp.path().join("layer-5").join("ffn_gate.bin").exists());
        // ...but layer 1 should write.
        maybe_save_stage(Some(&plan), SaveTensorStage::FfnGate, 1, &values).unwrap();
        assert!(tmp.path().join("layer-1").join("ffn_gate.bin").exists());
    }

    #[test]
    fn maybe_save_writes_bytes_verbatim_including_nans() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = SaveTensorPlan::from_cli("embedding", "0..1", tmp.path().to_path_buf()).unwrap();
        // Round-trip pin: NaN + Inf + signed zero must be preserved bit-exactly.
        let values = vec![1.0_f32, f32::NAN, -0.0, f32::INFINITY];
        maybe_save_stage(Some(&plan), SaveTensorStage::Embedding, 0, &values).unwrap();

        let bytes = read_back(&tmp.path().join("layer-0").join("embedding.bin"));
        let body = &bytes[12..];
        assert_eq!(body.len(), values.len() * 4);
        for (i, expected) in values.iter().enumerate() {
            let read = f32::from_le_bytes([
                body[i * 4],
                body[i * 4 + 1],
                body[i * 4 + 2],
                body[i * 4 + 3],
            ]);
            assert_eq!(
                read.to_bits(),
                expected.to_bits(),
                "bit-identical round-trip required at index {i}"
            );
        }
    }

    #[test]
    fn write_stage_file_creates_missing_parent_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        // Use a deeply nested output dir that does not exist yet.
        let nested: PathBuf = tmp.path().join("a").join("b").join("c");
        let values = vec![1.0_f32];
        write_stage_file(&nested, SaveTensorStage::Embedding, 7, &values).unwrap();
        assert!(nested.join("layer-7").join("embedding.bin").exists());
    }
}
