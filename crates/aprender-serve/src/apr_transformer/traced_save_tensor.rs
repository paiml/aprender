//! SHIP-007 PR-C-real-step1: thin `forward_traced_with_save_tensor` wrapper.
//!
//! Contract: [`contracts/apr-cli-trace-save-tensor-v1.yaml`] v1.0.0 (PROPOSED).
//!
//! ## Role in the cascade
//!
//! - PR-A (#1405, MERGED) — `apr trace --save-tensor` clap surface
//! - PR-B (#1406, MERGED) — [`SaveTensorPlan`] plan-builder
//! - PR-B-prep (#1407, MERGED) — plan ↔ writer integration tests
//! - **This file** — public API surface that connects the two:
//!   [`AprTransformer::forward_traced_with_save_tensor`] delegates to
//!   [`AprTransformer::forward_traced`] and emits the **embedding** stage
//!   to disk if the plan selects it.
//!
//! ## Why "step 1" (just embedding)
//!
//! The embedding stage is the one APR forward stage that can be re-extracted
//! by calling `self.embed(token_ids)` a second time (cheap, deterministic) —
//! no internal forward-pass instrumentation required. This makes step 1
//! shippable without touching the 360-line `forward_traced` body.
//!
//! Subsequent SHIP-007 steps will thread `Option<&SaveTensorPlan>` through
//! `forward_traced` itself so the per-layer stages (qkv_matmul, ffn_gate,
//! …) emit during the single forward pass instead of requiring re-runs.
//!
//! ## Why the wrapper exists at all
//!
//! Keeps the call sites in `apr-cli/src/dispatch.rs` simple (one method
//! call, one error return) and lets `forward_traced` stay free of
//! `&SaveTensorPlan` plumbing in the early steps. When the plan threads
//! all the way through (a later PR), this wrapper becomes a pure delegator
//! and can be deleted in favour of `forward_traced(tokens, Some(plan))`.

use std::io::{BufWriter, Write};

use crate::apr_transformer::{AprTransformer, ForwardTrace};
use crate::error::{RealizarError, Result};
use crate::inference_trace::save_tensor::{write_tensor_file, WHOLE_MODEL_LAYER};
use crate::inference_trace::save_tensor_paths::ensure_layer_dir;
use crate::inference_trace::save_tensor_plan::SaveTensorPlan;
use crate::inference_trace::save_tensor_stage::SaveTensorStage;

/// Errors specific to `forward_traced_with_save_tensor` (in addition to any
/// [`RealizarError`](crate::error::RealizarError) propagated from
/// `forward_traced` itself).
#[derive(Debug, thiserror::Error)]
pub enum SaveTensorEmitError {
    /// Failed to ensure the output directory exists.
    #[error("save-tensor: failed to create output dir: {0}")]
    CreateDir(std::io::Error),
    /// Failed to write the tensor file.
    #[error("save-tensor: failed to write tensor: {0}")]
    Write(std::io::Error),
    /// Failed to flush after writing.
    #[error("save-tensor: failed to flush: {0}")]
    Flush(std::io::Error),
}

impl AprTransformer {
    /// SHIP-007 PR-C-real (step 1 + step 2): forward pass with layer-by-layer
    /// tracing AND optional per-stage F32 tensor capture.
    ///
    /// This is a thin wrapper around [`AprTransformer::forward_traced`].
    /// It does the same work, returns the same [`ForwardTrace`], and
    /// additionally writes selected stage tensors to disk:
    ///
    /// - **`Embedding`** (step 1): re-extracted via `self.embed(token_ids)` —
    ///   cheap (token-table lookup), no internal forward_traced surgery.
    /// - **`LmHead`** (step 2): pulled from `trace.logits` — already returned
    ///   by `forward_traced`, no recompute, no surgery.
    ///
    /// Per-layer stages other than `Embedding` (qkv_matmul, ffn_gate, etc.)
    /// and the `FinalNorm` whole-model stage still require threading
    /// `Option<&SaveTensorPlan>` through `forward_traced` itself; they are
    /// captured by subsequent steps and silently skipped here.
    ///
    /// Pass an empty plan (e.g.
    /// `SaveTensorPlan::from_cli("embedding", "0..0_does_not_parse", _)`)
    /// to skip all writes; in that case this is exactly equivalent to
    /// calling `forward_traced` directly. (In practice, callers will
    /// only invoke this method when the user passed `--save-tensor`,
    /// so the plan's stage list is always non-empty.)
    ///
    /// # Errors
    ///
    /// - Propagates any [`RealizarError`](crate::error::RealizarError)
    ///   from [`AprTransformer::forward_traced`].
    /// - Returns [`SaveTensorEmitError`] (boxed into the project error
    ///   type) if directory creation, file writing, or flushing fails.
    pub fn forward_traced_with_save_tensor(
        &self,
        token_ids: &[u32],
        plan: &SaveTensorPlan,
    ) -> Result<ForwardTrace> {
        // Run the standard traced forward pass first. If it errors, we
        // never write a partial save_tensor file (atomic-by-construction).
        let trace = self.forward_traced(token_ids)?;

        // Step-1 scope: emit ONLY the embedding stage.
        if plan.should_save(SaveTensorStage::Embedding, 0) {
            // Re-extract the embedding F32 buffer. This is a cheap second
            // call (token-table lookup, no matmuls). Subsequent SHIP-007
            // steps replace this with a direct buffer pass-through inside
            // forward_traced so we don't compute embeddings twice.
            let embedding = self.embed(token_ids);

            // Build the destination path via the plan; that way file
            // layout stays in sync with `apr diff --stage` (PR-D) which
            // reads the same path.
            let path = plan.stage_path(SaveTensorStage::Embedding, 0);
            // Embedding is a per-layer stage with layer=0; ensure_layer_dir
            // creates `<output_dir>/layer-0/`.
            ensure_layer_dir(&plan.output_dir, 0).map_err(|e| RealizarError::IoError {
                message: format!("save_tensor::ensure_layer_dir: {e}"),
            })?;
            let file = std::fs::File::create(&path).map_err(|e| RealizarError::IoError {
                message: format!("save_tensor::create({}): {e}", path.display()),
            })?;
            let mut writer = BufWriter::new(file);
            // Layer index in the file header == 0 for embedding (which is
            // a per-layer stage). Per-layer stages do NOT use the
            // WHOLE_MODEL_LAYER sentinel.
            write_tensor_file(&mut writer, 0, &embedding).map_err(|e| RealizarError::IoError {
                message: format!("save_tensor::write({}): {e}", path.display()),
            })?;
            writer.flush().map_err(|e| RealizarError::IoError {
                message: format!("save_tensor::flush({}): {e}", path.display()),
            })?;
        }

        // Step-2 scope: emit the LmHead (final logits) whole-model stage.
        // `trace.logits` is already a Vec<f32> returned by forward_traced — no
        // recompute needed, no internal forward_traced surgery. This is the
        // same low-risk pattern as Embedding above; the corresponding
        // forward-pass surgery for per-layer stages (qkv, ffn_*, etc.) is
        // deferred to subsequent SHIP-007 steps.
        if plan.should_save(SaveTensorStage::LmHead, WHOLE_MODEL_LAYER) {
            let path = plan.stage_path(SaveTensorStage::LmHead, WHOLE_MODEL_LAYER);
            // Whole-model stage: ensure_layer_dir(WHOLE_MODEL_LAYER) creates
            // `<output_dir>/` (no per-layer subdir).
            ensure_layer_dir(&plan.output_dir, WHOLE_MODEL_LAYER).map_err(|e| {
                RealizarError::IoError {
                    message: format!("save_tensor::ensure_layer_dir(lm_head): {e}"),
                }
            })?;
            let file = std::fs::File::create(&path).map_err(|e| RealizarError::IoError {
                message: format!("save_tensor::create({}): {e}", path.display()),
            })?;
            let mut writer = BufWriter::new(file);
            // Whole-model stages use WHOLE_MODEL_LAYER as the header layer
            // field per `save_tensor::write_header` contract.
            write_tensor_file(&mut writer, WHOLE_MODEL_LAYER, &trace.logits).map_err(|e| {
                RealizarError::IoError {
                    message: format!("save_tensor::write({}): {e}", path.display()),
                }
            })?;
            writer.flush().map_err(|e| RealizarError::IoError {
                message: format!("save_tensor::flush({}): {e}", path.display()),
            })?;
        }

        Ok(trace)
    }
}

#[cfg(test)]
mod traced_save_tensor_step2_tests {
    //! SHIP-007 PR-C-real step 2 — `LmHead` branch pin tests.
    //!
    //! These tests do NOT instantiate a full `AprTransformer` (loading a
    //! real APR model is heavyweight). Instead they pin the SHAPE of the
    //! wrapper's LmHead branch: the same plan API → ensure_dir → File::create
    //! → write_tensor_file → flush sequence. A drift-prevention pin: if any
    //! step of the chain breaks (renamed function, changed path layout,
    //! header byte format change), this test fails before the wrapper does.
    //!
    //! Live discharge of the wrapper running on a real model is left to
    //! SHIP-007 PR-E (`apr trace --save-tensor lm_head` end-to-end).
    use super::{
        ensure_layer_dir, write_tensor_file, SaveTensorPlan, SaveTensorStage, WHOLE_MODEL_LAYER,
    };
    use std::io::{BufWriter, Write};

    /// Mirror the exact byte-flow the wrapper's step-2 branch performs.
    fn simulate_step2_lm_head_branch(
        plan: &SaveTensorPlan,
        logits: &[f32],
    ) -> std::io::Result<Option<std::path::PathBuf>> {
        if !plan.should_save(SaveTensorStage::LmHead, WHOLE_MODEL_LAYER) {
            return Ok(None);
        }
        let path = plan.stage_path(SaveTensorStage::LmHead, WHOLE_MODEL_LAYER);
        ensure_layer_dir(&plan.output_dir, WHOLE_MODEL_LAYER)?;
        let file = std::fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);
        write_tensor_file(&mut writer, WHOLE_MODEL_LAYER, logits)?;
        writer.flush()?;
        Ok(Some(path))
    }

    #[test]
    fn step2_lm_head_writes_to_output_root_not_per_layer_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = SaveTensorPlan::from_cli("lm_head", "0..1", tmp.path().to_path_buf()).unwrap();
        let logits = vec![1.5_f32, 2.5, -3.5, 4.5];

        let written = simulate_step2_lm_head_branch(&plan, &logits)
            .unwrap()
            .expect("LmHead is selected — should write");

        // Per `output_path_whole_model_no_layer_segment` test in
        // save_tensor_paths.rs: whole-model stages land at <root>/lm_head.bin.
        assert_eq!(written, tmp.path().join("lm_head.bin"));
        assert!(written.exists());
        // No layer-N subdirectory should be created for the lm_head stage.
        assert!(!tmp.path().join("layer-0").join("lm_head.bin").exists());
    }

    #[test]
    fn step2_lm_head_header_uses_whole_model_sentinel() {
        let tmp = tempfile::tempdir().unwrap();
        let plan = SaveTensorPlan::from_cli("lm_head", "0..1", tmp.path().to_path_buf()).unwrap();
        let logits = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5];

        let path = simulate_step2_lm_head_branch(&plan, &logits)
            .unwrap()
            .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() >= 12, "header is 12 bytes minimum");
        assert_eq!(&bytes[0..4], b"APRT", "magic must be APRT");
        let layer = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(
            layer, WHOLE_MODEL_LAYER,
            "whole-model stages must use WHOLE_MODEL_LAYER sentinel"
        );
        let dim = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(dim, logits.len() as u32);
        assert_eq!(
            bytes.len(),
            12 + logits.len() * 4,
            "total file = 12-byte header + 4 bytes per f32"
        );
    }

    #[test]
    fn step2_lm_head_skipped_when_plan_does_not_select_it() {
        // Plan selects only Embedding — LmHead branch must NOT fire.
        let tmp = tempfile::tempdir().unwrap();
        let plan = SaveTensorPlan::from_cli("embedding", "0..1", tmp.path().to_path_buf()).unwrap();
        let logits = vec![9.9_f32];

        let written = simulate_step2_lm_head_branch(&plan, &logits).unwrap();
        assert!(
            written.is_none(),
            "LmHead branch must be a no-op when not selected"
        );
        assert!(!tmp.path().join("lm_head.bin").exists());
    }

    #[test]
    fn step2_lm_head_writes_logits_bytes_verbatim() {
        // Determinism + bit-identity pin: the bytes after the header are
        // exactly the f32 LE values from the input slice, no quantization,
        // no NaN-masking.
        let tmp = tempfile::tempdir().unwrap();
        let plan = SaveTensorPlan::from_cli("lm_head", "0..1", tmp.path().to_path_buf()).unwrap();
        let logits = vec![1.0_f32, f32::NAN, -0.0, f32::INFINITY];

        let path = simulate_step2_lm_head_branch(&plan, &logits)
            .unwrap()
            .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let body = &bytes[12..];
        assert_eq!(body.len(), logits.len() * 4);
        // Verify each f32 round-trips bit-exactly (NaN bits preserved).
        for (i, expected) in logits.iter().enumerate() {
            let read = f32::from_le_bytes([
                body[i * 4],
                body[i * 4 + 1],
                body[i * 4 + 2],
                body[i * 4 + 3],
            ]);
            // NaN != NaN, so compare bit patterns instead.
            assert_eq!(
                read.to_bits(),
                expected.to_bits(),
                "bit-identical round-trip required at index {i}"
            );
        }
    }
}
