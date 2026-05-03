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
use crate::inference_trace::save_tensor::write_tensor_file;
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
    /// SHIP-007 PR-C-real-step1: forward pass with layer-by-layer tracing
    /// AND optional per-stage F32 tensor capture.
    ///
    /// This is a thin wrapper around [`AprTransformer::forward_traced`].
    /// It does the same work, returns the same [`ForwardTrace`], and
    /// additionally writes the embedding stage to disk if the supplied
    /// plan selects it (`plan.should_save(SaveTensorStage::Embedding, 0)`).
    ///
    /// Whole-model stages (`final_norm`, `lm_head`) and per-layer stages
    /// other than embedding are NOT yet captured by this wrapper — those
    /// will be added in subsequent PRs that thread the plan through
    /// `forward_traced` internals.
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

        Ok(trace)
    }
}
