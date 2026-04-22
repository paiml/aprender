//! Pre-flight gates for `apr pretrain` (extracted from `pretrain.rs` to
//! keep the file-size invariant).
//!
//! Bound by two contracts:
//!   * `contracts/pretraining-corpus-v1.yaml` v2.0.0 §FALSIFY-CORPUS-004
//!     — dispatch-budget gate (GATE-CORPUS-PREFLIGHT)
//!   * `contracts/model-families/llama-370m-sovereign-v1.yaml`
//!     §GATE-ARCH-370M-011 — tokenizer↔model vocab parity gate
//!
//! Both gates fire BEFORE any trainer allocation so a wrong-sized corpus
//! or a mismatched tokenizer costs zero GPU time.

use crate::error::{CliError, Result};
use entrenar::models::llama_370m::{assert_tokenizer_vocab_matches_model, Llama370MConfig};
use entrenar::train::shard_reader::ShardBatchIter;
use std::path::Path;

/// GATE-CORPUS-PREFLIGHT pre-flight: refuse to dispatch when the
/// operator's planned token budget exceeds the corpus's actual token
/// count unless `--allow-shard-cycle` is set.
///
/// Formula (per FALSIFY-CORPUS-004): `planned_tokens =
/// num_steps × batch_size × seq_length`. Corpus total is counted
/// directly from `.bin` shard file sizes so there is no manifest
/// round-trip and no cache-staleness window.
///
/// Contract: `contracts/pretraining-corpus-v1.yaml` v2.0.0
/// §FALSIFY-CORPUS-004. This gate is the pre-flight peer of
/// `GATE-TRAIN-EXHAUST` — together they close the task #141 silent
/// `(1.0, 1.0)` placeholder loophole.
///
/// Returns `(planned_tokens, total_tokens)` on success so the caller
/// can log the resolved decision (cycle vs no-cycle) before wiring
/// the iterator.
pub(super) fn preflight_dispatch_budget(
    dataset: &Path,
    num_steps: usize,
    batch_size: usize,
    seq_length: usize,
    allow_shard_cycle: bool,
) -> Result<(u64, u64)> {
    let planned_tokens = (num_steps as u64)
        .saturating_mul(batch_size as u64)
        .saturating_mul(seq_length as u64);
    let total_tokens = ShardBatchIter::count_tokens(dataset).map_err(|e| {
        CliError::ValidationFailed(format!(
            "GATE-CORPUS-PREFLIGHT: cannot count corpus tokens in {} ({e})",
            dataset.display()
        ))
    })?;
    if planned_tokens > total_tokens && !allow_shard_cycle {
        let factor = if total_tokens == 0 {
            f64::INFINITY
        } else {
            planned_tokens as f64 / total_tokens as f64
        };
        return Err(CliError::ValidationFailed(format!(
            "GATE-CORPUS-PREFLIGHT: planned_tokens={planned_tokens} exceeds \
             corpus total_tokens={total_tokens} (factor {factor:.1}×). \
             Either enlarge the corpus so total_tokens covers the planned \
             (num_steps × batch_size × seq_length) budget, or pass \
             `--allow-shard-cycle` to opt into shard-stream cycling \
             (INV-TRAIN-011 path a). See \
             contracts/pretraining-corpus-v1.yaml v2.0.0 FALSIFY-CORPUS-004."
        )));
    }
    Ok((planned_tokens, total_tokens))
}

/// GATE-ARCH-370M-011 pre-flight: count the tokenizer's vocabulary entries
/// from `vocab.json` and assert the count matches `Llama370MConfig::VOCAB_SIZE`
/// before any trainer allocation. Any mismatch aborts the dispatch with a
/// clear error naming both values and the violated invariant — the N-09 OOB
/// escape in `Embedding::forward` would otherwise silently corrupt training.
pub(super) fn preflight_tokenizer_vocab_matches_model(tokenizer_dir: &Path) -> Result<()> {
    let vocab_path = tokenizer_dir.join("vocab.json");
    let vocab_json = std::fs::read_to_string(&vocab_path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "GATE-ARCH-370M-011 pre-flight: cannot read {} ({e})",
            vocab_path.display()
        ))
    })?;
    let vocab: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&vocab_json)
        .map_err(|e| {
            CliError::ValidationFailed(format!(
                "GATE-ARCH-370M-011 pre-flight: {} is not a valid vocab.json: {e}",
                vocab_path.display()
            ))
        })?;
    assert_tokenizer_vocab_matches_model(vocab.len(), Llama370MConfig::VOCAB_SIZE)
        .map_err(CliError::ValidationFailed)
}
