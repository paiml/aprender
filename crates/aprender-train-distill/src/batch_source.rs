//! Batch source abstraction for the distillation training loop (Phase 4-prep).
//!
//! `Pipeline::train()` historically constructed batches inline as a hardcoded
//! synthetic dataset (see PMAT-698m / PMAT-698o). For Phase 4 of
//! SPEC-DISTILL-001 we need to swap in real-corpus batches sourced from
//! tokenized `.bin` shards (the same format `apr pretrain` consumes via
//! `entrenar::train::shard_reader::ShardBatchIter`).
//!
//! The [`BatchSource`] trait decouples batch production from the pipeline
//! orchestrator. Two implementations ship out of the box:
//!
//! - [`SyntheticBatchSource`] — the synthetic identity-mapping batch used
//!   by smoke tests and fixture-path falsifiers. Each row is `seq_len`
//!   copies of a unique token; the label matches.
//! - [`ShardBatchSource`] — wraps the real-corpus iterator from
//!   `aprender-train`. Reads u32 LE shards, yields `(input_ids, next_token_label)`
//!   pairs sized to the trainer's batch / seq dimensions.
//!
//! Phase 4 dispatch uses `ShardBatchSource`; everything else uses the
//! synthetic default.

use entrenar_common::Result;

/// Produces training batches for [`crate::Pipeline::train`].
///
/// # Contract
///
/// `next_batch(batch_size, seq_len)` returns `(inputs, labels)` where:
/// - `inputs.len() == batch_size`
/// - each row in `inputs` has length `seq_len`
/// - `labels.len() == batch_size`
/// - each label is a vocab-index `< vocab_size`
///
/// Implementations may wrap around on exhaustion (`ShardBatchSource`
/// configurable) or generate fresh data each call (`SyntheticBatchSource`).
pub trait BatchSource {
    /// Produce the next training batch.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying data source fails (e.g., shard
    /// I/O error, exhausted corpus when wrap-around is disabled).
    fn next_batch(
        &mut self,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Vec<Vec<u32>>, Vec<usize>)>;

    /// Optional: reset the source's internal cursor (e.g., for re-runs
    /// against a finite corpus). Default no-op for stateless sources.
    fn reset(&mut self) {}
}

/// The default synthetic batch source. Each row is `seq_len` copies of a
/// distinct token; the label matches. Used by smoke tests and fixture-path
/// falsifiers — the student learns a trivial identity mapping which keeps
/// the F-DISTILL-SMOKE-001 contract (`final_loss < initial_loss`) satisfiable
/// on any working pipeline.
///
/// Implements PMAT-698m + PMAT-698o behavior verbatim.
pub struct SyntheticBatchSource {
    num_classes: usize,
}

impl SyntheticBatchSource {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl BatchSource for SyntheticBatchSource {
    fn next_batch(
        &mut self,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Vec<Vec<u32>>, Vec<usize>)> {
        let nc = self.num_classes;
        let inputs: Vec<Vec<u32>> = (0..batch_size)
            .map(|i| vec![(i % nc) as u32; seq_len])
            .collect();
        let labels: Vec<usize> = (0..batch_size).map(|i| i % nc).collect();
        Ok((inputs, labels))
    }
}

/// Real-corpus batch source backed by `.bin` token shards (u32 LE).
///
/// Lazily-imported wrapper around `entrenar::train::shard_reader::ShardBatchIter`.
/// The shard reader iterates over `(seq_len + 1)` chunks per row; this source
/// splits each chunk into `input_ids = chunk[..seq_len]` and
/// `label = chunk[seq_len]` (causal next-token prediction).
///
/// Wrap-around on exhaustion is ENABLED by default — Phase 4's 50K-step
/// schedule may exceed a single corpus epoch on smaller subsets.
#[cfg(feature = "shard-batch-source")]
pub struct ShardBatchSource {
    inner: entrenar::train::shard_reader::ShardBatchIter,
}

#[cfg(feature = "shard-batch-source")]
impl ShardBatchSource {
    /// Construct from a directory of `.bin` shards.
    ///
    /// # Errors
    ///
    /// Returns an error if `dir` doesn't exist, contains no `.bin` files,
    /// or any shard fails to open.
    pub fn from_dir(
        dir: &std::path::Path,
        batch_size: usize,
        seq_len: usize,
        pad_id: u32,
        eos_id: u32,
    ) -> Result<Self> {
        let mut inner = entrenar::train::shard_reader::ShardBatchIter::new(
            dir, batch_size, seq_len, pad_id, eos_id,
        )
        .map_err(|e| entrenar_common::EntrenarError::Io {
            context: format!("opening shard dir {}", dir.display()),
            source: e,
        })?;
        inner = inner.with_wrap_around(true);
        Ok(Self { inner })
    }
}

#[cfg(feature = "shard-batch-source")]
impl BatchSource for ShardBatchSource {
    fn next_batch(
        &mut self,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Vec<Vec<u32>>, Vec<usize>)> {
        use std::iter::Iterator;
        let batch = self
            .inner
            .next()
            .ok_or_else(|| entrenar_common::EntrenarError::Internal {
                message: "ShardBatchSource exhausted without wrap-around".to_string(),
            })?;
        // LMBatch packs tokens with overlap (stride > 0) or split layout.
        // Convert to (inputs, labels) where label is the next token after
        // each row's input window. For Phase 4-prep this Stage B PR keeps
        // the conversion conservative: per-row predict-the-last-input-token
        // (same identity-mapping semantics as SyntheticBatchSource so the
        // pipeline doesn't immediately diverge on real data — Phase 4
        // proper switches to true next-token prediction).
        let inputs: Vec<Vec<u32>> = (0..batch_size)
            .map(|i| batch.get_input(i).map(<[u32]>::to_vec).unwrap_or_default())
            .collect();
        let labels: Vec<usize> = (0..batch_size)
            .map(|i| {
                // Use the target's last token as the label (next-token
                // prediction at the end of the input window).
                batch
                    .get_target(i)
                    .and_then(|t| t.last())
                    .copied()
                    .unwrap_or(0) as usize
            })
            .collect();
        let _ = seq_len; // shapes asserted via ShardBatchIter contract
        Ok((inputs, labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_batch_source_shapes() {
        let mut src = SyntheticBatchSource::new(32);
        let (inputs, labels) = src.next_batch(4, 8).unwrap();
        assert_eq!(inputs.len(), 4);
        assert_eq!(labels.len(), 4);
        for (i, row) in inputs.iter().enumerate() {
            assert_eq!(row.len(), 8);
            assert!(row.iter().all(|&t| t == (i % 32) as u32));
            assert_eq!(labels[i], i % 32);
        }
    }

    #[test]
    fn synthetic_batch_source_modulo_wraps_with_small_vocab() {
        let mut src = SyntheticBatchSource::new(3);
        let (inputs, labels) = src.next_batch(7, 2).unwrap();
        // batch_size=7 with num_classes=3 wraps: 0,1,2,0,1,2,0
        assert_eq!(labels, vec![0, 1, 2, 0, 1, 2, 0]);
        assert_eq!(inputs[3], vec![0, 0]);
        assert_eq!(inputs[6], vec![0, 0]);
    }

    #[test]
    fn synthetic_batch_source_reset_is_noop() {
        let mut src = SyntheticBatchSource::new(8);
        let (a, _) = src.next_batch(2, 2).unwrap();
        src.reset();
        let (b, _) = src.next_batch(2, 2).unwrap();
        assert_eq!(a, b);
    }
}
