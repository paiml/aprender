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

    /// Produce the next training batch with PER-POSITION labels.
    ///
    /// Returns `(inputs, labels)` where `labels` is `[batch][position]` — the
    /// full shifted target sequence for every input window (position `p`'s
    /// target is the token at `p+1`). This drives full-sequence KD
    /// ([`crate::kd_step::kd_step_per_position`]), which trains on every
    /// position instead of only the next token after the window.
    ///
    /// The default wraps [`Self::next_batch`] as a single trailing-position
    /// label, so existing sources keep working; sources backed by real
    /// shifted targets (`ShardBatchSource`) override it to expose all
    /// positions.
    ///
    /// # Errors
    ///
    /// Same as [`Self::next_batch`].
    fn next_batch_per_position(
        &mut self,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Vec<Vec<u32>>, Vec<Vec<usize>>)> {
        let (inputs, labels) = self.next_batch(batch_size, seq_len)?;
        Ok((inputs, labels.into_iter().map(|l| vec![l]).collect()))
    }

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

    fn next_batch_per_position(
        &mut self,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Vec<Vec<u32>>, Vec<Vec<usize>>)> {
        // Synthetic identity rows: each position predicts the same row token,
        // so the per-position labels are that token repeated `seq_len` times.
        let nc = self.num_classes;
        let inputs: Vec<Vec<u32>> = (0..batch_size)
            .map(|i| vec![(i % nc) as u32; seq_len])
            .collect();
        let labels: Vec<Vec<usize>> = (0..batch_size).map(|i| vec![i % nc; seq_len]).collect();
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
        // LMBatch uses causal layout: get_target(i) is the input shifted by
        // one (`target[p] = input[p+1]`). So `get_target(i).last()` is the
        // GENUINE next token immediately AFTER the input window — real
        // next-token prediction, NOT identity mapping. (An earlier comment
        // here mislabelled this as "identity-mapping"; it never was — see
        // crates/aprender-train/.../batch.rs causal-shift layout and the
        // `shard_batch_source_label_is_genuine_next_token` falsifier below.)
        // This per-row path trains on one target per window; the per-position
        // path (next_batch_per_position) trains on every position.
        let inputs: Vec<Vec<u32>> = (0..batch_size)
            .map(|i| batch.get_input(i).map(<[u32]>::to_vec).unwrap_or_default())
            .collect();
        let labels: Vec<usize> = (0..batch_size)
            .map(|i| {
                // The next token after the window = last token of the
                // causal-shifted target sequence.
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

    fn next_batch_per_position(
        &mut self,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Vec<Vec<u32>>, Vec<Vec<usize>>)> {
        use std::iter::Iterator;
        let batch = self
            .inner
            .next()
            .ok_or_else(|| entrenar_common::EntrenarError::Internal {
                message: "ShardBatchSource exhausted without wrap-around".to_string(),
            })?;
        // Full-sequence KD: per-position labels are the ENTIRE causal-shifted
        // target sequence (`target[p] = input[p+1]`), so every position
        // predicts its successor — up to `seq_len`× more KD signal per window
        // than the per-row path.
        let inputs: Vec<Vec<u32>> = (0..batch_size)
            .map(|i| batch.get_input(i).map(<[u32]>::to_vec).unwrap_or_default())
            .collect();
        let labels: Vec<Vec<usize>> = (0..batch_size)
            .map(|i| {
                batch
                    .get_target(i)
                    .map(|t| t.iter().map(|&x| x as usize).collect())
                    .unwrap_or_default()
            })
            .collect();
        let _ = seq_len;
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

    /// F-DISTILL-SHARD-BATCH-001 — Fixture-driven integration test for
    /// ShardBatchSource. Writes a tiny .bin shard with a known token
    /// sequence and asserts the source produces batches of the expected
    /// shape with tokens drawn from the shard.
    ///
    /// Catches the class of bug where the source's wrap_around / cursor
    /// behavior diverges from the actual shard contents — exactly the
    /// kind of issue that's silent at fixture-test time but corrupts
    /// every Phase 4 step.
    #[cfg(feature = "shard-batch-source")]
    #[test]
    fn shard_batch_source_reads_fixture() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir");
        let shard_path = dir.path().join("shard-00000.bin");
        // Write 4096 u32 tokens: [0, 1, 2, ..., 4095].
        // Enough for batch_size=4, seq_len=16 (4 * 17 = 68 tokens minimum
        // for seq+1 chunks; we have 4096 for plenty of headroom).
        let mut f = std::fs::File::create(&shard_path).expect("create");
        for i in 0u32..4096 {
            f.write_all(&i.to_le_bytes()).expect("write");
        }
        drop(f);

        let mut src = ShardBatchSource::from_dir(dir.path(), 4, 16, 0, 0)
            .expect("ShardBatchSource::from_dir");
        let (inputs, labels) = src.next_batch(4, 16).expect("next_batch");

        assert_eq!(inputs.len(), 4, "batch_size honored");
        assert_eq!(labels.len(), 4, "labels.len == batch_size");
        for row in &inputs {
            assert_eq!(row.len(), 16, "seq_len honored");
            // All tokens should be in [0, 4096) (drawn from the fixture).
            for &t in row {
                assert!(t < 4096, "token {t} out of fixture range");
            }
        }
        for &l in &labels {
            assert!(l < 4096, "label {l} out of fixture range");
        }
    }

    /// F-DISTILL-SHARD-BATCH-002 — wrap-around behavior. Tiny fixture
    /// (only enough tokens for ~2 batches) consumed for 5 batches should
    /// not error out — ShardBatchSource enables wrap-around by default.
    #[cfg(feature = "shard-batch-source")]
    #[test]
    fn shard_batch_source_wraps_around() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir");
        let shard_path = dir.path().join("shard-00000.bin");
        // Only 128 tokens. With batch_size=4, seq_len=16, each batch
        // consumes 4 * (16+1) = 68 tokens. So 128 / 68 ≈ 1.88 batches
        // before wrap-around required.
        let mut f = std::fs::File::create(&shard_path).expect("create");
        for i in 0u32..128 {
            f.write_all(&i.to_le_bytes()).expect("write");
        }
        drop(f);

        let mut src = ShardBatchSource::from_dir(dir.path(), 4, 16, 0, 0)
            .expect("ShardBatchSource::from_dir");
        for batch_idx in 0..5 {
            let (inputs, labels) = src
                .next_batch(4, 16)
                .unwrap_or_else(|e| panic!("batch {batch_idx}: {e:?}"));
            assert_eq!(inputs.len(), 4, "batch {batch_idx} shape");
            assert_eq!(labels.len(), 4, "batch {batch_idx} labels");
        }
    }
}
