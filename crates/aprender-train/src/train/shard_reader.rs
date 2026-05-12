//! Minimal tokenized-shard reader for MODEL-2 pretrain MVP (task #111).
//!
//! Reads a directory of `.bin` files containing little-endian u32 tokens,
//! chunks them into `seq_length + 1` sequences, and yields `LMBatch`es of
//! `batch_size` sequences. No licensing filter, no MinHash dedup, no PII
//! scrub — those belong to `apr-corpus-ingest run`.
//!
//! Contract: `contracts/dataset-thestack-python-v1.yaml` (shard format).

use crate::train::transformer_trainer::LMBatch;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

/// Streaming iterator over `LMBatch`es produced from a directory of
/// `.bin` token shards (little-endian u32).
///
/// Default behaviour matches the historical contract: when the last
/// shard is exhausted, `next()` returns `None`. For training paths
/// where the corpus is finite but the run extends beyond a single
/// epoch, opt in to loop-on-exhaust via `with_wrap_around(true)`.
/// This is the standard PyTorch/HuggingFace behaviour; `apr pretrain`
/// uses it in the real-corpus drive paths so that step-count budgets
/// are honoured even on small datasets.
pub struct ShardBatchIter {
    shards: Vec<PathBuf>,
    cursor_shard: usize,
    cursor_reader: Option<BufReader<File>>,
    batch_size: usize,
    seq_plus_one: usize,
    pad_id: u32,
    eos_id: u32,
    wrap_around: bool,
    epochs_completed: u64,
}

impl ShardBatchIter {
    /// Build an iterator that yields `LMBatch` with `batch_size` sequences
    /// of length `seq_length + 1` (for causal shift).
    ///
    /// Returns `Err` if `dataset_dir` is missing or contains no `.bin` shards.
    pub fn new(
        dataset_dir: &Path,
        batch_size: usize,
        seq_length: usize,
        pad_id: u32,
        eos_id: u32,
    ) -> io::Result<Self> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dataset_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "bin"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no .bin shards in {}", dataset_dir.display()),
            ));
        }
        Ok(Self {
            shards,
            cursor_shard: 0,
            cursor_reader: None,
            batch_size,
            seq_plus_one: seq_length + 1,
            pad_id,
            eos_id,
            wrap_around: false,
            epochs_completed: 0,
        })
    }

    /// Enable corpus wrap-around: when the last shard is exhausted,
    /// reset the cursor to shard 0 and continue.
    ///
    /// This is the standard ML-training behaviour. Without it, an
    /// 18M-token corpus exhausts in ~2 epochs of a 5K-step run with
    /// batch=16 seq=512, and the upstream `StepFn` falls back to
    /// returning placeholder loss `(1.0, 1.0)` — silently producing
    /// garbage data that breaks convergence. See spec §22 (PR #1073)
    /// for the corpus-bottleneck investigation.
    #[must_use]
    pub fn with_wrap_around(mut self, wrap_around: bool) -> Self {
        self.wrap_around = wrap_around;
        self
    }

    /// Number of times the iterator has cycled through the entire
    /// shard set. Increments each time the last shard is exhausted
    /// AND `wrap_around` was true (so a reset happened).
    #[must_use]
    pub fn epochs_completed(&self) -> u64 {
        self.epochs_completed
    }

    fn ensure_reader(&mut self) -> io::Result<bool> {
        if self.cursor_reader.is_some() {
            return Ok(true);
        }
        while self.cursor_shard < self.shards.len() {
            match File::open(&self.shards[self.cursor_shard]) {
                Ok(f) => {
                    self.cursor_reader = Some(BufReader::new(f));
                    return Ok(true);
                }
                Err(_) => {
                    self.cursor_shard += 1;
                }
            }
        }
        Ok(false)
    }

    fn read_one_sequence(&mut self) -> io::Result<Option<Vec<u32>>> {
        let tokens_per_seq = self.seq_plus_one;
        let mut buf = vec![0u8; tokens_per_seq * 4];
        loop {
            if !self.ensure_reader()? {
                // All shards exhausted. If wrap-around is on, reset cursor
                // and start over; else return None as before.
                if self.wrap_around {
                    self.epochs_completed += 1;
                    self.cursor_shard = 0;
                    self.cursor_reader = None;
                    if !self.ensure_reader()? {
                        // Still no readable shard after reset — give up
                        // to avoid infinite loop on a broken shard set.
                        return Ok(None);
                    }
                } else {
                    return Ok(None);
                }
            }
            let reader = self.cursor_reader.as_mut().expect("reader set above");
            match reader.read_exact(&mut buf) {
                Ok(()) => {
                    let mut seq = Vec::with_capacity(tokens_per_seq);
                    for chunk in buf.chunks_exact(4) {
                        seq.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                    return Ok(Some(seq));
                }
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    self.cursor_reader = None;
                    self.cursor_shard += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

impl Iterator for ShardBatchIter {
    type Item = LMBatch;

    fn next(&mut self) -> Option<LMBatch> {
        let mut seqs: Vec<Vec<u32>> = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            match self.read_one_sequence() {
                Ok(Some(seq)) => seqs.push(seq),
                Ok(None) => break,
                Err(_) => break,
            }
        }
        if seqs.is_empty() {
            None
        } else {
            Some(LMBatch::from_sequences(&seqs, self.pad_id, self.eos_id))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_shard(dir: &Path, name: &str, tokens: &[u32]) {
        let path = dir.join(name);
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for t in tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        std::fs::write(&path, bytes).expect("shard write");
    }

    /// Wrap-around regression guard: with `with_wrap_around(true)`,
    /// the iterator MUST keep yielding batches past the natural shard
    /// boundary. This is the SHIP-007-adjacent corpus-bottleneck fix —
    /// without wrap-around an N-token corpus exhausts in 1 epoch and
    /// the Cuda*StepFn falls back to placeholder `(1.0, 1.0)` losses,
    /// silently producing garbage gradients (observed 2026-04-26 on a
    /// 5K-step run that early-stopped at epoch 4 with train_loss=1.0).
    #[test]
    fn wrap_around_continues_past_shard_exhaustion() {
        let tmp = TempDir::new().expect("tempdir");
        let tokens: Vec<u32> = (0u32..40).collect(); // 8 sequences of len 5
        write_shard(tmp.path(), "shard-0.bin", &tokens);
        let mut iter =
            ShardBatchIter::new(tmp.path(), 2, 4, 0, 0).expect("iter").with_wrap_around(true);
        // Without wrap-around, we'd get 4 batches then None forever.
        // With wrap-around, we should get 12 batches (3 epochs of 4).
        let mut batches = Vec::new();
        for _ in 0..12 {
            batches.push(iter.next().expect("wrap-around must keep yielding"));
        }
        assert_eq!(batches.len(), 12, "12 batches across 3 simulated epochs");
        assert!(
            iter.epochs_completed() >= 2,
            "epochs_completed = {} should reflect at least 2 wrap resets",
            iter.epochs_completed()
        );
    }

    /// Default behaviour (wrap_around=false) preserved: returns None
    /// after the corpus is exhausted, matching the historical contract.
    #[test]
    fn no_wrap_around_terminates_on_exhaustion() {
        let tmp = TempDir::new().expect("tempdir");
        let tokens: Vec<u32> = (0u32..40).collect();
        write_shard(tmp.path(), "shard-0.bin", &tokens);
        let iter = ShardBatchIter::new(tmp.path(), 2, 4, 0, 0).expect("iter");
        let batches: Vec<_> = iter.collect();
        assert_eq!(batches.len(), 4, "default: 8 seqs / batch=2 = 4 batches then None");
    }

    #[test]
    fn single_shard_yields_expected_batch_count() {
        let tmp = TempDir::new().expect("tempdir");
        let tokens: Vec<u32> = (0u32..40).collect(); // 40 tokens = 8 × (seq=4+1)
        write_shard(tmp.path(), "shard-0.bin", &tokens);
        let iter = ShardBatchIter::new(tmp.path(), 2, 4, 0, 0).expect("iter");
        let batches: Vec<_> = iter.collect();
        assert_eq!(batches.len(), 4, "8 seqs / batch_size=2 = 4 batches");
        assert_eq!(batches[0].batch_size, 2);
        assert_eq!(batches[0].seq_len, 4);
    }

    #[test]
    fn empty_dir_errors() {
        let tmp = TempDir::new().expect("tempdir");
        let res = ShardBatchIter::new(tmp.path(), 2, 4, 0, 0);
        assert!(res.is_err(), "empty dir must error");
    }

    #[test]
    fn multi_shard_ordering_is_lexical() {
        let tmp = TempDir::new().expect("tempdir");
        write_shard(tmp.path(), "shard-0.bin", &(0u32..10).collect::<Vec<_>>());
        write_shard(tmp.path(), "shard-1.bin", &(100u32..110).collect::<Vec<_>>());
        let mut iter = ShardBatchIter::new(tmp.path(), 1, 4, 0, 0).expect("iter");
        let first = iter.next().expect("first batch");
        assert_eq!(first.get_input(0).expect("input0")[0], 0, "shard-0 first");
    }
}
