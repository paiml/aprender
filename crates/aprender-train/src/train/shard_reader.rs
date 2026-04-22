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
pub struct ShardBatchIter {
    shards: Vec<PathBuf>,
    cursor_shard: usize,
    cursor_reader: Option<BufReader<File>>,
    batch_size: usize,
    seq_plus_one: usize,
    pad_id: u32,
    eos_id: u32,
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
        })
    }

    /// Count the total u32 tokens across every `.bin` shard in
    /// `dataset_dir` without opening a reader. Each token is 4 bytes
    /// little-endian, so `token_count = Σ(file_size / 4)`.
    ///
    /// Used by the `apr pretrain` pre-flight gate
    /// (`contracts/pretraining-corpus-v1.yaml` v2.0.0 §FALSIFY-CORPUS-004)
    /// to compare the operator's planned token budget against the
    /// actual corpus size and refuse over-dispatch unless
    /// `--allow-shard-cycle` is set.
    ///
    /// Returns `Err` if `dataset_dir` is missing / unreadable. An
    /// empty directory (no `.bin` files) returns `Ok(0)` — the caller
    /// decides whether zero tokens is a refusal condition.
    pub fn count_tokens(dataset_dir: &Path) -> io::Result<u64> {
        let mut total: u64 = 0;
        for entry in std::fs::read_dir(dataset_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "bin") {
                let meta = std::fs::metadata(&path)?;
                total = total.saturating_add(meta.len() / 4);
            }
        }
        Ok(total)
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
                return Ok(None);
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
    fn count_tokens_sums_bin_sizes_divided_by_four() {
        // GATE-CORPUS-PREFLIGHT: count across 2 shards (10 + 20 = 30 u32
        // tokens => 120 bytes). Non-.bin files must be ignored.
        let tmp = TempDir::new().expect("tempdir");
        write_shard(tmp.path(), "a.bin", &(0u32..10).collect::<Vec<_>>());
        write_shard(tmp.path(), "b.bin", &(0u32..20).collect::<Vec<_>>());
        std::fs::write(tmp.path().join("README.md"), b"ignore me").expect("write readme");
        assert_eq!(
            ShardBatchIter::count_tokens(tmp.path()).expect("count"),
            30,
            "count_tokens must sum .bin sizes/4 and skip non-.bin files"
        );
    }

    #[test]
    fn count_tokens_empty_dir_returns_zero() {
        // An empty directory is NOT an error — the pre-flight caller
        // decides whether zero tokens is acceptable.
        let tmp = TempDir::new().expect("tempdir");
        assert_eq!(
            ShardBatchIter::count_tokens(tmp.path()).expect("count empty"),
            0,
            "empty dir must return Ok(0), not Err"
        );
    }

    #[test]
    fn count_tokens_missing_dir_errors() {
        let missing = std::path::PathBuf::from("/nonexistent/_shardreader_count_tokens_missing");
        assert!(
            ShardBatchIter::count_tokens(&missing).is_err(),
            "missing directory must surface io::Error"
        );
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
