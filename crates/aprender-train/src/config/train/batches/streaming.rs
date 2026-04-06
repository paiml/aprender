#![allow(dead_code)]
//! Streaming Parquet data loader with file-level sharding for distributed training.
//!
//! # Architecture
//!
//! For DDP pretraining, each worker loads a disjoint subset of Parquet files.
//! File-level sharding avoids duplicate samples across workers and is simpler
//! than sequence-level sharding (no coordination needed).
//!
//! Worker N loads files: {f | f % world_size == rank}
//!
//! # Contract
//!
//! C-SHARD-001: Disjointness — no file is assigned to two workers.
//! C-SHARD-001: Completeness — every file is assigned to exactly one worker.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// Configuration for data sharding across distributed workers.
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// This worker's global rank
    pub rank: usize,
    /// Total number of workers
    pub world_size: usize,
    /// Base random seed for epoch shuffling
    pub seed: u64,
}

impl ShardConfig {
    /// Create a single-worker (no sharding) config.
    pub fn single() -> Self {
        Self { rank: 0, world_size: 1, seed: 42 }
    }
}

/// Streaming Parquet data loader with prefetch and file-level sharding.
///
/// Loads Parquet files lazily, keeping only a bounded buffer of batches
/// in memory. Supports epoch-level reshuffling while maintaining shard
/// assignment invariants.
///
/// # Example
///
/// ```ignore
/// let loader = StreamingParquetLoader::new(
///     &data_dir,
///     ShardConfig { rank: 0, world_size: 2, seed: 42 },
///     4,    // batch_size
///     2048, // seq_len
/// )?;
/// ```
#[derive(Debug)]
pub struct StreamingParquetLoader {
    /// All Parquet files discovered in the data directory
    all_files: Vec<PathBuf>,
    /// Files assigned to this worker (after sharding)
    my_files: Vec<PathBuf>,
    /// Shard configuration
    shard_config: ShardConfig,
    /// Batch size for LMBatch construction
    batch_size: usize,
    /// Sequence length
    seq_len: usize,
    /// Buffer of pre-loaded sequences (token ID vectors)
    buffer: VecDeque<Vec<u32>>,
    /// Index of next file to load from `my_files`
    next_file_idx: usize,
    /// Current epoch (for shuffling)
    epoch: usize,
}

impl StreamingParquetLoader {
    /// Create a new streaming loader.
    ///
    /// Discovers all `.parquet` files in `data_dir`, assigns a subset to
    /// this worker based on `shard_config`, and prepares for iteration.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - `data_dir` doesn't exist or is unreadable
    /// - Fewer files than `world_size` (C-SHARD-001 violation)
    pub fn new(
        data_dir: &Path,
        shard_config: ShardConfig,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Self, String> {
        let mut all_files = discover_parquet_files(data_dir)?;
        all_files.sort(); // Deterministic ordering

        if all_files.len() < shard_config.world_size {
            return Err(format!(
                "insufficient files for sharding: {} files < {} workers (C-SHARD-001)",
                all_files.len(),
                shard_config.world_size,
            ));
        }

        let my_files = shard_files(&all_files, shard_config.rank, shard_config.world_size);

        Ok(Self {
            all_files,
            my_files,
            shard_config,
            batch_size,
            seq_len,
            buffer: VecDeque::new(),
            next_file_idx: 0,
            epoch: 0,
        })
    }

    /// Number of files assigned to this worker.
    pub fn num_files(&self) -> usize {
        self.my_files.len()
    }

    /// Total number of files across all workers.
    pub fn total_files(&self) -> usize {
        self.all_files.len()
    }

    /// Get the files assigned to this worker.
    pub fn my_files(&self) -> &[PathBuf] {
        &self.my_files
    }

    /// Reset for a new epoch, reshuffling file order.
    pub fn reset_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
        self.next_file_idx = 0;
        self.buffer.clear();
        // Shuffle file order using epoch-specific seed
        shuffle_files(&mut self.my_files, self.shard_config.seed, epoch);
    }

    /// Get batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Load the next shard file and return its `LMBatch`es.
    ///
    /// Returns `Ok(None)` when all shards for this epoch are exhausted.
    /// Each call loads one Parquet file, extracts pre-tokenized sequences,
    /// creates `LMBatch`es, and drops the raw data. Peak memory = one shard.
    #[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
    pub fn next_batches(
        &mut self,
    ) -> std::result::Result<Option<Vec<crate::train::LMBatch>>, String> {
        use crate::train::LMBatch;

        if self.next_file_idx >= self.my_files.len() {
            return Ok(None);
        }

        let path = &self.my_files[self.next_file_idx];
        self.next_file_idx += 1;

        // Load and extract pre-tokenized sequences from this shard
        let sequences = load_pretokenized_from_parquet(path)?;

        if sequences.is_empty() {
            return Ok(Some(Vec::new()));
        }

        // Create LMBatches from sequences (sequences dropped after this)
        let pad_id = 0u32;
        let eos_id = 2u32;
        let num_batches = sequences.len().div_ceil(self.batch_size);
        let mut batches = Vec::with_capacity(num_batches);
        for chunk in sequences.chunks(self.batch_size) {
            batches.push(LMBatch::from_sequences(chunk, pad_id, eos_id));
        }

        Ok(Some(batches))
    }

    /// Check if all files have been consumed for this epoch.
    pub fn is_epoch_exhausted(&self) -> bool {
        self.next_file_idx >= self.my_files.len() && self.buffer.is_empty()
    }

    /// Resume data loading from a specific file index (ALB-120).
    ///
    /// After checkpoint restore, call this to skip to the correct position.
    /// Contract: C-DATARESUME-001.
    pub fn resume_from(&mut self, file_idx: usize) {
        self.next_file_idx = file_idx.min(self.my_files.len());
        self.buffer.clear();
    }

    /// Current file index (for checkpointing).
    pub fn current_file_idx(&self) -> usize {
        self.next_file_idx
    }

    /// Current epoch (for checkpointing).
    pub fn current_epoch(&self) -> usize {
        self.epoch
    }
}

/// Discover all `.parquet` files in a directory (non-recursive).
fn discover_parquet_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !dir.exists() {
        return Err(format!("data directory does not exist: {}", dir.display()));
    }

    let mut files = Vec::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read directory {}: {e}", dir.display()))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to read dir entry: {e}"))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("parquet") {
            files.push(path);
        }
    }

    if files.is_empty() {
        return Err(format!("no .parquet files found in {}", dir.display()));
    }

    Ok(files)
}

/// Assign files to a worker using modular sharding.
///
/// Worker `rank` gets files at indices where `index % world_size == rank`.
///
/// # Contract (C-SHARD-001)
///
/// - Disjointness: `shard_files(_, r1, N) ∩ shard_files(_, r2, N) == ∅` for r1 ≠ r2
/// - Completeness: `∪_{r=0}^{N-1} shard_files(_, r, N) == all_files`
fn shard_files(all_files: &[PathBuf], rank: usize, world_size: usize) -> Vec<PathBuf> {
    all_files
        .iter()
        .enumerate()
        .filter(|(i, _)| i % world_size == rank)
        .map(|(_, f)| f.clone())
        .collect()
}

/// Shuffle files deterministically using a seed derived from base_seed + epoch.
///
/// Uses Fisher-Yates with a simple LCG PRNG for reproducibility.
fn shuffle_files(files: &mut [PathBuf], base_seed: u64, epoch: usize) {
    let mut rng_state = base_seed.wrapping_add(epoch as u64);
    for i in (1..files.len()).rev() {
        // LCG: state = state * 6364136223846793005 + 1442695040888963407
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = (rng_state >> 33) as usize % (i + 1);
        files.swap(i, j);
    }
}

/// Load pre-tokenized sequences from a single Parquet file.
///
/// Looks for `input_ids` or `token_ids` columns containing integer list arrays.
/// Returns the sequences as `Vec<Vec<u32>>`. The `ArrowDataset` is dropped before
/// returning, so only the extracted token IDs remain in memory.
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn load_pretokenized_from_parquet(path: &Path) -> std::result::Result<Vec<Vec<u32>>, String> {
    use alimentar::{ArrowDataset, Dataset};
    use arrow::array::{Array, ListArray};

    let dataset = ArrowDataset::from_parquet(path)
        .map_err(|e| format!("Failed to load parquet {}: {e}", path.display()))?;

    let schema = dataset.schema();
    let column_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

    let token_col = column_names.iter().find(|&&n| n == "input_ids" || n == "token_ids").copied();

    let token_col = match token_col {
        Some(col) => col,
        None => {
            return Err(format!(
                "No pre-tokenized column (input_ids/token_ids) in {}",
                path.display()
            ));
        }
    };

    let col_idx = schema.index_of(token_col).map_err(|e| format!("Column index error: {e}"))?;

    let mut sequences = Vec::with_capacity(dataset.len());

    for batch in dataset.iter() {
        let col = batch.column(col_idx);
        if let Some(list_arr) = col.as_any().downcast_ref::<ListArray>() {
            for i in 0..list_arr.len() {
                if list_arr.is_null(i) {
                    continue;
                }
                let values = list_arr.value(i);
                let seq = extract_u32_values(&*values);
                if !seq.is_empty() {
                    sequences.push(seq);
                }
            }
        }
    }
    // dataset dropped here — Arrow memory freed

    Ok(sequences)
}

/// Extract u32 token IDs from an Arrow array (inner values of a ListArray).
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn extract_u32_values(array: &dyn arrow::array::Array) -> Vec<u32> {
    use arrow::array::{Int32Array, Int64Array, UInt32Array};

    if let Some(arr) = array.as_any().downcast_ref::<UInt32Array>() {
        arr.values().to_vec()
    } else if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
        arr.values().iter().map(|&v| v as u32).collect()
    } else if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
        arr.values().iter().map(|&v| v as u32).collect()
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn create_temp_dir_with_files(n: usize) -> (tempfile::TempDir, Vec<PathBuf>) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let mut files = Vec::new();
        for i in 0..n {
            let path = dir.path().join(format!("shard_{i:04}.parquet"));
            fs::write(&path, format!("fake parquet {i}")).expect("write file");
            files.push(path);
        }
        (dir, files)
    }

    #[test]
    fn test_shard_files_disjointness() {
        let files: Vec<PathBuf> = (0..10).map(|i| PathBuf::from(format!("f{i}.parquet"))).collect();
        let s0 = shard_files(&files, 0, 3);
        let s1 = shard_files(&files, 1, 3);
        let s2 = shard_files(&files, 2, 3);

        // Disjointness
        for f in &s0 {
            assert!(!s1.contains(f));
            assert!(!s2.contains(f));
        }
        for f in &s1 {
            assert!(!s2.contains(f));
        }

        // Completeness
        assert_eq!(s0.len() + s1.len() + s2.len(), 10);
    }

    #[test]
    fn test_shard_files_assignment() {
        let files: Vec<PathBuf> = (0..10).map(|i| PathBuf::from(format!("f{i}.parquet"))).collect();
        let s0 = shard_files(&files, 0, 3);
        assert_eq!(s0.len(), 4); // 0,3,6,9
        let s1 = shard_files(&files, 1, 3);
        assert_eq!(s1.len(), 3); // 1,4,7
        let s2 = shard_files(&files, 2, 3);
        assert_eq!(s2.len(), 3); // 2,5,8
    }

    #[test]
    fn test_shard_files_two_workers() {
        let files: Vec<PathBuf> = (0..7).map(|i| PathBuf::from(format!("f{i}.parquet"))).collect();
        let s0 = shard_files(&files, 0, 2);
        let s1 = shard_files(&files, 1, 2);
        assert_eq!(s0.len(), 4); // 0,2,4,6
        assert_eq!(s1.len(), 3); // 1,3,5
    }

    #[test]
    fn test_discover_parquet_files() {
        let (dir, _) = create_temp_dir_with_files(5);
        // Add a non-parquet file
        fs::write(dir.path().join("readme.txt"), "not parquet").expect("write");
        let found = discover_parquet_files(dir.path()).expect("discover");
        assert_eq!(found.len(), 5);
    }

    #[test]
    fn test_discover_parquet_files_empty_dir() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let result = discover_parquet_files(dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no .parquet files"));
    }

    #[test]
    fn test_streaming_loader_insufficient_files() {
        let (dir, _) = create_temp_dir_with_files(1);
        let config = ShardConfig { rank: 0, world_size: 2, seed: 42 };
        let result = StreamingParquetLoader::new(dir.path(), config, 4, 2048);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("insufficient files"));
    }

    #[test]
    fn test_streaming_loader_basic() {
        let (dir, _) = create_temp_dir_with_files(4);
        let config = ShardConfig { rank: 0, world_size: 2, seed: 42 };
        let loader =
            StreamingParquetLoader::new(dir.path(), config, 4, 2048).expect("create loader");
        assert_eq!(loader.num_files(), 2);
        assert_eq!(loader.total_files(), 4);
    }

    #[test]
    fn test_shuffle_files_deterministic() {
        let mut a: Vec<PathBuf> = (0..10).map(|i| PathBuf::from(format!("f{i}"))).collect();
        let mut b = a.clone();
        shuffle_files(&mut a, 42, 0);
        shuffle_files(&mut b, 42, 0);
        assert_eq!(a, b, "same seed + epoch must produce same order");
    }

    #[test]
    fn test_shuffle_files_different_epochs() {
        let mut a: Vec<PathBuf> = (0..10).map(|i| PathBuf::from(format!("f{i}"))).collect();
        let mut b = a.clone();
        shuffle_files(&mut a, 42, 0);
        shuffle_files(&mut b, 42, 1);
        assert_ne!(a, b, "different epochs must produce different orders");
    }

    #[test]
    fn test_reset_epoch() {
        let (dir, _) = create_temp_dir_with_files(4);
        let config = ShardConfig { rank: 0, world_size: 2, seed: 42 };
        let mut loader = StreamingParquetLoader::new(dir.path(), config, 4, 2048).expect("create");
        let files_epoch0 = loader.my_files().to_vec();
        loader.reset_epoch(1);
        let files_epoch1 = loader.my_files().to_vec();
        // Same set of files (sorted), potentially different order
        let mut s0 = files_epoch0.clone();
        let mut s1 = files_epoch1.clone();
        s0.sort();
        s1.sort();
        assert_eq!(s0, s1, "same files assigned across epochs");
    }

    #[test]
    fn test_resume_from_skips_files() {
        let (dir, _files) = create_temp_dir_with_files(5);
        let mut loader =
            StreamingParquetLoader::new(dir.path(), ShardConfig::single(), 4, 128).unwrap();
        assert_eq!(loader.current_file_idx(), 0);
        loader.resume_from(3);
        assert_eq!(loader.current_file_idx(), 3);
        loader.resume_from(100);
        assert_eq!(loader.current_file_idx(), loader.num_files());
        assert!(loader.is_epoch_exhausted());
    }
}
