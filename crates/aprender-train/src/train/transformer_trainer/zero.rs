//! ZeRO-1 optimizer state sharding for distributed pretraining.
//!
//! Implements optimizer state partitioning (ZeRO Stage 1) where each worker
//! holds only 1/N of the optimizer states (Adam m and v vectors). After
//! gradient AllReduce, each worker runs the optimizer step only for its
//! assigned shard, then all-gathers the updated weights.
//!
//! # Memory Savings
//!
//! For 350M model: AdamW stores m + v (2 × param_count f32).
//! - Without ZeRO: each worker holds ~2.8 GB optimizer state
//! - With ZeRO-1 (N=2): each worker holds ~1.4 GB
//! - With ZeRO-1 (N=4): each worker holds ~0.7 GB
//!
//! # Contract (C-ZERO-001)
//!
//! - After all-gather, all workers hold identical updated weights
//! - Each worker's optimizer shard produces the same result as full optimizer
//! - Shards are contiguous and non-overlapping, covering all parameters

/// Optimizer state shard assignment for a single worker.
///
/// Each worker owns a contiguous range of parameter indices and holds
/// only the Adam m/v states for those parameters.
#[derive(Debug, Clone)]
pub struct OptimizerShard {
    /// This worker's rank
    pub rank: usize,
    /// Total workers in the ZeRO group
    pub world_size: usize,
    /// Start index (inclusive) in the flattened parameter vector
    pub param_start: usize,
    /// End index (exclusive) in the flattened parameter vector
    pub param_end: usize,
    /// Total parameter count across all shards
    pub total_params: usize,
}

impl OptimizerShard {
    /// Compute shard assignment for a given rank.
    ///
    /// Divides `total_params` into `world_size` contiguous shards.
    /// Last shard absorbs any remainder.
    ///
    /// # Contract (C-ZERO-001)
    ///
    /// - Union of all shards == [0, total_params)
    /// - Intersection of any two shards == empty
    /// - Each shard size within 1 of floor(total_params / world_size)
    pub fn for_rank(rank: usize, world_size: usize, total_params: usize) -> Self {
        let shard_size = total_params / world_size;
        let remainder = total_params % world_size;

        // First `remainder` ranks get one extra element
        let param_start = if rank < remainder {
            rank * (shard_size + 1)
        } else {
            remainder * (shard_size + 1) + (rank - remainder) * shard_size
        };

        let param_end =
            if rank < remainder { param_start + shard_size + 1 } else { param_start + shard_size };

        Self { rank, world_size, param_start, param_end, total_params }
    }

    /// Number of parameters in this shard.
    pub fn shard_size(&self) -> usize {
        self.param_end - self.param_start
    }

    /// Check if a parameter index belongs to this shard.
    pub fn owns_param(&self, param_idx: usize) -> bool {
        param_idx >= self.param_start && param_idx < self.param_end
    }

    /// Estimated memory savings ratio compared to full replication.
    ///
    /// Returns fraction of memory saved (e.g., 0.5 for 2 workers).
    pub fn memory_savings(&self) -> f64 {
        1.0 - (1.0 / self.world_size as f64)
    }

    /// Estimated optimizer memory for this shard in bytes.
    ///
    /// AdamW stores m + v (2 × shard_size × sizeof(f32)).
    pub fn shard_memory_bytes(&self) -> usize {
        self.shard_size() * 2 * std::mem::size_of::<f32>()
    }

    /// Full optimizer memory without sharding in bytes.
    pub fn full_memory_bytes(&self) -> usize {
        self.total_params * 2 * std::mem::size_of::<f32>()
    }
}

/// Block-level optimizer shard map.
///
/// Maps transformer blocks to worker ranks. In ZeRO-1, each block's
/// optimizer state is owned by exactly one worker. The owner runs the
/// optimizer step for that block after gradient AllReduce, then broadcasts
/// updated weights.
#[derive(Debug, Clone)]
pub struct ZeroShardMap {
    /// Which worker owns each block's optimizer state.
    /// `block_owners[i]` = rank that owns block i.
    pub block_owners: Vec<usize>,
    /// Which worker owns the LM head optimizer state.
    pub lm_head_owner: usize,
    /// Which worker owns the final norm optimizer state.
    pub final_norm_owner: usize,
    /// Which worker owns the embedding optimizer state.
    pub embedding_owner: usize,
    /// World size
    pub world_size: usize,
}

impl ZeroShardMap {
    /// Create a shard map distributing blocks round-robin across workers.
    ///
    /// Non-block components (LM head, final norm, embedding) are assigned
    /// to rank 0 by default since they're relatively small.
    pub fn round_robin(num_blocks: usize, world_size: usize) -> Self {
        let block_owners: Vec<usize> = (0..num_blocks).map(|i| i % world_size).collect();

        Self { block_owners, lm_head_owner: 0, final_norm_owner: 0, embedding_owner: 0, world_size }
    }

    /// Create a shard map distributing blocks in contiguous chunks.
    ///
    /// Preferred for pipeline parallelism compatibility: worker 0 gets
    /// blocks 0..N/W, worker 1 gets N/W..2N/W, etc.
    pub fn contiguous(num_blocks: usize, world_size: usize) -> Self {
        let blocks_per_worker = num_blocks / world_size;
        let remainder = num_blocks % world_size;
        let mut block_owners = Vec::with_capacity(num_blocks);

        for rank in 0..world_size {
            let count = blocks_per_worker + usize::from(rank < remainder);
            for _ in 0..count {
                block_owners.push(rank);
            }
        }

        Self { block_owners, lm_head_owner: 0, final_norm_owner: 0, embedding_owner: 0, world_size }
    }

    /// Get the owning rank for a given block index.
    pub fn block_owner(&self, block_idx: usize) -> usize {
        self.block_owners[block_idx]
    }

    /// Check if this rank owns a given block's optimizer state.
    pub fn rank_owns_block(&self, rank: usize, block_idx: usize) -> bool {
        self.block_owners[block_idx] == rank
    }

    /// Get all block indices owned by a given rank.
    pub fn blocks_for_rank(&self, rank: usize) -> Vec<usize> {
        self.block_owners
            .iter()
            .enumerate()
            .filter(|(_, &owner)| owner == rank)
            .map(|(i, _)| i)
            .collect()
    }

    /// Number of blocks owned by a given rank.
    pub fn num_blocks_for_rank(&self, rank: usize) -> usize {
        self.block_owners.iter().filter(|&&owner| owner == rank).count()
    }

    /// Memory savings for a given rank (ratio of blocks owned / total blocks).
    ///
    /// Returns the fraction of optimizer memory this rank holds.
    pub fn memory_fraction_for_rank(&self, rank: usize) -> f64 {
        let owned = self.num_blocks_for_rank(rank) as f64;
        let total = self.block_owners.len() as f64;
        owned / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_shard_basic() {
        // 100 params, 4 workers → 25 each
        let shard = OptimizerShard::for_rank(0, 4, 100);
        assert_eq!(shard.shard_size(), 25);
        assert_eq!(shard.param_start, 0);
        assert_eq!(shard.param_end, 25);
        assert!(shard.owns_param(0));
        assert!(shard.owns_param(24));
        assert!(!shard.owns_param(25));
    }

    #[test]
    fn test_optimizer_shard_remainder() {
        // 10 params, 3 workers → 4, 3, 3
        let s0 = OptimizerShard::for_rank(0, 3, 10);
        let s1 = OptimizerShard::for_rank(1, 3, 10);
        let s2 = OptimizerShard::for_rank(2, 3, 10);

        assert_eq!(s0.shard_size(), 4); // gets extra
        assert_eq!(s1.shard_size(), 3);
        assert_eq!(s2.shard_size(), 3);

        // Non-overlapping and complete
        assert_eq!(s0.param_start, 0);
        assert_eq!(s0.param_end, 4);
        assert_eq!(s1.param_start, 4);
        assert_eq!(s1.param_end, 7);
        assert_eq!(s2.param_start, 7);
        assert_eq!(s2.param_end, 10);
    }

    #[test]
    fn test_optimizer_shard_completeness() {
        // C-ZERO-001: union of shards == full range
        let total = 1_000_003; // prime number to test remainder handling
        let world_size = 7;
        let mut covered = vec![false; total];
        for rank in 0..world_size {
            let shard = OptimizerShard::for_rank(rank, world_size, total);
            for i in shard.param_start..shard.param_end {
                assert!(!covered[i], "param {i} covered by multiple shards");
                covered[i] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "not all params covered");
    }

    #[test]
    fn test_optimizer_shard_memory_savings() {
        let shard = OptimizerShard::for_rank(0, 4, 1_000_000);
        assert!((shard.memory_savings() - 0.75).abs() < 1e-10);
        // Full memory: 1M × 2 × 4 = 8 MB
        assert_eq!(shard.full_memory_bytes(), 8_000_000);
        // Shard memory: 250K × 2 × 4 = 2 MB
        assert_eq!(shard.shard_memory_bytes(), 2_000_000);
    }

    #[test]
    fn test_zero_shard_map_round_robin() {
        let map = ZeroShardMap::round_robin(24, 4);
        assert_eq!(map.block_owner(0), 0);
        assert_eq!(map.block_owner(1), 1);
        assert_eq!(map.block_owner(2), 2);
        assert_eq!(map.block_owner(3), 3);
        assert_eq!(map.block_owner(4), 0);

        assert_eq!(map.num_blocks_for_rank(0), 6);
        assert_eq!(map.num_blocks_for_rank(1), 6);

        let blocks = map.blocks_for_rank(0);
        assert_eq!(blocks, vec![0, 4, 8, 12, 16, 20]);
    }

    #[test]
    fn test_zero_shard_map_contiguous() {
        let map = ZeroShardMap::contiguous(24, 4);
        // 24 blocks / 4 workers = 6 each
        assert_eq!(map.blocks_for_rank(0), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(map.blocks_for_rank(1), vec![6, 7, 8, 9, 10, 11]);
        assert_eq!(map.blocks_for_rank(2), vec![12, 13, 14, 15, 16, 17]);
        assert_eq!(map.blocks_for_rank(3), vec![18, 19, 20, 21, 22, 23]);
    }

    #[test]
    fn test_zero_shard_map_contiguous_uneven() {
        let map = ZeroShardMap::contiguous(10, 3);
        // 10/3 = 3 rem 1 → worker 0 gets 4, workers 1,2 get 3
        assert_eq!(map.num_blocks_for_rank(0), 4);
        assert_eq!(map.num_blocks_for_rank(1), 3);
        assert_eq!(map.num_blocks_for_rank(2), 3);

        // All blocks covered
        let total: usize = (0..3).map(|r| map.num_blocks_for_rank(r)).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_zero_shard_map_memory_fraction() {
        let map = ZeroShardMap::round_robin(24, 4);
        let frac = map.memory_fraction_for_rank(0);
        assert!((frac - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_zero_shard_map_rank_owns_block() {
        let map = ZeroShardMap::contiguous(12, 3);
        assert!(map.rank_owns_block(0, 0));
        assert!(map.rank_owns_block(0, 3));
        assert!(!map.rank_owns_block(0, 4));
        assert!(map.rank_owns_block(1, 4));
    }

    #[test]
    fn test_zero_shard_350m() {
        // 350M model: 24 blocks, 4 GPUs
        let map = ZeroShardMap::contiguous(24, 4);
        // Each GPU owns 6 blocks → 25% of optimizer memory
        for rank in 0..4 {
            assert_eq!(map.num_blocks_for_rank(rank), 6);
            let frac = map.memory_fraction_for_rank(rank);
            assert!((frac - 0.25).abs() < 1e-10);
        }
    }
}
