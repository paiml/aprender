//! Tensor parallelism for transformer pretraining.
//!
//! Splits weight matrices across GPUs along the hidden/head dimension.
//! Each GPU holds a shard of Q/K/V/O projections and FFN layers.
//!
//! # Architecture (Megatron-LM style)
//!
//! ## Attention (Column + Row parallel):
//! ```text
//! Input X [S, H]
//!   ├── GPU 0: Q₀ = X × W_q[:, :H/2]  →  heads 0..N/2
//!   └── GPU 1: Q₁ = X × W_q[:, H/2:]  →  heads N/2..N
//!                    ↓ attention ↓
//!   ├── GPU 0: O₀ = attn₀ × W_o[:H/2, :]
//!   └── GPU 1: O₁ = attn₁ × W_o[H/2:, :]
//!              AllReduce(O₀ + O₁)
//! ```
//!
//! ## FFN (Column + Row parallel):
//! ```text
//! Input X [S, H]
//!   ├── GPU 0: gate₀ = X × W_gate[:, :I/2], up₀ = X × W_up[:, :I/2]
//!   └── GPU 1: gate₁ = X × W_gate[:, I/2:], up₁ = X × W_up[:, I/2:]
//!                     ↓ SiLU(gate) * up ↓
//!   ├── GPU 0: down₀ = act₀ × W_down[:I/2, :]
//!   └── GPU 1: down₁ = act₁ × W_down[I/2:, :]
//!              AllReduce(down₀ + down₁)
//! ```
//!
//! # Communication
//!
//! 2 AllReduce per block (attention output + FFN output).
//! Each AllReduce is [S, H] = seq_len × hidden_size × 4 bytes.
//! For S=1024, H=1024: 4 MB per AllReduce, 8 MB per block.
//!
//! # Contract (C-TP-001)
//!
//! - Column parallel: output[i, j] = input[i, :] · weight[:, shard_start + j]
//! - Row parallel: after AllReduce, output = Σ(partial_outputs) across shards
//! - Weight shards are contiguous slices along the parallelism dimension

/// Tensor parallel configuration for a single GPU.
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// This GPU's rank within the TP group
    pub tp_rank: usize,
    /// Total GPUs in the TP group
    pub tp_size: usize,
    /// Full hidden size (before sharding)
    pub hidden_size: usize,
    /// Full intermediate size (before sharding)
    pub intermediate_size: usize,
    /// Number of attention heads (must be divisible by tp_size)
    pub num_heads: usize,
    /// Number of KV heads (must be divisible by tp_size)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl TensorParallelConfig {
    /// Create a new TP config.
    ///
    /// # Panics
    /// Panics if heads or intermediate size not divisible by tp_size.
    pub fn new(
        tp_rank: usize,
        tp_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Self {
        assert!(
            num_heads.is_multiple_of(tp_size),
            "num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"
        );
        assert!(
            num_kv_heads.is_multiple_of(tp_size),
            "num_kv_heads ({num_kv_heads}) must be divisible by tp_size ({tp_size})"
        );
        assert!(
            intermediate_size.is_multiple_of(tp_size),
            "intermediate_size ({intermediate_size}) must be divisible by tp_size ({tp_size})"
        );

        let head_dim = hidden_size / num_heads;

        Self { tp_rank, tp_size, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim }
    }

    /// Number of Q heads on this GPU.
    pub fn local_num_heads(&self) -> usize {
        self.num_heads / self.tp_size
    }

    /// Number of KV heads on this GPU.
    pub fn local_num_kv_heads(&self) -> usize {
        self.num_kv_heads / self.tp_size
    }

    /// Local Q projection size: local_heads × head_dim.
    pub fn local_q_size(&self) -> usize {
        self.local_num_heads() * self.head_dim
    }

    /// Local KV projection size: local_kv_heads × head_dim.
    pub fn local_kv_size(&self) -> usize {
        self.local_num_kv_heads() * self.head_dim
    }

    /// Local intermediate (FFN) size.
    pub fn local_intermediate_size(&self) -> usize {
        self.intermediate_size / self.tp_size
    }

    /// Memory savings from tensor parallelism (ratio).
    ///
    /// TP shards attention + FFN weights. Embedding and norms are replicated.
    pub fn weight_memory_fraction(&self) -> f64 {
        1.0 / self.tp_size as f64
    }
}

/// Weight shard specification for a column-parallel layer.
///
/// Column parallel: input is replicated, output is sharded.
/// Each GPU holds weight[:, shard_start:shard_end].
#[derive(Debug, Clone)]
pub struct ColumnParallelShard {
    /// Input dimension (full, not sharded)
    pub input_dim: usize,
    /// Output dimension per GPU (sharded)
    pub local_output_dim: usize,
    /// Start column index in the full weight matrix
    pub col_start: usize,
    /// End column index (exclusive)
    pub col_end: usize,
}

impl ColumnParallelShard {
    /// Create a column-parallel shard for Q/K/V projection or FFN gate/up.
    pub fn new(input_dim: usize, full_output_dim: usize, tp_rank: usize, tp_size: usize) -> Self {
        let local_output_dim = full_output_dim / tp_size;
        let col_start = tp_rank * local_output_dim;
        let col_end = col_start + local_output_dim;

        Self { input_dim, local_output_dim, col_start, col_end }
    }

    /// Number of elements in the local weight shard.
    pub fn num_elements(&self) -> usize {
        self.input_dim * self.local_output_dim
    }

    /// Extract this shard from a full weight matrix (row-major).
    ///
    /// Full weight shape: [input_dim, full_output_dim]
    /// Returns: [input_dim, local_output_dim]
    pub fn extract_shard(&self, full_weights: &[f32], full_output_dim: usize) -> Vec<f32> {
        let mut shard = Vec::with_capacity(self.num_elements());
        for row in 0..self.input_dim {
            let row_start = row * full_output_dim;
            shard.extend_from_slice(
                &full_weights[row_start + self.col_start..row_start + self.col_end],
            );
        }
        shard
    }
}

/// Weight shard specification for a row-parallel layer.
///
/// Row parallel: input is sharded, output is partial (needs AllReduce).
/// Each GPU holds weight[shard_start:shard_end, :].
#[derive(Debug, Clone)]
pub struct RowParallelShard {
    /// Input dimension per GPU (sharded)
    pub local_input_dim: usize,
    /// Output dimension (full, not sharded)
    pub output_dim: usize,
    /// Start row index in the full weight matrix
    pub row_start: usize,
    /// End row index (exclusive)
    pub row_end: usize,
}

impl RowParallelShard {
    /// Create a row-parallel shard for O projection or FFN down.
    pub fn new(full_input_dim: usize, output_dim: usize, tp_rank: usize, tp_size: usize) -> Self {
        let local_input_dim = full_input_dim / tp_size;
        let row_start = tp_rank * local_input_dim;
        let row_end = row_start + local_input_dim;

        Self { local_input_dim, output_dim, row_start, row_end }
    }

    /// Number of elements in the local weight shard.
    pub fn num_elements(&self) -> usize {
        self.local_input_dim * self.output_dim
    }

    /// Extract this shard from a full weight matrix (row-major).
    ///
    /// Full weight shape: [full_input_dim, output_dim]
    /// Returns: [local_input_dim, output_dim]
    pub fn extract_shard(&self, full_weights: &[f32], _full_input_dim: usize) -> Vec<f32> {
        let start = self.row_start * self.output_dim;
        let end = self.row_end * self.output_dim;
        full_weights[start..end].to_vec()
    }
}

/// Communication cost estimate for tensor parallelism.
#[derive(Debug, Clone)]
pub struct TpCommCost {
    /// Bytes per AllReduce call
    pub bytes_per_allreduce: usize,
    /// Number of AllReduce calls per block (2: attention + FFN)
    pub allreduces_per_block: usize,
    /// Total blocks
    pub num_blocks: usize,
}

impl TpCommCost {
    /// Estimate TP communication cost.
    pub fn estimate(seq_len: usize, hidden_size: usize, num_blocks: usize) -> Self {
        Self {
            bytes_per_allreduce: seq_len * hidden_size * std::mem::size_of::<f32>(),
            allreduces_per_block: 2,
            num_blocks,
        }
    }

    /// Total bytes communicated per training step.
    pub fn total_bytes_per_step(&self) -> usize {
        self.bytes_per_allreduce * self.allreduces_per_block * self.num_blocks
    }

    /// Estimated overhead in milliseconds (assumes 10 GB/s intra-node bandwidth).
    pub fn estimated_overhead_ms(&self, bandwidth_gbps: f64) -> f64 {
        let total_bytes = self.total_bytes_per_step() as f64;
        let bandwidth_bytes_per_ms = bandwidth_gbps * 1e9 / 8.0 / 1000.0;
        total_bytes / bandwidth_bytes_per_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tp_config_basic() {
        // 350M: H=1024, I=4096, 16 heads, 4 KV heads, TP=2
        let tp = TensorParallelConfig::new(0, 2, 1024, 4096, 16, 4);
        assert_eq!(tp.local_num_heads(), 8);
        assert_eq!(tp.local_num_kv_heads(), 2);
        assert_eq!(tp.local_q_size(), 8 * 64); // 512
        assert_eq!(tp.local_kv_size(), 2 * 64); // 128
        assert_eq!(tp.local_intermediate_size(), 2048);
        assert!((tp.weight_memory_fraction() - 0.5).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "num_heads")]
    fn test_tp_config_indivisible_heads() {
        TensorParallelConfig::new(0, 3, 1024, 4096, 16, 4); // 16 % 3 != 0
    }

    #[test]
    fn test_column_parallel_shard() {
        // Q projection: [1024, 1024] split across 2 GPUs
        let shard0 = ColumnParallelShard::new(1024, 1024, 0, 2);
        let shard1 = ColumnParallelShard::new(1024, 1024, 1, 2);

        assert_eq!(shard0.col_start, 0);
        assert_eq!(shard0.col_end, 512);
        assert_eq!(shard0.local_output_dim, 512);
        assert_eq!(shard0.num_elements(), 1024 * 512);

        assert_eq!(shard1.col_start, 512);
        assert_eq!(shard1.col_end, 1024);
    }

    #[test]
    fn test_column_parallel_extract() {
        // Small example: [2, 4] split into [2, 2] each
        let full = vec![
            1.0, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
        ];
        let shard0 = ColumnParallelShard::new(2, 4, 0, 2);
        let shard1 = ColumnParallelShard::new(2, 4, 1, 2);

        let s0 = shard0.extract_shard(&full, 4);
        assert_eq!(s0, vec![1.0, 2.0, 5.0, 6.0]);

        let s1 = shard1.extract_shard(&full, 4);
        assert_eq!(s1, vec![3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_row_parallel_shard() {
        // O projection: [1024, 1024] split across 2 GPUs
        let shard0 = RowParallelShard::new(1024, 1024, 0, 2);
        let shard1 = RowParallelShard::new(1024, 1024, 1, 2);

        assert_eq!(shard0.row_start, 0);
        assert_eq!(shard0.row_end, 512);
        assert_eq!(shard0.num_elements(), 512 * 1024);

        assert_eq!(shard1.row_start, 512);
        assert_eq!(shard1.row_end, 1024);
    }

    #[test]
    fn test_row_parallel_extract() {
        // Small example: [4, 2] split into [2, 2] each
        let full = vec![
            1.0, 2.0, // row 0
            3.0, 4.0, // row 1
            5.0, 6.0, // row 2
            7.0, 8.0, // row 3
        ];
        let shard0 = RowParallelShard::new(4, 2, 0, 2);
        let shard1 = RowParallelShard::new(4, 2, 1, 2);

        let s0 = shard0.extract_shard(&full, 4);
        assert_eq!(s0, vec![1.0, 2.0, 3.0, 4.0]);

        let s1 = shard1.extract_shard(&full, 4);
        assert_eq!(s1, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_tp_comm_cost() {
        // 350M: S=1024, H=1024, L=24
        let cost = TpCommCost::estimate(1024, 1024, 24);
        assert_eq!(cost.bytes_per_allreduce, 1024 * 1024 * 4); // 4 MB
        assert_eq!(cost.allreduces_per_block, 2);
        assert_eq!(cost.total_bytes_per_step(), 4 * 1024 * 1024 * 2 * 24); // 192 MB

        // At 100 Gbps NVLink: ~15 ms overhead
        let overhead = cost.estimated_overhead_ms(100.0);
        assert!(overhead > 0.0);
        assert!(overhead < 100.0); // sanity check
    }

    #[test]
    fn test_tp_config_4way() {
        let tp = TensorParallelConfig::new(2, 4, 1024, 4096, 16, 4);
        assert_eq!(tp.local_num_heads(), 4);
        assert_eq!(tp.local_num_kv_heads(), 1);
        assert_eq!(tp.local_q_size(), 4 * 64);
        assert_eq!(tp.local_intermediate_size(), 1024);
    }
}
