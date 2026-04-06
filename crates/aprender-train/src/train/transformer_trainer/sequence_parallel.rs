//! Sequence parallelism for transformer pretraining.
//!
//! Distributes the sequence dimension across multiple GPUs. Each GPU
//! processes a contiguous chunk of the sequence, with all-to-all
//! communication for attention computation.
//!
//! # Architecture (Ring Attention)
//!
//! ```text
//! GPU 0: tokens[0..S/2]      GPU 1: tokens[S/2..S]
//! ──────────────────────     ──────────────────────
//! Q₀ = embed(tok[0..S/2])   Q₁ = embed(tok[S/2..S])
//! K₀ = proj(Q₀)             K₁ = proj(Q₁)
//! V₀ = proj(Q₀)             V₁ = proj(Q₁)
//!
//! Ring step 1: attn(Q₀, K₀, V₀) + recv K₁,V₁ from GPU 1
//! Ring step 2: attn(Q₀, K₁, V₁) + send K₀,V₀ to GPU 1
//! ─── Reduce attention outputs ───
//! ```
//!
//! # Communication Pattern
//!
//! Each GPU sends its K,V to the next GPU in the ring and receives K,V
//! from the previous GPU. After N-1 ring steps, each GPU has computed
//! attention against all K,V chunks.
//!
//! # When to Use
//!
//! Most valuable when sequence length >> hidden size (8K+ sequences).
//! Reduces peak memory from O(S² × H) to O((S/N)² × H × N) = O(S²/N × H).
//!
//! # Contract (C-SP-001)
//!
//! - Sequence chunks are contiguous and non-overlapping
//! - Each GPU's attention output is identical to the full-sequence result
//! - Ring communication maintains causal mask correctness

/// Sequence parallel configuration.
#[derive(Debug, Clone)]
pub struct SequenceParallelConfig {
    /// This GPU's rank in the SP group
    pub sp_rank: usize,
    /// Total GPUs in the SP group
    pub sp_size: usize,
    /// Full sequence length (before sharding)
    pub full_seq_len: usize,
    /// Hidden size (not sharded in SP)
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl SequenceParallelConfig {
    /// Create a new SP config.
    ///
    /// # Panics
    /// Panics if sequence length is not divisible by sp_size.
    pub fn new(
        sp_rank: usize,
        sp_size: usize,
        full_seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
    ) -> Self {
        assert!(
            full_seq_len.is_multiple_of(sp_size),
            "seq_len ({full_seq_len}) must be divisible by sp_size ({sp_size})"
        );

        let head_dim = hidden_size / num_heads;

        Self { sp_rank, sp_size, full_seq_len, hidden_size, num_heads, head_dim }
    }

    /// Local sequence length on this GPU.
    pub fn local_seq_len(&self) -> usize {
        self.full_seq_len / self.sp_size
    }

    /// Start token index for this GPU's chunk.
    pub fn seq_start(&self) -> usize {
        self.sp_rank * self.local_seq_len()
    }

    /// End token index (exclusive) for this GPU's chunk.
    pub fn seq_end(&self) -> usize {
        self.seq_start() + self.local_seq_len()
    }

    /// Memory savings for attention scores.
    ///
    /// Attention score matrix: [num_heads, local_seq, full_seq] per GPU.
    /// Without SP: [num_heads, full_seq, full_seq].
    /// Savings: 1 - 1/sp_size (e.g., 50% with 2 GPUs).
    pub fn attention_memory_savings(&self) -> f64 {
        1.0 - (1.0 / self.sp_size as f64)
    }

    /// Number of ring communication steps needed.
    ///
    /// Each GPU must see all other GPUs' K,V → sp_size - 1 steps.
    pub fn ring_steps(&self) -> usize {
        self.sp_size - 1
    }
}

/// Ring attention schedule for a single GPU.
///
/// Generates the sequence of (send_to, recv_from) pairs for each ring step.
#[derive(Debug, Clone)]
pub struct RingAttentionSchedule {
    /// Steps in the ring attention protocol
    pub steps: Vec<RingStep>,
    /// This GPU's rank
    pub rank: usize,
    /// Total GPUs
    pub world_size: usize,
}

/// A single step in the ring attention protocol.
#[derive(Debug, Clone, Copy)]
pub struct RingStep {
    /// Ring step index (0-based)
    pub step: usize,
    /// Rank to send K,V to
    pub send_to: usize,
    /// Rank to receive K,V from
    pub recv_from: usize,
    /// The chunk index being processed (which rank's K,V we're using)
    pub kv_chunk_source: usize,
}

impl RingAttentionSchedule {
    /// Generate a ring attention schedule for a given rank.
    ///
    /// In each step, GPU sends its K,V to the right neighbor and
    /// receives K,V from the left neighbor.
    pub fn new(rank: usize, world_size: usize) -> Self {
        let mut steps = Vec::with_capacity(world_size - 1);

        for step in 0..world_size - 1 {
            let send_to = (rank + 1) % world_size;
            let recv_from = (rank + world_size - 1) % world_size;
            // After `step` rotations, we have K,V from rank (rank - step - 1) mod N
            let kv_chunk_source = (rank + world_size - step - 1) % world_size;

            steps.push(RingStep { step, send_to, recv_from, kv_chunk_source });
        }

        Self { steps, rank, world_size }
    }

    /// Check if a ring step requires a causal mask adjustment.
    ///
    /// For causal (autoregressive) attention, tokens can only attend
    /// to earlier tokens. When processing K,V from a later chunk,
    /// the causal mask must block attention to future tokens.
    pub fn needs_causal_mask(&self, step: usize, local_seq_len: usize) -> CausalMaskType {
        let kv_source = self.steps[step].kv_chunk_source;
        let q_start = self.rank * local_seq_len;
        let kv_start = kv_source * local_seq_len;

        if kv_start + local_seq_len <= q_start {
            // KV chunk is entirely before Q chunk → full attention
            CausalMaskType::FullAttention
        } else if kv_start >= q_start + local_seq_len {
            // KV chunk is entirely after Q chunk → no attention (skip)
            CausalMaskType::NoAttention
        } else {
            // KV chunk overlaps with Q chunk → apply causal mask
            CausalMaskType::CausalMask
        }
    }
}

/// Type of causal mask needed for a ring attention step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalMaskType {
    /// All tokens can attend (KV is before Q)
    FullAttention,
    /// No tokens can attend (KV is after Q) — skip computation
    NoAttention,
    /// Standard causal mask needed (KV overlaps with Q)
    CausalMask,
}

/// Communication cost estimate for sequence parallelism.
#[derive(Debug, Clone)]
pub struct SpCommCost {
    /// Bytes per K,V send (local_seq × head_dim × num_kv_heads × sizeof(f32))
    pub kv_bytes_per_send: usize,
    /// Number of ring steps (sp_size - 1)
    pub ring_steps: usize,
    /// Number of blocks
    pub num_blocks: usize,
}

impl SpCommCost {
    /// Estimate SP communication cost.
    pub fn estimate(
        local_seq_len: usize,
        head_dim: usize,
        num_kv_heads: usize,
        sp_size: usize,
        num_blocks: usize,
    ) -> Self {
        // K + V = 2 × local_seq × kv_dim
        let kv_bytes_per_send =
            2 * local_seq_len * head_dim * num_kv_heads * std::mem::size_of::<f32>();

        Self { kv_bytes_per_send, ring_steps: sp_size - 1, num_blocks }
    }

    /// Total bytes communicated per training step.
    pub fn total_bytes_per_step(&self) -> usize {
        self.kv_bytes_per_send * self.ring_steps * self.num_blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sp_config_basic() {
        let sp = SequenceParallelConfig::new(0, 2, 2048, 1024, 16);
        assert_eq!(sp.local_seq_len(), 1024);
        assert_eq!(sp.seq_start(), 0);
        assert_eq!(sp.seq_end(), 1024);
        assert!((sp.attention_memory_savings() - 0.5).abs() < 1e-10);
        assert_eq!(sp.ring_steps(), 1);
    }

    #[test]
    fn test_sp_config_4way() {
        let sp = SequenceParallelConfig::new(2, 4, 8192, 1024, 16);
        assert_eq!(sp.local_seq_len(), 2048);
        assert_eq!(sp.seq_start(), 4096);
        assert_eq!(sp.seq_end(), 6144);
        assert!((sp.attention_memory_savings() - 0.75).abs() < 1e-10);
        assert_eq!(sp.ring_steps(), 3);
    }

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn test_sp_config_indivisible() {
        SequenceParallelConfig::new(0, 3, 1000, 1024, 16); // 1000 % 3 != 0
    }

    #[test]
    fn test_ring_attention_schedule_2gpu() {
        let sched = RingAttentionSchedule::new(0, 2);
        assert_eq!(sched.steps.len(), 1);
        assert_eq!(sched.steps[0].send_to, 1);
        assert_eq!(sched.steps[0].recv_from, 1);
        assert_eq!(sched.steps[0].kv_chunk_source, 1);
    }

    #[test]
    fn test_ring_attention_schedule_4gpu() {
        let sched = RingAttentionSchedule::new(0, 4);
        assert_eq!(sched.steps.len(), 3);

        // Step 0: send to 1, recv from 3, processing chunk from rank 3
        assert_eq!(sched.steps[0].send_to, 1);
        assert_eq!(sched.steps[0].recv_from, 3);
        assert_eq!(sched.steps[0].kv_chunk_source, 3);

        // Step 1: send to 1, recv from 3, processing chunk from rank 2
        assert_eq!(sched.steps[1].kv_chunk_source, 2);

        // Step 2: processing chunk from rank 1
        assert_eq!(sched.steps[2].kv_chunk_source, 1);
    }

    #[test]
    fn test_ring_attention_all_chunks_seen() {
        // Each GPU must see K,V from all other GPUs
        let world_size = 4;
        for rank in 0..world_size {
            let sched = RingAttentionSchedule::new(rank, world_size);
            let mut seen: Vec<usize> = sched.steps.iter().map(|s| s.kv_chunk_source).collect();
            seen.push(rank); // own chunk (processed locally)
            seen.sort_unstable();
            assert_eq!(seen, vec![0, 1, 2, 3], "rank {rank} didn't see all chunks");
        }
    }

    #[test]
    fn test_causal_mask_type() {
        // 4 GPUs, seq_len=1024, local=256
        let sched = RingAttentionSchedule::new(2, 4); // rank 2: tokens 512..768
        let local_seq = 256;

        // Step 0: kv from rank 1 (tokens 256..512) — before us → full attention
        let mask = sched.needs_causal_mask(0, local_seq);
        assert_eq!(mask, CausalMaskType::FullAttention);

        // Step 2: kv from rank 3 (tokens 768..1024) — after us → no attention
        let mask = sched.needs_causal_mask(2, local_seq);
        assert_eq!(mask, CausalMaskType::NoAttention);
    }

    #[test]
    fn test_sp_comm_cost() {
        // 2 GPUs, seq=2048 (local=1024), head_dim=64, 4 KV heads, 24 blocks
        let cost = SpCommCost::estimate(1024, 64, 4, 2, 24);
        // K+V = 2 × 1024 × 64 × 4 × 4 = 2 MB per send
        assert_eq!(cost.kv_bytes_per_send, 2 * 1024 * 64 * 4 * 4);
        assert_eq!(cost.ring_steps, 1);
        assert_eq!(cost.total_bytes_per_step(), cost.kv_bytes_per_send * 24);
    }
}
