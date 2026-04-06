//! Pipeline parallelism for transformer pretraining.
//!
//! Splits transformer blocks across multiple workers by layer range.
//! Each worker runs forward/backward only for its assigned blocks.
//! Inter-worker communication passes activations (forward) and
//! gradients (backward) between adjacent pipeline stages.
//!
//! # Architecture
//!
//! ```text
//! Worker 0 (blocks 0-11)    Worker 1 (blocks 12-23)
//! ─────────────────────    ──────────────────────────
//! embed → block[0..12]  →  block[12..24] → lm_head
//!                       ←  grad_activations
//! ```
//!
//! # Schedule: 1F1B (One Forward, One Backward)
//!
//! With M micro-batches:
//! 1. Warmup: pipeline fills with forward passes (M-1 forwards)
//! 2. Steady state: alternate 1 forward + 1 backward
//! 3. Cooldown: drain remaining backward passes
//!
//! Pipeline bubble = (P-1) / M of total compute, where P = pipeline stages.
//!
//! # Contract (C-PIPE-001)
//!
//! - Each block is owned by exactly one pipeline stage
//! - Activation tensor shapes are consistent at stage boundaries
//! - Gradient flow is continuous across stage boundaries

/// Pipeline stage assignment for a worker.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage index (0 = first, closest to embedding)
    pub stage_id: usize,
    /// Total number of pipeline stages
    pub num_stages: usize,
    /// First block index (inclusive)
    pub block_start: usize,
    /// Last block index (exclusive)
    pub block_end: usize,
    /// Whether this stage owns the embedding layer
    pub has_embedding: bool,
    /// Whether this stage owns the LM head
    pub has_lm_head: bool,
    /// Number of micro-batches for 1F1B schedule
    pub num_micro_batches: usize,
}

impl PipelineStage {
    /// Create a pipeline stage assignment.
    ///
    /// # Arguments
    /// * `stage_id` - This worker's stage (0-indexed)
    /// * `num_stages` - Total pipeline stages (typically 2 or 4)
    /// * `num_blocks` - Total transformer blocks
    /// * `num_micro_batches` - Micro-batches for 1F1B (must be >= num_stages)
    ///
    /// # Panics
    /// Panics if `num_micro_batches < num_stages` (can't fill pipeline).
    pub fn new(
        stage_id: usize,
        num_stages: usize,
        num_blocks: usize,
        num_micro_batches: usize,
    ) -> Self {
        assert!(
            num_micro_batches >= num_stages,
            "need at least {num_stages} micro-batches to fill pipeline, got {num_micro_batches}"
        );

        let blocks_per_stage = num_blocks / num_stages;
        let remainder = num_blocks % num_stages;

        let block_start = if stage_id < remainder {
            stage_id * (blocks_per_stage + 1)
        } else {
            remainder * (blocks_per_stage + 1) + (stage_id - remainder) * blocks_per_stage
        };

        let block_end = if stage_id < remainder {
            block_start + blocks_per_stage + 1
        } else {
            block_start + blocks_per_stage
        };

        Self {
            stage_id,
            num_stages,
            block_start,
            block_end,
            has_embedding: stage_id == 0,
            has_lm_head: stage_id == num_stages - 1,
            num_micro_batches,
        }
    }

    /// Number of blocks in this stage.
    pub fn num_blocks(&self) -> usize {
        self.block_end - self.block_start
    }

    /// Whether this is the first pipeline stage.
    pub fn is_first(&self) -> bool {
        self.stage_id == 0
    }

    /// Whether this is the last pipeline stage.
    pub fn is_last(&self) -> bool {
        self.stage_id == self.num_stages - 1
    }

    /// Compute pipeline bubble fraction.
    ///
    /// Returns the fraction of time spent idle due to pipeline bubbles.
    /// Bubble = (P - 1) / M where P = stages, M = micro-batches.
    pub fn bubble_fraction(&self) -> f64 {
        (self.num_stages as f64 - 1.0) / self.num_micro_batches as f64
    }

    /// Compute pipeline efficiency (1 - bubble fraction).
    pub fn efficiency(&self) -> f64 {
        1.0 - self.bubble_fraction()
    }

    /// Generate 1F1B schedule for this stage.
    ///
    /// Returns a sequence of (action, micro_batch_id) pairs.
    /// Action: Forward or Backward.
    pub fn schedule_1f1b(&self) -> Vec<PipelineAction> {
        let m = self.num_micro_batches;
        let p = self.num_stages;
        let mut actions = Vec::new();

        // Warmup phase: forward passes to fill pipeline
        let warmup_forwards = p - self.stage_id - 1;
        for mb in 0..warmup_forwards.min(m) {
            actions.push(PipelineAction::Forward(mb));
        }

        // Steady state: 1F1B pairs
        let steady_start = warmup_forwards.min(m);
        let mut next_fwd = steady_start;
        let mut next_bwd = 0;

        while next_fwd < m || next_bwd < m {
            // One forward (if remaining)
            if next_fwd < m {
                actions.push(PipelineAction::Forward(next_fwd));
                next_fwd += 1;
            }
            // One backward (if remaining)
            if next_bwd < m {
                actions.push(PipelineAction::Backward(next_bwd));
                next_bwd += 1;
            }
        }

        actions
    }
}

/// Action in a 1F1B pipeline schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineAction {
    /// Run forward pass for micro-batch N
    Forward(usize),
    /// Run backward pass for micro-batch N
    Backward(usize),
}

/// Activation buffer between pipeline stages.
///
/// Stores intermediate activations sent from stage N to stage N+1
/// during forward, and gradients sent from stage N+1 to stage N
/// during backward.
#[derive(Debug, Clone)]
pub struct PipelineActivationBuffer {
    /// Stored activations per micro-batch: `[micro_batch][seq_len * hidden_size]`
    pub forward_activations: Vec<Vec<f32>>,
    /// Stored gradients per micro-batch (from downstream stage)
    pub backward_gradients: Vec<Vec<f32>>,
    /// Number of micro-batches
    pub num_micro_batches: usize,
    /// Elements per activation tensor (seq_len * hidden_size)
    pub activation_size: usize,
}

impl PipelineActivationBuffer {
    /// Create a new activation buffer.
    ///
    /// # Arguments
    /// * `num_micro_batches` - Number of micro-batches
    /// * `seq_len` - Sequence length
    /// * `hidden_size` - Hidden dimension
    pub fn new(num_micro_batches: usize, seq_len: usize, hidden_size: usize) -> Self {
        let activation_size = seq_len * hidden_size;
        Self {
            forward_activations: vec![Vec::new(); num_micro_batches],
            backward_gradients: vec![Vec::new(); num_micro_batches],
            num_micro_batches,
            activation_size,
        }
    }

    /// Store forward activation for a micro-batch.
    pub fn store_activation(&mut self, micro_batch: usize, activation: Vec<f32>) {
        assert_eq!(
            activation.len(),
            self.activation_size,
            "activation size mismatch: expected {}, got {}",
            self.activation_size,
            activation.len()
        );
        self.forward_activations[micro_batch] = activation;
    }

    /// Store backward gradient for a micro-batch.
    pub fn store_gradient(&mut self, micro_batch: usize, gradient: Vec<f32>) {
        assert_eq!(
            gradient.len(),
            self.activation_size,
            "gradient size mismatch: expected {}, got {}",
            self.activation_size,
            gradient.len()
        );
        self.backward_gradients[micro_batch] = gradient;
    }

    /// Get forward activation for a micro-batch.
    pub fn get_activation(&self, micro_batch: usize) -> &[f32] {
        &self.forward_activations[micro_batch]
    }

    /// Get backward gradient for a micro-batch.
    pub fn get_gradient(&self, micro_batch: usize) -> &[f32] {
        &self.backward_gradients[micro_batch]
    }

    /// Total memory used by this buffer in bytes.
    pub fn memory_bytes(&self) -> usize {
        let fwd: usize = self.forward_activations.iter().map(|v| v.len() * 4).sum();
        let bwd: usize = self.backward_gradients.iter().map(|v| v.len() * 4).sum();
        fwd + bwd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_stage_basic() {
        // 24 blocks, 2 stages, 4 micro-batches
        let stage0 = PipelineStage::new(0, 2, 24, 4);
        let stage1 = PipelineStage::new(1, 2, 24, 4);

        assert_eq!(stage0.block_start, 0);
        assert_eq!(stage0.block_end, 12);
        assert_eq!(stage0.num_blocks(), 12);
        assert!(stage0.has_embedding);
        assert!(!stage0.has_lm_head);

        assert_eq!(stage1.block_start, 12);
        assert_eq!(stage1.block_end, 24);
        assert!(stage1.has_lm_head);
        assert!(!stage1.has_embedding);
    }

    #[test]
    fn test_pipeline_stage_4way() {
        // 24 blocks, 4 stages → 6 each
        for i in 0..4 {
            let stage = PipelineStage::new(i, 4, 24, 8);
            assert_eq!(stage.num_blocks(), 6);
            assert_eq!(stage.block_start, i * 6);
            assert_eq!(stage.block_end, (i + 1) * 6);
        }
    }

    #[test]
    fn test_pipeline_stage_uneven() {
        // 10 blocks, 3 stages → 4, 3, 3
        let s0 = PipelineStage::new(0, 3, 10, 6);
        let s1 = PipelineStage::new(1, 3, 10, 6);
        let s2 = PipelineStage::new(2, 3, 10, 6);

        assert_eq!(s0.num_blocks(), 4);
        assert_eq!(s1.num_blocks(), 3);
        assert_eq!(s2.num_blocks(), 3);

        // Complete coverage
        assert_eq!(s0.block_end, s1.block_start);
        assert_eq!(s1.block_end, s2.block_start);
        assert_eq!(s2.block_end, 10);
    }

    #[test]
    fn test_pipeline_bubble_fraction() {
        // 2 stages, 4 micro-batches → bubble = 1/4 = 25%
        let stage = PipelineStage::new(0, 2, 24, 4);
        assert!((stage.bubble_fraction() - 0.25).abs() < 1e-10);
        assert!((stage.efficiency() - 0.75).abs() < 1e-10);

        // 4 stages, 8 micro-batches → bubble = 3/8 = 37.5%
        let stage = PipelineStage::new(0, 4, 24, 8);
        assert!((stage.bubble_fraction() - 0.375).abs() < 1e-10);

        // 2 stages, 16 micro-batches → bubble = 1/16 = 6.25%
        let stage = PipelineStage::new(0, 2, 24, 16);
        assert!((stage.bubble_fraction() - 0.0625).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_1f1b_schedule() {
        let stage = PipelineStage::new(0, 2, 24, 4);
        let schedule = stage.schedule_1f1b();

        // Count forwards and backwards
        let fwd_count = schedule.iter().filter(|a| matches!(a, PipelineAction::Forward(_))).count();
        let bwd_count =
            schedule.iter().filter(|a| matches!(a, PipelineAction::Backward(_))).count();

        assert_eq!(fwd_count, 4, "should have 4 forwards");
        assert_eq!(bwd_count, 4, "should have 4 backwards");

        // All micro-batches covered
        let mut fwd_ids: Vec<_> = schedule
            .iter()
            .filter_map(|a| match a {
                PipelineAction::Forward(id) => Some(*id),
                _ => None,
            })
            .collect();
        fwd_ids.sort_unstable();
        assert_eq!(fwd_ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_pipeline_activation_buffer() {
        let mut buf = PipelineActivationBuffer::new(2, 512, 1024);
        assert_eq!(buf.activation_size, 512 * 1024);

        let act = vec![1.0f32; 512 * 1024];
        buf.store_activation(0, act.clone());
        assert_eq!(buf.get_activation(0).len(), 512 * 1024);
        assert_eq!(buf.get_activation(0)[0], 1.0);

        let grad = vec![0.5f32; 512 * 1024];
        buf.store_gradient(1, grad);
        assert_eq!(buf.get_gradient(1)[0], 0.5);
    }

    #[test]
    fn test_pipeline_first_last_stage() {
        let s0 = PipelineStage::new(0, 3, 12, 6);
        let s1 = PipelineStage::new(1, 3, 12, 6);
        let s2 = PipelineStage::new(2, 3, 12, 6);

        assert!(s0.is_first());
        assert!(!s0.is_last());
        assert!(!s1.is_first());
        assert!(!s1.is_last());
        assert!(!s2.is_first());
        assert!(s2.is_last());
    }

    #[test]
    #[should_panic(expected = "need at least")]
    fn test_pipeline_too_few_micro_batches() {
        PipelineStage::new(0, 4, 24, 2); // 2 < 4 stages
    }
}
