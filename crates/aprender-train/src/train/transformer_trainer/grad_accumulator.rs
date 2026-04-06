#![allow(dead_code)]
// Per-block gradient accumulation for distributed data-parallel pretraining.
//
// The `CudaTransformerTrainer` uses a shared `CudaGradWorkspace` that is
// overwritten for each block during backward. For DDP, we need to:
//
// 1. After block[i]'s backward, copy workspace gradients into `block_grads[i]`
// 2. AllReduce `block_grads[i]` across workers
// 3. Run optimizer_step for block[i] with averaged gradients
//
// This module provides CPU-side accumulation buffers that hold the per-block
// gradients after they are downloaded from GPU. These buffers are what get
// sent over the wire for AllReduce.
//
// # Contract
//
// C-DDP-001: Per-block gradient buffers match CudaGradWorkspace component sizes.

/// Number of gradient components per transformer block.
/// Matches CudaGradWorkspace: w_q, w_k, w_v, w_o, gate, up, down, input_norm, post_attn_norm
pub const BLOCK_GRAD_COMPONENTS: usize = 9;

/// Component indices for transformer block gradients.
pub mod component {
    pub const W_Q: usize = 0;
    pub const W_K: usize = 1;
    pub const W_V: usize = 2;
    pub const W_O: usize = 3;
    pub const GATE: usize = 4;
    pub const UP: usize = 5;
    pub const DOWN: usize = 6;
    pub const INPUT_NORM: usize = 7;
    pub const POST_ATTN_NORM: usize = 8;
}

/// Non-block component IDs (for wire protocol).
pub mod non_block {
    pub const LM_HEAD: u8 = 0;
    pub const FINAL_NORM: u8 = 1;
    pub const EMBEDDING: u8 = 2;
}

/// Gradient set for a single transformer block (CPU-side).
///
/// Contains 9 flattened f32 gradient vectors, one per CudaGradWorkspace component.
/// These are downloaded from GPU after backward and before AllReduce.
#[derive(Debug, Clone)]
pub struct BlockGradientSet {
    /// Gradient components, indexed by `component::*` constants
    pub components: Vec<Vec<f32>>,
}

impl BlockGradientSet {
    /// Create a new zeroed gradient set with the given component sizes.
    ///
    /// # Arguments
    /// * `sizes` - Element count for each of the 9 components
    pub fn zeroed(sizes: &[usize; BLOCK_GRAD_COMPONENTS]) -> Self {
        let components = sizes.iter().map(|&sz| vec![0.0f32; sz]).collect();
        Self { components }
    }

    /// Total number of f32 elements across all components.
    pub fn total_elements(&self) -> usize {
        self.components.iter().map(Vec::len).sum()
    }

    /// Get component sizes as u32 (for wire protocol).
    pub fn component_sizes_u32(&self) -> Vec<u32> {
        self.components.iter().map(|c| c.len() as u32).collect()
    }

    /// Flatten all components into a single contiguous Vec<f32>.
    pub fn flatten(&self) -> Vec<f32> {
        let total = self.total_elements();
        let mut flat = Vec::with_capacity(total);
        for comp in &self.components {
            flat.extend_from_slice(comp);
        }
        flat
    }

    /// Reconstruct from a flat gradient vector and component sizes.
    ///
    /// # Panics
    /// Panics if `flat.len() != sum(sizes)`.
    pub fn from_flat(flat: &[f32], sizes: &[u32]) -> Self {
        let total: usize = sizes.iter().map(|&s| s as usize).sum();
        assert_eq!(flat.len(), total, "flat gradient length mismatch");
        let mut components = Vec::with_capacity(sizes.len());
        let mut offset = 0;
        for &sz in sizes {
            let sz = sz as usize;
            components.push(flat[offset..offset + sz].to_vec());
            offset += sz;
        }
        Self { components }
    }

    /// Zero all gradient components (reuse buffers).
    pub fn zero(&mut self) {
        for comp in &mut self.components {
            for x in comp.iter_mut() {
                *x = 0.0;
            }
        }
    }

    /// Element-wise add another gradient set into this one.
    ///
    /// # Panics
    /// Panics if component sizes don't match.
    pub fn accumulate(&mut self, other: &BlockGradientSet) {
        assert_eq!(self.components.len(), other.components.len());
        for (dst, src) in self.components.iter_mut().zip(&other.components) {
            assert_eq!(dst.len(), src.len(), "component size mismatch");
            for (d, s) in dst.iter_mut().zip(src) {
                *d += s;
            }
        }
    }

    /// Divide all gradient elements by a scalar (for averaging).
    pub fn scale(&mut self, divisor: f32) {
        let inv = 1.0 / divisor;
        for comp in &mut self.components {
            for x in comp.iter_mut() {
                *x *= inv;
            }
        }
    }

    /// Check if any element is NaN or Inf (Jidoka safety check).
    pub fn has_non_finite(&self) -> bool {
        self.components.iter().any(|comp| comp.iter().any(|x| !x.is_finite()))
    }
}

/// Per-block gradient accumulator for the full model.
///
/// Holds one `BlockGradientSet` per transformer block, plus separate
/// buffers for LM head, final norm, and embedding gradients.
///
/// # VRAM Cost
///
/// For 350M model (H=1024, I=4096, D_kv=256, L=24):
/// - Per block: ~2.8M f32 = ~11.2 MB
/// - 24 blocks: ~268 MB total
/// - Non-block (LM head + final norm + embedding): ~67 MB
/// - Total: ~335 MB CPU RAM (not VRAM — these are CPU-side buffers)
#[derive(Debug)]
pub struct PerBlockGradientAccumulator {
    /// Per-block gradient buffers
    pub block_grads: Vec<BlockGradientSet>,
    /// LM head weight gradient [vocab_size * hidden_size]
    pub lm_head_grad: Vec<f32>,
    /// Final norm weight gradient [hidden_size]
    pub final_norm_grad: Vec<f32>,
    /// Embedding weight gradient [vocab_size * hidden_size]
    pub embedding_grad: Vec<f32>,
    /// Number of accumulated micro-batches
    pub accumulated_count: usize,
    /// Component sizes per block (cached for wire protocol)
    pub block_component_sizes: [usize; BLOCK_GRAD_COMPONENTS],
}

impl PerBlockGradientAccumulator {
    /// Create a new accumulator with zeroed buffers.
    ///
    /// # Arguments
    /// * `num_blocks` - Number of transformer layers (e.g., 24 for 350M)
    /// * `block_sizes` - Element count for each of the 9 gradient components per block
    /// * `vocab_size` - Vocabulary size (for LM head and embedding)
    /// * `hidden_size` - Hidden dimension (for final norm)
    pub fn new(
        num_blocks: usize,
        block_sizes: [usize; BLOCK_GRAD_COMPONENTS],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Self {
        let block_grads = (0..num_blocks).map(|_| BlockGradientSet::zeroed(&block_sizes)).collect();

        Self {
            block_grads,
            lm_head_grad: vec![0.0; vocab_size * hidden_size],
            final_norm_grad: vec![0.0; hidden_size],
            embedding_grad: vec![0.0; vocab_size * hidden_size],
            accumulated_count: 0,
            block_component_sizes: block_sizes,
        }
    }

    /// Compute the per-block component sizes from model architecture.
    ///
    /// # Arguments
    /// * `hidden_size` - H
    /// * `kv_hidden_size` - D_kv = (H / num_heads) * num_kv_heads
    /// * `intermediate_size` - I (FFN intermediate dimension)
    pub fn compute_block_sizes(
        hidden_size: usize,
        kv_hidden_size: usize,
        intermediate_size: usize,
    ) -> [usize; BLOCK_GRAD_COMPONENTS] {
        [
            hidden_size * hidden_size,       // w_q
            hidden_size * kv_hidden_size,    // w_k
            hidden_size * kv_hidden_size,    // w_v
            hidden_size * hidden_size,       // w_o
            hidden_size * intermediate_size, // gate
            hidden_size * intermediate_size, // up
            intermediate_size * hidden_size, // down
            hidden_size,                     // input_norm
            hidden_size,                     // post_attn_norm
        ]
    }

    /// Zero all accumulated gradients (call at the start of each step).
    pub fn zero_all(&mut self) {
        for block_grad in &mut self.block_grads {
            block_grad.zero();
        }
        self.lm_head_grad.iter_mut().for_each(|x| *x = 0.0);
        self.final_norm_grad.iter_mut().for_each(|x| *x = 0.0);
        self.embedding_grad.iter_mut().for_each(|x| *x = 0.0);
        self.accumulated_count = 0;
    }

    /// Average accumulated gradients by dividing by the accumulated count.
    pub fn average(&mut self) {
        if self.accumulated_count <= 1 {
            return;
        }
        let n = self.accumulated_count as f32;
        for block_grad in &mut self.block_grads {
            block_grad.scale(n);
        }
        let inv = 1.0 / n;
        for x in &mut self.lm_head_grad {
            *x *= inv;
        }
        for x in &mut self.final_norm_grad {
            *x *= inv;
        }
        for x in &mut self.embedding_grad {
            *x *= inv;
        }
    }

    /// Check if any block has NaN or Inf gradients (Jidoka).
    pub fn has_non_finite(&self) -> bool {
        self.block_grads.iter().any(BlockGradientSet::has_non_finite)
            || self.lm_head_grad.iter().any(|x| !x.is_finite())
            || self.final_norm_grad.iter().any(|x| !x.is_finite())
            || self.embedding_grad.iter().any(|x| !x.is_finite())
    }

    /// Number of transformer blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_grads.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_gradient_set_zeroed() {
        let sizes = [100, 50, 50, 100, 200, 200, 200, 10, 10];
        let bg = BlockGradientSet::zeroed(&sizes);
        assert_eq!(bg.components.len(), 9);
        assert_eq!(bg.total_elements(), 920);
        assert!(bg.components[0].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_block_gradient_set_flatten_roundtrip() {
        let sizes = [4, 2, 2, 4, 8, 8, 8, 1, 1];
        let mut bg = BlockGradientSet::zeroed(&sizes);
        // Fill with test data
        for (i, comp) in bg.components.iter_mut().enumerate() {
            for (j, val) in comp.iter_mut().enumerate() {
                *val = (i * 100 + j) as f32;
            }
        }
        let flat = bg.flatten();
        assert_eq!(flat.len(), 38);

        let sizes_u32 = bg.component_sizes_u32();
        let reconstructed = BlockGradientSet::from_flat(&flat, &sizes_u32);
        for (orig, recon) in bg.components.iter().zip(&reconstructed.components) {
            assert_eq!(orig, recon);
        }
    }

    #[test]
    fn test_block_gradient_set_accumulate() {
        let sizes = [2, 2, 2, 2, 2, 2, 2, 1, 1];
        let mut a = BlockGradientSet::zeroed(&sizes);
        let mut b = BlockGradientSet::zeroed(&sizes);
        a.components[0] = vec![1.0, 2.0];
        b.components[0] = vec![3.0, 4.0];
        a.accumulate(&b);
        assert_eq!(a.components[0], vec![4.0, 6.0]);
    }

    #[test]
    fn test_block_gradient_set_scale() {
        let sizes = [2, 1, 1, 1, 1, 1, 1, 1, 1];
        let mut bg = BlockGradientSet::zeroed(&sizes);
        bg.components[0] = vec![6.0, 9.0];
        bg.scale(3.0);
        assert!((bg.components[0][0] - 2.0).abs() < 1e-6);
        assert!((bg.components[0][1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_block_gradient_set_has_non_finite() {
        let sizes = [2, 1, 1, 1, 1, 1, 1, 1, 1];
        let mut bg = BlockGradientSet::zeroed(&sizes);
        assert!(!bg.has_non_finite());
        bg.components[0][0] = f32::NAN;
        assert!(bg.has_non_finite());
    }

    #[test]
    fn test_accumulator_new() {
        let sizes = PerBlockGradientAccumulator::compute_block_sizes(1024, 256, 4096);
        let acc = PerBlockGradientAccumulator::new(24, sizes, 32768, 1024);
        assert_eq!(acc.num_blocks(), 24);
        assert_eq!(acc.lm_head_grad.len(), 32768 * 1024);
        assert_eq!(acc.final_norm_grad.len(), 1024);
        assert_eq!(acc.embedding_grad.len(), 32768 * 1024);
    }

    #[test]
    fn test_accumulator_compute_block_sizes_350m() {
        // 350M: H=1024, num_heads=16, num_kv_heads=4, kv_dim=256, I=4096
        let sizes = PerBlockGradientAccumulator::compute_block_sizes(1024, 256, 4096);
        assert_eq!(sizes[component::W_Q], 1024 * 1024); // 1M
        assert_eq!(sizes[component::W_K], 1024 * 256); // 256K
        assert_eq!(sizes[component::W_V], 1024 * 256); // 256K
        assert_eq!(sizes[component::W_O], 1024 * 1024); // 1M
        assert_eq!(sizes[component::GATE], 1024 * 4096); // 4M
        assert_eq!(sizes[component::UP], 1024 * 4096); // 4M
        assert_eq!(sizes[component::DOWN], 4096 * 1024); // 4M
        assert_eq!(sizes[component::INPUT_NORM], 1024);
        assert_eq!(sizes[component::POST_ATTN_NORM], 1024);
    }

    #[test]
    fn test_accumulator_zero_all() {
        let sizes = [2, 1, 1, 1, 1, 1, 1, 1, 1];
        let mut acc = PerBlockGradientAccumulator::new(2, sizes, 10, 2);
        acc.block_grads[0].components[0] = vec![1.0, 2.0];
        acc.lm_head_grad[0] = 5.0;
        acc.accumulated_count = 3;
        acc.zero_all();
        assert!(acc.block_grads[0].components[0].iter().all(|&x| x == 0.0));
        assert_eq!(acc.lm_head_grad[0], 0.0);
        assert_eq!(acc.accumulated_count, 0);
    }

    #[test]
    fn test_accumulator_has_non_finite() {
        let sizes = [2, 1, 1, 1, 1, 1, 1, 1, 1];
        let mut acc = PerBlockGradientAccumulator::new(2, sizes, 10, 2);
        assert!(!acc.has_non_finite());
        acc.lm_head_grad[0] = f32::INFINITY;
        assert!(acc.has_non_finite());
    }
}
