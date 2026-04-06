//! PMAT-291: Transformer layer graph builder for Qwen2.5 architecture.
//!
//! Builds a ComputeGraph representing one transformer layer's decode step.
//! The graph has 14 nodes (1 leaf + 13 ops) vs the current ~15 individual
//! kernel launches per layer. Combined with CUDA graph capture across 28
//! layers, total dispatches drop from ~430 to ~392 (14 x 28) then to 1
//! graph replay.

use trueno_gpu::graph::{ComputeGraph, OpParams, TensorOp};

use crate::cuda::types::ValidatedLayerWeights;

/// Build a compute graph for one transformer decoder layer (Qwen2.5 architecture).
///
/// The graph represents:
/// ```text
/// Phase 1 (Attention):
///   input -> rmsnorm -> Q_proj -> K_proj -> V_proj -> attention(+RoPE+KV) -> O_proj -> residual_1
/// Phase 2 (FFN):
///   residual_1 -> rmsnorm -> gate_proj -> up_proj -> swiglu(gate,up) -> down_proj -> residual_2
/// ```
///
/// Returns the graph and the index of the final output node.
#[allow(clippy::too_many_arguments)]
pub fn build_layer_graph(
    layer_weights: &ValidatedLayerWeights,
    input_ptr: u64,
    hidden_dim: u32,
    intermediate_dim: u32,
    q_dim: u32,
    kv_dim: u32,
    m: u32,
    epsilon: f32,
    layer_idx: usize,
    // Workspace buffer pointers (pre-allocated by init_batched_workspace)
    hidden_buf1_ptr: u64,
    hidden_buf2_ptr: u64,
    q_buf_ptr: u64,
    k_buf_ptr: u64,
    v_buf_ptr: u64,
    attn_out_ptr: u64,
    ffn_gate_ptr: u64,
    ffn_up_ptr: u64,
    ffn_act_ptr: u64,
    input_staging_ptr: u64,
) -> (ComputeGraph, usize) {
    let mut g = ComputeGraph::new();

    // ========== Leaf: input hidden state ==========
    let input = g.add_leaf(input_ptr, [hidden_dim, 1, m, 0]);

    // ========== Phase 1: Attention ==========

    // 1. Pre-attention RMSNorm
    let normed_attn = g.add_op(
        TensorOp::RmsNorm,
        hidden_buf1_ptr,
        [hidden_dim, 1, m, 0],
        vec![input],
        OpParams {
            gamma_ptr: layer_weights.attn_norm_ptr,
            scalar: epsilon,
            ..Default::default()
        },
    );

    // 2. Q projection (Q4K or Q6K GEMV/GEMM)
    let q = g.add_op(
        TensorOp::MulMat,
        q_buf_ptr,
        [q_dim, hidden_dim, m, 0],
        vec![normed_attn],
        OpParams {
            weight_ptr: layer_weights.attn_q_ptr,
            ..Default::default()
        },
    );

    // 3. K projection
    let k = g.add_op(
        TensorOp::MulMat,
        k_buf_ptr,
        [kv_dim, hidden_dim, m, 0],
        vec![normed_attn],
        OpParams {
            weight_ptr: layer_weights.attn_k_ptr,
            ..Default::default()
        },
    );

    // 4. V projection
    let v = g.add_op(
        TensorOp::MulMat,
        v_buf_ptr,
        [kv_dim, hidden_dim, m, 0],
        vec![normed_attn],
        OpParams {
            weight_ptr: layer_weights.attn_v_ptr,
            ..Default::default()
        },
    );

    // 5. Attention (compound: RoPE + KV scatter + incremental attention)
    // The dispatcher handles all sub-operations internally via dispatch_attention.
    // Positions are passed through CudaExecutor::graph_dispatch_positions side-channel.
    let attn_out = g.add_op(
        TensorOp::SoftMax,
        attn_out_ptr,
        [q_dim, 1, m, 0],
        vec![q, k, v],
        OpParams {
            int_param: layer_idx as u32,
            ..Default::default()
        },
    );

    // 6. Output projection
    let o_proj = g.add_op(
        TensorOp::MulMat,
        hidden_buf1_ptr,
        [hidden_dim, q_dim, m, 0],
        vec![attn_out],
        OpParams {
            weight_ptr: layer_weights.attn_output_ptr,
            ..Default::default()
        },
    );

    // 7. First residual: input + o_proj -> input_staging
    let residual_1 = g.add_op(
        TensorOp::Add,
        input_staging_ptr,
        [hidden_dim, 1, m, 0],
        vec![input, o_proj],
        OpParams::default(),
    );

    // ========== Phase 2: FFN ==========

    // 8. Pre-FFN RMSNorm
    let normed_ffn = g.add_op(
        TensorOp::RmsNorm,
        hidden_buf1_ptr,
        [hidden_dim, 1, m, 0],
        vec![residual_1],
        OpParams {
            gamma_ptr: layer_weights.ffn_norm_ptr,
            scalar: epsilon,
            ..Default::default()
        },
    );

    // 9. Gate projection
    let gate = g.add_op(
        TensorOp::MulMat,
        ffn_gate_ptr,
        [intermediate_dim, hidden_dim, m, 0],
        vec![normed_ffn],
        OpParams {
            weight_ptr: layer_weights.ffn_gate_ptr,
            ..Default::default()
        },
    );

    // 10. Up projection
    let up = g.add_op(
        TensorOp::MulMat,
        ffn_up_ptr,
        [intermediate_dim, hidden_dim, m, 0],
        vec![normed_ffn],
        OpParams {
            weight_ptr: layer_weights.ffn_up_ptr,
            ..Default::default()
        },
    );

    // 11. SwiGLU: gate * silu(up) — element-wise, dispatched as Mul(gate, up)
    // The CudaExecutor dispatch_mul maps this to batched_swiglu_into.
    let ffn_act = g.add_op(
        TensorOp::Mul,
        ffn_act_ptr,
        [intermediate_dim, 1, m, 0],
        vec![gate, up],
        OpParams::default(),
    );

    // 12. Down projection
    let down = g.add_op(
        TensorOp::MulMat,
        hidden_buf1_ptr,
        [hidden_dim, intermediate_dim, m, 0],
        vec![ffn_act],
        OpParams {
            weight_ptr: layer_weights.ffn_down_ptr,
            ..Default::default()
        },
    );

    // 13. Second residual: input_staging + down -> hidden_buf2
    let residual_2 = g.add_op(
        TensorOp::Add,
        hidden_buf2_ptr,
        [hidden_dim, 1, m, 0],
        vec![residual_1, down],
        OpParams::default(),
    );

    (g, residual_2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::types::{IndexedLayerWeights, WeightQuantType};

    fn mock_weights() -> ValidatedLayerWeights {
        let raw = IndexedLayerWeights {
            attn_q_ptr: 0x10000,
            attn_q_len: 1024,
            attn_q_qtype: WeightQuantType::Q4K,
            attn_k_ptr: 0x20000,
            attn_k_len: 512,
            attn_k_qtype: WeightQuantType::Q4K,
            attn_v_ptr: 0x30000,
            attn_v_len: 512,
            attn_v_qtype: WeightQuantType::Q6K,
            attn_output_ptr: 0x40000,
            attn_output_len: 1024,
            attn_output_qtype: WeightQuantType::Q4K,
            ffn_gate_ptr: 0x50000,
            ffn_gate_len: 2048,
            ffn_gate_qtype: WeightQuantType::Q4K,
            ffn_up_ptr: 0x60000,
            ffn_up_len: 2048,
            ffn_up_qtype: WeightQuantType::Q4K,
            ffn_down_ptr: 0x70000,
            ffn_down_len: 2048,
            ffn_down_qtype: WeightQuantType::Q4K,
            attn_norm_ptr: 0x80000,
            attn_norm_len: 256,
            ffn_norm_ptr: 0x90000,
            ffn_norm_len: 256,
            attn_q_bias_ptr: 0,
            attn_q_bias_len: 0,
            attn_k_bias_ptr: 0,
            attn_k_bias_len: 0,
            attn_v_bias_ptr: 0,
            attn_v_bias_len: 0,
            attn_q_norm_ptr: 0,
            attn_q_norm_len: 0,
            attn_k_norm_ptr: 0,
            attn_k_norm_len: 0,
        };
        ValidatedLayerWeights::new_unchecked(raw)
    }

    #[test]
    fn test_layer_graph_node_count() {
        let weights = mock_weights();
        let (graph, output_idx) = build_layer_graph(
            &weights, 0xA0000, 1536, 8960, 1536, 256, 4, 1e-6, 0, 0xB0000, 0xC0000, 0xD0000,
            0xE0000, 0xF0000, 0x100000, 0x110000, 0x120000, 0x130000, 0x140000,
        );

        // 1 leaf + 13 ops = 14 nodes total
        assert_eq!(graph.nodes.len(), 14);
        assert_eq!(graph.n_leafs, 1);
        assert_eq!(graph.n_ops(), 13);
        assert_eq!(output_idx, 13);
    }

    #[test]
    fn test_layer_graph_execution_count() {
        use trueno_gpu::graph::execute_graph;

        let weights = mock_weights();
        let (graph, _) = build_layer_graph(
            &weights, 0xA0000, 1536, 8960, 1536, 256, 4, 1e-6, 0, 0xB0000, 0xC0000, 0xD0000,
            0xE0000, 0xF0000, 0x100000, 0x110000, 0x120000, 0x130000, 0x140000,
        );

        struct Counter(usize);
        impl trueno_gpu::graph::KernelDispatch for Counter {
            fn dispatch_mul_mat(
                &mut self,
                _: &trueno_gpu::graph::TensorNode,
                _: u64,
                _: u64,
                _: u32,
                _: u32,
                _: u32,
            ) -> Result<(), trueno_gpu::GpuError> {
                self.0 += 1;
                Ok(())
            }
            fn dispatch_rms_norm(
                &mut self,
                _: &trueno_gpu::graph::TensorNode,
                _: u64,
                _: u64,
                _: u32,
                _: u32,
                _: f32,
            ) -> Result<(), trueno_gpu::GpuError> {
                self.0 += 1;
                Ok(())
            }
            fn dispatch_add(
                &mut self,
                _: u64,
                _: u64,
                _: u64,
                _: usize,
            ) -> Result<(), trueno_gpu::GpuError> {
                self.0 += 1;
                Ok(())
            }
            fn dispatch_rope(
                &mut self,
                _: &trueno_gpu::graph::TensorNode,
                _: u64,
                _: &[u32],
                _: u32,
                _: u32,
            ) -> Result<(), trueno_gpu::GpuError> {
                self.0 += 1;
                Ok(())
            }
            fn dispatch_attention(
                &mut self,
                _: &trueno_gpu::graph::TensorNode,
                _: u64,
                _: u64,
                _: u64,
                _: u64,
                _: u32,
                _: usize,
            ) -> Result<(), trueno_gpu::GpuError> {
                self.0 += 1;
                Ok(())
            }
            fn dispatch_copy(
                &mut self,
                _: u64,
                _: u64,
                _: usize,
            ) -> Result<(), trueno_gpu::GpuError> {
                self.0 += 1;
                Ok(())
            }
            fn dispatch_mul(
                &mut self,
                _: u64,
                _: u64,
                _: u64,
                _: usize,
            ) -> Result<(), trueno_gpu::GpuError> {
                self.0 += 1;
                Ok(())
            }
            fn dispatch_silu(
                &mut self,
                _: u64,
                _: u64,
                _: usize,
            ) -> Result<(), trueno_gpu::GpuError> {
                self.0 += 1;
                Ok(())
            }
        }

        let mut counter = Counter(0);
        let n = execute_graph(&graph, &mut counter).unwrap();

        // 13 ops per layer: 2 rmsnorm + 7 mul_mat + 1 attention + 2 add + 1 swiglu
        assert_eq!(n, 13);
        assert_eq!(counter.0, 13);
    }
}
