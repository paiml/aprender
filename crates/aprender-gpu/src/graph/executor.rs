//! PMAT-291: Graph executor -- dispatches tensor operations to CUDA kernels.
//!
//! Each TensorOp maps to ONE kernel launch. The executor walks the compute
//! graph in topological order and dispatches the appropriate kernel for each
//! node. Combined with CUDA graph capture, this reduces 430 launches to
//! ~15 tensor-level dispatches, then 1 graph replay.
//!
//! # Kernel Mapping
//!
//! | TensorOp  | Kernel                        | Source |
//! |-----------|-------------------------------|--------|
//! | MulMat    | BatchedHwDp4aQ4KGemvKernel    | trueno-gpu/kernels/quantize/q4k/ |
//! | RmsNorm   | BatchedVectorizedRmsNormKernel | trueno-gpu/kernels/layernorm/ |
//! | Add       | BatchedResidualAddKernel       | trueno-gpu/kernels/ |
//! | Rope      | BatchedRopeKernel              | trueno-gpu/kernels/ |
//! | Mul       | FusedGateUpSwigluKernel        | trueno-gpu/kernels/quantize/q4k/ |
//! | SoftMax   | (attention dispatch)           | realizr attention module |
//! | Copy      | cuMemcpyDtoDAsync              | trueno driver |

use super::{ComputeGraph, TensorNode, TensorOp};

/// Result of executing a compute graph.
#[derive(Debug)]
pub struct GraphExecResult {
    /// Number of kernel launches performed
    pub n_launches: usize,
    /// Total execution time in microseconds (if timing enabled)
    pub elapsed_us: Option<u64>,
}

/// Trait for dispatching tensor operations to GPU kernels.
///
/// Implementors provide the actual kernel launch logic for each TensorOp.
/// This decouples the graph execution from the specific kernel implementations,
/// allowing realizr to plug in its own kernel dispatch (DP4A, FP8, cuBLASLt).
pub trait KernelDispatch {
    /// Dispatch a MulMat operation (quantized GEMV or GEMM).
    ///
    /// # Arguments
    /// * `node` - The tensor node with weight_ptr in params and input data
    /// * `input_ptr` - Device pointer to input activation
    /// * `output_ptr` - Device pointer to output buffer
    /// * `m` - Batch size
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    fn dispatch_mul_mat(
        &mut self,
        node: &TensorNode,
        input_ptr: u64,
        output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), crate::GpuError>;

    /// Dispatch a RmsNorm operation.
    fn dispatch_rms_norm(
        &mut self,
        node: &TensorNode,
        input_ptr: u64,
        output_ptr: u64,
        hidden_dim: u32,
        m: u32,
        epsilon: f32,
    ) -> Result<(), crate::GpuError>;

    /// Dispatch an element-wise Add (residual connection).
    fn dispatch_add(
        &mut self,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n_elements: usize,
    ) -> Result<(), crate::GpuError>;

    /// Dispatch RoPE position embedding.
    fn dispatch_rope(
        &mut self,
        node: &TensorNode,
        qk_ptr: u64,
        positions: &[u32],
        head_dim: u32,
        num_heads: u32,
    ) -> Result<(), crate::GpuError>;

    /// Dispatch attention (incremental or flash).
    fn dispatch_attention(
        &mut self,
        node: &TensorNode,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        output_ptr: u64,
        m: u32,
        layer_idx: usize,
    ) -> Result<(), crate::GpuError>;

    /// Dispatch KV cache scatter (copy).
    fn dispatch_copy(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        size_bytes: usize,
    ) -> Result<(), crate::GpuError>;

    /// Dispatch element-wise multiply (SwiGLU gate).
    fn dispatch_mul(
        &mut self,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n_elements: usize,
    ) -> Result<(), crate::GpuError>;

    /// Dispatch SiLU activation.
    fn dispatch_silu(
        &mut self,
        input_ptr: u64,
        output_ptr: u64,
        n_elements: usize,
    ) -> Result<(), crate::GpuError>;
}

/// Execute a compute graph using the provided kernel dispatcher.
///
/// Walks nodes in topological order. Leaf nodes (TensorOp::None) are
/// skipped -- they represent input tensors whose data is already on device.
///
/// Returns the number of kernel launches performed.
pub fn execute_graph<D: KernelDispatch>(
    graph: &ComputeGraph,
    dispatcher: &mut D,
) -> Result<usize, crate::GpuError> {
    let mut n_launches = 0;

    for node in &graph.nodes {
        match node.op {
            TensorOp::None => {
                // Leaf node -- input data already on device, nothing to dispatch
            }
            TensorOp::MulMat => {
                let input_idx = node.inputs.first().copied().unwrap_or(0);
                let input_ptr = graph.nodes[input_idx].data_ptr;
                dispatcher.dispatch_mul_mat(
                    node,
                    input_ptr,
                    node.data_ptr,
                    node.shape[2], // m (batch)
                    node.shape[0], // n (output dim)
                    node.shape[1], // k (input dim)
                )?;
                n_launches += 1;
            }
            TensorOp::RmsNorm => {
                let input_idx = node.inputs.first().copied().unwrap_or(0);
                let input_ptr = graph.nodes[input_idx].data_ptr;
                dispatcher.dispatch_rms_norm(
                    node,
                    input_ptr,
                    node.data_ptr,
                    node.shape[0],      // hidden_dim
                    node.shape[2],      // m (batch)
                    node.params.scalar, // epsilon
                )?;
                n_launches += 1;
            }
            TensorOp::Add => {
                let a_idx = node.inputs.first().copied().unwrap_or(0);
                let b_idx = node.inputs.get(1).copied().unwrap_or(0);
                let a_ptr = graph.nodes[a_idx].data_ptr;
                let b_ptr = graph.nodes[b_idx].data_ptr;
                let n_elements = (node.shape[0] * node.shape[2]) as usize;
                dispatcher.dispatch_add(a_ptr, b_ptr, node.data_ptr, n_elements)?;
                n_launches += 1;
            }
            TensorOp::Rope => {
                let input_idx = node.inputs.first().copied().unwrap_or(0);
                let input_ptr = graph.nodes[input_idx].data_ptr;
                // positions passed via params.int_param as base position
                dispatcher.dispatch_rope(
                    node,
                    input_ptr,
                    &[],           // positions filled at runtime
                    node.shape[0], // head_dim
                    node.shape[1], // num_heads
                )?;
                n_launches += 1;
            }
            TensorOp::SoftMax => {
                // Attention is dispatched as a compound operation
                // The dispatcher handles Q/K/V/output internally
                let q_idx = node.inputs.first().copied().unwrap_or(0);
                let k_idx = node.inputs.get(1).copied().unwrap_or(0);
                let v_idx = node.inputs.get(2).copied().unwrap_or(0);
                dispatcher.dispatch_attention(
                    node,
                    graph.nodes[q_idx].data_ptr,
                    graph.nodes[k_idx].data_ptr,
                    graph.nodes[v_idx].data_ptr,
                    node.data_ptr,
                    node.shape[2],                  // m
                    node.params.int_param as usize, // layer_idx
                )?;
                n_launches += 1;
            }
            TensorOp::Copy => {
                let src_idx = node.inputs.first().copied().unwrap_or(0);
                let src_ptr = graph.nodes[src_idx].data_ptr;
                let size = (node.shape[0] * node.shape[1] * 4) as usize; // f32
                dispatcher.dispatch_copy(src_ptr, node.data_ptr, size)?;
                n_launches += 1;
            }
            TensorOp::Mul => {
                let a_idx = node.inputs.first().copied().unwrap_or(0);
                let b_idx = node.inputs.get(1).copied().unwrap_or(0);
                let n_elements = (node.shape[0] * node.shape[2]) as usize;
                dispatcher.dispatch_mul(
                    graph.nodes[a_idx].data_ptr,
                    graph.nodes[b_idx].data_ptr,
                    node.data_ptr,
                    n_elements,
                )?;
                n_launches += 1;
            }
            TensorOp::Silu => {
                let input_idx = node.inputs.first().copied().unwrap_or(0);
                let n_elements = (node.shape[0] * node.shape[2]) as usize;
                dispatcher.dispatch_silu(
                    graph.nodes[input_idx].data_ptr,
                    node.data_ptr,
                    n_elements,
                )?;
                n_launches += 1;
            }
        }
    }

    Ok(n_launches)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock dispatcher that counts launches
    struct CountingDispatcher {
        launches: usize,
    }

    impl KernelDispatch for CountingDispatcher {
        fn dispatch_mul_mat(
            &mut self,
            _: &TensorNode,
            _: u64,
            _: u64,
            _: u32,
            _: u32,
            _: u32,
        ) -> Result<(), crate::GpuError> {
            self.launches += 1;
            Ok(())
        }
        fn dispatch_rms_norm(
            &mut self,
            _: &TensorNode,
            _: u64,
            _: u64,
            _: u32,
            _: u32,
            _: f32,
        ) -> Result<(), crate::GpuError> {
            self.launches += 1;
            Ok(())
        }
        fn dispatch_add(
            &mut self,
            _: u64,
            _: u64,
            _: u64,
            _: usize,
        ) -> Result<(), crate::GpuError> {
            self.launches += 1;
            Ok(())
        }
        fn dispatch_rope(
            &mut self,
            _: &TensorNode,
            _: u64,
            _: &[u32],
            _: u32,
            _: u32,
        ) -> Result<(), crate::GpuError> {
            self.launches += 1;
            Ok(())
        }
        fn dispatch_attention(
            &mut self,
            _: &TensorNode,
            _: u64,
            _: u64,
            _: u64,
            _: u64,
            _: u32,
            _: usize,
        ) -> Result<(), crate::GpuError> {
            self.launches += 1;
            Ok(())
        }
        fn dispatch_copy(&mut self, _: u64, _: u64, _: usize) -> Result<(), crate::GpuError> {
            self.launches += 1;
            Ok(())
        }
        fn dispatch_mul(
            &mut self,
            _: u64,
            _: u64,
            _: u64,
            _: usize,
        ) -> Result<(), crate::GpuError> {
            self.launches += 1;
            Ok(())
        }
        fn dispatch_silu(&mut self, _: u64, _: u64, _: usize) -> Result<(), crate::GpuError> {
            self.launches += 1;
            Ok(())
        }
    }

    #[test]
    fn test_execute_empty_graph() {
        let g = ComputeGraph::new();
        let mut d = CountingDispatcher { launches: 0 };
        let n = execute_graph(&g, &mut d).unwrap();
        assert_eq!(n, 0);
        assert_eq!(d.launches, 0);
    }

    #[test]
    fn test_execute_single_layer_graph() {
        use super::super::OpParams;

        let mut g = ComputeGraph::new();

        // Build a minimal transformer layer graph:
        // input -> rmsnorm -> mul_mat(Q) -> attention -> add(residual)
        let input = g.add_leaf(0x1000, [1536, 1, 4, 0]);
        let normed = g.add_op(
            TensorOp::RmsNorm,
            0x2000,
            [1536, 1, 4, 0],
            vec![input],
            OpParams { gamma_ptr: 0x3000, scalar: 1e-6, ..Default::default() },
        );
        let q = g.add_op(
            TensorOp::MulMat,
            0x4000,
            [1536, 1536, 4, 0],
            vec![normed],
            OpParams { weight_ptr: 0x5000, ..Default::default() },
        );
        let k = g.add_op(
            TensorOp::MulMat,
            0x6000,
            [256, 1536, 4, 0],
            vec![normed],
            OpParams { weight_ptr: 0x7000, ..Default::default() },
        );
        let v = g.add_op(
            TensorOp::MulMat,
            0x8000,
            [256, 1536, 4, 0],
            vec![normed],
            OpParams { weight_ptr: 0x9000, ..Default::default() },
        );
        let attn = g.add_op(
            TensorOp::SoftMax,
            0xA000,
            [1536, 1, 4, 0],
            vec![q, k, v],
            OpParams { int_param: 0, ..Default::default() },
        );
        let _residual = g.add_op(
            TensorOp::Add,
            0xB000,
            [1536, 1, 4, 0],
            vec![input, attn],
            OpParams::default(),
        );

        let mut d = CountingDispatcher { launches: 0 };
        let n = execute_graph(&g, &mut d).unwrap();

        // 6 ops: rmsnorm + 3 mul_mat + attention + add = 6 launches
        assert_eq!(n, 6);
        assert_eq!(d.launches, 6);
        assert_eq!(g.n_ops(), 6);
    }
}
