//! PMAT-291: Tensor Compute Graph for GPU Inference
//!
//! Inspired by ggml's compute graph pattern (transpiled via decy for reference).
//! Pure Rust, no FFI. Reduces ~430 individual cuLaunchKernel dispatches to ~15
//! tensor-level operations per decode step.
//!
//! # Design (from cross-project analysis)
//!
//! - ggml: C tensor graph with ~15 nodes, CUDA graph replay = 1 launch
//! - vLLM: PyTorch/inductor IR fusion + CUDA graphs (~80 nodes)
//! - realizr current: 430 individual kernel dispatches
//! - realizr target: ~15 tensor ops via this module + CUDA graph replay
//!
//! # Academic References
//!
//! - [Kwon et al., SOSP 2023] PagedAttention (arxiv:2309.06180)
//! - [Yu et al., OSDI 2022] Orca iteration-level scheduling
//! - [Dao, NeurIPS 2022] FlashAttention (arxiv:2205.14135)

pub mod executor;

pub use executor::{execute_graph, KernelDispatch};

/// Tensor operation types for decoder inference.
///
/// Each variant maps to ONE kernel dispatch. The goal is to express
/// an entire transformer layer as ~5 operations:
/// 1. RmsNorm (pre-attention)
/// 2. QKV+Attention (fused projection + attention + output projection)
/// 3. Residual add
/// 4. RmsNorm (pre-FFN)
/// 5. FFN (gate+up+swiglu+down fused)
/// 6. Residual add
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorOp {
    /// Matrix-vector multiply (Q4K dequant+GEMV or cuBLASLt GEMM)
    MulMat,
    /// Element-wise add (residual connections)
    Add,
    /// RMS normalization
    RmsNorm,
    /// Rotary position embedding
    Rope,
    /// Softmax (attention scores)
    SoftMax,
    /// Element-wise multiply (SwiGLU gate)
    Mul,
    /// SiLU activation
    Silu,
    /// Memory copy (KV cache scatter)
    Copy,
    /// No-op (input tensor, leaf node)
    None,
}

/// A node in the compute graph.
///
/// Each node represents a tensor operation with device-memory pointers
/// to its data and references to input nodes (by index).
#[derive(Debug, Clone)]
pub struct TensorNode {
    /// Operation to perform
    pub op: TensorOp,
    /// Device pointer to output data
    pub data_ptr: u64,
    /// Output dimensions [rows, cols, batch, unused]
    pub shape: [u32; 4],
    /// Indices of input nodes in the graph (max 3: src0, src1, src2)
    pub inputs: Vec<usize>,
    /// Operation-specific parameters (e.g., epsilon for RmsNorm, position for RoPE)
    pub params: OpParams,
}

/// Operation-specific parameters.
#[derive(Debug, Clone)]
pub struct OpParams {
    /// Weight pointer (for MulMat: quantized weights on device)
    pub weight_ptr: u64,
    /// Normalization gamma pointer (for RmsNorm)
    pub gamma_ptr: u64,
    /// Scalar parameter (epsilon for RmsNorm, etc.)
    pub scalar: f32,
    /// Integer parameter (position for RoPE, etc.)
    pub int_param: u32,
    /// PERF-050 (aprender#2753): GGML quantization type code of `weight_ptr`.
    ///
    /// Without this the graph carried only a pointer, and the dispatcher had no way to know
    /// what it pointed at -- so it assumed Q4_K for every MulMat. In a q4_k_m model that is
    /// right for 170 tensors and wrong for 29: attn_v, ffn_down and the LM head are Q6_K, and
    /// dequantizing Q6_K with the Q4_K kernel produced garbage of magnitude ~5e8 with NaNs.
    ///
    /// The numeric GGML code is used rather than a Rust enum because this crate must not depend
    /// on the serving crate's `WeightQuantType`. 12 = Q4_K, which is also the `Default`, so
    /// nodes that do not set it behave exactly as before.
    pub weight_qtype: u32,
}

/// GGML type code for Q4_K, the historical hardcoded assumption and this field's default.
pub const GGML_TYPE_Q4_K: u32 = 12;

/// Compute graph: topologically sorted list of tensor operations.
///
/// Built once per model architecture, reused every decode step.
/// Only the data pointers and parameters change between steps.
#[derive(Debug, Clone)]
pub struct ComputeGraph {
    /// Nodes in topological order (leafs first, output last)
    pub nodes: Vec<TensorNode>,
    /// Number of leaf nodes (inputs, no operation)
    pub n_leafs: usize,
}

impl ComputeGraph {
    /// Create an empty compute graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            n_leafs: 0,
        }
    }

    /// Add a leaf node (input tensor, no operation).
    pub fn add_leaf(&mut self, data_ptr: u64, shape: [u32; 4]) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::None,
            data_ptr,
            shape,
            inputs: Vec::new(),
            params: OpParams::default(),
        });
        self.n_leafs += 1;
        idx
    }

    /// Add an operation node with inputs.
    pub fn add_op(
        &mut self,
        op: TensorOp,
        data_ptr: u64,
        shape: [u32; 4],
        inputs: Vec<usize>,
        params: OpParams,
    ) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TensorNode {
            op,
            data_ptr,
            shape,
            inputs,
            params,
        });
        idx
    }

    /// Number of operation nodes (excludes leafs).
    pub fn n_ops(&self) -> usize {
        self.nodes.len() - self.n_leafs
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let g = ComputeGraph::new();
        assert_eq!(g.nodes.len(), 0);
        assert_eq!(g.n_leafs, 0);
        assert_eq!(g.n_ops(), 0);
    }

    #[test]
    fn test_simple_graph() {
        let mut g = ComputeGraph::new();

        // Input tensor (leaf)
        let input = g.add_leaf(0x1000, [1536, 1, 1, 0]);

        // RmsNorm
        let normed = g.add_op(
            TensorOp::RmsNorm,
            0x2000,
            [1536, 1, 1, 0],
            vec![input],
            OpParams {
                gamma_ptr: 0x3000,
                scalar: 1e-6,
                ..Default::default()
            },
        );

        // MulMat (Q projection)
        let q = g.add_op(
            TensorOp::MulMat,
            0x4000,
            [1536, 1, 1, 0],
            vec![normed],
            OpParams {
                weight_ptr: 0x5000,
                ..Default::default()
            },
        );

        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.n_leafs, 1);
        assert_eq!(g.n_ops(), 2);
        assert_eq!(g.nodes[q].op, TensorOp::MulMat);
        assert_eq!(g.nodes[q].inputs, vec![normed]);
    }
}

impl Default for OpParams {
    fn default() -> Self {
        Self {
            weight_ptr: 0,
            gamma_ptr: 0,
            scalar: 0.0,
            int_param: 0,
            // PERF-050: Q4_K, matching the assumption this field replaces, so an unset node is
            // bit-identical to the old behaviour.
            weight_qtype: GGML_TYPE_Q4_K,
        }
    }
}
