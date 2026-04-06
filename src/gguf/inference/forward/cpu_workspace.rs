// PMAT-305: Pre-allocated CPU inference workspace.
// Eliminates ~364 heap allocations per token by reusing buffers.
// PMAT-304 showed realizr IPC 1.60 vs llama.cpp 1.01 — extra cycles
// from malloc/free + cache line pollution from touching cold memory.

/// Pre-allocated buffers for CPU inference forward pass.
/// All buffers sized for M=1 decode (single token).
pub(crate) struct CpuWorkspace {
    /// Buffer A: hidden_dim sized (RMSNorm output, residuals)
    pub buf_hidden_a: Vec<f32>,
    /// Buffer B: hidden_dim sized
    pub buf_hidden_b: Vec<f32>,
    /// Buffer for QKV projection output: qkv_dim sized
    pub buf_qkv: Vec<f32>,
    /// Buffer for attention output: q_dim sized
    pub buf_attn: Vec<f32>,
    /// Buffer for FFN gate output: intermediate_dim sized
    pub buf_ffn_gate: Vec<f32>,
    /// Buffer for FFN up output: intermediate_dim sized
    pub buf_ffn_up: Vec<f32>,
    /// Buffer for FFN act (gate*silu(up)): intermediate_dim sized
    pub buf_ffn_act: Vec<f32>,
    /// Buffer for output projection: hidden_dim sized
    pub buf_output: Vec<f32>,
}

impl CpuWorkspace {
    /// Create workspace sized for the given model dimensions.
    pub fn new(
        hidden_dim: usize,
        qkv_dim: usize,
        intermediate_dim: usize,
    ) -> Self {
        Self {
            buf_hidden_a: vec![0.0; hidden_dim],
            buf_hidden_b: vec![0.0; hidden_dim],
            buf_qkv: vec![0.0; qkv_dim],
            buf_attn: vec![0.0; hidden_dim], // q_dim <= hidden_dim
            buf_ffn_gate: vec![0.0; intermediate_dim],
            buf_ffn_up: vec![0.0; intermediate_dim],
            buf_ffn_act: vec![0.0; intermediate_dim],
            buf_output: vec![0.0; hidden_dim],
        }
    }
}
