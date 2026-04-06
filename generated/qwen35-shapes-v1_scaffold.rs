/// Contract: Qwen3.5-9B concrete shape instantiation and RoPE frequency scaling v1.0.0
/// Paper: Qwen3.5 Fine-Tune Spec — model configuration
/// Paper: Su et al. (2021) RoFormer — Rotary Position Embedding
pub trait KernelContract {
    /// [n_kv * d_k, hidden] = [4*256, 4096] = [1024, 4096]
    /// Domain: Qwen3.5-9B config: n_kv=4, d_k=256
    /// INVARIANT: GQA ratio: n_h / n_kv = 4
    /// INVARIANT (Q projection shape): n_h * d_k = 4096 for Qwen3.5-9B
    /// INVARIANT (KV projection shape): n_kv * d_k = 1024 for Qwen3.5-9B
    /// INVARIANT (SwiGLU expansion ratio): 12288 / 4096 = 3.0
    /// INVARIANT (O projection transpose): shape(o_proj) == reverse(shape(q_proj))
    /// INVARIANT (RoPE frequency vector length): len(freqs) == d_k / 2
    /// MONOTONICITY (RoPE frequency decreasing): freq_i > freq_{i+1} for all i
    /// EQUIVALENCE (SIMD shape equivalence):
    fn kv_projection_shape(&self, input: &[f32], output: &mut [f32]);
    /// shape(o_proj) == transpose(shape(q_proj)) = [hidden, n_h * d_k]
    /// Domain: Standard transformer
    /// INVARIANT: O projection reverses Q projection dimensions
    /// INVARIANT (Q projection shape): n_h * d_k = 4096 for Qwen3.5-9B
    /// INVARIANT (KV projection shape): n_kv * d_k = 1024 for Qwen3.5-9B
    /// INVARIANT (SwiGLU expansion ratio): 12288 / 4096 = 3.0
    /// INVARIANT (O projection transpose): shape(o_proj) == reverse(shape(q_proj))
    /// INVARIANT (RoPE frequency vector length): len(freqs) == d_k / 2
    /// MONOTONICITY (RoPE frequency decreasing): freq_i > freq_{i+1} for all i
    /// EQUIVALENCE (SIMD shape equivalence):
    fn o_projection_transpose(&self, input: &[f32], output: &mut [f32]);
    /// [n_h * d_k, hidden] = [16*256, 4096] = [4096, 4096]
    /// Domain: Qwen3.5-9B config: n_h=16, d_k=256, hidden=4096
    /// INVARIANT: Q projection is square for this config
    /// INVARIANT (Q projection shape): n_h * d_k = 4096 for Qwen3.5-9B
    /// INVARIANT (KV projection shape): n_kv * d_k = 1024 for Qwen3.5-9B
    /// INVARIANT (SwiGLU expansion ratio): 12288 / 4096 = 3.0
    /// INVARIANT (O projection transpose): shape(o_proj) == reverse(shape(q_proj))
    /// INVARIANT (RoPE frequency vector length): len(freqs) == d_k / 2
    /// MONOTONICITY (RoPE frequency decreasing): freq_i > freq_{i+1} for all i
    /// EQUIVALENCE (SIMD shape equivalence):
    fn q_projection_shape(&self, input: &[f32], output: &mut [f32]);
    /// freq_i = base^(-2i/d_k) for i in [0, d_k/2)
    /// Domain: base = rope_theta, d_k = head_dim
    /// INVARIANT: len(freqs) = d_k / 2
    /// INVARIANT: freq_0 = 1.0
    /// INVARIANT: Strictly decreasing
    /// INVARIANT (Q projection shape): n_h * d_k = 4096 for Qwen3.5-9B
    /// INVARIANT (KV projection shape): n_kv * d_k = 1024 for Qwen3.5-9B
    /// INVARIANT (SwiGLU expansion ratio): 12288 / 4096 = 3.0
    /// INVARIANT (O projection transpose): shape(o_proj) == reverse(shape(q_proj))
    /// INVARIANT (RoPE frequency vector length): len(freqs) == d_k / 2
    /// MONOTONICITY (RoPE frequency decreasing): freq_i > freq_{i+1} for all i
    /// EQUIVALENCE (SIMD shape equivalence):
    fn rope_frequency(&self, input: &[f32], output: &mut [f32]);
    /// intermediate / hidden = 12288 / 4096 = 3.0
    /// Domain: Qwen3.5-9B config
    /// INVARIANT: Expansion ratio is exactly 3.0
    /// INVARIANT (Q projection shape): n_h * d_k = 4096 for Qwen3.5-9B
    /// INVARIANT (KV projection shape): n_kv * d_k = 1024 for Qwen3.5-9B
    /// INVARIANT (SwiGLU expansion ratio): 12288 / 4096 = 3.0
    /// INVARIANT (O projection transpose): shape(o_proj) == reverse(shape(q_proj))
    /// INVARIANT (RoPE frequency vector length): len(freqs) == d_k / 2
    /// MONOTONICITY (RoPE frequency decreasing): freq_i > freq_{i+1} for all i
    /// EQUIVALENCE (SIMD shape equivalence):
    fn swiglu_ratio(&self, input: &[f32], output: &mut [f32]);
}

// Qwen3.5-9B config constants
const HIDDEN: usize = 4096;
const N_HEADS: usize = 16;
const N_KV_HEADS: usize = 4;
const D_K: usize = 256;
const INTERMEDIATE: usize = 12288;
const ROPE_THETA: f64 = 1_000_000.0;

/// Concrete verifier implementing the Qwen3.5-9B shapes contract
pub struct Qwen35ShapesVerifier;

impl KernelContract for Qwen35ShapesVerifier {
    fn q_projection_shape(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 2);
        output[0] = (N_HEADS * D_K) as f32; // 4096
        output[1] = HIDDEN as f32; // 4096
    }

    fn kv_projection_shape(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 2);
        output[0] = (N_KV_HEADS * D_K) as f32; // 1024
        output[1] = HIDDEN as f32; // 4096
    }

    fn o_projection_transpose(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 2);
        output[0] = HIDDEN as f32; // 4096
        output[1] = (N_HEADS * D_K) as f32; // 4096
    }

    fn rope_frequency(&self, _input: &[f32], output: &mut [f32]) {
        let freq_len = D_K / 2;
        assert!(output.len() >= freq_len);
        for i in 0..freq_len {
            output[i] = ROPE_THETA.powf(-2.0 * i as f64 / D_K as f64) as f32;
        }
    }

    fn swiglu_ratio(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 1);
        output[0] = INTERMEDIATE as f32 / HIDDEN as f32; // 3.0
    }
}
