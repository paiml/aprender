/// Contract: Qwen2/2.5-7B concrete shape instantiation and RoPE frequency scaling v1.0.0
/// Paper: Qwen2.5 Technical Report — model configuration
/// Paper: Su et al. (2021) RoFormer — Rotary Position Embedding
pub trait KernelContract {
    /// d_k = hidden_size / num_attention_heads = 3584 / 28 = 128
    /// Domain: Qwen2.5-7B config
    /// INVARIANT: hidden_size is evenly divisible by num_attention_heads
    /// INVARIANT: d_k = 128 (standard head dimension)
    /// INVARIANT (Q projection shape): n_h * d_k = 3584 for Qwen2.5-7B
    /// INVARIANT (KV projection shape): n_kv * d_k = 512 for Qwen2.5-7B
    /// INVARIANT (GQA divisibility): n_h mod n_kv = 28 mod 4 = 0
    /// INVARIANT (SwiGLU gate/up shape): gate_proj.shape = up_proj.shape = [18944, 3584]
    /// INVARIANT (O projection transpose): shape(o_proj) == reverse(shape(q_proj))
    /// INVARIANT (RoPE frequency vector length): len(freqs) == d_k / 2 = 64
    /// MONOTONICITY (RoPE frequency decreasing): freq_i > freq_{i+1} for all i
    /// INVARIANT (Head dimension consistency): 3584 mod 28 = 0 and 3584 / 28 = 128
    /// EQUIVALENCE (SIMD shape equivalence):
    fn head_dim_consistency(&self, input: &[f32], output: &mut [f32]);
    /// [n_kv * d_k, hidden] = [4*128, 3584] = [512, 3584]
    /// Domain: Qwen2.5-7B config: n_kv=4, d_k=128
    /// INVARIANT: GQA ratio: n_h / n_kv = 7
    /// INVARIANT (Q projection shape): n_h * d_k = 3584 for Qwen2.5-7B
    /// INVARIANT (KV projection shape): n_kv * d_k = 512 for Qwen2.5-7B
    /// INVARIANT (GQA divisibility): n_h mod n_kv = 28 mod 4 = 0
    /// INVARIANT (SwiGLU gate/up shape): gate_proj.shape = up_proj.shape = [18944, 3584]
    /// INVARIANT (O projection transpose): shape(o_proj) == reverse(shape(q_proj))
    /// INVARIANT (RoPE frequency vector length): len(freqs) == d_k / 2 = 64
    /// MONOTONICITY (RoPE frequency decreasing): freq_i > freq_{i+1} for all i
    /// INVARIANT (Head dimension consistency): 3584 mod 28 = 0 and 3584 / 28 = 128
    /// EQUIVALENCE (SIMD shape equivalence):
    fn kv_projection_shape(&self, input: &[f32], output: &mut [f32]);
    /// shape(o_proj) == transpose(shape(q_proj)) = [hidden, n_h * d_k]
    /// Domain: Standard transformer
    /// INVARIANT: O projection reverses Q projection dimensions
    /// INVARIANT: For Qwen2.5-7B: [3584, 3584] (square, self-transpose)
    /// INVARIANT (Q projection shape): n_h * d_k = 3584 for Qwen2.5-7B
    /// INVARIANT (KV projection shape): n_kv * d_k = 512 for Qwen2.5-7B
    /// INVARIANT (GQA divisibility): n_h mod n_kv = 28 mod 4 = 0
    /// INVARIANT (SwiGLU gate/up shape): gate_proj.shape = up_proj.shape = [18944, 3584]
    /// INVARIANT (O projection transpose): shape(o_proj) == reverse(shape(q_proj))
    /// INVARIANT (RoPE frequency vector length): len(freqs) == d_k / 2 = 64
    /// MONOTONICITY (RoPE frequency decreasing): freq_i > freq_{i+1} for all i
    /// INVARIANT (Head dimension consistency): 3584 mod 28 = 0 and 3584 / 28 = 128
    /// EQUIVALENCE (SIMD shape equivalence):
    fn o_projection_transpose(&self, input: &[f32], output: &mut [f32]);
    /// [n_h * d_k, hidden] = [28*128, 3584] = [3584, 3584]
    /// Domain: Qwen2.5-7B config: n_h=28, d_k=128, hidden=3584
    /// INVARIANT: Q projection is square for this config
    /// INVARIANT (Q projection shape): n_h * d_k = 3584 for Qwen2.5-7B
    /// INVARIANT (KV projection shape): n_kv * d_k = 512 for Qwen2.5-7B
    /// INVARIANT (GQA divisibility): n_h mod n_kv = 28 mod 4 = 0
    /// INVARIANT (SwiGLU gate/up shape): gate_proj.shape = up_proj.shape = [18944, 3584]
    /// INVARIANT (O projection transpose): shape(o_proj) == reverse(shape(q_proj))
    /// INVARIANT (RoPE frequency vector length): len(freqs) == d_k / 2 = 64
    /// MONOTONICITY (RoPE frequency decreasing): freq_i > freq_{i+1} for all i
    /// INVARIANT (Head dimension consistency): 3584 mod 28 = 0 and 3584 / 28 = 128
    /// EQUIVALENCE (SIMD shape equivalence):
    fn q_projection_shape(&self, input: &[f32], output: &mut [f32]);
    /// freq_i = base^(-2i/d_k) for i in [0, d_k/2)
    /// Domain: base = 1000000.0, d_k = 128
    /// INVARIANT: len(freqs) = d_k / 2 = 64
    /// INVARIANT: freq_0 = 1.0
    /// INVARIANT: Strictly decreasing
    /// INVARIANT (Q projection shape): n_h * d_k = 3584 for Qwen2.5-7B
    /// INVARIANT (KV projection shape): n_kv * d_k = 512 for Qwen2.5-7B
    /// INVARIANT (GQA divisibility): n_h mod n_kv = 28 mod 4 = 0
    /// INVARIANT (SwiGLU gate/up shape): gate_proj.shape = up_proj.shape = [18944, 3584]
    /// INVARIANT (O projection transpose): shape(o_proj) == reverse(shape(q_proj))
    /// INVARIANT (RoPE frequency vector length): len(freqs) == d_k / 2 = 64
    /// MONOTONICITY (RoPE frequency decreasing): freq_i > freq_{i+1} for all i
    /// INVARIANT (Head dimension consistency): 3584 mod 28 = 0 and 3584 / 28 = 128
    /// EQUIVALENCE (SIMD shape equivalence):
    fn rope_frequency(&self, input: &[f32], output: &mut [f32]);
    /// intermediate / hidden = 18944 / 3584 = 37/7 ≈ 5.286
    /// Domain: Qwen2.5-7B config
    /// INVARIANT: Expansion ratio is 37/7 (non-integer, divisible check: 18944 mod 3584 = 0 is false)
    /// INVARIANT: gate_proj and up_proj both have shape [18944, 3584]
    /// INVARIANT: down_proj has shape [3584, 18944]
    /// INVARIANT (Q projection shape): n_h * d_k = 3584 for Qwen2.5-7B
    /// INVARIANT (KV projection shape): n_kv * d_k = 512 for Qwen2.5-7B
    /// INVARIANT (GQA divisibility): n_h mod n_kv = 28 mod 4 = 0
    /// INVARIANT (SwiGLU gate/up shape): gate_proj.shape = up_proj.shape = [18944, 3584]
    /// INVARIANT (O projection transpose): shape(o_proj) == reverse(shape(q_proj))
    /// INVARIANT (RoPE frequency vector length): len(freqs) == d_k / 2 = 64
    /// MONOTONICITY (RoPE frequency decreasing): freq_i > freq_{i+1} for all i
    /// INVARIANT (Head dimension consistency): 3584 mod 28 = 0 and 3584 / 28 = 128
    /// EQUIVALENCE (SIMD shape equivalence):
    fn swiglu_ratio(&self, input: &[f32], output: &mut [f32]);
}

// Qwen2.5-7B config constants
const HIDDEN: usize = 3584;
const N_HEADS: usize = 28;
const N_KV_HEADS: usize = 4;
const D_K: usize = 128;
const INTERMEDIATE: usize = 18944;
const ROPE_THETA: f64 = 1_000_000.0;

/// Concrete verifier implementing the Qwen2.5-7B shape contract
pub struct Qwen2ShapesVerifier;

impl KernelContract for Qwen2ShapesVerifier {
    fn head_dim_consistency(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 3);
        assert_eq!(HIDDEN % N_HEADS, 0);
        let d_k = HIDDEN / N_HEADS;
        assert_eq!(d_k, D_K);
        output[0] = HIDDEN as f32;
        output[1] = N_HEADS as f32;
        output[2] = d_k as f32;
    }

    fn kv_projection_shape(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 2);
        let kv_dim = N_KV_HEADS * D_K;
        output[0] = kv_dim as f32;
        output[1] = HIDDEN as f32;
    }

    fn o_projection_transpose(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 2);
        // O projection shape is transpose of Q: [hidden, n_h * d_k]
        output[0] = HIDDEN as f32;
        output[1] = (N_HEADS * D_K) as f32;
    }

    fn q_projection_shape(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 2);
        // Q projection: [n_h * d_k, hidden]
        output[0] = (N_HEADS * D_K) as f32;
        output[1] = HIDDEN as f32;
    }

    fn rope_frequency(&self, _input: &[f32], output: &mut [f32]) {
        let n_freqs = D_K / 2;
        assert!(output.len() >= n_freqs);
        for i in 0..n_freqs {
            output[i] = ROPE_THETA.powf(-2.0 * i as f64 / D_K as f64) as f32;
        }
    }

    fn swiglu_ratio(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 3);
        // gate/up shape: [intermediate, hidden]
        output[0] = INTERMEDIATE as f32;
        output[1] = HIDDEN as f32;
        // expansion ratio
        output[2] = (INTERMEDIATE as f64 / HIDDEN as f64) as f32;
    }
}
