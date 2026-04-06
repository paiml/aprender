/// Contract: Qwen3-235B-A22B (MoE) concrete shape instantiation, MoE routing, and RoPE frequency scaling v1.0.0
/// Paper: Qwen3 Technical Report — MoE architecture with top-8 routing
/// Paper: Su et al. (2021) RoFormer — Rotary Position Embedding
/// Paper: Fedus et al. (2022) Switch Transformers — MoE scaling
pub trait KernelContract {
    /// [n_kv * d_k, hidden] = [4*128, 4096] = [512, 4096]
    fn kv_projection_shape(&self, input: &[f32], output: &mut [f32]);
    /// expert_i: gate[moe_inter, hidden] * up[moe_inter, hidden] -> down[hidden, moe_inter]
    fn moe_expert_shape(&self, input: &[f32], output: &mut [f32]);
    /// router: [num_experts, hidden] = [128, 4096]
    fn moe_router_shape(&self, input: &[f32], output: &mut [f32]);
    /// shape(o_proj) = [hidden, n_h * d_k] = [4096, 8192]
    fn o_projection_transpose(&self, input: &[f32], output: &mut [f32]);
    /// [n_h * d_k, hidden] = [64*128, 4096] = [8192, 4096]
    fn q_projection_shape(&self, input: &[f32], output: &mut [f32]);
    /// freq_i = base^(-2i/d_k) for i in [0, d_k/2)
    fn rope_frequency(&self, input: &[f32], output: &mut [f32]);
    /// moe_intermediate / hidden = 1536 / 4096 = 0.375
    fn swiglu_ratio(&self, input: &[f32], output: &mut [f32]);
}

// Qwen3-235B-A22B config constants
const HIDDEN: usize = 4096;
const N_HEADS: usize = 64;
const N_KV_HEADS: usize = 4;
const D_K: usize = 128;
const MOE_INTERMEDIATE: usize = 1536;
const N_EXPERTS: usize = 128;
const N_EXPERTS_PER_TOK: usize = 8;
const ROPE_THETA: f64 = 1_000_000.0;

/// Concrete verifier implementing the Qwen3-235B-A22B MoE shapes contract
pub struct Qwen3MoeShapesVerifier;

impl KernelContract for Qwen3MoeShapesVerifier {
    fn q_projection_shape(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 2);
        output[0] = (N_HEADS * D_K) as f32; // 8192
        output[1] = HIDDEN as f32; // 4096
    }

    fn kv_projection_shape(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 2);
        output[0] = (N_KV_HEADS * D_K) as f32; // 512
        output[1] = HIDDEN as f32; // 4096
    }

    fn o_projection_transpose(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 2);
        output[0] = HIDDEN as f32; // 4096
        output[1] = (N_HEADS * D_K) as f32; // 8192
    }

    fn moe_expert_shape(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 3);
        let per_expert = 3 * HIDDEN * MOE_INTERMEDIATE;
        output[0] = per_expert as f32; // per-expert params
        output[1] = (N_EXPERTS * per_expert) as f32; // total expert params per layer
        output[2] = (N_EXPERTS_PER_TOK * per_expert) as f32; // active expert params
    }

    fn moe_router_shape(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 2);
        output[0] = N_EXPERTS as f32; // 128
        output[1] = HIDDEN as f32; // 4096
    }

    fn rope_frequency(&self, _input: &[f32], output: &mut [f32]) {
        let freq_len = D_K / 2;
        assert!(output.len() >= freq_len);
        for i in 0..freq_len {
            output[i] = ROPE_THETA.powf(-2.0 * i as f64 / D_K as f64) as f32;
        }
    }

    fn swiglu_ratio(&self, _input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= 2);
        output[0] = MOE_INTERMEDIATE as f32 / HIDDEN as f32; // 0.375 per-expert
        output[1] = (N_EXPERTS_PER_TOK * MOE_INTERMEDIATE) as f32 / HIDDEN as f32; // 3.0 effective
    }
}
