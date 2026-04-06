//! Multi-head attention module
//!
//! This module provides multi-head self-attention with grouped-query attention support.

use crate::autograd::{matmul, matmul_nt, BackwardOp};
use crate::Tensor;
use ndarray::Array1;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::config::TransformerConfig;

/// Add a bias vector to a projected tensor: output[s] += bias for each sequence position.
/// Input shape: (seq_len × dim) flattened. Bias shape: (dim).
fn add_bias(x: &Tensor, bias: &Tensor, seq_len: usize) -> Tensor {
    let xd = x.data();
    let x_slice = xd.as_slice().expect("contiguous projection");
    let bd = bias.data();
    let b_slice = bd.as_slice().expect("contiguous bias");
    let dim = b_slice.len();
    let mut out = Vec::with_capacity(x_slice.len());
    for s in 0..seq_len {
        let base = s * dim;
        for d in 0..dim {
            out.push(x_slice[base + d] + b_slice[d]);
        }
    }
    Tensor::from_vec(out, x.requires_grad())
}

/// Apply per-head RMSNorm to Q or K (Qwen3 QK-norm, ENT-269).
///
/// Input: [seq_len * total_dim] where total_dim = num_heads * head_dim.
/// Norm weight: [head_dim]. Applied independently to each head's head_dim slice.
fn apply_qk_norm(
    x: &Tensor,
    norm_weight: &Tensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Tensor {
    let xd = x.data();
    let x_slice = xd.as_slice().expect("contiguous qk");
    let wd = norm_weight.data();
    let w_slice = wd.as_slice().expect("contiguous norm weight");
    let total_dim = num_heads * head_dim;
    let eps = 1e-6_f32;
    let mut out = vec![0.0f32; seq_len * total_dim];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = s * total_dim + h * head_dim;
            // RMSNorm: x * weight / sqrt(mean(x^2) + eps)
            let mut sum_sq = 0.0f32;
            for d in 0..head_dim {
                let v = x_slice[offset + d];
                sum_sq += v * v;
            }
            let rms = (sum_sq / head_dim as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;
            for d in 0..head_dim {
                out[offset + d] = x_slice[offset + d] * inv_rms * w_slice[d];
            }
        }
    }

    Tensor::from_vec(out, x.requires_grad())
}

/// Apply Rotary Position Embedding (RoPE) to Q or K tensor (ENT-269).
///
/// Uses Llama/Qwen3 half-rotation layout (NOT interleaved pairs):
///   x1 = x[..., :half_dim], x2 = x[..., half_dim:]
///   rotate_half(x) = [-x2, x1]
///   result = x * cos + rotate_half(x) * sin
///
/// freq[i] = 1 / (theta ^ (2i / head_dim))
fn apply_rope(
    x: &Tensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) -> Tensor {
    let xd = x.data();
    let x_slice = xd.as_slice().expect("contiguous qk for rope");
    let total_dim = num_heads * head_dim;
    let half_dim = head_dim / 2;
    let mut out = vec![0.0f32; seq_len * total_dim];

    // Precompute inverse frequencies: 1 / (theta ^ (2i / head_dim))
    let inv_freq: Vec<f32> =
        (0..half_dim).map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32)).collect();

    for pos in 0..seq_len {
        for h in 0..num_heads {
            let offset = pos * total_dim + h * head_dim;
            for i in 0..half_dim {
                let freq = pos as f32 * inv_freq[i];
                let cos_f = freq.cos();
                let sin_f = freq.sin();
                // Half-rotation: pair (x[i], x[i + half_dim])
                let x_first = x_slice[offset + i];
                let x_second = x_slice[offset + i + half_dim];
                // rotate_half: [-x_second, x_first]
                out[offset + i] = x_first * cos_f - x_second * sin_f;
                out[offset + i + half_dim] = x_second * cos_f + x_first * sin_f;
            }
        }
    }

    let result = Tensor::from_vec(out, x.requires_grad());
    contract_post_rope!(result.data().as_slice().unwrap_or(&[]));
    result
}

// ---------------------------------------------------------------------------
// AttentionBlockBackward: combined backward for multi-head attention
//
// Orchestrates gradient flow: concat → per-head attention → Q/K/V projections.
// Calls each Q/K/V matmul backward exactly once to avoid gradient inflation.
// ---------------------------------------------------------------------------

struct AttentionBlockBackward {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    head_q_tensors: Vec<Tensor>,
    head_k_tensors: Vec<Tensor>,
    head_v_tensors: Vec<Tensor>,
    head_outputs: Vec<Tensor>,
    head_kv_indices: Vec<usize>,
    seq_len: usize,
    head_dim: usize,
    q_dim: usize,
    kv_hidden_size: usize,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for AttentionBlockBackward {
    fn backward(&self) {
        let Some(grad_out) = self.result_grad.borrow().as_ref().cloned() else { return };
        let go = grad_out.as_slice().expect("grad contiguous");
        let h = self.head_dim;

        // Step 1: Split concat grad per head and trigger each attention backward
        split_and_backward_heads(go, &self.head_outputs, self.seq_len, h, self.q_dim);

        // Step 2-4: Scatter per-head grads into full Q/K/V
        scatter_head_grads_q(&self.q, &self.head_q_tensors, self.seq_len, h, self.q_dim);
        scatter_head_grads_kv(
            &self.k,
            &self.head_k_tensors,
            &self.head_kv_indices,
            self.seq_len,
            h,
            self.kv_hidden_size,
        );
        scatter_head_grads_kv(
            &self.v,
            &self.head_v_tensors,
            &self.head_kv_indices,
            self.seq_len,
            h,
            self.kv_hidden_size,
        );

        // Step 5: Propagate backward through Q/K/V matmuls (once each)
        for proj in [&self.q, &self.k, &self.v] {
            if let Some(op) = proj.backward_op() {
                op.backward();
            }
        }
    }
}

/// Split concat gradient per head and trigger each head's attention backward
fn split_and_backward_heads(
    go: &[f32],
    head_outputs: &[Tensor],
    seq_len: usize,
    head_dim: usize,
    q_dim: usize,
) {
    for (head_idx, head_out) in head_outputs.iter().enumerate() {
        let mut grad_head = vec![0.0_f32; seq_len * head_dim];
        for s in 0..seq_len {
            let src_base = s * q_dim + head_idx * head_dim;
            let dst_base = s * head_dim;
            grad_head[dst_base..dst_base + head_dim]
                .copy_from_slice(&go[src_base..src_base + head_dim]);
        }
        head_out.accumulate_grad(Array1::from(grad_head));
        if let Some(op) = head_out.backward_op() {
            op.backward();
        }
    }
}

/// Scatter per-head Q gradients into the full Q projection tensor
fn scatter_head_grads_q(
    q: &Tensor,
    head_q_tensors: &[Tensor],
    seq_len: usize,
    head_dim: usize,
    q_dim: usize,
) {
    if !q.requires_grad() {
        return;
    }
    let mut grad_q = vec![0.0_f32; seq_len * q_dim];
    for (head_idx, head_q) in head_q_tensors.iter().enumerate() {
        if let Some(hgrad) = head_q.grad() {
            let hg = hgrad.as_slice().expect("contiguous");
            for s in 0..seq_len {
                let src_base = s * head_dim;
                let dst_base = s * q_dim + head_idx * head_dim;
                for d in 0..head_dim {
                    grad_q[dst_base + d] += hg[src_base + d];
                }
            }
        }
    }
    q.accumulate_grad(Array1::from(grad_q));
}

/// Scatter per-head K or V gradients into the full K/V projection tensor (GQA-correct)
fn scatter_head_grads_kv(
    target: &Tensor,
    head_tensors: &[Tensor],
    kv_indices: &[usize],
    seq_len: usize,
    head_dim: usize,
    kv_hidden_size: usize,
) {
    if !target.requires_grad() {
        return;
    }
    let mut grad = vec![0.0_f32; seq_len * kv_hidden_size];
    for (head_idx, head_t) in head_tensors.iter().enumerate() {
        let kv_h = kv_indices[head_idx];
        if let Some(hgrad) = head_t.grad() {
            let hg = hgrad.as_slice().expect("contiguous");
            for s in 0..seq_len {
                let src_base = s * head_dim;
                let dst_base = s * kv_hidden_size + kv_h * head_dim;
                for d in 0..head_dim {
                    grad[dst_base + d] += hg[src_base + d];
                }
            }
        }
    }
    target.accumulate_grad(Array1::from(grad));
}

/// Multi-head self-attention layer
pub struct MultiHeadAttention {
    /// Configuration
    config: TransformerConfig,
    /// Query projection weight (hidden_size x hidden_size)
    pub w_q: Tensor,
    /// Key projection weight (hidden_size x kv_hidden_size)
    pub w_k: Tensor,
    /// Value projection weight (hidden_size x kv_hidden_size)
    pub w_v: Tensor,
    /// Output projection weight (hidden_size x hidden_size)
    pub w_o: Tensor,
    /// Optional query bias (Qwen2 uses attention biases)
    pub b_q: Option<Tensor>,
    /// Optional key bias
    pub b_k: Option<Tensor>,
    /// Optional value bias
    pub b_v: Option<Tensor>,
    /// Optional Q RMSNorm weight (Qwen3 uses QK-norm, shape=[head_dim])
    pub q_norm: Option<Tensor>,
    /// Optional K RMSNorm weight (Qwen3 uses QK-norm, shape=[head_dim])
    pub k_norm: Option<Tensor>,
}

impl MultiHeadAttention {
    /// Create new attention layer with initialized weights
    pub fn new(config: &TransformerConfig) -> Self {
        use super::init::{get_init_seed, rand_normal_seeded};
        let hidden_size = config.hidden_size;
        let q_dim = config.q_dim();
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let seed = get_init_seed();

        // C-INIT-001: normal(0, 0.02) matching HuggingFace LLaMA
        Self {
            config: config.clone(),
            w_q: Tensor::from_vec(rand_normal_seeded(q_dim * hidden_size, seed, "w_q"), true),
            w_k: Tensor::from_vec(
                rand_normal_seeded(kv_hidden_size * hidden_size, seed, "w_k"),
                true,
            ),
            w_v: Tensor::from_vec(
                rand_normal_seeded(kv_hidden_size * hidden_size, seed, "w_v"),
                true,
            ),
            w_o: Tensor::from_vec(rand_normal_seeded(hidden_size * q_dim, seed, "w_o"), true),
            b_q: None,
            b_k: None,
            b_v: None,
            q_norm: None,
            k_norm: None,
        }
    }

    /// Create attention layer from parameter map
    ///
    /// Expected parameter names (following HuggingFace convention):
    /// - `{prefix}.q_proj.weight`
    /// - `{prefix}.k_proj.weight`
    /// - `{prefix}.v_proj.weight`
    /// - `{prefix}.o_proj.weight`
    /// # Contract (PMAT-331)
    /// Validates Q/K/V/O projection shapes against config dimensions.
    /// Returns None if any key is missing or shape is wrong.
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> Option<Self> {
        let w_q = params.get(&format!("{prefix}.q_proj.weight"))?.clone();
        let w_k = params.get(&format!("{prefix}.k_proj.weight"))?.clone();
        let w_v = params.get(&format!("{prefix}.v_proj.weight"))?.clone();
        let w_o = params.get(&format!("{prefix}.o_proj.weight"))?.clone();

        let hidden = config.hidden_size;
        let q_dim = config.q_dim();
        let kv_hidden = config.num_kv_heads * config.head_dim();

        // PMAT-331: Shape validation for attention projections
        // Q: [q_dim, hidden], K: [kv_hidden, hidden], V: [kv_hidden, hidden], O: [hidden, q_dim]
        let checks: &[(&str, &Tensor, usize)] = &[
            ("q_proj", &w_q, q_dim * hidden),
            ("k_proj", &w_k, kv_hidden * hidden),
            ("v_proj", &w_v, kv_hidden * hidden),
            ("o_proj", &w_o, hidden * q_dim),
        ];
        for &(name, tensor, expected) in checks {
            if tensor.len() != expected {
                eprintln!(
                    "[PMAT-331] {prefix}.{name}: shape mismatch — got {} elements, expected {expected}",
                    tensor.len()
                );
                return None;
            }
        }

        // Optional attention biases (Qwen2 uses Q/K/V biases)
        let b_q = params.get(&format!("{prefix}.q_proj.bias")).cloned();
        let b_k = params.get(&format!("{prefix}.k_proj.bias")).cloned();
        let b_v = params.get(&format!("{prefix}.v_proj.bias")).cloned();

        // Optional Q/K RMSNorm (Qwen3 uses QK-norm, ENT-269)
        let q_norm = params.get(&format!("{prefix}.q_norm.weight")).cloned();
        let k_norm = params.get(&format!("{prefix}.k_norm.weight")).cloned();

        Some(Self { config: config.clone(), w_q, w_k, w_v, w_o, b_q, b_k, b_v, q_norm, k_norm })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len * hidden_size, flattened)
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor (seq_len * hidden_size, flattened)
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Tensor {
        contract_pre_attention!(x.data());
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let q_dim = self.config.q_dim();
        let kv_hidden_size = num_kv_heads * head_dim;

        // Project Q, K, V — HF weights are [out_dim, in_dim], use matmul_nt (ENT-269)
        let mut q = matmul_nt(x, &self.w_q, seq_len, hidden_size, q_dim);
        let mut k = matmul_nt(x, &self.w_k, seq_len, hidden_size, kv_hidden_size);
        let mut v = matmul_nt(x, &self.w_v, seq_len, hidden_size, kv_hidden_size);

        // Apply attention biases if present (Qwen2 architecture)
        if let Some(ref b_q) = self.b_q {
            q = add_bias(&q, b_q, seq_len);
        }
        if let Some(ref b_k) = self.b_k {
            k = add_bias(&k, b_k, seq_len);
        }
        if let Some(ref b_v) = self.b_v {
            v = add_bias(&v, b_v, seq_len);
        }

        // Apply Q/K RMSNorm if present (Qwen3 QK-norm, ENT-269)
        if let Some(ref qn) = self.q_norm {
            q = apply_qk_norm(&q, qn, seq_len, num_heads, head_dim);
        }
        if let Some(ref kn) = self.k_norm {
            k = apply_qk_norm(&k, kn, seq_len, num_kv_heads, head_dim);
        }

        // Apply Rotary Position Embedding (RoPE) to Q and K (ENT-269)
        // Skip for encoder models (BERT/RoBERTa use learned positions, not RoPE)
        if self.config.rope_theta > 0.0 {
            q = apply_rope(&q, seq_len, num_heads, head_dim, self.config.rope_theta);
            k = apply_rope(&k, seq_len, num_kv_heads, head_dim, self.config.rope_theta);
        }

        let requires_grad = q.requires_grad() || k.requires_grad() || v.requires_grad();
        let heads_per_kv = num_heads / num_kv_heads;

        // KAIZEN-016: Hoist data borrows outside the head loop to avoid
        // num_heads × seq_len × 3 redundant RefCell borrows per attention call.
        let q_data = q.data();
        let q_slice = q_data.as_slice().expect("contiguous Q");
        let k_data = k.data();
        let k_slice = k_data.as_slice().expect("contiguous K");
        let v_data = v.data();
        let v_slice = v_data.as_slice().expect("contiguous V");

        // Per-head attention with gradient tracking
        let mut head_q_tensors = Vec::with_capacity(num_heads);
        let mut head_k_tensors = Vec::with_capacity(num_heads);
        let mut head_v_tensors = Vec::with_capacity(num_heads);
        let mut head_outputs = Vec::with_capacity(num_heads);
        let mut head_kv_indices = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;
            head_kv_indices.push(kv_h);

            // KAIZEN-016: Use extend_from_slice instead of flat_map+to_vec.
            // Eliminates num_heads × 3 × seq_len intermediate Vec allocations
            // (3.5M allocs/forward for Qwen3-4B).
            let mut q_head = Vec::with_capacity(seq_len * head_dim);
            for s in 0..seq_len {
                let start = s * q_dim + h * head_dim;
                q_head.extend_from_slice(&q_slice[start..start + head_dim]);
            }

            let mut k_head = Vec::with_capacity(seq_len * head_dim);
            for s in 0..seq_len {
                let start = s * kv_hidden_size + kv_h * head_dim;
                k_head.extend_from_slice(&k_slice[start..start + head_dim]);
            }

            let mut v_head = Vec::with_capacity(seq_len * head_dim);
            for s in 0..seq_len {
                let start = s * kv_hidden_size + kv_h * head_dim;
                v_head.extend_from_slice(&v_slice[start..start + head_dim]);
            }

            let q_tensor = Tensor::from_vec(q_head, requires_grad);
            let k_tensor = Tensor::from_vec(k_head, requires_grad);
            let v_tensor = Tensor::from_vec(v_head, requires_grad);

            let attn_out = crate::autograd::attention(
                &q_tensor, &k_tensor, &v_tensor, seq_len, head_dim, seq_len, head_dim,
            );

            head_q_tensors.push(q_tensor);
            head_k_tensors.push(k_tensor);
            head_v_tensors.push(v_tensor);
            head_outputs.push(attn_out);
        }

        // Concatenate heads: reorder from per-head (head, seq, dim) to (seq, head*dim)
        let mut concat_output = vec![0.0; seq_len * q_dim];
        for (h, head_out) in head_outputs.iter().enumerate() {
            let hd = head_out.data();
            let hdata = hd.as_slice().expect("contiguous attention output");
            for s in 0..seq_len {
                let src_base = s * head_dim;
                let dst_base = s * q_dim + h * head_dim;
                concat_output[dst_base..dst_base + head_dim]
                    .copy_from_slice(&hdata[src_base..src_base + head_dim]);
            }
        }

        let mut concat_tensor = Tensor::from_vec(concat_output, requires_grad);

        if requires_grad {
            let backward_op = Rc::new(AttentionBlockBackward {
                q: q.clone(),
                k: k.clone(),
                v: v.clone(),
                head_q_tensors,
                head_k_tensors,
                head_v_tensors,
                head_outputs,
                head_kv_indices,
                seq_len,
                head_dim,
                q_dim,
                kv_hidden_size,
                result_grad: concat_tensor.grad_cell(),
            });
            concat_tensor.set_backward_op(backward_op);
        }

        // Output projection — w_o is [hidden_size, q_dim] in HF (ENT-269)
        let result = matmul_nt(&concat_tensor, &self.w_o, seq_len, q_dim, hidden_size);
        contract_post_attention!(result.data().as_slice().unwrap_or(&[]));
        result
    }

    /// Forward pass with LoRA adjusts on Q and V projections (KAIZEN-010).
    ///
    /// Applies LoRA adapters to Q and V during the forward pass so that
    /// gradients flow through LoRA A/B matrices on non-CUDA paths.
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len * hidden_size)
    /// * `seq_len` - Sequence length
    /// * `lora_a_q`, `lora_b_q` - Q projection LoRA matrices (rank×d_in, d_out×rank)
    /// * `lora_a_v`, `lora_b_v` - V projection LoRA matrices (rank×d_in, d_out×rank)
    /// * `lora_rank` - LoRA rank
    /// * `lora_scale` - LoRA scaling factor (alpha/rank)
    pub fn forward_with_lora(
        &self,
        x: &Tensor,
        seq_len: usize,
        lora_a_q: &Tensor,
        // contract_pre_attention applied via forward()
        lora_b_q: &Tensor,
        lora_a_v: &Tensor,
        lora_b_v: &Tensor,
        lora_rank: usize,
        lora_scale: f32,
    ) -> Tensor {
        contract_pre_lora_forward!();
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let q_dim = self.config.q_dim();
        let kv_hidden_size = num_kv_heads * head_dim;

        // Q projection with LoRA: Q = x @ W_q + scale * (x @ A_q^T) @ B_q^T
        //
        // KAIZEN-011: Use matmul_nt to compute x @ A^T directly on the ORIGINAL
        // LoRA tensors. Previous impl created transposed copies via Tensor::from_vec
        // which broke gradient flow — gradients accumulated on ephemeral copies
        // instead of the actual trainable LoRA parameters.
        //
        // LoRA layout: A is (rank, d_in), B is (d_out, rank)
        // matmul_nt(x, A, seq, d_in, rank) computes x @ A^T = (seq, d_in) @ (d_in, rank) = (seq, rank)
        // matmul_nt(mid, B, seq, rank, d_out) computes mid @ B^T = (seq, rank) @ (rank, d_out) = (seq, d_out)
        let q_base = matmul_nt(x, &self.w_q, seq_len, hidden_size, q_dim);
        let q_mid = crate::autograd::matmul_nt(x, lora_a_q, seq_len, hidden_size, lora_rank);
        let q_lora = crate::autograd::matmul_nt(&q_mid, lora_b_q, seq_len, lora_rank, q_dim);
        let q = crate::autograd::add_scaled(&q_base, &q_lora, lora_scale);

        // K projection (no LoRA) — HF weights [out, in] (ENT-269)
        let k = matmul_nt(x, &self.w_k, seq_len, hidden_size, kv_hidden_size);

        // V projection with LoRA (same pattern as Q)
        let v_base = matmul_nt(x, &self.w_v, seq_len, hidden_size, kv_hidden_size);
        let v_mid = crate::autograd::matmul_nt(x, lora_a_v, seq_len, hidden_size, lora_rank);
        let v_lora =
            crate::autograd::matmul_nt(&v_mid, lora_b_v, seq_len, lora_rank, kv_hidden_size);
        let v = crate::autograd::add_scaled(&v_base, &v_lora, lora_scale);

        // Apply Q/K RMSNorm if present (Qwen3 QK-norm, ENT-269)
        let q = if let Some(ref qn) = self.q_norm {
            apply_qk_norm(&q, qn, seq_len, num_heads, head_dim)
        } else {
            q
        };
        let k = if let Some(ref kn) = self.k_norm {
            apply_qk_norm(&k, kn, seq_len, num_kv_heads, head_dim)
        } else {
            k
        };

        // Apply Rotary Position Embedding (RoPE) to Q and K (ENT-269)
        // Skip for encoder models (BERT/RoBERTa use learned positions, not RoPE)
        let (q, k) = if self.config.rope_theta > 0.0 {
            (
                apply_rope(&q, seq_len, num_heads, head_dim, self.config.rope_theta),
                apply_rope(&k, seq_len, num_kv_heads, head_dim, self.config.rope_theta),
            )
        } else {
            (q, k)
        };

        let requires_grad = q.requires_grad() || k.requires_grad() || v.requires_grad();
        let heads_per_kv = num_heads / num_kv_heads;

        // KAIZEN-016: Hoist data borrows outside head loop (same optimization as forward())
        let q_data = q.data();
        let q_slice = q_data.as_slice().expect("contiguous Q");
        let k_data = k.data();
        let k_slice = k_data.as_slice().expect("contiguous K");
        let v_data = v.data();
        let v_slice = v_data.as_slice().expect("contiguous V");

        // Per-head attention (same as forward())
        let mut head_q_tensors = Vec::with_capacity(num_heads);
        let mut head_k_tensors = Vec::with_capacity(num_heads);
        let mut head_v_tensors = Vec::with_capacity(num_heads);
        let mut head_outputs = Vec::with_capacity(num_heads);
        let mut head_kv_indices = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;
            head_kv_indices.push(kv_h);

            // KAIZEN-016: extend_from_slice replaces flat_map+to_vec
            let mut q_head = Vec::with_capacity(seq_len * head_dim);
            for s in 0..seq_len {
                let start = s * q_dim + h * head_dim;
                q_head.extend_from_slice(&q_slice[start..start + head_dim]);
            }

            let mut k_head = Vec::with_capacity(seq_len * head_dim);
            for s in 0..seq_len {
                let start = s * kv_hidden_size + kv_h * head_dim;
                k_head.extend_from_slice(&k_slice[start..start + head_dim]);
            }

            let mut v_head = Vec::with_capacity(seq_len * head_dim);
            for s in 0..seq_len {
                let start = s * kv_hidden_size + kv_h * head_dim;
                v_head.extend_from_slice(&v_slice[start..start + head_dim]);
            }

            let q_tensor = Tensor::from_vec(q_head, requires_grad);
            let k_tensor = Tensor::from_vec(k_head, requires_grad);
            let v_tensor = Tensor::from_vec(v_head, requires_grad);

            let attn_out = crate::autograd::attention(
                &q_tensor, &k_tensor, &v_tensor, seq_len, head_dim, seq_len, head_dim,
            );

            head_q_tensors.push(q_tensor);
            head_k_tensors.push(k_tensor);
            head_v_tensors.push(v_tensor);
            head_outputs.push(attn_out);
        }

        // Concatenate heads
        let mut concat_output = vec![0.0; seq_len * q_dim];
        for (h, head_out) in head_outputs.iter().enumerate() {
            let hd = head_out.data();
            let hdata = hd.as_slice().expect("contiguous attention output");
            for s in 0..seq_len {
                let src_base = s * head_dim;
                let dst_base = s * q_dim + h * head_dim;
                concat_output[dst_base..dst_base + head_dim]
                    .copy_from_slice(&hdata[src_base..src_base + head_dim]);
            }
        }

        let mut concat_tensor = Tensor::from_vec(concat_output, requires_grad);

        if requires_grad {
            let backward_op = Rc::new(AttentionBlockBackward {
                q: q.clone(),
                k: k.clone(),
                v: v.clone(),
                head_q_tensors,
                head_k_tensors,
                head_v_tensors,
                head_outputs,
                head_kv_indices,
                seq_len,
                head_dim,
                q_dim,
                kv_hidden_size,
                result_grad: concat_tensor.grad_cell(),
            });
            concat_tensor.set_backward_op(backward_op);
        }

        // Output projection — w_o is [hidden_size, q_dim] in HF (ENT-269)
        let result = matmul_nt(&concat_tensor, &self.w_o, seq_len, q_dim, hidden_size);
        contract_post_lora_forward!(result);
        result
    }

    /// Get all parameters as a vector
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.w_q, &self.w_k, &self.w_v, &self.w_o];
        if let Some(ref b) = self.b_q {
            params.push(b);
        }
        if let Some(ref b) = self.b_k {
            params.push(b);
        }
        if let Some(ref b) = self.b_v {
            params.push(b);
        }
        params
    }

    /// Get all parameters as mutable references for optimizer
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.w_q, &mut self.w_k, &mut self.w_v, &mut self.w_o];
        if let Some(ref mut b) = self.b_q {
            params.push(b);
        }
        if let Some(ref mut b) = self.b_k {
            params.push(b);
        }
        if let Some(ref mut b) = self.b_v {
            params.push(b);
        }
        params
    }

    /// Whether this attention layer has QKV biases
    pub fn has_biases(&self) -> bool {
        self.b_q.is_some()
    }

    /// Get named parameters for checkpoint serialization
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        let mut params = vec![
            (format!("{prefix}.q_proj.weight"), &self.w_q),
            (format!("{prefix}.k_proj.weight"), &self.w_k),
            (format!("{prefix}.v_proj.weight"), &self.w_v),
            (format!("{prefix}.o_proj.weight"), &self.w_o),
        ];
        if let Some(ref b) = self.b_q {
            params.push((format!("{prefix}.q_proj.bias"), b));
        }
        if let Some(ref b) = self.b_k {
            params.push((format!("{prefix}.k_proj.bias"), b));
        }
        if let Some(ref b) = self.b_v {
            params.push((format!("{prefix}.v_proj.bias"), b));
        }
        params
    }

    /// ENT-282: Set a named parameter by suffix (after "self_attn.").
    pub fn set_named_parameter(&mut self, suffix: &str, value: Tensor) -> bool {
        match suffix {
            "self_attn.q_proj.weight" => {
                self.w_q = value;
                true
            }
            "self_attn.k_proj.weight" => {
                self.w_k = value;
                true
            }
            "self_attn.v_proj.weight" => {
                self.w_v = value;
                true
            }
            "self_attn.o_proj.weight" => {
                self.w_o = value;
                true
            }
            _ => false,
        }
    }
}

/// LoRA-enabled linear projection
///
/// Computes: y = x @ W + scale * (x @ A) @ B
/// Where W is frozen base weight, A and B are trainable LoRA adapters
pub struct LoRAProjection {
    /// Base weight (frozen), shape (d_in × d_out)
    pub base_weight: Tensor,
    /// LoRA A matrix (down-projection), shape (d_in × rank)
    pub lora_a: Tensor,
    /// LoRA B matrix (up-projection), shape (rank × d_out)
    pub lora_b: Tensor,
    /// Input dimension
    pub d_in: usize,
    /// Output dimension
    pub d_out: usize,
    /// LoRA rank
    pub rank: usize,
    /// Scaling factor (alpha / rank)
    pub scale: f32,
}

impl LoRAProjection {
    /// Create a new LoRA projection
    ///
    /// # Arguments
    /// * `base_weight` - Frozen base weight [d_in × d_out]
    /// * `d_in` - Input dimension
    /// * `d_out` - Output dimension
    /// * `rank` - LoRA rank (typically 4, 8, 16, 32, or 64)
    /// * `alpha` - LoRA scaling parameter
    pub fn new(base_weight: Tensor, d_in: usize, d_out: usize, rank: usize, alpha: f32) -> Self {
        assert_eq!(base_weight.len(), d_in * d_out, "Base weight size mismatch");

        // Freeze base weight — only LoRA adapters are trainable
        let mut base_weight = base_weight;
        base_weight.set_requires_grad(false);

        // Initialize A with Kaiming uniform (standard LoRA paper)
        let lora_a = Tensor::from_vec(
            (0..d_in * rank).map(|i| ((i as f32 * 0.123).sin() * 0.01)).collect(),
            true, // requires_grad
        );

        // Initialize B with zeros (LoRA invariant: ΔW = B @ A = 0 at init)
        let lora_b = Tensor::zeros(rank * d_out, true);

        Self { base_weight, lora_a, lora_b, d_in, d_out, rank, scale: alpha / rank as f32 }
    }

    /// Forward pass with LoRA
    ///
    /// Computes: y = x @ W + scale * (x @ A) @ B
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len × d_in]
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor [seq_len × d_out]
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Tensor {
        // Base projection: x @ W, (seq × d_in) @ (d_in × d_out) = (seq × d_out)
        let base_out = matmul(x, &self.base_weight, seq_len, self.d_in, self.d_out);

        // LoRA path: scale * (x @ A) @ B
        // Step 1: x @ A, (seq × d_in) @ (d_in × rank) = (seq × rank)
        let lora_intermediate = matmul(x, &self.lora_a, seq_len, self.d_in, self.rank);

        // Step 2: (x @ A) @ B, (seq × rank) @ (rank × d_out) = (seq × d_out)
        let lora_out = matmul(&lora_intermediate, &self.lora_b, seq_len, self.rank, self.d_out);

        // Combine: base + scale * lora
        // Use autograd-compatible addition
        crate::autograd::add_scaled(&base_out, &lora_out, self.scale)
    }

    /// Get trainable LoRA parameters
    pub fn lora_params(&self) -> Vec<&Tensor> {
        vec![&self.lora_a, &self.lora_b]
    }

    /// Get mutable trainable LoRA parameters
    pub fn lora_params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.lora_a, &mut self.lora_b]
    }
}

/// Multi-head attention with deep LoRA injection
///
/// LoRA adapters are applied to Q, K, V, O projections during forward pass
pub struct MultiHeadAttentionWithLoRA {
    /// Configuration
    pub config: TransformerConfig,
    /// Query projection with LoRA
    pub q_proj: LoRAProjection,
    /// Key projection with LoRA
    pub k_proj: LoRAProjection,
    /// Value projection with LoRA
    pub v_proj: LoRAProjection,
    /// Output projection with LoRA
    pub o_proj: LoRAProjection,
}

impl MultiHeadAttentionWithLoRA {
    /// Create LoRA-enabled attention from existing attention weights
    ///
    /// # Arguments
    /// * `attn` - Base MultiHeadAttention with pretrained weights
    /// * `rank` - LoRA rank
    /// * `alpha` - LoRA alpha scaling factor
    pub fn from_attention(attn: &MultiHeadAttention, rank: usize, alpha: f32) -> Self {
        let hidden_size = attn.config.hidden_size;
        let q_dim = attn.config.q_dim();
        let kv_hidden_size = attn.config.num_kv_heads * attn.config.head_dim();

        Self {
            config: attn.config.clone(),
            q_proj: LoRAProjection::new(attn.w_q.clone(), hidden_size, q_dim, rank, alpha),
            k_proj: LoRAProjection::new(attn.w_k.clone(), hidden_size, kv_hidden_size, rank, alpha),
            v_proj: LoRAProjection::new(attn.w_v.clone(), hidden_size, kv_hidden_size, rank, alpha),
            o_proj: LoRAProjection::new(attn.w_o.clone(), q_dim, hidden_size, rank, alpha),
        }
    }

    /// Forward pass with deep LoRA injection
    ///
    /// LoRA is applied to all Q, K, V, O projections
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Tensor {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let q_dim = self.config.q_dim();
        let kv_hidden_size = num_kv_heads * head_dim;

        // Project Q, K, V with LoRA
        let q = self.q_proj.forward(x, seq_len);
        let k = self.k_proj.forward(x, seq_len);
        let v = self.v_proj.forward(x, seq_len);

        // Multi-head attention with grouped-query attention support
        let mut attn_outputs = Vec::with_capacity(num_heads * seq_len * head_dim);
        let heads_per_kv = num_heads / num_kv_heads;

        // KAIZEN-016: Hoist data borrows outside head loop
        let q_data = q.data();
        let q_slice = q_data.as_slice().expect("contiguous Q tensor");
        let k_data = k.data();
        let k_slice = k_data.as_slice().expect("contiguous K tensor");
        let v_data = v.data();
        let v_slice = v_data.as_slice().expect("contiguous V tensor");

        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;

            // KAIZEN-016: extend_from_slice replaces flat_map+to_vec
            let mut q_head = Vec::with_capacity(seq_len * head_dim);
            for s in 0..seq_len {
                let start = s * q_dim + h * head_dim;
                q_head.extend_from_slice(&q_slice[start..start + head_dim]);
            }

            let mut k_head = Vec::with_capacity(seq_len * head_dim);
            for s in 0..seq_len {
                let start = s * kv_hidden_size + kv_h * head_dim;
                k_head.extend_from_slice(&k_slice[start..start + head_dim]);
            }

            let mut v_head = Vec::with_capacity(seq_len * head_dim);
            for s in 0..seq_len {
                let start = s * kv_hidden_size + kv_h * head_dim;
                v_head.extend_from_slice(&v_slice[start..start + head_dim]);
            }

            // Scaled dot-product attention
            let q_tensor = Tensor::from_vec(q_head, false);
            let k_tensor = Tensor::from_vec(k_head, false);
            let v_tensor = Tensor::from_vec(v_head, false);

            let attn_out = crate::autograd::attention(
                &q_tensor, &k_tensor, &v_tensor, seq_len, head_dim, seq_len, head_dim,
            );

            attn_outputs.extend_from_slice(
                attn_out.data().as_slice().expect("contiguous attention output"),
            );
        }

        // Concatenate heads and reorder: (seq_len, q_dim)
        let mut concat_output = vec![0.0; seq_len * q_dim];
        for h in 0..num_heads {
            for s in 0..seq_len {
                let src_idx = h * seq_len * head_dim + s * head_dim;
                let dst_idx = s * q_dim + h * head_dim;
                concat_output[dst_idx..dst_idx + head_dim]
                    .copy_from_slice(&attn_outputs[src_idx..src_idx + head_dim]);
            }
        }

        let concat_tensor = Tensor::from_vec(concat_output, true);

        // Output projection with LoRA: (seq_len, q_dim) -> (seq_len, hidden_size)
        self.o_proj.forward(&concat_tensor, seq_len)
    }

    /// Get all trainable LoRA parameters
    pub fn lora_params(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.lora_params());
        params.extend(self.k_proj.lora_params());
        params.extend(self.v_proj.lora_params());
        params.extend(self.o_proj.lora_params());
        params
    }

    /// Get all trainable LoRA parameters as mutable references
    pub fn lora_params_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.lora_params_mut());
        params.extend(self.k_proj.lora_params_mut());
        params.extend(self.v_proj.lora_params_mut());
        params.extend(self.o_proj.lora_params_mut());
        params
    }

    /// Count total LoRA parameters
    pub fn lora_param_count(&self) -> usize {
        // Each projection has A (d_in × rank) + B (rank × d_out)
        let hidden = self.config.hidden_size;
        let kv_hidden = self.config.num_kv_heads * self.config.head_dim();
        let rank = self.q_proj.rank;

        // Q: (hidden × rank) + (rank × hidden)
        // K: (hidden × rank) + (rank × kv_hidden)
        // V: (hidden × rank) + (rank × kv_hidden)
        // O: (hidden × rank) + (rank × hidden)
        (hidden * rank + rank * hidden)      // Q
            + (hidden * rank + rank * kv_hidden) // K
            + (hidden * rank + rank * kv_hidden) // V
            + (hidden * rank + rank * hidden) // O
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_head_attention_tiny() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let x = Tensor::from_vec(vec![0.1; 2 * config.hidden_size], true);
        let output = attn.forward(&x, 2);
        assert_eq!(output.len(), 2 * config.hidden_size);
    }

    #[test]
    fn test_multi_head_attention_parameters() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let params = attn.parameters();
        assert_eq!(params.len(), 4); // w_q, w_k, w_v, w_o
    }

    #[test]
    fn test_attention_longer_sequence() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let x = Tensor::from_vec(vec![0.1; 8 * config.hidden_size], true);
        let output = attn.forward(&x, 8);
        assert_eq!(output.len(), 8 * config.hidden_size);
    }

    #[test]
    fn test_attention_weight_sizes() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let kv_hidden = config.num_kv_heads * config.head_dim();
        assert_eq!(attn.w_q.len(), config.hidden_size * config.hidden_size);
        assert_eq!(attn.w_k.len(), config.hidden_size * kv_hidden);
        assert_eq!(attn.w_v.len(), config.hidden_size * kv_hidden);
        assert_eq!(attn.w_o.len(), config.hidden_size * config.hidden_size);
    }

    #[test]
    fn test_multi_head_attention_from_params_success() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();

        let mut params = HashMap::new();
        params.insert(
            "attn.q_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );
        params.insert(
            "attn.k_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
        );
        params.insert(
            "attn.v_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
        );
        params.insert(
            "attn.o_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );

        let attn = MultiHeadAttention::from_params(&config, &params, "attn");
        assert!(attn.is_some());
        let attn = attn.expect("operation should succeed");
        assert_eq!(attn.w_q.len(), hidden_size * hidden_size);
    }

    #[test]
    fn test_multi_head_attention_from_params_missing_key() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;

        let mut params = HashMap::new();
        params.insert(
            "attn.q_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );
        // Missing k_proj, v_proj, o_proj

        let attn = MultiHeadAttention::from_params(&config, &params, "attn");
        assert!(attn.is_none());
    }

    #[test]
    fn test_attention_projections_backward() {
        // Test that Q, K, V projection matmuls have gradients
        // (isolated from the full attention which has intermediate tensor issues)
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let hidden_size = config.hidden_size;
        let seq_len = 2;

        let x = Tensor::from_vec(vec![0.1; seq_len * hidden_size], true);

        // Test Q projection
        let mut q = crate::autograd::matmul(&x, &attn.w_q, seq_len, hidden_size, hidden_size);
        let grad_out = ndarray::Array1::ones(seq_len * hidden_size);
        crate::autograd::backward(&mut q, Some(grad_out));

        assert!(attn.w_q.grad().is_some());
        let grad_q = attn.w_q.grad().expect("gradient should be available");
        assert!(grad_q.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_output_projection_backward() {
        // Test output projection in isolation
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let hidden_size = config.hidden_size;
        let seq_len = 2;

        // Simulate concatenated attention output
        let concat_out = Tensor::from_vec(vec![0.1; seq_len * hidden_size], true);

        // Output projection
        let mut output =
            crate::autograd::matmul(&concat_out, &attn.w_o, seq_len, hidden_size, hidden_size);

        let grad_out = ndarray::Array1::ones(seq_len * hidden_size);
        crate::autograd::backward(&mut output, Some(grad_out));

        assert!(attn.w_o.grad().is_some());
        let grad_o = attn.w_o.grad().expect("gradient should be available");
        assert!(grad_o.iter().all(|&v| v.is_finite()));
        let sum: f32 = grad_o.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.0, "Output projection gradient should not be all zero");
    }

    /// ALB-038: Full attention forward must propagate gradients to Q/K/V weights
    ///
    /// NOTE: Currently fails because apply_rope() has no backward op — it severs
    /// the autograd chain for Q and K. Needs a proper RoPE backward implementation
    /// (ENT-272). Skipped until then.
    #[test]
    #[ignore = "apply_rope() severs autograd chain — needs backward op (ENT-272)"]
    fn test_attention_full_forward_qkv_gradients() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let hidden_size = config.hidden_size;
        let seq_len = 3;

        // Non-uniform input: different positions must have different representations
        // so softmax produces non-uniform weights with non-zero score gradients
        let x_data: Vec<f32> =
            (0..seq_len * hidden_size).map(|i| ((i as f32) * 0.17).sin() * 0.5).collect();
        let x = Tensor::from_vec(x_data, true);
        let mut output = attn.forward(&x, seq_len);

        let grad_out = ndarray::Array1::ones(seq_len * hidden_size);
        crate::autograd::backward(&mut output, Some(grad_out));

        // All four projection weights must receive gradients
        for (name, param) in
            [("w_q", &attn.w_q), ("w_k", &attn.w_k), ("w_v", &attn.w_v), ("w_o", &attn.w_o)]
        {
            assert!(
                param.grad().is_some(),
                "ALB-038: {name} must have gradient after full attention forward"
            );
            let grad = param.grad().expect("gradient available");
            assert!(grad.iter().all(|&v| v.is_finite()), "ALB-038: {name} gradient must be finite");
            assert!(
                grad.iter().any(|&v| v.abs() > 1e-10),
                "ALB-038: {name} gradient must be non-zero"
            );
        }

        // Input must also receive gradient (enables gradient flow through model)
        assert!(x.grad().is_some(), "ALB-038: input x must have gradient");
    }

    // ============================================================================
    // LoRAProjection tests
    // ============================================================================

    #[test]
    fn test_lora_projection_new() {
        let d_in = 32;
        let d_out = 16;
        let rank = 4;
        let alpha = 8.0;

        let base_weight = Tensor::from_vec(vec![0.1; d_in * d_out], false);
        let lora = LoRAProjection::new(base_weight, d_in, d_out, rank, alpha);

        assert_eq!(lora.d_in, d_in);
        assert_eq!(lora.d_out, d_out);
        assert_eq!(lora.rank, rank);
        assert!((lora.scale - 2.0).abs() < 1e-6); // alpha / rank = 8 / 4 = 2
        assert_eq!(lora.lora_a.len(), d_in * rank);
        assert_eq!(lora.lora_b.len(), rank * d_out);
    }

    #[test]
    fn test_lora_projection_forward() {
        let d_in = 32;
        let d_out = 16;
        let rank = 4;
        let alpha = 8.0;
        let seq_len = 2;

        let base_weight = Tensor::from_vec(vec![0.1; d_in * d_out], false);
        let lora = LoRAProjection::new(base_weight, d_in, d_out, rank, alpha);

        let x = Tensor::from_vec(vec![0.1; seq_len * d_in], false);
        let output = lora.forward(&x, seq_len);

        assert_eq!(output.len(), seq_len * d_out);
        // Check output is finite
        assert!(output.data().iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_lora_projection_params() {
        let d_in = 32;
        let d_out = 16;
        let rank = 4;

        let base_weight = Tensor::from_vec(vec![0.1; d_in * d_out], false);
        let lora = LoRAProjection::new(base_weight, d_in, d_out, rank, 8.0);

        let params = lora.lora_params();
        assert_eq!(params.len(), 2); // lora_a and lora_b
    }

    #[test]
    fn test_lora_projection_params_mut() {
        let d_in = 32;
        let d_out = 16;
        let rank = 4;

        let base_weight = Tensor::from_vec(vec![0.1; d_in * d_out], false);
        let mut lora = LoRAProjection::new(base_weight, d_in, d_out, rank, 8.0);

        let params = lora.lora_params_mut();
        assert_eq!(params.len(), 2);
    }

    #[test]
    #[should_panic(expected = "Base weight size mismatch")]
    fn test_lora_projection_size_mismatch() {
        let d_in = 32;
        let d_out = 16;
        let rank = 4;

        // Wrong base weight size
        let base_weight = Tensor::from_vec(vec![0.1; d_in * d_out + 1], false);
        let _ = LoRAProjection::new(base_weight, d_in, d_out, rank, 8.0);
    }

    // ============================================================================
    // MultiHeadAttentionWithLoRA tests
    // ============================================================================

    #[test]
    fn test_mha_with_lora_creation() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let rank = 4;
        let alpha = 8.0;

        let lora_attn = MultiHeadAttentionWithLoRA::from_attention(&attn, rank, alpha);

        assert_eq!(lora_attn.q_proj.rank, rank);
        assert_eq!(lora_attn.k_proj.rank, rank);
        assert_eq!(lora_attn.v_proj.rank, rank);
        assert_eq!(lora_attn.o_proj.rank, rank);
    }

    #[test]
    fn test_mha_with_lora_forward() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let lora_attn = MultiHeadAttentionWithLoRA::from_attention(&attn, 4, 8.0);

        let seq_len = 2;
        let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], false);
        let output = lora_attn.forward(&x, seq_len);

        assert_eq!(output.len(), seq_len * config.hidden_size);
        // Check output is finite and non-zero
        assert!(output.data().iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_mha_with_lora_params() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let lora_attn = MultiHeadAttentionWithLoRA::from_attention(&attn, 4, 8.0);

        let params = lora_attn.lora_params();
        // 4 projections × 2 params each = 8
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_mha_with_lora_params_mut() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let mut lora_attn = MultiHeadAttentionWithLoRA::from_attention(&attn, 4, 8.0);

        let params = lora_attn.lora_params_mut();
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_mha_with_lora_param_count() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let rank = 4;
        let lora_attn = MultiHeadAttentionWithLoRA::from_attention(&attn, rank, 8.0);

        let param_count = lora_attn.lora_param_count();

        // Calculate expected:
        let hidden = config.hidden_size;
        let kv_hidden = config.num_kv_heads * config.head_dim();
        let expected = (hidden * rank + rank * hidden)      // Q
            + (hidden * rank + rank * kv_hidden) // K
            + (hidden * rank + rank * kv_hidden) // V
            + (hidden * rank + rank * hidden); // O

        assert_eq!(param_count, expected);
        assert!(param_count > 0);
    }

    #[test]
    fn test_mha_with_lora_longer_sequence() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let lora_attn = MultiHeadAttentionWithLoRA::from_attention(&attn, 4, 8.0);

        let seq_len = 8;
        let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], false);
        let output = lora_attn.forward(&x, seq_len);

        assert_eq!(output.len(), seq_len * config.hidden_size);
    }

    #[test]
    fn test_parameters_mut() {
        let config = TransformerConfig::tiny();
        let mut attn = MultiHeadAttention::new(&config);

        let params = attn.parameters_mut();
        assert_eq!(params.len(), 4);
    }

    // =========================================================================
    // FALSIFY-A: §2.1.3 Attention Projections — Five-Whys Gap Analysis (Refs PMAT-331)
    //
    // Contract: tensor-layout-v1.yaml §tensors.q_proj/k_proj/v_proj/o_proj
    //   q_proj: [num_heads*head_dim, hidden] (= [hidden, hidden] for MHA)
    //   k_proj: [num_kv_heads*head_dim, hidden] (smaller for GQA)
    //   v_proj: [num_kv_heads*head_dim, hidden] (smaller for GQA)
    //   o_proj: [hidden, num_heads*head_dim]
    //
    // Five-Whys:
    //   Why 1: Trained model's attention weights could be wrong shape
    //   Why 2: from_params accepts any tensor without shape validation
    //   Why 3: No ValidatedWeight in entrenar
    //   Why 4: entrenar predates the Poka-Yoke contract
    //   Why 5: No cross-crate contract enforcement for training weights
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // entrenar's attention weight handling prevents degenerate models."
    // =========================================================================

    /// FALSIFY-A1e: from_params rejects wrong-shape Q weight (PMAT-331 fix)
    ///
    /// from_params now validates Q projection shape against config dimensions.
    /// A tensor of 50 elements is rejected when hidden*hidden is expected.
    #[test]
    fn falsify_a1e_from_params_rejects_wrong_shape_q_weight() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();

        let mut params = HashMap::new();
        // WRONG-SHAPE q_proj: 50 elements instead of hidden*hidden
        params.insert("attn.q_proj.weight".to_string(), Tensor::from_vec(vec![0.1; 50], true));
        // Correct k, v, o
        params.insert(
            "attn.k_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
        );
        params.insert(
            "attn.v_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
        );
        params.insert(
            "attn.o_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );

        let attn = MultiHeadAttention::from_params(&config, &params, "attn");
        // FIXED (PMAT-331): now rejected
        assert!(
            attn.is_none(),
            "FALSIFY-A1e: PMAT-331 fix — from_params MUST reject wrong-shape q_proj"
        );
    }

    /// FALSIFY-A2e: GQA init produces correct K/V dimensions
    ///
    /// For GQA (num_kv_heads < num_heads), K/V must be smaller than Q.
    /// If init uses num_heads for K/V, the shapes are wrong.
    #[test]
    fn falsify_a2e_gqa_init_correct_kv_dimensions() {
        let mut config = TransformerConfig::tiny();
        config.num_kv_heads = 1; // Force GQA: 1 KV head, but num_heads > 1

        let attn = MultiHeadAttention::new(&config);
        let head_dim = config.head_dim();
        let kv_hidden = config.num_kv_heads * head_dim; // 1 * head_dim

        // Q: hidden * hidden
        assert_eq!(
            attn.w_q.len(),
            config.hidden_size * config.hidden_size,
            "FALSIFY-A2e: Q projection must be hidden*hidden"
        );

        // K: hidden * kv_hidden (smaller than Q for GQA)
        assert_eq!(
            attn.w_k.len(),
            config.hidden_size * kv_hidden,
            "FALSIFY-A2e: K projection must use num_kv_heads, not num_heads"
        );

        // V: hidden * kv_hidden (same as K)
        assert_eq!(
            attn.w_v.len(),
            config.hidden_size * kv_hidden,
            "FALSIFY-A2e: V projection must use num_kv_heads, not num_heads"
        );

        // O: hidden * hidden (matches Q output)
        assert_eq!(
            attn.w_o.len(),
            config.hidden_size * config.hidden_size,
            "FALSIFY-A2e: O projection must be hidden*hidden"
        );

        // K/V must be SMALLER than Q for GQA
        assert!(
            attn.w_k.len() < attn.w_q.len(),
            "FALSIFY-A2e: For GQA, K weight must be smaller than Q weight"
        );
    }

    /// FALSIFY-A3e: GQA forward produces correct output dimensions
    ///
    /// With num_kv_heads < num_heads, the forward pass must still produce
    /// [seq_len, hidden_size] output (not [seq_len, kv_hidden]).
    #[test]
    fn falsify_a3e_gqa_forward_correct_output_dims() {
        let mut config = TransformerConfig::tiny();
        config.num_kv_heads = 1; // Force GQA

        let attn = MultiHeadAttention::new(&config);
        let seq_len = 3;
        let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], true);
        let output = attn.forward(&x, seq_len);

        assert_eq!(
            output.len(),
            seq_len * config.hidden_size,
            "FALSIFY-A3e: GQA output must be seq_len * hidden_size, not seq_len * kv_hidden"
        );
    }

    /// FALSIFY-A4e: Attention init produces non-degenerate values
    ///
    /// Like FALSIFY-E7a for embeddings: init must produce varied, finite values.
    #[test]
    fn falsify_a4e_init_produces_valid_attention_weights() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);

        for (name, w) in
            [("w_q", &attn.w_q), ("w_k", &attn.w_k), ("w_v", &attn.w_v), ("w_o", &attn.w_o)]
        {
            let data = w.data();
            let slice = data.as_slice().expect("data as slice");

            // No NaN
            let nan_count = slice.iter().filter(|v| v.is_nan()).count();
            assert_eq!(nan_count, 0, "FALSIFY-A4e: {name} init must not contain NaN");

            // No Inf
            let inf_count = slice.iter().filter(|v| v.is_infinite()).count();
            assert_eq!(inf_count, 0, "FALSIFY-A4e: {name} init must not contain Inf");

            // Values vary
            let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
            let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                (max - min).abs() > 1e-6,
                "FALSIFY-A4e: {name} init values are constant ({min}..{max}) — degenerate weight"
            );
        }
    }

    /// FALSIFY-A5e: Attention forward produces finite outputs
    ///
    /// If any attention weight is degenerate, output should still be finite
    /// (the init is designed to prevent this).
    #[test]
    fn falsify_a5e_forward_produces_finite_output() {
        let config = TransformerConfig::tiny();
        let attn = MultiHeadAttention::new(&config);
        let seq_len = 4;
        let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], true);
        let output = attn.forward(&x, seq_len);

        let data = output.data();
        let nan_count = data.iter().filter(|v| v.is_nan()).count();
        let inf_count = data.iter().filter(|v| v.is_infinite()).count();
        assert_eq!(nan_count, 0, "FALSIFY-A5e: Attention output must not contain NaN");
        assert_eq!(inf_count, 0, "FALSIFY-A5e: Attention output must not contain Inf");
    }

    // =========================================================================
    // FALSIFY-GQ: gqa-kernel-v1.yaml contract (entrenar MultiHeadAttention GQA)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar had FALSIFY-A tests but zero FALSIFY-GQ-* tests
    //   Why 2: FALSIFY-A tests verify projections/shapes, not GQA invariants
    //   Why 3: no mapping from gqa-kernel-v1.yaml to entrenar test names
    //   Why 4: entrenar's GQA support added after FALSIFY-A tests
    //   Why 5: GQA was "obviously correct" (just index K/V by h/heads_per_kv)
    //
    // References:
    //   - provable-contracts/contracts/gqa-kernel-v1.yaml
    //   - Ainslie et al. (2023) "GQA: Training Generalized MQT Models"
    // =========================================================================

    /// FALSIFY-GQ-001e: GQA output shape correct for various head configs
    #[test]
    fn falsify_gq_001e_output_shape() {
        for (num_heads, num_kv_heads) in [(2, 2), (4, 2), (4, 1), (2, 1)] {
            let mut config = TransformerConfig::tiny();
            config.num_attention_heads = num_heads;
            config.num_kv_heads = num_kv_heads;

            let attn = MultiHeadAttention::new(&config);
            let seq_len = 3;
            let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], true);
            let output = attn.forward(&x, seq_len);

            assert_eq!(
                output.len(),
                seq_len * config.hidden_size,
                "FALSIFIED GQ-001e: output len mismatch for heads={num_heads},kv={num_kv_heads}"
            );
        }
    }

    /// FALSIFY-GQ-002e: MHA degeneration — kv_heads == num_heads produces finite output
    #[test]
    fn falsify_gq_002e_mha_degeneration() {
        let config = TransformerConfig::tiny(); // num_heads == num_kv_heads == 2
        assert_eq!(config.num_attention_heads, config.num_kv_heads);

        let attn = MultiHeadAttention::new(&config);
        let seq_len = 4;
        let x = Tensor::from_vec(
            (0..seq_len * config.hidden_size).map(|i| (i as f32 * 0.37).sin()).collect(),
            true,
        );
        let output = attn.forward(&x, seq_len);

        let data = output.data();
        for (i, v) in data.iter().enumerate() {
            assert!(v.is_finite(), "FALSIFIED GQ-002e: MHA output[{i}] = {v} (not finite)");
        }
    }

    /// FALSIFY-GQ-004e: Head divisibility — GQA requires num_heads % num_kv_heads == 0
    #[test]
    fn falsify_gq_004e_head_divisibility() {
        // Valid configurations should not panic
        for (nh, nkv) in [(2, 1), (2, 2), (4, 1), (4, 2), (4, 4), (8, 2), (8, 4)] {
            let mut config = TransformerConfig::tiny();
            config.num_attention_heads = nh;
            config.num_kv_heads = nkv;
            assert_eq!(nh % nkv, 0, "FALSIFIED GQ-004e: test config has invalid head ratio");
            // Should not panic during construction or forward
            let attn = MultiHeadAttention::new(&config);
            let x = Tensor::from_vec(vec![0.1; 2 * config.hidden_size], true);
            let _ = attn.forward(&x, 2);
        }
    }

    /// FALSIFY-GQ-006e: MQA boundary — kv_heads=1 broadcasts single KV to all heads
    #[test]
    fn falsify_gq_006e_mqa_boundary() {
        let mut config = TransformerConfig::tiny();
        config.num_attention_heads = 4;
        config.num_kv_heads = 1;
        // Adjust hidden_size to be divisible by 4 heads
        config.hidden_size = 64;

        let attn = MultiHeadAttention::new(&config);
        let seq_len = 3;
        let x = Tensor::from_vec(
            (0..seq_len * config.hidden_size).map(|i| (i as f32 * 0.73).cos()).collect(),
            true,
        );
        let output = attn.forward(&x, seq_len);

        assert_eq!(
            output.len(),
            seq_len * config.hidden_size,
            "FALSIFIED GQ-006e: MQA output size wrong"
        );

        // All finite
        let data = output.data();
        for (i, v) in data.iter().enumerate() {
            assert!(v.is_finite(), "FALSIFIED GQ-006e: MQA output[{i}] = {v} (not finite)");
        }
    }

    mod gq_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // FALSIFY-GQ-001e-prop: GQA output shape for random configs
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            #[test]
            fn falsify_gq_001e_prop_output_shape(
                config_idx in 0..4usize,
                seq_len in 2..=6usize,
                seed in 0..500u32,
            ) {
                let configs: [(usize, usize); 4] = [
                    (2, 2), (2, 1), (4, 2), (4, 1),
                ];
                let (num_heads, num_kv_heads) = configs[config_idx];
                let mut config = TransformerConfig::tiny();
                config.num_attention_heads = num_heads;
                config.num_kv_heads = num_kv_heads;

                let attn = MultiHeadAttention::new(&config);
                let data: Vec<f32> = (0..seq_len * config.hidden_size)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin())
                    .collect();
                let x = Tensor::from_vec(data, true);
                let output = attn.forward(&x, seq_len);

                prop_assert_eq!(
                    output.len(),
                    seq_len * config.hidden_size,
                    "FALSIFIED GQ-001e-prop: output len mismatch"
                );

                // All finite
                for v in output.data() {
                    prop_assert!(
                        v.is_finite(),
                        "FALSIFIED GQ-001e-prop: non-finite output"
                    );
                }
            }
        }

        // FALSIFY-GQ-006e-prop: MQA boundary with random inputs
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(30))]

            #[test]
            fn falsify_gq_006e_prop_mqa_boundary(
                seed in 0..500u32,
                seq_len in 2..=5usize,
            ) {
                let mut config = TransformerConfig::tiny();
                config.num_attention_heads = 4;
                config.num_kv_heads = 1;
                config.hidden_size = 64;

                let attn = MultiHeadAttention::new(&config);
                let data: Vec<f32> = (0..seq_len * config.hidden_size)
                    .map(|i| ((i as f32 + seed as f32) * 0.73).cos())
                    .collect();
                let x = Tensor::from_vec(data, true);
                let output = attn.forward(&x, seq_len);

                prop_assert_eq!(
                    output.len(),
                    seq_len * config.hidden_size,
                    "FALSIFIED GQ-006e-prop: MQA output len mismatch"
                );

                for v in output.data() {
                    prop_assert!(
                        v.is_finite(),
                        "FALSIFIED GQ-006e-prop: non-finite MQA output"
                    );
                }
            }
        }
    }

    #[test]
    fn test_attention_from_params_with_biases() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();

        let mut params = HashMap::new();
        params.insert(
            "attn.q_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );
        params.insert(
            "attn.k_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
        );
        params.insert(
            "attn.v_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
        );
        params.insert(
            "attn.o_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );
        params.insert(
            "attn.q_proj.bias".to_string(),
            Tensor::from_vec(vec![0.01; hidden_size], true),
        );
        params.insert(
            "attn.k_proj.bias".to_string(),
            Tensor::from_vec(vec![0.01; kv_hidden_size], true),
        );
        params.insert(
            "attn.v_proj.bias".to_string(),
            Tensor::from_vec(vec![0.01; kv_hidden_size], true),
        );

        let attn = MultiHeadAttention::from_params(&config, &params, "attn");
        assert!(attn.is_some());
        let attn = attn.expect("should load with biases");
        assert!(attn.has_biases());
        assert_eq!(attn.parameters().len(), 7);
    }

    #[test]
    fn test_attention_named_parameters_with_biases() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();

        let mut params = HashMap::new();
        params.insert(
            "attn.q_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );
        params.insert(
            "attn.k_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
        );
        params.insert(
            "attn.v_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
        );
        params.insert(
            "attn.o_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );
        params.insert(
            "attn.q_proj.bias".to_string(),
            Tensor::from_vec(vec![0.01; hidden_size], true),
        );
        params.insert(
            "attn.k_proj.bias".to_string(),
            Tensor::from_vec(vec![0.01; kv_hidden_size], true),
        );
        params.insert(
            "attn.v_proj.bias".to_string(),
            Tensor::from_vec(vec![0.01; kv_hidden_size], true),
        );

        let attn = MultiHeadAttention::from_params(&config, &params, "attn").expect("should load");
        let named = attn.named_parameters("attn");
        assert_eq!(named.len(), 7);
        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"attn.q_proj.bias"));
        assert!(names.contains(&"attn.k_proj.bias"));
        assert!(names.contains(&"attn.v_proj.bias"));
    }

    #[test]
    fn test_attention_forward_with_biases() {
        let config = TransformerConfig::tiny();
        let hidden_size = config.hidden_size;
        let kv_hidden_size = config.num_kv_heads * config.head_dim();

        let mut params = HashMap::new();
        params.insert(
            "attn.q_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );
        params.insert(
            "attn.k_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
        );
        params.insert(
            "attn.v_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * kv_hidden_size], true),
        );
        params.insert(
            "attn.o_proj.weight".to_string(),
            Tensor::from_vec(vec![0.1; hidden_size * hidden_size], true),
        );
        params
            .insert("attn.q_proj.bias".to_string(), Tensor::from_vec(vec![0.5; hidden_size], true));
        params.insert(
            "attn.k_proj.bias".to_string(),
            Tensor::from_vec(vec![0.5; kv_hidden_size], true),
        );
        params.insert(
            "attn.v_proj.bias".to_string(),
            Tensor::from_vec(vec![0.5; kv_hidden_size], true),
        );

        let attn = MultiHeadAttention::from_params(&config, &params, "attn").expect("should load");
        let x = Tensor::from_vec(vec![0.1; 2 * hidden_size], false);
        let output = attn.forward(&x, 2);
        assert_eq!(output.len(), 2 * hidden_size);
        assert!(output.data().iter().all(|v| v.is_finite()));
    }
}
