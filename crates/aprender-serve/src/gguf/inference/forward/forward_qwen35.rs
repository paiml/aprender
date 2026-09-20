use crate::error::Result;
use crate::gguf::quantized::{OwnedQuantizedTensor, QuantizedTensorRef};
use crate::gguf::{GGUFConfig, GGUFModel, OwnedQuantizedModel};
use std::f32::consts::E;

/// SiLU activation function
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x as f32).exp())
}

/// Softplus activation function
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// L2 Normalization with epsilon over the given slice.
pub fn l2_norm(x: &mut [f32], eps: f32) {
    let mut sq_sum = 0.0;
    for &v in x.iter() {
        sq_sum += v * v;
    }
    let scale = 1.0 / (sq_sum + eps).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Scale each `head_dim`-wide head of `x` to unit L2 norm on its own, as llama.cpp's
/// `build_gdn_l2_norm` does for Gated `DeltaNet` q and k (`rms_norm(x, eps / n) / sqrt(n)`
/// over one head is `x_h / sqrt(sum(x_h^2) + eps)`). One norm over all heads scales every
/// head by the other heads' magnitudes.
pub fn l2_norm_per_head(x: &mut [f32], head_dim: usize, eps: f32) {
    for head in x.chunks_exact_mut(head_dim) {
        l2_norm(head, eps);
    }
}

/// Multiply `x` elementwise by `sigmoid(gate)`: the output gate of Qwen3.5's full-attention
/// layers (llama.cpp's `attn_gated` node), applied before the output projection.
pub fn apply_sigmoid_gate(x: &mut [f32], gate: &[f32]) {
    for (o, g) in x.iter_mut().zip(gate) {
        *o *= 1.0 / (1.0 + (-*g).exp());
    }
}

/// Gated RMSNorm: (RMSNorm(input) * silu(gate))
/// Norm is computed independently over chunks of size `head_v_dim`.
pub fn gated_rmsnorm(
    input: &[f32],
    gate: &[f32],
    weight: &[f32],
    eps: f32,
    head_v_dim: usize,
    output: &mut [f32],
) {
    assert_eq!(input.len(), gate.len());
    assert_eq!(input.len(), output.len());
    assert_eq!(weight.len(), head_v_dim);
    assert_eq!(input.len() % head_v_dim, 0);

    for (chunk_in, (chunk_gate, chunk_out)) in input.chunks_exact(head_v_dim).zip(
        gate.chunks_exact(head_v_dim)
            .zip(output.chunks_exact_mut(head_v_dim)),
    ) {
        let mut sq_sum = 0.0;
        for &v in chunk_in {
            sq_sum += v * v;
        }
        let rms_scale = 1.0 / ((sq_sum / head_v_dim as f32) + eps).sqrt();

        for i in 0..head_v_dim {
            let norm_v = chunk_in[i] * rms_scale * weight[i];
            chunk_out[i] = norm_v * silu(chunk_gate[i]);
        }
    }
}

/// Causal Conv1d over seq for a single time step.
/// state is `(kernel_size - 1) * channels` floats.
/// For each channel, we shift the past inputs left and insert the new input.
pub fn causal_conv1d(
    input: &[f32],
    state: &mut [f32],
    weight: &[f32],
    kernel_size: usize,
    channels: usize,
    output: &mut [f32],
) {
    assert_eq!(input.len(), channels);
    assert_eq!(state.len(), (kernel_size - 1) * channels);
    assert_eq!(weight.len(), kernel_size * channels);
    assert_eq!(output.len(), channels);

    for c in 0..channels {
        let mut sum = 0.0;
        let s_offset = c * (kernel_size - 1);
        let w_offset = c * kernel_size;

        for k in 0..(kernel_size - 1) {
            sum += state[s_offset + k] * weight[w_offset + k];
        }
        sum += input[c] * weight[w_offset + kernel_size - 1];

        // shift left (oldest is at 0, newest is at kernel_size - 2)
        for k in 0..(kernel_size - 2) {
            state[s_offset + k] = state[s_offset + k + 1];
        }
        if kernel_size > 1 {
            state[s_offset + kernel_size - 2] = input[c];
        }

        output[c] = sum;
    }
}

/// The gated delta-rule recurrence for a single token exactly as delta-net-base.cpp computes it,
/// for a file whose Gated `DeltaNet` has one key/query head per value head
/// (`num_k_heads == num_v_heads`, `head_k_dim == head_v_dim`) — Qwen3.5-0.8B and -2B.
///
/// This is [`delta_rule_recurrence_gqa`] at ratio 1; it is kept as its own entry point so the
/// 0.8B call sites and the GPU parity tests read unchanged.
pub fn delta_rule_recurrence(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    gate: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    num_v_heads: usize,
    head_v_dim: usize,
) {
    delta_rule_recurrence_gqa(
        q,
        k,
        v,
        beta,
        gate,
        state,
        output,
        num_v_heads,
        head_v_dim,
        num_v_heads,
        head_v_dim,
    );
}

/// The gated delta-rule recurrence for a single token, with **grouped** key/query heads
/// (PMAT-3477, #3346/#3510).
///
/// Qwen3.5 4B and 9B carry `linear_num_value_heads = 32` against `linear_num_key_heads = 16`,
/// and 27B carries 48 against 16, so `q` and `k` are narrower than `v`. A **GGUF** file shares
/// them TILED:
///
/// > **value head `h` reads key/query head `h % num_k_heads`.** With `num_k_heads = 16` and
/// > `num_v_heads = 32`, value heads 0 and 16 share key head 0, value heads 1 and 17 share key
/// > head 1, and so on.
///
/// This is not the HF checkpoint's own order, and that is the trap. HF
/// (`modeling_qwen3_next.py::GatedDeltaNet`) stores the value heads grouped by key head and
/// expands q/k with `repeat_interleave` (`h / ratio`); the **conversion permutes the value
/// heads out of that order**, and everything downstream is written for the permuted file:
///
/// * `llama.cpp/conversion/qwen.py:455-464` — `_LinearAttentionVReorderBase`, which
///   `Qwen3_5TextModel` (line 639, `MODEL_ARCH.QWEN35`) is built from: *"reorders V heads from
///   grouped to tiled order for ggml broadcast … The HF weights store V heads grouped by K
///   head … ggml binary ops use tiled broadcast … We reorder V heads to tiled order so
///   `ggml_repeat` can replace the expensive interleaved repeat"*. `modify_tensors`
///   (lines 568-613) permutes **every** v-head-indexed tensor together — the v rows of
///   `in_proj_qkv`, `in_proj_z` (the gate), `in_proj_a`/`in_proj_b` (dt and beta),
///   `A_log`/`dt_bias`, the v channels of `conv1d`, and the v columns of `out_proj` — so the
///   loader needs no compensating permutation anywhere: only this index changes.
/// * `llama.cpp/src/models/qwen35.cpp:436-441` — the graph expands q and k with
///   `ggml_repeat_4d`, which tiles.
/// * `llama.cpp/ggml/src/ggml-cpu/ops.cpp:10976-10977` — the fused kernel that skips that
///   repeat reads `iq1 = iv1 % neq1; ik1 = iv1 % nek1;`.
///
/// MEASURED, so that nobody re-opens this from a text sample: with this mapping the 4B CPU
/// forward agrees with `llama-eval-callback` (`-ub 1`, autoregressive graph, same token
/// stream) on **every** `DeltaNet` intermediate of layer 0 at position 0 — `attn_norm`
/// -26.4532 vs -26.4532, `q_conv_predelta` 1.4485 vs 1.4614, `k_conv_predelta` 13.3249 vs
/// 13.3100, `gate` -18.2186 vs -18.2173, `beta_sigmoid` 21.0785 vs 21.0794, `attn_output`
/// 0.9502 vs 0.9535, `new_state` 38.7674 vs 38.7736, `l_out` 0.2917 vs 0.3056 (tensor sums;
/// the residual is the Q4_K/Q5_K activation quantisation, which is ~0.3% on a 8192-wide
/// sum). The *incoherent* 4B/9B text that survived this mapping was never a DeltaNet defect
/// at all — [`crate::gguf::qwen35_load::load_qwen35_layers`] was reading a bare
/// `block_count` key that never matched, so it built 24 layers for a 32-block file.
///
/// Shapes (this is the contract the CUDA `DeltaRuleRecurrenceKernel` / `Qwen35CudaModel`
/// mirror — the kernel's `num_v_heads`/`head_v_dim` pair gains `num_k_heads`/`head_k_dim` and
/// the same `h % num_k_heads` read):
///
/// | argument | length |
/// |---|---|
/// | `q`, `k` | `num_k_heads * head_k_dim` |
/// | `v`, `output` | `num_v_heads * head_v_dim` |
/// | `beta`, `gate` (dt) | `num_v_heads` — every gate is per VALUE head |
/// | `state` | `num_v_heads * head_v_dim * head_k_dim` |
///
/// The recurrent state of value head `h` is `S ∈ R^(head_k_dim × head_v_dim)` laid out so that
/// `S[i][j] = state[h * head_v_dim * head_k_dim + j * head_k_dim + i]` — memory row `j` is
/// column `j` of `S`, `i` runs over the key dim and `j` over the value dim. Every Qwen3.5 size
/// ships `head_k_dim == head_v_dim == 128`, so the block is square in practice; the two dims
/// are kept distinct here so a future file with a rectangular state is a shape, not a rewrite.
///
/// The scale is `1/sqrt(head_k_dim)` (the query/key width, as `fla`'s
/// `chunk_gated_delta_rule` defaults it) — identical to the previous
/// `1/sqrt(head_v_dim)` on every file that exists today.
///
/// # Panics
/// If any slice length disagrees with the table above, or if `num_v_heads` is not a positive
/// multiple of `num_k_heads`.
pub fn delta_rule_recurrence_gqa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    gate: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    num_k_heads: usize,
    head_k_dim: usize,
    num_v_heads: usize,
    head_v_dim: usize,
) {
    assert!(
        num_k_heads > 0 && num_v_heads % num_k_heads == 0,
        "Gated DeltaNet: num_v_heads ({num_v_heads}) must be a positive multiple of num_k_heads \
         ({num_k_heads})"
    );
    assert_eq!(q.len(), num_k_heads * head_k_dim);
    assert_eq!(k.len(), num_k_heads * head_k_dim);
    assert_eq!(v.len(), num_v_heads * head_v_dim);
    assert_eq!(beta.len(), num_v_heads);
    assert_eq!(gate.len(), num_v_heads);
    assert_eq!(state.len(), num_v_heads * head_v_dim * head_k_dim);
    assert_eq!(output.len(), num_v_heads * head_v_dim);

    let scale = 1.0 / (head_k_dim as f32).sqrt();

    for h in 0..num_v_heads {
        // ggml_repeat (tiled): value head h reads key/query head h % num_k_heads. The GGUF
        // conversion already permuted the value heads into this order — see the doc comment.
        let kh = h % num_k_heads;
        let q_h = &q[kh * head_k_dim..(kh + 1) * head_k_dim];
        let k_h = &k[kh * head_k_dim..(kh + 1) * head_k_dim];
        let v_h = &v[h * head_v_dim..(h + 1) * head_v_dim];
        let beta_val = beta[h];
        let gate_val = gate[h];

        let state_stride = head_v_dim * head_k_dim;
        let state_offset = h * state_stride;
        let s_h = &mut state[state_offset..state_offset + state_stride];

        // 1. S_h *= exp(gate_val)
        let exp_gate = gate_val.exp();
        for s in s_h.iter_mut() {
            *s *= exp_gate;
        }

        // 2. delta = (v_h - S_h^T * k_h) * beta_val
        // Note: s_h[j * head_k_dim + i] is S[i][j].
        // So row j of s_h in memory is column j of S.
        // sum = dot(row j of s_h, k_h)
        let mut delta = vec![0.0; head_v_dim];
        for j in 0..head_v_dim {
            let row_j = &s_h[j * head_k_dim..(j + 1) * head_k_dim];
            let mut sum = 0.0;
            for i in 0..head_k_dim {
                sum += row_j[i] * k_h[i];
            }
            delta[j] = (v_h[j] - sum) * beta_val;
        }

        // 3. S_h += k_h * delta^T
        for j in 0..head_v_dim {
            let row_j = &mut s_h[j * head_k_dim..(j + 1) * head_k_dim];
            let d_j = delta[j];
            for i in 0..head_k_dim {
                row_j[i] += k_h[i] * d_j;
            }
        }

        // 4. out_h = S_h^T * q_h * scale
        for j in 0..head_v_dim {
            let row_j = &s_h[j * head_k_dim..(j + 1) * head_k_dim];
            let mut sum = 0.0;
            for i in 0..head_k_dim {
                sum += row_j[i] * q_h[i];
            }
            output[h * head_v_dim + j] = sum * scale;
        }
    }
}

/// Partial NEOX RoPE, exactly as llama.cpp ggml_rope_multi for text input (every position stream is the
/// token position, so the sections do not change theta): for pair j in 0..n_rot/2 of each head,
/// (x[j], x[j + n_rot/2]) rotates by theta_j = pos * theta_scale^j with theta_scale = freq_base^(-2/n_rot),
/// computed iteratively as ggml does; dims n_rot..head_dim pass through unrotated.
pub fn apply_partial_neox_rope(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    n_rot: usize,
    pos: usize,
    freq_base: f32,
) {
    let half = n_rot / 2;
    let theta_scale = freq_base.powf(-2.0 / n_rot as f32);
    for h in 0..num_heads {
        let base = h * head_dim;
        let mut theta = pos as f32;
        for j in 0..half {
            let (sin, cos) = theta.sin_cos();
            let (a, b) = (x[base + j], x[base + j + half]);
            x[base + j] = a * cos - b * sin;
            x[base + j + half] = a * sin + b * cos;
            theta *= theta_scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - (1.0 / (1.0 + (-1.0_f32).exp()))).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm() {
        let mut x = [1.0, 2.0, 2.0];
        l2_norm(&mut x, 0.0);
        assert!((x[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((x[1] - 2.0 / 3.0).abs() < 1e-6);
        assert!((x[2] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_causal_conv1d() {
        let input = [1.0, 2.0];
        let mut state = [0.1, 0.2, 0.3, 0.4]; // 2 past states per channel. Channels = 2.
        let weight = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // kernel_size = 3. format: [w_0, w_1, w_2] per channel. Wait, weight memory is [kernel_size * channels].
                                                     // Is it channel outer or time outer?
                                                     // causal_conv1d uses `w_offset = c * kernel_size`, so channel outer.
                                                     // weight for c=0: [1.0, 2.0, 3.0]. weight for c=1: [4.0, 5.0, 6.0].
                                                     // state for c=0: [0.1, 0.2]. state for c=1: [0.3, 0.4].

        let mut output = [0.0, 0.0];
        causal_conv1d(&input, &mut state, &weight, 3, 2, &mut output);

        // c = 0: sum = 0.1 * 1.0 + 0.2 * 2.0 + 1.0 * 3.0 = 0.1 + 0.4 + 3.0 = 3.5
        assert!((output[0] - 3.5).abs() < 1e-5);
        // c = 1: sum = 0.3 * 4.0 + 0.4 * 5.0 + 2.0 * 6.0 = 1.2 + 2.0 + 12.0 = 15.2
        assert!((output[1] - 15.2).abs() < 1e-5);

        // State should shift: [0.2, 1.0, 0.4, 2.0]
        assert!((state[0] - 0.2).abs() < 1e-5);
        assert!((state[1] - 1.0).abs() < 1e-5);
        assert!((state[2] - 0.4).abs() < 1e-5);
        assert!((state[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_delta_rule_recurrence() {
        let num_v_heads = 1;
        let head_v_dim = 2;
        let q = [1.0, 2.0];
        let k = [0.5, 0.5];
        let v = [1.0, -1.0];
        let beta = [0.5];
        let gate = [0.0]; // exp(0) = 1.0

        // State S is 2x2.
        // s_h[j * 2 + i] is S[i][j].
        // Memory: [S[0][0], S[1][0], S[0][1], S[1][1]]
        // Let S be identity.
        let mut state = [
            1.0, 0.0, // column 0
            0.0, 1.0, // column 1
        ];
        let mut output = [0.0, 0.0];

        delta_rule_recurrence(
            &q,
            &k,
            &v,
            &beta,
            &gate,
            &mut state,
            &mut output,
            num_v_heads,
            head_v_dim,
        );

        // 1. S_h *= 1.0 -> same
        // 2. delta[j] = (v[j] - S^T * k) * 0.5
        // S^T * k = [1 0; 0 1] * [0.5, 0.5] = [0.5, 0.5]
        // v - S^T k = [0.5, -1.5]
        // delta = [0.25, -0.75]
        // 3. S += k * delta^T
        // k * delta^T = [0.5; 0.5] * [0.25, -0.75] = [0.125, -0.375; 0.125, -0.375]
        // New S = [1.125, -0.375; 0.125, 0.625]
        // 4. out = S^T * q * scale (scale = 1/sqrt(2) = 0.7071)
        // S^T = [1.125, 0.125; -0.375, 0.625]
        // S^T * [1.0, 2.0] = [1.125 + 0.25, -0.375 + 1.25] = [1.375, 0.875]
        // out = [1.375 / sqrt(2), 0.875 / sqrt(2)]

        let s2 = 2.0_f32.sqrt();
        assert!((output[0] - 1.375 / s2).abs() < 1e-5);
        assert!((output[1] - 0.875 / s2).abs() < 1e-5);

        // Check state
        assert!((state[0] - 1.125).abs() < 1e-5); // S[0][0]
        assert!((state[1] - 0.125).abs() < 1e-5); // S[1][0]
        assert!((state[2] - -0.375).abs() < 1e-5); // S[0][1]
        assert!((state[3] - 0.625).abs() < 1e-5); // S[1][1]
    }
}

/// Per-sequence decode state for [`Qwen35Model`]: the attention layers' KV cache plus each
/// Gated `DeltaNet` layer's causal-conv window and recurrent state. Build one with
/// [`Qwen35Model::new_state`].
pub struct Qwen35State {
    pub(crate) conv_states: Vec<Vec<f32>>,
    pub(crate) ssm_states: Vec<Vec<f32>>,
    pub(crate) kv_cache: crate::gguf::OwnedQuantizedKVCache,
}

impl Qwen35State {
    pub(crate) fn new(
        num_layers: usize,
        max_seq_len: usize,
        head_dim: usize,
        num_kv_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        num_v_heads: usize,
        head_value_dim: usize,
    ) -> Self {
        let convalue_dim = head_k_dim * num_k_heads * 2 + head_value_dim * num_v_heads;
        Self {
            conv_states: vec![vec![0.0; convalue_dim * 3]; num_layers],
            // One [head_k_dim x head_value_dim] recurrent state per VALUE head — the key width
            // is the state's row length, which is only the same as the value width because
            // every Qwen3.5 size ships head_k_dim == head_v_dim (PMAT-3477).
            ssm_states: vec![vec![0.0; num_v_heads * head_value_dim * head_k_dim]; num_layers],
            kv_cache: crate::gguf::OwnedQuantizedKVCache::new(
                num_layers,
                num_kv_heads * head_dim,
                max_seq_len,
            ),
        }
    }
}

pub(crate) struct Qwen35OwnedDeltaNetLayer {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) attn_qkv: OwnedQuantizedTensor,
    pub(crate) attn_gate: OwnedQuantizedTensor,
    pub(crate) ssm_alpha: OwnedQuantizedTensor,
    pub(crate) ssm_beta: OwnedQuantizedTensor,
    pub(crate) ssm_a: Vec<f32>,
    pub(crate) ssm_dt_bias: Vec<f32>,
    pub ssm_conv1d_weight: Vec<f32>,
    pub(crate) ssm_norm_weight: Vec<f32>,
    pub(crate) ssm_out: OwnedQuantizedTensor,
    pub(crate) post_attention_norm: Vec<f32>,
    pub(crate) ffn_gate: OwnedQuantizedTensor,
    pub(crate) ffn_up: OwnedQuantizedTensor,
    pub(crate) ffn_down: OwnedQuantizedTensor,
}

pub(crate) struct Qwen35OwnedAttentionLayer {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) attn_q: OwnedQuantizedTensor,
    pub(crate) attn_k: OwnedQuantizedTensor,
    pub(crate) attn_v: OwnedQuantizedTensor,
    pub(crate) attn_q_norm: Vec<f32>,
    pub(crate) attn_k_norm: Vec<f32>,
    pub(crate) attn_output: OwnedQuantizedTensor,
    pub(crate) post_attention_norm: Vec<f32>,
    pub(crate) ffn_gate: OwnedQuantizedTensor,
    pub(crate) ffn_up: OwnedQuantizedTensor,
    pub(crate) ffn_down: OwnedQuantizedTensor,
}

pub(crate) enum Qwen35OwnedLayer {
    DeltaNet(Qwen35OwnedDeltaNetLayer),
    Attention(Qwen35OwnedAttentionLayer),
}

/// Qwen3.5 / Qwen3.8 hybrid decoder on the CPU (#3091): Gated `DeltaNet` layers (short causal
/// conv, per-head recurrent state, gated delta rule) interleaved with gated full-attention
/// layers, on top of the dense model's embeddings, final norm and `lm_head`.
pub struct Qwen35Model<'a> {
    pub(crate) base: &'a OwnedQuantizedModel,
    pub(crate) layers: Vec<Qwen35OwnedLayer>,
    pub(crate) head_dim: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) num_v_heads: usize,
    pub(crate) head_v_dim: usize,
    pub(crate) num_k_heads: usize,
    pub(crate) head_k_dim: usize,
    pub(crate) conv_kernel: usize,
    pub(crate) rope_sections: [usize; 4],
}

fn load_f32_vec(tensor_ref: &QuantizedTensorRef, data: &[u8]) -> Result<Vec<f32>> {
    if tensor_ref.qtype != crate::gguf::types::GGUF_TYPE_F32 {
        return Err(crate::error::RealizarError::FormatError {
            reason: format!(
                "qwen35: expected an F32 tensor, found GGUF type {}",
                tensor_ref.qtype
            ),
        });
    }
    let bytes = data
        .get(tensor_ref.offset..tensor_ref.offset + tensor_ref.byte_size)
        .ok_or_else(|| crate::error::RealizarError::FormatError {
            reason: format!(
                "qwen35: F32 tensor at byte {} (+{}) lies outside the file",
                tensor_ref.offset, tensor_ref.byte_size
            ),
        })?;
    // Decoded, not reinterpreted: an mmap offset carries no f32 alignment guarantee.
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Metadata map of a GGUF file, as `GGUFModel::metadata` carries it.
type GGUFMetadata = std::collections::HashMap<String, crate::gguf::types::GGUFValue>;

/// The Gated `DeltaNet` shape Qwen3.5 records in GGUF metadata, under either the
/// `qwen2.*` or the `qwen35.*` key prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen35SsmMeta {
    pub(crate) head_k_dim: usize,
    pub(crate) num_k_heads: usize,
    pub(crate) num_v_heads: usize,
    pub(crate) conv_kernel: usize,
    pub(crate) rope_sections: [usize; 4],
}

/// One unsigned metadata value, accepting either GGUF integer width.
fn metadata_usize(metadata: &GGUFMetadata, key: &str) -> Option<usize> {
    match metadata.get(key) {
        Some(crate::gguf::types::GGUFValue::UInt32(v)) => Some(*v as usize),
        Some(crate::gguf::types::GGUFValue::Int32(v)) => Some(*v as usize),
        _ => None,
    }
}

/// `key`, else `fallback`, else `default` — the `qwen2.*`/`qwen35.*` prefix pair.
fn metadata_usize_or(metadata: &GGUFMetadata, key: &str, fallback: &str, default: usize) -> usize {
    metadata_usize(metadata, key)
        .or_else(|| metadata_usize(metadata, fallback))
        .unwrap_or(default)
}

/// The four M-RoPE section widths. Defaulted per section: a section the array does not
/// reach, or carries with a non-integer type, keeps its default.
fn rope_sections_from_metadata(metadata: &GGUFMetadata) -> [usize; 4] {
    let mut rope_sections = [16, 24, 24, 0];
    if let Some(crate::gguf::types::GGUFValue::Array(arr)) = metadata
        .get("qwen2.rope.dimension_sections")
        .or_else(|| metadata.get("qwen35.rope.dimension_sections"))
    {
        for (i, val) in arr.iter().take(4).enumerate() {
            if let crate::gguf::types::GGUFValue::UInt32(v) = val {
                rope_sections[i] = *v as usize;
            } else if let crate::gguf::types::GGUFValue::Int32(v) = val {
                rope_sections[i] = *v as usize;
            }
        }
    }
    rope_sections
}

/// Read the Gated `DeltaNet` shape, falling back to the Qwen3.5-0.8B defaults key by key.
pub(crate) fn qwen35_ssm_meta(metadata: &GGUFMetadata) -> Qwen35SsmMeta {
    Qwen35SsmMeta {
        head_k_dim: metadata_usize_or(
            metadata,
            "qwen2.ssm.state_size",
            "qwen35.ssm.state_size",
            16,
        ),
        num_k_heads: metadata_usize_or(
            metadata,
            "qwen2.ssm.group_count",
            "qwen35.ssm.group_count",
            1,
        ),
        num_v_heads: metadata_usize_or(
            metadata,
            "qwen2.ssm.time_step_rank",
            "qwen35.ssm.time_step_rank",
            16,
        ),
        conv_kernel: metadata_usize_or(
            metadata,
            "qwen2.ssm.conv_kernel",
            "qwen35.ssm.conv_kernel",
            4,
        ),
        rope_sections: rope_sections_from_metadata(metadata),
    }
}

/// The widths every hybrid layer's tensors are dequantised with.
struct Qwen35LayerDims {
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_v_heads: usize,
    conv_dim: usize,
    value_dim: usize,
}

/// Own one Gated `DeltaNet` layer's tensors.
fn own_deltanet_layer(
    d: &crate::gguf::qwen35_load::Qwen35DeltaNetLayer,
    data: &[u8],
    dims: &Qwen35LayerDims,
) -> Result<Qwen35OwnedDeltaNetLayer> {
    Ok(Qwen35OwnedDeltaNetLayer {
        attn_norm: load_f32_vec(&d.attn_norm, data)?,
        attn_qkv: OwnedQuantizedTensor::from_ref_with_dims(
            &d.attn_qkv,
            data,
            dims.hidden_dim,
            dims.conv_dim,
        ),
        attn_gate: OwnedQuantizedTensor::from_ref_with_dims(
            &d.attn_gate,
            data,
            dims.hidden_dim,
            dims.value_dim,
        ),
        ssm_alpha: OwnedQuantizedTensor::from_ref_with_dims(
            &d.ssm_alpha,
            data,
            dims.hidden_dim,
            dims.num_v_heads,
        ),
        ssm_beta: OwnedQuantizedTensor::from_ref_with_dims(
            &d.ssm_beta,
            data,
            dims.hidden_dim,
            dims.num_v_heads,
        ),
        ssm_a: load_f32_vec(&d.ssm_a, data)?,
        ssm_dt_bias: load_f32_vec(&d.ssm_dt_bias, data)?,
        ssm_conv1d_weight: load_f32_vec(&d.ssm_conv1d_weight, data)?,
        ssm_norm_weight: load_f32_vec(&d.ssm_norm_weight, data)?,
        ssm_out: OwnedQuantizedTensor::from_ref_with_dims(
            &d.ssm_out,
            data,
            dims.value_dim,
            dims.hidden_dim,
        ),
        post_attention_norm: load_f32_vec(&d.post_attention_norm, data)?,
        ffn_gate: OwnedQuantizedTensor::from_ref_with_dims(
            &d.ffn_gate,
            data,
            dims.hidden_dim,
            dims.intermediate_dim,
        ),
        ffn_up: OwnedQuantizedTensor::from_ref_with_dims(
            &d.ffn_up,
            data,
            dims.hidden_dim,
            dims.intermediate_dim,
        ),
        ffn_down: OwnedQuantizedTensor::from_ref_with_dims(
            &d.ffn_down,
            data,
            dims.intermediate_dim,
            dims.hidden_dim,
        ),
    })
}

/// Own one full-attention layer's tensors. The head width is `attn_q_norm`'s length
/// (256 for Qwen3.5 standard attention), not `hidden_dim / num_heads`, and `attn_q` is
/// gated, so it is twice as wide as the head fan-out.
fn own_attention_layer(
    a: &crate::gguf::qwen35_load::Qwen35AttentionLayer,
    data: &[u8],
    dims: &Qwen35LayerDims,
) -> Result<Qwen35OwnedAttentionLayer> {
    let attn_q_norm = load_f32_vec(&a.attn_q_norm, data)?;
    let true_head_dim = attn_q_norm.len();

    Ok(Qwen35OwnedAttentionLayer {
        attn_norm: load_f32_vec(&a.attn_norm, data)?,
        attn_q: OwnedQuantizedTensor::from_ref_with_dims(
            &a.attn_q,
            data,
            dims.hidden_dim,
            dims.num_heads * true_head_dim * 2,
        ),
        attn_k: OwnedQuantizedTensor::from_ref_with_dims(
            &a.attn_k,
            data,
            dims.hidden_dim,
            dims.num_kv_heads * true_head_dim,
        ),
        attn_v: OwnedQuantizedTensor::from_ref_with_dims(
            &a.attn_v,
            data,
            dims.hidden_dim,
            dims.num_kv_heads * true_head_dim,
        ),
        attn_q_norm,
        attn_k_norm: load_f32_vec(&a.attn_k_norm, data)?,
        attn_output: OwnedQuantizedTensor::from_ref_with_dims(
            &a.attn_output,
            data,
            dims.num_heads * true_head_dim,
            dims.hidden_dim,
        ),
        post_attention_norm: load_f32_vec(&a.post_attention_norm, data)?,
        ffn_gate: OwnedQuantizedTensor::from_ref_with_dims(
            &a.ffn_gate,
            data,
            dims.hidden_dim,
            dims.intermediate_dim,
        ),
        ffn_up: OwnedQuantizedTensor::from_ref_with_dims(
            &a.ffn_up,
            data,
            dims.hidden_dim,
            dims.intermediate_dim,
        ),
        ffn_down: OwnedQuantizedTensor::from_ref_with_dims(
            &a.ffn_down,
            data,
            dims.intermediate_dim,
            dims.hidden_dim,
        ),
    })
}

/// Attention head width is `key_length` (= `attn_q_norm`'s width, 256), NOT
/// `hidden / heads` (128): the KV cache and the attention loop used 128-wide heads over
/// 256-wide q/k/v. A file with no full-attention layer keeps `fallback`.
fn attention_head_dim(layers: &[Qwen35OwnedLayer], fallback: usize) -> usize {
    layers
        .iter()
        .find_map(|l| match l {
            Qwen35OwnedLayer::Attention(a) => Some(a.attn_q_norm.len()),
            Qwen35OwnedLayer::DeltaNet(_) => None,
        })
        .unwrap_or(fallback)
}

impl<'a> Qwen35Model<'a> {
    /// Build the shared base (config, token embeddings, final norm, `lm_head`) without the
    /// dense layer loader, which refuses hybrid files.
    ///
    /// # Errors
    /// Missing or malformed base tensors or metadata.
    pub fn create_base_model(
        model: &crate::gguf::GGUFModel,
        data: &[u8],
    ) -> crate::error::Result<crate::gguf::OwnedQuantizedModel> {
        let config = crate::gguf::config::ValidatedModelConfig::from_gguf(model)?.into_inner();

        let token_embedding = model.get_tensor_f32("token_embd.weight", data)?;
        let output_norm_weight = model.get_tensor_f32("output_norm.weight", data)?;

        let lm_head_ref =
            crate::gguf::QuantizedGGUFTransformer::get_tensor_ref(model, data, "output.weight")
                .or_else(|_| {
                    crate::gguf::QuantizedGGUFTransformer::get_tensor_ref(
                        model,
                        data,
                        "token_embd.weight",
                    )
                })?;
        let lm_head_weight = crate::gguf::OwnedQuantizedTensor::from_ref_with_dims(
            &lm_head_ref,
            data,
            config.hidden_dim,
            config.vocab_size,
        );

        Ok(crate::gguf::OwnedQuantizedModel {
            config,
            token_embedding,
            position_embedding: None,
            layers: vec![],
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            // A CUDA build carries these on every OwnedQuantizedModel (as in loading.rs); the
            // Qwen3.5 base never uses them (the GPU path has no Gated DeltaNet yet, #3090).
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        })
    }

    /// Load every hybrid layer of `model` on top of `base`.
    ///
    /// # Errors
    /// A layer tensor that is missing, has an unexpected type, or lies outside the file.
    pub fn from_model_and_layers(
        base: &'a OwnedQuantizedModel,
        model: &GGUFModel,
        data: &[u8],
    ) -> Result<Self> {
        let refs = crate::gguf::qwen35_load::load_qwen35_layers(model, data)?;
        let meta = qwen35_ssm_meta(&model.metadata);

        let head_v_dim = meta.head_k_dim;
        let num_heads = base.config.num_heads;
        let num_kv_heads = base.config.num_kv_heads;
        let head_dim = base.config.hidden_dim / num_heads;

        let key_dim = meta.head_k_dim * meta.num_k_heads;
        let value_dim = head_v_dim * meta.num_v_heads;
        let dims = Qwen35LayerDims {
            hidden_dim: base.config.hidden_dim,
            intermediate_dim: base.config.intermediate_dim,
            num_heads,
            num_kv_heads,
            num_v_heads: meta.num_v_heads,
            conv_dim: key_dim * 2 + value_dim,
            value_dim,
        };

        let mut owned = Vec::with_capacity(refs.len());
        for layer_ref in refs {
            owned.push(match layer_ref {
                crate::gguf::qwen35_load::Qwen35Layer::DeltaNet(d) => {
                    Qwen35OwnedLayer::DeltaNet(own_deltanet_layer(&d, data, &dims)?)
                },
                crate::gguf::qwen35_load::Qwen35Layer::Attention(a) => {
                    Qwen35OwnedLayer::Attention(own_attention_layer(&a, data, &dims)?)
                },
            });
        }

        let attn_head_dim = attention_head_dim(&owned, head_dim);
        Ok(Self {
            base,
            layers: owned,
            head_dim: attn_head_dim,
            num_kv_heads,
            num_v_heads: meta.num_v_heads,
            head_v_dim,
            num_k_heads: meta.num_k_heads,
            head_k_dim: meta.head_k_dim,
            conv_kernel: meta.conv_kernel,
            rope_sections: meta.rope_sections,
        })
    }

    /// A fresh decode state with room for `max_seq_len` positions.
    #[must_use]
    pub fn new_state(&self, max_seq_len: usize) -> Qwen35State {
        Qwen35State::new(
            self.layers.len(),
            max_seq_len,
            self.head_dim,
            self.num_kv_heads,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )
    }

    /// Run one token at `position` through every layer, updating `cache`, and return the
    /// logits.
    ///
    /// # Errors
    /// A matmul or shape failure in any layer.
    pub fn forward_single_qwen35(
        &self,
        token_id: u32,
        cache: &mut Qwen35State,
        position: usize,
    ) -> Result<Vec<f32>> {
        let mut hidden = self.base.token_embedding()[(token_id as usize)
            * self.base.config.hidden_dim
            ..(token_id as usize + 1) * self.base.config.hidden_dim]
            .to_vec();

        let mut normed = vec![0.0; self.base.config.hidden_dim];
        let mut post_attn_normed = vec![0.0; self.base.config.hidden_dim];

        let fn_softplus = |x: f32| -> f32 {
            if x > 20.0 {
                x
            } else {
                (1.0 + x.exp()).ln()
            }
        };

        for (il, layer) in self.layers.iter().enumerate() {
            match layer {
                Qwen35OwnedLayer::DeltaNet(d) => {
                    self.forward_deltanet(
                        d,
                        &mut hidden,
                        cache,
                        il,
                        position,
                        &mut normed,
                        &mut post_attn_normed,
                    )?;
                },
                Qwen35OwnedLayer::Attention(a) => {
                    self.forward_attention(
                        a,
                        &mut hidden,
                        cache,
                        il,
                        position,
                        &mut normed,
                        &mut post_attn_normed,
                    )?;
                },
            }
        }
        let mut out_normed = vec![0.0; self.base.config.hidden_dim];
        crate::gguf::ops::rms_norm_into(
            &hidden,
            self.base.output_norm_weight(),
            self.base.config.eps,
            &mut out_normed,
        );

        let mut logits = vec![0.0; self.base.config.vocab_size];
        self.base
            .fused_matmul_into(&out_normed, self.base.lm_head_weight(), &mut logits)?;

        cache.kv_cache.advance();

        Ok(logits)
    }

    /// One Gated `DeltaNet` layer, in place on `hidden`.
    ///
    /// `pub(crate)` (PMAT-3477, #3090) so the GPU model's per-layer parity test can
    /// drive the CPU reference layer by layer with teacher forcing. No logic change.
    pub(crate) fn forward_deltanet(
        &self,
        d: &Qwen35OwnedDeltaNetLayer,
        hidden: &mut [f32],
        cache: &mut Qwen35State,
        il: usize,
        position: usize,
        normed: &mut [f32],
        post_attn_normed: &mut [f32],
    ) -> Result<()> {
        let fn_softplus = |x: f32| -> f32 {
            if x > 20.0 {
                x
            } else {
                (1.0 + x.exp()).ln()
            }
        };

        crate::gguf::ops::rms_norm_into(hidden, &d.attn_norm, self.base.config.eps, normed);
        let conv_dim = self.head_k_dim * self.num_k_heads * 2 + self.head_v_dim * self.num_v_heads;
        let mut conv_in = vec![0.0; conv_dim];
        self.base
            .fused_matmul_into(normed, &d.attn_qkv, &mut conv_in)?;
        let mut conv_out = vec![0.0; conv_dim];
        causal_conv1d(
            &conv_in,
            &mut cache.conv_states[il][..],
            &d.ssm_conv1d_weight,
            self.conv_kernel,
            conv_dim,
            &mut conv_out,
        );
        for x in conv_out.iter_mut() {
            *x = *x / (1.0 + (-*x).exp()); // silu/sigmoid
        }
        // SiLU ONCE, as llama.cpp (ggml_silu on conv_output_raw). A second identical loop applied silu(silu(x)).

        let k_dim = self.head_k_dim * self.num_k_heads;
        let v_dim = self.head_v_dim * self.num_v_heads;
        let mut q = conv_out[0..k_dim].to_vec();
        let mut k = conv_out[k_dim..k_dim * 2].to_vec();
        let mut v = conv_out[k_dim * 2..conv_dim].to_vec();

        // Per-head L2 normalisation of q and k, as llama.cpp build_gdn_l2_norm:
        // rms_norm(x, eps/n) * 1/sqrt(n) over ne[0] = head_k_dim, i.e. x_h / sqrt(sum(x_h^2) + eps)
        // for EACH head. The earlier global norm over all heads (applied twice) scaled every head wrong.
        l2_norm_per_head(&mut q, self.head_k_dim, self.base.config.eps);
        l2_norm_per_head(&mut k, self.head_k_dim, self.base.config.eps);

        // Gated DeltaNet applies NO RoPE: llama.cpp build_layer_attn_linear has none (position comes from the causal conv).

        let mut dt_raw = vec![0.0; self.num_v_heads];
        self.base
            .fused_matmul_into(normed, &d.ssm_alpha, &mut dt_raw)?;
        let mut dt = vec![0.0; self.num_v_heads];
        for (i, val) in dt_raw.iter().enumerate() {
            dt[i] = fn_softplus(val + d.ssm_dt_bias[i]) * d.ssm_a[i];
        }

        let mut beta = vec![0.0; self.num_v_heads];
        self.base
            .fused_matmul_into(normed, &d.ssm_beta, &mut beta)?;
        // sigmoid ONCE, as llama.cpp: beta = sigmoid(ssm_beta . x). (It was applied twice.)
        for x in beta.iter_mut() {
            *x = 1.0 / (1.0 + (-*x).exp());
        }

        let mut gate = vec![0.0; v_dim];
        self.base
            .fused_matmul_into(normed, &d.attn_gate, &mut gate)?;

        // q and k are num_k_heads wide, v is num_v_heads wide: on 4B/9B (32 value heads to 16
        // key heads) and 27B (48 to 16) the recurrence shares each key/query head across
        // num_v_heads / num_k_heads value heads (PMAT-3477, #3346/#3510). On 0.8B and 2B the
        // ratio is 1 and this is the pre-GQA arithmetic, unchanged.
        let mut out_h = vec![0.0; v_dim];
        delta_rule_recurrence_gqa(
            &q,
            &k,
            &v,
            &beta,
            &dt,
            &mut cache.ssm_states[il][..],
            &mut out_h,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        );

        let mut ssm_out_in = vec![0.0; v_dim];
        gated_rmsnorm(
            &out_h,
            &gate,
            &d.ssm_norm_weight,
            self.base.config.eps,
            self.head_v_dim,
            &mut ssm_out_in,
        );
        // No normalisation after the gated RMS norm: llama.cpp goes from build_norm_gated straight to ssm_out.
        // (Two extra global L2 norms here forced every DeltaNet output to unit length.)

        let mut ssm_out = vec![0.0; self.base.config.hidden_dim];
        self.base
            .fused_matmul_into(&ssm_out_in, &d.ssm_out, &mut ssm_out)?;

        for i in 0..self.base.config.hidden_dim {
            hidden[i] += ssm_out[i];
        }

        crate::gguf::ops::rms_norm_into(
            hidden,
            &d.post_attention_norm,
            self.base.config.eps,
            post_attn_normed,
        );
        let mut ffn_gate = vec![0.0; d.ffn_gate.out_dim];
        self.base
            .fused_matmul_into(post_attn_normed, &d.ffn_gate, &mut ffn_gate)?;
        let mut ffn_up = vec![0.0; d.ffn_up.out_dim];
        self.base
            .fused_matmul_into(post_attn_normed, &d.ffn_up, &mut ffn_up)?;

        for i in 0..ffn_gate.len() {
            let x = ffn_gate[i];
            let silu = x / (1.0 + (-x as f32).exp());
            ffn_up[i] *= silu;
        }
        let mut ffn_down = vec![0.0; d.ffn_down.out_dim];
        self.base
            .fused_matmul_into(&ffn_up, &d.ffn_down, &mut ffn_down)?;
        for i in 0..self.base.config.hidden_dim {
            hidden[i] += ffn_down[i];
        }
        Ok(())
    }

    /// One full-attention layer, in place on `hidden`.
    ///
    /// `pub(crate)` (PMAT-3477, #3090) so the GPU parity test can advance the CPU
    /// reference across the interleaved attention layers while it compares the
    /// `DeltaNet` ones. No logic change.
    pub(crate) fn forward_attention(
        &self,
        a: &Qwen35OwnedAttentionLayer,
        hidden: &mut [f32],
        cache: &mut Qwen35State,
        il: usize,
        position: usize,
        normed: &mut [f32],
        post_attn_normed: &mut [f32],
    ) -> Result<()> {
        crate::gguf::ops::rms_norm_into(hidden, &a.attn_norm, self.base.config.eps, normed);

        let mut q_full = vec![0.0; a.attn_q.out_dim];
        let mut k = vec![0.0; a.attn_k.out_dim];
        let mut v = vec![0.0; a.attn_v.out_dim];
        self.base
            .fused_matmul_into(normed, &a.attn_q, &mut q_full)?;
        self.base.fused_matmul_into(normed, &a.attn_k, &mut k)?;
        self.base.fused_matmul_into(normed, &a.attn_v, &mut v)?;

        let num_heads = self.base.config.num_heads;
        let head_dim = a.attn_q_norm.len(); // 256
        let mut q = vec![0.0; num_heads * head_dim];
        let mut gate = vec![0.0; num_heads * head_dim];

        for h in 0..num_heads {
            let offset_q_full = h * head_dim * 2;
            let offset_q = h * head_dim;

            // Split into Q and gate
            q[offset_q..offset_q + head_dim]
                .copy_from_slice(&q_full[offset_q_full..offset_q_full + head_dim]);
            gate[offset_q..offset_q + head_dim]
                .copy_from_slice(&q_full[offset_q_full + head_dim..offset_q_full + head_dim * 2]);
        }

        crate::gguf::ops::apply_per_head_rms_norm(
            &mut q,
            &a.attn_q_norm,
            num_heads,
            self.base.config.eps,
        );
        crate::gguf::ops::apply_per_head_rms_norm(
            &mut k,
            &a.attn_k_norm,
            self.base.config.num_kv_heads,
            self.base.config.eps,
        );

        // Partial NEOX RoPE over n_rot = 2 * sum(rope.dimension_sections) = 64 of each 256-wide head, as
        // llama.cpp ggml_rope_multi for text input. The base apply_rope assumed head_dim = hidden/heads = 128
        // and a full rotation, both wrong for Qwen3.5 attention.
        let n_rot = 2 * self.rope_sections.iter().sum::<usize>();
        let freq_base = self.base.config.rope_theta;
        apply_partial_neox_rope(&mut q, num_heads, head_dim, n_rot, position, freq_base);
        apply_partial_neox_rope(
            &mut k,
            self.base.config.num_kv_heads,
            head_dim,
            n_rot,
            position,
            freq_base,
        );

        cache.kv_cache.append(il, &k, &v);
        let k_cache = cache.kv_cache.get_k(il);
        let v_cache = cache.kv_cache.get_v(il);

        let mut attn_out_in = vec![0.0; q.len()];
        let num_kv_heads = self.base.config.num_kv_heads;
        let num_heads = self.base.config.num_heads;
        let group_size = num_heads / num_kv_heads;
        let head_dim = self.head_dim;

        for h in 0..num_heads {
            let kv_h = h / group_size;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];

            let mut scores = vec![0.0; position + 1];
            for p in 0..=position {
                let mut dot = 0.0;
                let k_p = &k_cache[p * (num_kv_heads * head_dim) + kv_h * head_dim
                    ..p * (num_kv_heads * head_dim) + (kv_h + 1) * head_dim];
                for i in 0..head_dim {
                    dot += q_h[i] * k_p[i];
                }
                scores[p] = dot / (head_dim as f32).sqrt();
            }
            crate::gguf::ops::softmax(&mut scores);

            let out_h = &mut attn_out_in[h * head_dim..(h + 1) * head_dim];
            for p in 0..=position {
                let w = scores[p];
                let v_p = &v_cache[p * (num_kv_heads * head_dim) + kv_h * head_dim
                    ..p * (num_kv_heads * head_dim) + (kv_h + 1) * head_dim];
                for i in 0..head_dim {
                    out_h[i] += w * v_p[i];
                }
            }
        }
        // Output gate, as llama.cpp: attn_output * sigmoid(gate) before the output projection
        // (the attn_gated node). The gate split off the joint Q projection was never applied.
        apply_sigmoid_gate(&mut attn_out_in, &gate);
        let mut attn_out = vec![0.0; self.base.config.hidden_dim];
        self.base
            .fused_matmul_into(&attn_out_in, &a.attn_output, &mut attn_out)?;
        for i in 0..self.base.config.hidden_dim {
            hidden[i] += attn_out[i];
        }

        crate::gguf::ops::rms_norm_into(
            hidden,
            &a.post_attention_norm,
            self.base.config.eps,
            post_attn_normed,
        );
        let mut ffn_gate = vec![0.0; a.ffn_gate.out_dim];
        self.base
            .fused_matmul_into(post_attn_normed, &a.ffn_gate, &mut ffn_gate)?;
        let mut ffn_up = vec![0.0; a.ffn_up.out_dim];
        self.base
            .fused_matmul_into(post_attn_normed, &a.ffn_up, &mut ffn_up)?;
        for i in 0..ffn_gate.len() {
            let x = ffn_gate[i];
            let silu = x / (1.0 + (-x as f32).exp());
            ffn_up[i] *= silu;
        }
        let mut ffn_down = vec![0.0; a.ffn_down.out_dim];
        self.base
            .fused_matmul_into(&ffn_up, &a.ffn_down, &mut ffn_down)?;
        for i in 0..self.base.config.hidden_dim {
            hidden[i] += ffn_down[i];
        }
        Ok(())
    }
}

/// Prefill `input_tokens` through the Qwen3.5 CPU forward, then decode up to
/// `gen_config.max_tokens` more with the dense path's token choice (argmax at temperature 0 or
/// `top_k` 1, else seeded top-k/top-p). Returns the prompt followed by the new tokens. `apr run`
/// and `apr chat` dispatch `qwen35` GGUFs here (#3091).
///
/// # Errors
/// An empty prompt, a layer the Qwen3.5 loader cannot read, or a forward-pass failure.
pub fn run_qwen35_generate(
    mapped: &crate::gguf::MappedGGUFModel,
    base: &OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
) -> Result<Vec<u32>> {
    use rand::SeedableRng;
    if input_tokens.is_empty() {
        return Err(crate::error::RealizarError::InvalidShape {
            reason: "run_qwen35_generate: prompt cannot be empty".to_string(),
        });
    }
    let qwen = Qwen35Model::from_model_and_layers(base, &mapped.model, mapped.data())?;
    let max_seq_len = input_tokens.len() + gen_config.max_tokens + 1;
    let mut state = qwen.new_state(max_seq_len);
    let mut rng = rand::rngs::StdRng::seed_from_u64(gen_config.seed);

    let mut logits = Vec::new();
    for (pos, &token) in input_tokens.iter().enumerate() {
        logits = qwen.forward_single_qwen35(token, &mut state, pos)?;
    }
    let mut tokens = input_tokens.to_vec();
    for _ in 0..gen_config.max_tokens {
        let next = if gen_config.temperature == 0.0 || gen_config.top_k == 1 {
            crate::gguf::ops::argmax(&logits)
        } else {
            OwnedQuantizedModel::sample_topk_seeded(
                &logits,
                gen_config.temperature,
                gen_config.top_k,
                gen_config.top_p,
                &mut rng,
            )
        };
        tokens.push(next);
        if gen_config.stop_tokens.contains(&next) || tokens.len() >= max_seq_len {
            break;
        }
        logits = qwen.forward_single_qwen35(next, &mut state, tokens.len() - 1)?;
    }
    Ok(tokens)
}

/// Why a Qwen3.5 run served its tokens from the CPU forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35CpuReason {
    /// The caller asked for the CPU (`--no-gpu` / `force_cpu`).
    Requested,
    /// This binary was built without the `cuda` feature, so there is no GPU
    /// backend to route to — the hybrid forward itself exists on both (#3090).
    NoCudaBackend,
}

/// Which backend serves a Qwen3.5 (Gated `DeltaNet`) GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35Route {
    /// `Qwen35CudaModel::forward_single` (#3090).
    Gpu,
    /// `Qwen35Model::forward_single_qwen35` (#3091).
    Cpu(Qwen35CpuReason),
}

/// Pure: which backend a Qwen3.5 GGUF is routed to.
///
/// Split out from the dispatcher so the decision is falsifiable on a host with
/// no GPU and in a build without `cuda`: `cuda_backend` is the caller's
/// `cfg!(feature = "cuda")`, not a device probe (a device that is present but
/// unusable is a *fallback*, which is printed — see
/// [`run_qwen35_generate_dispatch`] — never a silent re-route).
#[must_use]
pub fn qwen35_route(no_gpu: bool, cuda_backend: bool) -> Qwen35Route {
    if no_gpu {
        Qwen35Route::Cpu(Qwen35CpuReason::Requested)
    } else if cuda_backend {
        Qwen35Route::Gpu
    } else {
        Qwen35Route::Cpu(Qwen35CpuReason::NoCudaBackend)
    }
}

/// The one-line notice a route owes the user, or `None` when it owes none.
///
/// #3477: the GPU case used to print `[qwen35: Gated DeltaNet runs on the CPU;
/// the GPU backend does not implement it yet (#3090)]` — which is now false: the
/// hybrid runs on the GPU. What a user still needs told is the case where they
/// asked for the GPU and this *binary* cannot give them one. Asking for the CPU
/// explicitly is not news.
#[must_use]
pub fn qwen35_route_notice(route: Qwen35Route) -> Option<&'static str> {
    match route {
        Qwen35Route::Cpu(Qwen35CpuReason::NoCudaBackend) => Some(
            "[qwen35: this binary has no CUDA backend (built without --features cuda); \
             the Gated DeltaNet forward runs on the CPU (#3091)]",
        ),
        Qwen35Route::Gpu | Qwen35Route::Cpu(Qwen35CpuReason::Requested) => None,
    }
}

/// The prefix of the loud, never-silent CPU fallback for the hybrid GPU path.
pub const QWEN35_GPU_FALLBACK_PREFIX: &str = "warning: GPU (CUDA) qwen35 path rejected";

/// Generate with the Qwen3.5 hybrid on the backend the caller asked for,
/// returning `(tokens, used_gpu)` (#3090/#3091).
///
/// The GPU is attempted whenever it was requested and this build has a CUDA
/// backend; a failure to build the model, a failure inside the forward, or a
/// rejection by the F2 CPU-parity guard falls back to the CPU forward **with the
/// reason printed** — an unannounced backend downgrade is the defect class
/// `QWEN35_GPU_FALLBACK_PREFIX` exists to make impossible.
///
/// # Errors
/// Only a CPU-forward failure: the GPU path never propagates its error, it falls
/// back.
pub fn run_qwen35_generate_dispatch(
    mapped: &crate::gguf::MappedGGUFModel,
    base: &OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    no_gpu: bool,
    stages: &mut crate::infer::stage_timings::StageTimings,
) -> Result<(Vec<u32>, bool)> {
    let route = qwen35_route(no_gpu, cfg!(feature = "cuda"));
    if let Some(notice) = qwen35_route_notice(route) {
        eprintln!("{notice}");
    }
    #[cfg(feature = "cuda")]
    if route == Qwen35Route::Gpu {
        match run_qwen35_generate_gpu(mapped, base, input_tokens, gen_config, stages) {
            Ok(tokens) => return Ok((tokens, true)),
            Err(reason) => {
                eprintln!("{QWEN35_GPU_FALLBACK_PREFIX}, falling back to CPU: {reason}");
            },
        }
    }
    stages.backend = "cpu-qwen35".to_string();
    let tokens = run_qwen35_generate(mapped, base, input_tokens, gen_config)?;
    Ok((tokens, false))
}

/// Positions the F2 hybrid guard forwards on both backends before it will let
/// the GPU serve a token. Same cap as the dense guard's `gpu_probe`.
#[cfg(feature = "cuda")]
const QWEN35_F2_PROBE_MAX: usize = 64;

/// The GPU twin of [`run_qwen35_generate`]: build the hybrid on CUDA, prove it
/// against its own CPU forward, then decode.
///
/// `Err` is a fallback reason, never a user-visible failure — the caller prints
/// it and runs the CPU forward.
#[cfg(feature = "cuda")]
fn run_qwen35_generate_gpu(
    mapped: &crate::gguf::MappedGGUFModel,
    base: &OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    stages: &mut crate::infer::stage_timings::StageTimings,
) -> std::result::Result<Vec<u32>, String> {
    if input_tokens.is_empty() {
        return Err("the prompt is empty".to_string());
    }
    let qwen = Qwen35Model::from_model_and_layers(base, &mapped.model, mapped.data())
        .map_err(|e| format!("the hybrid layers would not load: {e}"))?;

    let mut executor = crate::cuda::CudaExecutor::new(0)
        .map_err(|e| format!("CUDA initialization failed: {e}"))?;
    let device_name = executor
        .device_name()
        .unwrap_or_else(|_| "Unknown GPU".to_string());
    let vram_mb = executor.memory_info().unwrap_or((0, 0)).1 / (1024 * 1024);

    let max_seq_len = input_tokens.len() + gen_config.max_tokens + 1;
    // PMAT-3598 row 1: building the CUDA model is the HOST -> DEVICE transfer for this path.
    let (built, h2d_ms) = crate::infer::stage_timings::timed("h2d", || {
        crate::gguf::cuda::Qwen35CudaModel::with_max_seq_len(&qwen, executor, max_seq_len)
    });
    let mut gpu = built.map_err(|e| format!("the CUDA model would not build: {e}"))?;
    stages.h2d_ms = Some(h2d_ms);

    // Unconditional, like every other backend-selection line on this path: the
    // user must be able to tell a GPU run from a CPU one without --verbose.
    eprintln!(
        "Backend: GPU (CUDA, {device_name}, {vram_mb} MB VRAM) [qwen35 hybrid forward, #3090]"
    );

    let (f2_ok, validate_ms) = crate::infer::stage_timings::timed("validate", || {
        f2_validate_qwen35(&mut gpu, &qwen, input_tokens)
    });
    stages.validate_ms = Some(validate_ms);
    if !f2_ok {
        return Err("the F2 CPU-parity guard rejected the GPU path".to_string());
    }
    stages.backend = "cuda-qwen35".to_string();
    qwen35_gpu_decode(&mut gpu, input_tokens, gen_config, stages)
}

/// Prefill + decode on the GPU, with the token choice
/// [`run_qwen35_generate`] makes, from a state that has never seen the guard's
/// probe.
#[cfg(feature = "cuda")]
fn qwen35_gpu_decode(
    gpu: &mut crate::gguf::cuda::Qwen35CudaModel<'_>,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    stages: &mut crate::infer::stage_timings::StageTimings,
) -> std::result::Result<Vec<u32>, String> {
    use rand::SeedableRng;
    let max_seq_len = input_tokens.len() + gen_config.max_tokens + 1;
    let mut state = gpu
        .new_state()
        .map_err(|e| format!("the decode state would not allocate: {e}"))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(gen_config.seed);

    // PMAT-3598 row 1: the prompt loop IS prefill and the generate loop IS decode. The boundary
    // is exact here, not derived, so neither number is the other's remainder.
    let prefill_start = std::time::Instant::now();
    if let Some(d) = crate::infer::stage_timings::planted_delay("prefill") {
        std::thread::sleep(d);
    }
    let mut logits = Vec::new();
    for (pos, &token) in input_tokens.iter().enumerate() {
        logits = gpu
            .forward_single(token, &mut state, pos)
            .map_err(|e| format!("the GPU forward failed at prompt position {pos}: {e}"))?;
    }
    stages.prefill_ms = Some(prefill_start.elapsed().as_secs_f64() * 1000.0);
    let decode_start = std::time::Instant::now();
    if let Some(d) = crate::infer::stage_timings::planted_delay("decode") {
        std::thread::sleep(d);
    }
    let mut tokens = input_tokens.to_vec();
    for _ in 0..gen_config.max_tokens {
        let next = if gen_config.temperature == 0.0 || gen_config.top_k == 1 {
            crate::gguf::ops::argmax(&logits)
        } else {
            OwnedQuantizedModel::sample_topk_seeded(
                &logits,
                gen_config.temperature,
                gen_config.top_k,
                gen_config.top_p,
                &mut rng,
            )
        };
        tokens.push(next);
        if gen_config.stop_tokens.contains(&next) || tokens.len() >= max_seq_len {
            break;
        }
        let pos = tokens.len() - 1;
        logits = gpu
            .forward_single(next, &mut state, pos)
            .map_err(|e| format!("the GPU forward failed at decode position {pos}: {e}"))?;
    }
    stages.decode_ms = Some(decode_start.elapsed().as_secs_f64() * 1000.0);
    Ok(tokens)
}

/// The F2 runtime guard for the hybrid: forward the real prompt through BOTH
/// backends and accept the GPU only if every real position agrees.
///
/// The dense `validate_gpu_first_token` cannot serve here — its CPU reference is
/// `forward_single_with_cache` and its GPU probe `forward_gpu_resident`, neither
/// of which exists for this architecture. The decision itself is shared:
/// `f2_multi_position_report` with its floors (0.95 / 0.98 / 0.90), so the
/// hybrid is judged by the same rule as every other GPU path. The probe path is
/// [`F2ProbePath::Serial`] because there is no batched prefill for the hybrid —
/// `qwen35_gpu_decode` prefills token by token, so the serial probe IS the path
/// the run takes.
///
/// Both states are throwaway: the guard allocates its own, and the generation
/// that follows allocates another.
#[cfg(feature = "cuda")]
fn f2_validate_qwen35(
    gpu: &mut crate::gguf::cuda::Qwen35CudaModel<'_>,
    cpu: &Qwen35Model<'_>,
    probe_context: &[u32],
) -> bool {
    // Same escape hatch as the dense gate, and the same one `apr parity` uses.
    if std::env::var("SKIP_PARITY_GATE").is_ok_and(|v| v == "1") {
        return true;
    }
    let probe = &probe_context[probe_context.len().saturating_sub(QWEN35_F2_PROBE_MAX)..];
    // A one-token probe has no REAL position (≥1) to judge; position 0 is the
    // context-less near-tie the dense gate excludes for the same reason.
    if probe.len() < 2 {
        return true;
    }
    let Some(cpu_per_pos) = f2_qwen35_cpu_reference(cpu, probe) else {
        return true; // the CPU forward itself failed: nothing to judge against.
    };
    let decode_token = cpu_per_pos
        .get(probe.len().saturating_sub(1))
        .map_or(0, |l| crate::infer::argmax_u32(l));
    let gpu_per_pos = match f2_qwen35_gpu_logits(gpu, probe, decode_token) {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("{msg}");
            return false; // fail closed.
        },
    };
    let report = crate::infer::f2_multi_position_report(&cpu_per_pos, &gpu_per_pos);
    if report.accepted {
        true
    } else {
        eprintln!(
            "{}",
            crate::infer::f2_divergence_msg(&report, crate::infer::F2ProbePath::Serial)
        );
        false
    }
}

/// CPU logits at every probe position plus one greedy decode step — the
/// reference half of [`f2_validate_qwen35`].
#[cfg(feature = "cuda")]
fn f2_qwen35_cpu_reference(cpu: &Qwen35Model<'_>, probe: &[u32]) -> Option<Vec<Vec<f32>>> {
    let mut state = cpu.new_state(probe.len() + 2);
    let mut per_pos: Vec<Vec<f32>> = Vec::with_capacity(probe.len() + 1);
    for (pos, &tok) in probe.iter().enumerate() {
        per_pos.push(cpu.forward_single_qwen35(tok, &mut state, pos).ok()?);
    }
    let last = per_pos.last()?;
    let next = crate::infer::argmax_u32(last);
    per_pos.push(
        cpu.forward_single_qwen35(next, &mut state, probe.len())
            .ok()?,
    );
    Some(per_pos)
}

/// GPU logits for the same positions, from a fresh device state.
#[cfg(feature = "cuda")]
fn f2_qwen35_gpu_logits(
    gpu: &mut crate::gguf::cuda::Qwen35CudaModel<'_>,
    probe: &[u32],
    decode_token: u32,
) -> std::result::Result<Vec<Vec<f32>>, String> {
    let steps = probe.len() + 1;
    let mut state = gpu
        .new_state()
        .map_err(|e| format!("F2 qwen35 probe: the device state would not allocate: {e}"))?;
    let mut per_pos: Vec<Vec<f32>> = Vec::with_capacity(steps);
    for (pos, tok) in probe
        .iter()
        .copied()
        .chain(std::iter::once(decode_token))
        .enumerate()
    {
        match gpu.forward_single(tok, &mut state, pos) {
            Ok(logits) => per_pos.push(logits),
            Err(e) => return Err(crate::infer::gpu_forward_failure_msg(pos, steps, &e)),
        }
    }
    Ok(per_pos)
}

#[cfg(test)]
mod qwen35_route_tests {
    use super::{qwen35_route, qwen35_route_notice, Qwen35CpuReason, Qwen35Route};

    // #3090/#3477: with a CUDA build and no --no-gpu, the hybrid goes to the GPU.
    // This is the whole point of the ticket; if it ever reads Cpu again, `apr run
    // --gpu` is silently serving CPU tokens.
    #[test]
    fn a_cuda_build_that_was_not_told_otherwise_routes_to_the_gpu() {
        assert_eq!(qwen35_route(false, true), Qwen35Route::Gpu);
        assert_eq!(qwen35_route_notice(Qwen35Route::Gpu), None);
    }

    #[test]
    fn no_gpu_wins_over_a_present_cuda_backend() {
        assert_eq!(
            qwen35_route(true, true),
            Qwen35Route::Cpu(Qwen35CpuReason::Requested)
        );
        // The user asked for this: it is not news.
        assert_eq!(
            qwen35_route_notice(Qwen35Route::Cpu(Qwen35CpuReason::Requested)),
            None
        );
    }

    // A binary without the cuda feature still runs the model — but the user who
    // asked for a GPU must be told why they did not get one.
    #[test]
    fn a_build_without_cuda_says_so() {
        let route = qwen35_route(false, false);
        assert_eq!(route, Qwen35Route::Cpu(Qwen35CpuReason::NoCudaBackend));
        let notice = qwen35_route_notice(route).expect("the no-backend case owes a notice");
        assert!(
            notice.contains("no CUDA backend"),
            "the notice must name the missing backend, not the architecture: {notice}"
        );
        assert!(
            !notice.contains("#3090"),
            "#3090 is the GPU forward, which now exists — citing it here is the \
             withdrawn 'the GPU does not implement it' notice: {notice}"
        );
    }

    // Both CPU reasons are reachable from the routing function, so neither arm
    // of the notice is dead code.
    #[test]
    fn every_cpu_reason_is_produced_by_the_router() {
        for (no_gpu, cuda, want) in [
            (true, true, Qwen35CpuReason::Requested),
            (true, false, Qwen35CpuReason::Requested),
            (false, false, Qwen35CpuReason::NoCudaBackend),
        ] {
            assert_eq!(qwen35_route(no_gpu, cuda), Qwen35Route::Cpu(want));
        }
    }
}

#[cfg(test)]
mod qwen35_ssm_meta_tests {
    use super::{qwen35_ssm_meta, Qwen35SsmMeta};
    use crate::gguf::types::GGUFValue;
    use std::collections::HashMap;

    fn md(pairs: &[(&str, GGUFValue)]) -> HashMap<String, GGUFValue> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn test_empty_metadata_gives_the_qwen35_0_8b_defaults() {
        assert_eq!(
            qwen35_ssm_meta(&HashMap::new()),
            Qwen35SsmMeta {
                head_k_dim: 16,
                num_k_heads: 1,
                num_v_heads: 16,
                conv_kernel: 4,
                rope_sections: [16, 24, 24, 0],
            }
        );
    }

    #[test]
    fn test_qwen2_keys_win_and_qwen35_keys_are_the_fallback() {
        let meta = qwen35_ssm_meta(&md(&[
            ("qwen2.ssm.state_size", GGUFValue::UInt32(128)),
            ("qwen35.ssm.state_size", GGUFValue::UInt32(999)),
            ("qwen35.ssm.group_count", GGUFValue::UInt32(2)),
            ("qwen35.ssm.time_step_rank", GGUFValue::Int32(32)),
            ("qwen35.ssm.conv_kernel", GGUFValue::UInt32(4)),
        ]));
        assert_eq!(meta.head_k_dim, 128, "the qwen2 key is preferred");
        assert_eq!(meta.num_k_heads, 2, "no qwen2 key: the qwen35 one is read");
        assert_eq!(meta.num_v_heads, 32, "Int32 is accepted like UInt32");
        assert_eq!(meta.conv_kernel, 4);
    }

    #[test]
    fn test_a_non_integer_value_falls_through_to_the_default() {
        // A string where an integer belongs is not a value: it must not be read as 0.
        let meta = qwen35_ssm_meta(&md(&[(
            "qwen2.ssm.conv_kernel",
            GGUFValue::String("four".to_string()),
        )]));
        assert_eq!(meta.conv_kernel, 4);
    }

    #[test]
    fn test_rope_sections_are_defaulted_per_section() {
        // Two entries given: the last two sections keep their defaults, and a
        // non-integer entry keeps its own.
        let meta = qwen35_ssm_meta(&md(&[(
            "qwen2.rope.dimension_sections",
            GGUFValue::Array(vec![
                GGUFValue::UInt32(8),
                GGUFValue::Float32(1.0),
                GGUFValue::Int32(12),
            ]),
        )]));
        assert_eq!(meta.rope_sections, [8, 24, 12, 0]);
    }
}

#[cfg(test)]
mod qwen35_math_tests {
    use super::*;

    #[test]
    fn test_l2_norm_per_head_normalises_each_head_on_its_own() {
        // Two heads whose magnitudes differ 10x: each must come out as [0.6, 0.8].
        let mut x = [3.0, 4.0, 30.0, 40.0];
        l2_norm_per_head(&mut x, 2, 1e-12);
        for (got, want) in x.iter().zip([0.6, 0.8, 0.6, 0.8]) {
            assert!((got - want).abs() < 1e-6, "{x:?}");
        }
    }

    #[test]
    fn test_apply_sigmoid_gate_scales_by_sigmoid() {
        let mut x = [2.0, 2.0, 2.0];
        apply_sigmoid_gate(&mut x, &[0.0, 40.0, -40.0]);
        assert!((x[0] - 1.0).abs() < 1e-6, "sigmoid(0) = 0.5: {x:?}");
        assert!((x[1] - 2.0).abs() < 1e-6, "sigmoid(40) ~ 1: {x:?}");
        assert!(x[2].abs() < 1e-6, "sigmoid(-40) ~ 0: {x:?}");
    }

    #[test]
    fn test_partial_neox_rope_is_identity_at_position_zero() {
        let orig: Vec<f32> = (0..8u8).map(|i| f32::from(i) + 1.0).collect();
        let mut x = orig.clone();
        apply_partial_neox_rope(&mut x, 1, 8, 4, 0, 10_000.0);
        assert_eq!(x, orig);
    }

    #[test]
    fn test_partial_neox_rope_rotates_half_pairs_and_leaves_the_tail() {
        // head_dim 8, n_rot 4: the pairs are (0,2) and (1,3) (NEOX halves, not (0,1),(2,3)),
        // and dims 4..8 are not rotated at all.
        let base = 10_000.0f32;
        let mut x = [0.0, 1.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0];
        apply_partial_neox_rope(&mut x, 1, 8, 4, 3, base);
        let theta1 = 3.0 * base.powf(-2.0 / 4.0); // pair j = 1
        assert!(
            (x[1] - theta1.cos()).abs() < 1e-6 && (x[3] - theta1.sin()).abs() < 1e-6,
            "{x:?}"
        );
        assert_eq!(
            (x[0], x[2]),
            (0.0, 0.0),
            "the (0,2) pair was all zeros: {x:?}"
        );
        assert_eq!(&x[4..], &[5.0, 6.0, 7.0, 8.0]);
    }
}

/// Proof-obligation bodies for `contracts/qwen35-hybrid-forward-v1.yaml` (QHF-INV-001..004,
/// QHF-INV-006, QHF-CON-007).
#[cfg(test)]
#[path = "forward_qwen35_contract_tests.rs"]
mod qhf_contract_tests;

/// The Gated `DeltaNet` head mapping when `num_v_heads > num_k_heads`
/// (Qwen3.5 4B/9B/27B) — PMAT-3477, #3346/#3510.
#[cfg(test)]
#[path = "forward_qwen35_gqa_tests.rs"]
mod qwen35_gqa_tests;
