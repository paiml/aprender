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

/// The gated delta-rule recurrence for a single token exactly as delta-net-base.cpp computes it.
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
    assert_eq!(q.len(), num_v_heads * head_v_dim);
    assert_eq!(k.len(), num_v_heads * head_v_dim);
    assert_eq!(v.len(), num_v_heads * head_v_dim);
    assert_eq!(beta.len(), num_v_heads);
    assert_eq!(gate.len(), num_v_heads);
    assert_eq!(state.len(), num_v_heads * head_v_dim * head_v_dim);
    assert_eq!(output.len(), num_v_heads * head_v_dim);

    let scale = 1.0 / (head_v_dim as f32).sqrt();

    for h in 0..num_v_heads {
        let q_h = &q[h * head_v_dim..(h + 1) * head_v_dim];
        let k_h = &k[h * head_v_dim..(h + 1) * head_v_dim];
        let v_h = &v[h * head_v_dim..(h + 1) * head_v_dim];
        let beta_val = beta[h];
        let gate_val = gate[h];

        let state_offset = h * head_v_dim * head_v_dim;
        let s_h = &mut state[state_offset..state_offset + head_v_dim * head_v_dim];

        // 1. S_h *= exp(gate_val)
        let exp_gate = gate_val.exp();
        for s in s_h.iter_mut() {
            *s *= exp_gate;
        }

        // 2. delta = (v_h - S_h^T * k_h) * beta_val
        // Note: s_h[j * head_v_dim + i] is S[i][j].
        // So row j of s_h in memory is column j of S.
        // sum = dot(row j of s_h, k_h)
        let mut delta = vec![0.0; head_v_dim];
        for j in 0..head_v_dim {
            let row_j = &s_h[j * head_v_dim..(j + 1) * head_v_dim];
            let mut sum = 0.0;
            for i in 0..head_v_dim {
                sum += row_j[i] * k_h[i];
            }
            delta[j] = (v_h[j] - sum) * beta_val;
        }

        // 3. S_h += k_h * delta^T
        for j in 0..head_v_dim {
            let row_j = &mut s_h[j * head_v_dim..(j + 1) * head_v_dim];
            let d_j = delta[j];
            for i in 0..head_v_dim {
                row_j[i] += k_h[i] * d_j;
            }
        }

        // 4. out_h = S_h^T * q_h * scale
        for j in 0..head_v_dim {
            let row_j = &s_h[j * head_v_dim..(j + 1) * head_v_dim];
            let mut sum = 0.0;
            for i in 0..head_v_dim {
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
            ssm_states: vec![vec![0.0; num_v_heads * head_value_dim * head_value_dim]; num_layers],
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

        let get_u32 = |key: &str| -> Result<usize> {
            match model.metadata.get(key) {
                Some(crate::gguf::types::GGUFValue::UInt32(v)) => Ok(*v as usize),
                Some(crate::gguf::types::GGUFValue::Int32(v)) => Ok(*v as usize),
                _ => Err(crate::RealizarError::InvalidShape {
                    reason: format!("Missing or invalid {}", key),
                }),
            }
        };

        let head_k_dim = get_u32("qwen2.ssm.state_size")
            .or_else(|_| get_u32("qwen35.ssm.state_size"))
            .unwrap_or(16);
        let head_v_dim = head_k_dim;
        let num_k_heads = get_u32("qwen2.ssm.group_count")
            .or_else(|_| get_u32("qwen35.ssm.group_count"))
            .unwrap_or(1);
        let num_v_heads = get_u32("qwen2.ssm.time_step_rank")
            .or_else(|_| get_u32("qwen35.ssm.time_step_rank"))
            .unwrap_or(16);
        let conv_kernel = get_u32("qwen2.ssm.conv_kernel")
            .or_else(|_| get_u32("qwen35.ssm.conv_kernel"))
            .unwrap_or(4);
        let mut rope_sections = [16, 24, 24, 0];
        if let Some(crate::gguf::types::GGUFValue::Array(arr)) = model
            .metadata
            .get("qwen2.rope.dimension_sections")
            .or_else(|| model.metadata.get("qwen35.rope.dimension_sections"))
        {
            for (i, val) in arr.iter().take(4).enumerate() {
                if let crate::gguf::types::GGUFValue::UInt32(v) = val {
                    rope_sections[i] = *v as usize;
                } else if let crate::gguf::types::GGUFValue::Int32(v) = val {
                    rope_sections[i] = *v as usize;
                }
            }
        }

        let hidden_dim = base.config.hidden_dim;
        let intermediate_dim = base.config.intermediate_dim;
        let num_heads = base.config.num_heads;
        let num_kv_heads = base.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;

        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;

        let mut owned = Vec::with_capacity(refs.len());
        for layer_ref in refs {
            match layer_ref {
                crate::gguf::qwen35_load::Qwen35Layer::DeltaNet(d) => {
                    owned.push(Qwen35OwnedLayer::DeltaNet(Qwen35OwnedDeltaNetLayer {
                        attn_norm: load_f32_vec(&d.attn_norm, data)?,
                        attn_qkv: OwnedQuantizedTensor::from_ref_with_dims(
                            &d.attn_qkv,
                            data,
                            hidden_dim,
                            conv_dim,
                        ),
                        attn_gate: OwnedQuantizedTensor::from_ref_with_dims(
                            &d.attn_gate,
                            data,
                            hidden_dim,
                            value_dim,
                        ),
                        ssm_alpha: OwnedQuantizedTensor::from_ref_with_dims(
                            &d.ssm_alpha,
                            data,
                            hidden_dim,
                            num_v_heads,
                        ),
                        ssm_beta: OwnedQuantizedTensor::from_ref_with_dims(
                            &d.ssm_beta,
                            data,
                            hidden_dim,
                            num_v_heads,
                        ),
                        ssm_a: load_f32_vec(&d.ssm_a, data)?,
                        ssm_dt_bias: load_f32_vec(&d.ssm_dt_bias, data)?,
                        ssm_conv1d_weight: load_f32_vec(&d.ssm_conv1d_weight, data)?,
                        ssm_norm_weight: load_f32_vec(&d.ssm_norm_weight, data)?,
                        ssm_out: OwnedQuantizedTensor::from_ref_with_dims(
                            &d.ssm_out, data, value_dim, hidden_dim,
                        ),
                        post_attention_norm: load_f32_vec(&d.post_attention_norm, data)?,
                        ffn_gate: OwnedQuantizedTensor::from_ref_with_dims(
                            &d.ffn_gate,
                            data,
                            hidden_dim,
                            intermediate_dim,
                        ),
                        ffn_up: OwnedQuantizedTensor::from_ref_with_dims(
                            &d.ffn_up,
                            data,
                            hidden_dim,
                            intermediate_dim,
                        ),
                        ffn_down: OwnedQuantizedTensor::from_ref_with_dims(
                            &d.ffn_down,
                            data,
                            intermediate_dim,
                            hidden_dim,
                        ),
                    }));
                },
                crate::gguf::qwen35_load::Qwen35Layer::Attention(a) => {
                    let attn_q_norm = load_f32_vec(&a.attn_q_norm, data)?;
                    let true_head_dim = attn_q_norm.len(); // 256 for Qwen3.5 standard attention

                    owned.push(Qwen35OwnedLayer::Attention(Qwen35OwnedAttentionLayer {
                        attn_norm: load_f32_vec(&a.attn_norm, data)?,
                        attn_q: OwnedQuantizedTensor::from_ref_with_dims(
                            &a.attn_q,
                            data,
                            hidden_dim,
                            num_heads * true_head_dim * 2,
                        ),
                        attn_k: OwnedQuantizedTensor::from_ref_with_dims(
                            &a.attn_k,
                            data,
                            hidden_dim,
                            num_kv_heads * true_head_dim,
                        ),
                        attn_v: OwnedQuantizedTensor::from_ref_with_dims(
                            &a.attn_v,
                            data,
                            hidden_dim,
                            num_kv_heads * true_head_dim,
                        ),
                        attn_q_norm,
                        attn_k_norm: load_f32_vec(&a.attn_k_norm, data)?,
                        attn_output: OwnedQuantizedTensor::from_ref_with_dims(
                            &a.attn_output,
                            data,
                            num_heads * true_head_dim,
                            hidden_dim,
                        ),
                        post_attention_norm: load_f32_vec(&a.post_attention_norm, data)?,
                        ffn_gate: OwnedQuantizedTensor::from_ref_with_dims(
                            &a.ffn_gate,
                            data,
                            hidden_dim,
                            intermediate_dim,
                        ),
                        ffn_up: OwnedQuantizedTensor::from_ref_with_dims(
                            &a.ffn_up,
                            data,
                            hidden_dim,
                            intermediate_dim,
                        ),
                        ffn_down: OwnedQuantizedTensor::from_ref_with_dims(
                            &a.ffn_down,
                            data,
                            intermediate_dim,
                            hidden_dim,
                        ),
                    }));
                },
            }
        }

        // Attention head width is key_length (= attn_q_norm width, 256), NOT hidden/heads (128):
        // the KV cache and the attention loop used 128-wide heads over 256-wide q/k/v.
        let attn_head_dim = owned
            .iter()
            .find_map(|l| match l {
                Qwen35OwnedLayer::Attention(a) => Some(a.attn_q_norm.len()),
                Qwen35OwnedLayer::DeltaNet(_) => None,
            })
            .unwrap_or(head_dim);
        Ok(Self {
            base,
            layers: owned,
            head_dim: attn_head_dim,
            num_kv_heads,
            num_v_heads,
            head_v_dim,
            num_k_heads,
            head_k_dim,
            conv_kernel,
            rope_sections,
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

    fn forward_deltanet(
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

        let mut out_h = vec![0.0; v_dim];
        delta_rule_recurrence(
            &q,
            &k,
            &v,
            &beta,
            &dt,
            &mut cache.ssm_states[il][..],
            &mut out_h,
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

    fn forward_attention(
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
