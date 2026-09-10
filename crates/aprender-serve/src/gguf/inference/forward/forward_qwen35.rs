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

/// Partial RoPE over rope_dimension_count dims honouring rope.dimension_sections.
/// Sections are lengths of [rope, zero, rope, zero] applied to the feature dimension.
pub fn apply_rope_sections(
    q: &mut [f32],
    k: &mut [f32],
    pos: u32,
    head_dim: usize,
    num_q_heads: usize,
    num_k_heads: usize,
    freq_base: f32,
    sections: &[usize; 4],
) {
    let mut compute_rope = |x: &mut [f32], num_heads: usize| {
        for h in 0..num_heads {
            let x_h = &mut x[h * head_dim..(h + 1) * head_dim];
            let mut offset = 0;

            // Section 0: RoPE
            let sec0 = sections[0];
            for i in (0..sec0).step_by(2) {
                let theta = (pos as f32) / freq_base.powf((i as f32) / (sec0 as f32));
                let cos = theta.cos();
                let sin = theta.sin();

                let idx0 = offset + i;
                let idx1 = offset + i + 1;
                let x0 = x_h[idx0];
                let x1 = x_h[idx1];
                x_h[idx0] = x0 * cos - x1 * sin;
                x_h[idx1] = x0 * sin + x1 * cos;
            }
            offset += sec0;

            // Section 1: Skip
            offset += sections[1];

            // Section 2: RoPE
            let sec2 = sections[2];
            for i in (0..sec2).step_by(2) {
                let theta = (pos as f32) / freq_base.powf((i as f32) / (sec2 as f32));
                let cos = theta.cos();
                let sin = theta.sin();

                let idx0 = offset + i;
                let idx1 = offset + i + 1;
                let x0 = x_h[idx0];
                let x1 = x_h[idx1];
                x_h[idx0] = x0 * cos - x1 * sin;
                x_h[idx1] = x0 * sin + x1 * cos;
            }
        }
    };

    compute_rope(q, num_q_heads);
    compute_rope(k, num_k_heads);
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

pub struct Qwen35State {
    pub conv_states: Vec<Vec<f32>>,
    pub ssm_states: Vec<Vec<f32>>,
    pub kv_cache: crate::gguf::OwnedQuantizedKVCache,
}

impl Qwen35State {
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        head_dim: usize,
        num_kv_heads: usize,
        num_v_heads: usize,
        head_v_dim: usize,
    ) -> Self {
        let conv_dim = head_v_dim * num_kv_heads * 2 + head_v_dim * num_v_heads;
        Self {
            conv_states: vec![vec![0.0; conv_dim * 3]; num_layers], // kernel_size = 4, so (4-1)*conv_dim = 3*conv_dim
            ssm_states: vec![vec![0.0; num_v_heads * head_v_dim * head_v_dim]; num_layers],
            kv_cache: crate::gguf::OwnedQuantizedKVCache::new(
                num_layers,
                num_kv_heads * head_dim,
                max_seq_len,
            ),
        }
    }
}

pub struct Qwen35OwnedDeltaNetLayer {
    pub attn_norm: Vec<f32>,
    pub attn_qkv: OwnedQuantizedTensor,
    pub attn_gate: OwnedQuantizedTensor,
    pub ssm_alpha: OwnedQuantizedTensor,
    pub ssm_beta: OwnedQuantizedTensor,
    pub ssm_a: Vec<f32>,
    pub ssm_dt_bias: Vec<f32>,
    pub ssm_conv1d_weight: Vec<f32>,
    pub ssm_norm_weight: Vec<f32>,
    pub ssm_out: OwnedQuantizedTensor,
    pub post_attention_norm: Vec<f32>,
    pub ffn_gate: OwnedQuantizedTensor,
    pub ffn_up: OwnedQuantizedTensor,
    pub ffn_down: OwnedQuantizedTensor,
}

pub struct Qwen35OwnedAttentionLayer {
    pub attn_norm: Vec<f32>,
    pub attn_q: OwnedQuantizedTensor,
    pub attn_k: OwnedQuantizedTensor,
    pub attn_v: OwnedQuantizedTensor,
    pub attn_q_norm: Vec<f32>,
    pub attn_k_norm: Vec<f32>,
    pub attn_output: OwnedQuantizedTensor,
    pub post_attention_norm: Vec<f32>,
    pub ffn_gate: OwnedQuantizedTensor,
    pub ffn_up: OwnedQuantizedTensor,
    pub ffn_down: OwnedQuantizedTensor,
}

pub enum Qwen35OwnedLayer {
    DeltaNet(Qwen35OwnedDeltaNetLayer),
    Attention(Qwen35OwnedAttentionLayer),
}

pub struct Qwen35Model<'a> {
    pub base: &'a OwnedQuantizedModel,
    pub layers: Vec<Qwen35OwnedLayer>,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub num_v_heads: usize,
    pub head_v_dim: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub conv_kernel: usize,
    pub rope_sections: [usize; 4],
}

fn load_f32_vec(tensor_ref: &QuantizedTensorRef, data: &[u8]) -> Vec<f32> {
    assert_eq!(
        tensor_ref.qtype,
        crate::gguf::types::GGUF_TYPE_F32,
        "Expected F32"
    );
    let bytes = &data[tensor_ref.offset..tensor_ref.offset + tensor_ref.byte_size];
    let (head, body, tail) = unsafe { bytes.align_to::<f32>() };
    assert!(head.is_empty() && tail.is_empty(), "Unaligned tensor data");
    body.to_vec()
}

impl<'a> Qwen35Model<'a> {
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
        for layer_ref in refs.into_iter() {
            match layer_ref {
                crate::gguf::qwen35_load::Qwen35Layer::DeltaNet(d) => {
                    owned.push(Qwen35OwnedLayer::DeltaNet(Qwen35OwnedDeltaNetLayer {
                        attn_norm: load_f32_vec(&d.attn_norm, data),
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
                        ssm_a: load_f32_vec(&d.ssm_a, data),
                        ssm_dt_bias: load_f32_vec(&d.ssm_dt_bias, data),
                        ssm_conv1d_weight: load_f32_vec(&d.ssm_conv1d_weight, data),
                        ssm_norm_weight: load_f32_vec(&d.ssm_norm_weight, data),
                        ssm_out: OwnedQuantizedTensor::from_ref_with_dims(
                            &d.ssm_out, data, value_dim, hidden_dim,
                        ),
                        post_attention_norm: load_f32_vec(&d.post_attention_norm, data),
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
                    owned.push(Qwen35OwnedLayer::Attention(Qwen35OwnedAttentionLayer {
                        attn_norm: load_f32_vec(&a.attn_norm, data),
                        attn_q: OwnedQuantizedTensor::from_ref_with_dims(
                            &a.attn_q,
                            data,
                            hidden_dim,
                            num_heads * head_dim,
                        ),
                        attn_k: OwnedQuantizedTensor::from_ref_with_dims(
                            &a.attn_k,
                            data,
                            hidden_dim,
                            num_kv_heads * head_dim,
                        ),
                        attn_v: OwnedQuantizedTensor::from_ref_with_dims(
                            &a.attn_v,
                            data,
                            hidden_dim,
                            num_kv_heads * head_dim,
                        ),
                        attn_q_norm: load_f32_vec(&a.attn_q_norm, data),
                        attn_k_norm: load_f32_vec(&a.attn_k_norm, data),
                        attn_output: OwnedQuantizedTensor::from_ref_with_dims(
                            &a.attn_output,
                            data,
                            num_heads * head_dim,
                            hidden_dim,
                        ),
                        post_attention_norm: load_f32_vec(&a.post_attention_norm, data),
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

        Ok(Self {
            base,
            layers: owned,
            head_dim,
            num_kv_heads,
            num_v_heads,
            head_v_dim,
            num_k_heads,
            head_k_dim,
            conv_kernel,
            rope_sections,
        })
    }

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

        let k_dim = self.head_k_dim * self.num_k_heads;
        let v_dim = self.head_v_dim * self.num_v_heads;
        let mut q = conv_out[0..k_dim].to_vec();
        let mut k = conv_out[k_dim..k_dim * 2].to_vec();
        let mut v = conv_out[k_dim * 2..conv_dim].to_vec();

        apply_rope_sections(
            &mut q,
            &mut k,
            position as u32,
            self.head_k_dim,
            self.num_k_heads,
            self.num_k_heads,
            self.base.config.rope_theta,
            &self.rope_sections,
        );

        let mut dt_raw = vec![0.0; self.num_v_heads];
        self.base
            .fused_matmul_into(normed, &d.ssm_alpha, &mut dt_raw)?;
        let mut dt = vec![0.0; self.num_v_heads];
        for (i, val) in dt_raw.iter().enumerate() {
            dt[i] = fn_softplus(val + d.ssm_dt_bias[i]);
        }

        let mut beta = vec![0.0; self.num_v_heads];
        self.base
            .fused_matmul_into(normed, &d.ssm_beta, &mut beta)?;

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

        let mut q = vec![0.0; a.attn_q.out_dim];
        let mut k = vec![0.0; a.attn_k.out_dim];
        let mut v = vec![0.0; a.attn_v.out_dim];
        self.base.fused_matmul_into(normed, &a.attn_q, &mut q)?;
        self.base.fused_matmul_into(normed, &a.attn_k, &mut k)?;
        self.base.fused_matmul_into(normed, &a.attn_v, &mut v)?;

        crate::gguf::ops::apply_per_head_rms_norm(
            &mut q,
            &a.attn_q_norm,
            self.base.config.num_heads,
            self.base.config.eps,
        );
        crate::gguf::ops::apply_per_head_rms_norm(
            &mut k,
            &a.attn_k_norm,
            self.base.config.num_kv_heads,
            self.base.config.eps,
        );

        self.base
            .apply_rope(&mut q, position, self.base.config.num_heads);
        self.base
            .apply_rope(&mut k, position, self.base.config.num_kv_heads);

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
            for p in 0..position + 1 {
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
            for p in 0..position + 1 {
                let w = scores[p];
                let v_p = &v_cache[p * (num_kv_heads * head_dim) + kv_h * head_dim
                    ..p * (num_kv_heads * head_dim) + (kv_h + 1) * head_dim];
                for i in 0..head_dim {
                    out_h[i] += w * v_p[i];
                }
            }
        }
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
