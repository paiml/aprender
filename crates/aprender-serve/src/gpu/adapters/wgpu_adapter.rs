//! PMAT-333: WGPU adapter — dequantize OwnedQuantizedModel → WgslForwardPass weights.
//!
//! Converts Q4K/Q6K quantized weights to F32 and uploads to trueno's
//! WgslForwardPass for inference on AMD/Intel/Apple GPUs via Vulkan/Metal/WebGPU.

use crate::error::{RealizarError, Result};
use crate::gguf::OwnedQuantizedModel;
use crate::quantize::{dequantize_q4_k, dequantize_q5_k, dequantize_q6_k};

/// PMAT-333: Dequantize all model weights to F32 for WGPU upload.
///
/// Returns a map of weight name → (F32 data, rows, cols) ready for
/// `WgslForwardPass::upload_weight()`.
#[provable_contracts_macros::contract("wgpu-forward-pass-v1", equation = "dequant_correctness")]
pub fn dequant_model_weights(
    model: &OwnedQuantizedModel,
) -> Result<Vec<(String, Vec<f32>, usize, usize)>> {
    let config = &model.config;
    let hidden = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim();
    let intermediate = config.intermediate_dim;
    let num_layers = model.layers().len();

    let mut weights = Vec::new();

    eprintln!(
        "[PMAT-333] Dequantizing {} layers (hidden={}, heads={}/{}, intermediate={})",
        num_layers, hidden, num_heads, num_kv_heads, intermediate,
    );

    for (i, layer) in model.layers().iter().enumerate() {
        let prefix = format!("layer.{i}");

        // Norm weights (already F32)
        weights.push((
            format!("{prefix}.attn_norm"),
            layer.attn_norm_weight.clone(),
            1,
            hidden,
        ));

        if let Some(ref ffn_norm) = layer.ffn_norm_weight {
            weights.push((format!("{prefix}.ffn_norm"), ffn_norm.clone(), 1, hidden));
        }

        // QKV weights — dequantize from quantized format
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        match &layer.qkv_weight {
            crate::gguf::OwnedQKVWeights::Fused(tensor) => {
                let f32_data = dequant_tensor_public(tensor)?;
                let total_out = q_dim + 2 * kv_dim;
                // Split fused QKV into separate Q, K, V
                let q_data = f32_data[..q_dim * hidden].to_vec();
                let k_data = f32_data[q_dim * hidden..(q_dim + kv_dim) * hidden].to_vec();
                let v_data = f32_data[(q_dim + kv_dim) * hidden..total_out * hidden].to_vec();
                weights.push((format!("{prefix}.q_proj"), q_data, q_dim, hidden));
                weights.push((format!("{prefix}.k_proj"), k_data, kv_dim, hidden));
                weights.push((format!("{prefix}.v_proj"), v_data, kv_dim, hidden));
            },
            crate::gguf::OwnedQKVWeights::Separate { q, k, v } => {
                weights.push((
                    format!("{prefix}.q_proj"),
                    dequant_tensor_public(q)?,
                    q_dim,
                    hidden,
                ));
                weights.push((
                    format!("{prefix}.k_proj"),
                    dequant_tensor_public(k)?,
                    kv_dim,
                    hidden,
                ));
                weights.push((
                    format!("{prefix}.v_proj"),
                    dequant_tensor_public(v)?,
                    kv_dim,
                    hidden,
                ));
            },
        }

        // PMAT-342: QKV biases (required for Qwen2)
        if let Some(ref bias) = layer.qkv_bias {
            // Fused QKV bias: split into q_bias, k_bias, v_bias
            if bias.len() >= q_dim + 2 * kv_dim {
                weights.push((format!("{prefix}.q_bias"), bias[..q_dim].to_vec(), 1, q_dim));
                weights.push((
                    format!("{prefix}.k_bias"),
                    bias[q_dim..q_dim + kv_dim].to_vec(),
                    1,
                    kv_dim,
                ));
                weights.push((
                    format!("{prefix}.v_bias"),
                    bias[q_dim + kv_dim..q_dim + 2 * kv_dim].to_vec(),
                    1,
                    kv_dim,
                ));
            }
        }

        // O projection
        weights.push((
            format!("{prefix}.o_proj"),
            dequant_tensor_public(&layer.attn_output_weight)?,
            hidden,
            q_dim,
        ));

        // FFN weights
        if let Some(ref gate) = layer.ffn_gate_weight {
            weights.push((
                format!("{prefix}.gate_proj"),
                dequant_tensor_public(gate)?,
                intermediate,
                hidden,
            ));
        }
        weights.push((
            format!("{prefix}.up_proj"),
            dequant_tensor_public(&layer.ffn_up_weight)?,
            intermediate,
            hidden,
        ));
        weights.push((
            format!("{prefix}.down_proj"),
            dequant_tensor_public(&layer.ffn_down_weight)?,
            hidden,
            intermediate,
        ));

        if (i + 1) % 7 == 0 || i == num_layers - 1 {
            eprintln!("  Dequantized layer {}/{}", i + 1, num_layers);
        }
    }

    // LM head
    weights.push((
        "lm_head".to_string(),
        dequant_tensor_public(model.lm_head_weight())?,
        config.vocab_size,
        hidden,
    ));

    // PMAT-345: Weight layout analysis.
    // GGUF stores [ne0, ne1] with data layout data[i0 + i1*ne0].
    // For a weight W with GGUF dims [in_dim, out_dim]:
    //   data[in + out*in_dim] → this IS row-major [out_dim, in_dim]
    // The dequant_tensor produces data in this same order.
    // Our (rows=out_dim, cols=in_dim) labels match the data layout.
    // WGSL GEMV: w[row * K + col] = data[out * in_dim + in] ← CORRECT
    // NO TRANSPOSE NEEDED — GGUF layout is already row-major for [out, in].
    //
    // Previous transpose was WRONG — it double-transposed, causing garbled output.

    let total_bytes: usize = weights.iter().map(|(_, d, _, _)| d.len() * 4).sum();
    eprintln!(
        "[PMAT-333] Dequantized {} weights, {:.1} MB F32",
        weights.len(),
        total_bytes as f64 / 1e6,
    );

    Ok(weights)
}

/// PMAT-364: Extract raw Q4K weight bytes for fused dequant+GEMV on GPU.
/// Returns (name, raw_bytes, rows, cols) for Q4K tensors only. Other types skipped.
pub fn raw_q4k_weights(model: &OwnedQuantizedModel) -> Vec<(String, Vec<u8>, usize, usize)> {
    const GGUF_TYPE_Q4_K: u32 = 12;
    let config = &model.config;
    let hidden = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim();
    let intermediate = config.intermediate_dim;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let mut raw = Vec::new();

    for (i, layer) in model.layers().iter().enumerate() {
        let prefix = format!("layer.{i}");
        // Only extract Q4K projection weights (skip norms, biases)
        let projections: Vec<(&str, &crate::gguf::OwnedQuantizedTensor, usize, usize)> = vec![
            ("o_proj", &layer.attn_output_weight, hidden, q_dim),
            ("up_proj", &layer.ffn_up_weight, intermediate, hidden),
            ("down_proj", &layer.ffn_down_weight, hidden, intermediate),
        ];
        if let Some(ref gate) = layer.ffn_gate_weight {
            raw.push((
                format!("{prefix}.gate_proj"),
                gate.data.clone(),
                intermediate,
                hidden,
            ));
        }
        for (name, tensor, rows, cols) in projections {
            if tensor.qtype == GGUF_TYPE_Q4_K {
                raw.push((format!("{prefix}.{name}"), tensor.data.clone(), rows, cols));
            }
        }
        // QKV: handle separate weights
        if let crate::gguf::OwnedQKVWeights::Separate { q, k, v } = &layer.qkv_weight {
            if q.qtype == GGUF_TYPE_Q4_K {
                raw.push((format!("{prefix}.q_proj"), q.data.clone(), q_dim, hidden));
            }
            if k.qtype == GGUF_TYPE_Q4_K {
                raw.push((format!("{prefix}.k_proj"), k.data.clone(), kv_dim, hidden));
            }
            if v.qtype == GGUF_TYPE_Q4_K {
                raw.push((format!("{prefix}.v_proj"), v.data.clone(), kv_dim, hidden));
            }
        }
    }
    raw
}

/// Dequantize a single OwnedQuantizedTensor to F32
/// GH-560: Public per-tensor dequantization for streaming wgpu weight upload.
pub fn dequant_tensor_public(tensor: &crate::gguf::OwnedQuantizedTensor) -> Result<Vec<f32>> {
    const GGUF_TYPE_Q4_K: u32 = 12;
    const GGUF_TYPE_Q6_K: u32 = 14;
    const GGUF_TYPE_Q5_K: u32 = 13;
    const GGUF_TYPE_F32: u32 = 0;
    const GGUF_TYPE_F16: u32 = 1;

    match tensor.qtype {
        GGUF_TYPE_Q4_K => dequantize_q4_k(&tensor.data),
        GGUF_TYPE_Q6_K => dequantize_q6_k(&tensor.data),
        GGUF_TYPE_Q5_K => dequantize_q5_k(&tensor.data),
        GGUF_TYPE_F32 => Ok(tensor
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        GGUF_TYPE_F16 => Ok(tensor
            .data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect()),
        other => Err(RealizarError::FormatError {
            reason: format!("Unsupported quantization type {} for WGPU dequant", other),
        }),
    }
}
