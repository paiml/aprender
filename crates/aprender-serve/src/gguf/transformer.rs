//! Quantized GGUF transformer types
//!
//! This module contains the quantized transformer layer and model structures
//! that enable fused dequantization operations for memory-efficient inference.

use crate::error::{RealizarError, Result};
use crate::quantize::QK_K;

use super::config::{GGUFConfig, ValidatedModelConfig};
use super::quantized::{QKVWeights, QuantizedTensorRef};
use super::types::{
    GGUFModel, GGUF_TYPE_F32, GGUF_TYPE_Q2_K, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K,
    GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
};

/// Quantized transformer layer weights (stored as byte references)
///
/// Unlike `GGUFTransformerLayer` which stores dequantized Vec<f32>,
/// this stores references to quantized data for fused operations.
pub struct QuantizedGGUFTransformerLayer {
    /// Attention norm weight (kept as f32 - small, read once per token)
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias (optional)
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weights (quantized) - supports fused or separate
    pub qkv_weight: QKVWeights,
    /// QKV bias (optional, f32)
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection (quantized)
    pub attn_output_weight: QuantizedTensorRef,
    /// Attention output bias (optional, f32)
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN up projection (quantized)
    pub ffn_up_weight: QuantizedTensorRef,
    /// FFN up bias (optional, f32)
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection (quantized)
    pub ffn_down_weight: QuantizedTensorRef,
    /// FFN down bias (optional, f32)
    pub ffn_down_bias: Option<Vec<f32>>,
    /// FFN gate projection (quantized, SwiGLU models like LLaMA)
    pub ffn_gate_weight: Option<QuantizedTensorRef>,
    /// FFN gate bias (optional, f32)
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN norm weight (pre-FFN layer norm, LLaMA-style)
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias (optional, f32)
    pub ffn_norm_bias: Option<Vec<f32>>,
    /// GH-279: Per-head Q RMSNorm weight [head_dim] (Qwen3)
    pub attn_q_norm_weight: Option<Vec<f32>>,
    /// GH-279: Per-head K RMSNorm weight [head_dim] (Qwen3)
    pub attn_k_norm_weight: Option<Vec<f32>>,
}

/// Quantized GGUF Transformer for fused inference
///
/// Per Williams et al. (2009) roofline model, LLM inference is memory-bound.
/// This transformer stores weights in quantized form and uses fused
/// dequant+dot operations to minimize memory bandwidth.
///
/// # Performance Benefits
///
/// - **8x bandwidth reduction** for Q4_K vs f32 (144 bytes vs 1024 bytes per 256 values)
/// - **Zero intermediate buffers** - dequantization happens inline with dot product
/// - **SIMD acceleration** - AVX2/FMA fused operations when available
/// - **Zero-copy loading** - weights stay in memory-mapped file
///
/// # Architecture
///
/// ```text
/// [Memory-mapped Q4_K bytes] → [fused_q4k_dot_simd] → [f32 result]
///                               ↑
///                         No intermediate Vec<f32>!
/// ```
pub struct QuantizedGGUFTransformer<'a> {
    /// Model configuration
    pub config: GGUFConfig,
    /// Reference to memory-mapped file data
    pub data: &'a [u8],
    /// Token embedding (kept as f32 for lookup)
    pub token_embedding: Vec<f32>,
    /// GH-278: Position embedding [context_length, hidden_dim] (GPT-2 only)
    pub position_embedding: Option<Vec<f32>>,
    /// Quantized layer weights
    pub layers: Vec<QuantizedGGUFTransformerLayer>,
    /// Output norm weight (f32)
    pub output_norm_weight: Vec<f32>,
    /// Output norm bias (optional)
    pub output_norm_bias: Option<Vec<f32>>,
    /// LM head weight (quantized for large vocab)
    pub lm_head_weight: QuantizedTensorRef,
    /// LM head bias (optional, f32)
    pub lm_head_bias: Option<Vec<f32>>,
}

impl<'a> QuantizedGGUFTransformer<'a> {
    /// Load quantized transformer from memory-mapped GGUF model
    ///
    /// # Arguments
    ///
    /// * `model` - Parsed GGUF model metadata
    /// * `data` - Memory-mapped file data (zero-copy)
    ///
    /// # Errors
    ///
    /// Returns error if required tensors are missing or have unsupported format
    pub fn from_gguf(model: &GGUFModel, data: &'a [u8]) -> Result<Self> {
        // Phase 2: Validate config at construction boundary.
        let config = ValidatedModelConfig::from_gguf(model)?.into_inner();

        // Token embedding - keep as f32 for efficient lookup
        let token_embedding = model.get_tensor_f32("token_embd.weight", data)?;
        // GH-278: Position embedding — standard GGUF + legacy + aprender export fallback
        let position_embedding = model
            .get_tensor_f32("position_embd.weight", data)
            .or_else(|_| model.get_tensor_f32("token_pos_embd.weight", data))
            .or_else(|_| model.get_tensor_f32("model.position_embedding.weight", data))
            .ok();

        // Load layers with quantized weight references
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer = Self::load_quantized_layer(model, data, layer_idx)?;
            layers.push(layer);
        }

        // Output norm - small, keep as f32
        let output_norm_weight = model.get_tensor_f32("output_norm.weight", data)?;
        // GH-278: Output norm bias — standard + aprender fallback
        let output_norm_bias = model
            .get_tensor_f32("output_norm.bias", data)
            .or_else(|_| model.get_tensor_f32("model.norm.bias", data))
            .ok();

        // LM head - large, keep quantized
        // Fall back to token_embd.weight for tied embeddings (Qwen2, some LLaMA variants)
        let lm_head_weight = Self::get_tensor_ref(model, data, "output.weight")
            .or_else(|_| Self::get_tensor_ref(model, data, "token_embd.weight"))?;
        let lm_head_bias = model.get_tensor_f32("output.bias", data).ok();

        Ok(Self {
            config,
            data,
            token_embedding,
            position_embedding,
            layers,
            output_norm_weight,
            output_norm_bias,
            lm_head_weight,
            lm_head_bias,
        })
    }

    /// Calculate byte size for a quantized tensor based on its type and dimensions.
    fn tensor_byte_size(qtype: u32, num_elements: usize, dims: &[u64]) -> Result<usize> {
        /// Row-padded K-quant byte size: each row pads to super-block boundaries.
        fn k_quant_bytes(dims: &[u64], super_block_bytes: usize) -> usize {
            if dims.len() == 2 {
                let rows = dims[0] as usize;
                let cols = dims[1] as usize;
                rows * cols.div_ceil(QK_K) * super_block_bytes
            } else {
                let n: usize = dims.iter().map(|&d| d as usize).product();
                n.div_ceil(QK_K) * super_block_bytes
            }
        }

        match qtype {
            GGUF_TYPE_F32 => Ok(num_elements * 4),
            GGUF_TYPE_Q4_0 => Ok(num_elements.div_ceil(32) * 18),
            GGUF_TYPE_Q8_0 => Ok(num_elements.div_ceil(32) * 34),
            GGUF_TYPE_Q2_K => Ok(num_elements.div_ceil(QK_K) * 84),
            GGUF_TYPE_Q4_1 => Ok(num_elements.div_ceil(32) * 20),
            GGUF_TYPE_Q5_0 => Ok(num_elements.div_ceil(32) * 22),
            GGUF_TYPE_Q4_K => Ok(k_quant_bytes(dims, 144)),
            GGUF_TYPE_Q5_K => Ok(k_quant_bytes(dims, 176)),
            GGUF_TYPE_Q6_K => Ok(k_quant_bytes(dims, 210)),
            _ => Err(RealizarError::UnsupportedOperation {
                operation: "tensor_byte_size".to_string(),
                reason: format!("Unsupported quantization type: {qtype}"),
            }),
        }
    }

    /// PAR-058: Auto-correct qtype when header claims wrong type.
    fn resolve_qtype(
        name: &str,
        claimed_qtype: u32,
        byte_size: usize,
        num_elements: usize,
        offset: usize,
        data_len: usize,
    ) -> (usize, u32) {
        if offset + byte_size <= data_len {
            return (byte_size, claimed_qtype);
        }
        let avail = data_len.saturating_sub(offset);
        let q4_0_size = num_elements.div_ceil(32) * 18;
        if q4_0_size <= avail && q4_0_size > 0 {
            eprintln!(
                "[PAR-058-RESOLVED] Tensor '{name}' qtype mismatch: header says {claimed_qtype} but byte size suggests Q4_0. Using Q4_0."
            );
            return (q4_0_size, GGUF_TYPE_Q4_0);
        }
        let q8_0_size = num_elements.div_ceil(32) * 34;
        if q8_0_size <= avail && q8_0_size > 0 {
            eprintln!(
                "[PAR-058-RESOLVED] Tensor '{name}' qtype mismatch: header says {claimed_qtype} but byte size suggests Q8_0. Using Q8_0."
            );
            return (q8_0_size, GGUF_TYPE_Q8_0);
        }
        (byte_size, claimed_qtype)
    }

    /// Get tensor reference (offset + size + qtype) without dequantization
    fn get_tensor_ref(model: &GGUFModel, data: &[u8], name: &str) -> Result<QuantizedTensorRef> {
        let tensor = model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: format!("Tensor '{}' not found", name),
            })?;

        let num_elements: usize = tensor.dims.iter().map(|&d| d as usize).product();
        let offset = model.tensor_data_start + tensor.offset as usize;
        let byte_size = Self::tensor_byte_size(tensor.qtype, num_elements, &tensor.dims)?;
        let (byte_size, actual_qtype) = Self::resolve_qtype(
            name,
            tensor.qtype,
            byte_size,
            num_elements,
            offset,
            data.len(),
        );

        if offset + byte_size > data.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Tensor '{}' data range [{}, {}) exceeds file size {}",
                    name,
                    offset,
                    offset + byte_size,
                    data.len()
                ),
            });
        }

        Ok(QuantizedTensorRef {
            offset,
            byte_size,
            num_elements,
            qtype: actual_qtype,
        })
    }

    /// Load a single quantized transformer layer
    fn load_quantized_layer(
        model: &GGUFModel,
        data: &[u8],
        layer_idx: usize,
    ) -> Result<QuantizedGGUFTransformerLayer> {
        let prefix = format!("blk.{}", layer_idx);

        // Attention norm - small, keep as f32
        let attn_norm_weight =
            model.get_tensor_f32(&format!("{}.attn_norm.weight", prefix), data)?;
        // GH-278: Attention norm bias — standard GGUF + aprender fallback
        let attn_norm_bias = model
            .get_tensor_f32(&format!("{}.attn_norm.bias", prefix), data)
            .or_else(|_| model.get_tensor_f32(&format!("{}.input_layernorm.bias", prefix), data))
            .ok();

        // QKV - large, keep quantized
        // Try fused first (phi-2 style), fall back to separate (llama style)
        let (qkv_weight, qkv_bias) = if let Ok(fused) =
            Self::get_tensor_ref(model, data, &format!("{}.attn_qkv.weight", prefix))
        {
            // phi-2 style: fused QKV tensor
            let bias = model
                .get_tensor_f32(&format!("{}.attn_qkv.bias", prefix), data)
                .ok();
            (QKVWeights::Fused(fused), bias)
        } else {
            // llama style: separate Q, K, V tensors
            let q = Self::get_tensor_ref(model, data, &format!("{}.attn_q.weight", prefix))?;
            let k = Self::get_tensor_ref(model, data, &format!("{}.attn_k.weight", prefix))?;
            let v = Self::get_tensor_ref(model, data, &format!("{}.attn_v.weight", prefix))?;

            // Try to get biases (llama usually doesn't have them)
            let q_bias = model
                .get_tensor_f32(&format!("{}.attn_q.bias", prefix), data)
                .ok();
            let k_bias = model
                .get_tensor_f32(&format!("{}.attn_k.bias", prefix), data)
                .ok();
            let v_bias = model
                .get_tensor_f32(&format!("{}.attn_v.bias", prefix), data)
                .ok();

            let bias = match (q_bias, k_bias, v_bias) {
                (Some(qb), Some(kb), Some(vb)) => {
                    let mut combined = Vec::with_capacity(qb.len() + kb.len() + vb.len());
                    combined.extend_from_slice(&qb);
                    combined.extend_from_slice(&kb);
                    combined.extend_from_slice(&vb);
                    Some(combined)
                },
                _ => None,
            };

            (QKVWeights::Separate { q, k, v }, bias)
        };

        // Attention output - large, keep quantized
        let attn_output_weight =
            Self::get_tensor_ref(model, data, &format!("{}.attn_output.weight", prefix))?;
        let attn_output_bias = model
            .get_tensor_f32(&format!("{}.attn_output.bias", prefix), data)
            .ok();

        // FFN - large, keep quantized
        let ffn_up_weight =
            Self::get_tensor_ref(model, data, &format!("{}.ffn_up.weight", prefix))?;
        // GH-278: FFN biases — standard GGUF + aprender fallback
        let ffn_up_bias = model
            .get_tensor_f32(&format!("{}.ffn_up.bias", prefix), data)
            .or_else(|_| model.get_tensor_f32(&format!("{}.mlp.up_proj.bias", prefix), data))
            .ok();
        let ffn_down_weight =
            Self::get_tensor_ref(model, data, &format!("{}.ffn_down.weight", prefix))?;
        let ffn_down_bias = model
            .get_tensor_f32(&format!("{}.ffn_down.bias", prefix), data)
            .or_else(|_| model.get_tensor_f32(&format!("{}.mlp.down_proj.bias", prefix), data))
            .ok();

        // FFN gate - SwiGLU models like LLaMA have this
        let ffn_gate_weight =
            Self::get_tensor_ref(model, data, &format!("{}.ffn_gate.weight", prefix)).ok();
        let ffn_gate_bias = model
            .get_tensor_f32(&format!("{}.ffn_gate.bias", prefix), data)
            .ok();

        // FFN norm - LLaMA-style pre-FFN layer norm
        let ffn_norm_weight = model
            .get_tensor_f32(&format!("{}.ffn_norm.weight", prefix), data)
            .ok();
        // GH-278: FFN norm bias — standard GGUF + aprender fallback
        let ffn_norm_bias = model
            .get_tensor_f32(&format!("{}.ffn_norm.bias", prefix), data)
            .or_else(|_| {
                model.get_tensor_f32(&format!("{}.post_attention_layernorm.bias", prefix), data)
            })
            .ok();

        // GH-279: QK norm weights (Qwen3 per-head RMSNorm on Q and K)
        let attn_q_norm_weight = model
            .get_tensor_f32(&format!("{}.attn_q_norm.weight", prefix), data)
            .ok();
        let attn_k_norm_weight = model
            .get_tensor_f32(&format!("{}.attn_k_norm.weight", prefix), data)
            .ok();

        Ok(QuantizedGGUFTransformerLayer {
            attn_norm_weight,
            attn_norm_bias,
            qkv_weight,
            qkv_bias,
            attn_output_weight,
            attn_output_bias,
            ffn_up_weight,
            ffn_up_bias,
            ffn_down_weight,
            ffn_down_bias,
            ffn_gate_weight,
            ffn_gate_bias,
            ffn_norm_weight,
            ffn_norm_bias,
            attn_q_norm_weight,
            attn_k_norm_weight,
        })
    }
}

include!("transformer_quantized_layer_field.rs");
