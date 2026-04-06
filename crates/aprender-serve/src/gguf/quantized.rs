//! Quantized tensor types for GGUF models
//!
//! This module contains the fundamental quantized tensor types that form
//! the backbone of efficient LLM inference:
//!
//! - `QuantizedTensorRef`: Reference to quantized data in memory-mapped file
//! - `OwnedQuantizedTensor`: Owned copy of quantized data
//! - `QKVWeights`: Fused or separate QKV projection weights (borrowed)
//! - `OwnedQKVWeights`: Fused or separate QKV projection weights (owned)
//!
//! Per Wulf & McKee (1995) "Hitting the Memory Wall", memory bandwidth is the
//! bottleneck for LLM inference. These types support 8x bandwidth reduction
//! via Q4_K quantization.

// ============================================================================
// QuantizedTensorRef - Reference to quantized data in mmap
// ============================================================================

/// Reference to quantized tensor data in memory-mapped file
///
/// Per Wulf & McKee (1995) "Hitting the Memory Wall", memory bandwidth is the
/// bottleneck for LLM inference. By keeping weights in quantized form and
/// dequantizing inline during computation, we achieve 8x memory bandwidth
/// reduction for Q4_K format.
#[derive(Debug, Clone)]
pub struct QuantizedTensorRef {
    /// Byte offset in file where tensor data starts
    pub offset: usize,
    /// Size in bytes of the quantized data
    pub byte_size: usize,
    /// Number of elements after dequantization
    pub num_elements: usize,
    /// Quantization type (GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K, etc.)
    pub qtype: u32,
}

// ============================================================================
// QKVWeights - Borrowed QKV weight storage
// ============================================================================

/// QKV weight storage - supports both fused (phi-2) and separate (llama) formats
///
/// Five Whys Root Cause Fix: TinyLlama and other LLaMA-style models use separate
/// Q, K, V tensors while phi-2 style models use fused QKV. This enum supports both.
#[derive(Clone)]
pub enum QKVWeights {
    /// Fused QKV tensor (phi-2 style): single [hidden_dim, 3*hidden_dim] tensor
    Fused(QuantizedTensorRef),
    /// Separate Q, K, V tensors (llama style): three separate tensors
    Separate {
        /// Query projection [hidden_dim, hidden_dim]
        q: QuantizedTensorRef,
        /// Key projection [hidden_dim, kv_dim] (may differ for GQA)
        k: QuantizedTensorRef,
        /// Value projection [hidden_dim, kv_dim]
        v: QuantizedTensorRef,
    },
}

impl QKVWeights {
    /// Calculate the output dimension per position (q_dim + k_dim + v_dim)
    pub fn out_dim(&self, hidden_dim: usize) -> usize {
        match self {
            Self::Fused(ref weight) => weight.num_elements / hidden_dim,
            Self::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                let q_dim = q.num_elements / hidden_dim;
                let k_dim = k.num_elements / hidden_dim;
                let v_dim = v.num_elements / hidden_dim;
                q_dim + k_dim + v_dim
            },
        }
    }

    /// Get the Q dimension (query projection output dimension)
    pub fn q_dim(&self, hidden_dim: usize) -> usize {
        match self {
            Self::Fused(ref weight) => weight.num_elements / hidden_dim / 3,
            Self::Separate { ref q, .. } => q.num_elements / hidden_dim,
        }
    }
}

// ============================================================================
// OwnedQuantizedTensor - Owned copy of quantized data
// ============================================================================

/// Owned quantized tensor - copies data to avoid lifetime issues
///
/// IMP-100: This allows storing quantized models in AppState with 'static lifetime
#[derive(Debug, Clone)]
pub struct OwnedQuantizedTensor {
    /// Raw quantized data (owned copy)
    pub data: Vec<u8>,
    /// Input dimension
    pub in_dim: usize,
    /// Output dimension
    pub out_dim: usize,
    /// Quantization type
    pub qtype: u32,
}

impl OwnedQuantizedTensor {
    /// Create owned tensor from a tensor reference and data slice with explicit dimensions
    #[must_use]
    pub fn from_ref_with_dims(
        tensor_ref: &QuantizedTensorRef,
        data: &[u8],
        in_dim: usize,
        out_dim: usize,
    ) -> Self {
        let start = tensor_ref.offset;
        let end = start + tensor_ref.byte_size;
        let tensor_data = if end <= data.len() {
            data[start..end].to_vec()
        } else {
            Vec::new()
        };

        Self {
            data: tensor_data,
            in_dim,
            out_dim,
            qtype: tensor_ref.qtype,
        }
    }
}

// ============================================================================
// OwnedQKVWeights - Owned QKV weight storage
// ============================================================================

/// Owned QKV weight storage - supports both fused (phi-2) and separate (llama) formats
#[derive(Debug, Clone)]
pub enum OwnedQKVWeights {
    /// Fused QKV tensor (phi-2 style)
    Fused(OwnedQuantizedTensor),
    /// Separate Q, K, V tensors (llama style)
    Separate {
        /// Query projection weights
        q: OwnedQuantizedTensor,
        /// Key projection weights
        k: OwnedQuantizedTensor,
        /// Value projection weights
        v: OwnedQuantizedTensor,
    },
}

impl OwnedQKVWeights {
    /// Create from borrowed QKVWeights
    #[must_use]
    pub fn from_borrowed(qkv: &QKVWeights, data: &[u8], hidden_dim: usize) -> Self {
        match qkv {
            QKVWeights::Fused(ref tensor) => {
                let qkv_dim = 3 * hidden_dim;
                OwnedQKVWeights::Fused(OwnedQuantizedTensor::from_ref_with_dims(
                    tensor, data, hidden_dim, qkv_dim,
                ))
            },
            QKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                let q_dim = q.num_elements / hidden_dim;
                let k_dim = k.num_elements / hidden_dim;
                let v_dim = v.num_elements / hidden_dim;
                OwnedQKVWeights::Separate {
                    q: OwnedQuantizedTensor::from_ref_with_dims(q, data, hidden_dim, q_dim),
                    k: OwnedQuantizedTensor::from_ref_with_dims(k, data, hidden_dim, k_dim),
                    v: OwnedQuantizedTensor::from_ref_with_dims(v, data, hidden_dim, v_dim),
                }
            },
        }
    }

    /// Get the output dimension (total Q+K+V dim)
    #[must_use]
    pub fn out_dim(&self) -> usize {
        match self {
            OwnedQKVWeights::Fused(t) => t.out_dim,
            OwnedQKVWeights::Separate { q, k, v } => q.out_dim + k.out_dim + v.out_dim,
        }
    }

    /// Get the Q dimension (query projection output dimension)
    ///
    /// NOTE: For GQA models, use `q_dim_for_config` instead as this method
    /// assumes MHA (out_dim / 3) which is incorrect for GQA.
    #[must_use]
    pub fn q_dim(&self) -> usize {
        match self {
            OwnedQKVWeights::Fused(t) => t.out_dim / 3,
            OwnedQKVWeights::Separate { q, .. } => q.out_dim,
        }
    }

    /// Get the Q dimension for GQA-aware models
    ///
    /// GH-305: `head_dim` comes from GGUF metadata — may differ from `hidden_dim / num_heads`.
    /// For Qwen3-0.6B: `q_dim = 16 * 128 = 2048` while `hidden_dim = 1024`.
    #[must_use]
    pub fn q_dim_for_config(
        &self,
        num_heads: usize,
        _num_kv_heads: usize,
        _hidden_dim: usize,
        head_dim: usize,
    ) -> usize {
        match self {
            OwnedQKVWeights::Fused(_) => num_heads * head_dim,
            OwnedQKVWeights::Separate { q, .. } => q.out_dim,
        }
    }

    /// Get the K dimension for GQA-aware models
    ///
    /// For GQA: k_dim = num_kv_heads * head_dim (smaller than q_dim)
    #[must_use]
    pub fn k_dim_for_config(
        &self,
        _num_heads: usize,
        num_kv_heads: usize,
        _hidden_dim: usize,
        head_dim: usize,
    ) -> usize {
        match self {
            OwnedQKVWeights::Fused(_) => num_kv_heads * head_dim,
            OwnedQKVWeights::Separate { k, .. } => k.out_dim,
        }
    }

    /// Get the V dimension for GQA-aware models
    ///
    /// For GQA: v_dim = num_kv_heads * head_dim (same as k_dim)
    #[must_use]
    pub fn v_dim_for_config(
        &self,
        _num_heads: usize,
        num_kv_heads: usize,
        _hidden_dim: usize,
        head_dim: usize,
    ) -> usize {
        match self {
            OwnedQKVWeights::Fused(_) => num_kv_heads * head_dim,
            OwnedQKVWeights::Separate { v, .. } => v.out_dim,
        }
    }

    /// GH-129: Total size of owned weight data in bytes.
    #[must_use]
    pub fn data_bytes(&self) -> usize {
        match self {
            OwnedQKVWeights::Fused(t) => t.data.len(),
            OwnedQKVWeights::Separate { q, k, v } => q.data.len() + k.data.len() + v.data.len(),
        }
    }

    /// GH-129: Free owned weight data (replace with empty Vec).
    pub fn free_data(&mut self) {
        match self {
            OwnedQKVWeights::Fused(t) => t.data = Vec::new(),
            OwnedQKVWeights::Separate { q, k, v } => {
                q.data = Vec::new();
                k.data = Vec::new();
                v.data = Vec::new();
            },
        }
    }
}

// ============================================================================
// OwnedQuantizedLayer - Owned transformer layer weights
// ============================================================================

/// Owned quantized transformer layer - copies all weight data
///
/// IMP-100: Allows storing in Arc without lifetime parameters
#[derive(Debug, Clone)]
pub struct OwnedQuantizedLayer {
    /// Attention norm weight (f32, small)
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias (optional)
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weights (owned quantized data) - supports fused or separate
    pub qkv_weight: OwnedQKVWeights,
    /// QKV bias (optional, f32)
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection weights
    pub attn_output_weight: OwnedQuantizedTensor,
    /// Attention output bias (optional)
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN up projection weights
    pub ffn_up_weight: OwnedQuantizedTensor,
    /// FFN up bias (optional)
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection weights
    pub ffn_down_weight: OwnedQuantizedTensor,
    /// FFN down bias (optional)
    pub ffn_down_bias: Option<Vec<f32>>,
    /// FFN gate projection weights (SwiGLU models like LLaMA)
    pub ffn_gate_weight: Option<OwnedQuantizedTensor>,
    /// FFN gate bias (optional)
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN norm weight (pre-FFN layer norm, LLaMA-style)
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias (optional)
    pub ffn_norm_bias: Option<Vec<f32>>,
    /// GH-279: Per-head Q RMSNorm weight [head_dim] (Qwen3)
    pub attn_q_norm_weight: Option<Vec<f32>>,
    /// GH-279: Per-head K RMSNorm weight [head_dim] (Qwen3)
    pub attn_k_norm_weight: Option<Vec<f32>>,
}

impl OwnedQuantizedLayer {
    /// GH-129: Free projection weight data (keep norms, biases).
    /// After GPU preload, CPU copies are redundant on unified memory.
    pub fn free_projection_weights(&mut self) {
        self.qkv_weight.free_data();
        self.attn_output_weight.data = Vec::new();
        self.ffn_up_weight.data = Vec::new();
        self.ffn_down_weight.data = Vec::new();
        if let Some(ref mut gate) = self.ffn_gate_weight {
            gate.data = Vec::new();
        }
    }

    /// Convert from borrowed layer with data reference and model config
    #[must_use]
    pub fn from_borrowed(
        layer: &crate::gguf::QuantizedGGUFTransformerLayer,
        data: &[u8],
        config: &crate::gguf::GGUFConfig,
    ) -> Self {
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;

        Self {
            attn_norm_weight: layer.attn_norm_weight.clone(),
            attn_norm_bias: layer.attn_norm_bias.clone(),
            qkv_weight: OwnedQKVWeights::from_borrowed(&layer.qkv_weight, data, hidden_dim),
            qkv_bias: layer.qkv_bias.clone(),
            attn_output_weight: OwnedQuantizedTensor::from_ref_with_dims(
                &layer.attn_output_weight,
                data,
                // GH-307: Gemma-2 has q_dim (2048) != hidden_dim (2304) due to
                // non-standard head_dim. attn_output projects from q_dim to hidden_dim.
                config.q_dim(),
                hidden_dim,
            ),
            attn_output_bias: layer.attn_output_bias.clone(),
            ffn_up_weight: {
                // GH-306: When ffn_gate is absent AND the model uses SwiGLU (has_gate_ffn),
                // ffn_up is a fused gate_up tensor with out_dim = 2 * intermediate_dim.
                // GH-309: Models with GELU activation (Phi-2, GPT-2) also have no gate
                // weight but their ffn_up is NOT fused — it's just intermediate_dim.
                let is_fused_gate_up =
                    layer.ffn_gate_weight.is_none() && config.constraints.has_gate_ffn();
                let up_out_dim = if is_fused_gate_up {
                    intermediate_dim * 2
                } else {
                    intermediate_dim
                };
                OwnedQuantizedTensor::from_ref_with_dims(
                    &layer.ffn_up_weight,
                    data,
                    hidden_dim,
                    up_out_dim,
                )
            },
            ffn_up_bias: layer.ffn_up_bias.clone(),
            ffn_down_weight: OwnedQuantizedTensor::from_ref_with_dims(
                &layer.ffn_down_weight,
                data,
                intermediate_dim,
                hidden_dim,
            ),
            ffn_down_bias: layer.ffn_down_bias.clone(),
            ffn_gate_weight: layer.ffn_gate_weight.as_ref().map(|gate_ref| {
                OwnedQuantizedTensor::from_ref_with_dims(
                    gate_ref,
                    data,
                    hidden_dim,
                    intermediate_dim,
                )
            }),
            ffn_gate_bias: layer.ffn_gate_bias.clone(),
            ffn_norm_weight: layer.ffn_norm_weight.clone(),
            ffn_norm_bias: layer.ffn_norm_bias.clone(),
            attn_q_norm_weight: layer.attn_q_norm_weight.clone(),
            attn_k_norm_weight: layer.attn_k_norm_weight.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::types::GGUF_TYPE_Q4_K;

    #[test]
    fn test_quantized_tensor_ref() {
        let tensor = QuantizedTensorRef {
            offset: 1024,
            byte_size: 4096,
            num_elements: 8192,
            qtype: GGUF_TYPE_Q4_K,
        };

        assert_eq!(tensor.offset, 1024);
        assert_eq!(tensor.byte_size, 4096);
        assert_eq!(tensor.num_elements, 8192);
        assert_eq!(tensor.qtype, GGUF_TYPE_Q4_K);
    }

    #[test]
    fn test_qkv_weights_fused() {
        let tensor = QuantizedTensorRef {
            offset: 0,
            byte_size: 1024,
            num_elements: 4096 * 3, // 3 * hidden_dim
            qtype: GGUF_TYPE_Q4_K,
        };
        let qkv = QKVWeights::Fused(tensor);

        assert_eq!(qkv.out_dim(4096), 3); // 12288 / 4096 = 3
        assert_eq!(qkv.q_dim(4096), 1); // 3 / 3 = 1
    }

    #[test]
    fn test_qkv_weights_separate() {
        let q = QuantizedTensorRef {
            offset: 0,
            byte_size: 1024,
            num_elements: 4096 * 4096, // hidden_dim * hidden_dim
            qtype: GGUF_TYPE_Q4_K,
        };
        let k = QuantizedTensorRef {
            offset: 1024,
            byte_size: 256,
            num_elements: 4096 * 512, // hidden_dim * kv_dim
            qtype: GGUF_TYPE_Q4_K,
        };
        let v = QuantizedTensorRef {
            offset: 1280,
            byte_size: 256,
            num_elements: 4096 * 512,
            qtype: GGUF_TYPE_Q4_K,
        };

        let qkv = QKVWeights::Separate { q, k, v };

        assert_eq!(qkv.out_dim(4096), 4096 + 512 + 512);
        assert_eq!(qkv.q_dim(4096), 4096);
    }

    #[test]
    fn test_owned_quantized_tensor() {
        let tensor_ref = QuantizedTensorRef {
            offset: 0,
            byte_size: 8,
            num_elements: 16,
            qtype: GGUF_TYPE_Q4_K,
        };
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let owned = OwnedQuantizedTensor::from_ref_with_dims(&tensor_ref, &data, 4, 4);

        assert_eq!(owned.data, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(owned.in_dim, 4);
        assert_eq!(owned.out_dim, 4);
        assert_eq!(owned.qtype, GGUF_TYPE_Q4_K);
    }

    #[test]
    fn test_owned_qkv_weights() {
        let tensor = QuantizedTensorRef {
            offset: 0,
            byte_size: 12,
            num_elements: 12, // 4 * 3
            qtype: GGUF_TYPE_Q4_K,
        };
        let qkv_borrowed = QKVWeights::Fused(tensor);
        let data = vec![0u8; 20];

        let owned = OwnedQKVWeights::from_borrowed(&qkv_borrowed, &data, 4);

        assert_eq!(owned.out_dim(), 12); // 3 * 4
        assert_eq!(owned.q_dim(), 4); // 12 / 3
    }

    #[test]
    fn test_owned_quantized_tensor_bounds() {
        let tensor_ref = QuantizedTensorRef {
            offset: 100,
            byte_size: 50,
            num_elements: 100,
            qtype: GGUF_TYPE_Q4_K,
        };
        // Data too small - offset 100, needs 50 bytes
        let data = vec![0u8; 50];

        let owned = OwnedQuantizedTensor::from_ref_with_dims(&tensor_ref, &data, 10, 10);

        // Should return empty data when out of bounds
        assert!(owned.data.is_empty());
    }
}
