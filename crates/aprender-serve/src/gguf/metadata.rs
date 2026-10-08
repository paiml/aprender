impl GGUFModel {

    /// Extract tensor data by name with dequantization
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to extract
    /// * `file_data` - Complete GGUF file bytes
    ///
    /// # Returns
    ///
    /// Dequantized f32 tensor data
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tensor not found
    /// - Unsupported quantization type
    /// - Invalid data at offset
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let file_data = std::fs::read("model.gguf")?;
    /// let model = GGUFModel::from_bytes(&file_data)?;
    /// let weights = model.get_tensor_f32("layer.0.weight", &file_data)?;
    /// ```
    pub fn get_tensor_f32(&self, name: &str, file_data: &[u8]) -> Result<Vec<f32>> {
        // Find tensor info
        let tensor = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        // Calculate tensor size in elements
        let size: usize = tensor
            .dims
            .iter()
            .try_fold(1usize, |acc, &dim| {
                usize::try_from(dim).ok().and_then(|d| acc.checked_mul(d))
            })
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: format!("Tensor dimensions overflow: {:?}", tensor.dims),
            })?;

        // Convert tensor offset to usize and add tensor data start
        let tensor_offset =
            usize::try_from(tensor.offset).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_offset".to_string(),
                reason: format!("Offset {} exceeds platform usize limit", tensor.offset),
            })?;
        let offset = self.tensor_data_start + tensor_offset;

        // Extract and dequantize based on qtype. A block type reads whole blocks and trims the
        // padding the last block carries; every type checks its byte range against the file first.
        use crate::quantize::{
            dequantize_f16, dequantize_q2_k, dequantize_q3_k, dequantize_q4_0, dequantize_q4_1,
            dequantize_q4_k_simd, dequantize_q5_0, dequantize_q5_1, dequantize_q5_k,
            dequantize_q6_k, dequantize_q8_0_simd, QK_K,
        };
        type Dequant = fn(&[u8]) -> Result<Vec<f32>>;
        // (elements per block, bytes per block, dequantizer)
        let blocked: Option<(usize, usize, Dequant)> = match tensor.qtype {
            GGUF_TYPE_Q4_0 => Some((32, 18, dequantize_q4_0)),
            GGUF_TYPE_Q8_0 => Some((32, 34, dequantize_q8_0_simd)),
            GGUF_TYPE_Q2_K => Some((QK_K, 84, dequantize_q2_k)),
            GGUF_TYPE_Q3_K => Some((QK_K, 110, dequantize_q3_k)),
            GGUF_TYPE_Q4_K => Some((QK_K, 144, dequantize_q4_k_simd)),
            GGUF_TYPE_Q5_K => Some((QK_K, 176, dequantize_q5_k)),
            GGUF_TYPE_Q6_K => Some((QK_K, 210, dequantize_q6_k)),
            GGUF_TYPE_Q4_1 => Some((32, 20, dequantize_q4_1)),
            GGUF_TYPE_Q5_0 => Some((32, 22, dequantize_q5_0)),
            GGUF_TYPE_Q5_1 => Some((32, 24, dequantize_q5_1)),
            _ => None,
        };
        if let Some((block_elems, block_bytes, dequant)) = blocked {
            let bytes = tensor_byte_range(file_data, offset, size.div_ceil(block_elems) * block_bytes)?;
            let mut values = dequant(bytes)?;
            values.truncate(size);
            return Ok(values);
        }
        match tensor.qtype {
            GGUF_TYPE_F32 => Ok(tensor_byte_range(file_data, offset, size * 4)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect()),
            GGUF_TYPE_F16 => dequantize_f16(tensor_byte_range(file_data, offset, size * 2)?),
            // BF16 (bfloat16): 2 bytes/elem, no block structure. Reuses the existing SIMD
            // converter (y = from_bits((bits as u32) << 16)) already used by the safetensors
            // loaders + gguf/embedding.rs. #1893-class fix: without this arm a BF16 GGUF's
            // embeddings/norms/lm_head hit the catch-all "Unsupported quantization type: 30".
            GGUF_TYPE_BF16 => Ok(crate::inference::simd_bf16_to_f32(tensor_byte_range(
                file_data,
                offset,
                size * 2,
            )?)),
            _ => Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Unsupported quantization type: {}", tensor.qtype),
            }),
        }
    }

    /// Extract model architecture from metadata
    pub fn architecture(&self) -> Option<&str> {
        if let Some(GGUFValue::String(arch)) = self.metadata.get(crate::gguf::keys::GENERAL_ARCHITECTURE) {
            Some(arch.as_str())
        } else {
            None
        }
    }

    /// Get embedding dimension from metadata
    pub fn embedding_dim(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::EMBEDDING_LENGTH);
        if let Some(GGUFValue::UInt32(dim)) = self.metadata.get(&key) {
            Some(*dim as usize)
        } else {
            None
        }
    }

    /// Get number of layers from metadata
    pub fn num_layers(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::BLOCK_COUNT);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// Get number of attention heads from metadata
    pub fn num_heads(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTENTION_HEAD_COUNT);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// Get context length from metadata
    pub fn context_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::CONTEXT_LENGTH);
        if let Some(GGUFValue::UInt32(len)) = self.metadata.get(&key) {
            Some(*len as usize)
        } else {
            None
        }
    }

    /// Get number of key-value heads from metadata (for GQA)
    pub fn num_kv_heads(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTENTION_HEAD_COUNT_KV);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// Get attention key length (head dimension) from metadata.
    ///
    /// This is the per-head dimension for Q/K projections. For most models
    /// this equals `hidden_dim / num_heads`, but Qwen3-0.6B has `head_dim=128`
    /// while `hidden_dim=1024` and `num_heads=16` (so `q_dim=2048 ≠ hidden_dim`).
    ///
    /// GGUF key: `{arch}.attention.key_length`
    pub fn key_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTENTION_KEY_LENGTH);
        if let Some(GGUFValue::UInt32(len)) = self.metadata.get(&key) {
            Some(*len as usize)
        } else {
            None
        }
    }

    /// Get attention value length (value head dimension) from metadata.
    ///
    /// GGUF key: `{arch}.attention.value_length`
    pub fn value_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTENTION_VALUE_LENGTH);
        if let Some(GGUFValue::UInt32(len)) = self.metadata.get(&key) {
            Some(*len as usize)
        } else {
            None
        }
    }

    /// Get RoPE frequency base from metadata
    /// Different models use different bases (LLaMA: 10000, Qwen2: 1000000)
    pub fn rope_freq_base(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ROPE_FREQ_BASE);
        if let Some(GGUFValue::Float32(base)) = self.metadata.get(&key) {
            Some(*base)
        } else {
            None
        }
    }

    /// PMAT-810: Get Gemma2 attention-logit softcap (`{arch}.attn_logit_softcapping`).
    /// 50.0 for gemma-2-*. Returns None when absent (non-Gemma2, or weights-only GGUF).
    pub fn attn_logit_softcapping(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::ATTN_LOGIT_SOFTCAPPING);
        if let Some(GGUFValue::Float32(cap)) = self.metadata.get(&key) {
            Some(*cap)
        } else {
            None
        }
    }

    /// PMAT-810: Get Gemma2 final-logit softcap (`{arch}.final_logit_softcapping`).
    /// 30.0 for gemma-2-*. Returns None when absent (non-Gemma2, or weights-only GGUF).
    pub fn final_logit_softcapping(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::FINAL_LOGIT_SOFTCAPPING);
        if let Some(GGUFValue::Float32(cap)) = self.metadata.get(&key) {
            Some(*cap)
        } else {
            None
        }
    }

    /// PMAT-810: Get Gemma2 pre-attention query scale denominator
    /// (`{arch}.attention.query_pre_attn_scalar`). The attention scale is
    /// `1/sqrt(query_pre_attn_scalar)` rather than `1/sqrt(head_dim)`. 256 for
    /// gemma-2-2b (== head_dim, no-op), 224 for 9b/27b. Returns None when absent
    /// (non-Gemma2, weights-only GGUF, or older converts) so callers fall back to
    /// `head_dim` — exactly llama.cpp's default (`n_embd_head_k`).
    pub fn query_pre_attn_scalar(&self) -> Option<f32> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::QUERY_PRE_ATTN_SCALAR);
        match self.metadata.get(&key) {
            Some(GGUFValue::UInt32(v)) => Some(*v as f32),
            Some(GGUFValue::Float32(v)) => Some(*v),
            _ => None,
        }
    }

    /// M32c.2.2.2.1.2: Get MoE expert count (`{arch}.expert_count`).
    /// 128 for Qwen3-Coder-30B-A3B-Instruct. Returns None for dense models.
    pub fn expert_count(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::EXPERT_COUNT);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// M32c.2.2.2.1.2: Get MoE top-k expert count (`{arch}.expert_used_count`).
    /// 8 for Qwen3-Coder-30B-A3B-Instruct. Returns None for dense models.
    pub fn expert_used_count(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key = crate::gguf::keys::arch_key(arch, crate::gguf::keys::EXPERT_USED_COUNT);
        if let Some(GGUFValue::UInt32(count)) = self.metadata.get(&key) {
            Some(*count as usize)
        } else {
            None
        }
    }

    /// M32c.2.2.2.1.2: Get MoE per-expert FFN intermediate dim
    /// (`{arch}.expert_feed_forward_length`). 768 for Qwen3-Coder-30B-A3B.
    /// Returns None for dense models.
    pub fn expert_feed_forward_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        let key =
            crate::gguf::keys::arch_key(arch, crate::gguf::keys::EXPERT_FEED_FORWARD_LENGTH);
        if let Some(GGUFValue::UInt32(len)) = self.metadata.get(&key) {
            Some(*len as usize)
        } else {
            None
        }
    }
}

/// `file_data[offset..offset + byte_size]`, or the `get_tensor_f32` refusal naming the range when
/// it runs past the end of the file.
fn tensor_byte_range(file_data: &[u8], offset: usize, byte_size: usize) -> Result<&[u8]> {
    if offset + byte_size > file_data.len() {
        return Err(RealizarError::UnsupportedOperation {
            operation: "get_tensor_f32".to_string(),
            reason: format!(
                "Data range [{}, {}) exceeds file size {}",
                offset,
                offset + byte_size,
                file_data.len()
            ),
        });
    }
    Ok(&file_data[offset..offset + byte_size])
}
