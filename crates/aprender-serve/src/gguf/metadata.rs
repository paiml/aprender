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

        // Extract and dequantize based on qtype
        match tensor.qtype {
            GGUF_TYPE_F32 => {
                // Unquantized F32 data
                let byte_size = size * 4; // 4 bytes per f32
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

                let bytes = &file_data[offset..offset + byte_size];
                let values = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(values)
            },
            GGUF_TYPE_Q4_0 => {
                // Q4_0 quantized data
                use crate::quantize::dequantize_q4_0;

                // Q4_0 block: 32 elements
                // Layout: 1×f16 scale (2 bytes) + 16 bytes (32×4-bit values) = 18 bytes
                const BLOCK_BYTES: usize = 18;
                const BLOCK_SIZE: usize = 32;

                let num_blocks = size.div_ceil(BLOCK_SIZE);
                let byte_size = num_blocks * BLOCK_BYTES;

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

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q4_0(bytes)?;

                // Trim to exact size (dequantization pads to block boundaries)
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q8_0 => {
                // Q8_0 quantized data - use SIMD-parallel for faster loading
                use crate::quantize::dequantize_q8_0_simd;

                // Q8_0 block size: 34 bytes (2 for f16 scale + 32 for quants)
                const BLOCK_BYTES: usize = 34;
                const BLOCK_SIZE: usize = 32;

                let num_blocks = size.div_ceil(BLOCK_SIZE);
                let byte_size = num_blocks * BLOCK_BYTES;

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

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q8_0_simd(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q2_K => {
                // Q2_K quantized data (K-quantization) - 2 bits per weight
                use crate::quantize::{dequantize_q2_k, QK_K};

                // Q2_K super-block size: 84 bytes for 256 values
                const SUPER_BLOCK_BYTES: usize = 84;

                let num_super_blocks = size.div_ceil(QK_K);
                let byte_size = num_super_blocks * SUPER_BLOCK_BYTES;

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

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q2_k(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q4_K => {
                // Q4_K quantized data (K-quantization) - use SIMD-parallel for faster loading
                use crate::quantize::{dequantize_q4_k_simd, QK_K};

                // Q4_K super-block size: 144 bytes for 256 values
                const SUPER_BLOCK_BYTES: usize = 144;

                let num_super_blocks = size.div_ceil(QK_K);
                let byte_size = num_super_blocks * SUPER_BLOCK_BYTES;

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

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q4_k_simd(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q5_K => {
                // Q5_K quantized data (K-quantization)
                use crate::quantize::{dequantize_q5_k, QK_K};

                // Q5_K super-block size: 176 bytes for 256 values
                const SUPER_BLOCK_BYTES: usize = 176;

                let num_super_blocks = size.div_ceil(QK_K);
                let byte_size = num_super_blocks * SUPER_BLOCK_BYTES;

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

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q5_k(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q6_K => {
                // Q6_K quantized data (K-quantization)
                use crate::quantize::{dequantize_q6_k, QK_K};

                // Q6_K super-block size: 210 bytes for 256 values
                const SUPER_BLOCK_BYTES: usize = 210;

                let num_super_blocks = size.div_ceil(QK_K);
                let byte_size = num_super_blocks * SUPER_BLOCK_BYTES;

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

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q6_k(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_F16 => {
                // F16 (half-precision float) data
                use crate::quantize::dequantize_f16;

                let byte_size = size * 2; // 2 bytes per f16
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

                let bytes = &file_data[offset..offset + byte_size];
                let values = dequantize_f16(bytes)?;
                Ok(values)
            },
            GGUF_TYPE_Q4_1 => {
                // Q4_1 quantized data
                use crate::quantize::dequantize_q4_1;

                // Q4_1 block size: 20 bytes (2 for scale + 2 for min + 16 for quants)
                const BLOCK_BYTES: usize = 20;
                const BLOCK_SIZE: usize = 32;

                let num_blocks = size.div_ceil(BLOCK_SIZE);
                let byte_size = num_blocks * BLOCK_BYTES;

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

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q4_1(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q5_0 => {
                // Q5_0 quantized data
                use crate::quantize::dequantize_q5_0;

                // Q5_0 block size: 22 bytes (2 for scale + 4 for high bits + 16 for quants)
                const BLOCK_BYTES: usize = 22;
                const BLOCK_SIZE: usize = 32;

                let num_blocks = size.div_ceil(BLOCK_SIZE);
                let byte_size = num_blocks * BLOCK_BYTES;

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

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q5_0(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
            GGUF_TYPE_Q5_1 => {
                // Q5_1 quantized data
                use crate::quantize::dequantize_q5_1;

                // Q5_1 block size: 24 bytes (2 for scale + 2 for min + 4 for high bits + 16 for quants)
                const BLOCK_BYTES: usize = 24;
                const BLOCK_SIZE: usize = 32;

                let num_blocks = size.div_ceil(BLOCK_SIZE);
                let byte_size = num_blocks * BLOCK_BYTES;

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

                let bytes = &file_data[offset..offset + byte_size];
                let mut values = dequantize_q5_1(bytes)?;

                // Trim to exact size
                values.truncate(size);
                Ok(values)
            },
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
}
