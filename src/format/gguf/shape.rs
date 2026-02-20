impl GgufReader {

    /// Extract a tensor as F32 data (dequantizing if needed)
    pub fn get_tensor_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let meta = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{name}' not found in GGUF"),
            })?;

        let shape: Vec<usize> = meta.dims.iter().map(|&d| d as usize).collect();

        // BUG-GGUF-002 FIX: Use checked multiplication to prevent integer overflow
        let num_elements = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| AprenderError::FormatError {
                message: format!(
                    "Tensor '{}' shape {:?} causes integer overflow (malicious file?)",
                    name, shape
                ),
            })?;

        // BUG-GGUF-002 FIX: Validate total elements against reasonable limit
        if num_elements > MAX_TENSOR_ELEMENTS {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Tensor '{}' has {} elements, exceeds max {} (possible malicious file)",
                    name, num_elements, MAX_TENSOR_ELEMENTS
                ),
            });
        }

        let tensor_start = self.data_offset + meta.offset as usize;

        let data = match meta.dtype {
            0 => {
                // F32 - direct copy
                // BUG-GGUF-002 FIX: Use checked_mul for byte size calculation
                let byte_size =
                    num_elements
                        .checked_mul(4)
                        .ok_or_else(|| AprenderError::FormatError {
                            message: format!("Tensor '{}' byte size calculation overflow", name),
                        })?;
                if tensor_start + byte_size > self.data.len() {
                    return Err(AprenderError::FormatError {
                        message: format!("Tensor '{name}' data exceeds file size"),
                    });
                }
                let bytes = &self.data[tensor_start..tensor_start + byte_size];
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            1 => {
                // F16 - convert to F32
                // BUG-GGUF-002 FIX: Use checked_mul for byte size calculation
                let byte_size =
                    num_elements
                        .checked_mul(2)
                        .ok_or_else(|| AprenderError::FormatError {
                            message: format!("Tensor '{}' byte size calculation overflow", name),
                        })?;
                if tensor_start + byte_size > self.data.len() {
                    return Err(AprenderError::FormatError {
                        message: format!("Tensor '{name}' data exceeds file size"),
                    });
                }
                let bytes = &self.data[tensor_start..tensor_start + byte_size];
                bytes
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect()
            }
            // GGML dtype values (from ggml.h):
            // 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1, 8=Q8_0, 9=Q8_1
            // 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K
            // 16+=IQ variants
            2 => {
                // Q4_0 - dequantize
                super::dequantize_q4_0(&self.data, tensor_start, num_elements)?
            }
            3 => {
                // Q4_1 - dequantize (blocks of 32 with scale and min)
                super::dequantize_q4_1(&self.data, tensor_start, num_elements)?
            }
            6 => {
                // Q5_0 - dequantize (blocks of 32 with 5-bit quants)
                super::dequantize_q5_0(&self.data, tensor_start, num_elements)?
            }
            7 => {
                // Q5_1 - dequantize (blocks of 32 with 5-bit quants + min)
                dequantize_q5_1(&self.data, tensor_start, num_elements)?
            }
            8 => {
                // Q8_0 - dequantize
                super::dequantize_q8_0(&self.data, tensor_start, num_elements)?
            }
            10 => {
                // Q2_K - dequantize (super blocks of 256)
                dequantize_q2_k(&self.data, tensor_start, num_elements)?
            }
            11 => {
                // Q3_K - dequantize (super blocks of 256)
                dequantize_q3_k(&self.data, tensor_start, num_elements)?
            }
            12 => {
                // Q4_K - dequantize (super blocks of 256 elements, 144 bytes/block)
                dequantize_q4_k(&self.data, tensor_start, num_elements)?
            }
            13 => {
                // Q5_K - dequantize (super blocks of 256 elements, 176 bytes/block)
                dequantize_q5_k(&self.data, tensor_start, num_elements)?
            }
            14 => {
                // Q6_K - dequantize (super blocks of 256 elements, 210 bytes/block)
                dequantize_q6_k(&self.data, tensor_start, num_elements)?
            }
            // I-quants (importance matrix quantization) - complex formats
            // For now, approximate with simpler dequantization
            16..=23 => {
                // IQ2_XXS=16, IQ2_XS=17, IQ2_S=18, IQ3_XXS=19, IQ3_S=20, IQ1_S=21, IQ4_NL=22, IQ4_XS=23
                // Fall back to zero-filled tensor with warning
                eprintln!(
                    "Warning: I-quant dtype {} for tensor '{}' not fully supported, using approximation",
                    meta.dtype, name
                );
                dequantize_iq_approximate(&self.data, tensor_start, num_elements, meta.dtype)
            }
            _ => {
                return Err(AprenderError::FormatError {
                    message: format!("Unsupported GGUF dtype {} for tensor '{name}'", meta.dtype),
                });
            }
        };

        Ok((data, shape))
    }

    /// Get all tensors as F32
    pub fn get_all_tensors_f32(&self) -> Result<TensorDataMap> {
        let mut result = BTreeMap::new();
        for meta in &self.tensors {
            let (data, shape) = self.get_tensor_f32(&meta.name)?;
            result.insert(meta.name.clone(), (data, shape));
        }
        Ok(result)
    }

    /// Get raw tensor bytes without dequantization (preserves Q4K/Q6K)
    ///
    /// Returns (raw_bytes, shape, ggml_dtype) where dtype is per GGML spec:
    /// - 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 8=Q8_0
    /// - 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K
    pub fn get_tensor_raw(&self, name: &str) -> Result<(Vec<u8>, Vec<usize>, u32)> {
        let meta = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{name}' not found in GGUF"),
            })?;

        let shape: Vec<usize> = meta.dims.iter().map(|&d| d as usize).collect();

        // BUG-GGUF-002 FIX: Use checked multiplication to prevent integer overflow
        let num_elements = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| AprenderError::FormatError {
                message: format!(
                    "Tensor '{}' shape {:?} causes integer overflow (malicious file?)",
                    name, shape
                ),
            })?;

        // BUG-GGUF-002 FIX: Validate total elements against reasonable limit
        if num_elements > MAX_TENSOR_ELEMENTS {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Tensor '{}' has {} elements, exceeds max {} (possible malicious file)",
                    name, num_elements, MAX_TENSOR_ELEMENTS
                ),
            });
        }

        let tensor_start = self.data_offset + meta.offset as usize;

        // Calculate byte size based on dtype (GGML dtype values)
        // See llama.cpp ggml.h for type definitions
        // GGML enum: 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1, 8=Q8_0, 9=Q8_1
        //           10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K, 15=Q8_K
        // BUG-GGUF-002 FIX: Use checked arithmetic to prevent overflow in byte size calc
        // Note: Some dtypes share byte sizes but are documented separately for clarity
        #[allow(clippy::match_same_arms)]
        let byte_size = match meta.dtype {
            0 => num_elements.checked_mul(4),         // F32
            1 => num_elements.checked_mul(2),         // F16
            2 => (num_elements / 32).checked_mul(18), // Q4_0: 32 elements = 2 (d) + 16 (qs)
            3 => (num_elements / 32).checked_mul(20), // Q4_1: 32 elements = 2 (d) + 2 (m) + 16 (qs)
            // dtype 4,5 = removed (Q4_2, Q4_3)
            6 => (num_elements / 32).checked_mul(22), // Q5_0: 32 elements = 2 (d) + 4 (qh) + 16 (ql)
            7 => (num_elements / 32).checked_mul(24), // Q5_1: 32 elements = 2 (d) + 2 (m) + 4 (qh) + 16 (ql)
            8 => (num_elements / 32).checked_mul(34), // Q8_0: 32 elements = 2 (d) + 32 (qs)
            9 => (num_elements / 32).checked_mul(36), // Q8_1: 32 elements = 2 (d) + 2 (s) + 32 (qs)
            10 => (num_elements / 256).checked_mul(84), // Q2_K: 256 elements = 84 bytes
            11 => (num_elements / 256).checked_mul(110), // Q3_K: 256 elements = 110 bytes
            12 => (num_elements / 256).checked_mul(144), // Q4_K: 256 elements = 144 bytes
            13 => (num_elements / 256).checked_mul(176), // Q5_K: 256 elements = 176 bytes
            14 => (num_elements / 256).checked_mul(210), // Q6_K: 256 elements = 210 bytes
            15 => (num_elements / 256).checked_mul(292), // Q8_K: 256 elements = 292 bytes
            30 => num_elements.checked_mul(2),        // BF16: 2 bytes per element
            _ => {
                return Err(AprenderError::FormatError {
                    message: format!("Unsupported dtype {} for raw extraction", meta.dtype),
                });
            }
        }
        .ok_or_else(|| AprenderError::FormatError {
            message: format!(
                "Tensor '{}' byte size calculation overflow (dtype: {})",
                name, meta.dtype
            ),
        })?;

        if tensor_start + byte_size > self.data.len() {
            return Err(AprenderError::FormatError {
                message: format!("Tensor '{name}' data exceeds file size"),
            });
        }

        let bytes = self.data[tensor_start..tensor_start + byte_size].to_vec();
        Ok((bytes, shape, meta.dtype))
    }

    /// Get all tensors as raw bytes (preserves quantization)
    ///
    /// Returns BTreeMap of name -> (raw_bytes, shape, ggml_dtype)
    pub fn get_all_tensors_raw(&self) -> Result<BTreeMap<String, (Vec<u8>, Vec<usize>, u32)>> {
        let mut result = BTreeMap::new();
        for meta in &self.tensors {
            let (data, shape, dtype) = self.get_tensor_raw(&meta.name)?;
            result.insert(meta.name.clone(), (data, shape, dtype));
        }
        Ok(result)
    }
}
