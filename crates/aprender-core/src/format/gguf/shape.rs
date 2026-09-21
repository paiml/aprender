impl GgufReader {

    /// Extract a tensor as F32 data (dequantizing if needed)
    ///
    /// Postcondition: data.len() == shape.iter().product()
    #[ensures(ret.as_ref().map_or(true, |(data, shape)| data.len() == shape.iter().product::<usize>()))]
    pub fn get_tensor_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let meta = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{name}' not found in GGUF"),
            })?;
        let (shape, num_elements) = tensor_shape_and_elements(meta)?;
        let tensor_start = self.data_offset + meta.offset as usize;
        let data = tensor_f32_from(&self.data, tensor_start, num_elements, meta.dtype, name)?;
        Ok((data, shape))
    }

    /// Get all tensors as F32
    pub fn get_all_tensors_f32(&self) -> Result<TensorDataMap> {
        self.get_all_tensors_f32_with_progress(|_, _, _| {})
    }

    /// Get all tensors as F32 with per-tensor progress callback.
    ///
    /// Contract: GH-692 — progress feedback for large GGUF dequantization.
    /// Callback receives (current_index, total_count, tensor_name).
    pub fn get_all_tensors_f32_with_progress(
        &self,
        progress: impl Fn(usize, usize, &str),
    ) -> Result<TensorDataMap> {
        let total = self.tensors.len();
        let mut result = BTreeMap::new();
        for (i, meta) in self.tensors.iter().enumerate() {
            progress(i + 1, total, &meta.name);
            let (data, shape) = self.get_tensor_f32(&meta.name)?;
            result.insert(meta.name.clone(), (data, shape));
        }
        Ok(result)
    }

    /// Get raw tensor bytes without dequantization (preserves Q4K/Q6K)
    ///
    /// Returns (raw_bytes, shape, ggml_dtype) where dtype is the GGML type id. Every
    /// type live in upstream ggml is sized (`trueno_quant::TRAITS`); an id upstream
    /// removed, or one newer than that table, is refused by name.
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

        // #3601: size from ggml's own `type_traits` (`trueno_quant::TRAITS`, extracted
        // from upstream and fixture-checked by PMAT-3430) rather than a hand-typed match.
        // That match knew 15 of ggml's 35 live ids, so every IQ*/TQ* tensor was refused
        // as "Unsupported dtype 23 for raw extraction" — which read as a corrupt file.
        // Sizing a type is not a claim that anything here can dequantize it.
        let ggml_type = trueno_quant::GgmlType::try_from_id(meta.dtype)
            .map_err(|e| unsized_ggml_type_error(name, &e))?;

        // Whole blocks, rounding DOWN: the arithmetic the hand-typed arms used, kept so
        // no id that was already sized changes its byte count.
        // BUG-GGUF-002 FIX: Use checked arithmetic to prevent overflow in byte size calc
        let byte_size = (num_elements / ggml_type.block_size())
            .checked_mul(ggml_type.block_bytes())
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

/// #3601: the refusal for a tensor whose ggml type id has no size in `trueno_quant::TRAITS`.
///
/// Every id live upstream is sized, so this is reached only by an id upstream REMOVED or one
/// newer than the pinned table. The message says which — "Unsupported dtype 23" sent readers
/// looking for file corruption when the gap was in apr.
fn unsized_ggml_type_error(tensor: &str, e: &trueno_quant::GgmlTypeError) -> AprenderError {
    let why = match e {
        trueno_quant::GgmlTypeError::Removed { .. } => {
            "no current ggml defines a layout for it, so the file must be re-quantized from \
             its source weights"
        }
        trueno_quant::GgmlTypeError::Unknown { .. } => {
            "it is newer than the ggml type table this apr was built with (pinned by \
             scripts/llama_pin.toml) — a gap in apr, not a corrupt file; please report it at \
             https://github.com/paiml/aprender/issues"
        }
    };
    AprenderError::FormatError {
        message: format!(
            "GGUF tensor '{tensor}': {e}; {why}. The ids apr sizes are listed in \
             contracts/ggml-type-v1.yaml"
        ),
    }
}

/// A tensor's shape and element count, with the BUG-GGUF-002 guards (overflow, and the element
/// limit against a malicious file).
fn tensor_shape_and_elements(meta: &GgufTensorMeta) -> Result<(Vec<usize>, usize)> {
    let name = &meta.name;
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
    Ok((shape, num_elements))
}

/// Dequantize one tensor to F32 from `data`, where its bytes start at `tensor_start` (#3790: the
/// whole file for [`GgufReader::get_tensor_f32`], or just the tensor's own bytes for
/// [`read_tensor_f32`]). A tensor whose bytes run past `data` is an error, never a short read.
fn tensor_f32_from(
    data: &[u8],
    tensor_start: usize,
    num_elements: usize,
    dtype: u32,
    name: &str,
) -> Result<Vec<f32>> {
let data = match dtype {
        0 => {
            // F32 - direct copy
            // BUG-GGUF-002 FIX: Use checked_mul for byte size calculation
            let byte_size =
                num_elements
                    .checked_mul(4)
                    .ok_or_else(|| AprenderError::FormatError {
                        message: format!("Tensor '{}' byte size calculation overflow", name),
                    })?;
            if tensor_start + byte_size > data.len() {
                return Err(AprenderError::FormatError {
                    message: format!("Tensor '{name}' data exceeds file size"),
                });
            }
            let bytes = &data[tensor_start..tensor_start + byte_size];
            bytes
                .as_chunks::<4>()
                .0
                .iter()
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
            if tensor_start + byte_size > data.len() {
                return Err(AprenderError::FormatError {
                    message: format!("Tensor '{name}' data exceeds file size"),
                });
            }
            let bytes = &data[tensor_start..tensor_start + byte_size];
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        // GGML dtype values (from ggml.h):
        // 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1, 8=Q8_0, 9=Q8_1
        // 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K
        // 16+=IQ variants
        2 => {
            // Q4_0 - dequantize
            super::dequantize_q4_0(data, tensor_start, num_elements)?
        }
        3 => {
            // Q4_1 - dequantize (blocks of 32 with scale and min)
            super::dequantize_q4_1(data, tensor_start, num_elements)?
        }
        6 => {
            // Q5_0 - dequantize (blocks of 32 with 5-bit quants)
            super::dequantize_q5_0(data, tensor_start, num_elements)?
        }
        7 => {
            // Q5_1 - dequantize (blocks of 32 with 5-bit quants + min)
            dequantize_q5_1(data, tensor_start, num_elements)?
        }
        8 => {
            // Q8_0 - dequantize
            super::dequantize_q8_0(data, tensor_start, num_elements)?
        }
        10 => {
            // Q2_K - dequantize (super blocks of 256)
            dequantize_q2_k(data, tensor_start, num_elements)?
        }
        11 => {
            // Q3_K - dequantize (super blocks of 256)
            dequantize_q3_k(data, tensor_start, num_elements)?
        }
        12 => {
            // Q4_K - dequantize (super blocks of 256 elements, 144 bytes/block)
            dequantize_q4_k(data, tensor_start, num_elements)?
        }
        13 => {
            // Q5_K - dequantize (super blocks of 256 elements, 176 bytes/block)
            dequantize_q5_k(data, tensor_start, num_elements)?
        }
        14 => {
            // Q6_K - dequantize (super blocks of 256 elements, 210 bytes/block)
            dequantize_q6_k(data, tensor_start, num_elements)?
        }
        // #3656: IQ types (16..=23) used to go to `dequantize_iq_approximate`, which
        // mapped each raw byte to `(b - 128) * 0.01` and returned Ok — `apr convert`
        // wrote those invented weights (std 36.6x the real tensor) and exited 0. With
        // no real dequantizer here, the only honest answer is a refusal.
        _ => {
            let type_name = trueno_quant::GgmlType::from_id(dtype)
                .map_or("a type ggml does not define", trueno_quant::GgmlType::as_str);
            return Err(AprenderError::FormatError {
                message: format!(
                    "GGUF tensor '{name}' is {type_name} (ggml type {}): aprender-core has \
                     no dequantizer for it, so it cannot be read as F32. Refusing rather \
                     than approximating, which would invent its weights",
                    dtype
                ),
            });
        }
    };

    Ok(data)
}

/// The bytes one tensor occupies, from ggml's own type table: `None` for a type aprender-core
/// has no dequantizer for (reading it would be wasted: [`tensor_f32_from`] refuses it by name).
fn tensor_byte_extent(dtype: u32, num_elements: usize) -> Option<usize> {
    match dtype {
        0 => num_elements.checked_mul(4),
        1 => num_elements.checked_mul(2),
        2 | 3 | 6 | 7 | 8 | 10 | 11 | 12 | 13 | 14 => {
            let t = trueno_quant::GgmlType::from_id(dtype)?;
            num_elements
                .div_ceil(t.block_size())
                .checked_mul(t.block_bytes())
        }
        _ => None,
    }
}

/// One tensor as F32, read from `file` at its own extent: its bytes, never the rest of the model
/// (#3790). `data_offset` is the header's; `file_len` cuts the read at end of file, so a tensor
/// that runs past it fails in the dequantizer exactly as it does over the whole file.
pub(crate) fn read_tensor_f32(
    file: &std::fs::File,
    file_len: u64,
    data_offset: usize,
    meta: &GgufTensorMeta,
) -> Result<(Vec<f32>, Vec<usize>)> {
    use std::os::unix::fs::FileExt;
    let (shape, num_elements) = tensor_shape_and_elements(meta)?;
    let start = (data_offset as u64).saturating_add(meta.offset);
    let wanted = tensor_byte_extent(meta.dtype, num_elements).unwrap_or(0) as u64;
    let n = wanted.min(file_len.saturating_sub(start));
    let mut bytes = vec![0u8; usize::try_from(n).map_err(|_| AprenderError::FormatError {
        message: format!("Tensor '{}' extent does not fit in memory", meta.name),
    })?];
    file.read_exact_at(&mut bytes, start)
        .map_err(AprenderError::Io)?;
    let data = tensor_f32_from(&bytes, 0, num_elements, meta.dtype, &meta.name)?;
    Ok((data, shape))
}
