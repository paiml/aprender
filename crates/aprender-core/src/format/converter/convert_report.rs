
impl ConvertReport {
    /// Build a report, computing the reduction ratio from original/converted sizes
    fn build(
        original_size: usize,
        output_path: &Path,
        tensor_count: usize,
        quantization: Option<QuantizationType>,
        compression: Option<Compression>,
    ) -> Self {
        let converted_size = fs::metadata(output_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let reduction_ratio = if converted_size > 0 {
            original_size as f64 / converted_size as f64
        } else {
            0.0
        };
        Self {
            original_size,
            converted_size,
            tensor_count,
            quantization,
            compression,
            reduction_ratio,
        }
    }

    /// Format reduction as percentage string
    #[must_use]
    pub fn reduction_percent(&self) -> String {
        if self.original_size > 0 && self.converted_size > 0 {
            let reduction = 100.0 * (1.0 - self.converted_size as f64 / self.original_size as f64);
            format!("{:.1}%", reduction)
        } else {
            "N/A".to_string()
        }
    }
}

/// PMAT-271: Detect format via magic bytes first, extension fallback.
/// Handles extensionless HF cache blob paths.
fn detect_format(path: &Path) -> Result<crate::format::rosetta::FormatType> {
    use crate::format::rosetta::FormatType;
    FormatType::from_magic(path).or_else(|_| FormatType::from_extension(path))
}

/// Load tensors from model file
///
/// Supports: SafeTensors, APR, GGUF (GH-164 fix)
/// GGUF tensors are dequantized to F32 during loading.
/// PMAT-271: Uses magic byte detection first, extension fallback for extensionless HF cache blobs.
pub fn load_model_tensors(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    use crate::format::rosetta::FormatType;
    let format = detect_format(path)?;

    match format {
        FormatType::SafeTensors => load_safetensors_tensors(path),
        FormatType::Apr => load_apr_tensors_f32(path),
        FormatType::Gguf => load_gguf_tensors_f32(path),
    }
}

/// Load tensors with provenance tracking (DOUBLE-QUANT-001).
///
/// Returns `TensorProvenance::Native` for SafeTensors and unquantized APR sources,
/// `TensorProvenance::Dequantized` for GGUF and quantized APR sources.
/// This enables compile-time prevention of double quantization.
/// PMAT-271: Uses magic byte detection first, extension fallback for extensionless HF cache blobs.
pub(crate) fn load_model_tensors_provenance(path: &Path) -> Result<TensorProvenance> {
    use crate::format::rosetta::FormatType;
    let format = detect_format(path)?;

    match format {
        FormatType::SafeTensors => {
            let tensors = load_safetensors_tensors(path)?;
            Ok(TensorProvenance::Native(NativeF32Tensors::new(tensors)))
        }
        FormatType::Apr => {
            // Check if source has quantized tensors
            if let Some(quant) = export::detect_apr_quantization(path) {
                let tensors = load_apr_tensors_f32(path)?;
                Ok(TensorProvenance::Dequantized(DequantizedTensors::new(
                    tensors, quant,
                )))
            } else {
                let tensors = load_apr_tensors_f32(path)?;
                Ok(TensorProvenance::Native(NativeF32Tensors::new(tensors)))
            }
        }
        FormatType::Gguf => {
            // GGUF models are always quantized (Q4K, Q6K, etc.)
            let tensors = load_gguf_tensors_f32(path)?;
            Ok(TensorProvenance::Dequantized(DequantizedTensors::new(
                tensors,
                QuantizationType::Q4K,
            )))
        }
    }
}

/// Load GGUF tensors and dequantize to F32 (GH-164)
///
/// Uses GgufReader::get_all_tensors_f32() which handles:
/// - Q4_K, Q5_K, Q6_K dequantization
/// - Q4_0, Q5_0, Q8_0 dequantization
/// - F16, F32 direct loading
///
/// PMAT-187: Validates all tensors after loading to catch corruption early.
/// GH-692: Shows per-tensor progress on interactive terminals.
fn load_gguf_tensors_f32(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let reader = GgufReader::from_file(path)?;

    // GH-692: Show dequantization progress on interactive terminals
    use std::io::{IsTerminal, Write};
    let is_tty = std::io::stderr().is_terminal();
    let tensors = reader.get_all_tensors_f32_with_progress(|current, total, name| {
        if is_tty {
            eprint!("\r  [{current}/{total}] Dequantizing {name}");
            let _ = std::io::stderr().flush();
        }
    })?;
    if is_tty {
        // Clear the progress line
        eprint!("\r{}\r", " ".repeat(72));
        let _ = std::io::stderr().flush();
    }

    // PMAT-187: Validate all tensors after loading (Jidoka - stop the line)
    for (name, (data, _shape)) in &tensors {
        validate_tensor_values(name, data)?;
    }

    // GH-208: MANDATORY CONTRACT ENFORCEMENT
    // Shape transformation goes through enforce_import_contract().
    // See: contracts/tensor-layout-v1.yaml, Five Whys in layout_contract.rs
    //
    // NOTE: This F32 path is for dequantized tensors. The raw quantized path
    // (import.rs:apr_import_gguf_raw) also uses enforce_import_contract.
    use crate::format::layout_contract::enforce_import_contract;

    let tensors = tensors
        .into_iter()
        .map(|(name, (data, shape))| {
            // Use CONTRACT for shape transformation (vocab_size/hidden_dim=0 means unknown)
            let (apr_shape, needs_data_transpose) = enforce_import_contract(&name, &shape, 0, 0);

            // GH-208: Data transpose should NEVER be needed
            assert!(
                !needs_data_transpose,
                "CONTRACT BUG: enforce_import_contract returned needs_data_transpose=true for '{}'. \
                 GGUF→APR NEVER needs data transpose.",
                name
            );

            (name, (data, apr_shape))
        })
        .collect();

    Ok(tensors)
}

/// Load APR tensors and dequantize to F32 (PMAT-174, GH-196 fix)
///
/// Uses `AprV2Reader` for correct parsing of v2 format files produced by
/// `AprV2Writer`. Falls back to manual v1 parsing for legacy files.
///
/// Handles all APR dtypes: F32, F16, BF16, Q4_K, Q6_K, Q8_0
fn load_apr_tensors_f32(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    use crate::format::v2::AprV2Reader;
    use crate::format::AprV2DequantExt; // issue #2231 re-attached accessor

    let data = fs::read(path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to read APR file: {e}"),
    })?;

    // Use AprV2Reader for correct v2 format parsing
    let reader = AprV2Reader::from_bytes(&data).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to parse APR file: {e:?}"),
    })?;

    let mut tensors = BTreeMap::new();
    for name in reader.tensor_names() {
        let entry = reader
            .get_tensor(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{}' missing from index", name),
            })?;
        let shape = entry.shape.clone();

        // get_tensor_as_f32 handles all dtypes (F32, F16, Q8, Q4, etc.)
        let f32_data =
            reader
                .get_tensor_as_f32(name)
                .ok_or_else(|| AprenderError::FormatError {
                    message: format!("Failed to dequantize tensor '{}'", name),
                })?;

        // PMAT-187: Validate tensor values after dequantization (Jidoka - stop the line)
        validate_tensor_values(name, &f32_data)?;

        tensors.insert(name.to_string(), (f32_data, shape));
    }

    Ok(tensors)
}

/// PMAT-187: Validate tensor values for NaN/Inf/explosive corruption
///
/// Toyota Way Jidoka: Stop the line on quality defects, don't pass defects downstream.
/// This catches corruption introduced during dequantization before it propagates.
///
/// # Errors
///
/// Returns error if tensor contains NaN, Inf, or explosive values (mean > 100)
pub(crate) fn validate_tensor_values(name: &str, data: &[f32]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }

    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut sum: f64 = 0.0;

    for &value in data {
        if value.is_nan() {
            nan_count += 1;
        } else if value.is_infinite() {
            inf_count += 1;
        } else {
            sum += value as f64;
        }
    }

    // Fail fast on NaN
    if nan_count > 0 {
        return Err(AprenderError::FormatError {
            message: format!(
                "PMAT-187: Tensor '{}' contains {} NaN values (data corruption detected). \
                 Toyota Way: Stop the line - do not pass defects downstream.",
                name, nan_count
            ),
        });
    }

    // Fail fast on Inf
    if inf_count > 0 {
        return Err(AprenderError::FormatError {
            message: format!(
                "PMAT-187: Tensor '{}' contains {} Inf values (numerical overflow detected). \
                 Toyota Way: Stop the line - do not pass defects downstream.",
                name, inf_count
            ),
        });
    }

    // Check for explosive mean (indicates corrupted scale factors)
    let valid_count = data.len() - nan_count - inf_count;
    if valid_count > 0 {
        let mean = sum / valid_count as f64;
        if mean.abs() > 100.0 {
            return Err(AprenderError::FormatError {
                message: format!(
                    "PMAT-187: Tensor '{}' has explosive mean={:.2e} (expected [-100, 100]). \
                     This indicates corrupted quantization scale factors. \
                     Toyota Way: Stop the line - do not pass defects downstream.",
                    name, mean
                ),
            });
        }
    }

    Ok(())
}

/// Dequantize F16 to F32 (PMAT-174)
#[cfg(test)]
fn dequantize_f16_to_f32(bytes: &[u8], _num_elements: usize) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f16_to_f32(bits)
        })
        .collect()
}

/// Dequantize BF16 to F32 (PMAT-174)
#[cfg(test)]
fn dequantize_bf16_to_f32(bytes: &[u8], _num_elements: usize) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            // BF16 is F32 with lower 16 mantissa bits zeroed
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

/// Dequantize Q8_0 to F32 (PMAT-174)
/// Q8_0: 32-element blocks with f16 scale + 32 int8 quants
#[cfg(test)]
fn dequantize_q8_0_to_f32(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    // f16 scale + 32 int8s. MSRV-compatible div_ceil: (n + d - 1) / d
    const BLOCK_BYTES: usize = 2 + 32;
    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = Vec::with_capacity(num_elements);

    for i in 0..num_blocks {
        let block_start = i * BLOCK_BYTES;
        if block_start + BLOCK_BYTES > bytes.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([bytes[block_start], bytes[block_start + 1]]);
        // GH-186 FIX: Clamp NaN/Inf/subnormal to prevent propagation
        let scale_raw = f16_to_f32(scale_bits);
        // Uses shared F16_MIN_NORMAL from crate::format::f16_safety (P2 fix)
        let scale =
            if scale_raw.is_nan() || scale_raw.is_infinite() || scale_raw.abs() < F16_MIN_NORMAL {
                0.0
            } else {
                scale_raw
            };

        for j in 0..BLOCK_SIZE {
            if result.len() >= num_elements {
                break;
            }
            let q = bytes[block_start + 2 + j] as i8;
            result.push(q as f32 * scale);
        }
    }

    result
}

/// Calculate total tensor size in bytes (f32)
pub(crate) fn calculate_tensor_size(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> usize {
    tensors.values().map(|(data, _)| data.len() * 4).sum()
}

/// Apply quantization to tensors.
///
/// DOUBLE-QUANT-001: Only accepts `NativeF32Tensors` (natively F32 sources).
/// Passing `DequantizedTensors` is a compile error, preventing the lossy
/// Q4K→F32→Q4K double quantization that destroyed weight fidelity (PMAT-252).
///
/// BUG-EXPORT-004 FIX: Embedding tensors are SKIPPED from quantization.
/// Embeddings are lookup tables with small values that would round to zero
/// under global-scale Int4/Int8 quantization (78% of values become zero).
/// They are kept in F32 and exported as F32 in GGUF.
pub(crate) fn quantize_tensors(
    tensors: &NativeF32Tensors,
    quant_type: &QuantizationType,
) -> Result<NativeF32Tensors> {
    contract_pre_quantization_bounds!();
    let mut result = BTreeMap::new();

    for (name, (data, shape)) in tensors.as_ref() {
        // BUG-EXPORT-004: Skip quantization for embedding tensors
        // Embeddings have small values (-0.3 to 0.2) that would mostly round to 0
        // under global-scale Int4 quantization (scale=0.042, values<0.021 → 0)
        let is_embedding = name.contains("embed_tokens")
            || name.contains("token_embd")
            || name.contains("wte")  // GPT-style
            || name.contains("word_embeddings"); // BERT-style

        // GH-234: lm_head.weight has same small-value distribution as embeddings
        // (especially when weight-tied). Quantization causes 4:1 packing / all-zeros.
        let is_lm_head = name.contains("lm_head") || name == "output.weight";

        let quantized_data = if is_embedding || is_lm_head {
            // Keep embeddings and lm_head in original F32 - no quantization
            data.clone()
        } else {
            match quant_type {
                QuantizationType::Fp16 => quantize_fp16(data),
                QuantizationType::Int8 => quantize_int8(data),
                QuantizationType::Int4 => quantize_int4(data),
                QuantizationType::Q4K => {
                    // Q4K: quantize to packed bytes then dequantize back to f32
                    // This preserves the shape but shows quantization error
                    let q4k_bytes = quantize_q4_k(data);
                    dequantize_q4_k_to_f32(&q4k_bytes, data.len())
                }
            }
        };
        result.insert(name.clone(), (quantized_data, shape.clone()));
    }

    let result = NativeF32Tensors::new(result);
    contract_post_quantization_bounds!(&result);
    Ok(result)
}

// NOTE: dequantize_q4_k_to_f32 moved to trueno-quant crate (Toyota Way consolidation)

/// Quantize to fp16 - TRUE PACKING (2 bytes per value)
/// Returns dequantized f32 for now (proper f16 storage requires format change)
fn quantize_fp16(data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|&v| {
            // Convert f32 to f16 bits then back - TRUE precision reduction
            let f16_bits = f32_to_f16(v);
            f16_to_f32(f16_bits)
        })
        .collect()
}

/// Convert an `f32` to IEEE 754 half-precision (`f16`) bits.
///
/// PMAT-905 (CPU f16-RNE sweep): bit-identical to `half::f16::from_f32` across
/// the **entire** 2^32 `f32` domain (verified exhaustively, NaN payloads
/// treated as equal). This is the **canonical** f16 encoder for the whole
/// `converter` module — it feeds `quantize_fp16` (the `apr convert --quantize
/// fp16` precision-reduction round-trip) and, via `f32_to_f16_bits` →
/// `f32_slice_to_f16_le_bytes`, the SafeTensors FP16 export byte path.
///
/// The previous implementation rounded **toward zero**: the normal path
/// truncated the mantissa (`mantissa >> 13`, no sticky bit) and the subnormal
/// path rounded **half-up** (`saturating_add(round_bit)`), with f32 subnormals
/// flushed to zero and NaN payloads collapsed. It diverged from
/// `half::f16::from_f32` in ~251.6M of 2^32 inputs, e.g. `255.99 → 0x5BFF`
/// (should be `0x5C00`, mantissa carry) and `65520.0 → 0x7BFF` (should be `+Inf
/// 0x7C00`, overflow tie). Every `apr convert --quantize fp16` weight was thus
/// biased ~0.5–1 ULP low with a mis-encoded overflow boundary.
///
/// Re-expressed as IEEE round-to-nearest-even across normal AND subnormal grids
/// with a carry that propagates into the exponent (and onward to Inf), matching
/// `half` exactly including ties-to-even, overflow→Inf, and NaN payloads.
fn f32_to_f16(value: f32) -> u16 {
    let x: u32 = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = (x >> 23) & 0xFF; // biased f32 exponent (8 bits)
    let mantissa = x & 0x007F_FFFF;

    // Inf / NaN
    if exp == 0xFF {
        return if mantissa == 0 {
            sign | 0x7C00 // ±Inf
        } else {
            // NaN: canonical quiet bit + top mantissa bits (matches `half`)
            sign | 0x7E00 | ((mantissa >> 13) as u16)
        };
    }

    // Rebias exponent: f32 bias = 127, f16 bias = 15.
    let half_exp = (exp as i32) - 127 + 15;

    if half_exp >= 0x1F {
        // Overflow → ±Inf
        return sign | 0x7C00;
    }

    if half_exp <= 0 {
        // Subnormal or zero. `14 - half_exp` is the right-shift; >24 means every
        // significant bit is shifted out (true zero).
        if 14 - half_exp > 24 {
            return sign;
        }
        let m = mantissa | 0x0080_0000; // restore implicit leading 1
        let shift = (14 - half_exp) as u32;
        let mut half_mant = m >> shift;
        // Round to nearest even: round up iff the highest dropped bit is set AND
        // the remaining dropped/sticky bits make it strictly past the half-way tie.
        let round_bit = 1u32 << (shift - 1);
        if (m & round_bit) != 0 && (m & (3 * round_bit - 1)) != 0 {
            half_mant += 1;
        }
        return sign | half_mant as u16;
    }

    // Normal number. Round the 13 dropped mantissa bits to nearest even.
    let half_mant = (mantissa >> 13) as u16;
    let mut result = sign | ((half_exp as u16) << 10) | half_mant;
    let round_bit = 0x0000_1000u32;
    if (mantissa & round_bit) != 0 && (mantissa & (3 * round_bit - 1)) != 0 {
        // A mantissa carry-out propagates into the exponent (and onward to Inf)
        // because the mantissa field sits directly below the exponent field.
        result += 1;
    }
    result
}
