/// Bytes per element for GGML data types (table lookup, O(1)).
///
/// Block-quantized types use exact bytes-per-element = block_bytes / block_elems.
/// K-quant super-blocks pack QK_K=256 elements (ggml-common.h):
///   Q2_K=84, Q3_K=110, Q4_K=144, Q5_K=176, Q6_K=210, Q8_K=292 bytes
///   → 84/256, 110/256, 144/256, 176/256, 210/256, 292/256 (all exact dyadic).
/// Unknown dtypes default to 4.0 (F32 size) as a conservative overestimate.
/// See contracts/gguf-kquant-element-size-v1.yaml (PMAT-869).
fn ggml_dtype_element_size(dtype: u32) -> f64 {
    // Index: [F32, F16, Q4_0, Q4_1, (4), (5), Q5_0, Q5_1, Q8_0, Q8_1,
    //         Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ2_XXS, IQ2_XS,
    //         IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, I8, I16,
    //         BF16, I32, I64, F64, IQ1_M]
    const SIZES: [f64; 31] = [
        4.0, 2.0, 0.5625, 0.625, 4.0, 4.0, 0.6875, 0.75, 1.0625, 1.125, 0.328_125, 0.429_687_5,
        0.5625, 0.6875, 0.820_312_5, 1.140_625, 0.5625, 0.625, 0.6875, 0.4375, 0.5625, 0.4375,
        0.625, 0.5, 1.0, 2.0, 2.0, 4.0, 8.0, 8.0, 0.375,
    ];
    SIZES.get(dtype as usize).copied().unwrap_or(4.0)
}

/// List tensors from GGUF file bytes
fn list_tensors_gguf(data: &[u8], options: TensorListOptions) -> Result<TensorListResult> {
    let reader = GgufReader::from_bytes(data.to_vec()).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to parse GGUF: {e}"),
    })?;

    // #2569: every row below asserts that `size_bytes` of tensor data exist at a
    // declared offset. Prove that before printing it. Run over ALL tensors, ahead
    // of the `--filter` loop, so a filtered listing cannot hide a truncation.
    let extents: Vec<(&str, u64, u64)> = reader
        .tensors
        .iter()
        .map(|meta| {
            let num_elements: u64 = meta.dims.iter().product();
            let size_bytes = (num_elements as f64 * ggml_dtype_element_size(meta.dtype)) as u64;
            (meta.name.as_str(), meta.offset, size_bytes)
        })
        .collect();
    check_tensor_table_fits("GGUF", data.len() as u64, reader.data_offset as u64, extents)?;

    let mut tensors = Vec::new();
    let mut total_size = 0usize;
    let mut total_matching = 0usize;

    for meta in &reader.tensors {
        // Apply filter
        if let Some(ref pattern) = options.filter {
            if !meta.name.contains(pattern.as_str()) {
                continue;
            }
        }

        let shape: Vec<usize> = meta.dims.iter().map(|&d| d as usize).collect();
        let num_elements: usize = shape.iter().product();
        let size_bytes = (num_elements as f64 * ggml_dtype_element_size(meta.dtype)) as usize;

        total_size += size_bytes;
        total_matching += 1;

        // Only collect details up to the limit
        if tensors.len() < options.limit {
            let mut info = TensorInfo {
                name: meta.name.clone(),
                shape,
                dtype: ggml_dtype_name(meta.dtype).to_string(),
                size_bytes,
                mean: None,
                std: None,
                min: None,
                max: None,
                nan_count: None,
                inf_count: None,
            };

            if options.compute_stats {
                // #2569: this was `if let Ok(...)`, so a tensor whose data could not
                // be read printed em-dashes in the mean/std/range columns — byte-for-byte
                // what a run WITHOUT `--stats` prints. The user asked for statistics;
                // "could not compute them" and "you did not ask" must not render the
                // same. Fail closed and name the tensor and the reason.
                let (f32_data, _shape) = reader.get_tensor_f32(&meta.name).map_err(|e| {
                    AprenderError::FormatError {
                        message: format!(
                            "--stats requested but tensor '{}' ({}) could not be read: {e}",
                            meta.name,
                            ggml_dtype_name(meta.dtype)
                        ),
                    }
                })?;
                compute_tensor_stats(&mut info, &f32_data);
            }

            tensors.push(info);
        }
    }

    Ok(TensorListResult {
        file: String::new(),
        format_version: format!("GGUF v{}", reader.version),
        tensor_count: total_matching,
        total_size_bytes: total_size,
        tensors,
    })
}

// ============================================================================
// SafeTensors Format Support (PMAT-ROSETTA-001)
// ============================================================================

/// Parse and validate the SafeTensors JSON header, returning the parsed header
/// as a `serde_json::Value` (guaranteed to be an object) and the byte offset
/// where tensor data begins.
fn parse_safetensors_header(data: &[u8]) -> Result<(serde_json::Value, usize)> {
    if data.len() < 8 {
        return Err(AprenderError::FormatError {
            message: "SafeTensors file too small".to_string(),
        });
    }

    let header_len =
        u64::from_le_bytes(
            data[0..8]
                .try_into()
                .map_err(|_| AprenderError::FormatError {
                    message: "Failed to read SafeTensors header length".to_string(),
                })?,
        ) as usize;

    if data.len() < 8 + header_len {
        return Err(AprenderError::FormatError {
            message: "SafeTensors file truncated (header extends past EOF)".to_string(),
        });
    }

    let header_json = &data[8..8 + header_len];
    let header: serde_json::Value =
        serde_json::from_slice(header_json).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to parse SafeTensors JSON header: {e}"),
        })?;

    if !header.is_object() {
        return Err(AprenderError::FormatError {
            message: "SafeTensors header is not a JSON object".to_string(),
        });
    }

    let data_start = 8 + header_len;
    Ok((header, data_start))
}

/// Extract a `TensorInfo` from a SafeTensors JSON tensor entry.
/// Returns the info and the relative byte offsets `(start, end)` within the
/// data section (if present in the entry).
fn extract_safetensors_tensor_info(
    name: &str,
    value: &serde_json::Value,
) -> (TensorInfo, Option<(usize, usize)>) {
    let dtype = value
        .get("dtype")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let shape: Vec<usize> = value
        .get("shape")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default();

    let relative_offsets = value
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            let start = arr.first()?.as_u64()? as usize;
            let end = arr.get(1)?.as_u64()? as usize;
            Some((start, end))
        });

    let size_bytes = relative_offsets
        .map(|(start, end)| end - start)
        .unwrap_or(0);

    let info = TensorInfo {
        name: name.to_string(),
        shape,
        dtype,
        size_bytes,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };

    (info, relative_offsets)
}

/// Compute and populate stats on a `TensorInfo` from its SafeTensors byte
/// range. `data` is the full file buffer; `data_start` is the byte offset
/// where the tensor data section begins; `relative_offsets` are
/// `(start, end)` relative to that section.
fn populate_safetensors_stats(
    info: &mut TensorInfo,
    data: &[u8],
    data_start: usize,
    relative_offsets: (usize, usize),
) {
    let (start, end) = relative_offsets;
    let abs_start = data_start + start;
    let abs_end = data_start + end;
    if abs_end > data.len() {
        return;
    }
    let tensor_bytes = &data[abs_start..abs_end];
    let f32_data = safetensors_bytes_to_f32(tensor_bytes, &info.dtype);
    compute_tensor_stats(info, &f32_data);
}

/// Check whether a tensor name passes the optional filter pattern.
fn matches_filter(name: &str, filter: Option<&String>) -> bool {
    match filter {
        Some(pattern) => name.contains(pattern.as_str()),
        None => true,
    }
}

/// List tensors from SafeTensors file bytes by parsing the JSON header
fn list_tensors_safetensors(data: &[u8], options: TensorListOptions) -> Result<TensorListResult> {
    let (header, data_start) = parse_safetensors_header(data)?;

    // Safety: parse_safetensors_header validated this is an object
    let obj = header
        .as_object()
        .expect("parse_safetensors_header guarantees object");

    let mut tensors = Vec::new();
    let mut total_size = 0usize;
    let mut total_matching = 0usize;

    // Collect and sort tensor names for deterministic output
    let mut tensor_entries: Vec<(&String, &serde_json::Value)> =
        obj.iter().filter(|(k, _)| *k != "__metadata__").collect();
    tensor_entries.sort_by_key(|(k, _)| *k);

    for (name, value) in tensor_entries {
        if !matches_filter(name, options.filter.as_ref()) {
            continue;
        }

        let (mut info, relative_offsets) = extract_safetensors_tensor_info(name, value);

        total_size += info.size_bytes;
        total_matching += 1;

        if tensors.len() >= options.limit {
            continue;
        }

        if options.compute_stats {
            if let Some(offsets) = relative_offsets {
                populate_safetensors_stats(&mut info, data, data_start, offsets);
            }
        }

        tensors.push(info);
    }

    Ok(TensorListResult {
        file: String::new(),
        format_version: "SafeTensors".to_string(),
        tensor_count: total_matching,
        total_size_bytes: total_size,
        tensors,
    })
}

/// Convert SafeTensors raw bytes to f32 based on dtype
fn safetensors_bytes_to_f32(bytes: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F32" => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "F16" => bytes
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f16_to_f32(bits)
            })
            .collect(),
        "BF16" => bytes
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                bf16_to_f32(bits)
            })
            .collect(),
        _ => Vec::new(), // Unknown dtype, skip stats
    }
}

/// Convert IEEE 754 half-precision float to f32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Denormalized: convert to normalized f32
        let mut e = 1u32;
        let mut m = mantissa;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        // PMAT-843: `e` starts at 1, so the reconstructed f32 exponent needs a
        // +2 bias correction (was +1); without it every subnormal is halved.
        let f32_exp = (127 - 15 - e + 2) << 23;
        let f32_mant = (m & 0x3FF) << 13;
        f32::from_bits(sign | f32_exp | f32_mant)
    } else if exponent == 31 {
        // Inf/NaN
        let f32_exp = 0xFF << 23;
        let f32_mant = mantissa << 13;
        f32::from_bits(sign | f32_exp | f32_mant)
    } else {
        let f32_exp = (exponent + 127 - 15) << 23;
        let f32_mant = mantissa << 13;
        f32::from_bits(sign | f32_exp | f32_mant)
    }
}

/// Convert BFloat16 to f32 (simple: just shift left by 16)
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ============================================================================
// Path-Based Format Dispatch (PMAT-ROSETTA-001)
// ============================================================================

/// Convert tensor index entry to TensorInfo
fn tensor_info_from_entry(entry: &TensorIndexEntry) -> TensorInfo {
    TensorInfo {
        name: entry.name.clone(),
        shape: entry.shape.clone(),
        dtype: entry.dtype.name().to_string(),
        size_bytes: entry.size as usize,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    }
}

// ============================================================================
// Tensor Listing - From File
// ============================================================================

/// List tensors from a model file (APR, GGUF, or SafeTensors)
///
/// Uses magic byte detection for reliable format identification,
/// then delegates to the appropriate format-specific reader.
///
/// # Arguments
/// * `path` - Path to model file
/// * `options` - Listing options
///
/// # Errors
/// Returns error if the file doesn't exist or is invalid.
pub fn list_tensors(
    path: impl AsRef<Path>,
    options: TensorListOptions,
) -> Result<TensorListResult> {
    let path = path.as_ref();

    // For SafeTensors, prefer MappedSafeTensors (mmap-based, handles large files)
    if let Ok(FormatType::SafeTensors) = FormatType::from_magic(path) {
        let mut result = list_tensors_safetensors_path(path, options)?;
        result.file = path.display().to_string();
        return Ok(result);
    }

    // For APR v2, use mmap + AprV2ReaderRef (realizar#136 — no full-file read)
    if let Ok(FormatType::Apr) = FormatType::from_magic(path) {
        let mut magic = [0u8; 4];
        let mut f = File::open(path)?;
        std::io::Read::read_exact(&mut f, &mut magic)?;
        drop(f);
        if &magic == b"APR\0" {
            let mapped = crate::bundle::MappedFile::open(path)?;
            let mut result = list_tensors_v2_mmap(mapped.as_slice(), options)?;
            result.file = path.display().to_string();
            return Ok(result);
        }
    }

    // For GGUF and APR v1, read into memory and dispatch
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;

    let mut result = list_tensors_from_bytes(&data, options)?;
    result.file = path.display().to_string();

    Ok(result)
}

#[cfg(test)]
mod f16_tests {
    use super::f16_to_f32;

    /// Self-contained, bit-exact IEEE-754 binary16 -> binary32 reference
    /// (golden oracle). Verified bit-identical to `half::f16::to_f32()` across
    /// all 65536 patterns (NaN excluded). PMAT-843.
    fn golden_f16_to_f32(bits: u16) -> f32 {
        let sign = (bits >> 15) & 1;
        let exp = i32::from((bits >> 10) & 0x1F);
        let man = u32::from(bits & 0x3FF);
        let s = (sign as f32).mul_add(-2.0, 1.0); // +1.0 if sign=0, -1.0 if sign=1
        if exp == 0 {
            // zero or subnormal: value = mantissa * 2^-24
            s * (man as f32) * 2f32.powi(-24)
        } else if exp == 0x1F {
            if man == 0 {
                if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            }
        } else {
            // normal: value = (1 + mantissa/1024) * 2^(exp-15)
            s * (1.0 + (man as f32) / 1024.0) * 2f32.powi(exp - 15)
        }
    }

    /// PMAT-843 falsifier: smallest positive subnormal must NOT be halved.
    /// RED (buggy): 0x33000000 (2.9802322e-8). GREEN: 0x33800000 (5.9604645e-8).
    #[test]
    fn smallest_subnormal_not_halved() {
        let got = f16_to_f32(0x0001).to_bits();
        assert_eq!(
            got, 0x3380_0000,
            "f16_to_f32(0x0001) = {got:#010x}, expected 0x33800000 (5.9604645e-8)"
        );
    }

    /// PMAT-843 strong falsifier: every f16 bit pattern (NaN excluded) must
    /// convert bit-exactly to the golden oracle. RED count (buggy) = 2046
    /// (all subnormals halved); GREEN = 0.
    #[test]
    fn all_bit_patterns_match_golden() {
        let mut mismatches = 0u32;
        for bits in 0..=u16::MAX {
            let exp = (bits >> 10) & 0x1F;
            let man = bits & 0x3FF;
            if exp == 0x1F && man != 0 {
                continue; // skip NaN (bit pattern not canonical)
            }
            if f16_to_f32(bits).to_bits() != golden_f16_to_f32(bits).to_bits() {
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches, 0,
            "{mismatches} f16->f32 conversions disagree with golden oracle"
        );
    }
}
