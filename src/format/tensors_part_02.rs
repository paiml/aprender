
/// Bytes per element for GGML data types (approximate for block types)
fn ggml_dtype_element_size(dtype: u32) -> f64 {
    match dtype {
        0 => 4.0,                // F32
        1 => 2.0,                // F16
        2 => 0.5 + 2.0 / 32.0,   // Q4_0: 4-bit + scale
        3 => 0.5 + 4.0 / 32.0,   // Q4_1: 4-bit + scale + min
        6 => 0.625 + 2.0 / 32.0, // Q5_0
        7 => 0.625 + 4.0 / 32.0, // Q5_1
        8 => 1.0 + 2.0 / 32.0,   // Q8_0
        9 => 1.0 + 4.0 / 32.0,   // Q8_1
        10 => 0.3125,            // Q2_K
        11 => 0.4375,            // Q3_K
        12 => 0.5625,            // Q4_K
        13 => 0.6875,            // Q5_K
        14 => 0.8125,            // Q6_K
        15 => 1.0625,            // Q8_K
        26 => 2.0,               // BF16
        // GGML I-quants (importance matrix quantization)
        16 => 0.5625, // IQ2_XXS
        17 => 0.625,  // IQ2_XS
        18 => 0.6875, // IQ3_XXS
        19 => 0.4375, // IQ1_S
        20 => 0.5625, // IQ4_NL
        21 => 0.4375, // IQ3_S
        22 => 0.625,  // IQ2_S
        23 => 0.5,    // IQ4_XS
        24 => 1.0,    // I8
        25 => 2.0,    // I16
        27 => 4.0,    // I32
        28 => 8.0,    // I64
        29 => 8.0,    // F64
        30 => 0.375,  // IQ1_M
        // Unknown dtype: use F32 size (4 bytes) as conservative estimate.
        // This is intentional — for size estimation purposes, overestimating
        // is safer than underestimating. The dtype name function above will
        // report "unknown" for diagnostics.
        _ => 4.0,
    }
}

/// List tensors from GGUF file bytes
fn list_tensors_gguf(data: &[u8], options: TensorListOptions) -> Result<TensorListResult> {
    let reader = GgufReader::from_bytes(data.to_vec()).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to parse GGUF: {e}"),
    })?;

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
                if let Ok((f32_data, _shape)) = reader.get_tensor_f32(&meta.name) {
                    compute_tensor_stats(&mut info, &f32_data);
                }
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
        let f32_exp = (127 - 15 - e + 1) << 23;
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

    // For APR and GGUF, read into memory and dispatch
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;

    let mut result = list_tensors_from_bytes(&data, options)?;
    result.file = path.display().to_string();

    Ok(result)
}
