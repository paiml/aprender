
fn validate_and_read_header(bytes: &[u8]) -> Result<usize, String> {
    if bytes.len() < 8 {
        return Err(format!(
            "Invalid SafeTensors file: file is {} bytes, need at least 8 bytes for header",
            bytes.len()
        ));
    }

    let header_bytes: [u8; 8] = bytes[0..8]
        .try_into()
        .map_err(|_| "Failed to read header bytes".to_string())?;
    let metadata_len = u64::from_le_bytes(header_bytes) as usize;

    if metadata_len == 0 {
        return Err("Invalid SafeTensors file: metadata length is 0".to_string());
    }

    if 8 + metadata_len > bytes.len() {
        return Err(format!(
            "Invalid SafeTensors file: metadata length {metadata_len} exceeds file size"
        ));
    }

    Ok(metadata_len)
}

fn parse_metadata(
    bytes: &[u8],
    metadata_len: usize,
) -> Result<(SafeTensorsMetadata, UserMetadata), String> {
    let metadata_json = &bytes[8..8 + metadata_len];
    let metadata_str = std::str::from_utf8(metadata_json)
        .map_err(|e| format!("Metadata is not valid UTF-8: {e}"))?;

    let raw_metadata: serde_json::Value =
        serde_json::from_str(metadata_str).map_err(|e| format!("JSON parsing failed: {e}"))?;

    let serde_json::Value::Object(map) = raw_metadata else {
        return Ok((SafeTensorsMetadata::new(), UserMetadata::new()));
    };

    let mut metadata = SafeTensorsMetadata::new();
    let mut user_metadata = UserMetadata::new();

    for (key, value) in map {
        if key == "__metadata__" {
            // PMAT-223: Extract user metadata instead of discarding it
            extract_user_metadata(value, &mut user_metadata);
            continue;
        }
        if key.starts_with("__") {
            continue;
        }
        if let Ok(tensor_meta) = serde_json::from_value::<TensorMetadata>(value) {
            metadata.insert(key, tensor_meta);
        }
    }

    Ok((metadata, user_metadata))
}

/// Extracts string key-value pairs from a `__metadata__` JSON object into `UserMetadata`.
fn extract_user_metadata(value: serde_json::Value, user_metadata: &mut UserMetadata) {
    let serde_json::Value::Object(meta_map) = value else {
        return;
    };
    for (mk, mv) in meta_map {
        if let serde_json::Value::String(s) = mv {
            user_metadata.insert(mk, s);
        }
    }
}

/// Extracts a tensor from raw `SafeTensors` data.
///
/// # Arguments
///
/// * `raw_data` - Raw tensor bytes from `SafeTensors` file
/// * `tensor_meta` - Metadata for the tensor to extract
///
/// # Returns
///
/// Vector of F32 values (BF16/F16 are converted to F32)
///
/// # Errors
///
/// Returns an error if:
/// - Data offsets are invalid
/// - Data size doesn't match dtype requirements
/// - Unsupported dtype
pub fn extract_tensor(raw_data: &[u8], tensor_meta: &TensorMetadata) -> Result<Vec<f32>, String> {
    let [start, end] = tensor_meta.data_offsets;

    // Validate offsets
    if end > raw_data.len() {
        return Err(format!(
            "Invalid data offset: end={} exceeds data size={}",
            end,
            raw_data.len()
        ));
    }

    if start >= end {
        return Err(format!("Invalid data offset: start={start} >= end={end}"));
    }

    // Extract bytes
    let tensor_bytes = &raw_data[start..end];

    // Handle different dtypes
    match tensor_meta.dtype.as_str() {
        "F32" => extract_f32(tensor_bytes),
        "BF16" => extract_bf16_to_f32(tensor_bytes),
        "F16" => extract_f16_to_f32(tensor_bytes),
        other => Err(format!(
            "Unsupported dtype: {other}. Supported: F32, BF16, F16"
        )),
    }
}

/// Extract F32 tensor data
fn extract_f32(tensor_bytes: &[u8]) -> Result<Vec<f32>, String> {
    if tensor_bytes.len() % 4 != 0 {
        return Err(format!(
            "Invalid F32 tensor data: size {} is not a multiple of 4 bytes",
            tensor_bytes.len()
        ));
    }

    let values: Vec<f32> = tensor_bytes
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().expect("chunk is 4 bytes");
            f32::from_le_bytes(bytes)
        })
        .collect();

    Ok(values)
}

/// Extract BF16 tensor data and convert to F32
fn extract_bf16_to_f32(tensor_bytes: &[u8]) -> Result<Vec<f32>, String> {
    if tensor_bytes.len() % 2 != 0 {
        return Err(format!(
            "Invalid BF16 tensor data: size {} is not a multiple of 2 bytes",
            tensor_bytes.len()
        ));
    }

    let values: Vec<f32> = tensor_bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bytes: [u8; 2] = chunk.try_into().expect("chunk is 2 bytes");
            bf16_to_f32(bytes)
        })
        .collect();

    Ok(values)
}

/// Extract F16 tensor data and convert to F32
fn extract_f16_to_f32(tensor_bytes: &[u8]) -> Result<Vec<f32>, String> {
    if tensor_bytes.len() % 2 != 0 {
        return Err(format!(
            "Invalid F16 tensor data: size {} is not a multiple of 2 bytes",
            tensor_bytes.len()
        ));
    }

    let values: Vec<f32> = tensor_bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bytes: [u8; 2] = chunk.try_into().expect("chunk is 2 bytes");
            f16_to_f32(bytes)
        })
        .collect();

    Ok(values)
}

/// Convert BF16 (Brain Float 16) to F32
///
/// BF16 has the same exponent range as F32 (8 bits) but only 7 mantissa bits.
/// Conversion is done by zero-padding the mantissa.
#[inline]
fn bf16_to_f32(bytes: [u8; 2]) -> f32 {
    // BF16 is the upper 16 bits of an F32
    let bits = u32::from_le_bytes([0, 0, bytes[0], bytes[1]]);
    f32::from_bits(bits)
}

/// Convert F16 (IEEE 754 half-precision) to F32
///
/// F16 has 5 exponent bits and 10 mantissa bits.
#[inline]
fn f16_to_f32(bytes: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(bytes);

    // Extract components
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            // Zero (positive or negative)
            u32::from(sign) << 31
        } else {
            // Subnormal: convert to normalized F32
            let mut m = mant;
            let mut e: i32 = -14; // F16 subnormal exponent bias adjustment
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF; // Remove implicit 1
            let exp32 = (e + 127) as u32; // F32 bias
            (u32::from(sign) << 31) | (exp32 << 23) | (u32::from(m) << 13)
        }
    } else if exp == 31 {
        // Inf or NaN
        let mant32 = u32::from(mant) << 13;
        (u32::from(sign) << 31) | (0xFF << 23) | mant32
    } else {
        // Normal number
        // GH-205 FIX: Rearrange to avoid underflow in debug mode
        // F16 bias is 15, F32 bias is 127, so we add 112 (127 - 15)
        let exp32 = u32::from(exp) + 112; // Was: exp - 15 + 127 (underflows if exp < 15)
        let mant32 = u32::from(mant) << 13;
        (u32::from(sign) << 31) | (exp32 << 23) | mant32
    };

    f32::from_bits(f32_bits)
}
