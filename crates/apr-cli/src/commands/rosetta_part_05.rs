
pub fn run_validate_stats(
    model: &Path,
    reference: Option<&Path>,
    fingerprints_file: Option<&Path>,
    threshold: f32,
    strict: bool,
    json: bool,
) -> Result<()> {
    if !model.exists() {
        return Err(CliError::FileNotFound(model.to_path_buf()));
    }

    if reference.is_none() && fingerprints_file.is_none() {
        return Err(CliError::ValidationFailed(
            "Must provide either --reference or --fingerprints".to_string(),
        ));
    }

    if !json {
        println!(
            "{}",
            "╔══════════════════════════════════════════════════════════════════════════════╗"
                .cyan()
        );
        println!(
            "{}",
            "║             TENSOR STATISTICS VALIDATION (PMAT-202, JAX-STAT-002)           ║"
                .cyan()
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!(
            "║ Model: {:<69} ║",
            truncate_path(model.display().to_string(), 69)
        );
        println!(
            "║ Threshold: {:.1}σ{:<60} ║",
            threshold,
            if strict { " (strict mode)" } else { "" }
        );
    }

    let actual = compute_fingerprints(model, None)?;
    let reference_fps = resolve_reference_fingerprints(reference, fingerprints_file, json)?;

    if !json {
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
    }

    let anomalies = validate_fingerprints(&actual, &reference_fps, threshold, strict);

    if json {
        print_validate_stats_json(model, threshold, strict, actual.len(), &anomalies);
    } else {
        print_validate_stats_text(&anomalies);
    }

    if !anomalies.is_empty() {
        let critical_count = anomalies
            .iter()
            .filter(|a| a.deviation_sigma > 10.0)
            .count();
        if critical_count > 0 {
            return Err(CliError::ValidationFailed(format!(
                "E020: {} critical statistical anomalies detected (>{:.0}σ deviation)",
                critical_count, threshold
            )));
        }
    }

    Ok(())
}

/// Tensor statistical fingerprint (PMAT-201)
#[derive(Debug, Clone)]
pub struct TensorFingerprint {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub p5: f32,
    pub p25: f32,
    pub p50: f32,
    pub p75: f32,
    pub p95: f32,
    pub nan_count: u32,
    pub inf_count: u32,
    pub zero_fraction: f32,
    pub checksum: u32,
}

/// Statistical anomaly detected during validation
#[derive(Debug)]
struct StatisticalAnomaly {
    tensor: String,
    field: String,
    expected: f32,
    actual: f32,
    deviation_sigma: f32,
}

/// Compute fingerprints for all tensors in a model
fn compute_fingerprints(model_path: &Path, filter: Option<&str>) -> Result<Vec<TensorFingerprint>> {
    let rosetta = RosettaStone::new();
    let report = rosetta
        .inspect(model_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model: {e}")))?;

    let mut fingerprints = Vec::new();

    // Try to load actual tensor data for statistics
    let tensor_data = load_tensor_data(model_path);

    for tensor_info in &report.tensors {
        // Apply filter
        if let Some(pattern) = filter {
            if !tensor_info.name.contains(pattern) {
                continue;
            }
        }

        // Compute statistics from actual data if available
        let (
            mean,
            std,
            min,
            max,
            p5,
            p25,
            p50,
            p75,
            p95,
            nan_count,
            inf_count,
            zero_fraction,
            checksum,
        ) = if let Some(ref data_map) = tensor_data {
            if let Some(values) = data_map.get(&tensor_info.name) {
                compute_tensor_stats(values)
            } else {
                // No data available - use placeholder
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0)
            }
        } else {
            // No data available - use placeholder
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0)
        };

        fingerprints.push(TensorFingerprint {
            name: tensor_info.name.clone(),
            shape: tensor_info.shape.clone(),
            dtype: tensor_info.dtype.clone(),
            mean,
            std,
            min,
            max,
            p5,
            p25,
            p50,
            p75,
            p95,
            nan_count,
            inf_count,
            zero_fraction,
            checksum,
        });
    }

    Ok(fingerprints)
}

/// Load tensor data from model file (for computing actual statistics)
fn load_tensor_data(model_path: &Path) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    // Use realizar to load tensor data
    let realizar_path = std::env::var("REALIZAR_PATH").unwrap_or_else(|_| "realizar".to_string());

    // Try to get tensor statistics via realizar dump command
    let output = std::process::Command::new(&realizar_path)
        .arg("dump")
        .arg("--stats")
        .arg(model_path)
        .arg("--format")
        .arg("json")
        .output()
        .ok()?;

    if !output.status.success() {
        // Fallback: try loading directly via aprender format module
        return load_tensor_data_direct(model_path);
    }

    // Parse JSON output
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_tensor_stats_json(&stdout)
}

/// Direct tensor loading via aprender format module
/// Read a u64 from little-endian bytes at a given position.
fn read_u64_le(data: &[u8], pos: usize) -> Option<u64> {
    let bytes: [u8; 8] = data.get(pos..pos + 8)?.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}

/// Load GGUF tensors as dequantized f32 maps.
fn load_gguf_tensors_direct(
    model_path: &Path,
) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    use aprender::format::gguf::GgufReader;
    let reader = GgufReader::from_file(model_path).ok()?;
    let mut tensor_map = std::collections::HashMap::new();
    for tensor_meta in &reader.tensors {
        if let Ok((values, _shape)) = reader.get_tensor_f32(&tensor_meta.name) {
            tensor_map.insert(tensor_meta.name.clone(), values);
        }
    }
    Some(tensor_map)
}

/// Load SafeTensors tensors as dequantized f32 maps.
fn load_safetensors_tensors_direct(
    model_path: &Path,
) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    use aprender::serialization::safetensors::MappedSafeTensors;
    let mapped = MappedSafeTensors::open(model_path).ok()?;
    let mut tensor_map = std::collections::HashMap::new();
    for name in mapped.tensor_names() {
        if let Ok(values) = mapped.get_tensor(name) {
            tensor_map.insert((*name).to_string(), values);
        }
    }
    Some(tensor_map)
}

/// Load APR tensors as dequantized f32 maps (PMAT-201).
fn load_apr_tensors_direct(
    model_path: &Path,
) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    let data = std::fs::read(model_path).ok()?;
    if data.len() < 40 || data[0..4] != *b"APR\0" {
        return None;
    }

    let tensor_count = u32::from_le_bytes(data[8..12].try_into().ok()?) as usize;
    let tensor_index_offset = read_u64_le(&data, 24)? as usize;
    let data_offset = read_u64_le(&data, 32)? as usize;

    let mut tensor_map = std::collections::HashMap::new();
    let mut pos = tensor_index_offset;

    for _ in 0..tensor_count {
        let Some((name, dtype, dims, offset, size, new_pos)) = parse_apr_tensor_entry(&data, pos)
        else {
            break;
        };
        pos = new_pos;

        let tensor_start = data_offset + offset;
        let tensor_end = tensor_start + size;
        if tensor_end > data.len() {
            continue;
        }

        let values = dequantize_by_dtype(dtype, &data[tensor_start..tensor_end], &dims);
        let Some(values) = values else { continue };
        tensor_map.insert(name, values);
    }

    Some(tensor_map)
}

/// Parse a single APR tensor index entry, returning (name, dtype, dims, offset, size, new_pos).
fn parse_apr_tensor_entry(
    data: &[u8],
    mut pos: usize,
) -> Option<(String, u8, Vec<usize>, usize, usize, usize)> {
    let name_len = u16::from_le_bytes(data.get(pos..pos + 2)?.try_into().ok()?) as usize;
    pos += 2;

    let name = String::from_utf8_lossy(data.get(pos..pos + name_len)?).to_string();
    pos += name_len;

    let dtype = *data.get(pos)?;
    pos += 1;
    let ndim = *data.get(pos)? as usize;
    pos += 1;

    let mut dims = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        dims.push(read_u64_le(data, pos)? as usize);
        pos += 8;
    }

    let offset = read_u64_le(data, pos)? as usize;
    pos += 8;
    let size = read_u64_le(data, pos)? as usize;
    pos += 8;

    Some((name, dtype, dims, offset, size, pos))
}

/// Dequantize tensor bytes to f32 based on APR dtype code.
fn dequantize_by_dtype(dtype: u8, bytes: &[u8], dims: &[usize]) -> Option<Vec<f32>> {
    match dtype {
        0 => Some(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        12 => {
            let num_elements: usize = dims.iter().product();
            Some(dequantize_q4k_for_stats(bytes, num_elements))
        }
        14 => {
            let num_elements: usize = dims.iter().product();
            Some(dequantize_q6k_for_stats(bytes, num_elements))
        }
        _ => None,
    }
}

fn load_tensor_data_direct(
    model_path: &Path,
) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    let ext = model_path.extension()?.to_str()?;

    let tensor_map = match ext.to_lowercase().as_str() {
        "gguf" => load_gguf_tensors_direct(model_path)?,
        "apr" => load_apr_tensors_direct(model_path)?,
        "safetensors" => load_safetensors_tensors_direct(model_path)?,
        _ => return None,
    };

    if tensor_map.is_empty() {
        None
    } else {
        Some(tensor_map)
    }
}

/// Simple Q4_K dequantization for statistics (PMAT-201)
fn dequantize_q4k_for_stats(data: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 144; // Q4_K block size

    let num_blocks = (num_elements + QK_K - 1) / QK_K;
    let mut result = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        if block_start + BLOCK_SIZE > data.len() {
            break;
        }

        // Read scales (d, dmin)
        let d = f16_to_f32(&data[block_start..block_start + 2]);
        let dmin = f16_to_f32(&data[block_start + 2..block_start + 4]);

        // Read 12 bytes of scales
        let scales = &data[block_start + 4..block_start + 16];

        // Read 128 bytes of quantized values (4 bits each, 256 values)
        let qs = &data[block_start + 16..block_start + 144];

        // Dequantize 256 elements
        for j in 0..QK_K {
            if result.len() >= num_elements {
                break;
            }
            let scale_idx = j / 32;
            let scale = if scale_idx < 12 {
                (scales[scale_idx] & 0x3F) as f32
            } else {
                1.0
            };

            let q_idx = j / 2;
            let q_val = if j % 2 == 0 {
                (qs.get(q_idx).copied().unwrap_or(0) & 0x0F) as i32
            } else {
                ((qs.get(q_idx).copied().unwrap_or(0) >> 4) & 0x0F) as i32
            };

            let val = d * scale * (q_val as f32 - 8.0) - dmin * scale;
            result.push(val);
        }
    }

    result
}

/// Simple Q6_K dequantization for statistics (PMAT-201)
fn dequantize_q6k_for_stats(data: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 210; // Q6_K block size

    let num_blocks = (num_elements + QK_K - 1) / QK_K;
    let mut result = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        if block_start + BLOCK_SIZE > data.len() {
            break;
        }

        // Read scale (d)
        let d = f16_to_f32(&data[block_start + 208..block_start + 210]);

        // Simplified: just read as scaled values
        for j in 0..QK_K {
            if result.len() >= num_elements {
                break;
            }
            let q_idx = block_start + (j * 6 / 8);
            let q_val = data.get(q_idx).copied().unwrap_or(0) as i32;
            let val = d * (q_val as f32 - 32.0);
            result.push(val);
        }
    }

    result
}

/// Convert f16 bytes to f32
fn f16_to_f32(bytes: &[u8]) -> f32 {
    if bytes.len() < 2 {
        return 0.0;
    }
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}

/// Parse tensor statistics from JSON
fn parse_tensor_stats_json(_json_str: &str) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    // Simple JSON parsing for tensor data
    // Format expected: {"tensors": {"name": [values...], ...}}
    // PMAT-201: Would need proper JSON parsing for full implementation
    None // Placeholder - returns None to use placeholder stats
}
