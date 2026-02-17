
fn print_gguf_tensor_hex(
    opts: &HexOptions,
    tensor: &GgufTensorEntry,
    info: &GgufInfo,
) -> Result<(), CliError> {
    println!("{}", "═".repeat(70).dimmed());
    println!("{}: {}", "Tensor".bold(), tensor.name.cyan());
    println!("{}", "═".repeat(70).dimmed());

    let dims_str: Vec<String> = tensor
        .dims
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let num_elements: u64 = tensor.dims.iter().product();
    println!(
        "{}: [{}] = {} elements",
        "Shape".bold(),
        dims_str.join(", ").white(),
        output::count_fmt(num_elements as usize).green()
    );
    println!(
        "{}: {} ({})",
        "Dtype".bold(),
        output::dtype_color(ggml_dtype_name(tensor.dtype)),
        format!("{}", tensor.dtype).dimmed()
    );
    println!(
        "{}: {} {}",
        "Offset".bold(),
        format!("0x{:X}", info.data_offset + tensor.offset as usize).cyan(),
        format!("(data section + 0x{:X})", tensor.offset).dimmed()
    );

    match get_gguf_tensor_f32(&opts.file, &tensor.name) {
        Ok((data, _shape)) => {
            if opts.stats {
                print_tensor_stats(&data);
            }
            print_hex_dump(&data, opts.limit);
        }
        Err(e) => {
            println!("  {} Cannot dequantize: {e}", "Note:".yellow());
        }
    }
    Ok(())
}

// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn list_gguf_tensors(
    tensors: &[GgufTensorEntry],
    filter: Option<&str>,
    json_output: bool,
) -> Result<(), CliError> {
    let filtered: Vec<&GgufTensorEntry> = tensors
        .iter()
        .filter(|t| filter.map_or(true, |f| t.name.contains(f)))
        .collect();

    if json_output {
        let names: Vec<&str> = filtered.iter().map(|t| t.name.as_str()).collect();
        let json = serde_json::json!({
            "tensors": names,
            "count": filtered.len()
        });
        if let Ok(s) = serde_json::to_string_pretty(&json) {
            println!("{s}");
        }
    } else {
        println!("{}", "Tensors:".bold());
        for tensor in &filtered {
            let dims_str: Vec<String> = tensor
                .dims
                .iter()
                .map(std::string::ToString::to_string)
                .collect();
            println!(
                "  {} {} {}",
                tensor.name.cyan(),
                output::dtype_color(ggml_dtype_name(tensor.dtype)),
                format!("[{}]", dims_str.join(", ")).dimmed()
            );
        }
        println!("\n{} tensors total", filtered.len().to_string().cyan());
    }
    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn output_gguf_json(
    path: &Path,
    tensors: &[&GgufTensorEntry],
    limit: usize,
    show_stats: bool,
) -> Result<(), CliError> {
    use serde::Serialize;

    #[derive(Serialize)]
    struct TensorDump {
        name: String,
        dims: Vec<u64>,
        dtype: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        stats: Option<JsonStats>,
        sample_values: Vec<f32>,
    }

    #[derive(Serialize)]
    struct JsonStats {
        min: f32,
        max: f32,
        mean: f32,
        std: f32,
    }

    let mut results = Vec::new();
    for tensor in tensors {
        let data = get_gguf_tensor_f32(path, &tensor.name).ok();
        let stats = if show_stats {
            data.as_ref().map(|(d, _)| {
                let (min, max, mean, std) = compute_stats(d);
                JsonStats {
                    min,
                    max,
                    mean,
                    std,
                }
            })
        } else {
            None
        };

        let sample_values = data
            .as_ref()
            .map(|(d, _)| d.iter().take(limit).copied().collect())
            .unwrap_or_default();

        results.push(TensorDump {
            name: tensor.name.clone(),
            dims: tensor.dims.clone(),
            dtype: ggml_dtype_name(tensor.dtype).to_string(),
            stats,
            sample_values,
        });
    }

    if let Ok(json) = serde_json::to_string_pretty(&results) {
        println!("{json}");
    }
    Ok(())
}

// ============================================================================
// SafeTensors mode
// ============================================================================

/// Parsed SafeTensors header info.
struct SafeTensorsHeader {
    header_len: usize,
    header: serde_json::Value,
}

fn parse_safetensors_header(bytes: &[u8]) -> Result<SafeTensorsHeader, CliError> {
    if bytes.len() < 9 {
        return Err(CliError::InvalidFormat(
            "SafeTensors file too small".to_string(),
        ));
    }
    let header_len = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]) as usize;
    if 8 + header_len > bytes.len() {
        return Err(CliError::InvalidFormat(
            "SafeTensors header length exceeds file size".to_string(),
        ));
    }
    let header_json = std::str::from_utf8(&bytes[8..8 + header_len])
        .map_err(|e| CliError::InvalidFormat(format!("Invalid SafeTensors header UTF-8: {e}")))?;
    let header: serde_json::Value = serde_json::from_str(header_json)
        .map_err(|e| CliError::InvalidFormat(format!("Invalid SafeTensors JSON: {e}")))?;
    Ok(SafeTensorsHeader { header_len, header })
}

#[allow(clippy::redundant_closure_for_method_calls)]
fn run_safetensors(opts: &HexOptions, bytes: &[u8]) -> Result<(), CliError> {
    let parsed = parse_safetensors_header(bytes)?;
    let header_len = parsed.header_len;

    let tensor_map = parsed.header.as_object().ok_or_else(|| {
        CliError::InvalidFormat("SafeTensors header is not a JSON object".to_string())
    })?;

    let tensor_names: Vec<&String> = tensor_map.keys().filter(|k| *k != "__metadata__").collect();

    output::header(&format!(
        "SafeTensors Binary Forensics: {}",
        opts.file.display()
    ));
    output::metric("Tensors", output::count_fmt(tensor_names.len()), "");
    output::metric("Header size", output::format_size(header_len as u64), "");
    output::metric("File size", output::format_size(bytes.len() as u64), "");
    output::metric("Data offset", format!("0x{:X}", 8 + header_len), "");

    if opts.list {
        return list_safetensor_names(&tensor_names, tensor_map);
    }
    if opts.contract {
        println!(
            "{}",
            output::badge_info("Layout contract not applicable for SafeTensors")
        );
        return Ok(());
    }
    if opts.blocks {
        println!(
            "{}",
            output::badge_info("Block view not applicable for SafeTensors (no quantization)")
        );
        return Ok(());
    }

    // Show filtered tensor info
    let filter = opts.tensor.as_deref();
    let matching: Vec<&&String> = tensor_names
        .iter()
        .filter(|n| filter.map_or(true, |f| n.contains(f)))
        .collect();

    if matching.is_empty() {
        println!("\n{}", "No tensors match the filter pattern".yellow());
        return Ok(());
    }

    for name in &matching {
        if let Some(info) = tensor_map.get(name.as_str()) {
            print_safetensor_entry(name, info, bytes, header_len, opts);
        }
    }
    Ok(())
}

#[allow(clippy::redundant_closure_for_method_calls)]
fn list_safetensor_names(
    names: &[&String],
    tensor_map: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), CliError> {
    println!("\n{}", "Tensors:".bold());
    for name in names {
        if let Some(info) = tensor_map.get(name.as_str()) {
            let dtype = info.get("dtype").and_then(|v| v.as_str()).unwrap_or("?");
            let shape = info
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64())
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            println!("  {} {} [{}]", name, output::dtype_color(dtype), shape);
        }
    }
    println!("\n{} tensors total", names.len().to_string().cyan());
    Ok(())
}

/// Print a single SafeTensors tensor entry with its hex dump.
fn print_safetensor_entry(
    name: &str,
    info: &serde_json::Value,
    bytes: &[u8],
    header_len: usize,
    opts: &HexOptions,
) {
    println!("\n{}", "═".repeat(70));
    println!("{}: {}", "Tensor".bold(), name.cyan());

    let (Some(dtype), Some(shape), Some(offsets)) = (
        info.get("dtype").and_then(serde_json::Value::as_str),
        info.get("shape").and_then(serde_json::Value::as_array),
        info.get("data_offsets")
            .and_then(serde_json::Value::as_array),
    ) else {
        return;
    };

    let shape_str: Vec<String> = shape
        .iter()
        .filter_map(serde_json::Value::as_u64)
        .map(|d| d.to_string())
        .collect();
    let num_elements: u64 = shape.iter().filter_map(serde_json::Value::as_u64).product();
    println!(
        "{}: [{}] = {} elements",
        "Shape".bold(),
        shape_str.join(", "),
        output::count_fmt(num_elements as usize).green()
    );
    println!("{}: {}", "Dtype".bold(), output::dtype_color(dtype));

    let (Some(start), Some(end)) = (
        offsets.first().and_then(serde_json::Value::as_u64),
        offsets.get(1).and_then(serde_json::Value::as_u64),
    ) else {
        return;
    };

    let abs_start = 8 + header_len + start as usize;
    let abs_end = 8 + header_len + end as usize;
    println!(
        "{}: 0x{:X}..0x{:X} ({} bytes)",
        "Offset".bold(),
        abs_start,
        abs_end,
        output::format_size(end - start)
    );

    if abs_end > bytes.len() {
        return;
    }
    let tensor_bytes = &bytes[abs_start..abs_end];

    let f32_data: Option<Vec<f32>> = match dtype {
        "F32" => Some(
            tensor_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        "F16" => Some(
            tensor_bytes
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        ),
        "BF16" => Some(
            tensor_bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u32::from_le_bytes([0, 0, c[0], c[1]]);
                    f32::from_bits(bits)
                })
                .collect(),
        ),
        _ => None,
    };

    if let Some(ref data) = f32_data {
        if opts.stats {
            print_tensor_stats(data);
        }
        if opts.distribution {
            let analysis = compute_distribution(data);
            print_distribution(&analysis);
        }
        print_hex_dump(data, opts.limit);
    }
}

// ============================================================================
// --slice: Extract a range of elements from a tensor
// ============================================================================

/// Parse a slice string like "0:3" into (start, end).
fn parse_slice(s: &str) -> Result<(usize, usize), CliError> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(CliError::InvalidFormat(
            format!("Invalid slice format '{s}': expected 'start:end' (e.g., 0:3)"),
        ));
    }
    let start: usize = parts[0]
        .parse()
        .map_err(|e| CliError::InvalidFormat(format!("Invalid slice start: {e}")))?;
    let end: usize = parts[1]
        .parse()
        .map_err(|e| CliError::InvalidFormat(format!("Invalid slice end: {e}")))?;
    if start >= end {
        return Err(CliError::InvalidFormat(
            format!("Invalid slice range {start}:{end}: start must be less than end"),
        ));
    }
    Ok((start, end))
}

/// Dispatch slice extraction based on format.
fn run_slice(opts: &HexOptions, bytes: &[u8], format: FileFormat) -> Result<(), CliError> {
    let slice_str = opts.slice.as_deref().ok_or_else(|| {
        CliError::InvalidFormat("--slice requires a range".to_string())
    })?;
    let tensor_filter = opts.tensor.as_deref().ok_or_else(|| {
        CliError::InvalidFormat("--slice requires --tensor".to_string())
    })?;
    let (start, end) = parse_slice(slice_str)?;

    match format {
        FileFormat::SafeTensors => slice_safetensors(opts, bytes, tensor_filter, start, end),
        FileFormat::Gguf => slice_gguf(opts, tensor_filter, start, end),
        FileFormat::Apr => slice_apr(opts, tensor_filter, start, end),
    }
}

/// Slice extraction for SafeTensors format.
#[allow(clippy::disallowed_methods)]
fn slice_safetensors(
    opts: &HexOptions,
    bytes: &[u8],
    tensor_name: &str,
    start: usize,
    end: usize,
) -> Result<(), CliError> {
    let parsed = parse_safetensors_header(bytes)?;
    let tensor_map = parsed.header.as_object().ok_or_else(|| {
        CliError::InvalidFormat("SafeTensors header is not a JSON object".to_string())
    })?;

    let info = tensor_map.get(tensor_name).ok_or_else(|| {
        CliError::InvalidFormat(format!("Tensor '{tensor_name}' not found"))
    })?;

    let (dtype, shape, data_offsets) = extract_st_tensor_info(info, tensor_name)?;
    let num_elements: usize = shape.iter().product();

    if end > num_elements {
        return Err(CliError::InvalidFormat(format!(
            "Slice end {end} exceeds tensor size {num_elements}"
        )));
    }

    let abs_start = 8 + parsed.header_len + data_offsets.0;
    let abs_end = 8 + parsed.header_len + data_offsets.1;
    if abs_end > bytes.len() {
        return Err(CliError::InvalidFormat("Tensor data exceeds file bounds".to_string()));
    }
    let tensor_bytes = &bytes[abs_start..abs_end];

    let values = decode_st_slice(tensor_bytes, &dtype, start, end)?;
    let slice_count = end - start;

    output_slice_result(opts, tensor_name, start, end, &dtype, slice_count, &values)
}

/// Extract SafeTensors tensor metadata: dtype, shape, (data_start, data_end).
fn extract_st_tensor_info(
    info: &serde_json::Value,
    tensor_name: &str,
) -> Result<(String, Vec<usize>, (usize, usize)), CliError> {
    let dtype = info
        .get("dtype")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            CliError::InvalidFormat(format!("Missing dtype for tensor '{tensor_name}'"))
        })?
        .to_string();

    let shape: Vec<usize> = info
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            CliError::InvalidFormat(format!("Missing shape for tensor '{tensor_name}'"))
        })?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();

    let offsets = info
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            CliError::InvalidFormat(format!("Missing data_offsets for tensor '{tensor_name}'"))
        })?;

    let data_start = offsets
        .first()
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            CliError::InvalidFormat(format!("Invalid data_offsets for tensor '{tensor_name}'"))
        })? as usize;

    let data_end = offsets
        .get(1)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            CliError::InvalidFormat(format!("Invalid data_offsets for tensor '{tensor_name}'"))
        })? as usize;

    Ok((dtype, shape, (data_start, data_end)))
}

/// Decode a slice of elements from raw SafeTensors bytes for a given dtype.
fn decode_st_slice(
    tensor_bytes: &[u8],
    dtype: &str,
    start: usize,
    end: usize,
) -> Result<Vec<f32>, CliError> {
    match dtype {
        "F32" => {
            let byte_start = start * 4;
            let byte_end = end * 4;
            if byte_end > tensor_bytes.len() {
                return Err(CliError::InvalidFormat("Slice exceeds tensor data".to_string()));
            }
            Ok(tensor_bytes[byte_start..byte_end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        "F16" => {
            let byte_start = start * 2;
            let byte_end = end * 2;
            if byte_end > tensor_bytes.len() {
                return Err(CliError::InvalidFormat("Slice exceeds tensor data".to_string()));
            }
            Ok(tensor_bytes[byte_start..byte_end]
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        "BF16" => {
            let byte_start = start * 2;
            let byte_end = end * 2;
            if byte_end > tensor_bytes.len() {
                return Err(CliError::InvalidFormat("Slice exceeds tensor data".to_string()));
            }
            Ok(tensor_bytes[byte_start..byte_end]
                .chunks_exact(2)
                .map(|c| {
                    let bits = u32::from_le_bytes([0, 0, c[0], c[1]]);
                    f32::from_bits(bits)
                })
                .collect())
        }
        _ => Err(CliError::InvalidFormat(format!(
            "Unsupported dtype '{dtype}' for --slice (supported: F32, F16, BF16)"
        ))),
    }
}

/// Slice extraction for GGUF format (dequantize then slice).
#[allow(clippy::disallowed_methods)]
fn slice_gguf(
    opts: &HexOptions,
    tensor_name: &str,
    start: usize,
    end: usize,
) -> Result<(), CliError> {
    let (data, shape) = get_gguf_tensor_f32(&opts.file, tensor_name)?;
    let num_elements = data.len();

    if end > num_elements {
        return Err(CliError::InvalidFormat(format!(
            "Slice end {end} exceeds tensor size {num_elements}"
        )));
    }

    let values: Vec<f32> = data[start..end].to_vec();
    let info = parse_gguf(&opts.file)?;
    let dtype_name = info
        .tensors
        .iter()
        .find(|t| t.name == tensor_name)
        .map(|t| ggml_dtype_name(t.dtype))
        .unwrap_or("Unknown");

    let _ = shape; // shape available but not needed for flat slice
    let slice_count = end - start;
    output_slice_result(opts, tensor_name, start, end, dtype_name, slice_count, &values)
}

/// Slice extraction for APR format.
#[allow(clippy::disallowed_methods)]
fn slice_apr(
    opts: &HexOptions,
    tensor_name: &str,
    start: usize,
    end: usize,
) -> Result<(), CliError> {
    let file_bytes = std::fs::read(&opts.file)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to read file: {e}")))?;
    let reader = AprV2Reader::from_bytes(&file_bytes)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to read APR: {e}")))?;

    let data = reader.get_tensor_as_f32(tensor_name).ok_or_else(|| {
        CliError::InvalidFormat(format!("Tensor '{tensor_name}' not found or cannot be read as f32"))
    })?;

    let num_elements = data.len();
    if end > num_elements {
        return Err(CliError::InvalidFormat(format!(
            "Slice end {end} exceeds tensor size {num_elements}"
        )));
    }

    let values: Vec<f32> = data[start..end].to_vec();
    let dtype_name = reader
        .get_tensor(tensor_name)
        .map(|e| format!("{:?}", e.dtype))
        .unwrap_or_else(|| "Unknown".to_string());

    let slice_count = end - start;
    output_slice_result(opts, tensor_name, start, end, &dtype_name, slice_count, &values)
}

/// Output slice result as JSON or text.
#[allow(clippy::disallowed_methods)]
fn output_slice_result(
    opts: &HexOptions,
    tensor_name: &str,
    start: usize,
    end: usize,
    dtype: &str,
    count: usize,
    values: &[f32],
) -> Result<(), CliError> {
    if opts.json {
        let json = serde_json::json!({
            "tensor": tensor_name,
            "slice": format!("{start}:{end}"),
            "dtype": dtype,
            "shape": [count],
            "values": values,
        });
        if let Ok(s) = serde_json::to_string_pretty(&json) {
            println!("{s}");
        }
    } else {
        println!("{}: {}", "Tensor".bold(), tensor_name.cyan());
        println!("{}: {start}:{end} ({count} elements)", "Slice".bold());
        println!("{}: {}", "Dtype".bold(), output::dtype_color(dtype));
        println!("{}: {:?}", "Values".bold(), values);
    }
    Ok(())
}

// ============================================================================
// --header: Annotated file header
// ============================================================================

fn print_file_header(bytes: &[u8], format: FileFormat) {
    output::header(&format!("{} File Header", format_display_name(format)));

    match format {
        FileFormat::Gguf => print_gguf_file_header(bytes),
        FileFormat::Apr => print_apr_file_header(bytes),
        FileFormat::SafeTensors => print_safetensors_file_header(bytes),
    }
}

fn print_gguf_file_header(bytes: &[u8]) {
    if bytes.len() < 24 {
        println!("  {} File too small for GGUF header", "Error:".red());
        return;
    }

    print_annotated_field(
        0,
        &bytes[0..4],
        "magic",
        &format!(
            "\"{}\"",
            std::str::from_utf8(&bytes[0..4]).unwrap_or("????")
        ),
    );

    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    print_annotated_field(4, &bytes[4..8], "version", &version.to_string());

    let tensor_count = u64::from_le_bytes([
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    ]);
    print_annotated_field(
        8,
        &bytes[8..16],
        "tensor_count",
        &output::count_fmt(tensor_count as usize),
    );

    let metadata_kv_count = u64::from_le_bytes([
        bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
    ]);
    print_annotated_field(
        16,
        &bytes[16..24],
        "metadata_kv_count",
        &output::count_fmt(metadata_kv_count as usize),
    );
}

fn print_apr_file_header(bytes: &[u8]) {
    if bytes.len() < 8 {
        println!("  {} File too small for APR header", "Error:".red());
        return;
    }

    let magic_str = std::str::from_utf8(&bytes[0..4]).unwrap_or("????");
    print_annotated_field(0, &bytes[0..4], "magic", &format!("\"{magic_str}\""));

    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    print_annotated_field(4, &bytes[4..8], "version", &version.to_string());

    if bytes.len() >= 12 {
        let model_type = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        print_annotated_field(8, &bytes[8..12], "model_type", &model_type.to_string());
    }
    if bytes.len() >= 20 {
        let metadata_size = u64::from_le_bytes([
            bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
        ]);
        print_annotated_field(
            12,
            &bytes[12..20],
            "metadata_size",
            &output::format_size(metadata_size),
        );
    }
}
