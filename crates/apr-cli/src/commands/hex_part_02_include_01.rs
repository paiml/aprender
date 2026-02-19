/// Extract SafeTensors tensor metadata: dtype, shape, (data_start, data_end).
#[allow(clippy::type_complexity)]
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
        .map_or("Unknown", |t| ggml_dtype_name(t.dtype));

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
        .map_or_else(|| "Unknown".to_string(), |e| format!("{:?}", e.dtype));

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
