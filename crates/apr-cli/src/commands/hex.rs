//! Hex dump and binary forensics (GH-122)
//!
//! Toyota Way: Genchi Genbutsu - Go and see the actual bytes.
//! Format-aware binary inspection for APR, GGUF, and SafeTensors files.
//!
//! Usage:
//!   apr hex model.gguf --header              # Annotated file header
//!   apr hex model.gguf --blocks --tensor "attn_q"  # Q4K/Q6K super-block view
//!   apr hex model.gguf --distribution        # Value histogram + entropy + kurtosis
//!   apr hex model.gguf --contract            # Layout contract overlay
//!   apr hex model.gguf --entropy             # Per-region byte entropy
//!   apr hex model.gguf --raw --width 32      # Format-aware xxd
//!   apr hex model.apr --stats --list         # APR backward compat

use crate::error::CliError;
use crate::output;
#[cfg(test)]
use aprender::format::v2::TensorDType;
use aprender::format::v2::{AprV2Reader, TensorIndexEntry};
use colored::Colorize;
use std::path::{Path, PathBuf};

// ============================================================================
// HexOptions
// ============================================================================

/// Options for the hex dump command
pub(crate) struct HexOptions {
    pub file: PathBuf,
    pub tensor: Option<String>,
    pub limit: usize,
    pub stats: bool,
    pub list: bool,
    pub json: bool,
    pub header: bool,
    pub blocks: bool,
    pub distribution: bool,
    pub contract: bool,
    pub entropy: bool,
    pub raw: bool,
    pub offset: usize,
    pub width: usize,
}

impl Default for HexOptions {
    fn default() -> Self {
        Self {
            file: PathBuf::new(),
            tensor: None,
            limit: 64,
            stats: false,
            list: false,
            json: false,
            header: false,
            blocks: false,
            distribution: false,
            contract: false,
            entropy: false,
            raw: false,
            offset: 0,
            width: 16,
        }
    }
}

// ============================================================================
// Format detection
// ============================================================================

/// Detected file format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileFormat {
    Apr,
    Gguf,
    SafeTensors,
}

/// Detect format from first 8 bytes of file
fn detect_format(bytes: &[u8]) -> Option<FileFormat> {
    if bytes.len() < 4 {
        return None;
    }
    if &bytes[..4] == b"GGUF" {
        return Some(FileFormat::Gguf);
    }
    if &bytes[..4] == b"APRN"
        || &bytes[..4] == b"APR1"
        || &bytes[..4] == b"APR2"
        || bytes[..4] == [0x41, 0x50, 0x52, 0x00]
    {
        return Some(FileFormat::Apr);
    }
    // SafeTensors: first 8 bytes = u64 LE header length, followed by JSON '{'
    if bytes.len() >= 9 {
        let header_len = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        // Reasonable header size (< 100MB) and starts with JSON
        if header_len < 100_000_000 && bytes[8] == b'{' {
            return Some(FileFormat::SafeTensors);
        }
    }
    None
}

/// Format name for display
fn format_display_name(fmt: FileFormat) -> &'static str {
    match fmt {
        FileFormat::Apr => "APR",
        FileFormat::Gguf => "GGUF",
        FileFormat::SafeTensors => "SafeTensors",
    }
}

// ============================================================================
// Main entry point
// ============================================================================

/// Run the hex dump command
pub(crate) fn run(opts: &HexOptions) -> Result<(), CliError> {
    if !opts.file.exists() {
        return Err(CliError::FileNotFound(opts.file.clone()));
    }

    let bytes = std::fs::read(&opts.file)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to read file: {e}")))?;

    let format = detect_format(&bytes).ok_or_else(|| {
        CliError::InvalidFormat("Cannot detect format (not APR, GGUF, or SafeTensors)".to_string())
    })?;

    // Header mode — annotated file header (any format)
    if opts.header {
        print_file_header(&bytes, format);
        return Ok(());
    }

    // Raw hex mode — format-aware xxd (any format)
    if opts.raw {
        print_raw_hex(&bytes, opts.offset, opts.limit, opts.width);
        return Ok(());
    }

    // Entropy mode — byte entropy analysis (any format)
    if opts.entropy {
        print_entropy_analysis(&bytes, format);
        return Ok(());
    }

    // Format-specific modes
    match format {
        FileFormat::Apr => run_apr(opts),
        FileFormat::Gguf => run_gguf(opts, &bytes),
        FileFormat::SafeTensors => run_safetensors(opts, &bytes),
    }
}

// ============================================================================
// APR mode (v2 only — v1 removed)
// ============================================================================

fn run_apr(opts: &HexOptions) -> Result<(), CliError> {
    let data = std::fs::read(&opts.file)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to read file: {e}")))?;

    let reader = AprV2Reader::from_bytes(&data)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to read APR: {e}")))?;

    let tensor_names = reader.tensor_names();

    // Apply filter
    let filtered: Vec<&str> = tensor_names
        .iter()
        .filter(|name| {
            opts.tensor
                .as_ref()
                .map_or(true, |f| name.contains(f.as_str()))
        })
        .copied()
        .collect();

    if filtered.is_empty() {
        return print_empty_filter(opts.json);
    }

    if opts.list {
        return list_tensors_v2(&reader, &filtered, opts.json);
    }

    if opts.json {
        return output_json_v2(&reader, &filtered, opts.limit, opts.stats);
    }

    if opts.distribution {
        return print_apr_distributions_v2(&reader, &filtered);
    }
    if opts.contract {
        println!(
            "{}",
            output::badge_info("Layout contract not applicable for APR files (use with GGUF)")
        );
        return Ok(());
    }
    if opts.blocks {
        println!(
            "{}",
            output::badge_info("Block view requires GGUF quantized tensors")
        );
        return Ok(());
    }

    print_apr_hex_dump(&reader, &filtered, opts.limit, opts.stats);
    Ok(())
}

/// Print hex dump for each APR tensor
fn print_apr_hex_dump(reader: &AprV2Reader, names: &[&str], limit: usize, show_stats: bool) {
    for name in names {
        if let Some(entry) = reader.get_tensor(name) {
            print_tensor_header_v2(entry);
        }
        if let Some(raw_data) = reader.get_tensor_data(name) {
            if show_stats {
                if let Some(f32_data) = reader.get_tensor_as_f32(name) {
                    print_tensor_stats(&f32_data);
                }
            }
            let byte_limit = if limit == 0 { raw_data.len() } else { limit.min(raw_data.len()) };
            print_raw_hex(raw_data, 0, byte_limit, 16);
        }
        println!();
    }
}

fn print_empty_filter(json: bool) -> Result<(), CliError> {
    if !json {
        println!("{}", "No tensors match the filter pattern".yellow());
    }
    Ok(())
}

fn print_apr_distributions_v2(
    reader: &AprV2Reader,
    filtered: &[&str],
) -> Result<(), CliError> {
    for name in filtered {
        if let Some(data) = reader.get_tensor_as_f32(name) {
            println!("{}: {}", "Distribution".bold(), name.cyan());
            let analysis = compute_distribution(&data);
            print_distribution(&analysis);
            println!();
        }
    }
    Ok(())
}

// ============================================================================
// GGUF mode
// ============================================================================

/// GGUF tensor info (extracted from raw bytes, no GgufReader dependency)
struct GgufInfo {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    tensors: Vec<GgufTensorEntry>,
    data_offset: usize,
}

struct GgufTensorEntry {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    offset: u64,
}

/// Parse GGUF file using aprender's GgufReader
fn parse_gguf(path: &Path) -> Result<GgufInfo, CliError> {
    use aprender::format::gguf::GgufReader;

    let reader = GgufReader::from_file(path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to parse GGUF: {e}")))?;

    let tensors = reader
        .tensors
        .iter()
        .map(|t| GgufTensorEntry {
            name: t.name.clone(),
            dims: t.dims.clone(),
            dtype: t.dtype,
            offset: t.offset,
        })
        .collect();

    Ok(GgufInfo {
        version: reader.version,
        tensor_count: reader.tensor_count,
        metadata_kv_count: reader.metadata.len() as u64,
        tensors,
        data_offset: reader.data_offset,
    })
}

/// Get dequantized f32 tensor data from GGUF
fn get_gguf_tensor_f32(path: &Path, name: &str) -> Result<(Vec<f32>, Vec<usize>), CliError> {
    use aprender::format::gguf::GgufReader;

    let reader = GgufReader::from_file(path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to parse GGUF: {e}")))?;
    reader
        .get_tensor_f32(name)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to read tensor '{name}': {e}")))
}

fn run_gguf(opts: &HexOptions, bytes: &[u8]) -> Result<(), CliError> {
    let info = parse_gguf(&opts.file)?;

    let filtered: Vec<&GgufTensorEntry> = info
        .tensors
        .iter()
        .filter(|t| {
            opts.tensor
                .as_ref()
                .map_or(true, |f| t.name.contains(f.as_str()))
        })
        .collect();

    if opts.list {
        return list_gguf_tensors(&info.tensors, opts.tensor.as_deref(), opts.json);
    }

    if opts.contract {
        print_contract_overlay(&info);
        return Ok(());
    }

    if opts.blocks {
        return print_gguf_blocks(&filtered, bytes, &info);
    }

    if opts.distribution {
        return print_gguf_distributions(&opts.file, &filtered);
    }

    // Default: show file summary + hex dump
    print_gguf_summary(&opts.file, &info, bytes);

    if filtered.is_empty() {
        println!("\n{}", "No tensors match the filter pattern".yellow());
        return Ok(());
    }

    if opts.json {
        return output_gguf_json(&opts.file, &filtered, opts.limit, opts.stats);
    }

    for tensor in &filtered {
        print_gguf_tensor_hex(opts, tensor, &info)?;
        println!();
    }
    Ok(())
}

fn print_gguf_blocks(
    filtered: &[&GgufTensorEntry],
    bytes: &[u8],
    info: &GgufInfo,
) -> Result<(), CliError> {
    if filtered.is_empty() {
        println!("{}", "No tensors match the filter pattern".yellow());
        return Ok(());
    }
    for tensor in filtered {
        let byte_offset = info.data_offset + tensor.offset as usize;
        print_tensor_blocks(bytes, tensor, byte_offset)?;
    }
    Ok(())
}

fn print_gguf_distributions(path: &Path, filtered: &[&GgufTensorEntry]) -> Result<(), CliError> {
    if filtered.is_empty() {
        println!("{}", "No tensors match the filter pattern".yellow());
        return Ok(());
    }
    for tensor in filtered {
        match get_gguf_tensor_f32(path, &tensor.name) {
            Ok((data, _shape)) => {
                println!("{}: {}", "Distribution".bold(), tensor.name.cyan());
                let analysis = compute_distribution(&data);
                print_distribution(&analysis);
                println!();
            }
            Err(e) => println!("  {} {}: {e}", "Skip".yellow(), tensor.name),
        }
    }
    Ok(())
}

fn print_gguf_summary(path: &Path, info: &GgufInfo, bytes: &[u8]) {
    output::header(&format!("GGUF Binary Forensics: {}", path.display()));
    output::metric("Format", format!("GGUF v{}", info.version), "");
    output::metric("Tensors", output::count_fmt(info.tensor_count as usize), "");
    output::metric(
        "Metadata",
        output::count_fmt(info.metadata_kv_count as usize),
        "KV pairs",
    );
    output::metric("Data offset", format!("0x{:X}", info.data_offset), "");
    output::metric("File size", output::format_size(bytes.len() as u64), "");
}

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

    if dtype == "F32" && abs_end <= bytes.len() {
        let tensor_bytes = &bytes[abs_start..abs_end];
        let f32_data: Vec<f32> = tensor_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if opts.stats {
            print_tensor_stats(&f32_data);
        }
        if opts.distribution {
            let analysis = compute_distribution(&f32_data);
            print_distribution(&analysis);
        }
        print_hex_dump(&f32_data, opts.limit);
    }
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

fn print_safetensors_file_header(bytes: &[u8]) {
    if bytes.len() < 9 {
        println!("  {} File too small for SafeTensors header", "Error:".red());
        return;
    }

    let header_len = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    print_annotated_field(
        0,
        &bytes[0..8],
        "header_length",
        &format!("{header_len} bytes"),
    );

    let header_end = (8 + header_len as usize).min(bytes.len());
    let preview_end = header_end.min(8 + 200); // Show first 200 chars of JSON
    if let Ok(json_preview) = std::str::from_utf8(&bytes[8..preview_end]) {
        println!("\n  {} (first 200 chars):", "JSON Header".bold());
        // Pretty-print if it's valid JSON
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(
            std::str::from_utf8(&bytes[8..header_end]).unwrap_or(""),
        ) {
            if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                for (i, line) in pretty.lines().take(20).enumerate() {
                    println!("    {line}");
                    if i == 19 {
                        println!("    ...");
                    }
                }
            }
        } else {
            println!("    {json_preview}");
        }
    }
}

// ============================================================================
// --raw: Format-aware xxd
// ============================================================================

fn print_raw_hex(bytes: &[u8], offset: usize, limit: usize, width: usize) {
    let width = if width == 0 { 16 } else { width };
    let start = offset.min(bytes.len());
    let end = (start + limit).min(bytes.len());
    let slice = &bytes[start..end];

    if slice.is_empty() {
        println!("{}", "No bytes to display at this offset".yellow());
        return;
    }

    for (i, chunk) in slice.chunks(width).enumerate() {
        let addr = start + i * width;
        print_raw_hex_row(addr, chunk, width);
    }

    if end < bytes.len() {
        println!(
            "... {} more bytes",
            output::count_fmt(bytes.len() - end).dimmed()
        );
    }
}

/// Print a single raw hex dump row: offset | hex bytes | ASCII.
fn print_raw_hex_row(addr: usize, chunk: &[u8], width: usize) {
    print!("{}", format!("{addr:08X}: ").dimmed());

    // Hex bytes with midpoint separator
    for (j, &b) in chunk.iter().enumerate() {
        if j == width / 2 && width >= 8 {
            print!(" ");
        }
        print!("{}", format!("{b:02X} ").yellow());
    }
    // Pad short rows
    let missing = width - chunk.len();
    for j in 0..missing {
        if chunk.len() + j == width / 2 && width >= 8 {
            print!(" ");
        }
        print!("   ");
    }

    // ASCII column
    print!(" |");
    for &b in chunk {
        if b.is_ascii_graphic() || b == b' ' {
            print!("{}", (b as char).to_string().white());
        } else {
            print!("{}", ".".dimmed());
        }
    }
    for _ in 0..missing {
        print!(" ");
    }
    println!("|");
}

// ============================================================================
// --blocks: Quantization super-block inspection
// ============================================================================

/// Q4K block size: 144 bytes per 256 elements
const Q4K_BLOCK_SIZE: usize = 144;
/// Q6K block size: 210 bytes per 256 elements
const Q6K_BLOCK_SIZE: usize = 210;
/// Q8_0 block size: 34 bytes per 32 elements
const Q8_0_BLOCK_SIZE: usize = 34;

fn print_tensor_blocks(
    file_bytes: &[u8],
    tensor: &GgufTensorEntry,
    byte_offset: usize,
) -> Result<(), CliError> {
    let dtype_name = ggml_dtype_name(tensor.dtype);
    let dims_str: Vec<String> = tensor
        .dims
        .iter()
        .map(std::string::ToString::to_string)
        .collect();

    output::header(&format!(
        "Block View: {} ({}, [{}])",
        tensor.name,
        dtype_name,
        dims_str.join(", ")
    ));

    match tensor.dtype {
        12 => print_q4k_blocks(file_bytes, byte_offset, 3), // Q4K, show 3 blocks
        14 => print_q6k_blocks(file_bytes, byte_offset, 3), // Q6K
        8 => print_q8_0_blocks(file_bytes, byte_offset, 3), // Q8_0
        _ => {
            println!(
                "  {}",
                output::badge_info(&format!("Block view not applicable for dtype {dtype_name}"))
            );
        }
    }
    Ok(())
}

fn print_q4k_blocks(file_bytes: &[u8], base_offset: usize, count: usize) {
    for block_idx in 0..count {
        let offset = base_offset + block_idx * Q4K_BLOCK_SIZE;
        if offset + Q4K_BLOCK_SIZE > file_bytes.len() {
            println!(
                "  {} Block #{block_idx} exceeds file bounds",
                "Warn:".yellow()
            );
            break;
        }
        let block = &file_bytes[offset..offset + Q4K_BLOCK_SIZE];

        println!(
            "\n  {} (256 elements, {Q4K_BLOCK_SIZE} bytes):",
            format!("Q4_K Super-Block #{block_idx}").cyan().bold()
        );

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        print_annotated_field(0, &block[0..2], "d (scale)", &format!("{d:.5} (f16)"));

        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        print_annotated_field(2, &block[2..4], "dmin", &format!("{dmin:.5} (f16)"));

        print_annotated_field(4, &block[4..16], "scales[0-11]", "12 sub-block scales");
        print_annotated_field(
            16,
            &block[16..Q4K_BLOCK_SIZE],
            "qs[0-127]",
            "4-bit packed (256 values)",
        );
    }
}

fn print_q6k_blocks(file_bytes: &[u8], base_offset: usize, count: usize) {
    for block_idx in 0..count {
        let offset = base_offset + block_idx * Q6K_BLOCK_SIZE;
        if offset + Q6K_BLOCK_SIZE > file_bytes.len() {
            println!(
                "  {} Block #{block_idx} exceeds file bounds",
                "Warn:".yellow()
            );
            break;
        }
        let block = &file_bytes[offset..offset + Q6K_BLOCK_SIZE];

        println!(
            "\n  {} (256 elements, {Q6K_BLOCK_SIZE} bytes):",
            format!("Q6_K Super-Block #{block_idx}").cyan().bold()
        );

        print_annotated_field(
            0,
            &block[0..128],
            "ql[0-127]",
            "low 4 bits (256 values, 2/byte)",
        );
        print_annotated_field(
            128,
            &block[128..192],
            "qh[0-63]",
            "high 2 bits (256 values, 4/byte)",
        );
        print_annotated_field(192, &block[192..208], "scales[0-15]", "16 sub-block scales");

        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        print_annotated_field(208, &block[208..210], "d (scale)", &format!("{d:.5} (f16)"));
    }
}

fn print_q8_0_blocks(file_bytes: &[u8], base_offset: usize, count: usize) {
    for block_idx in 0..count {
        let offset = base_offset + block_idx * Q8_0_BLOCK_SIZE;
        if offset + Q8_0_BLOCK_SIZE > file_bytes.len() {
            println!(
                "  {} Block #{block_idx} exceeds file bounds",
                "Warn:".yellow()
            );
            break;
        }
        let block = &file_bytes[offset..offset + Q8_0_BLOCK_SIZE];

        println!(
            "\n  {} (32 elements, {Q8_0_BLOCK_SIZE} bytes):",
            format!("Q8_0 Block #{block_idx}").cyan().bold()
        );

        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        print_annotated_field(0, &block[0..2], "scale", &format!("{scale:.5} (f16)"));
        print_annotated_field(
            2,
            &block[2..Q8_0_BLOCK_SIZE],
            "quants[0-31]",
            "i8 values [-128..127]",
        );
    }
}

// ============================================================================
// --distribution: Histogram + entropy + kurtosis
// ============================================================================

struct DistributionAnalysis {
    histogram: Vec<(f64, f64, usize)>, // (bin_start, bin_end, count)
    total: usize,
    entropy: f64,
    kurtosis: f64,
    skewness: f64,
    nan_count: usize,
    inf_count: usize,
    zero_count: usize,
    min: f32,
    max: f32,
    mean: f64,
    std: f64,
}

/// First-pass scan of f32 data: counts NaN/Inf/zero, min/max, sum, valid count.
struct ValueScan {
    nan_count: usize,
    inf_count: usize,
    zero_count: usize,
    min: f32,
    max: f32,
    sum: f64,
    valid_count: usize,
}

fn scan_values(data: &[f32]) -> ValueScan {
    let mut s = ValueScan {
        nan_count: 0,
        inf_count: 0,
        zero_count: 0,
        min: f32::INFINITY,
        max: f32::NEG_INFINITY,
        sum: 0.0,
        valid_count: 0,
    };
    for &x in data {
        if x.is_nan() {
            s.nan_count += 1;
            continue;
        }
        if x.is_infinite() {
            s.inf_count += 1;
            continue;
        }
        if x == 0.0 {
            s.zero_count += 1;
        }
        s.min = s.min.min(x);
        s.max = s.max.max(x);
        s.sum += f64::from(x);
        s.valid_count += 1;
    }
    s
}

/// Second-pass: variance, skewness, kurtosis from mean.
fn compute_moments(data: &[f32], mean: f64, valid_count: usize) -> (f64, f64, f64) {
    let (mut m2, mut m3, mut m4) = (0.0_f64, 0.0_f64, 0.0_f64);
    for &x in data {
        if x.is_nan() || x.is_infinite() {
            continue;
        }
        let d = f64::from(x) - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    let variance = m2 / valid_count as f64;
    let std = variance.sqrt();
    let skewness = if std > 0.0 {
        (m3 / valid_count as f64) / (std * std * std)
    } else {
        0.0
    };
    let kurtosis = if std > 0.0 {
        (m4 / valid_count as f64) / (variance * variance)
    } else {
        0.0
    };
    (std, skewness, kurtosis)
}

/// Build histogram bins for valid (non-NaN, non-Inf) values.
fn build_histogram(
    data: &[f32],
    min: f32,
    max: f32,
    num_bins: usize,
    valid_count: usize,
) -> (Vec<(f64, f64, usize)>, f64) {
    let range = f64::from(max) - f64::from(min);
    let bin_width = if range > 0.0 {
        range / num_bins as f64
    } else {
        1.0
    };
    let mut bins = vec![0usize; num_bins];
    for &x in data {
        if x.is_nan() || x.is_infinite() {
            continue;
        }
        let idx = (((f64::from(x) - f64::from(min)) / bin_width) as usize).min(num_bins - 1);
        bins[idx] += 1;
    }
    let histogram: Vec<(f64, f64, usize)> = bins
        .iter()
        .enumerate()
        .map(|(i, &count)| {
            let start = f64::from(min) + i as f64 * bin_width;
            (start, start + bin_width, count)
        })
        .collect();
    let entropy: f64 = histogram
        .iter()
        .filter(|(_, _, c)| *c > 0)
        .map(|(_, _, c)| {
            let p = *c as f64 / valid_count as f64;
            -p * p.log2()
        })
        .sum();
    (histogram, entropy)
}

fn empty_distribution(total: usize, scan: &ValueScan) -> DistributionAnalysis {
    DistributionAnalysis {
        histogram: Vec::new(),
        total,
        entropy: 0.0,
        kurtosis: 0.0,
        skewness: 0.0,
        nan_count: scan.nan_count,
        inf_count: scan.inf_count,
        zero_count: scan.zero_count,
        min: 0.0,
        max: 0.0,
        mean: 0.0,
        std: 0.0,
    }
}

fn compute_distribution(data: &[f32]) -> DistributionAnalysis {
    if data.is_empty() {
        let empty_scan = ValueScan {
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            sum: 0.0,
            valid_count: 0,
        };
        return empty_distribution(0, &empty_scan);
    }

    let scan = scan_values(data);
    if scan.valid_count == 0 {
        return empty_distribution(data.len(), &scan);
    }

    let mean = scan.sum / scan.valid_count as f64;
    let (std, skewness, kurtosis) = compute_moments(data, mean, scan.valid_count);
    let (histogram, entropy) = build_histogram(data, scan.min, scan.max, 10, scan.valid_count);

    DistributionAnalysis {
        histogram,
        total: data.len(),
        entropy,
        kurtosis,
        skewness,
        nan_count: scan.nan_count,
        inf_count: scan.inf_count,
        zero_count: scan.zero_count,
        min: scan.min,
        max: scan.max,
        mean,
        std,
    }
}

fn print_distribution(analysis: &DistributionAnalysis) {
    if analysis.total == 0 {
        println!("  {}", "No data".dimmed());
        return;
    }

    let max_count = analysis
        .histogram
        .iter()
        .map(|(_, _, c)| *c)
        .max()
        .unwrap_or(1);

    for (start, end, count) in &analysis.histogram {
        let bar_width = if max_count > 0 {
            (*count * 40) / max_count
        } else {
            0
        };
        let pct = if analysis.total > 0 {
            *count as f64 / analysis.total as f64 * 100.0
        } else {
            0.0
        };
        let bar = "█".repeat(bar_width);
        println!(
            "  {} {} {}",
            format!("[{start:>8.3}, {end:>8.3})").dimmed(),
            format!("{bar:<40}").green(),
            format!("{pct:>5.1}%").white().bold()
        );
    }

    println!();
    output::metric("Entropy", format!("{:.2} bits", analysis.entropy), "");
    output::metric("Kurtosis", format!("{:.2}", analysis.kurtosis), "");
    output::metric("Skewness", format!("{:.4}", analysis.skewness), "");
    output::metric("Min", format!("{:.6}", analysis.min), "");
    output::metric("Max", format!("{:.6}", analysis.max), "");
    output::metric("Mean", format!("{:.6}", analysis.mean), "");
    output::metric("Std", format!("{:.6}", analysis.std), "");
    if analysis.nan_count > 0 {
        println!(
            "  {} {} NaN values",
            "Warning:".red().bold(),
            output::count_fmt(analysis.nan_count)
        );
    }
    if analysis.inf_count > 0 {
        println!(
            "  {} {} Inf values",
            "Warning:".red().bold(),
            output::count_fmt(analysis.inf_count)
        );
    }
    if analysis.zero_count > 0 {
        output::metric(
            "Zero values",
            format!(
                "{} ({:.1}%)",
                output::count_fmt(analysis.zero_count),
                analysis.zero_count as f64 / analysis.total as f64 * 100.0
            ),
            "",
        );
    }
}

// ============================================================================
// --entropy: Byte entropy analysis
// ============================================================================

/// Compute Shannon entropy of byte distribution (0.0 = all same, 8.0 = uniform random)
fn compute_byte_entropy(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }

    let mut counts = [0u64; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }

    let total = bytes.len() as f64;
    let mut entropy = 0.0_f64;
    for &count in &counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn print_entropy_analysis(bytes: &[u8], format: FileFormat) {
    output::header(&format!(
        "Byte Entropy Analysis ({})",
        format_display_name(format)
    ));

    if bytes.is_empty() {
        println!("  {}", "Empty file".dimmed());
        return;
    }

    let total_entropy = compute_byte_entropy(bytes);
    output::metric(
        "Total entropy",
        format!("{total_entropy:.4} bits"),
        "(0.0 = uniform, 8.0 = random)",
    );
    output::metric("File size", output::format_size(bytes.len() as u64), "");

    // Expected entropy ranges by format
    let expected = match format {
        FileFormat::Gguf => "Q4K/Q6K: 7.5-8.0, F32: 5.0-7.5, F16: 6.0-7.5",
        FileFormat::Apr => "F32: 5.0-7.5, F16: 6.0-7.5",
        FileFormat::SafeTensors => "F32: 5.0-7.5, F16: 6.0-7.5, BF16: 6.0-7.5",
    };
    output::metric("Expected range", expected, "");

    // Sliding window analysis (4KB windows)
    let window_size = 4096;
    if bytes.len() >= window_size {
        let mut min_entropy = f64::MAX;
        let mut max_entropy = f64::MIN;
        let mut min_offset = 0;
        let mut max_offset = 0;
        let mut anomalous_regions = Vec::new();

        let step = (bytes.len() / 100).max(window_size); // ~100 samples
        let mut offset = 0;
        while offset + window_size <= bytes.len() {
            let window = &bytes[offset..offset + window_size];
            let e = compute_byte_entropy(window);

            if e < min_entropy {
                min_entropy = e;
                min_offset = offset;
            }
            if e > max_entropy {
                max_entropy = e;
                max_offset = offset;
            }

            // Flag anomalous regions (very low entropy = all-zeros or corruption)
            if e < 1.0 {
                anomalous_regions.push((offset, e));
            }

            offset += step;
        }

        output::subheader("Sliding Window (4KB)");
        output::metric(
            "Min entropy",
            format!("{min_entropy:.4} at 0x{min_offset:X}"),
            "",
        );
        output::metric(
            "Max entropy",
            format!("{max_entropy:.4} at 0x{max_offset:X}"),
            "",
        );

        if !anomalous_regions.is_empty() {
            println!(
                "\n  {} {} anomalous regions (entropy < 1.0):",
                "Warning:".yellow().bold(),
                anomalous_regions.len()
            );
            for (off, e) in anomalous_regions.iter().take(5) {
                println!("    0x{off:08X}: entropy={e:.4} (possible all-zeros or padding)");
            }
            if anomalous_regions.len() > 5 {
                println!("    ... and {} more", anomalous_regions.len() - 5);
            }
        }
    }
}

// ============================================================================
// --contract: Layout contract overlay
// ============================================================================

fn print_contract_overlay(info: &GgufInfo) {
    use aprender::format::layout_contract::contract;

    output::header("Layout Contract Overlay (GGUF → APR)");

    let layout = contract();
    let mut pass_count = 0;
    let mut miss_count = 0;

    let headers = &["GGUF Name", "APR Name", "Transpose", "Critical", "Status"];
    let mut rows = Vec::new();

    for tensor in &info.tensors {
        if let Some(tc) = layout.get_gguf_contract(&tensor.name) {
            let status = output::badge_pass("Mapped");
            rows.push(vec![
                tensor.name.clone(),
                tc.apr_name.to_string(),
                if tc.should_transpose {
                    "Yes".to_string()
                } else {
                    "No".to_string()
                },
                if tc.is_critical {
                    "CRITICAL".to_string()
                } else {
                    "-".to_string()
                },
                status,
            ]);
            pass_count += 1;
        } else {
            miss_count += 1;
        }
    }

    if !rows.is_empty() {
        println!("{}", output::table(headers, &rows));
    }

    println!();
    output::metric("Mapped tensors", output::count_fmt(pass_count), "");
    output::metric(
        "Unmapped tensors",
        output::count_fmt(miss_count),
        "(norm weights, etc.)",
    );
    output::metric("Total", output::count_fmt(info.tensors.len()), "");
}

// ============================================================================
// Utility functions
// ============================================================================

/// Convert IEEE 754 half-precision (f16) to single-precision (f32)
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // +/- zero
            return f32::from_bits(sign << 31);
        }
        // Subnormal: normalize
        let mut e = 0_i32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let exp32 = (127 - 15 + 1 - e) as u32;
        let mant32 = (m & 0x3FF) << 13;
        return f32::from_bits((sign << 31) | (exp32 << 23) | mant32);
    }

    if exp == 31 {
        if mant == 0 {
            // +/- infinity
            return f32::from_bits((sign << 31) | (0xFF << 23));
        }
        // NaN
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }

    // Normal number: (exp - 15 + 127) rewritten to avoid u32 underflow
    let exp32 = exp + 112; // 112 = 127 - 15
    let mant32 = mant << 13;
    f32::from_bits((sign << 31) | (exp32 << 23) | mant32)
}

/// GGML dtype name from u32 discriminant
fn ggml_dtype_name(dtype: u32) -> &'static str {
    match dtype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        _ => "Unknown",
    }
}

/// Print an annotated hex field
fn print_annotated_field(offset: usize, bytes: &[u8], label: &str, value: &str) {
    print!("  {}", format!("{offset:08X}: ").dimmed());

    // Show up to 8 bytes of hex
    let display_len = bytes.len().min(8);
    for b in &bytes[..display_len] {
        print!("{}", format!("{b:02X} ").yellow());
    }
    if bytes.len() > 8 {
        print!("{}", ".. ".yellow());
    }

    // Pad to align annotations (8 bytes * 3 chars = 24, plus ".. " = 27)
    let hex_width = display_len * 3 + if bytes.len() > 8 { 3 } else { 0 };
    let padding = 28usize.saturating_sub(hex_width);
    print!("{:width$}", "", width = padding);

    println!("{}: {}", label.white().bold(), value.cyan());
}

/// Parse hex offset string (supports "0x" prefix)
pub(crate) fn parse_hex_offset(s: &str) -> Result<usize, String> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        usize::from_str_radix(hex, 16).map_err(|e| format!("Invalid hex offset: {e}"))
    } else {
        s.parse::<usize>()
            .map_err(|e| format!("Invalid offset: {e}"))
    }
}

// ============================================================================
// Preserved helpers (APR tensor display)
// ============================================================================

/// List tensor names (v2 reader)
#[allow(clippy::unnecessary_wraps)]
#[allow(clippy::disallowed_methods)]
fn list_tensors_v2(reader: &AprV2Reader, filtered: &[&str], json_output: bool) -> Result<(), CliError> {
    if json_output {
        let json = serde_json::json!({
            "tensors": filtered,
            "count": filtered.len()
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!("{}", "Tensors:".bold());
        for name in filtered {
            if let Some(entry) = reader.get_tensor(name) {
                println!(
                    "  {}: {:?} ({} bytes, dtype={:?})",
                    name.cyan(),
                    entry.shape,
                    entry.size,
                    entry.dtype
                );
            } else {
                println!("  {}", name);
            }
        }
        println!("\n{} tensors total", filtered.len().to_string().cyan());
    }
    Ok(())
}

/// Print tensor header information (v2 reader)
fn print_tensor_header_v2(entry: &TensorIndexEntry) {
    println!("{}", "═".repeat(70));
    println!("{}: {}", "Tensor".bold(), entry.name.cyan());
    println!("{}", "═".repeat(70));

    let num_elements: usize = entry.shape.iter().product();
    println!(
        "{}: {:?} = {} elements",
        "Shape".bold(),
        entry.shape,
        num_elements.to_string().green()
    );
    println!("{}: {:?}", "Dtype".bold(), entry.dtype);
    println!(
        "{}: 0x{:08X} ({} bytes)",
        "Offset".bold(),
        entry.offset,
        entry.offset
    );
    println!(
        "{}: {} bytes",
        "Size".bold(),
        entry.size.to_string().yellow()
    );
}

/// Check for tensor anomalies and print warnings
fn print_tensor_anomalies(min: f32, max: f32, mean: f32, std: f32) {
    if min.is_nan() || max.is_nan() || mean.is_nan() {
        println!("  {} NaN values detected!", "Warning:".red());
    }
    if min.is_infinite() || max.is_infinite() {
        println!("  {} Infinite values detected!", "Warning:".red());
    }
    if std < 1e-10 {
        println!(
            "  {} Very low variance - possible collapsed weights!",
            "Warning:".yellow()
        );
    }
}

/// Print statistics for tensor data
fn print_tensor_stats(data: &[f32]) {
    println!();
    println!("{}", "Statistics:".bold());
    let (min, max, mean, std) = compute_stats(data);
    println!("  min={min:.6}  max={max:.6}  mean={mean:.6}  std={std:.6}");
    print_tensor_anomalies(min, max, mean, std);
}

/// Print a hex dump row for a chunk of float values
fn print_hex_row(chunk: &[&f32], row_offset: usize) {
    print!("{}", format!("{row_offset:08X}: ").dimmed());

    for &val in chunk {
        let bytes = val.to_le_bytes();
        for b in &bytes {
            print!("{}", format!("{b:02X} ").yellow());
        }
    }

    let padding = (4 - chunk.len()) * 12;
    print!("{:width$}", "", width = padding);

    print!("{}", " | ".dimmed());
    for &val in chunk {
        let color_val = if *val == 0.0 {
            format!("{val:>10.4} ").dimmed().to_string()
        } else if *val < 0.0 {
            format!("{val:>10.4} ").red().to_string()
        } else {
            format!("{val:>10.4} ").green().to_string()
        };
        print!("{color_val}");
    }
    println!();
}

/// Print hex dump of tensor data
fn print_hex_dump(data: &[f32], limit: usize) {
    println!();
    println!(
        "{} (first {} bytes):",
        "Hex dump".bold(),
        (limit * 4).min(data.len() * 4)
    );

    let bytes_to_show = limit.min(data.len());
    for (i, chunk) in data
        .iter()
        .take(bytes_to_show)
        .collect::<Vec<_>>()
        .chunks(4)
        .enumerate()
    {
        print_hex_row(chunk, i * 16);
    }

    if data.len() > bytes_to_show {
        println!(
            "... {} more elements",
            (data.len() - bytes_to_show).to_string().dimmed()
        );
    }
}

// print_tensor_hex removed — v2 path uses inline hex dump in run_apr()

/// Compute basic statistics
fn compute_stats(data: &[f32]) -> (f32, f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;

    for &x in data {
        min = min.min(x);
        max = max.max(x);
        sum += f64::from(x);
    }

    let mean = (sum / data.len() as f64) as f32;

    let variance: f32 = (data
        .iter()
        .map(|&x| {
            let diff = f64::from(x) - f64::from(mean);
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64) as f32;

    let std = variance.sqrt();

    (min, max, mean, std)
}

/// Output as JSON (v2 reader)
#[allow(clippy::unnecessary_wraps)]
#[allow(clippy::disallowed_methods)]
fn output_json_v2(
    reader: &AprV2Reader,
    filtered: &[&str],
    limit: usize,
    show_stats: bool,
) -> Result<(), CliError> {
    use serde::Serialize;

    #[derive(Serialize)]
    struct TensorDump {
        name: String,
        shape: Vec<usize>,
        dtype: String,
        offset: u64,
        size_bytes: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        stats: Option<TensorStatsJson>,
        sample_values: Vec<f32>,
    }

    #[derive(Serialize)]
    struct TensorStatsJson {
        min: f32,
        max: f32,
        mean: f32,
        std: f32,
    }

    let mut results = Vec::new();

    for name in filtered {
        let entry = reader.get_tensor(name);
        let data = reader.get_tensor_as_f32(name);
        let stats = if show_stats {
            data.as_ref().map(|d| {
                let (min, max, mean, std) = compute_stats(d);
                TensorStatsJson {
                    min,
                    max,
                    mean,
                    std,
                }
            })
        } else {
            None
        };

        let sample_values: Vec<f32> = data
            .as_ref()
            .map(|d| d.iter().take(limit).copied().collect())
            .unwrap_or_default();

        if let Some(e) = entry {
            results.push(TensorDump {
                name: e.name.clone(),
                shape: e.shape.clone(),
                dtype: format!("{:?}", e.dtype),
                offset: e.offset,
                size_bytes: e.size,
                stats,
                sample_values,
            });
        }
    }

    if let Ok(json) = serde_json::to_string_pretty(&results) {
        println!("{json}");
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ========================================================================
    // HexOptions defaults
    // ========================================================================

    #[test]
    fn test_hex_options_defaults() {
        let opts = HexOptions::default();
        assert_eq!(opts.limit, 64);
        assert_eq!(opts.width, 16);
        assert_eq!(opts.offset, 0);
        assert!(!opts.header);
        assert!(!opts.blocks);
        assert!(!opts.distribution);
        assert!(!opts.contract);
        assert!(!opts.entropy);
        assert!(!opts.raw);
        assert!(!opts.stats);
        assert!(!opts.list);
        assert!(!opts.json);
        assert!(opts.tensor.is_none());
    }

    // ========================================================================
    // Format detection
    // ========================================================================

    #[test]
    fn test_detect_format_gguf() {
        let bytes = b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(detect_format(bytes), Some(FileFormat::Gguf));
    }

    #[test]
    fn test_detect_format_apr_aprn() {
        let bytes = b"APRN\x02\x00\x00\x00";
        assert_eq!(detect_format(bytes), Some(FileFormat::Apr));
    }

    #[test]
    fn test_detect_format_apr_apr0() {
        let bytes = [0x41, 0x50, 0x52, 0x00, 0x02, 0x00, 0x00, 0x00];
        assert_eq!(detect_format(&bytes), Some(FileFormat::Apr));
    }

    #[test]
    fn test_detect_format_safetensors() {
        // header_length = 50 (LE u64), then '{' starts JSON
        let mut bytes = vec![50, 0, 0, 0, 0, 0, 0, 0, b'{'];
        bytes.extend_from_slice(b"\"test\":{}");
        assert_eq!(detect_format(&bytes), Some(FileFormat::SafeTensors));
    }

    #[test]
    fn test_detect_format_unknown() {
        let bytes = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(detect_format(bytes), None);
    }

    #[test]
    fn test_detect_format_too_short() {
        assert_eq!(detect_format(&[0x41, 0x50]), None);
        assert_eq!(detect_format(&[]), None);
    }

    // ========================================================================
    // f16 to f32 conversion
    // ========================================================================

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0_f32);
    }

    #[test]
    fn test_f16_to_f32_neg_zero() {
        let val = f16_to_f32(0x8000);
        assert_eq!(val.to_bits(), 0x8000_0000); // -0.0
    }

    #[test]
    fn test_f16_to_f32_one() {
        let val = f16_to_f32(0x3C00); // 1.0 in f16
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_neg_one() {
        let val = f16_to_f32(0xBC00); // -1.0 in f16
        assert!((val - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_infinity() {
        let val = f16_to_f32(0x7C00); // +Inf in f16
        assert!(val.is_infinite() && val.is_sign_positive());
    }

    #[test]
    fn test_f16_to_f32_neg_infinity() {
        let val = f16_to_f32(0xFC00); // -Inf in f16
        assert!(val.is_infinite() && val.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_nan() {
        let val = f16_to_f32(0x7C01); // NaN in f16
        assert!(val.is_nan());
    }

    #[test]
    fn test_f16_to_f32_half() {
        let val = f16_to_f32(0x3800); // 0.5 in f16
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        let val = f16_to_f32(0x0001); // Smallest subnormal f16
        assert!(val > 0.0 && val < 1e-6);
    }

    // ========================================================================
    // ggml_dtype_name
    // ========================================================================

    #[test]
    fn test_ggml_dtype_name_known() {
        assert_eq!(ggml_dtype_name(0), "F32");
        assert_eq!(ggml_dtype_name(1), "F16");
        assert_eq!(ggml_dtype_name(2), "Q4_0");
        assert_eq!(ggml_dtype_name(8), "Q8_0");
        assert_eq!(ggml_dtype_name(12), "Q4_K");
        assert_eq!(ggml_dtype_name(14), "Q6_K");
    }

    #[test]
    fn test_ggml_dtype_name_unknown() {
        assert_eq!(ggml_dtype_name(99), "Unknown");
        assert_eq!(ggml_dtype_name(255), "Unknown");
    }

    // ========================================================================
    // parse_hex_offset
    // ========================================================================

    #[test]
    fn test_parse_hex_offset_decimal() {
        assert_eq!(parse_hex_offset("256"), Ok(256));
        assert_eq!(parse_hex_offset("0"), Ok(0));
    }

    #[test]
    fn test_parse_hex_offset_hex() {
        assert_eq!(parse_hex_offset("0x100"), Ok(256));
        assert_eq!(parse_hex_offset("0xFF"), Ok(255));
        assert_eq!(parse_hex_offset("0X1A"), Ok(26));
    }

    #[test]
    fn test_parse_hex_offset_invalid() {
        assert!(parse_hex_offset("0xGG").is_err());
        assert!(parse_hex_offset("abc").is_err());
    }

    // ========================================================================
    // compute_byte_entropy
    // ========================================================================

    #[test]
    fn test_byte_entropy_empty() {
        assert_eq!(compute_byte_entropy(&[]), 0.0);
    }

    #[test]
    fn test_byte_entropy_all_same() {
        let bytes = vec![0x42; 1000];
        assert_eq!(compute_byte_entropy(&bytes), 0.0);
    }

    #[test]
    fn test_byte_entropy_two_values() {
        // Equal distribution of two values → entropy = 1.0
        let mut bytes = vec![0u8; 500];
        bytes.extend(vec![1u8; 500]);
        let entropy = compute_byte_entropy(&bytes);
        assert!((entropy - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_byte_entropy_uniform() {
        // All 256 byte values equally represented → entropy = 8.0
        let bytes: Vec<u8> = (0..=255).cycle().take(256 * 100).collect();
        let entropy = compute_byte_entropy(&bytes);
        assert!((entropy - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_byte_entropy_single_byte() {
        assert_eq!(compute_byte_entropy(&[42]), 0.0);
    }

    // ========================================================================
    // compute_distribution
    // ========================================================================

    #[test]
    fn test_distribution_empty() {
        let analysis = compute_distribution(&[]);
        assert_eq!(analysis.total, 0);
        assert_eq!(analysis.entropy, 0.0);
        assert!(analysis.histogram.is_empty());
    }

    #[test]
    fn test_distribution_single_value() {
        let analysis = compute_distribution(&[1.0]);
        assert_eq!(analysis.total, 1);
        assert_eq!(analysis.min, 1.0);
        assert_eq!(analysis.max, 1.0);
        assert_eq!(analysis.nan_count, 0);
    }

    #[test]
    fn test_distribution_uniform() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let analysis = compute_distribution(&data);
        assert_eq!(analysis.total, 1000);
        assert!(analysis.min >= 0.0);
        assert!(analysis.max < 1.0);
        assert!(analysis.entropy > 2.0); // Should be high entropy
        assert_eq!(analysis.histogram.len(), 10);
    }

    #[test]
    fn test_distribution_with_nan_inf() {
        let data = [1.0, 2.0, f32::NAN, f32::INFINITY, 3.0, f32::NEG_INFINITY];
        let analysis = compute_distribution(&data);
        assert_eq!(analysis.nan_count, 1);
        assert_eq!(analysis.inf_count, 2);
        assert_eq!(analysis.total, 6);
    }

    #[test]
    fn test_distribution_all_zeros() {
        let data = vec![0.0_f32; 100];
        let analysis = compute_distribution(&data);
        assert_eq!(analysis.zero_count, 100);
        assert_eq!(analysis.min, 0.0);
        assert_eq!(analysis.max, 0.0);
    }

    #[test]
    fn test_distribution_skewed() {
        // Heavily right-skewed: many small values, few large
        let mut data: Vec<f32> = vec![0.1; 900];
        data.extend(vec![10.0; 100]);
        let analysis = compute_distribution(&data);
        assert!(analysis.skewness > 0.0, "Should be right-skewed");
    }

    // ========================================================================
    // print_annotated_field
    // ========================================================================

    #[test]
    fn test_print_annotated_field_short() {
        // Should not panic
        print_annotated_field(0, &[0x41, 0x50], "magic", "AP");
    }

    #[test]
    fn test_print_annotated_field_long() {
        // More than 8 bytes → should show ".."
        print_annotated_field(0x100, &[0u8; 16], "data", "16 bytes");
    }

    #[test]
    fn test_print_annotated_field_exact_8() {
        print_annotated_field(0, &[1, 2, 3, 4, 5, 6, 7, 8], "field", "value");
    }

    // ========================================================================
    // Statistics Tests (preserved)
    // ========================================================================

    #[test]
    fn test_compute_stats_empty() {
        let (min, max, mean, std) = compute_stats(&[]);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_single_value() {
        let (min, max, mean, std) = compute_stats(&[5.0]);
        assert_eq!(min, 5.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 5.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_simple_range() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 3.0);
        assert!((std - 1.4142).abs() < 0.01);
    }

    #[test]
    fn test_compute_stats_negative_values() {
        let data = [-5.0, -2.0, 0.0, 2.0, 5.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -5.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_compute_stats_all_same() {
        let data = [7.0, 7.0, 7.0, 7.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 7.0);
        assert_eq!(max, 7.0);
        assert_eq!(mean, 7.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_large_values() {
        let data = [1e6, 2e6, 3e6];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, 1e6);
        assert_eq!(max, 3e6);
        assert_eq!(mean, 2e6);
    }

    #[test]
    fn test_compute_stats_tiny_values() {
        let data = [1e-6, 2e-6, 3e-6];
        let (min, max, mean, _std) = compute_stats(&data);
        assert!((min - 1e-6).abs() < 1e-9);
        assert!((max - 3e-6).abs() < 1e-9);
        assert!((mean - 2e-6).abs() < 1e-9);
    }

    #[test]
    fn test_compute_stats_with_nan() {
        let data = [1.0, f32::NAN, 3.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert!(min.is_nan() || min == 1.0);
        assert!(mean.is_nan());
        let _ = (max, std);
    }

    #[test]
    fn test_compute_stats_all_nan() {
        let data = [f32::NAN, f32::NAN, f32::NAN];
        let (_min, _max, mean, std) = compute_stats(&data);
        assert!(mean.is_nan());
        assert!(std.is_nan());
    }

    #[test]
    fn test_compute_stats_with_infinity() {
        let data = [1.0, f32::INFINITY, -1.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -1.0);
        assert_eq!(max, f32::INFINITY);
        assert!(mean.is_infinite());
    }

    #[test]
    fn test_compute_stats_with_neg_infinity() {
        let data = [1.0, f32::NEG_INFINITY, 3.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, f32::NEG_INFINITY);
        assert_eq!(max, 3.0);
        assert!(mean.is_infinite());
    }

    #[test]
    fn test_compute_stats_two_values() {
        let data = [0.0, 10.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 10.0);
        assert_eq!(mean, 5.0);
        assert!((std - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_stats_all_zeros() {
        let data = [0.0, 0.0, 0.0, 0.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_large_array() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 999.0);
        assert!((mean - 499.5).abs() < 0.1);
        assert!((std - 288.67).abs() < 1.0);
    }

    #[test]
    fn test_compute_stats_mixed_positive_negative() {
        let data = [-100.0, -50.0, 0.0, 50.0, 100.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -100.0);
        assert_eq!(max, 100.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_compute_stats_subnormal_values() {
        let data = [
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE * 2.0,
            f32::MIN_POSITIVE * 3.0,
        ];
        let (min, max, _mean, _std) = compute_stats(&data);
        assert_eq!(min, f32::MIN_POSITIVE);
        assert_eq!(max, f32::MIN_POSITIVE * 3.0);
    }

    // ========================================================================
    // print_tensor_anomalies tests (preserved)
    // ========================================================================

    #[test]
    fn test_print_tensor_anomalies_no_issues() {
        print_tensor_anomalies(0.0, 1.0, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_nan_min() {
        print_tensor_anomalies(f32::NAN, 1.0, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_nan_max() {
        print_tensor_anomalies(0.0, f32::NAN, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_nan_mean() {
        print_tensor_anomalies(0.0, 1.0, f32::NAN, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_all_nan() {
        print_tensor_anomalies(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
    }

    #[test]
    fn test_print_tensor_anomalies_infinite_min() {
        print_tensor_anomalies(f32::NEG_INFINITY, 1.0, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_infinite_max() {
        print_tensor_anomalies(0.0, f32::INFINITY, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_low_variance() {
        print_tensor_anomalies(0.0, 1.0, 0.5, 1e-12);
    }

    #[test]
    fn test_print_tensor_anomalies_zero_variance() {
        print_tensor_anomalies(5.0, 5.0, 5.0, 0.0);
    }

    #[test]
    fn test_print_tensor_anomalies_exactly_threshold() {
        print_tensor_anomalies(0.0, 1.0, 0.5, 1e-10);
    }

    #[test]
    fn test_print_tensor_anomalies_above_threshold() {
        print_tensor_anomalies(0.0, 1.0, 0.5, 2e-10);
    }

    #[test]
    fn test_print_tensor_anomalies_nan_and_infinite_together() {
        print_tensor_anomalies(f32::NAN, f32::INFINITY, f32::NAN, 0.0);
    }

    // ========================================================================
    // print_tensor_header tests (preserved)
    // ========================================================================

    fn make_entry(
        name: &str,
        shape: Vec<usize>,
        dtype: TensorDType,
        offset: u64,
        size: u64,
    ) -> TensorIndexEntry {
        TensorIndexEntry::new(name, dtype, shape, offset, size)
    }

    #[test]
    fn test_print_tensor_header_basic() {
        let entry = make_entry(
            "model.layers.0.weight",
            vec![768, 3072],
            TensorDType::F32,
            0,
            (768 * 3072 * 4) as u64,
        );
        print_tensor_header_v2(&entry);
    }

    #[test]
    fn test_print_tensor_header_empty_shape() {
        let entry = make_entry("scalar_param", vec![], TensorDType::F32, 0, 4);
        print_tensor_header_v2(&entry);
    }

    #[test]
    fn test_print_tensor_header_single_dim() {
        let entry = make_entry("bias", vec![512], TensorDType::F32, 1024, 512 * 4);
        print_tensor_header_v2(&entry);
    }

    #[test]
    fn test_print_tensor_header_large_offset() {
        let entry = make_entry(
            "lm_head.weight",
            vec![32000, 4096],
            TensorDType::F16,
            0xFFFF_FFFF,
            32000 * 4096 * 2,
        );
        print_tensor_header_v2(&entry);
    }

    #[test]
    fn test_print_tensor_header_zero_size() {
        let entry = make_entry("empty", vec![0], TensorDType::F32, 0, 0);
        print_tensor_header_v2(&entry);
    }

    #[test]
    fn test_print_tensor_header_3d_shape() {
        let entry = make_entry("conv.weight", vec![64, 3, 3], TensorDType::F32, 512, (64 * 3 * 3 * 4) as u64);
        print_tensor_header_v2(&entry);
    }

    // ========================================================================
    // print_hex_row tests (preserved)
    // ========================================================================

    #[test]
    fn test_print_hex_row_full_chunk() {
        let vals = [1.0_f32, 2.0, 3.0, 4.0];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 0);
    }

    #[test]
    fn test_print_hex_row_partial_chunk() {
        let vals = [1.0_f32, 2.0];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 16);
    }

    #[test]
    fn test_print_hex_row_single_value() {
        let vals = [42.0_f32];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 0);
    }

    #[test]
    fn test_print_hex_row_three_values() {
        let vals = [0.0_f32, -1.0, f32::MAX];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 48);
    }

    #[test]
    fn test_print_hex_row_special_values() {
        let vals = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 0);
    }

    #[test]
    fn test_print_hex_row_large_offset() {
        let vals = [1.0_f32];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 0xDEAD_BEEF);
    }

    #[test]
    fn test_print_hex_row_negative_values() {
        let vals = [-0.5_f32, -100.0, -1e-6, -f32::MAX];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 32);
    }

    // ========================================================================
    // print_hex_dump tests (preserved)
    // ========================================================================

    #[test]
    fn test_print_hex_dump_empty_data() {
        print_hex_dump(&[], 100);
    }

    #[test]
    fn test_print_hex_dump_data_smaller_than_limit() {
        let data = [1.0_f32, 2.0, 3.0];
        print_hex_dump(&data, 100);
    }

    #[test]
    fn test_print_hex_dump_data_equal_to_limit() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        print_hex_dump(&data, 4);
    }

    #[test]
    fn test_print_hex_dump_data_larger_than_limit() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        print_hex_dump(&data, 10);
    }

    #[test]
    fn test_print_hex_dump_limit_zero() {
        let data = [1.0_f32, 2.0, 3.0];
        print_hex_dump(&data, 0);
    }

    #[test]
    fn test_print_hex_dump_single_element() {
        print_hex_dump(&[42.0], 1);
    }

    #[test]
    fn test_print_hex_dump_exactly_one_row() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        print_hex_dump(&data, 4);
    }

    #[test]
    fn test_print_hex_dump_two_rows() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        print_hex_dump(&data, 8);
    }

    #[test]
    fn test_print_hex_dump_partial_last_row() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        print_hex_dump(&data, 5);
    }

    // ========================================================================
    // print_tensor_stats tests (preserved)
    // ========================================================================

    #[test]
    fn test_print_tensor_stats_normal_data() {
        print_tensor_stats(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_print_tensor_stats_empty_data() {
        print_tensor_stats(&[]);
    }

    #[test]
    fn test_print_tensor_stats_single_value() {
        print_tensor_stats(&[42.0]);
    }

    #[test]
    fn test_print_tensor_stats_with_nan() {
        print_tensor_stats(&[1.0, f32::NAN, 3.0]);
    }

    #[test]
    fn test_print_tensor_stats_all_same() {
        let data = [3.14_f32; 100];
        print_tensor_stats(&data);
    }

    // list_tensors_v2 tests require a real AprV2Reader — tested via integration tests

    // ========================================================================
    // Run command tests (updated for HexOptions)
    // ========================================================================

    fn make_opts(file: &Path) -> HexOptions {
        HexOptions {
            file: file.to_path_buf(),
            ..HexOptions::default()
        }
    }

    #[test]
    fn test_run_file_not_found() {
        let opts = make_opts(Path::new("/nonexistent/model.apr"));
        let result = run(&opts);
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_invalid_apr_file() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"APRN\x00\x00\x00\x00not valid")
            .expect("write");
        let opts = make_opts(file.path());
        let result = run(&opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_unknown_format() {
        let mut file = NamedTempFile::with_suffix(".bin").expect("create temp file");
        file.write_all(b"\x00\x00\x00\x00\x00\x00\x00\x00\x00")
            .expect("write");
        let opts = make_opts(file.path());
        let result = run(&opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_tensor_filter() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"APRN\x00\x00\x00\x00not valid data")
            .expect("write");
        let mut opts = make_opts(file.path());
        opts.tensor = Some("encoder".to_string());
        let result = run(&opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_raw_mode_with_data() {
        let mut file = NamedTempFile::with_suffix(".bin").expect("create temp file");
        // Write GGUF magic so format detection works
        file.write_all(b"GGUF\x03\x00\x00\x00").expect("write");
        file.write_all(&[0u8; 100]).expect("write");
        let mut opts = make_opts(file.path());
        opts.raw = true;
        opts.limit = 32;
        let result = run(&opts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_header_mode_gguf() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        // Write minimal GGUF header
        file.write_all(b"GGUF").expect("write");
        file.write_all(&3u32.to_le_bytes()).expect("write"); // version
        file.write_all(&0u64.to_le_bytes()).expect("write"); // tensor_count
        file.write_all(&0u64.to_le_bytes()).expect("write"); // metadata_kv_count
        let mut opts = make_opts(file.path());
        opts.header = true;
        let result = run(&opts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_header_mode_apr() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"APRN\x02\x00\x00\x00\x00\x00\x00\x00")
            .expect("write");
        file.write_all(&[0u8; 20]).expect("write");
        let mut opts = make_opts(file.path());
        opts.header = true;
        let result = run(&opts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_entropy_mode() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        // Write GGUF header + some data
        file.write_all(b"GGUF").expect("write");
        file.write_all(&3u32.to_le_bytes()).expect("write");
        file.write_all(&0u64.to_le_bytes()).expect("write");
        file.write_all(&0u64.to_le_bytes()).expect("write");
        file.write_all(&[0x42u8; 8192]).expect("write");
        let mut opts = make_opts(file.path());
        opts.entropy = true;
        let result = run(&opts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_nonexistent_path_returns_file_not_found() {
        let opts = make_opts(Path::new("/tmp/this_does_not_exist_apr_test.apr"));
        let result = run(&opts);
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(p)) => {
                assert_eq!(p, Path::new("/tmp/this_does_not_exist_apr_test.apr"));
            }
            other => panic!("Expected FileNotFound, got {:?}", other),
        }
    }

    #[test]
    fn test_run_empty_file() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let opts = make_opts(file.path());
        let result = run(&opts);
        assert!(result.is_err());
    }

    // ========================================================================
    // Raw hex output tests
    // ========================================================================

    #[test]
    fn test_print_raw_hex_basic() {
        let bytes = [0x41, 0x50, 0x52, 0x4E, 0x00, 0x01, 0x02, 0x03];
        print_raw_hex(&bytes, 0, 8, 16);
    }

    #[test]
    fn test_print_raw_hex_with_offset() {
        let bytes = [0u8; 256];
        print_raw_hex(&bytes, 128, 32, 16);
    }

    #[test]
    fn test_print_raw_hex_width_32() {
        let bytes = [0xAA; 64];
        print_raw_hex(&bytes, 0, 64, 32);
    }

    #[test]
    fn test_print_raw_hex_empty() {
        print_raw_hex(&[], 0, 100, 16);
    }

    #[test]
    fn test_print_raw_hex_offset_past_end() {
        let bytes = [0u8; 10];
        print_raw_hex(&bytes, 100, 10, 16);
    }

    #[test]
    fn test_print_raw_hex_width_zero_uses_default() {
        let bytes = [0x42; 32];
        print_raw_hex(&bytes, 0, 32, 0);
    }

    // ========================================================================
    // format_display_name
    // ========================================================================

    #[test]
    fn test_format_display_name() {
        assert_eq!(format_display_name(FileFormat::Apr), "APR");
        assert_eq!(format_display_name(FileFormat::Gguf), "GGUF");
        assert_eq!(format_display_name(FileFormat::SafeTensors), "SafeTensors");
    }

    // ========================================================================
    // Q4K/Q6K/Q8_0 block print tests (no panic)
    // ========================================================================

    #[test]
    fn test_print_q4k_blocks_synthetic() {
        let mut bytes = vec![0u8; Q4K_BLOCK_SIZE * 2];
        // Set scale bytes to known f16 value (1.0 = 0x3C00)
        bytes[0] = 0x00;
        bytes[1] = 0x3C;
        print_q4k_blocks(&bytes, 0, 1);
    }

    #[test]
    fn test_print_q6k_blocks_synthetic() {
        let bytes = vec![0u8; Q6K_BLOCK_SIZE * 2];
        print_q6k_blocks(&bytes, 0, 1);
    }

    #[test]
    fn test_print_q8_0_blocks_synthetic() {
        let bytes = vec![0u8; Q8_0_BLOCK_SIZE * 2];
        print_q8_0_blocks(&bytes, 0, 1);
    }

    #[test]
    fn test_print_blocks_exceeds_bounds() {
        // Should print warning, not panic
        let bytes = vec![0u8; 10];
        print_q4k_blocks(&bytes, 0, 1);
    }
}
