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
    pub slice: Option<String>,
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
            slice: None,
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

    // Slice mode — extract a range of elements from a specific tensor
    if opts.slice.is_some() && opts.tensor.is_some() {
        return run_slice(opts, &bytes, format);
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
            let byte_limit = if limit == 0 {
                raw_data.len()
            } else {
                limit.min(raw_data.len())
            };
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

fn print_apr_distributions_v2(reader: &AprV2Reader, filtered: &[&str]) -> Result<(), CliError> {
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

include!("safe_tensors_header.rs");
include!("hex_part_03.rs");
include!("sliding_window_entropy.rs");
include!("hex_part_05.rs");
include!("hex_part_06.rs");
