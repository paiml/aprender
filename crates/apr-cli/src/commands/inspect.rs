//! Inspect command implementation (PMAT-225)
//!
//! Toyota Way: Genchi Genbutsu - Go to the source to understand.
//! Inspect APR v2 model metadata, architecture, tensors, and structure.

use crate::error::CliError;
use crate::output;
use aprender::format::v2::{AprV2Flags, AprV2Header, AprV2Metadata, HEADER_SIZE_V2, MAGIC_V2};
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

// ============================================================================
// Data Structures
// ============================================================================

/// Model inspection result for JSON output
#[derive(Serialize)]
struct InspectResult {
    file: String,
    valid: bool,
    format: String,
    version: String,
    tensor_count: u32,
    size_bytes: u64,
    checksum_valid: bool,
    /// GH-249: Top-level architecture field for parity checker compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    architecture: Option<String>,
    /// GH-249: Top-level num_layers for parity checker compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    num_layers: Option<usize>,
    /// GH-249: Top-level num_heads for parity checker compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    num_heads: Option<usize>,
    /// GH-249: Top-level hidden_size for parity checker compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    hidden_size: Option<usize>,
    /// GH-249: Top-level vocab_size for parity checker compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    vocab_size: Option<usize>,
    flags: FlagsInfo,
    metadata: MetadataInfo,
}

#[derive(Serialize)]
struct FlagsInfo {
    lz4_compressed: bool,
    zstd_compressed: bool,
    encrypted: bool,
    signed: bool,
    sharded: bool,
    quantized: bool,
    has_vocab: bool,
}

#[derive(Serialize, Default)]
struct MetadataInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    model_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    author: Option<String>,
    // C-APR-PROVENANCE / INV-APR-PROV-002 / FM-APR-PROV-SILENT-SKIP:
    // provenance keys MUST always serialize (null when absent) so auditors
    // never see them silently skipped. Do NOT add skip_serializing_if here.
    license: Option<String>,
    data_source: Option<String>,
    data_license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    original_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_at: Option<String>,
    architecture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    param_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vocab_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hidden_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_layers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_kv_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    intermediate_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_position_embeddings: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rope_theta: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    special_tokens: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_metadata: Option<serde_json::Value>,
}

/// Parsed v2 header data
struct HeaderData {
    version: (u8, u8),
    flags: AprV2Flags,
    tensor_count: u32,
    metadata_offset: u64,
    metadata_size: u32,
    #[allow(dead_code)]
    tensor_index_offset: u64,
    data_offset: u64,
    checksum_valid: bool,
}

// ============================================================================
// Command Entry Point
// ============================================================================

/// Run the inspect command
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
pub(crate) fn run(
    path: &Path,
    show_vocab: bool,
    show_filters: bool,
    show_weights: bool,
    json_output: bool,
) -> Result<(), CliError> {
    validate_path(path)?;

    // Detect format via magic bytes (Rosetta Stone dispatch)
    let format = aprender::format::rosetta::FormatType::from_magic(path)
        .or_else(|_| aprender::format::rosetta::FormatType::from_extension(path));

    match format {
        Ok(
            aprender::format::rosetta::FormatType::Gguf
            | aprender::format::rosetta::FormatType::SafeTensors,
            // GH-685: forward all flags to rosetta path (were dropped)
        ) => {
            let result = run_rosetta_inspect(path, show_vocab, show_weights, json_output);
            // GH-685: --filters on GGUF/SafeTensors — acknowledge the flag
            if show_filters && !json_output {
                println!();
                println!("  (--filters: GGUF/SafeTensors format has no security filter metadata)");
            }
            result
        }
        _ => {
            // Default: APR v2 inspect (existing path)
            let file = File::open(path)?;
            let file_size = file.metadata()?.len();
            let mut reader = BufReader::new(file);

            let header = read_and_parse_header(&mut reader)?;
            let metadata_info = read_metadata(&mut reader, &header);

            if json_output {
                output_json(path, file_size, &header, metadata_info);
            } else {
                output_text(
                    path,
                    file_size,
                    &header,
                    &metadata_info,
                    show_vocab,
                    show_filters,
                    show_weights,
                );
            }
            Ok(())
        }
    }
}

/// GGUF/SafeTensors inspect via RosettaStone
/// Print rosetta inspection report as JSON.
fn output_rosetta_json(path: &Path, report: &aprender::format::rosetta::InspectionReport) {
    let mut json_map = serde_json::Map::new();
    json_map.insert(
        "file".to_string(),
        serde_json::Value::String(path.display().to_string()),
    );
    json_map.insert(
        "format".to_string(),
        serde_json::Value::String(report.format.to_string()),
    );
    json_map.insert(
        "file_size".to_string(),
        serde_json::Value::Number(serde_json::Number::from(report.file_size)),
    );
    json_map.insert(
        "total_params".to_string(),
        serde_json::Value::Number(serde_json::Number::from(report.total_params)),
    );
    // GH-249: Always include architecture and quantization (use "unknown" if absent)
    json_map.insert(
        "architecture".to_string(),
        serde_json::Value::String(
            report
                .architecture
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
        ),
    );
    json_map.insert(
        "quantization".to_string(),
        serde_json::Value::String(
            report
                .quantization
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
        ),
    );
    json_map.insert(
        "tensor_count".to_string(),
        serde_json::Value::Number(serde_json::Number::from(report.tensors.len())),
    );
    let metadata: serde_json::Value = report
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect::<serde_json::Map<_, _>>()
        .into();
    json_map.insert("metadata".to_string(), metadata);

    if let Ok(json) = serde_json::to_string_pretty(&json_map) {
        println!("{json}");
    }
}

/// Print rosetta inspection report as rich text.
fn output_rosetta_text(report: &aprender::format::rosetta::InspectionReport) {
    output::header("Rosetta Stone Inspection");

    let mut pairs: Vec<(&str, String)> = vec![
        ("Format", report.format.to_string()),
        ("File Size", output::format_size(report.file_size as u64)),
        ("Parameters", output::count_fmt(report.total_params)),
    ];
    if let Some(ref arch) = report.architecture {
        pairs.push(("Architecture", arch.clone()));
    }
    if let Some(ref quant) = report.quantization {
        pairs.push(("Quantization", quant.clone()));
    }
    println!("{}", output::kv_table(&pairs));

    if !report.metadata.is_empty() {
        output::subheader(&format!("Metadata ({} keys)", report.metadata.len()));
        let meta_pairs: Vec<(&str, String)> = report
            .metadata
            .iter()
            .map(|(k, v)| {
                let display_v = if v.len() > 60 {
                    format!("{}...", &v[..60])
                } else {
                    v.clone()
                };
                (k.as_str(), display_v)
            })
            .collect();
        println!("{}", output::kv_table(&meta_pairs));
    }

    output::subheader(&format!("Tensors ({} total)", report.tensors.len()));
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (i, t) in report.tensors.iter().enumerate() {
        if i < 10 || i >= report.tensors.len().saturating_sub(2) {
            rows.push(vec![
                t.name.clone(),
                format!("{}", output::dtype_color(&t.dtype)),
                format!("{:?}", t.shape),
                output::format_size(t.size_bytes as u64),
            ]);
        } else if i == 10 {
            rows.push(vec![
                format!("... {} more ...", report.tensors.len().saturating_sub(12)),
                String::new(),
                String::new(),
                String::new(),
            ]);
        }
    }
    println!(
        "{}",
        output::table(&["Name", "DType", "Shape", "Size"], &rows)
    );
}

/// GH-682: Display vocabulary/tokenizer metadata from GGUF/SafeTensors models.
fn output_rosetta_vocab(report: &aprender::format::rosetta::InspectionReport) {
    let vocab_keys: Vec<(&str, &str)> = report
        .metadata
        .iter()
        .filter(|(k, _)| k.starts_with("tokenizer.ggml."))
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    if vocab_keys.is_empty() {
        println!("\n  (no tokenizer metadata present in model)");
        return;
    }

    output::subheader("Vocabulary");
    let pairs: Vec<(&str, String)> = vocab_keys
        .iter()
        .filter(|(k, _)| {
            // Show concise tokenizer info, skip large arrays
            !k.ends_with(".tokens") && !k.ends_with(".merges") && !k.ends_with(".token_type")
        })
        .map(|(k, v)| {
            let short_key = k.strip_prefix("tokenizer.ggml.").unwrap_or(k);
            (short_key, v.to_string())
        })
        .collect();
    println!("{}", output::kv_table(&pairs));

    // Show token counts for the large arrays
    for (k, v) in &vocab_keys {
        if k.ends_with(".tokens") || k.ends_with(".merges") {
            let short_key = k.strip_prefix("tokenizer.ggml.").unwrap_or(k);
            if let Some(len) = v.strip_prefix("[len=").and_then(|s| s.strip_suffix(']')) {
                println!("  {short_key}: {len} entries");
            }
        }
    }
}

// Contract: apr-inspect-flags-v1 F-INSPECT-FLAGS-001
fn run_rosetta_inspect(
    path: &Path,
    show_vocab: bool,
    show_weights: bool,
    json_output: bool,
) -> Result<(), CliError> {
    use aprender::format::rosetta::RosettaStone;

    let rosetta = RosettaStone::new();
    let report = rosetta
        .inspect(path)
        .map_err(|e| CliError::InvalidFormat(format!("Inspection failed: {e}")))?;

    if json_output {
        output_rosetta_json(path, &report);
    } else {
        output_rosetta_text(&report);
    }

    // GH-682: --vocab flag was dropped by rosetta dispatch path
    if show_vocab {
        output_rosetta_vocab(&report);
    }

    // GH-685: --weights shows per-tensor statistics
    if show_weights {
        output_rosetta_weights(&report);
    }

    Ok(())
}

/// GH-685: Display per-tensor weight statistics from GGUF/SafeTensors.
fn output_rosetta_weights(report: &aprender::format::rosetta::InspectionReport) {
    output::subheader("Weight Statistics");
    let mut rows: Vec<Vec<String>> = Vec::new();
    for t in &report.tensors {
        let elements: usize = t.shape.iter().product();
        rows.push(vec![
            t.name.clone(),
            format!("{}", output::dtype_color(&t.dtype)),
            format!("{elements}"),
            output::format_size(t.size_bytes as u64),
        ]);
    }
    if rows.len() > 12 {
        let total = rows.len();
        let mut display_rows: Vec<Vec<String>> = rows[..5].to_vec();
        display_rows.push(vec![
            format!("... {} more ...", total - 10),
            String::new(),
            String::new(),
            String::new(),
        ]);
        display_rows.extend_from_slice(&rows[total - 5..]);
        println!(
            "{}",
            output::table(&["Tensor", "DType", "Elements", "Size"], &display_rows)
        );
    } else {
        println!(
            "{}",
            output::table(&["Tensor", "DType", "Elements", "Size"], &rows)
        );
    }
    let total_elements: usize = report
        .tensors
        .iter()
        .map(|t| t.shape.iter().product::<usize>())
        .sum();
    let total_bytes: u64 = report.tensors.iter().map(|t| t.size_bytes as u64).sum();
    println!(
        "  Total: {} tensors, {} elements, {}",
        report.tensors.len(),
        output::count_fmt(total_elements),
        output::format_size(total_bytes)
    );
}

// ============================================================================
// Parsing
// ============================================================================

fn validate_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    Ok(())
}

fn read_and_parse_header(reader: &mut BufReader<File>) -> Result<HeaderData, CliError> {
    let mut header_bytes = [0u8; HEADER_SIZE_V2];
    reader.read_exact(&mut header_bytes).map_err(|_| {
        CliError::InvalidFormat(
            "File too small to contain valid APR header (need 64 bytes)".to_string(),
        )
    })?;

    // Check magic - only APR\0 (v2) is supported for detailed inspection
    // BUG-INSPECT-001 FIX: Distinguish GGUF from legacy APR formats
    let magic = &header_bytes[0..4];
    if magic != MAGIC_V2 {
        if magic == output::MAGIC_GGUF {
            return Err(CliError::InvalidFormat(
                "GGUF format detected. Use 'apr inspect' with --format gguf flag \
                 or convert to APR format with 'apr import'."
                    .to_string(),
            ));
        }
        if output::is_valid_magic(magic) {
            return Err(CliError::InvalidFormat(
                "Legacy APR format detected (APRN/APR1/APR2). Only APR v2 (APR\\0) is supported. \
                 Re-import the model to create a v2 file."
                    .to_string(),
            ));
        }
        return Err(CliError::InvalidFormat(format!(
            "Invalid magic bytes: expected APR\\0, got {:02x}{:02x}{:02x}{:02x}",
            magic[0], magic[1], magic[2], magic[3]
        )));
    }

    let header = AprV2Header::from_bytes(&header_bytes)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to parse v2 header: {e}")))?;

    let checksum_valid = header.verify_checksum();

    Ok(HeaderData {
        version: header.version,
        flags: header.flags,
        tensor_count: header.tensor_count,
        metadata_offset: header.metadata_offset,
        metadata_size: header.metadata_size,
        tensor_index_offset: header.tensor_index_offset,
        data_offset: header.data_offset,
        checksum_valid,
    })
}

fn read_metadata(reader: &mut BufReader<File>, header: &HeaderData) -> MetadataInfo {
    if header.metadata_size == 0 {
        return MetadataInfo::default();
    }

    // Seek to metadata offset
    if reader
        .seek(SeekFrom::Start(header.metadata_offset))
        .is_err()
    {
        return MetadataInfo::default();
    }

    let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
    if reader.read_exact(&mut metadata_bytes).is_err() {
        return MetadataInfo::default();
    }

    // Parse JSON metadata (v2 uses JSON, not msgpack)
    match AprV2Metadata::from_json(&metadata_bytes) {
        Ok(meta) => {
            let source_metadata = meta.custom.get("source_metadata").cloned();

            MetadataInfo {
                model_type: if meta.model_type.is_empty() {
                    None
                } else {
                    Some(meta.model_type)
                },
                name: meta.name,
                description: meta.description,
                author: meta.author,
                // C-APR-PROVENANCE: these three are copied unconditionally
                // (even if None) so inspect always emits the keys.
                license: meta.license,
                data_source: meta.data_source,
                data_license: meta.data_license,
                source: meta.source,
                original_format: meta.original_format,
                created_at: meta.created_at,
                // GH-249: Always include architecture (never empty)
                architecture: meta
                    .architecture
                    .filter(|a| !a.is_empty())
                    .or_else(|| Some("unknown".to_string())),
                param_count: if meta.param_count > 0 {
                    Some(meta.param_count)
                } else {
                    None
                },
                vocab_size: meta.vocab_size,
                hidden_size: meta.hidden_size,
                num_layers: meta.num_layers,
                num_heads: meta.num_heads,
                num_kv_heads: meta.num_kv_heads,
                intermediate_size: meta.intermediate_size,
                max_position_embeddings: meta.max_position_embeddings,
                rope_theta: meta.rope_theta,
                chat_template: meta.chat_template,
                chat_format: meta.chat_format,
                special_tokens: meta
                    .special_tokens
                    .and_then(|st| serde_json::to_value(st).ok()),
                source_metadata,
            }
        }
        Err(_) => MetadataInfo::default(),
    }
}

include!("inspect_output_json.rs");
include!("inspect_03.rs");
include!("inspect_tests.rs");
