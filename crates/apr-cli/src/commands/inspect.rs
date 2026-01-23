//! Inspect command implementation
//!
//! Toyota Way: Genchi Genbutsu - Go to the source to understand.
//! Inspect model metadata, vocab, filters, and structure without loading payload.

use crate::error::CliError;
use crate::output;
use aprender::format::{self, ModelType, HEADER_SIZE};
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Model inspection result for JSON output
#[derive(Serialize)]
struct InspectResult {
    file: String,
    valid: bool,
    model_type: String,
    version: String,
    size_bytes: u64,
    compressed_size: u64,
    uncompressed_size: u64,
    flags: FlagsInfo,
    metadata: MetadataInfo,
}

#[derive(Serialize)]
#[allow(clippy::struct_excessive_bools)] // Flags are naturally booleans
struct FlagsInfo {
    encrypted: bool,
    signed: bool,
    compressed: bool,
    streaming: bool,
    quantized: bool,
}

#[derive(Serialize, Default)]
struct MetadataInfo {
    created_at: Option<String>,
    aprender_version: Option<String>,
    model_name: Option<String>,
    description: Option<String>,
    vocab_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hyperparameters: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics: Option<serde_json::Value>,
    // Chat template info (CTA-07: apr inspect Shows Template)
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    special_tokens: Option<serde_json::Value>,
}

/// Parsed header data
#[allow(clippy::struct_excessive_bools)] // Flags are naturally booleans
struct HeaderData {
    version: (u8, u8),
    model_type: String,
    metadata_size: u32,
    payload_size: u32,
    uncompressed_size: u32,
    compressed: bool,
    encrypted: bool,
    signed: bool,
    streaming: bool,
    quantized: bool,
}

/// Run the inspect command
#[allow(clippy::fn_params_excessive_bools)] // CLI flags are naturally booleans
pub(crate) fn run(
    path: &Path,
    show_vocab: bool,
    show_filters: bool,
    show_weights: bool,
    json_output: bool,
) -> Result<(), CliError> {
    validate_path(path)?;

    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let header = read_and_parse_header(&mut reader)?;
    let metadata_info = read_metadata(&mut reader, header.metadata_size);

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
    let mut header_bytes = [0u8; HEADER_SIZE];
    reader.read_exact(&mut header_bytes).map_err(|_| {
        CliError::InvalidFormat("File too small to contain valid header".to_string())
    })?;

    if !output::is_valid_magic(&header_bytes[0..4]) {
        return Err(CliError::InvalidFormat(format!(
            "Invalid magic bytes: expected APRN, APR1, APR2, or APR\\0, got {:?}",
            &header_bytes[0..4]
        )));
    }

    let version = (header_bytes[4], header_bytes[5]);
    let model_type_raw = u16::from_le_bytes([header_bytes[6], header_bytes[7]]);
    let metadata_size = u32::from_le_bytes([
        header_bytes[8],
        header_bytes[9],
        header_bytes[10],
        header_bytes[11],
    ]);
    let payload_size = u32::from_le_bytes([
        header_bytes[12],
        header_bytes[13],
        header_bytes[14],
        header_bytes[15],
    ]);
    let uncompressed_size = u32::from_le_bytes([
        header_bytes[16],
        header_bytes[17],
        header_bytes[18],
        header_bytes[19],
    ]);
    let compression = header_bytes[20];
    let flags_byte = header_bytes[21];

    let model_type = ModelType::from_u16(model_type_raw).map_or_else(
        || format!("Unknown(0x{model_type_raw:04X})"),
        |t| format!("{t:?}"),
    );

    Ok(HeaderData {
        version,
        model_type,
        metadata_size,
        payload_size,
        uncompressed_size,
        compressed: compression != 0,
        encrypted: flags_byte & 0x01 != 0,
        signed: flags_byte & 0x02 != 0,
        streaming: flags_byte & 0x04 != 0,
        quantized: flags_byte & 0x20 != 0,
    })
}

fn read_metadata(reader: &mut BufReader<File>, metadata_size: u32) -> MetadataInfo {
    if metadata_size == 0 {
        return MetadataInfo::default();
    }

    let mut metadata_bytes = vec![0u8; metadata_size as usize];
    if reader.read_exact(&mut metadata_bytes).is_err() {
        return MetadataInfo::default();
    }

    match rmp_serde::from_slice::<format::Metadata>(&metadata_bytes) {
        Ok(meta) => MetadataInfo {
            created_at: Some(meta.created_at.clone()),
            aprender_version: Some(meta.aprender_version.clone()),
            model_name: meta.model_name.clone(),
            description: meta.description.clone(),
            vocab_size: meta
                .metrics
                .get("vocab_size")
                .and_then(|v: &serde_json::Value| v.as_u64())
                .map(|v| v as usize),
            hyperparameters: if meta.hyperparameters.is_empty() {
                None
            } else {
                serde_json::to_value(&meta.hyperparameters).ok()
            },
            metrics: if meta.metrics.is_empty() {
                None
            } else {
                serde_json::to_value(&meta.metrics).ok()
            },
            // Chat template info (from v1 metadata, may be empty)
            chat_template: None,
            chat_format: None,
            special_tokens: None,
        },
        Err(_) => MetadataInfo::default(),
    }
}

fn output_json(path: &Path, file_size: u64, header: &HeaderData, metadata: MetadataInfo) {
    let (v_maj, v_min) = header.version;
    let result = InspectResult {
        file: path.display().to_string(),
        valid: true,
        model_type: header.model_type.clone(),
        version: format!("{v_maj}.{v_min}"),
        size_bytes: file_size,
        compressed_size: u64::from(header.payload_size),
        uncompressed_size: u64::from(header.uncompressed_size),
        flags: FlagsInfo {
            encrypted: header.encrypted,
            signed: header.signed,
            compressed: header.compressed,
            streaming: header.streaming,
            quantized: header.quantized,
        },
        metadata,
    };
    if let Ok(json) = serde_json::to_string_pretty(&result) {
        println!("{json}");
    }
}

fn output_text(
    path: &Path,
    file_size: u64,
    header: &HeaderData,
    metadata: &MetadataInfo,
    show_vocab: bool,
    show_filters: bool,
    show_weights: bool,
) {
    output::section(&path.display().to_string());
    println!();

    let (v_maj, v_min) = header.version;
    output::kv("Type", &header.model_type);
    output::kv("Version", format!("{v_maj}.{v_min}"));
    output::kv("Size", output::format_size(file_size));

    if header.compressed {
        let ratio = if header.payload_size > 0 {
            f64::from(header.uncompressed_size) / f64::from(header.payload_size)
        } else {
            1.0
        };
        output::kv(
            "Compressed",
            format!(
                "{} (ratio: {ratio:.2}x)",
                output::format_size(u64::from(header.payload_size))
            ),
        );
    }

    output_flags(header);
    output_metadata_text(metadata);

    if show_vocab {
        println!("\n  Vocabulary: (not yet implemented)");
    }
    if show_filters {
        println!("\n  Filters: (not yet implemented)");
    }
    if show_weights {
        println!("\n  Weights: (not yet implemented)");
    }
}

fn output_flags(header: &HeaderData) {
    let mut flag_list = Vec::new();
    if header.compressed {
        flag_list.push("COMPRESSED");
    }
    if header.encrypted {
        flag_list.push("ENCRYPTED");
    }
    if header.signed {
        flag_list.push("SIGNED");
    }
    if header.streaming {
        flag_list.push("STREAMING");
    }
    if header.quantized {
        flag_list.push("QUANTIZED");
    }

    if !flag_list.is_empty() {
        output::kv("Flags", flag_list.join(" | "));
    }
}

fn output_metadata_text(metadata: &MetadataInfo) {
    if let Some(created) = &metadata.created_at {
        output::kv("Created", created);
    }
    if let Some(version) = &metadata.aprender_version {
        output::kv("Framework", format!("aprender {version}"));
    }
    if let Some(name) = &metadata.model_name {
        output::kv("Name", name);
    }
    if let Some(desc) = &metadata.description {
        output::kv("Description", desc);
    }
    if let Some(vocab) = metadata.vocab_size {
        output::kv("Vocab Size", vocab);
    }

    if let Some(hp) = &metadata.hyperparameters {
        println!("\n  Hyperparameters:");
        if let Some(obj) = hp.as_object() {
            for (k, v) in obj {
                println!("    {k}: {v}");
            }
        }
    }

    if let Some(m) = &metadata.metrics {
        println!("\n  Metrics:");
        if let Some(obj) = m.as_object() {
            for (k, v) in obj {
                println!("    {k}: {v}");
            }
        }
    }

    // Chat template info (CTA-07: apr inspect Shows Template)
    if metadata.chat_template.is_some() || metadata.chat_format.is_some() {
        println!("\n  Chat Template:");
        if let Some(format) = &metadata.chat_format {
            println!("    Format: {format}");
        }
        if let Some(template) = &metadata.chat_template {
            // Truncate long templates for display
            let display_template = if template.len() > 100 {
                format!("{}... ({} chars)", &template[..100], template.len())
            } else {
                template.clone()
            };
            println!("    Template: {display_template}");
        }
        if let Some(tokens) = &metadata.special_tokens {
            println!("    Special Tokens:");
            if let Some(obj) = tokens.as_object() {
                for (k, v) in obj {
                    if !v.is_null() {
                        println!("      {k}: {v}");
                    }
                }
            }
        }
    }
}
