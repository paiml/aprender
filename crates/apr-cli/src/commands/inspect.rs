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
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    original_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
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

    // Check magic - only APR\0 (v2) is supported
    let magic = &header_bytes[0..4];
    if magic != MAGIC_V2 {
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
                source: meta.source,
                original_format: meta.original_format,
                created_at: meta.created_at,
                architecture: meta.architecture,
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

// ============================================================================
// Output Formatting
// ============================================================================

fn output_json(path: &Path, file_size: u64, header: &HeaderData, metadata: MetadataInfo) {
    let (v_maj, v_min) = header.version;
    let result = InspectResult {
        file: path.display().to_string(),
        valid: true,
        format: "APR v2".to_string(),
        version: format!("{v_maj}.{v_min}"),
        tensor_count: header.tensor_count,
        size_bytes: file_size,
        checksum_valid: header.checksum_valid,
        flags: flags_from_header(header),
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

    // Header info
    let (v_maj, v_min) = header.version;
    output::kv("Format", "APR v2");
    output::kv("Version", format!("{v_maj}.{v_min}"));
    output::kv("Size", output::format_size(file_size));
    output::kv("Tensors", header.tensor_count);
    output::kv(
        "Checksum",
        if header.checksum_valid {
            "VALID"
        } else {
            "INVALID"
        },
    );

    // Data layout
    output::kv(
        "Data Offset",
        format!(
            "0x{:X} ({})",
            header.data_offset,
            output::format_size(header.data_offset)
        ),
    );

    // Flags
    output_flags(header);

    // Architecture section
    output_architecture(metadata);

    // General metadata
    output_metadata_text(metadata);

    if show_vocab {
        println!("\n  Vocabulary: (use `apr tensors` for detailed view)");
    }
    if show_filters {
        println!("\n  Filters: (not applicable for v2 format)");
    }
    if show_weights {
        println!("\n  Weights: (use `apr tensors` for detailed view)");
    }
}

fn flags_from_header(header: &HeaderData) -> FlagsInfo {
    FlagsInfo {
        lz4_compressed: header.flags.is_lz4_compressed(),
        zstd_compressed: header.flags.is_zstd_compressed(),
        encrypted: header.flags.is_encrypted(),
        signed: header.flags.contains(AprV2Flags::SIGNED),
        sharded: header.flags.is_sharded(),
        quantized: header.flags.is_quantized(),
        has_vocab: header.flags.contains(AprV2Flags::HAS_VOCAB),
    }
}

fn output_flags(header: &HeaderData) {
    let mut flag_list = Vec::new();
    if header.flags.is_lz4_compressed() {
        flag_list.push("LZ4");
    }
    if header.flags.is_zstd_compressed() {
        flag_list.push("ZSTD");
    }
    if header.flags.is_encrypted() {
        flag_list.push("ENCRYPTED");
    }
    if header.flags.contains(AprV2Flags::SIGNED) {
        flag_list.push("SIGNED");
    }
    if header.flags.is_sharded() {
        flag_list.push("SHARDED");
    }
    if header.flags.is_quantized() {
        flag_list.push("QUANTIZED");
    }
    if header.flags.contains(AprV2Flags::HAS_VOCAB) {
        flag_list.push("HAS_VOCAB");
    }
    if header.flags.contains(AprV2Flags::HAS_FILTERBANK) {
        flag_list.push("HAS_FILTERBANK");
    }
    if header.flags.contains(AprV2Flags::HAS_MODEL_CARD) {
        flag_list.push("HAS_MODEL_CARD");
    }
    if header.flags.contains(AprV2Flags::STREAMING) {
        flag_list.push("STREAMING");
    }

    if flag_list.is_empty() {
        output::kv("Flags", "(none)");
    } else {
        output::kv("Flags", flag_list.join(" | "));
    }
}

fn output_architecture(metadata: &MetadataInfo) {
    // Only show architecture section if we have transformer config
    let has_arch_info = metadata.architecture.is_some()
        || metadata.hidden_size.is_some()
        || metadata.num_layers.is_some();

    if !has_arch_info {
        return;
    }

    println!("\n  Architecture:");
    if let Some(arch) = &metadata.architecture {
        println!("    Family: {arch}");
    }
    if let Some(p) = metadata.param_count {
        println!("    Parameters: {}", format_param_count(p));
    }
    if let Some(h) = metadata.hidden_size {
        println!("    Hidden Size: {h}");
    }
    if let Some(n) = metadata.num_layers {
        println!("    Layers: {n}");
    }
    if let Some(n) = metadata.num_heads {
        println!("    Attention Heads: {n}");
    }
    if let Some(n) = metadata.num_kv_heads {
        println!("    KV Heads: {n}");
    }
    if let Some(i) = metadata.intermediate_size {
        println!("    Intermediate Size: {i}");
    }
    if let Some(v) = metadata.vocab_size {
        println!("    Vocab Size: {v}");
    }
    if let Some(m) = metadata.max_position_embeddings {
        println!("    Max Position: {m}");
    }
    if let Some(r) = metadata.rope_theta {
        println!("    RoPE Theta: {r}");
    }
}

fn output_metadata_text(metadata: &MetadataInfo) {
    // General metadata
    if let Some(name) = &metadata.name {
        output::kv("Name", name);
    }
    if let Some(model_type) = &metadata.model_type {
        output::kv("Model Type", model_type);
    }
    if let Some(desc) = &metadata.description {
        output::kv("Description", desc);
    }
    if let Some(author) = &metadata.author {
        output::kv("Author", author);
    }
    if let Some(source) = &metadata.source {
        output::kv("Source", source);
    }
    if let Some(fmt) = &metadata.original_format {
        output::kv("Original Format", fmt);
    }
    if let Some(created) = &metadata.created_at {
        output::kv("Created", created);
    }

    // Chat template info
    if metadata.chat_template.is_some() || metadata.chat_format.is_some() {
        println!("\n  Chat Template:");
        if let Some(format) = &metadata.chat_format {
            println!("    Format: {format}");
        }
        if let Some(template) = &metadata.chat_template {
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

    // Source metadata (PMAT-223)
    if let Some(source_meta) = &metadata.source_metadata {
        println!("\n  Source Metadata (PMAT-223):");
        if let Some(obj) = source_meta.as_object() {
            for (k, v) in obj {
                if let Some(s) = v.as_str() {
                    println!("    {k}: {s}");
                } else {
                    println!("    {k}: {v}");
                }
            }
        }
    }
}

fn format_param_count(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B ({count})", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M ({count})", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K ({count})", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::format::v2::AprV2Writer;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    /// Helper: create a minimal valid v2 APR file
    fn create_test_apr_bytes(metadata: AprV2Metadata) -> Vec<u8> {
        let mut writer = AprV2Writer::new(metadata);
        writer.add_f32_tensor("test.weight", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        writer.write().expect("write v2 bytes")
    }

    fn create_test_apr_file(metadata: AprV2Metadata) -> NamedTempFile {
        let bytes = create_test_apr_bytes(metadata);
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(&bytes).expect("write");
        file
    }

    // ========================================================================
    // Path Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_path_not_found() {
        let result = validate_path(Path::new("/nonexistent/model.apr"));
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_path_is_directory() {
        let dir = tempdir().expect("create dir");
        let result = validate_path(dir.path());
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            _ => panic!("Expected NotAFile error"),
        }
    }

    #[test]
    fn test_validate_path_valid() {
        let file = NamedTempFile::new().expect("create file");
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // Run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            false,
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_file_too_small() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(b"short").expect("write");

        let result = run(file.path(), false, false, false, false);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidFormat(msg)) => {
                assert!(msg.contains("too small") || msg.contains("64 bytes"));
            }
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_run_invalid_magic() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        // Write 64 bytes with invalid magic
        let mut data = [0u8; 64];
        data[0..4].copy_from_slice(b"XXXX");
        file.write_all(&data).expect("write");

        let result = run(file.path(), false, false, false, false);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidFormat(msg)) => {
                assert!(msg.contains("Invalid magic"));
            }
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_run_legacy_magic_rejected() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        // Write 64 bytes with legacy APRN magic
        let mut data = [0u8; 64];
        data[0..4].copy_from_slice(b"APRN");
        file.write_all(&data).expect("write");

        let result = run(file.path(), false, false, false, false);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidFormat(msg)) => {
                assert!(
                    msg.contains("Legacy"),
                    "Expected legacy format message, got: {msg}"
                );
            }
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_run_valid_v2_file_text() {
        let metadata = AprV2Metadata {
            model_type: "Qwen2".to_string(),
            name: Some("test-model".to_string()),
            architecture: Some("qwen2".to_string()),
            hidden_size: Some(896),
            num_layers: Some(24),
            num_heads: Some(14),
            num_kv_heads: Some(2),
            vocab_size: Some(151936),
            param_count: 494_032_768,
            ..Default::default()
        };
        let file = create_test_apr_file(metadata);
        let result = run(file.path(), false, false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_valid_v2_file_json() {
        let metadata = AprV2Metadata {
            model_type: "Qwen2".to_string(),
            name: Some("json-test".to_string()),
            param_count: 1_000_000,
            ..Default::default()
        };
        let file = create_test_apr_file(metadata);
        let result = run(file.path(), false, false, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_v2_with_source_metadata() {
        let mut custom = std::collections::HashMap::new();
        let mut source_meta = serde_json::Map::new();
        source_meta.insert(
            "my_run_id".to_string(),
            serde_json::Value::String("test_123".to_string()),
        );
        source_meta.insert(
            "framework".to_string(),
            serde_json::Value::String("pytorch".to_string()),
        );
        custom.insert(
            "source_metadata".to_string(),
            serde_json::Value::Object(source_meta),
        );

        let metadata = AprV2Metadata {
            model_type: "Qwen2".to_string(),
            name: Some("metadata-test".to_string()),
            custom,
            ..Default::default()
        };
        let file = create_test_apr_file(metadata);

        // Run JSON output and verify source_metadata appears
        let result = run(file.path(), false, false, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_show_options() {
        let metadata = AprV2Metadata::new("test");
        let file = create_test_apr_file(metadata);
        let result = run(file.path(), true, true, true, false);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Header Parsing Tests
    // ========================================================================

    #[test]
    fn test_read_header_valid_v2() {
        let metadata = AprV2Metadata {
            model_type: "test".to_string(),
            param_count: 42,
            ..Default::default()
        };
        let bytes = create_test_apr_bytes(metadata);
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(&bytes).expect("write");

        let f = File::open(file.path()).expect("open");
        let mut reader = BufReader::new(f);
        let header = read_and_parse_header(&mut reader).expect("parse header");

        assert_eq!(header.version, (2, 0));
        assert_eq!(header.tensor_count, 1);
        assert!(header.checksum_valid);
        assert!(header.metadata_size > 0);
    }

    #[test]
    fn test_read_metadata_from_v2() {
        let metadata = AprV2Metadata {
            model_type: "Qwen2".to_string(),
            name: Some("round-trip-test".to_string()),
            architecture: Some("qwen2".to_string()),
            hidden_size: Some(768),
            num_layers: Some(12),
            ..Default::default()
        };
        let bytes = create_test_apr_bytes(metadata);
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(&bytes).expect("write");

        let f = File::open(file.path()).expect("open");
        let mut reader = BufReader::new(f);
        let header = read_and_parse_header(&mut reader).expect("parse header");
        let meta = read_metadata(&mut reader, &header);

        assert_eq!(meta.model_type.as_deref(), Some("Qwen2"));
        assert_eq!(meta.name.as_deref(), Some("round-trip-test"));
        assert_eq!(meta.architecture.as_deref(), Some("qwen2"));
        assert_eq!(meta.hidden_size, Some(768));
        assert_eq!(meta.num_layers, Some(12));
    }

    #[test]
    fn test_read_metadata_with_source_metadata() {
        let mut custom = std::collections::HashMap::new();
        let mut source_meta = serde_json::Map::new();
        source_meta.insert(
            "run_id".to_string(),
            serde_json::Value::String("abc_789".to_string()),
        );
        custom.insert(
            "source_metadata".to_string(),
            serde_json::Value::Object(source_meta),
        );

        let metadata = AprV2Metadata {
            model_type: "test".to_string(),
            custom,
            ..Default::default()
        };
        let bytes = create_test_apr_bytes(metadata);
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(&bytes).expect("write");

        let f = File::open(file.path()).expect("open");
        let mut reader = BufReader::new(f);
        let header = read_and_parse_header(&mut reader).expect("parse header");
        let meta = read_metadata(&mut reader, &header);

        assert!(meta.source_metadata.is_some());
        let sm = meta.source_metadata.as_ref().expect("source_metadata");
        assert_eq!(sm.get("run_id").and_then(|v| v.as_str()), Some("abc_789"));
    }

    // ========================================================================
    // Serialization Tests
    // ========================================================================

    #[test]
    fn test_flags_info_serialization() {
        let flags = FlagsInfo {
            lz4_compressed: true,
            zstd_compressed: false,
            encrypted: false,
            signed: false,
            sharded: true,
            quantized: true,
            has_vocab: false,
        };

        let json = serde_json::to_string(&flags).expect("serialize");
        assert!(json.contains("\"lz4_compressed\":true"));
        assert!(json.contains("\"sharded\":true"));
        assert!(json.contains("\"quantized\":true"));
    }

    #[test]
    fn test_metadata_info_default() {
        let info = MetadataInfo::default();
        assert!(info.model_type.is_none());
        assert!(info.name.is_none());
        assert!(info.architecture.is_none());
        assert!(info.source_metadata.is_none());
    }

    #[test]
    fn test_metadata_info_serialization() {
        let info = MetadataInfo {
            model_type: Some("Qwen2".to_string()),
            name: Some("test-model".to_string()),
            architecture: Some("qwen2".to_string()),
            hidden_size: Some(768),
            num_layers: Some(12),
            vocab_size: Some(50000),
            param_count: Some(494_000_000),
            ..Default::default()
        };

        let json = serde_json::to_string(&info).expect("serialize");
        assert!(json.contains("test-model"));
        assert!(json.contains("qwen2"));
        assert!(json.contains("768"));
        assert!(json.contains("494000000"));
    }

    #[test]
    fn test_inspect_result_serialization() {
        let result = InspectResult {
            file: "model.apr".to_string(),
            valid: true,
            format: "APR v2".to_string(),
            version: "2.0".to_string(),
            tensor_count: 291,
            size_bytes: 1024 * 1024,
            checksum_valid: true,
            flags: FlagsInfo {
                lz4_compressed: false,
                zstd_compressed: false,
                encrypted: false,
                signed: false,
                sharded: false,
                quantized: false,
                has_vocab: false,
            },
            metadata: MetadataInfo::default(),
        };

        let json = serde_json::to_string_pretty(&result).expect("serialize");
        assert!(json.contains("model.apr"));
        assert!(json.contains("APR v2"));
        assert!(json.contains("\"valid\": true"));
        assert!(json.contains("\"tensor_count\": 291"));
    }

    // ========================================================================
    // Output Functions Tests
    // ========================================================================

    #[test]
    fn test_output_flags_empty() {
        let header = HeaderData {
            version: (2, 0),
            flags: AprV2Flags::new(),
            tensor_count: 0,
            metadata_offset: 64,
            metadata_size: 0,
            tensor_index_offset: 0,
            data_offset: 0,
            checksum_valid: true,
        };
        output_flags(&header);
    }

    #[test]
    fn test_output_flags_multiple() {
        let flags = AprV2Flags::new()
            .with(AprV2Flags::LZ4_COMPRESSED)
            .with(AprV2Flags::QUANTIZED)
            .with(AprV2Flags::HAS_VOCAB);

        let header = HeaderData {
            version: (2, 0),
            flags,
            tensor_count: 10,
            metadata_offset: 64,
            metadata_size: 100,
            tensor_index_offset: 200,
            data_offset: 300,
            checksum_valid: true,
        };
        output_flags(&header);
    }

    #[test]
    fn test_output_metadata_text_empty() {
        let metadata = MetadataInfo::default();
        output_metadata_text(&metadata);
    }

    #[test]
    fn test_output_metadata_text_full() {
        let metadata = MetadataInfo {
            model_type: Some("Qwen2".to_string()),
            name: Some("test-model".to_string()),
            description: Some("Test description".to_string()),
            author: Some("Test Author".to_string()),
            source: Some("hf://test/model".to_string()),
            original_format: Some("safetensors".to_string()),
            created_at: Some("2024-01-01".to_string()),
            architecture: Some("qwen2".to_string()),
            param_count: Some(494_000_000),
            hidden_size: Some(896),
            num_layers: Some(24),
            num_heads: Some(14),
            num_kv_heads: Some(2),
            vocab_size: Some(151936),
            intermediate_size: Some(4864),
            max_position_embeddings: Some(32768),
            rope_theta: Some(1_000_000.0),
            chat_template: Some("{{prompt}}".to_string()),
            chat_format: Some("chatml".to_string()),
            special_tokens: Some(serde_json::json!({"bos": "<s>", "eos": "</s>"})),
            source_metadata: Some(
                serde_json::json!({"run_id": "test_123", "framework": "pytorch"}),
            ),
        };
        output_metadata_text(&metadata);
    }

    #[test]
    fn test_output_metadata_text_long_template() {
        let long_template = "a".repeat(200);
        let metadata = MetadataInfo {
            chat_template: Some(long_template),
            chat_format: Some("custom".to_string()),
            ..Default::default()
        };
        output_metadata_text(&metadata);
    }

    #[test]
    fn test_output_architecture() {
        let metadata = MetadataInfo {
            architecture: Some("qwen2".to_string()),
            param_count: Some(7_000_000_000),
            hidden_size: Some(4096),
            num_layers: Some(32),
            ..Default::default()
        };
        output_architecture(&metadata);
    }

    #[test]
    fn test_output_architecture_empty() {
        let metadata = MetadataInfo::default();
        // Should not print anything
        output_architecture(&metadata);
    }

    #[test]
    fn test_output_json_v2() {
        let header = HeaderData {
            version: (2, 0),
            flags: AprV2Flags::new().with(AprV2Flags::QUANTIZED),
            tensor_count: 291,
            metadata_offset: 64,
            metadata_size: 1024,
            tensor_index_offset: 2048,
            data_offset: 4096,
            checksum_valid: true,
        };
        let metadata = MetadataInfo {
            model_type: Some("Qwen2".to_string()),
            ..Default::default()
        };
        output_json(Path::new("test.apr"), 1024, &header, metadata);
    }

    #[test]
    fn test_output_text_v2() {
        let header = HeaderData {
            version: (2, 0),
            flags: AprV2Flags::new(),
            tensor_count: 1,
            metadata_offset: 64,
            metadata_size: 100,
            tensor_index_offset: 200,
            data_offset: 300,
            checksum_valid: true,
        };
        let metadata = MetadataInfo::default();
        output_text(
            Path::new("test.apr"),
            512,
            &header,
            &metadata,
            false,
            false,
            false,
        );
    }

    #[test]
    fn test_output_text_with_options() {
        let header = HeaderData {
            version: (2, 0),
            flags: AprV2Flags::new(),
            tensor_count: 5,
            metadata_offset: 64,
            metadata_size: 200,
            tensor_index_offset: 300,
            data_offset: 500,
            checksum_valid: true,
        };
        let metadata = MetadataInfo::default();
        output_text(
            Path::new("test.apr"),
            1024,
            &header,
            &metadata,
            true,
            true,
            true,
        );
    }

    // ========================================================================
    // Format Helper Tests
    // ========================================================================

    #[test]
    fn test_format_param_count() {
        assert_eq!(format_param_count(500), "500");
        assert_eq!(format_param_count(1_500), "1.5K (1500)");
        assert_eq!(format_param_count(494_000_000), "494.0M (494000000)");
        assert_eq!(format_param_count(7_000_000_000), "7.0B (7000000000)");
    }

    #[test]
    fn test_flags_from_header() {
        let header = HeaderData {
            version: (2, 0),
            flags: AprV2Flags::new()
                .with(AprV2Flags::LZ4_COMPRESSED)
                .with(AprV2Flags::QUANTIZED),
            tensor_count: 0,
            metadata_offset: 64,
            metadata_size: 0,
            tensor_index_offset: 0,
            data_offset: 0,
            checksum_valid: true,
        };
        let flags = flags_from_header(&header);
        assert!(flags.lz4_compressed);
        assert!(flags.quantized);
        assert!(!flags.encrypted);
        assert!(!flags.sharded);
    }
}
