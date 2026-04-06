
/// Validate shape of a critical tensor against the layout contract.
fn validate_critical_tensor_shape(
    report: &mut LintReport,
    layout: &crate::format::layout_contract::LayoutContract,
    tc: &crate::format::layout_contract::TensorContract,
    tensor: &TensorLintInfo,
    vocab_size: usize,
    hidden_dim: usize,
) {
    if !tc.is_critical || tensor.shape.is_empty() {
        return;
    }

    if let Err(e) = layout.validate_apr_shape(&tensor.name, &tensor.shape, vocab_size, hidden_dim) {
        report.add_issue(
            LintIssue::layout_error(format!("F-LAYOUT-CONTRACT-002 violation: {}", e))
                .with_suggestion(format!(
                    "Expected shape {} per contract",
                    tc.apr_shape_formula
                )),
        );
    }
}

/// Validate that a 2D tensor requiring transpose has correct dimensions.
fn validate_transpose_dimensions(
    report: &mut LintReport,
    tc: &crate::format::layout_contract::TensorContract,
    tensor: &TensorLintInfo,
    vocab_size: usize,
) {
    if !tc.should_transpose || tensor.shape.len() != 2 {
        return;
    }

    let dim0 = tensor.shape[0];

    // For lm_head specifically, dim0 should be vocab_size
    if tensor.name.contains("lm_head") && dim0 != vocab_size {
        report.add_issue(
            LintIssue::layout_warn(format!(
                "lm_head.weight shape[0]={} but expected vocab_size={}",
                dim0, vocab_size
            ))
            .with_suggestion("Shape should be [vocab_size, hidden_dim] after transpose"),
        );
    }
}

/// Check efficiency requirements
fn check_efficiency(report: &mut LintReport, info: &ModelLintInfo) {
    const ALIGNMENT_TARGET: usize = 64;
    const LARGE_TENSOR_THRESHOLD: usize = 1024 * 1024; // 1MB

    let mut unaligned_count = 0;
    let mut uncompressed_large_count = 0;

    for tensor in &info.tensors {
        // Check alignment
        if tensor.alignment < ALIGNMENT_TARGET && tensor.alignment > 0 {
            unaligned_count += 1;
        }

        // Check for uncompressed large tensors
        if !tensor.is_compressed && tensor.size_bytes > LARGE_TENSOR_THRESHOLD {
            uncompressed_large_count += 1;
        }
    }

    if unaligned_count > 0 {
        report.add_issue(LintIssue::efficiency_info(format!(
            "{} tensors could be aligned to 64 bytes (currently unaligned)",
            unaligned_count
        )));
    }

    if uncompressed_large_count > 0 {
        report.add_issue(LintIssue::efficiency_info(format!(
            "{} uncompressed tensors exceed 1MB - consider compression",
            uncompressed_large_count
        )));
    }
}

/// Check if license info exists in header/metadata.
fn has_license_info(header: &crate::format::Header, metadata: &crate::format::Metadata) -> bool {
    metadata.license.is_some()
        || metadata.custom.contains_key("license")
        || header.flags.is_licensed()
}

/// Check if model card info exists in header/metadata.
fn has_model_card_info(header: &crate::format::Header, metadata: &crate::format::Metadata) -> bool {
    metadata.model_card.is_some()
        || metadata.custom.contains_key("model_card")
        || header.flags.has_model_card()
}

/// Check if provenance info exists in metadata.
fn has_provenance_info(metadata: &crate::format::Metadata) -> bool {
    metadata.distillation.is_some()
        || metadata.distillation_info.is_some()
        || metadata.training.is_some()
        || metadata.custom.contains_key("provenance")
        || metadata.custom.contains_key("author")
}

/// Lint any model file from disk (APR, GGUF, or SafeTensors)
///
/// Detects format via magic bytes and runs format-appropriate lint checks:
/// - **Universal**: metadata (license, provenance), tensor naming, NaN/Inf
/// - **APR-only**: CRC32 integrity, 64-byte alignment, compression
/// - **GGUF**: metadata KV pairs for license/model_card
/// - **SafeTensors**: `__metadata__` for license/description/author
pub fn lint_model_file(path: impl AsRef<Path>) -> Result<LintReport> {
    use crate::format::rosetta::FormatType;

    let path = path.as_ref();
    let format = FormatType::from_magic(path).or_else(|_| FormatType::from_extension(path))?;

    match format {
        FormatType::Apr => lint_apr_file(path),
        FormatType::Gguf => lint_gguf_file(path),
        FormatType::SafeTensors => lint_safetensors_file(path),
    }
}

/// Lint a GGUF file for best practices
fn lint_gguf_file(path: &Path) -> Result<LintReport> {
    use crate::format::gguf::GgufReader;

    let reader = GgufReader::from_file(path)?;
    let mut info = ModelLintInfo::default();

    // Check GGUF metadata KV pairs for standard fields
    info.has_license = reader
        .metadata
        .keys()
        .any(|k| k.contains("license") || k.contains("License"));
    info.has_model_card = reader
        .metadata
        .keys()
        .any(|k| k.contains("model_card") || k.contains("description"));
    info.has_provenance = reader
        .metadata
        .keys()
        .any(|k| k.contains("author") || k.contains("source") || k.contains("url"));

    // Build tensor lint info and extract model config
    for meta in &reader.tensors {
        let shape: Vec<usize> = meta.dims.iter().map(|&d| d as usize).collect();
        let num_elements: usize = shape.iter().product();

        // Extract vocab_size and hidden_dim from known tensors
        if meta.name == "output.weight" || meta.name.contains("lm_head") {
            // GGUF stores as [hidden, vocab], APR as [vocab, hidden]
            if shape.len() == 2 {
                info.hidden_dim = Some(shape[0]);
                info.vocab_size = Some(shape[1]);
            }
        }

        info.tensors.push(TensorLintInfo {
            name: meta.name.clone(),
            size_bytes: num_elements * 4, // approximate
            alignment: 32,                // GGUF uses 32-byte alignment
            is_compressed: false,
            shape: shape.clone(),
        });
    }

    Ok(lint_model(&info))
}

/// Lint a SafeTensors file for best practices
fn lint_safetensors_file(path: &Path) -> Result<LintReport> {
    use crate::serialization::safetensors::MappedSafeTensors;

    let mapped =
        MappedSafeTensors::open(path).map_err(|e| crate::error::AprenderError::FormatError {
            message: format!("SafeTensors open failed: {e}"),
        })?;

    let mut info = ModelLintInfo::default();

    let data = std::fs::read(path)?;
    extract_safetensors_metadata(&data, &mut info);
    collect_safetensors_tensors(&mapped, &mut info);

    Ok(lint_model(&info))
}

/// Extract metadata flags from the SafeTensors `__metadata__` header section.
fn extract_safetensors_metadata(data: &[u8], info: &mut ModelLintInfo) {
    if data.len() < 8 {
        return;
    }

    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0u8; 8])) as usize;
    if data.len() < 8 + header_len {
        return;
    }

    let Ok(header) = serde_json::from_slice::<serde_json::Value>(&data[8..8 + header_len]) else {
        return;
    };

    let Some(meta) = header.get("__metadata__").and_then(|v| v.as_object()) else {
        return;
    };

    info.has_license = meta.contains_key("license");
    info.has_model_card = meta.contains_key("description") || meta.contains_key("model_card");
    info.has_provenance = meta.contains_key("author") || meta.contains_key("source");
}

/// Build tensor lint info from a mapped SafeTensors file, extracting model dimensions.
fn collect_safetensors_tensors(
    mapped: &crate::serialization::safetensors::MappedSafeTensors,
    info: &mut ModelLintInfo,
) {
    for name in mapped.tensor_names() {
        let Some(meta) = mapped.get_metadata(name) else {
            continue;
        };

        let size_bytes = meta.data_offsets[1] - meta.data_offsets[0];
        let shape: Vec<usize> = meta.shape.clone();

        // Extract vocab_size and hidden_dim from lm_head
        if name.contains("lm_head") && shape.len() == 2 {
            info.vocab_size = Some(shape[0]);
            info.hidden_dim = Some(shape[1]);
        }

        info.tensors.push(TensorLintInfo {
            name: name.to_string(),
            size_bytes,
            alignment: 0, // SafeTensors doesn't guarantee alignment
            is_compressed: false,
            shape,
        });
    }
}

/// Lint an APR file from disk (supports both v1 and v2 formats)
pub fn lint_apr_file(path: impl AsRef<Path>) -> Result<LintReport> {
    use std::fs::File;
    use std::io::{BufReader, Read};

    let path = path.as_ref();
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read first 4 bytes to detect version
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;

    // Detect APR v1 (APRN) vs v2 (APR\0 or APR2)
    if &magic == b"APRN" {
        // APR v1 format - use v1 Header
        lint_apr_v1_file(path)
    } else if magic[0..3] == *b"APR" {
        // APR v2 format - use v2 reader
        lint_apr_v2_file(path)
    } else {
        Err(crate::error::AprenderError::FormatError {
            message: format!(
                "Invalid APR magic: {:02X}{:02X}{:02X}{:02X}",
                magic[0], magic[1], magic[2], magic[3]
            ),
        })
    }
}

/// Lint an APR v1 file (APRN magic)
fn lint_apr_v1_file(path: &Path) -> Result<LintReport> {
    use crate::format::{Header, Metadata, HEADER_SIZE};
    use std::fs::File;
    use std::io::{BufReader, Read};

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header
    let mut header_bytes = [0u8; HEADER_SIZE];
    reader.read_exact(&mut header_bytes)?;
    let header = Header::from_bytes(&header_bytes)?;

    // Read metadata
    let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
    reader.read_exact(&mut metadata_bytes)?;
    let metadata: Metadata = rmp_serde::from_slice(&metadata_bytes).unwrap_or_default();

    // Build lint info from header/metadata
    let mut info = ModelLintInfo::default();
    info.has_license = has_license_info(&header, &metadata);
    info.has_model_card = has_model_card_info(&header, &metadata);
    info.has_provenance = has_provenance_info(&metadata);
    info.is_compressed = header.compression != crate::format::Compression::None;

    // For tensor info, we need to read tensor index
    let payload_size = header.payload_size as usize;
    if payload_size > 0 {
        info.tensors.push(TensorLintInfo {
            name: "payload".to_string(),
            size_bytes: payload_size,
            alignment: 64, // Assume aligned for now
            is_compressed: info.is_compressed,
            shape: vec![], // v1 format doesn't store shape info in header
        });
    }

    Ok(lint_model(&info))
}

/// Lint an APR v2 file (APR\0 or APR2 magic)
fn lint_apr_v2_file(path: &Path) -> Result<LintReport> {
    use crate::format::v2::AprV2Reader;
    use std::fs;

    // Read file and create reader
    let data = fs::read(path)?;
    let reader =
        AprV2Reader::from_bytes(&data).map_err(|e| crate::error::AprenderError::FormatError {
            message: format!("Failed to parse APR v2: {e}"),
        })?;

    // Build lint info from v2 metadata
    let mut info = ModelLintInfo::default();
    let metadata = reader.metadata();

    // Check for standard metadata fields
    info.has_license = metadata.license.is_some();
    info.has_model_card = metadata.description.is_some();
    info.has_provenance = metadata.author.is_some()
        || metadata.source.is_some()
        || metadata.original_format.is_some();

    // Add tensor info (use tensor_names and get_tensor)
    for name in reader.tensor_names() {
        if let Some(tensor) = reader.get_tensor(name) {
            let shape: Vec<usize> = tensor.shape.clone();

            // Extract vocab_size and hidden_dim from lm_head
            if name.contains("lm_head") && shape.len() == 2 {
                info.vocab_size = Some(shape[0]);
                info.hidden_dim = Some(shape[1]);
            }

            info.tensors.push(TensorLintInfo {
                name: name.to_string(),
                size_bytes: tensor.size as usize,
                alignment: 64, // v2 uses 64-byte alignment
                is_compressed: false,
                shape,
            });
        }
    }

    Ok(lint_model(&info))
}

// ============================================================================
// TESTS - Written first following EXTREME TDD
// ============================================================================

#[cfg(test)]
mod tests;
