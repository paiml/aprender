
/// Update inferred config from a single tensor's name and shape.
fn infer_q4k_single_tensor(cfg: &mut InferredQ4kConfig, name: &str, shape: &[usize]) {
    // Infer hidden_size from norm weights (1D tensor of hidden_dim)
    if name.contains("input_layernorm.weight") && shape.len() == 1 {
        cfg.hidden_size = Some(shape[0]);
    }
    // Infer vocab_size from embedding [vocab_size, hidden_dim]
    if name.contains("embed_tokens.weight") && shape.len() == 2 {
        cfg.vocab_size = Some(shape[0]);
        if cfg.hidden_size.is_none() {
            cfg.hidden_size = Some(shape[1]);
        }
    }
    // Count layers
    if let Some(idx) = name.strip_prefix("model.layers.") {
        if let Some(layer_num) = idx.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
            cfg.num_layers = Some(
                cfg.num_layers
                    .map_or(layer_num + 1, |n| n.max(layer_num + 1)),
            );
        }
    }
    // Infer kv_heads from k_proj shape [kv_dim, hidden_dim]
    if name.contains("k_proj.weight") && shape.len() == 2 && cfg.hidden_size.is_some() {
        // kv_dim = shape[0], hidden_dim = shape[1]
        // num_kv_heads = kv_dim / head_dim where head_dim = hidden_dim / num_heads
        // For Qwen2-0.5B: kv_dim=128, hidden_dim=896, head_dim=64, num_kv_heads=2
        cfg.num_kv_heads = Some(shape[0] / 64); // Assume head_dim=64 for now
    }
    // Infer num_heads from q_proj shape [q_dim, hidden_dim]
    if name.contains("q_proj.weight") && shape.len() == 2 {
        // q_dim = hidden_dim for standard attention
        // num_heads = hidden_dim / head_dim = hidden_dim / 64
        cfg.num_heads = Some(shape[0] / 64);
    }
    // Infer intermediate_size from gate_proj [intermediate, hidden]
    if name.contains("gate_proj.weight") && shape.len() == 2 {
        cfg.intermediate_size = Some(shape[0]);
    }
}

/// Build APR v2 metadata for a Q4K-quantized model.
///
/// Populates architecture fields from the inferred config and sets Qwen2-specific
/// defaults for RoPE, norm epsilon, and position embeddings.
fn build_q4k_metadata(cfg: &InferredQ4kConfig, param_count: u64) -> AprV2Metadata {
    AprV2Metadata {
        model_type: "qwen2".to_string(),
        name: Some("Quantized Model".to_string()),
        description: Some("Q4K quantized from SafeTensors".to_string()),
        author: None,
        license: None,
        version: Some("1.0.0".to_string()),
        source: None,
        original_format: Some("safetensors".to_string()),
        created_at: None,
        total_size: 0,
        param_count,
        quantization: Some(QuantizationMetadata {
            quant_type: "q4_k".to_string(),
            bits: 4,
            block_size: Some(256),
            symmetric: false,
        }),
        sharding: None,
        chat_template: None,
        chat_format: None,
        special_tokens: None,
        architecture: Some("qwen2".to_string()),
        hidden_size: cfg.hidden_size,
        num_layers: cfg.num_layers,
        num_heads: cfg.num_heads,
        num_kv_heads: cfg.num_kv_heads,
        vocab_size: cfg.vocab_size,
        intermediate_size: cfg.intermediate_size,
        max_position_embeddings: Some(32768), // Default for Qwen2
        rope_theta: Some(1000000.0),          // Default for Qwen2
        rope_type: Some(2),                   // NEOX style for Qwen2 (PMAT-114)
        rms_norm_eps: Some(1e-6),             // Default for Qwen2
        custom: std::collections::HashMap::new(),
    }
}

/// Determine whether a tensor should be quantized to Q4K.
///
/// Returns `true` for large (>= 256 elements) multi-dimensional weight tensors,
/// excluding biases, norms, scales, and embeddings which are kept as F32.
fn should_quantize_tensor(name: &str, shape: &[usize], data_len: usize) -> bool {
    shape.len() >= 2
        && data_len >= 256 // Minimum size for Q4K (one super-block)
        && !name.contains("bias")
        && !name.contains("norm")
        && !name.contains("scale")
        && !name.contains("embed") // Keep embeddings as F32 for now
}

/// Serialize APR writer output and write the resulting bytes to a file.
fn write_q4k_apr_file(mut writer: AprV2Writer, output: &Path) -> Result<()> {
    use std::io::Write as IoWrite;

    let bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to serialize APR format: {e}"),
    })?;

    let mut file = fs::File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;

    file.write_all(&bytes)
        .map_err(|e| AprenderError::FormatError {
            message: format!("Failed to write APR file: {e}"),
        })
}

/// Save model tensors with Q4K quantization in APR format
///
/// Selectively quantizes large weight tensors while keeping biases and norms as F32.
/// Uses APR format with proper Q4K dtype for GPU-accelerated inference.
fn save_model_tensors_q4k(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
) -> Result<()> {
    let cfg = infer_q4k_config(tensors);
    let param_count: u64 = tensors.values().map(|(data, _)| data.len() as u64).sum();
    let metadata = build_q4k_metadata(&cfg, param_count);
    let mut writer = AprV2Writer::new(metadata);

    // Add tensors, selectively quantizing to Q4K
    // GH-202 FIX: Use quantize_q4_k_matrix for 2D tensors to ensure proper
    // row-aligned block layout. quantize_q4_k treats data as flat, which
    // produces wrong block boundaries when row width != multiple of 256.
    for (name, (data, shape)) in tensors {
        if should_quantize_tensor(name, shape, data.len()) {
            // GH-202 FIX: Use matrix-aware quantization for proper row padding
            let q4k_bytes = quantize_q4_k_matrix(data, shape);
            writer.add_q4k_raw_tensor(name, shape.clone(), q4k_bytes);
        } else {
            // Keep as F32
            writer.add_f32_tensor(name, shape.clone(), data);
        }
    }

    write_q4k_apr_file(writer, output)
}

// ============================================================================

// Write functions extracted to write.rs (PMAT-197)
mod write;
pub(crate) use write::{write_apr_file, write_apr_file_raw};

// Import pipeline extracted to import.rs (PMAT-197)
mod import;
pub use import::apr_import;

// Export functionality extracted to export.rs (PMAT-197)
mod export;
pub use export::{apr_export, ExportFormat, ExportOptions, ExportReport};

// Merge functionality extracted to merge.rs (PMAT-197)
mod merge;
pub use merge::{apr_merge, MergeOptions, MergeReport, MergeStrategy};
// For tests

// Tests extracted to tests.rs (PMAT-197)
#[cfg(test)]
mod tests;
