//! APR Converter - Export to SafeTensors/GGUF (APR-SPEC §4.6)
//! PMAT-197: Extracted from mod.rs for file size reduction

use crate::error::{AprenderError, Result};
use crate::format::converter_types::QuantizationType;
use crate::serialization::safetensors::{
    save_safetensors, save_safetensors_with_metadata, UserMetadata,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

// Import shared functions from parent module
use super::{calculate_tensor_size, load_model_tensors, quantize_tensors};

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// `SafeTensors` format (.safetensors) - `HuggingFace` ecosystem
    SafeTensors,
    /// GGUF format (.gguf) - llama.cpp / local inference
    Gguf,
    /// ONNX format (.onnx) - Cross-framework inference (not yet implemented)
    Onnx,
    /// `TorchScript` format (.pt) - `PyTorch` deployment (not yet implemented)
    TorchScript,
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "safetensors" | "st" => Ok(Self::SafeTensors),
            "gguf" => Ok(Self::Gguf),
            "onnx" => Ok(Self::Onnx),
            "torchscript" | "pt" | "torch" => Ok(Self::TorchScript),
            _ => Err(format!("Unknown export format: {s}")),
        }
    }
}

impl ExportFormat {
    /// Get default file extension
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Gguf => "gguf",
            Self::Onnx => "onnx",
            Self::TorchScript => "pt",
        }
    }

    /// Check if format is supported
    #[must_use]
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::Gguf)
    }
}

/// Options for model export
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Target format
    pub format: ExportFormat,
    /// Optional quantization
    pub quantize: Option<QuantizationType>,
    /// GH-182: Include tokenizer.json companion file (SafeTensors only)
    pub include_tokenizer: bool,
    /// GH-182: Include config.json companion file (SafeTensors only)
    pub include_config: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::SafeTensors,
            quantize: None,
            include_tokenizer: true, // Default to true for HuggingFace compatibility
            include_config: true,
        }
    }
}

/// Report from export operation
#[derive(Debug, Clone)]
pub struct ExportReport {
    /// Original size in bytes
    pub original_size: usize,
    /// Exported size in bytes
    pub exported_size: usize,
    /// Number of tensors exported
    pub tensor_count: usize,
    /// Export format used
    pub format: ExportFormat,
    /// Quantization applied
    pub quantization: Option<QuantizationType>,
}

/// Export APR/SafeTensors model to another format
///
/// # Arguments
///
/// * `input` - Input model path (.apr or .safetensors)
/// * `output` - Output file path
/// * `options` - Export options
///
/// # Returns
///
/// Export report with size and format information
///
/// # Errors
///
/// Returns error if:
/// - Input file doesn't exist
/// - Format not supported
/// - Export fails
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{apr_export, ExportOptions, ExportFormat};
///
/// let options = ExportOptions {
///     format: ExportFormat::Gguf,
///     quantize: None,
/// };
/// let report = apr_export("model.apr", "model.gguf", options)?;
/// ```
pub fn apr_export<P: AsRef<Path>>(
    input: P,
    output: P,
    options: ExportOptions,
) -> Result<ExportReport> {
    let input_path = input.as_ref();
    let output_path = output.as_ref();

    // Validate input exists
    if !input_path.exists() {
        return Err(AprenderError::FormatError {
            message: format!("Input file not found: {}", input_path.display()),
        });
    }

    // Check if format is supported
    if !options.format.is_supported() {
        return Err(AprenderError::FormatError {
            message: format!(
                "Export format {:?} is not yet supported. Use 'safetensors' or 'gguf'.",
                options.format
            ),
        });
    }

    // Load tensors
    let tensors = load_model_tensors(input_path)?;
    let original_size = calculate_tensor_size(&tensors);
    let original_count = tensors.len();

    // Apply quantization if requested
    let tensors = if let Some(ref quant_type) = options.quantize {
        quantize_tensors(&tensors, quant_type)?
    } else {
        tensors
    };

    // Export to target format
    match options.format {
        ExportFormat::SafeTensors => {
            // PMAT-223: Extract user metadata from APR custom field for round-trip
            let user_metadata = extract_user_metadata(input_path);
            if user_metadata.is_empty() {
                save_safetensors(output_path, &tensors).map_err(|e| {
                    AprenderError::FormatError {
                        message: format!("Failed to export to SafeTensors: {e}"),
                    }
                })?;
            } else {
                eprintln!(
                    "[PMAT-223] Restoring {} user metadata key(s) to SafeTensors __metadata__",
                    user_metadata.len()
                );
                save_safetensors_with_metadata(output_path, &tensors, &user_metadata).map_err(
                    |e| AprenderError::FormatError {
                        message: format!("Failed to export to SafeTensors: {e}"),
                    },
                )?;
            }

            // GH-182: Write companion files alongside SafeTensors
            let output_dir = output_path.parent().unwrap_or(Path::new("."));

            if options.include_config {
                let config = infer_model_config(&tensors);
                let config_path = output_dir.join("config.json");
                if let Err(e) = fs::write(&config_path, config) {
                    eprintln!("[GH-182] Warning: Failed to write config.json: {e}");
                }
            }

            if options.include_tokenizer {
                // Try to extract tokenizer from APR input if available
                let tokenizer_json = infer_tokenizer_json(input_path);
                if !tokenizer_json.is_empty() {
                    let tokenizer_path = output_dir.join("tokenizer.json");
                    if let Err(e) = fs::write(&tokenizer_path, &tokenizer_json) {
                        eprintln!("[GH-182] Warning: Failed to write tokenizer.json: {e}");
                    }
                }
            }
        }
        ExportFormat::Gguf => {
            export_to_gguf(&tensors, output_path)?;
        }
        ExportFormat::Onnx | ExportFormat::TorchScript => {
            return Err(AprenderError::FormatError {
                message: format!("Export format {:?} is not yet implemented", options.format),
            });
        }
    }

    // Get exported file size
    let exported_size = fs::metadata(output_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    Ok(ExportReport {
        original_size,
        exported_size,
        tensor_count: original_count,
        format: options.format,
        quantization: options.quantize,
    })
}

/// Export tensors to GGUF format
fn export_to_gguf(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>, output: &Path) -> Result<()> {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use std::fs::File;
    use std::io::BufWriter;

    // Convert tensors to GGUF format
    let gguf_tensors: Vec<GgufTensor> = tensors
        .iter()
        .map(|(name, (data, shape))| {
            // Convert f32 data to bytes
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

            GgufTensor {
                name: name.clone(),
                shape: shape.iter().map(|&d| d as u64).collect(),
                dtype: GgmlType::F32,
                data: bytes,
            }
        })
        .collect();

    // Basic metadata
    let metadata = vec![
        (
            "general.name".to_string(),
            GgufValue::String("model".to_string()),
        ),
        (
            "general.quantization_version".to_string(),
            GgufValue::Uint32(1),
        ),
    ];

    // Write to file
    let file = File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;
    let mut writer = BufWriter::new(file);

    export_tensors_to_gguf(&mut writer, &gguf_tensors, &metadata)
}

// ============================================================================
// GH-182: COMPANION FILE HELPERS
// ============================================================================

/// Infer model config.json from tensor shapes (GH-182, GH-193)
///
/// Creates a HuggingFace-compatible config.json based on tensor analysis.
/// GH-193: Now includes all required fields for SafeTensors inference:
/// - num_attention_heads, intermediate_size, max_position_embeddings, etc.
fn infer_model_config(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> String {
    // Infer hidden_size from embedding or first layer weight
    let hidden_size = tensors
        .iter()
        .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embd"))
        .map(|(_, (_, shape))| shape.last().copied().unwrap_or(4096))
        .unwrap_or(4096);

    // Count layers by looking for layer patterns
    let num_layers = tensors
        .keys()
        .filter_map(|name| {
            if name.contains("layers.") || name.contains("blk.") {
                // Extract layer number from "layers.N." or "blk.N."
                let parts: Vec<&str> = name.split(&['.', '_'][..]).collect();
                for (i, part) in parts.iter().enumerate() {
                    if (*part == "layers" || *part == "blk") && i + 1 < parts.len() {
                        return parts[i + 1].parse::<usize>().ok();
                    }
                }
            }
            None
        })
        .max()
        .map(|n| n + 1)
        .unwrap_or(12);

    // Infer vocab_size from lm_head or output weight
    let vocab_size = tensors
        .iter()
        .find(|(name, _)| name.contains("lm_head") || name.contains("output.weight"))
        .map(|(_, (_, shape))| shape.first().copied().unwrap_or(32000))
        .unwrap_or(32000);

    // GH-193: Infer num_attention_heads from attention Q/K/V weights
    // Shape is typically [hidden_size, num_heads * head_dim]
    let num_attention_heads = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("self_attn.q_proj")
                || name.contains("attn.q_proj")
                || name.contains("attention.wq")
        })
        .map(|(_, (_, _shape))| {
            // Common head dimensions: 64, 128 - infer num_heads from hidden_size
            // Most models use head_dim = hidden_size / num_heads
            // Common configs: 4096/32=128 head_dim, 2048/16=128, etc.
            let head_dim = if hidden_size >= 4096 { 128 } else { 64 };
            hidden_size / head_dim
        })
        .unwrap_or_else(|| {
            // Fallback: standard ratios
            match hidden_size {
                896 => 14,                       // Qwen2.5-0.5B
                1536 => 12,                      // Qwen2.5-1.5B
                2048 => 16,                      // Llama-7B style
                4096 => 32,                      // Llama-7B
                5120 => 40,                      // Llama-13B
                8192 => 64,                      // Llama-70B
                _ => (hidden_size / 128).max(1), // Default: head_dim=128
            }
        });

    // GH-193: Infer intermediate_size from MLP weights
    let intermediate_size = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("mlp.gate_proj")
                || name.contains("mlp.up_proj")
                || name.contains("feed_forward.w1")
        })
        .map(|(_, (_, shape))| shape.first().copied().unwrap_or(hidden_size * 4))
        .unwrap_or(hidden_size * 4); // Default to 4x hidden_size (common in transformers)

    // GH-193: Infer head_dim (guard against division by zero)
    let head_dim = if num_attention_heads > 0 {
        hidden_size / num_attention_heads
    } else {
        64 // Default head dimension
    };

    // GH-193: Infer num_key_value_heads (GQA support)
    // Look for k_proj shape to detect if using GQA (grouped query attention)
    let num_key_value_heads = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("self_attn.k_proj")
                || name.contains("attn.k_proj")
                || name.contains("attention.wk")
        })
        .map(|(_, (_, shape))| {
            // For GQA: k_proj shape is [hidden_size, num_kv_heads * head_dim]
            // If shape[0] < hidden_size, it's GQA
            let kv_dim = shape.first().copied().unwrap_or(hidden_size);
            // Guard against division by zero
            if head_dim > 0 {
                (kv_dim / head_dim).max(1)
            } else {
                1
            }
        })
        .unwrap_or(num_attention_heads); // Default: same as num_attention_heads (MHA)

    // Create HuggingFace-compatible config.json with all required fields (GH-193)
    format!(
        r#"{{
  "architectures": ["Qwen2ForCausalLM"],
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": {hidden_size},
  "initializer_range": 0.02,
  "intermediate_size": {intermediate_size},
  "max_position_embeddings": 32768,
  "model_type": "qwen2",
  "num_attention_heads": {num_attention_heads},
  "num_hidden_layers": {num_layers},
  "num_key_value_heads": {num_key_value_heads},
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": {vocab_size}
}}"#
    )
}

/// Extract tokenizer.json from APR input file (GH-182)
///
/// If the input is APR format with embedded tokenizer, extract it.
/// Otherwise return empty string.
fn infer_tokenizer_json(input_path: &Path) -> String {
    let extension = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    if extension == "apr" {
        // Try to read APR metadata which may contain tokenizer
        if let Ok(data) = fs::read(input_path) {
            if data.len() > 44 {
                // Check for embedded tokenizer in APR metadata
                // APR format: header (44 bytes) + metadata (JSON) + tensors
                // Look for "tokenizer" in metadata section
                let metadata_start = 44;
                if let Some(metadata_end) = data[metadata_start..]
                    .windows(4)
                    .position(|w| w == b"}\n\n\n" || w == b"}\r\n\r")
                    .map(|p| metadata_start + p + 1)
                {
                    if let Ok(metadata_str) =
                        std::str::from_utf8(&data[metadata_start..metadata_end])
                    {
                        // Check if tokenizer data is embedded
                        if metadata_str.contains("\"tokenizer\"")
                            || metadata_str.contains("\"vocabulary\"")
                        {
                            // For now, return a minimal tokenizer.json
                            // In production, we'd extract the actual vocabulary
                            return r#"{"version": "1.0", "model": {"type": "BPE"}}"#.to_string();
                        }
                    }
                }
            }
        }
    }

    // Return empty if no tokenizer found
    String::new()
}

/// PMAT-223: Extract user metadata from APR file's custom field.
///
/// Reads the APR metadata JSON and looks for the `"source_metadata"` key
/// that was preserved during import from SafeTensors.
fn extract_user_metadata(apr_path: &Path) -> UserMetadata {
    let data = match fs::read(apr_path) {
        Ok(d) => d,
        Err(_) => return UserMetadata::new(),
    };

    // APR v2 format: magic(4) + version(4) + metadata_len(8) + metadata_json
    if data.len() < 16 {
        return UserMetadata::new();
    }

    let metadata_len = u64::from_le_bytes(data[8..16].try_into().unwrap_or([0u8; 8])) as usize;

    if data.len() < 16 + metadata_len {
        return UserMetadata::new();
    }

    let metadata_json = match std::str::from_utf8(&data[16..16 + metadata_len]) {
        Ok(s) => s,
        Err(_) => return UserMetadata::new(),
    };

    let parsed: serde_json::Value = match serde_json::from_str(metadata_json) {
        Ok(v) => v,
        Err(_) => return UserMetadata::new(),
    };

    // Look for custom.source_metadata
    if let Some(serde_json::Value::Object(map)) =
        parsed.get("custom").and_then(|c| c.get("source_metadata"))
    {
        let mut result = UserMetadata::new();
        for (k, v) in map {
            if let serde_json::Value::String(s) = v {
                result.insert(k.clone(), s.clone());
            }
        }
        return result;
    }

    UserMetadata::new()
}
