//! APR Converter - Export to SafeTensors/GGUF (APR-SPEC §4.6)
//! PMAT-197: Extracted from mod.rs for file size reduction

use crate::error::{AprenderError, Result};
use crate::format::converter_types::QuantizationType;
use crate::serialization::safetensors::save_safetensors;
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
            save_safetensors(output_path, &tensors).map_err(|e| AprenderError::FormatError {
                message: format!("Failed to export to SafeTensors: {e}"),
            })?;

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

/// Infer model config.json from tensor shapes (GH-182)
///
/// Creates a HuggingFace-compatible config.json based on tensor analysis.
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

    // Create minimal config.json
    format!(
        r#"{{
  "architectures": ["AutoModelForCausalLM"],
  "hidden_size": {hidden_size},
  "num_hidden_layers": {num_layers},
  "vocab_size": {vocab_size},
  "model_type": "llama"
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
