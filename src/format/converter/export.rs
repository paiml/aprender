//! APR Converter - Export to SafeTensors/GGUF (APR-SPEC §4.6)
//! PMAT-197: Extracted from mod.rs for file size reduction

use crate::error::{AprenderError, Result};
use crate::format::converter_types::{Architecture, QuantizationType};
use crate::format::gguf::GgufReader;
use crate::format::layout_contract::contract;
use crate::serialization::safetensors::{
    save_safetensors, save_safetensors_typed, save_safetensors_with_metadata,
    save_safetensors_with_metadata_typed, UserMetadata,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

// Import shared functions from parent module
use super::{
    calculate_tensor_size, load_model_tensors_provenance, map_tensor_names, quantize_tensors,
    NativeF32Tensors,
};
// NOTE: quantize_q4_k_matrix imported via super:: through mod.rs

// GH-202 FIX: Removed transpose_f32_rowmajor_to_colmajor.
// GGML data layout is C row-major with reversed shape [ne0, ne1].
// No data transposition is needed for GGUF export — only shape reversal.

/// GH-236: Resolve architecture name from APR metadata.
///
/// Checks `architecture` field first, then falls back to `model_type` (for models
/// imported without explicit architecture, e.g. GPT-2 from SafeTensors), then
/// defaults to "qwen2" as the most common architecture.
fn resolve_architecture(apr_metadata: &crate::format::v2::AprV2Metadata) -> &str {
    apr_metadata
        .architecture
        .as_deref()
        .or_else(|| {
            let mt = &apr_metadata.model_type;
            if mt.is_empty() || mt == "unknown" {
                None
            } else {
                Some(mt.as_str())
            }
        })
        .unwrap_or("qwen2")
}

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// `SafeTensors` format (.safetensors) - `HuggingFace` ecosystem
    SafeTensors,
    /// GGUF format (.gguf) - llama.cpp / local inference
    Gguf,
    /// MLX format (directory with .safetensors + config.json) - Apple Silicon
    Mlx,
    /// ONNX format (.onnx) - Cross-framework inference (not yet implemented)
    Onnx,
    /// OpenVINO IR format (.xml + .bin) - Intel inference (not yet implemented)
    OpenVino,
    /// CoreML format (.mlpackage) - iOS/macOS deployment (not yet implemented)
    CoreMl,
    /// `TorchScript` format (.pt) - `PyTorch` deployment (not yet implemented)
    TorchScript,
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "safetensors" | "st" => Ok(Self::SafeTensors),
            "gguf" => Ok(Self::Gguf),
            "mlx" => Ok(Self::Mlx),
            "onnx" => Ok(Self::Onnx),
            "openvino" | "ov" => Ok(Self::OpenVino),
            "coreml" | "mlpackage" => Ok(Self::CoreMl),
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
            Self::Mlx => "mlx",
            Self::Onnx => "onnx",
            Self::OpenVino => "xml",
            Self::CoreMl => "mlpackage",
            Self::TorchScript => "pt",
        }
    }

    /// Check if format is supported
    #[must_use]
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::Gguf | Self::Mlx)
    }

    /// Human-readable name for display
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::SafeTensors => "SafeTensors",
            Self::Gguf => "GGUF",
            Self::Mlx => "MLX",
            Self::Onnx => "ONNX",
            Self::OpenVino => "OpenVINO",
            Self::CoreMl => "CoreML",
            Self::TorchScript => "TorchScript",
        }
    }

    /// All known export formats
    #[must_use]
    pub fn all() -> &'static [ExportFormat] {
        &[
            Self::SafeTensors,
            Self::Gguf,
            Self::Mlx,
            Self::Onnx,
            Self::OpenVino,
            Self::CoreMl,
            Self::TorchScript,
        ]
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

/// GH-253-4: Validated GGUF metadata — compile-time enforcement via newtype.
///
/// Private inner data ensures metadata can ONLY be constructed through `validate()`,
/// which checks that required keys are present and consistent.
///
/// Required keys:
/// - `general.architecture` (always)
/// - `tokenizer.ggml.model` (if vocabulary present)
/// - `tokenizer.ggml.tokens` and `tokenizer.ggml.model` must appear together
#[derive(Debug)]
pub(crate) struct ValidatedGgufMetadata {
    inner: Vec<(String, crate::format::gguf::GgufValue)>,
}

impl ValidatedGgufMetadata {
    /// Validate and construct metadata. Fails if required keys are missing.
    pub(crate) fn validate(
        metadata: Vec<(String, crate::format::gguf::GgufValue)>,
    ) -> Result<Self> {
        let has_key = |k: &str| metadata.iter().any(|(name, _)| name == k);

        // REQUIRED: general.architecture must always be present
        if !has_key("general.architecture") {
            return Err(AprenderError::FormatError {
                message: "[GH-253-4] GGUF export missing required key: general.architecture"
                    .to_string(),
            });
        }

        // CONSISTENCY: if tokens present, model type must also be present (and vice versa)
        let has_tokens = has_key("tokenizer.ggml.tokens");
        let has_model = has_key("tokenizer.ggml.model");
        if has_tokens && !has_model {
            return Err(AprenderError::FormatError {
                message:
                    "[GH-253-4] GGUF export has tokenizer.ggml.tokens but missing tokenizer.ggml.model"
                        .to_string(),
            });
        }
        if has_model && !has_tokens {
            return Err(AprenderError::FormatError {
                message:
                    "[GH-253-4] GGUF export has tokenizer.ggml.model but missing tokenizer.ggml.tokens"
                        .to_string(),
            });
        }

        Ok(Self { inner: metadata })
    }

    /// Access validated metadata as a slice for GGUF writing.
    pub(crate) fn as_slice(&self) -> &[(String, crate::format::gguf::GgufValue)] {
        &self.inner
    }
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

    validate_export_inputs(input_path, &options)?;

    // PMAT-252: For APR→GGUF with quantized source, use raw block passthrough
    if let Some(report) = try_raw_gguf_passthrough(input_path, output_path, &options)? {
        return Ok(report);
    }

    let provenance = load_model_tensors_provenance(input_path)?;
    let original_size = calculate_tensor_size(provenance.as_map());
    let tensors = provenance.into_map();

    // PMAT-260: Capture original dtypes from SafeTensors source for round-trip preservation.
    // BF16 tensors must be written back as BF16, not widened to F32.
    let original_dtypes = extract_source_dtypes(input_path);

    let tensors = prepare_tensors_for_export(tensors, input_path, &options);
    enforce_contract_violations(&tensors)?;
    let tensors = apply_export_quantization(tensors, input_path, &options)?;

    dispatch_export(&tensors, input_path, output_path, &options, &original_dtypes)?;

    let exported_size = if output_path.is_dir() {
        // GH-246: For directory-based exports (MLX), sum all file sizes
        dir_total_size(output_path)
    } else {
        fs::metadata(output_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0)
    };
    Ok(ExportReport {
        original_size,
        exported_size,
        tensor_count: tensors.len(),
        format: options.format,
        quantization: options.quantize,
    })
}

/// GH-246: Calculate total size of all files in a directory (for MLX exports).
fn dir_total_size(dir: &Path) -> usize {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len() as usize)
                .sum()
        })
        .unwrap_or(0)
}

/// Validate export inputs (file exists, format supported).
fn validate_export_inputs(input_path: &Path, options: &ExportOptions) -> Result<()> {
    if !input_path.exists() {
        return Err(AprenderError::FormatError {
            message: format!("Input file not found: {}", input_path.display()),
        });
    }
    if !options.format.is_supported() {
        return Err(AprenderError::FormatError {
            message: format!(
                "Export format {:?} is not yet supported. Use 'safetensors', 'gguf', or 'mlx'.",
                options.format
            ),
        });
    }
    Ok(())
}

/// PMAT-252: Try raw block passthrough for quantized APR→GGUF.
fn try_raw_gguf_passthrough(
    input_path: &Path,
    output_path: &Path,
    options: &ExportOptions,
) -> Result<Option<ExportReport>> {
    if options.format != ExportFormat::Gguf
        || options.quantize.is_some()
        || input_path.extension().and_then(|e| e.to_str()) != Some("apr")
    {
        return Ok(None);
    }
    match detect_apr_quantization(input_path) {
        Some(detected) => {
            eprintln!(
                "[PMAT-252] Raw passthrough: detected {detected:?} in APR source. Copying blocks directly (zero loss)."
            );
            export_apr_to_gguf_raw(input_path, output_path).map(Some)
        }
        None => Ok(None),
    }
}

/// Prepare tensors: name mapping, QKV unfusing, lm_head removal.
fn prepare_tensors_for_export(
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    options: &ExportOptions,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    // GH-200: Map GGUF names to HF canonical format
    let tensors = if input_path.extension().and_then(|e| e.to_str()) == Some("gguf") {
        map_tensor_names(&tensors, detect_gguf_architecture(input_path))
    } else {
        tensors
    };
    let tensors = unfuse_qkv_tensors(tensors, input_path);
    if options.format == ExportFormat::SafeTensors {
        remove_tied_lm_head(tensors, input_path)
    } else {
        tensors
    }
}

/// GH-237: Validate tensor shapes against layout contract, returning error on violations.
///
/// Previously this was advisory (eprintln only). Now violations are collected and
/// returned as an error to prevent corrupt data from being written to disk.
fn enforce_contract_violations(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> Result<()> {
    let layout_contract = contract();
    let (vocab_size, hidden_dim) = infer_vocab_hidden(tensors);
    if vocab_size == 0 || hidden_dim == 0 {
        return Ok(());
    }
    let mut violations = Vec::new();
    for (name, (_data, shape)) in tensors {
        if let Err(e) = layout_contract.validate_apr_shape(name, shape, vocab_size, hidden_dim) {
            violations.push(format!("{name}: {e}"));
        }
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(AprenderError::FormatError {
            message: format!(
                "[CONTRACT-VIOLATION] Export validation failed for {} tensor(s):\n  {}",
                violations.len(),
                violations.join("\n  ")
            ),
        })
    }
}

/// DOUBLE-QUANT-001: Apply quantization only to natively F32 tensors.
fn apply_export_quantization(
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    options: &ExportOptions,
) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let Some(ref quant_type) = options.quantize else {
        return Ok(tensors);
    };
    if let Some(detected) = detect_apr_quantization(input_path) {
        return Err(AprenderError::FormatError {
            message: format!(
                "DOUBLE-QUANT-001: Cannot re-quantize to {quant_type:?}: source is already \
                 {detected:?}. Remove --quantize flag to use raw passthrough."
            ),
        });
    }
    let native = NativeF32Tensors::new(tensors);
    Ok(quantize_tensors(&native, quant_type)?.into_inner())
}

/// Dispatch to format-specific export.
/// PMAT-260: Extract original dtype map from a SafeTensors source file.
///
/// Returns a map of tensor name → dtype string (e.g. "BF16", "F16", "F32").
/// Returns empty map for non-SafeTensors sources (APR, GGUF).
fn extract_source_dtypes(input_path: &Path) -> BTreeMap<String, String> {
    if input_path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return BTreeMap::new();
    }
    match crate::serialization::safetensors::MappedSafeTensors::open(input_path) {
        Ok(mapped) => mapped.dtype_map(),
        Err(_) => BTreeMap::new(),
    }
}

fn dispatch_export(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    output_path: &Path,
    options: &ExportOptions,
    original_dtypes: &BTreeMap<String, String>,
) -> Result<()> {
    match options.format {
        ExportFormat::SafeTensors => export_safetensors_with_companions(
            tensors,
            input_path,
            output_path,
            options,
            original_dtypes,
        ),
        ExportFormat::Gguf => {
            export_to_gguf(tensors, output_path, input_path, options.quantize.as_ref())
        }
        ExportFormat::Mlx => export_mlx(tensors, input_path, output_path, options),
        ExportFormat::Onnx
        | ExportFormat::OpenVino
        | ExportFormat::CoreMl
        | ExportFormat::TorchScript => Err(AprenderError::FormatError {
            message: format!(
                "Export format {} is not yet implemented. Supported: safetensors, gguf, mlx",
                options.format.display_name()
            ),
        }),
    }
}

include!("export_part_02.rs");
include!("export_part_03.rs");
include!("export_part_04.rs");
include!("export_part_05.rs");
