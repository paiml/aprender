//! APR Converter - Export to SafeTensors/GGUF (APR-SPEC §4.6)
//! PMAT-197: Extracted from mod.rs for file size reduction

use crate::error::{AprenderError, Result};
use crate::format::converter_types::{Architecture, QuantizationType};
use crate::format::gguf::GgufReader;
use crate::format::layout_contract::contract;
use crate::serialization::safetensors::{
    save_safetensors, save_safetensors_with_metadata, UserMetadata,
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

    let tensors = prepare_tensors_for_export(tensors, input_path, &options);
    enforce_contract_violations(&tensors)?;
    let tensors = apply_export_quantization(tensors, input_path, &options)?;

    dispatch_export(&tensors, input_path, output_path, &options)?;

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
fn dispatch_export(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    output_path: &Path,
    options: &ExportOptions,
) -> Result<()> {
    match options.format {
        ExportFormat::SafeTensors => {
            export_safetensors_with_companions(tensors, input_path, output_path, options)
        }
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

/// Export to SafeTensors with optional companion files (config.json, tokenizer.json)
fn export_safetensors_with_companions(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    output_path: &Path,
    options: &ExportOptions,
) -> Result<()> {
    // PMAT-223: Extract user metadata from APR custom field for round-trip
    let user_metadata = extract_user_metadata(input_path);
    if user_metadata.is_empty() {
        save_safetensors(output_path, tensors).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to export to SafeTensors: {e}"),
        })?;
    } else {
        eprintln!(
            "[PMAT-223] Restoring {} user metadata key(s) to SafeTensors __metadata__",
            user_metadata.len()
        );
        save_safetensors_with_metadata(output_path, tensors, &user_metadata).map_err(|e| {
            AprenderError::FormatError {
                message: format!("Failed to export to SafeTensors: {e}"),
            }
        })?;
    }

    // GH-182: Write companion files alongside SafeTensors
    let output_dir = output_path.parent().unwrap_or(Path::new("."));

    if options.include_config {
        let config = infer_model_config(tensors);
        let config_path = output_dir.join("config.json");
        if let Err(e) = fs::write(&config_path, config) {
            eprintln!("[GH-182] Warning: Failed to write config.json: {e}");
        }
    }

    if options.include_tokenizer {
        let tokenizer_json = infer_tokenizer_json(input_path);
        if !tokenizer_json.is_empty() {
            let tokenizer_path = output_dir.join("tokenizer.json");
            if let Err(e) = fs::write(&tokenizer_path, &tokenizer_json) {
                eprintln!("[GH-182] Warning: Failed to write tokenizer.json: {e}");
            }
        }
    }

    Ok(())
}

/// Export tensors to GGUF format (GGUF-EXPORT-001 fix)
///
/// Reads APR metadata to populate GGUF KV pairs and maps tensor names
/// from HuggingFace convention to GGUF convention.
///
/// BUG-1 FIX: Now supports Q4_K quantization for GGUF inference compatibility.
/// F32 GGUF files don't work with realizar's fused matmul kernels.
///
/// BUG-EXPORT-004 FIX: Now includes tokenizer metadata for realizar inference.
/// Without BOS/EOS token IDs, the model produces empty output.
/// Resolved GGUF export configuration (APR metadata with inferred fallbacks).
struct GgufExportConfig {
    arch: String,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    vocab_size: usize,
    intermediate_size: usize,
    max_pos: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    head_dim: usize,
    model_name: String,
}

/// Resolve GGUF export config from APR metadata + inferred fallbacks.
fn resolve_gguf_config(
    apr_metadata: Option<&crate::format::v2::AprV2Metadata>,
    inferred: Option<&crate::format::gguf::GgufModelConfig>,
) -> GgufExportConfig {
    /// Resolve a field: APR metadata → inferred → default.
    fn resolve<T: Copy>(
        apr: Option<&crate::format::v2::AprV2Metadata>,
        inf: Option<&crate::format::gguf::GgufModelConfig>,
        apr_f: impl Fn(&crate::format::v2::AprV2Metadata) -> Option<T>,
        inf_f: impl Fn(&crate::format::gguf::GgufModelConfig) -> Option<T>,
        default: T,
    ) -> T {
        apr.and_then(&apr_f)
            .or_else(|| inf.and_then(&inf_f))
            .unwrap_or(default)
    }

    let num_heads = resolve(apr_metadata, inferred, |m| m.num_heads, |c| c.num_heads, 32);
    let hidden_size = resolve(
        apr_metadata,
        inferred,
        |m| m.hidden_size,
        |c| c.hidden_size,
        4096,
    );

    GgufExportConfig {
        arch: apr_metadata
            .and_then(|m| m.architecture.clone())
            .or_else(|| inferred.and_then(|c| c.architecture.clone()))
            .unwrap_or_else(|| "qwen2".to_string()),
        hidden_size,
        num_layers: resolve(
            apr_metadata,
            inferred,
            |m| m.num_layers,
            |c| c.num_layers,
            32,
        ),
        num_heads,
        num_kv_heads: resolve(
            apr_metadata,
            inferred,
            |m| m.num_kv_heads,
            |c| c.num_kv_heads,
            num_heads,
        ),
        vocab_size: resolve(
            apr_metadata,
            inferred,
            |m| m.vocab_size,
            |c| c.vocab_size,
            32000,
        ),
        intermediate_size: resolve(
            apr_metadata,
            inferred,
            |m| m.intermediate_size,
            |c| c.intermediate_size,
            11008,
        ),
        max_pos: apr_metadata
            .and_then(|m| m.max_position_embeddings)
            .unwrap_or(32768),
        rope_theta: apr_metadata
            .and_then(|m| m.rope_theta)
            .unwrap_or(1_000_000.0),
        rms_norm_eps: apr_metadata.and_then(|m| m.rms_norm_eps).unwrap_or(1e-6),
        head_dim: if num_heads > 0 {
            hidden_size / num_heads
        } else {
            128
        },
        model_name: apr_metadata
            .and_then(|m| m.name.clone())
            .unwrap_or_else(|| "model".to_string()),
    }
}

/// Build GGUF architecture metadata KV pairs from resolved config.
fn build_gguf_config_metadata(
    cfg: &GgufExportConfig,
) -> Vec<(String, crate::format::gguf::GgufValue)> {
    use crate::format::gguf::GgufValue;
    let arch = &cfg.arch;
    vec![
        (
            "general.architecture".to_string(),
            GgufValue::String(arch.clone()),
        ),
        (
            "general.name".to_string(),
            GgufValue::String(cfg.model_name.clone()),
        ),
        (
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        ),
        ("general.file_type".to_string(), GgufValue::Uint32(0)),
        (
            format!("{arch}.context_length"),
            GgufValue::Uint32(cfg.max_pos as u32),
        ),
        (
            format!("{arch}.embedding_length"),
            GgufValue::Uint32(cfg.hidden_size as u32),
        ),
        (
            format!("{arch}.block_count"),
            GgufValue::Uint32(cfg.num_layers as u32),
        ),
        (
            format!("{arch}.feed_forward_length"),
            GgufValue::Uint32(cfg.intermediate_size as u32),
        ),
        (
            format!("{arch}.attention.head_count"),
            GgufValue::Uint32(cfg.num_heads as u32),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            GgufValue::Uint32(cfg.num_kv_heads as u32),
        ),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::Float32(cfg.rms_norm_eps),
        ),
        (
            format!("{arch}.rope.dimension_count"),
            GgufValue::Uint32(cfg.head_dim as u32),
        ),
        (
            format!("{arch}.rope.freq_base"),
            GgufValue::Float32(cfg.rope_theta),
        ),
        (
            format!("{arch}.vocab_size"),
            GgufValue::Uint32(cfg.vocab_size as u32),
        ),
    ]
}

/// Build tokenizer metadata KV pairs for GGUF export.
fn build_tokenizer_gguf_metadata(
    tokenizer: &crate::format::gguf::GgufTokenizer,
    arch: &str,
) -> Vec<(String, crate::format::gguf::GgufValue)> {
    use crate::format::gguf::GgufValue;
    let mut metadata = Vec::new();
    let model_type = tokenizer.model_type.as_deref().unwrap_or("gpt2");

    metadata.push((
        "tokenizer.ggml.model".to_string(),
        GgufValue::String(model_type.to_lowercase()),
    ));
    metadata.push((
        "tokenizer.ggml.pre".to_string(),
        GgufValue::String(arch.to_string()),
    ));

    if let Some(bos) = tokenizer.bos_token_id {
        metadata.push((
            "tokenizer.ggml.bos_token_id".to_string(),
            GgufValue::Uint32(bos),
        ));
    }
    if let Some(eos) = tokenizer.eos_token_id {
        metadata.push((
            "tokenizer.ggml.eos_token_id".to_string(),
            GgufValue::Uint32(eos),
        ));
    }
    if !tokenizer.vocabulary.is_empty() {
        metadata.push((
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayString(tokenizer.vocabulary.clone()),
        ));
        eprintln!(
            "[BUG-EXPORT-004] Added tokenizer metadata: model={}, vocab_size={}, bos={:?}, eos={:?}",
            model_type, tokenizer.vocabulary.len(), tokenizer.bos_token_id, tokenizer.eos_token_id
        );
    }
    if !tokenizer.merges.is_empty() {
        metadata.push((
            "tokenizer.ggml.merges".to_string(),
            GgufValue::ArrayString(tokenizer.merges.clone()),
        ));
    }
    metadata
}

fn export_to_gguf(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    input: &Path,
    quantize: Option<&QuantizationType>,
) -> Result<()> {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor};
    use crate::format::v2::AprV2Reader;
    use std::fs::File;
    use std::io::BufWriter;

    let tokenizer = super::import::load_tokenizer_from_json(input);

    let apr_metadata = if input.extension().and_then(|e| e.to_str()) == Some("apr") {
        fs::read(input)
            .ok()
            .and_then(|d| AprV2Reader::from_bytes(&d).ok())
            .map(|r| r.metadata().clone())
    } else {
        None
    };
    let inferred = super::import::infer_model_config_from_tensors(tensors);
    let cfg = resolve_gguf_config(apr_metadata.as_ref(), inferred.as_ref());

    let mut metadata = build_gguf_config_metadata(&cfg);
    append_tokenizer_to_metadata(
        &mut metadata,
        tokenizer.as_ref(),
        apr_metadata.as_ref(),
        &cfg.arch,
        input,
    );

    eprintln!(
        "[GGUF-EXPORT-001] Writing {} metadata keys (arch={}, layers={}, heads={}/{}kv, hidden={})",
        metadata.len(),
        cfg.arch,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.hidden_size
    );

    // GGUF-EXPORT-001: Map tensor names from HF convention to GGUF convention
    // PMAT-222 FIX: Reverse 2D shapes from standard [rows, cols] to GGML [ne0, ne1]
    // GGML convention: ne[0] is the contiguous dimension (cols), ne[1] is rows.
    // This is the inverse of write.rs:520 which reverses GGML→standard on import.
    //
    // BUG-1 FIX: Support Q4_K quantization for GGUF inference compatibility.
    // F32 GGUF files don't work with realizar's fused matmul kernels which
    // only support Q4_0/Q8_0/Q4_K/Q5_K/Q6_K types.
    let use_q4k = matches!(
        quantize,
        Some(QuantizationType::Q4K | QuantizationType::Int4)
    );

    // GH-202 FIX: Build GGUF tensors WITHOUT data transpose.
    //
    // CRITICAL INSIGHT: GGML data layout data[i0 + i1*ne0] is IDENTICAL to
    // C row-major data[row*cols + col] when shape is reversed. The data does
    // NOT need transposing for GGUF export — only shape needs reversal from
    // standard [rows, cols] back to GGML [ne0=cols, ne1=rows].
    let gguf_tensors: Vec<GgufTensor> = tensors
        .iter()
        .map(|(name, (data, shape))| {
            let gguf_name = hf_to_gguf_name(name);

            // lm_head and embeddings skip quantization — keep F32 to preserve full precision
            let is_embedding = gguf_name == "token_embd.weight" || name.contains("embed_tokens");
            let is_lm_head = gguf_name == "output.weight" || name.contains("lm_head");

            // Reverse shape for GGUF: [rows, cols] → [ne0=cols, ne1=rows]
            let gguf_shape = if shape.len() == 2 {
                vec![shape[1] as u64, shape[0] as u64]
            } else {
                shape.iter().map(|&d| d as u64).collect()
            };

            // GH-202 FIX: No data transpose needed. Data is row-major in APR,
            // and GGML's layout with reversed shape is identical.
            let (dtype, bytes) =
                if use_q4k && shape.len() == 2 && data.len() >= 256 && !is_embedding && !is_lm_head
                {
                    // Quantize row-major F32 to Q4K using GGUF shape [ne0, ne1]
                    // quantize_q4_k_matrix processes per-row with ne0 elements per row
                    let gguf_shape_usize = vec![shape[1], shape[0]]; // [ne0=cols, ne1=rows]
                    let q4k_bytes = super::quantize_q4_k_matrix(data, &gguf_shape_usize);
                    (GgmlType::Q4K, q4k_bytes)
                } else {
                    // F32 (weights, embeddings, 1D) - just convert to bytes, no transpose
                    let f32_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                    (GgmlType::F32, f32_bytes)
                };

            GgufTensor {
                name: gguf_name,
                shape: gguf_shape,
                dtype,
                data: bytes,
            }
        })
        .collect();

    // BUG-4 FIX: For tied embedding models, create Q4K output.weight from embedding
    let has_lm_head = gguf_tensors.iter().any(|t| t.name == "output.weight");
    let mut gguf_tensors = gguf_tensors;

    if use_q4k && !has_lm_head {
        if let Some(tied) = build_tied_output_weight(tensors) {
            gguf_tensors.push(tied);
        }
    }

    // Write to file
    let file = File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;
    let mut writer = BufWriter::new(file);

    export_tensors_to_gguf(&mut writer, &gguf_tensors, &metadata)
}

/// Append tokenizer metadata to GGUF metadata, preferring tokenizer.json over APR fallback.
fn append_tokenizer_to_metadata(
    metadata: &mut Vec<(String, crate::format::gguf::GgufValue)>,
    tokenizer: Option<&crate::format::gguf::GgufTokenizer>,
    apr_metadata: Option<&crate::format::v2::AprV2Metadata>,
    arch: &str,
    input: &Path,
) {
    if let Some(tok) = tokenizer {
        metadata.extend(build_tokenizer_gguf_metadata(tok, arch));
        return;
    }

    eprintln!(
        "[BUG-EXPORT-004] Warning: No tokenizer.json found near {}, GGUF may lack tokenizer metadata",
        input.display()
    );

    // GH-211: Fallback — extract tokenizer from APR metadata when no tokenizer.json
    let Some(apr_meta) = apr_metadata else {
        return;
    };
    let apr_tok_entries = extract_apr_tokenizer_for_gguf(apr_meta);
    if !apr_tok_entries.is_empty() {
        eprintln!(
            "[GH-211] Extracted {} tokenizer entries from APR metadata",
            apr_tok_entries.len()
        );
        metadata.extend(apr_tok_entries);
    }
}

/// Build a Q4K output.weight tensor from embedding data for tied-embedding models (BUG-4).
fn build_tied_output_weight(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Option<crate::format::gguf::GgufTensor> {
    use crate::format::gguf::{GgmlType, GgufTensor};

    let (_, (data, shape)) = tensors
        .iter()
        .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embedding"))?;

    if shape.len() != 2 || data.len() < 256 {
        return None;
    }

    eprintln!("[BUG-4-FIX] Creating Q4K output.weight from embedding for tied embeddings");

    let gguf_shape_usize = vec![shape[1], shape[0]]; // [ne0=cols, ne1=rows]
    let q4k_bytes = super::quantize_q4_k_matrix(data, &gguf_shape_usize);
    let gguf_shape = vec![shape[1] as u64, shape[0] as u64];

    Some(GgufTensor {
        name: "output.weight".to_string(),
        shape: gguf_shape,
        dtype: GgmlType::Q4K,
        data: q4k_bytes,
    })
}

/// Build GGUF architecture metadata from APR model metadata
fn build_gguf_arch_metadata(
    apr_metadata: &crate::format::v2::AprV2Metadata,
) -> Vec<(String, crate::format::gguf::GgufValue)> {
    use crate::format::gguf::GgufValue;

    let arch = resolve_architecture(apr_metadata);
    let hidden_size = apr_metadata.hidden_size.unwrap_or(4096);
    let num_layers = apr_metadata.num_layers.unwrap_or(32);
    let num_heads = apr_metadata.num_heads.unwrap_or(32);
    let num_kv_heads = apr_metadata.num_kv_heads.unwrap_or(num_heads);
    let vocab_size = apr_metadata.vocab_size.unwrap_or(32000);
    let intermediate_size = apr_metadata.intermediate_size.unwrap_or(11008);
    let max_pos = apr_metadata.max_position_embeddings.unwrap_or(32768);
    let rope_theta = apr_metadata.rope_theta.unwrap_or(1_000_000.0);
    let rms_norm_eps = apr_metadata.rms_norm_eps.unwrap_or(1e-6);
    let head_dim = if num_heads > 0 {
        hidden_size / num_heads
    } else {
        128
    };
    let model_name = apr_metadata
        .name
        .clone()
        .unwrap_or_else(|| "model".to_string());

    vec![
        (
            "general.architecture".to_string(),
            GgufValue::String(arch.to_string()),
        ),
        ("general.name".to_string(), GgufValue::String(model_name)),
        (
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        ),
        ("general.file_type".to_string(), GgufValue::Uint32(0)),
        (
            format!("{arch}.context_length"),
            GgufValue::Uint32(max_pos as u32),
        ),
        (
            format!("{arch}.embedding_length"),
            GgufValue::Uint32(hidden_size as u32),
        ),
        (
            format!("{arch}.block_count"),
            GgufValue::Uint32(num_layers as u32),
        ),
        (
            format!("{arch}.feed_forward_length"),
            GgufValue::Uint32(intermediate_size as u32),
        ),
        (
            format!("{arch}.attention.head_count"),
            GgufValue::Uint32(num_heads as u32),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            GgufValue::Uint32(num_kv_heads as u32),
        ),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::Float32(rms_norm_eps),
        ),
        (
            format!("{arch}.rope.dimension_count"),
            GgufValue::Uint32(head_dim as u32),
        ),
        (
            format!("{arch}.rope.freq_base"),
            GgufValue::Float32(rope_theta),
        ),
        (
            format!("{arch}.vocab_size"),
            GgufValue::Uint32(vocab_size as u32),
        ),
    ]
}

/// Push a string array from APR custom fields to GGUF entries.
fn push_string_array(
    entries: &mut Vec<(String, crate::format::gguf::GgufValue)>,
    custom: &std::collections::HashMap<String, serde_json::Value>,
    src_key: &str,
    gguf_key: &str,
) {
    let arr = custom.get(src_key).and_then(|v| v.as_array());
    let Some(arr) = arr else { return };
    let strings: Vec<String> = arr
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    if !strings.is_empty() {
        entries.push((
            gguf_key.to_string(),
            crate::format::gguf::GgufValue::ArrayString(strings),
        ));
    }
}

/// Push a u32 value from APR custom fields to GGUF entries.
fn push_u32_field(
    entries: &mut Vec<(String, crate::format::gguf::GgufValue)>,
    custom: &std::collections::HashMap<String, serde_json::Value>,
    src_key: &str,
    gguf_key: &str,
) {
    if let Some(val) = custom.get(src_key).and_then(|v| v.as_u64()) {
        entries.push((
            gguf_key.to_string(),
            crate::format::gguf::GgufValue::Uint32(val as u32),
        ));
    }
}

/// Push an i32 array from APR custom fields to GGUF entries.
fn push_i32_array(
    entries: &mut Vec<(String, crate::format::gguf::GgufValue)>,
    custom: &std::collections::HashMap<String, serde_json::Value>,
    src_key: &str,
    gguf_key: &str,
) {
    let arr = custom.get(src_key).and_then(|v| v.as_array());
    let Some(arr) = arr else { return };
    let types: Vec<i32> = arr
        .iter()
        .filter_map(|v| v.as_i64().map(|n| n as i32))
        .collect();
    if !types.is_empty() {
        entries.push((
            gguf_key.to_string(),
            crate::format::gguf::GgufValue::ArrayInt32(types),
        ));
    }
}

/// Extract tokenizer metadata from APR custom fields for GGUF export (GH-253)
fn extract_apr_tokenizer_for_gguf(
    apr_metadata: &crate::format::v2::AprV2Metadata,
) -> Vec<(String, crate::format::gguf::GgufValue)> {
    use crate::format::gguf::GgufValue;

    let mut entries = Vec::new();
    let custom = &apr_metadata.custom;
    let arch = resolve_architecture(apr_metadata);

    // Tokenizer model type: "gpt2" for byte-level BPE (Qwen, GPT-2), "llama" for SentencePiece
    // GH-253-3: APR stores raw model_type from GGUF which may be "bpe" — map to "gpt2"
    let raw_model_type = custom
        .get("tokenizer.model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt2");
    let model_type = match raw_model_type {
        "bpe" => "gpt2",
        other => other,
    };
    entries.push((
        "tokenizer.ggml.model".to_string(),
        GgufValue::String(model_type.to_string()),
    ));
    entries.push((
        "tokenizer.ggml.pre".to_string(),
        GgufValue::String(arch.to_string()),
    ));

    push_string_array(
        &mut entries,
        custom,
        "tokenizer.vocabulary",
        "tokenizer.ggml.tokens",
    );
    push_string_array(
        &mut entries,
        custom,
        "tokenizer.merges",
        "tokenizer.ggml.merges",
    );
    push_u32_field(
        &mut entries,
        custom,
        "tokenizer.bos_token_id",
        "tokenizer.ggml.bos_token_id",
    );
    push_u32_field(
        &mut entries,
        custom,
        "tokenizer.eos_token_id",
        "tokenizer.ggml.eos_token_id",
    );
    push_i32_array(
        &mut entries,
        custom,
        "tokenizer.token_type",
        "tokenizer.ggml.token_type",
    );
    push_u32_field(
        &mut entries,
        custom,
        "tokenizer.padding_token_id",
        "tokenizer.ggml.padding_token_id",
    );

    // GH-253-1: add_bos_token flag
    if let Some(add_bos) = custom
        .get("tokenizer.add_bos_token")
        .and_then(|v| v.as_bool())
    {
        entries.push((
            "tokenizer.ggml.add_bos_token".to_string(),
            GgufValue::Bool(add_bos),
        ));
    }

    // GH-253-1: Chat template (Jinja2)
    let chat_tmpl = apr_metadata.chat_template.as_deref().or_else(|| {
        custom
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
    });
    if let Some(tmpl) = chat_tmpl {
        entries.push((
            "tokenizer.chat_template".to_string(),
            GgufValue::String(tmpl.to_string()),
        ));
    }

    entries
}

/// GH-246: Export to MLX format (Apple Silicon).
///
/// MLX models are stored as a directory containing:
/// - `model.safetensors` — weights in SafeTensors format
/// - `config.json` — model configuration (HuggingFace-compatible)
/// - `tokenizer.json` — tokenizer (optional, from APR metadata)
///
/// This reuses the SafeTensors export path since MLX uses SafeTensors as its
/// underlying weight format. The key difference is the directory structure.
fn export_mlx(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    output_path: &Path,
    options: &ExportOptions,
) -> Result<()> {
    // Output path is the directory
    fs::create_dir_all(output_path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create MLX output directory: {e}"),
    })?;

    // Write model.safetensors
    let weights_path = output_path.join("model.safetensors");
    let user_metadata = extract_user_metadata(input_path);
    if user_metadata.is_empty() {
        save_safetensors(&weights_path, tensors).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to write MLX weights: {e}"),
        })?;
    } else {
        save_safetensors_with_metadata(&weights_path, tensors, &user_metadata).map_err(|e| {
            AprenderError::FormatError {
                message: format!("Failed to write MLX weights: {e}"),
            }
        })?;
    }

    // Write config.json
    let config = infer_model_config(tensors);
    let config_path = output_path.join("config.json");
    fs::write(&config_path, config).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write MLX config.json: {e}"),
    })?;

    // Write tokenizer.json if available
    if options.include_tokenizer {
        let tokenizer_json = infer_tokenizer_json(input_path);
        if !tokenizer_json.is_empty() {
            let tokenizer_path = output_path.join("tokenizer.json");
            if let Err(e) = fs::write(&tokenizer_path, &tokenizer_json) {
                eprintln!("[GH-246] Warning: Failed to write tokenizer.json: {e}");
            }
        }
    }

    Ok(())
}

/// PMAT-252: Raw block passthrough for APR→GGUF export.
///
/// Reads raw tensor bytes directly from APR file (Q4K super-blocks, F32 vectors,
/// etc.) and writes them to GGUF without any dequantization/requantization.
/// This is LOSSLESS for quantized data — zero quality degradation.
///
/// The key insight: APR and GGUF both store Q4K blocks in the same binary format
/// (256-element super-blocks, 144 bytes each). The only differences are:
/// 1. Tensor names (HF convention in APR → GGML convention in GGUF)
/// 2. Shape representation (APR [rows, cols] → GGUF [ne0=cols, ne1=rows])
/// 3. File-level metadata (APR header → GGUF KV pairs)
fn export_apr_to_gguf_raw(input: &Path, output: &Path) -> Result<ExportReport> {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor};
    use crate::format::v2::{AprV2Reader, TensorDType};
    use std::fs::File;
    use std::io::BufWriter;

    let data = fs::read(input).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to read APR file: {e}"),
    })?;
    let original_size = data.len();

    let reader = AprV2Reader::from_bytes(&data).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to parse APR file: {e:?}"),
    })?;

    let apr_metadata = reader.metadata().clone();

    let arch = resolve_architecture(&apr_metadata);
    let num_layers = apr_metadata.num_layers.unwrap_or(32);
    let num_heads = apr_metadata.num_heads.unwrap_or(32);
    let num_kv_heads = apr_metadata.num_kv_heads.unwrap_or(num_heads);
    let hidden_size = apr_metadata.hidden_size.unwrap_or(4096);

    // Build metadata from architecture config + tokenizer custom fields
    let mut metadata = build_gguf_arch_metadata(&apr_metadata);
    metadata.extend(extract_apr_tokenizer_for_gguf(&apr_metadata));

    // GH-253-4: Validate metadata completeness before writing
    let validated = ValidatedGgufMetadata::validate(metadata)?;

    eprintln!(
        "[PMAT-252] Writing {} metadata keys (arch={}, layers={}, heads={}/{}kv, hidden={})",
        validated.as_slice().len(),
        arch,
        num_layers,
        num_heads,
        num_kv_heads,
        hidden_size
    );

    // Build GGUF tensors with raw byte passthrough
    let tensor_names = reader.tensor_names();
    let mut gguf_tensors = Vec::with_capacity(tensor_names.len());

    for name in &tensor_names {
        let entry = reader
            .get_tensor(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{}' missing from index", name),
            })?;
        let raw_bytes = reader
            .get_tensor_data(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{}' data not found", name),
            })?;

        let gguf_name = hf_to_gguf_name(name);

        // Map APR dtype → GGUF dtype (same discriminant values)
        let gguf_dtype = match entry.dtype {
            TensorDType::F32 => GgmlType::F32,
            TensorDType::F16 => GgmlType::F16,
            TensorDType::Q4K => GgmlType::Q4K,
            TensorDType::Q6K => GgmlType::Q6K,
            TensorDType::Q8 => GgmlType::Q8_0,
            _ => GgmlType::F32, // Fallback for BF16, I32, etc.
        };

        // Reverse shape for GGUF: [rows, cols] → [ne0=cols, ne1=rows]
        let gguf_shape = if entry.shape.len() == 2 {
            vec![entry.shape[1] as u64, entry.shape[0] as u64]
        } else {
            entry.shape.iter().map(|&d| d as u64).collect()
        };

        eprintln!(
            "[PMAT-252] '{}': {} bytes (dtype={:?})",
            gguf_name,
            raw_bytes.len(),
            entry.dtype
        );

        gguf_tensors.push(GgufTensor {
            name: gguf_name,
            shape: gguf_shape,
            dtype: gguf_dtype,
            data: raw_bytes.to_vec(),
        });
    }

    // Write to file
    let file = File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;
    let mut writer = BufWriter::new(file);

    export_tensors_to_gguf(&mut writer, &gguf_tensors, validated.as_slice())?;

    let exported_size = fs::metadata(output).map(|m| m.len() as usize).unwrap_or(0);

    Ok(ExportReport {
        original_size,
        exported_size,
        tensor_count: gguf_tensors.len(),
        format: ExportFormat::Gguf,
        quantization: Some(QuantizationType::Q4K),
    })
}

/// Map HuggingFace-style tensor names to GGUF convention (GGUF-EXPORT-001)
///
/// Reverse of `Architecture::qwen2_map_name()` which maps GGUF→HF.
/// This maps HF→GGUF for export.
fn hf_to_gguf_name(name: &str) -> String {
    // Handle layer tensors: model.layers.N.suffix → blk.N.suffix
    if let Some(rest) = name.strip_prefix("model.layers.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];

            let gguf_suffix = match suffix {
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.q_proj.bias" => "attn_q.bias",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.k_proj.bias" => "attn_k.bias",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.v_proj.bias" => "attn_v.bias",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.o_proj.bias" => "attn_output.bias",
                "self_attn.qkv_proj.weight" => "attn_qkv.weight",
                "self_attn.qkv_proj.bias" => "attn_qkv.bias",
                "input_layernorm.weight" => "attn_norm.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                other => other, // Preserve unknown suffixes
            };

            return format!("blk.{layer_num}.{gguf_suffix}");
        }
    }

    // Handle non-layer tensors
    match name {
        "model.embed_tokens.weight" => "token_embd.weight".to_string(),
        "lm_head.weight" => "output.weight".to_string(),
        "model.norm.weight" => "output_norm.weight".to_string(),
        _ => name.to_string(), // Preserve unknown names
    }
}

// ============================================================================
// GH-182: COMPANION FILE HELPERS
// ============================================================================

/// Infer hidden_size from embedding tensor (BUG-EXPORT-001: pick smaller dim)
fn infer_hidden_size(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> (usize, bool) {
    tensors
        .iter()
        .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embd"))
        .map(|(name, (_, shape))| {
            let dim = if shape.len() >= 2 {
                let inferred = shape[0].min(shape[1]);
                eprintln!(
                    "[GH-197] Inferred hidden_size={inferred} from tensor '{name}' \
                     (shape={shape:?}, picked smaller dim)"
                );
                inferred
            } else {
                shape.last().copied().unwrap_or(4096)
            };
            (dim, true)
        })
        .unwrap_or((4096, false))
}

/// Count transformer layers from tensor name patterns
fn infer_num_layers(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> usize {
    let max_layer: Option<usize> = tensors
        .keys()
        .filter_map(|name| {
            if name.contains("layers.") || name.contains("blk.") {
                let parts: Vec<&str> = name.split(&['.', '_'][..]).collect();
                for (i, part) in parts.iter().enumerate() {
                    if (*part == "layers" || *part == "blk") && i + 1 < parts.len() {
                        return parts[i + 1].parse::<usize>().ok();
                    }
                }
            }
            None
        })
        .max();

    if let Some(max) = max_layer {
        let count = max + 1;
        eprintln!("[GH-197] Inferred num_layers={count} from layer indices 0..{max}");
        count
    } else {
        12
    }
}

/// Infer vocab_size from lm_head, output, or embedding tensor (BUG-EXPORT-001: pick larger dim)
fn infer_vocab_size(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> (usize, bool) {
    tensors
        .iter()
        .find(|(name, _)| name.contains("lm_head") || name.contains("output.weight"))
        .or_else(|| {
            tensors
                .iter()
                .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embd"))
        })
        .map(|(name, (_, shape))| {
            let dim = if shape.len() >= 2 {
                let inferred = shape[0].max(shape[1]);
                eprintln!(
                    "[GH-197] Inferred vocab_size={inferred} from tensor '{name}' \
                     (shape={shape:?}, picked larger dim)"
                );
                inferred
            } else {
                shape.first().copied().unwrap_or(32000)
            };
            (dim, true)
        })
        .unwrap_or((32000, false))
}

/// Infer model config.json from tensor shapes (GH-182, GH-193)
fn infer_model_config(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> String {
    let (hidden_size, hidden_inferred) = infer_hidden_size(tensors);
    let num_layers = infer_num_layers(tensors);
    let (vocab_size, vocab_inferred) = infer_vocab_size(tensors);

    // GH-197 FIX: Sanity validation
    if vocab_inferred && hidden_inferred && vocab_size < hidden_size {
        eprintln!(
            "[GH-197] WARNING: vocab_size ({vocab_size}) < hidden_size ({hidden_size}). \
             This is unusual for LLMs - dimensions may be swapped!"
        );
    }

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
    if input_path.extension().and_then(|e| e.to_str()) != Some("apr") {
        return String::new();
    }
    extract_apr_tokenizer_hint(input_path).unwrap_or_default()
}

/// Try to extract tokenizer hint from APR metadata section.
fn extract_apr_tokenizer_hint(input_path: &Path) -> Option<String> {
    let data = fs::read(input_path).ok()?;
    if data.len() <= 44 {
        return None;
    }
    let metadata_start = 44;
    let metadata_end = data[metadata_start..]
        .windows(4)
        .position(|w| w == b"}\n\n\n" || w == b"}\r\n\r")
        .map(|p| metadata_start + p + 1)?;
    let metadata_str = std::str::from_utf8(&data[metadata_start..metadata_end]).ok()?;
    if metadata_str.contains("\"tokenizer\"") || metadata_str.contains("\"vocabulary\"") {
        Some(r#"{"version": "1.0", "model": {"type": "BPE"}}"#.to_string())
    } else {
        None
    }
}

/// ROSETTA-003: Read APR v2 metadata from file.
///
/// Returns `None` for non-APR files or on any read/parse failure.
fn read_apr_metadata(apr_path: &Path) -> Option<crate::format::v2::AprV2Metadata> {
    if apr_path.extension().and_then(|e| e.to_str()) != Some("apr") {
        return None;
    }
    let data = fs::read(apr_path).ok()?;
    let reader = crate::format::v2::AprV2Reader::from_bytes(&data).ok()?;
    Some(reader.metadata().clone())
}

/// ROSETTA-003: Unfuse legacy QKV tensors for lossless round-trip export.
///
/// Old APR files (pre-ROSETTA-003) stored fused `qkv_proj.weight` tensors.
/// This function splits them back into separate `q_proj`, `k_proj`, `v_proj`
/// for correct GGUF/SafeTensors export. New APR files with separate Q/K/V
/// pass through unchanged.
fn unfuse_qkv_tensors(
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    apr_path: &Path,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let has_fused = tensors.keys().any(|k| k.contains("qkv_proj."));
    if !has_fused {
        return tensors;
    }

    let metadata = read_apr_metadata(apr_path);
    let (hidden_size, num_heads, num_kv_heads) = match &metadata {
        Some(m) => {
            let hs = m.hidden_size.unwrap_or(0);
            let nh = m.num_heads.unwrap_or(0);
            let nkv = m.num_kv_heads.unwrap_or(nh);
            (hs, nh, nkv)
        }
        None => return tensors,
    };

    if hidden_size == 0 || num_heads == 0 {
        return tensors;
    }

    let head_dim = hidden_size / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let mut result = BTreeMap::new();

    for (name, (data, shape)) in tensors {
        if name.contains("qkv_proj.weight") {
            // Split [Q;K;V] weight back into separate tensors.
            // Shape was [qkv_dim, hidden_dim] where qkv_dim = hidden_size + 2*kv_dim.
            let hidden_dim = if shape.len() >= 2 {
                shape[1]
            } else {
                hidden_size
            };
            let q_elements = hidden_size * hidden_dim;
            let kv_elements = kv_dim * hidden_dim;

            if data.len() >= q_elements + 2 * kv_elements {
                let prefix = name.strip_suffix("qkv_proj.weight").unwrap_or(&name);

                result.insert(
                    format!("{prefix}q_proj.weight"),
                    (data[..q_elements].to_vec(), vec![hidden_size, hidden_dim]),
                );
                result.insert(
                    format!("{prefix}k_proj.weight"),
                    (
                        data[q_elements..q_elements + kv_elements].to_vec(),
                        vec![kv_dim, hidden_dim],
                    ),
                );
                result.insert(
                    format!("{prefix}v_proj.weight"),
                    (
                        data[q_elements + kv_elements..q_elements + 2 * kv_elements].to_vec(),
                        vec![kv_dim, hidden_dim],
                    ),
                );
            } else {
                result.insert(name, (data, shape));
            }
        } else if name.contains("qkv_proj.bias") {
            // Split [Q_bias; K_bias; V_bias] back into separate biases.
            let qkv_dim = hidden_size + 2 * kv_dim;
            if data.len() == qkv_dim {
                let prefix = name.strip_suffix("qkv_proj.bias").unwrap_or(&name);

                result.insert(
                    format!("{prefix}q_proj.bias"),
                    (data[..hidden_size].to_vec(), vec![hidden_size]),
                );
                result.insert(
                    format!("{prefix}k_proj.bias"),
                    (
                        data[hidden_size..hidden_size + kv_dim].to_vec(),
                        vec![kv_dim],
                    ),
                );
                result.insert(
                    format!("{prefix}v_proj.bias"),
                    (data[hidden_size + kv_dim..].to_vec(), vec![kv_dim]),
                );
            } else {
                result.insert(name, (data, shape));
            }
        } else {
            result.insert(name, (data, shape));
        }
    }

    result
}

/// ROSETTA-003: Remove synthesized `lm_head.weight` for SafeTensors export.
///
/// When the APR metadata has `tied_embeddings: true`, the lm_head was copied
/// from embed_tokens during import. For SafeTensors round-trip fidelity,
/// remove it so the exported file matches the original HuggingFace convention.
fn remove_tied_lm_head(
    mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    apr_path: &Path,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let metadata = read_apr_metadata(apr_path);
    let is_tied = metadata
        .as_ref()
        .and_then(|m| m.custom.get("tied_embeddings"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if is_tied {
        tensors.remove("lm_head.weight");
    }

    tensors
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

/// Detect predominant quantization type from an APR file (PMAT-252).
///
/// Reads the tensor index and checks the dtype of 2D weight tensors.
/// Returns the `QuantizationType` if the majority of weights use a
/// quantized format (Q4K, Q6K), or `None` for F32/F16 files.
pub(crate) fn detect_apr_quantization(apr_path: &Path) -> Option<QuantizationType> {
    use crate::format::v2::{AprV2Reader, TensorDType};

    let data = fs::read(apr_path).ok()?;
    let reader = AprV2Reader::from_bytes(&data).ok()?;

    // Count dtypes across 2D weight tensors (skip 1D biases/norms)
    let mut q4k_count = 0usize;
    let mut q6k_count = 0usize;
    let mut other_count = 0usize;

    for name in reader.tensor_names() {
        if let Some(entry) = reader.get_tensor(name) {
            if entry.shape.len() >= 2 {
                match entry.dtype {
                    TensorDType::Q4K => q4k_count += 1,
                    TensorDType::Q6K => q6k_count += 1,
                    _ => other_count += 1,
                }
            }
        }
    }

    let total = q4k_count + q6k_count + other_count;
    if total == 0 {
        return None;
    }

    // If majority of 2D tensors are Q4K, default to Q4K export
    if q4k_count > q6k_count && q4k_count > other_count {
        return Some(QuantizationType::Q4K);
    }

    // Q6K not yet in QuantizationType — treat as no auto-detect
    None
}

/// Detect GGUF model architecture for tensor name mapping (GH-200).
/// GH-236: Added GPT-2 recognition — was falling through to Qwen2 default,
/// causing metadata key mismatch (writes "qwen2.embedding_length" but reader
/// looks for "gpt2.embedding_length") → hidden_dim=0 on reimport.
fn detect_gguf_architecture(path: &Path) -> Architecture {
    GgufReader::from_file(path)
        .ok()
        .and_then(|r| r.architecture())
        .map(|a| Architecture::from_model_type(&a).unwrap_or(Architecture::Qwen2))
        .unwrap_or(Architecture::Qwen2)
}

/// Infer vocab_size and hidden_dim from tensor shapes.
///
/// Find the first 2D tensor matching any of the given name patterns.
fn find_2d_tensor_shape<'a>(
    tensors: &'a BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    patterns: &[&str],
) -> Option<&'a [usize]> {
    tensors.iter().find_map(|(name, (_, shape))| {
        if shape.len() == 2 && patterns.iter().any(|p| name.contains(p)) {
            Some(shape.as_slice())
        } else {
            None
        }
    })
}

/// Used for contract validation during export.
fn infer_vocab_hidden(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> (usize, usize) {
    // Try embedding, then lm_head for [vocab_size, hidden_dim]
    if let Some(shape) = find_2d_tensor_shape(tensors, &["embed_tokens", "token_embd"]) {
        return (shape[0], shape[1]);
    }
    if let Some(shape) = find_2d_tensor_shape(tensors, &["lm_head", "output.weight"]) {
        return (shape[0], shape[1]);
    }
    // Fallback: get hidden_dim from q_proj
    let hidden = find_2d_tensor_shape(tensors, &["q_proj"]).map_or(0, |s| s[1]);
    (0, hidden)
}

#[cfg(test)]
#[path = "export_tests.rs"]
mod tests;
