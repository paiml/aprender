//! APR Converter Module - Import Pipeline
//!
//! Implements Section 13 of APR-SPEC.md: Import/Convert Pipeline
//!
//! Supports:
//! - `HuggingFace` Hub downloads (<hf://org/repo>)
//! - `SafeTensors` conversion
//! - Inline validation during conversion
//! - Quantization and compression

use crate::error::{AprenderError, Result};

// Toyota Way: ONE source of truth for quantization (trueno-quant crate)
// PMAT-230: trueno-quant created 2026-02-03, resolves cyclic dependency
// Re-export constants and key functions for external use
pub use trueno_quant::F16_MIN_NORMAL;

// Internal imports - use trueno_quant as the canonical implementation
// Re-exported as pub(crate) so tests can access them
// Note: Only import functions actually used in this module
pub(crate) use trueno_quant::{dequantize_q4_k_to_f32, quantize_q4_k, quantize_q4_k_matrix};

use crate::format::gguf::{
    load_gguf_raw, load_gguf_with_tokenizer, GgufModelConfig, GgufRawTensor, GgufReader,
    GgufTokenizer,
};
use crate::format::v2::{AprV2Metadata, AprV2Writer, QuantizationMetadata};
use crate::format::Compression;
use crate::serialization::safetensors::{save_safetensors, SafeTensorsMetadata, TensorMetadata};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

// PMAT-197: Re-export types from converter_types module for backward compatibility
#[cfg(feature = "hf-hub-integration")]
pub use crate::format::converter_types::parse_import_error;
pub use crate::format::converter_types::{
    detect_sharded_model, Architecture, DequantizedTensors, ImportError, ImportOptions,
    NativeF32Tensors, QuantizationType, ShardedIndex, Source, TensorExpectation, TensorProvenance,
    ValidationConfig,
};

// PMAT-197: Import functions moved to import.rs that are used elsewhere
pub(crate) use import::infer_model_config_from_tensors;
use import::load_safetensors_tensors;
pub(crate) use import::map_tensor_names;
// GH-268: Sanitize non-standard JSON from HuggingFace config files
pub use import::sanitize_hf_json;

// For tests: re-export helper types and functions
#[cfg(test)]
pub(crate) use crate::format::validation::{AprValidator, TensorStats};
#[cfg(test)]
pub(crate) use import::{
    compute_std, compute_tensor_stats, parse_tokenizer_json, validate_single_tensor,
    TensorAccumulator,
};
#[cfg(test)]
pub(crate) use merge::calculate_merge_weights;
#[cfg(test)]
pub(crate) use std::path::PathBuf;

// HF Hub integration is used via hf_hub::api::sync::ApiBuilder in download_from_hf()

// ============================================================================
// Converter
// ============================================================================

/// APR Converter with builder pattern
#[derive(Debug)]
pub struct AprConverter {
    source: Option<Source>,
    architecture: Architecture,
    validation: ValidationConfig,
    quantize: Option<QuantizationType>,
    compress: Option<Compression>,
}

impl AprConverter {
    /// Create a new converter
    #[must_use]
    pub fn new() -> Self {
        Self {
            source: None,
            architecture: Architecture::Auto,
            validation: ValidationConfig::Strict,
            quantize: None,
            compress: None,
        }
    }

    /// Set the source
    pub fn source(mut self, source: &str) -> Result<Self> {
        self.source = Some(Source::parse(source)?);
        Ok(self)
    }

    /// Set the architecture
    #[must_use]
    pub fn architecture(mut self, arch: Architecture) -> Self {
        self.architecture = arch;
        self
    }

    /// Set validation config
    #[must_use]
    pub fn validate(mut self, config: ValidationConfig) -> Self {
        self.validation = config;
        self
    }

    /// Set quantization
    #[must_use]
    pub fn quantize(mut self, quant: QuantizationType) -> Self {
        self.quantize = Some(quant);
        self
    }

    /// Set compression
    #[must_use]
    pub fn compress(mut self, comp: Compression) -> Self {
        self.compress = Some(comp);
        self
    }

    /// Run the conversion
    pub fn convert(self) -> Result<Vec<u8>> {
        let source = self.source.ok_or_else(|| AprenderError::FormatError {
            message: "No source specified".to_string(),
        })?;

        // NOTE: Full conversion pipeline is tracked in GH-80 (metaheuristics milestone)
        // Current limitation: Returns error for unsupported sources
        Err(AprenderError::FormatError {
            message: format!(
                "Conversion from {:?} not yet implemented - see GH-80",
                source
            ),
        })
    }
}

impl Default for AprConverter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Model Conversion (apr convert)
// ============================================================================

/// Options for model conversion
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Quantization method (int8, int4, fp16)
    pub quantize: Option<QuantizationType>,
    /// Compression method
    pub compress: Option<Compression>,
    /// Validate after conversion
    pub validate: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            quantize: None,
            compress: None,
            validate: true,
        }
    }
}

/// GH-181: Attempt Q4K raw byte pass-through for GGUF sources already quantized as Q4K.
///
/// When the source GGUF contains Q4_K/Q5_K/Q6_K tensors and the target is Q4K,
/// we skip the lossy dequant->requant round-trip and copy raw bytes directly.
/// Returns `Ok(Some(report))` on success, `Ok(None)` if pass-through is not applicable.
fn try_gguf_q4k_passthrough(
    input_path: &Path,
    output_path: &Path,
    options: &ConvertOptions,
) -> Result<Option<ConvertReport>> {
    let raw_result = match load_gguf_raw(input_path) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };

    // Check if source contains Q4_K tensors (dtype 12, 13, 14 = Q4_K, Q5_K, Q6_K)
    let has_q4k = raw_result
        .tensors
        .values()
        .any(|t| t.dtype == 12 || t.dtype == 13 || t.dtype == 14);

    if !has_q4k {
        return Ok(None);
    }

    eprintln!("[GH-181] Detected Q4K source, using raw byte pass-through");

    // Map tensor names to APR canonical format (GH-190 fix: bare names)
    let mapped_tensors: BTreeMap<String, GgufRawTensor> = raw_result
        .tensors
        .into_iter()
        .map(|(name, tensor)| {
            let mapped_name = Architecture::Qwen2.map_name(&name);
            (mapped_name, tensor)
        })
        .collect();

    let original_size: usize = mapped_tensors.values().map(|t| t.data.len()).sum();
    let original_count = mapped_tensors.len();

    // Write APR file with raw quantized tensors (preserves block alignment)
    let import_opts = ImportOptions {
        architecture: Architecture::Qwen2,
        allow_no_config: true,
        ..Default::default()
    };
    write_apr_file_raw(
        &mapped_tensors,
        output_path,
        &import_opts,
        Some(&raw_result.tokenizer),
        Some(&raw_result.model_config),
    )?;

    Ok(Some(ConvertReport::build(
        original_size,
        output_path,
        original_count,
        options.quantize,
        options.compress,
    )))
}

/// F-REGR-231 / PMAT-113: Extract GGUF model config and tokenizer.
///
/// For GGUF inputs, loads the full config to preserve rope_type (Qwen2.5 requires
/// rope_type=2 NEOX style) and extracts the tokenizer for APR embedding.
/// Returns `(None, None)` if loading fails so callers can fall back to inference.
fn extract_gguf_config(input_path: &Path) -> (Option<GgufModelConfig>, Option<GgufTokenizer>) {
    match load_gguf_with_tokenizer(input_path) {
        Ok(result) => {
            eprintln!(
                "[PMAT-113] Extracted tokenizer with {} vocabulary tokens",
                result.tokenizer.vocabulary.len()
            );
            (Some(result.model_config), Some(result.tokenizer))
        }
        Err(_) => (None, None),
    }
}

/// PMAT-205 / GH-190: Map GGUF tensor names to APR canonical format.
///
/// GGUF uses names like "blk.0.attn_q.weight" but APR loaders expect
/// bare names like "layers.0.self_attn.q_proj.weight" (no "model." prefix).
fn apply_gguf_name_mapping(
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    eprintln!(
        "[PMAT-205] Mapping {} GGUF tensor names to APR canonical format...",
        tensors.len()
    );
    let mapped = map_tensor_names(&tensors, Architecture::Qwen2);
    for (i, name) in mapped.keys().take(5).enumerate() {
        eprintln!("[PMAT-205]   {}: {}", i, name);
    }
    mapped
}

// GH-237: apply_quantization removed — convert path now dispatches through
// add_tensor_with_quantization() for real packing. Export path uses
// quantize_tensors() directly in apply_export_quantization (export.rs).

/// Save tensors to output, using GGUF config if available for metadata fidelity.
///
/// F-REGR-231: Preserves rope_type and other GGUF metadata.
/// PMAT-113: Embeds tokenizer for standalone APR inference.
fn save_output(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output_path: &Path,
    compression: Option<Compression>,
    gguf_config: Option<&GgufModelConfig>,
    gguf_tokenizer: Option<&GgufTokenizer>,
    quantize: Option<QuantizationType>,
) -> Result<()> {
    if let Some(config) = gguf_config {
        save_model_tensors_with_gguf_config_and_tokenizer(
            tensors,
            output_path,
            compression,
            config,
            gguf_tokenizer,
            quantize,
        )
    } else {
        save_model_tensors(tensors, output_path, compression, quantize)
    }
}

/// Convert a model with quantization and/or compression
///
/// # Arguments
/// * `input` - Input model path (.safetensors or .apr)
/// * `output` - Output model path
/// * `options` - Conversion options
///
/// # Returns
/// * `ConvertReport` with size reduction stats
///
/// # Example
/// ```rust,ignore
/// use aprender::format::{apr_convert, ConvertOptions, QuantizationType};
///
/// let options = ConvertOptions {
///     quantize: Some(QuantizationType::Int8),
///     ..Default::default()
/// };
/// let report = apr_convert("model.safetensors", "model-int8.apr", options)?;
/// println!("Reduced from {} to {} bytes", report.original_size, report.converted_size);
/// ```
pub fn apr_convert<P: AsRef<Path>>(
    input: P,
    output: P,
    options: ConvertOptions,
) -> Result<ConvertReport> {
    let input_path = input.as_ref();
    let output_path = output.as_ref();
    // PMAT-271: Use magic bytes first, extension fallback for extensionless HF cache blobs
    let is_gguf = crate::format::rosetta::FormatType::from_magic(input_path)
        .map(|f| matches!(f, crate::format::rosetta::FormatType::Gguf))
        .unwrap_or_else(|_| input_path.extension().and_then(|e| e.to_str()) == Some("gguf"));

    // GH-181: Q4K GGUF pass-through (skip dequant->requant for already-quantized sources)
    if is_gguf && options.quantize == Some(QuantizationType::Q4K) {
        if let Some(report) = try_gguf_q4k_passthrough(input_path, output_path, &options)? {
            return Ok(report);
        }
    }

    // F-REGR-231 / PMAT-113: Extract GGUF config and tokenizer for metadata fidelity
    let (gguf_config, gguf_tokenizer) = if is_gguf {
        extract_gguf_config(input_path)
    } else {
        (None, None)
    };

    // Step 1: Load tensors
    let tensors = load_model_tensors(input_path)?;
    // PMAT-274 FIX: Use input FILE size (not raw F32 tensor bytes) for fair comparison.
    // calculate_tensor_size() counts decompressed F32 bytes (no headers/metadata), but
    // ConvertReport::build() reads output FILE size (with headers/metadata/alignment).
    // This apples-to-oranges comparison made small-model quantization report ratio ≈ 1.0.
    let original_size = fs::metadata(input_path)
        .map(|m| m.len() as usize)
        .unwrap_or_else(|_| calculate_tensor_size(&tensors));
    let original_count = tensors.len();

    // Step 1b: PMAT-205 / GH-190: Map GGUF tensor names to APR canonical format
    let tensors = if is_gguf {
        apply_gguf_name_mapping(tensors)
    } else {
        tensors
    };

    // Step 2: Handle Q4K specially - store raw Q4K bytes in APR format
    if options.quantize == Some(QuantizationType::Q4K) {
        save_model_tensors_q4k(&tensors, output_path)?;
        return Ok(ConvertReport::build(
            original_size,
            output_path,
            original_count,
            options.quantize,
            options.compress,
        ));
    }

    // Step 2b: Quantization is applied during save via writer dispatch (GH-237)
    // Previously, apply_quantization() did a round-trip simulation (quantize→dequantize
    // to f32) and save functions always called add_f32_tensor(). Now quantization type
    // is passed through to save functions which call add_q8_tensor/add_q4_tensor/
    // add_f16_tensor directly, producing real packed bytes.

    // Step 3: Save output with GGUF config if available
    save_output(
        &tensors,
        output_path,
        options.compress,
        gguf_config.as_ref(),
        gguf_tokenizer.as_ref(),
        options.quantize,
    )?;

    // Step 4: Build report
    Ok(ConvertReport::build(
        original_size,
        output_path,
        original_count,
        options.quantize,
        options.compress,
    ))
}

/// Report from model conversion
#[derive(Debug, Clone)]
pub struct ConvertReport {
    /// Original model size in bytes
    pub original_size: usize,
    /// Converted model size in bytes
    pub converted_size: usize,
    /// Number of tensors
    pub tensor_count: usize,
    /// Quantization applied
    pub quantization: Option<QuantizationType>,
    /// Compression applied
    pub compression: Option<Compression>,
    /// Size reduction ratio (original/converted)
    pub reduction_ratio: f64,
}

include!("mod_part_02.rs");
include!("mod_part_03.rs");
include!("mod_part_04.rs");
