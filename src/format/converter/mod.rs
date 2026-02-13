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
use crate::serialization::safetensors::save_safetensors;
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

/// Apply optional quantization (Fp16, Int8, Int4) to loaded tensors.
///
/// DOUBLE-QUANT-001: Wraps tensors in `NativeF32Tensors` (safe because
/// `apr_convert` always loads from file; the raw Q4K passthrough handles
/// quantized GGUF sources separately).
fn apply_quantization(
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    quant_type: &QuantizationType,
) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let native = NativeF32Tensors::new(tensors);
    Ok(quantize_tensors(&native, quant_type)?.into_inner())
}

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
) -> Result<()> {
    if let Some(config) = gguf_config {
        save_model_tensors_with_gguf_config_and_tokenizer(
            tensors,
            output_path,
            compression,
            config,
            gguf_tokenizer,
        )
    } else {
        save_model_tensors(tensors, output_path, compression)
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
    let extension = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let is_gguf = extension == "gguf";

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
    let original_size = calculate_tensor_size(&tensors);
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

    // Step 2b: Apply other quantization types (Fp16, Int8, Int4)
    let tensors = match options.quantize {
        Some(ref quant_type) => apply_quantization(tensors, quant_type)?,
        None => tensors,
    };

    // Step 3: Save output with GGUF config if available
    save_output(
        &tensors,
        output_path,
        options.compress,
        gguf_config.as_ref(),
        gguf_tokenizer.as_ref(),
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

impl ConvertReport {
    /// Build a report, computing the reduction ratio from original/converted sizes
    fn build(
        original_size: usize,
        output_path: &Path,
        tensor_count: usize,
        quantization: Option<QuantizationType>,
        compression: Option<Compression>,
    ) -> Self {
        let converted_size = fs::metadata(output_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let reduction_ratio = if converted_size > 0 {
            original_size as f64 / converted_size as f64
        } else {
            0.0
        };
        Self {
            original_size,
            converted_size,
            tensor_count,
            quantization,
            compression,
            reduction_ratio,
        }
    }

    /// Format reduction as percentage string
    #[must_use]
    pub fn reduction_percent(&self) -> String {
        if self.original_size > 0 && self.converted_size > 0 {
            let reduction = 100.0 * (1.0 - self.converted_size as f64 / self.original_size as f64);
            format!("{:.1}%", reduction)
        } else {
            "N/A".to_string()
        }
    }
}

/// Load tensors from model file
///
/// Supports: SafeTensors, APR, GGUF (GH-164 fix)
/// GGUF tensors are dequantized to F32 during loading.
pub(crate) fn load_model_tensors(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension {
        "safetensors" => load_safetensors_tensors(path),
        "apr" => load_apr_tensors_f32(path),
        "gguf" => load_gguf_tensors_f32(path),
        other => Err(AprenderError::FormatError {
            message: format!("Unsupported format for conversion: .{other}"),
        }),
    }
}

/// Load tensors with provenance tracking (DOUBLE-QUANT-001).
///
/// Returns `TensorProvenance::Native` for SafeTensors and unquantized APR sources,
/// `TensorProvenance::Dequantized` for GGUF and quantized APR sources.
/// This enables compile-time prevention of double quantization.
pub(crate) fn load_model_tensors_provenance(path: &Path) -> Result<TensorProvenance> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension {
        "safetensors" => {
            let tensors = load_safetensors_tensors(path)?;
            Ok(TensorProvenance::Native(NativeF32Tensors::new(tensors)))
        }
        "apr" => {
            // Check if source has quantized tensors
            if let Some(quant) = export::detect_apr_quantization(path) {
                let tensors = load_apr_tensors_f32(path)?;
                Ok(TensorProvenance::Dequantized(DequantizedTensors::new(
                    tensors, quant,
                )))
            } else {
                let tensors = load_apr_tensors_f32(path)?;
                Ok(TensorProvenance::Native(NativeF32Tensors::new(tensors)))
            }
        }
        "gguf" => {
            // GGUF models are always quantized (Q4K, Q6K, etc.)
            let tensors = load_gguf_tensors_f32(path)?;
            Ok(TensorProvenance::Dequantized(DequantizedTensors::new(
                tensors,
                QuantizationType::Q4K,
            )))
        }
        other => Err(AprenderError::FormatError {
            message: format!("Unsupported format for conversion: .{other}"),
        }),
    }
}

/// Load GGUF tensors and dequantize to F32 (GH-164)
///
/// Uses GgufReader::get_all_tensors_f32() which handles:
/// - Q4_K, Q5_K, Q6_K dequantization
/// - Q4_0, Q5_0, Q8_0 dequantization
/// - F16, F32 direct loading
///
/// PMAT-187: Validates all tensors after loading to catch corruption early.
fn load_gguf_tensors_f32(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let reader = GgufReader::from_file(path)?;
    let tensors = reader.get_all_tensors_f32()?;

    // PMAT-187: Validate all tensors after loading (Jidoka - stop the line)
    for (name, (data, _shape)) in &tensors {
        validate_tensor_values(name, data)?;
    }

    // GH-208: MANDATORY CONTRACT ENFORCEMENT
    // Shape transformation goes through enforce_import_contract().
    // See: contracts/tensor-layout-v1.yaml, Five Whys in layout_contract.rs
    //
    // NOTE: This F32 path is for dequantized tensors. The raw quantized path
    // (import.rs:apr_import_gguf_raw) also uses enforce_import_contract.
    use crate::format::layout_contract::enforce_import_contract;

    let tensors = tensors
        .into_iter()
        .map(|(name, (data, shape))| {
            // Use CONTRACT for shape transformation (vocab_size/hidden_dim=0 means unknown)
            let (apr_shape, needs_data_transpose) = enforce_import_contract(&name, &shape, 0, 0);

            // GH-208: Data transpose should NEVER be needed
            assert!(
                !needs_data_transpose,
                "CONTRACT BUG: enforce_import_contract returned needs_data_transpose=true for '{}'. \
                 GGUF→APR NEVER needs data transpose.",
                name
            );

            (name, (data, apr_shape))
        })
        .collect();

    Ok(tensors)
}

/// Load APR tensors and dequantize to F32 (PMAT-174, GH-196 fix)
///
/// Uses `AprV2Reader` for correct parsing of v2 format files produced by
/// `AprV2Writer`. Falls back to manual v1 parsing for legacy files.
///
/// Handles all APR dtypes: F32, F16, BF16, Q4_K, Q6_K, Q8_0
fn load_apr_tensors_f32(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    use crate::format::v2::AprV2Reader;

    let data = fs::read(path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to read APR file: {e}"),
    })?;

    // Use AprV2Reader for correct v2 format parsing
    let reader = AprV2Reader::from_bytes(&data).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to parse APR file: {e:?}"),
    })?;

    let mut tensors = BTreeMap::new();
    for name in reader.tensor_names() {
        let entry = reader
            .get_tensor(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{}' missing from index", name),
            })?;
        let shape = entry.shape.clone();

        // get_tensor_as_f32 handles all dtypes (F32, F16, Q8, Q4, etc.)
        let f32_data =
            reader
                .get_tensor_as_f32(name)
                .ok_or_else(|| AprenderError::FormatError {
                    message: format!("Failed to dequantize tensor '{}'", name),
                })?;

        // PMAT-187: Validate tensor values after dequantization (Jidoka - stop the line)
        validate_tensor_values(name, &f32_data)?;

        tensors.insert(name.to_string(), (f32_data, shape));
    }

    Ok(tensors)
}

/// PMAT-187: Validate tensor values for NaN/Inf/explosive corruption
///
/// Toyota Way Jidoka: Stop the line on quality defects, don't pass defects downstream.
/// This catches corruption introduced during dequantization before it propagates.
///
/// # Errors
///
/// Returns error if tensor contains NaN, Inf, or explosive values (mean > 100)
pub(crate) fn validate_tensor_values(name: &str, data: &[f32]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }

    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut sum: f64 = 0.0;

    for &value in data {
        if value.is_nan() {
            nan_count += 1;
        } else if value.is_infinite() {
            inf_count += 1;
        } else {
            sum += value as f64;
        }
    }

    // Fail fast on NaN
    if nan_count > 0 {
        return Err(AprenderError::FormatError {
            message: format!(
                "PMAT-187: Tensor '{}' contains {} NaN values (data corruption detected). \
                 Toyota Way: Stop the line - do not pass defects downstream.",
                name, nan_count
            ),
        });
    }

    // Fail fast on Inf
    if inf_count > 0 {
        return Err(AprenderError::FormatError {
            message: format!(
                "PMAT-187: Tensor '{}' contains {} Inf values (numerical overflow detected). \
                 Toyota Way: Stop the line - do not pass defects downstream.",
                name, inf_count
            ),
        });
    }

    // Check for explosive mean (indicates corrupted scale factors)
    let valid_count = data.len() - nan_count - inf_count;
    if valid_count > 0 {
        let mean = sum / valid_count as f64;
        if mean.abs() > 100.0 {
            return Err(AprenderError::FormatError {
                message: format!(
                    "PMAT-187: Tensor '{}' has explosive mean={:.2e} (expected [-100, 100]). \
                     This indicates corrupted quantization scale factors. \
                     Toyota Way: Stop the line - do not pass defects downstream.",
                    name, mean
                ),
            });
        }
    }

    Ok(())
}

/// Dequantize F16 to F32 (PMAT-174)
#[allow(dead_code)] // Retained for GGUF conversion paths
fn dequantize_f16_to_f32(bytes: &[u8], _num_elements: usize) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f16_to_f32(bits)
        })
        .collect()
}

/// Dequantize BF16 to F32 (PMAT-174)
#[allow(dead_code)] // Retained for GGUF conversion paths
fn dequantize_bf16_to_f32(bytes: &[u8], _num_elements: usize) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            // BF16 is F32 with lower 16 mantissa bits zeroed
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

/// Dequantize Q8_0 to F32 (PMAT-174)
/// Q8_0: 32-element blocks with f16 scale + 32 int8 quants
#[allow(dead_code)] // Retained for GGUF conversion paths
fn dequantize_q8_0_to_f32(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 32; // f16 scale + 32 int8s
                                       // MSRV-compatible div_ceil: (n + d - 1) / d
    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = Vec::with_capacity(num_elements);

    for i in 0..num_blocks {
        let block_start = i * BLOCK_BYTES;
        if block_start + BLOCK_BYTES > bytes.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([bytes[block_start], bytes[block_start + 1]]);
        // GH-186 FIX: Clamp NaN/Inf/subnormal to prevent propagation
        let scale_raw = f16_to_f32(scale_bits);
        // Uses shared F16_MIN_NORMAL from crate::format::f16_safety (P2 fix)
        let scale =
            if scale_raw.is_nan() || scale_raw.is_infinite() || scale_raw.abs() < F16_MIN_NORMAL {
                0.0
            } else {
                scale_raw
            };

        for j in 0..BLOCK_SIZE {
            if result.len() >= num_elements {
                break;
            }
            let q = bytes[block_start + 2 + j] as i8;
            result.push(q as f32 * scale);
        }
    }

    result
}

/// Calculate total tensor size in bytes (f32)
pub(crate) fn calculate_tensor_size(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> usize {
    tensors.values().map(|(data, _)| data.len() * 4).sum()
}

/// Apply quantization to tensors.
///
/// DOUBLE-QUANT-001: Only accepts `NativeF32Tensors` (natively F32 sources).
/// Passing `DequantizedTensors` is a compile error, preventing the lossy
/// Q4K→F32→Q4K double quantization that destroyed weight fidelity (PMAT-252).
///
/// BUG-EXPORT-004 FIX: Embedding tensors are SKIPPED from quantization.
/// Embeddings are lookup tables with small values that would round to zero
/// under global-scale Int4/Int8 quantization (78% of values become zero).
/// They are kept in F32 and exported as F32 in GGUF.
pub(crate) fn quantize_tensors(
    tensors: &NativeF32Tensors,
    quant_type: &QuantizationType,
) -> Result<NativeF32Tensors> {
    let mut result = BTreeMap::new();

    for (name, (data, shape)) in tensors.as_ref() {
        // BUG-EXPORT-004: Skip quantization for embedding tensors
        // Embeddings have small values (-0.3 to 0.2) that would mostly round to 0
        // under global-scale Int4 quantization (scale=0.042, values<0.021 → 0)
        let is_embedding = name.contains("embed_tokens")
            || name.contains("token_embd")
            || name.contains("wte")  // GPT-style
            || name.contains("word_embeddings"); // BERT-style

        // GH-234: lm_head.weight has same small-value distribution as embeddings
        // (especially when weight-tied). Quantization causes 4:1 packing / all-zeros.
        let is_lm_head = name.contains("lm_head") || name == "output.weight";

        let quantized_data = if is_embedding || is_lm_head {
            // Keep embeddings and lm_head in original F32 - no quantization
            data.clone()
        } else {
            match quant_type {
                QuantizationType::Fp16 => quantize_fp16(data),
                QuantizationType::Int8 => quantize_int8(data),
                QuantizationType::Int4 => quantize_int4(data),
                QuantizationType::Q4K => {
                    // Q4K: quantize to packed bytes then dequantize back to f32
                    // This preserves the shape but shows quantization error
                    let q4k_bytes = quantize_q4_k(data);
                    dequantize_q4_k_to_f32(&q4k_bytes, data.len())
                }
            }
        };
        result.insert(name.clone(), (quantized_data, shape.clone()));
    }

    Ok(NativeF32Tensors::new(result))
}

// NOTE: dequantize_q4_k_to_f32 moved to trueno-quant crate (Toyota Way consolidation)

/// Quantize to fp16 - TRUE PACKING (2 bytes per value)
/// Returns dequantized f32 for now (proper f16 storage requires format change)
fn quantize_fp16(data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|&v| {
            // Convert f32 to f16 bits then back - TRUE precision reduction
            let f16_bits = f32_to_f16(v);
            f16_to_f32(f16_bits)
        })
        .collect()
}

/// Convert f32 to f16 (IEEE 754 half-precision)
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0 {
        // Zero or denormal
        sign
    } else if exp == 0xFF {
        // Inf or NaN
        sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }
    } else {
        // Normal number
        let new_exp = exp as i32 - 127 + 15;
        if new_exp >= 31 {
            // Overflow to infinity
            sign | 0x7C00
        } else if new_exp <= 0 {
            // Subnormal: f16 can represent down to 2^(-24) ≈ 5.96e-8
            // Add implicit 1 bit to mantissa (bit 23)
            let full_mantissa = mantissa | 0x800000;
            // Calculate shift: 14 bits base when new_exp=0, more for negative
            let shift = 14 - new_exp;
            if shift <= 24 {
                // Round to nearest: add 0.5 at the bit position being truncated
                let round_bit = 1u32 << (shift - 1);
                let rounded = full_mantissa.saturating_add(round_bit);
                let subnormal = (rounded >> shift) as u16;
                sign | (subnormal & 0x3FF)
            } else {
                // Too small for f16 subnormals, underflow to zero
                sign
            }
        } else {
            let new_mantissa = (mantissa >> 13) as u16;
            sign | ((new_exp as u16) << 10) | new_mantissa
        }
    }
}

/// Convert f16 to f32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            f32::from_bits(sign)
        } else {
            // Denormal
            let mut m = mantissa;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let new_exp = ((127 - 15 + 1 + e) as u32) << 23;
            let new_mantissa = (m & 0x3FF) << 13;
            f32::from_bits(sign | new_exp | new_mantissa)
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits(sign | 0x7F800000 | (mantissa << 13))
    } else {
        // exp is 1-30, bias conversion: f16 bias=15, f32 bias=127
        let new_exp = (exp as u32 + 127 - 15) << 23;
        let new_mantissa = mantissa << 13;
        f32::from_bits(sign | new_exp | new_mantissa)
    }
}

/// Quantize to int8 (symmetric quantization)
fn quantize_int8(data: &[f32]) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }

    // Find scale factor (max absolute value)
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    if max_abs == 0.0 {
        return vec![0.0; data.len()];
    }

    let scale = max_abs / 127.0;

    // Quantize and dequantize
    data.iter()
        .map(|&v| {
            let quantized = (v / scale).round().clamp(-127.0, 127.0) as i8;
            f32::from(quantized) * scale
        })
        .collect()
}

/// Quantize to int4 (symmetric quantization)
fn quantize_int4(data: &[f32]) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }

    // Find scale factor
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    if max_abs == 0.0 {
        return vec![0.0; data.len()];
    }

    let scale = max_abs / 7.0; // 4-bit signed range: -8 to 7

    // Quantize and dequantize
    data.iter()
        .map(|&v| {
            let quantized = (v / scale).round().clamp(-8.0, 7.0) as i8;
            f32::from(quantized) * scale
        })
        .collect()
}

// NOTE: quantize_q4_k moved to trueno-quant crate (Toyota Way consolidation)

// NOTE: quantize_q6_k moved to trueno-quant crate (Toyota Way consolidation)

// NOTE: quantize_q6_k_matrix moved to trueno-quant crate (Toyota Way consolidation)

// NOTE: quantize_q5_k moved to trueno-quant crate (Toyota Way consolidation)

// NOTE: quantize_q5_k_matrix moved to trueno-quant crate (Toyota Way consolidation)

// NOTE: quantize_q4_k_matrix moved to trueno-quant crate (Toyota Way consolidation)

// Transpose Q4K data for matmul kernel compatibility (LAYOUT-002)
// GGUF stores weight matrices in column-major order (GGML convention).
// NOTE: transpose_q4k_for_matmul moved to trueno-quant crate (Toyota Way consolidation)
// NOTE: transpose_q5k_for_matmul moved to trueno-quant crate (Toyota Way consolidation)
// NOTE: transpose_q6k_for_matmul moved to trueno-quant crate (Toyota Way consolidation)
// NOTE: dequantize_q6_k_to_f32 moved to trueno-quant crate (Toyota Way consolidation)
// NOTE: dequantize_q5_k_to_f32 moved to trueno-quant crate (Toyota Way consolidation)

/// Check if a tensor name represents a 2D weight that needs transposition
///
/// Note: Scaffolding for PMAT-103 layout conversion optimization.
#[allow(dead_code)]
fn needs_transpose(name: &str, shape: &[usize]) -> bool {
    // Only transpose 2D weight tensors
    if shape.len() != 2 {
        return false;
    }

    // Transpose these weight tensors for matmul compatibility
    let weight_patterns = [
        "attn_output.weight",
        "attn_k.weight",
        "attn_q.weight",
        "attn_v.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "output.weight",
        "lm_head.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    ];

    weight_patterns.iter().any(|pattern| name.contains(pattern))
}

/// Save model tensors with optional compression
///
/// Note: For .apr output, use save_model_tensors_with_config() instead to embed metadata.
fn save_model_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    compression: Option<Compression>,
) -> Result<()> {
    // GH-165 FIX: If output is .apr, use APR format with embedded config
    let extension = output.extension().and_then(|e| e.to_str()).unwrap_or("");
    if extension == "apr" {
        return save_model_tensors_with_config(tensors, output, compression);
    }

    // For non-APR output (e.g., .safetensors), use plain SafeTensors
    save_safetensors(output, tensors).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to save converted model: {e}"),
    })
}

/// Save model tensors to APR format with embedded config metadata (GH-165 fix)
///
/// Infers model configuration from tensor shapes and embeds it in APR metadata.
/// This ensures AprTransformer can load with correct dimensions.
/// If config cannot be inferred (generic tensors), saves with minimal metadata.
fn save_model_tensors_with_config(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    _compression: Option<Compression>,
) -> Result<()> {
    // Try to infer model configuration from tensor shapes
    let config = infer_model_config_from_tensors(tensors);

    // Build AprV2Metadata with inferred config (or defaults)
    let mut metadata = AprV2Metadata::new("unknown");
    metadata.original_format = Some("safetensors".to_string());

    if let Some(cfg) = config {
        metadata.model_type = "qwen2".to_string(); // Detected transformer model
        metadata.hidden_size = cfg.hidden_size;
        metadata.num_layers = cfg.num_layers;
        metadata.vocab_size = cfg.vocab_size;
        metadata.num_heads = cfg.num_heads;
        metadata.num_kv_heads = cfg.num_kv_heads;
        metadata.intermediate_size = cfg.intermediate_size;
    }

    // Create writer and add all tensors
    let mut writer = AprV2Writer::new(metadata);
    for (name, (data, shape)) in tensors {
        writer.add_f32_tensor(name, shape.clone(), data);
    }

    // Write to file
    let apr_bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write APR format: {e}"),
    })?;

    fs::write(output, apr_bytes).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write output file: {e}"),
    })
}

/// Save model tensors to APR format with GGUF model config (F-REGR-231 fix)
///
/// This function preserves critical GGUF metadata including:
/// - rope_type: RoPE style (0=NORM, 2=NEOX) - CRITICAL for Qwen2.5 models
/// - rope_theta: Position encoding frequency
/// - rms_norm_eps: RMS normalization epsilon
/// - All other model dimensions from GGUF
///
/// Without this, APR defaults to rope_type=0 which produces garbage for Qwen2.5.
#[allow(dead_code)] // Superseded by save_model_tensors_with_gguf_config_and_tokenizer
fn save_model_tensors_with_gguf_config(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    _compression: Option<Compression>,
    gguf_config: &GgufModelConfig,
) -> Result<()> {
    // Build AprV2Metadata with GGUF config (not inferred from tensor shapes)
    let mut metadata = AprV2Metadata::new(gguf_config.architecture.as_deref().unwrap_or("qwen2"));
    metadata.original_format = Some("gguf".to_string());
    metadata.model_type = gguf_config
        .architecture
        .clone()
        .unwrap_or_else(|| "qwen2".to_string());

    // Copy all GGUF config fields to APR metadata
    metadata.hidden_size = gguf_config.hidden_size;
    metadata.num_layers = gguf_config.num_layers;
    metadata.num_heads = gguf_config.num_heads;
    metadata.num_kv_heads = gguf_config.num_kv_heads;
    metadata.vocab_size = gguf_config.vocab_size;
    metadata.intermediate_size = gguf_config.intermediate_size;
    metadata.max_position_embeddings = gguf_config.max_position_embeddings;

    // F-REGR-231 FIX: These fields are CRITICAL for correct inference
    metadata.rope_theta = gguf_config.rope_theta;
    metadata.rope_type = gguf_config.rope_type;
    metadata.rms_norm_eps = gguf_config.rms_norm_eps;

    // Create writer and add all tensors
    let mut writer = AprV2Writer::new(metadata);
    for (name, (data, shape)) in tensors {
        writer.add_f32_tensor(name, shape.clone(), data);
    }

    // Write to file
    let apr_bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write APR format: {e}"),
    })?;

    fs::write(output, apr_bytes).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write output file: {e}"),
    })
}

/// Save model tensors to APR format with GGUF config AND tokenizer (PMAT-113 fix)
///
/// This extends `save_model_tensors_with_gguf_config` to also embed the tokenizer
/// vocabulary for standalone APR inference without sibling tokenizer.json files.
fn save_model_tensors_with_gguf_config_and_tokenizer(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    _compression: Option<Compression>,
    gguf_config: &GgufModelConfig,
    tokenizer: Option<&GgufTokenizer>,
) -> Result<()> {
    // Build AprV2Metadata with GGUF config (not inferred from tensor shapes)
    let mut metadata = AprV2Metadata::new(gguf_config.architecture.as_deref().unwrap_or("qwen2"));
    metadata.original_format = Some("gguf".to_string());
    metadata.model_type = gguf_config
        .architecture
        .clone()
        .unwrap_or_else(|| "qwen2".to_string());
    // PMAT-113 FIX: Set architecture for chat template detection
    metadata.architecture = gguf_config.architecture.clone();

    // Copy all GGUF config fields to APR metadata
    metadata.hidden_size = gguf_config.hidden_size;
    metadata.num_layers = gguf_config.num_layers;
    metadata.num_heads = gguf_config.num_heads;
    metadata.num_kv_heads = gguf_config.num_kv_heads;
    metadata.vocab_size = gguf_config.vocab_size;
    metadata.intermediate_size = gguf_config.intermediate_size;
    metadata.max_position_embeddings = gguf_config.max_position_embeddings;

    // F-REGR-231 FIX: These fields are CRITICAL for correct inference
    metadata.rope_theta = gguf_config.rope_theta;
    metadata.rope_type = gguf_config.rope_type;
    metadata.rms_norm_eps = gguf_config.rms_norm_eps;

    // PMAT-113 FIX: Embed tokenizer vocabulary for standalone APR inference
    if let Some(tok) = tokenizer {
        if !tok.vocabulary.is_empty() {
            eprintln!(
                "[PMAT-113] Embedding {} vocabulary tokens into APR metadata",
                tok.vocabulary.len()
            );
            // Store vocabulary as JSON array in custom metadata
            let vocab_array: Vec<serde_json::Value> = tok
                .vocabulary
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            metadata.custom.insert(
                "tokenizer.vocabulary".to_string(),
                serde_json::Value::Array(vocab_array),
            );
            metadata.custom.insert(
                "tokenizer.vocab_size".to_string(),
                serde_json::Value::Number(serde_json::Number::from(tok.vocabulary.len())),
            );
        }
        if let Some(ref model_type) = tok.model_type {
            metadata.custom.insert(
                "tokenizer.model".to_string(),
                serde_json::Value::String(model_type.clone()),
            );
        }
        if let Some(bos_id) = tok.bos_token_id {
            metadata.custom.insert(
                "tokenizer.bos_token_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(bos_id)),
            );
        }
        if let Some(eos_id) = tok.eos_token_id {
            metadata.custom.insert(
                "tokenizer.eos_token_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(eos_id)),
            );
        }
        // PMAT-171: Embed BPE merge rules for standalone APR encoding
        if !tok.merges.is_empty() {
            eprintln!(
                "[PMAT-171] Embedding {} BPE merge rules into APR metadata",
                tok.merges.len()
            );
            let merges_array: Vec<serde_json::Value> = tok
                .merges
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            metadata.custom.insert(
                "tokenizer.merges".to_string(),
                serde_json::Value::Array(merges_array),
            );
        }
    }

    // Create writer and add all tensors
    let mut writer = AprV2Writer::new(metadata);
    for (name, (data, shape)) in tensors {
        writer.add_f32_tensor(name, shape.clone(), data);
    }

    // Write to file
    let apr_bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write APR format: {e}"),
    })?;

    fs::write(output, apr_bytes).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write output file: {e}"),
    })
}

/// Inferred model configuration from tensor shapes for Q4K quantization.
///
/// Used by `save_model_tensors_q4k` to populate APR metadata fields.
struct InferredQ4kConfig {
    hidden_size: Option<usize>,
    num_layers: Option<usize>,
    num_kv_heads: Option<usize>,
    vocab_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_heads: Option<usize>,
}

/// Infer model configuration from tensor names and shapes.
///
/// Scans the tensor map for well-known naming patterns (norm weights, embeddings,
/// layer indices, projection matrices, gate projections) and extracts architecture
/// dimensions. Assumes `head_dim=64` for head-count inference.
fn infer_q4k_config(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> InferredQ4kConfig {
    let mut cfg = InferredQ4kConfig {
        hidden_size: None,
        num_layers: None,
        num_kv_heads: None,
        vocab_size: None,
        intermediate_size: None,
        num_heads: None,
    };

    for (name, (_, shape)) in tensors {
        infer_q4k_single_tensor(&mut cfg, name, shape);
    }

    cfg
}

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
