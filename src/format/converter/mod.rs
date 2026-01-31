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
    detect_sharded_model, Architecture, ImportError, ImportOptions, QuantizationType, ShardedIndex,
    Source, TensorExpectation, ValidationConfig,
};

// PMAT-197: Import functions moved to import.rs that are used elsewhere
use import::{infer_model_config_from_tensors, load_safetensors_tensors, map_tensor_names};

// For tests: re-export helper types and functions
#[cfg(test)]
pub(crate) use crate::format::validation::{AprValidator, TensorStats};
#[cfg(test)]
pub(crate) use import::{
    compute_std, compute_tensor_stats, validate_single_tensor, TensorAccumulator,
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

    eprintln!(
        "[DEBUG apr_convert] input: {:?}, extension: {:?}",
        input_path, extension
    );

    // GH-181 FIX: Preserve Q4_K_M block alignment by using raw byte pass-through
    // When source is already Q4K quantized GGUF and target is Q4K, skip dequant→requant
    if extension == "gguf" && options.quantize == Some(QuantizationType::Q4K) {
        if let Ok(raw_result) = load_gguf_raw(input_path) {
            // Check if source contains Q4_K tensors (dtype 12)
            let has_q4k = raw_result
                .tensors
                .values()
                .any(|t| t.dtype == 12 || t.dtype == 13 || t.dtype == 14); // Q4_K, Q5_K, Q6_K

            if has_q4k {
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

                // Calculate original size from raw bytes
                let original_size: usize = mapped_tensors.values().map(|t| t.data.len()).sum();
                let original_count = mapped_tensors.len();

                // Write APR file with raw quantized tensors (preserves block alignment)
                let import_opts = ImportOptions {
                    architecture: Architecture::Qwen2,
                    ..Default::default()
                };
                write_apr_file_raw(
                    &mapped_tensors,
                    output_path,
                    &import_opts,
                    Some(&raw_result.tokenizer),
                    Some(&raw_result.model_config),
                )?;

                let converted_size = fs::metadata(output_path)
                    .map(|m| m.len() as usize)
                    .unwrap_or(0);

                return Ok(ConvertReport {
                    original_size,
                    converted_size,
                    tensor_count: original_count,
                    quantization: options.quantize,
                    compression: options.compress,
                    reduction_ratio: if converted_size > 0 {
                        original_size as f64 / converted_size as f64
                    } else {
                        0.0
                    },
                });
            }
        }
    }

    // F-REGR-231 FIX: For GGUF input, load with full config to preserve rope_type
    // Qwen2.5 models require rope_type=2 (NEOX style), not default 0 (NORM style)
    // PMAT-113 FIX: Also preserve tokenizer for APR embedding
    let (gguf_config, gguf_tokenizer) = if extension == "gguf" {
        match load_gguf_with_tokenizer(input_path) {
            Ok(result) => {
                eprintln!(
                    "[PMAT-113] Extracted tokenizer with {} vocabulary tokens",
                    result.tokenizer.vocabulary.len()
                );
                (Some(result.model_config), Some(result.tokenizer))
            }
            Err(_) => (None, None), // Fall back to inference if GGUF loading fails
        }
    } else {
        (None, None)
    };

    // Step 1: Load tensors
    let tensors = load_model_tensors(input_path)?;
    let original_size = calculate_tensor_size(&tensors);
    let original_count = tensors.len();

    // Step 1b: Map GGUF tensor names to APR canonical format (PMAT-205 fix / GH-190)
    // GGUF uses names like "blk.0.attn_q.weight" but APR loaders expect
    // bare names like "layers.0.self_attn.q_proj.weight" (no "model." prefix)
    let tensors = if extension == "gguf" {
        eprintln!(
            "[PMAT-205] Mapping {} GGUF tensor names to APR canonical format...",
            tensors.len()
        );
        let mapped = map_tensor_names(&tensors, Architecture::Qwen2);
        // Debug: show a few mapped names
        for (i, name) in mapped.keys().take(5).enumerate() {
            eprintln!("[PMAT-205]   {}: {}", i, name);
        }
        mapped
    } else {
        tensors
    };

    // Step 2: Handle Q4K specially - store raw Q4K bytes in APR format
    if options.quantize == Some(QuantizationType::Q4K) {
        save_model_tensors_q4k(&tensors, output_path)?;

        let converted_size = fs::metadata(output_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);

        return Ok(ConvertReport {
            original_size,
            converted_size,
            tensor_count: original_count,
            quantization: options.quantize,
            compression: options.compress,
            reduction_ratio: if converted_size > 0 {
                original_size as f64 / converted_size as f64
            } else {
                0.0
            },
        });
    }

    // Step 2b: Apply other quantization types (Fp16, Int8, Int4)
    let tensors = if let Some(quant_type) = &options.quantize {
        quantize_tensors(&tensors, quant_type)?
    } else {
        tensors
    };

    // Step 3: Save output with GGUF config if available (F-REGR-231 fix)
    // PMAT-113 FIX: Also embed tokenizer for standalone APR inference
    if let Some(ref config) = gguf_config {
        save_model_tensors_with_gguf_config_and_tokenizer(
            &tensors,
            output_path,
            options.compress,
            config,
            gguf_tokenizer.as_ref(),
        )?;
    } else {
        save_model_tensors(&tensors, output_path, options.compress)?;
    }

    // Step 4: Calculate stats
    let converted_size = fs::metadata(output_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    Ok(ConvertReport {
        original_size,
        converted_size,
        tensor_count: original_count,
        quantization: options.quantize,
        compression: options.compress,
        reduction_ratio: if converted_size > 0 {
            original_size as f64 / converted_size as f64
        } else {
            0.0
        },
    })
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

    Ok(tensors)
}

/// Load APR tensors and dequantize to F32 (PMAT-174)
///
/// APR binary format:
/// - Header (44 bytes): magic, version, flags, tensor_count, offsets, checksum
/// - Metadata: JSON config
/// - Tensor Index: binary tensor entries
/// - Tensor Data: raw bytes
///
/// Handles all APR dtypes: F32, F16, BF16, Q4_K, Q6_K, Q8_0
fn load_apr_tensors_f32(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    use std::io::Read;

    // Read entire file
    let mut file = fs::File::open(path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to open APR file: {e}"),
    })?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| AprenderError::FormatError {
            message: format!("Failed to read APR file: {e}"),
        })?;

    // Validate header (44 bytes minimum)
    if data.len() < 44 {
        return Err(AprenderError::FormatError {
            message: "APR file too small for header".to_string(),
        });
    }

    // Check magic "APR\0" (0x00525041 in little-endian)
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 0x0052_5041 {
        // "APR\0" in little-endian
        return Err(AprenderError::FormatError {
            message: format!("Invalid APR magic: 0x{magic:08X}, expected APR"),
        });
    }

    // Parse header
    let tensor_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let tensor_index_offset = u64::from_le_bytes([
        data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
    ]) as usize;
    let data_offset = u64::from_le_bytes([
        data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
    ]) as usize;

    // Parse tensor index
    let mut tensors = BTreeMap::new();
    let mut pos = tensor_index_offset;

    for _ in 0..tensor_count {
        if pos + 4 > data.len() {
            break;
        }

        // Name: len (2 bytes) + bytes
        let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
        pos += name_len;

        // Dtype (1 byte)
        let dtype_byte = data[pos];
        pos += 1;

        // Shape: ndim (1 byte) + dims (8 bytes each)
        let ndim = data[pos] as usize;
        pos += 1;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let dim = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]) as usize;
            pos += 8;
            shape.push(dim);
        }

        // Offset and size
        let offset = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;
        pos += 8;
        let size = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;
        pos += 8;

        // Load tensor data
        let tensor_start = data_offset + offset;
        let tensor_end = tensor_start + size;
        if tensor_end > data.len() {
            continue;
        }
        let tensor_bytes = &data[tensor_start..tensor_end];
        let num_elements: usize = shape.iter().product();

        // Dequantize based on dtype
        let f32_data = match dtype_byte {
            0 => {
                // F32
                tensor_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            1 => {
                // F16
                dequantize_f16_to_f32(tensor_bytes, num_elements)
            }
            2 => {
                // BF16
                dequantize_bf16_to_f32(tensor_bytes, num_elements)
            }
            8 => {
                // Q4_K
                dequantize_q4_k_to_f32(tensor_bytes, num_elements)
            }
            9 => {
                // Q6_K
                dequantize_q6_k_to_f32(tensor_bytes, num_elements)
            }
            10 => {
                // Q8_0
                dequantize_q8_0_to_f32(tensor_bytes, num_elements)
            }
            _ => {
                // Default to F32 interpretation
                tensor_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
        };

        // PMAT-187: Validate tensor values after dequantization (Jidoka - stop the line)
        validate_tensor_values(&name, &f32_data)?;

        tensors.insert(name, (f32_data, shape));
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
        let scale = f16_to_f32(scale_bits);

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

/// Apply quantization to tensors
pub(crate) fn quantize_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    quant_type: &QuantizationType,
) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let mut result = BTreeMap::new();

    for (name, (data, shape)) in tensors {
        let quantized_data = match quant_type {
            QuantizationType::Fp16 => quantize_fp16(data),
            QuantizationType::Int8 => quantize_int8(data),
            QuantizationType::Int4 => quantize_int4(data),
            QuantizationType::Q4K => {
                // Q4K: quantize to packed bytes then dequantize back to f32
                // This preserves the shape but shows quantization error
                let q4k_bytes = quantize_q4_k(data);
                dequantize_q4_k_to_f32(&q4k_bytes, data.len())
            }
        };
        result.insert(name.clone(), (quantized_data, shape.clone()));
    }

    Ok(result)
}

/// Dequantize Q4_K bytes back to F32 (for verification/testing)
/// Dequantize Q4_K data to f32 (llama.cpp compatible)
///
/// Matches the encoder format and realizar's `dequantize_q4_k_apr`:
/// - Scale packing: blocks 0-3 in lower 6 bits, blocks 4-7 use upper bits
/// - Value packing: 64-value chunks with low/high nibble interleaving
fn dequantize_q4_k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 144;
    // PMAT-177: Minimum valid f16 normal value (~6.1e-5), clamp scales to avoid NaN
    const F16_MIN_NORMAL: f32 = 6.1e-5;

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * SUPER_BLOCK_SIZE];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * SUPER_BLOCK_SIZE;

        if sb_start + SUPER_BLOCK_BYTES > data.len() {
            break;
        }

        // Read d and dmin (f16) - PMAT-177: Validate for NaN/Inf
        let d_raw = f16_to_f32(u16::from_le_bytes([data[sb_start], data[sb_start + 1]]));
        let dmin_raw = f16_to_f32(u16::from_le_bytes([data[sb_start + 2], data[sb_start + 3]]));

        // PMAT-177: Replace NaN/Inf/subnormal with safe values to prevent corruption
        let d = if d_raw.is_nan() || d_raw.is_infinite() || d_raw.abs() < F16_MIN_NORMAL {
            0.0
        } else {
            d_raw
        };
        let dmin = if dmin_raw.is_nan() || dmin_raw.is_infinite() || dmin_raw.abs() < F16_MIN_NORMAL
        {
            0.0
        } else {
            dmin_raw
        };

        // Unpack scales and mins (llama.cpp format)
        let scales_bytes = &data[sb_start + 4..sb_start + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        for i in 0..4 {
            // Blocks 0-3: lower 6 bits of bytes 0-3 and 4-7
            scales[i] = scales_bytes[i] & 0x3F;
            mins[i] = scales_bytes[i + 4] & 0x3F;
            // Blocks 4-7: lower 4 bits from bytes 8-11, upper 2 bits from bytes 0-3/4-7
            scales[i + 4] = (scales_bytes[i + 8] & 0x0F) | ((scales_bytes[i] >> 6) << 4);
            mins[i + 4] = (scales_bytes[i + 8] >> 4) | ((scales_bytes[i + 4] >> 6) << 4);
        }

        // Read quantized values (128 bytes = 256 4-bit values)
        let qs = &data[sb_start + 16..sb_start + 144];

        // PMAT-190 FIX: Match gguf.rs dequantize_q4_k layout exactly
        // Q4_K uses 8 sub-blocks of 32 elements each
        // Each sub-block uses ONE scale for ALL 32 elements (not different for low/high!)
        // Layout: 16 bytes per sub-block, each byte → 2 values (low nibble, high nibble)
        let mut ys_index = out_start;

        for j in 0..8 {
            // One scale per sub-block (same for both nibbles!)
            let scale = d * f32::from(scales[j]);
            let min_val = dmin * f32::from(mins[j]);

            // Process 16 bytes → 32 values
            for l in 0..16 {
                let q_byte = qs[j * 16 + l];
                let q0 = f32::from(q_byte & 0x0F);
                let q1 = f32::from(q_byte >> 4);
                result[ys_index] = q0 * scale - min_val;
                result[ys_index + 1] = q1 * scale - min_val;
                ys_index += 2;
            }
        }
    }

    result.truncate(num_elements);
    result
}

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

/// Quantize F32 data to Q4_K format (GGML K-quants)
///
/// Q4_K format: 256-element super-blocks, each with:
/// - d (f16, 2 bytes): scale for scales
/// - dmin (f16, 2 bytes): scale for mins (offsets)
/// - scales (12 bytes): 8 6-bit scale values packed
/// - qs (128 bytes): 256 4-bit quantized values
///
/// Decoding formula: `value = q * (d * scales[j]) - (dmin * mins[j])`
/// Total: 144 bytes per 256 elements = 4.5 bits/weight
///
/// Returns packed Q4_K bytes ready for APR storage.
/// Quantize f32 data to Q4_K format (llama.cpp compatible)
///
/// Q4_K super-block layout (144 bytes per 256 elements):
/// - d: 2 bytes (f16 global scale)
/// - dmin: 2 bytes (f16 global min scale)
/// - scales: 12 bytes (packed 6-bit scales and mins for 8 sub-blocks)
/// - qs: 128 bytes (4-bit quantized values, interleaved low/high nibbles)
///
/// Scale packing (llama.cpp get_scale_min_k4):
/// - Blocks 0-3: scales[j] = scale_6bit, scales[j+4] = min_6bit
/// - Blocks 4-7: packed in bytes 8-11 using high bits of bytes 0-7
///
/// Value packing (candle/llama.cpp layout):
/// - For each 64-value chunk: 32 bytes store low nibbles first, then high nibbles
/// - Low nibbles use scale[is], high nibbles use scale[is+1]
fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUB_BLOCK_SIZE: usize = 32;
    const SUPER_BLOCK_BYTES: usize = 144; // 2 + 2 + 12 + 128
                                          // PMAT-177: Minimum valid f16 normal value (~6.1e-5) - prevents NaN on round-trip
    const F16_MIN_NORMAL: f32 = 6.1e-5;

    if data.is_empty() {
        return vec![];
    }

    let num_blocks = (data.len() + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = Vec::with_capacity(num_blocks * SUPER_BLOCK_BYTES);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * SUPER_BLOCK_SIZE;
        let block_end = (block_start + SUPER_BLOCK_SIZE).min(data.len());
        let block_data = &data[block_start..block_end];

        // Pad to 256 if needed
        let mut padded = [0.0f32; SUPER_BLOCK_SIZE];
        padded[..block_data.len()].copy_from_slice(block_data);

        // Compute per-sub-block statistics (8 sub-blocks of 32 elements each)
        // Q4_K decoding: value = q * d * scale - dmin * min
        let mut sub_scales = [0.0f32; 8];
        let mut sub_mins = [0.0f32; 8];

        for (j, sub_block) in padded.chunks(SUB_BLOCK_SIZE).enumerate().take(8) {
            let min = sub_block.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = sub_block.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let range = max - min;

            // PMAT-177: Clamp to F16_MIN_NORMAL to prevent underflow in f16 encoding
            sub_scales[j] = if range > F16_MIN_NORMAL {
                range / 15.0
            } else {
                F16_MIN_NORMAL
            };
            sub_mins[j] = (-min).max(0.0); // Store as positive offset
        }

        // Find global scale factors d and dmin
        let max_scale = sub_scales.iter().fold(0.0f32, |a, &b| a.max(b));
        let max_min = sub_mins.iter().fold(0.0f32, |a, &b| a.max(b));

        // PMAT-177: Clamp d/dmin to F16_MIN_NORMAL to prevent NaN after f16 round-trip
        let d = if max_scale > F16_MIN_NORMAL {
            max_scale / 63.0
        } else {
            F16_MIN_NORMAL
        };
        let dmin = if max_min > F16_MIN_NORMAL {
            max_min / 63.0
        } else {
            F16_MIN_NORMAL
        };

        // Compute 6-bit scales and mins for each sub-block
        let mut scales_6bit = [0u8; 8];
        let mut mins_6bit = [0u8; 8];

        for j in 0..8 {
            scales_6bit[j] = ((sub_scales[j] / d).round() as u8).min(63);
            mins_6bit[j] = ((sub_mins[j] / dmin).round() as u8).min(63);
        }

        // Write d (f16) - 2 bytes
        let d_f16 = f32_to_f16(d);
        result.extend_from_slice(&d_f16.to_le_bytes());

        // Write dmin (f16) - 2 bytes
        let dmin_f16 = f32_to_f16(dmin);
        result.extend_from_slice(&dmin_f16.to_le_bytes());

        // Pack scales and mins into 12 bytes (llama.cpp format)
        // Decoder expects:
        // - Blocks 0-3: scale = scales[j] & 63, min = scales[j+4] & 63
        // - Blocks 4-7: scale = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
        //               min = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
        let mut scales_packed = [0u8; 12];

        // Blocks 0-3: store lower 6 bits directly, use upper 2 bits for blocks 4-7
        for i in 0..4 {
            // Lower 6 bits of scale[i], upper 2 bits store part of scale[i+4]
            scales_packed[i] = (scales_6bit[i] & 0x3F) | ((scales_6bit[i + 4] & 0x30) << 2);
            // Lower 6 bits of min[i], upper 2 bits store part of min[i+4]
            scales_packed[i + 4] = (mins_6bit[i] & 0x3F) | ((mins_6bit[i + 4] & 0x30) << 2);
        }

        // Blocks 4-7: store lower 4 bits of scale and min in bytes 8-11
        for i in 0..4 {
            scales_packed[i + 8] = (scales_6bit[i + 4] & 0x0F) | ((mins_6bit[i + 4] & 0x0F) << 4);
        }
        result.extend_from_slice(&scales_packed);

        // PMAT-190 FIX: Quantize to match gguf.rs dequantize_q4_k layout
        // Q4_K: 8 sub-blocks of 32 elements, ONE scale per sub-block
        // 16 bytes per sub-block, each byte packs TWO CONSECUTIVE values
        let mut qs = [0u8; 128];

        for j in 0..8 {
            // One scale per sub-block (same for both nibbles!)
            let scale = d * f32::from(scales_6bit[j]);
            let min_val = dmin * f32::from(mins_6bit[j]);

            // Process 32 values → 16 bytes
            for l in 0..16 {
                let idx0 = j * 32 + l * 2; // First value of pair
                let idx1 = j * 32 + l * 2 + 1; // Second value of pair

                // Quantize: q = (value + min_val) / scale
                let q0 = if scale > 1e-10 {
                    ((padded[idx0] + min_val) / scale).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };
                let q1 = if scale > 1e-10 {
                    ((padded[idx1] + min_val) / scale).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };

                // Pack: low nibble = q0, high nibble = q1
                qs[j * 16 + l] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
            }
        }
        result.extend_from_slice(&qs);
    }

    result
}

/// Transpose Q4K data for matmul kernel compatibility (PMAT-103)
///
/// GGUF stores weight matrices in column-major order (GGML convention) for `x @ W`.
/// The trueno Q4K kernel expects row-major order for `W @ x`.
/// These are transposes of each other.
///
/// This function:
/// 1. Dequantizes Q4K to F32
/// 2. Transposes from [rows, cols] to [cols, rows]
/// 3. Re-quantizes to Q4K
///
/// Returns: (transposed_q4k_bytes, transposed_shape)
///
/// Note: GH-189 FIX - Now used in write.rs for GGUF→APR conversion.
pub(crate) fn transpose_q4k_for_matmul(data: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    // Only transpose 2D tensors
    if shape.len() != 2 {
        return (data.to_vec(), shape.to_vec());
    }

    let rows = shape[0];
    let cols = shape[1];
    let num_elements = rows * cols;

    // Step 1: Dequantize Q4K to F32
    let f32_data = dequantize_q4_k_to_f32(data, num_elements);

    // Step 2: Transpose the F32 matrix from [rows, cols] to [cols, rows]
    // Original: data[i * cols + j] = element at row i, column j
    // Transposed: data[j * rows + i] = element at row j, column i
    let mut transposed_f32 = vec![0.0f32; num_elements];
    for i in 0..rows {
        for j in 0..cols {
            transposed_f32[j * rows + i] = f32_data[i * cols + j];
        }
    }

    // Step 3: Re-quantize to Q4K
    let transposed_q4k = quantize_q4_k(&transposed_f32);

    // Return with swapped dimensions
    (transposed_q4k, vec![cols, rows])
}

/// Transpose Q6K data for matmul kernel compatibility (PMAT-103)
///
/// Same as transpose_q4k_for_matmul but for Q6K format.
///
/// Note: GH-189 FIX - Now used in write.rs for GGUF→APR conversion.
/// Currently outputs Q4K for re-quantized transpose until Q6K encoder is added.
pub(crate) fn transpose_q6k_for_matmul(data: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    // Only transpose 2D tensors
    if shape.len() != 2 {
        return (data.to_vec(), shape.to_vec());
    }

    let rows = shape[0];
    let cols = shape[1];
    let num_elements = rows * cols;

    // Step 1: Dequantize Q6K to F32
    let f32_data = dequantize_q6_k_to_f32(data, num_elements);

    // Step 2: Transpose the F32 matrix
    let mut transposed_f32 = vec![0.0f32; num_elements];
    for i in 0..rows {
        for j in 0..cols {
            transposed_f32[j * rows + i] = f32_data[i * cols + j];
        }
    }

    // Step 3: Re-quantize to Q6K (for now, convert to Q4K since we don't have Q6K encoder)
    // Note: Proper Q6K quantization will be added when Q6K encoder is implemented
    let transposed_q4k = quantize_q4_k(&transposed_f32);

    // Return with swapped dimensions
    (transposed_q4k, vec![cols, rows])
}

/// Dequantize Q6_K data to f32 (for transpose)
///
/// Note: Scaffolding for PMAT-103 layout conversion optimization.
#[allow(dead_code)]
fn dequantize_q6_k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 210;
    // PMAT-177: Minimum valid f16 normal value (~6.1e-5), clamp scales to avoid NaN
    const F16_MIN_NORMAL: f32 = 6.1e-5;

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * SUPER_BLOCK_SIZE];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * SUPER_BLOCK_SIZE;

        if sb_start + SUPER_BLOCK_BYTES > data.len() {
            break;
        }

        // Q6_K layout: ql (128B) + qh (64B) + scales (16B) + d (2B)
        let ql = &data[sb_start..sb_start + 128];
        let qh = &data[sb_start + 128..sb_start + 192];
        let scales_raw = &data[sb_start + 192..sb_start + 208];
        let d_raw = f16_to_f32(u16::from_le_bytes([
            data[sb_start + 208],
            data[sb_start + 209],
        ]));

        // PMAT-177: Replace NaN/Inf/subnormal with safe values to prevent corruption
        let d = if d_raw.is_nan() || d_raw.is_infinite() || d_raw.abs() < F16_MIN_NORMAL {
            0.0
        } else {
            d_raw
        };

        // Decode scales as signed i8
        let mut scales = [0i8; 16];
        for i in 0..16 {
            scales[i] = scales_raw[i] as i8;
        }

        // Dequantize 256 values
        for j in 0..256 {
            // Get 6-bit quantized value
            let ql_byte = ql[j / 2];
            let ql_val = if j % 2 == 0 {
                ql_byte & 0x0F
            } else {
                ql_byte >> 4
            };

            let qh_byte = qh[j / 4];
            let qh_val = (qh_byte >> ((j % 4) * 2)) & 0x03;

            let q6 = (ql_val as i32) | ((qh_val as i32) << 4);
            let q6_signed = q6 - 32; // Q6K uses offset encoding

            // Get scale for this 16-element block
            let scale_idx = j / 16;
            let scale = scales[scale_idx] as f32;

            result[out_start + j] = d * scale * q6_signed as f32;
        }
    }

    result.truncate(num_elements);
    result
}

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

/// Save model tensors with Q4K quantization in APR format
///
/// Selectively quantizes large weight tensors while keeping biases and norms as F32.
/// Uses APR format with proper Q4K dtype for GPU-accelerated inference.
fn save_model_tensors_q4k(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
) -> Result<()> {
    use std::io::Write as IoWrite;

    // Infer model configuration from tensor shapes
    let mut hidden_size: Option<usize> = None;
    let mut num_layers: Option<usize> = None;
    let mut num_kv_heads: Option<usize> = None;
    let mut vocab_size: Option<usize> = None;
    let mut intermediate_size: Option<usize> = None;
    let mut num_heads: Option<usize> = None;

    for (name, (_, shape)) in tensors {
        // Infer hidden_size from norm weights (1D tensor of hidden_dim)
        if name.contains("input_layernorm.weight") && shape.len() == 1 {
            hidden_size = Some(shape[0]);
        }
        // Infer vocab_size from embedding [vocab_size, hidden_dim]
        if name.contains("embed_tokens.weight") && shape.len() == 2 {
            vocab_size = Some(shape[0]);
            if hidden_size.is_none() {
                hidden_size = Some(shape[1]);
            }
        }
        // Count layers
        if let Some(idx) = name.strip_prefix("model.layers.") {
            if let Some(layer_num) = idx.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                num_layers = Some(num_layers.map_or(layer_num + 1, |n| n.max(layer_num + 1)));
            }
        }
        // Infer kv_heads from k_proj shape [kv_dim, hidden_dim]
        if name.contains("k_proj.weight") && shape.len() == 2 && hidden_size.is_some() {
            // kv_dim = shape[0], hidden_dim = shape[1]
            // num_kv_heads = kv_dim / head_dim where head_dim = hidden_dim / num_heads
            // For Qwen2-0.5B: kv_dim=128, hidden_dim=896, head_dim=64, num_kv_heads=2
            num_kv_heads = Some(shape[0] / 64); // Assume head_dim=64 for now
        }
        // Infer num_heads from q_proj shape [q_dim, hidden_dim]
        if name.contains("q_proj.weight") && shape.len() == 2 {
            // q_dim = hidden_dim for standard attention
            // num_heads = hidden_dim / head_dim = hidden_dim / 64
            num_heads = Some(shape[0] / 64);
        }
        // Infer intermediate_size from gate_proj [intermediate, hidden]
        if name.contains("gate_proj.weight") && shape.len() == 2 {
            intermediate_size = Some(shape[0]);
        }
    }

    // Create APR metadata
    let param_count: u64 = tensors.values().map(|(data, _)| data.len() as u64).sum();

    let metadata = AprV2Metadata {
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
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_size,
        max_position_embeddings: Some(32768), // Default for Qwen2
        rope_theta: Some(1000000.0),          // Default for Qwen2
        rope_type: Some(2),                   // NEOX style for Qwen2 (PMAT-114)
        rms_norm_eps: Some(1e-6),             // Default for Qwen2
        custom: std::collections::HashMap::new(),
    };

    let mut writer = AprV2Writer::new(metadata);

    // Add tensors, selectively quantizing to Q4K
    for (name, (data, shape)) in tensors {
        // Skip quantization for small tensors (biases, norms, scales)
        // and for 1D tensors which are typically biases/norms
        let should_quantize = shape.len() >= 2
            && data.len() >= 256  // Minimum size for Q4K (one super-block)
            && !name.contains("bias")
            && !name.contains("norm")
            && !name.contains("scale")
            && !name.contains("embed"); // Keep embeddings as F32 for now

        if should_quantize {
            // Quantize to Q4K bytes
            let q4k_bytes = quantize_q4_k(data);
            writer.add_q4k_raw_tensor(name, shape.clone(), q4k_bytes);
        } else {
            // Keep as F32
            writer.add_f32_tensor(name, shape.clone(), data);
        }
    }

    // Write to file
    let bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to serialize APR format: {e}"),
    })?;

    let mut file = fs::File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;

    file.write_all(&bytes)
        .map_err(|e| AprenderError::FormatError {
            message: format!("Failed to write APR file: {e}"),
        })?;

    Ok(())
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
