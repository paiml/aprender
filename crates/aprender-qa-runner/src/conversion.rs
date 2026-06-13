//! Format Conversion Testing (P0 CRITICAL)
//!
//! Implements bi-directional format conversion testing across all backends.
//! This is the most critical requirement of the entire project.
//!
//! # Five Whys
//!
//! 1. Why format conversion testing? Models exist in multiple formats.
//! 2. Why is it critical? Incorrect conversion corrupts all inference.
//! 3. Why are subtle errors dangerous? They pass basic checks but produce wrong outputs.
//! 4. Why can't normal tests catch this? They verify "runs" not "identical output".
//! 5. Why P0? A single bit flip invalidates millions of inferences.
//!
//! # Bug Classification (GH-187)
//!
//! This module implements detection for common conversion bugs that have
//! occurred 50+ times:
//!
//! - **EMBEDDING_TRANSPOSITION**: Embedding stored as `[hidden_dim, vocab_size]`
//!   but `embed()` expects `[vocab_size, hidden_dim]`. Causes garbage output.
//! - **TOKENIZER_MISSING**: APR file doesn't include embedded tokenizer.
//! - **WEIGHT_CORRUPTION**: Tensor values corrupted during conversion.
//! - **SHAPE_MISMATCH**: Tensor dimensions don't match expected config.

#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::if_not_else)]
#![allow(clippy::use_self)]

use crate::error::{Error, Result};
use crate::evidence::Evidence;
use aprender_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Isolated output directory for conversion test artifacts.
///
/// Implements ISO-OUT-001: All conversion test outputs are written to an isolated
/// directory, never to the source model location.
///
/// # Directory Structure
///
/// ```text
/// {base}/conversions/{org}/{repo}/{test_type}/
/// ```
///
/// Where `test_type` is one of: `basic`, `semantic`, `idempotency`, `comparison`, `round-trip`
#[derive(Debug, Clone)]
pub struct ConversionOutputDir {
    base: PathBuf,
    org: String,
    repo: String,
}

impl ConversionOutputDir {
    /// Create a new conversion output directory for a model.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Base output directory (e.g., `output/`)
    /// * `model_id` - Model identifier containing org/repo
    #[must_use]
    pub fn new(output_dir: &Path, model_id: &ModelId) -> Self {
        Self {
            base: output_dir.to_path_buf(),
            org: model_id.org.clone(),
            repo: model_id.name.clone(),
        }
    }

    /// Get the base conversions directory for this model.
    fn model_dir(&self) -> PathBuf {
        self.base
            .join("conversions")
            .join(&self.org)
            .join(&self.repo)
    }

    /// Get output directory for basic conversion tests.
    #[must_use]
    pub fn basic_dir(&self) -> PathBuf {
        self.model_dir().join("basic")
    }

    /// Get output directory for semantic conversion tests.
    #[must_use]
    pub fn semantic_dir(&self) -> PathBuf {
        self.model_dir().join("semantic")
    }

    /// Get output directory for idempotency tests.
    #[must_use]
    pub fn idempotency_dir(&self) -> PathBuf {
        self.model_dir().join("idempotency")
    }

    /// Get output directory for comparison tests.
    #[must_use]
    pub fn comparison_dir(&self) -> PathBuf {
        self.model_dir().join("comparison")
    }

    /// Get output directory for round-trip tests.
    #[must_use]
    pub fn round_trip_dir(&self) -> PathBuf {
        self.model_dir().join("round-trip")
    }

    /// Generate an output path for a converted model file.
    ///
    /// # Arguments
    ///
    /// * `test_type` - Type of test (used as subdirectory)
    /// * `source_name` - Original model filename (without extension)
    /// * `tag` - Test-specific tag (e.g., "idem1", "direct")
    /// * `target_format` - Target format for extension
    #[must_use]
    pub fn output_path(
        &self,
        test_type: &str,
        source_name: &str,
        tag: &str,
        target_format: Format,
    ) -> PathBuf {
        let ext = match target_format {
            Format::Gguf => "gguf",
            Format::SafeTensors => "safetensors",
            Format::Apr => "apr",
        };
        let dir = self.model_dir().join(test_type);
        dir.join(format!("{source_name}.{tag}.{ext}"))
    }

    /// Ensure the output directory exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    pub fn ensure_dir(&self, test_type: &str) -> std::io::Result<PathBuf> {
        let dir = self.model_dir().join(test_type);
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    /// Clean up all conversion artifacts for this model.
    ///
    /// # Errors
    ///
    /// Returns an error if cleanup fails.
    pub fn cleanup(&self) -> std::io::Result<()> {
        let dir = self.model_dir();
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }
}

/// Resolve a model directory path to an actual model file for a specific format.
///
/// Handles multiple directory structures:
/// - **File mode**: If `base_path` is already a file, validates extension matches format
/// - **APR cache**: `{base_path}/{format}/model.{ext}` (e.g., `model_cache/gguf/model.gguf`)
/// - **HuggingFace cache**: `{base_path}/model.{ext}` (flat structure in snapshot directory)
///
/// # Errors
///
/// Returns an error if the path cannot be resolved to a valid model file.
pub fn resolve_model_path(base_path: &Path, format: Format) -> Result<std::path::PathBuf> {
    if base_path.is_file() {
        return resolve_file_by_format(base_path, format);
    }

    let ext = format_extension(format);

    // Try APR cache structure: {base}/{ext}/model.{ext}
    let resolved = base_path.join(ext).join(format!("model.{ext}"));
    if resolved.exists() {
        return Ok(resolved);
    }

    // Try sharded SafeTensors index
    if ext == "safetensors" {
        let sharded_index = base_path.join(ext).join("model.safetensors.index.json");
        if sharded_index.exists() {
            return Ok(sharded_index);
        }
    }

    // Try HuggingFace cache structure: {base}/model.{ext} (flat)
    let flat_resolved = base_path.join(format!("model.{ext}"));
    if flat_resolved.exists() {
        return Ok(flat_resolved);
    }

    // Search format subdir, then base dir for any matching file
    let format_dir = base_path.join(ext);
    find_file_by_extension(&format_dir, ext)
        .or_else(|| find_file_by_extension(base_path, ext))
        .ok_or_else(|| {
            Error::Execution(format!(
                "No {ext} file found in {}/{ext}/ or {}/",
                base_path.display(),
                base_path.display()
            ))
        })
}

fn resolve_file_by_format(path: &Path, format: Format) -> Result<std::path::PathBuf> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let expected = format_extension(format);
    if ext == expected {
        Ok(path.to_path_buf())
    } else {
        Err(Error::Execution(format!(
            "File extension mismatch: expected .{expected}, got .{ext}"
        )))
    }
}

fn format_extension(format: Format) -> &'static str {
    match format {
        Format::Gguf => "gguf",
        Format::Apr => "apr",
        Format::SafeTensors => "safetensors",
    }
}

/// Build the in-place conversion output path for `source` with `target_ext`,
/// idempotently (PMAT-743). `Path::with_extension("converted.<ext>")` replaces only
/// the final extension component, so a source already ending `.converted.<x>` would
/// compound into `.converted.converted.<x>` and pollute the model cache unboundedly
/// across re-conversions (observed in the HF cache as
/// `*.converted.converted.safetensors`). Normalizing the stem first makes the output
/// stable; for a source that does NOT already contain `.converted` the result is
/// identical to the old `with_extension` form.
fn converted_output_path(source: &Path, target_ext: &str) -> PathBuf {
    let mut stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    while let Some(base) = stem.strip_suffix(".converted") {
        stem = base;
    }
    source.with_file_name(format!("{stem}.converted.{target_ext}"))
}

fn find_file_by_extension(dir: &Path, ext: &str) -> Option<std::path::PathBuf> {
    std::fs::read_dir(dir).ok()?.flatten().find_map(|entry| {
        let p = entry.path();
        if p.extension().is_some_and(|e| e == ext) {
            Some(p)
        } else {
            None
        }
    })
}

/// Tolerance for floating-point comparison
pub const EPSILON: f64 = 1e-6;

/// Classification of conversion bugs (GH-187)
///
/// These bugs have been observed 50+ times in production.
/// Detection enables faster root cause analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversionBugType {
    /// Embedding stored as [hidden_dim, vocab_size] instead of [vocab_size, hidden_dim]
    /// Symptom: Output is garbage tokens (often PAD tokens or random sequences)
    EmbeddingTransposition,
    /// APR file missing embedded tokenizer from GGUF metadata
    /// Symptom: [PMAT-172] error, output doesn't match prompt semantics
    TokenizerMissing,
    /// Tensor values corrupted during conversion (NaN, Inf, zeros)
    /// Symptom: All-zero output or NaN propagation
    WeightCorruption,
    /// Tensor dimensions don't match model config
    /// Symptom: Runtime shape mismatch errors
    ShapeMismatch,
    /// Output semantically wrong but structurally valid
    /// Symptom: Model "runs" but produces completely wrong answers
    SemanticDrift,
    /// Unknown bug type - requires manual investigation
    Unknown,
}

impl ConversionBugType {
    /// Get the gate ID for this bug type
    #[must_use]
    pub fn gate_id(&self) -> &'static str {
        match self {
            Self::EmbeddingTransposition => "F-CONV-EMBED-001",
            Self::TokenizerMissing => "F-CONV-TOK-001",
            Self::WeightCorruption => "F-CONV-WEIGHT-001",
            Self::ShapeMismatch => "F-CONV-SHAPE-001",
            Self::SemanticDrift => "F-CONV-SEMANTIC-001",
            Self::Unknown => "F-CONV-UNKNOWN-001",
        }
    }

    /// Get a human-readable description
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::EmbeddingTransposition => "Embedding tensor transposition bug",
            Self::TokenizerMissing => "Embedded tokenizer missing from APR file",
            Self::WeightCorruption => "Weight tensor corruption (NaN/Inf/zeros)",
            Self::ShapeMismatch => "Tensor shape mismatch with model config",
            Self::SemanticDrift => "Semantic drift - structurally valid but wrong output",
            Self::Unknown => "Unknown conversion bug - requires investigation",
        }
    }
}

/// Tensor naming convention
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorNaming {
    /// HuggingFace convention (e.g., model.layers.0.self_attn.q_proj.weight)
    HuggingFace,
    /// GGUF convention (e.g., blk.0.attn_q.weight)
    Gguf,
    /// APR convention
    Apr,
    /// Unknown naming convention
    Unknown(String),
}

/// Quantization type for tolerance selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantType {
    /// Full precision 32-bit float
    F32,
    /// Half precision 16-bit float
    F16,
    /// Brain floating point 16-bit
    BF16,
    /// 4-bit K-quant medium
    Q4KM,
    /// 6-bit K-quant
    Q6K,
    /// 5-bit K-quant medium
    Q5KM,
    /// 4-bit quantization (legacy)
    Q4_0,
    /// 8-bit quantization
    Q8_0,
    /// Unknown quantization type
    Unknown,
}

impl QuantType {
    /// Parse quantization type from a string label
    #[must_use]
    pub fn from_str_label(label: &str) -> Self {
        match label.to_lowercase().replace('-', "_").as_str() {
            "f32" | "fp32" | "float32" => Self::F32,
            "f16" | "fp16" | "float16" => Self::F16,
            "bf16" | "bfloat16" => Self::BF16,
            "q4_k_m" | "q4km" => Self::Q4KM,
            "q5_k_m" | "q5km" => Self::Q5KM,
            "q6_k" | "q6k" => Self::Q6K,
            "q4_0" | "q40" => Self::Q4_0,
            "q8_0" | "q80" => Self::Q8_0,
            _ => Self::Unknown,
        }
    }
}

/// Typed conversion failure classification (§3.4)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversionFailureType {
    /// Tensor names differ between source and target
    TensorNameMismatch,
    /// Dequantization produced incorrect values
    DequantizationFailure,
    /// Config metadata (hidden_size, num_layers) doesn't match
    ConfigMetadataMismatch,
    /// Required artifact (config.json, tokenizer) is missing
    MissingArtifact,
    /// Inference failed after conversion
    InferenceFailure,
    /// Unknown failure type
    Unknown,
}

impl ConversionFailureType {
    /// Get the gate ID for this failure type
    #[must_use]
    pub fn gate_id(&self) -> &'static str {
        match self {
            Self::TensorNameMismatch => "F-CONV-TNAME-001",
            Self::DequantizationFailure => "F-CONV-DEQUANT-001",
            Self::ConfigMetadataMismatch => "F-CONV-CONFIG-001",
            Self::MissingArtifact => "F-CONV-MISSING-001",
            Self::InferenceFailure => "F-CONV-INFER-001",
            Self::Unknown => "F-CONV-UNKNOWN-002",
        }
    }

    /// Get a human-readable key for defect mapping
    #[must_use]
    pub fn key(&self) -> &'static str {
        match self {
            Self::TensorNameMismatch => "tensor_name_mismatch",
            Self::DequantizationFailure => "dequantization_failure",
            Self::ConfigMetadataMismatch => "config_metadata_mismatch",
            Self::MissingArtifact => "missing_artifact",
            Self::InferenceFailure => "inference_failure",
            Self::Unknown => "unknown",
        }
    }
}

/// Tolerance configuration for a specific quantization type (§3.7)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionTolerance {
    /// Quantization type this tolerance applies to
    pub quant_type: QuantType,
    /// Absolute tolerance
    pub atol: f64,
    /// Relative tolerance
    pub rtol: f64,
    /// Expected pygmy fixture name (for defect mapping)
    pub expected_pygmy_fixture: String,
}

include!("conversion_tolerances.rs");
include!("semantic_conversion_test.rs");
include!("conversion_strategies.rs");

#[cfg(test)]
mod pmat743_converted_path_tests {
    use super::converted_output_path;
    use std::path::Path;

    #[test]
    fn first_conversion_appends_converted_once() {
        let out = converted_output_path(Path::new("/m/qwen-q4_k_m.gguf"), "safetensors");
        assert_eq!(out, Path::new("/m/qwen-q4_k_m.converted.safetensors"));
    }

    #[test]
    fn reconverting_an_artifact_does_not_compound() {
        // The bug: with_extension on `*.converted.safetensors` produced
        // `*.converted.converted.safetensors`. Must stay single `.converted`.
        let out = converted_output_path(Path::new("/m/qwen-q4_k_m.converted.safetensors"), "apr");
        assert_eq!(out, Path::new("/m/qwen-q4_k_m.converted.apr"));
    }

    #[test]
    fn deeply_compounded_artifact_is_normalized() {
        let out = converted_output_path(
            Path::new("/m/qwen.converted.converted.converted.safetensors"),
            "safetensors",
        );
        assert_eq!(out, Path::new("/m/qwen.converted.safetensors"));
    }

    #[test]
    fn no_extension_source_is_handled() {
        let out = converted_output_path(Path::new("/m/model"), "gguf");
        assert_eq!(out, Path::new("/m/model.converted.gguf"));
    }
}
