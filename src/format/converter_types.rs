//! APR Converter Types
//!
//! Type definitions extracted from converter.rs for better modularity.
//! Part of PMAT-197: File size reduction.

use crate::error::{AprenderError, Result};
use crate::format::validation::TensorStats;
use crate::format::Compression;
use std::collections::BTreeMap;
use std::path::PathBuf;

// ============================================================================
// Source Parsing
// ============================================================================

/// Parsed source location
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    /// `HuggingFace` Hub: <hf://org/repo> or <hf://org/repo/file.safetensors>
    HuggingFace {
        org: String,
        repo: String,
        file: Option<String>,
    },
    /// Local file path
    Local(PathBuf),
    /// HTTP/HTTPS URL
    Url(String),
}

impl Source {
    /// Parse a source string into a Source enum
    pub fn parse(source: &str) -> Result<Self> {
        if source.starts_with("hf://") {
            Self::parse_hf(source)
        } else if source.starts_with("http://") || source.starts_with("https://") {
            Ok(Self::Url(source.to_string()))
        } else {
            Ok(Self::Local(PathBuf::from(source)))
        }
    }

    fn parse_hf(source: &str) -> Result<Self> {
        let path = source.strip_prefix("hf://").unwrap_or(source);
        let parts: Vec<&str> = path.split('/').collect();

        if parts.len() < 2 {
            return Err(AprenderError::FormatError {
                message: format!("Invalid HuggingFace source: {source}. Expected hf://org/repo"),
            });
        }

        let org = parts[0].to_string();
        let repo = parts[1].to_string();
        let file = if parts.len() > 2 {
            let joined = parts[2..].join("/");
            // GH-221: Strip HuggingFace web URL path components.
            // Users copy URLs like hf://org/repo/resolve/main/file.safetensors
            // or hf://org/repo/blob/main/file.safetensors from the browser.
            let cleaned = joined
                .strip_prefix("resolve/main/")
                .or_else(|| joined.strip_prefix("blob/main/"))
                .unwrap_or(&joined);
            // Also handle bare "resolve/main" or "blob/main" (no trailing slash, no file)
            let cleaned = if cleaned == "resolve/main" || cleaned == "blob/main" {
                ""
            } else {
                cleaned
            };
            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned.to_string())
            }
        } else {
            None
        };

        Ok(Self::HuggingFace { org, repo, file })
    }

    /// Get the default model file for this source
    #[must_use]
    pub fn default_file(&self) -> &str {
        match self {
            Self::HuggingFace { file: Some(f), .. } => f,
            Self::HuggingFace { file: None, .. } => "model.safetensors",
            Self::Local(p) => p.to_str().unwrap_or("model.safetensors"),
            Self::Url(u) => u.rsplit('/').next().unwrap_or("model.safetensors"),
        }
    }
}

// ============================================================================
// Architecture / Name Mapping
// ============================================================================

/// Model architecture for tensor name mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Architecture {
    /// Auto-detect from tensor names
    #[default]
    Auto,
    /// `OpenAI` Whisper
    Whisper,
    /// Meta `LLaMA` (also SmolLM2, Granite, Nemotron derivatives)
    Llama,
    /// Google BERT
    Bert,
    /// Alibaba Qwen2 (includes Qwen2.5, `QwenCoder`)
    Qwen2,
    /// Alibaba Qwen3
    Qwen3,
    /// `OpenAI` GPT-2
    Gpt2,
    /// Microsoft Phi (Phi-3, Phi-4)
    Phi,
}

impl Architecture {
    /// Map a source tensor name to APR canonical name
    #[must_use]
    pub fn map_name(&self, source_name: &str) -> String {
        match self {
            Self::Auto => Self::auto_map_name(source_name),
            Self::Whisper => Self::whisper_map_name(source_name),
            Self::Llama => Self::llama_map_name(source_name),
            Self::Bert => Self::bert_map_name(source_name),
            Self::Qwen2 => Self::qwen2_map_name(source_name),
            Self::Qwen3 => Self::qwen2_map_name(source_name), // Qwen3 uses same GGUF naming as Qwen2
            Self::Gpt2 => Self::gpt2_map_name(source_name),
            Self::Phi => Self::llama_map_name(source_name), // Phi uses HuggingFace model.layers naming
        }
    }

    /// PMAT-224: Check if this architecture has verified inference support.
    ///
    /// Returns true only for architectures with tested tensor name mapping
    /// and confirmed realizar inference compatibility.
    #[must_use]
    pub fn is_inference_verified(&self) -> bool {
        matches!(self, Self::Qwen2 | Self::Qwen3 | Self::Llama | Self::Phi)
    }

    /// PMAT-224: Get a human-readable name for warning messages.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Auto => "auto-detected",
            Self::Whisper => "Whisper",
            Self::Llama => "LLaMA",
            Self::Bert => "BERT",
            Self::Qwen2 => "Qwen2",
            Self::Qwen3 => "Qwen3",
            Self::Gpt2 => "GPT-2",
            Self::Phi => "Phi",
        }
    }

    /// Parse a `model_type` string (from config.json or GGUF metadata) into an Architecture.
    ///
    /// Returns None for unrecognized types. Centralizes the mapping used by
    /// `infer_architecture()` (import.rs) and `detect_gguf_architecture()` (export.rs).
    #[must_use]
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type.to_lowercase().as_str() {
            "qwen2" | "qwen" | "qwen2.5" => Some(Self::Qwen2),
            "qwen3" => Some(Self::Qwen3),
            "llama" | "llama2" | "llama3" => Some(Self::Llama),
            "whisper" => Some(Self::Whisper),
            "bert" => Some(Self::Bert),
            "gpt2" => Some(Self::Gpt2),
            "phi" | "phi3" | "phi4" => Some(Self::Phi),
            // LLaMA derivatives
            "smollm" | "smollm2" | "granite" | "granite3" | "nemotron" | "mistral" | "gemma"
            | "gemma2" | "gemma3" => Some(Self::Llama),
            _ => None,
        }
    }

    fn auto_map_name(name: &str) -> String {
        // PMAT-099: Preserve original tensor names for AprTransformer compatibility
        // AprTransformer::from_apr_bytes expects model.* prefixes for HuggingFace models
        name.to_string()
    }

    fn whisper_map_name(name: &str) -> String {
        // PMAT-099: Preserve model. prefix for Whisper
        name.to_string()
    }

    fn llama_map_name(name: &str) -> String {
        // PMAT-099: Preserve model. prefix for LLaMA
        name.to_string()
    }

    fn bert_map_name(name: &str) -> String {
        // BERT uses "bert." prefix - preserve it
        name.to_string()
    }

    fn qwen2_map_name(name: &str) -> String {
        // PMAT-205 FIX (GH-190): Map GGUF tensor names to APR canonical format.
        // APR uses BARE names WITHOUT "model." prefix to match the Qwen2 loader
        // contract (models/qwen2/mod.rs:1046-1131).
        //
        // GGUF: blk.N.attn_q.weight → APR: layers.N.self_attn.q_proj.weight
        //
        // PMAT-113 originally added "model." prefix, but the loader expects bare
        // names. This mismatch caused GH-190: 196 tensors unfindable → garbage.

        // Handle layer-specific tensors (blk.N.*)
        if let Some(rest) = name.strip_prefix("blk.") {
            if let Some(dot_pos) = rest.find('.') {
                let layer_num = &rest[..dot_pos];
                let suffix = &rest[dot_pos + 1..];

                // Map GGUF tensor suffixes to APR canonical names
                let apr_suffix = match suffix {
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_q.bias" => "self_attn.q_proj.bias",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_k.bias" => "self_attn.k_proj.bias",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_v.bias" => "self_attn.v_proj.bias",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "attn_output.bias" => "self_attn.o_proj.bias",
                    "attn_norm.weight" => "input_layernorm.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    "ffn_norm.weight" => "post_attention_layernorm.weight",
                    other => other, // Preserve unknown suffixes
                };

                // PMAT-222 FIX: Add "model." prefix to match SafeTensors convention
                // GH-190 was wrong - realizar DOES expect "model.layers.N.suffix"
                return format!("model.layers.{layer_num}.{apr_suffix}");
            }
        }

        // PMAT-222 FIX: Handle non-layer tensors with "model." prefix to match SafeTensors
        // Realizar's AprTransformer looks for "model.embed_tokens.weight" not "embed_tokens.weight"
        match name {
            "token_embd.weight" => "model.embed_tokens.weight".to_string(),
            "output.weight" => "lm_head.weight".to_string(),
            "output_norm.weight" => "model.norm.weight".to_string(),
            _ => name.to_string(), // Preserve unknown names
        }
    }

    /// GH-233: Map GPT-2 tensor names to APR canonical format.
    ///
    /// GPT-2 uses `transformer.h.N.*` naming. The fused `c_attn` tensor is
    /// preserved here and split by `split_gpt2_fused_qkv()` after mapping.
    fn gpt2_map_name(name: &str) -> String {
        // Handle layer-specific tensors (transformer.h.N.*)
        if let Some(rest) = name.strip_prefix("transformer.h.") {
            if let Some(dot_pos) = rest.find('.') {
                let layer_num = &rest[..dot_pos];
                let suffix = &rest[dot_pos + 1..];

                let apr_suffix = match suffix {
                    "ln_1.weight" => "input_layernorm.weight",
                    "ln_1.bias" => "input_layernorm.bias",
                    "ln_2.weight" => "post_attention_layernorm.weight",
                    "ln_2.bias" => "post_attention_layernorm.bias",
                    "attn.c_attn.weight" => "self_attn.c_attn.weight",
                    "attn.c_attn.bias" => "self_attn.c_attn.bias",
                    "attn.c_proj.weight" => "self_attn.o_proj.weight",
                    "attn.c_proj.bias" => "self_attn.o_proj.bias",
                    "mlp.c_fc.weight" => "mlp.up_proj.weight",
                    "mlp.c_fc.bias" => "mlp.up_proj.bias",
                    "mlp.c_proj.weight" => "mlp.down_proj.weight",
                    "mlp.c_proj.bias" => "mlp.down_proj.bias",
                    other => other,
                };

                return format!("model.layers.{layer_num}.{apr_suffix}");
            }
        }

        // Non-layer tensors
        match name {
            "wte.weight" => "model.embed_tokens.weight".to_string(),
            "wpe.weight" => "model.position_embedding.weight".to_string(),
            "ln_f.weight" => "model.norm.weight".to_string(),
            "ln_f.bias" => "model.norm.bias".to_string(),
            _ => name.to_string(),
        }
    }

    /// GH-233: Split GPT-2 fused QKV tensors into separate Q, K, V projections.
    ///
    /// GPT-2's `c_attn` has shape `[3*hidden, hidden]` — split dim 0 into 3 equal parts.
    /// Call this AFTER `map_tensor_names()` when architecture is `Gpt2`.
    pub fn split_gpt2_fused_qkv(tensors: &mut BTreeMap<String, (Vec<f32>, Vec<usize>)>) {
        // Collect fused c_attn tensor names
        let fused_keys: Vec<String> = tensors
            .keys()
            .filter(|k| k.contains("self_attn.c_attn."))
            .cloned()
            .collect();

        for fused_name in fused_keys {
            let (data, shape) = match tensors.remove(&fused_name) {
                Some(v) => v,
                None => continue,
            };

            let is_bias = fused_name
                .rsplit_once('.')
                .is_some_and(|(_, ext)| ext.eq_ignore_ascii_case("bias"));

            if is_bias {
                // Bias: 1D tensor of shape [3*hidden] — split into 3 equal parts
                if data.len() % 3 != 0 {
                    // Can't split evenly, put it back
                    tensors.insert(fused_name, (data, shape));
                    continue;
                }
                let chunk = data.len() / 3;
                let base = fused_name.replace("self_attn.c_attn.bias", "");

                tensors.insert(
                    format!("{base}self_attn.q_proj.bias"),
                    (data[..chunk].to_vec(), vec![chunk]),
                );
                tensors.insert(
                    format!("{base}self_attn.k_proj.bias"),
                    (data[chunk..2 * chunk].to_vec(), vec![chunk]),
                );
                tensors.insert(
                    format!("{base}self_attn.v_proj.bias"),
                    (data[2 * chunk..].to_vec(), vec![chunk]),
                );
            } else {
                // Weight: 2D tensor of shape [3*hidden, hidden] — split dim 0
                if shape.len() != 2 || shape[0] % 3 != 0 {
                    tensors.insert(fused_name, (data, shape));
                    continue;
                }
                let rows_per_proj = shape[0] / 3;
                let cols = shape[1];
                let chunk = rows_per_proj * cols;
                let base = fused_name.replace("self_attn.c_attn.weight", "");

                tensors.insert(
                    format!("{base}self_attn.q_proj.weight"),
                    (data[..chunk].to_vec(), vec![rows_per_proj, cols]),
                );
                tensors.insert(
                    format!("{base}self_attn.k_proj.weight"),
                    (data[chunk..2 * chunk].to_vec(), vec![rows_per_proj, cols]),
                );
                tensors.insert(
                    format!("{base}self_attn.v_proj.weight"),
                    (data[2 * chunk..].to_vec(), vec![rows_per_proj, cols]),
                );
            }

            eprintln!(
                "[GH-233] Split fused c_attn tensor: {} → q_proj + k_proj + v_proj",
                fused_name
            );
        }
    }
}

// ============================================================================
// Tensor Expectations
// ============================================================================

/// Expected statistics for a tensor type
#[derive(Debug, Clone)]
pub struct TensorExpectation {
    /// Expected mean range (min, max)
    pub mean_range: (f32, f32),
    /// Expected std range (min, max)
    pub std_range: Option<(f32, f32)>,
    /// Description for error messages
    pub description: &'static str,
}

impl TensorExpectation {
    /// `LayerNorm` weight: gamma initialized to ~1.0
    pub const LAYER_NORM_WEIGHT: Self = Self {
        mean_range: (0.5, 3.0),
        std_range: Some((0.0, 2.0)),
        description: "LayerNorm weight (gamma)",
    };

    /// `LayerNorm` bias: beta initialized to ~0.0
    pub const LAYER_NORM_BIAS: Self = Self {
        mean_range: (-0.5, 0.5),
        std_range: Some((0.0, 1.0)),
        description: "LayerNorm bias (beta)",
    };

    /// Linear/Attention weight: Xavier/He initialized, mean ~0
    pub const LINEAR_WEIGHT: Self = Self {
        mean_range: (-0.1, 0.1),
        std_range: None,
        description: "Linear/Attention weight",
    };

    /// Embedding weight: varies by initialization
    pub const EMBEDDING: Self = Self {
        mean_range: (-1.0, 1.0),
        std_range: None,
        description: "Embedding",
    };

    /// `RMSNorm` weight: gamma initialized to ~1.0 but varies after training
    /// Trained models show means from ~0.0 to ~2.0 (`TinyLlama`: 0.005-0.5)
    /// Issue #46: Qwen2.5-Coder-0.5B has mean=7.23 and std=2.11, widen ranges
    pub const RMSNORM_WEIGHT: Self = Self {
        mean_range: (-1.0, 10.0),    // Wider range for Qwen and other architectures
        std_range: Some((0.0, 5.0)), // Qwen has std up to 2.11
        description: "RMSNorm weight (gamma)",
    };

    /// Get expectation for a tensor name
    #[must_use]
    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    pub fn for_tensor(name: &str) -> Option<Self> {
        // RMSNorm patterns (LLaMA, Qwen2, TinyLlama, GGUF) - check BEFORE generic LayerNorm
        // These use gamma initialized to 1.0, not the 0-centered LayerNorm
        // Fix #163: Also match GGUF attn_norm/ffn_norm patterns
        if (name.contains("input_layernorm")
            || name.contains("post_attention_layernorm")
            || name.contains("rms_norm")
            || name.contains("attn_norm") // GGUF pattern (blk.N.attn_norm.weight)
            || name.contains("ffn_norm")) // GGUF pattern (blk.N.ffn_norm.weight)
            && name.ends_with(".weight")
        {
            return Some(Self::RMSNORM_WEIGHT);
        }

        // Traditional LayerNorm patterns (BERT, older models)
        if name.contains("layer_norm") || name.contains("ln_") {
            if name.ends_with(".weight") || name.ends_with(".gamma") {
                return Some(Self::LAYER_NORM_WEIGHT);
            }
            if name.ends_with(".bias") || name.ends_with(".beta") {
                return Some(Self::LAYER_NORM_BIAS);
            }
        }

        // Final norm layer (often RMSNorm in modern LLMs)
        if name == "norm.weight" || name.ends_with(".norm.weight") {
            return Some(Self::RMSNORM_WEIGHT);
        }

        if name.contains("embed") {
            return Some(Self::EMBEDDING);
        }

        if name.ends_with(".weight") {
            return Some(Self::LINEAR_WEIGHT);
        }

        None
    }

    /// Check if stats match expectation
    pub fn check(&self, stats: &TensorStats) -> Result<()> {
        let (min_mean, max_mean) = self.mean_range;

        if stats.mean < min_mean || stats.mean > max_mean {
            return Err(AprenderError::FormatError {
                message: format!(
                    "{}: mean={:.4} outside expected range [{:.1}, {:.1}]",
                    self.description, stats.mean, min_mean, max_mean
                ),
            });
        }

        if let Some((min_std, max_std)) = self.std_range {
            if stats.std < min_std || stats.std > max_std {
                return Err(AprenderError::FormatError {
                    message: format!(
                        "{}: std={:.4} outside expected range [{:.1}, {:.1}]",
                        self.description, stats.std, min_std, max_std
                    ),
                });
            }
        }

        Ok(())
    }
}

// ============================================================================
// Validation Config
// ============================================================================

/// Validation strictness configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationConfig {
    /// No validation
    None,
    /// Basic checks (NaN, Inf only)
    Basic,
    /// Full statistical validation
    Strict,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self::Strict
    }
}

impl ValidationConfig {
    /// Create strict validation config
    #[must_use]
    pub fn strict() -> Self {
        Self::Strict
    }
}

// ============================================================================
// Import Options
// ============================================================================

/// Quantization type for import pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// 8-bit integer quantization
    Int8,
    /// 4-bit integer quantization
    Int4,
    /// 16-bit float
    Fp16,
    /// GGUF Q4_K format (K-quants, ~4.5 bits/weight)
    /// 256-element super-blocks with nested 32-element sub-blocks
    /// Achieves ~7x memory bandwidth improvement over F32
    Q4K,
}

// ============================================================================
// DOUBLE-QUANT-001: Compile-Time Double Quantization Prevention
// ============================================================================

/// F32 tensors that were NATIVELY F32 (e.g., SafeTensors F32/BF16 sources).
///
/// This is the ONLY type that `quantize_tensors()` accepts.
/// Constructed from sources known to be unquantized (SafeTensors, F32 APR files).
///
/// There is intentionally NO `From<DequantizedTensors> for NativeF32Tensors` —
/// attempting to pass dequantized tensors to `quantize_tensors()` is a compile error.
#[derive(Debug)]
pub struct NativeF32Tensors(BTreeMap<String, (Vec<f32>, Vec<usize>)>);

impl NativeF32Tensors {
    /// Wrap a tensor map as natively F32.
    #[must_use]
    pub fn new(map: BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> Self {
        Self(map)
    }

    /// Consume the wrapper and return the inner map.
    #[must_use]
    pub fn into_inner(self) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        self.0
    }
}

impl AsRef<BTreeMap<String, (Vec<f32>, Vec<usize>)>> for NativeF32Tensors {
    fn as_ref(&self) -> &BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        &self.0
    }
}

/// F32 tensors that were DEQUANTIZED from a quantized format (Q4K, Q6K, etc.).
///
/// CANNOT be passed to `quantize_tensors()` — compile error.
/// Re-quantizing dequantized data is a lossy double quantization that destroys
/// weight fidelity (PMAT-252).
#[derive(Debug)]
pub struct DequantizedTensors {
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    /// The original quantization format these tensors came from.
    pub original_quant: QuantizationType,
}

impl DequantizedTensors {
    /// Wrap a tensor map as dequantized from the given quantization type.
    #[must_use]
    pub fn new(map: BTreeMap<String, (Vec<f32>, Vec<usize>)>, quant: QuantizationType) -> Self {
        Self {
            tensors: map,
            original_quant: quant,
        }
    }

    /// Consume the wrapper and return the inner map.
    #[must_use]
    pub fn into_inner(self) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        self.tensors
    }
}

impl AsRef<BTreeMap<String, (Vec<f32>, Vec<usize>)>> for DequantizedTensors {
    fn as_ref(&self) -> &BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        &self.tensors
    }
}

/// Tensor provenance: either natively F32 or dequantized from quantized format.
///
/// Used by export/convert pipelines to enforce compile-time safety against
/// double quantization (DOUBLE-QUANT-001).
#[derive(Debug)]
pub enum TensorProvenance {
    /// Tensors that are natively F32 — safe to quantize.
    Native(NativeF32Tensors),
    /// Tensors dequantized from a quantized format — must NOT be re-quantized.
    Dequantized(DequantizedTensors),
}

impl TensorProvenance {
    /// Get a read-only reference to the underlying tensor map.
    #[must_use]
    pub fn as_map(&self) -> &BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        match self {
            Self::Native(n) => n.as_ref(),
            Self::Dequantized(d) => d.as_ref(),
        }
    }

    /// Consume the provenance wrapper and return the inner map.
    #[must_use]
    pub fn into_map(self) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        match self {
            Self::Native(n) => n.into_inner(),
            Self::Dequantized(d) => d.into_inner(),
        }
    }
}

// ============================================================================

/// Options for the import pipeline
#[derive(Debug, Clone)]
pub struct ImportOptions {
    /// Target architecture for name mapping
    pub architecture: Architecture,
    /// Validation configuration
    pub validation: ValidationConfig,
    /// Quantization (None = keep original precision)
    pub quantize: Option<QuantizationType>,
    /// Compression algorithm
    pub compress: Option<Compression>,
    /// GH-196: Strict mode rejects unverified architectures.
    /// Default is permissive (imports proceed with a warning).
    /// Use `--strict` on CLI to enable this.
    pub strict: bool,
    /// Cache downloaded files
    pub cache: bool,
    /// PMAT-232: External tokenizer.json path for weights-only GGUF files.
    /// If the GGUF has no embedded tokenizer, this file will be used instead.
    pub tokenizer_path: Option<std::path::PathBuf>,
    /// GH-223: Allow import without config.json. By default, SafeTensors import
    /// errors when config.json is missing (inferred hyperparameters like rope_theta
    /// may be wrong). Pass `--allow-no-config` to proceed with warnings only.
    pub allow_no_config: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            architecture: Architecture::Auto,
            validation: ValidationConfig::Strict,
            quantize: None,
            compress: None,
            strict: false,
            cache: true,
            tokenizer_path: None,
            allow_no_config: false,
        }
    }
}

// ============================================================================
// Import Error
// ============================================================================

/// Import-specific errors (GH-129: actionable error messages)
#[derive(Debug, Clone)]
pub enum ImportError {
    /// Download failed
    DownloadFailed { source: String, reason: String },
    /// Unsupported format
    UnsupportedFormat { extension: String },
    /// Tensor validation failed
    ValidationFailed { name: String, reason: String },
    /// Unknown tensor name
    UnknownTensor { source_name: String },
    /// Missing required tensor
    MissingTensor { name: String },
    /// Resource not found (404)
    NotFound { resource: String, status: u16 },
    /// Rate limited by server
    RateLimited { retry_after: Option<u64> },
    /// Authentication required (gated model)
    AuthRequired { resource: String },
    /// Model requires sharded loading (GH-127)
    ShardingRequired { model_size: u64, shard_count: usize },
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DownloadFailed { source, reason } => {
                write!(f, "Download failed: {source} - {reason}")
            }
            Self::UnsupportedFormat { extension } => {
                write!(f, "Unsupported format: {extension}")
            }
            Self::ValidationFailed { name, reason } => {
                write!(f, "Tensor validation failed: {name} - {reason}")
            }
            Self::UnknownTensor { source_name } => {
                write!(f, "Unknown tensor: {source_name}")
            }
            Self::MissingTensor { name } => {
                write!(f, "Missing required tensor: {name}")
            }
            // GH-129: Actionable error messages
            Self::NotFound { resource, status } => {
                write!(
                    f,
                    "Resource not found ({status}): {resource}. \
                     Fix: verify the model name exists on huggingface.co/models"
                )
            }
            Self::RateLimited { retry_after } => {
                if let Some(secs) = retry_after {
                    write!(
                        f,
                        "Rate limited by server. Retry after {secs} seconds. \
                         Fix: wait and retry, or use --cache to avoid re-downloads"
                    )
                } else {
                    write!(
                        f,
                        "Rate limited by server. \
                         Fix: wait a few minutes and retry"
                    )
                }
            }
            Self::AuthRequired { resource } => {
                write!(
                    f,
                    "Authentication required for {resource}. \
                     Fix: set HF_TOKEN environment variable with your HuggingFace token"
                )
            }
            Self::ShardingRequired {
                model_size,
                shard_count,
            } => {
                let size_gb = *model_size as f64 / 1_000_000_000.0;
                write!(
                    f,
                    "Model too large ({size_gb:.1} GB, {shard_count} shards) for single-file loading. \
                     Fix: use streaming import with --sharded flag"
                )
            }
        }
    }
}

impl std::error::Error for ImportError {}

impl From<ImportError> for AprenderError {
    fn from(err: ImportError) -> Self {
        AprenderError::FormatError {
            message: err.to_string(),
        }
    }
}

/// Parse error message to detect specific error types (GH-129)
#[cfg(feature = "hf-hub-integration")]
pub fn parse_import_error(error_msg: &str, resource: &str) -> ImportError {
    let msg_lower = error_msg.to_lowercase();

    // Check for 404 / not found
    if msg_lower.contains("404")
        || msg_lower.contains("not found")
        || msg_lower.contains("does not exist")
        || msg_lower.contains("no such")
    {
        return ImportError::NotFound {
            resource: resource.to_string(),
            status: 404,
        };
    }

    // Check for authentication / 401 / 403
    if msg_lower.contains("401")
        || msg_lower.contains("403")
        || msg_lower.contains("unauthorized")
        || msg_lower.contains("forbidden")
        || msg_lower.contains("gated")
        || msg_lower.contains("access denied")
    {
        return ImportError::AuthRequired {
            resource: resource.to_string(),
        };
    }

    // Check for rate limiting / 429
    if msg_lower.contains("429")
        || msg_lower.contains("rate limit")
        || msg_lower.contains("too many requests")
    {
        // Try to extract retry-after
        let retry_after = if let Some(pos) = msg_lower.find("retry") {
            msg_lower[pos..]
                .split_whitespace()
                .find_map(|s| s.parse::<u64>().ok())
        } else {
            None
        };
        return ImportError::RateLimited { retry_after };
    }

    // Default to download failed
    ImportError::DownloadFailed {
        source: resource.to_string(),
        reason: error_msg.to_string(),
    }
}

// ============================================================================
// GH-127: Sharded Model Support
// ============================================================================

/// Parsed sharded model index (model.safetensors.index.json)
///
/// `HuggingFace` uses this format for large models split across multiple shards.
/// Example: Llama-2-7b has 2 shards, Llama-2-70b has 15 shards.
#[derive(Debug, Clone)]
pub struct ShardedIndex {
    /// Map of tensor name → shard filename
    weight_map: std::collections::HashMap<String, String>,
    /// Optional total size in bytes
    total_size: Option<u64>,
}

impl ShardedIndex {
    /// Parse a sharded index from JSON string
    ///
    /// # Example JSON format
    /// ```json
    /// {
    ///   "metadata": {"total_size": 14000000000},
    ///   "weight_map": {
    ///     "model.encoder.weight": "model-00001-of-00002.safetensors",
    ///     "model.decoder.weight": "model-00002-of-00002.safetensors"
    ///   }
    /// }
    /// ```
    pub fn parse(json: &str) -> Result<Self> {
        // Minimal JSON parsing without serde dependency
        // Look for "weight_map" key and parse the object

        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return Err(AprenderError::FormatError {
                message: "Invalid JSON: expected object".to_string(),
            });
        }

        // Find weight_map section
        let weight_map_start =
            json.find("\"weight_map\"")
                .ok_or_else(|| AprenderError::FormatError {
                    message: "Missing 'weight_map' key in index.json".to_string(),
                })?;

        // Parse weight_map object
        let after_key = &json[weight_map_start + 12..]; // Skip "weight_map"
        let obj_start = after_key
            .find('{')
            .ok_or_else(|| AprenderError::FormatError {
                message: "Invalid weight_map: expected object".to_string(),
            })?;

        let obj_content = &after_key[obj_start..];
        let mut weight_map = std::collections::HashMap::new();
        let mut depth = 0;
        let mut obj_end = 0;

        for (i, c) in obj_content.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        obj_end = i;
                        break;
                    }
                }
                _ => {}
            }
        }

        let inner = &obj_content[1..obj_end];

        // Parse key-value pairs: "tensor_name": "shard_file"
        for pair in inner.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }

            let parts: Vec<&str> = pair.splitn(2, ':').collect();
            if parts.len() == 2 {
                let key = parts[0].trim().trim_matches('"');
                let val = parts[1].trim().trim_matches('"');
                if !key.is_empty() && !val.is_empty() {
                    weight_map.insert(key.to_string(), val.to_string());
                }
            }
        }

        // Parse optional total_size from metadata
        let total_size = json.find("\"total_size\"").and_then(|pos| {
            let after = &json[pos + 12..];
            let colon = after.find(':')?;
            let after_colon = after[colon + 1..].trim_start();
            let end = after_colon.find(|c: char| !c.is_ascii_digit())?;
            after_colon[..end].parse::<u64>().ok()
        });

        Ok(Self {
            weight_map,
            total_size,
        })
    }

    /// Number of unique shard files
    #[must_use]
    pub fn shard_count(&self) -> usize {
        let unique: std::collections::HashSet<_> = self.weight_map.values().collect();
        unique.len()
    }

    /// Number of tensors in the index
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    /// Total model size in bytes (if available)
    #[must_use]
    pub fn total_size(&self) -> Option<u64> {
        self.total_size
    }

    /// Get the shard file containing a specific tensor
    #[must_use]
    pub fn shard_for_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }

    /// Get all tensor names in a specific shard
    #[must_use]
    pub fn tensors_in_shard(&self, shard_file: &str) -> Vec<&str> {
        self.weight_map
            .iter()
            .filter(|(_, v)| v.as_str() == shard_file)
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Get sorted list of shard files
    #[must_use]
    pub fn shard_files(&self) -> Vec<&str> {
        let mut files: Vec<_> = self
            .weight_map
            .values()
            .map(String::as_str)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        files.sort_unstable();
        files
    }
}

/// Detect if a model directory contains a sharded model
///
/// Checks for `model.safetensors.index.json` which indicates sharding.
#[must_use]
pub fn detect_sharded_model(dir: &std::path::Path, base_name: &str) -> Option<PathBuf> {
    let index_name = format!("{base_name}.index.json");
    let index_path = dir.join(&index_name);

    if index_path.exists() {
        Some(index_path)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_parse_resolve_main_stripped() {
        // GH-221: Users copy URLs with /resolve/main/ from HuggingFace browser
        let src = Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors")
            .expect("should parse");
        match src {
            Source::HuggingFace { org, repo, file } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-1.5B-Instruct");
                assert_eq!(file, Some("model.safetensors".to_string()));
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_blob_main_stripped() {
        let src = Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct/blob/main/model.safetensors")
            .expect("should parse");
        match src {
            Source::HuggingFace { org, repo, file } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-1.5B-Instruct");
                assert_eq!(file, Some("model.safetensors".to_string()));
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_direct_file_unchanged() {
        let src = Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct/model.safetensors")
            .expect("should parse");
        match src {
            Source::HuggingFace { org, repo, file } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-1.5B-Instruct");
                assert_eq!(file, Some("model.safetensors".to_string()));
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_no_file() {
        let src = Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct").expect("should parse");
        match src {
            Source::HuggingFace { org, repo, file } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-1.5B-Instruct");
                assert_eq!(file, None);
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_nested_path_preserved() {
        // Paths that don't start with resolve/main/ or blob/main/ are preserved
        let src = Source::parse("hf://org/repo/subdir/model.safetensors").expect("should parse");
        match src {
            Source::HuggingFace { file, .. } => {
                assert_eq!(file, Some("subdir/model.safetensors".to_string()));
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_resolve_main_no_trailing_slash() {
        // Edge case: hf://org/repo/resolve/main (no file, no trailing slash)
        let src =
            Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct/resolve/main").expect("should parse");
        match src {
            Source::HuggingFace { org, repo, file } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-1.5B-Instruct");
                assert_eq!(file, None);
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_blob_main_no_trailing_slash() {
        let src = Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct/blob/main").expect("should parse");
        match src {
            Source::HuggingFace { file, .. } => {
                assert_eq!(file, None);
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_hf_parse_resolve_main_nested_file() {
        // resolve/main/ with nested subdir path
        let src =
            Source::parse("hf://Qwen/Qwen2.5-1.5B-Instruct/resolve/main/sub/model.safetensors")
                .expect("should parse");
        match src {
            Source::HuggingFace { file, .. } => {
                assert_eq!(file, Some("sub/model.safetensors".to_string()));
            }
            _ => panic!("expected HuggingFace variant"),
        }
    }
}
