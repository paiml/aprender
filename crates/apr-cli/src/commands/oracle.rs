//! Oracle command implementation (PMAT-244..247)
//!
//! Model Oracle: identifies model family, size variant, constraints, and contract
//! compliance from local files, HuggingFace URIs, or contract descriptions.
//!
//! 3X Enhancement: Statistical analysis, architecture explanations, kernel
//! compatibility, and cross-validation against HuggingFace ground truth.
//!
//! Toyota Way: Genchi Genbutsu - Go to the source to understand the model.

use crate::error::CliError;
use crate::output;
use aprender::format::model_family::{
    AttentionType, FamilyRegistry, MlpType, ModelConstraints, ModelFamily, ModelFamilyConfig,
    ModelSizeConfig, NormType, PositionalEncoding,
};
use aprender::format::model_family_loader::load_family_registry;
use aprender::format::rosetta::RosettaStone;
use serde::Serialize;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

// ============================================================================
// Report Types (spec §6.5)
// ============================================================================

/// Analysis mode for the oracle
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OracleMode {
    /// Local file analysis (PMAT-244)
    Local,
    /// HuggingFace API query (PMAT-245)
    HuggingFace,
    /// Contract description (PMAT-246)
    Family,
}

/// Complete oracle report for a model (spec §6.5)
#[derive(Debug, Clone, Serialize)]
pub struct ModelOracleReport {
    /// Source path, HF URI, or family name
    pub source: String,
    /// Analysis mode
    pub mode: OracleMode,
    /// Detected or specified model family
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<FamilyInfo>,
    /// Detected size variant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_variant: Option<SizeVariantInfo>,
    /// Format information (for local files)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<FormatInfo>,
    /// Contract compliance result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compliance: Option<ComplianceResult>,
    /// Certification status
    #[serde(skip_serializing_if = "Option::is_none")]
    pub certification: Option<CertificationInfo>,
    /// Tensor list (for --tensors flag)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensors: Option<Vec<TensorComplianceEntry>>,
    /// Statistical analysis (--stats or --full)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<StatisticalAnalysis>,
    /// Architecture explanation (--explain or --full)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<ArchitectureExplanation>,
    /// Kernel compatibility report (--kernels or --full)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_compatibility: Option<KernelCompatibility>,
    /// Cross-validation against HF ground truth (--validate or --full)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cross_validation: Option<CrossValidation>,
    /// HuggingFace data (populated in HF mode)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_data: Option<HuggingFaceData>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FamilyInfo {
    pub name: String,
    pub display_name: String,
    pub vendor: String,
    pub architectures: Vec<String>,
    pub constraints: ConstraintsSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_format: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConstraintsSummary {
    pub attention: String,
    pub activation: String,
    pub norm: String,
    pub bias: bool,
    pub tied_embeddings: bool,
    pub mlp: String,
    pub positional_encoding: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SizeVariantInfo {
    pub name: String,
    pub parameters: String,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub expected_tensor_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FormatInfo {
    pub format_type: String,
    pub file_size: usize,
    pub tensor_count: usize,
    pub total_params: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComplianceResult {
    pub is_compliant: bool,
    pub tensor_count_match: bool,
    pub missing_tensors: Vec<String>,
    pub unexpected_tensors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CertificationInfo {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub playbook_path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorComplianceEntry {
    pub name: String,
    pub present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

// ============================================================================
// Phase 1: Statistical Analysis
// ============================================================================

/// Statistical analysis of model architecture (pure computation, no I/O)
#[derive(Debug, Clone, Serialize)]
pub struct StatisticalAnalysis {
    // GQA Analysis
    pub gqa_ratio: f64,
    pub kv_cache_reduction: f64,

    // Memory Estimation (inference)
    pub model_params: u64,
    pub model_size_f16_mb: f64,
    pub model_size_q4_mb: f64,
    pub kv_cache_per_token_bytes: u64,
    pub kv_cache_4k_mb: f64,

    // FFN Analysis
    pub ffn_expansion_ratio: f64,
    pub ffn_type_explanation: String,

    // RoPE Analysis
    pub rope_max_wavelength: f64,
    pub effective_context_window: usize,

    // Attention Complexity
    pub attention_flops_per_token: u64,
    pub ffn_flops_per_token: u64,
}

/// Compute GQA ratio and KV cache reduction
pub fn compute_gqa_analysis(size: &ModelSizeConfig) -> (f64, f64) {
    if size.num_heads == 0 {
        return (0.0, 0.0);
    }
    let ratio = size.num_kv_heads as f64 / size.num_heads as f64;
    let reduction = 1.0 - ratio;
    (ratio, reduction)
}

/// Compute model parameter count from architecture config
pub fn compute_param_count(size: &ModelSizeConfig, constraints: &ModelConstraints) -> u64 {
    let h = size.hidden_dim as u64;
    let v = size.vocab_size as u64;
    let l = size.num_layers as u64;
    let n_heads = size.num_heads as u64;
    let n_kv = size.num_kv_heads as u64;
    let head_d = size.head_dim as u64;
    let inter = size.intermediate_dim as u64;

    // Embedding: vocab_size * hidden_dim
    let embedding = v * h;

    // Per-layer attention: Q, K, V projections + O projection
    // Q: hidden_dim * (num_heads * head_dim)
    // K: hidden_dim * (num_kv_heads * head_dim)
    // V: hidden_dim * (num_kv_heads * head_dim)
    // O: (num_heads * head_dim) * hidden_dim
    let attn =
        h * (n_heads * head_d) + h * (n_kv * head_d) + h * (n_kv * head_d) + (n_heads * head_d) * h;

    // Bias for attention (if applicable)
    let attn_bias = if constraints.has_bias {
        (n_heads * head_d) + (n_kv * head_d) + (n_kv * head_d) + h
    } else {
        0
    };

    // FFN: gated (3 matrices) vs standard (2 matrices)
    let is_gated = matches!(constraints.mlp_type, MlpType::SwiGlu | MlpType::GatedMlp);
    let ffn = if is_gated {
        h * inter * 3 // gate_proj + up_proj + down_proj
    } else {
        h * inter * 2 // fc1 + fc2
    };

    // Norms per layer (2 norms: input + post-attention, each has hidden_dim weights)
    let norms_per_layer = h * 2;

    let per_layer = attn + attn_bias + ffn + norms_per_layer;

    // LM head: vocab_size * hidden_dim (unless tied to embedding)
    let lm_head = if constraints.tied_embeddings {
        0
    } else {
        v * h
    };

    // Final norm
    let final_norm = h;

    embedding + (per_layer * l) + lm_head + final_norm
}

/// Compute memory estimates for different precisions
pub fn compute_memory_estimates(
    size: &ModelSizeConfig,
    constraints: &ModelConstraints,
) -> (f64, f64) {
    let params = compute_param_count(size, constraints);
    let f16_mb = (params as f64 * 2.0) / (1024.0 * 1024.0);
    let q4_mb = (params as f64 * 0.5) / (1024.0 * 1024.0);
    (f16_mb, q4_mb)
}

/// Compute KV cache size per token and for 4K context
pub fn compute_kv_cache(size: &ModelSizeConfig) -> (u64, f64) {
    // KV cache per token: 2 (K+V) * num_layers * num_kv_heads * head_dim * 2 (f16 bytes)
    let per_token =
        2_u64 * size.num_layers as u64 * size.num_kv_heads as u64 * size.head_dim as u64 * 2; // f16 = 2 bytes
    let cache_4k_mb = (per_token as f64 * 4096.0) / (1024.0 * 1024.0);
    (per_token, cache_4k_mb)
}

/// Compute FFN expansion ratio and explanation
pub fn compute_ffn_analysis(
    size: &ModelSizeConfig,
    constraints: &ModelConstraints,
) -> (f64, String) {
    if size.hidden_dim == 0 {
        return (0.0, String::new());
    }
    let ratio = size.intermediate_dim as f64 / size.hidden_dim as f64;
    let explanation = match constraints.mlp_type {
        MlpType::SwiGlu => format!(
            "SwiGLU uses 2/3 of 4x expansion: FFN(x) = (W_up * x * SiLU(W_gate * x)) * W_down. \
             Ratio {ratio:.2}x with 3 weight matrices."
        ),
        MlpType::GatedMlp => format!(
            "GeGLU gated MLP: FFN(x) = (W_up * x * GELU(W_gate * x)) * W_down. \
             Ratio {ratio:.2}x with 3 weight matrices."
        ),
        MlpType::GeluMlp => format!(
            "Standard GELU MLP: FFN(x) = W2 * GELU(W1 * x). \
             Ratio {ratio:.2}x with 2 weight matrices."
        ),
    };
    (ratio, explanation)
}

/// Compute RoPE wavelength analysis
pub fn compute_rope_analysis(size: &ModelSizeConfig) -> (f64, usize) {
    let wavelength = if size.rope_theta > 0.0 {
        2.0 * std::f64::consts::PI * size.rope_theta
    } else {
        0.0
    };
    (wavelength, size.max_position_embeddings)
}

/// Compute approximate FLOPS per token
pub fn compute_flops_estimate(
    size: &ModelSizeConfig,
    constraints: &ModelConstraints,
) -> (u64, u64) {
    let h = size.hidden_dim as u64;
    let n_heads = size.num_heads as u64;
    let n_kv = size.num_kv_heads as u64;
    let head_d = size.head_dim as u64;
    let inter = size.intermediate_dim as u64;

    // Attention FLOPS per token per layer (simplified):
    // QKV projections: 2 * h * (n_heads + 2*n_kv) * head_d
    // Attention scores: 2 * n_heads * head_d (per position, amortized)
    // Output projection: 2 * n_heads * head_d * h
    let attn_flops = 2 * h * (n_heads + 2 * n_kv) * head_d + 2 * n_heads * head_d * h;

    // FFN FLOPS per token per layer:
    let is_gated = matches!(constraints.mlp_type, MlpType::SwiGlu | MlpType::GatedMlp);
    let ffn_flops = if is_gated {
        2 * h * inter * 3 // 3 matmuls
    } else {
        2 * h * inter * 2 // 2 matmuls
    };

    // Total per layer, then multiply by num_layers
    let l = size.num_layers as u64;
    (attn_flops * l, ffn_flops * l)
}

/// Build complete statistical analysis from size config and constraints
pub fn build_statistical_analysis(
    size: &ModelSizeConfig,
    constraints: &ModelConstraints,
) -> StatisticalAnalysis {
    let (gqa_ratio, kv_cache_reduction) = compute_gqa_analysis(size);
    let params = compute_param_count(size, constraints);
    let (f16_mb, q4_mb) = compute_memory_estimates(size, constraints);
    let (kv_per_token, kv_4k_mb) = compute_kv_cache(size);
    let (ffn_ratio, ffn_explanation) = compute_ffn_analysis(size, constraints);
    let (rope_wavelength, context_window) = compute_rope_analysis(size);
    let (attn_flops, ffn_flops) = compute_flops_estimate(size, constraints);

    StatisticalAnalysis {
        gqa_ratio,
        kv_cache_reduction,
        model_params: params,
        model_size_f16_mb: f16_mb,
        model_size_q4_mb: q4_mb,
        kv_cache_per_token_bytes: kv_per_token,
        kv_cache_4k_mb: kv_4k_mb,
        ffn_expansion_ratio: ffn_ratio,
        ffn_type_explanation: ffn_explanation,
        rope_max_wavelength: rope_wavelength,
        effective_context_window: context_window,
        attention_flops_per_token: attn_flops,
        ffn_flops_per_token: ffn_flops,
    }
}

// ============================================================================
// Phase 2: HuggingFace Data (ureq + serde_json)
// ============================================================================

/// Data fetched from HuggingFace API
#[derive(Debug, Clone, Serialize)]
pub struct HuggingFaceData {
    pub repo: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downloads: Option<u64>,
    pub config_fields: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<serde_json::Value>,
}

/// Fetch a URL via ureq and return body as string.
fn fetch_url(url: &str) -> Result<String, CliError> {
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(30))
        .build();
    let response = agent
        .get(url)
        .call()
        .map_err(|e| CliError::NetworkError(format!("Failed to fetch {url}: {e}")))?;

    response
        .into_string()
        .map_err(|e| CliError::NetworkError(format!("Failed to read response from {url}: {e}")))
}

/// Fetch and parse JSON from a URL, returning None on failure (non-critical).
fn fetch_json_optional(url: &str) -> Option<serde_json::Value> {
    fetch_url(url)
        .ok()
        .and_then(|body| serde_json::from_str(&body).ok())
}

/// Fetch full HuggingFace data for a repo.
fn fetch_hf_data(repo: &str) -> Result<HuggingFaceData, CliError> {
    // Fetch config.json (required)
    let config_url = format!("https://huggingface.co/{repo}/raw/main/config.json");
    let config_body = fetch_url(&config_url)?;
    let config_json: serde_json::Value = serde_json::from_str(&config_body)
        .map_err(|e| CliError::InvalidFormat(format!("Invalid config.json from {repo}: {e}")))?;

    let model_type = config_json["model_type"].as_str().map(String::from);

    // Fetch generation_config.json (optional)
    let gen_config_url = format!("https://huggingface.co/{repo}/raw/main/generation_config.json");
    let generation_config = fetch_json_optional(&gen_config_url);

    // Fetch API metadata (optional, for downloads/pipeline_tag)
    let api_url = format!("https://huggingface.co/api/models/{repo}");
    let api_data = fetch_json_optional(&api_url);

    let pipeline_tag = api_data
        .as_ref()
        .and_then(|d| d["pipeline_tag"].as_str().map(String::from));
    let downloads = api_data.as_ref().and_then(|d| d["downloads"].as_u64());

    Ok(HuggingFaceData {
        repo: repo.to_string(),
        model_type,
        pipeline_tag,
        downloads,
        config_fields: config_json,
        generation_config,
    })
}

include!("oracle_part_02.rs");
include!("oracle_part_03.rs");
include!("oracle_part_04.rs");
include!("oracle_part_05.rs");
include!("oracle_part_06.rs");
include!("oracle_part_07.rs");
