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

// ============================================================================
// Phase 3: Cross-Validation
// ============================================================================

/// Cross-validation of contract YAML vs HuggingFace config.json
#[derive(Debug, Clone, Serialize)]
pub struct CrossValidation {
    pub matches: Vec<CrossValidationEntry>,
    pub mismatches: Vec<CrossValidationEntry>,
    pub contract_only: Vec<String>,
    pub hf_only: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CrossValidationEntry {
    pub field: String,
    pub contract_value: String,
    pub hf_value: String,
    pub status: String,
}

/// Cross-validate contract size config against HF config.json
pub fn cross_validate(
    size: &ModelSizeConfig,
    constraints: &ModelConstraints,
    hf_config: &serde_json::Value,
) -> CrossValidation {
    let mut matches = Vec::new();
    let mut mismatches = Vec::new();
    let mut contract_only = Vec::new();
    let mut hf_only = Vec::new();

    type FieldMapping = (&'static str, &'static str, fn(&ModelSizeConfig) -> String);
    let field_mappings: &[FieldMapping] = &[
        ("hidden_size", "hidden_dim", |s| s.hidden_dim.to_string()),
        ("num_hidden_layers", "num_layers", |s| {
            s.num_layers.to_string()
        }),
        ("num_attention_heads", "num_heads", |s| {
            s.num_heads.to_string()
        }),
        ("num_key_value_heads", "num_kv_heads", |s| {
            s.num_kv_heads.to_string()
        }),
        ("intermediate_size", "intermediate_dim", |s| {
            s.intermediate_dim.to_string()
        }),
        ("vocab_size", "vocab_size", |s| s.vocab_size.to_string()),
        ("max_position_embeddings", "max_position_embeddings", |s| {
            s.max_position_embeddings.to_string()
        }),
    ];

    for (hf_key, contract_key, getter) in field_mappings {
        let contract_val = getter(size);
        if let Some(hf_val) = hf_config.get(hf_key) {
            let hf_str = match hf_val {
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let entry = CrossValidationEntry {
                field: (*contract_key).to_string(),
                contract_value: contract_val.clone(),
                hf_value: hf_str.clone(),
                status: if contract_val == hf_str {
                    "match".to_string()
                } else {
                    "mismatch".to_string()
                },
            };
            if contract_val == hf_str {
                matches.push(entry);
            } else {
                mismatches.push(entry);
            }
        } else {
            contract_only.push(format!("{contract_key}={contract_val}"));
        }
    }

    cross_validate_rope_theta(size, hf_config, &mut matches, &mut mismatches);
    cross_validate_norm_eps(size, hf_config, &mut matches, &mut mismatches);
    cross_validate_model_type(constraints, hf_config, &mut matches);
    collect_untracked_hf_fields(hf_config, &mut hf_only);

    CrossValidation {
        matches,
        mismatches,
        contract_only,
        hf_only,
    }
}

fn cross_validate_rope_theta(
    size: &ModelSizeConfig,
    hf_config: &serde_json::Value,
    matches: &mut Vec<CrossValidationEntry>,
    mismatches: &mut Vec<CrossValidationEntry>,
) {
    let Some(hf_theta) = hf_config
        .get("rope_theta")
        .and_then(serde_json::Value::as_f64)
    else {
        return;
    };
    let contract_theta = size.rope_theta;
    let status = if (contract_theta - hf_theta).abs() < 1.0 {
        "match"
    } else if (contract_theta - hf_theta).abs() / hf_theta.max(1.0) < 0.01 {
        "approximate"
    } else {
        "mismatch"
    };
    let entry = CrossValidationEntry {
        field: "rope_theta".to_string(),
        contract_value: format!("{contract_theta}"),
        hf_value: format!("{hf_theta}"),
        status: status.to_string(),
    };
    if status == "mismatch" {
        mismatches.push(entry);
    } else {
        matches.push(entry);
    }
}

fn cross_validate_norm_eps(
    size: &ModelSizeConfig,
    hf_config: &serde_json::Value,
    matches: &mut Vec<CrossValidationEntry>,
    mismatches: &mut Vec<CrossValidationEntry>,
) {
    let hf_eps = hf_config
        .get("rms_norm_eps")
        .or_else(|| hf_config.get("layer_norm_eps"))
        .or_else(|| hf_config.get("layer_norm_epsilon"))
        .and_then(serde_json::Value::as_f64);
    let Some(hf_eps_val) = hf_eps else { return };
    let contract_eps = size.norm_eps;
    let status = if (contract_eps - hf_eps_val).abs() < 1e-12 {
        "match"
    } else {
        "mismatch"
    };
    let entry = CrossValidationEntry {
        field: "norm_eps".to_string(),
        contract_value: format!("{contract_eps:.1e}"),
        hf_value: format!("{hf_eps_val:.1e}"),
        status: status.to_string(),
    };
    if status == "mismatch" {
        mismatches.push(entry);
    } else {
        matches.push(entry);
    }
}

fn cross_validate_model_type(
    constraints: &ModelConstraints,
    hf_config: &serde_json::Value,
    matches: &mut Vec<CrossValidationEntry>,
) {
    if let Some(hf_model_type) = hf_config["model_type"].as_str() {
        matches.push(CrossValidationEntry {
            field: "model_type".to_string(),
            contract_value: format!("{:?}", constraints.attention_type),
            hf_value: hf_model_type.to_string(),
            status: "info".to_string(),
        });
    }
}

fn collect_untracked_hf_fields(hf_config: &serde_json::Value, hf_only: &mut Vec<String>) {
    const TRACKED: &[&str] = &[
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
        "rms_norm_eps",
        "layer_norm_eps",
        "layer_norm_epsilon",
        "model_type",
    ];
    const INTERESTING: &[&str] = &[
        "rope_scaling",
        "sliding_window",
        "attention_dropout",
        "use_cache",
        "tie_word_embeddings",
    ];
    let Some(obj) = hf_config.as_object() else {
        return;
    };
    for key in obj.keys() {
        if !TRACKED.contains(&key.as_str()) && INTERESTING.contains(&key.as_str()) {
            hf_only.push(format!("{key}={}", obj[key]));
        }
    }
}

// ============================================================================
// Phase 4: Architecture Explanation
// ============================================================================

/// Template-based architecture explanations referencing literature
#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureExplanation {
    pub attention_explanation: String,
    pub ffn_explanation: String,
    pub norm_explanation: String,
    pub positional_explanation: String,
    pub scaling_analysis: String,
}

pub fn build_architecture_explanation(
    size: &ModelSizeConfig,
    constraints: &ModelConstraints,
    stats: &StatisticalAnalysis,
) -> ArchitectureExplanation {
    let attention_explanation = match constraints.attention_type {
        AttentionType::Gqa => {
            let mha_kv_mb = stats.kv_cache_4k_mb / stats.gqa_ratio.max(0.01);
            format!(
                "GQA with ratio {:.2} reduces KV cache by {:.0}% vs MHA (Ainslie et al., 2023). \
                 For this model at 4K context: KV cache = {:.1} MB vs {:.1} MB with MHA.",
                stats.gqa_ratio,
                stats.kv_cache_reduction * 100.0,
                stats.kv_cache_4k_mb,
                mha_kv_mb
            )
        }
        AttentionType::Mha => format!(
            "Multi-Head Attention with {} heads, each of dimension {}. \
             Full KV cache: {:.1} MB at 4K context (no sharing).",
            size.num_heads, size.head_dim, stats.kv_cache_4k_mb
        ),
        AttentionType::Mqa => format!(
            "Multi-Query Attention: single KV head shared across {} query heads. \
             Maximum KV cache reduction. Cache: {:.1} MB at 4K context.",
            size.num_heads, stats.kv_cache_4k_mb
        ),
    };

    let ffn_explanation = match constraints.mlp_type {
        MlpType::SwiGlu => format!(
            "SwiGLU (Shazeer, 2020) uses gated activation: \
             FFN(x) = (W_up * x * SiLU(W_gate * x)) * W_down. \
             Expansion ratio {:.2}x compensates for the gating bottleneck (standard FFN uses 4x).",
            stats.ffn_expansion_ratio
        ),
        MlpType::GatedMlp => format!(
            "GeGLU gated MLP: FFN(x) = (W_up * x * GELU(W_gate * x)) * W_down. \
             Expansion ratio {:.2}x with GELU gating.",
            stats.ffn_expansion_ratio
        ),
        MlpType::GeluMlp => format!(
            "Standard GELU MLP: FFN(x) = W2 * GELU(W1 * x + b1) + b2. \
             Expansion ratio {:.2}x (standard 4x for non-gated architectures).",
            stats.ffn_expansion_ratio
        ),
    };

    let norm_explanation = match constraints.norm_type {
        NormType::RmsNorm => format!(
            "RMSNorm (Zhang & Sennrich, 2019) omits mean-centering, reducing compute by ~50% \
             per normalization with minimal quality loss. eps={:.1e}.",
            size.norm_eps
        ),
        NormType::LayerNorm => format!(
            "LayerNorm (Ba et al., 2016) centers and scales activations. \
             eps={:.1e}. More compute than RMSNorm but standard for encoder models.",
            size.norm_eps
        ),
    };

    let positional_explanation = match constraints.positional_encoding {
        PositionalEncoding::Rope => {
            let extrapolated = size.max_position_embeddings * 4;
            format!(
                "RoPE theta={:.0} supports {} positions natively. \
                 Max wavelength: {:.0} tokens. \
                 YaRN/NTK scaling can extend to ~{}.",
                size.rope_theta,
                size.max_position_embeddings,
                stats.rope_max_wavelength,
                extrapolated
            )
        }
        PositionalEncoding::Absolute => format!(
            "Absolute position embeddings: learned vectors for positions 0..{}. \
             No extrapolation beyond training length.",
            size.max_position_embeddings
        ),
        PositionalEncoding::Alibi => format!(
            "ALiBi (Press et al., 2022): linear attention bias, no learned embeddings. \
             Max context: {} (train-time length).",
            size.max_position_embeddings
        ),
        PositionalEncoding::Relative => format!(
            "Relative positional encoding: encodes pairwise distances. \
             Max context: {}.",
            size.max_position_embeddings
        ),
    };

    let params_b = stats.model_params as f64 / 1e9;
    let layers = size.num_layers;
    let hidden = size.hidden_dim;
    let chinchilla_tokens = (params_b * 20.0 * 1e9) as u64;
    let training_flops = 6.0 * stats.model_params as f64 * chinchilla_tokens as f64;
    let scaling_analysis = format!(
        "At {params_b:.1}B parameters with {layers}L x {hidden}H: \
         estimated training compute ~{training_flops:.1e} FLOPs \
         (Chinchilla-optimal tokens: {:.0}B).",
        chinchilla_tokens as f64 / 1e9
    );

    ArchitectureExplanation {
        attention_explanation,
        ffn_explanation,
        norm_explanation,
        positional_explanation,
        scaling_analysis,
    }
}

// ============================================================================
// Phase 5: Kernel Compatibility
// ============================================================================

/// Kernel compatibility report for realizar runtime
#[derive(Debug, Clone, Serialize)]
pub struct KernelCompatibility {
    pub supported_quantizations: Vec<QuantizationSupport>,
    pub attention_kernel: String,
    pub ffn_kernel: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_tps_cpu: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_tps_gpu: Option<f64>,
    pub memory_required_mb: f64,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantizationSupport {
    pub format: String,
    pub supported: bool,
    pub kernel: String,
    pub bits_per_weight: f64,
    pub estimated_size_mb: f64,
}

pub fn build_kernel_compatibility(
    size: &ModelSizeConfig,
    constraints: &ModelConstraints,
    stats: &StatisticalAnalysis,
) -> KernelCompatibility {
    let params = stats.model_params;
    let params_b = params as f64 / 1e9;

    // Build quantization support entries
    let quant_entries = vec![
        QuantizationSupport {
            format: "F16".to_string(),
            supported: true,
            kernel: "trueno::f16_matvec".to_string(),
            bits_per_weight: 16.0,
            estimated_size_mb: (params as f64 * 2.0) / (1024.0 * 1024.0),
        },
        QuantizationSupport {
            format: "Q8_0".to_string(),
            supported: true,
            kernel: "trueno::q8_matvec".to_string(),
            bits_per_weight: 8.0,
            estimated_size_mb: (params as f64 * 1.0) / (1024.0 * 1024.0),
        },
        QuantizationSupport {
            format: "Q4_K_M".to_string(),
            supported: true,
            kernel: "fused_q4k_parallel_matvec (row-major)".to_string(),
            bits_per_weight: 4.5,
            estimated_size_mb: (params as f64 * 0.5625) / (1024.0 * 1024.0),
        },
        QuantizationSupport {
            format: "Q6_K".to_string(),
            supported: true,
            kernel: "fused_q6k_parallel_matvec (row-major)".to_string(),
            bits_per_weight: 6.5,
            estimated_size_mb: (params as f64 * 0.8125) / (1024.0 * 1024.0),
        },
    ];

    // Attention kernel description
    let attention_kernel = match constraints.attention_type {
        AttentionType::Gqa => format!(
            "GQA fused QKV (row-major): {kv}KV groups, {}Q heads per group",
            size.num_heads / size.num_kv_heads.max(1),
            kv = size.num_kv_heads
        ),
        AttentionType::Mha => format!(
            "MHA standard QKV (row-major): {} heads x {} dim",
            size.num_heads, size.head_dim
        ),
        AttentionType::Mqa => format!(
            "MQA single-KV QKV (row-major): 1 KV, {} Q heads",
            size.num_heads
        ),
    };

    // FFN kernel description
    let ffn_kernel = match constraints.mlp_type {
        MlpType::SwiGlu => "fused gated SwiGLU matvec (gate+up fused, row-major)".to_string(),
        MlpType::GatedMlp => "fused gated GeGLU matvec (gate+up fused, row-major)".to_string(),
        MlpType::GeluMlp => "standard GELU MLP matvec (row-major)".to_string(),
    };

    // Performance estimation (memory bandwidth model)
    // CPU: ~50 GB/s DDR5, GPU: ~900 GB/s HBM3
    let q4_size_gb = (params as f64 * 0.5625) / (1024.0 * 1024.0 * 1024.0);
    let estimated_tps_cpu = if params_b > 0.0 {
        Some(50.0 / q4_size_gb)
    } else {
        None
    };
    let estimated_tps_gpu = if params_b > 0.0 {
        Some(900.0 / q4_size_gb)
    } else {
        None
    };

    let memory_required_mb = (params as f64 * 0.5625) / (1024.0 * 1024.0) + stats.kv_cache_4k_mb;

    let mut notes = Vec::new();
    notes.push("All kernels use ROW-MAJOR layout (LAYOUT-001/002)".to_string());
    if constraints.has_bias {
        notes.push("Bias terms included in attention projections".to_string());
    }
    if size.num_kv_heads < size.num_heads {
        notes.push(format!(
            "GQA: {} query heads share {} KV groups ({}:1 ratio)",
            size.num_heads,
            size.num_kv_heads,
            size.num_heads / size.num_kv_heads.max(1)
        ));
    }

    KernelCompatibility {
        supported_quantizations: quant_entries,
        attention_kernel,
        ffn_kernel,
        estimated_tps_cpu,
        estimated_tps_gpu,
        memory_required_mb,
        notes,
    }
}

// ============================================================================
// Phase 6: CLI Options
// ============================================================================

/// Oracle enhancement flags (passed from CLI layer)
#[derive(Debug, Clone, Copy, Default)]
pub struct OracleFlags {
    pub stats: bool,
    pub explain: bool,
    pub kernels: bool,
    pub validate: bool,
    pub full: bool,
}

impl OracleFlags {
    fn show_stats(self) -> bool {
        self.stats || self.full
    }
    fn show_explain(self) -> bool {
        self.explain || self.full
    }
    fn show_kernels(self) -> bool {
        self.kernels || self.full
    }
    fn show_validate(self) -> bool {
        self.validate || self.full
    }
}

// ============================================================================
// Command Entry Point
// ============================================================================

/// Run the oracle command.
///
/// Dispatches to the appropriate mode based on arguments:
/// - Local file: `apr oracle model.gguf`
/// - HuggingFace: `apr oracle hf://org/repo`
/// - Family: `apr oracle --family qwen2`
#[allow(clippy::fn_params_excessive_bools)]
pub(crate) fn run(
    source: Option<&String>,
    family_name: Option<&String>,
    size_filter: Option<&String>,
    show_compliance: bool,
    show_tensors: bool,
    json_output: bool,
    verbose: bool,
    offline: bool,
    flags: OracleFlags,
) -> Result<(), CliError> {
    // Mode 3: Contract description (--family)
    if let Some(family) = family_name {
        return run_family_mode(
            family,
            size_filter.map(String::as_str),
            json_output,
            verbose,
            flags,
        );
    }

    // Require a source for modes 1 and 2
    let source = source.ok_or_else(|| {
        CliError::InvalidFormat(
            "Either <SOURCE> or --family is required. Usage: apr oracle <FILE|hf://...> or apr oracle --family <NAME>".to_string(),
        )
    })?;

    // Mode 2: HuggingFace API query
    if source.starts_with("hf://") || source.starts_with("huggingface://") {
        if offline {
            return Err(CliError::NetworkError(
                "Cannot query HuggingFace API in --offline mode".to_string(),
            ));
        }
        return run_hf_mode(source, json_output, verbose, flags);
    }

    // Mode 1: Local file analysis
    let path = PathBuf::from(source);
    run_local_mode(
        &path,
        show_compliance,
        show_tensors,
        json_output,
        verbose,
        flags,
    )
}

// ============================================================================
// Mode 1: Local File Analysis (PMAT-244)
// ============================================================================

fn run_local_mode(
    path: &Path,
    show_compliance: bool,
    show_tensors: bool,
    json_output: bool,
    verbose: bool,
    flags: OracleFlags,
) -> Result<(), CliError> {
    // Validate file exists
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }

    // Inspect using RosettaStone
    let rosetta = RosettaStone::new();
    let report = rosetta.inspect(path).map_err(|e| {
        CliError::InvalidFormat(format!("Failed to inspect {}: {e}", path.display()))
    })?;

    // Load family registry
    let registry = load_registry()?;

    // Collect tensor names for family detection
    let tensor_names: Vec<&str> = report.tensors.iter().map(|t| t.name.as_str()).collect();

    // Detect family — try tensor names first (SafeTensors), then GGUF architecture metadata
    let detected_family = registry.detect_family(&tensor_names).or_else(|| {
        // GGUF uses different tensor naming (blk.0.attn_q vs model.layers.0.self_attn.q_proj).
        // Fall back to GGUF architecture metadata (e.g., "qwen2" → Qwen2 family).
        report
            .architecture
            .as_deref()
            .and_then(|arch| registry.detect_from_model_type(arch))
    });

    // Build format info
    let format_info = FormatInfo {
        format_type: format!("{}", report.format),
        file_size: report.file_size,
        tensor_count: report.tensors.len(),
        total_params: report.total_params,
        quantization: report.quantization.clone(),
        architecture: report.architecture.clone(),
    };

    // Build family info and size variant
    let (family_info, size_variant_info) = if let Some(family) = detected_family {
        let config = family.config();
        let fi = build_family_info(config);

        // Detect size from metadata
        let size_info = detect_size_from_inspection(family, &report);

        (Some(fi), size_info)
    } else {
        (None, None)
    };

    // Build compliance result
    let compliance = if show_compliance {
        detected_family.map(|family| {
            build_compliance(
                family,
                &tensor_names,
                size_variant_info.as_ref().map(|s| s.name.as_str()),
            )
        })
    } else {
        None
    };

    // Build tensor list
    let tensors_list = if show_tensors {
        Some(build_tensor_list(&report, detected_family))
    } else {
        None
    };

    // Build certification info (PMAT-247)
    let certification = detected_family.and_then(|family| {
        build_certification(
            family.config(),
            size_variant_info.as_ref().map(|s| s.name.as_str()),
        )
    });

    // Build enhanced sections
    let (stats, explanation, kernel_compat) = build_enhanced_sections(
        detected_family,
        size_variant_info.as_ref().map(|s| s.name.as_str()),
        flags,
    );

    let oracle_report = ModelOracleReport {
        source: path.display().to_string(),
        mode: OracleMode::Local,
        family: family_info,
        size_variant: size_variant_info,
        format: Some(format_info),
        compliance,
        certification,
        tensors: tensors_list,
        stats,
        explanation,
        kernel_compatibility: kernel_compat,
        cross_validation: None, // No HF data in local mode
        hf_data: None,
    };

    if json_output {
        output_json(&oracle_report)?;
    } else {
        output_text(&oracle_report, verbose);
    }

    Ok(())
}

// ============================================================================
// Mode 2: HuggingFace API Query (PMAT-245)
// ============================================================================

fn run_hf_mode(
    source: &str,
    json_output: bool,
    verbose: bool,
    flags: OracleFlags,
) -> Result<(), CliError> {
    // Parse HF URI → org/repo
    let repo = source
        .strip_prefix("hf://")
        .or_else(|| source.strip_prefix("huggingface://"))
        .ok_or_else(|| CliError::InvalidFormat(format!("Invalid HF URI: {source}")))?;

    // Fetch HF data via ureq
    let hf_data = fetch_hf_data(repo)?;

    // Load family registry
    let registry = load_registry()?;

    // Extract model_type from config.json
    let model_type = hf_data.config_fields["model_type"].as_str();
    let hidden_size = hf_data.config_fields["hidden_size"]
        .as_u64()
        .map(|v| v as usize);
    let num_layers = hf_data.config_fields["num_hidden_layers"]
        .as_u64()
        .map(|v| v as usize);

    // Detect family from model_type
    let detected_family = model_type.and_then(|mt| registry.detect_from_model_type(mt));

    // Build family info and size variant
    let (family_info, size_variant_info) = if let Some(family) = detected_family {
        let config = family.config();
        let fi = build_family_info(config);

        // Detect size from config.json values
        let size_info = match (hidden_size, num_layers) {
            (Some(h), Some(l)) => family.detect_size(h, l).and_then(|size_name| {
                family
                    .size_config(&size_name)
                    .map(|sc| build_size_info(&size_name, sc, family))
            }),
            _ => None,
        };

        (Some(fi), size_info)
    } else {
        (None, None)
    };

    // Build certification info
    let certification = detected_family.and_then(|family| {
        build_certification(
            family.config(),
            size_variant_info.as_ref().map(|s| s.name.as_str()),
        )
    });

    // Build enhanced sections
    let (stats, explanation, kernel_compat) = build_enhanced_sections(
        detected_family,
        size_variant_info.as_ref().map(|s| s.name.as_str()),
        flags,
    );

    // Cross-validation
    let cross_validation = if flags.show_validate() {
        detected_family.and_then(|family| {
            let size_name = size_variant_info.as_ref().map(|s| s.name.as_str())?;
            let size_config = family.size_config(size_name)?;
            Some(cross_validate(
                size_config,
                family.constraints(),
                &hf_data.config_fields,
            ))
        })
    } else {
        None
    };

    let oracle_report = ModelOracleReport {
        source: source.to_string(),
        mode: OracleMode::HuggingFace,
        family: family_info,
        size_variant: size_variant_info,
        format: None,
        compliance: None,
        certification,
        tensors: None,
        stats,
        explanation,
        kernel_compatibility: kernel_compat,
        cross_validation,
        hf_data: Some(hf_data),
    };

    if json_output {
        output_json(&oracle_report)?;
    } else {
        output_text(&oracle_report, verbose);
    }

    Ok(())
}

// ============================================================================
// Mode 3: Contract Description (PMAT-246)
// ============================================================================

fn run_family_mode(
    family_name: &str,
    size_filter: Option<&str>,
    json_output: bool,
    verbose: bool,
    flags: OracleFlags,
) -> Result<(), CliError> {
    let registry = load_registry()?;
    let family = registry
        .detect_from_model_type(family_name)
        .ok_or_else(|| {
            // List known families for helpful error
            let known: Vec<String> = (0..registry.len())
            .filter_map(|_| None::<String>) // We can't iterate, so list empty
            .collect();
            let _ = known; // avoid unused
            CliError::InvalidFormat(format!(
                "Unknown model family: '{family_name}'. Use tensor names or HF model_type."
            ))
        })?;

    let config = family.config();

    if json_output {
        // Build a report with family details
        let fi = build_family_info(config);

        // If size filter, include that variant only
        let size_info = size_filter.and_then(|size| {
            family
                .size_config(size)
                .map(|sc| build_size_info(size, sc, family))
        });

        // Build enhanced sections
        let (stats, explanation, kernel_compat) =
            build_enhanced_sections(Some(family), size_filter, flags);

        let report = ModelOracleReport {
            source: family_name.to_string(),
            mode: OracleMode::Family,
            family: Some(fi),
            size_variant: size_info,
            format: None,
            compliance: None,
            certification: build_certification(config, size_filter),
            tensors: None,
            stats,
            explanation,
            kernel_compatibility: kernel_compat,
            cross_validation: None,
            hf_data: None,
        };
        output_json(&report)?;
    } else {
        output_family_description(config, size_filter, verbose, flags, family);
    }

    Ok(())
}

// ============================================================================
// Enhanced Sections Builder
// ============================================================================

fn build_enhanced_sections(
    family: Option<&dyn ModelFamily>,
    size_name: Option<&str>,
    flags: OracleFlags,
) -> (
    Option<StatisticalAnalysis>,
    Option<ArchitectureExplanation>,
    Option<KernelCompatibility>,
) {
    let Some(family) = family else {
        return (None, None, None);
    };

    let Some(size_name) = size_name else {
        return (None, None, None);
    };

    let Some(size) = family.size_config(size_name) else {
        return (None, None, None);
    };

    let constraints = family.constraints();

    let stats = if flags.show_stats() || flags.show_explain() || flags.show_kernels() {
        Some(build_statistical_analysis(size, constraints))
    } else {
        None
    };

    let explanation = if flags.show_explain() {
        stats
            .as_ref()
            .map(|s| build_architecture_explanation(size, constraints, s))
    } else {
        None
    };

    let kernel_compat = if flags.show_kernels() {
        stats
            .as_ref()
            .map(|s| build_kernel_compatibility(size, constraints, s))
    } else {
        None
    };

    // Only return stats if explicitly requested
    let stats = if flags.show_stats() { stats } else { None };

    (stats, explanation, kernel_compat)
}

// ============================================================================
// Registry Loading
// ============================================================================

/// Load the family registry from contracts/ directory.
/// Searches for contracts relative to the executable, then falls back to CWD.
fn contracts_candidate_paths() -> Vec<PathBuf> {
    let mut candidates = vec![PathBuf::from("contracts")];

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(exe_dir.join("contracts"));
            if let Some(parent) = exe_dir.parent() {
                candidates.push(parent.join("contracts"));
            }
        }
    }

    for ancestor in [".", "..", "../..", "../../.."] {
        candidates.push(PathBuf::from(ancestor).join("contracts"));
    }

    candidates
}

fn load_registry() -> Result<FamilyRegistry, CliError> {
    for candidate in contracts_candidate_paths() {
        if candidate.join("model-families").exists() {
            return load_family_registry(&candidate)
                .map_err(|e| CliError::Aprender(format!("Failed to load family contracts: {e}")));
        }
    }
    // Return empty registry rather than error — graceful degradation
    Ok(FamilyRegistry::new())
}

// ============================================================================
// Helper Functions
// ============================================================================

fn build_family_info(config: &ModelFamilyConfig) -> FamilyInfo {
    let constraints = &config.constraints;
    FamilyInfo {
        name: config.family.clone(),
        display_name: config.display_name.clone(),
        vendor: config.vendor.clone(),
        architectures: config.architectures.clone(),
        constraints: ConstraintsSummary {
            attention: format!("{}", constraints.attention_type),
            activation: format!("{}", constraints.activation),
            norm: format!("{}", constraints.norm_type),
            bias: constraints.has_bias,
            mlp: format!("{}", constraints.mlp_type),
            tied_embeddings: constraints.tied_embeddings,
            positional_encoding: format!("{}", constraints.positional_encoding),
        },
        chat_template_format: config.chat_template.as_ref().map(|ct| ct.format.clone()),
    }
}

fn build_size_info(
    size_name: &str,
    sc: &ModelSizeConfig,
    family: &dyn ModelFamily,
) -> SizeVariantInfo {
    SizeVariantInfo {
        name: size_name.to_string(),
        parameters: sc.parameters.clone(),
        hidden_dim: sc.hidden_dim,
        num_layers: sc.num_layers,
        num_heads: sc.num_heads,
        num_kv_heads: sc.num_kv_heads,
        intermediate_dim: sc.intermediate_dim,
        vocab_size: sc.vocab_size,
        expected_tensor_count: family.expected_tensor_count(size_name).unwrap_or(0),
    }
}

fn detect_size_from_inspection(
    family: &dyn ModelFamily,
    report: &aprender::format::rosetta::InspectionReport,
) -> Option<SizeVariantInfo> {
    // Try to extract hidden_dim from metadata
    let hidden_dim = report
        .metadata
        .get("hidden_size")
        .or_else(|| report.metadata.get("embedding_length"))
        .or_else(|| report.metadata.get("hidden_dim"))
        .and_then(|v| v.parse::<usize>().ok());

    let num_layers = report
        .metadata
        .get("num_hidden_layers")
        .or_else(|| report.metadata.get("block_count"))
        .or_else(|| report.metadata.get("num_layers"))
        .and_then(|v| v.parse::<usize>().ok());

    // Also try to infer from tensor shapes (embedding shape[1] = hidden_dim)
    let hidden_from_tensors = report.tensors.iter().find_map(|t| {
        if t.name.contains("embed") && t.shape.len() == 2 {
            Some(t.shape[1])
        } else {
            None
        }
    });

    let layers_from_tensors = {
        let mut max_layer: Option<usize> = None;
        for t in &report.tensors {
            // Match patterns like "layers.N." or "blk.N."
            if let Some(rest) = t
                .name
                .strip_prefix("model.layers.")
                .or_else(|| t.name.strip_prefix("blk."))
                .or_else(|| t.name.strip_prefix("encoder.layers."))
            {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(n) = rest[..dot_pos].parse::<usize>() {
                        max_layer = Some(max_layer.map_or(n, |m: usize| m.max(n)));
                    }
                }
            }
        }
        max_layer.map(|m| m + 1) // 0-indexed → count
    };

    let h = hidden_dim.or(hidden_from_tensors)?;
    let l = num_layers.or(layers_from_tensors)?;

    let size_name = family.detect_size(h, l)?;
    let sc = family.size_config(&size_name)?;
    Some(build_size_info(&size_name, sc, family))
}

fn build_compliance(
    family: &dyn ModelFamily,
    tensor_names: &[&str],
    size_name: Option<&str>,
) -> ComplianceResult {
    let size = size_name.unwrap_or("unknown");

    // Get expected tensors from template
    let config = family.config();
    let expected = expand_tensor_template(&config.tensor_template, config, size);

    let actual_set: std::collections::HashSet<&str> = tensor_names.iter().copied().collect();
    let expected_set: std::collections::HashSet<&str> =
        expected.iter().map(String::as_str).collect();

    let missing: Vec<String> = expected_set
        .difference(&actual_set)
        .map(|s| (*s).to_string())
        .collect();

    let unexpected: Vec<String> = actual_set
        .difference(&expected_set)
        .map(|s| (*s).to_string())
        .collect();

    let expected_count = family.expected_tensor_count(size).unwrap_or(expected.len());
    let tensor_count_match = tensor_names.len() == expected_count;

    ComplianceResult {
        is_compliant: missing.is_empty() && tensor_count_match,
        tensor_count_match,
        missing_tensors: missing,
        unexpected_tensors: unexpected,
    }
}

fn expand_tensor_template(
    template: &aprender::format::model_family::TensorTemplate,
    config: &ModelFamilyConfig,
    size_name: &str,
) -> Vec<String> {
    let mut names = Vec::new();

    // Global tensors
    if !template.embedding.is_empty() {
        names.push(template.embedding.clone());
    }
    if let Some(ref lm_head) = template.lm_head {
        names.push(lm_head.clone());
    }
    if let Some(ref final_norm) = template.final_norm {
        names.push(final_norm.clone());
    }

    // Per-layer tensors
    let num_layers = config
        .size_variants
        .get(size_name)
        .map_or(0, |sc| sc.num_layers);

    for layer_idx in 0..num_layers {
        for pat in template.per_layer.values().flatten() {
            names.push(pat.replace("{n}", &layer_idx.to_string()));
        }
    }

    names
}

fn build_tensor_list(
    report: &aprender::format::rosetta::InspectionReport,
    _family: Option<&dyn ModelFamily>,
) -> Vec<TensorComplianceEntry> {
    report
        .tensors
        .iter()
        .map(|t| TensorComplianceEntry {
            name: t.name.clone(),
            present: true,
            dtype: Some(t.dtype.clone()),
            shape: Some(t.shape.clone()),
            note: None,
        })
        .collect()
}

fn build_certification(
    config: &ModelFamilyConfig,
    size_name: Option<&str>,
) -> Option<CertificationInfo> {
    let cert = config.certification.as_ref()?;

    let playbook = size_name.map(|size| cert.playbook_path.replace("{size}", size));

    Some(CertificationInfo {
        status: "PENDING".to_string(),
        playbook_path: playbook,
    })
}

fn format_params(params: usize) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else if params >= 1_000 {
        format!("{:.1}K", params as f64 / 1_000.0)
    } else {
        format!("{params}")
    }
}

// ============================================================================
// Output Formatting
// ============================================================================

fn output_json(report: &ModelOracleReport) -> Result<(), CliError> {
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| CliError::Aprender(format!("JSON serialization failed: {e}")))?;
    println!("{json}");
    Ok(())
}

fn format_text_report(report: &ModelOracleReport) -> String {
    let mut out = String::new();
    writeln!(out, "  Source: {}", report.source).ok();
    writeln!(out, "  Mode: {:?}", report.mode).ok();
    out
}

fn output_text(report: &ModelOracleReport, verbose: bool) {
    output::header("Model Oracle Report");
    print!("{}", format_text_report(report));

    output_text_format(report.format.as_ref());
    output_text_family(report.family.as_ref(), verbose);
    output_text_size(report.size_variant.as_ref());
    output_text_constraints(report.family.as_ref());
    output_text_stats(report.stats.as_ref());
    output_text_explanation(report.explanation.as_ref());
    output_text_kernels(report.kernel_compatibility.as_ref());
    output_text_cross_validation(report.cross_validation.as_ref());
    output_text_hf(report.hf_data.as_ref(), verbose);
    output_text_compliance(report.compliance.as_ref(), verbose);
    output_text_certification(report.certification.as_ref());
    output_text_tensors(report.tensors.as_ref(), verbose);
}

fn format_text_format(fmt: &FormatInfo) -> String {
    let mut out = String::new();
    writeln!(out, "  Format: {}", fmt.format_type).ok();
    writeln!(
        out,
        "  File Size: {}",
        output::format_size(fmt.file_size as u64)
    )
    .ok();
    writeln!(out, "  Tensors: {}", fmt.tensor_count).ok();
    writeln!(out, "  Parameters: {}", format_params(fmt.total_params)).ok();
    if let Some(ref q) = fmt.quantization {
        writeln!(out, "  Quantization: {q}").ok();
    }
    if let Some(ref arch) = fmt.architecture {
        writeln!(out, "  Architecture: {arch}").ok();
    }
    out
}

fn output_text_format(fmt: Option<&FormatInfo>) {
    let Some(fmt) = fmt else { return };
    print!("{}", format_text_format(fmt));
}

fn format_text_family(family: &FamilyInfo, verbose: bool) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  Family: {} ({})", family.name, family.display_name).ok();
    writeln!(out, "  Vendor: {}", family.vendor).ok();
    if verbose {
        writeln!(out, "  Architectures: {}", family.architectures.join(", ")).ok();
    }
    if let Some(ref ct) = family.chat_template_format {
        writeln!(out, "  Chat Template: {ct}").ok();
    }
    out
}

fn output_text_family(family: Option<&FamilyInfo>, verbose: bool) {
    let Some(family) = family else {
        println!();
        output::kv("Family", "UNKNOWN (no matching contract)");
        return;
    };
    print!("{}", format_text_family(family, verbose));
}

fn format_text_size(size: &SizeVariantInfo) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(
        out,
        "  Size: {} (hidden={}, layers={}, heads={}, kv_heads={})",
        size.parameters, size.hidden_dim, size.num_layers, size.num_heads, size.num_kv_heads
    )
    .ok();
    writeln!(out, "  Intermediate Dim: {}", size.intermediate_dim).ok();
    writeln!(out, "  Vocab Size: {}", size.vocab_size).ok();
    writeln!(out, "  Expected Tensors: {}", size.expected_tensor_count).ok();
    out
}

fn output_text_size(size: Option<&SizeVariantInfo>) {
    let Some(size) = size else { return };
    print!("{}", format_text_size(size));
}

fn format_text_constraints(family: &FamilyInfo) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  Constraints:").ok();
    let c = &family.constraints;
    writeln!(
        out,
        "    Attention: {} | Activation: {} | Norm: {}",
        c.attention, c.activation, c.norm
    )
    .ok();
    writeln!(
        out,
        "    Bias: {} | Tied: {} | MLP: {} | Position: {}",
        if c.bias { "yes" } else { "no" },
        if c.tied_embeddings { "yes" } else { "no" },
        c.mlp,
        c.positional_encoding
    )
    .ok();
    out
}

fn output_text_constraints(family: Option<&FamilyInfo>) {
    let Some(family) = family else { return };
    print!("{}", format_text_constraints(family));
}

fn format_text_stats(stats: &StatisticalAnalysis) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(
        out,
        "  GQA Ratio: {:.2} ({:.0}% KV cache reduction)",
        stats.gqa_ratio,
        stats.kv_cache_reduction * 100.0
    )
    .ok();
    writeln!(
        out,
        "  Model Parameters: {}",
        format_params(stats.model_params as usize)
    )
    .ok();
    writeln!(out, "  Model Size (F16): {:.1} MB", stats.model_size_f16_mb).ok();
    writeln!(
        out,
        "  Model Size (Q4_K_M): {:.1} MB",
        stats.model_size_q4_mb
    )
    .ok();
    writeln!(
        out,
        "  KV Cache/Token: {} bytes",
        stats.kv_cache_per_token_bytes
    )
    .ok();
    writeln!(out, "  KV Cache (4K ctx): {:.1} MB", stats.kv_cache_4k_mb).ok();
    writeln!(out, "  FFN Expansion: {:.2}x", stats.ffn_expansion_ratio).ok();
    writeln!(out, "  FFN Type: {}", stats.ffn_type_explanation).ok();
    if stats.rope_max_wavelength > 0.0 {
        writeln!(out, "  RoPE Wavelength: {:.0}", stats.rope_max_wavelength).ok();
    }
    writeln!(out, "  Context Window: {}", stats.effective_context_window).ok();
    writeln!(
        out,
        "  Attn FLOPS/tok: {:.2e}",
        stats.attention_flops_per_token as f64
    )
    .ok();
    writeln!(
        out,
        "  FFN FLOPS/tok: {:.2e}",
        stats.ffn_flops_per_token as f64
    )
    .ok();
    out
}

fn output_text_stats(stats: Option<&StatisticalAnalysis>) {
    let Some(stats) = stats else { return };
    output::section("Statistical Analysis");
    print!("{}", format_text_stats(stats));
}

fn format_text_explanation(expl: &ArchitectureExplanation) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  Attention: {}", expl.attention_explanation).ok();
    writeln!(out).ok();
    writeln!(out, "  FFN: {}", expl.ffn_explanation).ok();
    writeln!(out).ok();
    writeln!(out, "  Normalization: {}", expl.norm_explanation).ok();
    writeln!(out).ok();
    writeln!(out, "  Position: {}", expl.positional_explanation).ok();
    writeln!(out).ok();
    writeln!(out, "  Scaling: {}", expl.scaling_analysis).ok();
    out
}

fn output_text_explanation(expl: Option<&ArchitectureExplanation>) {
    let Some(expl) = expl else { return };
    output::section("Architecture Explanation");
    print!("{}", format_text_explanation(expl));
}

fn format_text_kernels(kern: &KernelCompatibility) -> String {
    let mut out = String::new();
    writeln!(out, "  Attention Kernel: {}", kern.attention_kernel).ok();
    writeln!(out, "  FFN Kernel: {}", kern.ffn_kernel).ok();
    writeln!(out).ok();
    writeln!(out, "  Quantization Support:").ok();
    writeln!(
        out,
        "    {:<12} {:<10} {:<8} {:<12} Kernel",
        "Format", "Supported", "BPW", "Size (MB)"
    )
    .ok();
    for q in &kern.supported_quantizations {
        writeln!(
            out,
            "    {:<12} {:<10} {:<8.1} {:<12.1} {}",
            q.format,
            if q.supported { "yes" } else { "no" },
            q.bits_per_weight,
            q.estimated_size_mb,
            q.kernel
        )
        .ok();
    }
    if let Some(tps_cpu) = kern.estimated_tps_cpu {
        writeln!(out, "  Est. CPU tok/s (Q4_K_M): {tps_cpu:.0}").ok();
    }
    if let Some(tps_gpu) = kern.estimated_tps_gpu {
        writeln!(out, "  Est. GPU tok/s (Q4_K_M): {tps_gpu:.0}").ok();
    }
    writeln!(
        out,
        "  Memory Required (Q4+KV): {:.1} MB",
        kern.memory_required_mb
    )
    .ok();
    for note in &kern.notes {
        writeln!(out, "    * {note}").ok();
    }
    out
}

fn output_text_kernels(kern: Option<&KernelCompatibility>) {
    let Some(kern) = kern else { return };
    println!();
    output::section("Kernel Compatibility");
    print!("{}", format_text_kernels(kern));
}

fn format_text_cross_validation(cv: &CrossValidation) -> String {
    let mut out = String::new();
    if !cv.matches.is_empty() {
        writeln!(out, "  Matches ({}):", cv.matches.len()).ok();
        for entry in &cv.matches {
            writeln!(
                out,
                "    [OK] {}: {} == {}",
                entry.field, entry.contract_value, entry.hf_value
            )
            .ok();
        }
    }
    if !cv.mismatches.is_empty() {
        writeln!(out, "  Mismatches ({}):", cv.mismatches.len()).ok();
        for entry in &cv.mismatches {
            writeln!(
                out,
                "    [!!] {}: contract={} vs hf={}",
                entry.field, entry.contract_value, entry.hf_value
            )
            .ok();
        }
    }
    if !cv.contract_only.is_empty() {
        writeln!(out, "  Contract-only: {}", cv.contract_only.join(", ")).ok();
    }
    if !cv.hf_only.is_empty() {
        writeln!(out, "  HF-only: {}", cv.hf_only.join(", ")).ok();
    }
    out
}

fn output_text_cross_validation(cv: Option<&CrossValidation>) {
    let Some(cv) = cv else { return };
    println!();
    output::section("Cross-Validation (Contract vs HF)");
    print!("{}", format_text_cross_validation(cv));
}

fn format_text_hf(hf: &HuggingFaceData) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  HF Repo: {}", hf.repo).ok();
    if let Some(ref mt) = hf.model_type {
        writeln!(out, "  HF model_type: {mt}").ok();
    }
    if let Some(ref pt) = hf.pipeline_tag {
        writeln!(out, "  HF pipeline_tag: {pt}").ok();
    }
    if let Some(dl) = hf.downloads {
        writeln!(out, "  HF Downloads: {dl}").ok();
    }
    out
}

fn output_text_hf(hf: Option<&HuggingFaceData>, verbose: bool) {
    let Some(hf) = hf else { return };
    if !verbose {
        return;
    }
    print!("{}", format_text_hf(hf));
}

fn format_text_compliance(compliance: &ComplianceResult, verbose: bool) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    if compliance.is_compliant {
        writeln!(out, "  Contract: COMPLIANT").ok();
        return out;
    }
    writeln!(out, "  Contract: NON-COMPLIANT").ok();
    if !compliance.tensor_count_match {
        writeln!(out, "  Tensor Count: MISMATCH").ok();
    }
    if !compliance.missing_tensors.is_empty() {
        writeln!(
            out,
            "  Missing Tensors: {} tensor(s)",
            compliance.missing_tensors.len()
        )
        .ok();
        if verbose {
            for t in &compliance.missing_tensors {
                writeln!(out, "    - {t}").ok();
            }
        }
    }
    if !compliance.unexpected_tensors.is_empty() && verbose {
        writeln!(
            out,
            "  Unexpected Tensors: {} tensor(s)",
            compliance.unexpected_tensors.len()
        )
        .ok();
        for t in &compliance.unexpected_tensors {
            writeln!(out, "    + {t}").ok();
        }
    }
    out
}

fn output_text_compliance(compliance: Option<&ComplianceResult>, verbose: bool) {
    let Some(compliance) = compliance else { return };
    print!("{}", format_text_compliance(compliance, verbose));
}

fn format_text_certification(cert: &CertificationInfo) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  Certification: {}", cert.status).ok();
    if let Some(ref pb) = cert.playbook_path {
        writeln!(out, "  Playbook: {pb}").ok();
    }
    out
}

fn output_text_certification(cert: Option<&CertificationInfo>) {
    let Some(cert) = cert else { return };
    print!("{}", format_text_certification(cert));
}

fn format_text_tensors(tensors: &[TensorComplianceEntry], verbose: bool) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  Tensors ({} total):", tensors.len()).ok();
    let max_show = if verbose { tensors.len() } else { 20 };
    for (i, t) in tensors.iter().enumerate() {
        if i >= max_show && i < tensors.len() - 2 {
            if i == max_show {
                writeln!(out, "    ... ({} more) ...", tensors.len() - max_show - 2).ok();
            }
            continue;
        }
        let shape_str = t
            .shape
            .as_ref()
            .map(|s| format!("{s:?}"))
            .unwrap_or_default();
        let dtype_str = t.dtype.as_deref().unwrap_or("");
        writeln!(out, "    {:<50} {} {}", t.name, dtype_str, shape_str).ok();
    }
    out
}

fn output_text_tensors(tensors: Option<&Vec<TensorComplianceEntry>>, verbose: bool) {
    let Some(tensors) = tensors else { return };
    print!("{}", format_text_tensors(tensors, verbose));
}

/// Format the family description header (config metadata + constraints).
fn format_family_description_header(config: &ModelFamilyConfig) -> String {
    let mut out = String::new();
    writeln!(out, "  Family: {}", config.family).ok();
    writeln!(out, "  Vendor: {}", config.vendor).ok();
    writeln!(out, "  Architectures: {}", config.architectures.join(", ")).ok();
    writeln!(out, "  HF Pattern: {}", config.hf_pattern).ok();

    let c = &config.constraints;
    writeln!(out).ok();
    writeln!(out, "  Constraints:").ok();
    writeln!(
        out,
        "    Attention: {} | Activation: {} | Norm: {}",
        c.attention_type, c.activation, c.norm_type
    )
    .ok();
    writeln!(
        out,
        "    Bias: {} | Tied: {} | MLP: {} | Position: {}",
        if c.has_bias { "yes" } else { "no" },
        if c.tied_embeddings { "yes" } else { "no" },
        c.mlp_type,
        c.positional_encoding
    )
    .ok();
    out
}

/// Format a single size variant block.
fn format_family_size_variant(name: &str, sc: &ModelSizeConfig) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  Size Variant: {name} ({})", sc.parameters).ok();
    writeln!(out, "    hidden_dim: {}", sc.hidden_dim).ok();
    writeln!(out, "    num_layers: {}", sc.num_layers).ok();
    writeln!(out, "    num_heads: {}", sc.num_heads).ok();
    writeln!(out, "    num_kv_heads: {}", sc.num_kv_heads).ok();
    writeln!(out, "    intermediate_dim: {}", sc.intermediate_dim).ok();
    writeln!(out, "    vocab_size: {}", sc.vocab_size).ok();
    writeln!(out, "    head_dim: {}", sc.head_dim).ok();
    if sc.rope_theta > 0.0 {
        writeln!(out, "    rope_theta: {}", sc.rope_theta).ok();
    }
    writeln!(out, "    norm_eps: {}", sc.norm_eps).ok();
    out
}

/// Format per-variant stats summary.
fn format_family_variant_stats(stats: &StatisticalAnalysis) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(
        out,
        "    GQA Ratio: {:.2} ({:.0}% KV reduction)",
        stats.gqa_ratio,
        stats.kv_cache_reduction * 100.0
    )
    .ok();
    writeln!(
        out,
        "    Est. Parameters: {}",
        format_params(stats.model_params as usize)
    )
    .ok();
    writeln!(
        out,
        "    Model Size (F16): {:.1} MB",
        stats.model_size_f16_mb
    )
    .ok();
    writeln!(out, "    Model Size (Q4): {:.1} MB", stats.model_size_q4_mb).ok();
    writeln!(out, "    KV Cache (4K): {:.1} MB", stats.kv_cache_4k_mb).ok();
    writeln!(out, "    FFN Ratio: {:.2}x", stats.ffn_expansion_ratio).ok();
    out
}

/// Format per-variant explanation summary.
fn format_family_variant_explain(expl: &ArchitectureExplanation) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "    Attention: {}", expl.attention_explanation).ok();
    writeln!(out, "    FFN: {}", expl.ffn_explanation).ok();
    writeln!(out, "    Scaling: {}", expl.scaling_analysis).ok();
    out
}

/// Format per-variant kernel summary.
fn format_family_variant_kernels(kern: &KernelCompatibility) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "    Attn Kernel: {}", kern.attention_kernel).ok();
    writeln!(out, "    FFN Kernel: {}", kern.ffn_kernel).ok();
    if let Some(tps) = kern.estimated_tps_cpu {
        writeln!(out, "    Est. CPU tok/s: {tps:.0}").ok();
    }
    if let Some(tps) = kern.estimated_tps_gpu {
        writeln!(out, "    Est. GPU tok/s: {tps:.0}").ok();
    }
    writeln!(out, "    Memory (Q4+KV): {:.1} MB", kern.memory_required_mb).ok();
    out
}

/// Format family description footer (tensor template, quantizations, chat template).
fn format_family_description_footer(config: &ModelFamilyConfig, verbose: bool) -> String {
    let mut out = String::new();
    if verbose {
        writeln!(out).ok();
        writeln!(out, "  Tensor Template:").ok();
        writeln!(out, "    Embedding: {}", config.tensor_template.embedding).ok();
        if let Some(ref lm) = config.tensor_template.lm_head {
            writeln!(out, "    LM Head: {lm}").ok();
        }
        if let Some(ref norm) = config.tensor_template.final_norm {
            writeln!(out, "    Final Norm: {norm}").ok();
        }
        writeln!(out, "    Per-layer:").ok();
        for (role, pattern) in &config.tensor_template.per_layer {
            if let Some(pat) = pattern {
                writeln!(out, "      {role}: {pat}").ok();
            }
        }
    }
    if !config.quantizations.is_empty() {
        writeln!(out).ok();
        writeln!(out, "  Quantizations: {}", config.quantizations.join(", ")).ok();
    }
    if let Some(ref ct) = config.chat_template {
        writeln!(out).ok();
        writeln!(out, "  Chat Template: {}", ct.format).ok();
        writeln!(out, "    BOS: {}", ct.bos_token).ok();
        writeln!(out, "    EOS: {}", ct.eos_token).ok();
    }
    out
}

fn output_family_description(
    config: &ModelFamilyConfig,
    size_filter: Option<&str>,
    verbose: bool,
    flags: OracleFlags,
    family: &dyn ModelFamily,
) {
    output::section(&format!("{} Family Contract", config.display_name));
    print!("{}", format_family_description_header(config));

    // Size variants
    let variants: Vec<(&String, &ModelSizeConfig)> = if let Some(size) = size_filter {
        config
            .size_variants
            .iter()
            .filter(|(k, _)| k.as_str() == size)
            .collect()
    } else {
        let mut v: Vec<_> = config.size_variants.iter().collect();
        v.sort_by_key(|(_, sc)| sc.hidden_dim);
        v
    };

    for (name, sc) in &variants {
        print!("{}", format_family_size_variant(name, sc));

        if flags.show_stats() || flags.show_explain() || flags.show_kernels() {
            let stats = build_statistical_analysis(sc, family.constraints());

            if flags.show_stats() {
                print!("{}", format_family_variant_stats(&stats));
            }

            if flags.show_explain() {
                let expl = build_architecture_explanation(sc, family.constraints(), &stats);
                print!("{}", format_family_variant_explain(&expl));
            }

            if flags.show_kernels() {
                let kern = build_kernel_compatibility(sc, family.constraints(), &stats);
                print!("{}", format_family_variant_kernels(&kern));
            }
        }
    }

    if size_filter.is_some() && variants.is_empty() {
        println!();
        output::kv(
            "Error",
            format!("Size '{}' not found", size_filter.unwrap_or("")),
        );
        let available: Vec<&String> = config.size_variants.keys().collect();
        output::kv("Available", format!("{available:?}"));
    }

    print!("{}", format_family_description_footer(config, verbose));
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oracle_mode_serialize() {
        let mode = OracleMode::Local;
        let json = serde_json::to_string(&mode).expect("serialize mode");
        assert_eq!(json, "\"local\"");

        let mode = OracleMode::HuggingFace;
        let json = serde_json::to_string(&mode).expect("serialize mode");
        assert_eq!(json, "\"hugging_face\"");

        let mode = OracleMode::Family;
        let json = serde_json::to_string(&mode).expect("serialize mode");
        assert_eq!(json, "\"family\"");
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(1_500), "1.5K");
        assert_eq!(format_params(1_500_000), "1.5M");
        assert_eq!(format_params(1_500_000_000), "1.5B");
        assert_eq!(format_params(7_000_000_000), "7.0B");
    }

    #[test]
    fn test_family_info_build() {
        use aprender::format::model_family::*;
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test Model".to_string(),
            vendor: "TestCo".to_string(),
            architectures: vec!["TestForCausalLM".to_string()],
            hf_pattern: "test/*".to_string(),
            size_variants: HashMap::new(),
            constraints: ModelConstraints {
                attention_type: AttentionType::Gqa,
                activation: Activation::Silu,
                norm_type: NormType::RmsNorm,
                has_bias: true,
                tied_embeddings: false,
                positional_encoding: PositionalEncoding::Rope,
                mlp_type: MlpType::SwiGlu,
            },
            tensor_template: TensorTemplate {
                embedding: "embed.weight".to_string(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec!["q4_k_m".to_string()],
            chat_template: None,
            certification: None,
        };

        let fi = build_family_info(&config);
        assert_eq!(fi.name, "test");
        assert_eq!(fi.vendor, "TestCo");
        assert_eq!(fi.constraints.attention, "GQA");
        assert!(fi.constraints.bias);
    }

    #[test]
    fn test_compliance_result_serialize() {
        let cr = ComplianceResult {
            is_compliant: true,
            tensor_count_match: true,
            missing_tensors: vec![],
            unexpected_tensors: vec![],
        };
        let json = serde_json::to_string(&cr).expect("serialize");
        assert!(json.contains("\"is_compliant\":true"));
    }

    #[test]
    fn test_report_json_roundtrip() {
        let report = ModelOracleReport {
            source: "test.gguf".to_string(),
            mode: OracleMode::Local,
            family: Some(FamilyInfo {
                name: "qwen2".to_string(),
                display_name: "Qwen2".to_string(),
                vendor: "Alibaba".to_string(),
                architectures: vec!["Qwen2ForCausalLM".to_string()],
                constraints: ConstraintsSummary {
                    attention: "GQA".to_string(),
                    activation: "SiLU".to_string(),
                    norm: "RMSNorm".to_string(),
                    bias: true,
                    tied_embeddings: false,
                    mlp: "SwiGLU".to_string(),
                    positional_encoding: "RoPE".to_string(),
                },
                chat_template_format: Some("chatml".to_string()),
            }),
            size_variant: Some(SizeVariantInfo {
                name: "1.5b".to_string(),
                parameters: "1.5B".to_string(),
                hidden_dim: 1536,
                num_layers: 28,
                num_heads: 12,
                num_kv_heads: 2,
                intermediate_dim: 8960,
                vocab_size: 151936,
                expected_tensor_count: 339,
            }),
            format: Some(FormatInfo {
                format_type: "GGUF".to_string(),
                file_size: 1_000_000,
                tensor_count: 339,
                total_params: 1_500_000_000,
                quantization: Some("Q4_K_M".to_string()),
                architecture: Some("qwen2".to_string()),
            }),
            compliance: None,
            certification: None,
            tensors: None,
            stats: None,
            explanation: None,
            kernel_compatibility: None,
            cross_validation: None,
            hf_data: None,
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("\"source\": \"test.gguf\""));
        assert!(json.contains("\"mode\": \"local\""));
        assert!(json.contains("\"family\""));
        assert!(json.contains("\"hidden_dim\": 1536"));
    }

    #[test]
    fn test_certification_build_with_size() {
        use aprender::format::model_family::CertificationConfig;
        use std::collections::HashMap;

        let mut size_cats = HashMap::new();
        size_cats.insert("1.5b".to_string(), "small".to_string());

        let config = ModelFamilyConfig {
            family: "qwen2".to_string(),
            display_name: "Qwen2".to_string(),
            vendor: "Alibaba".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: aprender::format::model_family::ModelConstraints {
                attention_type: aprender::format::model_family::AttentionType::Gqa,
                activation: aprender::format::model_family::Activation::Silu,
                norm_type: aprender::format::model_family::NormType::RmsNorm,
                has_bias: false,
                tied_embeddings: false,
                positional_encoding: aprender::format::model_family::PositionalEncoding::Rope,
                mlp_type: aprender::format::model_family::MlpType::SwiGlu,
            },
            tensor_template: aprender::format::model_family::TensorTemplate {
                embedding: String::new(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: Some(CertificationConfig {
                playbook_path: "../playbooks/{size}.yaml".to_string(),
                csv_family_key: "qwen2".to_string(),
                size_categories: size_cats,
            }),
        };

        let cert = build_certification(&config, Some("1.5b"));
        assert!(cert.is_some());
        let cert = cert.expect("cert exists");
        assert_eq!(cert.status, "PENDING");
        assert_eq!(
            cert.playbook_path,
            Some("../playbooks/1.5b.yaml".to_string())
        );
    }

    #[test]
    fn test_source_required_error() {
        let flags = OracleFlags::default();
        let result = run(None, None, None, false, false, false, false, false, flags);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidFormat(msg)) => {
                assert!(msg.contains("required"));
            }
            other => panic!("Expected InvalidFormat, got: {other:?}"),
        }
    }

    #[test]
    fn test_file_not_found() {
        let src = "/nonexistent/model.gguf".to_string();
        let flags = OracleFlags::default();
        let result = run(
            Some(&src),
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            flags,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            other => panic!("Expected FileNotFound, got: {other:?}"),
        }
    }

    #[test]
    fn test_offline_hf_rejected() {
        let src = "hf://Qwen/Qwen2.5-Coder-1.5B".to_string();
        let flags = OracleFlags::default();
        let result = run(
            Some(&src),
            None,
            None,
            false,
            false,
            false,
            false,
            true, // offline
            flags,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::NetworkError(msg)) => {
                assert!(msg.contains("offline"));
            }
            other => panic!("Expected NetworkError, got: {other:?}"),
        }
    }

    #[test]
    fn test_load_registry_graceful_degradation() {
        // Should not error even if contracts dir doesn't exist nearby
        let registry = load_registry();
        // This should succeed (might be empty or populated depending on CWD)
        assert!(registry.is_ok());
    }

    #[test]
    fn test_tensor_compliance_entry_serialize() {
        let entry = TensorComplianceEntry {
            name: "model.embed_tokens.weight".to_string(),
            present: true,
            dtype: Some("F16".to_string()),
            shape: Some(vec![151936, 1536]),
            note: None,
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("embed_tokens"));
        assert!(!json.contains("note")); // skip_serializing_if
    }

    // ========================================================================
    // Phase 1: Statistical Analysis Tests
    // ========================================================================

    fn make_test_size() -> ModelSizeConfig {
        ModelSizeConfig {
            parameters: "1.5B".to_string(),
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            intermediate_dim: 8960,
            vocab_size: 151936,
            max_position_embeddings: 32768,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-6,
        }
    }

    fn make_test_constraints() -> ModelConstraints {
        ModelConstraints {
            attention_type: AttentionType::Gqa,
            activation: aprender::format::model_family::Activation::Silu,
            norm_type: NormType::RmsNorm,
            has_bias: true,
            tied_embeddings: false,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
        }
    }

    #[test]
    fn test_gqa_analysis() {
        let size = make_test_size();
        let (ratio, reduction) = compute_gqa_analysis(&size);
        // 2 kv heads / 12 heads = 1/6
        assert!((ratio - 1.0 / 6.0).abs() < 0.01);
        assert!((reduction - 5.0 / 6.0).abs() < 0.01);
    }

    #[test]
    fn test_gqa_analysis_mha() {
        let mut size = make_test_size();
        size.num_kv_heads = size.num_heads; // MHA: ratio = 1.0
        let (ratio, reduction) = compute_gqa_analysis(&size);
        assert!((ratio - 1.0).abs() < 0.01);
        assert!(reduction.abs() < 0.01);
    }

    #[test]
    fn test_param_count_nonzero() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let params = compute_param_count(&size, &constraints);
        assert!(params > 0);
        // Qwen2 1.5B should be in the ballpark of 1.5B params
        assert!(params > 1_000_000_000, "params too small: {params}");
        assert!(params < 3_000_000_000, "params too large: {params}");
    }

    #[test]
    fn test_memory_estimates() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let (f16_mb, q4_mb) = compute_memory_estimates(&size, &constraints);
        // F16 should be about 2x Q4
        assert!(f16_mb > q4_mb * 3.0, "F16 should be much larger than Q4");
        assert!(f16_mb > 0.0);
        assert!(q4_mb > 0.0);
    }

    #[test]
    fn test_kv_cache() {
        let size = make_test_size();
        let (per_token, cache_4k) = compute_kv_cache(&size);
        assert!(per_token > 0);
        assert!(cache_4k > 0.0);
        // Per-token should be 2 * 28 * 2 * 128 * 2 = 28672 bytes
        assert_eq!(per_token, 2 * 28 * 2 * 128 * 2);
    }

    #[test]
    fn test_ffn_analysis() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let (ratio, explanation) = compute_ffn_analysis(&size, &constraints);
        // 8960 / 1536 ≈ 5.83
        assert!(
            ratio > 5.0 && ratio < 6.5,
            "FFN ratio {ratio} out of expected range"
        );
        assert!(explanation.contains("SwiGLU"));
    }

    #[test]
    fn test_rope_analysis() {
        let size = make_test_size();
        let (wavelength, ctx) = compute_rope_analysis(&size);
        assert!(wavelength > 0.0);
        assert_eq!(ctx, 32768);
    }

    #[test]
    fn test_flops_estimate() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let (attn, ffn) = compute_flops_estimate(&size, &constraints);
        assert!(attn > 0);
        assert!(ffn > 0);
    }

    #[test]
    fn test_statistical_analysis_complete() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);

        // Verify all fields are populated sensibly
        assert!(stats.gqa_ratio > 0.0 && stats.gqa_ratio <= 1.0);
        assert!(stats.kv_cache_reduction >= 0.0 && stats.kv_cache_reduction < 1.0);
        assert!(stats.model_params > 0);
        assert!(stats.model_size_f16_mb > 0.0);
        assert!(stats.model_size_q4_mb > 0.0);
        assert!(stats.kv_cache_per_token_bytes > 0);
        assert!(stats.kv_cache_4k_mb > 0.0);
        assert!(stats.ffn_expansion_ratio > 1.0);
        assert!(!stats.ffn_type_explanation.is_empty());
        assert!(stats.rope_max_wavelength > 0.0);
        assert!(stats.effective_context_window > 0);
        assert!(stats.attention_flops_per_token > 0);
        assert!(stats.ffn_flops_per_token > 0);

        // Verify JSON serialization
        let json = serde_json::to_string(&stats).expect("serialize stats");
        assert!(json.contains("gqa_ratio"));
        assert!(json.contains("model_params"));
    }

    // ========================================================================
    // Phase 3: Cross-Validation Tests
    // ========================================================================

    #[test]
    fn test_cross_validation_match() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config: serde_json::Value = serde_json::json!({
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "intermediate_size": 8960,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "model_type": "qwen2"
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        assert!(
            cv.mismatches.is_empty(),
            "Expected no mismatches, got: {:?}",
            cv.mismatches
        );
        assert!(!cv.matches.is_empty(), "Expected matches");
    }

    #[test]
    fn test_cross_validation_mismatch() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config: serde_json::Value = serde_json::json!({
            "hidden_size": 2048,  // MISMATCH: 2048 vs 1536
            "num_hidden_layers": 28,
            "model_type": "qwen2"
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        assert!(
            cv.mismatches.iter().any(|e| e.field == "hidden_dim"),
            "Expected hidden_dim mismatch, got: {:?}",
            cv.mismatches
        );
    }

    // ========================================================================
    // Phase 4: Architecture Explanation Tests
    // ========================================================================

    #[test]
    fn test_architecture_explanation() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.attention_explanation.contains("GQA"));
        assert!(expl.ffn_explanation.contains("SwiGLU"));
        assert!(expl.norm_explanation.contains("RMSNorm"));
        assert!(expl.positional_explanation.contains("RoPE"));
        assert!(expl.scaling_analysis.contains("parameters"));
    }

    // ========================================================================
    // Phase 5: Kernel Compatibility Tests
    // ========================================================================

    #[test]
    fn test_kernel_compatibility() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert_eq!(kern.supported_quantizations.len(), 4);
        assert!(kern.supported_quantizations.iter().all(|q| q.supported));
        assert!(kern.attention_kernel.contains("GQA"));
        assert!(kern.ffn_kernel.contains("SwiGLU"));
        assert!(kern.estimated_tps_cpu.is_some());
        assert!(kern.estimated_tps_gpu.is_some());
        assert!(kern.memory_required_mb > 0.0);
        assert!(!kern.notes.is_empty());
    }

    // ========================================================================
    // Phase 6: OracleFlags Tests
    // ========================================================================

    #[test]
    fn test_oracle_flags_default() {
        let flags = OracleFlags::default();
        assert!(!flags.show_stats());
        assert!(!flags.show_explain());
        assert!(!flags.show_kernels());
        assert!(!flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_full() {
        let flags = OracleFlags {
            full: true,
            ..OracleFlags::default()
        };
        assert!(flags.show_stats());
        assert!(flags.show_explain());
        assert!(flags.show_kernels());
        assert!(flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_individual() {
        let flags = OracleFlags {
            stats: true,
            ..OracleFlags::default()
        };
        assert!(flags.show_stats());
        assert!(!flags.show_explain());
        assert!(!flags.show_kernels());
        assert!(!flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_explain_only() {
        let flags = OracleFlags {
            explain: true,
            ..OracleFlags::default()
        };
        assert!(!flags.show_stats());
        assert!(flags.show_explain());
        assert!(!flags.show_kernels());
        assert!(!flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_kernels_only() {
        let flags = OracleFlags {
            kernels: true,
            ..OracleFlags::default()
        };
        assert!(!flags.show_stats());
        assert!(!flags.show_explain());
        assert!(flags.show_kernels());
        assert!(!flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_validate_only() {
        let flags = OracleFlags {
            validate: true,
            ..OracleFlags::default()
        };
        assert!(!flags.show_stats());
        assert!(!flags.show_explain());
        assert!(!flags.show_kernels());
        assert!(flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_debug() {
        let flags = OracleFlags::default();
        let debug = format!("{flags:?}");
        assert!(debug.contains("OracleFlags"));
    }

    #[test]
    fn test_oracle_flags_clone() {
        let flags = OracleFlags {
            stats: true,
            explain: true,
            kernels: false,
            validate: false,
            full: false,
        };
        let cloned = flags;
        assert!(cloned.show_stats());
        assert!(cloned.show_explain());
    }

    // ========================================================================
    // GQA Analysis Edge Cases
    // ========================================================================

    #[test]
    fn test_gqa_analysis_zero_heads() {
        let mut size = make_test_size();
        size.num_heads = 0;
        let (ratio, reduction) = compute_gqa_analysis(&size);
        assert_eq!(ratio, 0.0);
        assert_eq!(reduction, 0.0);
    }

    #[test]
    fn test_gqa_analysis_single_kv_head() {
        let mut size = make_test_size();
        size.num_heads = 32;
        size.num_kv_heads = 1; // MQA-like
        let (ratio, reduction) = compute_gqa_analysis(&size);
        assert!((ratio - 1.0 / 32.0).abs() < 0.001);
        assert!((reduction - 31.0 / 32.0).abs() < 0.001);
    }

    #[test]
    fn test_gqa_analysis_equal_heads() {
        let mut size = make_test_size();
        size.num_heads = 32;
        size.num_kv_heads = 32; // MHA
        let (ratio, reduction) = compute_gqa_analysis(&size);
        assert!((ratio - 1.0).abs() < 0.001);
        assert!(reduction.abs() < 0.001);
    }

    // ========================================================================
    // Param Count Edge Cases
    // ========================================================================

    #[test]
    fn test_param_count_no_bias() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.has_bias = false;
        let params_no_bias = compute_param_count(&size, &constraints);

        constraints.has_bias = true;
        let params_with_bias = compute_param_count(&size, &constraints);

        assert!(
            params_with_bias > params_no_bias,
            "Bias should add parameters"
        );
    }

    #[test]
    fn test_param_count_tied_embeddings() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.tied_embeddings = false;
        let params_untied = compute_param_count(&size, &constraints);

        constraints.tied_embeddings = true;
        let params_tied = compute_param_count(&size, &constraints);

        // Tied embeddings removes lm_head = vocab_size * hidden_dim
        let lm_head_params = (size.vocab_size as u64) * (size.hidden_dim as u64);
        assert_eq!(params_untied - params_tied, lm_head_params);
    }

    #[test]
    fn test_param_count_gated_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GatedMlp;
        let params_gated = compute_param_count(&size, &constraints);

        constraints.mlp_type = MlpType::SwiGlu;
        let params_swiglu = compute_param_count(&size, &constraints);

        // Both gated, should have same FFN param count
        assert_eq!(params_gated, params_swiglu);
    }

    #[test]
    fn test_param_count_standard_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GeluMlp;
        let params_standard = compute_param_count(&size, &constraints);

        constraints.mlp_type = MlpType::SwiGlu;
        let params_gated = compute_param_count(&size, &constraints);

        // Standard uses 2 matrices, gated uses 3
        assert!(
            params_gated > params_standard,
            "Gated should have more params"
        );
    }

    #[test]
    fn test_param_count_minimal_model() {
        let size = ModelSizeConfig {
            parameters: "tiny".to_string(),
            hidden_dim: 64,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 128,
            vocab_size: 100,
            max_position_embeddings: 512,
            head_dim: 32,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        };
        let constraints = ModelConstraints {
            attention_type: AttentionType::Mha,
            activation: aprender::format::model_family::Activation::Silu,
            norm_type: NormType::RmsNorm,
            has_bias: false,
            tied_embeddings: true,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::GeluMlp,
        };
        let params = compute_param_count(&size, &constraints);
        assert!(params > 0, "Even minimal model should have params");
        // Embedding: 100*64 = 6400, no lm_head (tied), 1 layer with small dims
        assert!(
            params < 1_000_000,
            "Tiny model shouldn't have millions of params"
        );
    }

    // ========================================================================
    // Memory Estimates Edge Cases
    // ========================================================================

    #[test]
    fn test_memory_estimates_f16_is_4x_q4() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let (f16_mb, q4_mb) = compute_memory_estimates(&size, &constraints);
        // F16 = 2 bytes/param, Q4 = 0.5 bytes/param, ratio = 4
        assert!(
            (f16_mb / q4_mb - 4.0).abs() < 0.01,
            "F16/Q4 ratio should be ~4"
        );
    }

    // ========================================================================
    // KV Cache Edge Cases
    // ========================================================================

    #[test]
    fn test_kv_cache_zero_layers() {
        let mut size = make_test_size();
        size.num_layers = 0;
        let (per_token, cache_4k) = compute_kv_cache(&size);
        assert_eq!(per_token, 0);
        assert_eq!(cache_4k, 0.0);
    }

    #[test]
    fn test_kv_cache_zero_kv_heads() {
        let mut size = make_test_size();
        size.num_kv_heads = 0;
        let (per_token, cache_4k) = compute_kv_cache(&size);
        assert_eq!(per_token, 0);
        assert_eq!(cache_4k, 0.0);
    }

    #[test]
    fn test_kv_cache_large_context() {
        let size = make_test_size();
        let (per_token, cache_4k) = compute_kv_cache(&size);
        // 4K cache = per_token * 4096 / (1024*1024)
        let expected_4k = (per_token as f64 * 4096.0) / (1024.0 * 1024.0);
        assert!((cache_4k - expected_4k).abs() < 0.001);
    }

    // ========================================================================
    // FFN Analysis Edge Cases
    // ========================================================================

    #[test]
    fn test_ffn_analysis_zero_hidden_dim() {
        let mut size = make_test_size();
        size.hidden_dim = 0;
        let constraints = make_test_constraints();
        let (ratio, explanation) = compute_ffn_analysis(&size, &constraints);
        assert_eq!(ratio, 0.0);
        assert!(explanation.is_empty());
    }

    #[test]
    fn test_ffn_analysis_gated_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GatedMlp;
        let (ratio, explanation) = compute_ffn_analysis(&size, &constraints);
        assert!(ratio > 0.0);
        assert!(explanation.contains("GeGLU"));
    }

    #[test]
    fn test_ffn_analysis_gelu_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GeluMlp;
        let (ratio, explanation) = compute_ffn_analysis(&size, &constraints);
        assert!(ratio > 0.0);
        assert!(explanation.contains("Standard GELU"));
    }

    // ========================================================================
    // RoPE Analysis Edge Cases
    // ========================================================================

    #[test]
    fn test_rope_analysis_zero_theta() {
        let mut size = make_test_size();
        size.rope_theta = 0.0;
        let (wavelength, ctx) = compute_rope_analysis(&size);
        assert_eq!(wavelength, 0.0);
        assert_eq!(ctx, size.max_position_embeddings);
    }

    #[test]
    fn test_rope_analysis_standard_theta() {
        let mut size = make_test_size();
        size.rope_theta = 10000.0;
        let (wavelength, _) = compute_rope_analysis(&size);
        let expected = 2.0 * std::f64::consts::PI * 10000.0;
        assert!((wavelength - expected).abs() < 1.0);
    }

    // ========================================================================
    // FLOPS Estimate Edge Cases
    // ========================================================================

    #[test]
    fn test_flops_estimate_standard_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GeluMlp;
        let (attn_gelu, ffn_gelu) = compute_flops_estimate(&size, &constraints);

        constraints.mlp_type = MlpType::SwiGlu;
        let (attn_swiglu, ffn_swiglu) = compute_flops_estimate(&size, &constraints);

        // Attention FLOPS should be the same regardless of MLP type
        assert_eq!(attn_gelu, attn_swiglu);
        // Gated has 50% more FFN FLOPS (3 matmuls vs 2)
        assert_eq!(ffn_swiglu, ffn_gelu * 3 / 2);
    }

    #[test]
    fn test_flops_estimate_zero_layers() {
        let mut size = make_test_size();
        size.num_layers = 0;
        let constraints = make_test_constraints();
        let (attn, ffn) = compute_flops_estimate(&size, &constraints);
        assert_eq!(attn, 0);
        assert_eq!(ffn, 0);
    }

    // ========================================================================
    // Cross-Validation Extended Tests
    // ========================================================================

    #[test]
    fn test_cross_validation_empty_hf_config() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({});

        let cv = cross_validate(&size, &constraints, &hf_config);
        assert!(cv.matches.is_empty());
        assert!(cv.mismatches.is_empty());
        // All contract fields are contract_only since HF has none
        assert!(!cv.contract_only.is_empty());
    }

    #[test]
    fn test_cross_validation_rope_theta_approximate() {
        let mut size = make_test_size();
        size.rope_theta = 1_000_000.0;
        let constraints = make_test_constraints();
        // Within 1.0 absolute tolerance => "match"
        let hf_config = serde_json::json!({
            "rope_theta": 1_000_000.5
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let rope_entry = cv.matches.iter().find(|e| e.field == "rope_theta");
        assert!(
            rope_entry.is_some(),
            "rope_theta should match within tolerance"
        );
    }

    #[test]
    fn test_cross_validation_rope_theta_mismatch() {
        let mut size = make_test_size();
        size.rope_theta = 1_000_000.0;
        let constraints = make_test_constraints();
        // Way off => "mismatch"
        let hf_config = serde_json::json!({
            "rope_theta": 500_000.0
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let rope_mismatch = cv.mismatches.iter().find(|e| e.field == "rope_theta");
        assert!(rope_mismatch.is_some(), "rope_theta should mismatch");
    }

    #[test]
    fn test_cross_validation_norm_eps_match() {
        let mut size = make_test_size();
        size.norm_eps = 1e-6;
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "rms_norm_eps": 1e-6
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let eps_entry = cv.matches.iter().find(|e| e.field == "norm_eps");
        assert!(eps_entry.is_some(), "norm_eps should match");
    }

    #[test]
    fn test_cross_validation_norm_eps_layer_norm_variant() {
        let mut size = make_test_size();
        size.norm_eps = 1e-5;
        let constraints = make_test_constraints();
        // Uses layer_norm_eps key instead of rms_norm_eps
        let hf_config = serde_json::json!({
            "layer_norm_eps": 1e-5
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let eps_entry = cv.matches.iter().find(|e| e.field == "norm_eps");
        assert!(
            eps_entry.is_some(),
            "norm_eps should match via layer_norm_eps key"
        );
    }

    #[test]
    fn test_cross_validation_norm_eps_epsilon_variant() {
        let mut size = make_test_size();
        size.norm_eps = 1e-5;
        let constraints = make_test_constraints();
        // Uses layer_norm_epsilon key
        let hf_config = serde_json::json!({
            "layer_norm_epsilon": 1e-5
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let eps_entry = cv.matches.iter().find(|e| e.field == "norm_eps");
        assert!(
            eps_entry.is_some(),
            "norm_eps should match via layer_norm_epsilon key"
        );
    }

    #[test]
    fn test_cross_validation_norm_eps_mismatch() {
        let mut size = make_test_size();
        size.norm_eps = 1e-6;
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "rms_norm_eps": 1e-5  // Different
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let eps_mismatch = cv.mismatches.iter().find(|e| e.field == "norm_eps");
        assert!(eps_mismatch.is_some(), "norm_eps should mismatch");
    }

    #[test]
    fn test_cross_validation_hf_only_interesting_fields() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "rope_scaling": {"type": "dynamic"},
            "sliding_window": 4096,
            "attention_dropout": 0.0,
            "use_cache": true,
            "tie_word_embeddings": false,
            "some_other_field": 42
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        // Should find 5 interesting HF-only fields
        assert!(
            cv.hf_only.len() >= 5,
            "Expected at least 5 HF-only fields, got {}",
            cv.hf_only.len()
        );
    }

    #[test]
    fn test_cross_validation_model_type_info() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "model_type": "qwen2"
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let model_type_entry = cv.matches.iter().find(|e| e.field == "model_type");
        assert!(model_type_entry.is_some());
        assert_eq!(model_type_entry.expect("exists").status, "info");
    }

    #[test]
    fn test_cross_validation_hf_string_value() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        // String-valued HF field instead of number
        let hf_config = serde_json::json!({
            "hidden_size": "1536"
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        // Should match since both are "1536"
        let entry = cv.matches.iter().find(|e| e.field == "hidden_dim");
        assert!(entry.is_some(), "String-valued HF field should match");
    }

    #[test]
    fn test_cross_validation_contract_only_fields() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        // HF config missing most fields
        let hf_config = serde_json::json!({
            "hidden_size": 1536
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        // Should have several contract_only fields
        assert!(
            cv.contract_only.len() >= 5,
            "Expected many contract-only fields"
        );
    }

    // ========================================================================
    // Architecture Explanation Extended Tests
    // ========================================================================

    #[test]
    fn test_architecture_explanation_mha() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.attention_type = AttentionType::Mha;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.attention_explanation.contains("Multi-Head Attention"));
        assert!(expl
            .attention_explanation
            .contains(&size.num_heads.to_string()));
    }

    #[test]
    fn test_architecture_explanation_mqa() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.attention_type = AttentionType::Mqa;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.attention_explanation.contains("Multi-Query Attention"));
    }

    #[test]
    fn test_architecture_explanation_geglu() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GatedMlp;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.ffn_explanation.contains("GeGLU"));
    }

    #[test]
    fn test_architecture_explanation_gelu_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GeluMlp;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.ffn_explanation.contains("Standard GELU MLP"));
    }

    #[test]
    fn test_architecture_explanation_layer_norm() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.norm_type = NormType::LayerNorm;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.norm_explanation.contains("LayerNorm"));
    }

    #[test]
    fn test_architecture_explanation_absolute_pos() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.positional_encoding = PositionalEncoding::Absolute;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.positional_explanation.contains("Absolute position"));
    }

    #[test]
    fn test_architecture_explanation_alibi_pos() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.positional_encoding = PositionalEncoding::Alibi;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.positional_explanation.contains("ALiBi"));
    }

    #[test]
    fn test_architecture_explanation_relative_pos() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.positional_encoding = PositionalEncoding::Relative;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.positional_explanation.contains("Relative"));
    }

    #[test]
    fn test_architecture_explanation_scaling_analysis() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.scaling_analysis.contains("parameters"));
        assert!(expl.scaling_analysis.contains("FLOPs"));
        assert!(expl.scaling_analysis.contains("Chinchilla"));
    }

    // ========================================================================
    // Kernel Compatibility Extended Tests
    // ========================================================================

    #[test]
    fn test_kernel_compatibility_mha() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.attention_type = AttentionType::Mha;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert!(kern.attention_kernel.contains("MHA"));
    }

    #[test]
    fn test_kernel_compatibility_mqa() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.attention_type = AttentionType::Mqa;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert!(kern.attention_kernel.contains("MQA"));
    }

    #[test]
    fn test_kernel_compatibility_gelu_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GeluMlp;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert!(kern.ffn_kernel.contains("standard GELU"));
    }

    #[test]
    fn test_kernel_compatibility_geglu() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GatedMlp;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert!(kern.ffn_kernel.contains("GeGLU"));
    }

    #[test]
    fn test_kernel_compatibility_no_bias() {
        let mut size = make_test_size();
        size.num_kv_heads = size.num_heads; // MHA to remove GQA note
        let mut constraints = make_test_constraints();
        constraints.has_bias = false;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        // No bias note should be absent
        assert!(!kern.notes.iter().any(|n| n.contains("Bias")));
    }

    #[test]
    fn test_kernel_compatibility_with_bias() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.has_bias = true;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert!(kern.notes.iter().any(|n| n.contains("Bias")));
    }

    #[test]
    fn test_kernel_compatibility_gqa_note() {
        let mut size = make_test_size();
        size.num_heads = 32;
        size.num_kv_heads = 8;
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert!(kern.notes.iter().any(|n| n.contains("GQA")));
    }

    #[test]
    fn test_kernel_compatibility_equal_heads_no_gqa_note() {
        let mut size = make_test_size();
        size.num_heads = 12;
        size.num_kv_heads = 12; // MHA
        let mut constraints = make_test_constraints();
        constraints.has_bias = false;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        // Should only have layout note, no GQA or bias notes
        assert!(
            kern.notes.len() == 1,
            "Expected only layout note, got: {:?}",
            kern.notes
        );
        assert!(kern.notes[0].contains("ROW-MAJOR"));
    }

    #[test]
    fn test_kernel_compatibility_quantization_sizes() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        // F16 (16bpw) should be largest, Q4_K_M (4.5bpw) should be smallest
        let f16_entry = kern
            .supported_quantizations
            .iter()
            .find(|q| q.format == "F16")
            .expect("F16");
        let q4_entry = kern
            .supported_quantizations
            .iter()
            .find(|q| q.format == "Q4_K_M")
            .expect("Q4_K_M");
        assert!(
            f16_entry.estimated_size_mb > q4_entry.estimated_size_mb,
            "F16 ({:.1} MB) should be larger than Q4_K_M ({:.1} MB)",
            f16_entry.estimated_size_mb,
            q4_entry.estimated_size_mb
        );

        // All sizes should be positive
        for q in &kern.supported_quantizations {
            assert!(
                q.estimated_size_mb > 0.0,
                "{} size should be positive",
                q.format
            );
        }
    }

    #[test]
    fn test_kernel_compatibility_tps_estimates() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        let cpu_tps = kern.estimated_tps_cpu.expect("should have CPU estimate");
        let gpu_tps = kern.estimated_tps_gpu.expect("should have GPU estimate");
        // GPU should be much faster than CPU (900 GB/s vs 50 GB/s bandwidth)
        assert!(
            gpu_tps > cpu_tps * 10.0,
            "GPU should be >10x faster than CPU"
        );
    }

    #[test]
    fn test_kernel_compatibility_memory_includes_kv() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        // Memory should be > Q4 model size (includes KV cache)
        assert!(kern.memory_required_mb > stats.model_size_q4_mb);
    }

    // ========================================================================
    // format_params Extended Tests
    // ========================================================================

    #[test]
    fn test_format_params_boundary_1000() {
        assert_eq!(format_params(999), "999");
        assert_eq!(format_params(1000), "1.0K");
    }

    #[test]
    fn test_format_params_boundary_1m() {
        assert_eq!(format_params(999_999), "1000.0K");
        assert_eq!(format_params(1_000_000), "1.0M");
    }

    #[test]
    fn test_format_params_boundary_1b() {
        assert_eq!(format_params(999_999_999), "1000.0M");
        assert_eq!(format_params(1_000_000_000), "1.0B");
    }

    #[test]
    fn test_format_params_zero() {
        assert_eq!(format_params(0), "0");
    }

    // ========================================================================
    // build_statistical_analysis Integration Tests
    // ========================================================================

    #[test]
    fn test_statistical_analysis_with_mha() {
        let mut size = make_test_size();
        size.num_kv_heads = size.num_heads;
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);

        assert!((stats.gqa_ratio - 1.0).abs() < 0.01);
        assert!(stats.kv_cache_reduction.abs() < 0.01);
    }

    #[test]
    fn test_statistical_analysis_serialization() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);

        let json = serde_json::to_string_pretty(&stats).expect("serialize");
        assert!(json.contains("gqa_ratio"));
        assert!(json.contains("kv_cache_reduction"));
        assert!(json.contains("model_params"));
        assert!(json.contains("model_size_f16_mb"));
        assert!(json.contains("model_size_q4_mb"));
        assert!(json.contains("kv_cache_per_token_bytes"));
        assert!(json.contains("kv_cache_4k_mb"));
        assert!(json.contains("ffn_expansion_ratio"));
        assert!(json.contains("ffn_type_explanation"));
        assert!(json.contains("rope_max_wavelength"));
        assert!(json.contains("effective_context_window"));
        assert!(json.contains("attention_flops_per_token"));
        assert!(json.contains("ffn_flops_per_token"));
    }

    // ========================================================================
    // Serialization Tests for Report Types
    // ========================================================================

    #[test]
    fn test_family_info_serialize() {
        let fi = FamilyInfo {
            name: "llama".to_string(),
            display_name: "LLaMA".to_string(),
            vendor: "Meta".to_string(),
            architectures: vec!["LlamaForCausalLM".to_string()],
            constraints: ConstraintsSummary {
                attention: "GQA".to_string(),
                activation: "SiLU".to_string(),
                norm: "RMSNorm".to_string(),
                bias: false,
                tied_embeddings: false,
                mlp: "SwiGLU".to_string(),
                positional_encoding: "RoPE".to_string(),
            },
            chat_template_format: None,
        };
        let json = serde_json::to_string(&fi).expect("serialize");
        assert!(json.contains("\"name\":\"llama\""));
        // chat_template_format should be skipped
        assert!(!json.contains("chat_template_format"));
    }

    #[test]
    fn test_size_variant_info_serialize() {
        let svi = SizeVariantInfo {
            name: "7b".to_string(),
            parameters: "7B".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_dim: 14336,
            vocab_size: 32000,
            expected_tensor_count: 291,
        };
        let json = serde_json::to_string(&svi).expect("serialize");
        assert!(json.contains("\"hidden_dim\":4096"));
        assert!(json.contains("\"expected_tensor_count\":291"));
    }

    #[test]
    fn test_format_info_serialize() {
        let fi = FormatInfo {
            format_type: "GGUF".to_string(),
            file_size: 4_000_000_000,
            tensor_count: 291,
            total_params: 7_000_000_000,
            quantization: Some("Q4_K_M".to_string()),
            architecture: Some("llama".to_string()),
        };
        let json = serde_json::to_string(&fi).expect("serialize");
        assert!(json.contains("\"format_type\":\"GGUF\""));
        assert!(json.contains("\"quantization\":\"Q4_K_M\""));
    }

    #[test]
    fn test_format_info_serialize_no_optional() {
        let fi = FormatInfo {
            format_type: "SafeTensors".to_string(),
            file_size: 2_000_000_000,
            tensor_count: 200,
            total_params: 1_500_000_000,
            quantization: None,
            architecture: None,
        };
        let json = serde_json::to_string(&fi).expect("serialize");
        assert!(!json.contains("quantization"));
        assert!(!json.contains("architecture"));
    }

    #[test]
    fn test_certification_info_serialize() {
        let ci = CertificationInfo {
            status: "PENDING".to_string(),
            playbook_path: Some("/playbooks/7b.yaml".to_string()),
        };
        let json = serde_json::to_string(&ci).expect("serialize");
        assert!(json.contains("PENDING"));
        assert!(json.contains("playbook_path"));
    }

    #[test]
    fn test_certification_info_no_playbook() {
        let ci = CertificationInfo {
            status: "APPROVED".to_string(),
            playbook_path: None,
        };
        let json = serde_json::to_string(&ci).expect("serialize");
        assert!(!json.contains("playbook_path"));
    }

    #[test]
    fn test_cross_validation_entry_serialize() {
        let entry = CrossValidationEntry {
            field: "hidden_dim".to_string(),
            contract_value: "1536".to_string(),
            hf_value: "1536".to_string(),
            status: "match".to_string(),
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("\"status\":\"match\""));
    }

    #[test]
    fn test_cross_validation_serialize() {
        let cv = CrossValidation {
            matches: vec![CrossValidationEntry {
                field: "hidden_dim".to_string(),
                contract_value: "1536".to_string(),
                hf_value: "1536".to_string(),
                status: "match".to_string(),
            }],
            mismatches: vec![],
            contract_only: vec!["norm_eps=1e-6".to_string()],
            hf_only: vec!["rope_scaling=dynamic".to_string()],
        };
        let json = serde_json::to_string(&cv).expect("serialize");
        assert!(json.contains("matches"));
        assert!(json.contains("mismatches"));
        assert!(json.contains("contract_only"));
        assert!(json.contains("hf_only"));
    }

    #[test]
    fn test_kernel_compatibility_serialize() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![QuantizationSupport {
                format: "Q4_K_M".to_string(),
                supported: true,
                kernel: "fused_q4k".to_string(),
                bits_per_weight: 4.5,
                estimated_size_mb: 500.0,
            }],
            attention_kernel: "GQA fused".to_string(),
            ffn_kernel: "SwiGLU fused".to_string(),
            estimated_tps_cpu: Some(100.0),
            estimated_tps_gpu: Some(1000.0),
            memory_required_mb: 600.0,
            notes: vec!["ROW-MAJOR".to_string()],
        };
        let json = serde_json::to_string(&kern).expect("serialize");
        assert!(json.contains("supported_quantizations"));
        assert!(json.contains("estimated_tps_cpu"));
    }

    #[test]
    fn test_kernel_compatibility_no_tps() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![],
            attention_kernel: "none".to_string(),
            ffn_kernel: "none".to_string(),
            estimated_tps_cpu: None,
            estimated_tps_gpu: None,
            memory_required_mb: 0.0,
            notes: vec![],
        };
        let json = serde_json::to_string(&kern).expect("serialize");
        assert!(!json.contains("estimated_tps_cpu"));
        assert!(!json.contains("estimated_tps_gpu"));
    }

    #[test]
    fn test_architecture_explanation_serialize() {
        let expl = ArchitectureExplanation {
            attention_explanation: "GQA with ratio 0.17".to_string(),
            ffn_explanation: "SwiGLU gated".to_string(),
            norm_explanation: "RMSNorm".to_string(),
            positional_explanation: "RoPE theta=1000000".to_string(),
            scaling_analysis: "1.5B params".to_string(),
        };
        let json = serde_json::to_string(&expl).expect("serialize");
        assert!(json.contains("attention_explanation"));
        assert!(json.contains("ffn_explanation"));
        assert!(json.contains("norm_explanation"));
        assert!(json.contains("positional_explanation"));
        assert!(json.contains("scaling_analysis"));
    }

    #[test]
    fn test_quantization_support_serialize() {
        let qs = QuantizationSupport {
            format: "F16".to_string(),
            supported: true,
            kernel: "trueno::f16_matvec".to_string(),
            bits_per_weight: 16.0,
            estimated_size_mb: 3000.0,
        };
        let json = serde_json::to_string(&qs).expect("serialize");
        assert!(json.contains("\"format\":\"F16\""));
        assert!(json.contains("\"supported\":true"));
    }

    #[test]
    fn test_huggingface_data_serialize() {
        let hf = HuggingFaceData {
            repo: "Qwen/Qwen2.5-1.5B".to_string(),
            model_type: Some("qwen2".to_string()),
            pipeline_tag: Some("text-generation".to_string()),
            downloads: Some(1000),
            config_fields: serde_json::json!({"hidden_size": 1536}),
            generation_config: None,
        };
        let json = serde_json::to_string(&hf).expect("serialize");
        assert!(json.contains("\"repo\":\"Qwen/Qwen2.5-1.5B\""));
        assert!(!json.contains("generation_config"));
    }

    #[test]
    fn test_huggingface_data_all_none() {
        let hf = HuggingFaceData {
            repo: "test/model".to_string(),
            model_type: None,
            pipeline_tag: None,
            downloads: None,
            config_fields: serde_json::json!({}),
            generation_config: None,
        };
        let json = serde_json::to_string(&hf).expect("serialize");
        assert!(!json.contains("model_type"));
        assert!(!json.contains("pipeline_tag"));
        assert!(!json.contains("downloads"));
    }

    // ========================================================================
    // Report with All Optional Fields Populated
    // ========================================================================

    #[test]
    fn test_report_with_all_fields() {
        let report = ModelOracleReport {
            source: "test.gguf".to_string(),
            mode: OracleMode::Local,
            family: Some(FamilyInfo {
                name: "qwen2".to_string(),
                display_name: "Qwen2".to_string(),
                vendor: "Alibaba".to_string(),
                architectures: vec!["Qwen2ForCausalLM".to_string()],
                constraints: ConstraintsSummary {
                    attention: "GQA".to_string(),
                    activation: "SiLU".to_string(),
                    norm: "RMSNorm".to_string(),
                    bias: true,
                    tied_embeddings: false,
                    mlp: "SwiGLU".to_string(),
                    positional_encoding: "RoPE".to_string(),
                },
                chat_template_format: Some("chatml".to_string()),
            }),
            size_variant: None,
            format: None,
            compliance: Some(ComplianceResult {
                is_compliant: false,
                tensor_count_match: false,
                missing_tensors: vec!["layer.0.attn.q_proj.weight".to_string()],
                unexpected_tensors: vec!["extra.weight".to_string()],
            }),
            certification: Some(CertificationInfo {
                status: "PENDING".to_string(),
                playbook_path: Some("/playbooks/1.5b.yaml".to_string()),
            }),
            tensors: Some(vec![TensorComplianceEntry {
                name: "embed.weight".to_string(),
                present: true,
                dtype: Some("F16".to_string()),
                shape: Some(vec![151936, 1536]),
                note: Some("embedding".to_string()),
            }]),
            stats: None,
            explanation: None,
            kernel_compatibility: None,
            cross_validation: Some(CrossValidation {
                matches: vec![],
                mismatches: vec![],
                contract_only: vec![],
                hf_only: vec![],
            }),
            hf_data: None,
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("compliance"));
        assert!(json.contains("certification"));
        assert!(json.contains("tensors"));
        assert!(json.contains("cross_validation"));
    }

    // ========================================================================
    // OracleMode Tests
    // ========================================================================

    #[test]
    fn test_oracle_mode_debug() {
        let mode = OracleMode::Local;
        let debug = format!("{mode:?}");
        assert!(debug.contains("Local"));
    }

    #[test]
    fn test_oracle_mode_clone() {
        let mode = OracleMode::HuggingFace;
        let cloned = mode.clone();
        let json = serde_json::to_string(&cloned).expect("serialize");
        assert_eq!(json, "\"hugging_face\"");
    }

    // ========================================================================
    // build_certification Tests
    // ========================================================================

    #[test]
    fn test_build_certification_no_cert_config() {
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: aprender::format::model_family::TensorTemplate {
                embedding: String::new(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let cert = build_certification(&config, Some("7b"));
        assert!(cert.is_none());
    }

    #[test]
    fn test_build_certification_no_size() {
        use aprender::format::model_family::CertificationConfig;
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: aprender::format::model_family::TensorTemplate {
                embedding: String::new(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: Some(CertificationConfig {
                playbook_path: "../playbooks/{size}.yaml".to_string(),
                csv_family_key: "test".to_string(),
                size_categories: HashMap::new(),
            }),
        };

        let cert = build_certification(&config, None);
        assert!(cert.is_some());
        let cert = cert.expect("cert exists");
        assert!(cert.playbook_path.is_none());
    }

    // ========================================================================
    // Tensor Compliance Entry Tests
    // ========================================================================

    #[test]
    fn test_tensor_compliance_entry_missing_tensor() {
        let entry = TensorComplianceEntry {
            name: "model.layers.0.attn.q_proj.weight".to_string(),
            present: false,
            dtype: None,
            shape: None,
            note: Some("MISSING".to_string()),
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("\"present\":false"));
        assert!(json.contains("\"note\":\"MISSING\""));
        // dtype and shape should be skipped
        assert!(!json.contains("\"dtype\""));
        assert!(!json.contains("\"shape\""));
    }

    // ========================================================================
    // Constraints Summary Tests
    // ========================================================================

    #[test]
    fn test_constraints_summary_serialize() {
        let cs = ConstraintsSummary {
            attention: "GQA".to_string(),
            activation: "SiLU".to_string(),
            norm: "RMSNorm".to_string(),
            bias: true,
            tied_embeddings: false,
            mlp: "SwiGLU".to_string(),
            positional_encoding: "RoPE".to_string(),
        };
        let json = serde_json::to_string(&cs).expect("serialize");
        assert!(json.contains("\"bias\":true"));
        assert!(json.contains("\"tied_embeddings\":false"));
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_cross_validation_rope_theta_relative_approximate() {
        // Test the "approximate" path: abs diff > 1.0 but relative diff < 1%
        let mut size = make_test_size();
        size.rope_theta = 100_000.0;
        let constraints = make_test_constraints();
        // abs diff = 500 > 1.0, but relative = 500/100000 = 0.5% < 1%
        let hf_config = serde_json::json!({
            "rope_theta": 100_500.0
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let rope_entry = cv.matches.iter().find(|e| e.field == "rope_theta");
        assert!(
            rope_entry.is_some(),
            "rope_theta should approximately match"
        );
        assert_eq!(
            rope_entry.expect("exists").status,
            "approximate",
            "Status should be 'approximate'"
        );
    }

    #[test]
    fn test_cross_validation_hf_value_bool() {
        // Test when HF config has a non-number, non-string value (e.g. bool/object)
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "hidden_size": true
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        // "true" != "1536" => mismatch
        let entry = cv.mismatches.iter().find(|e| e.field == "hidden_dim");
        assert!(
            entry.is_some(),
            "bool HF value should mismatch with numeric contract value"
        );
    }

    #[test]
    fn test_cross_validation_hf_value_array() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "hidden_size": [1536]
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let entry = cv.mismatches.iter().find(|e| e.field == "hidden_dim");
        assert!(entry.is_some(), "array HF value should mismatch");
    }

    #[test]
    fn test_expand_tensor_template_empty() {
        use std::collections::HashMap;

        let template = aprender::format::model_family::TensorTemplate {
            embedding: String::new(),
            lm_head: None,
            final_norm: None,
            per_layer: HashMap::new(),
        };
        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: template.clone(),
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let names = expand_tensor_template(&template, &config, "7b");
        assert!(names.is_empty(), "Empty template should produce no names");
    }

    #[test]
    fn test_expand_tensor_template_with_globals() {
        use std::collections::HashMap;

        let template = aprender::format::model_family::TensorTemplate {
            embedding: "model.embed_tokens.weight".to_string(),
            lm_head: Some("lm_head.weight".to_string()),
            final_norm: Some("model.norm.weight".to_string()),
            per_layer: HashMap::new(),
        };
        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: template.clone(),
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let names = expand_tensor_template(&template, &config, "nonexistent_size");
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(names.contains(&"lm_head.weight".to_string()));
        assert!(names.contains(&"model.norm.weight".to_string()));
    }

    #[test]
    fn test_expand_tensor_template_with_per_layer() {
        use std::collections::HashMap;

        let mut per_layer = HashMap::new();
        per_layer.insert(
            "q_proj".to_string(),
            Some("model.layers.{n}.self_attn.q_proj.weight".to_string()),
        );
        per_layer.insert(
            "k_proj".to_string(),
            Some("model.layers.{n}.self_attn.k_proj.weight".to_string()),
        );

        let template = aprender::format::model_family::TensorTemplate {
            embedding: "embed.weight".to_string(),
            lm_head: None,
            final_norm: None,
            per_layer,
        };

        let mut size_variants = HashMap::new();
        size_variants.insert(
            "tiny".to_string(),
            ModelSizeConfig {
                parameters: "tiny".to_string(),
                hidden_dim: 64,
                num_layers: 2,
                num_heads: 2,
                num_kv_heads: 2,
                intermediate_dim: 128,
                vocab_size: 100,
                max_position_embeddings: 512,
                head_dim: 32,
                rope_theta: 10000.0,
                norm_eps: 1e-5,
            },
        );

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants,
            constraints: make_test_constraints(),
            tensor_template: template.clone(),
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let names = expand_tensor_template(&template, &config, "tiny");
        // 1 embedding + 2 layers * 2 per-layer tensors = 5
        assert_eq!(names.len(), 5);
        assert!(names.contains(&"embed.weight".to_string()));
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.0.self_attn.k_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.self_attn.k_proj.weight".to_string()));
    }

    #[test]
    fn test_expand_tensor_template_per_layer_with_none_value() {
        use std::collections::HashMap;

        let mut per_layer = HashMap::new();
        per_layer.insert(
            "q_proj".to_string(),
            Some("model.layers.{n}.q.weight".to_string()),
        );
        per_layer.insert("bias".to_string(), None); // Optional tensor not present

        let template = aprender::format::model_family::TensorTemplate {
            embedding: "embed.weight".to_string(),
            lm_head: None,
            final_norm: None,
            per_layer,
        };

        let mut size_variants = HashMap::new();
        size_variants.insert(
            "tiny".to_string(),
            ModelSizeConfig {
                parameters: "tiny".to_string(),
                hidden_dim: 64,
                num_layers: 1,
                num_heads: 2,
                num_kv_heads: 2,
                intermediate_dim: 128,
                vocab_size: 100,
                max_position_embeddings: 512,
                head_dim: 32,
                rope_theta: 10000.0,
                norm_eps: 1e-5,
            },
        );

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants,
            constraints: make_test_constraints(),
            tensor_template: template.clone(),
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let names = expand_tensor_template(&template, &config, "tiny");
        // 1 embedding + 1 layer * 1 per-layer (None skipped by flatten) = 2
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"embed.weight".to_string()));
        assert!(names.contains(&"model.layers.0.q.weight".to_string()));
    }

    #[test]
    fn test_oracle_flags_combined_stats_and_explain() {
        let flags = OracleFlags {
            stats: true,
            explain: true,
            kernels: false,
            validate: false,
            full: false,
        };
        assert!(flags.show_stats());
        assert!(flags.show_explain());
        assert!(!flags.show_kernels());
        assert!(!flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_combined_kernels_and_validate() {
        let flags = OracleFlags {
            stats: false,
            explain: false,
            kernels: true,
            validate: true,
            full: false,
        };
        assert!(!flags.show_stats());
        assert!(!flags.show_explain());
        assert!(flags.show_kernels());
        assert!(flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_full_overrides_individual() {
        let flags = OracleFlags {
            stats: false,
            explain: false,
            kernels: false,
            validate: false,
            full: true,
        };
        // full=true should make all show_* true even if individual flags are false
        assert!(flags.show_stats());
        assert!(flags.show_explain());
        assert!(flags.show_kernels());
        assert!(flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_copy_semantics() {
        let flags = OracleFlags {
            stats: true,
            explain: false,
            kernels: true,
            validate: false,
            full: false,
        };
        let copied = flags;
        // After copy, original should still work (Copy trait)
        assert!(flags.show_stats());
        assert!(copied.show_stats());
        assert!(flags.show_kernels());
        assert!(copied.show_kernels());
    }

    #[test]
    fn test_report_all_none_fields_serialize() {
        let report = ModelOracleReport {
            source: "minimal.gguf".to_string(),
            mode: OracleMode::Local,
            family: None,
            size_variant: None,
            format: None,
            compliance: None,
            certification: None,
            tensors: None,
            stats: None,
            explanation: None,
            kernel_compatibility: None,
            cross_validation: None,
            hf_data: None,
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("\"source\": \"minimal.gguf\""));
        assert!(json.contains("\"mode\": \"local\""));
        // All optional fields should be absent due to skip_serializing_if
        assert!(!json.contains("\"family\""));
        assert!(!json.contains("\"size_variant\""));
        assert!(!json.contains("\"format\""));
        assert!(!json.contains("\"compliance\""));
        assert!(!json.contains("\"certification\""));
        assert!(!json.contains("\"tensors\""));
        assert!(!json.contains("\"stats\""));
        assert!(!json.contains("\"explanation\""));
        assert!(!json.contains("\"kernel_compatibility\""));
        assert!(!json.contains("\"cross_validation\""));
        assert!(!json.contains("\"hf_data\""));
    }

    #[test]
    fn test_huggingface_data_with_generation_config() {
        let hf = HuggingFaceData {
            repo: "test/model".to_string(),
            model_type: Some("llama".to_string()),
            pipeline_tag: None,
            downloads: Some(42),
            config_fields: serde_json::json!({"hidden_size": 4096}),
            generation_config: Some(serde_json::json!({
                "temperature": 0.7,
                "top_p": 0.9,
                "max_length": 2048
            })),
        };
        let json = serde_json::to_string(&hf).expect("serialize");
        assert!(json.contains("generation_config"));
        assert!(json.contains("temperature"));
        assert!(json.contains("0.7"));
    }

    #[test]
    fn test_compute_param_count_embedding_contribution() {
        // Verify embedding = vocab_size * hidden_dim
        let size = ModelSizeConfig {
            parameters: "test".to_string(),
            hidden_dim: 100,
            num_layers: 0, // zero layers to isolate embedding
            num_heads: 1,
            num_kv_heads: 1,
            intermediate_dim: 200,
            vocab_size: 1000,
            max_position_embeddings: 512,
            head_dim: 100,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        };
        let mut constraints = make_test_constraints();
        constraints.tied_embeddings = false;

        let params = compute_param_count(&size, &constraints);
        // embedding (1000*100) + lm_head (1000*100) + final_norm (100) + 0 layers
        let expected = 1000 * 100 + 1000 * 100 + 100;
        assert_eq!(params, expected);
    }

    #[test]
    fn test_compute_param_count_tied_removes_lm_head() {
        let size = ModelSizeConfig {
            parameters: "test".to_string(),
            hidden_dim: 100,
            num_layers: 0,
            num_heads: 1,
            num_kv_heads: 1,
            intermediate_dim: 200,
            vocab_size: 1000,
            max_position_embeddings: 512,
            head_dim: 100,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        };
        let mut constraints = make_test_constraints();
        constraints.tied_embeddings = true;

        let params = compute_param_count(&size, &constraints);
        // embedding (1000*100) + final_norm (100) + 0 layers, no lm_head
        let expected = 1000 * 100 + 100;
        assert_eq!(params, expected);
    }

    #[test]
    fn test_compute_memory_estimates_ratio() {
        // F16 = 2 bytes/param, Q4 = 0.5 bytes/param
        // So F16/Q4 ratio should be exactly 4.0
        let size = make_test_size();
        let constraints = make_test_constraints();
        let (f16_mb, q4_mb) = compute_memory_estimates(&size, &constraints);
        let ratio = f16_mb / q4_mb;
        assert!(
            (ratio - 4.0).abs() < 1e-10,
            "F16/Q4 ratio should be exactly 4.0, got {ratio}"
        );
    }

    #[test]
    fn test_kv_cache_formula_correctness() {
        // KV = 2 * L * kv_heads * head_dim * 2(f16)
        let size = ModelSizeConfig {
            parameters: "test".to_string(),
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 4,
            intermediate_dim: 512,
            vocab_size: 100,
            max_position_embeddings: 2048,
            head_dim: 32,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        };
        let (per_token, _) = compute_kv_cache(&size);
        let expected: u64 = 2 * 4 * 4 * 32 * 2;
        assert_eq!(per_token, expected);
    }

    #[test]
    fn test_ffn_analysis_swiglu_explanation_contains_ratio() {
        let mut size = make_test_size();
        size.hidden_dim = 1000;
        size.intermediate_dim = 3000;
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::SwiGlu;
        let (ratio, explanation) = compute_ffn_analysis(&size, &constraints);
        assert!((ratio - 3.0).abs() < 0.01);
        assert!(explanation.contains("3.00x"));
    }

    #[test]
    fn test_ffn_analysis_gelu_explanation_contains_ratio() {
        let mut size = make_test_size();
        size.hidden_dim = 1000;
        size.intermediate_dim = 4000;
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GeluMlp;
        let (ratio, explanation) = compute_ffn_analysis(&size, &constraints);
        assert!((ratio - 4.0).abs() < 0.01);
        assert!(explanation.contains("4.00x"));
    }

    #[test]
    fn test_rope_analysis_negative_theta() {
        let mut size = make_test_size();
        size.rope_theta = -1.0; // Negative should produce 0.0
        let (wavelength, _) = compute_rope_analysis(&size);
        assert_eq!(wavelength, 0.0);
    }

    #[test]
    fn test_flops_estimate_scales_with_layers() {
        let mut size = make_test_size();
        let constraints = make_test_constraints();

        size.num_layers = 10;
        let (attn_10, ffn_10) = compute_flops_estimate(&size, &constraints);

        size.num_layers = 20;
        let (attn_20, ffn_20) = compute_flops_estimate(&size, &constraints);

        // Doubling layers should double FLOPS
        assert_eq!(attn_20, attn_10 * 2);
        assert_eq!(ffn_20, ffn_10 * 2);
    }

    #[test]
    fn test_build_statistical_analysis_zero_rope_theta() {
        let mut size = make_test_size();
        size.rope_theta = 0.0;
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        assert_eq!(stats.rope_max_wavelength, 0.0);
    }

    #[test]
    fn test_build_statistical_analysis_zero_hidden_dim() {
        let mut size = make_test_size();
        size.hidden_dim = 0;
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        assert_eq!(stats.ffn_expansion_ratio, 0.0);
        assert!(stats.ffn_type_explanation.is_empty());
    }

    #[test]
    fn test_kernel_compatibility_geglu_kernel_string() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GatedMlp;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);
        assert!(kern.ffn_kernel.contains("GeGLU"));
        assert!(kern.ffn_kernel.contains("row-major"));
    }

    #[test]
    fn test_kernel_compatibility_f16_bits_per_weight() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        let f16 = kern
            .supported_quantizations
            .iter()
            .find(|q| q.format == "F16")
            .expect("F16");
        assert!((f16.bits_per_weight - 16.0).abs() < 0.001);

        let q8 = kern
            .supported_quantizations
            .iter()
            .find(|q| q.format == "Q8_0")
            .expect("Q8_0");
        assert!((q8.bits_per_weight - 8.0).abs() < 0.001);

        let q4 = kern
            .supported_quantizations
            .iter()
            .find(|q| q.format == "Q4_K_M")
            .expect("Q4_K_M");
        assert!((q4.bits_per_weight - 4.5).abs() < 0.001);

        let q6 = kern
            .supported_quantizations
            .iter()
            .find(|q| q.format == "Q6_K")
            .expect("Q6_K");
        assert!((q6.bits_per_weight - 6.5).abs() < 0.001);
    }

    #[test]
    fn test_kernel_compatibility_row_major_note_always_present() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);
        assert!(kern.notes.iter().any(|n| n.contains("ROW-MAJOR")));
    }

    #[test]
    fn test_architecture_explanation_gqa_kv_cache_comparison() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);
        // GQA explanation should mention cache reduction percentage
        assert!(expl.attention_explanation.contains("reduces KV cache"));
        assert!(expl.attention_explanation.contains("MB"));
    }

    #[test]
    fn test_architecture_explanation_rope_extrapolation() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);
        // RoPE explanation should mention extrapolation
        assert!(expl.positional_explanation.contains("YaRN"));
        let extrapolated = size.max_position_embeddings * 4;
        assert!(expl
            .positional_explanation
            .contains(&extrapolated.to_string()));
    }

    #[test]
    fn test_architecture_explanation_chinchilla_tokens() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);
        assert!(expl.scaling_analysis.contains("Chinchilla"));
        assert!(expl.scaling_analysis.contains("FLOPs"));
    }

    #[test]
    fn test_build_family_info_with_chat_template() {
        use aprender::format::model_family::*;
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "qwen2".to_string(),
            display_name: "Qwen2".to_string(),
            vendor: "Alibaba".to_string(),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            hf_pattern: "Qwen/Qwen2*".to_string(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: TensorTemplate {
                embedding: "embed.weight".to_string(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: Some(ChatTemplateConfig {
                format: "chatml".to_string(),
                template: String::new(),
                bos_token: "<|im_start|>".to_string(),
                eos_token: "<|im_end|>".to_string(),
                special_tokens: HashMap::new(),
            }),
            certification: None,
        };

        let fi = build_family_info(&config);
        assert_eq!(fi.chat_template_format, Some("chatml".to_string()));
    }

    #[test]
    fn test_build_family_info_display_types() {
        use aprender::format::model_family::*;
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "gpt2".to_string(),
            display_name: "GPT-2".to_string(),
            vendor: "OpenAI".to_string(),
            architectures: vec!["GPT2LMHeadModel".to_string()],
            hf_pattern: "openai/gpt2*".to_string(),
            size_variants: HashMap::new(),
            constraints: ModelConstraints {
                attention_type: AttentionType::Mha,
                activation: Activation::Gelu,
                norm_type: NormType::LayerNorm,
                has_bias: true,
                tied_embeddings: true,
                positional_encoding: PositionalEncoding::Absolute,
                mlp_type: MlpType::GeluMlp,
            },
            tensor_template: TensorTemplate {
                embedding: "wte.weight".to_string(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let fi = build_family_info(&config);
        assert_eq!(fi.constraints.attention, "MHA");
        assert_eq!(fi.constraints.norm, "LayerNorm");
        assert_eq!(fi.constraints.mlp, "GELU MLP");
        assert_eq!(fi.constraints.positional_encoding, "Absolute");
        assert!(fi.constraints.bias);
        assert!(fi.constraints.tied_embeddings);
    }

    #[test]
    fn test_cross_validation_all_fields_match() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "intermediate_size": 8960,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "model_type": "qwen2"
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        assert!(cv.mismatches.is_empty());
        assert!(cv.contract_only.is_empty());
        // 7 size fields + rope_theta + norm_eps + model_type = 10 matches
        assert!(
            cv.matches.len() >= 9,
            "Expected at least 9 matches, got {}",
            cv.matches.len()
        );
    }

    #[test]
    fn test_cross_validation_multiple_mismatches() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "hidden_size": 9999,
            "num_hidden_layers": 99,
            "num_attention_heads": 99,
            "num_key_value_heads": 99,
            "intermediate_size": 9999,
            "vocab_size": 9999,
            "max_position_embeddings": 9999
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        assert_eq!(cv.mismatches.len(), 7, "All 7 size fields should mismatch");
    }

    #[test]
    fn test_cross_validation_norm_eps_mismatch_value() {
        let mut size = make_test_size();
        size.norm_eps = 1e-6;
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "rms_norm_eps": 1e-5
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let entry = cv
            .mismatches
            .iter()
            .find(|e| e.field == "norm_eps")
            .expect("should mismatch");
        assert_eq!(entry.status, "mismatch");
    }

    #[test]
    fn test_format_params_large_values() {
        assert_eq!(format_params(70_000_000_000), "70.0B");
        assert_eq!(format_params(175_000_000_000), "175.0B");
    }

    #[test]
    fn test_format_params_exact_boundaries() {
        assert_eq!(format_params(1), "1");
        assert_eq!(format_params(10), "10");
        assert_eq!(format_params(100), "100");
    }

    #[test]
    fn test_statistical_analysis_complete_field_coverage() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);

        // Verify specific computed values
        let (expected_ratio, expected_reduction) = compute_gqa_analysis(&size);
        assert!((stats.gqa_ratio - expected_ratio).abs() < 1e-10);
        assert!((stats.kv_cache_reduction - expected_reduction).abs() < 1e-10);

        let expected_params = compute_param_count(&size, &constraints);
        assert_eq!(stats.model_params, expected_params);

        let (expected_per_token, expected_4k) = compute_kv_cache(&size);
        assert_eq!(stats.kv_cache_per_token_bytes, expected_per_token);
        assert!((stats.kv_cache_4k_mb - expected_4k).abs() < 1e-10);

        let (expected_ffn_ratio, _) = compute_ffn_analysis(&size, &constraints);
        assert!((stats.ffn_expansion_ratio - expected_ffn_ratio).abs() < 1e-10);

        let (expected_wavelength, expected_ctx) = compute_rope_analysis(&size);
        assert!((stats.rope_max_wavelength - expected_wavelength).abs() < 1e-10);
        assert_eq!(stats.effective_context_window, expected_ctx);

        let (expected_attn_flops, expected_ffn_flops) = compute_flops_estimate(&size, &constraints);
        assert_eq!(stats.attention_flops_per_token, expected_attn_flops);
        assert_eq!(stats.ffn_flops_per_token, expected_ffn_flops);
    }

    #[test]
    fn test_build_certification_size_template_replacement() {
        use aprender::format::model_family::CertificationConfig;
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: aprender::format::model_family::TensorTemplate {
                embedding: String::new(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: Some(CertificationConfig {
                playbook_path: "playbooks/{size}/run.yaml".to_string(),
                csv_family_key: "test".to_string(),
                size_categories: HashMap::new(),
            }),
        };

        let cert = build_certification(&config, Some("13b")).expect("cert exists");
        assert_eq!(
            cert.playbook_path,
            Some("playbooks/13b/run.yaml".to_string())
        );
    }

    #[test]
    fn test_compliance_result_non_compliant_serialize() {
        let cr = ComplianceResult {
            is_compliant: false,
            tensor_count_match: false,
            missing_tensors: vec![
                "layer.0.q.weight".to_string(),
                "layer.0.k.weight".to_string(),
            ],
            unexpected_tensors: vec!["extra.bias".to_string()],
        };
        let json = serde_json::to_string(&cr).expect("serialize");
        assert!(json.contains("\"is_compliant\":false"));
        assert!(json.contains("\"tensor_count_match\":false"));
        assert!(json.contains("layer.0.q.weight"));
        assert!(json.contains("extra.bias"));
    }

    #[test]
    fn test_kernel_compatibility_memory_calculation() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        // memory_required_mb = Q4 model size + KV cache
        let q4_size = (stats.model_params as f64 * 0.5625) / (1024.0 * 1024.0);
        let expected_mem = q4_size + stats.kv_cache_4k_mb;
        assert!(
            (kern.memory_required_mb - expected_mem).abs() < 0.01,
            "Memory should be Q4 model + KV cache"
        );
    }

    #[test]
    fn test_oracle_mode_family_serialize() {
        let mode = OracleMode::Family;
        let json = serde_json::to_string(&mode).expect("serialize");
        assert_eq!(json, "\"family\"");
    }

    #[test]
    fn test_cross_validation_entry_debug() {
        let entry = CrossValidationEntry {
            field: "hidden_dim".to_string(),
            contract_value: "1536".to_string(),
            hf_value: "2048".to_string(),
            status: "mismatch".to_string(),
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("CrossValidationEntry"));
        assert!(debug.contains("hidden_dim"));
    }

    #[test]
    fn test_cross_validation_debug() {
        let cv = CrossValidation {
            matches: vec![],
            mismatches: vec![],
            contract_only: vec![],
            hf_only: vec![],
        };
        let debug = format!("{cv:?}");
        assert!(debug.contains("CrossValidation"));
    }

    #[test]
    fn test_statistical_analysis_debug() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let debug = format!("{stats:?}");
        assert!(debug.contains("StatisticalAnalysis"));
        assert!(debug.contains("gqa_ratio"));
    }

    #[test]
    fn test_architecture_explanation_debug() {
        let expl = ArchitectureExplanation {
            attention_explanation: "test".to_string(),
            ffn_explanation: "test".to_string(),
            norm_explanation: "test".to_string(),
            positional_explanation: "test".to_string(),
            scaling_analysis: "test".to_string(),
        };
        let debug = format!("{expl:?}");
        assert!(debug.contains("ArchitectureExplanation"));
    }

    #[test]
    fn test_kernel_compatibility_debug() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![],
            attention_kernel: "test".to_string(),
            ffn_kernel: "test".to_string(),
            estimated_tps_cpu: None,
            estimated_tps_gpu: None,
            memory_required_mb: 0.0,
            notes: vec![],
        };
        let debug = format!("{kern:?}");
        assert!(debug.contains("KernelCompatibility"));
    }

    #[test]
    fn test_quantization_support_debug() {
        let qs = QuantizationSupport {
            format: "Q4_K_M".to_string(),
            supported: true,
            kernel: "fused_q4k".to_string(),
            bits_per_weight: 4.5,
            estimated_size_mb: 500.0,
        };
        let debug = format!("{qs:?}");
        assert!(debug.contains("QuantizationSupport"));
    }

    #[test]
    fn test_huggingface_data_debug() {
        let hf = HuggingFaceData {
            repo: "test/model".to_string(),
            model_type: None,
            pipeline_tag: None,
            downloads: None,
            config_fields: serde_json::json!({}),
            generation_config: None,
        };
        let debug = format!("{hf:?}");
        assert!(debug.contains("HuggingFaceData"));
    }

    #[test]
    fn test_offline_hf_huggingface_prefix_rejected() {
        let src = "huggingface://Qwen/Qwen2.5-1.5B".to_string();
        let flags = OracleFlags::default();
        let result = run(
            Some(&src),
            None,
            None,
            false,
            false,
            false,
            false,
            true,
            flags,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::NetworkError(msg)) => {
                assert!(msg.contains("offline"));
            }
            other => panic!("Expected NetworkError, got: {other:?}"),
        }
    }

    // ========================================================================
    // Format Function Tests (coverage for output formatting)
    // ========================================================================

    #[test]
    fn test_format_text_report_basic() {
        let report = ModelOracleReport {
            source: "test.gguf".to_string(),
            mode: OracleMode::Local,
            family: None,
            size_variant: None,
            format: None,
            compliance: None,
            certification: None,
            tensors: None,
            stats: None,
            explanation: None,
            kernel_compatibility: None,
            cross_validation: None,
            hf_data: None,
        };
        let out = format_text_report(&report);
        assert!(out.contains("test.gguf"));
        assert!(out.contains("Local"));
    }

    #[test]
    fn test_format_text_report_hf_mode() {
        let report = ModelOracleReport {
            source: "hf://Qwen/Qwen2.5-1.5B".to_string(),
            mode: OracleMode::HuggingFace,
            family: None,
            size_variant: None,
            format: None,
            compliance: None,
            certification: None,
            tensors: None,
            stats: None,
            explanation: None,
            kernel_compatibility: None,
            cross_validation: None,
            hf_data: None,
        };
        let out = format_text_report(&report);
        assert!(out.contains("hf://Qwen/Qwen2.5-1.5B"));
        assert!(out.contains("HuggingFace"));
    }

    #[test]
    fn test_format_text_report_family_mode() {
        let report = ModelOracleReport {
            source: "qwen2".to_string(),
            mode: OracleMode::Family,
            family: None,
            size_variant: None,
            format: None,
            compliance: None,
            certification: None,
            tensors: None,
            stats: None,
            explanation: None,
            kernel_compatibility: None,
            cross_validation: None,
            hf_data: None,
        };
        let out = format_text_report(&report);
        assert!(out.contains("qwen2"));
        assert!(out.contains("Family"));
    }

    #[test]
    fn test_format_text_format_basic() {
        let fmt = FormatInfo {
            format_type: "GGUF".to_string(),
            file_size: 4_000_000_000,
            tensor_count: 291,
            total_params: 7_000_000_000,
            quantization: Some("Q4_K_M".to_string()),
            architecture: Some("LlamaForCausalLM".to_string()),
        };
        let out = format_text_format(&fmt);
        assert!(out.contains("GGUF"));
        assert!(out.contains("291"));
        assert!(out.contains("Q4_K_M"));
        assert!(out.contains("LlamaForCausalLM"));
        assert!(out.contains("7.0B"));
    }

    #[test]
    fn test_format_text_format_no_optionals() {
        let fmt = FormatInfo {
            format_type: "SafeTensors".to_string(),
            file_size: 1_000_000,
            tensor_count: 100,
            total_params: 500_000,
            quantization: None,
            architecture: None,
        };
        let out = format_text_format(&fmt);
        assert!(out.contains("SafeTensors"));
        assert!(out.contains("100"));
        assert!(!out.contains("Quantization"));
        assert!(!out.contains("Architecture"));
    }

    #[test]
    fn test_format_text_format_small_params() {
        let fmt = FormatInfo {
            format_type: "APR".to_string(),
            file_size: 1024,
            tensor_count: 5,
            total_params: 500,
            quantization: None,
            architecture: None,
        };
        let out = format_text_format(&fmt);
        assert!(out.contains("APR"));
        assert!(out.contains("500"));
    }

    #[test]
    fn test_format_text_family_basic() {
        let fi = FamilyInfo {
            name: "qwen2".to_string(),
            display_name: "Qwen2".to_string(),
            vendor: "Alibaba".to_string(),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            constraints: ConstraintsSummary {
                attention: "GQA".to_string(),
                activation: "SiLU".to_string(),
                norm: "RMSNorm".to_string(),
                bias: true,
                tied_embeddings: false,
                mlp: "SwiGLU".to_string(),
                positional_encoding: "RoPE".to_string(),
            },
            chat_template_format: Some("chatml".to_string()),
        };
        let out = format_text_family(&fi, false);
        assert!(out.contains("qwen2 (Qwen2)"));
        assert!(out.contains("Alibaba"));
        assert!(!out.contains("Qwen2ForCausalLM")); // not verbose
        assert!(out.contains("chatml"));
    }

    #[test]
    fn test_format_text_family_verbose() {
        let fi = FamilyInfo {
            name: "llama".to_string(),
            display_name: "LLaMA".to_string(),
            vendor: "Meta".to_string(),
            architectures: vec!["LlamaForCausalLM".to_string(), "LlamaModel".to_string()],
            constraints: ConstraintsSummary {
                attention: "GQA".to_string(),
                activation: "SiLU".to_string(),
                norm: "RMSNorm".to_string(),
                bias: false,
                tied_embeddings: false,
                mlp: "SwiGLU".to_string(),
                positional_encoding: "RoPE".to_string(),
            },
            chat_template_format: None,
        };
        let out = format_text_family(&fi, true);
        assert!(out.contains("LlamaForCausalLM, LlamaModel"));
        assert!(!out.contains("Chat Template"));
    }

    #[test]
    fn test_format_text_family_no_chat_template() {
        let fi = FamilyInfo {
            name: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "TestCo".to_string(),
            architectures: vec![],
            constraints: ConstraintsSummary {
                attention: "MHA".to_string(),
                activation: "GELU".to_string(),
                norm: "LayerNorm".to_string(),
                bias: true,
                tied_embeddings: true,
                mlp: "GELU MLP".to_string(),
                positional_encoding: "Absolute".to_string(),
            },
            chat_template_format: None,
        };
        let out = format_text_family(&fi, false);
        assert!(!out.contains("Chat Template"));
    }

    #[test]
    fn test_format_text_family_empty_architectures_verbose() {
        let fi = FamilyInfo {
            name: "t".to_string(),
            display_name: "T".to_string(),
            vendor: "V".to_string(),
            architectures: vec![],
            constraints: ConstraintsSummary {
                attention: "MHA".to_string(),
                activation: "GELU".to_string(),
                norm: "LayerNorm".to_string(),
                bias: false,
                tied_embeddings: false,
                mlp: "GELU MLP".to_string(),
                positional_encoding: "Absolute".to_string(),
            },
            chat_template_format: None,
        };
        let out = format_text_family(&fi, true);
        assert!(out.contains("Architectures:"));
    }

    #[test]
    fn test_format_text_size_basic() {
        let svi = SizeVariantInfo {
            name: "1.5b".to_string(),
            parameters: "1.5B".to_string(),
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            intermediate_dim: 8960,
            vocab_size: 151936,
            expected_tensor_count: 339,
        };
        let out = format_text_size(&svi);
        assert!(out.contains("1.5B"));
        assert!(out.contains("hidden=1536"));
        assert!(out.contains("layers=28"));
        assert!(out.contains("heads=12"));
        assert!(out.contains("kv_heads=2"));
        assert!(out.contains("8960"));
        assert!(out.contains("151936"));
        assert!(out.contains("339"));
    }

    #[test]
    fn test_format_text_size_large_model() {
        let svi = SizeVariantInfo {
            name: "70b".to_string(),
            parameters: "70B".to_string(),
            hidden_dim: 8192,
            num_layers: 80,
            num_heads: 64,
            num_kv_heads: 8,
            intermediate_dim: 28672,
            vocab_size: 128256,
            expected_tensor_count: 723,
        };
        let out = format_text_size(&svi);
        assert!(out.contains("70B"));
        assert!(out.contains("hidden=8192"));
        assert!(out.contains("723"));
    }

    #[test]
    fn test_format_text_size_minimal() {
        let svi = SizeVariantInfo {
            name: "tiny".to_string(),
            parameters: "10M".to_string(),
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 128,
            vocab_size: 100,
            expected_tensor_count: 20,
        };
        let out = format_text_size(&svi);
        assert!(out.contains("10M"));
        assert!(out.contains("Intermediate Dim: 128"));
    }

    #[test]
    fn test_format_text_constraints_with_bias() {
        let fi = FamilyInfo {
            name: "qwen2".to_string(),
            display_name: "Qwen2".to_string(),
            vendor: "Alibaba".to_string(),
            architectures: vec![],
            constraints: ConstraintsSummary {
                attention: "GQA".to_string(),
                activation: "SiLU".to_string(),
                norm: "RMSNorm".to_string(),
                bias: true,
                tied_embeddings: false,
                mlp: "SwiGLU".to_string(),
                positional_encoding: "RoPE".to_string(),
            },
            chat_template_format: None,
        };
        let out = format_text_constraints(&fi);
        assert!(out.contains("GQA"));
        assert!(out.contains("SiLU"));
        assert!(out.contains("RMSNorm"));
        assert!(out.contains("Bias: yes"));
        assert!(out.contains("Tied: no"));
        assert!(out.contains("SwiGLU"));
        assert!(out.contains("RoPE"));
    }

    #[test]
    fn test_format_text_constraints_no_bias_tied() {
        let fi = FamilyInfo {
            name: "gpt2".to_string(),
            display_name: "GPT-2".to_string(),
            vendor: "OpenAI".to_string(),
            architectures: vec![],
            constraints: ConstraintsSummary {
                attention: "MHA".to_string(),
                activation: "GELU".to_string(),
                norm: "LayerNorm".to_string(),
                bias: false,
                tied_embeddings: true,
                mlp: "GELU MLP".to_string(),
                positional_encoding: "Absolute".to_string(),
            },
            chat_template_format: None,
        };
        let out = format_text_constraints(&fi);
        assert!(out.contains("Bias: no"));
        assert!(out.contains("Tied: yes"));
        assert!(out.contains("MHA"));
        assert!(out.contains("LayerNorm"));
    }

    #[test]
    fn test_format_text_constraints_header() {
        let fi = FamilyInfo {
            name: "t".to_string(),
            display_name: "T".to_string(),
            vendor: "V".to_string(),
            architectures: vec![],
            constraints: ConstraintsSummary {
                attention: "MQA".to_string(),
                activation: "SiLU".to_string(),
                norm: "RMSNorm".to_string(),
                bias: false,
                tied_embeddings: false,
                mlp: "SwiGLU".to_string(),
                positional_encoding: "ALiBi".to_string(),
            },
            chat_template_format: None,
        };
        let out = format_text_constraints(&fi);
        assert!(out.contains("Constraints:"));
        assert!(out.contains("MQA"));
        assert!(out.contains("ALiBi"));
    }

    #[test]
    fn test_format_text_stats_basic() {
        let stats = StatisticalAnalysis {
            gqa_ratio: 0.167,
            kv_cache_reduction: 0.833,
            model_params: 1_500_000_000,
            model_size_f16_mb: 2861.0,
            model_size_q4_mb: 715.0,
            kv_cache_per_token_bytes: 28672,
            kv_cache_4k_mb: 112.0,
            ffn_expansion_ratio: 5.83,
            ffn_type_explanation: "SwiGLU gated".to_string(),
            rope_max_wavelength: 6283185.0,
            effective_context_window: 32768,
            attention_flops_per_token: 100_000_000,
            ffn_flops_per_token: 200_000_000,
        };
        let out = format_text_stats(&stats);
        assert!(out.contains("0.17"));
        assert!(out.contains("83%"));
        assert!(out.contains("1.5B"));
        assert!(out.contains("2861.0 MB"));
        assert!(out.contains("715.0 MB"));
        assert!(out.contains("28672 bytes"));
        assert!(out.contains("112.0 MB"));
        assert!(out.contains("5.83x"));
        assert!(out.contains("SwiGLU gated"));
        assert!(out.contains("32768"));
    }

    #[test]
    fn test_format_text_stats_no_rope() {
        let stats = StatisticalAnalysis {
            gqa_ratio: 1.0,
            kv_cache_reduction: 0.0,
            model_params: 100_000,
            model_size_f16_mb: 0.2,
            model_size_q4_mb: 0.05,
            kv_cache_per_token_bytes: 100,
            kv_cache_4k_mb: 0.4,
            ffn_expansion_ratio: 4.0,
            ffn_type_explanation: "Standard GELU".to_string(),
            rope_max_wavelength: 0.0,
            effective_context_window: 2048,
            attention_flops_per_token: 1000,
            ffn_flops_per_token: 2000,
        };
        let out = format_text_stats(&stats);
        assert!(!out.contains("RoPE Wavelength"));
        assert!(out.contains("Standard GELU"));
    }

    #[test]
    fn test_format_text_stats_with_rope() {
        let stats = StatisticalAnalysis {
            gqa_ratio: 0.25,
            kv_cache_reduction: 0.75,
            model_params: 7_000_000_000,
            model_size_f16_mb: 13000.0,
            model_size_q4_mb: 3250.0,
            kv_cache_per_token_bytes: 65536,
            kv_cache_4k_mb: 256.0,
            ffn_expansion_ratio: 3.5,
            ffn_type_explanation: "SwiGLU".to_string(),
            rope_max_wavelength: 62831.0,
            effective_context_window: 131072,
            attention_flops_per_token: 500_000_000,
            ffn_flops_per_token: 800_000_000,
        };
        let out = format_text_stats(&stats);
        assert!(out.contains("RoPE Wavelength: 62831"));
        assert!(out.contains("131072"));
    }

    #[test]
    fn test_format_text_stats_flops_format() {
        let stats = StatisticalAnalysis {
            gqa_ratio: 0.5,
            kv_cache_reduction: 0.5,
            model_params: 1_000_000,
            model_size_f16_mb: 1.9,
            model_size_q4_mb: 0.48,
            kv_cache_per_token_bytes: 512,
            kv_cache_4k_mb: 2.0,
            ffn_expansion_ratio: 4.0,
            ffn_type_explanation: "test".to_string(),
            rope_max_wavelength: 100.0,
            effective_context_window: 1024,
            attention_flops_per_token: 123_456_789,
            ffn_flops_per_token: 987_654_321,
        };
        let out = format_text_stats(&stats);
        assert!(out.contains("Attn FLOPS/tok:"));
        assert!(out.contains("FFN FLOPS/tok:"));
        assert!(out.contains("e"));
    }

    #[test]
    fn test_format_text_explanation_basic() {
        let expl = ArchitectureExplanation {
            attention_explanation: "GQA with ratio 0.17".to_string(),
            ffn_explanation: "SwiGLU gated activation".to_string(),
            norm_explanation: "RMSNorm eps=1e-6".to_string(),
            positional_explanation: "RoPE theta=1000000".to_string(),
            scaling_analysis: "1.5B parameters, Chinchilla-optimal".to_string(),
        };
        let out = format_text_explanation(&expl);
        assert!(out.contains("GQA with ratio 0.17"));
        assert!(out.contains("SwiGLU gated activation"));
        assert!(out.contains("RMSNorm eps=1e-6"));
        assert!(out.contains("RoPE theta=1000000"));
        assert!(out.contains("Chinchilla-optimal"));
    }

    #[test]
    fn test_format_text_explanation_sections_labeled() {
        let expl = ArchitectureExplanation {
            attention_explanation: "attn".to_string(),
            ffn_explanation: "ffn".to_string(),
            norm_explanation: "norm".to_string(),
            positional_explanation: "pos".to_string(),
            scaling_analysis: "scale".to_string(),
        };
        let out = format_text_explanation(&expl);
        assert!(out.contains("Attention: attn"));
        assert!(out.contains("FFN: ffn"));
        assert!(out.contains("Normalization: norm"));
        assert!(out.contains("Position: pos"));
        assert!(out.contains("Scaling: scale"));
    }

    #[test]
    fn test_format_text_explanation_empty_strings() {
        let expl = ArchitectureExplanation {
            attention_explanation: String::new(),
            ffn_explanation: String::new(),
            norm_explanation: String::new(),
            positional_explanation: String::new(),
            scaling_analysis: String::new(),
        };
        let out = format_text_explanation(&expl);
        assert!(out.contains("Attention:"));
        assert!(out.contains("FFN:"));
    }

    #[test]
    fn test_format_text_kernels_basic() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![
                QuantizationSupport {
                    format: "F16".to_string(),
                    supported: true,
                    kernel: "trueno::f16_matvec".to_string(),
                    bits_per_weight: 16.0,
                    estimated_size_mb: 3000.0,
                },
                QuantizationSupport {
                    format: "Q4_K_M".to_string(),
                    supported: true,
                    kernel: "fused_q4k".to_string(),
                    bits_per_weight: 4.5,
                    estimated_size_mb: 750.0,
                },
            ],
            attention_kernel: "GQA fused QKV".to_string(),
            ffn_kernel: "SwiGLU fused".to_string(),
            estimated_tps_cpu: Some(60.0),
            estimated_tps_gpu: Some(1200.0),
            memory_required_mb: 850.0,
            notes: vec!["ROW-MAJOR layout".to_string(), "GQA: 6:1 ratio".to_string()],
        };
        let out = format_text_kernels(&kern);
        assert!(out.contains("GQA fused QKV"));
        assert!(out.contains("SwiGLU fused"));
        assert!(out.contains("F16"));
        assert!(out.contains("Q4_K_M"));
        assert!(out.contains("yes"));
        assert!(out.contains("60"));
        assert!(out.contains("1200"));
        assert!(out.contains("850.0 MB"));
        assert!(out.contains("ROW-MAJOR layout"));
        assert!(out.contains("GQA: 6:1 ratio"));
    }

    #[test]
    fn test_format_text_kernels_no_tps() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![],
            attention_kernel: "MHA standard".to_string(),
            ffn_kernel: "GELU MLP".to_string(),
            estimated_tps_cpu: None,
            estimated_tps_gpu: None,
            memory_required_mb: 100.0,
            notes: vec![],
        };
        let out = format_text_kernels(&kern);
        assert!(!out.contains("Est. CPU"));
        assert!(!out.contains("Est. GPU"));
        assert!(out.contains("100.0 MB"));
    }

    #[test]
    fn test_format_text_kernels_unsupported_quant() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![QuantizationSupport {
                format: "Q2_K".to_string(),
                supported: false,
                kernel: "none".to_string(),
                bits_per_weight: 2.5,
                estimated_size_mb: 300.0,
            }],
            attention_kernel: "test".to_string(),
            ffn_kernel: "test".to_string(),
            estimated_tps_cpu: None,
            estimated_tps_gpu: None,
            memory_required_mb: 0.0,
            notes: vec![],
        };
        let out = format_text_kernels(&kern);
        assert!(out.contains("Q2_K"));
        assert!(out.contains("no"));
    }

    #[test]
    fn test_format_text_kernels_header_line() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![],
            attention_kernel: "a".to_string(),
            ffn_kernel: "f".to_string(),
            estimated_tps_cpu: None,
            estimated_tps_gpu: None,
            memory_required_mb: 0.0,
            notes: vec![],
        };
        let out = format_text_kernels(&kern);
        assert!(out.contains("Quantization Support:"));
        assert!(out.contains("Format"));
        assert!(out.contains("Supported"));
        assert!(out.contains("BPW"));
    }

    #[test]
    fn test_format_text_cross_validation_all_match() {
        let cv = CrossValidation {
            matches: vec![
                CrossValidationEntry {
                    field: "hidden_dim".to_string(),
                    contract_value: "1536".to_string(),
                    hf_value: "1536".to_string(),
                    status: "match".to_string(),
                },
                CrossValidationEntry {
                    field: "num_layers".to_string(),
                    contract_value: "28".to_string(),
                    hf_value: "28".to_string(),
                    status: "match".to_string(),
                },
            ],
            mismatches: vec![],
            contract_only: vec![],
            hf_only: vec![],
        };
        let out = format_text_cross_validation(&cv);
        assert!(out.contains("Matches (2)"));
        assert!(out.contains("[OK] hidden_dim: 1536 == 1536"));
        assert!(out.contains("[OK] num_layers: 28 == 28"));
        assert!(!out.contains("Mismatches"));
    }

    #[test]
    fn test_format_text_cross_validation_with_mismatches() {
        let cv = CrossValidation {
            matches: vec![],
            mismatches: vec![CrossValidationEntry {
                field: "hidden_dim".to_string(),
                contract_value: "1536".to_string(),
                hf_value: "2048".to_string(),
                status: "mismatch".to_string(),
            }],
            contract_only: vec!["vocab_size=151936".to_string()],
            hf_only: vec!["rope_scaling=dynamic".to_string()],
        };
        let out = format_text_cross_validation(&cv);
        assert!(out.contains("Mismatches (1)"));
        assert!(out.contains("[!!] hidden_dim: contract=1536 vs hf=2048"));
        assert!(out.contains("Contract-only: vocab_size=151936"));
        assert!(out.contains("HF-only: rope_scaling=dynamic"));
    }

    #[test]
    fn test_format_text_cross_validation_empty() {
        let cv = CrossValidation {
            matches: vec![],
            mismatches: vec![],
            contract_only: vec![],
            hf_only: vec![],
        };
        let out = format_text_cross_validation(&cv);
        assert!(out.is_empty());
    }

    #[test]
    fn test_format_text_cross_validation_multiple_contract_only() {
        let cv = CrossValidation {
            matches: vec![],
            mismatches: vec![],
            contract_only: vec!["a=1".to_string(), "b=2".to_string(), "c=3".to_string()],
            hf_only: vec![],
        };
        let out = format_text_cross_validation(&cv);
        assert!(out.contains("Contract-only: a=1, b=2, c=3"));
    }

    #[test]
    fn test_format_text_hf_all_fields() {
        let hf = HuggingFaceData {
            repo: "Qwen/Qwen2.5-1.5B".to_string(),
            model_type: Some("qwen2".to_string()),
            pipeline_tag: Some("text-generation".to_string()),
            downloads: Some(50000),
            config_fields: serde_json::json!({}),
            generation_config: None,
        };
        let out = format_text_hf(&hf);
        assert!(out.contains("Qwen/Qwen2.5-1.5B"));
        assert!(out.contains("qwen2"));
        assert!(out.contains("text-generation"));
        assert!(out.contains("50000"));
    }

    #[test]
    fn test_format_text_hf_minimal() {
        let hf = HuggingFaceData {
            repo: "test/model".to_string(),
            model_type: None,
            pipeline_tag: None,
            downloads: None,
            config_fields: serde_json::json!({}),
            generation_config: None,
        };
        let out = format_text_hf(&hf);
        assert!(out.contains("test/model"));
        assert!(!out.contains("model_type"));
        assert!(!out.contains("pipeline_tag"));
        assert!(!out.contains("Downloads"));
    }

    #[test]
    fn test_format_text_hf_partial_fields() {
        let hf = HuggingFaceData {
            repo: "org/repo".to_string(),
            model_type: Some("llama".to_string()),
            pipeline_tag: None,
            downloads: Some(42),
            config_fields: serde_json::json!({}),
            generation_config: None,
        };
        let out = format_text_hf(&hf);
        assert!(out.contains("llama"));
        assert!(out.contains("42"));
        assert!(!out.contains("pipeline_tag"));
    }

    #[test]
    fn test_format_text_compliance_compliant() {
        let cr = ComplianceResult {
            is_compliant: true,
            tensor_count_match: true,
            missing_tensors: vec![],
            unexpected_tensors: vec![],
        };
        let out = format_text_compliance(&cr, false);
        assert!(out.contains("COMPLIANT"));
        assert!(!out.contains("NON-COMPLIANT"));
    }

    #[test]
    fn test_format_text_compliance_non_compliant() {
        let cr = ComplianceResult {
            is_compliant: false,
            tensor_count_match: false,
            missing_tensors: vec![
                "layer.0.q.weight".to_string(),
                "layer.0.k.weight".to_string(),
            ],
            unexpected_tensors: vec!["extra.bias".to_string()],
        };
        let out = format_text_compliance(&cr, false);
        assert!(out.contains("NON-COMPLIANT"));
        assert!(out.contains("MISMATCH"));
        assert!(out.contains("2 tensor(s)"));
        assert!(!out.contains("layer.0.q.weight")); // not verbose
    }

    #[test]
    fn test_format_text_compliance_non_compliant_verbose() {
        let cr = ComplianceResult {
            is_compliant: false,
            tensor_count_match: false,
            missing_tensors: vec!["layer.0.q.weight".to_string()],
            unexpected_tensors: vec!["extra.bias".to_string()],
        };
        let out = format_text_compliance(&cr, true);
        assert!(out.contains("NON-COMPLIANT"));
        assert!(out.contains("- layer.0.q.weight"));
        assert!(out.contains("+ extra.bias"));
        assert!(out.contains("Unexpected Tensors: 1 tensor(s)"));
    }

    #[test]
    fn test_format_text_compliance_count_mismatch_only() {
        let cr = ComplianceResult {
            is_compliant: false,
            tensor_count_match: false,
            missing_tensors: vec![],
            unexpected_tensors: vec![],
        };
        let out = format_text_compliance(&cr, false);
        assert!(out.contains("NON-COMPLIANT"));
        assert!(out.contains("MISMATCH"));
        assert!(!out.contains("Missing"));
    }

    #[test]
    fn test_format_text_compliance_unexpected_hidden_not_verbose() {
        let cr = ComplianceResult {
            is_compliant: false,
            tensor_count_match: true,
            missing_tensors: vec![],
            unexpected_tensors: vec!["extra.weight".to_string()],
        };
        let out = format_text_compliance(&cr, false);
        assert!(!out.contains("Unexpected")); // not shown when not verbose
    }

    #[test]
    fn test_format_text_certification_with_playbook() {
        let cert = CertificationInfo {
            status: "PENDING".to_string(),
            playbook_path: Some("/playbooks/1.5b.yaml".to_string()),
        };
        let out = format_text_certification(&cert);
        assert!(out.contains("PENDING"));
        assert!(out.contains("/playbooks/1.5b.yaml"));
    }

    #[test]
    fn test_format_text_certification_no_playbook() {
        let cert = CertificationInfo {
            status: "APPROVED".to_string(),
            playbook_path: None,
        };
        let out = format_text_certification(&cert);
        assert!(out.contains("APPROVED"));
        assert!(!out.contains("Playbook"));
    }

    #[test]
    fn test_format_text_tensors_few() {
        let tensors = vec![
            TensorComplianceEntry {
                name: "model.embed_tokens.weight".to_string(),
                present: true,
                dtype: Some("F16".to_string()),
                shape: Some(vec![151936, 1536]),
                note: None,
            },
            TensorComplianceEntry {
                name: "lm_head.weight".to_string(),
                present: true,
                dtype: Some("F16".to_string()),
                shape: Some(vec![151936, 1536]),
                note: None,
            },
        ];
        let out = format_text_tensors(&tensors, false);
        assert!(out.contains("Tensors (2 total)"));
        assert!(out.contains("model.embed_tokens.weight"));
        assert!(out.contains("lm_head.weight"));
        assert!(out.contains("F16"));
        assert!(out.contains("[151936, 1536]"));
    }

    #[test]
    fn test_format_text_tensors_truncated() {
        // Create 25 tensors — should truncate at 20 when not verbose
        let tensors: Vec<TensorComplianceEntry> = (0..25)
            .map(|i| TensorComplianceEntry {
                name: format!("layer.{i}.weight"),
                present: true,
                dtype: Some("F16".to_string()),
                shape: Some(vec![100, 100]),
                note: None,
            })
            .collect();
        let out = format_text_tensors(&tensors, false);
        assert!(out.contains("Tensors (25 total)"));
        assert!(out.contains("... (3 more) ...")); // 25 - 20 - 2 = 3
        assert!(out.contains("layer.0.weight")); // first shown
        assert!(out.contains("layer.23.weight")); // last 2 always shown
        assert!(out.contains("layer.24.weight"));
    }

    #[test]
    fn test_format_text_tensors_verbose_all_shown() {
        let tensors: Vec<TensorComplianceEntry> = (0..25)
            .map(|i| TensorComplianceEntry {
                name: format!("layer.{i}.weight"),
                present: true,
                dtype: Some("F16".to_string()),
                shape: Some(vec![100]),
                note: None,
            })
            .collect();
        let out = format_text_tensors(&tensors, true);
        assert!(!out.contains("more")); // no truncation in verbose
        for i in 0..25 {
            assert!(
                out.contains(&format!("layer.{i}.weight")),
                "Missing layer.{i}.weight"
            );
        }
    }

    #[test]
    fn test_format_text_tensors_no_dtype_no_shape() {
        let tensors = vec![TensorComplianceEntry {
            name: "unknown.weight".to_string(),
            present: false,
            dtype: None,
            shape: None,
            note: Some("MISSING".to_string()),
        }];
        let out = format_text_tensors(&tensors, false);
        assert!(out.contains("unknown.weight"));
        assert!(out.contains("Tensors (1 total)"));
    }

    #[test]
    fn test_format_family_description_header_basic() {
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "qwen2".to_string(),
            display_name: "Qwen2".to_string(),
            vendor: "Alibaba".to_string(),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            hf_pattern: "Qwen/Qwen2*".to_string(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: aprender::format::model_family::TensorTemplate {
                embedding: String::new(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };
        let out = format_family_description_header(&config);
        assert!(out.contains("qwen2"));
        assert!(out.contains("Alibaba"));
        assert!(out.contains("Qwen2ForCausalLM"));
        assert!(out.contains("Qwen/Qwen2*"));
        assert!(out.contains("Constraints:"));
        assert!(out.contains("GQA"));
        assert!(out.contains("SiLU"));
        assert!(out.contains("Bias: yes"));
    }

    #[test]
    fn test_format_family_size_variant_basic() {
        let sc = make_test_size();
        let out = format_family_size_variant("1.5b", &sc);
        assert!(out.contains("1.5b (1.5B)"));
        assert!(out.contains("hidden_dim: 1536"));
        assert!(out.contains("num_layers: 28"));
        assert!(out.contains("num_heads: 12"));
        assert!(out.contains("num_kv_heads: 2"));
        assert!(out.contains("intermediate_dim: 8960"));
        assert!(out.contains("vocab_size: 151936"));
        assert!(out.contains("head_dim: 128"));
        assert!(out.contains("rope_theta: 1000000"));
        assert!(out.contains("norm_eps:"));
    }

    #[test]
    fn test_format_family_size_variant_no_rope() {
        let mut sc = make_test_size();
        sc.rope_theta = 0.0;
        let out = format_family_size_variant("test", &sc);
        assert!(!out.contains("rope_theta"));
    }

    #[test]
    fn test_format_family_variant_stats_basic() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let out = format_family_variant_stats(&stats);
        assert!(out.contains("GQA Ratio:"));
        assert!(out.contains("KV reduction"));
        assert!(out.contains("Est. Parameters:"));
        assert!(out.contains("Model Size (F16):"));
        assert!(out.contains("Model Size (Q4):"));
        assert!(out.contains("KV Cache (4K):"));
        assert!(out.contains("FFN Ratio:"));
    }

    #[test]
    fn test_format_family_variant_explain_basic() {
        let expl = ArchitectureExplanation {
            attention_explanation: "GQA attention".to_string(),
            ffn_explanation: "SwiGLU FFN".to_string(),
            norm_explanation: "RMSNorm".to_string(),
            positional_explanation: "RoPE".to_string(),
            scaling_analysis: "1.5B scaling".to_string(),
        };
        let out = format_family_variant_explain(&expl);
        assert!(out.contains("Attention: GQA attention"));
        assert!(out.contains("FFN: SwiGLU FFN"));
        assert!(out.contains("Scaling: 1.5B scaling"));
    }

    #[test]
    fn test_format_family_variant_kernels_basic() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![],
            attention_kernel: "GQA fused".to_string(),
            ffn_kernel: "SwiGLU fused".to_string(),
            estimated_tps_cpu: Some(55.0),
            estimated_tps_gpu: Some(1100.0),
            memory_required_mb: 900.0,
            notes: vec![],
        };
        let out = format_family_variant_kernels(&kern);
        assert!(out.contains("Attn Kernel: GQA fused"));
        assert!(out.contains("FFN Kernel: SwiGLU fused"));
        assert!(out.contains("Est. CPU tok/s: 55"));
        assert!(out.contains("Est. GPU tok/s: 1100"));
        assert!(out.contains("900.0 MB"));
    }

    #[test]
    fn test_format_family_variant_kernels_no_tps() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![],
            attention_kernel: "test".to_string(),
            ffn_kernel: "test".to_string(),
            estimated_tps_cpu: None,
            estimated_tps_gpu: None,
            memory_required_mb: 0.0,
            notes: vec![],
        };
        let out = format_family_variant_kernels(&kern);
        assert!(!out.contains("Est. CPU"));
        assert!(!out.contains("Est. GPU"));
    }

    #[test]
    fn test_format_family_description_footer_verbose() {
        use aprender::format::model_family::*;
        use std::collections::HashMap;

        let mut per_layer = HashMap::new();
        per_layer.insert(
            "q_proj".to_string(),
            Some("model.layers.{n}.q_proj.weight".to_string()),
        );

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: TensorTemplate {
                embedding: "embed.weight".to_string(),
                lm_head: Some("lm_head.weight".to_string()),
                final_norm: Some("norm.weight".to_string()),
                per_layer,
            },
            shape_template: ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec!["q4_k_m".to_string(), "q6_k".to_string()],
            chat_template: Some(ChatTemplateConfig {
                format: "chatml".to_string(),
                template: String::new(),
                bos_token: "<|im_start|>".to_string(),
                eos_token: "<|im_end|>".to_string(),
                special_tokens: HashMap::new(),
            }),
            certification: None,
        };

        let out = format_family_description_footer(&config, true);
        assert!(out.contains("Tensor Template:"));
        assert!(out.contains("Embedding: embed.weight"));
        assert!(out.contains("LM Head: lm_head.weight"));
        assert!(out.contains("Final Norm: norm.weight"));
        assert!(out.contains("Per-layer:"));
        assert!(out.contains("q_proj: model.layers.{n}.q_proj.weight"));
        assert!(out.contains("Quantizations: q4_k_m, q6_k"));
        assert!(out.contains("Chat Template: chatml"));
        assert!(out.contains("BOS: <|im_start|>"));
        assert!(out.contains("EOS: <|im_end|>"));
    }

    #[test]
    fn test_format_family_description_footer_not_verbose() {
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: aprender::format::model_family::TensorTemplate {
                embedding: "embed.weight".to_string(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let out = format_family_description_footer(&config, false);
        assert!(!out.contains("Tensor Template")); // hidden when not verbose
    }

    #[test]
    fn test_format_family_description_footer_empty_quantizations() {
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: aprender::format::model_family::TensorTemplate {
                embedding: String::new(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let out = format_family_description_footer(&config, false);
        assert!(!out.contains("Quantizations"));
    }
}
