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
    let attn = h * (n_heads * head_d) + h * (n_kv * head_d) + h * (n_kv * head_d) + (n_heads * head_d) * h;

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
    let lm_head = if constraints.tied_embeddings { 0 } else { v * h };

    // Final norm
    let final_norm = h;

    embedding + (per_layer * l) + lm_head + final_norm
}

/// Compute memory estimates for different precisions
pub fn compute_memory_estimates(size: &ModelSizeConfig, constraints: &ModelConstraints) -> (f64, f64) {
    let params = compute_param_count(size, constraints);
    let f16_mb = (params as f64 * 2.0) / (1024.0 * 1024.0);
    let q4_mb = (params as f64 * 0.5) / (1024.0 * 1024.0);
    (f16_mb, q4_mb)
}

/// Compute KV cache size per token and for 4K context
pub fn compute_kv_cache(size: &ModelSizeConfig) -> (u64, f64) {
    // KV cache per token: 2 (K+V) * num_layers * num_kv_heads * head_dim * 2 (f16 bytes)
    let per_token = 2_u64
        * size.num_layers as u64
        * size.num_kv_heads as u64
        * size.head_dim as u64
        * 2; // f16 = 2 bytes
    let cache_4k_mb = (per_token as f64 * 4096.0) / (1024.0 * 1024.0);
    (per_token, cache_4k_mb)
}

/// Compute FFN expansion ratio and explanation
pub fn compute_ffn_analysis(size: &ModelSizeConfig, constraints: &ModelConstraints) -> (f64, String) {
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
pub fn compute_flops_estimate(size: &ModelSizeConfig, constraints: &ModelConstraints) -> (u64, u64) {
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
    let downloads = api_data
        .as_ref()
        .and_then(|d| d["downloads"].as_u64());

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
        ("num_hidden_layers", "num_layers", |s| s.num_layers.to_string()),
        ("num_attention_heads", "num_heads", |s| s.num_heads.to_string()),
        ("num_key_value_heads", "num_kv_heads", |s| s.num_kv_heads.to_string()),
        ("intermediate_size", "intermediate_dim", |s| s.intermediate_dim.to_string()),
        ("vocab_size", "vocab_size", |s| s.vocab_size.to_string()),
        ("max_position_embeddings", "max_position_embeddings", |s| s.max_position_embeddings.to_string()),
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
                status: if contract_val == hf_str { "match".to_string() } else { "mismatch".to_string() },
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

    // Check rope_theta (float comparison with tolerance)
    if let Some(hf_theta) = hf_config.get("rope_theta").and_then(serde_json::Value::as_f64) {
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

    // Check norm_eps (float comparison)
    let hf_eps = hf_config
        .get("rms_norm_eps")
        .or_else(|| hf_config.get("layer_norm_eps"))
        .or_else(|| hf_config.get("layer_norm_epsilon"))
        .and_then(serde_json::Value::as_f64);
    if let Some(hf_eps_val) = hf_eps {
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

    // Check model_type vs family constraints
    if let Some(hf_model_type) = hf_config["model_type"].as_str() {
        let entry = CrossValidationEntry {
            field: "model_type".to_string(),
            contract_value: format!("{:?}", constraints.attention_type),
            hf_value: hf_model_type.to_string(),
            status: "info".to_string(),
        };
        matches.push(entry);
    }

    // Find HF fields we don't track
    let tracked_hf_fields = [
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
    if let Some(obj) = hf_config.as_object() {
        for key in obj.keys() {
            if !tracked_hf_fields.contains(&key.as_str()) {
                let interesting = [
                    "rope_scaling",
                    "sliding_window",
                    "attention_dropout",
                    "use_cache",
                    "tie_word_embeddings",
                ];
                if interesting.contains(&key.as_str()) {
                    hf_only.push(format!("{key}={}", obj[key]));
                }
            }
        }
    }

    CrossValidation {
        matches,
        mismatches,
        contract_only,
        hf_only,
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
        return run_family_mode(family, size_filter.map(String::as_str), json_output, verbose, flags);
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
    run_local_mode(&path, show_compliance, show_tensors, json_output, verbose, flags)
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

    // Detect family
    let detected_family = registry.detect_family(&tensor_names);

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

fn run_hf_mode(source: &str, json_output: bool, verbose: bool, flags: OracleFlags) -> Result<(), CliError> {
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
    let hidden_size = hf_data.config_fields["hidden_size"].as_u64().map(|v| v as usize);
    let num_layers = hf_data.config_fields["num_hidden_layers"].as_u64().map(|v| v as usize);

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
            Some(cross_validate(size_config, family.constraints(), &hf_data.config_fields))
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
    let family = registry.detect_from_model_type(family_name).ok_or_else(|| {
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
        let (stats, explanation, kernel_compat) = build_enhanced_sections(
            Some(family),
            size_filter,
            flags,
        );

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
fn load_registry() -> Result<FamilyRegistry, CliError> {
    // Try contracts/ relative to CWD first
    let cwd_contracts = PathBuf::from("contracts");
    if cwd_contracts.join("model-families").exists() {
        return load_family_registry(&cwd_contracts)
            .map_err(|e| CliError::Aprender(format!("Failed to load family contracts: {e}")));
    }

    // Try relative to executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let exe_contracts = exe_dir.join("contracts");
            if exe_contracts.join("model-families").exists() {
                return load_family_registry(&exe_contracts).map_err(|e| {
                    CliError::Aprender(format!("Failed to load family contracts: {e}"))
                });
            }
            // Try one level up (for cargo run scenarios)
            if let Some(parent) = exe_dir.parent() {
                let parent_contracts = parent.join("contracts");
                if parent_contracts.join("model-families").exists() {
                    return load_family_registry(&parent_contracts).map_err(|e| {
                        CliError::Aprender(format!("Failed to load family contracts: {e}"))
                    });
                }
            }
        }
    }

    // Try workspace root (common in dev)
    for ancestor in [".", "..", "../..", "../../.."] {
        let contracts = PathBuf::from(ancestor).join("contracts");
        if contracts.join("model-families").exists() {
            return load_family_registry(&contracts)
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
        expected_tensor_count: family
            .expected_tensor_count(size_name)
            .unwrap_or(0),
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
    let expected_set: std::collections::HashSet<&str> = expected.iter().map(String::as_str).collect();

    let missing: Vec<String> = expected_set
        .difference(&actual_set)
        .map(|s| (*s).to_string())
        .collect();

    let unexpected: Vec<String> = actual_set
        .difference(&expected_set)
        .map(|s| (*s).to_string())
        .collect();

    let expected_count = family
        .expected_tensor_count(size)
        .unwrap_or(expected.len());
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

fn output_text(report: &ModelOracleReport, verbose: bool) {
    output::section("Model Oracle Report");

    output::kv("Source", &report.source);
    output::kv("Mode", format!("{:?}", report.mode));

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

fn output_text_format(fmt: Option<&FormatInfo>) {
    let Some(fmt) = fmt else { return };
    output::kv("Format", &fmt.format_type);
    output::kv("File Size", output::format_size(fmt.file_size as u64));
    output::kv("Tensors", fmt.tensor_count);
    output::kv("Parameters", format_params(fmt.total_params));
    if let Some(ref q) = fmt.quantization {
        output::kv("Quantization", q);
    }
    if let Some(ref arch) = fmt.architecture {
        output::kv("Architecture", arch);
    }
}

fn output_text_family(family: Option<&FamilyInfo>, verbose: bool) {
    let Some(family) = family else {
        println!();
        output::kv("Family", "UNKNOWN (no matching contract)");
        return;
    };
    println!();
    output::kv(
        "Family",
        format!("{} ({})", family.name, family.display_name),
    );
    output::kv("Vendor", &family.vendor);
    if verbose {
        output::kv("Architectures", family.architectures.join(", "));
    }
    if let Some(ref ct) = family.chat_template_format {
        output::kv("Chat Template", ct);
    }
}

fn output_text_size(size: Option<&SizeVariantInfo>) {
    let Some(size) = size else { return };
    println!();
    output::kv(
        "Size",
        format!(
            "{} (hidden={}, layers={}, heads={}, kv_heads={})",
            size.parameters, size.hidden_dim, size.num_layers, size.num_heads, size.num_kv_heads
        ),
    );
    output::kv("Intermediate Dim", size.intermediate_dim);
    output::kv("Vocab Size", size.vocab_size);
    output::kv("Expected Tensors", size.expected_tensor_count);
}

fn output_text_constraints(family: Option<&FamilyInfo>) {
    let Some(family) = family else { return };
    println!();
    println!("  Constraints:");
    let c = &family.constraints;
    println!(
        "    Attention: {} | Activation: {} | Norm: {}",
        c.attention, c.activation, c.norm
    );
    println!(
        "    Bias: {} | Tied: {} | MLP: {} | Position: {}",
        if c.bias { "yes" } else { "no" },
        if c.tied_embeddings { "yes" } else { "no" },
        c.mlp,
        c.positional_encoding
    );
}

fn output_text_stats(stats: Option<&StatisticalAnalysis>) {
    let Some(stats) = stats else { return };
    println!();
    output::section("Statistical Analysis");
    output::kv("GQA Ratio", format!("{:.2} ({:.0}% KV cache reduction)", stats.gqa_ratio, stats.kv_cache_reduction * 100.0));
    output::kv("Model Parameters", format_params(stats.model_params as usize));
    output::kv("Model Size (F16)", format!("{:.1} MB", stats.model_size_f16_mb));
    output::kv("Model Size (Q4_K_M)", format!("{:.1} MB", stats.model_size_q4_mb));
    output::kv("KV Cache/Token", format!("{} bytes", stats.kv_cache_per_token_bytes));
    output::kv("KV Cache (4K ctx)", format!("{:.1} MB", stats.kv_cache_4k_mb));
    output::kv("FFN Expansion", format!("{:.2}x", stats.ffn_expansion_ratio));
    output::kv("FFN Type", &stats.ffn_type_explanation);
    if stats.rope_max_wavelength > 0.0 {
        output::kv("RoPE Wavelength", format!("{:.0}", stats.rope_max_wavelength));
    }
    output::kv("Context Window", stats.effective_context_window);
    output::kv("Attn FLOPS/tok", format!("{:.2e}", stats.attention_flops_per_token as f64));
    output::kv("FFN FLOPS/tok", format!("{:.2e}", stats.ffn_flops_per_token as f64));
}

fn output_text_explanation(expl: Option<&ArchitectureExplanation>) {
    let Some(expl) = expl else { return };
    println!();
    output::section("Architecture Explanation");
    println!();
    println!("  Attention: {}", expl.attention_explanation);
    println!();
    println!("  FFN: {}", expl.ffn_explanation);
    println!();
    println!("  Normalization: {}", expl.norm_explanation);
    println!();
    println!("  Position: {}", expl.positional_explanation);
    println!();
    println!("  Scaling: {}", expl.scaling_analysis);
}

fn output_text_kernels(kern: Option<&KernelCompatibility>) {
    let Some(kern) = kern else { return };
    println!();
    output::section("Kernel Compatibility");
    output::kv("Attention Kernel", &kern.attention_kernel);
    output::kv("FFN Kernel", &kern.ffn_kernel);

    println!();
    println!("  Quantization Support:");
    println!("    {:<12} {:<10} {:<8} {:<12} Kernel", "Format", "Supported", "BPW", "Size (MB)");
    for q in &kern.supported_quantizations {
        println!(
            "    {:<12} {:<10} {:<8.1} {:<12.1} {}",
            q.format,
            if q.supported { "yes" } else { "no" },
            q.bits_per_weight,
            q.estimated_size_mb,
            q.kernel
        );
    }

    if let Some(tps_cpu) = kern.estimated_tps_cpu {
        output::kv("Est. CPU tok/s (Q4_K_M)", format!("{tps_cpu:.0}"));
    }
    if let Some(tps_gpu) = kern.estimated_tps_gpu {
        output::kv("Est. GPU tok/s (Q4_K_M)", format!("{tps_gpu:.0}"));
    }
    output::kv("Memory Required (Q4+KV)", format!("{:.1} MB", kern.memory_required_mb));

    for note in &kern.notes {
        println!("    * {note}");
    }
}

fn output_text_cross_validation(cv: Option<&CrossValidation>) {
    let Some(cv) = cv else { return };
    println!();
    output::section("Cross-Validation (Contract vs HF)");

    if !cv.matches.is_empty() {
        println!("  Matches ({}):", cv.matches.len());
        for entry in &cv.matches {
            println!("    [OK] {}: {} == {}", entry.field, entry.contract_value, entry.hf_value);
        }
    }
    if !cv.mismatches.is_empty() {
        println!("  Mismatches ({}):", cv.mismatches.len());
        for entry in &cv.mismatches {
            println!(
                "    [!!] {}: contract={} vs hf={}",
                entry.field, entry.contract_value, entry.hf_value
            );
        }
    }
    if !cv.contract_only.is_empty() {
        println!("  Contract-only: {}", cv.contract_only.join(", "));
    }
    if !cv.hf_only.is_empty() {
        println!("  HF-only: {}", cv.hf_only.join(", "));
    }
}

fn output_text_hf(hf: Option<&HuggingFaceData>, verbose: bool) {
    let Some(hf) = hf else { return };
    if !verbose { return; }
    println!();
    output::kv("HF Repo", &hf.repo);
    if let Some(ref mt) = hf.model_type {
        output::kv("HF model_type", mt);
    }
    if let Some(ref pt) = hf.pipeline_tag {
        output::kv("HF pipeline_tag", pt);
    }
    if let Some(dl) = hf.downloads {
        output::kv("HF Downloads", format!("{dl}"));
    }
}

fn output_text_compliance(compliance: Option<&ComplianceResult>, verbose: bool) {
    let Some(compliance) = compliance else { return };
    println!();
    if compliance.is_compliant {
        output::kv("Contract", "COMPLIANT");
        return;
    }
    output::kv("Contract", "NON-COMPLIANT");
    if !compliance.tensor_count_match {
        output::kv("Tensor Count", "MISMATCH");
    }
    if !compliance.missing_tensors.is_empty() {
        output::kv(
            "Missing Tensors",
            format!("{} tensor(s)", compliance.missing_tensors.len()),
        );
        if verbose {
            for t in &compliance.missing_tensors {
                println!("    - {t}");
            }
        }
    }
    if !compliance.unexpected_tensors.is_empty() && verbose {
        output::kv(
            "Unexpected Tensors",
            format!("{} tensor(s)", compliance.unexpected_tensors.len()),
        );
        for t in &compliance.unexpected_tensors {
            println!("    + {t}");
        }
    }
}

fn output_text_certification(cert: Option<&CertificationInfo>) {
    let Some(cert) = cert else { return };
    println!();
    output::kv("Certification", &cert.status);
    if let Some(ref pb) = cert.playbook_path {
        output::kv("Playbook", pb);
    }
}

fn output_text_tensors(tensors: Option<&Vec<TensorComplianceEntry>>, verbose: bool) {
    let Some(tensors) = tensors else { return };
    println!();
    println!("  Tensors ({} total):", tensors.len());
    let max_show = if verbose { tensors.len() } else { 20 };
    for (i, t) in tensors.iter().enumerate() {
        if i >= max_show && i < tensors.len() - 2 {
            if i == max_show {
                println!("    ... ({} more) ...", tensors.len() - max_show - 2);
            }
            continue;
        }
        let shape_str = t
            .shape
            .as_ref()
            .map(|s| format!("{s:?}"))
            .unwrap_or_default();
        let dtype_str = t.dtype.as_deref().unwrap_or("");
        println!("    {:<50} {} {}", t.name, dtype_str, shape_str);
    }
}

fn output_family_description(
    config: &ModelFamilyConfig,
    size_filter: Option<&str>,
    verbose: bool,
    flags: OracleFlags,
    family: &dyn ModelFamily,
) {
    output::section(&format!("{} Family Contract", config.display_name));

    output::kv("Family", &config.family);
    output::kv("Vendor", &config.vendor);
    output::kv("Architectures", config.architectures.join(", "));
    output::kv("HF Pattern", &config.hf_pattern);

    // Constraints
    let c = &config.constraints;
    println!();
    println!("  Constraints:");
    println!(
        "    Attention: {} | Activation: {} | Norm: {}",
        c.attention_type, c.activation, c.norm_type
    );
    println!(
        "    Bias: {} | Tied: {} | MLP: {} | Position: {}",
        if c.has_bias { "yes" } else { "no" },
        if c.tied_embeddings { "yes" } else { "no" },
        c.mlp_type,
        c.positional_encoding
    );

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
        println!();
        output::kv("Size Variant", format!("{name} ({})", sc.parameters));
        output::kv("  hidden_dim", sc.hidden_dim);
        output::kv("  num_layers", sc.num_layers);
        output::kv("  num_heads", sc.num_heads);
        output::kv("  num_kv_heads", sc.num_kv_heads);
        output::kv("  intermediate_dim", sc.intermediate_dim);
        output::kv("  vocab_size", sc.vocab_size);
        output::kv("  head_dim", sc.head_dim);
        if sc.rope_theta > 0.0 {
            output::kv("  rope_theta", sc.rope_theta);
        }
        output::kv("  norm_eps", sc.norm_eps);

        // Enhanced sections per size variant
        if flags.show_stats() || flags.show_explain() || flags.show_kernels() {
            let stats = build_statistical_analysis(sc, family.constraints());

            if flags.show_stats() {
                println!();
                output::kv("  GQA Ratio", format!("{:.2} ({:.0}% KV reduction)", stats.gqa_ratio, stats.kv_cache_reduction * 100.0));
                output::kv("  Est. Parameters", format_params(stats.model_params as usize));
                output::kv("  Model Size (F16)", format!("{:.1} MB", stats.model_size_f16_mb));
                output::kv("  Model Size (Q4)", format!("{:.1} MB", stats.model_size_q4_mb));
                output::kv("  KV Cache (4K)", format!("{:.1} MB", stats.kv_cache_4k_mb));
                output::kv("  FFN Ratio", format!("{:.2}x", stats.ffn_expansion_ratio));
            }

            if flags.show_explain() {
                let expl = build_architecture_explanation(sc, family.constraints(), &stats);
                println!();
                println!("    Attention: {}", expl.attention_explanation);
                println!("    FFN: {}", expl.ffn_explanation);
                println!("    Scaling: {}", expl.scaling_analysis);
            }

            if flags.show_kernels() {
                let kern = build_kernel_compatibility(sc, family.constraints(), &stats);
                println!();
                output::kv("  Attn Kernel", &kern.attention_kernel);
                output::kv("  FFN Kernel", &kern.ffn_kernel);
                if let Some(tps) = kern.estimated_tps_cpu {
                    output::kv("  Est. CPU tok/s", format!("{tps:.0}"));
                }
                if let Some(tps) = kern.estimated_tps_gpu {
                    output::kv("  Est. GPU tok/s", format!("{tps:.0}"));
                }
                output::kv("  Memory (Q4+KV)", format!("{:.1} MB", kern.memory_required_mb));
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

    // Tensor template
    if verbose {
        println!();
        println!("  Tensor Template:");
        println!("    Embedding: {}", config.tensor_template.embedding);
        if let Some(ref lm) = config.tensor_template.lm_head {
            println!("    LM Head: {lm}");
        }
        if let Some(ref norm) = config.tensor_template.final_norm {
            println!("    Final Norm: {norm}");
        }
        println!("    Per-layer:");
        for (role, pattern) in &config.tensor_template.per_layer {
            if let Some(pat) = pattern {
                println!("      {role}: {pat}");
            }
        }
    }

    // Quantizations
    if !config.quantizations.is_empty() {
        println!();
        output::kv("Quantizations", config.quantizations.join(", "));
    }

    // Chat template
    if let Some(ref ct) = config.chat_template {
        println!();
        output::kv("Chat Template", &ct.format);
        output::kv("  BOS", &ct.bos_token);
        output::kv("  EOS", &ct.eos_token);
    }
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
        assert!(ratio > 5.0 && ratio < 6.5, "FFN ratio {ratio} out of expected range");
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
        assert!(cv.mismatches.is_empty(), "Expected no mismatches, got: {:?}", cv.mismatches);
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
}
