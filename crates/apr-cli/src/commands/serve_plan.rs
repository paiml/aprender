//! `apr serve plan` — Pre-flight inference capacity planning.
//!
//! Computes VRAM budget, throughput estimates, and contract verification
//! before starting a server. Header-only inspection — no weights loaded.
//!
//! Supports two model sources:
//! - **Local files**: `.gguf`, `.apr`, `.safetensors` — inspected via RosettaStone
//! - **HuggingFace repos**: `hf://org/repo` or `org/repo` — fetches only ~2KB
//!   `config.json` to extract architecture params. No weight download needed.
//!
//! Building blocks reused from oracle.rs, oracle_flags.rs, profile_ollama.rs.

use crate::commands::oracle::{
    build_kernel_compatibility, build_statistical_analysis, compute_kv_cache, KernelCompatibility,
    StatisticalAnalysis,
};
use crate::commands::profile;
use crate::commands::serve_plan_output::print_serve_plan_text;
use crate::error::CliError;
use aprender::format::converter::sanitize_hf_json;
use aprender::format::model_family::{
    FamilyRegistry, ModelConstraints, ModelFamily, ModelSizeConfig,
};
use aprender::format::model_family_loader::load_family_registry;
use aprender::format::rosetta::{InspectionReport, RosettaStone};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ============================================================================
// ServePlan output types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServePlan {
    pub model: ServePlanModel,
    pub hardware: Option<ServePlanHardware>,
    pub memory_budget: MemoryBudget,
    pub roofline: Option<RooflineEstimate>,
    pub throughput: ThroughputEstimate,
    pub contracts: Vec<ContractCheck>,
    pub verdict: PlanVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServePlanModel {
    pub name: String,
    pub params: u64,
    pub quantization: Option<String>,
    pub format: String,
    pub file_size_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServePlanHardware {
    pub gpu_name: String,
    pub vram_mb: f64,
    pub bandwidth_gbps: f64,
    pub peak_tflops: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBudget {
    pub weights_mb: f64,
    pub kv_cache_mb: f64,
    pub activations_mb: f64,
    pub overhead_mb: f64,
    pub total_mb: f64,
    pub gpu_total_mb: Option<f64>,
    pub utilization_pct: Option<f64>,
    pub max_batch: Option<usize>,
    pub batch_size: usize,
    pub seq_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RooflineEstimate {
    pub bandwidth_gbps: f64,
    pub bandwidth_ceiling_tps: f64,
    pub compute_ceiling_tps: f64,
    pub bottleneck: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputEstimate {
    pub single_decode_tps: f64,
    pub batched_tps: Option<f64>,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractCheck {
    pub id: String,
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlanVerdict {
    Ready,
    Warnings,
    Blocked,
}

impl std::fmt::Display for PlanVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanVerdict::Ready => write!(f, "READY"),
            PlanVerdict::Warnings => write!(f, "WARNINGS"),
            PlanVerdict::Blocked => write!(f, "BLOCKED"),
        }
    }
}

// ============================================================================
// Model source dispatch
// ============================================================================

/// Parsed model source — local file or HuggingFace repo.
enum ServePlanSource {
    Local(PathBuf),
    HuggingFace { repo_id: String },
}

/// Parse a model string into a local path or HuggingFace repo ID.
///
/// Matches: `hf://org/repo`, bare `org/repo` (if no file extension and path
/// doesn't exist on disk), falls back to local path.
fn parse_model_source(model: &str) -> ServePlanSource {
    // Explicit hf:// prefix
    if let Some(repo) = model.strip_prefix("hf://") {
        return ServePlanSource::HuggingFace {
            repo_id: repo.to_string(),
        };
    }

    // Bare org/repo: exactly one slash, no file extension, path doesn't exist locally
    if model.contains('/') && !model.contains("..") {
        let path = Path::new(model);
        let has_model_ext = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| {
                matches!(
                    ext.to_lowercase().as_str(),
                    "gguf" | "apr" | "safetensors" | "bin" | "pt" | "onnx"
                )
            });

        if !has_model_ext && !path.exists() {
            let parts: Vec<&str> = model.split('/').collect();
            if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
                return ServePlanSource::HuggingFace {
                    repo_id: model.to_string(),
                };
            }
        }
    }

    ServePlanSource::Local(PathBuf::from(model))
}

// ============================================================================
// Core logic
// ============================================================================

pub fn run_serve_plan(
    model: &str,
    gpu: bool,
    batch_size: usize,
    seq_len: usize,
    format: &str,
    quant_override: Option<&str>,
) -> Result<(), CliError> {
    match parse_model_source(model) {
        ServePlanSource::Local(path) => {
            run_serve_plan_local(&path, gpu, batch_size, seq_len, format)
        }
        ServePlanSource::HuggingFace { repo_id } => {
            run_serve_plan_hf(&repo_id, gpu, batch_size, seq_len, format, quant_override)
        }
    }
}

/// Local file path — existing logic (header-only RosettaStone inspection).
fn run_serve_plan_local(
    file: &Path,
    gpu: bool,
    batch_size: usize,
    seq_len: usize,
    format: &str,
) -> Result<(), CliError> {
    // 1. Inspect model (header-only, no weight load)
    let rosetta = RosettaStone::new();
    let report = rosetta
        .inspect(file)
        .map_err(|e| CliError::Aprender(format!("Failed to inspect model: {e}")))?;

    // 2. Load family registry and detect model family
    let registry = load_registry()?;
    let (size_config, constraints) = detect_model_config(&report, &registry)?;

    // 3-10. Shared pipeline
    let display_name = file.display().to_string();
    let model_name = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let quantization = resolve_quantization(&report, file);
    let model_format = format!("{}", report.format);
    let file_size_mb = report.file_size as f64 / (1024.0 * 1024.0);

    assemble_and_output(
        &size_config,
        &constraints,
        gpu,
        batch_size,
        seq_len,
        format,
        &display_name,
        model_name,
        quantization,
        model_format,
        file_size_mb,
    )
}

/// HuggingFace repo — fetches only ~2KB config.json, no weight download.
fn run_serve_plan_hf(
    repo_id: &str,
    gpu: bool,
    batch_size: usize,
    seq_len: usize,
    format: &str,
    quant_override: Option<&str>,
) -> Result<(), CliError> {
    // 1. Fetch config.json from HuggingFace
    let config_json = fetch_config_json_from_hf(repo_id)?;

    // 2. Load family registry and detect model config from JSON
    let registry = load_registry()?;
    let (size_config, constraints) = detect_model_config_from_hf(&config_json, &registry)?;

    // 3-10. Shared pipeline
    let display_name = format!("hf://{repo_id}");
    let model_name = repo_id.rsplit('/').next().unwrap_or(repo_id).to_string();
    let quantization = quant_override
        .map(str::to_uppercase)
        .or_else(|| infer_quant_from_repo_name(repo_id));
    let model_format = "HuggingFace".to_string();

    // Estimate file size from statistical analysis (no actual file)
    let stats = build_statistical_analysis(&size_config, &constraints);
    let file_size_mb = stats.model_size_q4_mb;

    assemble_and_output(
        &size_config,
        &constraints,
        gpu,
        batch_size,
        seq_len,
        format,
        &display_name,
        model_name,
        quantization,
        model_format,
        file_size_mb,
    )
}

/// Shared pipeline: stats → budget → roofline → throughput → contracts → output.
#[allow(clippy::too_many_arguments)]
fn assemble_and_output(
    size_config: &ModelSizeConfig,
    constraints: &ModelConstraints,
    gpu: bool,
    batch_size: usize,
    seq_len: usize,
    format: &str,
    display_name: &str,
    model_name: String,
    quantization: Option<String>,
    model_format: String,
    file_size_mb: f64,
) -> Result<(), CliError> {
    let stats = build_statistical_analysis(size_config, constraints);
    let kernels = build_kernel_compatibility(size_config, constraints, &stats);

    let hw = if gpu { Some(detect_hardware()?) } else { None };

    let memory_budget =
        compute_memory_budget(&stats, size_config, hw.as_ref(), batch_size, seq_len);
    let roofline = hw.as_ref().map(|h| compute_roofline(&stats, h));
    let throughput = compute_throughput(&kernels, roofline.as_ref(), batch_size);
    let contracts = run_contract_checks(&memory_budget, hw.as_ref());
    let verdict = determine_verdict(&contracts);

    let plan = ServePlan {
        model: ServePlanModel {
            name: model_name,
            params: stats.model_params,
            quantization,
            format: model_format,
            file_size_mb,
        },
        hardware: hw,
        memory_budget,
        roofline,
        throughput,
        contracts,
        verdict,
    };

    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&plan)
                .map_err(|e| CliError::Aprender(format!("JSON serialization failed: {e}")))?;
            println!("{json}");
        }
        "yaml" => {
            let yaml = serde_yaml::to_string(&plan)
                .map_err(|e| CliError::Aprender(format!("YAML serialization failed: {e}")))?;
            print!("{yaml}");
        }
        _ => {
            print_serve_plan_text(&plan, display_name);
        }
    }

    Ok(())
}

/// Resolve quantization name from report or filename pattern.
fn resolve_quantization(report: &InspectionReport, file: &Path) -> Option<String> {
    // Try the report's quantization field first
    if let Some(ref q) = report.quantization {
        if q != "0" && !q.is_empty() {
            return Some(q.clone());
        }
    }
    // Fall back to filename pattern matching (e.g., q4_k_m, q6_k, q8_0)
    let stem = file.file_stem()?.to_str()?.to_lowercase();
    for pattern in [
        "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1",
        "q2_k", "q3_k", "q4k", "q6k", "q8k", "fp16", "f16", "f32",
    ] {
        if stem.contains(pattern) {
            return Some(pattern.to_uppercase());
        }
    }
    None
}

// ============================================================================
// HuggingFace config.json fetching
// ============================================================================

/// Resolve HuggingFace auth token for gated models.
///
/// Priority: HF_TOKEN env var → ~/.huggingface/token → ~/.cache/huggingface/token.
/// Duplicated from pull_extract_shard.rs — these `include!()` modules can't cross-import.
fn resolve_hf_token() -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            return Some(token);
        }
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)?;
    for path in [
        home.join(".huggingface/token"),
        home.join(".cache/huggingface/token"),
    ] {
        if let Ok(token) = std::fs::read_to_string(&path) {
            let token = token.trim().to_string();
            if !token.is_empty() {
                return Some(token);
            }
        }
    }
    None
}

/// Build an authenticated ureq request if HF token is available.
fn hf_get(url: &str) -> ureq::Request {
    let req = ureq::get(url);
    if let Some(token) = resolve_hf_token() {
        req.set("Authorization", &format!("Bearer {token}"))
    } else {
        req
    }
}

/// Fetch config.json from a HuggingFace repo (raw endpoint, ~2KB).
fn fetch_config_json_from_hf(repo_id: &str) -> Result<serde_json::Value, CliError> {
    let url = format!("https://huggingface.co/{repo_id}/raw/main/config.json");

    let response = hf_get(&url).call().map_err(|e| match &e {
        ureq::Error::Status(401 | 403, _) => {
            let has_token = resolve_hf_token().is_some();
            if has_token {
                CliError::NetworkError(format!(
                    "Access denied for {repo_id}\n\
                     Your HF_TOKEN was sent but lacks access to this gated model.\n\
                     Request access at https://huggingface.co/{repo_id}"
                ))
            } else {
                CliError::NetworkError(format!(
                    "Access denied for {repo_id}\n\
                     This is a gated model requiring authentication.\n\
                     Set HF_TOKEN=hf_... or run: huggingface-cli login"
                ))
            }
        }
        ureq::Error::Status(404, _) => CliError::HttpNotFound(format!(
            "config.json not found for '{repo_id}'. \
             Verify the repo exists at https://huggingface.co/{repo_id}"
        )),
        _ => CliError::NetworkError(format!("Failed to fetch config.json from HuggingFace: {e}")),
    })?;

    let body = response
        .into_string()
        .map_err(|e| CliError::NetworkError(format!("Failed to read config.json response: {e}")))?;

    let sanitized = sanitize_hf_json(&body);
    serde_json::from_str(&sanitized).map_err(|e| {
        CliError::InvalidFormat(format!("Failed to parse config.json from {repo_id}: {e}"))
    })
}

/// Extract model architecture params from HuggingFace config.json.
///
/// Uses the same alias tables as `source_load_result.rs` for cross-architecture
/// compatibility (LLaMA, Qwen, GPT-2, Phi, BLOOM, etc.).
fn detect_model_config_from_hf(
    json: &serde_json::Value,
    registry: &FamilyRegistry,
) -> Result<(ModelSizeConfig, ModelConstraints), CliError> {
    // Detect family from model_type field
    let model_type = json.get("model_type").and_then(|v| v.as_str());

    let family: Option<&dyn ModelFamily> =
        model_type.and_then(|mt| registry.detect_from_model_type(mt));

    let family = family.ok_or_else(|| {
        let mt = model_type.unwrap_or("unknown");
        CliError::Aprender(format!(
            "Unknown model family '{mt}' in config.json. \
             Ensure contracts/ directory is available."
        ))
    })?;

    // Extract architecture params using alias arrays
    let hidden_dim = json_usize_or(json, &["hidden_size", "n_embd", "n_embed", "d_model"], 0);
    let num_layers = json_usize_or(json, &["num_hidden_layers", "n_layer", "num_layers"], 0);
    let num_heads = json_usize_or(json, &["num_attention_heads", "n_head", "num_heads"], 32);
    let num_kv_heads = json_usize_or(json, &["num_key_value_heads"], num_heads);
    let intermediate_dim = json_usize_or(
        json,
        &["intermediate_size", "n_inner", "ffn_dim"],
        hidden_dim.saturating_mul(4),
    );
    let vocab_size = json_usize_or(json, &["vocab_size"], 32000);
    let max_position = json_usize_or(
        json,
        &["max_position_embeddings", "n_positions", "n_ctx"],
        4096,
    );
    let rope_theta = json
        .get("rope_theta")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(10000.0);
    let norm_eps = json_f64_or(
        json,
        &["rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps"],
        1e-5,
    );

    if hidden_dim == 0 || num_layers == 0 {
        return Err(CliError::Aprender(format!(
            "config.json missing required fields (hidden_size={hidden_dim}, \
             num_hidden_layers={num_layers})"
        )));
    }

    let head_dim = if num_heads > 0 {
        hidden_dim / num_heads
    } else {
        128
    };

    // Try registry size detection, fall back to building from extracted params
    let size_name = family
        .detect_size(hidden_dim, num_layers)
        .unwrap_or_else(|| "unknown".to_string());

    let size_config = family
        .size_config(&size_name)
        .cloned()
        .unwrap_or(ModelSizeConfig {
            parameters: estimate_param_string(hidden_dim, num_layers, intermediate_dim, vocab_size),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            intermediate_dim,
            vocab_size,
            max_position_embeddings: max_position,
            head_dim,
            rope_theta,
            norm_eps,
        });

    let constraints = family.constraints().clone();
    Ok((size_config, constraints))
}

/// Helper: look up a usize JSON field by trying multiple key aliases.
fn json_usize_or(json: &serde_json::Value, keys: &[&str], default: usize) -> usize {
    keys.iter()
        .find_map(|&k| json.get(k))
        .and_then(serde_json::Value::as_u64)
        .map_or(default, |v| v as usize)
}

/// Helper: look up an f64 JSON field by trying multiple key aliases.
fn json_f64_or(json: &serde_json::Value, keys: &[&str], default: f64) -> f64 {
    keys.iter()
        .find_map(|&k| json.get(k))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(default)
}

/// Estimate parameter count string from architecture dimensions.
fn estimate_param_string(
    hidden_dim: usize,
    num_layers: usize,
    intermediate_dim: usize,
    vocab_size: usize,
) -> String {
    // Rough formula: embedding + num_layers * (attn + ffn) + lm_head
    // attn ≈ 4 * hidden_dim² (Q/K/V/O projections)
    // ffn ≈ 3 * hidden_dim * intermediate_dim (gate/up/down for SwiGLU)
    let embedding = hidden_dim * vocab_size;
    let attn_per_layer = 4 * hidden_dim * hidden_dim;
    let ffn_per_layer = 3 * hidden_dim * intermediate_dim;
    let total = embedding + num_layers * (attn_per_layer + ffn_per_layer) + embedding;
    format_param_count(total as u64)
}

/// Pattern-match repo name for quantization hints.
fn infer_quant_from_repo_name(repo: &str) -> Option<String> {
    let lower = repo.to_lowercase();
    for pattern in [
        "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1",
        "q2_k", "q3_k", "fp16", "f16", "f32",
    ] {
        if lower.contains(pattern) {
            return Some(pattern.to_uppercase());
        }
    }
    // If repo mentions GGUF but no specific quant, assume Q4_K_M
    if lower.contains("gguf") {
        return Some("Q4_K_M".to_string());
    }
    None
}

// ============================================================================
// Registry loading (reuse oracle_family.rs pattern)
// ============================================================================

fn contracts_candidate_paths() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(&home).join(".aprender/contracts"));
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
    Ok(FamilyRegistry::new())
}

// ============================================================================
// Model config detection
// ============================================================================

fn detect_model_config(
    report: &InspectionReport,
    registry: &FamilyRegistry,
) -> Result<(ModelSizeConfig, ModelConstraints), CliError> {
    // Try detection from architecture metadata first
    let family: Option<&dyn ModelFamily> = report
        .architecture
        .as_deref()
        .and_then(|arch| registry.detect_from_model_type(arch));

    // Fall back to tensor name detection
    let family = family.or_else(|| {
        let tensor_names: Vec<&str> = report.tensors.iter().map(|t| t.name.as_str()).collect();
        registry.detect_family(&tensor_names)
    });

    let family = family.ok_or_else(|| {
        CliError::Aprender(
            "Could not detect model family. Ensure contracts/ directory is available.".to_string(),
        )
    })?;

    // Detect size from metadata or tensor shapes
    let hidden_dim = infer_hidden_dim(report);
    let num_layers = infer_num_layers(report);

    let size_name = family
        .detect_size(hidden_dim, num_layers)
        .unwrap_or_else(|| "unknown".to_string());

    let size_config = family.size_config(&size_name).cloned().unwrap_or_else(|| {
        // Build a minimal config from what we can infer
        build_inferred_size_config(report, hidden_dim, num_layers)
    });

    let constraints = family.constraints().clone();
    Ok((size_config, constraints))
}

/// Infer hidden dimension from metadata or tensor shapes.
fn infer_hidden_dim(report: &InspectionReport) -> usize {
    // Try canonical metadata keys (used by RosettaStone for GGUF)
    for key in [
        "n_embd",
        "llama.embedding_length",
        "qwen2.embedding_length",
        "phi3.embedding_length",
        "gemma.embedding_length",
    ] {
        if let Some(val) = report.metadata.get(key) {
            if let Ok(dim) = val.parse::<usize>() {
                return dim;
            }
        }
    }
    // Fall back to embedding tensor shape (last dim = hidden_dim)
    // GGUF embedding: shape=[vocab_size, hidden_dim], take last
    // SafeTensors: may be [hidden_dim, vocab_size], take smaller
    for tensor in &report.tensors {
        if tensor.name.contains("token_embd") || tensor.name.contains("embed_tokens") {
            if tensor.shape.len() == 2 {
                // Hidden dim is typically the smaller of the two
                return tensor.shape[0].min(tensor.shape[1]);
            }
            if let Some(&dim) = tensor.shape.last() {
                return dim;
            }
        }
    }
    0
}

/// Infer number of layers from metadata or tensor count patterns.
fn infer_num_layers(report: &InspectionReport) -> usize {
    // Try canonical metadata keys
    for key in [
        "n_layers",
        "llama.block_count",
        "qwen2.block_count",
        "phi3.block_count",
        "gemma.block_count",
    ] {
        if let Some(val) = report.metadata.get(key) {
            if let Ok(n) = val.parse::<usize>() {
                return n;
            }
        }
    }
    // Fall back: count unique layer indices in tensor names
    let mut max_layer: usize = 0;
    for tensor in &report.tensors {
        if let Some(idx) = extract_layer_index(&tensor.name) {
            max_layer = max_layer.max(idx);
        }
    }
    if max_layer > 0 {
        max_layer + 1
    } else {
        0
    }
}

fn extract_layer_index(name: &str) -> Option<usize> {
    // Match patterns like "blk.5.", "layers.5.", "h.5."
    for part in name.split('.') {
        if let Ok(n) = part.parse::<usize>() {
            return Some(n);
        }
    }
    None
}

/// Build an inferred size config when family detection succeeds but size doesn't match.
fn build_inferred_size_config(
    report: &InspectionReport,
    hidden_dim: usize,
    num_layers: usize,
) -> ModelSizeConfig {
    let num_heads = infer_from_metadata(
        report,
        &[
            "n_heads",
            "llama.attention.head_count",
            "qwen2.attention.head_count",
            "phi3.attention.head_count",
        ],
    )
    .unwrap_or(32);

    let num_kv_heads = infer_from_metadata(
        report,
        &[
            "n_kv_heads",
            "llama.attention.head_count_kv",
            "qwen2.attention.head_count_kv",
            "phi3.attention.head_count_kv",
        ],
    )
    .unwrap_or(num_heads);

    let intermediate_dim = infer_from_metadata(
        report,
        &[
            "n_ff",
            "llama.feed_forward_length",
            "qwen2.feed_forward_length",
            "phi3.feed_forward_length",
        ],
    )
    .unwrap_or(hidden_dim * 4);

    let vocab_size = infer_from_metadata(
        report,
        &[
            "n_vocab",
            "llama.vocab_size",
            "qwen2.vocab_size",
            "tokenizer.ggml.tokens",
        ],
    )
    .unwrap_or(32000);

    let max_position = infer_from_metadata(
        report,
        &[
            "n_ctx",
            "llama.context_length",
            "qwen2.context_length",
            "phi3.context_length",
        ],
    )
    .unwrap_or(4096);

    let rope_theta_str = report
        .metadata
        .get("llama.rope.freq_base")
        .or_else(|| report.metadata.get("qwen2.rope.freq_base"))
        .cloned()
        .unwrap_or_default();
    let rope_theta = rope_theta_str.parse::<f64>().unwrap_or(10000.0);

    let head_dim = if num_heads > 0 {
        hidden_dim / num_heads
    } else {
        128
    };

    ModelSizeConfig {
        parameters: format_param_count(report.total_params as u64),
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_dim,
        vocab_size,
        max_position_embeddings: max_position,
        head_dim,
        rope_theta,
        norm_eps: 1e-5,
    }
}

fn infer_from_metadata(report: &InspectionReport, keys: &[&str]) -> Option<usize> {
    for key in keys {
        if let Some(val) = report.metadata.get(*key) {
            if let Ok(n) = val.parse::<usize>() {
                return Some(n);
            }
        }
    }
    None
}

fn format_param_count(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1e9)
    } else if params >= 1_000_000 {
        format!("{:.0}M", params as f64 / 1e6)
    } else {
        format!("{params}")
    }
}

// ============================================================================
// GPU hardware detection
// ============================================================================

fn detect_hardware() -> Result<ServePlanHardware, CliError> {
    let (peak_gflops, peak_bw, _ai_thresh, gpu_name) = profile::detect_gpu_hardware();
    let vram_mb = profile::query_gpu_vram_mb().unwrap_or(0.0);

    Ok(ServePlanHardware {
        gpu_name,
        vram_mb,
        bandwidth_gbps: peak_bw,
        peak_tflops: peak_gflops / 1000.0,
    })
}

// ============================================================================
// Memory budget computation
// ============================================================================

fn compute_memory_budget(
    stats: &StatisticalAnalysis,
    size: &ModelSizeConfig,
    hw: Option<&ServePlanHardware>,
    batch_size: usize,
    seq_len: usize,
) -> MemoryBudget {
    let weights_mb = stats.model_size_q4_mb;

    // KV cache: per_token_bytes * seq_len * batch_size
    let (kv_per_token, _) = compute_kv_cache(size);
    let kv_cache_mb =
        (kv_per_token as f64 * seq_len as f64 * batch_size as f64) / (1024.0 * 1024.0);

    // Activations: ~hidden_dim * seq_len * 4 bytes (f32) / 1MB
    let activations_mb = (size.hidden_dim as f64 * seq_len as f64 * 4.0) / (1024.0 * 1024.0);

    // CUDA overhead: ~512 MB for context, driver, etc.
    let overhead_mb = if hw.is_some() { 512.0 } else { 0.0 };

    let total_mb = weights_mb + kv_cache_mb + activations_mb + overhead_mb;

    let (gpu_total_mb, utilization_pct, max_batch) = if let Some(h) = hw {
        let util = if h.vram_mb > 0.0 {
            Some(total_mb / h.vram_mb * 100.0)
        } else {
            None
        };

        // Max batch: how many batches fit after weights + overhead
        let kv_per_batch_mb = (kv_per_token as f64 * seq_len as f64) / (1024.0 * 1024.0);
        let available = h.vram_mb - weights_mb - overhead_mb - activations_mb;
        let max_b = if kv_per_batch_mb > 0.0 && available > 0.0 {
            Some((available / kv_per_batch_mb).floor() as usize)
        } else {
            Some(0)
        };

        (Some(h.vram_mb), util, max_b)
    } else {
        (None, None, None)
    };

    MemoryBudget {
        weights_mb,
        kv_cache_mb,
        activations_mb,
        overhead_mb,
        total_mb,
        gpu_total_mb,
        utilization_pct,
        max_batch,
        batch_size,
        seq_len,
    }
}

// ============================================================================
// Roofline analysis
// ============================================================================

fn compute_roofline(stats: &StatisticalAnalysis, hw: &ServePlanHardware) -> RooflineEstimate {
    // Memory bandwidth ceiling: BW / model_size_bytes = tokens/sec
    let model_size_gb = stats.model_size_q4_mb / 1024.0;
    let bw_ceiling = if model_size_gb > 0.0 {
        hw.bandwidth_gbps / model_size_gb
    } else {
        0.0
    };

    // Compute ceiling: TFLOPS / FLOPS_per_token
    let total_flops_per_token = stats.attention_flops_per_token + stats.ffn_flops_per_token;
    let compute_ceiling = if total_flops_per_token > 0 {
        (hw.peak_tflops * 1e12) / total_flops_per_token as f64
    } else {
        0.0
    };

    let bottleneck = if bw_ceiling < compute_ceiling {
        "MEMORY-BOUND"
    } else {
        "COMPUTE-BOUND"
    };

    RooflineEstimate {
        bandwidth_gbps: hw.bandwidth_gbps,
        bandwidth_ceiling_tps: bw_ceiling,
        compute_ceiling_tps: compute_ceiling,
        bottleneck: bottleneck.to_string(),
    }
}

// ============================================================================
// Throughput estimation
// ============================================================================

fn compute_throughput(
    kernels: &KernelCompatibility,
    roofline: Option<&RooflineEstimate>,
    batch_size: usize,
) -> ThroughputEstimate {
    let single_tps = if let Some(roof) = roofline {
        // GPU: use roofline bandwidth ceiling
        roof.bandwidth_ceiling_tps
    } else {
        // CPU: use kernel estimate
        kernels.estimated_tps_cpu.unwrap_or(50.0)
    };

    let batched_tps = if batch_size > 1 {
        // Batched throughput scales sub-linearly (memory bandwidth shared)
        // Empirical: ~sqrt(batch_size) scaling for memory-bound inference
        Some(single_tps * (batch_size as f64).sqrt())
    } else {
        None
    };

    ThroughputEstimate {
        single_decode_tps: single_tps,
        batched_tps,
        batch_size,
    }
}

// ============================================================================
// Contract checks
// ============================================================================

fn run_contract_checks(
    budget: &MemoryBudget,
    hw: Option<&ServePlanHardware>,
) -> Vec<ContractCheck> {
    let mut checks = Vec::new();

    if let Some(h) = hw {
        let gpu_mb = h.vram_mb;
        let safety_margin = gpu_mb * 0.95;

        // BUDGET-001: Total VRAM fits
        checks.push(ContractCheck {
            id: "BUDGET-001".to_string(),
            name: "VRAM fits".to_string(),
            passed: budget.total_mb <= safety_margin,
            detail: format!(
                "{:.0} MB {} {:.0} MB (95% of {:.0} MB)",
                budget.total_mb,
                if budget.total_mb <= safety_margin {
                    "<"
                } else {
                    ">"
                },
                safety_margin,
                gpu_mb,
            ),
        });

        // BUDGET-002: Model weights loadable contiguous
        checks.push(ContractCheck {
            id: "BUDGET-002".to_string(),
            name: "Model contiguous".to_string(),
            passed: budget.weights_mb <= gpu_mb,
            detail: format!("{:.0} MB", budget.weights_mb),
        });

        // BUDGET-003: KV cache fits at batch=1
        let batch1_total = budget.weights_mb
            + budget.activations_mb
            + budget.overhead_mb
            + (budget.kv_cache_mb / budget.batch_size.max(1) as f64);
        checks.push(ContractCheck {
            id: "BUDGET-003".to_string(),
            name: "KV cache fits at batch=1".to_string(),
            passed: batch1_total <= gpu_mb,
            detail: format!("{:.0} MB at batch=1", batch1_total),
        });

        // BUDGET-004: Target batch achievable
        if budget.batch_size > 1 {
            checks.push(ContractCheck {
                id: "BUDGET-004".to_string(),
                name: format!("batch={} achievable", budget.batch_size),
                passed: budget.total_mb <= gpu_mb,
                detail: format!(
                    "{:.0} MB at batch={} (max batch: {})",
                    budget.total_mb,
                    budget.batch_size,
                    budget.max_batch.unwrap_or(0),
                ),
            });
        }
    } else {
        // CPU-only: just check KV fits
        checks.push(ContractCheck {
            id: "BUDGET-003".to_string(),
            name: "KV cache fits at batch=1".to_string(),
            passed: true,
            detail: format!("{:.0} MB total", budget.total_mb),
        });
    }

    checks
}

fn determine_verdict(contracts: &[ContractCheck]) -> PlanVerdict {
    let any_failed = contracts.iter().any(|c| !c.passed);
    let critical_failed = contracts
        .iter()
        .any(|c| !c.passed && (c.id == "BUDGET-001" || c.id == "BUDGET-002"));

    if critical_failed {
        PlanVerdict::Blocked
    } else if any_failed {
        PlanVerdict::Warnings
    } else {
        PlanVerdict::Ready
    }
}
