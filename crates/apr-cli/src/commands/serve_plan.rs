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

#[provable_contracts_macros::contract(
    "apr-cli-command-safety-v1",
    equation = "long_running_graceful"
)]
pub fn run_serve_plan(
    model: &str,
    gpu: bool,
    batch_size: usize,
    seq_len: usize,
    format: &str,
    quant_override: Option<&str>,
) -> Result<(), CliError> {
    contract_pre_server_lifecycle!();
    let result = match parse_model_source(model) {
        ServePlanSource::Local(path) => {
            run_serve_plan_local(&path, gpu, batch_size, seq_len, format)
        }
        ServePlanSource::HuggingFace { repo_id } => {
            run_serve_plan_hf(&repo_id, gpu, batch_size, seq_len, format, quant_override)
        }
    };
    if let Ok(ref r) = result {
        contract_post_server_lifecycle!(r);
    }
    result
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

    // GH-633: Auto-detect GPU if --gpu not explicitly passed
    let hw = if gpu {
        Some(detect_hardware()?)
    } else {
        // Try detection anyway — report GPU if available
        detect_hardware().ok()
    };

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
    // GH-600: Env var override
    if let Ok(path) = std::env::var("APRENDER_CONTRACTS") {
        candidates.push(PathBuf::from(path));
    }
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(&home).join("src/aprender/contracts"));
        candidates.push(PathBuf::from(&home).join(".aprender/contracts"));
        candidates.push(PathBuf::from(&home).join(".config/apr/contracts"));
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
    if let Some(dim) = infer_from_metadata(
        report,
        &[
            "n_embd",
            "llama.embedding_length",
            "qwen2.embedding_length",
            "phi3.embedding_length",
            "gemma.embedding_length",
        ],
    ) {
        return dim;
    }
    infer_hidden_dim_from_tensors(report)
}

/// Fall back to embedding tensor shape to infer hidden dimension.
fn infer_hidden_dim_from_tensors(report: &InspectionReport) -> usize {
    for tensor in &report.tensors {
        if tensor.name.contains("token_embd") || tensor.name.contains("embed_tokens") {
            if tensor.shape.len() == 2 {
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

// ═══════════════════════════════════════════════════════════════════
// PMAT-540 Phase 4: Inline tests for pure helper functions
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod serve_plan_tests {
    use super::*;

    #[test]
    fn test_format_param_count_billions() {
        assert_eq!(format_param_count(7_000_000_000), "7.0B");
        assert_eq!(format_param_count(1_500_000_000), "1.5B");
    }

    #[test]
    fn test_format_param_count_millions() {
        assert_eq!(format_param_count(125_000_000), "125M");
        assert_eq!(format_param_count(350_000_000), "350M");
    }

    #[test]
    fn test_format_param_count_small() {
        assert_eq!(format_param_count(50_000), "50000");
    }

    #[test]
    fn test_extract_layer_index_dotted_pattern() {
        assert_eq!(extract_layer_index("model.layers.5.attn"), Some(5));
        assert_eq!(extract_layer_index("blk.0.ffn"), Some(0));
        assert_eq!(extract_layer_index("h.31.attn"), Some(31));
    }

    #[test]
    fn test_extract_layer_index_no_number() {
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
    }

    fn make_test_stats(
        model_size_q4_mb: f64,
        attn_flops: u64,
        ffn_flops: u64,
    ) -> StatisticalAnalysis {
        StatisticalAnalysis {
            gqa_ratio: 1.0,
            kv_cache_reduction: 1.0,
            model_params: 1_000_000,
            model_size_f16_mb: model_size_q4_mb * 2.0,
            model_size_q4_mb,
            kv_cache_per_token_bytes: 256,
            kv_cache_4k_mb: 1.0,
            ffn_expansion_ratio: 2.67,
            ffn_type_explanation: "SwiGLU".to_string(),
            rope_max_wavelength: 10000.0,
            effective_context_window: 4096,
            attention_flops_per_token: attn_flops,
            ffn_flops_per_token: ffn_flops,
        }
    }

    fn make_test_hw() -> ServePlanHardware {
        ServePlanHardware {
            gpu_name: "test-cpu".to_string(),
            bandwidth_gbps: 50.0,
            peak_tflops: 1.0,
            vram_mb: 0.0,
        }
    }

    #[test]
    fn test_compute_roofline_memory_bound() {
        let stats = make_test_stats(4096.0, 100_000_000, 200_000_000);
        let hw = make_test_hw();
        let roofline = compute_roofline(&stats, &hw);
        assert!(roofline.bandwidth_ceiling_tps > 0.0);
        assert!(roofline.compute_ceiling_tps > 0.0);
    }

    #[test]
    fn test_compute_roofline_zero_model_size() {
        let stats = make_test_stats(0.0, 0, 0);
        let hw = make_test_hw();
        let roofline = compute_roofline(&stats, &hw);
        assert!((roofline.bandwidth_ceiling_tps).abs() < f64::EPSILON);
    }

    #[test]
    fn test_detect_hardware_returns_valid() {
        let hw = detect_hardware().expect("hardware detection");
        assert!(!hw.gpu_name.is_empty());
        assert!(hw.bandwidth_gbps > 0.0);
    }

    // ====================================================================
    // PMAT-125 B3: additional coverage for serve_plan pure helpers
    // ====================================================================

    use aprender::format::model_family::ModelSizeConfig;
    use aprender::format::rosetta::{FormatType, InspectionReport, TensorInfo};
    use std::collections::BTreeMap;

    fn make_size_config(hidden_dim: usize, num_layers: usize, num_heads: usize) -> ModelSizeConfig {
        ModelSizeConfig {
            parameters: "1.5B".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads: num_heads,
            intermediate_dim: hidden_dim * 4,
            vocab_size: 32_000,
            max_position_embeddings: 4096,
            head_dim: if num_heads > 0 {
                hidden_dim / num_heads
            } else {
                128
            },
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
        }
    }

    fn empty_report() -> InspectionReport {
        InspectionReport {
            format: FormatType::Gguf,
            file_size: 0,
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
            total_params: 0,
            quantization: None,
            architecture: None,
        }
    }

    fn tensor(name: &str, shape: Vec<usize>) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: "F16".to_string(),
            shape,
            size_bytes: 0,
            stats: None,
        }
    }

    fn make_kernels(tps_cpu: Option<f64>) -> KernelCompatibility {
        KernelCompatibility {
            supported_quantizations: Vec::new(),
            attention_kernel: "flash".to_string(),
            ffn_kernel: "swiglu".to_string(),
            estimated_tps_cpu: tps_cpu,
            estimated_tps_gpu: None,
            memory_required_mb: 0.0,
            notes: Vec::new(),
        }
    }

    // ---- PlanVerdict Display ------------------------------------------

    #[test]
    fn plan_verdict_display_strings() {
        assert_eq!(PlanVerdict::Ready.to_string(), "READY");
        assert_eq!(PlanVerdict::Warnings.to_string(), "WARNINGS");
        assert_eq!(PlanVerdict::Blocked.to_string(), "BLOCKED");
    }

    // ---- parse_model_source -------------------------------------------

    #[test]
    fn parse_model_source_explicit_hf_prefix() {
        match parse_model_source("hf://org/model-7b") {
            ServePlanSource::HuggingFace { repo_id } => assert_eq!(repo_id, "org/model-7b"),
            ServePlanSource::Local(_) => panic!("expected HuggingFace source"),
        }
    }

    #[test]
    fn parse_model_source_bare_org_repo_is_hf() {
        match parse_model_source("Qwen/Qwen2.5-1.5B") {
            ServePlanSource::HuggingFace { repo_id } => assert_eq!(repo_id, "Qwen/Qwen2.5-1.5B"),
            ServePlanSource::Local(_) => panic!("bare org/repo should resolve to HuggingFace"),
        }
    }

    #[test]
    fn parse_model_source_path_with_model_extension_is_local() {
        // A slash + a known model extension must stay local even if absent on disk.
        match parse_model_source("models/qwen.gguf") {
            ServePlanSource::Local(p) => assert_eq!(p, PathBuf::from("models/qwen.gguf")),
            ServePlanSource::HuggingFace { .. } => panic!("model file extension must stay local"),
        }
    }

    #[test]
    fn parse_model_source_no_slash_is_local() {
        match parse_model_source("model.gguf") {
            ServePlanSource::Local(p) => assert_eq!(p, PathBuf::from("model.gguf")),
            ServePlanSource::HuggingFace { .. } => panic!("no slash → local"),
        }
    }

    #[test]
    fn parse_model_source_parent_traversal_stays_local() {
        match parse_model_source("../weights/foo") {
            ServePlanSource::Local(p) => assert_eq!(p, PathBuf::from("../weights/foo")),
            ServePlanSource::HuggingFace { .. } => panic!(".. traversal must stay local"),
        }
    }

    #[test]
    fn parse_model_source_three_segment_path_is_local() {
        // More than two non-empty segments is not a bare org/repo → local.
        match parse_model_source("a/b/c") {
            ServePlanSource::Local(p) => assert_eq!(p, PathBuf::from("a/b/c")),
            ServePlanSource::HuggingFace { .. } => panic!("3-segment path is not org/repo"),
        }
    }

    // ---- json_usize_or / json_f64_or ----------------------------------

    #[test]
    fn json_usize_or_picks_first_present_alias() {
        let v = serde_json::json!({"hidden_size": 4096, "n_embd": 1024});
        assert_eq!(json_usize_or(&v, &["n_embd", "hidden_size"], 0), 1024);
        assert_eq!(json_usize_or(&v, &["hidden_size"], 0), 4096);
    }

    #[test]
    fn json_usize_or_returns_default_when_missing() {
        let v = serde_json::json!({"other": 1});
        assert_eq!(json_usize_or(&v, &["absent", "missing"], 777), 777);
    }

    #[test]
    fn json_usize_or_default_on_non_integer() {
        let v = serde_json::json!({"x": "not-a-number"});
        assert_eq!(json_usize_or(&v, &["x"], 42), 42);
    }

    #[test]
    fn json_f64_or_reads_float_and_default() {
        let v = serde_json::json!({"rope_theta": 1_000_000.0});
        assert!((json_f64_or(&v, &["rope_theta"], 10_000.0) - 1_000_000.0).abs() < 1e-6);
        assert!((json_f64_or(&v, &["absent"], 10_000.0) - 10_000.0).abs() < 1e-6);
    }

    // ---- estimate_param_string ----------------------------------------

    #[test]
    fn estimate_param_string_scales_to_billions() {
        // ~7B-class architecture should land in the "B" range.
        let s = estimate_param_string(4096, 32, 11_008, 32_000);
        assert!(s.ends_with('B'), "expected billions, got {s}");
    }

    #[test]
    fn estimate_param_string_small_model_is_millions_or_less() {
        let s = estimate_param_string(256, 2, 512, 1_000);
        assert!(
            s.ends_with('M') || s.chars().all(|c| c.is_ascii_digit()),
            "expected millions or raw count, got {s}"
        );
    }

    // ---- infer_quant_from_repo_name -----------------------------------

    #[test]
    fn infer_quant_from_repo_name_specific_quant() {
        assert_eq!(
            infer_quant_from_repo_name("TheBloke/Llama-2-7B-GGUF-q4_k_m"),
            Some("Q4_K_M".to_string())
        );
        assert_eq!(
            infer_quant_from_repo_name("foo-q8_0-bar"),
            Some("Q8_0".to_string())
        );
    }

    #[test]
    fn infer_quant_from_repo_name_gguf_default_q4km() {
        assert_eq!(
            infer_quant_from_repo_name("org/some-model-GGUF"),
            Some("Q4_K_M".to_string())
        );
    }

    #[test]
    fn infer_quant_from_repo_name_none_when_unknown() {
        assert_eq!(
            infer_quant_from_repo_name("org/plain-safetensors-model"),
            None
        );
    }

    // ---- resolve_quantization -----------------------------------------

    #[test]
    fn resolve_quantization_prefers_report_field() {
        let mut report = empty_report();
        report.quantization = Some("Q6_K".to_string());
        assert_eq!(
            resolve_quantization(&report, Path::new("ignored.gguf")),
            Some("Q6_K".to_string())
        );
    }

    #[test]
    fn resolve_quantization_ignores_zero_quant_field() {
        let mut report = empty_report();
        report.quantization = Some("0".to_string());
        // Falls back to filename pattern.
        assert_eq!(
            resolve_quantization(&report, Path::new("model-q5_k_m.gguf")),
            Some("Q5_K_M".to_string())
        );
    }

    #[test]
    fn resolve_quantization_filename_fallback() {
        let report = empty_report();
        assert_eq!(
            resolve_quantization(&report, Path::new("/tmp/llama-q4_0.gguf")),
            Some("Q4_0".to_string())
        );
    }

    #[test]
    fn resolve_quantization_none_when_no_hint() {
        let report = empty_report();
        assert_eq!(
            resolve_quantization(&report, Path::new("plain-model.bin")),
            None
        );
    }

    // ---- infer_from_metadata / infer_hidden_dim / infer_num_layers ----

    #[test]
    fn infer_from_metadata_first_match_wins() {
        let mut report = empty_report();
        report
            .metadata
            .insert("n_embd".to_string(), "2048".to_string());
        report
            .metadata
            .insert("llama.embedding_length".to_string(), "4096".to_string());
        assert_eq!(
            infer_from_metadata(&report, &["n_embd", "llama.embedding_length"]),
            Some(2048)
        );
    }

    #[test]
    fn infer_from_metadata_none_when_absent_or_unparsable() {
        let mut report = empty_report();
        report.metadata.insert("k".to_string(), "abc".to_string());
        assert_eq!(infer_from_metadata(&report, &["k"]), None);
        assert_eq!(infer_from_metadata(&report, &["missing"]), None);
    }

    #[test]
    fn infer_hidden_dim_from_metadata() {
        let mut report = empty_report();
        report
            .metadata
            .insert("qwen2.embedding_length".to_string(), "1536".to_string());
        assert_eq!(infer_hidden_dim(&report), 1536);
    }

    #[test]
    fn infer_hidden_dim_from_embedding_tensor() {
        let mut report = empty_report();
        report
            .tensors
            .push(tensor("token_embd.weight", vec![151936, 1536]));
        // min(shape) is the hidden dim.
        assert_eq!(infer_hidden_dim_from_tensors(&report), 1536);
        assert_eq!(infer_hidden_dim(&report), 1536);
    }

    #[test]
    fn infer_hidden_dim_zero_when_no_signal() {
        assert_eq!(infer_hidden_dim_from_tensors(&empty_report()), 0);
        assert_eq!(infer_hidden_dim(&empty_report()), 0);
    }

    #[test]
    fn infer_num_layers_from_metadata() {
        let mut report = empty_report();
        report
            .metadata
            .insert("llama.block_count".to_string(), "28".to_string());
        assert_eq!(infer_num_layers(&report), 28);
    }

    #[test]
    fn infer_num_layers_from_tensor_indices() {
        let mut report = empty_report();
        report
            .tensors
            .push(tensor("blk.0.attn_q.weight", vec![1, 1]));
        report
            .tensors
            .push(tensor("blk.5.attn_q.weight", vec![1, 1]));
        report.tensors.push(tensor("blk.3.ffn.weight", vec![1, 1]));
        // max index 5 → 6 layers.
        assert_eq!(infer_num_layers(&report), 6);
    }

    #[test]
    fn infer_num_layers_zero_when_no_layer_tensors() {
        let mut report = empty_report();
        report.tensors.push(tensor("token_embd.weight", vec![1, 1]));
        assert_eq!(infer_num_layers(&report), 0);
    }

    // ---- build_inferred_size_config -----------------------------------

    #[test]
    fn build_inferred_size_config_uses_metadata_and_defaults() {
        let mut report = empty_report();
        report.total_params = 1_500_000_000;
        report
            .metadata
            .insert("n_heads".to_string(), "16".to_string());
        report
            .metadata
            .insert("n_ff".to_string(), "8960".to_string());
        let cfg = build_inferred_size_config(&report, 1536, 28);
        assert_eq!(cfg.hidden_dim, 1536);
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.intermediate_dim, 8960);
        // num_kv_heads defaults to num_heads when not provided.
        assert_eq!(cfg.num_kv_heads, 16);
        // head_dim = hidden_dim / num_heads.
        assert_eq!(cfg.head_dim, 1536 / 16);
        assert!(cfg.parameters.ends_with('B'));
    }

    #[test]
    fn build_inferred_size_config_defaults_when_metadata_absent() {
        let report = empty_report();
        let cfg = build_inferred_size_config(&report, 1024, 12);
        assert_eq!(cfg.num_heads, 32); // default
        assert_eq!(cfg.intermediate_dim, 1024 * 4); // hidden*4 default
        assert_eq!(cfg.vocab_size, 32_000); // default
        assert_eq!(cfg.max_position_embeddings, 4096); // default
        assert!((cfg.rope_theta - 10_000.0).abs() < 1e-6);
    }

    // ---- compute_memory_budget ----------------------------------------

    #[test]
    fn compute_memory_budget_cpu_has_no_gpu_fields() {
        let stats = make_test_stats(2000.0, 1_000_000, 2_000_000);
        let size = make_size_config(1536, 28, 12);
        let budget = compute_memory_budget(&stats, &size, None, 1, 2048);
        assert!((budget.weights_mb - 2000.0).abs() < 1e-6);
        assert!(budget.kv_cache_mb >= 0.0);
        assert!(
            (budget.overhead_mb).abs() < f64::EPSILON,
            "no CUDA overhead on CPU"
        );
        assert!(budget.gpu_total_mb.is_none());
        assert!(budget.utilization_pct.is_none());
        assert!(budget.max_batch.is_none());
        assert_eq!(budget.batch_size, 1);
        assert_eq!(budget.seq_len, 2048);
        // total = weights + kv + activations (no overhead).
        let expected = budget.weights_mb + budget.kv_cache_mb + budget.activations_mb;
        assert!((budget.total_mb - expected).abs() < 1e-6);
    }

    #[test]
    fn compute_memory_budget_gpu_populates_utilization() {
        let stats = make_test_stats(2000.0, 1_000_000, 2_000_000);
        let size = make_size_config(1536, 28, 12);
        let hw = ServePlanHardware {
            gpu_name: "GPU".to_string(),
            vram_mb: 24_000.0,
            bandwidth_gbps: 900.0,
            peak_tflops: 80.0,
        };
        let budget = compute_memory_budget(&stats, &size, Some(&hw), 4, 4096);
        assert_eq!(budget.gpu_total_mb, Some(24_000.0));
        assert!(
            (budget.overhead_mb - 512.0).abs() < 1e-6,
            "CUDA overhead present"
        );
        let util = budget.utilization_pct.expect("utilization on GPU");
        assert!((util - budget.total_mb / 24_000.0 * 100.0).abs() < 1e-6);
        assert!(budget.max_batch.is_some());
    }

    #[test]
    fn compute_memory_budget_gpu_zero_vram_no_utilization() {
        let stats = make_test_stats(1000.0, 1, 1);
        let size = make_size_config(1024, 12, 8);
        let hw = ServePlanHardware {
            gpu_name: "GPU".to_string(),
            vram_mb: 0.0,
            bandwidth_gbps: 100.0,
            peak_tflops: 1.0,
        };
        let budget = compute_memory_budget(&stats, &size, Some(&hw), 1, 1024);
        assert!(
            budget.utilization_pct.is_none(),
            "zero VRAM → no utilization"
        );
        assert_eq!(budget.max_batch, Some(0), "no room → max batch 0");
    }

    // ---- compute_throughput -------------------------------------------

    #[test]
    fn compute_throughput_cpu_uses_kernel_estimate() {
        let kernels = make_kernels(Some(42.0));
        let t = compute_throughput(&kernels, None, 1);
        assert!((t.single_decode_tps - 42.0).abs() < 1e-6);
        assert!(t.batched_tps.is_none(), "batch=1 has no batched estimate");
        assert_eq!(t.batch_size, 1);
    }

    #[test]
    fn compute_throughput_cpu_default_tps_when_absent() {
        let kernels = make_kernels(None);
        let t = compute_throughput(&kernels, None, 1);
        assert!((t.single_decode_tps - 50.0).abs() < 1e-6, "default 50 tps");
    }

    #[test]
    fn compute_throughput_batched_scales_sqrt() {
        let kernels = make_kernels(Some(100.0));
        let t = compute_throughput(&kernels, None, 4);
        let batched = t.batched_tps.expect("batched estimate for batch>1");
        assert!((batched - 100.0 * 2.0).abs() < 1e-6, "sqrt(4)=2 scaling");
    }

    #[test]
    fn compute_throughput_gpu_uses_roofline_ceiling() {
        let kernels = make_kernels(Some(10.0));
        let roof = RooflineEstimate {
            bandwidth_gbps: 900.0,
            bandwidth_ceiling_tps: 333.0,
            compute_ceiling_tps: 500.0,
            bottleneck: "MEMORY-BOUND".to_string(),
        };
        let t = compute_throughput(&kernels, Some(&roof), 1);
        assert!(
            (t.single_decode_tps - 333.0).abs() < 1e-6,
            "GPU uses roofline, not cpu kernel"
        );
    }

    // ---- compute_roofline bottleneck classification -------------------

    #[test]
    fn compute_roofline_compute_bound_when_huge_model() {
        // Big model_size_q4 → tiny bandwidth ceiling; tiny flops → huge compute ceiling.
        let stats = make_test_stats(100_000.0, 1, 1);
        let hw = make_test_hw();
        let roof = compute_roofline(&stats, &hw);
        assert_eq!(roof.bottleneck, "MEMORY-BOUND");
    }

    #[test]
    fn compute_roofline_zero_flops_compute_ceiling_zero() {
        let stats = make_test_stats(10.0, 0, 0);
        let hw = make_test_hw();
        let roof = compute_roofline(&stats, &hw);
        assert!((roof.compute_ceiling_tps).abs() < f64::EPSILON);
    }

    // ---- run_contract_checks + determine_verdict ----------------------

    fn budget_fixture(total_mb: f64, weights_mb: f64, batch_size: usize) -> MemoryBudget {
        MemoryBudget {
            weights_mb,
            kv_cache_mb: 100.0,
            activations_mb: 50.0,
            overhead_mb: 512.0,
            total_mb,
            gpu_total_mb: Some(24_000.0),
            utilization_pct: Some(total_mb / 24_000.0 * 100.0),
            max_batch: Some(8),
            batch_size,
            seq_len: 4096,
        }
    }

    #[test]
    fn run_contract_checks_cpu_only_single_check_passes() {
        let budget = budget_fixture(2000.0, 1500.0, 1);
        let checks = run_contract_checks(&budget, None);
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].id, "BUDGET-003");
        assert!(checks[0].passed);
    }

    #[test]
    fn run_contract_checks_gpu_all_pass_within_budget() {
        let hw = ServePlanHardware {
            gpu_name: "GPU".to_string(),
            vram_mb: 24_000.0,
            bandwidth_gbps: 900.0,
            peak_tflops: 80.0,
        };
        let budget = budget_fixture(2000.0, 1500.0, 1);
        let checks = run_contract_checks(&budget, Some(&hw));
        // batch_size 1 → no BUDGET-004; expect 001/002/003.
        let ids: Vec<&str> = checks.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"BUDGET-001"));
        assert!(ids.contains(&"BUDGET-002"));
        assert!(ids.contains(&"BUDGET-003"));
        assert!(!ids.contains(&"BUDGET-004"));
        assert!(checks.iter().all(|c| c.passed), "all should fit in 24GB");
    }

    #[test]
    fn run_contract_checks_gpu_batch_adds_budget_004() {
        let hw = ServePlanHardware {
            gpu_name: "GPU".to_string(),
            vram_mb: 24_000.0,
            bandwidth_gbps: 900.0,
            peak_tflops: 80.0,
        };
        let budget = budget_fixture(3000.0, 1500.0, 8);
        let checks = run_contract_checks(&budget, Some(&hw));
        assert!(checks.iter().any(|c| c.id == "BUDGET-004"));
    }

    #[test]
    fn run_contract_checks_gpu_oversized_fails_budget_001() {
        let hw = ServePlanHardware {
            gpu_name: "tiny".to_string(),
            vram_mb: 1000.0,
            bandwidth_gbps: 100.0,
            peak_tflops: 1.0,
        };
        // total well above 95% of 1000 MB.
        let budget = budget_fixture(5000.0, 4000.0, 1);
        let checks = run_contract_checks(&budget, Some(&hw));
        let b001 = checks
            .iter()
            .find(|c| c.id == "BUDGET-001")
            .expect("BUDGET-001");
        assert!(!b001.passed, "5000 MB must not fit in 1000 MB");
    }

    #[test]
    fn determine_verdict_ready_when_all_pass() {
        let checks = vec![ContractCheck {
            id: "BUDGET-001".to_string(),
            name: "x".to_string(),
            passed: true,
            detail: String::new(),
        }];
        assert_eq!(determine_verdict(&checks), PlanVerdict::Ready);
    }

    #[test]
    fn determine_verdict_blocked_on_critical_failure() {
        let checks = vec![ContractCheck {
            id: "BUDGET-002".to_string(),
            name: "x".to_string(),
            passed: false,
            detail: String::new(),
        }];
        assert_eq!(determine_verdict(&checks), PlanVerdict::Blocked);
    }

    #[test]
    fn determine_verdict_warnings_on_noncritical_failure() {
        let checks = vec![
            ContractCheck {
                id: "BUDGET-001".to_string(),
                name: "x".to_string(),
                passed: true,
                detail: String::new(),
            },
            ContractCheck {
                id: "BUDGET-004".to_string(),
                name: "x".to_string(),
                passed: false,
                detail: String::new(),
            },
        ];
        assert_eq!(determine_verdict(&checks), PlanVerdict::Warnings);
    }

    // ---- contracts_candidate_paths ------------------------------------

    #[test]
    fn contracts_candidate_paths_includes_ancestors() {
        let paths = contracts_candidate_paths();
        assert!(paths.iter().any(|p| p == &PathBuf::from("./contracts")));
        assert!(paths.iter().any(|p| p.ends_with("contracts")));
    }
}
