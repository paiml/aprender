//! Trace command implementation
//!
//! Layer-by-layer analysis of APR models.
//! Toyota Way: Visualization - Make hidden problems visible.
//!
//! This command traces through model layers, computing statistics at each stage
//! to help identify where numerical issues or divergences occur.

use crate::error::CliError;
use crate::output;
use aprender::format::HEADER_SIZE;
use colored::Colorize;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Layer trace information
#[derive(Serialize, Clone)]
pub(crate) struct LayerTrace {
    /// Layer name/type
    pub name: String,
    /// Layer index (if applicable)
    pub index: Option<usize>,
    /// Input statistics
    pub input_stats: Option<TensorStats>,
    /// Output statistics
    pub output_stats: Option<TensorStats>,
    /// Weight statistics (if layer has weights)
    pub weight_stats: Option<TensorStats>,
    /// Anomalies detected
    pub anomalies: Vec<String>,
}

/// Tensor statistics for tracing
#[derive(Serialize, Clone)]
#[allow(dead_code)]
pub(crate) struct TensorStats {
    /// Number of elements
    pub count: usize,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// L2 norm
    pub l2_norm: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Maximum absolute value
    pub max_abs: f32,
    /// Count of NaN values
    pub nan_count: usize,
    /// Count of Inf values
    pub inf_count: usize,
}

impl TensorStats {
    /// Compute statistics from a slice of f32 values
    #[allow(dead_code, clippy::cast_lossless)]
    pub(crate) fn from_slice(data: &[f32]) -> Self {
        let count = data.len();
        if count == 0 {
            return Self {
                count: 0,
                mean: 0.0,
                std: 0.0,
                l2_norm: 0.0,
                min: 0.0,
                max: 0.0,
                max_abs: 0.0,
                nan_count: 0,
                inf_count: 0,
            };
        }

        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut max_abs = 0.0_f32;
        let mut nan_count = 0;
        let mut inf_count = 0;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
                continue;
            }
            if v.is_infinite() {
                inf_count += 1;
                continue;
            }
            sum += v as f64;
            sum_sq += (v as f64) * (v as f64);
            min = min.min(v);
            max = max.max(v);
            max_abs = max_abs.max(v.abs());
        }

        let valid_count = count - nan_count - inf_count;
        let mean = if valid_count > 0 {
            (sum / valid_count as f64) as f32
        } else {
            0.0
        };
        let variance = if valid_count > 1 {
            ((sum_sq / valid_count as f64) - (mean as f64).powi(2)).max(0.0)
        } else {
            0.0
        };
        let std = (variance as f32).sqrt();
        let l2_norm = (sum_sq as f32).sqrt();

        Self {
            count,
            mean,
            std,
            l2_norm,
            min: if min.is_finite() { min } else { 0.0 },
            max: if max.is_finite() { max } else { 0.0 },
            max_abs,
            nan_count,
            inf_count,
        }
    }

    /// Check for anomalies
    #[allow(dead_code)]
    pub(crate) fn detect_anomalies(&self, name: &str) -> Vec<String> {
        let mut anomalies = Vec::new();

        if self.nan_count > 0 {
            anomalies.push(format!(
                "{name}: {}/{} NaN values",
                self.nan_count, self.count
            ));
        }
        if self.inf_count > 0 {
            anomalies.push(format!(
                "{name}: {}/{} Inf values",
                self.inf_count, self.count
            ));
        }
        if self.std < 1e-8 && self.count > 1 {
            anomalies.push(format!("{name}: near-zero variance (std={:.2e})", self.std));
        }
        if self.max_abs > 100.0 {
            anomalies.push(format!(
                "{name}: large values (max_abs={:.2})",
                self.max_abs
            ));
        }
        if self.mean.abs() > 10.0 {
            anomalies.push(format!("{name}: large mean bias ({:.4})", self.mean));
        }

        anomalies
    }
}

/// Trace result for JSON output
#[derive(Serialize)]
struct TraceResult {
    file: String,
    format: String,
    layers: Vec<LayerTrace>,
    summary: TraceSummary,
}

/// Summary of trace analysis
#[derive(Serialize)]
struct TraceSummary {
    total_layers: usize,
    total_parameters: usize,
    anomaly_count: usize,
    anomalies: Vec<String>,
}

/// Handle special trace modes (interactive, payload, diff).
/// Returns `Some(Ok(()))` if a mode handled the request, `None` to continue.
fn handle_special_modes(
    path: &Path,
    reference: Option<&Path>,
    payload: bool,
    diff: bool,
    interactive: bool,
) -> Option<Result<(), CliError>> {
    handle_special_modes_with_json(path, reference, payload, diff, interactive, false)
}

/// JSON-aware variant of `handle_special_modes`.
///
/// When `json && payload`, traced inference output is emitted as a single
/// JSON object instead of the human-readable text format. This is the exit
/// criterion shape for M34 FAST PATH Step 2 (companion
/// claude-code-parity-apr docs/specifications/claude-code-parity-apr-poc.md
/// § "M32d FAST PATH"):
///
///     "apr trace --json --payload <gguf> --prompt 'What is 2+2?' returns
///      non-null output_stats for every transformer_block_N entry, with
///      finite L2 norms."
fn handle_special_modes_with_json(
    path: &Path,
    reference: Option<&Path>,
    payload: bool,
    diff: bool,
    interactive: bool,
    json: bool,
) -> Option<Result<(), CliError>> {
    if interactive {
        println!("Starting interactive trace (TUI) for {}", path.display());
        println!("(TUI mode not yet fully implemented)");
        return Some(Ok(()));
    }

    if payload {
        if json {
            // M32d Step 2 exit-criterion shape: machine-readable JSON output
            // for `apr trace --json --payload`. Text-mode fallback if the
            // JSON path doesn't apply (e.g. SafeTensors).
            return Some(run_traced_inference_json(path));
        }
        return Some(run_traced_inference(path));
    }

    match DiffMode::resolve(diff, reference) {
        Err(refusal) => return Some(Err(refusal)),
        Ok(Some(DiffMode { reference })) => {
            println!(
                "Diffing trace between {} and {}",
                path.display(),
                reference.display()
            );
        }
        Ok(None) => {}
    }

    None
}

/// `--diff`, resolved: a diff always has something to diff against.
///
/// dogfood-0.63.0, issue #2394 finding 6: `apr trace model.gguf --diff`
/// printed "Diff mode requires --reference", then ignored its own requirement,
/// ran an ordinary single-model trace and exited 0 — so a scripted diff
/// produced a plain trace that looked like a successful comparison. The
/// invalid combination is now unrepresentable: a `DiffMode` cannot be built
/// without a reference path, and `resolve` is the only way to build one.
struct DiffMode<'a> {
    reference: &'a Path,
}

impl<'a> DiffMode<'a> {
    /// `Ok(None)` when `--diff` was not requested; `Err` when it was requested
    /// without the reference it needs.
    fn resolve(diff: bool, reference: Option<&'a Path>) -> Result<Option<Self>, CliError> {
        if !diff {
            return Ok(None);
        }
        match reference {
            Some(reference) => Ok(Some(Self { reference })),
            None => Err(CliError::ValidationFailed(
                "--diff requires --reference <MODEL>: there is nothing to diff against".to_string(),
            )),
        }
    }
}

/// Resolve a model path: download from HuggingFace if `hf://` URI, else return unchanged.
fn resolve_model_path(path: &Path) -> Result<std::path::PathBuf, CliError> {
    use super::run::{download_hf_model, ModelSource};

    let path_str = path.to_string_lossy();
    if !path_str.starts_with("hf://") {
        println!("Model: {}", path.display());
        println!();
        return Ok(path.to_path_buf());
    }

    let source = ModelSource::parse(&path_str)?;
    match source {
        ModelSource::HuggingFace { org, repo, file } => {
            println!(
                "Model: hf://{}/{}{}",
                org,
                repo,
                file.as_ref().map(|f| format!("/{}", f)).unwrap_or_default()
            );
            println!();
            eprintln!("{}", "Downloading from HuggingFace...".yellow());
            download_hf_model(&org, &repo, file.as_deref())
        }
        _ => Ok(path.to_path_buf()),
    }
}

/// PMAT-235: Pre-flight contract validation before traced inference.
fn preflight_contract_check(local_path: &Path) {
    use aprender::format::rosetta::RosettaStone;

    let rosetta = RosettaStone::new();
    match rosetta.validate(local_path) {
        Ok(report) => {
            let contract_failures: Vec<String> = report
                .tensors
                .iter()
                .flat_map(|t| t.failures.iter().map(move |f| format!("{}: {}", t.name, f)))
                .collect();
            if contract_failures.is_empty() {
                println!(
                    "{}",
                    format!(
                        "Contract: {} tensors pass PMAT-235 gates",
                        report.tensor_count
                    )
                    .green()
                );
            } else {
                println!(
                    "{}",
                    format!(
                        "Contract: {} violations in {} tensors",
                        contract_failures.len(),
                        report.failed_tensor_count
                    )
                    .red()
                    .bold()
                );
                for failure in contract_failures.iter().take(5) {
                    println!("  {}", failure.red());
                }
                if contract_failures.len() > 5 {
                    println!("  ... and {} more", contract_failures.len() - 5);
                }
                println!();
                println!(
                    "{}",
                    "WARNING: Contract violations may cause garbage output."
                        .yellow()
                        .bold()
                );
            }
        }
        Err(e) => {
            println!("{}", format!("Contract: validation skipped ({e})").yellow());
        }
    }
    println!();
}

/// Dispatch traced inference to the format-specific implementation based on file extension.
fn dispatch_by_format(local_path: &Path) -> Result<(), CliError> {
    let ext = local_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext.to_lowercase().as_str() {
        "gguf" => run_traced_inference_gguf(local_path),
        "apr" => run_traced_inference_apr(local_path),
        "safetensors" => run_traced_inference_safetensors(local_path),
        _ => Err(CliError::InvalidFormat(format!(
            "Unknown format: {}. Supported: .gguf, .apr, .safetensors",
            ext
        ))),
    }
}

/// Run traced inference through the model to debug layer-by-layer outputs.
/// This is the core functionality for debugging garbage output (BUG-GGUF-001).
fn run_traced_inference(path: &Path) -> Result<(), CliError> {
    output::section("Traced Inference (APR-TRACE-001)");
    let local_path = resolve_model_path(path)?;
    preflight_contract_check(&local_path);
    dispatch_by_format(&local_path)
}

/// Traced inference for GGUF models (primary path for BUG-GGUF-001 debugging)
#[cfg(feature = "inference")]
fn run_traced_inference_gguf(path: &Path) -> Result<(), CliError> {
    use colored::Colorize;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

    println!("{}", "Format: GGUF (quantized)".cyan());
    println!();

    // Load GGUF via mmap
    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF: {e}")))?;

    // Create quantized model
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to create quantized model: {e}")))?;

    let config = model.config();
    println!("Architecture: {}", config.architecture);
    println!("  Layers: {}", config.num_layers);
    println!("  Hidden dim: {}", config.hidden_dim);
    println!("  Vocab size: {}", config.vocab_size);
    println!(
        "  Heads: {} (KV: {})",
        config.num_heads, config.num_kv_heads
    );
    println!();

    // Encode test prompt using GGUF's embedded tokenizer
    let test_prompt = "What is 2+2?";
    let test_tokens = mapped
        .model
        .encode(test_prompt)
        .unwrap_or_else(|| vec![1u32]);

    println!("{}", format!("Test prompt: {:?}", test_prompt).cyan());
    println!("{}", format!("Encoded tokens: {:?}", test_tokens).cyan());
    println!();

    // SHIP-007 §26.4 P3: forward_traced emits per-layer LayerActivation
    // stats for APR-vs-GGUF bisection (the §23 layer-3 ffn_swigl 17×
    // anomaly comparison). Run BEFORE generation so the trace reflects
    // a pristine forward pass on the test prompt.
    //
    // M32d Step 2 (claude-code-parity-apr-poc.md § "M32d FAST PATH"):
    // qwen3_moe-arch GGUF dispatches to forward_qwen3_moe_traced because
    // the dense path's forward_traced doesn't exercise the MoE FFN
    // dispatch — it would silently skip the MoE-specific computation.
    println!("{}", "FORWARD PASS (with layer tracing):".green().bold());
    let canonical_arch = realizar::tensor_names::normalize_architecture(&config.architecture);
    // Accept both canonical "qwen3_moe" and raw GGUF-reported "qwen3moe"
    // (no underscore) — the build.rs codegen sometimes lags on the YAML
    // alias mapping `qwen3moe → qwen3_moe`. Robust string comparison is
    // cheaper than relying on the generator cache being current.
    let raw_arch_lower = config.architecture.to_lowercase();
    let is_qwen3_moe = canonical_arch == "qwen3_moe"
        || raw_arch_lower == "qwen3moe"
        || raw_arch_lower == "qwen3_moe";
    let trace_result = if is_qwen3_moe {
        run_qwen3_moe_traced_forward(&mapped, &model, &test_tokens)
    } else {
        model.forward_traced(&test_tokens)
    };
    match trace_result {
        Ok(trace) => {
            println!();
            println!("{}", "EMBEDDING:".cyan().bold());
            print_activation_stats_colored("  ", &trace.embed_stats);

            print_layer_activations(&trace.layer_activations);

            println!();
            println!("{}", "FINAL LAYER NORM:".cyan().bold());
            print_activation_stats("  ", &trace.final_norm_stats);

            print_logit_predictions(&trace.logits);
            print_trace_summary(&trace.layer_activations, &trace.logits);
        }
        Err(e) => {
            eprintln!(
                "{}",
                format!("forward_traced unavailable for this GGUF: {e}").yellow()
            );
            println!("  Layer-by-layer tracing not available (e.g., encoder-decoder model).");
        }
    }
    println!();

    // Run generation with small max_tokens to see what comes out.
    //
    // M32d Step 2 (claude-code-parity-apr-poc.md § "M32d FAST PATH"):
    // qwen3_moe-arch GGUF cannot use generate_with_cache because that
    // method calls the dense FFN path on the placeholder zero weights
    // (per M32c.2.2 LAZY-FUSED-MATVEC strategy — dense FFN fields are
    // empty stubs for MoE models). Skip generation for qwen3_moe; the
    // traced forward output above is the load-bearing diagnostic for
    // FAST PATH Step 3 (per-layer cosine bisection vs HF FP16).
    if is_qwen3_moe {
        println!(
            "{}",
            "GENERATION: skipped for qwen3_moe (use `apr run` for text generation)".yellow()
        );
        println!();
        return Ok(());
    }

    println!("{}", "GENERATION (max 8 tokens):".green().bold());
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 8,
        temperature: 0.0, // Greedy for reproducibility
        top_k: 1,
        ..Default::default()
    };

    let output_tokens = model
        .generate_with_cache(&test_tokens, &gen_config)
        .map_err(|e| CliError::InferenceFailed(format!("Generation failed: {e}")))?;

    let generated = &output_tokens[test_tokens.len()..];
    println!("  Generated token IDs: {:?}", generated);

    // Decode each token individually to see where garbage starts
    println!();
    println!("{}", "TOKEN-BY-TOKEN DECODE:".green().bold());
    for (i, &token_id) in generated.iter().enumerate() {
        let decoded = mapped.model.decode(&[token_id]);
        let is_garbage = is_likely_garbage(&decoded);
        if is_garbage {
            println!(
                "  {}. token_id={} → {:?} {}",
                i + 1,
                token_id,
                decoded,
                "⚠ GARBAGE".red().bold()
            );
        } else {
            println!("  {}. token_id={} → {:?}", i + 1, token_id, decoded);
        }
    }

    // Full decoded output
    let full_decoded = mapped.model.decode(generated);
    println!();
    println!("{}", "FULL OUTPUT:".green().bold());
    println!("  {:?}", full_decoded);

    // Garbage detection
    println!();
    if is_likely_garbage(&full_decoded) {
        println!("{}", "⚠ GARBAGE OUTPUT DETECTED!".red().bold());
        println!();
        println!("Likely causes:");
        println!("  1. LAYOUT-001: Column-major vs row-major kernel mismatch");
        println!("  2. Weight tensor corruption during loading");
        println!("  3. Tokenizer vocabulary mismatch");
        println!();
        println!("Debug steps:");
        println!("  1. Check if SafeTensors produces correct output (same model)");
        println!("  2. Compare token IDs between GGUF and SafeTensors");
        println!("  3. Verify quantization type is supported");
    } else {
        println!("{}", "✓ Output appears reasonable".green());
    }

    Ok(())
}

/// JSON-output variant of `run_traced_inference` — emits a single JSON
/// object with `embedding`, `layers[]`, `final_norm`, `logits` shaped
/// for `apr trace --json --payload` consumers.
///
/// Companion spec exit-criterion shape: M34 FAST PATH Step 2 (claude-
/// code-parity-apr docs/specifications/claude-code-parity-apr-poc.md §
/// "M32d FAST PATH"). Schema:
///
/// ```jsonc
/// {
///   "format": "GGUF (qwen3moe)",
///   "architecture": "qwen3moe",
///   "num_layers": 48, "hidden_dim": 2048, "vocab_size": 151936,
///   "prompt": "What is 2+2?",
///   "encoded_tokens": [...],
///   "embedding": { "min": ..., "max": ..., "mean": ..., "std_dev": ..., "count": 2048 },
///   "layers": [
///     { "layer_idx": 0,
///       "attn_norm": {...}, "qkv": {...}, "attn_out": {...},
///       "ffn_norm": {...}, "ffn_out": {...}, "output": {...} },
///     ...
///   ],
///   "final_norm": {...},
///   "logits": { "vocab_size": 151936, "l2_norm": ..., "top_k": [{"token_id": ..., "logit": ...}, ...] }
/// }
/// ```
#[cfg(feature = "inference")]
fn run_traced_inference_json(path: &Path) -> Result<(), CliError> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
    use serde_json::json;

    // JSON mode: skip the human-readable "Model: ..." / "Contract: ..."
    // preamble that resolve_model_path + preflight_contract_check print —
    // those break `apr trace --json --payload | jq` consumers.
    let local_path: std::path::PathBuf = if path.to_string_lossy().starts_with("hf://") {
        resolve_model_path(path)?
    } else {
        path.to_path_buf()
    };

    let mapped = MappedGGUFModel::from_path(&local_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF: {e}")))?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to create quantized model: {e}")))?;

    let config = model.config();
    let test_prompt = "What is 2+2?";
    let test_tokens = mapped
        .model
        .encode(test_prompt)
        .unwrap_or_else(|| vec![1u32]);

    let canonical_arch = realizar::tensor_names::normalize_architecture(&config.architecture);
    let raw_arch_lower = config.architecture.to_lowercase();
    let is_qwen3_moe = canonical_arch == "qwen3_moe"
        || raw_arch_lower == "qwen3moe"
        || raw_arch_lower == "qwen3_moe";
    let trace = if is_qwen3_moe {
        run_qwen3_moe_traced_forward(&mapped, &model, &test_tokens)
    } else {
        model.forward_traced(&test_tokens)
    }
    .map_err(|e| CliError::InferenceFailed(format!("forward_traced: {e}")))?;

    let stats_to_json = |s: &realizar::apr_transformer::ActivationStats| {
        json!({
            "min": s.min, "max": s.max, "mean": s.mean, "std_dev": s.std_dev,
            "nan_count": s.nan_count, "inf_count": s.inf_count,
            "zero_count": s.zero_count, "count": s.count,
        })
    };

    let layers_json: Vec<_> = trace
        .layer_activations
        .iter()
        .map(|la| {
            json!({
                "layer_idx": la.layer_idx,
                "attn_norm": stats_to_json(&la.attn_norm_stats),
                "qkv": stats_to_json(&la.qkv_stats),
                "attn_out": stats_to_json(&la.attn_out_stats),
                "ffn_norm": stats_to_json(&la.ffn_norm_stats),
                "ffn_out": stats_to_json(&la.ffn_out_stats),
                "output": stats_to_json(&la.output_stats),
            })
        })
        .collect();

    let logits_l2: f32 = trace.logits.iter().map(|v| v * v).sum::<f32>().sqrt();
    let mut indexed: Vec<(usize, f32)> = trace.logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k: Vec<_> = indexed
        .iter()
        .take(5)
        .map(|(i, v)| json!({ "token_id": *i, "logit": *v }))
        .collect();

    let arch_label = if is_qwen3_moe {
        format!("GGUF ({})", config.architecture)
    } else {
        "GGUF (quantized)".to_string()
    };
    let out = json!({
        "format": arch_label,
        "architecture": config.architecture,
        "num_layers": config.num_layers,
        "hidden_dim": config.hidden_dim,
        "vocab_size": config.vocab_size,
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "prompt": test_prompt,
        "encoded_tokens": test_tokens,
        "embedding": stats_to_json(&trace.embed_stats),
        "layers": layers_json,
        "final_norm": stats_to_json(&trace.final_norm_stats),
        "logits_stats": stats_to_json(&trace.logits_stats),
        "logits": {
            "vocab_size": trace.logits.len(),
            "l2_norm": logits_l2,
            "top_k": top_k,
        },
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&out)
            .map_err(|e| { CliError::InferenceFailed(format!("JSON serialization: {e}")) })?
    );
    Ok(())
}

#[cfg(not(feature = "inference"))]
fn run_traced_inference_json(_path: &Path) -> Result<(), CliError> {
    Err(CliError::FeatureDisabled(
        "Traced inference requires the 'inference' feature.".to_string(),
    ))
}

/// M32d Step 2 — qwen3_moe-arch traced forward dispatch helper.
///
/// Reads MoE config (num_experts, num_experts_per_tok, moe_intermediate)
/// from GGUF metadata, loads per-layer Qwen3MoeQuantizedLayer descriptors,
/// then calls forward_qwen3_moe_traced. Returns ForwardTrace with one
/// LayerActivation per decoder layer; sub-FFN slots are zero (no globally
/// meaningful SwiGLU breakdown in MoE).
///
/// Companion spec: paiml/claude-code-parity-apr docs/specifications/
/// claude-code-parity-apr-poc.md § "M32d FAST PATH" Step 2.
#[cfg(feature = "inference")]
fn run_qwen3_moe_traced_forward(
    mapped: &realizar::gguf::MappedGGUFModel,
    model: &realizar::gguf::OwnedQuantizedModel,
    test_tokens: &[u32],
) -> realizar::error::Result<realizar::apr_transformer::ForwardTrace> {
    let num_experts = mapped.model.expert_count().ok_or_else(|| {
        realizar::error::RealizarError::InvalidShape {
            reason: "qwen3_moe trace: missing 'expert_count' in GGUF metadata".to_string(),
        }
    })?;
    let num_experts_per_tok = mapped.model.expert_used_count().ok_or_else(|| {
        realizar::error::RealizarError::InvalidShape {
            reason: "qwen3_moe trace: missing 'expert_used_count' in GGUF metadata".to_string(),
        }
    })?;
    let moe_intermediate = mapped.model.expert_feed_forward_length().ok_or_else(|| {
        realizar::error::RealizarError::InvalidShape {
            reason: "qwen3_moe trace: missing 'expert_feed_forward_length' in GGUF metadata"
                .to_string(),
        }
    })?;

    let data = mapped.data();
    let num_layers = model.config().num_layers;
    let mut moe_layers = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        moe_layers.push(realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer(
            &mapped.model,
            data,
            layer_idx,
        )?);
    }

    model.forward_qwen3_moe_traced(
        test_tokens,
        &moe_layers,
        num_experts,
        num_experts_per_tok,
        moe_intermediate,
        data,
    )
}

/// Stub for GGUF inference when inference feature is disabled
#[cfg(not(feature = "inference"))]
fn run_traced_inference_gguf(_path: &Path) -> Result<(), CliError> {
    Err(CliError::FeatureDisabled(
        "Traced inference for GGUF models requires the 'inference' feature. Build with --features inference".to_string(),
    ))
}

include!("vector_stats.rs");
include!("trace_likely_has_repeated.rs");
include!("layer.rs");
include!("trace_05.rs");
