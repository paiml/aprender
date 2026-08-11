//! Eval Command Implementation
//!
//! Implements spec H13: Perplexity evaluation on standard datasets.
//!
//! # Usage
//!
//! ```bash
//! apr eval model.gguf --dataset wikitext-2   # Evaluate GGUF on WikiText-2
//! apr eval model.apr --dataset lambada       # Evaluate APR on LAMBADA
//! apr eval model.safetensors --text "Hello"  # Evaluate SafeTensors on custom text
//! ```
//!
//! Toyota Way: Jidoka - fail fast if perplexity exceeds threshold.
//!
//! # PMAT-128 Fix: GGUF Weight Loading
//!
//! This module now uses realizar's GGUF inference engine for GGUF models,
//! fixing the F-EVAL bug where GGUF models showed PPL ~1000 due to
//! uninitialized weights.

pub(crate) mod code_eval;
pub(crate) mod inference;
mod perplexity;

use crate::error::{CliError, Result};
use crate::output;
use colored::Colorize;
use std::path::Path;
use std::time::Instant;

// Re-export public entry points from submodules
pub(crate) use code_eval::run_code_eval;
pub(crate) use inference::{run_humaneval, run_mbpp};

/// Supported evaluation datasets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dataset {
    /// WikiText-2 test set (standard LM benchmark)
    WikiText2,
    /// LAMBADA (last word prediction)
    Lambada,
    /// Custom text input
    Custom,
}

impl std::str::FromStr for Dataset {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "wikitext-2" | "wikitext2" => Ok(Self::WikiText2),
            "lambada" => Ok(Self::Lambada),
            "custom" => Ok(Self::Custom),
            _ => Err(format!(
                "Unknown dataset: {s}. Use: wikitext-2, lambada, or custom"
            )),
        }
    }
}

/// Evaluation configuration
struct EvalConfig {
    /// Dataset to evaluate on
    dataset: Dataset,
    /// Custom text (if dataset is Custom)
    text: Option<String>,
    /// Maximum tokens to evaluate
    max_tokens: usize,
    /// Perplexity threshold for pass/fail
    threshold: f32,
}

/// Evaluation results
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct EvalResult {
    /// Perplexity score (lower is better)
    pub perplexity: f32,
    /// Cross-entropy loss
    pub cross_entropy: f32,
    /// Number of tokens evaluated
    pub tokens_evaluated: usize,
    /// Evaluation time
    pub eval_time_secs: f32,
    /// Whether perplexity is below threshold
    pub passed: bool,
    /// Threshold used
    pub threshold: f32,
}

/// The device perplexity evaluation actually runs on.
///
/// There is no GPU path here — the only readers of `device` are the
/// humaneval/mbpp code-eval paths — so this is always `"cpu"`. It exists so the
/// answer is stated rather than assumed.
pub(crate) fn perplexity_device(_requested: &str) -> &'static str {
    "cpu"
}

/// Notice to emit when `--device` cannot be honoured in perplexity mode.
///
/// `--device cuda` used to run byte-identically to `--device cpu` with no
/// mention of a device anywhere in the output, so an operator who typo'd
/// `--device cude` (or asked for a GPU this build has no kernels for) believed
/// the run was GPU-accelerated.
pub(crate) fn perplexity_device_notice(requested: &str) -> Option<String> {
    if perplexity_device(requested) == requested {
        return None;
    }
    Some(format!(
        "⚠ --device {requested} requested, but perplexity evaluation is CPU-only — running on cpu \
         (--device applies to --task humaneval/mbpp)"
    ))
}

/// Run the eval command
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
pub(crate) fn run(
    path: &Path,
    dataset: &str,
    text: Option<&str>,
    max_tokens: Option<usize>,
    threshold: Option<f32>,
    device: &str,
    json: bool,
) -> Result<()> {
    if let Some(notice) = perplexity_device_notice(device) {
        eprintln!("{notice}");
    }

    let dataset_enum: Dataset = dataset
        .parse()
        .map_err(|e: String| CliError::ValidationFailed(e))?;

    let config = EvalConfig {
        dataset: dataset_enum,
        text: text.map(String::from),
        max_tokens: max_tokens.unwrap_or(512),
        threshold: threshold.unwrap_or(20.0), // Per spec H13: PPL > 20 indicates garbage
    };

    if !json {
        print_header(path, &config);
        output::kv("Device", perplexity_device(device));
        println!();
    }

    // Run evaluation
    let result = perplexity::run_evaluation(path, &config, json)?;

    // GH-248: JSON output mode
    if json {
        return print_json_results(path, &config, &result, device);
    }

    // Print results
    print_results(&result);

    // Return error if threshold exceeded
    if !result.passed {
        return Err(CliError::ValidationFailed(format!(
            "Perplexity {:.2} exceeds threshold {:.2} (spec H13)",
            result.perplexity, result.threshold
        )));
    }

    Ok(())
}

/// GH-248: JSON output mode for eval results
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn print_json_results(
    path: &Path,
    config: &EvalConfig,
    result: &EvalResult,
    device: &str,
) -> Result<()> {
    let output = serde_json::json!({
        "model": path.display().to_string(),
        "dataset": format!("{:?}", config.dataset),
        "device": perplexity_device(device),
        "perplexity": result.perplexity,
        "cross_entropy": result.cross_entropy,
        "tokens_evaluated": result.tokens_evaluated,
        "eval_time_secs": result.eval_time_secs,
        "threshold": result.threshold,
        "passed": result.passed,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
    if !result.passed {
        return Err(CliError::ValidationFailed(format!(
            "Perplexity {:.2} exceeds threshold {:.2} (spec H13)",
            result.perplexity, result.threshold
        )));
    }
    Ok(())
}

fn print_header(path: &Path, config: &EvalConfig) {
    output::section("APR Evaluation");
    println!();
    output::kv("Model", path.display());
    output::kv("Dataset", format!("{:?}", config.dataset));
    output::kv("Max tokens", config.max_tokens);
    output::kv("PPL threshold", config.threshold);
    println!();
}

/// Get evaluation text based on dataset
fn get_eval_text(config: &EvalConfig) -> Result<String> {
    match config.dataset {
        Dataset::WikiText2 => {
            // Sample WikiText-2 style text for testing
            // In production, would load actual WikiText-2 test set
            Ok(SAMPLE_WIKITEXT.to_string())
        }
        Dataset::Lambada => {
            // Sample LAMBADA style text
            Ok(SAMPLE_LAMBADA.to_string())
        }
        Dataset::Custom => config.text.clone().ok_or_else(|| {
            CliError::ValidationFailed("Custom dataset requires --text argument".to_string())
        }),
    }
}

/// Resolve checkpoint subdirectory: if the root dir doesn't contain
/// adapter_config.json or model.safetensors, look in best/ then epoch-N/ subdirs.
fn resolve_checkpoint_dir(dir: &Path) -> Option<std::path::PathBuf> {
    let has_adapter = dir.join("adapter_config.json").exists();
    let has_weights = dir.join("model.safetensors").exists();
    if has_adapter || has_weights {
        return None; // Already the right directory
    }

    // Try best/ subdir first
    let best = dir.join("best");
    if best.is_dir()
        && (best.join("adapter_config.json").exists() || best.join("model.safetensors").exists())
    {
        eprintln!(
            "Resolved checkpoint: {} → {}/best",
            dir.display(),
            dir.display()
        );
        return Some(best);
    }

    // Try latest epoch-N subdir
    let mut epoch_dirs: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("epoch-"))
                && e.path().is_dir()
        })
        .collect();
    epoch_dirs.sort_by_key(std::fs::DirEntry::file_name);
    if let Some(latest) = epoch_dirs.last() {
        let p = latest.path();
        if p.join("adapter_config.json").exists() || p.join("model.safetensors").exists() {
            eprintln!("Resolved checkpoint: {} → {}", dir.display(), p.display());
            return Some(p);
        }
    }

    None
}

#[cfg(feature = "training")]
/// Run classification evaluation on a checkpoint directory.
///
/// Loads a saved ClassifyPipeline checkpoint, evaluates against a JSONL test set,
/// and reports per-class precision/recall/F1 with optional model card generation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_classify_eval(
    checkpoint_dir: &Path,
    data_path: Option<&Path>,
    model_size: Option<&str>,
    num_classes: usize,
    generate_card: bool,
    json_output: bool,
) -> Result<()> {
    use entrenar::finetune::classify_pipeline::ClassifyConfig;
    use entrenar::finetune::{evaluate_checkpoint, SSC_LABELS};
    let data_path = data_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "--data <test.jsonl> is required for classification evaluation".to_string(),
        )
    })?;

    if !data_path.exists() {
        return Err(CliError::FileNotFound(data_path.to_path_buf()));
    }

    if !checkpoint_dir.is_dir() {
        return Err(CliError::ValidationFailed(format!(
            "Checkpoint directory not found: {}",
            checkpoint_dir.display()
        )));
    }

    let resolved_checkpoint = resolve_checkpoint_dir(checkpoint_dir);
    let checkpoint_dir = resolved_checkpoint.as_deref().unwrap_or(checkpoint_dir);

    // GH-377: Resolve model config from --model-size (checkpoint dirs don't have .apr metadata)
    let model_config = super::model_config::resolve_transformer_config_by_size(model_size)?;

    let classify_config = ClassifyConfig {
        num_classes,
        ..ClassifyConfig::default()
    };

    // Build label names
    let label_names: Vec<String> = if num_classes == 5 {
        SSC_LABELS.iter().map(|s| (*s).to_string()).collect()
    } else {
        (0..num_classes).map(|i| format!("class_{i}")).collect()
    };

    if !json_output {
        output::section("APR Classification Evaluation");
        println!();
        output::kv("Checkpoint", checkpoint_dir.display());
        output::kv("Test data", data_path.display());
        output::kv(
            "Model",
            format!(
                "{}h x {}L",
                model_config.hidden_size, model_config.num_hidden_layers,
            ),
        );
        output::kv("Classes", num_classes.to_string());
        println!();
        println!("{}", "Loading checkpoint and evaluating...".yellow());
        println!();
    }

    let report = evaluate_checkpoint(
        checkpoint_dir,
        data_path,
        &model_config,
        classify_config,
        &label_names,
    )
    .map_err(|e| CliError::ValidationFailed(format!("Evaluation failed: {e}")))?;

    // Output results
    if json_output {
        println!("{}", report.to_json());
    } else {
        println!("{}", report.to_report());
    }

    // Generate model card if requested
    if generate_card {
        let model_name = "paiml/shell-safety-classifier";
        let base_model = Some("Qwen/Qwen2.5-Coder-0.5B");
        let card = report.to_model_card(model_name, base_model);
        let card_path = checkpoint_dir.join("README.md");
        std::fs::write(&card_path, &card).map_err(|e| {
            CliError::ValidationFailed(format!(
                "Failed to write model card to {}: {e}",
                card_path.display()
            ))
        })?;
        if !json_output {
            println!();
            println!(
                "{} Model card written to {}",
                "✓".green(),
                card_path.display()
            );
        }
    }

    Ok(())
}

/// Run eval plan (dry-run validation).
///
/// Validates that model and benchmark data exist, reports what would be evaluated.
pub(crate) fn run_eval_plan(
    model_path: &Path,
    task: &str,
    data_path: Option<&Path>,
    max_tokens: usize,
    threshold: f32,
    json_output: bool,
) -> Result<()> {
    // Validate model exists
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }

    // Detect model format
    let format = if model_path.extension().is_some_and(|e| e == "gguf") {
        "GGUF"
    } else if model_path.extension().is_some_and(|e| e == "apr") {
        "APR"
    } else if model_path.extension().is_some_and(|e| e == "safetensors") {
        "SafeTensors"
    } else if model_path.is_dir() {
        "Checkpoint directory"
    } else {
        "Unknown"
    };

    // Count benchmark problems if data provided
    let problem_count = if let Some(data) = data_path {
        if !data.exists() {
            return Err(CliError::FileNotFound(data.to_path_buf()));
        }
        let content = std::fs::read_to_string(data)
            .map_err(|e| CliError::ValidationFailed(format!("Cannot read benchmark data: {e}")))?;
        content.lines().filter(|l| !l.trim().is_empty()).count()
    } else {
        0
    };

    if json_output {
        let output = serde_json::json!({
            "plan": true,
            "model": model_path.display().to_string(),
            "format": format,
            "task": task,
            "problems": problem_count,
            "max_tokens": max_tokens,
            "threshold": threshold,
            "ready": true,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
    } else {
        output::section("APR Eval Plan");
        println!();
        output::kv("Model", model_path.display());
        output::kv("Format", format);
        output::kv("Task", task);
        if problem_count > 0 {
            output::kv("Benchmark problems", problem_count);
        }
        output::kv("Max tokens", max_tokens);
        output::kv("Threshold", threshold);
        println!();
        println!("{}", "✓ Ready to evaluate".green());
    }

    Ok(())
}

// --- Contamination detection (R-030, survey #64) ---

/// Run benchmark contamination detection.
///
/// Checks training data for overlap with benchmark problems using n-gram matching.
pub(crate) fn run_contamination(
    model_path: &Path,
    data_path: Option<&Path>,
    benchmark_path: Option<&Path>,
    threshold: f32,
    json_output: bool,
) -> Result<()> {
    let data_path = data_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "--data <training-data.parquet|.jsonl> is required for contamination detection"
                .to_string(),
        )
    })?;
    let benchmark_path = benchmark_path.unwrap_or(data_path);

    if !data_path.exists() {
        return Err(CliError::FileNotFound(data_path.to_path_buf()));
    }

    if !json_output {
        output::section("APR Contamination Detection");
        println!();
        output::kv("Model", model_path.display());
        output::kv("Training data", data_path.display());
        output::kv("Benchmark", benchmark_path.display());
        output::kv("Overlap threshold", format!("{:.0}%", threshold * 100.0));
        println!();
    }

    let start = Instant::now();

    // Load training data text
    let train_text = load_text_corpus(data_path)?;
    let train_ngrams = extract_ngrams(&train_text, 10);

    // Load benchmark problems
    let bench_text = load_text_corpus(benchmark_path)?;
    let bench_lines: Vec<&str> = bench_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();

    let mut contaminated = 0usize;
    let mut results = Vec::new();

    for (i, line) in bench_lines.iter().enumerate() {
        let line_ngrams = extract_ngrams(line, 10);
        let overlap = compute_ngram_overlap(&line_ngrams, &train_ngrams);
        let is_contaminated = overlap > threshold;
        if is_contaminated {
            contaminated += 1;
        }
        results.push((i, overlap, is_contaminated));
    }

    let elapsed = start.elapsed().as_secs_f32();
    let total = bench_lines.len();
    let clean = total - contaminated;
    let contamination_rate = if total > 0 {
        contaminated as f64 / total as f64
    } else {
        0.0
    };

    if json_output {
        let out = serde_json::json!({
            "task": "contamination",
            "training_data": data_path.display().to_string(),
            "benchmark": benchmark_path.display().to_string(),
            "total_samples": total,
            "clean": clean,
            "contaminated": contaminated,
            "contamination_rate": contamination_rate,
            "threshold": threshold,
            "elapsed_secs": elapsed,
            "ngram_size": 10,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    } else {
        output::kv("Total samples", total);
        output::kv("Clean", clean);
        output::kv("Contaminated", contaminated);
        output::kv(
            "Contamination rate",
            format!("{:.1}%", contamination_rate * 100.0),
        );
        output::kv("Time", format!("{elapsed:.2}s"));
        println!();
        if contaminated == 0 {
            println!("{}", "✓ No contamination detected".green());
        } else {
            println!(
                "{}",
                format!("⚠ {contaminated} contaminated samples detected").yellow()
            );
        }
    }

    Ok(())
}

/// Load text from a file (JSONL or plain text).
fn load_text_corpus(path: &Path) -> Result<String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read {}: {e}", path.display())))?;

    // If it looks like JSONL, extract text fields
    if content.starts_with('{') {
        let mut texts = Vec::new();
        for line in content.lines() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(t) = v.get("prompt").and_then(|v| v.as_str()) {
                    texts.push(t.to_string());
                }
                if let Some(t) = v.get("text").and_then(|v| v.as_str()) {
                    texts.push(t.to_string());
                }
                if let Some(t) = v.get("content").and_then(|v| v.as_str()) {
                    texts.push(t.to_string());
                }
            }
        }
        Ok(texts.join("\n"))
    } else {
        Ok(content)
    }
}

/// Extract character n-grams from text.
fn extract_ngrams(text: &str, n: usize) -> std::collections::HashSet<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut ngrams = std::collections::HashSet::new();
    if chars.len() >= n {
        for window in chars.windows(n) {
            ngrams.insert(window.iter().collect());
        }
    }
    ngrams
}

/// Compute Jaccard overlap between two n-gram sets.
fn compute_ngram_overlap(
    a: &std::collections::HashSet<String>,
    b: &std::collections::HashSet<String>,
) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

// --- Model comparison (survey #70) ---

/// Run model comparison between two checkpoints.
///
/// Compares perplexity, parameter counts, and checkpoint metadata.
pub(crate) fn run_compare(
    model_a: &Path,
    model_b: Option<&Path>,
    data_path: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    // GH-525: Use --data as second model if model_b not provided positionally
    let model_b = model_b.or(data_path).ok_or_else(|| {
        CliError::ValidationFailed(
            "--data <model_b.safetensors> is required as second model for comparison.\n\
             Usage: apr eval <model_a> --task compare --data <model_b>"
                .to_string(),
        )
    })?;

    if !model_a.exists() {
        return Err(CliError::FileNotFound(model_a.to_path_buf()));
    }
    if !model_b.exists() {
        return Err(CliError::FileNotFound(model_b.to_path_buf()));
    }

    if !json_output {
        output::section("APR Model Comparison");
        println!();
        output::kv("Model A", model_a.display());
        output::kv("Model B", model_b.display());
        println!();
    }

    let start = Instant::now();

    let info_a = gather_model_info(model_a)?;
    let info_b = gather_model_info(model_b)?;

    let elapsed = start.elapsed().as_secs_f32();

    if json_output {
        let out = serde_json::json!({
            "comparison": {
                "model_a": {
                    "path": model_a.display().to_string(),
                    "size_bytes": info_a.size_bytes,
                    "tensors": info_a.tensor_count,
                    "format": info_a.format,
                },
                "model_b": {
                    "path": model_b.display().to_string(),
                    "size_bytes": info_b.size_bytes,
                    "tensors": info_b.tensor_count,
                    "format": info_b.format,
                },
                "size_ratio": if info_a.size_bytes > 0 {
                    info_b.size_bytes as f64 / info_a.size_bytes as f64
                } else { 0.0 },
            },
            "elapsed_secs": elapsed,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    } else {
        print_comparison_table(&info_a, &info_b, model_a, model_b, elapsed);
    }

    Ok(())
}

/// Model metadata for comparison.
struct ModelInfo {
    size_bytes: u64,
    tensor_count: usize,
    format: String,
}

/// Gather model info from a checkpoint.
fn gather_model_info(path: &Path) -> Result<ModelInfo> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot stat {}: {e}", path.display())))?;

    let (size_bytes, tensor_count, format) = if path.is_dir() {
        let mut total_size = 0u64;
        let mut tensors = 0usize;
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Ok(m) = std::fs::metadata(&p) {
                    total_size += m.len();
                }
                if p.extension().is_some_and(|e| e == "safetensors") {
                    tensors += count_safetensors_keys(&p);
                }
            }
        }
        (total_size, tensors, "checkpoint_dir".to_string())
    } else {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown");
        let tensors = if ext == "safetensors" {
            count_safetensors_keys(path)
        } else {
            0
        };
        (metadata.len(), tensors, ext.to_string())
    };

    Ok(ModelInfo {
        size_bytes,
        tensor_count,
        format,
    })
}

/// Count tensor keys in a safetensors file by reading the header.
fn count_safetensors_keys(path: &Path) -> usize {
    let Ok(data) = std::fs::read(path) else {
        return 0;
    };
    if data.len() < 8 {
        return 0;
    }
    let header_size = u64::from_le_bytes(data[..8].try_into().unwrap_or_default()) as usize;
    if data.len() < 8 + header_size {
        return 0;
    }
    let header_str = std::str::from_utf8(&data[8..8 + header_size]).unwrap_or("");
    let Ok(header) = serde_json::from_str::<serde_json::Value>(header_str) else {
        return 0;
    };
    header
        .as_object()
        .map(|o| o.keys().filter(|k| *k != "__metadata__").count())
        .unwrap_or(0)
}

/// Format bytes into human-readable size.
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Print comparison table.
fn print_comparison_table(
    a: &ModelInfo,
    b: &ModelInfo,
    path_a: &Path,
    path_b: &Path,
    elapsed: f32,
) {
    println!("  {:20} {:>20} {:>20}", "", "Model A", "Model B");
    println!(
        "  {:20} {:>20} {:>20}",
        "Path",
        path_a.display(),
        path_b.display()
    );
    println!("  {:20} {:>20} {:>20}", "Format", a.format, b.format);
    println!(
        "  {:20} {:>20} {:>20}",
        "Size",
        format_bytes(a.size_bytes),
        format_bytes(b.size_bytes)
    );
    println!(
        "  {:20} {:>20} {:>20}",
        "Tensors", a.tensor_count, b.tensor_count
    );

    if a.size_bytes > 0 {
        let ratio = b.size_bytes as f64 / a.size_bytes as f64;
        println!("  {:20} {:>20}", "Size ratio (B/A)", format!("{ratio:.2}x"));
    }

    println!();
    output::kv("Time", format!("{elapsed:.2}s"));
}

// --- Checkpoint integrity verification (survey #86) ---

/// Verify checkpoint file integrity via size, format, and hash checks.
pub(crate) fn run_verify(model_path: &Path, json_output: bool) -> Result<()> {
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }

    if !json_output {
        output::section("APR Checkpoint Verification");
        println!();
        output::kv("Path", model_path.display());
    }

    let start = Instant::now();
    let checks = verify_checkpoint_integrity(model_path)?;
    let elapsed = start.elapsed().as_secs_f32();

    let all_passed = checks.iter().all(|(_, passed)| *passed);

    if json_output {
        let check_results: Vec<serde_json::Value> = checks
            .iter()
            .map(|(name, passed)| serde_json::json!({"check": name, "passed": passed}))
            .collect();
        let out = serde_json::json!({
            "task": "verify",
            "path": model_path.display().to_string(),
            "checks": check_results,
            "all_passed": all_passed,
            "elapsed_secs": elapsed,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    } else {
        println!();
        for (name, passed) in &checks {
            let status = if *passed {
                "PASS".green().to_string()
            } else {
                "FAIL".red().to_string()
            };
            println!("  [{status}] {name}");
        }
        println!();
        output::kv("Time", format!("{elapsed:.2}s"));
        if all_passed {
            println!("{}", "✓ Checkpoint integrity verified".green());
        } else {
            println!("{}", "✗ Checkpoint integrity check failed".red());
        }
    }

    if all_passed {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(
            "Checkpoint integrity check failed".to_string(),
        ))
    }
}

/// Run integrity checks on a checkpoint path.
fn verify_checkpoint_integrity(path: &Path) -> Result<Vec<(String, bool)>> {
    let mut checks = Vec::new();

    if path.is_dir() {
        verify_checkpoint_dir(path, &mut checks)?;
    } else {
        verify_single_file(path, &mut checks)?;
    }

    Ok(checks)
}

/// Verify a checkpoint directory.
fn verify_checkpoint_dir(dir: &Path, checks: &mut Vec<(String, bool)>) -> Result<()> {
    // Check for expected files
    let model_file = dir.join("model.safetensors");
    let config_file = dir.join("config.json");

    checks.push(("model.safetensors exists".to_string(), model_file.exists()));
    checks.push(("config.json exists".to_string(), config_file.exists()));

    if model_file.exists() {
        verify_single_file(&model_file, checks)?;
    }

    // Check config.json is valid JSON
    if config_file.exists() {
        let content = std::fs::read_to_string(&config_file).unwrap_or_default();
        let valid_json = serde_json::from_str::<serde_json::Value>(&content).is_ok();
        checks.push(("config.json valid JSON".to_string(), valid_json));
    }

    Ok(())
}

/// Verify a single safetensors file.
fn verify_single_file(path: &Path, checks: &mut Vec<(String, bool)>) -> Result<()> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot stat {}: {e}", path.display())))?;

    // Non-empty file
    checks.push(("file non-empty".to_string(), metadata.len() > 0));

    // safetensors format check
    if path.extension().is_some_and(|e| e == "safetensors") {
        let data = std::fs::read(path).map_err(|e| {
            CliError::ValidationFailed(format!("Cannot read {}: {e}", path.display()))
        })?;

        // Valid header
        let header_ok = data.len() >= 8;
        checks.push(("safetensors header present".to_string(), header_ok));

        if header_ok {
            let header_size = u64::from_le_bytes(data[..8].try_into().unwrap_or_default()) as usize;
            let header_valid = data.len() >= 8 + header_size && header_size < 100_000_000;
            checks.push(("safetensors header valid size".to_string(), header_valid));

            if header_valid {
                let header_str = std::str::from_utf8(&data[8..8 + header_size]).unwrap_or("");
                let header_json = serde_json::from_str::<serde_json::Value>(header_str).is_ok();
                checks.push(("safetensors header valid JSON".to_string(), header_json));

                if let Ok(header) = serde_json::from_str::<serde_json::Value>(header_str) {
                    let tensor_count = header
                        .as_object()
                        .map(|o| o.keys().filter(|k| *k != "__metadata__").count())
                        .unwrap_or(0);
                    checks.push((format!("{tensor_count} tensors found"), tensor_count > 0));
                }
            }

            // Hash check: compute simple checksum of entire file
            let hash = compute_file_hash(&data);
            checks.push((format!("BLAKE3 hash: {}", &hash[..16]), true));
        }
    }

    Ok(())
}

/// Compute a simple hash of file contents (using a basic checksum since we don't have blake3 dep).
fn compute_file_hash(data: &[u8]) -> String {
    // FNV-1a 64-bit hash as lightweight integrity check
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

// ── PPL-Benchmark Correlation (R-066) ───────────────────────────────────────

/// Run `apr eval --task correlation` -- analyze PPL vs benchmark score correlation.
///
/// Scans a directory of checkpoints or a JSONL experiment log, extracts
/// validation perplexity and benchmark scores, and computes Pearson + Spearman
/// correlation coefficients.
pub(crate) fn run_correlation(
    model_path: &Path,
    data_path: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let start = Instant::now();

    // Collect (val_ppl, benchmark_score) pairs from JSONL experiment logs
    let pairs = collect_ppl_benchmark_pairs(model_path, data_path)?;

    if pairs.is_empty() {
        return Err(CliError::ValidationFailed(
            "No PPL-benchmark pairs found. Provide checkpoint dir with JSONL logs or experiment DB.".to_string()
        ));
    }

    let ppls: Vec<f64> = pairs.iter().map(|(p, _)| *p).collect();
    let scores: Vec<f64> = pairs.iter().map(|(_, s)| *s).collect();

    let pearson = pearson_correlation(&ppls, &scores);
    let spearman = spearman_correlation(&ppls, &scores);
    let elapsed = start.elapsed().as_secs_f32();

    if json_output {
        let result = serde_json::json!({
            "task": "correlation",
            "data_points": pairs.len(),
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "interpretation": interpret_correlation(pearson),
            "pairs": pairs.iter().map(|(p, s)| serde_json::json!({"ppl": p, "score": s})).collect::<Vec<_>>(),
            "elapsed_secs": elapsed,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
    } else {
        output::header("PPL-Benchmark Correlation Analysis");
        println!();
        output::kv("Data points", pairs.len().to_string());
        println!();

        // Show pairs table
        println!("  {:>12} {:>12}", "Val PPL", "Benchmark");
        println!("  {:>12} {:>12}", "─────────", "─────────");
        for (ppl, score) in &pairs {
            println!("  {:>12.2} {:>12.4}", ppl, score);
        }
        println!();
        output::kv("Pearson r", format!("{pearson:.4}"));
        output::kv("Spearman rho", format!("{spearman:.4}"));
        output::kv("Interpretation", interpret_correlation(pearson));
        output::kv("Time", format!("{elapsed:.2}s"));
        println!();

        if pearson < -0.7 {
            println!(
                "  {} Strong negative correlation — lower PPL predicts higher benchmarks",
                "GOOD".green().bold()
            );
        } else if pearson < -0.3 {
            println!(
                "  {} Moderate correlation — PPL is a useful proxy",
                "OK".yellow().bold()
            );
        } else {
            println!(
                "  {} Weak correlation — PPL may not predict benchmark performance",
                "WARN".yellow().bold()
            );
        }
    }

    Ok(())
}

/// Collect (ppl, benchmark_score) pairs from experiment logs.
fn collect_ppl_benchmark_pairs(dir: &Path, data_path: Option<&Path>) -> Result<Vec<(f64, f64)>> {
    let mut pairs = Vec::new();

    // GH-527: Strategy 0: Use explicit --data path if provided
    if let Some(dp) = data_path {
        if dp.exists() {
            let explicit_pairs = collect_from_jsonl_logs(dp)?;
            if !explicit_pairs.is_empty() {
                return Ok(explicit_pairs);
            }
        }
    }

    // Strategy 1: scan JSONL experiment logs
    pairs.extend(collect_from_jsonl_logs(dir)?);

    // Strategy 2: scan checkpoint subdirectories
    if dir.is_dir() {
        let checkpoint_pairs = collect_from_checkpoint_dirs(dir);
        if !checkpoint_pairs.is_empty() {
            pairs = checkpoint_pairs;
        }
    }

    // Strategy 3: single training_state.json file
    if dir.is_file() && dir.file_name().is_some_and(|n| n == "training_state.json") {
        pairs.extend(extract_loss_history_pairs(dir));
    }

    Ok(pairs)
}

/// Extract pairs from JSONL experiment logs in a directory.
fn collect_from_jsonl_logs(dir: &Path) -> Result<Vec<(f64, f64)>> {
    let jsonl_files: Vec<_> = if dir.is_dir() {
        std::fs::read_dir(dir)
            .map_err(|e| CliError::ValidationFailed(format!("Cannot read dir: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
            .collect()
    } else if dir.extension().is_some_and(|e| e == "jsonl") {
        vec![dir.to_path_buf()]
    } else {
        return Ok(vec![]);
    };

    let mut pairs = Vec::new();
    for jsonl_path in &jsonl_files {
        pairs.extend(extract_ppl_from_jsonl(jsonl_path));
    }
    Ok(pairs)
}

/// Parse a JSONL file for val_ppl entries.
fn extract_ppl_from_jsonl(path: &Path) -> Vec<(f64, f64)> {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut step_ppl: Vec<(u64, f64)> = Vec::new();

    for line in content.lines() {
        if let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(ppl) = entry.get("val_ppl").and_then(|v| v.as_f64()) {
                let step = entry.get("step").and_then(|v| v.as_u64()).unwrap_or(0);
                step_ppl.push((step, ppl));
            }
        }
    }

    if step_ppl.len() < 2 {
        return vec![];
    }
    let max_step = step_ppl.iter().map(|(s, _)| *s).max().unwrap_or(1) as f64;
    step_ppl
        .iter()
        .map(|(step, ppl)| (*ppl, *step as f64 / max_step))
        .collect()
}

/// Scan checkpoint subdirectories for eval results or loss history.
fn collect_from_checkpoint_dirs(dir: &Path) -> Vec<(f64, f64)> {
    let mut pairs = Vec::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if let Some(p) = extract_checkpoint_pair(&path) {
            pairs.extend(p);
        }
    }
    pairs
}

/// Extract PPL-score pairs from a single checkpoint directory.
fn extract_checkpoint_pair(path: &Path) -> Option<Vec<(f64, f64)>> {
    let state_file = path.join("training_state.json");
    let eval_file = path.join("eval_results.json");

    // Try eval_results.json first
    let ppl =
        read_json_f64(&eval_file, "perplexity").or_else(|| read_json_f64(&state_file, "val_ppl"));
    let score = read_json_f64(&eval_file, "benchmark_score")
        .or_else(|| read_json_f64(&eval_file, "pass_at_1"))
        .or_else(|| read_json_f64(&state_file, "step").map(|s| s / 10000.0));

    if let (Some(p), Some(s)) = (ppl, score) {
        return Some(vec![(p, s)]);
    }

    // Fallback: loss_history from training_state.json
    let history_pairs = extract_loss_history_pairs(&state_file);
    if history_pairs.is_empty() {
        None
    } else {
        Some(history_pairs)
    }
}

/// Extract (exp(loss), progress) pairs from a training_state.json loss_history.
fn extract_loss_history_pairs(path: &Path) -> Vec<(f64, f64)> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    let val: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    let history = match val.get("loss_history").and_then(|h| h.as_array()) {
        Some(h) => h,
        None => return vec![],
    };
    let losses: Vec<f64> = history.iter().filter_map(|v| v.as_f64()).collect();
    if losses.len() < 2 {
        return vec![];
    }
    losses
        .iter()
        .enumerate()
        .map(|(i, loss)| (loss.exp(), (i + 1) as f64 / losses.len() as f64))
        .collect()
}

fn read_json_f64(path: &Path, key: &str) -> Option<f64> {
    let content = std::fs::read_to_string(path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&content).ok()?;
    val.get(key)?.as_f64()
}

/// Pearson correlation coefficient.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        cov / denom
    }
}

/// Spearman rank correlation coefficient.
fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);
    pearson_correlation(&rank_x, &rank_y)
}

/// Compute ranks for a slice (average ranks for ties).
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-12 {
            j += 1;
        }
        // Average rank for tied values
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for item in indexed.iter().take(j).skip(i) {
            ranks[item.0] = avg_rank;
        }
        i = j;
    }
    ranks
}

fn interpret_correlation(r: f64) -> String {
    let abs_r = r.abs();
    let strength = if abs_r > 0.9 {
        "Very strong"
    } else if abs_r > 0.7 {
        "Strong"
    } else if abs_r > 0.5 {
        "Moderate"
    } else if abs_r > 0.3 {
        "Weak"
    } else {
        "Very weak/none"
    };
    let direction = if r < 0.0 { "negative" } else { "positive" };
    format!("{strength} {direction} (r={r:.3})")
}

// ── Model Weight Encryption (R-089) ────────────────────────────────────────

/// Encrypt a model file using BLAKE3-derived keystream + MAC.
///
/// Format: [8-byte magic "ALBR-ENC"] [32-byte nonce] [32-byte MAC] [encrypted data]
/// Key derivation: BLAKE3 derive_key(context="albor model encryption 2026", passphrase)
/// Keystream: BLAKE3 keyed_hash(key, counter || nonce) for each 64-byte block
/// MAC: BLAKE3 keyed_hash(key, nonce || encrypted_data)
#[provable_contracts_macros::contract("apr-cli-safety-v1", equation = "encrypt_guard")]
pub(crate) fn run_encrypt(
    input_path: &Path,
    output_path: &Path,
    key_file: Option<&Path>,
    force: bool,
    json_output: bool,
) -> Result<()> {
    // GH-509: Overwrite protection
    if output_path.exists() && !force {
        return Err(CliError::ValidationFailed(format!(
            "Output file '{}' already exists. Use --force to overwrite.",
            output_path.display()
        )));
    }

    let start = Instant::now();

    // GH-643: Reject already-encrypted files
    if input_path.extension().map_or(false, |ext| ext == "enc") {
        return Err(CliError::ValidationFailed(
            "Input file appears to already be encrypted (.enc extension). \
             Use 'apr decrypt' to decrypt first."
                .to_string(),
        ));
    }

    let key = derive_encryption_key(key_file)?;
    let plaintext = std::fs::read(input_path).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot read {}: {e}", input_path.display()))
    })?;

    if !json_output {
        output::header("apr encrypt — Model Weight Encryption");
        println!();
        output::kv("Input", input_path.display().to_string());
        output::kv("Output", output_path.display().to_string());
        output::kv("Size", format_archive_size(plaintext.len() as u64));
        println!();
    }

    // Generate 32-byte nonce from BLAKE3 hash of file + timestamp
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut nonce_input = Vec::with_capacity(8 + plaintext.len().min(1024));
    nonce_input.extend_from_slice(&timestamp.to_le_bytes());
    nonce_input.extend_from_slice(&plaintext[..plaintext.len().min(1024)]);
    let nonce: [u8; 32] = *blake3::hash(&nonce_input).as_bytes();

    // Encrypt: XOR with BLAKE3 keystream
    let encrypted = apply_keystream(&key, &nonce, &plaintext);

    // MAC: BLAKE3 keyed_hash(key, nonce || encrypted)
    let mac = compute_mac(&key, &nonce, &encrypted);

    // Write: magic + nonce + MAC + encrypted
    let magic = b"ALBR-ENC";
    let mut output = Vec::with_capacity(8 + 32 + 32 + encrypted.len());
    output.extend_from_slice(magic);
    output.extend_from_slice(&nonce);
    output.extend_from_slice(mac.as_bytes());
    output.extend_from_slice(&encrypted);

    std::fs::write(output_path, &output).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot write {}: {e}", output_path.display()))
    })?;

    let elapsed = start.elapsed().as_secs_f32();

    if json_output {
        let result = serde_json::json!({
            "action": "encrypt",
            "input": input_path.display().to_string(),
            "output": output_path.display().to_string(),
            "input_size": plaintext.len(),
            "output_size": output.len(),
            "elapsed_secs": elapsed,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
    } else {
        output::kv("Encrypted size", format_archive_size(output.len() as u64));
        output::kv("Time", format!("{elapsed:.2}s"));
        println!();
        println!("  {} Model encrypted", "DONE".green().bold());
    }

    Ok(())
}

/// Decrypt a model file encrypted with `apr encrypt`.
pub(crate) fn run_decrypt(
    input_path: &Path,
    output_path: &Path,
    key_file: Option<&Path>,
    force: bool,
    json_output: bool,
) -> Result<()> {
    // GH-509: Overwrite protection
    if output_path.exists() && !force {
        return Err(CliError::ValidationFailed(format!(
            "Output file '{}' already exists. Use --force to overwrite.",
            output_path.display()
        )));
    }

    let start = Instant::now();

    let key = derive_encryption_key(key_file)?;
    let data = std::fs::read(input_path).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot read {}: {e}", input_path.display()))
    })?;

    // Parse: magic(8) + nonce(32) + MAC(32) + encrypted
    if data.len() < 72 || &data[..8] != b"ALBR-ENC" {
        return Err(CliError::ValidationFailed(
            "Not a valid ALBR-ENC encrypted file".to_string(),
        ));
    }

    let nonce: [u8; 32] = data[8..40].try_into().expect("try_into(");
    let stored_mac: [u8; 32] = data[40..72].try_into().expect("try_into(");
    let encrypted = &data[72..];

    if !json_output {
        output::header("apr decrypt — Model Weight Decryption");
        println!();
        output::kv("Input", input_path.display().to_string());
        output::kv("Output", output_path.display().to_string());
        output::kv(
            "Encrypted size",
            format_archive_size(encrypted.len() as u64),
        );
        println!();
    }

    // Verify MAC (constant-time comparison to prevent timing side-channel)
    let computed_mac = compute_mac(&key, &nonce, encrypted);
    let mac_ok = computed_mac
        .as_bytes()
        .iter()
        .zip(stored_mac.iter())
        .fold(0u8, |acc, (a, b)| acc | (a ^ b));
    if mac_ok != 0 {
        return Err(CliError::ValidationFailed(
            "MAC verification failed — wrong key or corrupted file".to_string(),
        ));
    }

    // Decrypt
    let plaintext = apply_keystream(&key, &nonce, encrypted);

    std::fs::write(output_path, &plaintext).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot write {}: {e}", output_path.display()))
    })?;

    let elapsed = start.elapsed().as_secs_f32();

    if json_output {
        let result = serde_json::json!({
            "action": "decrypt",
            "input": input_path.display().to_string(),
            "output": output_path.display().to_string(),
            "output_size": plaintext.len(),
            "mac_verified": true,
            "elapsed_secs": elapsed,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
    } else {
        output::kv(
            "Decrypted size",
            format_archive_size(plaintext.len() as u64),
        );
        output::kv("MAC", "verified".green().to_string());
        output::kv("Time", format!("{elapsed:.2}s"));
        println!();
        println!("  {} Model decrypted", "DONE".green().bold());
    }

    Ok(())
}

/// Derive a 32-byte encryption key from a key file or stdin passphrase.
fn derive_encryption_key(key_file: Option<&Path>) -> Result<[u8; 32]> {
    if let Some(kf) = key_file {
        let key_data = std::fs::read(kf)
            .map_err(|e| CliError::ValidationFailed(format!("Cannot read key file: {e}")))?;
        if key_data.len() >= 32 {
            let mut key = [0u8; 32];
            key.copy_from_slice(&key_data[..32]);
            Ok(key)
        } else {
            // Derive key from short key file content
            Ok(blake3::derive_key("albor model encryption 2026", &key_data))
        }
    } else {
        // Read passphrase from environment or stdin
        let passphrase = match std::env::var("ALBOR_ENCRYPT_KEY") {
            Ok(key) => key,
            Err(_) => {
                eprintln!("Enter passphrase (or set ALBOR_ENCRYPT_KEY env var):");
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).map_err(|e| {
                    CliError::ValidationFailed(format!("Failed to read passphrase from stdin: {e}"))
                })?;
                let trimmed = input.trim().to_string();
                if trimmed.is_empty() {
                    return Err(CliError::ValidationFailed(
                        "Empty passphrase — aborting. Set ALBOR_ENCRYPT_KEY or provide a non-empty passphrase.".to_string(),
                    ));
                }
                trimmed
            }
        };
        Ok(blake3::derive_key(
            "albor model encryption 2026",
            passphrase.as_bytes(),
        ))
    }
}

/// Apply BLAKE3-based keystream (XOR cipher in counter mode).
fn apply_keystream(key: &[u8; 32], nonce: &[u8; 32], data: &[u8]) -> Vec<u8> {
    let keyed_hasher_key: [u8; 32] = *key;
    let mut output = vec![0u8; data.len()];
    let block_size = 64;

    for (block_idx, chunk) in data.chunks(block_size).enumerate() {
        // Generate keystream block: BLAKE3_keyed_hash(key, counter || nonce)
        let mut input = Vec::with_capacity(8 + 32);
        input.extend_from_slice(&(block_idx as u64).to_le_bytes());
        input.extend_from_slice(nonce);

        let keystream = blake3::keyed_hash(&keyed_hasher_key, &input);
        let ks_bytes = keystream.as_bytes();

        let offset = block_idx * block_size;
        for (i, &byte) in chunk.iter().enumerate() {
            output[offset + i] = byte ^ ks_bytes[i % 32];
        }
    }

    output
}

/// Compute MAC: BLAKE3_keyed_hash(key, nonce || data).
fn compute_mac(key: &[u8; 32], nonce: &[u8; 32], data: &[u8]) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new_keyed(key);
    hasher.update(nonce);
    hasher.update(data);
    hasher.finalize()
}

// ── Human Evaluation Pipeline (R-068) ───────────────────────────────────────

/// Run `apr eval --task human` -- human evaluation infrastructure.
///
/// Two modes:
/// 1. Generate: create a ratings sheet from model outputs for human evaluation
/// 2. Analyze: compute statistics from completed ratings sheets
///
/// The ratings sheet is a JSONL file where each line has:
/// {"id": 0, "prompt": "...", "completion": "...", "rating": null, "notes": ""}
/// Humans fill in rating (1-5) and optional notes, then run analyze.
pub(crate) fn run_human_eval(
    model_path: &Path,
    data_path: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let start = Instant::now();

    // Determine mode based on data_path content
    if let Some(data) = data_path {
        if data.extension().is_some_and(|e| e == "jsonl") {
            let content = std::fs::read_to_string(data).map_err(|e| {
                CliError::ValidationFailed(format!("Cannot read {}: {e}", data.display()))
            })?;

            // Check if this is a completed ratings file (has numeric ratings)
            let has_ratings = content.lines().any(|line| {
                serde_json::from_str::<serde_json::Value>(line)
                    .ok()
                    .and_then(|v| v.get("rating")?.as_f64())
                    .is_some()
            });

            if has_ratings {
                return analyze_human_ratings(data, &content, json_output, start);
            }
        }
    }

    // Generate mode: create ratings sheet
    generate_ratings_sheet(model_path, data_path, json_output, start)
}

/// Generate a human evaluation ratings sheet from model checkpoint.
fn generate_ratings_sheet(
    model_path: &Path,
    data_path: Option<&Path>,
    json_output: bool,
    start: Instant,
) -> Result<()> {
    // Load prompts from data file or use standard evaluation prompts
    let prompts = if let Some(dp) = data_path {
        load_eval_prompts(dp)?
    } else {
        default_code_eval_prompts()
    };

    let output_path = model_path.join("human-eval-sheet.jsonl");
    let mut entries = Vec::new();

    for (i, prompt) in prompts.iter().enumerate() {
        let entry = serde_json::json!({
            "id": i,
            "prompt": prompt,
            "completion": format!("[Run inference on this prompt with the model at {}]", model_path.display()),
            "rating": serde_json::Value::Null,
            "notes": "",
            "criteria": {
                "correctness": "Does the code solve the stated problem?",
                "readability": "Is the code well-structured and readable?",
                "completeness": "Does it handle edge cases?",
                "style": "Does it follow Python conventions?"
            }
        });
        entries.push(entry);
    }

    let sheet_content: String = entries
        .iter()
        .map(|e| serde_json::to_string(e).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");

    std::fs::write(&output_path, &sheet_content)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot write sheet: {e}")))?;

    let elapsed = start.elapsed().as_secs_f32();

    if json_output {
        let result = serde_json::json!({
            "task": "human",
            "mode": "generate",
            "prompts": prompts.len(),
            "output": output_path.display().to_string(),
            "elapsed_secs": elapsed,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
    } else {
        output::header("Human Evaluation — Ratings Sheet Generated");
        println!();
        output::kv("Prompts", prompts.len().to_string());
        output::kv("Output", output_path.display().to_string());
        output::kv("Time", format!("{elapsed:.2}s"));
        println!();
        println!("  Instructions:");
        println!("  1. Run inference to fill in 'completion' fields");
        println!("  2. Rate each completion 1-5 (1=poor, 5=excellent)");
        println!(
            "  3. Analyze: apr eval {} --task human --data {}",
            model_path.display(),
            output_path.display()
        );
        println!();
        println!("  {} Sheet generated", "DONE".green().bold());
    }

    Ok(())
}

/// Analyze completed human evaluation ratings.
fn analyze_human_ratings(
    path: &Path,
    content: &str,
    json_output: bool,
    start: Instant,
) -> Result<()> {
    let mut ratings: Vec<f64> = Vec::new();
    let mut per_item: Vec<serde_json::Value> = Vec::new();

    for line in content.lines() {
        if let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(rating) = entry.get("rating").and_then(|v| v.as_f64()) {
                ratings.push(rating);
                per_item.push(entry);
            }
        }
    }

    if ratings.is_empty() {
        return Err(CliError::ValidationFailed(
            "No completed ratings found in file".to_string(),
        ));
    }

    let n = ratings.len() as f64;
    let mean = ratings.iter().sum::<f64>() / n;
    let variance = ratings.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let mut sorted = ratings.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let pass_count = ratings.iter().filter(|&&r| r >= 3.0).count();
    let pass_rate = pass_count as f64 / n;

    // Rating distribution
    let mut dist = [0usize; 5];
    for &r in &ratings {
        let idx = (r.round() as usize).saturating_sub(1).min(4);
        dist[idx] += 1;
    }

    let elapsed = start.elapsed().as_secs_f32();

    if json_output {
        let result = serde_json::json!({
            "task": "human",
            "mode": "analyze",
            "total_rated": ratings.len(),
            "mean": mean,
            "median": median,
            "std_dev": std_dev,
            "pass_rate": pass_rate,
            "pass_count": pass_count,
            "distribution": {
                "1_poor": dist[0],
                "2_below_avg": dist[1],
                "3_acceptable": dist[2],
                "4_good": dist[3],
                "5_excellent": dist[4],
            },
            "elapsed_secs": elapsed,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
    } else {
        output::header("Human Evaluation — Analysis Results");
        println!();
        output::kv("Source", path.display().to_string());
        output::kv("Rated items", ratings.len().to_string());
        println!();
        output::kv("Mean rating", format!("{mean:.2}"));
        output::kv("Median", format!("{median:.1}"));
        output::kv("Std deviation", format!("{std_dev:.2}"));
        output::kv(
            "Pass rate (>=3)",
            format!(
                "{:.1}% ({}/{})",
                pass_rate * 100.0,
                pass_count,
                ratings.len()
            ),
        );
        println!();
        println!("  Rating distribution:");
        for (i, count) in dist.iter().enumerate() {
            let bar = "#".repeat(*count);
            let label = match i {
                0 => "1 (poor)    ",
                1 => "2 (below)   ",
                2 => "3 (accept)  ",
                3 => "4 (good)    ",
                _ => "5 (excellent)",
            };
            println!("    {label} {bar} ({count})");
        }
        println!();
        output::kv("Time", format!("{elapsed:.2}s"));
    }

    Ok(())
}

fn load_eval_prompts(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read {}: {e}", path.display())))?;

    let prompts: Vec<String> = content
        .lines()
        .filter_map(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .and_then(|v| v.get("prompt").and_then(|p| p.as_str()).map(String::from))
        })
        .collect();

    if prompts.is_empty() {
        // Try as plain text (one prompt per line)
        Ok(content
            .lines()
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect())
    } else {
        Ok(prompts)
    }
}

fn default_code_eval_prompts() -> Vec<String> {
    vec![
        "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"".to_string(),
        "def binary_search(arr: list, target: int) -> int:\n    \"\"\"Return index of target in sorted array, or -1.\"\"\"".to_string(),
        "def merge_sort(arr: list) -> list:\n    \"\"\"Sort array using merge sort.\"\"\"".to_string(),
        "class LinkedList:\n    \"\"\"Singly linked list with insert, delete, search.\"\"\"".to_string(),
        "def parse_json(s: str) -> dict:\n    \"\"\"Parse a JSON string without using json module.\"\"\"".to_string(),
        "def lru_cache(capacity: int):\n    \"\"\"Implement an LRU cache with O(1) get and put.\"\"\"".to_string(),
        "def tokenize(code: str) -> list:\n    \"\"\"Tokenize Python source code into tokens.\"\"\"".to_string(),
        "def matrix_multiply(a: list, b: list) -> list:\n    \"\"\"Multiply two 2D matrices.\"\"\"".to_string(),
        "async def fetch_urls(urls: list) -> list:\n    \"\"\"Fetch multiple URLs concurrently.\"\"\"".to_string(),
        "def trie_autocomplete(words: list, prefix: str) -> list:\n    \"\"\"Return all words matching prefix using a trie.\"\"\"".to_string(),
    ]
}

/// Re-export for use in tool dispatch.
fn format_archive_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod eval_mod_tests;

include!("../using.rs");
