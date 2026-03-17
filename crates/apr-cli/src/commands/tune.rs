//! ML Tuning Command (GH-176, PMAT-184)
//!
//! Provides LoRA/QLoRA fine-tuning capabilities via entrenar-lora.
//!
//! Toyota Way: Muda Elimination - Reuses entrenar instead of reimplementing.
//!
//! # Example
//!
//! ```bash
//! apr tune model.gguf --method lora --rank 8           # Plan LoRA config
//! apr tune model.gguf --method qlora --vram 16         # Plan QLoRA for 16GB VRAM
//! apr tune --plan 7B --vram 24                         # Memory planning
//! ```

use crate::error::CliError;
use crate::output;
use colored::Colorize;
use entrenar_lora::{plan, MemoryPlanner, Method};
use std::path::Path;

/// Tuning method selection
#[derive(Debug, Clone, Copy, Default)]
pub enum TuneMethod {
    #[default]
    Auto,
    Full,
    LoRA,
    QLoRA,
}

impl std::str::FromStr for TuneMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "full" => Ok(Self::Full),
            "lora" => Ok(Self::LoRA),
            "qlora" => Ok(Self::QLoRA),
            _ => Err(format!("Unknown method: {s}. Use: auto, full, lora, qlora")),
        }
    }
}

impl From<TuneMethod> for Method {
    fn from(m: TuneMethod) -> Self {
        match m {
            TuneMethod::Auto => Method::Auto,
            TuneMethod::Full => Method::Full,
            TuneMethod::LoRA => Method::LoRA,
            TuneMethod::QLoRA => Method::QLoRA,
        }
    }
}

/// Run the tune command
#[allow(clippy::too_many_arguments)]
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
pub fn run(
    model_path: Option<&Path>,
    method: TuneMethod,
    rank: Option<u32>,
    vram_gb: f64,
    plan_only: bool,
    model_size: Option<&str>,
    freeze_base: bool,
    train_data: Option<&Path>,
    json_output: bool,
) -> Result<(), CliError> {
    // GH-518: Warn on unimplemented tuning flags
    if freeze_base {
        eprintln!("Warning: --freeze-base is not yet implemented. Flag ignored.");
    }
    if train_data.is_some() {
        eprintln!("Warning: --train-data is not yet implemented. Flag ignored.");
    }

    if !json_output {
        output::section("apr tune (GH-176: ML Tuning via entrenar-lora)");
        println!();
    }

    // Determine model parameters
    let model_params = if let Some(size) = model_size {
        parse_model_size(size)?
    } else if let Some(path) = model_path {
        estimate_params_from_file(path)?
    } else {
        return Err(CliError::ValidationFailed(
            "Either --model or model path required".to_string(),
        ));
    };

    if !json_output {
        output::kv("Model parameters", format_params(model_params));
        output::kv("Available VRAM", format!("{:.1} GB", vram_gb));
        output::kv("Method", format!("{:?}", method));
        if let Some(r) = rank {
            output::kv("Requested rank", r.to_string());
        }
        println!();
    }

    // Plan optimal configuration using entrenar-lora
    let config = plan(model_params, vram_gb, method.into())
        .map_err(|e| CliError::ValidationFailed(format!("Failed to plan tuning config: {e}")))?;

    if json_output {
        // JSON output for CI integration
        let json = serde_json::json!({
            "model_params": model_params,
            "vram_gb": vram_gb,
            "recommended_method": format!("{:?}", config.method),
            "recommended_rank": config.rank,
            "recommended_alpha": config.alpha,
            "trainable_params": config.trainable_params,
            "trainable_percent": config.trainable_percent,
            "memory_gb": config.memory_gb,
            "utilization_percent": config.utilization_percent,
            "speedup": config.speedup,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
        return Ok(());
    }

    // Display results
    println!("{}", "RECOMMENDED CONFIGURATION".white().bold());
    println!("{}", "═".repeat(50));
    println!();

    println!(
        "  Method:           {}",
        format!("{:?}", config.method).cyan().bold()
    );
    println!("  Rank:             {}", config.rank.to_string().green());
    println!("  Alpha:            {:.1}", config.alpha);
    println!(
        "  Trainable params: {} ({:.2}%)",
        format_params(config.trainable_params).yellow(),
        config.trainable_percent
    );
    println!(
        "  Memory required:  {:.2} GB ({:.0}% utilization)",
        config.memory_gb, config.utilization_percent
    );
    println!(
        "  Speedup:          {:.1}x vs full fine-tuning",
        config.speedup
    );
    println!();

    // Memory breakdown
    println!("{}", "MEMORY BREAKDOWN".white().bold());
    println!("{}", "─".repeat(50));

    let planner = MemoryPlanner::new(model_params);
    let req = planner.estimate(config.method, config.rank);

    let model_gb = req.model_bytes as f64 / 1e9;
    let adapter_gb = req.adapter_bytes as f64 / 1e9;
    let optimizer_gb = req.optimizer_bytes as f64 / 1e9;
    let activation_gb = req.activation_bytes as f64 / 1e9;
    let total_gb = req.total_bytes as f64 / 1e9;

    println!("  Base model:       {:.2} GB", model_gb);
    println!("  Adapter:          {:.2} GB", adapter_gb);
    println!("  Optimizer states: {:.2} GB", optimizer_gb);
    println!("  Activations:      {:.2} GB", activation_gb);
    println!("{}", "─".repeat(50));
    println!("  {}:            {:.2} GB", "TOTAL".bold(), total_gb);
    println!(
        "  Savings:          {:.0}% vs full fine-tuning",
        req.savings_percent
    );
    println!();

    // Feasibility check
    if total_gb <= vram_gb {
        println!(
            "{} Configuration fits in {:.1} GB VRAM",
            "✓".green().bold(),
            vram_gb
        );
    } else {
        println!(
            "{} Configuration requires {:.2} GB but only {:.1} GB available",
            "⚠".yellow().bold(),
            total_gb,
            vram_gb
        );
        println!();
        println!("  Suggestions:");
        println!("    - Use QLoRA (4-bit quantization)");
        println!("    - Reduce rank (--rank 4)");
        println!("    - Use gradient checkpointing");
    }

    if plan_only {
        return Ok(());
    }

    // If training data provided, show next steps
    println!();
    println!("{}", "NEXT STEPS".white().bold());
    println!("{}", "─".repeat(50));
    println!("  1. Prepare training data in JSONL format");
    println!("  2. Run: apr tune model.gguf --train-data data.jsonl");
    println!(
        "  3. Output adapter saved to: model-lora-r{}.bin",
        config.rank
    );

    Ok(())
}

/// Parse model size string (e.g., "7B", "1.5B", "70B")
fn parse_model_size(size: &str) -> Result<u64, CliError> {
    let size = size.to_uppercase();
    let (num_str, multiplier) = if size.ends_with('B') {
        (&size[..size.len() - 1], 1_000_000_000u64)
    } else if size.ends_with('M') {
        (&size[..size.len() - 1], 1_000_000u64)
    } else {
        return Err(CliError::ValidationFailed(format!(
            "Invalid model size format: {size}. Use: 7B, 1.5B, 70B, etc."
        )));
    };

    let num: f64 = num_str.parse().map_err(|_| {
        CliError::ValidationFailed(format!("Invalid number in model size: {num_str}"))
    })?;

    Ok((num * multiplier as f64) as u64)
}

/// Estimate parameters from model file size.
///
/// GH-484: Use file extension to pick bytes-per-param ratio instead of
/// blindly assuming Q4 (which overestimates fp16/bf16 models by 4x).
fn estimate_params_from_file(path: &Path) -> Result<u64, CliError> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model file: {e}")))?;

    let size_bytes = metadata.len();

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let estimated_params = match ext {
        // GGUF models are typically quantized (Q4-Q8), ~0.5-1.0 bytes/param
        "gguf" => size_bytes * 2,
        // SafeTensors/APR/bin are typically fp16/bf16 (2 bytes/param)
        _ => size_bytes / 2,
    };

    Ok(estimated_params)
}

// ═══════════════════════════════════════════════════════════════════════
// Classify tune (SPEC-TUNE-2026-001)
// ═══════════════════════════════════════════════════════════════════════

/// Run automatic hyperparameter tuning for classification fine-tuning.
///
/// Orchestrates HPO search over LoRA + classifier configurations using
/// entrenar's ClassifyTuner with TPE/Grid/Random searchers and ASHA/Median schedulers.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
pub fn run_classify_tune(
    _model_path: Option<&Path>,
    budget: usize,
    strategy: &str,
    scheduler: &str,
    scout: bool,
    data_path: Option<&Path>,
    num_classes: usize,
    _model_size: Option<&str>,
    _from_scout: Option<&Path>,
    max_epochs: usize,
    _time_limit: Option<&str>,
    json_output: bool,
) -> Result<(), CliError> {
    use entrenar::finetune::{ClassifyTuner, SchedulerKind, TuneConfig, TuneStrategy};

    // Parse strategy
    let tune_strategy: TuneStrategy = strategy
        .parse()
        .map_err(|e: String| CliError::ValidationFailed(e))?;

    // Parse scheduler
    let sched_kind: SchedulerKind = scheduler
        .parse()
        .map_err(|e: String| CliError::ValidationFailed(e))?;

    // Validate data path
    if let Some(path) = data_path {
        if !path.exists() {
            return Err(CliError::ValidationFailed(format!(
                "FALSIFY-TUNE-003: data file not found: {}",
                path.display()
            )));
        }
    }

    // Build TuneConfig
    let tune_config = TuneConfig {
        budget,
        strategy: tune_strategy,
        scheduler: sched_kind,
        scout,
        max_epochs,
        num_classes,
        seed: 42,
        time_limit_secs: None,
    };

    // Create tuner (validates budget > 0 and num_classes > 0)
    let tuner =
        ClassifyTuner::new(tune_config).map_err(|e| CliError::ValidationFailed(e.to_string()))?;

    // Build searcher and scheduler to verify they work
    let mut searcher = tuner.build_searcher();
    let _scheduler_obj = tuner.build_scheduler();

    if json_output {
        return print_classify_tune_json(
            &mut searcher,
            strategy,
            scheduler,
            scout,
            budget,
            num_classes,
            max_epochs,
        );
    }

    print_classify_tune_text(
        &mut searcher,
        tune_strategy,
        scout,
        budget,
        num_classes,
        max_epochs,
        data_path,
    );
    Ok(())
}

/// Print classify tune results as JSON.
#[allow(clippy::disallowed_methods)]
fn print_classify_tune_json(
    searcher: &mut Box<dyn entrenar::finetune::TuneSearcher>,
    strategy: &str,
    scheduler: &str,
    scout: bool,
    budget: usize,
    num_classes: usize,
    max_epochs: usize,
) -> Result<(), CliError> {
    let mut trial_configs = Vec::new();
    for _ in 0..budget.min(3) {
        if let Ok(trial) = searcher.suggest() {
            trial_configs.push(trial.config);
        }
    }

    let json = serde_json::json!({
        "task": "classify",
        "strategy": strategy,
        "scheduler": scheduler,
        "mode": if scout { "scout" } else { "full" },
        "budget": budget,
        "num_classes": num_classes,
        "max_epochs": if scout { 1 } else { max_epochs },
        "search_space_params": 9,
        "sample_configs": trial_configs,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json).unwrap_or_default()
    );
    Ok(())
}

/// Print classify tune results as human-readable text.
fn print_classify_tune_text(
    searcher: &mut Box<dyn entrenar::finetune::TuneSearcher>,
    tune_strategy: entrenar::finetune::TuneStrategy,
    scout: bool,
    budget: usize,
    num_classes: usize,
    max_epochs: usize,
    data_path: Option<&Path>,
) {
    output::section("apr tune — Classification HPO (SPEC-TUNE-2026-001)");
    println!();
    output::kv("Task", "classify");
    output::kv("Strategy", format!("{tune_strategy}"));
    output::kv(
        "Mode",
        if scout {
            "scout (1 epoch/trial)"
        } else {
            "full"
        },
    );
    output::kv("Budget", format!("{budget} trials"));
    output::kv("Classes", num_classes.to_string());
    output::kv(
        "Max epochs",
        if scout {
            "1".to_string()
        } else {
            max_epochs.to_string()
        },
    );

    if let Some(path) = data_path {
        output::kv("Data", path.display().to_string());
    }
    println!();

    println!("{}", "SEARCH SPACE (9 parameters)".bold());
    println!("{}", "─".repeat(50));
    println!("  learning_rate:      5e-6 .. 5e-4 (log)");
    println!("  lora_rank:          4 .. 64 (step 4)");
    println!("  lora_alpha_ratio:   0.5 .. 2.0");
    println!("  batch_size:         [8, 16, 32, 64, 128]");
    println!("  warmup_fraction:    0.01 .. 0.2");
    println!("  gradient_clip_norm: 0.5 .. 5.0");
    println!("  class_weights:      [uniform, inverse_freq, sqrt_inverse]");
    println!("  target_modules:     [qv, qkv, all_linear]");
    println!("  lr_min_ratio:       0.001 .. 0.1 (log)");
    println!();

    println!("{}", "SAMPLE CONFIGURATIONS".bold());
    println!("{}", "─".repeat(50));
    for i in 0..budget.min(3) {
        if let Ok(trial) = searcher.suggest() {
            let (lr, rank, alpha, batch, warmup, clip, weights, targets, lr_min) =
                entrenar::finetune::extract_trial_params(&trial.config);
            println!(
                "  Trial {}: lr={:.2e} rank={} alpha={:.1} batch={} warmup={:.2} clip={:.1} wt={} tgt={} lr_min={:.4}",
                i + 1, lr, rank, alpha, batch, warmup, clip, weights, targets, lr_min
            );
        }
    }
    println!();

    if data_path.is_none() {
        println!("{}", "NEXT STEPS".bold());
        println!("{}", "─".repeat(50));
        println!("  Provide training data to start tuning:");
        println!(
            "  apr tune --task classify --data corpus.jsonl --budget {budget} {}",
            if scout { "--scout" } else { "" }
        );
    }
}

/// Format parameter count for display
fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else {
        format!("{}", params)
    }
}

#[cfg(test)]
#[path = "tune_tests.rs"]
mod tests;
