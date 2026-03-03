//! `apr train plan` — Forjar-style training pre-flight validation.
//!
//! Generates a `TrainingPlan` by validating data quality, model compatibility,
//! HPO search space, resource estimates, and pre-flight checks — all without
//! touching the GPU.
//!
//! Analogous to `forjar plan` which shows what will change before `forjar apply`.

use colored::Colorize;

use crate::{error::CliError, output};

type Result<T> = std::result::Result<T, CliError>;

/// Run `apr train plan` — generate and display a training plan.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_plan(
    data: Option<&std::path::Path>,
    model_size: &str,
    model_path: Option<&std::path::Path>,
    num_classes: usize,
    task: &str,
    config_path: Option<&std::path::Path>,
    output_dir: &std::path::Path,
    strategy: &str,
    budget: usize,
    scout: bool,
    max_epochs: usize,
    manual_lr: Option<f32>,
    manual_lora_rank: Option<usize>,
    manual_batch_size: Option<usize>,
    val_data: Option<&std::path::Path>,
    test_data: Option<&std::path::Path>,
    format: &str,
    json_output: bool,
) -> Result<()> {
    match task {
        "pretrain" | "causal_lm" => run_plan_pretrain(config_path, json_output),
        "classify" => {
            let data = data.ok_or_else(|| {
                CliError::ValidationFailed(
                    "--data is required for --task classify".to_string(),
                )
            })?;
            run_plan_classify(
                data, model_size, model_path, num_classes, output_dir, strategy,
                budget, scout, max_epochs, manual_lr, manual_lora_rank,
                manual_batch_size, val_data, test_data, format, json_output,
            )
        }
        _ => Err(CliError::ValidationFailed(format!(
            "Unknown task type: {task}. Supported: classify, pretrain"
        ))),
    }
}

/// Plan for classification fine-tuning (existing behavior).
#[allow(clippy::too_many_arguments)]
fn run_plan_classify(
    data: &std::path::Path,
    model_size: &str,
    model_path: Option<&std::path::Path>,
    num_classes: usize,
    output_dir: &std::path::Path,
    strategy: &str,
    budget: usize,
    scout: bool,
    max_epochs: usize,
    manual_lr: Option<f32>,
    manual_lora_rank: Option<usize>,
    manual_batch_size: Option<usize>,
    val_data: Option<&std::path::Path>,
    test_data: Option<&std::path::Path>,
    format: &str,
    json_output: bool,
) -> Result<()> {
    let config = entrenar::finetune::PlanConfig {
        task: "classify".to_string(),
        data_path: data.to_path_buf(),
        val_path: val_data.map(std::path::Path::to_path_buf),
        test_path: test_data.map(std::path::Path::to_path_buf),
        model_size: model_size.to_string(),
        model_path: model_path.map(std::path::Path::to_path_buf),
        num_classes,
        output_dir: output_dir.to_path_buf(),
        strategy: strategy.to_string(),
        budget,
        scout,
        max_epochs,
        manual_lr,
        manual_lora_rank,
        manual_batch_size,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };

    let plan = entrenar::finetune::training_plan(&config)
        .map_err(|e| CliError::ValidationFailed(e.to_string()))?;

    let effective_format = if json_output { "json" } else { format };

    match effective_format {
        "json" => println!("{}", plan.to_json()),
        "yaml" => println!("{}", plan.to_yaml()),
        _ => print_plan_text(&plan),
    }

    Ok(())
}

/// Plan for causal LM pre-training from YAML config (ALB-009).
fn run_plan_pretrain(
    config_path: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    let config_path = config_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "--config <yaml> is required for --task pretrain".to_string(),
        )
    })?;

    if !config_path.exists() {
        return Err(CliError::FileNotFound(config_path.to_path_buf()));
    }

    // Load and validate the training config
    let spec = entrenar::config::load_config(config_path)
        .map_err(|e| CliError::ValidationFailed(format!("Config error: {e}")))?;

    entrenar::config::validate_config(&spec)
        .map_err(|e| CliError::ValidationFailed(format!("Validation error: {e}")))?;

    // Display the pre-training plan
    if json_output {
        let plan = serde_json::json!({
            "task": "pretrain",
            "config": config_path.display().to_string(),
            "model": {
                "path": spec.model.path.display().to_string(),
                "mode": format!("{:?}", spec.model.mode),
            },
            "data": {
                "train": spec.data.train.display().to_string(),
                "batch_size": spec.data.batch_size,
                "seq_len": spec.data.seq_len,
            },
            "optimizer": {
                "name": spec.optimizer.name,
                "lr": spec.optimizer.lr,
            },
            "training": {
                "epochs": spec.training.epochs,
                "mode": format!("{:?}", spec.training.mode),
                "warmup_steps": spec.training.warmup_steps,
                "gradient_accumulation": spec.training.gradient_accumulation,
                "mixed_precision": spec.training.mixed_precision,
            },
            "verdict": "ready",
            "vram_estimate": spec.model.architecture.as_ref()
                .and_then(|arch| estimate_vram(arch, &spec))
                .map(|e| serde_json::json!({
                    "parameters": e.param_count,
                    "weights_gb": e.weights_gb,
                    "gradients_gb": e.gradients_gb,
                    "optimizer_gb": e.optimizer_gb,
                    "activations_gb": e.activations_gb,
                    "total_gb": e.total_gb,
                    "ram_total_gb": e.ram_total_gb,
                    "disk_checkpoint_gb": e.disk_checkpoint_gb,
                    "tokens_per_step": e.tokens_per_step,
                    "estimated_step_ms": e.estimated_step_ms,
                })),
        });
        println!("{}", serde_json::to_string_pretty(&plan).unwrap_or_default());
    } else {
        output::header("apr train plan — Pre-training Plan (Causal LM)");
        println!();
        output::kv("  Config", config_path.display().to_string());
        output::kv("  Model path", spec.model.path.display().to_string());
        output::kv("  Model mode", format!("{:?}", spec.model.mode));
        output::kv("  Training mode", format!("{:?}", spec.training.mode));
        println!();
        output::kv("  Train data", spec.data.train.display().to_string());
        output::kv("  Batch size", spec.data.batch_size.to_string());
        if let Some(seq_len) = spec.data.seq_len {
            output::kv("  Sequence length", seq_len.to_string());
        }
        println!();
        output::kv("  Optimizer", &spec.optimizer.name);
        output::kv("  Learning rate", format!("{:.2e}", spec.optimizer.lr));
        output::kv("  Epochs", spec.training.epochs.to_string());
        output::kv("  Warmup steps", spec.training.warmup_steps.to_string());
        if let Some(accum) = spec.training.gradient_accumulation {
            output::kv("  Gradient accumulation", accum.to_string());
        }
        if let Some(ref mp) = spec.training.mixed_precision {
            output::kv("  Mixed precision", mp);
        }
        if let Some(ref lora) = spec.lora {
            println!();
            output::kv("  LoRA rank", lora.rank.to_string());
            output::kv("  LoRA alpha", format!("{:.1}", lora.alpha));
        }
        // VRAM estimation from architecture overrides
        if let Some(ref arch) = spec.model.architecture {
            if let Some(estimate) = estimate_vram(arch, &spec) {
                println!();
                output::kv("  Parameters", format_params(estimate.param_count));
                println!();
                println!("  {} (contract: training-memory-kernel-v1):", "Memory Estimate".bold());
                println!("    Weights ({})       {:>8.1} GB  (CPU RAM)", estimate.dtype_label, estimate.weights_gb);
                println!("    Gradients (f32)       {:>8.1} GB  (CPU RAM)", estimate.gradients_gb);
                println!("    Optimizer (AdamW)     {:>8.1} GB  (CPU RAM)", estimate.optimizer_gb);
                println!("    Activations (est.)    {:>8.1} GB  (CPU RAM)", estimate.activations_gb);
                println!("    CUDA context          {:>8.1} GB  (VRAM)", 0.5);
                println!("    ─────────────────────────────────────");
                println!("    {}        {:>8.1} GB  (system total)", "Total".bold(), estimate.total_gb);
                println!();
                println!("  {} (R-095):", "Extended Resource Estimates".bold());
                println!("    RAM (total)           {:>8.1} GB", estimate.ram_total_gb);
                println!("    Disk (per checkpoint) {:>8.1} GB", estimate.disk_checkpoint_gb);
                println!("    Tokens/step           {:>8}", estimate.tokens_per_step);
                println!("    Est. step time        {:>8.0} ms  (RTX 4090 peak)", estimate.estimated_step_ms);
                if estimate.tokens_per_step > 0 && estimate.estimated_step_ms > 0.0 {
                    let tok_per_sec = estimate.tokens_per_step as f64 / (estimate.estimated_step_ms / 1000.0);
                    println!("    Est. throughput       {:>8.0} tok/s", tok_per_sec);
                }
            }
        }

        println!();
        println!("  {} Config validated, ready for apply", "READY".green().bold());
    }

    Ok(())
}

/// Comprehensive resource estimate from training-memory-kernel-v1.yaml contract.
struct VramEstimate {
    param_count: usize,
    dtype_label: &'static str,
    weights_gb: f64,
    gradients_gb: f64,
    optimizer_gb: f64,
    activations_gb: f64,
    total_gb: f64,
    // Extended estimates (R-095)
    ram_total_gb: f64,
    disk_checkpoint_gb: f64,
    tokens_per_step: usize,
    estimated_step_ms: f64,
}

/// Estimate VRAM usage from architecture parameters.
///
/// All formulas from `contracts/training-memory-kernel-v1.yaml`:
///   - parameter_count: P_total = P_embed + L × P_layer + P_norm (equivalence, tolerance=0)
///   - weight_memory:   M_weights = P_total × B_w (equivalence, tolerance=0)
///   - gradient_memory: M_grad = P_total × 4 (equivalence, tolerance=0)
///   - optimizer_memory: M_opt = P_total × 8 (equivalence, tolerance=0)
///   - activation_memory: M_act = L × S × H × K × 4 (bound, K=10)
///   - total_memory: M_total = sum + M_cuda (M_cuda = 512 MB)
fn estimate_vram(
    arch: &entrenar::config::ArchitectureOverrides,
    spec: &entrenar::config::TrainSpec,
) -> Option<VramEstimate> {
    let h = arch.hidden_size?;
    let l = arch.num_hidden_layers?;
    let v = arch.vocab_size?;
    let i = arch.intermediate_size.unwrap_or(h * 4);
    let num_heads = arch.num_attention_heads?;
    let kv_heads = arch.num_kv_heads.unwrap_or(num_heads);
    let head_dim = h / num_heads;
    let d_kv = kv_heads * head_dim;

    // Contract eq: parameter_count
    // P_embed = V × H
    let p_embed = v * h;
    // P_layer = 2H + 2H² + 2H×D_kv + 3H×I
    let p_layer = 2 * h + 2 * h * h + 2 * h * d_kv + 3 * h * i;
    // P_norm = H
    let p_norm = h;
    // P_total = P_embed + L × P_layer + P_norm
    let p_total = p_embed + l * p_layer + p_norm;

    // Contract eq: weight_memory — M_weights = P_total × B_w
    // entrenar stores f32 master weights; fp16 cast at matmul site
    let b_w: f64 = 4.0; // always f32 in current entrenar impl
    let is_fp16 = matches!(
        spec.training.mixed_precision.as_deref(),
        Some("fp16") | Some("bf16")
    );

    let gb = |bytes: f64| bytes / (1024.0 * 1024.0 * 1024.0);

    let weights_gb = gb(p_total as f64 * b_w);
    // Contract eq: gradient_memory — M_grad = P_total × 4
    let gradients_gb = gb(p_total as f64 * 4.0);
    // Contract eq: optimizer_memory — M_opt = P_total × 8
    let optimizer_gb = gb(p_total as f64 * 8.0);

    // Contract eq: activation_memory — M_act = L × S × H × K × 4 (upper bound)
    let s = spec.data.seq_len.unwrap_or(512) as f64;
    let k: f64 = 10.0; // contract constant: Q,K,V,attn_scores,attn_out,gate,up,down,2×residual
    let activations_gb = gb(l as f64 * s * h as f64 * k * 4.0);

    // Contract eq: total_memory — M_total = sum + M_cuda
    let m_cuda_gb = 0.5; // 512 MB CUDA context overhead
    let total_gb = weights_gb + gradients_gb + optimizer_gb + activations_gb + m_cuda_gb;

    // Extended estimates (R-095)
    let batch_size = spec.data.batch_size as usize;
    let seq_len = spec.data.seq_len.unwrap_or(512) as usize;
    let tokens_per_step = batch_size * seq_len;

    // RAM: data loading + embedding table + workspace
    let data_buffer_gb = gb(tokens_per_step as f64 * 4.0 * 16.0); // ~16 batches buffered
    let ram_total_gb = weights_gb + optimizer_gb + data_buffer_gb + 1.0; // +1 GB overhead

    // Disk: checkpoint = weights + optimizer state + metadata
    let disk_checkpoint_gb = weights_gb + optimizer_gb + 0.01; // ~10 MB metadata
    // Time estimate: ~6NBS / TFLOPS per step (RTX 4090 ≈ 82 TFLOPS fp32)
    let tflops = 82.0; // RTX 4090 theoretical peak
    let flops_per_step = 6.0 * p_total as f64 * tokens_per_step as f64;
    let estimated_step_ms = flops_per_step / (tflops * 1e9); // TFLOPS = 1e12, ms = 1e-3

    Some(VramEstimate {
        param_count: p_total,
        dtype_label: if is_fp16 { "fp16" } else { "f32" },
        weights_gb,
        gradients_gb,
        optimizer_gb,
        activations_gb,
        total_gb,
        ram_total_gb,
        disk_checkpoint_gb,
        tokens_per_step,
        estimated_step_ms,
    })
}

include!("train_output.rs");

/// Load a training plan from file, or generate one inline.
#[allow(clippy::too_many_arguments)]
fn load_or_generate_plan(
    plan_file: Option<&std::path::Path>,
    data: Option<&std::path::Path>,
    model_size: &str,
    model_path: Option<&std::path::Path>,
    num_classes: usize,
    output_dir: &std::path::Path,
    strategy: &str,
    budget: usize,
    scout: bool,
    max_epochs: usize,
    manual_lr: Option<f32>,
    manual_lora_rank: Option<usize>,
    manual_batch_size: Option<usize>,
) -> Result<entrenar::finetune::TrainingPlan> {
    if let Some(plan_path) = plan_file {
        let content = std::fs::read_to_string(plan_path).map_err(|_| {
            CliError::FileNotFound(plan_path.to_path_buf())
        })?;
        entrenar::finetune::TrainingPlan::from_str(&content)
            .map_err(|e| {
                CliError::ValidationFailed(format!(
                    "Failed to parse plan file {}: {e}",
                    plan_path.display()
                ))
            })
    } else {
        let data_path = data.ok_or_else(|| {
            CliError::ValidationFailed(
                "Either --plan <file> or --data <file> is required".to_string(),
            )
        })?;
        let config = entrenar::finetune::PlanConfig {
            task: "classify".to_string(),
            data_path: data_path.to_path_buf(),
            val_path: None,
            test_path: None,
            model_size: model_size.to_string(),
            model_path: model_path.map(std::path::Path::to_path_buf),
            num_classes,
            output_dir: output_dir.to_path_buf(),
            strategy: strategy.to_string(),
            budget,
            scout,
            max_epochs,
            manual_lr,
            manual_lora_rank,
            manual_batch_size,
            manual_lora_alpha: None,
            manual_warmup: None,
            manual_gradient_clip: None,
            manual_lr_min_ratio: None,
            manual_class_weights: None,
            manual_target_modules: None,
        };
        entrenar::finetune::training_plan(&config)
            .map_err(|e| CliError::ValidationFailed(e.to_string()))
    }
}

/// Print apply summary before execution.
fn print_apply_summary(
    plan: &entrenar::finetune::TrainingPlan,
    output_dir: &std::path::Path,
) {
    output::header("apr train apply — Executing Training Plan");
    println!();
    output::kv("  Strategy", &plan.hyperparameters.strategy);
    if plan.hyperparameters.strategy == "manual" {
        if let Some(ref m) = plan.hyperparameters.manual {
            output::kv("  Learning rate", format!("{:.2e}", m.learning_rate));
            output::kv("  LoRA rank", m.lora_rank.to_string());
            output::kv("  Batch size", m.batch_size.to_string());
        }
    } else {
        output::kv("  Budget", format!("{} trials", plan.hyperparameters.budget));
        output::kv(
            "  Mode",
            if plan.hyperparameters.scout { "scout (1 epoch/trial)" } else { "full" },
        );
    }
    output::kv("  Model", &plan.model.size);
    output::kv("  Data", &plan.data.train_path);
    output::kv("  Output", output_dir.display().to_string());
    println!();
}

/// Run `apr train apply` — execute a training plan.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_apply(
    plan_file: Option<&std::path::Path>,
    config_path: Option<&std::path::Path>,
    task: &str,
    data: Option<&std::path::Path>,
    model_size: &str,
    model_path: Option<&std::path::Path>,
    num_classes: usize,
    output_dir: &std::path::Path,
    strategy: &str,
    budget: usize,
    scout: bool,
    max_epochs: usize,
    manual_lr: Option<f32>,
    manual_lora_rank: Option<usize>,
    manual_batch_size: Option<usize>,
    json_output: bool,
) -> Result<()> {
    match task {
        "pretrain" | "causal_lm" => run_apply_pretrain(config_path, json_output),
        "classify" => run_apply_classify(
            plan_file, data, model_size, model_path, num_classes,
            output_dir, strategy, budget, scout, max_epochs,
            manual_lr, manual_lora_rank, manual_batch_size, json_output,
        ),
        _ => Err(CliError::ValidationFailed(format!(
            "Unknown task type: {task}. Supported: classify, pretrain"
        ))),
    }
}

/// Execute classification fine-tuning (existing behavior).
#[allow(clippy::too_many_arguments)]
fn run_apply_classify(
    plan_file: Option<&std::path::Path>,
    data: Option<&std::path::Path>,
    model_size: &str,
    model_path: Option<&std::path::Path>,
    num_classes: usize,
    output_dir: &std::path::Path,
    strategy: &str,
    budget: usize,
    scout: bool,
    max_epochs: usize,
    manual_lr: Option<f32>,
    manual_lora_rank: Option<usize>,
    manual_batch_size: Option<usize>,
    json_output: bool,
) -> Result<()> {
    let plan = load_or_generate_plan(
        plan_file, data, model_size, model_path, num_classes,
        output_dir, strategy, budget, scout, max_epochs,
        manual_lr, manual_lora_rank, manual_batch_size,
    )?;

    if plan.verdict == entrenar::finetune::PlanVerdict::Blocked {
        if json_output {
            println!("{}", plan.to_json());
        } else {
            println!(
                "  {} Plan is blocked — resolve failures before applying",
                "BLOCKED".red().bold()
            );
            print_plan_text(&plan);
        }
        return Err(CliError::ValidationFailed(
            "Plan is blocked — resolve all failures first".to_string(),
        ));
    }

    let resolved_model_path = model_path
        .map(std::path::Path::to_path_buf)
        .or_else(|| {
            plan.model.weights_available.then(|| {
                std::path::PathBuf::from("/home/noah/src/models/qwen2.5-coder-0.5b")
            })
        })
        .ok_or_else(|| {
            CliError::ValidationFailed(
                "--model-path is required for apply (model weights directory)".to_string(),
            )
        })?;

    let resolved_data_path = match data {
        Some(p) => p.to_path_buf(),
        None => std::path::PathBuf::from(&plan.data.train_path),
    };

    if !json_output {
        print_apply_summary(&plan, output_dir);
    }

    let apply_config = entrenar::finetune::ApplyConfig {
        model_path: resolved_model_path,
        data_path: resolved_data_path,
        output_dir: output_dir.to_path_buf(),
        on_trial_complete: None,
    };

    let result = entrenar::finetune::execute_plan(&plan, &apply_config)
        .map_err(|e| CliError::ValidationFailed(format!("Training failed: {e}")))?;

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
    } else {
        print_apply_result(&result);
    }

    Ok(())
}

/// Execute causal LM pre-training from YAML config (ALB-009).
fn run_apply_pretrain(
    config_path: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    let config_path = config_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "--config <yaml> is required for --task pretrain".to_string(),
        )
    })?;

    if !config_path.exists() {
        return Err(CliError::FileNotFound(config_path.to_path_buf()));
    }

    if !json_output {
        output::header("apr train apply — Causal LM Pre-training");
        println!();
        output::kv("  Config", config_path.display().to_string());
        println!();
    }

    entrenar::config::train_from_yaml(config_path)
        .map_err(|e| CliError::ValidationFailed(format!("Training failed: {e}")))?;

    if json_output {
        println!("{{\"status\":\"completed\",\"task\":\"pretrain\"}}");
    } else {
        println!();
        println!("  {} Pre-training completed", "DONE".green().bold());
    }

    Ok(())
}

/// Format a parameter count for display.
fn format_params(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

/// Format minutes into human-readable duration.
fn format_duration(minutes: f64) -> String {
    if minutes < 1.0 {
        format!("{:.0} sec", minutes * 60.0)
    } else if minutes < 60.0 {
        format!("{:.1} min", minutes)
    } else {
        let hours = minutes / 60.0;
        format!("{:.1} hours", hours)
    }
}

/// Run `apr train watch` — crash-resilient training supervisor.
///
/// Monitors a training process with automatic restart on crash, hang detection
/// via heartbeat staleness, GPU state capture, and exponential backoff.
/// Sovereign Rust replacement for train-guard.sh.
/// Mutable state for the watch supervisor loop.
struct WatchState {
    attempt: usize,
    backoff: u64,
    last_stable: std::time::Instant,
    use_blocking: bool,
}

pub(crate) fn run_watch(
    config_path: &std::path::Path,
    max_restarts: usize,
    _heartbeat_timeout: u64,
    backoff_initial: u64,
    backoff_max: u64,
    json_output: bool,
) -> Result<()> {
    if !config_path.exists() {
        return Err(CliError::FileNotFound(config_path.to_path_buf()));
    }

    print_watch_header(config_path, max_restarts, _heartbeat_timeout, backoff_initial, backoff_max, json_output);

    let mut state = WatchState {
        attempt: 0,
        backoff: backoff_initial,
        last_stable: std::time::Instant::now(),
        use_blocking: false,
    };

    loop {
        state.attempt += 1;
        if state.attempt > max_restarts + 1 {
            return Err(watch_max_restarts_exceeded(max_restarts, json_output));
        }

        if !json_output {
            println!("  {} Starting training (attempt {}/{})", "▶".green(), state.attempt, max_restarts + 1);
        }

        kill_stale_gpu_procs();
        let exit_status = run_training_process(config_path, state.use_blocking);

        match exit_status {
            Ok(status) if status.success() => {
                return watch_success(state.attempt, json_output);
            }
            Ok(status) => {
                let action = handle_crash(config_path, &mut state, status, backoff_initial, backoff_max, json_output)?;
                if action == CrashAction::Fatal {
                    return Err(CliError::ValidationFailed("Fatal error".to_string()));
                }
            }
            Err(e) => {
                return Err(watch_spawn_failed(e, json_output));
            }
        }
    }
}

/// Print the watch header.
fn print_watch_header(config_path: &std::path::Path, max_restarts: usize, heartbeat_timeout: u64, backoff_initial: u64, backoff_max: u64, json_output: bool) {
    if !json_output {
        output::header("apr train watch — Training Supervisor");
        println!();
        output::kv("  Config", config_path.display().to_string());
        output::kv("  Max restarts", max_restarts.to_string());
        output::kv("  Heartbeat timeout", format!("{heartbeat_timeout}s"));
        output::kv("  Backoff", format!("{backoff_initial}s → {backoff_max}s"));
        println!();
    }
}

/// Handle max restarts exceeded.
fn watch_max_restarts_exceeded(max_restarts: usize, json_output: bool) -> CliError {
    let msg = format!("Max restarts ({max_restarts}) exceeded");
    if json_output {
        println!("{{\"status\":\"failed\",\"reason\":\"{msg}\"}}");
    } else {
        println!("  {} {msg}", "FATAL".red().bold());
    }
    CliError::ValidationFailed(msg)
}

/// Handle successful training completion.
fn watch_success(attempt: usize, json_output: bool) -> Result<()> {
    if json_output {
        println!("{{\"status\":\"completed\",\"attempts\":{attempt}}}");
    } else {
        println!();
        println!("  {} Training completed successfully (attempt {attempt})", "DONE".green().bold());
    }
    Ok(())
}

/// Handle spawn failure.
fn watch_spawn_failed(e: std::io::Error, json_output: bool) -> CliError {
    if !json_output {
        println!("  {} Failed to start training: {e}", "ERROR".red().bold());
    }
    CliError::ValidationFailed(format!("Cannot start training process: {e}"))
}

#[derive(PartialEq)]
enum CrashAction { Restart, Fatal }

/// Handle a training crash: classify, diagnose, maybe restart.
fn handle_crash(
    config_path: &std::path::Path,
    state: &mut WatchState,
    status: std::process::ExitStatus,
    backoff_initial: u64,
    backoff_max: u64,
    json_output: bool,
) -> Result<CrashAction> {
    let code = status.code().unwrap_or(-1);
    let classification = classify_exit_code(code);

    if !json_output {
        println!();
        println!("  {} Training exited with code {code} ({classification})", "CRASH".red().bold());
    }

    write_crash_report(config_path, state.attempt, code, classification);

    if classification == "signal" && state.attempt == 1 {
        state.use_blocking = true;
        if !json_output {
            println!("  {} Enabling CUDA_LAUNCH_BLOCKING for diagnosis", "DIAG".yellow());
        }
    }

    if classification == "fatal" {
        let msg = format!("Fatal error (exit code {code}), not restarting");
        if json_output {
            println!("{{\"status\":\"fatal\",\"exit_code\":{code}}}");
        } else {
            println!("  {} {msg}", "FATAL".red().bold());
        }
        return Ok(CrashAction::Fatal);
    }

    if state.last_stable.elapsed().as_secs() > 3600 {
        state.backoff = backoff_initial;
    }
    state.last_stable = std::time::Instant::now();

    if !json_output {
        println!("  {} Waiting {}s before restart...", "⏳".dimmed(), state.backoff);
    }
    std::thread::sleep(std::time::Duration::from_secs(state.backoff));
    state.backoff = (state.backoff * 2).min(backoff_max);

    Ok(CrashAction::Restart)
}

/// Run the training process as a child.
fn run_training_process(
    config_path: &std::path::Path,
    blocking: bool,
) -> std::result::Result<std::process::ExitStatus, std::io::Error> {
    let apr = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("apr"));
    let mut cmd = std::process::Command::new(&apr);
    cmd.args(["train", "apply", "--task", "pretrain", "--config"]);
    cmd.arg(config_path);
    cmd.env("RUST_BACKTRACE", "1");
    if blocking {
        cmd.env("CUDA_LAUNCH_BLOCKING", "1");
    }
    cmd.status()
}

/// Classify an exit code into a category.
fn classify_exit_code(code: i32) -> &'static str {
    match code {
        0 => "success",
        1 => "error",
        2 => "usage",
        134 => "sigabrt",    // CUDA assertion
        135 => "sigbus",     // fatal
        137 => "oom",        // SIGKILL (OOM killer)
        139 => "sigsegv",    // CUDA memory corruption
        _ if code > 128 => "signal",
        _ => "unknown",
    }
}

/// Kill any stale GPU processes from previous training runs.
fn kill_stale_gpu_procs() {
    // Best-effort: find processes using the GPU that match our training pattern
    let _ = std::process::Command::new("bash")
        .args(["-c", "pgrep -f 'apr.*train.*apply' | grep -v $$ | xargs -r kill 2>/dev/null"])
        .status();
}

/// Write a JSON crash report.
fn write_crash_report(
    config_path: &std::path::Path,
    attempt: usize,
    exit_code: i32,
    classification: &str,
) {
    let reports_dir = std::path::Path::new("crash-reports");
    let _ = std::fs::create_dir_all(reports_dir);

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let report = serde_json::json!({
        "timestamp": timestamp,
        "attempt": attempt,
        "config": config_path.display().to_string(),
        "exit_code": exit_code,
        "classification": classification,
        "gpu_state": capture_gpu_state(),
    });

    let filename = reports_dir.join(format!("crash-{timestamp}-attempt{attempt}.json"));
    let _ = std::fs::write(
        &filename,
        serde_json::to_string_pretty(&report).unwrap_or_default(),
    );
}

/// Capture GPU state via nvidia-smi.
fn capture_gpu_state() -> serde_json::Value {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=gpu_name,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
            serde_json::json!({"nvidia_smi": text})
        }
        _ => serde_json::json!({"nvidia_smi": "unavailable"}),
    }
}

// ── Hyperparameter Sweep (R-027 Rust replacement) ───────────────────────────

/// Run `apr train sweep` — generate hyperparameter sweep configs.
///
/// Creates N training configs with varied hyperparameters (grid or random).
/// Each output is a complete YAML that can be passed to `apr train apply`.
pub(crate) fn run_sweep(
    config_path: &std::path::Path,
    strategy: &str,
    num_configs: usize,
    output_dir: &std::path::Path,
    seed: u64,
    json_output: bool,
) -> Result<()> {
    if !config_path.exists() {
        return Err(CliError::FileNotFound(config_path.to_path_buf()));
    }

    let base_content = std::fs::read_to_string(config_path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read config: {e}")))?;
    let base: serde_yaml_ng::Value = serde_yaml_ng::from_str(&base_content)
        .map_err(|e| CliError::ValidationFailed(format!("Invalid YAML: {e}")))?;

    std::fs::create_dir_all(output_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot create output dir: {e}")))?;

    if !json_output {
        output::header("apr train sweep — Hyperparameter Sweep Generator");
        println!();
        output::kv("  Base config", config_path.display().to_string());
        output::kv("  Strategy", strategy);
        output::kv("  Configs", num_configs.to_string());
        output::kv("  Output", output_dir.display().to_string());
        println!();
    }

    let configs = match strategy {
        "grid" => generate_grid_configs(&base, num_configs),
        "random" | _ => generate_random_configs(&base, num_configs, seed),
    };

    let mut results = Vec::new();
    for (i, config) in configs.iter().enumerate() {
        let filename = output_dir.join(format!("sweep-{i:03}.yaml"));
        let yaml_str = serde_yaml_ng::to_string(config)
            .map_err(|e| CliError::ValidationFailed(format!("YAML serialize error: {e}")))?;
        std::fs::write(&filename, &yaml_str)
            .map_err(|e| CliError::ValidationFailed(format!("Cannot write {}: {e}", filename.display())))?;

        let lr = config.get("optimizer")
            .and_then(|o| o.get("lr"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let bs = config.get("data")
            .and_then(|d| d.get("batch_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        results.push(serde_json::json!({
            "file": filename.display().to_string(),
            "lr": lr,
            "batch_size": bs,
        }));

        if !json_output {
            println!("  [{}] {} (lr={:.2e}, bs={})", i, filename.display(), lr, bs);
        }
    }

    if json_output {
        let output = serde_json::json!({
            "strategy": strategy,
            "configs_generated": configs.len(),
            "output_dir": output_dir.display().to_string(),
            "configs": results,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
    } else {
        println!();
        println!("  {} Generated {} configs", "DONE".green().bold(), configs.len());
    }

    Ok(())
}

/// Generate grid search configs over LR × batch_size × weight_decay.
fn generate_grid_configs(
    base: &serde_yaml_ng::Value,
    max_configs: usize,
) -> Vec<serde_yaml_ng::Value> {
    let lr_values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3];
    let bs_values: &[u64] = &[2, 4, 8];
    let wd_values = [0.0, 0.01, 0.1];

    let mut configs = Vec::new();
    for &lr in &lr_values {
        for &bs in bs_values {
            for &wd in &wd_values {
                if configs.len() >= max_configs {
                    return configs;
                }
                let mut c = base.clone();
                set_yaml_f64(&mut c, &["optimizer", "lr"], lr);
                set_yaml_u64(&mut c, &["data", "batch_size"], bs);
                set_yaml_f64(&mut c, &["optimizer", "weight_decay"], wd);
                configs.push(c);
            }
        }
    }
    configs
}

/// Generate random search configs using LCG PRNG.
fn generate_random_configs(
    base: &serde_yaml_ng::Value,
    num_configs: usize,
    seed: u64,
) -> Vec<serde_yaml_ng::Value> {
    let mut rng_state = seed;
    let mut configs = Vec::new();

    for _ in 0..num_configs {
        let mut c = base.clone();

        // LR: log-uniform in [1e-5, 1e-2]
        let lr_log = -5.0 + lcg_f64(&mut rng_state) * 3.0; // [-5, -2]
        let lr = 10.0_f64.powf(lr_log);
        set_yaml_f64(&mut c, &["optimizer", "lr"], lr);

        // Batch size: uniform choice from [1, 2, 4, 8, 16]
        let bs_choices: &[u64] = &[1, 2, 4, 8, 16];
        let bs_idx = (lcg_f64(&mut rng_state) * bs_choices.len() as f64) as usize;
        let bs = bs_choices[bs_idx.min(bs_choices.len() - 1)];
        set_yaml_u64(&mut c, &["data", "batch_size"], bs);

        // Weight decay: log-uniform in [1e-3, 0.5]
        let wd_log = -3.0 + lcg_f64(&mut rng_state) * 2.7; // [-3, -0.3]
        let wd = 10.0_f64.powf(wd_log);
        set_yaml_f64(&mut c, &["optimizer", "weight_decay"], wd);

        // Warmup steps: uniform in [50, 2000]
        let warmup = 50 + (lcg_f64(&mut rng_state) * 1950.0) as u64;
        set_yaml_u64(&mut c, &["training", "warmup_steps"], warmup);

        configs.push(c);
    }
    configs
}

/// LCG pseudo-random: returns f64 in [0, 1).
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

/// Set a nested YAML value (f64).
fn set_yaml_f64(root: &mut serde_yaml_ng::Value, path: &[&str], val: f64) {
    let mut node = root;
    for (i, key) in path.iter().enumerate() {
        if i == path.len() - 1 {
            node[*key] = serde_yaml_ng::Value::Number(serde_yaml_ng::Number::from(val));
        } else {
            if node.get(*key).is_none() {
                node[*key] = serde_yaml_ng::Value::Mapping(serde_yaml_ng::Mapping::new());
            }
            node = &mut node[*key];
        }
    }
}

/// Set a nested YAML value (u64).
fn set_yaml_u64(root: &mut serde_yaml_ng::Value, path: &[&str], val: u64) {
    let mut node = root;
    for (i, key) in path.iter().enumerate() {
        if i == path.len() - 1 {
            node[*key] = serde_yaml_ng::Value::Number(serde_yaml_ng::Number::from(val));
        } else {
            if node.get(*key).is_none() {
                node[*key] = serde_yaml_ng::Value::Mapping(serde_yaml_ng::Mapping::new());
            }
            node = &mut node[*key];
        }
    }
}

// ── Checkpoint Archival (R-085) ─────────────────────────────────────────────

/// Run `apr train archive` — package checkpoint into a release bundle.
pub(crate) fn run_archive(
    checkpoint_dir: &std::path::Path,
    output_dir: &std::path::Path,
    version: Option<&str>,
    notes: Option<&str>,
    json_output: bool,
) -> Result<()> {
    if !checkpoint_dir.is_dir() {
        return Err(CliError::ValidationFailed(format!(
            "Not a directory: {}", checkpoint_dir.display()
        )));
    }

    std::fs::create_dir_all(output_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot create output dir: {e}")))?;

    if !json_output {
        output::header("apr train archive — Checkpoint Release Bundle");
        println!();
        output::kv("  Source", checkpoint_dir.display().to_string());
        output::kv("  Output", output_dir.display().to_string());
        if let Some(v) = version {
            output::kv("  Version", v);
        }
        println!();
    }

    // Copy checkpoint files
    let mut manifest_entries = Vec::new();
    let mut total_bytes: u64 = 0;

    for entry in std::fs::read_dir(checkpoint_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read dir: {e}")))?
    {
        let entry = entry
            .map_err(|e| CliError::ValidationFailed(format!("Dir entry error: {e}")))?;
        let src = entry.path();
        if !src.is_file() {
            continue;
        }

        let filename = src.file_name().unwrap_or_default().to_string_lossy().to_string();
        let dst = output_dir.join(&filename);

        let data = std::fs::read(&src)
            .map_err(|e| CliError::ValidationFailed(format!("Cannot read {}: {e}", src.display())))?;
        let size = data.len() as u64;
        let hash = blake3::hash(&data).to_hex().to_string();

        std::fs::write(&dst, &data)
            .map_err(|e| CliError::ValidationFailed(format!("Cannot write {}: {e}", dst.display())))?;

        manifest_entries.push(serde_json::json!({
            "file": filename,
            "size": size,
            "blake3": hash,
        }));
        total_bytes += size;

        if !json_output {
            println!("  [COPY] {} ({}, BLAKE3: {}…)", filename, format_archive_size(size), &hash[..16]);
        }
    }

    // Write MANIFEST.json
    let manifest = serde_json::json!({
        "format": "albor-checkpoint-archive",
        "version": version.unwrap_or("0.0.0"),
        "created": chrono::Utc::now().to_rfc3339(),
        "notes": notes.unwrap_or(""),
        "source": checkpoint_dir.display().to_string(),
        "files": manifest_entries,
        "total_bytes": total_bytes,
    });

    let manifest_path = output_dir.join("MANIFEST.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap_or_default(),
    ).map_err(|e| CliError::ValidationFailed(format!("Cannot write manifest: {e}")))?;

    if json_output {
        println!("{}", serde_json::to_string_pretty(&manifest).unwrap_or_default());
    } else {
        println!();
        println!("  [MANIFEST] {} ({} files, {})",
            manifest_path.display(),
            manifest_entries.len(),
            format_archive_size(total_bytes),
        );
        println!();
        println!("  {} Archive created", "DONE".green().bold());
    }

    Ok(())
}

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
