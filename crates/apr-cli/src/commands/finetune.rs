//! Fine-tuning command implementation (GH-244)
//!
//! Surfaces entrenar's LoRA/QLoRA fine-tuning pipeline through the apr CLI.
//! Supports planning mode (VRAM estimation) and training execution.
//!
//! # Example
//!
//! ```bash
//! apr finetune model.apr --method lora --data train.jsonl -o adapter/
//! apr finetune model.apr --method qlora --rank 16 --plan --json
//! apr finetune merge model.apr --adapter adapter/ -o merged.apr
//! ```

use crate::error::{CliError, Result};
use crate::output;
use colored::Colorize;
use entrenar_lora::{plan, MemoryPlanner, MemoryRequirement, MergeEngine, Method, OptimalConfig};
use std::path::Path;

/// Fine-tuning method selection
#[derive(Debug, Clone, Copy, Default)]
pub enum FinetuneMethod {
    #[default]
    Auto,
    Full,
    LoRA,
    QLoRA,
}

impl std::str::FromStr for FinetuneMethod {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "full" => Ok(Self::Full),
            "lora" => Ok(Self::LoRA),
            "qlora" => Ok(Self::QLoRA),
            _ => Err(format!(
                "Unknown fine-tuning method: {s}. Use: auto, full, lora, qlora"
            )),
        }
    }
}

impl From<FinetuneMethod> for Method {
    fn from(m: FinetuneMethod) -> Self {
        match m {
            FinetuneMethod::Auto => Method::Auto,
            FinetuneMethod::Full => Method::Full,
            FinetuneMethod::LoRA => Method::LoRA,
            FinetuneMethod::QLoRA => Method::QLoRA,
        }
    }
}

/// Resolve model parameters from either --model-size flag or file inspection.
fn resolve_model_params(model_size: Option<&str>, model_path: Option<&Path>) -> Result<u64> {
    if let Some(size) = model_size {
        parse_model_size(size)
    } else if let Some(path) = model_path {
        estimate_params_from_file(path)
    } else {
        Err(CliError::ValidationFailed(
            "Either model path or --model-size required".to_string(),
        ))
    }
}

/// Display plan configuration as JSON.
#[allow(clippy::disallowed_methods)]
fn display_plan_json(
    config: &OptimalConfig,
    req: &MemoryRequirement,
    model_params: u64,
    vram_gb: f64,
    epochs: u32,
    learning_rate: f64,
    plan_only: bool,
) {
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
        "epochs": epochs,
        "learning_rate": learning_rate,
        "plan_only": plan_only,
        "memory_breakdown": {
            "model_bytes": req.model_bytes,
            "adapter_bytes": req.adapter_bytes,
            "optimizer_bytes": req.optimizer_bytes,
            "activation_bytes": req.activation_bytes,
            "total_bytes": req.total_bytes,
            "savings_percent": req.savings_percent,
        },
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json).unwrap_or_default()
    );
}

/// Display plan configuration as human-readable text.
fn display_plan_text(config: &OptimalConfig, req: &MemoryRequirement, vram_gb: f64) {
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

    display_memory_breakdown(req, vram_gb);
}

/// Display memory breakdown table with feasibility check.
fn display_memory_breakdown(req: &MemoryRequirement, vram_gb: f64) {
    println!("{}", "MEMORY BREAKDOWN".white().bold());
    println!("{}", "─".repeat(50));

    let model_gb = req.model_bytes as f64 / 1e9;
    let adapter_gb = req.adapter_bytes as f64 / 1e9;
    let optimizer_gb = req.optimizer_bytes as f64 / 1e9;
    let activation_gb = req.activation_bytes as f64 / 1e9;
    let total_gb = req.total_bytes as f64 / 1e9;

    println!("  Base model:       {model_gb:.2} GB");
    println!("  Adapter:          {adapter_gb:.2} GB");
    println!("  Optimizer states: {optimizer_gb:.2} GB");
    println!("  Activations:      {activation_gb:.2} GB");
    println!("{}", "─".repeat(50));
    println!("  {}:            {total_gb:.2} GB", "TOTAL".bold());
    println!(
        "  Savings:          {:.0}% vs full fine-tuning",
        req.savings_percent
    );
    println!();

    if total_gb <= vram_gb {
        println!(
            "{} Configuration fits in {vram_gb:.1} GB VRAM",
            "✓".green().bold(),
        );
    } else {
        println!(
            "{} Configuration requires {total_gb:.2} GB but only {vram_gb:.1} GB available",
            "⚠".yellow().bold(),
        );
        println!();
        println!("  Suggestions:");
        println!("    - Use QLoRA (4-bit quantization)");
        println!("    - Reduce rank (--rank 4)");
        println!("    - Use gradient checkpointing");
    }
}

/// Execute LoRA adapter creation from model tensors.
fn execute_training(
    model_path: &Path,
    config: &OptimalConfig,
    data_path: &Path,
    output_path: &Path,
    epochs: u32,
    learning_rate: f64,
    json_output: bool,
) -> Result<()> {
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let report = rosetta
        .inspect(model_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model: {e}")))?;

    let lora_targets: Vec<_> = report
        .tensors
        .iter()
        .filter(|t| t.shape.len() == 2 && is_lora_eligible(&t.name))
        .collect();

    if lora_targets.is_empty() {
        return Err(CliError::ValidationFailed(
            "No LoRA-eligible layers found in model".to_string(),
        ));
    }

    let lora_rank = config.rank;
    let lora_alpha = config.alpha;

    if !json_output {
        println!();
        output::pipeline_stage("Creating adapters", output::StageStatus::Running);
        println!("  LoRA targets: {} layers", lora_targets.len());
        println!("  Rank: {lora_rank}, Alpha: {lora_alpha:.1}");
    }

    let mut writer = aprender::serialization::apr::AprWriter::new();
    write_adapter_metadata(
        &mut writer,
        model_path,
        config,
        epochs,
        learning_rate,
        Some(data_path),
    );

    let (adapter_count, total_adapter_params) =
        create_lora_tensors(&mut writer, &lora_targets, lora_rank as usize);

    let bytes = writer
        .to_bytes()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to serialize adapters: {e}")))?;
    std::fs::write(output_path, &bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write adapter: {e}")))?;

    display_adapter_result(
        adapter_count,
        total_adapter_params,
        bytes.len() as u64,
        output_path,
        config,
        json_output,
    );
    Ok(())
}

/// Write LoRA adapter metadata to APR writer.
#[allow(clippy::disallowed_methods)]
fn write_adapter_metadata(
    writer: &mut aprender::serialization::apr::AprWriter,
    model_path: &Path,
    config: &OptimalConfig,
    epochs: u32,
    learning_rate: f64,
    data_path: Option<&Path>,
) {
    writer.set_metadata("adapter_type", serde_json::json!("lora"));
    writer.set_metadata("lora_rank", serde_json::json!(config.rank));
    writer.set_metadata("lora_alpha", serde_json::json!(config.alpha));
    writer.set_metadata("method", serde_json::json!(format!("{:?}", config.method)));
    writer.set_metadata(
        "source_model",
        serde_json::json!(model_path.display().to_string()),
    );
    writer.set_metadata("epochs", serde_json::json!(epochs));
    writer.set_metadata("learning_rate", serde_json::json!(learning_rate));
    if let Some(dp) = data_path {
        writer.set_metadata("data_path", serde_json::json!(dp.display().to_string()));
    }
}

/// Create LoRA A/B tensor pairs for all eligible layers.
fn create_lora_tensors(
    writer: &mut aprender::serialization::apr::AprWriter,
    lora_targets: &[&aprender::format::rosetta::TensorInfo],
    rank: usize,
) -> (u64, u64) {
    let mut adapter_count = 0u64;
    let mut total_adapter_params = 0u64;

    for ti in lora_targets {
        let rows = ti.shape[0];
        let cols = ti.shape[1];

        let bound = 1.0 / (cols as f32).sqrt();
        let a_data: Vec<f32> = (0..rank * cols)
            .map(|i| {
                let seed = hash_seed(&ti.name, i);
                (seed % 1000) as f32 / 1000.0 * 2.0 * bound - bound
            })
            .collect();
        writer.add_tensor_f32(format!("{}.lora_a", ti.name), vec![rank, cols], &a_data);

        let b_data = vec![0.0f32; rows * rank];
        writer.add_tensor_f32(format!("{}.lora_b", ti.name), vec![rows, rank], &b_data);

        adapter_count += 1;
        total_adapter_params += (rank * cols + rows * rank) as u64;
    }

    (adapter_count, total_adapter_params)
}

/// Display adapter creation results.
#[allow(clippy::disallowed_methods)]
fn display_adapter_result(
    adapter_count: u64,
    total_adapter_params: u64,
    output_size: u64,
    output_path: &Path,
    config: &OptimalConfig,
    json_output: bool,
) {
    if json_output {
        let json = serde_json::json!({
            "status": "adapter_created",
            "adapter_layers": adapter_count,
            "adapter_params": total_adapter_params,
            "output_size": output_size,
            "output": output_path.display().to_string(),
            "rank": config.rank,
            "alpha": config.alpha,
            "method": format!("{:?}", config.method),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::pipeline_stage("Creating adapters", output::StageStatus::Done);
        println!();
        output::subheader("Adapter Created");
        println!(
            "{}",
            output::kv_table(&[
                ("Layers adapted", adapter_count.to_string()),
                ("Adapter params", format_params(total_adapter_params)),
                (
                    "Output size",
                    humansize::format_size(output_size, humansize::BINARY)
                ),
                ("Output", output_path.display().to_string()),
            ])
        );
    }
}

/// Run the finetune command
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
pub(crate) fn run(
    model_path: Option<&Path>,
    method: &str,
    rank: Option<u32>,
    vram_gb: f64,
    plan_only: bool,
    data_path: Option<&Path>,
    output_path: Option<&Path>,
    adapter_path: Option<&Path>,
    merge_mode: bool,
    epochs: u32,
    learning_rate: f64,
    model_size: Option<&str>,
    task: Option<&str>,
    num_classes: usize,
    checkpoint_format: &str,
    oversample: bool,
    max_seq_len: Option<usize>,
    quantize_nf4: bool,
    gpus: Option<&str>,
    gpu_backend: &str,
    role: Option<&str>,
    bind: Option<&str>,
    coordinator: Option<&str>,
    expect_workers: Option<usize>,
    json_output: bool,
) -> Result<()> {
    if merge_mode {
        return run_merge(model_path, adapter_path, output_path, json_output);
    }

    // Dispatch to classification pipeline when --task classify
    if let Some("classify") = task {
        return run_classify(
            model_path,
            model_size,
            data_path,
            output_path,
            num_classes,
            rank.unwrap_or(16),
            epochs,
            learning_rate,
            plan_only,
            checkpoint_format,
            oversample,
            max_seq_len,
            quantize_nf4,
            gpus,
            gpu_backend,
            role,
            bind,
            coordinator,
            expect_workers,
            json_output,
        );
    }

    // Dispatch to instruction fine-tuning pipeline when --task instruct (GH-371)
    if let Some("instruct") = task {
        return run_instruct(
            model_path,
            model_size,
            data_path,
            output_path,
            rank.unwrap_or(16),
            epochs,
            learning_rate,
            plan_only,
            json_output,
            method,
            quantize_nf4,
            max_seq_len,
            vram_gb,
        );
    }

    let ft_method: FinetuneMethod = method.parse().map_err(CliError::ValidationFailed)?;

    if !json_output {
        output::section("apr finetune (GH-244: LoRA/QLoRA Fine-tuning)");
        println!();
    }

    let model_params = resolve_model_params(model_size, model_path)?;
    display_run_header(
        ft_method,
        model_params,
        vram_gb,
        rank,
        epochs,
        learning_rate,
        json_output,
    );

    let config = plan(model_params, vram_gb, ft_method.into())
        .map_err(|e| CliError::ValidationFailed(format!("Failed to plan config: {e}")))?;

    let planner = MemoryPlanner::new(model_params);
    let req = planner.estimate(config.method, config.rank);

    if json_output {
        display_plan_json(
            &config,
            &req,
            model_params,
            vram_gb,
            epochs,
            learning_rate,
            plan_only,
        );
    } else {
        display_plan_text(&config, &req, vram_gb);
    }

    if plan_only {
        return Ok(());
    }

    let Some(data) = data_path else {
        display_next_steps(json_output);
        return Ok(());
    };
    if !data.exists() {
        return Err(CliError::FileNotFound(data.to_path_buf()));
    }

    if !json_output {
        println!();
        output::pipeline_stage("Training", output::StageStatus::Running);
        println!("  Data: {}", data.display());
        println!("  Epochs: {epochs}");
        println!("  Learning rate: {learning_rate:.1e}");
    }

    let mp = model_path.ok_or_else(|| {
        CliError::ValidationFailed("Model path required for training".to_string())
    })?;
    if !mp.exists() {
        return Err(CliError::FileNotFound(mp.to_path_buf()));
    }

    let out = output_path.unwrap_or(Path::new("adapter.apr"));
    execute_training(mp, &config, data, out, epochs, learning_rate, json_output)
}

/// Display run header with model info.
fn display_run_header(
    ft_method: FinetuneMethod,
    model_params: u64,
    vram_gb: f64,
    rank: Option<u32>,
    epochs: u32,
    learning_rate: f64,
    json_output: bool,
) {
    if !json_output {
        output::kv("Model parameters", format_params(model_params));
        output::kv("Available VRAM", format!("{vram_gb:.1} GB"));
        output::kv("Method", format!("{ft_method:?}"));
        if let Some(r) = rank {
            output::kv("Requested rank", r.to_string());
        }
        output::kv("Epochs", epochs.to_string());
        output::kv("Learning rate", format!("{learning_rate:.1e}"));
        println!();
    }
}

include!("finetune_display_next_validate.rs");

// =============================================================================
// Distributed training config builder
// =============================================================================

/// Build distributed config from CLI flags, if `--role` is specified.
///
/// Returns `None` — distributed training requires unpublished entrenar APIs
/// (DistributedConfig, NodeRole). Stubbed until entrenar publishes the
/// distributed training subsystem.
fn build_distributed_config(
    role: Option<&str>,
    _bind: Option<&str>,
    _coordinator: Option<&str>,
    _expect_workers: Option<usize>,
) -> Result<()> {
    if role.is_some() {
        return Err(CliError::ValidationFailed(
            "Distributed training (--role) requires unreleased entrenar APIs. \
             Use single-machine training for now."
                .to_string(),
        ));
    }
    Ok(())
}

// =============================================================================
/// Print corpus stats (extracted to reduce cognitive complexity in run_classify).
fn print_corpus_stats(stats: &entrenar::finetune::SafetyCorpusStats) {
    output::subheader("Corpus");
    output::kv("Samples", stats.total.to_string());
    output::kv("Avg input length", format!("{} chars", stats.avg_input_len));
    for (i, count) in stats.class_counts.iter().enumerate() {
        output::kv(&format!("  Class {i}"), count.to_string());
    }
    println!();
}

// Classification fine-tuning (--task classify)
// =============================================================================

/// Build ClassifyConfig from CLI args, applying QLoRA defaults when appropriate.
///
/// Contract: provable-contracts/contracts/entrenar/qlora-hyperparameters-v1.yaml
fn build_classify_config(
    _model_config: &entrenar::transformer::TransformerConfig,
    num_classes: usize,
    rank: u32,
    epochs: u32,
    learning_rate: f64,
    max_seq_len: Option<usize>,
    quantize_nf4: bool,
) -> entrenar::finetune::classify_pipeline::ClassifyConfig {
    use entrenar::finetune::classify_pipeline::ClassifyConfig;

    if quantize_nf4 {
        eprintln!(
            "[warn] --quantize-nf4 requested but ClassifyConfig.quantize_nf4 is not \
             available in entrenar 0.7.5. Proceeding without NF4 quantization."
        );
    }

    let classify_config = ClassifyConfig {
        num_classes,
        lora_rank: rank as usize,
        lora_alpha: rank as f32,
        learning_rate: learning_rate as f32,
        epochs: epochs as usize,
        max_seq_len: max_seq_len.unwrap_or(ClassifyConfig::default().max_seq_len),
        ..ClassifyConfig::default()
    };

    classify_config
}

/// Display classify header info (model, config, GPU settings).
#[allow(clippy::too_many_arguments)]
fn display_classify_header(
    model_config: &entrenar::transformer::TransformerConfig,
    classify_config: &entrenar::finetune::classify_pipeline::ClassifyConfig,
    num_classes: usize,
    rank: u32,
    epochs: u32,
    learning_rate: f64,
    checkpoint_format: &str,
    gpu_backend: &str,
    gpus: Option<&str>,
) {
    output::kv(
        "Model",
        format!("{}h x {}L", model_config.hidden_size, model_config.num_hidden_layers),
    );
    output::kv("Classes", num_classes.to_string());
    output::kv("LoRA rank", rank.to_string());
    output::kv("Epochs", epochs.to_string());
    output::kv("Learning rate", format!("{learning_rate:.1e}"));
    output::kv("Max seq len", classify_config.max_seq_len.to_string());
    output::kv("Checkpoint format", checkpoint_format);
    output::kv("GPU backend", gpu_backend);
    if let Some(g) = gpus {
        output::kv("GPU indices", g);
    }
    println!();
}

/// Display distributed training config info.
///
/// Stubbed — distributed training requires unpublished entrenar APIs.
fn display_distributed_info() {
    // No-op: distributed training not yet available in published entrenar.
}

/// Display GPU/device info in non-JSON output.
fn display_device_info(gpu_info: &Option<(String, usize)>, gpu_backend: &str) {
    if let Some((ref name, _)) = gpu_info {
        output::kv("Device", format!("CUDA ({name})"));
    } else {
        let device_str = match gpu_backend {
            "wgpu" => "wgpu (GPU)".to_string(),
            "auto" => {
                if gpu_info.is_some() { "CUDA".to_string() }
                else { "CPU".to_string() }
            }
            _ => "CPU".to_string(),
        };
        output::kv("Device", device_str);
    }
}

/// Run classification fine-tuning pipeline via entrenar.
///
/// Creates a ClassifyPipeline, loads the corpus, and runs the full training
/// loop via ClassifyTrainer with epoch management, validation, LR scheduling,
/// checkpointing, and early stopping.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
fn run_classify(
    model_path: Option<&Path>,
    model_size: Option<&str>,
    data_path: Option<&Path>,
    output_path: Option<&Path>,
    num_classes: usize,
    rank: u32,
    epochs: u32,
    learning_rate: f64,
    plan_only: bool,
    checkpoint_format: &str,
    oversample: bool,
    max_seq_len: Option<usize>,
    quantize_nf4: bool,
    gpus: Option<&str>,
    gpu_backend: &str,
    role: Option<&str>,
    bind: Option<&str>,
    coordinator: Option<&str>,
    expect_workers: Option<usize>,
    json_output: bool,
) -> Result<()> {
    use entrenar::finetune::{ClassifyTrainer, TrainingConfig};

    if !json_output {
        output::section("apr finetune --task classify (Shell Safety Classification)");
        println!();
    }

    // GH-377: Read architecture from .apr metadata or --model-size fallback
    let model_config = super::model_config::resolve_transformer_config(model_path, model_size)?;

    let classify_config = build_classify_config(
        &model_config, num_classes, rank, epochs, learning_rate, max_seq_len, quantize_nf4,
    );

    if !json_output {
        display_classify_header(
            &model_config, &classify_config, num_classes, rank, epochs,
            learning_rate, checkpoint_format, gpu_backend, gpus,
        );
    }

    // Validate no distributed flags were passed (unsupported in entrenar 0.7.5)
    build_distributed_config(role, bind, coordinator, expect_workers)?;

    if !json_output {
        display_distributed_info();
    }

    let pipeline = load_classify_pipeline(model_path, &model_config, classify_config)?;

    // Capture GPU info before pipeline is moved into trainer
    let gpu_info: Option<(String, usize)> = pipeline.gpu_name().zip(pipeline.gpu_total_memory());

    if !json_output {
        display_device_info(&gpu_info, gpu_backend);
        println!("{}", pipeline.summary());
        println!();
    }

    if plan_only {
        display_classify_plan(&pipeline, &model_config, num_classes, rank, json_output);
        return Ok(());
    }

    let Some(data) = data_path else {
        display_classify_next_steps(json_output);
        return Ok(());
    };

    if !data.exists() {
        return Err(CliError::FileNotFound(data.to_path_buf()));
    }

    // Load corpus
    let samples = pipeline
        .load_corpus(data)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load corpus: {e}")))?;

    let stats = entrenar::finetune::corpus_stats(&samples, num_classes);

    if !json_output {
        print_corpus_stats(&stats);
    }

    // Resolve output directory for checkpoints
    let output_dir = output_path
        .unwrap_or(Path::new("checkpoints"))
        .to_path_buf();

    if oversample {
        eprintln!(
            "[warn] --oversample requested but TrainingConfig.oversample_minority is not \
             available in entrenar 0.7.5. Proceeding without oversampling."
        );
    }

    // Create TrainingConfig from CLI args
    let training_config = TrainingConfig {
        epochs: epochs as usize,
        val_split: 0.2,
        save_every: 5,
        early_stopping_patience: 10,
        checkpoint_dir: output_dir.clone(),
        seed: 42,
        log_interval: 1,
        ..TrainingConfig::default()
    };

    // Create trainer
    let mut trainer = ClassifyTrainer::new(pipeline, samples, training_config)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create trainer: {e}")))?;

    // Attach monitor writer for live TUI updates
    let model_name = model_size.unwrap_or("tiny");
    let experiment_id = format!(
        "classify-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let mut writer =
        entrenar::monitor::tui::TrainingStateWriter::new(&output_dir, &experiment_id, model_name);
    // Wire GPU telemetry into training state for `apr monitor`
    if let Some((ref name, mem)) = gpu_info {
        writer.set_gpu(name, mem as f32 / 1e9);
    }
    trainer.set_monitor_writer(writer);

    if !json_output {
        output::pipeline_stage("Training", output::StageStatus::Running);
        println!("  Output dir: {}", output_dir.display());
        println!("  Monitor:    apr monitor {}", output_dir.display());
        println!();
    }

    // Run training (single-machine mode; distributed requires unreleased entrenar APIs)
    let result = trainer.train();

    if !json_output {
        output::pipeline_stage("Training", output::StageStatus::Done);
        println!();
    }

    // Display results
    display_train_result(&result, &output_dir, checkpoint_format, json_output);

    Ok(())
}

/// Display training results: per-epoch metrics table, best epoch, and summary.
#[allow(clippy::disallowed_methods)]
fn display_train_result(
    result: &entrenar::finetune::TrainResult,
    output_dir: &Path,
    checkpoint_format: &str,
    json_output: bool,
) {
    if json_output {
        let epochs_json: Vec<serde_json::Value> = result
            .epoch_metrics
            .iter()
            .map(|m| {
                serde_json::json!({
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "train_accuracy": m.train_accuracy,
                    "val_loss": m.val_loss,
                    "val_accuracy": m.val_accuracy,
                    "learning_rate": m.learning_rate,
                    "epoch_time_ms": m.epoch_time_ms,
                    "samples_per_sec": m.samples_per_sec,
                })
            })
            .collect();

        let json = serde_json::json!({
            "status": "training_complete",
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "stopped_early": result.stopped_early,
            "total_time_ms": result.total_time_ms,
            "total_epochs": result.epoch_metrics.len(),
            "checkpoint_dir": output_dir.display().to_string(),
            "checkpoint_format": checkpoint_format,
            "monitor": format!("apr monitor {}", output_dir.display()),
            "training_state": output_dir.join("training_state.json").display().to_string(),
            "epoch_metrics": epochs_json,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
        return;
    }

    // Per-epoch metrics table
    output::subheader("Training Metrics");
    println!();
    println!(
        "  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}",
        "Epoch".white().bold(),
        "Train Loss".white().bold(),
        "Val Loss".white().bold(),
        "Train Acc".white().bold(),
        "Val Acc".white().bold(),
        "LR".white().bold(),
        "Time".white().bold(),
    );
    println!("  {}", "\u{2500}".repeat(72));

    for m in &result.epoch_metrics {
        let is_best = m.epoch == result.best_epoch;
        let marker = if is_best { "*" } else { " " };
        println!(
            " {}{:>4}  {:>10.4}  {:>10.4}  {:>9.1}%  {:>9.1}%  {:>10.2e}  {:>6}ms",
            marker,
            m.epoch + 1,
            m.train_loss,
            m.val_loss,
            m.train_accuracy * 100.0,
            m.val_accuracy * 100.0,
            m.learning_rate,
            m.epoch_time_ms,
        );
    }
    println!();

    // Summary
    output::subheader("Summary");
    let total_secs = result.total_time_ms as f64 / 1000.0;
    output::kv("Total epochs", result.epoch_metrics.len().to_string());
    output::kv(
        "Best epoch",
        format!(
            "{} (val_loss: {:.4})",
            result.best_epoch + 1,
            result.best_val_loss
        ),
    );
    if result.stopped_early {
        output::kv(
            "Early stopping",
            "Yes (patience exhausted)".yellow().to_string(),
        );
    } else {
        output::kv("Early stopping", "No (completed all epochs)");
    }
    output::kv("Total time", format!("{total_secs:.1}s"));
    output::kv("Checkpoints", output_dir.display().to_string());
    output::kv("Format", checkpoint_format);

    // Show final accuracy prominently
    if let Some(best) = result
        .epoch_metrics
        .iter()
        .find(|m| m.epoch == result.best_epoch)
    {
        println!();
        println!(
            "  {} Best validation accuracy: {:.1}%",
            "\u{2713}".green().bold(),
            best.val_accuracy * 100.0,
        );
    }
}

// =============================================================================
// Instruction fine-tuning (--task instruct) (GH-371)
// =============================================================================

// GH-376/GH-377: resolve_transformer_config and read_apr_architecture
// moved to shared module: super::model_config

/// Run instruction fine-tuning pipeline via entrenar.
///
/// Stubbed — the instruct fine-tuning subsystem (instruct_corpus, instruct_pipeline,
/// instruct_trainer, InstructTrainResult) is not published in entrenar 0.7.5.
/// Returns a clear error directing users to wait for the next entrenar release.
#[allow(clippy::too_many_arguments)]
fn run_instruct(
    _model_path: Option<&Path>,
    _model_size: Option<&str>,
    _data_path: Option<&Path>,
    _output_path: Option<&Path>,
    _rank: u32,
    _epochs: u32,
    _learning_rate: f64,
    _plan_only: bool,
    _json_output: bool,
    _method: &str,
    _quantize_nf4: bool,
    _max_seq_len: Option<usize>,
    _vram_gb: f64,
) -> Result<()> {
    Err(CliError::ValidationFailed(
        "Instruction fine-tuning (--task instruct) requires unreleased entrenar APIs \
         (instruct_corpus, instruct_pipeline, instruct_trainer). \
         This feature will be available once entrenar publishes the instruct subsystem. \
         Use --task classify for classification fine-tuning."
            .to_string(),
    ))
}

/// Load classify pipeline from model path (directory or new).
///
/// Note: `ClassifyPipeline::from_apr()` and `set_model_path()` are not available
/// in entrenar 0.7.5. `.apr` file loading falls back to `from_pretrained` with
/// the parent directory, and bare `new()` is used without model path injection.
fn load_classify_pipeline(
    model_path: Option<&Path>,
    model_config: &entrenar::transformer::TransformerConfig,
    config: entrenar::finetune::classify_pipeline::ClassifyConfig,
) -> Result<entrenar::finetune::classify_pipeline::ClassifyPipeline> {
    if let Some(mp) = model_path.filter(|p| p.is_dir()) {
        entrenar::finetune::classify_pipeline::ClassifyPipeline::from_pretrained(
            mp,
            model_config,
            config,
        )
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load pretrained model: {e}")))
    } else if let Some(mp) = model_path.filter(|p| p.is_file()) {
        // from_apr() not available in entrenar 0.7.5; use parent dir with from_pretrained
        let parent = mp.parent().unwrap_or(mp);
        entrenar::finetune::classify_pipeline::ClassifyPipeline::from_pretrained(
            parent,
            model_config,
            config,
        )
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR model: {e}")))
    } else {
        Ok(entrenar::finetune::classify_pipeline::ClassifyPipeline::new(model_config, config))
    }
}

/// Display plan-only output for classify fine-tuning.
fn display_classify_plan(
    pipeline: &entrenar::finetune::classify_pipeline::ClassifyPipeline,
    model_config: &entrenar::transformer::TransformerConfig,
    num_classes: usize,
    rank: u32,
    json_output: bool,
) {
    if json_output {
        let json = serde_json::json!({
            "task": "classify",
            "num_classes": num_classes,
            "lora_rank": rank,
            "hidden_size": model_config.hidden_size,
            "num_layers": model_config.num_hidden_layers,
            "trainable_params": pipeline.num_trainable_parameters(),
            "lora_adapters": pipeline.lora_layers.len(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    }
}

/// Display next-steps guidance when no classify training data is provided.
fn display_classify_next_steps(json_output: bool) {
    if !json_output {
        println!();
        println!("{}", "NEXT STEPS".white().bold());
        println!("{}", "\u{2500}".repeat(50));
        println!("  Provide --data <train.jsonl> to start training.");
        println!("  Example: apr finetune --task classify --data train.jsonl -o checkpoints/");
    }
}

// Instruct helper functions (load_instruct_pipeline, display_instruct_plan,
// display_instruct_next_steps, display_instruct_result, display_instruct_result_json)
// are removed — they depend on unpublished entrenar 0.7.5 APIs:
//   instruct_corpus, instruct_pipeline, instruct_trainer, InstructTrainResult.
// They will be restored when entrenar publishes the instruct subsystem.

#[cfg(test)]
#[path = "finetune_tests.rs"]
mod tests;
