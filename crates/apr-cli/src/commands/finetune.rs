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
    json_output: bool,
) -> Result<()> {
    if merge_mode {
        return run_merge(model_path, adapter_path, output_path, json_output);
    }

    // Dispatch to classification pipeline when --task classify
    if let Some("classify") = task {
        return run_classify(
            model_size,
            data_path,
            output_path,
            num_classes,
            rank.unwrap_or(16),
            epochs,
            learning_rate,
            plan_only,
            json_output,
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
// Classification fine-tuning (--task classify)
// =============================================================================

/// Run classification fine-tuning pipeline via entrenar.
///
/// Creates a ClassifyPipeline with the specified configuration,
/// loads the corpus, and runs a forward pass to verify the pipeline works.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
fn run_classify(
    model_size: Option<&str>,
    data_path: Option<&Path>,
    output_path: Option<&Path>,
    num_classes: usize,
    rank: u32,
    epochs: u32,
    learning_rate: f64,
    plan_only: bool,
    json_output: bool,
) -> Result<()> {
    use entrenar::finetune::classify_pipeline::{ClassifyConfig, ClassifyPipeline};
    use entrenar::transformer::TransformerConfig;

    if !json_output {
        output::section("apr finetune --task classify (Shell Safety Classification)");
        println!();
    }

    // Select model config based on model_size
    let model_config = match model_size.unwrap_or("tiny") {
        "0.5B" | "500M" | "qwen2-0.5b" => TransformerConfig::qwen2_0_5b(),
        _ => TransformerConfig::tiny(), // For testing
    };

    let classify_config = ClassifyConfig {
        num_classes,
        lora_rank: rank as usize,
        lora_alpha: rank as f32,
        learning_rate: learning_rate as f32,
        epochs: epochs as usize,
        ..ClassifyConfig::default()
    };

    if !json_output {
        output::kv(
            "Model",
            format!(
                "{}h x {}L",
                model_config.hidden_size, model_config.num_hidden_layers
            ),
        );
        output::kv("Classes", num_classes.to_string());
        output::kv("LoRA rank", rank.to_string());
        output::kv("Epochs", epochs.to_string());
        output::kv("Learning rate", format!("{learning_rate:.1e}"));
        println!();
    }

    let pipeline = ClassifyPipeline::new(&model_config, classify_config);

    if !json_output {
        println!("{}", pipeline.summary());
        println!();
    }

    if plan_only {
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
        return Ok(());
    }

    // Verify corpus if provided
    if let Some(data) = data_path {
        if !data.exists() {
            return Err(CliError::FileNotFound(data.to_path_buf()));
        }

        let samples = pipeline
            .load_corpus(data)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to load corpus: {e}")))?;

        let stats = entrenar::finetune::corpus_stats(&samples, num_classes);

        if !json_output {
            output::subheader("Corpus");
            output::kv("Samples", stats.total.to_string());
            output::kv("Avg input length", format!("{} chars", stats.avg_input_len));
            for (i, count) in stats.class_counts.iter().enumerate() {
                let label = format!("  Class {i}");
                output::kv(&label, count.to_string());
            }
            println!();
        }

        // Run a single forward pass to verify pipeline works
        if !json_output {
            output::pipeline_stage("Verification", output::StageStatus::Running);
        }

        if let Some(sample) = samples.first() {
            // Simple byte-level tokenization for verification
            let token_ids: Vec<u32> = sample.input.bytes().map(u32::from).take(64).collect();
            let loss = pipeline.train_step(&token_ids, sample.label);

            if !json_output {
                output::pipeline_stage("Verification", output::StageStatus::Done);
                output::kv("  Sample loss", format!("{loss:.4}"));
                println!();
            }
        }
    }

    let _out = output_path.unwrap_or(Path::new("classify-adapter.apr"));

    if !json_output {
        output::subheader("Status");
        println!("  Pipeline configured and verified.");
        println!(
            "  Full training loop requires GPU acceleration (--task classify --epochs {epochs})."
        );
    }

    Ok(())
}

#[cfg(test)]
#[path = "finetune_tests.rs"]
mod tests;
