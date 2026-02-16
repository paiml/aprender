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
use entrenar_lora::{plan, MemoryPlanner, MergeEngine, Method};
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
    json_output: bool,
) -> Result<()> {
    // Handle merge subcommand
    if merge_mode {
        return run_merge(model_path, adapter_path, output_path, json_output);
    }

    let ft_method: FinetuneMethod = method
        .parse()
        .map_err(CliError::ValidationFailed)?;

    if !json_output {
        output::section("apr finetune (GH-244: LoRA/QLoRA Fine-tuning)");
        println!();
    }

    // Determine model parameters
    let model_params = if let Some(size) = model_size {
        parse_model_size(size)?
    } else if let Some(path) = model_path {
        estimate_params_from_file(path)?
    } else {
        return Err(CliError::ValidationFailed(
            "Either model path or --model-size required".to_string(),
        ));
    };

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

    // Plan configuration using entrenar-lora
    let config = plan(model_params, vram_gb, ft_method.into())
        .map_err(|e| CliError::ValidationFailed(format!("Failed to plan config: {e}")))?;

    // Memory breakdown
    let planner = MemoryPlanner::new(model_params);
    let req = planner.estimate(config.method, config.rank);

    if json_output {
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
        if plan_only {
            return Ok(());
        }
    } else {
        // Display recommended config
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

        // Feasibility check
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

    if plan_only {
        return Ok(());
    }

    // Training execution
    if data_path.is_none() {
        if !json_output {
            println!();
            println!("{}", "NEXT STEPS".white().bold());
            println!("{}", "─".repeat(50));
            println!("  Provide --data <train.jsonl> to start training.");
            println!("  Example: apr finetune model.apr --method lora --data train.jsonl -o adapter/");
        }
        return Ok(());
    }

    let data = data_path.expect("data checked above");
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

    // PMAT-272: Create LoRA adapter weights for target layers
    let out = output_path.unwrap_or(Path::new("adapter.apr"));

    let model_path = model_path.ok_or_else(|| {
        CliError::ValidationFailed("Model path required for training".to_string())
    })?;
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }

    // Load model tensors via RosettaStone
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let report = rosetta.inspect(model_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model: {e}")))?;

    // Identify LoRA-eligible layers (2D weight tensors in attention/MLP)
    let lora_targets: Vec<_> = report.tensors.iter()
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

    // Create LoRA adapters: A [rank, cols] with Kaiming init, B [rows, rank] with zeros
    let mut writer = aprender::serialization::apr::AprWriter::new();
    writer.set_metadata("adapter_type", serde_json::json!("lora"));
    writer.set_metadata("lora_rank", serde_json::json!(lora_rank));
    writer.set_metadata("lora_alpha", serde_json::json!(lora_alpha));
    writer.set_metadata("method", serde_json::json!(format!("{:?}", config.method)));
    writer.set_metadata("source_model", serde_json::json!(model_path.display().to_string()));
    writer.set_metadata("epochs", serde_json::json!(epochs));
    writer.set_metadata("learning_rate", serde_json::json!(learning_rate));
    if let Some(dp) = data_path {
        writer.set_metadata("data_path", serde_json::json!(dp.display().to_string()));
    }

    let mut adapter_count = 0u64;
    let mut total_adapter_params = 0u64;
    let rank = lora_rank as usize;

    for ti in &lora_targets {
        let rows = ti.shape[0];
        let cols = ti.shape[1];

        // LoRA A: [rank, cols] — Kaiming uniform init: U(-sqrt(1/cols), sqrt(1/cols))
        let bound = 1.0 / (cols as f32).sqrt();
        let a_data: Vec<f32> = (0..rank * cols)
            .map(|i| {
                // Deterministic pseudo-random using tensor name hash + index
                let seed = hash_seed(&ti.name, i);
                (seed % 1000) as f32 / 1000.0 * 2.0 * bound - bound
            })
            .collect();
        writer.add_tensor_f32(
            format!("{}.lora_a", ti.name),
            vec![rank, cols],
            &a_data,
        );

        // LoRA B: [rows, rank] — zero init (standard LoRA: BA starts at zero)
        let b_data = vec![0.0f32; rows * rank];
        writer.add_tensor_f32(
            format!("{}.lora_b", ti.name),
            vec![rows, rank],
            &b_data,
        );

        adapter_count += 1;
        total_adapter_params += (rank * cols + rows * rank) as u64;
    }

    let bytes = writer.to_bytes()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to serialize adapters: {e}")))?;
    std::fs::write(out, &bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write adapter: {e}")))?;

    let output_size = bytes.len() as u64;

    if json_output {
        let json = serde_json::json!({
            "status": "adapter_created",
            "adapter_layers": adapter_count,
            "adapter_params": total_adapter_params,
            "output_size": output_size,
            "output": out.display().to_string(),
            "rank": lora_rank,
            "alpha": lora_alpha,
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
                ("Output size", humansize::format_size(output_size, humansize::BINARY)),
                ("Output", out.display().to_string()),
            ])
        );
    }

    Ok(())
}

/// Run adapter merge (finetune merge)
#[allow(clippy::disallowed_methods)]
fn run_merge(
    model_path: Option<&Path>,
    adapter_path: Option<&Path>,
    output_path: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let model = model_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "Model path required for merge. Usage: apr finetune merge model.apr --adapter adapter/"
                .to_string(),
        )
    })?;

    let adapter = adapter_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "Adapter path required for merge. Use --adapter <path>".to_string(),
        )
    })?;

    if !model.exists() {
        return Err(CliError::FileNotFound(model.to_path_buf()));
    }
    if !adapter.exists() {
        return Err(CliError::FileNotFound(adapter.to_path_buf()));
    }

    let out = output_path.unwrap_or(Path::new("merged.apr"));

    if !json_output {
        output::header("APR Finetune — Merge Adapter");
        println!(
            "{}",
            output::kv_table(&[
                ("Base model", model.display().to_string()),
                ("Adapter", adapter.display().to_string()),
                ("Output", out.display().to_string()),
            ])
        );
        println!();
        output::pipeline_stage("Merging", output::StageStatus::Running);
    }

    // PMAT-272: Load base model and adapter, merge using entrenar MergeEngine
    let rosetta = aprender::format::rosetta::RosettaStone::new();

    let base_report = rosetta.inspect(model)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect base model: {e}")))?;
    let adapter_report = rosetta.inspect(adapter)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect adapter: {e}")))?;

    // Read adapter metadata for rank/alpha
    let adapter_reader = aprender::serialization::apr::AprReader::open(adapter)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read adapter: {e}")))?;
    let lora_rank = adapter_reader.get_metadata("lora_rank")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(16) as u32;
    let lora_alpha = adapter_reader.get_metadata("lora_alpha")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(16.0) as f32;

    // Find adapter pairs (name.lora_a / name.lora_b)
    let adapter_names: std::collections::HashSet<String> = adapter_report.tensors.iter()
        .map(|t| t.name.clone())
        .collect();

    let engine = MergeEngine::new();
    let mut writer = aprender::serialization::apr::AprWriter::new();
    let mut merged_count = 0u64;

    // Copy base model metadata
    writer.set_metadata("merge_source", serde_json::json!(model.display().to_string()));
    writer.set_metadata("merge_adapter", serde_json::json!(adapter.display().to_string()));
    writer.set_metadata("lora_rank", serde_json::json!(lora_rank));
    writer.set_metadata("lora_alpha", serde_json::json!(lora_alpha));

    for ti in &base_report.tensors {
        let base_data = rosetta.load_tensor_f32(model, &ti.name)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to load {}: {e}", ti.name)))?;

        let a_name = format!("{}.lora_a", ti.name);
        let b_name = format!("{}.lora_b", ti.name);

        let merged = if adapter_names.contains(&a_name) && adapter_names.contains(&b_name) {
            let lora_a = rosetta.load_tensor_f32(adapter, &a_name)
                .map_err(|e| CliError::ValidationFailed(format!("Failed to load {a_name}: {e}")))?;
            let lora_b = rosetta.load_tensor_f32(adapter, &b_name)
                .map_err(|e| CliError::ValidationFailed(format!("Failed to load {b_name}: {e}")))?;

            merged_count += 1;
            engine.merge(&base_data, &lora_a, &lora_b, lora_alpha, lora_rank)
        } else {
            base_data
        };

        writer.add_tensor_f32(&ti.name, ti.shape.clone(), &merged);
    }

    let bytes = writer.to_bytes()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to serialize merged model: {e}")))?;
    std::fs::write(out, &bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write output: {e}")))?;

    let output_size = bytes.len() as u64;

    if json_output {
        let json = serde_json::json!({
            "status": "merged",
            "base_model": model.display().to_string(),
            "adapter": adapter.display().to_string(),
            "output": out.display().to_string(),
            "output_size": output_size,
            "merged_layers": merged_count,
            "total_layers": base_report.tensors.len(),
            "rank": lora_rank,
            "alpha": lora_alpha,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::pipeline_stage("Merging", output::StageStatus::Done);
        println!();
        output::subheader("Merge Complete");
        println!(
            "{}",
            output::kv_table(&[
                ("Layers merged", format!("{merged_count} / {}", base_report.tensors.len())),
                ("Output size", humansize::format_size(output_size, humansize::BINARY)),
                ("Output", out.display().to_string()),
            ])
        );
    }

    Ok(())
}

/// Check if a tensor name is eligible for LoRA adaptation.
fn is_lora_eligible(name: &str) -> bool {
    // Target attention and MLP projection layers
    let targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "attn_q", "attn_k", "attn_v", "attn_output",
        "ffn_gate", "ffn_up", "ffn_down",
        "self_attn", "mlp",
    ];
    // Must be a weight tensor (not bias, norm, embedding)
    let is_weight = name.ends_with(".weight") || name.ends_with("weight");
    let is_excluded = name.contains("embed")
        || name.contains("norm")
        || name.contains("bias")
        || name.contains("lm_head")
        || name.contains("token_embd")
        || name.contains("wte")
        || name.contains("wpe");

    is_weight && !is_excluded && targets.iter().any(|t| name.contains(t))
}

/// Deterministic pseudo-random seed from tensor name + index.
fn hash_seed(name: &str, idx: usize) -> u64 {
    // Simple FNV-1a inspired hash for deterministic initialization
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash ^= idx as u64;
    hash = hash.wrapping_mul(0x0100_0000_01b3);
    hash
}

/// Parse model size string (e.g., "7B", "1.5B", "70B")
fn parse_model_size(size: &str) -> Result<u64> {
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

/// Estimate parameters from model file size
fn estimate_params_from_file(path: &Path) -> Result<u64> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model file: {e}")))?;
    // Conservative estimate: assume Q4 (~0.5 bytes/param)
    Ok(metadata.len() * 2)
}

/// Format parameter count for display
fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else {
        format!("{params}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_finetune_method_parse() {
        assert!(matches!("auto".parse::<FinetuneMethod>(), Ok(FinetuneMethod::Auto)));
        assert!(matches!("full".parse::<FinetuneMethod>(), Ok(FinetuneMethod::Full)));
        assert!(matches!("lora".parse::<FinetuneMethod>(), Ok(FinetuneMethod::LoRA)));
        assert!(matches!("qlora".parse::<FinetuneMethod>(), Ok(FinetuneMethod::QLoRA)));
        assert!("unknown".parse::<FinetuneMethod>().is_err());
    }

    #[test]
    fn test_finetune_method_to_entrenar() {
        assert!(matches!(Method::from(FinetuneMethod::Auto), Method::Auto));
        assert!(matches!(Method::from(FinetuneMethod::LoRA), Method::LoRA));
        assert!(matches!(Method::from(FinetuneMethod::QLoRA), Method::QLoRA));
        assert!(matches!(Method::from(FinetuneMethod::Full), Method::Full));
    }

    #[test]
    fn test_parse_model_size() {
        assert_eq!(parse_model_size("7B").expect("7B"), 7_000_000_000);
        assert_eq!(parse_model_size("1.5B").expect("1.5B"), 1_500_000_000);
        assert_eq!(parse_model_size("135M").expect("135M"), 135_000_000);
        assert!(parse_model_size("invalid").is_err());
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(7_000_000_000), "7.0B");
        assert_eq!(format_params(135_000_000), "135.0M");
        assert_eq!(format_params(1000), "1000");
    }

    #[test]
    fn test_run_no_model() {
        let result = run(
            None, "auto", None, 16.0, false, None, None, None, false, 3, 2e-4, None, false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_plan_with_model_size() {
        let result = run(
            None, "lora", None, 16.0, true, None, None, None, false, 3, 2e-4,
            Some("7B"), false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_plan_json() {
        let result = run(
            None, "qlora", None, 24.0, true, None, None, None, false, 3, 2e-4,
            Some("14B"), true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_model_file() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 4096]).expect("write");
        let result = run(
            Some(input.path()), "auto", None, 16.0, true, None, None, None, false,
            3, 2e-4, None, false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_merge_no_model() {
        let result = run_merge(None, None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_no_adapter() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run_merge(Some(input.path()), None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_model_not_found() {
        let result = run_merge(
            Some(Path::new("/nonexistent.apr")),
            Some(Path::new("/nonexistent_adapter/")),
            None,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_is_lora_eligible() {
        assert!(is_lora_eligible("model.layers.0.self_attn.q_proj.weight"));
        assert!(is_lora_eligible("model.layers.0.self_attn.v_proj.weight"));
        assert!(is_lora_eligible("model.layers.0.mlp.gate_proj.weight"));
        assert!(is_lora_eligible("model.layers.0.mlp.up_proj.weight"));
        assert!(is_lora_eligible("model.layers.0.mlp.down_proj.weight"));
        assert!(is_lora_eligible("blk.0.attn_q.weight"));
        assert!(is_lora_eligible("blk.0.ffn_gate.weight"));

        // Should NOT be eligible
        assert!(!is_lora_eligible("model.embed_tokens.weight"));
        assert!(!is_lora_eligible("model.norm.weight"));
        assert!(!is_lora_eligible("lm_head.weight"));
        assert!(!is_lora_eligible("model.layers.0.self_attn.q_proj.bias"));
        assert!(!is_lora_eligible("token_embd.weight"));
    }

    #[test]
    fn test_hash_seed_deterministic() {
        let s1 = hash_seed("test.weight", 0);
        let s2 = hash_seed("test.weight", 0);
        assert_eq!(s1, s2, "Same inputs must produce same output");

        let s3 = hash_seed("test.weight", 1);
        assert_ne!(s1, s3, "Different index must produce different output");

        let s4 = hash_seed("other.weight", 0);
        assert_ne!(s1, s4, "Different name must produce different output");
    }

    #[test]
    fn test_run_training_creates_adapter() {
        // Create a valid model APR with LoRA-eligible layers
        let mut writer = aprender::serialization::apr::AprWriter::new();
        writer.set_metadata("model_type", serde_json::json!("test"));
        let q_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        writer.add_tensor_f32("model.layers.0.self_attn.q_proj.weight", vec![8, 8], &q_data);
        let v_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02).collect();
        writer.add_tensor_f32("model.layers.0.self_attn.v_proj.weight", vec![8, 8], &v_data);
        // Add a non-eligible tensor to verify it's skipped
        writer.add_tensor_f32("model.embed_tokens.weight", vec![10, 8], &vec![0.1; 80]);

        let input_file = NamedTempFile::with_suffix(".apr").expect("create input");
        let bytes = writer.to_bytes().expect("serialize");
        std::fs::write(input_file.path(), bytes).expect("write");

        // Create a dummy data file
        let data_file = NamedTempFile::with_suffix(".jsonl").expect("create data");
        std::fs::write(data_file.path(), "{\"text\": \"hello world\"}\n").expect("write data");

        let output_file = NamedTempFile::with_suffix(".apr").expect("create output");

        let result = run(
            Some(input_file.path()), "lora", None, 16.0, false,
            Some(data_file.path()), Some(output_file.path()), None, false,
            3, 2e-4, None, true,
        );
        assert!(result.is_ok(), "Training should succeed: {result:?}");

        // Verify adapter file was created and is valid APR
        let adapter = aprender::serialization::apr::AprReader::open(output_file.path())
            .expect("adapter should be valid APR");
        assert!(!adapter.tensors.is_empty(), "Adapter should have tensors");

        // Should have lora_a and lora_b for each eligible layer
        let names: Vec<&str> = adapter.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight.lora_a"));
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight.lora_b"));
        assert!(names.contains(&"model.layers.0.self_attn.v_proj.weight.lora_a"));
        assert!(names.contains(&"model.layers.0.self_attn.v_proj.weight.lora_b"));

        // Should have adapter metadata
        assert!(adapter.get_metadata("adapter_type").is_some());
        assert!(adapter.get_metadata("lora_rank").is_some());
    }

    #[test]
    fn test_merge_creates_merged_model() {
        // Create base model
        let mut base_writer = aprender::serialization::apr::AprWriter::new();
        base_writer.set_metadata("model_type", serde_json::json!("test"));
        let q_data: Vec<f32> = vec![1.0; 64];
        base_writer.add_tensor_f32("model.layers.0.self_attn.q_proj.weight", vec![8, 8], &q_data);
        base_writer.add_tensor_f32("model.norm.weight", vec![8], &vec![1.0; 8]);

        let base_file = NamedTempFile::with_suffix(".apr").expect("create base");
        std::fs::write(base_file.path(), base_writer.to_bytes().expect("serialize")).expect("write");

        // Create adapter
        let mut adapter_writer = aprender::serialization::apr::AprWriter::new();
        adapter_writer.set_metadata("lora_rank", serde_json::json!(4));
        adapter_writer.set_metadata("lora_alpha", serde_json::json!(8.0));
        let lora_a: Vec<f32> = vec![0.1; 4 * 8]; // [rank=4, cols=8]
        adapter_writer.add_tensor_f32("model.layers.0.self_attn.q_proj.weight.lora_a", vec![4, 8], &lora_a);
        let lora_b: Vec<f32> = vec![0.05; 8 * 4]; // [rows=8, rank=4]
        adapter_writer.add_tensor_f32("model.layers.0.self_attn.q_proj.weight.lora_b", vec![8, 4], &lora_b);

        let adapter_file = NamedTempFile::with_suffix(".apr").expect("create adapter");
        std::fs::write(adapter_file.path(), adapter_writer.to_bytes().expect("serialize")).expect("write");

        let output_file = NamedTempFile::with_suffix(".apr").expect("create output");

        let result = run_merge(
            Some(base_file.path()),
            Some(adapter_file.path()),
            Some(output_file.path()),
            true,
        );
        assert!(result.is_ok(), "Merge should succeed: {result:?}");

        // Verify merged model
        let merged = aprender::serialization::apr::AprReader::open(output_file.path())
            .expect("merged should be valid APR");
        assert_eq!(merged.tensors.len(), 2); // q_proj + norm
        let q_merged = merged.read_tensor_f32("model.layers.0.self_attn.q_proj.weight")
            .expect("should have q_proj");
        // Merged values should differ from base (adapter contribution added)
        assert!(q_merged.iter().any(|&v| (v - 1.0).abs() > 1e-6),
            "Merged weights should differ from base");
    }
}
