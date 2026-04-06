//! Multi-Adapter Training Example (GPU-SHARE Phase 2, GH-203/205/206)
//!
//! Demonstrates training N independent LoRA adapter sets on a single frozen
//! base model using `MultiAdapterPipeline`.
//!
//! ```bash
//! cargo run --example multi_adapter_training
//! cargo run --example multi_adapter_training -- --adapters 3 --epochs 2
//! cargo run --example multi_adapter_training -- --schedule priority
//! ```

use entrenar::finetune::instruct_corpus::InstructSample;
use entrenar::finetune::instruct_pipeline::{InstructConfig, InstructPipeline};
use entrenar::finetune::multi_adapter_pipeline::{
    AdapterConfig, AdapterSchedule, MultiAdapterPipeline,
};
use entrenar::transformer::TransformerConfig;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }

    let num_adapters = parse_arg(&args, "--adapters").unwrap_or(2);
    let epochs = parse_arg(&args, "--epochs").unwrap_or(2);
    let schedule = parse_schedule(&args);

    println!("=== Multi-Adapter Training Demo ===");
    println!();
    println!("Adapters:  {num_adapters}");
    println!("Epochs:    {epochs}");
    println!("Schedule:  {schedule:?}");
    println!();

    let (model_config, instruct_config) = build_configs(epochs);
    let base_pipeline = InstructPipeline::new(&model_config, instruct_config.clone());
    print_model_info(&model_config, &instruct_config);

    let mut multi = MultiAdapterPipeline::new(base_pipeline, schedule);
    register_adapters(&mut multi, num_adapters, &instruct_config);

    run_training(&mut multi, epochs, schedule);
    print_summary(&multi, num_adapters);
    save_checkpoints(&multi, epochs);
}

fn print_usage() {
    println!("Multi-Adapter Training Example (GPU-SHARE Phase 2)");
    println!();
    println!("Usage:");
    println!("  multi_adapter_training                      # 2 adapters, round-robin");
    println!("  multi_adapter_training --adapters 3         # 3 adapters");
    println!("  multi_adapter_training --epochs 5           # 5 epochs");
    println!("  multi_adapter_training --schedule priority  # PriorityValLoss scheduling");
    println!("  multi_adapter_training --schedule sync      # Synchronized scheduling");
}

fn build_configs(epochs: usize) -> (TransformerConfig, InstructConfig) {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig {
        lora_rank: 4,
        lora_alpha: 8.0,
        learning_rate: 1e-3,
        epochs,
        max_seq_len: 32,
        ..InstructConfig::default()
    };
    (model_config, instruct_config)
}

fn print_model_info(model_config: &TransformerConfig, instruct_config: &InstructConfig) {
    println!(
        "Base model: {}h x {}L (vocab {})",
        model_config.hidden_size, model_config.num_hidden_layers, model_config.vocab_size
    );
    println!("LoRA rank:  {}", instruct_config.lora_rank);
    println!();
}

fn register_adapters(
    multi: &mut MultiAdapterPipeline,
    num_adapters: usize,
    instruct_config: &InstructConfig,
) {
    for i in 0..num_adapters {
        let train_samples = generate_synthetic_samples(i, 20);
        let val_samples = generate_synthetic_samples(i + 100, 5);
        let checkpoint_dir = PathBuf::from(format!("/tmp/multi-adapter-demo/adapter-{i}"));

        let adapter_config = AdapterConfig {
            data_path: PathBuf::from(format!("synthetic-adapter-{i}.jsonl")),
            checkpoint_dir: checkpoint_dir.clone(),
            instruct_config: instruct_config.clone(),
        };

        multi.add_adapter(adapter_config, train_samples, val_samples);
        println!("Adapter {i}: 20 train, 5 val -> {}", checkpoint_dir.display());
    }
    println!();
}

fn run_training(multi: &mut MultiAdapterPipeline, epochs: usize, schedule: AdapterSchedule) {
    // Note: tiny model has no tokenizer, so train_step_adapter returns None.
    // With a real model (from_pretrained or from_apr), actual training occurs.
    println!("--- Training ---");
    if multi.base_pipeline.has_tokenizer() {
        run_real_training(multi, epochs);
    } else {
        println!("(Tiny model has no tokenizer — demonstrating pipeline orchestration only)");
        println!();
        for epoch in 0..epochs {
            multi.reset_epoch(epoch as u64);
            println!(
                "Epoch {} | {} adapters scheduled ({schedule:?})",
                epoch + 1,
                multi.num_adapters()
            );
            for i in 0..multi.num_adapters() {
                println!(
                    "  Adapter {i}: {} train samples, cursor=0",
                    multi.adapters[i].train_samples.len()
                );
            }
        }
    }
}

fn run_real_training(multi: &mut MultiAdapterPipeline, epochs: usize) {
    for epoch in 0..epochs {
        multi.reset_epoch(epoch as u64);
        let mut epoch_losses: Vec<Vec<f32>> = vec![Vec::new(); multi.num_adapters()];

        while !multi.all_exhausted() {
            if let Some(idx) = multi.select_next_adapter() {
                if let Some(result) = multi.train_step_adapter(idx) {
                    epoch_losses[idx].push(result.loss);
                }
            } else {
                break;
            }
        }

        for (i, losses) in epoch_losses.iter().enumerate() {
            let avg = if losses.is_empty() {
                0.0
            } else {
                losses.iter().sum::<f32>() / losses.len() as f32
            };
            println!(
                "Epoch {} | Adapter {i}: avg_loss={avg:.4} ({} steps)",
                epoch + 1,
                losses.len()
            );
        }
    }
}

fn print_summary(multi: &MultiAdapterPipeline, num_adapters: usize) {
    println!();
    println!("--- Summary ---");
    println!("Global steps: {}", multi.global_step);
    println!("Adapters:     {}", multi.num_adapters());
    println!();
    println!("VRAM savings vs {num_adapters} separate processes:");
    println!("  MPS:           {num_adapters} x base_model = {num_adapters}x VRAM");
    println!("  Multi-adapter: 1 x base_model + {num_adapters} x ~0.02 GB LoRA = ~1x VRAM");
}

fn save_checkpoints(multi: &MultiAdapterPipeline, epochs: usize) {
    println!();
    println!("--- Checkpointing ---");
    for i in 0..multi.num_adapters() {
        match multi.save_adapter_checkpoint(i, epochs - 1, 0.0) {
            Ok(path) => println!("Adapter {i}: saved to {}", path.display()),
            Err(e) => println!("Adapter {i}: checkpoint failed: {e}"),
        }
    }
}

/// Generate synthetic instruct samples for demonstration.
fn generate_synthetic_samples(seed: usize, count: usize) -> Vec<InstructSample> {
    (0..count)
        .map(|i| InstructSample {
            instruction: format!("Explain concept {seed}-{i} in simple terms"),
            response: format!("Concept {seed}-{i} is a fundamental idea that relates to..."),
            system: None,
            metadata: None,
        })
        .collect()
}

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok())
}

fn parse_schedule(args: &[String]) -> AdapterSchedule {
    let val = args
        .iter()
        .position(|a| a == "--schedule")
        .and_then(|i| args.get(i + 1))
        .map(std::string::String::as_str);
    match val {
        Some("priority") => AdapterSchedule::PriorityValLoss,
        Some("sync") => AdapterSchedule::Synchronized,
        _ => AdapterSchedule::RoundRobin,
    }
}
