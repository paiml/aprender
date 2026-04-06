//! Example: Declarative Training from YAML Configuration
//!
//! This example demonstrates Ludwig-style declarative training configuration.
//! Instead of writing imperative training code, you define your training
//! pipeline in a YAML file.

use entrenar::config::{DataConfig, LoRASpec, ModelRef, OptimSpec, TrainSpec, TrainingParams};
use std::collections::HashMap;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Declarative Training Configuration Example ===\n");

    // Example 1: Create a config programmatically
    println!("1. PROGRAMMATIC CONFIG CREATION\n");

    let spec = TrainSpec {
        model: ModelRef {
            path: PathBuf::from("model.gguf"),
            layers: vec!["q_proj".to_string(), "v_proj".to_string()],
            ..Default::default()
        },
        data: DataConfig {
            train: PathBuf::from("train.parquet"),
            val: Some(PathBuf::from("val.parquet")),
            batch_size: 16,
            auto_infer_types: true,
            seq_len: Some(2048),
            ..Default::default()
        },
        optimizer: OptimSpec {
            name: "adamw".to_string(),
            lr: 0.0001,
            params: {
                let mut params = HashMap::new();
                params.insert("beta1".to_string(), serde_json::json!(0.9));
                params.insert("beta2".to_string(), serde_json::json!(0.999));
                params.insert("weight_decay".to_string(), serde_json::json!(0.01));
                params
            },
        },
        lora: Some(LoRASpec {
            rank: 64,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            dropout: 0.1,
            lora_plus_ratio: 1.0,
            double_quantize: false,
            quantize_base: false,
        }),
        quantize: None,
        merge: None,
        training: TrainingParams {
            epochs: 3,
            grad_clip: Some(1.0),
            lr_scheduler: Some("cosine".to_string()),
            warmup_steps: 100,
            save_interval: 1,
            output_dir: PathBuf::from("./checkpoints"),
            ..Default::default()
        },
        publish: None,
    };

    // Serialize to YAML
    let yaml = serde_yaml::to_string(&spec)?;
    println!("Generated YAML configuration:\n");
    println!("{yaml}");

    // Example 2: Load from YAML file (commented out - requires actual files)
    println!("\n2. LOAD FROM YAML FILE\n");
    println!("To train from a YAML config file, run:");
    println!("  cargo run --example train_from_yaml examples/config.yaml\n");

    // Example 3: Show minimal config
    println!("3. MINIMAL CONFIG EXAMPLE\n");
    let minimal_yaml = r"
model:
  path: model.gguf

data:
  train: data.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001

training:
  epochs: 10
";
    println!("{minimal_yaml}");

    let minimal_spec: TrainSpec = serde_yaml::from_str(minimal_yaml)?;
    println!("✓ Minimal config parsed successfully");
    println!("  Epochs: {}", minimal_spec.training.epochs);
    println!("  LoRA: {}", if minimal_spec.lora.is_some() { "yes" } else { "no" });
    println!("  Quantization: {}", if minimal_spec.quantize.is_some() { "yes" } else { "no" });

    println!("\n=== Example Complete ===\n");

    Ok(())
}
