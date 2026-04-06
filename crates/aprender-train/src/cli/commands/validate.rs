//! Validate command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{load_config, validate_config, TrainSpec, ValidateArgs};

/// Format model information as a string
pub fn format_model_info(spec: &TrainSpec) -> String {
    let mode_str = format!("{:?}", spec.model.mode).to_lowercase();
    let mut lines = vec![
        format!("  Model path: {}", spec.model.path.display()),
        format!("  Model mode: {mode_str}"),
        format!("  Target layers: {:?}", spec.model.layers),
    ];
    if let Some(ref config) = spec.model.config {
        lines.push(format!("  Config preset: {config}"));
    }
    lines.join("\n")
}

/// Format data configuration as a string
pub fn format_data_info(spec: &TrainSpec) -> String {
    let mut lines = vec![format!("  Training data: {}", spec.data.train.display())];
    if let Some(val) = &spec.data.val {
        lines.push(format!("  Validation data: {}", val.display()));
    }
    lines.push(format!("  Batch size: {}", spec.data.batch_size));
    if let Some(ref tokenizer) = spec.data.tokenizer {
        lines.push(format!("  Tokenizer: {}", tokenizer.display()));
    }
    if let Some(seq_len) = spec.data.seq_len {
        lines.push(format!("  Sequence length: {seq_len}"));
    }
    if let Some(ref col) = spec.data.input_column {
        lines.push(format!("  Input column: {col}"));
    }
    if let Some(ref col) = spec.data.output_column {
        lines.push(format!("  Output column: {col}"));
    }
    if let Some(max_len) = spec.data.max_length {
        lines.push(format!("  Max length: {max_len}"));
    }
    lines.join("\n")
}

/// Format optimizer configuration as a string
pub fn format_optimizer_info(spec: &TrainSpec) -> String {
    let mut lines = vec![
        format!("  Optimizer: {}", spec.optimizer.name),
        format!("  Learning rate: {}", spec.optimizer.lr),
    ];
    if let Some(wd) = spec.optimizer.params.get("weight_decay") {
        lines.push(format!("  Weight decay: {wd}"));
    }
    lines.join("\n")
}

/// Format training configuration as a string
pub fn format_training_info(spec: &TrainSpec) -> String {
    let training_mode = format!("{:?}", spec.training.mode).to_lowercase();
    let mut lines = vec![
        format!("  Training mode: {training_mode}"),
        format!("  Epochs: {}", spec.training.epochs),
    ];
    if let Some(clip) = spec.training.grad_clip {
        lines.push(format!("  Gradient clipping: {clip}"));
    }
    if let Some(ref sched) = spec.training.lr_scheduler {
        let mut sched_str = format!("  Scheduler: {sched}");
        if spec.training.warmup_steps > 0 {
            sched_str.push_str(&format!(" (warmup={} steps)", spec.training.warmup_steps));
        }
        lines.push(sched_str);
        if let Some(ref params) = spec.training.scheduler_params {
            for (k, v) in params {
                lines.push(format!("    {k}: {v}"));
            }
        }
    }
    if let Some(ga) = spec.training.gradient_accumulation {
        lines.push(format!("  Gradient accumulation: {ga}"));
    }
    if let Some(ref mp) = spec.training.mixed_precision {
        lines.push(format!("  Mixed precision: {mp}"));
    }
    if let Some(seed) = spec.training.seed {
        lines.push(format!("  Seed: {seed}"));
    }
    lines.push(format!("  Output dir: {}", spec.training.output_dir.display()));
    lines.join("\n")
}

/// Format LoRA configuration as a string
pub fn format_lora_info(spec: &TrainSpec) -> Option<String> {
    spec.lora.as_ref().map(|lora| {
        let mut lines = vec![
            "  LoRA:".to_string(),
            format!("    Rank: {}", lora.rank),
            format!("    Alpha: {}", lora.alpha),
        ];
        if lora.dropout > 0.0 {
            lines.push(format!("    Dropout: {}", lora.dropout));
        }
        lines.join("\n")
    })
}

/// Format quantization configuration as a string
pub fn format_quant_info(spec: &TrainSpec) -> Option<String> {
    spec.quantize.as_ref().map(|quant| {
        format!("  Quantization:\n    Bits: {}\n    Symmetric: {}", quant.bits, quant.symmetric)
    })
}

/// Format merge configuration as a string
pub fn format_merge_info(spec: &TrainSpec) -> Option<String> {
    spec.merge.as_ref().map(|merge| {
        let mut lines = vec!["  Merge:".to_string(), format!("    Method: {}", merge.method)];
        if let Some(weight) = merge.params.get("weight") {
            lines.push(format!("    Weight: {weight}"));
        }
        lines.join("\n")
    })
}

/// Print detailed configuration summary
pub fn print_detailed_summary(spec: &TrainSpec) {
    println!();
    println!("Configuration Summary:");
    println!("{}", format_model_info(spec));
    println!();
    println!("{}", format_data_info(spec));
    println!();
    println!("{}", format_optimizer_info(spec));
    println!();
    println!("{}", format_training_info(spec));

    if let Some(lora_info) = format_lora_info(spec) {
        println!();
        println!("{lora_info}");
    }

    if let Some(quant_info) = format_quant_info(spec) {
        println!();
        println!("{quant_info}");
    }

    if let Some(merge_info) = format_merge_info(spec) {
        println!();
        println!("{merge_info}");
    }
}

pub fn run_validate(args: ValidateArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Normal, &format!("Validating config: {}", args.config.display()));

    let spec = load_config(&args.config).map_err(|e| format!("Config error: {e}"))?;

    validate_config(&spec).map_err(|e| format!("Validation failed: {e}"))?;

    log(level, LogLevel::Normal, "Configuration is valid");

    if args.detailed {
        print_detailed_summary(&spec);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DataConfig, LoRASpec, MergeSpec, ModelRef, OptimSpec, QuantSpec, TrainingParams,
    };
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn make_test_spec() -> TrainSpec {
        TrainSpec {
            model: ModelRef {
                path: PathBuf::from("/model/path"),
                layers: vec!["layer1".to_string()],
                ..Default::default()
            },
            data: DataConfig {
                train: PathBuf::from("/train.parquet"),
                val: Some(PathBuf::from("/val.parquet")),
                batch_size: 32,
                ..Default::default()
            },
            optimizer: OptimSpec {
                name: "adam".to_string(),
                lr: 0.001,
                params: {
                    let mut p = HashMap::new();
                    p.insert("weight_decay".to_string(), serde_json::json!(0.01));
                    p
                },
            },
            training: TrainingParams {
                epochs: 10,
                grad_clip: Some(1.0),
                output_dir: PathBuf::from("/output"),
                ..Default::default()
            },
            lora: Some(LoRASpec {
                rank: 16,
                alpha: 32.0,
                dropout: 0.1,
                target_modules: vec!["q_proj".to_string()],
                lora_plus_ratio: 1.0,
                double_quantize: false,
                quantize_base: false,
            }),
            quantize: Some(QuantSpec { bits: 4, symmetric: true, per_channel: true }),
            merge: Some(MergeSpec {
                method: "slerp".to_string(),
                params: {
                    let mut p = HashMap::new();
                    p.insert("weight".to_string(), serde_json::json!(0.5));
                    p
                },
            }),
            publish: None,
        }
    }

    #[test]
    fn test_format_model_info() {
        let spec = make_test_spec();
        let info = format_model_info(&spec);
        assert!(info.contains("/model/path"));
        assert!(info.contains("layer1"));
        assert!(info.contains("tabular"));
    }

    #[test]
    fn test_format_model_info_transformer() {
        let mut spec = make_test_spec();
        spec.model.mode = crate::config::ModelMode::Transformer;
        spec.model.config = Some("qwen2_1_5b".into());
        let info = format_model_info(&spec);
        assert!(info.contains("transformer"));
        assert!(info.contains("qwen2_1_5b"));
    }

    #[test]
    fn test_format_data_info() {
        let spec = make_test_spec();
        let info = format_data_info(&spec);
        assert!(info.contains("/train.parquet"));
        assert!(info.contains("/val.parquet"));
        assert!(info.contains("32"));
    }

    #[test]
    fn test_format_data_info_no_val() {
        let mut spec = make_test_spec();
        spec.data.val = None;
        let info = format_data_info(&spec);
        assert!(info.contains("/train.parquet"));
        assert!(!info.contains("Validation"));
    }

    #[test]
    fn test_format_data_info_llm_fields() {
        let mut spec = make_test_spec();
        spec.data.tokenizer = Some(std::path::PathBuf::from("./tokenizer.json"));
        spec.data.seq_len = Some(2048);
        spec.data.input_column = Some("text".into());
        spec.data.output_column = Some("label".into());
        spec.data.max_length = Some(512);
        let info = format_data_info(&spec);
        assert!(info.contains("tokenizer.json"));
        assert!(info.contains("2048"));
        assert!(info.contains("text"));
        assert!(info.contains("label"));
        assert!(info.contains("512"));
    }

    #[test]
    fn test_format_optimizer_info() {
        let spec = make_test_spec();
        let info = format_optimizer_info(&spec);
        assert!(info.contains("adam"));
        assert!(info.contains("0.001"));
        // weight_decay is in params, check it's present in output
        assert!(info.contains("Weight decay"));
    }

    #[test]
    fn test_format_training_info() {
        let spec = make_test_spec();
        let info = format_training_info(&spec);
        assert!(info.contains("10"));
        assert!(info.contains("regression"));
        assert!(info.contains("/output"));
    }

    #[test]
    fn test_format_training_info_full() {
        let mut spec = make_test_spec();
        spec.training.mode = crate::config::TrainingMode::CausalLm;
        spec.training.lr_scheduler = Some("cosine".into());
        spec.training.warmup_steps = 200;
        spec.training.gradient_accumulation = Some(8);
        spec.training.mixed_precision = Some("bf16".into());
        spec.training.seed = Some(42);
        let mut params = HashMap::new();
        params.insert("t_max".into(), serde_json::json!(1000));
        spec.training.scheduler_params = Some(params);
        let info = format_training_info(&spec);
        assert!(info.contains("causal"));
        assert!(info.contains("cosine"));
        assert!(info.contains("warmup=200"));
        assert!(info.contains("t_max"));
        assert!(info.contains('8'));
        assert!(info.contains("bf16"));
        assert!(info.contains("42"));
    }

    #[test]
    fn test_format_lora_info() {
        let spec = make_test_spec();
        let info = format_lora_info(&spec).expect("operation should succeed");
        assert!(info.contains("16"));
        assert!(info.contains("32"));
        assert!(info.contains("0.1"));
    }

    #[test]
    fn test_format_lora_info_none() {
        let mut spec = make_test_spec();
        spec.lora = None;
        assert!(format_lora_info(&spec).is_none());
    }

    #[test]
    fn test_format_quant_info() {
        let spec = make_test_spec();
        let info = format_quant_info(&spec).expect("operation should succeed");
        assert!(info.contains('4'));
        assert!(info.contains("true"));
    }

    #[test]
    fn test_format_quant_info_none() {
        let mut spec = make_test_spec();
        spec.quantize = None;
        assert!(format_quant_info(&spec).is_none());
    }

    #[test]
    fn test_format_merge_info() {
        let spec = make_test_spec();
        let info = format_merge_info(&spec).expect("operation should succeed");
        assert!(info.contains("slerp"));
        assert!(info.contains("0.5"));
    }

    #[test]
    fn test_format_merge_info_none() {
        let mut spec = make_test_spec();
        spec.merge = None;
        assert!(format_merge_info(&spec).is_none());
    }
}
