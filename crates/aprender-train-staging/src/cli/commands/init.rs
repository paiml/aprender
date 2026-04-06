//! Init command implementation
//!
//! Supports smart initialization with:
//! - `--base` for HF model IDs (auto-detects model size for LoRA rank)
//! - `--method` for training method selection (lora, qlora, full)
//! - Data format auto-detection (JSONL, Parquet, text, CSV)

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{InitArgs, InitTemplate, TrainingMethod};
use crate::yaml_mode::{generate_yaml, Template};

/// Estimated model size category for LoRA rank suggestion
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSize {
    /// < 1B parameters
    Small,
    /// 1B - 7B parameters
    Medium,
    /// 7B - 30B parameters
    Large,
    /// > 30B parameters
    XLarge,
}

/// Detected data format
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataFormat {
    Jsonl,
    Parquet,
    Csv,
    Text,
    Unknown,
}

/// Estimate model size from HF model name patterns
///
/// Parses common naming conventions like:
/// - "Qwen/Qwen2.5-Coder-0.5B" -> Small
/// - "meta-llama/Llama-3-7B" -> Medium
/// - "meta-llama/Llama-3-13B" -> Large
/// - "meta-llama/Llama-3-70B" -> XLarge
pub fn estimate_model_size(model_name: &str) -> ModelSize {
    // Extract the part after the last "/" (the model name itself)
    let name = model_name.rsplit('/').next().unwrap_or(model_name);
    let name_upper = name.to_uppercase();

    // Split by '-' and '_' only (not '.') to preserve decimal numbers like "0.5B"
    for segment in name_upper.split(['-', '_']) {
        if let Some(stripped) = segment.strip_suffix('B') {
            if let Ok(size) = stripped.parse::<f64>() {
                return categorize_param_count(size);
            }
        }
    }

    // Default to medium if we can't determine
    ModelSize::Medium
}

fn categorize_param_count(billions: f64) -> ModelSize {
    if billions < 1.0 {
        ModelSize::Small
    } else if billions <= 7.0 {
        ModelSize::Medium
    } else if billions <= 30.0 {
        ModelSize::Large
    } else {
        ModelSize::XLarge
    }
}

/// Suggest LoRA rank based on model size
pub fn suggest_lora_rank(size: ModelSize) -> u32 {
    match size {
        ModelSize::Small => 32,
        ModelSize::Medium => 64,
        ModelSize::Large => 128,
        ModelSize::XLarge => 256,
    }
}

/// Suggest learning rate based on model size
pub fn suggest_learning_rate(size: ModelSize) -> f64 {
    match size {
        ModelSize::Small => 3e-4,
        ModelSize::Medium => 2e-4,
        ModelSize::Large => 1e-4,
        ModelSize::XLarge => 5e-5,
    }
}

/// Detect data format from a path (file or directory)
pub fn detect_data_format(path: &str) -> DataFormat {
    let path = std::path::Path::new(path);

    // If it's a file, check the extension
    if path.is_file() {
        return format_from_extension(path);
    }

    // If it's a directory, scan for common data files
    if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                let fmt = format_from_extension(&p);
                if fmt != DataFormat::Unknown {
                    return fmt;
                }
            }
        }
    }

    // Try extension-based detection for paths that don't exist yet
    format_from_extension(path)
}

fn format_from_extension(path: &std::path::Path) -> DataFormat {
    match path.extension().and_then(|e| e.to_str()) {
        Some("jsonl" | "jsonlines") => DataFormat::Jsonl,
        Some("parquet" | "pq") => DataFormat::Parquet,
        Some("csv" | "tsv") => DataFormat::Csv,
        Some("txt" | "text") => DataFormat::Text,
        _ => DataFormat::Unknown,
    }
}

impl std::fmt::Display for DataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataFormat::Jsonl => write!(f, "jsonl"),
            DataFormat::Parquet => write!(f, "parquet"),
            DataFormat::Csv => write!(f, "csv"),
            DataFormat::Text => write!(f, "text"),
            DataFormat::Unknown => write!(f, "unknown"),
        }
    }
}

impl std::fmt::Display for ModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSize::Small => write!(f, "small (<1B)"),
            ModelSize::Medium => write!(f, "medium (1-7B)"),
            ModelSize::Large => write!(f, "large (7-30B)"),
            ModelSize::XLarge => write!(f, "xlarge (>30B)"),
        }
    }
}

pub fn run_init(args: InitArgs, level: LogLevel) -> Result<(), String> {
    // Resolve model source: --base takes precedence over --model
    let model_source = args.base.as_deref().or(args.model.as_deref());

    // Resolve template: --method overrides --template
    let template = if let Some(method) = &args.method {
        match method {
            TrainingMethod::Full => Template::Full,
            TrainingMethod::Lora => Template::Lora,
            TrainingMethod::Qlora => Template::Qlora,
        }
    } else {
        match args.template {
            InitTemplate::Minimal => Template::Minimal,
            InitTemplate::Lora => Template::Lora,
            InitTemplate::Qlora => Template::Qlora,
            InitTemplate::Full => Template::Full,
        }
    };

    // Detect model size if a base model is specified
    let model_size = model_source.map(estimate_model_size);
    let lora_rank = model_size.map(suggest_lora_rank);
    let lr = model_size.map(suggest_learning_rate);

    // Detect data format
    let data_format = args.data.as_deref().map(detect_data_format);

    // Log detected settings
    if let Some(source) = model_source {
        log(level, LogLevel::Normal, &format!("Model: {source}"));
    }
    if let Some(size) = model_size {
        log(
            level,
            LogLevel::Normal,
            &format!("Detected size: {size}, suggested LoRA rank: {}", lora_rank.unwrap_or(64)),
        );
    }
    if let Some(fmt) = data_format {
        if fmt != DataFormat::Unknown {
            log(level, LogLevel::Normal, &format!("Data format: {fmt}"));
        }
    }

    log(level, LogLevel::Normal, &format!("Generating {template:?} template for: {}", args.name));

    // Generate YAML manifest with smart defaults
    let yaml =
        generate_yaml(template, &args.name, model_source, args.data.as_deref(), lora_rank, lr);

    // Output to file or stdout
    if let Some(output_path) = &args.output {
        std::fs::write(output_path, &yaml).map_err(|e| format!("Failed to write file: {e}"))?;
        log(level, LogLevel::Normal, &format!("Manifest saved to: {}", output_path.display()));
    } else {
        println!("{yaml}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Model Size Detection ===

    #[test]
    fn test_estimate_model_size_small() {
        assert_eq!(estimate_model_size("Qwen/Qwen2.5-Coder-0.5B"), ModelSize::Small);
        assert_eq!(estimate_model_size("microsoft/phi-2-0.3B"), ModelSize::Small);
    }

    #[test]
    fn test_estimate_model_size_medium() {
        assert_eq!(estimate_model_size("meta-llama/Llama-3-7B"), ModelSize::Medium);
        assert_eq!(estimate_model_size("mistralai/Mistral-1.5B-Instruct"), ModelSize::Medium);
    }

    #[test]
    fn test_estimate_model_size_large() {
        assert_eq!(estimate_model_size("meta-llama/Llama-3-13B"), ModelSize::Large);
    }

    #[test]
    fn test_estimate_model_size_xlarge() {
        assert_eq!(estimate_model_size("meta-llama/Llama-3-70B"), ModelSize::XLarge);
    }

    #[test]
    fn test_estimate_model_size_unknown_defaults_medium() {
        assert_eq!(estimate_model_size("some-org/some-model"), ModelSize::Medium);
    }

    #[test]
    fn test_suggest_lora_rank() {
        assert_eq!(suggest_lora_rank(ModelSize::Small), 32);
        assert_eq!(suggest_lora_rank(ModelSize::Medium), 64);
        assert_eq!(suggest_lora_rank(ModelSize::Large), 128);
        assert_eq!(suggest_lora_rank(ModelSize::XLarge), 256);
    }

    #[test]
    fn test_suggest_learning_rate() {
        assert!((suggest_learning_rate(ModelSize::Small) - 3e-4).abs() < 1e-10);
        assert!((suggest_learning_rate(ModelSize::Medium) - 2e-4).abs() < 1e-10);
        assert!((suggest_learning_rate(ModelSize::Large) - 1e-4).abs() < 1e-10);
        assert!((suggest_learning_rate(ModelSize::XLarge) - 5e-5).abs() < 1e-10);
    }

    // === Data Format Detection ===

    #[test]
    fn test_detect_data_format_jsonl() {
        assert_eq!(detect_data_format("train.jsonl"), DataFormat::Jsonl);
        assert_eq!(detect_data_format("data/train.jsonlines"), DataFormat::Jsonl);
    }

    #[test]
    fn test_detect_data_format_parquet() {
        assert_eq!(detect_data_format("data.parquet"), DataFormat::Parquet);
        assert_eq!(detect_data_format("data.pq"), DataFormat::Parquet);
    }

    #[test]
    fn test_detect_data_format_csv() {
        assert_eq!(detect_data_format("train.csv"), DataFormat::Csv);
        assert_eq!(detect_data_format("train.tsv"), DataFormat::Csv);
    }

    #[test]
    fn test_detect_data_format_text() {
        assert_eq!(detect_data_format("corpus.txt"), DataFormat::Text);
    }

    #[test]
    fn test_detect_data_format_unknown() {
        assert_eq!(detect_data_format("data.bin"), DataFormat::Unknown);
        assert_eq!(detect_data_format("./my-data/"), DataFormat::Unknown);
    }

    #[test]
    fn test_data_format_display() {
        assert_eq!(format!("{}", DataFormat::Jsonl), "jsonl");
        assert_eq!(format!("{}", DataFormat::Parquet), "parquet");
        assert_eq!(format!("{}", DataFormat::Csv), "csv");
        assert_eq!(format!("{}", DataFormat::Text), "text");
        assert_eq!(format!("{}", DataFormat::Unknown), "unknown");
    }

    #[test]
    fn test_model_size_display() {
        assert_eq!(format!("{}", ModelSize::Small), "small (<1B)");
        assert_eq!(format!("{}", ModelSize::Medium), "medium (1-7B)");
        assert_eq!(format!("{}", ModelSize::Large), "large (7-30B)");
        assert_eq!(format!("{}", ModelSize::XLarge), "xlarge (>30B)");
    }

    // === Integration Tests ===

    #[test]
    fn test_run_init_with_base_flag() {
        let args = InitArgs {
            name: "test_project".to_string(),
            output: None,
            template: InitTemplate::Minimal,
            model: None,
            base: Some("Qwen/Qwen2.5-Coder-0.5B".to_string()),
            method: Some(TrainingMethod::Qlora),
            data: None,
        };

        let result = run_init(args, LogLevel::Quiet);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_init_method_overrides_template() {
        let args = InitArgs {
            name: "test_project".to_string(),
            output: None,
            template: InitTemplate::Minimal, // should be overridden by method
            model: None,
            base: Some("meta-llama/Llama-3-7B".to_string()),
            method: Some(TrainingMethod::Lora),
            data: Some("train.jsonl".to_string()),
        };

        let result = run_init(args, LogLevel::Quiet);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_init_base_overrides_model() {
        // --base should take precedence over --model
        let args = InitArgs {
            name: "test".to_string(),
            output: None,
            template: InitTemplate::Lora,
            model: Some("local-model.safetensors".to_string()),
            base: Some("Qwen/Qwen2.5-Coder-0.5B".to_string()),
            method: None,
            data: None,
        };

        let result = run_init(args, LogLevel::Quiet);
        assert!(result.is_ok());
    }

    #[test]
    fn test_categorize_param_count() {
        assert_eq!(categorize_param_count(0.5), ModelSize::Small);
        assert_eq!(categorize_param_count(1.0), ModelSize::Medium);
        assert_eq!(categorize_param_count(7.0), ModelSize::Medium);
        assert_eq!(categorize_param_count(13.0), ModelSize::Large);
        assert_eq!(categorize_param_count(30.0), ModelSize::Large);
        assert_eq!(categorize_param_count(70.0), ModelSize::XLarge);
    }
}
