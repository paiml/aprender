//! Info command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{load_config, InfoArgs, OutputFormat};

pub fn run_info(args: InfoArgs, level: LogLevel) -> Result<(), String> {
    let spec = load_config(&args.config).map_err(|e| format!("Config error: {e}"))?;

    match args.format {
        OutputFormat::Text => {
            log(level, LogLevel::Normal, "Configuration Info:");
            println!();
            println!("Model: {}", spec.model.path.display());
            println!("Optimizer: {} (lr={})", spec.optimizer.name, spec.optimizer.lr);
            println!("Epochs: {}", spec.training.epochs);
            println!("Batch size: {}", spec.data.batch_size);

            if spec.lora.is_some() {
                println!("LoRA: enabled");
            }
            if spec.quantize.is_some() {
                println!("Quantization: enabled");
            }
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&spec)
                .map_err(|e| format!("JSON serialization error: {e}"))?;
            println!("{json}");
        }
        OutputFormat::Yaml => {
            let yaml = serde_yaml::to_string(&spec)
                .map_err(|e| format!("YAML serialization error: {e}"))?;
            println!("{yaml}");
        }
    }

    Ok(())
}
