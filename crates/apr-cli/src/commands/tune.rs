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
pub fn run(
    model_path: Option<&Path>,
    method: TuneMethod,
    rank: Option<u32>,
    vram_gb: f64,
    plan_only: bool,
    model_size: Option<&str>,
    _freeze_base: bool,
    _train_data: Option<&Path>,
    json_output: bool,
) -> Result<(), CliError> {
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

/// Estimate parameters from model file size
fn estimate_params_from_file(path: &Path) -> Result<u64, CliError> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model file: {e}")))?;

    let size_bytes = metadata.len();

    // Rough estimation: 2 bytes per param for fp16, 0.5 for Q4
    // Use conservative estimate (assume Q4)
    let estimated_params = size_bytes * 2; // Q4: ~0.5 bytes/param

    Ok(estimated_params)
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
mod tests {
    use super::*;
    use std::fs;

    // =========================================================================
    // TuneMethod tests
    // =========================================================================

    #[test]
    fn test_tune_method_parse() {
        assert!(matches!(
            "lora".parse::<TuneMethod>().unwrap(),
            TuneMethod::LoRA
        ));
        assert!(matches!(
            "qlora".parse::<TuneMethod>().unwrap(),
            TuneMethod::QLoRA
        ));
        assert!(matches!(
            "auto".parse::<TuneMethod>().unwrap(),
            TuneMethod::Auto
        ));
        assert!(matches!(
            "full".parse::<TuneMethod>().unwrap(),
            TuneMethod::Full
        ));
    }

    #[test]
    fn test_tune_method_parse_case_insensitive() {
        assert!(matches!(
            "LORA".parse::<TuneMethod>().unwrap(),
            TuneMethod::LoRA
        ));
        assert!(matches!(
            "LoRa".parse::<TuneMethod>().unwrap(),
            TuneMethod::LoRA
        ));
        assert!(matches!(
            "QLORA".parse::<TuneMethod>().unwrap(),
            TuneMethod::QLoRA
        ));
    }

    #[test]
    fn test_tune_method_parse_invalid() {
        let result: Result<TuneMethod, _> = "invalid".parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown method"));
    }

    #[test]
    fn test_tune_method_default() {
        let method = TuneMethod::default();
        assert!(matches!(method, TuneMethod::Auto));
    }

    #[test]
    fn test_tune_method_debug() {
        assert_eq!(format!("{:?}", TuneMethod::Auto), "Auto");
        assert_eq!(format!("{:?}", TuneMethod::Full), "Full");
        assert_eq!(format!("{:?}", TuneMethod::LoRA), "LoRA");
        assert_eq!(format!("{:?}", TuneMethod::QLoRA), "QLoRA");
    }

    #[test]
    fn test_tune_method_clone() {
        let method = TuneMethod::LoRA;
        let cloned = method;
        assert!(matches!(cloned, TuneMethod::LoRA));
    }

    #[test]
    fn test_tune_method_copy() {
        let method = TuneMethod::QLoRA;
        let copied: TuneMethod = method;
        assert!(matches!(method, TuneMethod::QLoRA));
        assert!(matches!(copied, TuneMethod::QLoRA));
    }

    #[test]
    fn test_tune_method_into_entrenar_method() {
        let auto: Method = TuneMethod::Auto.into();
        assert!(matches!(auto, Method::Auto));

        let full: Method = TuneMethod::Full.into();
        assert!(matches!(full, Method::Full));

        let lora: Method = TuneMethod::LoRA.into();
        assert!(matches!(lora, Method::LoRA));

        let qlora: Method = TuneMethod::QLoRA.into();
        assert!(matches!(qlora, Method::QLoRA));
    }

    // =========================================================================
    // parse_model_size tests
    // =========================================================================

    #[test]
    fn test_parse_model_size() {
        assert_eq!(parse_model_size("7B").unwrap(), 7_000_000_000);
        assert_eq!(parse_model_size("1.5B").unwrap(), 1_500_000_000);
        assert_eq!(parse_model_size("70B").unwrap(), 70_000_000_000);
        assert_eq!(parse_model_size("500M").unwrap(), 500_000_000);
    }

    #[test]
    fn test_parse_model_size_case_insensitive() {
        assert_eq!(parse_model_size("7b").unwrap(), 7_000_000_000);
        assert_eq!(parse_model_size("1.5b").unwrap(), 1_500_000_000);
    }

    #[test]
    fn test_parse_model_size_invalid() {
        assert!(parse_model_size("7").is_err());
        assert!(parse_model_size("7GB").is_err());
        assert!(parse_model_size("abc").is_err());
    }

    #[test]
    fn test_parse_model_size_decimal() {
        assert_eq!(parse_model_size("0.5B").unwrap(), 500_000_000);
        assert_eq!(parse_model_size("2.7B").unwrap(), 2_700_000_000);
        assert_eq!(parse_model_size("13.5B").unwrap(), 13_500_000_000);
    }

    #[test]
    fn test_parse_model_size_millions() {
        assert_eq!(parse_model_size("125M").unwrap(), 125_000_000);
        assert_eq!(parse_model_size("350M").unwrap(), 350_000_000);
        assert_eq!(parse_model_size("1000M").unwrap(), 1_000_000_000);
    }

    #[test]
    fn test_parse_model_size_large() {
        assert_eq!(parse_model_size("180B").unwrap(), 180_000_000_000);
        assert_eq!(parse_model_size("405B").unwrap(), 405_000_000_000);
    }

    #[test]
    fn test_parse_model_size_invalid_number() {
        let result = parse_model_size("abcB");
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Invalid number"));
            }
            other => panic!("Expected ValidationFailed, got {:?}", other),
        }
    }

    // =========================================================================
    // format_params tests
    // =========================================================================

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(7_000_000_000), "7.0B");
        assert_eq!(format_params(1_500_000_000), "1.5B");
        assert_eq!(format_params(500_000_000), "500.0M");
    }

    #[test]
    fn test_format_params_small() {
        assert_eq!(format_params(100_000), "100000");
        assert_eq!(format_params(999_999), "999999");
    }

    #[test]
    fn test_format_params_millions() {
        assert_eq!(format_params(1_000_000), "1.0M");
        assert_eq!(format_params(125_000_000), "125.0M");
        assert_eq!(format_params(999_999_999), "1000.0M");
    }

    #[test]
    fn test_format_params_billions() {
        assert_eq!(format_params(1_000_000_000), "1.0B");
        assert_eq!(format_params(70_000_000_000), "70.0B");
        assert_eq!(format_params(405_000_000_000), "405.0B");
    }

    // =========================================================================
    // estimate_params_from_file tests
    // =========================================================================

    #[test]
    fn test_estimate_params_from_file() {
        let temp_dir = std::env::temp_dir().join("apr_tune_test");
        let _ = fs::create_dir_all(&temp_dir);

        // Create a 1MB file
        let test_file = temp_dir.join("test_model.bin");
        let data = vec![0u8; 1_000_000];
        let _ = fs::write(&test_file, &data);

        let params = estimate_params_from_file(&test_file).unwrap();
        // 1MB file * 2 (Q4 estimate) = 2M params
        assert_eq!(params, 2_000_000);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_estimate_params_from_file_not_found() {
        let result = estimate_params_from_file(Path::new("/nonexistent/model.bin"));
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Cannot read model file"));
            }
            other => panic!("Expected ValidationFailed, got {:?}", other),
        }
    }

    // =========================================================================
    // run() error cases tests
    // =========================================================================

    #[test]
    fn test_run_no_model_or_size() {
        let result = run(
            None, // No model path
            TuneMethod::Auto,
            None,
            16.0,
            true,
            None, // No model size
            false,
            None,
            false,
        );

        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Either --model or model path required"));
            }
            other => panic!("Expected ValidationFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_run_with_model_size() {
        let result = run(
            None,
            TuneMethod::LoRA,
            Some(8),
            24.0,
            true,
            Some("7B"),
            false,
            None,
            false,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_model_size_json_output() {
        let result = run(
            None,
            TuneMethod::QLoRA,
            Some(16),
            16.0,
            true,
            Some("1.5B"),
            false,
            None,
            true, // JSON output
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_run_plan_only() {
        let result = run(
            None,
            TuneMethod::Auto,
            None,
            8.0,
            true, // plan_only
            Some("3B"),
            false,
            None,
            false,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_rank() {
        let result = run(
            None,
            TuneMethod::LoRA,
            Some(4), // rank
            16.0,
            true,
            Some("7B"),
            false,
            None,
            false,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_model_file() {
        let temp_dir = std::env::temp_dir().join("apr_tune_run_test");
        let _ = fs::create_dir_all(&temp_dir);

        // Create a test model file (small for fast tests)
        let test_file = temp_dir.join("test_model.gguf");
        let data = vec![0u8; 100_000]; // 100KB
        let _ = fs::write(&test_file, &data);

        let result = run(
            Some(&test_file),
            TuneMethod::QLoRA,
            None,
            8.0,
            true,
            None,
            false,
            None,
            false,
        );

        assert!(result.is_ok());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_run_model_file_not_found() {
        let result = run(
            Some(Path::new("/nonexistent/model.gguf")),
            TuneMethod::Auto,
            None,
            16.0,
            true,
            None,
            false,
            None,
            false,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_model_size() {
        let result = run(
            None,
            TuneMethod::Auto,
            None,
            16.0,
            true,
            Some("invalid"), // Invalid size format
            false,
            None,
            false,
        );

        assert!(result.is_err());
    }
}
