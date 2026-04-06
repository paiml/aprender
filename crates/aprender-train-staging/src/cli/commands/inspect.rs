//! Inspect command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{InspectArgs, InspectMode};
use std::path::Path;

/// Inspect a SafeTensors model file
fn inspect_safetensors(path: &Path, level: LogLevel) -> Result<(), String> {
    use safetensors::SafeTensors;

    let data = std::fs::read(path).map_err(|e| format!("Failed to read file: {e}"))?;

    let tensors =
        SafeTensors::deserialize(&data).map_err(|e| format!("Failed to parse SafeTensors: {e}"))?;

    let tensor_names: Vec<String> = tensors.names().iter().map(|s| (*s).to_string()).collect();
    let total_params = count_total_parameters(&tensors, &tensor_names);
    let file_size = data.len();

    log_model_info(level, file_size, total_params, tensor_names.len());

    if level == LogLevel::Verbose {
        log_tensor_details(level, &tensors, &tensor_names);
    }

    Ok(())
}

/// Count total parameters across all tensors
fn count_total_parameters(tensors: &safetensors::SafeTensors<'_>, names: &[String]) -> u64 {
    let mut total: u64 = 0;
    for name in names {
        if let Ok(tensor) = tensors.tensor(name) {
            let params: u64 = tensor.shape().iter().product::<usize>() as u64;
            total += params;
        }
    }
    total
}

/// Log basic model information
fn log_model_info(level: LogLevel, file_size: usize, total_params: u64, tensor_count: usize) {
    log(level, LogLevel::Normal, "Model Information:");
    log(level, LogLevel::Normal, &format!("  File size: {:.2} MB", file_size as f64 / 1_000_000.0));
    log(level, LogLevel::Normal, &format!("  Parameters: {:.2}B", total_params as f64 / 1e9));
    log(level, LogLevel::Normal, &format!("  Tensors: {tensor_count}"));
}

/// Log detailed tensor information
fn log_tensor_details(level: LogLevel, tensors: &safetensors::SafeTensors<'_>, names: &[String]) {
    log(level, LogLevel::Verbose, "\nTensor Details:");
    for name in &names[..names.len().min(20)] {
        if let Ok(tensor) = tensors.tensor(name) {
            log(
                level,
                LogLevel::Verbose,
                &format!("  {}: {:?} ({:?})", name, tensor.shape(), tensor.dtype()),
            );
        }
    }
    if names.len() > 20 {
        log(level, LogLevel::Verbose, &format!("  ... and {} more tensors", names.len() - 20));
    }
}

/// Inspect a GGUF model file
fn inspect_gguf(path: &Path, level: LogLevel) -> Result<(), String> {
    let metadata = std::fs::metadata(path).map_err(|e| format!("Failed to read metadata: {e}"))?;

    log(level, LogLevel::Normal, "GGUF Model Information:");
    log(
        level,
        LogLevel::Normal,
        &format!("  File size: {:.2} MB", metadata.len() as f64 / 1_000_000.0),
    );
    log(level, LogLevel::Normal, "  Format: GGUF (llama.cpp compatible)");
    log(level, LogLevel::Normal, "  (Use llama.cpp for detailed GGUF inspection)");

    Ok(())
}

/// Inspect a data file (parquet or csv)
fn inspect_data_file(
    path: &Path,
    ext: &str,
    mode: InspectMode,
    z_threshold: f32,
    level: LogLevel,
) -> Result<(), String> {
    let metadata = std::fs::metadata(path).map_err(|e| format!("Failed to read metadata: {e}"))?;

    match mode {
        InspectMode::Summary => {
            log(level, LogLevel::Normal, "Data Summary:");
            log(
                level,
                LogLevel::Normal,
                &format!("  File size: {:.2} MB", metadata.len() as f64 / 1_000_000.0),
            );
            log(level, LogLevel::Normal, &format!("  Format: {ext}"));
        }
        InspectMode::Outliers => {
            log(
                level,
                LogLevel::Normal,
                &format!("Outlier Detection (z-threshold: {z_threshold}):"),
            );
            log(level, LogLevel::Normal, "  Load data with alimentar for outlier analysis");
        }
        InspectMode::Distribution => {
            log(level, LogLevel::Normal, "Distribution Statistics:");
            log(level, LogLevel::Normal, "  Load data with alimentar for distribution analysis");
        }
        InspectMode::Schema => {
            log(level, LogLevel::Normal, "Schema:");
            log(level, LogLevel::Normal, &format!("  Format: {ext}"));
        }
    }

    Ok(())
}

/// Get file extension as lowercase string
fn get_extension(path: &Path) -> &str {
    path.extension().and_then(|s| s.to_str()).unwrap_or("")
}

/// Inspect a LoRA adapter directory (ENT-LoRA-018)
fn inspect_lora_adapter(dir: &Path, level: LogLevel) -> Result<(), String> {
    let config_path = dir.join("adapter_config.json");
    let adapter_path = dir.join("adapter_model.safetensors");

    log(level, LogLevel::Normal, "LoRA Adapter:");
    log(level, LogLevel::Normal, &format!("  Directory: {}", dir.display()));

    // Read adapter_config.json
    if config_path.exists() {
        let config_str =
            std::fs::read_to_string(&config_path).map_err(|e| format!("Read config: {e}"))?;
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
            if let Some(rank) = config.get("r").and_then(serde_json::Value::as_u64) {
                log(level, LogLevel::Normal, &format!("  Rank: {rank}"));
            }
            if let Some(alpha) = config.get("lora_alpha").and_then(serde_json::Value::as_f64) {
                log(level, LogLevel::Normal, &format!("  Alpha: {alpha}"));
            }
            if let Some(modules) =
                config.get("target_modules").and_then(serde_json::Value::as_array)
            {
                let names: Vec<&str> =
                    modules.iter().filter_map(serde_json::Value::as_str).collect();
                log(level, LogLevel::Normal, &format!("  Target modules: {}", names.join(", ")));
            }
            if let Some(base) =
                config.get("base_model_name_or_path").and_then(serde_json::Value::as_str)
            {
                log(level, LogLevel::Normal, &format!("  Base model: {base}"));
            }
        }
    }

    // Read adapter_model.safetensors
    if adapter_path.exists() {
        let size = std::fs::metadata(&adapter_path).map(|m| m.len()).unwrap_or(0);
        log(level, LogLevel::Normal, &format!("  Adapter size: {:.2} MB", size as f64 / 1e6));

        let data = std::fs::read(&adapter_path).map_err(|e| format!("Read adapter: {e}"))?;
        if let Ok(tensors) = safetensors::SafeTensors::deserialize(&data) {
            let names: Vec<String> = tensors.names().iter().map(|s| (*s).to_string()).collect();
            log(level, LogLevel::Normal, &format!("  Adapter tensors: {}", names.len()));
            let total_params: u64 = names
                .iter()
                .filter_map(|n| tensors.tensor(n).ok())
                .map(|t| t.shape().iter().product::<usize>() as u64)
                .sum();
            log(level, LogLevel::Normal, &format!("  Trainable params: {total_params}"));
        }
    } else {
        log(level, LogLevel::Normal, "  (no adapter_model.safetensors found)");
    }

    Ok(())
}

pub fn run_inspect(args: InspectArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Normal, &format!("Inspecting: {}", args.input.display()));

    if !args.input.exists() {
        return Err(format!("File not found: {}", args.input.display()));
    }

    // ENT-LoRA-018: Check if this is a LoRA adapter directory
    if args.input.is_dir() && args.input.join("adapter_config.json").exists() {
        return inspect_lora_adapter(&args.input, level);
    }

    let ext = get_extension(&args.input);
    log(level, LogLevel::Normal, &format!("  Mode: {}", args.mode));

    match ext {
        "safetensors" => inspect_safetensors(&args.input, level),
        "gguf" => inspect_gguf(&args.input, level),
        "parquet" | "csv" => {
            inspect_data_file(&args.input, ext, args.mode, args.z_threshold, level)
        }
        _ => {
            if args.input.is_dir() {
                Err(format!(
                    "Directory {} does not contain adapter_config.json",
                    args.input.display()
                ))
            } else {
                Err(format!(
                    "Unsupported file format: {ext}. Use .safetensors, .gguf, .parquet, or .csv"
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_get_extension_safetensors() {
        let path = PathBuf::from("model.safetensors");
        assert_eq!(get_extension(&path), "safetensors");
    }

    #[test]
    fn test_get_extension_gguf() {
        let path = PathBuf::from("model.gguf");
        assert_eq!(get_extension(&path), "gguf");
    }

    #[test]
    fn test_get_extension_parquet() {
        let path = PathBuf::from("data.parquet");
        assert_eq!(get_extension(&path), "parquet");
    }

    #[test]
    fn test_get_extension_csv() {
        let path = PathBuf::from("data.csv");
        assert_eq!(get_extension(&path), "csv");
    }

    #[test]
    fn test_get_extension_none() {
        let path = PathBuf::from("noextension");
        assert_eq!(get_extension(&path), "");
    }

    #[test]
    fn test_run_inspect_file_not_found() {
        let args = InspectArgs {
            input: PathBuf::from("/nonexistent/path/model.safetensors"),
            mode: InspectMode::Summary,
            columns: None,
            z_threshold: 3.0,
        };
        let result = run_inspect(args, LogLevel::Normal);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("File not found"));
    }

    #[test]
    fn test_run_inspect_unsupported_format() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_inspect.xyz");
        std::fs::write(&path, "test").expect("file write should succeed");

        let args = InspectArgs {
            input: path.clone(),
            mode: InspectMode::Summary,
            columns: None,
            z_threshold: 3.0,
        };
        let result = run_inspect(args, LogLevel::Normal);

        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported file format"));
    }

    #[test]
    fn test_inspect_data_file_summary() {
        let temp = std::env::temp_dir().join("inspect_sum.csv");
        std::fs::write(&temp, "a,b\n1,2").expect("write csv");
        let r = inspect_data_file(&temp, "csv", InspectMode::Summary, 3.0, LogLevel::Normal);
        let _ = std::fs::remove_file(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_inspect_data_file_outliers() {
        let temp = std::env::temp_dir().join("inspect_out.csv");
        std::fs::write(&temp, "a,b\n1,2").expect("write csv");
        let r = inspect_data_file(&temp, "csv", InspectMode::Outliers, 2.5, LogLevel::Normal);
        let _ = std::fs::remove_file(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_inspect_data_file_distribution() {
        let temp = std::env::temp_dir().join("inspect_dist.csv");
        std::fs::write(&temp, "a,b\n1,2").expect("write csv");
        let r = inspect_data_file(&temp, "csv", InspectMode::Distribution, 3.0, LogLevel::Normal);
        let _ = std::fs::remove_file(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_inspect_data_file_schema() {
        let temp = std::env::temp_dir().join("inspect_sch.parquet");
        std::fs::write(&temp, "fake parquet").expect("write file");
        let r = inspect_data_file(&temp, "parquet", InspectMode::Schema, 3.0, LogLevel::Normal);
        let _ = std::fs::remove_file(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_inspect_gguf() {
        let temp = std::env::temp_dir().join("test_model.gguf");
        std::fs::write(&temp, "GGUF fake data 12345678").expect("write gguf");
        let r = inspect_gguf(&temp, LogLevel::Normal);
        let _ = std::fs::remove_file(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_count_total_parameters_single_scalar() {
        use safetensors::serialize;
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;
        // A scalar tensor with shape [1] = 1 parameter
        let data = [0u8; 4]; // 1 f32 = 4 bytes
        let tv = TensorView::new(Dtype::F32, vec![1], &data).unwrap();
        let tensors = vec![("scalar", tv)];
        let bytes = serialize(tensors, None).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let names: Vec<String> = st.names().iter().map(|s| (*s).to_string()).collect();
        let total = count_total_parameters(&st, &names);
        assert_eq!(total, 1);
    }

    #[test]
    fn test_count_total_parameters_with_data() {
        use safetensors::serialize;
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;
        let data = [0u8; 24];
        let tv = TensorView::new(Dtype::F32, vec![2, 3], &data).unwrap();
        let tensors = vec![("w", tv)];
        let bytes = serialize(tensors, None).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let names: Vec<String> = st.names().iter().map(|s| (*s).to_string()).collect();
        let total = count_total_parameters(&st, &names);
        assert_eq!(total, 6);
    }

    #[test]
    fn test_log_model_info() {
        log_model_info(LogLevel::Normal, 1_000_000, 500_000_000, 100);
    }

    #[test]
    fn test_log_model_info_verbose() {
        log_model_info(LogLevel::Verbose, 0, 0, 0);
    }

    #[test]
    fn test_run_inspect_csv_file() {
        let temp = std::env::temp_dir().join("ri_test.csv");
        std::fs::write(&temp, "col1,col2\nval1,val2").expect("write csv");
        let args = InspectArgs {
            input: temp.clone(),
            mode: InspectMode::Summary,
            columns: None,
            z_threshold: 3.0,
        };
        let r = run_inspect(args, LogLevel::Normal);
        let _ = std::fs::remove_file(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_run_inspect_gguf_file() {
        let temp = std::env::temp_dir().join("ri_test.gguf");
        std::fs::write(&temp, "GGUF fake").expect("write gguf");
        let args = InspectArgs {
            input: temp.clone(),
            mode: InspectMode::Summary,
            columns: None,
            z_threshold: 3.0,
        };
        let r = run_inspect(args, LogLevel::Normal);
        let _ = std::fs::remove_file(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_run_inspect_directory_no_adapter() {
        let temp = std::env::temp_dir().join("ri_dir_test");
        let _ = std::fs::create_dir_all(&temp);
        let args = InspectArgs {
            input: temp.clone(),
            mode: InspectMode::Summary,
            columns: None,
            z_threshold: 3.0,
        };
        let r = run_inspect(args, LogLevel::Normal);
        let _ = std::fs::remove_dir_all(&temp);
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("adapter_config.json"));
    }

    #[test]
    fn test_run_inspect_parquet_distribution() {
        let temp = std::env::temp_dir().join("ri_test.parquet");
        std::fs::write(&temp, "fake parquet data").expect("write");
        let args = InspectArgs {
            input: temp.clone(),
            mode: InspectMode::Distribution,
            columns: None,
            z_threshold: 3.0,
        };
        let r = run_inspect(args, LogLevel::Normal);
        let _ = std::fs::remove_file(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_run_inspect_outlier_mode() {
        let temp = std::env::temp_dir().join("ri_outlier.csv");
        std::fs::write(&temp, "x\n1\n2\n3").expect("write");
        let args = InspectArgs {
            input: temp.clone(),
            mode: InspectMode::Outliers,
            columns: None,
            z_threshold: 2.0,
        };
        let r = run_inspect(args, LogLevel::Normal);
        let _ = std::fs::remove_file(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_run_inspect_schema_mode() {
        let temp = std::env::temp_dir().join("ri_schema.csv");
        std::fs::write(&temp, "a,b\n1,2").expect("write");
        let args = InspectArgs {
            input: temp.clone(),
            mode: InspectMode::Schema,
            columns: None,
            z_threshold: 3.0,
        };
        let r = run_inspect(args, LogLevel::Normal);
        let _ = std::fs::remove_file(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_inspect_lora_adapter_empty_dir() {
        let temp = std::env::temp_dir().join("ri_lora_empty");
        let _ = std::fs::create_dir_all(&temp);
        std::fs::write(temp.join("adapter_config.json"), "{}").expect("write config");
        let r = inspect_lora_adapter(&temp, LogLevel::Normal);
        let _ = std::fs::remove_dir_all(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_inspect_lora_adapter_with_config() {
        let temp = std::env::temp_dir().join("ri_lora_cfg");
        let _ = std::fs::create_dir_all(&temp);
        let config = serde_json::json!({
            "r": 16,
            "lora_alpha": 32.0,
            "target_modules": ["q_proj", "v_proj"],
            "base_model_name_or_path": "test/model"
        });
        std::fs::write(temp.join("adapter_config.json"), config.to_string()).expect("write");
        let r = inspect_lora_adapter(&temp, LogLevel::Normal);
        let _ = std::fs::remove_dir_all(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_run_inspect_lora_adapter_dir() {
        let temp = std::env::temp_dir().join("ri_lora_run");
        let _ = std::fs::create_dir_all(&temp);
        std::fs::write(temp.join("adapter_config.json"), "{}").expect("write");
        let args = InspectArgs {
            input: temp.clone(),
            mode: InspectMode::Summary,
            columns: None,
            z_threshold: 3.0,
        };
        let r = run_inspect(args, LogLevel::Normal);
        let _ = std::fs::remove_dir_all(&temp);
        assert!(r.is_ok());
    }

    #[test]
    fn test_get_extension_compound() {
        let p = PathBuf::from("model.v2.safetensors");
        assert_eq!(get_extension(&p), "safetensors");
    }

    #[test]
    fn test_get_extension_dotfile() {
        let p = PathBuf::from(".hidden");
        assert_eq!(get_extension(&p), "");
    }
}
