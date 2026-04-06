//! Merge command implementation

use crate::autograd::Tensor;
use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{MergeArgs, MergeMethod};
use crate::merge::{
    dare_merge, ensemble_merge, slerp_merge, ties_merge, DareConfig, EnsembleConfig, Model,
    SlerpConfig, TiesConfig,
};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

pub fn run_merge(args: MergeArgs, level: LogLevel) -> Result<(), String> {
    // ENT-LoRA-017: LoRA adapter merge path
    if args.method == MergeMethod::LoraAdapter {
        return run_lora_adapter_merge(&args, level);
    }

    log_merge_start(&args, level);
    validate_model_count(&args)?;

    let models = load_all_models(&args.models, level)?;
    let merged = perform_merge(&models, &args)?;
    export_merged_model(&merged, &args)?;

    log_merge_complete(&merged, &args, level);
    Ok(())
}

/// Log merge operation start
fn log_merge_start(args: &MergeArgs, level: LogLevel) {
    log(
        level,
        LogLevel::Normal,
        &format!("Merging {} models using {:?}", args.models.len(), args.method),
    );

    for (i, model) in args.models.iter().enumerate() {
        log(level, LogLevel::Verbose, &format!("  Model {}: {}", i + 1, model.display()));
    }
    log(level, LogLevel::Verbose, &format!("  Output: {}", args.output.display()));
}

/// Validate we have enough models
fn validate_model_count(args: &MergeArgs) -> Result<(), String> {
    if args.models.len() < 2 {
        return Err("Need at least 2 models to merge".to_string());
    }
    Ok(())
}

/// Load all models from paths
fn load_all_models(paths: &[std::path::PathBuf], level: LogLevel) -> Result<Vec<Model>, String> {
    let mut models: Vec<Model> = Vec::new();
    for path in paths {
        let model = load_single_model(path)?;
        let tensor_count = model.len();
        models.push(model);

        log(
            level,
            LogLevel::Verbose,
            &format!("  Loaded {} tensors from {}", tensor_count, path.display()),
        );
    }
    Ok(models)
}

/// Load a single model from a SafeTensors file
fn load_single_model(path: &Path) -> Result<Model, String> {
    let data =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;

    let mut model: Model = HashMap::new();
    for name in tensors.names() {
        if let Some(tensor) = extract_f32_tensor(&tensors, name)? {
            model.insert((*name).to_string(), tensor);
        }
    }
    Ok(model)
}

/// Extract a tensor as f32 values (returns None for non-F32 tensors)
fn extract_f32_tensor(tensors: &SafeTensors<'_>, name: &str) -> Result<Option<Tensor>, String> {
    let tensor = tensors.tensor(name).map_err(|e| format!("Failed to get tensor {name}: {e}"))?;

    if tensor.dtype() != safetensors::tensor::Dtype::F32 {
        return Ok(None);
    }

    let bytes = tensor.data();
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(Some(Tensor::from_vec(values, false)))
}

/// Perform the merge based on the specified method
fn perform_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    match args.method {
        MergeMethod::Ties => perform_ties_merge(models, args),
        MergeMethod::Dare => perform_dare_merge(models, args),
        MergeMethod::Slerp => perform_slerp_merge(models, args),
        MergeMethod::Average => perform_average_merge(models, args),
        MergeMethod::LoraAdapter => {
            // Handled by early return in run_merge; shouldn't reach here
            Err("LoRA adapter merge uses dedicated path".to_string())
        }
    }
}

/// TIES merge: first model is base, rest are task-specific
fn perform_ties_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    let config = TiesConfig { density: args.density.unwrap_or(0.2) };
    let base = &models[0];
    ties_merge(models.get(1..).unwrap_or_default(), base, &config)
        .map_err(|e| format!("TIES merge failed: {e}"))
}

/// DARE merge with dropout
fn perform_dare_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    let config = DareConfig { drop_prob: 1.0 - args.density.unwrap_or(0.5), seed: None };
    let base = &models[0];
    dare_merge(models.get(1..).unwrap_or_default(), base, &config)
        .map_err(|e| format!("DARE merge failed: {e}"))
}

/// SLERP merge (requires exactly 2 models)
fn perform_slerp_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    if models.len() != 2 {
        return Err("SLERP requires exactly 2 models".to_string());
    }
    let config = SlerpConfig { t: args.weight.unwrap_or(0.5) };
    slerp_merge(&models[0], &models[1], &config).map_err(|e| format!("SLERP merge failed: {e}"))
}

/// Average/ensemble merge with optional weights
fn perform_average_merge(models: &[Model], args: &MergeArgs) -> Result<Model, String> {
    let config = build_ensemble_config(args)?;
    ensemble_merge(models, &config).map_err(|e| format!("Average merge failed: {e}"))
}

/// Build ensemble config from args
fn build_ensemble_config(args: &MergeArgs) -> Result<EnsembleConfig, String> {
    if let Some(w_str) = &args.weights {
        let weights: Vec<f32> = w_str
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Invalid weights: {e}"))?;
        Ok(EnsembleConfig::weighted_average(weights))
    } else {
        Ok(EnsembleConfig::uniform_average())
    }
}

/// Export merged model to file
fn export_merged_model(merged: &Model, args: &MergeArgs) -> Result<(), String> {
    let output_ext = args.output.extension().and_then(|s| s.to_str()).unwrap_or("json");

    if output_ext == "safetensors" {
        export_safetensors(merged, args)
    } else {
        export_json(merged, args)
    }
}

/// Export to SafeTensors format
fn export_safetensors(merged: &Model, args: &MergeArgs) -> Result<(), String> {
    use safetensors::tensor::{Dtype, TensorView};

    let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = merged
        .iter()
        .map(|(name, tensor)| {
            let data = tensor.data();
            let bytes: Vec<u8> = bytemuck::cast_slice(data.as_slice().unwrap_or(&[])).to_vec();
            let shape = vec![tensor.len()];
            (name.clone(), bytes, shape)
        })
        .collect();

    let views: Vec<(&str, TensorView<'_>)> = tensor_data
        .iter()
        .filter_map(|(name, bytes, shape)| {
            TensorView::new(Dtype::F32, shape.clone(), bytes).ok().map(|view| (name.as_str(), view))
        })
        .collect();

    let metadata = build_safetensor_metadata(merged, args);
    let safetensor_bytes = safetensors::serialize(views, Some(metadata))
        .map_err(|e| format!("Failed to serialize SafeTensors: {e}"))?;

    std::fs::write(&args.output, safetensor_bytes)
        .map_err(|e| format!("Failed to write output: {e}"))
}

/// Build SafeTensors metadata
fn build_safetensor_metadata(merged: &Model, args: &MergeArgs) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    metadata.insert("name".to_string(), "merged-model".to_string());
    metadata.insert("merge_method".to_string(), format!("{:?}", args.method));
    metadata.insert("tensor_count".to_string(), merged.len().to_string());
    metadata
}

/// Export to JSON format
fn export_json(merged: &Model, args: &MergeArgs) -> Result<(), String> {
    let output_data: HashMap<String, Vec<f32>> =
        merged.iter().map(|(name, tensor)| (name.clone(), tensor.data().to_vec())).collect();

    let json_data =
        serde_json::to_vec_pretty(&output_data).map_err(|e| format!("Failed to serialize: {e}"))?;

    std::fs::write(&args.output, &json_data).map_err(|e| format!("Failed to write output: {e}"))
}

/// Log merge completion
fn log_merge_complete(merged: &Model, args: &MergeArgs, level: LogLevel) {
    log(
        level,
        LogLevel::Normal,
        &format!("Merge complete: {} tensors written to {}", merged.len(), args.output.display()),
    );
}

/// Merge LoRA adapter into base model (ENT-LoRA-017)
///
/// Computes W_merged = W_base + scale * B @ A for each adapted module,
/// producing a standard safetensors model with no LoRA tensors.
fn run_lora_adapter_merge(args: &MergeArgs, level: LogLevel) -> Result<(), String> {
    let base_path = args.base.as_ref().ok_or("--base required for lora-adapter merge")?;
    let adapter_dir = args.adapter.as_ref().ok_or("--adapter required for lora-adapter merge")?;

    let config_path = adapter_dir.join("adapter_config.json");
    let adapter_path = adapter_dir.join("adapter_model.safetensors");

    if !base_path.exists() {
        return Err(format!("Base model not found: {}", base_path.display()));
    }
    if !config_path.exists() {
        return Err(format!("adapter_config.json not found in {}", adapter_dir.display()));
    }
    if !adapter_path.exists() {
        return Err(format!("adapter_model.safetensors not found in {}", adapter_dir.display()));
    }

    log(level, LogLevel::Normal, "LoRA adapter merge:");
    log(level, LogLevel::Normal, &format!("  Base: {}", base_path.display()));
    log(level, LogLevel::Normal, &format!("  Adapter: {}", adapter_dir.display()));

    // Read adapter config
    let config_str =
        std::fs::read_to_string(&config_path).map_err(|e| format!("Read adapter config: {e}"))?;
    let config: serde_json::Value =
        serde_json::from_str(&config_str).map_err(|e| format!("Parse adapter config: {e}"))?;

    let rank = config.get("r").and_then(serde_json::Value::as_u64).unwrap_or(8) as usize;
    let alpha =
        config.get("lora_alpha").and_then(serde_json::Value::as_f64).unwrap_or(rank as f64 * 2.0);
    let scale = alpha as f32 / rank as f32;

    log(level, LogLevel::Normal, &format!("  Rank: {rank}, Alpha: {alpha}, Scale: {scale:.4}"));

    // Load base model
    let base_data = std::fs::read(base_path).map_err(|e| format!("Read base model: {e}"))?;
    let base_tensors =
        SafeTensors::deserialize(&base_data).map_err(|e| format!("Parse base model: {e}"))?;

    // Load adapter
    let adapter_data = std::fs::read(&adapter_path).map_err(|e| format!("Read adapter: {e}"))?;
    let adapter_tensors =
        SafeTensors::deserialize(&adapter_data).map_err(|e| format!("Parse adapter: {e}"))?;

    // Merge: copy all base tensors, apply LoRA delta where adapters exist
    let adapter_names: Vec<String> =
        adapter_tensors.names().iter().map(|s| (*s).to_string()).collect();
    let base_names: Vec<String> = base_tensors.names().iter().map(|s| (*s).to_string()).collect();

    // Build map of adapter A/B pairs grouped by module path
    let lora_pairs = build_lora_pairs(&adapter_names, &adapter_tensors)?;
    let mut merged_count = 0usize;

    // Prepare output tensors
    let mut output_tensors: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

    for name in &base_names {
        let base_t = base_tensors.tensor(name).map_err(|e| format!("Get tensor {name}: {e}"))?;
        let shape: Vec<usize> = base_t.shape().to_vec();

        // Check if this weight has a LoRA adapter
        if let Some((a_data, b_data, a_shape, b_shape)) = lora_pairs.get(name.as_str()) {
            // W_merged = W_base + scale * B @ A
            let base_f32 = bytes_to_f32(base_t.data(), base_t.dtype());
            let a_f32 = bytes_to_f32(a_data, safetensors::tensor::Dtype::F32);
            let b_f32 = bytes_to_f32(b_data, safetensors::tensor::Dtype::F32);

            let d_out = b_shape[0];
            let r = b_shape[1];
            let d_in = a_shape[1];

            // Compute B @ A: [d_out, r] @ [r, d_in] -> [d_out, d_in]
            let mut ba = vec![0.0f32; d_out * d_in];
            for i in 0..d_out {
                for j in 0..d_in {
                    let mut sum = 0.0f32;
                    for k in 0..r {
                        sum += b_f32[i * r + k] * a_f32[k * d_in + j];
                    }
                    ba[i * d_in + j] = sum;
                }
            }

            // W_merged = W_base + scale * BA
            let mut merged: Vec<f32> = base_f32;
            for (i, val) in merged.iter_mut().enumerate() {
                *val += scale * ba[i];
            }

            let bytes: Vec<u8> = bytemuck::cast_slice(&merged).to_vec();
            output_tensors.push((name.clone(), bytes, shape));
            merged_count += 1;
        } else {
            // Pass through base tensor unchanged
            output_tensors.push((name.clone(), base_t.data().to_vec(), shape));
        }
    }

    // Serialize to safetensors
    let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = output_tensors
        .iter()
        .filter_map(|(name, bytes, shape)| {
            safetensors::tensor::TensorView::new(
                safetensors::tensor::Dtype::F32,
                shape.clone(),
                bytes,
            )
            .ok()
            .map(|view| (name.as_str(), view))
        })
        .collect();

    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "entrenar-merged-lora".to_string());
    metadata.insert("lora_rank".to_string(), rank.to_string());
    metadata.insert("lora_alpha".to_string(), format!("{alpha}"));

    let safetensor_bytes = safetensors::serialize(views, Some(metadata))
        .map_err(|e| format!("Serialize merged model: {e}"))?;

    std::fs::write(&args.output, safetensor_bytes)
        .map_err(|e| format!("Write merged model: {e}"))?;

    let output_size = std::fs::metadata(&args.output).map(|m| m.len()).unwrap_or(0);
    log(
        level,
        LogLevel::Normal,
        &format!("  Merged {merged_count} adapter weights into base model"),
    );
    log(
        level,
        LogLevel::Normal,
        &format!("  Output: {} ({:.2} MB)", args.output.display(), output_size as f64 / 1e6),
    );

    Ok(())
}

/// Build a map of base weight name -> (A_data, B_data, A_shape, B_shape)
fn build_lora_pairs<'a>(
    names: &[String],
    tensors: &'a SafeTensors<'a>,
) -> Result<HashMap<&'a str, (Vec<u8>, Vec<u8>, Vec<usize>, Vec<usize>)>, String> {
    let mut pairs: HashMap<String, (Option<(Vec<u8>, Vec<usize>)>, Option<(Vec<u8>, Vec<usize>)>)> =
        HashMap::new();

    for name in names {
        // PEFT naming: base_model.model.{path}.lora_A.weight / lora_B.weight
        let (base_name, is_a) = if let Some(stripped) = name.strip_suffix(".lora_A.weight") {
            (stripped.replace("base_model.model.", "") + ".weight", true)
        } else if let Some(stripped) = name.strip_suffix(".lora_B.weight") {
            (stripped.replace("base_model.model.", "") + ".weight", false)
        } else {
            continue;
        };

        let tensor = tensors.tensor(name).map_err(|e| format!("Get adapter tensor {name}: {e}"))?;
        let data = tensor.data().to_vec();
        let shape = tensor.shape().to_vec();

        let entry = pairs.entry(base_name).or_insert((None, None));
        if is_a {
            entry.0 = Some((data, shape));
        } else {
            entry.1 = Some((data, shape));
        }
    }

    let mut result = HashMap::new();
    for (base_name, (a, b)) in &pairs {
        if let (Some((a_data, a_shape)), Some((b_data, b_shape))) = (a, b) {
            // Leak the base_name string to get a &'a str — safe in this context
            // as the result lives only for the merge duration
            let key: &str = Box::leak(base_name.clone().into_boxed_str());
            result.insert(key, (a_data.clone(), b_data.clone(), a_shape.clone(), b_shape.clone()));
        }
    }
    Ok(result)
}

/// Convert tensor bytes to f32 based on dtype
fn bytes_to_f32(data: &[u8], dtype: safetensors::tensor::Dtype) -> Vec<f32> {
    match dtype {
        safetensors::tensor::Dtype::F32 => {
            data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        safetensors::tensor::Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect(),
        safetensors::tensor::Dtype::BF16 => data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::bf16::from_bits(bits).to_f32()
            })
            .collect(),
        _ => {
            // For other dtypes, treat as f32 (best effort)
            data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_validate_model_count_zero() {
        let args = MergeArgs {
            models: vec![],
            output: PathBuf::from("o.json"),
            method: MergeMethod::Ties,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(validate_model_count(&args).is_err());
    }

    #[test]
    fn test_validate_model_count_two_ok() {
        let args = MergeArgs {
            models: vec![PathBuf::from("a"), PathBuf::from("b")],
            output: PathBuf::from("o.json"),
            method: MergeMethod::Ties,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(validate_model_count(&args).is_ok());
    }

    #[test]
    fn test_build_ensemble_config_no_weights() {
        let args = MergeArgs {
            models: vec![],
            output: PathBuf::from("o.json"),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(build_ensemble_config(&args).is_ok());
    }

    #[test]
    fn test_build_ensemble_config_with_weights() {
        let args = MergeArgs {
            models: vec![],
            output: PathBuf::from("o.json"),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: Some("0.3, 0.7".into()),
            base: None,
            adapter: None,
        };
        assert!(build_ensemble_config(&args).is_ok());
    }

    #[test]
    fn test_build_ensemble_config_invalid() {
        let args = MergeArgs {
            models: vec![],
            output: PathBuf::from("o.json"),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: Some("abc".into()),
            base: None,
            adapter: None,
        };
        assert!(build_ensemble_config(&args).unwrap_err().contains("Invalid weights"));
    }

    fn mk(keys: &[(&str, &[f32])]) -> Model {
        keys.iter().map(|(n, v)| (n.to_string(), Tensor::from_vec(v.to_vec(), false))).collect()
    }

    #[test]
    fn test_slerp_wrong_count() {
        let ms = vec![mk(&[("w", &[1.0])]), mk(&[("w", &[2.0])]), mk(&[("w", &[3.0])])];
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Slerp,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(perform_slerp_merge(&ms, &a).unwrap_err().contains("SLERP requires exactly 2"));
    }

    #[test]
    fn test_merge_lora_err() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::LoraAdapter,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(perform_merge(&[], &a).is_err());
    }

    #[test]
    fn test_bytes_to_f32_f32() {
        let v = vec![1.0f32, 2.5];
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let r = bytes_to_f32(&b, safetensors::tensor::Dtype::F32);
        assert!((r[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bytes_to_f32_f16() {
        let b = half::f16::from_f32(1.0).to_le_bytes().to_vec();
        assert!((bytes_to_f32(&b, safetensors::tensor::Dtype::F16)[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bytes_to_f32_bf16() {
        let b = half::bf16::from_f32(2.0).to_le_bytes().to_vec();
        assert!((bytes_to_f32(&b, safetensors::tensor::Dtype::BF16)[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_bytes_to_f32_fallback() {
        let b: Vec<u8> = 42.0f32.to_le_bytes().to_vec();
        assert!((bytes_to_f32(&b, safetensors::tensor::Dtype::I8)[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_bytes_to_f32_empty() {
        assert!(bytes_to_f32(&[], safetensors::tensor::Dtype::F32).is_empty());
    }

    #[test]
    fn test_safetensor_metadata() {
        let m = mk(&[("a", &[1.0]), ("b", &[2.0])]);
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o.st"),
            method: MergeMethod::Dare,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        let md = build_safetensor_metadata(&m, &a);
        assert_eq!(md["name"], "merged-model");
        assert_eq!(md["tensor_count"], "2");
    }

    #[test]
    fn test_export_json() {
        let m = mk(&[("w", &[1.0])]);
        let t = std::env::temp_dir().join("ent_merge_j.json");
        let a = MergeArgs {
            models: vec![],
            output: t.clone(),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(export_merged_model(&m, &a).is_ok());
        let _ = std::fs::remove_file(&t);
    }

    #[test]
    fn test_export_safetensors() {
        let m = mk(&[("w", &[1.0])]);
        let t = std::env::temp_dir().join("ent_merge_s.safetensors");
        let a = MergeArgs {
            models: vec![],
            output: t.clone(),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(export_merged_model(&m, &a).is_ok());
        let _ = std::fs::remove_file(&t);
    }

    #[test]
    fn test_ties_merge_ok() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Ties,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        // ties_merge needs base + at least 2 delta models (3 total)
        assert!(perform_ties_merge(
            &[mk(&[("w", &[1.0, 2.0])]), mk(&[("w", &[1.1, 2.1])]), mk(&[("w", &[1.2, 2.2])]),],
            &a
        )
        .is_ok());
    }

    #[test]
    fn test_dare_merge_ok() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Dare,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(
            perform_dare_merge(&[mk(&[("w", &[1.0, 2.0])]), mk(&[("w", &[1.1, 2.1])])], &a).is_ok()
        );
    }

    #[test]
    fn test_average_merge() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        let r = perform_average_merge(&[mk(&[("w", &[2.0, 4.0])]), mk(&[("w", &[6.0, 8.0])])], &a)
            .unwrap();
        let s = r["w"].data().as_slice().unwrap().to_vec();
        assert!((s[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_merge_no_panic() {
        let a = MergeArgs {
            models: vec![PathBuf::from("a"), PathBuf::from("b")],
            output: PathBuf::from("o"),
            method: MergeMethod::Ties,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        log_merge_start(&a, LogLevel::Quiet);
        log_merge_start(&a, LogLevel::Verbose);
        log_merge_complete(&mk(&[("w", &[1.0])]), &a, LogLevel::Normal);
    }

    #[test]
    fn test_lora_missing_base() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::LoraAdapter,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: Some(PathBuf::from("/tmp")),
        };
        assert!(run_lora_adapter_merge(&a, LogLevel::Quiet)
            .unwrap_err()
            .contains("--base required"));
    }

    #[test]
    fn test_lora_missing_adapter() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::LoraAdapter,
            weight: None,
            density: None,
            weights: None,
            base: Some(PathBuf::from("/tmp/x")),
            adapter: None,
        };
        assert!(run_lora_adapter_merge(&a, LogLevel::Quiet)
            .unwrap_err()
            .contains("--adapter required"));
    }

    #[test]
    fn test_lora_base_not_found() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::LoraAdapter,
            weight: None,
            density: None,
            weights: None,
            base: Some(PathBuf::from("/no/base")),
            adapter: Some(PathBuf::from("/tmp")),
        };
        assert!(run_lora_adapter_merge(&a, LogLevel::Quiet)
            .unwrap_err()
            .contains("Base model not found"));
    }

    #[test]
    fn test_load_nonexistent() {
        assert!(load_single_model(std::path::Path::new("/no/m"))
            .unwrap_err()
            .contains("Failed to read"));
    }

    #[test]
    fn test_run_merge_too_few() {
        let a = MergeArgs {
            models: vec![PathBuf::from("a")],
            output: PathBuf::from("o"),
            method: MergeMethod::Ties,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(run_merge(a, LogLevel::Quiet).unwrap_err().contains("Need at least 2"));
    }

    #[test]
    fn test_run_merge_lora_routes() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::LoraAdapter,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(run_merge(a, LogLevel::Quiet).unwrap_err().contains("--base required"));
    }

    // ── perform_merge routing tests ─────────────────────────────────────

    #[test]
    fn test_perform_merge_ties_route() {
        let models =
            vec![mk(&[("w", &[1.0, 2.0])]), mk(&[("w", &[1.1, 2.1])]), mk(&[("w", &[1.2, 2.2])])];
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Ties,
            weight: None,
            density: Some(0.5),
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(perform_merge(&models, &a).is_ok());
    }

    #[test]
    fn test_perform_merge_dare_route() {
        let models = vec![mk(&[("w", &[1.0, 2.0])]), mk(&[("w", &[1.5, 2.5])])];
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Dare,
            weight: None,
            density: Some(0.3),
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(perform_merge(&models, &a).is_ok());
    }

    #[test]
    fn test_perform_merge_slerp_route() {
        let models = vec![mk(&[("w", &[1.0, 0.0])]), mk(&[("w", &[0.0, 1.0])])];
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Slerp,
            weight: Some(0.5),
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(perform_merge(&models, &a).is_ok());
    }

    #[test]
    fn test_perform_merge_average_route() {
        let models = vec![mk(&[("w", &[2.0])]), mk(&[("w", &[4.0])])];
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        let result = perform_merge(&models, &a).unwrap();
        let vals = result["w"].data().as_slice().unwrap().to_vec();
        assert!((vals[0] - 3.0).abs() < 1e-6);
    }

    // ── slerp merge with exactly 2 models ───────────────────────────────

    #[test]
    fn test_slerp_merge_two_models_ok() {
        let ms = vec![mk(&[("w", &[1.0, 0.0])]), mk(&[("w", &[0.0, 1.0])])];
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Slerp,
            weight: Some(0.3),
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(perform_slerp_merge(&ms, &a).is_ok());
    }

    #[test]
    fn test_slerp_merge_default_weight() {
        let ms = vec![mk(&[("w", &[1.0, 0.0])]), mk(&[("w", &[0.0, 1.0])])];
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Slerp,
            weight: None, // defaults to 0.5
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(perform_slerp_merge(&ms, &a).is_ok());
    }

    // ── ties merge with density ─────────────────────────────────────────

    #[test]
    fn test_ties_merge_with_density() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Ties,
            weight: None,
            density: Some(0.8),
            weights: None,
            base: None,
            adapter: None,
        };
        let models =
            vec![mk(&[("w", &[1.0, 2.0])]), mk(&[("w", &[1.5, 2.5])]), mk(&[("w", &[1.2, 2.2])])];
        let result = perform_ties_merge(&models, &a);
        assert!(result.is_ok());
    }

    // ── dare merge with density ─────────────────────────────────────────

    #[test]
    fn test_dare_merge_with_density() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Dare,
            weight: None,
            density: Some(0.9),
            weights: None,
            base: None,
            adapter: None,
        };
        let models = vec![mk(&[("w", &[1.0, 2.0])]), mk(&[("w", &[1.5, 2.5])])];
        assert!(perform_dare_merge(&models, &a).is_ok());
    }

    // ── average merge with explicit weights ─────────────────────────────

    #[test]
    fn test_average_merge_with_weights() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: Some("0.8,0.2".to_string()),
            base: None,
            adapter: None,
        };
        let models = vec![mk(&[("w", &[10.0])]), mk(&[("w", &[0.0])])];
        let result = perform_average_merge(&models, &a).unwrap();
        let vals = result["w"].data().as_slice().unwrap().to_vec();
        // 0.8 * 10.0 + 0.2 * 0.0 = 8.0
        assert!((vals[0] - 8.0).abs() < 1e-4);
    }

    // ── build_ensemble_config edge cases ────────────────────────────────

    #[test]
    fn test_build_ensemble_config_single_weight() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o.json"),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: Some("1.0".to_string()),
            base: None,
            adapter: None,
        };
        let config = build_ensemble_config(&a);
        assert!(config.is_ok());
    }

    #[test]
    fn test_build_ensemble_config_three_weights() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o.json"),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: Some("0.2, 0.3, 0.5".to_string()),
            base: None,
            adapter: None,
        };
        let config = build_ensemble_config(&a);
        assert!(config.is_ok());
    }

    #[test]
    fn test_build_ensemble_config_empty_weights_string() {
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o.json"),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: Some(String::new()),
            base: None,
            adapter: None,
        };
        // Empty string should fail to parse as f32
        assert!(build_ensemble_config(&a).is_err());
    }

    // ── validate_model_count edge cases ─────────────────────────────────

    #[test]
    fn test_validate_model_count_one() {
        let a = MergeArgs {
            models: vec![PathBuf::from("a")],
            output: PathBuf::from("o"),
            method: MergeMethod::Ties,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(validate_model_count(&a).is_err());
    }

    #[test]
    fn test_validate_model_count_three() {
        let a = MergeArgs {
            models: vec![PathBuf::from("a"), PathBuf::from("b"), PathBuf::from("c")],
            output: PathBuf::from("o"),
            method: MergeMethod::Ties,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(validate_model_count(&a).is_ok());
    }

    // ── export_merged_model extension detection ─────────────────────────

    #[test]
    fn test_export_merged_model_no_extension() {
        let m = mk(&[("w", &[1.0])]);
        let t = std::env::temp_dir().join("ent_merge_noext");
        let a = MergeArgs {
            models: vec![],
            output: t.clone(),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        // Should fall through to JSON export (default)
        assert!(export_merged_model(&m, &a).is_ok());
        let _ = std::fs::remove_file(&t);
    }

    // ── bytes_to_f32 additional edge cases ──────────────────────────────

    #[test]
    fn test_bytes_to_f32_f32_multiple() {
        let vals = vec![1.0f32, 2.0, 3.5, -1.0];
        let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::F32);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.5).abs() < 1e-6);
        assert!((result[3] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_bytes_to_f32_f16_multiple() {
        let vals = vec![half::f16::from_f32(0.5), half::f16::from_f32(1.5)];
        let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::F16);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.5).abs() < 0.01);
        assert!((result[1] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_bytes_to_f32_bf16_multiple() {
        let vals = vec![half::bf16::from_f32(3.0), half::bf16::from_f32(-1.0)];
        let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::BF16);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 3.0).abs() < 0.1);
        assert!((result[1] - (-1.0)).abs() < 0.1);
    }

    // ── build_safetensor_metadata tests ─────────────────────────────────

    #[test]
    fn test_safetensor_metadata_ties() {
        let m = mk(&[("a", &[1.0]), ("b", &[2.0]), ("c", &[3.0])]);
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o.st"),
            method: MergeMethod::Ties,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        let md = build_safetensor_metadata(&m, &a);
        assert_eq!(md["name"], "merged-model");
        assert_eq!(md["tensor_count"], "3");
        assert!(md["merge_method"].contains("Ties"));
    }

    #[test]
    fn test_safetensor_metadata_slerp() {
        let m = mk(&[("x", &[1.0])]);
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o.st"),
            method: MergeMethod::Slerp,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        let md = build_safetensor_metadata(&m, &a);
        assert!(md["merge_method"].contains("Slerp"));
    }

    // ── log_merge_start and log_merge_complete with different levels ────

    #[test]
    fn test_log_merge_start_normal() {
        let a = MergeArgs {
            models: vec![PathBuf::from("m1"), PathBuf::from("m2")],
            output: PathBuf::from("out"),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        log_merge_start(&a, LogLevel::Normal);
    }

    #[test]
    fn test_log_merge_complete_verbose() {
        let m = mk(&[("a", &[1.0, 2.0])]);
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("merged.json"),
            method: MergeMethod::Dare,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        log_merge_complete(&m, &a, LogLevel::Verbose);
    }

    // ── LoRA merge error paths ──────────────────────────────────────────

    #[test]
    fn test_lora_adapter_config_not_found() {
        // adapter dir exists but no adapter_config.json inside
        let dir = tempfile::tempdir().unwrap();
        // Create a fake base file
        let base_file = dir.path().join("base.safetensors");
        std::fs::write(&base_file, b"fake").unwrap();
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::LoraAdapter,
            weight: None,
            density: None,
            weights: None,
            base: Some(base_file),
            adapter: Some(dir.path().to_path_buf()),
        };
        let err = run_lora_adapter_merge(&a, LogLevel::Quiet).unwrap_err();
        assert!(err.contains("adapter_config.json"), "Error: {err}");
    }

    #[test]
    fn test_lora_adapter_model_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let base_file = dir.path().join("base.safetensors");
        std::fs::write(&base_file, b"fake").unwrap();
        // Create adapter_config.json but not adapter_model.safetensors
        std::fs::write(dir.path().join("adapter_config.json"), r#"{"r": 8, "lora_alpha": 16}"#)
            .unwrap();
        let a = MergeArgs {
            models: vec![],
            output: PathBuf::from("o"),
            method: MergeMethod::LoraAdapter,
            weight: None,
            density: None,
            weights: None,
            base: Some(base_file),
            adapter: Some(dir.path().to_path_buf()),
        };
        let err = run_lora_adapter_merge(&a, LogLevel::Quiet).unwrap_err();
        assert!(err.contains("adapter_model.safetensors"), "Error: {err}");
    }

    // ── run_merge with nonexistent model files ──────────────────────────

    #[test]
    fn test_run_merge_nonexistent_models() {
        let a = MergeArgs {
            models: vec![PathBuf::from("/no/m1"), PathBuf::from("/no/m2")],
            output: PathBuf::from("o"),
            method: MergeMethod::Ties,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        let err = run_merge(a, LogLevel::Quiet).unwrap_err();
        assert!(err.contains("Failed to read"), "Error: {err}");
    }

    // ── mk helper verify ────────────────────────────────────────────────

    #[test]
    fn test_mk_helper_creates_model() {
        let model = mk(&[("a", &[1.0, 2.0, 3.0]), ("b", &[4.0])]);
        assert_eq!(model.len(), 2);
        assert!(model.contains_key("a"));
        assert!(model.contains_key("b"));
        assert_eq!(model["a"].len(), 3);
        assert_eq!(model["b"].len(), 1);
    }

    // ── export safetensors with multiple tensors ────────────────────────

    #[test]
    fn test_export_safetensors_multiple_tensors() {
        let m = mk(&[("w1", &[1.0, 2.0]), ("w2", &[3.0, 4.0, 5.0])]);
        let t = std::env::temp_dir().join("ent_merge_multi.safetensors");
        let a = MergeArgs {
            models: vec![],
            output: t.clone(),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(export_merged_model(&m, &a).is_ok());
        // Verify file was created and has content
        assert!(t.exists());
        let _ = std::fs::remove_file(&t);
    }

    // ── export json roundtrip ───────────────────────────────────────────

    #[test]
    fn test_export_json_roundtrip() {
        let m = mk(&[("w1", &[1.0, 2.0]), ("w2", &[3.0])]);
        let t = std::env::temp_dir().join("ent_merge_roundtrip.json");
        let a = MergeArgs {
            models: vec![],
            output: t.clone(),
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        };
        assert!(export_merged_model(&m, &a).is_ok());
        let content = std::fs::read_to_string(&t).unwrap();
        let parsed: HashMap<String, Vec<f32>> = serde_json::from_str(&content).unwrap();
        assert!(parsed.contains_key("w1"));
        assert!(parsed.contains_key("w2"));
        assert_eq!(parsed["w1"].len(), 2);
        let _ = std::fs::remove_file(&t);
    }

    // =========================================================================
    // test_cov2_* — Additional coverage tests
    // =========================================================================

    /// Helper to build MergeArgs easily
    fn mk_args(method: MergeMethod) -> MergeArgs {
        MergeArgs {
            models: vec![],
            output: PathBuf::from("out.json"),
            method,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        }
    }

    // ── bytes_to_f32 with zero-value data ────────────────────────────────

    #[test]
    fn test_cov2_bytes_to_f32_f32_zeros() {
        let zeros = vec![0.0f32; 10];
        let bytes: Vec<u8> = zeros.iter().flat_map(|x| x.to_le_bytes()).collect();
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::F32);
        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_cov2_bytes_to_f32_f32_negative() {
        let vals = vec![-1.0f32, -100.0, -0.001];
        let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::F32);
        assert_eq!(result.len(), 3);
        assert!((result[0] - (-1.0)).abs() < 1e-6);
        assert!((result[1] - (-100.0)).abs() < 1e-6);
        assert!((result[2] - (-0.001)).abs() < 1e-6);
    }

    #[test]
    fn test_cov2_bytes_to_f32_f32_large() {
        let vals = vec![1e30f32, -1e30];
        let bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::F32);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1e30).abs() / 1e30 < 1e-6);
    }

    #[test]
    fn test_cov2_bytes_to_f32_f16_zero() {
        let zero = half::f16::from_f32(0.0);
        let bytes = zero.to_le_bytes().to_vec();
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::F16);
        assert_eq!(result.len(), 1);
        assert!((result[0]).abs() < 1e-6);
    }

    #[test]
    fn test_cov2_bytes_to_f32_bf16_zero() {
        let zero = half::bf16::from_f32(0.0);
        let bytes = zero.to_le_bytes().to_vec();
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::BF16);
        assert_eq!(result.len(), 1);
        assert!((result[0]).abs() < 1e-6);
    }

    #[test]
    fn test_cov2_bytes_to_f32_f16_negative() {
        let neg = half::f16::from_f32(-3.14);
        let bytes = neg.to_le_bytes().to_vec();
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::F16);
        assert_eq!(result.len(), 1);
        assert!((result[0] - (-3.14)).abs() < 0.01);
    }

    #[test]
    fn test_cov2_bytes_to_f32_bf16_negative() {
        let neg = half::bf16::from_f32(-5.0);
        let bytes = neg.to_le_bytes().to_vec();
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::BF16);
        assert_eq!(result.len(), 1);
        assert!((result[0] - (-5.0)).abs() < 0.5);
    }

    // ── bytes_to_f32 with truncated data (not aligned) ──────────────────

    #[test]
    fn test_cov2_bytes_to_f32_f32_truncated() {
        // 5 bytes → only 1 full f32 chunk (4 bytes), remainder ignored
        let bytes: Vec<u8> = vec![0, 0, 128, 63, 99];
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::F32);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cov2_bytes_to_f32_f16_truncated() {
        // 3 bytes → only 1 full f16 chunk (2 bytes), remainder ignored
        let val = half::f16::from_f32(2.0);
        let mut bytes = val.to_le_bytes().to_vec();
        bytes.push(0xFF);
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::F16);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.0).abs() < 0.01);
    }

    // ── bytes_to_f32 with I64 fallback (other dtype) ────────────────────

    #[test]
    fn test_cov2_bytes_to_f32_i64_fallback() {
        let v = 3.14f32;
        let bytes = v.to_le_bytes().to_vec();
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::I64);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_cov2_bytes_to_f32_u8_fallback() {
        let v = 7.0f32;
        let bytes = v.to_le_bytes().to_vec();
        let result = bytes_to_f32(&bytes, safetensors::tensor::Dtype::U8);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 7.0).abs() < 1e-6);
    }

    // ── build_ensemble_config with whitespace-padded weights ────────────

    #[test]
    fn test_cov2_build_ensemble_config_whitespace_weights() {
        let a = MergeArgs {
            weights: Some("  0.5 , 0.3 , 0.2  ".to_string()),
            ..mk_args(MergeMethod::Average)
        };
        let config = build_ensemble_config(&a);
        assert!(config.is_ok());
    }

    // ── build_ensemble_config with negative weights ─────────────────────

    #[test]
    fn test_cov2_build_ensemble_config_negative_weights() {
        let a =
            MergeArgs { weights: Some("-0.5, 1.5".to_string()), ..mk_args(MergeMethod::Average) };
        let config = build_ensemble_config(&a);
        // Parsing should succeed (negative floats are valid f32)
        assert!(config.is_ok());
    }

    // ── build_ensemble_config with large number of weights ──────────────

    #[test]
    fn test_cov2_build_ensemble_config_many_weights() {
        let w_str = (0..10).map(|_| "0.1").collect::<Vec<_>>().join(",");
        let a = MergeArgs { weights: Some(w_str), ..mk_args(MergeMethod::Average) };
        let config = build_ensemble_config(&a);
        assert!(config.is_ok());
    }

    // ── build_safetensor_metadata for each method ───────────────────────

    #[test]
    fn test_cov2_safetensor_metadata_average() {
        let m = mk(&[("w", &[1.0])]);
        let a = mk_args(MergeMethod::Average);
        let md = build_safetensor_metadata(&m, &a);
        assert!(md["merge_method"].contains("Average"));
        assert_eq!(md["tensor_count"], "1");
    }

    #[test]
    fn test_cov2_safetensor_metadata_dare() {
        let m = mk(&[("a", &[1.0]), ("b", &[2.0])]);
        let a = mk_args(MergeMethod::Dare);
        let md = build_safetensor_metadata(&m, &a);
        assert!(md["merge_method"].contains("Dare"));
        assert_eq!(md["tensor_count"], "2");
    }

    #[test]
    fn test_cov2_safetensor_metadata_lora() {
        let m = mk(&[("w", &[1.0])]);
        let a = mk_args(MergeMethod::LoraAdapter);
        let md = build_safetensor_metadata(&m, &a);
        assert!(md["merge_method"].contains("LoraAdapter"));
    }

    #[test]
    fn test_cov2_safetensor_metadata_empty_model() {
        let m: Model = HashMap::new();
        let a = mk_args(MergeMethod::Ties);
        let md = build_safetensor_metadata(&m, &a);
        assert_eq!(md["tensor_count"], "0");
    }

    // ── validate_model_count edge: exactly 2 ────────────────────────────

    #[test]
    fn test_cov2_validate_model_count_exactly_2() {
        let a = MergeArgs {
            models: vec![PathBuf::from("a"), PathBuf::from("b")],
            ..mk_args(MergeMethod::Average)
        };
        assert!(validate_model_count(&a).is_ok());
    }

    #[test]
    fn test_cov2_validate_model_count_large() {
        let models: Vec<PathBuf> = (0..100).map(|i| PathBuf::from(format!("m{i}"))).collect();
        let a = MergeArgs { models, ..mk_args(MergeMethod::Average) };
        assert!(validate_model_count(&a).is_ok());
    }

    // ── perform_merge LoRA early error message ──────────────────────────

    #[test]
    fn test_cov2_perform_merge_lora_error_msg() {
        let a = mk_args(MergeMethod::LoraAdapter);
        let err = perform_merge(&[], &a).unwrap_err();
        assert_eq!(err, "LoRA adapter merge uses dedicated path");
    }

    // ── export_merged_model to bad path ─────────────────────────────────

    #[test]
    fn test_cov2_export_json_bad_path() {
        let m = mk(&[("w", &[1.0])]);
        let a = MergeArgs {
            output: PathBuf::from("/nonexistent_dir_xxxx/output.json"),
            ..mk_args(MergeMethod::Average)
        };
        let result = export_merged_model(&m, &a);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to write"));
    }

    #[test]
    fn test_cov2_export_safetensors_bad_path() {
        let m = mk(&[("w", &[1.0])]);
        let a = MergeArgs {
            output: PathBuf::from("/nonexistent_dir_xxxx/output.safetensors"),
            ..mk_args(MergeMethod::Average)
        };
        let result = export_merged_model(&m, &a);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to write"));
    }

    // ── export safetensors roundtrip ────────────────────────────────────

    #[test]
    fn test_cov2_export_safetensors_roundtrip() {
        let m = mk(&[("layer1", &[1.0, 2.0, 3.0]), ("layer2", &[4.0, 5.0])]);
        let t = std::env::temp_dir().join("ent_merge_cov2_rt.safetensors");
        let a = MergeArgs { output: t.clone(), ..mk_args(MergeMethod::Ties) };
        assert!(export_merged_model(&m, &a).is_ok());
        // Read back and verify
        let data = std::fs::read(&t).unwrap();
        let tensors = SafeTensors::deserialize(&data).unwrap();
        let names: Vec<&str> = tensors.names().to_vec();
        assert!(names.contains(&"layer1"));
        assert!(names.contains(&"layer2"));
        let _ = std::fs::remove_file(&t);
    }

    // ── export json with empty model ────────────────────────────────────

    #[test]
    fn test_cov2_export_json_empty_model() {
        let m: Model = HashMap::new();
        let t = std::env::temp_dir().join("ent_merge_cov2_empty.json");
        let a = MergeArgs { output: t.clone(), ..mk_args(MergeMethod::Average) };
        assert!(export_merged_model(&m, &a).is_ok());
        let content = std::fs::read_to_string(&t).unwrap();
        let parsed: HashMap<String, Vec<f32>> = serde_json::from_str(&content).unwrap();
        assert!(parsed.is_empty());
        let _ = std::fs::remove_file(&t);
    }

    // ── export safetensors with empty model ─────────────────────────────

    #[test]
    fn test_cov2_export_safetensors_empty_model() {
        let m: Model = HashMap::new();
        let t = std::env::temp_dir().join("ent_merge_cov2_empty.safetensors");
        let a = MergeArgs { output: t.clone(), ..mk_args(MergeMethod::Average) };
        assert!(export_merged_model(&m, &a).is_ok());
        let _ = std::fs::remove_file(&t);
    }

    // ── perform_ties_merge with default density ─────────────────────────

    #[test]
    fn test_cov2_ties_merge_default_density() {
        let a = MergeArgs { density: None, ..mk_args(MergeMethod::Ties) };
        // density defaults to 0.2
        let models =
            vec![mk(&[("w", &[1.0, 2.0])]), mk(&[("w", &[1.5, 2.5])]), mk(&[("w", &[1.2, 2.2])])];
        assert!(perform_ties_merge(&models, &a).is_ok());
    }

    // ── perform_dare_merge with default density ─────────────────────────

    #[test]
    fn test_cov2_dare_merge_default_density() {
        let a = MergeArgs { density: None, ..mk_args(MergeMethod::Dare) };
        // density defaults to 0.5 → drop_prob = 0.5
        let models = vec![mk(&[("w", &[1.0, 2.0])]), mk(&[("w", &[1.5, 2.5])])];
        assert!(perform_dare_merge(&models, &a).is_ok());
    }

    // ── perform_slerp_merge with default weight ─────────────────────────

    #[test]
    fn test_cov2_slerp_merge_default_weight() {
        let a = MergeArgs { weight: None, ..mk_args(MergeMethod::Slerp) };
        let models = vec![mk(&[("w", &[1.0, 0.0])]), mk(&[("w", &[0.0, 1.0])])];
        let result = perform_slerp_merge(&models, &a);
        assert!(result.is_ok());
    }

    // ── slerp with single model → error ─────────────────────────────────

    #[test]
    fn test_cov2_slerp_single_model() {
        let a = mk_args(MergeMethod::Slerp);
        let models = vec![mk(&[("w", &[1.0])])];
        let err = perform_slerp_merge(&models, &a).unwrap_err();
        assert!(err.contains("SLERP requires exactly 2"));
    }

    // ── run_merge with LoRA routes to lora function ─────────────────────

    #[test]
    fn test_cov2_run_merge_lora_missing_both() {
        let a = MergeArgs {
            method: MergeMethod::LoraAdapter,
            base: None,
            adapter: None,
            ..mk_args(MergeMethod::LoraAdapter)
        };
        let err = run_merge(a, LogLevel::Quiet).unwrap_err();
        assert!(err.contains("--base required"));
    }

    #[test]
    fn test_cov2_run_merge_lora_has_base_no_adapter() {
        let a = MergeArgs {
            method: MergeMethod::LoraAdapter,
            base: Some(PathBuf::from("/tmp/some_base")),
            adapter: None,
            ..mk_args(MergeMethod::LoraAdapter)
        };
        let err = run_merge(a, LogLevel::Quiet).unwrap_err();
        assert!(err.contains("--adapter required"));
    }

    // ── load_single_model with empty file ───────────────────────────────

    #[test]
    fn test_cov2_load_single_model_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.safetensors");
        std::fs::write(&path, b"").unwrap();
        let err = load_single_model(&path).unwrap_err();
        assert!(err.contains("Failed to parse"));
    }

    // ── load_single_model with garbage data ─────────────────────────────

    #[test]
    fn test_cov2_load_single_model_garbage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("garbage.safetensors");
        std::fs::write(&path, b"this is not a safetensors file at all").unwrap();
        let err = load_single_model(&path).unwrap_err();
        assert!(err.contains("Failed to parse"));
    }

    // ── run_merge models don't exist → load error ───────────────────────

    #[test]
    fn test_cov2_run_merge_first_model_missing() {
        let dir = tempfile::tempdir().unwrap();
        let a = MergeArgs {
            models: vec![
                dir.path().join("no_exist_1.safetensors"),
                dir.path().join("no_exist_2.safetensors"),
            ],
            output: dir.path().join("out.json"),
            method: MergeMethod::Average,
            ..mk_args(MergeMethod::Average)
        };
        let err = run_merge(a, LogLevel::Quiet).unwrap_err();
        assert!(err.contains("Failed to read"));
    }

    // ── log functions with all log levels ────────────────────────────────

    #[test]
    fn test_cov2_log_merge_start_quiet() {
        let a = MergeArgs {
            models: vec![PathBuf::from("m1"), PathBuf::from("m2"), PathBuf::from("m3")],
            output: PathBuf::from("out.safetensors"),
            ..mk_args(MergeMethod::Dare)
        };
        log_merge_start(&a, LogLevel::Quiet);
    }

    #[test]
    fn test_cov2_log_merge_complete_quiet() {
        let m = mk(&[("w", &[1.0, 2.0, 3.0])]);
        let a = MergeArgs {
            output: PathBuf::from("merged.safetensors"),
            ..mk_args(MergeMethod::Average)
        };
        log_merge_complete(&m, &a, LogLevel::Quiet);
    }

    // ── average merge with multiple tensors per model ───────────────────

    #[test]
    fn test_cov2_average_merge_multi_tensor() {
        let a = mk_args(MergeMethod::Average);
        let m1 = mk(&[("a", &[1.0, 2.0]), ("b", &[3.0])]);
        let m2 = mk(&[("a", &[3.0, 4.0]), ("b", &[5.0])]);
        let result = perform_average_merge(&[m1, m2], &a).unwrap();
        let a_vals = result["a"].data().as_slice().unwrap().to_vec();
        let b_vals = result["b"].data().as_slice().unwrap().to_vec();
        assert!((a_vals[0] - 2.0).abs() < 1e-6);
        assert!((a_vals[1] - 3.0).abs() < 1e-6);
        assert!((b_vals[0] - 4.0).abs() < 1e-6);
    }

    // ── slerp merge with custom weight ──────────────────────────────────

    #[test]
    fn test_cov2_slerp_merge_weight_0() {
        let a = MergeArgs { weight: Some(0.0), ..mk_args(MergeMethod::Slerp) };
        let models = vec![mk(&[("w", &[1.0, 0.0])]), mk(&[("w", &[0.0, 1.0])])];
        let result = perform_slerp_merge(&models, &a).unwrap();
        let vals = result["w"].data().as_slice().unwrap().to_vec();
        // t=0 should give model 0's values
        assert!((vals[0] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_cov2_slerp_merge_weight_1() {
        let a = MergeArgs { weight: Some(1.0), ..mk_args(MergeMethod::Slerp) };
        let models = vec![mk(&[("w", &[1.0, 0.0])]), mk(&[("w", &[0.0, 1.0])])];
        let result = perform_slerp_merge(&models, &a).unwrap();
        let vals = result["w"].data().as_slice().unwrap().to_vec();
        // t=1 should give model 1's values
        assert!((vals[1] - 1.0).abs() < 0.1);
    }

    // ── ties merge with explicit density close to 1.0 ───────────────────

    #[test]
    fn test_cov2_ties_merge_high_density() {
        let a = MergeArgs { density: Some(0.99), ..mk_args(MergeMethod::Ties) };
        let models = vec![
            mk(&[("w", &[1.0, 2.0, 3.0])]),
            mk(&[("w", &[1.1, 2.1, 3.1])]),
            mk(&[("w", &[1.2, 2.2, 3.2])]),
        ];
        assert!(perform_ties_merge(&models, &a).is_ok());
    }

    // ── dare merge with density close to 0 ──────────────────────────────

    #[test]
    fn test_cov2_dare_merge_low_density() {
        let a = MergeArgs { density: Some(0.01), ..mk_args(MergeMethod::Dare) };
        let models = vec![mk(&[("w", &[1.0, 2.0])]), mk(&[("w", &[1.5, 2.5])])];
        assert!(perform_dare_merge(&models, &a).is_ok());
    }

    // ── LoRA merge: adapter dir exists, has config, but no model.safetensors ─

    #[test]
    fn test_cov2_lora_adapter_config_exists_no_model() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.safetensors");
        std::fs::write(&base, b"fake").unwrap();
        let adapter_dir = dir.path().join("adapter");
        std::fs::create_dir_all(&adapter_dir).unwrap();
        std::fs::write(adapter_dir.join("adapter_config.json"), r#"{"r":8}"#).unwrap();
        // No adapter_model.safetensors
        let a = MergeArgs {
            base: Some(base),
            adapter: Some(adapter_dir),
            ..mk_args(MergeMethod::LoraAdapter)
        };
        let err = run_lora_adapter_merge(&a, LogLevel::Quiet).unwrap_err();
        assert!(err.contains("adapter_model.safetensors"));
    }

    // ── LoRA merge: base path doesn't exist ─────────────────────────────

    #[test]
    fn test_cov2_lora_base_path_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let adapter_dir = dir.path().join("adapter");
        std::fs::create_dir_all(&adapter_dir).unwrap();
        let a = MergeArgs {
            base: Some(PathBuf::from("/definitely/not/exist/base.st")),
            adapter: Some(adapter_dir),
            ..mk_args(MergeMethod::LoraAdapter)
        };
        let err = run_lora_adapter_merge(&a, LogLevel::Quiet).unwrap_err();
        assert!(err.contains("Base model not found"));
    }

    // ── export extension detection ──────────────────────────────────────

    #[test]
    fn test_cov2_export_extension_safetensors() {
        let m = mk(&[("w", &[1.0])]);
        let t = std::env::temp_dir().join("ent_merge_cov2_ext.safetensors");
        let a = MergeArgs { output: t.clone(), ..mk_args(MergeMethod::Average) };
        assert!(export_merged_model(&m, &a).is_ok());
        // Verify file exists and is valid safetensors
        let data = std::fs::read(&t).unwrap();
        assert!(SafeTensors::deserialize(&data).is_ok());
        let _ = std::fs::remove_file(&t);
    }

    #[test]
    fn test_cov2_export_extension_json() {
        let m = mk(&[("w", &[1.0])]);
        let t = std::env::temp_dir().join("ent_merge_cov2_ext.json");
        let a = MergeArgs { output: t.clone(), ..mk_args(MergeMethod::Average) };
        assert!(export_merged_model(&m, &a).is_ok());
        let content = std::fs::read_to_string(&t).unwrap();
        assert!(serde_json::from_str::<HashMap<String, Vec<f32>>>(&content).is_ok());
        let _ = std::fs::remove_file(&t);
    }

    #[test]
    fn test_cov2_export_extension_unknown() {
        let m = mk(&[("w", &[1.0])]);
        let t = std::env::temp_dir().join("ent_merge_cov2_ext.bin");
        let a = MergeArgs { output: t.clone(), ..mk_args(MergeMethod::Average) };
        // Unknown extension → falls through to JSON
        assert!(export_merged_model(&m, &a).is_ok());
        let content = std::fs::read_to_string(&t).unwrap();
        assert!(serde_json::from_str::<HashMap<String, Vec<f32>>>(&content).is_ok());
        let _ = std::fs::remove_file(&t);
    }

    // ── mk helper edge cases ────────────────────────────────────────────

    #[test]
    fn test_cov2_mk_empty_model() {
        let model = mk(&[]);
        assert!(model.is_empty());
    }

    #[test]
    fn test_cov2_mk_single_empty_tensor() {
        let model = mk(&[("empty", &[])]);
        assert_eq!(model.len(), 1);
        assert_eq!(model["empty"].len(), 0);
    }

    // ── perform_merge dispatch coverage ─────────────────────────────────

    #[test]
    fn test_cov2_perform_merge_all_methods() {
        // Ties
        let models3 =
            vec![mk(&[("w", &[1.0, 2.0])]), mk(&[("w", &[1.1, 2.1])]), mk(&[("w", &[1.2, 2.2])])];
        assert!(perform_merge(&models3, &mk_args(MergeMethod::Ties)).is_ok());

        // Dare
        let models2 = vec![mk(&[("w", &[1.0, 2.0])]), mk(&[("w", &[1.5, 2.5])])];
        assert!(perform_merge(&models2, &mk_args(MergeMethod::Dare)).is_ok());

        // Slerp
        let models_s = vec![mk(&[("w", &[1.0, 0.0])]), mk(&[("w", &[0.0, 1.0])])];
        assert!(perform_merge(
            &models_s,
            &MergeArgs { weight: Some(0.5), ..mk_args(MergeMethod::Slerp) }
        )
        .is_ok());

        // Average
        let models_a = vec![mk(&[("w", &[2.0])]), mk(&[("w", &[4.0])])];
        assert!(perform_merge(&models_a, &mk_args(MergeMethod::Average)).is_ok());

        // LoraAdapter → error
        assert!(perform_merge(&[], &mk_args(MergeMethod::LoraAdapter)).is_err());
    }

    // ── large tensor export/import roundtrip ────────────────────────────

    #[test]
    fn test_cov2_large_tensor_roundtrip() {
        let large_data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
        let m = mk(&[("big", large_data.as_slice())]);
        let t = std::env::temp_dir().join("ent_merge_cov2_large.json");
        let a = MergeArgs { output: t.clone(), ..mk_args(MergeMethod::Average) };
        assert!(export_merged_model(&m, &a).is_ok());
        let content = std::fs::read_to_string(&t).unwrap();
        let parsed: HashMap<String, Vec<f32>> = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["big"].len(), 1000);
        assert!((parsed["big"][500] - 0.5).abs() < 1e-3);
        let _ = std::fs::remove_file(&t);
    }
}
