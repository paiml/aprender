//! Prune-then-Quantize pipeline
//!
//! Combines pruning and quantization into a single operation:
//! 1. Prune weights (set to zero based on importance)
//! 2. Quantize remaining weights (Q4_0 or Q8_0)
//! 3. Export the quantized/pruned weights

use crate::quant::{Q4_0, Q8_0};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Quantization format for the prune-quantize pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
enum PruneQuantFormat {
    /// 4-bit quantization
    Q4_0,
    /// 8-bit quantization
    Q8_0,
}

/// Configuration for prune-then-quantize pipeline
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PruneQuantConfig {
    /// Target sparsity (0.0 to 1.0)
    target_sparsity: f32,
    /// Quantization format
    quant_format: PruneQuantFormat,
}

/// Result of prune-then-quantize pipeline
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PruneQuantizeResult {
    /// Path to output file
    output_path: PathBuf,
    /// Achieved sparsity before quantization
    achieved_sparsity: f32,
    /// Quantization format used
    quant_format: PruneQuantFormat,
    /// Number of tensors
    num_tensors: usize,
    /// Output file size in bytes
    file_size: u64,
}

/// Apply magnitude pruning to weights (set smallest values to zero)
#[allow(dead_code)]
fn magnitude_prune(
    weights: &mut HashMap<String, Vec<f32>>,
    target_sparsity: f32,
) -> (usize, usize) {
    if target_sparsity <= 0.0 {
        let total: usize = weights.values().map(Vec::len).sum();
        return (0, total);
    }

    // Collect all magnitudes to determine threshold
    let mut all_magnitudes: Vec<f32> =
        weights.values().flat_map(|data| data.iter().map(|v| v.abs())).collect();
    all_magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let total = all_magnitudes.len();
    let prune_count = ((total as f32 * target_sparsity) as usize).min(total);
    let threshold = if prune_count < total { all_magnitudes[prune_count] } else { f32::MAX };

    // Apply pruning
    let mut pruned = 0;
    for data in weights.values_mut() {
        for val in data.iter_mut() {
            if val.abs() < threshold {
                *val = 0.0;
                pruned += 1;
            }
        }
    }

    (pruned, total)
}

/// Prune model weights and quantize, saving as SafeTensors with quantized data
///
/// Pipeline:
/// 1. Apply magnitude pruning to reach target sparsity
/// 2. Quantize all tensors to Q4_0 or Q8_0
/// 3. Dequantize back to f32 for SafeTensors storage (preserving quantization effects)
/// 4. Export as SafeTensors
#[allow(dead_code)]
fn prune_and_quantize(
    weights: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    config: &PruneQuantConfig,
    output_dir: impl AsRef<Path>,
    filename: &str,
) -> Result<PruneQuantizeResult, std::io::Error> {
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir)?;

    // Clone weights for pruning
    let mut pruned_weights = weights.clone();

    // Step 1: Prune
    let (pruned_count, total_count) = magnitude_prune(&mut pruned_weights, config.target_sparsity);
    let achieved_sparsity =
        if total_count > 0 { pruned_count as f32 / total_count as f32 } else { 0.0 };

    // Step 2: Quantize then dequantize (applies quantization rounding)
    let quantized_weights: HashMap<String, Vec<f32>> = pruned_weights
        .iter()
        .map(|(name, data)| {
            let deq = match config.quant_format {
                PruneQuantFormat::Q4_0 => Q4_0::quantize(data).dequantize(),
                PruneQuantFormat::Q8_0 => Q8_0::quantize(data).dequantize(),
            };
            (name.clone(), deq)
        })
        .collect();

    // Step 3: Export as SafeTensors
    use safetensors::tensor::{Dtype, TensorView};

    let mut sorted_names: Vec<&String> = quantized_weights.keys().collect();
    sorted_names.sort();

    let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = sorted_names
        .iter()
        .map(|name| {
            let data = &quantized_weights[*name];
            let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
            let shape = shapes.get(*name).cloned().unwrap_or_else(|| vec![data.len()]);
            ((*name).clone(), bytes, shape)
        })
        .collect();

    let views: Vec<(&str, TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, bytes, shape)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .expect("TensorView construction must not fail for valid F32 data");
            (name.as_str(), view)
        })
        .collect();

    let mut metadata = std::collections::HashMap::new();
    metadata.insert("sparsity".to_string(), format!("{achieved_sparsity:.4}"));
    metadata.insert(
        "quantization".to_string(),
        match config.quant_format {
            PruneQuantFormat::Q4_0 => "Q4_0".to_string(),
            PruneQuantFormat::Q8_0 => "Q8_0".to_string(),
        },
    );

    let safetensor_bytes = safetensors::serialize(views, Some(metadata))
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    let output_path = output_dir.join(filename);
    std::fs::write(&output_path, &safetensor_bytes)?;

    Ok(PruneQuantizeResult {
        output_path,
        achieved_sparsity,
        quant_format: config.quant_format,
        num_tensors: sorted_names.len(),
        file_size: safetensor_bytes.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_test_weights() -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();

        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        weights.insert("layer.0.weight".to_string(), data);
        shapes.insert("layer.0.weight".to_string(), vec![8, 8]);

        (weights, shapes)
    }

    #[test]
    fn test_prune_and_quantize_q4() {
        let (weights, shapes) = make_test_weights();
        let config =
            PruneQuantConfig { target_sparsity: 0.5, quant_format: PruneQuantFormat::Q4_0 };
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let result =
            prune_and_quantize(&weights, &shapes, &config, tmp.path(), "pruned.safetensors")
                .expect("operation should succeed");

        assert!(result.achieved_sparsity >= 0.3);
        assert_eq!(result.quant_format, PruneQuantFormat::Q4_0);
        assert!(result.output_path.exists());
        assert!(result.file_size > 0);
    }

    #[test]
    fn test_prune_and_quantize_q8() {
        let (weights, shapes) = make_test_weights();
        let config =
            PruneQuantConfig { target_sparsity: 0.3, quant_format: PruneQuantFormat::Q8_0 };
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let result =
            prune_and_quantize(&weights, &shapes, &config, tmp.path(), "pruned-q8.safetensors")
                .expect("operation should succeed");

        assert_eq!(result.quant_format, PruneQuantFormat::Q8_0);
        assert!(result.file_size > 0);
    }

    #[test]
    fn test_prune_and_quantize_no_sparsity() {
        let (weights, shapes) = make_test_weights();
        let config =
            PruneQuantConfig { target_sparsity: 0.0, quant_format: PruneQuantFormat::Q4_0 };
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let result =
            prune_and_quantize(&weights, &shapes, &config, tmp.path(), "unpruned.safetensors")
                .expect("operation should succeed");

        assert_eq!(result.achieved_sparsity, 0.0);
    }

    #[test]
    fn test_magnitude_prune_basic() {
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), vec![0.1, 0.5, 0.01, 0.8, 0.02, 0.9]);

        let (pruned, total) = magnitude_prune(&mut weights, 0.5);
        assert_eq!(total, 6);
        assert!(pruned >= 2);

        let data = &weights["w"];
        assert_eq!(data[2], 0.0); // 0.01 should be pruned
        assert_eq!(data[4], 0.0); // 0.02 should be pruned
    }

    #[test]
    fn test_output_safetensors_valid() {
        let (weights, shapes) = make_test_weights();
        let config =
            PruneQuantConfig { target_sparsity: 0.5, quant_format: PruneQuantFormat::Q4_0 };
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let result = prune_and_quantize(&weights, &shapes, &config, tmp.path(), "test.safetensors")
            .expect("config should be valid");

        let data = std::fs::read(&result.output_path).expect("file read should succeed");
        let loaded = safetensors::SafeTensors::deserialize(&data).expect("load should succeed");
        assert_eq!(loaded.len(), 1);

        // Check metadata
        let (_, meta) =
            safetensors::SafeTensors::read_metadata(&data).expect("deserialization should succeed");
        let md = meta.metadata().as_ref().expect("operation should succeed");
        assert!(md.contains_key("sparsity"));
        assert!(md.contains_key("quantization"));
    }
}
