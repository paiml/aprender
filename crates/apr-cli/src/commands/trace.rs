//! Trace command implementation
//!
//! Layer-by-layer analysis of APR models.
//! Toyota Way: Visualization - Make hidden problems visible.
//!
//! This command traces through model layers, computing statistics at each stage
//! to help identify where numerical issues or divergences occur.

use crate::error::CliError;
use crate::output;
use aprender::format::HEADER_SIZE;
use colored::Colorize;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Layer trace information
#[derive(Serialize, Clone)]
pub(crate) struct LayerTrace {
    /// Layer name/type
    pub name: String,
    /// Layer index (if applicable)
    pub index: Option<usize>,
    /// Input statistics
    pub input_stats: Option<TensorStats>,
    /// Output statistics
    pub output_stats: Option<TensorStats>,
    /// Weight statistics (if layer has weights)
    pub weight_stats: Option<TensorStats>,
    /// Anomalies detected
    pub anomalies: Vec<String>,
}

/// Tensor statistics for tracing
#[derive(Serialize, Clone)]
#[allow(dead_code)]
pub(crate) struct TensorStats {
    /// Number of elements
    pub count: usize,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// L2 norm
    pub l2_norm: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Maximum absolute value
    pub max_abs: f32,
    /// Count of NaN values
    pub nan_count: usize,
    /// Count of Inf values
    pub inf_count: usize,
}

impl TensorStats {
    /// Compute statistics from a slice of f32 values
    #[allow(dead_code, clippy::cast_lossless)]
    pub(crate) fn from_slice(data: &[f32]) -> Self {
        let count = data.len();
        if count == 0 {
            return Self {
                count: 0,
                mean: 0.0,
                std: 0.0,
                l2_norm: 0.0,
                min: 0.0,
                max: 0.0,
                max_abs: 0.0,
                nan_count: 0,
                inf_count: 0,
            };
        }

        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut max_abs = 0.0_f32;
        let mut nan_count = 0;
        let mut inf_count = 0;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
                continue;
            }
            if v.is_infinite() {
                inf_count += 1;
                continue;
            }
            sum += v as f64;
            sum_sq += (v as f64) * (v as f64);
            min = min.min(v);
            max = max.max(v);
            max_abs = max_abs.max(v.abs());
        }

        let valid_count = count - nan_count - inf_count;
        let mean = if valid_count > 0 {
            (sum / valid_count as f64) as f32
        } else {
            0.0
        };
        let variance = if valid_count > 1 {
            ((sum_sq / valid_count as f64) - (mean as f64).powi(2)).max(0.0)
        } else {
            0.0
        };
        let std = (variance as f32).sqrt();
        let l2_norm = (sum_sq as f32).sqrt();

        Self {
            count,
            mean,
            std,
            l2_norm,
            min: if min.is_finite() { min } else { 0.0 },
            max: if max.is_finite() { max } else { 0.0 },
            max_abs,
            nan_count,
            inf_count,
        }
    }

    /// Check for anomalies
    #[allow(dead_code)]
    pub(crate) fn detect_anomalies(&self, name: &str) -> Vec<String> {
        let mut anomalies = Vec::new();

        if self.nan_count > 0 {
            anomalies.push(format!(
                "{name}: {}/{} NaN values",
                self.nan_count, self.count
            ));
        }
        if self.inf_count > 0 {
            anomalies.push(format!(
                "{name}: {}/{} Inf values",
                self.inf_count, self.count
            ));
        }
        if self.std < 1e-8 && self.count > 1 {
            anomalies.push(format!("{name}: near-zero variance (std={:.2e})", self.std));
        }
        if self.max_abs > 100.0 {
            anomalies.push(format!(
                "{name}: large values (max_abs={:.2})",
                self.max_abs
            ));
        }
        if self.mean.abs() > 10.0 {
            anomalies.push(format!("{name}: large mean bias ({:.4})", self.mean));
        }

        anomalies
    }
}

/// Trace result for JSON output
#[derive(Serialize)]
struct TraceResult {
    file: String,
    format: String,
    layers: Vec<LayerTrace>,
    summary: TraceSummary,
}

/// Summary of trace analysis
#[derive(Serialize)]
struct TraceSummary {
    total_layers: usize,
    total_parameters: usize,
    anomaly_count: usize,
    anomalies: Vec<String>,
}

/// Run the trace command
pub(crate) fn run(
    path: &Path,
    layer_filter: Option<&str>,
    reference: Option<&Path>,
    json_output: bool,
    verbose: bool,
) -> Result<(), CliError> {
    validate_path(path)?;

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read and validate header
    let format_name = validate_header(&mut reader)?;

    // Read metadata
    let mut size_buf = [0u8; 4];
    reader.seek(SeekFrom::Start(8))?;
    reader.read_exact(&mut size_buf)?;
    let metadata_size = u32::from_le_bytes(size_buf) as usize;

    reader.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    let mut metadata_bytes = vec![0u8; metadata_size];
    reader.read_exact(&mut metadata_bytes)?;

    // Parse metadata and extract layer information
    let layers = trace_layers(&metadata_bytes, layer_filter, verbose);

    // Compute summary
    let all_anomalies: Vec<String> = layers.iter().flat_map(|l| l.anomalies.clone()).collect();

    let total_params: usize = layers
        .iter()
        .filter_map(|l| l.weight_stats.as_ref().map(|s| s.count))
        .sum();

    let summary = TraceSummary {
        total_layers: layers.len(),
        total_parameters: total_params,
        anomaly_count: all_anomalies.len(),
        anomalies: all_anomalies,
    };

    // Handle reference comparison
    if let Some(ref_path) = reference {
        compare_with_reference(path, ref_path, &layers, json_output)?;
        return Ok(());
    }

    if json_output {
        output_json(path, &format_name, &layers, &summary);
    } else {
        output_text(path, &format_name, &layers, &summary, verbose);
    }

    Ok(())
}

fn validate_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    Ok(())
}

fn validate_header(reader: &mut BufReader<File>) -> Result<String, CliError> {
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|_| CliError::InvalidFormat("File too small".to_string()))?;

    if !output::is_valid_magic(&magic) {
        return Err(CliError::InvalidFormat(format!(
            "Invalid magic: expected APRN or APR1, got {magic:?}"
        )));
    }

    Ok(output::format_name(&magic).to_string())
}

fn trace_layers(metadata_bytes: &[u8], filter: Option<&str>, _verbose: bool) -> Vec<LayerTrace> {
    // Parse metadata as MessagePack
    let metadata: BTreeMap<String, serde_json::Value> =
        rmp_serde::from_slice(metadata_bytes).unwrap_or_else(|_| BTreeMap::new());

    let mut layers = Vec::new();

    // Extract layer info from hyperparameters if available
    if let Some(hp) = metadata.get("hyperparameters") {
        if let Some(hp_obj) = hp.as_object() {
            // Create synthetic layer trace from model architecture
            let n_layers = hp_obj
                .get("n_layer")
                .or_else(|| hp_obj.get("n_layers"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize;

            let d_model = hp_obj
                .get("n_embd")
                .or_else(|| hp_obj.get("d_model"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize;

            // Add embedding layer
            layers.push(LayerTrace {
                name: "embedding".to_string(),
                index: None,
                input_stats: None,
                output_stats: Some(TensorStats {
                    count: d_model,
                    mean: 0.0,
                    std: 0.0,
                    l2_norm: 0.0,
                    min: 0.0,
                    max: 0.0,
                    max_abs: 0.0,
                    nan_count: 0,
                    inf_count: 0,
                }),
                weight_stats: None,
                anomalies: vec![],
            });

            // Add transformer layers
            for i in 0..n_layers {
                let layer_name = format!("transformer_block_{i}");

                if let Some(f) = filter {
                    if !layer_name.contains(f) {
                        continue;
                    }
                }

                layers.push(LayerTrace {
                    name: layer_name,
                    index: Some(i),
                    input_stats: None,
                    output_stats: None,
                    weight_stats: None,
                    anomalies: vec![],
                });
            }

            // Add final layer norm
            layers.push(LayerTrace {
                name: "final_layer_norm".to_string(),
                index: None,
                input_stats: None,
                output_stats: None,
                weight_stats: None,
                anomalies: vec![],
            });
        }
    }

    // Also check tensor_shapes for actual weight info
    // This section can be extended to extract more layer information from tensor shapes
    if let Some(shapes) = metadata.get("tensor_shapes") {
        if let Some(shapes_map) = shapes.as_object() {
            // Count tensors that match the filter (for potential future use)
            let _matching_tensors: Vec<_> = shapes_map
                .keys()
                .filter(|name| {
                    if let Some(f) = filter {
                        name.contains(f)
                    } else {
                        true
                    }
                })
                .filter(|name| !name.contains("layer") && !name.contains("block"))
                .collect();
            // Future: use _matching_tensors to add more layer info
        }
    }

    if layers.is_empty() {
        layers.push(LayerTrace {
            name: "(layer trace metadata not available)".to_string(),
            index: None,
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec!["No layer information in metadata".to_string()],
        });
    }

    layers
}

fn compare_with_reference(
    model_path: &Path,
    ref_path: &Path,
    _layers: &[LayerTrace],
    json_output: bool,
) -> Result<(), CliError> {
    validate_path(ref_path)?;

    if json_output {
        println!("{{\"comparison\": \"reference comparison not yet implemented\"}}");
    } else {
        output::section(&format!(
            "Layer Comparison: {} vs {}",
            model_path.display(),
            ref_path.display()
        ));
        println!();
        println!("{}", "Reference comparison coming soon...".yellow());
        println!();
        println!("Future features:");
        println!("  - Layer-by-layer output comparison");
        println!("  - Cosine similarity between activations");
        println!("  - Probar visual diff generation");
    }

    Ok(())
}

fn output_json(path: &Path, format: &str, layers: &[LayerTrace], summary: &TraceSummary) {
    let result = TraceResult {
        file: path.display().to_string(),
        format: format.to_string(),
        layers: layers.to_vec(),
        summary: TraceSummary {
            total_layers: summary.total_layers,
            total_parameters: summary.total_parameters,
            anomaly_count: summary.anomaly_count,
            anomalies: summary.anomalies.clone(),
        },
    };

    if let Ok(json) = serde_json::to_string_pretty(&result) {
        println!("{json}");
    }
}

fn output_text(
    path: &Path,
    format: &str,
    layers: &[LayerTrace],
    summary: &TraceSummary,
    verbose: bool,
) {
    output::section(&format!("Layer Trace: {}", path.display()));
    println!();

    output::kv("Format", format);
    output::kv("Layers", summary.total_layers);
    output::kv("Parameters", format!("{}", summary.total_parameters));

    if !summary.anomalies.is_empty() {
        println!();
        println!(
            "{}",
            format!("⚠ {} anomalies detected:", summary.anomaly_count)
                .yellow()
                .bold()
        );
        for anomaly in &summary.anomalies {
            println!("  - {}", anomaly.red());
        }
    }

    println!();
    println!("{}", "Layer Breakdown:".white().bold());

    for layer in layers {
        let idx_str = layer.index.map_or(String::new(), |i| format!("[{i}]"));
        println!("  {} {}", layer.name.cyan(), idx_str);

        if verbose {
            if let Some(ref stats) = layer.weight_stats {
                println!(
                    "    weights: {} params, mean={:.4}, std={:.4}, L2={:.4}",
                    stats.count, stats.mean, stats.std, stats.l2_norm
                );
            }

            if let Some(ref stats) = layer.output_stats {
                println!(
                    "    output:  mean={:.4}, std={:.4}, range=[{:.4}, {:.4}]",
                    stats.mean, stats.std, stats.min, stats.max
                );
            }
        }

        for anomaly in &layer.anomalies {
            println!("    {}", anomaly.red());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_stats_empty() {
        let stats = TensorStats::from_slice(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_tensor_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::from_slice(&data);

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_tensor_stats_with_nan() {
        let data = vec![1.0, f32::NAN, 3.0];
        let stats = TensorStats::from_slice(&data);

        assert_eq!(stats.nan_count, 1);
        assert!((stats.mean - 2.0).abs() < 1e-5); // Mean of 1 and 3
    }

    #[test]
    fn test_anomaly_detection() {
        let stats = TensorStats {
            count: 100,
            mean: 15.0, // Large mean
            std: 1.0,
            l2_norm: 100.0,
            min: 0.0,
            max: 20.0,
            max_abs: 20.0,
            nan_count: 0,
            inf_count: 0,
        };

        let anomalies = stats.detect_anomalies("test_layer");
        assert!(anomalies.iter().any(|a| a.contains("large mean")));
    }

    #[test]
    fn test_anomaly_detection_nan() {
        let stats = TensorStats {
            count: 100,
            mean: 0.0,
            std: 1.0,
            l2_norm: 10.0,
            min: -1.0,
            max: 1.0,
            max_abs: 1.0,
            nan_count: 5,
            inf_count: 0,
        };

        let anomalies = stats.detect_anomalies("test");
        assert!(anomalies.iter().any(|a| a.contains("NaN")));
    }
}
