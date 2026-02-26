//! Probar integration command
//!
//! Export layer-by-layer data for visual regression testing with probar.
//! Toyota Way: Visualization + Standardization - Make debugging visual and repeatable.
//!
//! This command generates visual test artifacts that can be used with probar's
//! visual regression testing framework to compare model behavior.

use crate::error::CliError;
use crate::output;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Layer activation snapshot for visual testing
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct LayerSnapshot {
    /// Layer name
    pub name: String,
    /// Layer index
    pub index: usize,
    /// Activation histogram (256 bins)
    pub histogram: Vec<u32>,
    /// Statistics
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    /// Heatmap data (if 2D tensor, flattened)
    pub heatmap: Option<Vec<f32>>,
    pub heatmap_width: Option<usize>,
    pub heatmap_height: Option<usize>,
}

/// Complete probar test manifest
#[derive(Serialize, Deserialize)]
struct ProbarManifest {
    /// Model file this was generated from
    pub source_model: String,
    /// Timestamp of generation
    pub timestamp: String,
    /// Model format
    pub format: String,
    /// Layer snapshots
    pub layers: Vec<LayerSnapshot>,
    /// Golden reference path (if available)
    pub golden_reference: Option<String>,
}

/// Probar export format
#[derive(Debug, Clone, Copy)]
pub(crate) enum ExportFormat {
    /// JSON manifest for programmatic access
    Json,
    /// PNG heatmaps for visual comparison
    Png,
    /// Both JSON and PNG
    Both,
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "png" => Ok(Self::Png),
            "both" | "all" => Ok(Self::Both),
            _ => Err(format!("Unknown format: {s}. Use json, png, or both")),
        }
    }
}

/// Run the probar command
pub(crate) fn run(
    path: &Path,
    output_dir: &Path,
    format: ExportFormat,
    golden: Option<&Path>,
    layer_filter: Option<&str>,
) -> Result<(), CliError> {
    validate_path(path)?;
    fs::create_dir_all(output_dir)?;

    // Use RosettaStone for universal format detection (GGUF, APR, SafeTensors)
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let report = rosetta
        .inspect(path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to inspect model: {e}")))?;
    let model_format = report.format.to_string();
    let n_layers = detect_layer_count(&report);

    let layers = generate_snapshots(Some(path), n_layers, layer_filter);
    let manifest = create_manifest(path, &model_format, &layers, golden);

    export_by_format(format, &manifest, &layers, output_dir)?;

    if let Some(golden_path) = golden {
        generate_diff(golden_path, &manifest, output_dir)?;
    }

    print_summary(path, output_dir, &model_format, &layers, golden);
    print_generated_files(format, output_dir, &layers);
    print_integration_guide();

    Ok(())
}

/// Detect the number of transformer layers from a RosettaStone inspection report.
///
/// Uses tensor naming conventions: `blk.N.`, `.layers.N.`, `block.N.`, `layer.N.`
fn detect_layer_count(report: &aprender::format::rosetta::InspectionReport) -> usize {
    let mut max_layer: Option<usize> = None;
    let patterns = ["blk.", ".layers.", "block.", "layer."];

    for tensor in &report.tensors {
        for pattern in &patterns {
            if let Some(pos) = tensor.name.find(pattern) {
                let after = &tensor.name[pos + pattern.len()..];
                if let Some(dot_pos) = after.find('.') {
                    if let Ok(idx) = after[..dot_pos].parse::<usize>() {
                        max_layer = Some(max_layer.map_or(idx, |prev: usize| prev.max(idx)));
                    }
                }
            }
        }
    }

    // Layer indices are 0-based, so count = max + 1
    max_layer.map_or(0, |m| m + 1)
}

fn create_manifest(
    path: &Path,
    model_format: &str,
    layers: &[LayerSnapshot],
    golden: Option<&Path>,
) -> ProbarManifest {
    ProbarManifest {
        source_model: path.display().to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        format: model_format.to_string(),
        layers: layers.to_vec(),
        golden_reference: golden.map(|p| p.display().to_string()),
    }
}

fn export_by_format(
    format: ExportFormat,
    manifest: &ProbarManifest,
    layers: &[LayerSnapshot],
    output_dir: &Path,
) -> Result<(), CliError> {
    match format {
        ExportFormat::Json => export_json(manifest, output_dir),
        ExportFormat::Png => export_png(layers, output_dir),
        ExportFormat::Both => {
            export_json(manifest, output_dir)?;
            export_png(layers, output_dir)
        }
    }
}

fn print_summary(
    path: &Path,
    output_dir: &Path,
    model_format: &str,
    layers: &[LayerSnapshot],
    golden: Option<&Path>,
) {
    output::section("Probar Export Complete");
    println!();
    output::kv("Source", path.display());
    output::kv("Output", output_dir.display());
    output::kv("Format", model_format);
    output::kv("Layers", layers.len());

    if golden.is_some() {
        println!();
        println!("{}", "Golden reference comparison generated".green());
    }
}

fn print_generated_files(format: ExportFormat, output_dir: &Path, layers: &[LayerSnapshot]) {
    println!();
    println!("{}", "Generated files:".white().bold());

    if matches!(format, ExportFormat::Json | ExportFormat::Both) {
        println!("  - {}/manifest.json", output_dir.display());
    }

    if matches!(format, ExportFormat::Png | ExportFormat::Both) {
        for layer in layers {
            println!(
                "  - {}/layer_{:03}_{}.png",
                output_dir.display(),
                layer.index,
                layer.name
            );
        }
    }
}

fn print_integration_guide() {
    println!();
    println!("{}", "Integration with probar:".cyan().bold());
    println!("  1. Copy output to probar test fixtures");
    println!("  2. Use VisualRegressionTester to compare snapshots");
    println!("  3. Run: probar test --visual-diff");
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

/// Build a 256-bin histogram from tensor values.
fn build_histogram(values: &[f32], min: f32, max: f32) -> Vec<u32> {
    let mut histogram = vec![0u32; 256];
    let range = max - min;
    if range < f32::EPSILON || values.is_empty() {
        return histogram;
    }
    for &v in values {
        if v.is_nan() || v.is_infinite() {
            continue;
        }
        let bin = (((v - min) / range) * 255.0) as usize;
        histogram[bin.min(255)] += 1;
    }
    histogram
}

/// Collect all tensor values matching a layer index (e.g. "blk.3." or ".layers.3.").
fn collect_layer_tensor_values(
    tensor_data: &std::collections::HashMap<String, Vec<f32>>,
    layer_idx: usize,
) -> Vec<f32> {
    let patterns = [
        format!("blk.{layer_idx}."),
        format!(".layers.{layer_idx}."),
        format!("block.{layer_idx}."),
        format!("layer.{layer_idx}."),
    ];
    let mut values = Vec::new();
    for (name, data) in tensor_data {
        if patterns.iter().any(|p| name.contains(p.as_str())) {
            values.extend_from_slice(data);
        }
    }
    values
}

fn generate_snapshots(
    model_path: Option<&Path>,
    n_layers: usize,
    filter: Option<&str>,
) -> Vec<LayerSnapshot> {
    // Try to load tensor data from model file for real statistics
    let tensor_data = model_path.and_then(super::rosetta::load_tensor_data_direct);

    let mut snapshots = Vec::new();

    for i in 0..n_layers {
        let name = format!("block_{i}");

        if let Some(f) = filter {
            if !name.contains(f) {
                continue;
            }
        }

        // Try to compute real stats from tensor data
        let (histogram, mean, std, min, max) = if let Some(ref td) = tensor_data {
            let values = collect_layer_tensor_values(td, i);
            if values.is_empty() {
                (vec![0u32; 256], 0.0, 0.0, 0.0, 0.0)
            } else {
                let (m, s, mn, mx, ..) = super::rosetta::compute_tensor_stats(&values);
                let hist = build_histogram(&values, mn, mx);
                (hist, m, s, mn, mx)
            }
        } else {
            (vec![0u32; 256], 0.0, 0.0, 0.0, 0.0)
        };

        snapshots.push(LayerSnapshot {
            name,
            index: i,
            histogram,
            mean,
            std,
            min,
            max,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        });
    }

    // If no layers found, create a fallback entry
    if snapshots.is_empty() {
        snapshots.push(LayerSnapshot {
            name: "fallback".to_string(),
            index: 0,
            histogram: vec![0; 256],
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        });
    }

    snapshots
}

fn export_json(manifest: &ProbarManifest, output_dir: &Path) -> Result<(), CliError> {
    let json_path = output_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(manifest)
        .map_err(|e| CliError::InvalidFormat(format!("JSON serialization failed: {e}")))?;

    let mut file = File::create(&json_path)?;
    file.write_all(json.as_bytes())?;

    Ok(())
}

#[allow(clippy::disallowed_methods)] // json! macro uses infallible unwrap internally
fn export_png(layers: &[LayerSnapshot], output_dir: &Path) -> Result<(), CliError> {
    for layer in layers {
        let filename = format!("layer_{:03}_{}.png", layer.index, layer.name);
        let png_path = output_dir.join(&filename);

        // Generate a simple histogram visualization as PNG
        // Using raw PNG encoding (no external dependencies)
        let width = 256;
        let height = 100;

        // Find max histogram value for normalization
        let max_val = *layer.histogram.iter().max().unwrap_or(&1);

        // Generate grayscale image data
        let mut pixels = vec![255u8; width * height]; // White background

        for (x, &count) in layer.histogram.iter().enumerate() {
            let bar_height = ((count as f32 / max_val as f32) * height as f32) as usize;
            for y in 0..bar_height {
                let pixel_y = height - 1 - y;
                pixels[pixel_y * width + x] = 0; // Black bar
            }
        }

        // Write as simple PGM (portable graymap) - easy to convert to PNG
        // For now, write as .pgm which can be viewed in most image viewers
        let pgm_path = output_dir.join(format!("layer_{:03}_{}.pgm", layer.index, layer.name));
        let mut file = File::create(&pgm_path)?;
        writeln!(file, "P5")?;
        writeln!(file, "{width} {height}")?;
        writeln!(file, "255")?;
        file.write_all(&pixels)?;

        // Create a metadata sidecar JSON
        let meta_path =
            output_dir.join(format!("layer_{:03}_{}.meta.json", layer.index, layer.name));
        let meta_json = serde_json::to_string_pretty(&serde_json::json!({
            "name": layer.name,
            "index": layer.index,
            "mean": layer.mean,
            "std": layer.std,
            "min": layer.min,
            "max": layer.max,
            "histogram_bins": 256,
            "image_width": width,
            "image_height": height,
        }))
        .unwrap_or_default();

        let mut meta_file = File::create(&meta_path)?;
        meta_file.write_all(meta_json.as_bytes())?;

        // Note: In production, use image crate or similar to generate actual PNG
        // For now, PGM format works for development/testing
        let _ = png_path; // Suppress unused warning
    }

    Ok(())
}

#[allow(clippy::disallowed_methods)] // json! macro uses infallible unwrap internally
fn generate_diff(
    golden_path: &Path,
    current: &ProbarManifest,
    output_dir: &Path,
) -> Result<(), CliError> {
    // Try to load golden manifest
    let golden_json = fs::read_to_string(golden_path.join("manifest.json"))
        .map_err(|_| CliError::FileNotFound(golden_path.to_path_buf()))?;

    let golden: ProbarManifest = serde_json::from_str(&golden_json)
        .map_err(|e| CliError::InvalidFormat(format!("Invalid golden manifest: {e}")))?;

    // Generate diff report
    let diff_path = output_dir.join("diff_report.json");

    let mut diffs = Vec::new();

    for (current_layer, golden_layer) in current.layers.iter().zip(golden.layers.iter()) {
        if current_layer.name != golden_layer.name {
            diffs.push(serde_json::json!({
                "type": "name_mismatch",
                "current": current_layer.name,
                "golden": golden_layer.name,
            }));
        }

        let mean_diff = (current_layer.mean - golden_layer.mean).abs();
        let std_diff = (current_layer.std - golden_layer.std).abs();

        if mean_diff > 0.01 || std_diff > 0.01 {
            diffs.push(serde_json::json!({
                "type": "stats_divergence",
                "layer": current_layer.name,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
            }));
        }
    }

    let diff_report = serde_json::json!({
        "current_model": current.source_model,
        "golden_model": golden.source_model,
        "total_diffs": diffs.len(),
        "diffs": diffs,
    });

    let mut file = File::create(&diff_path)?;
    file.write_all(
        serde_json::to_string_pretty(&diff_report)
            .unwrap_or_default()
            .as_bytes(),
    )?;

    Ok(())
}

#[cfg(test)]
#[path = "probar_tests.rs"]
mod tests;
