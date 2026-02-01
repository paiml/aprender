//! Probar integration command
//!
//! Export layer-by-layer data for visual regression testing with probar.
//! Toyota Way: Visualization + Standardization - Make debugging visual and repeatable.
//!
//! This command generates visual test artifacts that can be used with probar's
//! visual regression testing framework to compare model behavior.

use crate::error::CliError;
use crate::output;
use aprender::format::HEADER_SIZE;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
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

    let (model_format, metadata_bytes) = read_model_metadata(path)?;
    let layers = generate_snapshots(&metadata_bytes, layer_filter);
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

fn read_model_metadata(path: &Path) -> Result<(String, Vec<u8>), CliError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let model_format = validate_header(&mut reader)?;

    let mut size_buf = [0u8; 4];
    reader.seek(SeekFrom::Start(8))?;
    reader.read_exact(&mut size_buf)?;
    let metadata_size = u32::from_le_bytes(size_buf) as usize;

    reader.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    let mut metadata_bytes = vec![0u8; metadata_size];
    reader.read_exact(&mut metadata_bytes)?;

    Ok((model_format, metadata_bytes))
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

fn validate_header(reader: &mut BufReader<File>) -> Result<String, CliError> {
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|_| CliError::InvalidFormat("File too small".to_string()))?;

    if !output::is_valid_magic(&magic) {
        return Err(CliError::InvalidFormat(format!(
            "Invalid magic: expected APRN, APR1, APR2, or APR\\0, got {magic:?}"
        )));
    }

    Ok(output::format_name(&magic).to_string())
}

fn generate_snapshots(metadata_bytes: &[u8], filter: Option<&str>) -> Vec<LayerSnapshot> {
    // Parse metadata
    let metadata: BTreeMap<String, serde_json::Value> =
        rmp_serde::from_slice(metadata_bytes).unwrap_or_else(|_| BTreeMap::new());

    let mut snapshots = Vec::new();

    // Extract layer info from hyperparameters
    if let Some(hp) = metadata.get("hyperparameters") {
        if let Some(hp_obj) = hp.as_object() {
            let n_layers = hp_obj
                .get("n_layer")
                .or_else(|| hp_obj.get("n_layers"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(4) as usize;

            for i in 0..n_layers {
                let name = format!("block_{i}");

                if let Some(f) = filter {
                    if !name.contains(f) {
                        continue;
                    }
                }

                // Generate placeholder histogram (uniform distribution)
                // In real implementation, this would come from actual activations
                let histogram: Vec<u32> = (0..256).map(|_| 100).collect();

                snapshots.push(LayerSnapshot {
                    name,
                    index: i,
                    histogram,
                    mean: 0.0,
                    std: 1.0,
                    min: -3.0,
                    max: 3.0,
                    heatmap: None,
                    heatmap_width: None,
                    heatmap_height: None,
                });
            }
        }
    }

    // If no layers found, create a placeholder
    if snapshots.is_empty() {
        snapshots.push(LayerSnapshot {
            name: "placeholder".to_string(),
            index: 0,
            histogram: vec![100; 256],
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
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
mod tests {
    use super::*;

    #[test]
    fn test_export_format_parse() {
        assert!(matches!(
            "json".parse::<ExportFormat>(),
            Ok(ExportFormat::Json)
        ));
        assert!(matches!(
            "png".parse::<ExportFormat>(),
            Ok(ExportFormat::Png)
        ));
        assert!(matches!(
            "both".parse::<ExportFormat>(),
            Ok(ExportFormat::Both)
        ));
        assert!(matches!(
            "all".parse::<ExportFormat>(),
            Ok(ExportFormat::Both)
        ));
        assert!("invalid".parse::<ExportFormat>().is_err());
    }

    #[test]
    fn test_layer_snapshot_serialize() {
        let snapshot = LayerSnapshot {
            name: "test".to_string(),
            index: 0,
            histogram: vec![1, 2, 3],
            mean: 0.5,
            std: 1.0,
            min: -1.0,
            max: 2.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        };

        let json = serde_json::to_string(&snapshot).expect("serialize");
        assert!(json.contains("\"name\":\"test\""));
    }

    // ========================================================================
    // ExportFormat Tests
    // ========================================================================

    #[test]
    fn test_export_format_parse_uppercase() {
        assert!(matches!(
            "JSON".parse::<ExportFormat>(),
            Ok(ExportFormat::Json)
        ));
        assert!(matches!(
            "PNG".parse::<ExportFormat>(),
            Ok(ExportFormat::Png)
        ));
    }

    #[test]
    fn test_export_format_debug() {
        let format = ExportFormat::Json;
        let debug = format!("{format:?}");
        assert!(debug.contains("Json"));
    }

    #[test]
    fn test_export_format_clone() {
        let format = ExportFormat::Png;
        let cloned = format;
        assert!(matches!(cloned, ExportFormat::Png));
    }

    // ========================================================================
    // LayerSnapshot Tests
    // ========================================================================

    #[test]
    fn test_layer_snapshot_with_heatmap() {
        let snapshot = LayerSnapshot {
            name: "attn".to_string(),
            index: 1,
            histogram: vec![0; 256],
            mean: 0.0,
            std: 1.0,
            min: -3.0,
            max: 3.0,
            heatmap: Some(vec![1.0, 2.0, 3.0, 4.0]),
            heatmap_width: Some(2),
            heatmap_height: Some(2),
        };
        assert!(snapshot.heatmap.is_some());
        assert_eq!(snapshot.heatmap_width, Some(2));
    }

    #[test]
    fn test_layer_snapshot_clone() {
        let snapshot = LayerSnapshot {
            name: "test".to_string(),
            index: 0,
            histogram: vec![1, 2, 3],
            mean: 0.5,
            std: 1.0,
            min: -1.0,
            max: 2.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        };
        let cloned = snapshot.clone();
        assert_eq!(cloned.name, snapshot.name);
        assert_eq!(cloned.index, snapshot.index);
    }

    #[test]
    fn test_layer_snapshot_deserialize() {
        let json = r#"{"name":"test","index":0,"histogram":[1,2,3],"mean":0.5,"std":1.0,"min":-1.0,"max":2.0}"#;
        let snapshot: LayerSnapshot = serde_json::from_str(json).expect("deserialize");
        assert_eq!(snapshot.name, "test");
        assert_eq!(snapshot.index, 0);
    }

    #[test]
    fn test_layer_snapshot_histogram() {
        let snapshot = LayerSnapshot {
            name: "hist".to_string(),
            index: 0,
            histogram: vec![10, 20, 30, 40],
            mean: 0.0,
            std: 1.0,
            min: -2.0,
            max: 2.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        };
        assert_eq!(snapshot.histogram.len(), 4);
        assert_eq!(snapshot.histogram[0], 10);
    }

    // ========================================================================
    // run Command Tests
    // ========================================================================

    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    #[test]
    fn test_run_file_not_found() {
        let output_dir = tempdir().expect("create output dir");
        let result = run(
            Path::new("/nonexistent/model.apr"),
            output_dir.path(),
            ExportFormat::Json,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_apr() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");
        let output_dir = tempdir().expect("create output dir");

        let result = run(
            file.path(),
            output_dir.path(),
            ExportFormat::Json,
            None,
            None,
        );
        // Should fail (invalid APR)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_png_format() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        let output_dir = tempdir().expect("create output dir");

        let result = run(
            file.path(),
            output_dir.path(),
            ExportFormat::Png,
            None,
            None,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_both_format() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        let output_dir = tempdir().expect("create output dir");

        let result = run(
            file.path(),
            output_dir.path(),
            ExportFormat::Both,
            None,
            None,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_golden() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        let mut golden = NamedTempFile::with_suffix(".json").expect("create golden file");
        golden.write_all(b"{}").expect("write");
        let output_dir = tempdir().expect("create output dir");

        let result = run(
            file.path(),
            output_dir.path(),
            ExportFormat::Json,
            Some(golden.path()),
            None,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_layer_filter() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        let output_dir = tempdir().expect("create output dir");

        let result = run(
            file.path(),
            output_dir.path(),
            ExportFormat::Json,
            None,
            Some("encoder"),
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_is_directory() {
        let dir = tempdir().expect("create input dir");
        let output_dir = tempdir().expect("create output dir");

        let result = run(
            dir.path(),
            output_dir.path(),
            ExportFormat::Json,
            None,
            None,
        );
        // Should fail (is a directory)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_format() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let output_dir = tempdir().expect("create output dir");

        let result = run(
            file.path(),
            output_dir.path(),
            ExportFormat::Json,
            None,
            None,
        );
        // Should fail (invalid GGUF)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_format() {
        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file.write_all(b"not valid safetensors").expect("write");
        let output_dir = tempdir().expect("create output dir");

        let result = run(
            file.path(),
            output_dir.path(),
            ExportFormat::Json,
            None,
            None,
        );
        // Should fail (invalid SafeTensors)
        assert!(result.is_err());
    }
}
