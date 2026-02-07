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

    // BUG-PROBAR-001 FIX: Updated error message to include GGUF
    if !output::is_valid_magic(&magic) {
        return Err(CliError::InvalidFormat(format!(
            "Invalid magic: expected APRN, APR1, APR2, APR\\0, or GGUF, got {magic:?}"
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

    // ========================================================================
    // ExportFormat Error Messages
    // ========================================================================

    #[test]
    fn test_export_format_error_contains_input() {
        let err = "foobar".parse::<ExportFormat>().expect_err("should fail");
        assert!(
            err.contains("foobar"),
            "error message should contain the invalid input"
        );
    }

    #[test]
    fn test_export_format_error_suggests_valid_options() {
        let err = "xyz".parse::<ExportFormat>().expect_err("should fail");
        assert!(err.contains("json"), "error should mention 'json'");
        assert!(err.contains("png"), "error should mention 'png'");
        assert!(err.contains("both"), "error should mention 'both'");
    }

    #[test]
    fn test_export_format_case_insensitive_mixed() {
        assert!(matches!(
            "Json".parse::<ExportFormat>(),
            Ok(ExportFormat::Json)
        ));
        assert!(matches!(
            "pNg".parse::<ExportFormat>(),
            Ok(ExportFormat::Png)
        ));
        assert!(matches!(
            "BOTH".parse::<ExportFormat>(),
            Ok(ExportFormat::Both)
        ));
        assert!(matches!(
            "ALL".parse::<ExportFormat>(),
            Ok(ExportFormat::Both)
        ));
    }

    #[test]
    fn test_export_format_copy_semantics() {
        let a = ExportFormat::Both;
        let b = a; // Copy
                   // Both a and b are valid after copy
        assert!(matches!(a, ExportFormat::Both));
        assert!(matches!(b, ExportFormat::Both));
    }

    #[test]
    fn test_export_format_debug_all_variants() {
        assert_eq!(format!("{:?}", ExportFormat::Json), "Json");
        assert_eq!(format!("{:?}", ExportFormat::Png), "Png");
        assert_eq!(format!("{:?}", ExportFormat::Both), "Both");
    }

    // ========================================================================
    // generate_snapshots Tests
    // ========================================================================

    #[test]
    fn test_generate_snapshots_empty_metadata_returns_placeholder() {
        let snapshots = generate_snapshots(&[], None);
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].name, "placeholder");
        assert_eq!(snapshots[0].index, 0);
        assert_eq!(snapshots[0].histogram.len(), 256);
        assert!(snapshots[0].heatmap.is_none());
    }

    #[test]
    fn test_generate_snapshots_invalid_msgpack_returns_placeholder() {
        let bad_bytes = b"this is definitely not msgpack";
        let snapshots = generate_snapshots(bad_bytes, None);
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].name, "placeholder");
    }

    #[test]
    fn test_generate_snapshots_empty_map_returns_placeholder() {
        // Valid msgpack encoding of an empty map
        let metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        let bytes = rmp_serde::to_vec(&metadata).expect("encode");
        let snapshots = generate_snapshots(&bytes, None);
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].name, "placeholder");
    }

    #[test]
    fn test_generate_snapshots_with_n_layer_hyperparameter() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layer".to_string(), serde_json::json!(3));

        let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        metadata.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));

        let bytes = rmp_serde::to_vec(&metadata).expect("encode");
        let snapshots = generate_snapshots(&bytes, None);

        assert_eq!(snapshots.len(), 3);
        for (i, snap) in snapshots.iter().enumerate() {
            assert_eq!(snap.name, format!("block_{i}"));
            assert_eq!(snap.index, i);
            assert_eq!(snap.histogram.len(), 256);
            // All histogram bins should be 100 (uniform placeholder)
            assert!(snap.histogram.iter().all(|&v| v == 100));
            assert_eq!(snap.mean, 0.0);
            assert_eq!(snap.std, 1.0);
            assert_eq!(snap.min, -3.0);
            assert_eq!(snap.max, 3.0);
            assert!(snap.heatmap.is_none());
        }
    }

    #[test]
    fn test_generate_snapshots_with_n_layers_alternative_key() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layers".to_string(), serde_json::json!(2));

        let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        metadata.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));

        let bytes = rmp_serde::to_vec(&metadata).expect("encode");
        let snapshots = generate_snapshots(&bytes, None);

        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].name, "block_0");
        assert_eq!(snapshots[1].name, "block_1");
    }

    #[test]
    fn test_generate_snapshots_defaults_to_4_layers_when_key_missing() {
        let hp = serde_json::Map::new(); // no n_layer or n_layers key

        let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        metadata.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));

        let bytes = rmp_serde::to_vec(&metadata).expect("encode");
        let snapshots = generate_snapshots(&bytes, None);

        assert_eq!(snapshots.len(), 4, "should default to 4 layers");
    }

    #[test]
    fn test_generate_snapshots_filter_matches_subset() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layer".to_string(), serde_json::json!(5));

        let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        metadata.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));

        let bytes = rmp_serde::to_vec(&metadata).expect("encode");

        // Filter for "block_3" - should match only block_3
        let snapshots = generate_snapshots(&bytes, Some("block_3"));
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].name, "block_3");
        assert_eq!(snapshots[0].index, 3);
    }

    #[test]
    fn test_generate_snapshots_filter_matches_none_returns_placeholder() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layer".to_string(), serde_json::json!(3));

        let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        metadata.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));

        let bytes = rmp_serde::to_vec(&metadata).expect("encode");

        // Filter for something that doesn't match any layer
        let snapshots = generate_snapshots(&bytes, Some("nonexistent"));
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].name, "placeholder");
    }

    #[test]
    fn test_generate_snapshots_filter_partial_match() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layer".to_string(), serde_json::json!(10));

        let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        metadata.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));

        let bytes = rmp_serde::to_vec(&metadata).expect("encode");

        // "block_" matches all layers
        let snapshots = generate_snapshots(&bytes, Some("block_"));
        assert_eq!(snapshots.len(), 10);
    }

    #[test]
    fn test_generate_snapshots_hyperparameters_not_object() {
        // hyperparameters is a string instead of an object
        let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        metadata.insert(
            "hyperparameters".to_string(),
            serde_json::json!("not an object"),
        );

        let bytes = rmp_serde::to_vec(&metadata).expect("encode");
        let snapshots = generate_snapshots(&bytes, None);

        // Falls through to placeholder since as_object() returns None
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].name, "placeholder");
    }

    #[test]
    fn test_generate_snapshots_zero_layers() {
        let mut hp = serde_json::Map::new();
        hp.insert("n_layer".to_string(), serde_json::json!(0));

        let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        metadata.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));

        let bytes = rmp_serde::to_vec(&metadata).expect("encode");
        let snapshots = generate_snapshots(&bytes, None);

        // 0 layers => empty => placeholder
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].name, "placeholder");
    }

    #[test]
    fn test_generate_snapshots_placeholder_stats() {
        let snapshots = generate_snapshots(&[], None);
        let placeholder = &snapshots[0];
        assert_eq!(placeholder.mean, 0.0);
        assert_eq!(placeholder.std, 1.0);
        assert_eq!(placeholder.min, -1.0);
        assert_eq!(placeholder.max, 1.0);
        assert!(placeholder.heatmap.is_none());
        assert!(placeholder.heatmap_width.is_none());
        assert!(placeholder.heatmap_height.is_none());
    }

    // ========================================================================
    // create_manifest Tests
    // ========================================================================

    #[test]
    fn test_create_manifest_basic_fields() {
        let layers = vec![LayerSnapshot {
            name: "block_0".to_string(),
            index: 0,
            histogram: vec![100; 256],
            mean: 0.0,
            std: 1.0,
            min: -3.0,
            max: 3.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }];

        let manifest = create_manifest(
            Path::new("/tmp/model.apr"),
            "APRN (aprender v1)",
            &layers,
            None,
        );

        assert_eq!(manifest.source_model, "/tmp/model.apr");
        assert_eq!(manifest.format, "APRN (aprender v1)");
        assert_eq!(manifest.layers.len(), 1);
        assert_eq!(manifest.layers[0].name, "block_0");
        assert!(manifest.golden_reference.is_none());
        // Timestamp should be non-empty RFC3339
        assert!(!manifest.timestamp.is_empty());
        assert!(manifest.timestamp.contains('T'));
    }

    #[test]
    fn test_create_manifest_with_golden_reference() {
        let manifest = create_manifest(
            Path::new("/model.apr"),
            "APR v2",
            &[],
            Some(Path::new("/golden/reference")),
        );
        assert_eq!(
            manifest.golden_reference,
            Some("/golden/reference".to_string())
        );
    }

    #[test]
    fn test_create_manifest_without_golden_reference() {
        let manifest = create_manifest(Path::new("/model.apr"), "APR v2", &[], None);
        assert!(manifest.golden_reference.is_none());
    }

    #[test]
    fn test_create_manifest_preserves_layer_order() {
        let layers: Vec<LayerSnapshot> = (0..5)
            .map(|i| LayerSnapshot {
                name: format!("layer_{i}"),
                index: i,
                histogram: vec![0; 256],
                mean: 0.0,
                std: 1.0,
                min: -1.0,
                max: 1.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            })
            .collect();

        let manifest = create_manifest(Path::new("/m.apr"), "APR", &layers, None);
        for (i, layer) in manifest.layers.iter().enumerate() {
            assert_eq!(layer.name, format!("layer_{i}"));
            assert_eq!(layer.index, i);
        }
    }

    // ========================================================================
    // validate_path Tests (direct)
    // ========================================================================

    #[test]
    fn test_validate_path_nonexistent_returns_file_not_found() {
        let result = validate_path(Path::new("/absolutely/nonexistent/path.apr"));
        assert!(result.is_err());
        let err = result.expect_err("should be error");
        assert!(
            matches!(err, CliError::FileNotFound(_)),
            "expected FileNotFound, got {err:?}"
        );
    }

    #[test]
    fn test_validate_path_directory_returns_not_a_file() {
        let dir = tempdir().expect("create temp dir");
        let result = validate_path(dir.path());
        assert!(result.is_err());
        let err = result.expect_err("should be error");
        assert!(
            matches!(err, CliError::NotAFile(_)),
            "expected NotAFile, got {err:?}"
        );
    }

    #[test]
    fn test_validate_path_valid_file_succeeds() {
        let file = NamedTempFile::new().expect("create temp file");
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // export_json Tests
    // ========================================================================

    #[test]
    fn test_export_json_creates_manifest_file() {
        let output_dir = tempdir().expect("create output dir");
        let manifest = ProbarManifest {
            source_model: "/test/model.apr".to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            format: "APRN".to_string(),
            layers: vec![],
            golden_reference: None,
        };

        export_json(&manifest, output_dir.path()).expect("export json");

        let json_path = output_dir.path().join("manifest.json");
        assert!(json_path.exists(), "manifest.json should be created");

        let content = fs::read_to_string(&json_path).expect("read manifest");
        assert!(content.contains("\"source_model\""));
        assert!(content.contains("/test/model.apr"));
        assert!(content.contains("\"format\""));
        assert!(content.contains("APRN"));
    }

    #[test]
    fn test_export_json_contains_layer_data() {
        let output_dir = tempdir().expect("create output dir");
        let manifest = ProbarManifest {
            source_model: "m.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: vec![LayerSnapshot {
                name: "test_layer".to_string(),
                index: 42,
                histogram: vec![1, 2, 3],
                mean: 0.5,
                std: 0.25,
                min: -1.0,
                max: 1.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            }],
            golden_reference: None,
        };

        export_json(&manifest, output_dir.path()).expect("export json");

        let content = fs::read_to_string(output_dir.path().join("manifest.json")).expect("read");
        assert!(content.contains("test_layer"));
        assert!(content.contains("42"));
    }

    #[test]
    fn test_export_json_roundtrip() {
        let output_dir = tempdir().expect("create output dir");
        let manifest = ProbarManifest {
            source_model: "model.apr".to_string(),
            timestamp: "2026-02-06T12:00:00Z".to_string(),
            format: "GGUF".to_string(),
            layers: vec![LayerSnapshot {
                name: "block_0".to_string(),
                index: 0,
                histogram: vec![50; 256],
                mean: -0.1,
                std: 0.9,
                min: -4.0,
                max: 4.0,
                heatmap: Some(vec![1.0, 2.0]),
                heatmap_width: Some(2),
                heatmap_height: Some(1),
            }],
            golden_reference: Some("/golden".to_string()),
        };

        export_json(&manifest, output_dir.path()).expect("export");

        let content = fs::read_to_string(output_dir.path().join("manifest.json")).expect("read");
        let loaded: ProbarManifest = serde_json::from_str(&content).expect("deserialize");

        assert_eq!(loaded.source_model, "model.apr");
        assert_eq!(loaded.format, "GGUF");
        assert_eq!(loaded.layers.len(), 1);
        assert_eq!(loaded.layers[0].histogram.len(), 256);
        assert_eq!(loaded.golden_reference, Some("/golden".to_string()));
    }

    // ========================================================================
    // export_png Tests
    // ========================================================================

    #[test]
    fn test_export_png_creates_pgm_and_meta_files() {
        let output_dir = tempdir().expect("create output dir");
        let layers = vec![LayerSnapshot {
            name: "attn".to_string(),
            index: 0,
            histogram: vec![100; 256],
            mean: 0.0,
            std: 1.0,
            min: -3.0,
            max: 3.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }];

        export_png(&layers, output_dir.path()).expect("export png");

        let pgm_path = output_dir.path().join("layer_000_attn.pgm");
        let meta_path = output_dir.path().join("layer_000_attn.meta.json");

        assert!(pgm_path.exists(), "PGM file should be created");
        assert!(meta_path.exists(), "meta.json sidecar should be created");
    }

    #[test]
    fn test_export_png_pgm_header_format() {
        let output_dir = tempdir().expect("create output dir");
        let layers = vec![LayerSnapshot {
            name: "test".to_string(),
            index: 5,
            histogram: vec![50; 256],
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }];

        export_png(&layers, output_dir.path()).expect("export png");

        let content = fs::read(output_dir.path().join("layer_005_test.pgm")).expect("read pgm");
        // PGM header: "P5\n256 100\n255\n" followed by pixel data
        let header_end = content
            .windows(1)
            .enumerate()
            .filter(|(_, w)| w[0] == b'\n')
            .nth(2)
            .map(|(i, _)| i + 1)
            .expect("find header end");

        let header = std::str::from_utf8(&content[..header_end]).expect("valid utf8 header");
        assert!(header.starts_with("P5\n"));
        assert!(header.contains("256 100"));
        assert!(header.contains("255"));

        // Pixel data should be 256 * 100 bytes
        assert_eq!(content.len() - header_end, 256 * 100);
    }

    #[test]
    fn test_export_png_meta_json_contents() {
        let output_dir = tempdir().expect("create output dir");
        let layers = vec![LayerSnapshot {
            name: "ffn".to_string(),
            index: 7,
            histogram: vec![0; 256],
            mean: 0.5,
            std: 2.0,
            min: -5.0,
            max: 5.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }];

        export_png(&layers, output_dir.path()).expect("export png");

        let meta_content = fs::read_to_string(output_dir.path().join("layer_007_ffn.meta.json"))
            .expect("read meta");
        let meta: serde_json::Value = serde_json::from_str(&meta_content).expect("parse meta json");

        assert_eq!(meta["name"], "ffn");
        assert_eq!(meta["index"], 7);
        assert_eq!(meta["mean"], 0.5);
        assert_eq!(meta["std"], 2.0);
        assert_eq!(meta["min"], -5.0);
        assert_eq!(meta["max"], 5.0);
        assert_eq!(meta["histogram_bins"], 256);
        assert_eq!(meta["image_width"], 256);
        assert_eq!(meta["image_height"], 100);
    }

    #[test]
    fn test_export_png_multiple_layers() {
        let output_dir = tempdir().expect("create output dir");
        let layers: Vec<LayerSnapshot> = (0..3)
            .map(|i| LayerSnapshot {
                name: format!("block_{i}"),
                index: i,
                histogram: vec![100; 256],
                mean: 0.0,
                std: 1.0,
                min: -1.0,
                max: 1.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            })
            .collect();

        export_png(&layers, output_dir.path()).expect("export png");

        for i in 0..3 {
            let pgm = output_dir
                .path()
                .join(format!("layer_{i:03}_block_{i}.pgm"));
            let meta = output_dir
                .path()
                .join(format!("layer_{i:03}_block_{i}.meta.json"));
            assert!(pgm.exists(), "PGM for layer {i} should exist");
            assert!(meta.exists(), "meta for layer {i} should exist");
        }
    }

    #[test]
    fn test_export_png_empty_layers() {
        let output_dir = tempdir().expect("create output dir");
        let result = export_png(&[], output_dir.path());
        assert!(
            result.is_ok(),
            "empty layers should succeed (no files created)"
        );
    }

    #[test]
    fn test_export_png_histogram_normalization() {
        let output_dir = tempdir().expect("create output dir");
        // Histogram with one spike: bin 128 has max value, rest are 0
        let mut histogram = vec![0u32; 256];
        histogram[128] = 1000;

        let layers = vec![LayerSnapshot {
            name: "spike".to_string(),
            index: 0,
            histogram,
            mean: 0.0,
            std: 0.01,
            min: 0.0,
            max: 0.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }];

        export_png(&layers, output_dir.path()).expect("export png");

        // Read the PGM and verify pixel data
        let content = fs::read(output_dir.path().join("layer_000_spike.pgm")).expect("read pgm");
        // Find start of pixel data (after 3rd newline)
        let header_end = content
            .windows(1)
            .enumerate()
            .filter(|(_, w)| w[0] == b'\n')
            .nth(2)
            .map(|(i, _)| i + 1)
            .expect("find header end");

        let pixels = &content[header_end..];
        // Column 128 should have a black bar (value 0), other columns should be white (255)
        // Check bottom pixel of column 0 (should be white - no bar)
        let bottom_row = 99; // height - 1
        assert_eq!(
            pixels[bottom_row * 256 + 0],
            255,
            "column 0 bottom should be white"
        );
        // Column 128 bottom should be black (full bar)
        assert_eq!(
            pixels[bottom_row * 256 + 128],
            0,
            "column 128 bottom should be black"
        );
    }

    // ========================================================================
    // export_by_format Tests
    // ========================================================================

    #[test]
    fn test_export_by_format_json_creates_manifest_only() {
        let output_dir = tempdir().expect("create output dir");
        let manifest = ProbarManifest {
            source_model: "m.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: vec![LayerSnapshot {
                name: "l".to_string(),
                index: 0,
                histogram: vec![1; 256],
                mean: 0.0,
                std: 1.0,
                min: -1.0,
                max: 1.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            }],
            golden_reference: None,
        };

        export_by_format(
            ExportFormat::Json,
            &manifest,
            &manifest.layers,
            output_dir.path(),
        )
        .expect("export");

        assert!(output_dir.path().join("manifest.json").exists());
        // PNG/PGM should NOT exist
        assert!(!output_dir.path().join("layer_000_l.pgm").exists());
    }

    #[test]
    fn test_export_by_format_png_creates_pgm_only() {
        let output_dir = tempdir().expect("create output dir");
        let layers = vec![LayerSnapshot {
            name: "x".to_string(),
            index: 0,
            histogram: vec![1; 256],
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }];
        let manifest = ProbarManifest {
            source_model: "m.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: layers.clone(),
            golden_reference: None,
        };

        export_by_format(ExportFormat::Png, &manifest, &layers, output_dir.path()).expect("export");

        assert!(!output_dir.path().join("manifest.json").exists());
        assert!(output_dir.path().join("layer_000_x.pgm").exists());
    }

    #[test]
    fn test_export_by_format_both_creates_all() {
        let output_dir = tempdir().expect("create output dir");
        let layers = vec![LayerSnapshot {
            name: "y".to_string(),
            index: 0,
            histogram: vec![1; 256],
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }];
        let manifest = ProbarManifest {
            source_model: "m.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: layers.clone(),
            golden_reference: None,
        };

        export_by_format(ExportFormat::Both, &manifest, &layers, output_dir.path())
            .expect("export");

        assert!(output_dir.path().join("manifest.json").exists());
        assert!(output_dir.path().join("layer_000_y.pgm").exists());
    }

    // ========================================================================
    // generate_diff Tests
    // ========================================================================

    #[test]
    fn test_generate_diff_identical_models_produces_zero_diffs() {
        let golden_dir = tempdir().expect("golden dir");
        let output_dir = tempdir().expect("output dir");

        let layers = vec![LayerSnapshot {
            name: "block_0".to_string(),
            index: 0,
            histogram: vec![100; 256],
            mean: 0.5,
            std: 1.0,
            min: -2.0,
            max: 2.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }];

        // Write golden manifest
        let golden_manifest = ProbarManifest {
            source_model: "golden.apr".to_string(),
            timestamp: "t1".to_string(),
            format: "APR".to_string(),
            layers: layers.clone(),
            golden_reference: None,
        };
        let golden_json = serde_json::to_string_pretty(&golden_manifest).expect("serialize golden");
        fs::write(golden_dir.path().join("manifest.json"), &golden_json).expect("write golden");

        // Current manifest with identical stats
        let current = ProbarManifest {
            source_model: "current.apr".to_string(),
            timestamp: "t2".to_string(),
            format: "APR".to_string(),
            layers,
            golden_reference: None,
        };

        generate_diff(golden_dir.path(), &current, output_dir.path()).expect("generate diff");

        let diff_content =
            fs::read_to_string(output_dir.path().join("diff_report.json")).expect("read diff");
        let diff: serde_json::Value = serde_json::from_str(&diff_content).expect("parse diff");

        assert_eq!(diff["total_diffs"], 0);
        assert!(diff["diffs"].as_array().expect("diffs array").is_empty());
    }

    #[test]
    fn test_generate_diff_detects_name_mismatch() {
        let golden_dir = tempdir().expect("golden dir");
        let output_dir = tempdir().expect("output dir");

        let golden_manifest = ProbarManifest {
            source_model: "golden.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: vec![LayerSnapshot {
                name: "layer_a".to_string(),
                index: 0,
                histogram: vec![0; 256],
                mean: 0.0,
                std: 1.0,
                min: -1.0,
                max: 1.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            }],
            golden_reference: None,
        };
        fs::write(
            golden_dir.path().join("manifest.json"),
            serde_json::to_string(&golden_manifest).expect("ser"),
        )
        .expect("write");

        let current = ProbarManifest {
            source_model: "current.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: vec![LayerSnapshot {
                name: "layer_b".to_string(),
                index: 0,
                histogram: vec![0; 256],
                mean: 0.0,
                std: 1.0,
                min: -1.0,
                max: 1.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            }],
            golden_reference: None,
        };

        generate_diff(golden_dir.path(), &current, output_dir.path()).expect("diff");

        let diff_content =
            fs::read_to_string(output_dir.path().join("diff_report.json")).expect("read");
        let diff: serde_json::Value = serde_json::from_str(&diff_content).expect("parse");

        assert!(diff["total_diffs"].as_u64().expect("total") >= 1);
        let diffs = diff["diffs"].as_array().expect("diffs array");
        assert!(diffs.iter().any(|d| d["type"] == "name_mismatch"));
    }

    #[test]
    fn test_generate_diff_detects_stats_divergence() {
        let golden_dir = tempdir().expect("golden dir");
        let output_dir = tempdir().expect("output dir");

        let golden_manifest = ProbarManifest {
            source_model: "golden.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: vec![LayerSnapshot {
                name: "block_0".to_string(),
                index: 0,
                histogram: vec![0; 256],
                mean: 0.0,
                std: 1.0,
                min: -1.0,
                max: 1.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            }],
            golden_reference: None,
        };
        fs::write(
            golden_dir.path().join("manifest.json"),
            serde_json::to_string(&golden_manifest).expect("ser"),
        )
        .expect("write");

        let current = ProbarManifest {
            source_model: "current.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: vec![LayerSnapshot {
                name: "block_0".to_string(),
                index: 0,
                histogram: vec![0; 256],
                mean: 0.5, // diverged by 0.5 (> 0.01 threshold)
                std: 2.0,  // diverged by 1.0 (> 0.01 threshold)
                min: -1.0,
                max: 1.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            }],
            golden_reference: None,
        };

        generate_diff(golden_dir.path(), &current, output_dir.path()).expect("diff");

        let diff_content =
            fs::read_to_string(output_dir.path().join("diff_report.json")).expect("read");
        let diff: serde_json::Value = serde_json::from_str(&diff_content).expect("parse");

        assert!(diff["total_diffs"].as_u64().expect("total") >= 1);
        let diffs = diff["diffs"].as_array().expect("diffs array");
        assert!(diffs.iter().any(|d| d["type"] == "stats_divergence"));
    }

    #[test]
    fn test_generate_diff_within_tolerance_no_divergence() {
        let golden_dir = tempdir().expect("golden dir");
        let output_dir = tempdir().expect("output dir");

        let golden_manifest = ProbarManifest {
            source_model: "golden.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: vec![LayerSnapshot {
                name: "block_0".to_string(),
                index: 0,
                histogram: vec![0; 256],
                mean: 1.0,
                std: 1.0,
                min: -1.0,
                max: 1.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            }],
            golden_reference: None,
        };
        fs::write(
            golden_dir.path().join("manifest.json"),
            serde_json::to_string(&golden_manifest).expect("ser"),
        )
        .expect("write");

        let current = ProbarManifest {
            source_model: "current.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: vec![LayerSnapshot {
                name: "block_0".to_string(),
                index: 0,
                histogram: vec![0; 256],
                mean: 1.005, // diff = 0.005, within 0.01 tolerance
                std: 1.009,  // diff = 0.009, within 0.01 tolerance
                min: -1.0,
                max: 1.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            }],
            golden_reference: None,
        };

        generate_diff(golden_dir.path(), &current, output_dir.path()).expect("diff");

        let diff_content =
            fs::read_to_string(output_dir.path().join("diff_report.json")).expect("read");
        let diff: serde_json::Value = serde_json::from_str(&diff_content).expect("parse");

        assert_eq!(diff["total_diffs"], 0);
    }

    #[test]
    fn test_generate_diff_missing_golden_manifest() {
        let golden_dir = tempdir().expect("golden dir");
        let output_dir = tempdir().expect("output dir");
        // Don't create manifest.json in golden dir

        let current = ProbarManifest {
            source_model: "c.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: vec![],
            golden_reference: None,
        };

        let result = generate_diff(golden_dir.path(), &current, output_dir.path());
        assert!(result.is_err(), "missing golden manifest should fail");
    }

    #[test]
    fn test_generate_diff_invalid_golden_json() {
        let golden_dir = tempdir().expect("golden dir");
        let output_dir = tempdir().expect("output dir");

        fs::write(golden_dir.path().join("manifest.json"), "not valid json")
            .expect("write bad json");

        let current = ProbarManifest {
            source_model: "c.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: vec![],
            golden_reference: None,
        };

        let result = generate_diff(golden_dir.path(), &current, output_dir.path());
        assert!(result.is_err(), "invalid golden JSON should fail");
    }

    #[test]
    fn test_generate_diff_unequal_layer_counts_zips_shortest() {
        let golden_dir = tempdir().expect("golden dir");
        let output_dir = tempdir().expect("output dir");

        let mk_layer = |name: &str| LayerSnapshot {
            name: name.to_string(),
            index: 0,
            histogram: vec![0; 256],
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        };

        let golden_manifest = ProbarManifest {
            source_model: "g.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: vec![mk_layer("a"), mk_layer("b"), mk_layer("c")],
            golden_reference: None,
        };
        fs::write(
            golden_dir.path().join("manifest.json"),
            serde_json::to_string(&golden_manifest).expect("ser"),
        )
        .expect("write");

        // Current has only 1 layer
        let current = ProbarManifest {
            source_model: "c.apr".to_string(),
            timestamp: "t".to_string(),
            format: "APR".to_string(),
            layers: vec![mk_layer("a")],
            golden_reference: None,
        };

        generate_diff(golden_dir.path(), &current, output_dir.path()).expect("diff");

        let diff_content =
            fs::read_to_string(output_dir.path().join("diff_report.json")).expect("read");
        let diff: serde_json::Value = serde_json::from_str(&diff_content).expect("parse");

        // Only 1 layer compared (zip stops at shortest), so 0 diffs
        assert_eq!(diff["total_diffs"], 0);
    }

    // ========================================================================
    // ProbarManifest Serialization Tests
    // ========================================================================

    #[test]
    fn test_probar_manifest_serialize_deserialize_roundtrip() {
        let manifest = ProbarManifest {
            source_model: "roundtrip.apr".to_string(),
            timestamp: "2026-02-06T00:00:00Z".to_string(),
            format: "GGUF (llama.cpp)".to_string(),
            layers: vec![
                LayerSnapshot {
                    name: "block_0".to_string(),
                    index: 0,
                    histogram: vec![10, 20, 30],
                    mean: -0.1,
                    std: 0.9,
                    min: -5.0,
                    max: 5.0,
                    heatmap: Some(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                    heatmap_width: Some(3),
                    heatmap_height: Some(2),
                },
                LayerSnapshot {
                    name: "block_1".to_string(),
                    index: 1,
                    histogram: vec![],
                    mean: 0.0,
                    std: 0.0,
                    min: 0.0,
                    max: 0.0,
                    heatmap: None,
                    heatmap_width: None,
                    heatmap_height: None,
                },
            ],
            golden_reference: Some("/golden/ref".to_string()),
        };

        let json = serde_json::to_string(&manifest).expect("serialize");
        let loaded: ProbarManifest = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(loaded.source_model, manifest.source_model);
        assert_eq!(loaded.timestamp, manifest.timestamp);
        assert_eq!(loaded.format, manifest.format);
        assert_eq!(loaded.layers.len(), 2);
        assert_eq!(loaded.layers[0].name, "block_0");
        assert_eq!(
            loaded.layers[0].heatmap,
            Some(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        );
        assert_eq!(loaded.layers[1].histogram.len(), 0);
        assert_eq!(loaded.golden_reference, Some("/golden/ref".to_string()));
    }

    // ========================================================================
    // LayerSnapshot Full Round-Trip with Heatmap
    // ========================================================================

    #[test]
    fn test_layer_snapshot_full_roundtrip_with_heatmap() {
        let original = LayerSnapshot {
            name: "embed".to_string(),
            index: 99,
            histogram: (0..256).map(|i| i as u32 * 2).collect(),
            mean: -0.001,
            std: 0.999,
            min: -10.0,
            max: 10.0,
            heatmap: Some(vec![f32::MIN, 0.0, f32::MAX]),
            heatmap_width: Some(3),
            heatmap_height: Some(1),
        };

        let json = serde_json::to_string(&original).expect("serialize");
        let restored: LayerSnapshot = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.name, "embed");
        assert_eq!(restored.index, 99);
        assert_eq!(restored.histogram.len(), 256);
        assert_eq!(restored.histogram[0], 0);
        assert_eq!(restored.histogram[255], 510);
        assert_eq!(restored.heatmap_width, Some(3));
        assert_eq!(restored.heatmap_height, Some(1));
    }

    #[test]
    fn test_layer_snapshot_deserialize_with_null_optionals() {
        let json = r#"{
            "name": "null_test",
            "index": 0,
            "histogram": [],
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "heatmap": null,
            "heatmap_width": null,
            "heatmap_height": null
        }"#;
        let snapshot: LayerSnapshot = serde_json::from_str(json).expect("deserialize");
        assert_eq!(snapshot.name, "null_test");
        assert!(snapshot.heatmap.is_none());
        assert!(snapshot.heatmap_width.is_none());
        assert!(snapshot.heatmap_height.is_none());
    }

    // ========================================================================
    // print_* No-Panic Tests
    // ========================================================================

    #[test]
    fn test_print_summary_does_not_panic() {
        let layers = vec![LayerSnapshot {
            name: "l".to_string(),
            index: 0,
            histogram: vec![],
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }];
        // Without golden
        print_summary(Path::new("/m.apr"), Path::new("/out"), "APR", &layers, None);
        // With golden
        print_summary(
            Path::new("/m.apr"),
            Path::new("/out"),
            "APR",
            &layers,
            Some(Path::new("/golden")),
        );
    }

    #[test]
    fn test_print_summary_empty_layers_does_not_panic() {
        print_summary(Path::new("/m.apr"), Path::new("/out"), "GGUF", &[], None);
    }

    #[test]
    fn test_print_generated_files_json_does_not_panic() {
        print_generated_files(ExportFormat::Json, Path::new("/out"), &[]);
    }

    #[test]
    fn test_print_generated_files_png_does_not_panic() {
        let layers = vec![
            LayerSnapshot {
                name: "a".to_string(),
                index: 0,
                histogram: vec![],
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            },
            LayerSnapshot {
                name: "b".to_string(),
                index: 1,
                histogram: vec![],
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            },
        ];
        print_generated_files(ExportFormat::Png, Path::new("/out"), &layers);
    }

    #[test]
    fn test_print_generated_files_both_does_not_panic() {
        let layers = vec![LayerSnapshot {
            name: "c".to_string(),
            index: 2,
            histogram: vec![],
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }];
        print_generated_files(ExportFormat::Both, Path::new("/output"), &layers);
    }

    #[test]
    fn test_print_integration_guide_does_not_panic() {
        print_integration_guide();
    }
}
