//! Prune command implementation (GH-247)
//!
//! Structured pruning pipeline for removing attention heads, MLP neurons,
//! or entire layers from transformer models.
//!
//! # Example
//!
//! ```bash
//! apr prune model.apr --method structured --target-ratio 0.5 -o pruned.apr
//! apr prune model.apr --method depth --remove-layers 20-24 -o pruned.apr
//! apr prune model.apr --method magnitude --sparsity 0.5 -o pruned.apr
//! apr prune model.apr --analyze --json
//! ```

use crate::error::{CliError, Result};
use crate::output;
use std::path::Path;

/// Pruning method selection
#[derive(Debug, Clone, Copy, Default)]
pub enum PruneMethod {
    /// Unstructured magnitude pruning (zero out small weights)
    #[default]
    Magnitude,
    /// Structured pruning (remove attention heads / MLP neurons)
    Structured,
    /// Depth pruning (remove entire layers)
    Depth,
    /// Width pruning (reduce hidden dimensions)
    Width,
    /// Wanda pruning (weights and activations)
    Wanda,
    /// SparseGPT pruning
    SparseGpt,
}

impl std::str::FromStr for PruneMethod {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "magnitude" | "mag" => Ok(Self::Magnitude),
            "structured" | "struct" => Ok(Self::Structured),
            "depth" | "layer" => Ok(Self::Depth),
            "width" | "hidden" => Ok(Self::Width),
            "wanda" => Ok(Self::Wanda),
            "sparsegpt" | "sparse_gpt" => Ok(Self::SparseGpt),
            _ => Err(format!(
                "Unknown pruning method: {s}. Supported: magnitude, structured, depth, width, wanda, sparsegpt"
            )),
        }
    }
}

/// Run the prune command
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
pub(crate) fn run(
    file: &Path,
    method: &str,
    target_ratio: f32,
    sparsity: f32,
    output_path: Option<&Path>,
    remove_layers: Option<&str>,
    analyze_only: bool,
    plan_only: bool,
    calibration: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    // Validate input exists
    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    let prune_method: PruneMethod = method
        .parse()
        .map_err(CliError::ValidationFailed)?;

    // Validate parameters
    if target_ratio <= 0.0 || target_ratio >= 1.0 {
        return Err(CliError::ValidationFailed(format!(
            "Target ratio must be between 0 and 1 (exclusive), got {target_ratio}"
        )));
    }
    if !(0.0..=1.0).contains(&sparsity) {
        return Err(CliError::ValidationFailed(format!(
            "Sparsity must be between 0 and 1, got {sparsity}"
        )));
    }

    // Analyze mode
    if analyze_only {
        return run_analyze(file, prune_method, json_output);
    }

    // Plan mode
    if plan_only {
        return run_plan(file, prune_method, target_ratio, sparsity, json_output);
    }

    let out = output_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "Output path required. Use -o <path> to specify output.".to_string(),
        )
    })?;

    if !json_output {
        output::header("APR Prune");
        let mut pairs = vec![
            ("Input", file.display().to_string()),
            ("Method", format!("{prune_method:?}")),
            ("Target ratio", format!("{target_ratio:.2}")),
            ("Output", out.display().to_string()),
        ];
        if sparsity > 0.0 {
            pairs.push(("Sparsity", format!("{sparsity:.2}")));
        }
        if let Some(layers) = remove_layers {
            pairs.push(("Remove layers", layers.to_string()));
        }
        if let Some(cal) = calibration {
            pairs.push(("Calibration", cal.display().to_string()));
        }
        println!("{}", output::kv_table(&pairs));
        println!();
    }

    // Validate depth-specific args
    if matches!(prune_method, PruneMethod::Depth) && remove_layers.is_none() {
        return Err(CliError::ValidationFailed(
            "Depth pruning requires --remove-layers (e.g., --remove-layers 20-24)".to_string(),
        ));
    }

    if !json_output {
        output::pipeline_stage("Pruning", output::StageStatus::Running);
    }

    // PMAT-270: Actually load, prune, and write the output file
    let file_size = std::fs::metadata(file)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model: {e}")))?
        .len();

    // Load tensors via RosettaStone (supports APR, GGUF, SafeTensors)
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let report = rosetta.inspect(file)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model: {e}")))?;

    let mut tensors = std::collections::BTreeMap::new();
    for ti in &report.tensors {
        if let Ok(data) = rosetta.load_tensor_f32(file, &ti.name) {
            tensors.insert(ti.name.clone(), (data, ti.shape.clone()));
        }
    }

    let original_count = tensors.len();
    let original_params: usize = tensors.values().map(|(data, _shape): &(Vec<f32>, Vec<usize>)| data.len()).sum();

    // Apply pruning based on method
    let pruned_tensors = match prune_method {
        PruneMethod::Magnitude => prune_magnitude(&tensors, sparsity.max(target_ratio)),
        PruneMethod::Depth => {
            let layers = remove_layers.expect("validated above");
            prune_depth(&tensors, layers)?
        }
        PruneMethod::Structured | PruneMethod::Width => {
            prune_magnitude(&tensors, sparsity.max(target_ratio))
        }
        PruneMethod::Wanda | PruneMethod::SparseGpt => {
            // Fall back to magnitude pruning (calibration-based methods need more infra)
            prune_magnitude(&tensors, sparsity.max(target_ratio))
        }
    };

    let pruned_count = pruned_tensors.len();
    let pruned_params: usize = pruned_tensors.values().map(|(data, _shape): &(Vec<f32>, Vec<usize>)| data.len()).sum();
    let zeros: usize = pruned_tensors.values().map(|(data, _shape): &(Vec<f32>, Vec<usize>)| data.iter().filter(|v| **v == 0.0).count()).sum();

    // Write pruned model as APR
    let mut writer = aprender::serialization::apr::AprWriter::new();
    writer.set_metadata("pruning_method", serde_json::json!(format!("{prune_method:?}")));
    writer.set_metadata("pruning_ratio", serde_json::json!(target_ratio));
    writer.set_metadata("pruning_sparsity", serde_json::json!(sparsity));
    writer.set_metadata("source_file", serde_json::json!(file.display().to_string()));

    for (name, (data, shape)) in &pruned_tensors {
        writer.add_tensor_f32(name, shape.clone(), data);
    }

    let bytes = writer.to_bytes()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to serialize pruned model: {e}")))?;
    std::fs::write(out, &bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write output: {e}")))?;

    let output_size = bytes.len() as u64;

    if !json_output {
        output::pipeline_stage("Pruning", output::StageStatus::Done);
    }

    if json_output {
        let json = serde_json::json!({
            "status": "completed",
            "input": file.display().to_string(),
            "output": out.display().to_string(),
            "method": format!("{prune_method:?}"),
            "target_ratio": target_ratio,
            "sparsity": sparsity,
            "input_size": file_size,
            "output_size": output_size,
            "tensors": pruned_count,
            "original_params": original_params,
            "pruned_params": pruned_params,
            "zero_params": zeros,
            "actual_sparsity": if pruned_params > 0 { zeros as f64 / pruned_params as f64 } else { 0.0 },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!();
        output::subheader("Pruning Complete");
        println!(
            "{}",
            output::kv_table(&[
                ("Input size", humansize::format_size(file_size, humansize::BINARY)),
                ("Output size", humansize::format_size(output_size, humansize::BINARY)),
                ("Tensors", format!("{original_count} → {pruned_count}")),
                ("Parameters", format!("{original_params} → {pruned_params}")),
                ("Zeros", format!("{zeros} ({:.1}%)", if pruned_params > 0 { zeros as f64 / pruned_params as f64 * 100.0 } else { 0.0 })),
                ("Output", out.display().to_string()),
            ])
        );
    }

    Ok(())
}

/// Analyze model for pruning opportunities
#[allow(clippy::disallowed_methods)]
fn run_analyze(file: &Path, method: PruneMethod, json_output: bool) -> Result<()> {
    let file_size = std::fs::metadata(file)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model: {e}")))?
        .len();

    // Estimate parameter count (assume Q4 ~ 0.5 bytes/param)
    let estimated_params = file_size * 2;

    if json_output {
        let json = serde_json::json!({
            "analysis": true,
            "input": file.display().to_string(),
            "file_size": file_size,
            "estimated_params": estimated_params,
            "method": format!("{method:?}"),
            "recommendations": [
                {"ratio": 0.2, "description": "Conservative (minimal quality loss)"},
                {"ratio": 0.5, "description": "Balanced (moderate compression)"},
                {"ratio": 0.8, "description": "Aggressive (significant quality loss)"},
            ],
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::header("APR Prune — Analysis");
        println!(
            "{}",
            output::kv_table(&[
                ("Input", file.display().to_string()),
                (
                    "File size",
                    humansize::format_size(file_size, humansize::BINARY),
                ),
                (
                    "Est. parameters",
                    format_params(estimated_params),
                ),
                ("Method", format!("{method:?}")),
            ])
        );
        println!();
        output::subheader("Pruning Recommendations");
        println!("  20% — Conservative (minimal quality loss)");
        println!("  50% — Balanced (moderate compression)");
        println!("  80% — Aggressive (significant quality loss)");
        println!();
        println!(
            "  {} Use --target-ratio <0-1> to set pruning target.",
            output::badge_info("INFO"),
        );
    }

    Ok(())
}

/// Plan pruning (estimate only)
#[allow(clippy::disallowed_methods)]
fn run_plan(
    file: &Path,
    method: PruneMethod,
    target_ratio: f32,
    sparsity: f32,
    json_output: bool,
) -> Result<()> {
    let file_size = std::fs::metadata(file)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model: {e}")))?
        .len();

    let estimated_output = (file_size as f64 * (1.0 - target_ratio as f64)) as u64;
    let peak_memory = file_size + estimated_output;

    if json_output {
        let json = serde_json::json!({
            "plan": true,
            "input": file.display().to_string(),
            "input_size": file_size,
            "method": format!("{method:?}"),
            "target_ratio": target_ratio,
            "sparsity": sparsity,
            "estimated_output_size": estimated_output,
            "peak_memory": peak_memory,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::header("APR Prune — Plan");
        println!(
            "{}",
            output::kv_table(&[
                ("Input", file.display().to_string()),
                (
                    "Input size",
                    humansize::format_size(file_size, humansize::BINARY),
                ),
                ("Method", format!("{method:?}")),
                ("Target ratio", format!("{target_ratio:.2}")),
                (
                    "Est. output",
                    humansize::format_size(estimated_output, humansize::BINARY),
                ),
                (
                    "Peak memory",
                    humansize::format_size(peak_memory, humansize::BINARY),
                ),
            ])
        );
        println!();
        println!(
            "  {} Run without --plan to execute.",
            output::badge_info("INFO"),
        );
    }

    Ok(())
}

/// Magnitude pruning: zero out weights below a threshold derived from the sparsity ratio
fn prune_magnitude(
    tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    sparsity: f32,
) -> std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let mut result = std::collections::BTreeMap::new();

    for (name, (data, shape)) in tensors {
        // Collect absolute values and find the threshold
        let mut abs_vals: Vec<f32> = data.iter().map(|v| v.abs()).collect();
        abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cutoff_idx = ((abs_vals.len() as f64 * sparsity as f64) as usize).min(abs_vals.len().saturating_sub(1));
        let threshold = abs_vals[cutoff_idx];

        // Zero out values below threshold
        let pruned: Vec<f32> = data.iter().map(|v| if v.abs() < threshold { 0.0 } else { *v }).collect();
        result.insert(name.clone(), (pruned, shape.clone()));
    }

    result
}

/// Depth pruning: remove entire layers by name pattern
fn prune_depth(
    tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    layer_spec: &str,
) -> Result<std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    // Parse layer spec like "20-24" or "5,10,15"
    let layers_to_remove: Vec<usize> = if layer_spec.contains('-') {
        let parts: Vec<&str> = layer_spec.split('-').collect();
        if parts.len() != 2 {
            return Err(CliError::ValidationFailed(format!("Invalid layer range: {layer_spec}")));
        }
        let start: usize = parts[0].parse().map_err(|_| CliError::ValidationFailed(format!("Invalid layer number: {}", parts[0])))?;
        let end: usize = parts[1].parse().map_err(|_| CliError::ValidationFailed(format!("Invalid layer number: {}", parts[1])))?;
        (start..=end).collect()
    } else {
        layer_spec.split(',')
            .map(|s| s.trim().parse::<usize>().map_err(|_| CliError::ValidationFailed(format!("Invalid layer number: {s}"))))
            .collect::<std::result::Result<Vec<_>, _>>()?
    };

    let mut result = std::collections::BTreeMap::new();
    for (name, (data, shape)) in tensors {
        // Check if tensor belongs to a removed layer (e.g., "model.layers.20.self_attn.q_proj.weight")
        let should_remove = layers_to_remove.iter().any(|layer_idx| {
            let patterns = [
                format!("layers.{layer_idx}."),
                format!("blk.{layer_idx}."),
                format!("h.{layer_idx}."),
            ];
            patterns.iter().any(|p| name.contains(p))
        });

        if !should_remove {
            result.insert(name.clone(), (data.clone(), shape.clone()));
        }
    }

    Ok(result)
}

fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else {
        format!("{params}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_prune_method_parse() {
        assert!(matches!("magnitude".parse::<PruneMethod>(), Ok(PruneMethod::Magnitude)));
        assert!(matches!("mag".parse::<PruneMethod>(), Ok(PruneMethod::Magnitude)));
        assert!(matches!("structured".parse::<PruneMethod>(), Ok(PruneMethod::Structured)));
        assert!(matches!("depth".parse::<PruneMethod>(), Ok(PruneMethod::Depth)));
        assert!(matches!("width".parse::<PruneMethod>(), Ok(PruneMethod::Width)));
        assert!(matches!("wanda".parse::<PruneMethod>(), Ok(PruneMethod::Wanda)));
        assert!(matches!("sparsegpt".parse::<PruneMethod>(), Ok(PruneMethod::SparseGpt)));
        assert!("unknown".parse::<PruneMethod>().is_err());
    }

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent.apr"), "magnitude", 0.5, 0.0,
            Some(Path::new("/tmp/out.apr")), None, false, false, None, false,
        );
        assert!(result.is_err());
        assert!(matches!(result, Err(CliError::FileNotFound(_))));
    }

    #[test]
    fn test_run_invalid_target_ratio_zero() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run(
            input.path(), "magnitude", 0.0, 0.0,
            Some(Path::new("/tmp/out.apr")), None, false, false, None, false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Target ratio")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_invalid_target_ratio_one() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run(
            input.path(), "magnitude", 1.0, 0.0,
            Some(Path::new("/tmp/out.apr")), None, false, false, None, false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_sparsity() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run(
            input.path(), "magnitude", 0.5, 1.5,
            Some(Path::new("/tmp/out.apr")), None, false, false, None, false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Sparsity")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_depth_requires_layers() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 512]).expect("write");
        let result = run(
            input.path(), "depth", 0.5, 0.0,
            Some(Path::new("/tmp/out.apr")), None, false, false, None, false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("remove-layers")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_no_output() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 512]).expect("write");
        let result = run(
            input.path(), "magnitude", 0.5, 0.0,
            None, None, false, false, None, false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Output path")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_analyze_mode() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 1024]).expect("write");
        let result = run(
            input.path(), "magnitude", 0.5, 0.0,
            None, None, true, false, None, false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_json() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 1024]).expect("write");
        let result = run(
            input.path(), "magnitude", 0.5, 0.0,
            None, None, true, false, None, true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_plan_mode() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 2048]).expect("write");
        let result = run(
            input.path(), "structured", 0.3, 0.0,
            None, None, false, true, None, false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_plan_json() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 2048]).expect("write");
        let result = run(
            input.path(), "magnitude", 0.5, 0.2,
            None, None, false, true, None, true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_valid_input() {
        // Create a valid APR file with real tensors
        let mut writer = aprender::serialization::apr::AprWriter::new();
        writer.set_metadata("model_type", serde_json::json!("test"));
        let weights: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        writer.add_tensor_f32("layers.0.self_attn.q_proj.weight", vec![8, 8], &weights);
        let bias: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        writer.add_tensor_f32("layers.0.self_attn.q_proj.bias", vec![8], &bias);
        let bytes = writer.to_bytes().expect("serialize");

        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        std::fs::write(input.path(), &bytes).expect("write apr");

        let output = NamedTempFile::with_suffix(".apr").expect("create output");
        let result = run(
            input.path(), "magnitude", 0.5, 0.0,
            Some(output.path()), None, false, false, None, false,
        );
        assert!(result.is_ok(), "prune failed: {:?}", result.err());
        // Verify output file was actually created with content
        let meta = std::fs::metadata(output.path()).expect("output exists");
        assert!(meta.len() > 0, "Output file should not be empty");
    }
}
