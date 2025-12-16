//! Hex dump and tensor visualization (GH-122)
//!
//! Toyota Way: Genchi Genbutsu - Go and see the actual bytes.
//! Visualize tensor data at the byte level for debugging.
//!
//! Usage:
//!   apr hex model.apr --tensor "encoder.layers.0.self_attn.q_proj.weight"
//!   apr hex model.apr --tensor "decoder" --stats
//!   apr hex model.apr --list

use crate::error::CliError;
use aprender::serialization::apr::{AprReader, AprTensorDescriptor};
use colored::Colorize;
use std::path::Path;

/// Run the hex dump command
pub(crate) fn run(
    apr_path: &Path,
    tensor_filter: Option<&str>,
    limit: usize,
    show_stats: bool,
    list_only: bool,
    json_output: bool,
) -> Result<(), CliError> {
    if !apr_path.exists() {
        return Err(CliError::FileNotFound(apr_path.to_path_buf()));
    }

    let reader = AprReader::open(apr_path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to read APR: {e}")))?;

    // Filter tensors
    let filtered: Vec<&AprTensorDescriptor> = reader
        .tensors
        .iter()
        .filter(|t| tensor_filter.map_or(true, |f| t.name.contains(f)))
        .collect();

    if filtered.is_empty() {
        if !json_output {
            println!("{}", "No tensors match the filter pattern".yellow());
        }
        return Ok(());
    }

    if list_only {
        return list_tensors(&filtered, json_output);
    }

    if json_output {
        return output_json(&reader, &filtered, limit, show_stats);
    }

    // Process each matching tensor
    for tensor in &filtered {
        print_tensor_hex(&reader, tensor, limit, show_stats)?;
        println!();
    }

    Ok(())
}

/// List tensor names only
#[allow(clippy::unnecessary_wraps)] // Consistent with other command functions
#[allow(clippy::disallowed_methods)] // json! macro uses infallible unwrap internally
fn list_tensors(tensors: &[&AprTensorDescriptor], json_output: bool) -> Result<(), CliError> {
    if json_output {
        let names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
        let json = serde_json::json!({
            "tensors": names,
            "count": tensors.len()
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!("{}", "Tensors:".bold());
        for tensor in tensors {
            println!("  {}", tensor.name);
        }
        println!("\n{} tensors total", tensors.len().to_string().cyan());
    }
    Ok(())
}

/// Print tensor header information
fn print_tensor_header(tensor: &AprTensorDescriptor) {
    println!("{}", "═".repeat(70));
    println!("{}: {}", "Tensor".bold(), tensor.name.cyan());
    println!("{}", "═".repeat(70));

    let num_elements: usize = tensor.shape.iter().product();
    println!(
        "{}: {:?} = {} elements",
        "Shape".bold(),
        tensor.shape,
        num_elements.to_string().green()
    );
    println!("{}: {}", "Dtype".bold(), tensor.dtype);
    println!(
        "{}: 0x{:08X} ({} bytes)",
        "Offset".bold(),
        tensor.offset,
        tensor.offset
    );
    println!(
        "{}: {} bytes",
        "Size".bold(),
        tensor.size.to_string().yellow()
    );
}

/// Check for tensor anomalies and print warnings
fn print_tensor_anomalies(min: f32, max: f32, mean: f32, std: f32) {
    if min.is_nan() || max.is_nan() || mean.is_nan() {
        println!("  {} NaN values detected!", "⚠".red());
    }
    if min.is_infinite() || max.is_infinite() {
        println!("  {} Infinite values detected!", "⚠".red());
    }
    if std < 1e-10 {
        println!(
            "  {} Very low variance - possible collapsed weights!",
            "⚠".yellow()
        );
    }
}

/// Print statistics for tensor data
fn print_tensor_stats(data: &[f32]) {
    println!();
    println!("{}", "Statistics:".bold());
    let (min, max, mean, std) = compute_stats(data);
    println!("  min={min:.6}  max={max:.6}  mean={mean:.6}  std={std:.6}");
    print_tensor_anomalies(min, max, mean, std);
}

/// Print a hex dump row for a chunk of float values
fn print_hex_row(chunk: &[&f32], row_offset: usize) {
    print!("{row_offset:08X}: ");

    for &val in chunk {
        let bytes = val.to_le_bytes();
        for b in &bytes {
            print!("{b:02X} ");
        }
    }

    let padding = (4 - chunk.len()) * 12;
    print!("{:width$}", "", width = padding);

    print!(" | ");
    for &val in chunk {
        print!("{val:>10.4} ");
    }
    println!();
}

/// Print hex dump of tensor data
fn print_hex_dump(data: &[f32], limit: usize) {
    println!();
    println!(
        "{} (first {} bytes):",
        "Hex dump".bold(),
        (limit * 4).min(data.len() * 4)
    );

    let bytes_to_show = limit.min(data.len());
    for (i, chunk) in data
        .iter()
        .take(bytes_to_show)
        .collect::<Vec<_>>()
        .chunks(4)
        .enumerate()
    {
        print_hex_row(chunk, i * 16);
    }

    if data.len() > bytes_to_show {
        println!(
            "... {} more elements",
            (data.len() - bytes_to_show).to_string().dimmed()
        );
    }
}

/// Print hex dump for a single tensor
fn print_tensor_hex(
    reader: &AprReader,
    tensor: &AprTensorDescriptor,
    limit: usize,
    show_stats: bool,
) -> Result<(), CliError> {
    print_tensor_header(tensor);

    let data = reader
        .read_tensor_f32(&tensor.name)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to read tensor: {e}")))?;

    if show_stats {
        print_tensor_stats(&data);
    }

    print_hex_dump(&data, limit);

    Ok(())
}

/// Compute basic statistics
fn compute_stats(data: &[f32]) -> (f32, f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;

    for &x in data {
        min = min.min(x);
        max = max.max(x);
        sum += f64::from(x);
    }

    let mean = (sum / data.len() as f64) as f32;

    let variance: f32 = (data
        .iter()
        .map(|&x| {
            let diff = f64::from(x) - f64::from(mean);
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64) as f32;

    let std = variance.sqrt();

    (min, max, mean, std)
}

/// Output as JSON
#[allow(clippy::unnecessary_wraps)] // Consistent with other command functions
#[allow(clippy::disallowed_methods)] // unwrap_or_default is safe for empty vec
fn output_json(
    reader: &AprReader,
    tensors: &[&AprTensorDescriptor],
    limit: usize,
    show_stats: bool,
) -> Result<(), CliError> {
    use serde::Serialize;

    #[derive(Serialize)]
    struct TensorDump {
        name: String,
        shape: Vec<usize>,
        dtype: String,
        offset: usize,
        size_bytes: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        stats: Option<TensorStats>,
        sample_values: Vec<f32>,
    }

    #[derive(Serialize)]
    struct TensorStats {
        min: f32,
        max: f32,
        mean: f32,
        std: f32,
    }

    let mut results = Vec::new();

    for tensor in tensors {
        let data = reader.read_tensor_f32(&tensor.name).ok();
        let stats = if show_stats {
            data.as_ref().map(|d| {
                let (min, max, mean, std) = compute_stats(d);
                TensorStats {
                    min,
                    max,
                    mean,
                    std,
                }
            })
        } else {
            None
        };

        let sample_values = data
            .as_ref()
            .map(|d| d.iter().take(limit).copied().collect())
            .unwrap_or_default();

        results.push(TensorDump {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            dtype: tensor.dtype.clone(),
            offset: tensor.offset,
            size_bytes: tensor.size,
            stats,
            sample_values,
        });
    }

    if let Ok(json) = serde_json::to_string_pretty(&results) {
        println!("{json}");
    }

    Ok(())
}
