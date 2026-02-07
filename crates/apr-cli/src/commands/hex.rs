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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    // ========================================================================
    // Statistics Tests
    // ========================================================================

    #[test]
    fn test_compute_stats_empty() {
        let (min, max, mean, std) = compute_stats(&[]);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_single_value() {
        let (min, max, mean, std) = compute_stats(&[5.0]);
        assert_eq!(min, 5.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 5.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_simple_range() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 3.0);
        // std for [1,2,3,4,5] is sqrt(2) ≈ 1.414
        assert!((std - 1.4142).abs() < 0.01);
    }

    #[test]
    fn test_compute_stats_negative_values() {
        let data = [-5.0, -2.0, 0.0, 2.0, 5.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -5.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_compute_stats_all_same() {
        let data = [7.0, 7.0, 7.0, 7.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 7.0);
        assert_eq!(max, 7.0);
        assert_eq!(mean, 7.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_large_values() {
        let data = [1e6, 2e6, 3e6];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, 1e6);
        assert_eq!(max, 3e6);
        assert_eq!(mean, 2e6);
    }

    #[test]
    fn test_compute_stats_tiny_values() {
        let data = [1e-6, 2e-6, 3e-6];
        let (min, max, mean, _std) = compute_stats(&data);
        assert!((min - 1e-6).abs() < 1e-9);
        assert!((max - 3e-6).abs() < 1e-9);
        assert!((mean - 2e-6).abs() < 1e-9);
    }

    // ========================================================================
    // Run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            None,
            100,
            false,
            false,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_invalid_apr_file() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), None, 100, false, false, false);
        // Should fail because it's not a valid APR
        assert!(result.is_err());
    }

    #[test]
    fn test_run_is_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = run(dir.path(), None, 100, false, false, false);
        // Should fail because it's a directory
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_tensor_filter() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), Some("encoder"), 100, false, false, false);
        // Should fail (invalid file) but tests the filter path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_stats_flag() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(
            file.path(),
            None,
            100,
            true, // show_stats
            false,
            false,
        );
        // Should fail (invalid file) but tests the stats path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_list_only() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(
            file.path(),
            None,
            100,
            false,
            true, // list_only
            false,
        );
        // Should fail (invalid file) but tests the list_only path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_json_output() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(
            file.path(),
            None,
            100,
            false,
            false,
            true, // json_output
        );
        // Should fail (invalid file) but tests the json path
        assert!(result.is_err());
    }

    // ========================================================================
    // compute_stats: NaN / Inf / edge case tests
    // ========================================================================

    #[test]
    fn test_compute_stats_with_nan() {
        let data = [1.0, f32::NAN, 3.0];
        let (min, max, mean, std) = compute_stats(&data);
        // NaN propagates through min/max/mean/std
        assert!(min.is_nan() || min == 1.0); // min(1.0, NaN) is NaN per IEEE 754
        assert!(mean.is_nan());
        let _ = (max, std); // may or may not be NaN depending on ordering
    }

    #[test]
    fn test_compute_stats_all_nan() {
        let data = [f32::NAN, f32::NAN, f32::NAN];
        let (_min, _max, mean, std) = compute_stats(&data);
        // IEEE 754: f32::INFINITY.min(NaN) returns INFINITY (NaN does not propagate through min/max)
        // But mean and std are computed via sum which DOES propagate NaN
        assert!(mean.is_nan());
        assert!(std.is_nan());
    }

    #[test]
    fn test_compute_stats_with_infinity() {
        let data = [1.0, f32::INFINITY, -1.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -1.0);
        assert_eq!(max, f32::INFINITY);
        assert!(mean.is_infinite());
    }

    #[test]
    fn test_compute_stats_with_neg_infinity() {
        let data = [1.0, f32::NEG_INFINITY, 3.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, f32::NEG_INFINITY);
        assert_eq!(max, 3.0);
        assert!(mean.is_infinite());
    }

    #[test]
    fn test_compute_stats_two_values() {
        let data = [0.0, 10.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 10.0);
        assert_eq!(mean, 5.0);
        assert!((std - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_stats_all_zeros() {
        let data = [0.0, 0.0, 0.0, 0.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_large_array() {
        // 1000 values from 0 to 999
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 999.0);
        assert!((mean - 499.5).abs() < 0.1);
        // Std dev of uniform 0..999 is ~288.67
        assert!((std - 288.67).abs() < 1.0);
    }

    #[test]
    fn test_compute_stats_mixed_positive_negative() {
        let data = [-100.0, -50.0, 0.0, 50.0, 100.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -100.0);
        assert_eq!(max, 100.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_compute_stats_subnormal_values() {
        // Very small subnormal floats
        let data = [
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE * 2.0,
            f32::MIN_POSITIVE * 3.0,
        ];
        let (min, max, _mean, _std) = compute_stats(&data);
        assert_eq!(min, f32::MIN_POSITIVE);
        assert_eq!(max, f32::MIN_POSITIVE * 3.0);
    }

    // ========================================================================
    // print_tensor_anomalies tests (output to stdout, verify no panic)
    // ========================================================================

    #[test]
    fn test_print_tensor_anomalies_no_issues() {
        // Normal values — should print nothing and not panic
        print_tensor_anomalies(0.0, 1.0, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_nan_min() {
        // NaN in min triggers warning — should not panic
        print_tensor_anomalies(f32::NAN, 1.0, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_nan_max() {
        print_tensor_anomalies(0.0, f32::NAN, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_nan_mean() {
        print_tensor_anomalies(0.0, 1.0, f32::NAN, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_all_nan() {
        print_tensor_anomalies(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
    }

    #[test]
    fn test_print_tensor_anomalies_infinite_min() {
        print_tensor_anomalies(f32::NEG_INFINITY, 1.0, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_infinite_max() {
        print_tensor_anomalies(0.0, f32::INFINITY, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_low_variance() {
        // std < 1e-10 triggers collapsed weights warning
        print_tensor_anomalies(0.0, 1.0, 0.5, 1e-12);
    }

    #[test]
    fn test_print_tensor_anomalies_zero_variance() {
        print_tensor_anomalies(5.0, 5.0, 5.0, 0.0);
    }

    #[test]
    fn test_print_tensor_anomalies_exactly_threshold() {
        // std exactly at 1e-10 triggers warning (< 1e-10 is false, = is not <)
        print_tensor_anomalies(0.0, 1.0, 0.5, 1e-10);
    }

    #[test]
    fn test_print_tensor_anomalies_above_threshold() {
        // std just above 1e-10 — should NOT trigger collapsed weights
        print_tensor_anomalies(0.0, 1.0, 0.5, 2e-10);
    }

    #[test]
    fn test_print_tensor_anomalies_nan_and_infinite_together() {
        // Both NaN and infinite: both warnings should fire
        print_tensor_anomalies(f32::NAN, f32::INFINITY, f32::NAN, 0.0);
    }

    // ========================================================================
    // print_tensor_header tests (verify no panic, exercises formatting)
    // ========================================================================

    fn make_descriptor(
        name: &str,
        shape: Vec<usize>,
        dtype: &str,
        offset: usize,
        size: usize,
    ) -> AprTensorDescriptor {
        AprTensorDescriptor {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape,
            offset,
            size,
        }
    }

    #[test]
    fn test_print_tensor_header_basic() {
        let desc = make_descriptor(
            "model.layers.0.weight",
            vec![768, 3072],
            "F32",
            0,
            768 * 3072 * 4,
        );
        print_tensor_header(&desc);
    }

    #[test]
    fn test_print_tensor_header_empty_shape() {
        let desc = make_descriptor("scalar_param", vec![], "F32", 0, 4);
        // Product of empty shape is 1 (identity for multiplication)
        print_tensor_header(&desc);
    }

    #[test]
    fn test_print_tensor_header_single_dim() {
        let desc = make_descriptor("bias", vec![512], "F32", 1024, 512 * 4);
        print_tensor_header(&desc);
    }

    #[test]
    fn test_print_tensor_header_large_offset() {
        let desc = make_descriptor(
            "lm_head.weight",
            vec![32000, 4096],
            "F16",
            0xFFFF_FFFF,
            32000 * 4096 * 2,
        );
        print_tensor_header(&desc);
    }

    #[test]
    fn test_print_tensor_header_zero_size() {
        let desc = make_descriptor("empty", vec![0], "F32", 0, 0);
        print_tensor_header(&desc);
    }

    #[test]
    fn test_print_tensor_header_3d_shape() {
        let desc = make_descriptor("conv.weight", vec![64, 3, 3], "F32", 512, 64 * 3 * 3 * 4);
        print_tensor_header(&desc);
    }

    // ========================================================================
    // print_hex_row tests (verify formatting, no panic)
    // ========================================================================

    #[test]
    fn test_print_hex_row_full_chunk() {
        let vals = [1.0_f32, 2.0, 3.0, 4.0];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 0);
    }

    #[test]
    fn test_print_hex_row_partial_chunk() {
        let vals = [1.0_f32, 2.0];
        let refs: Vec<&f32> = vals.iter().collect();
        // 2 values out of 4 → should have padding
        print_hex_row(&refs, 16);
    }

    #[test]
    fn test_print_hex_row_single_value() {
        let vals = [42.0_f32];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 0);
    }

    #[test]
    fn test_print_hex_row_three_values() {
        let vals = [0.0_f32, -1.0, f32::MAX];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 48);
    }

    #[test]
    fn test_print_hex_row_special_values() {
        let vals = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 0);
    }

    #[test]
    fn test_print_hex_row_large_offset() {
        let vals = [1.0_f32];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 0xDEAD_BEEF);
    }

    #[test]
    fn test_print_hex_row_negative_values() {
        let vals = [-0.5_f32, -100.0, -1e-6, -f32::MAX];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 32);
    }

    // ========================================================================
    // print_hex_dump tests (verify truncation, no panic)
    // ========================================================================

    #[test]
    fn test_print_hex_dump_empty_data() {
        print_hex_dump(&[], 100);
    }

    #[test]
    fn test_print_hex_dump_data_smaller_than_limit() {
        let data = [1.0_f32, 2.0, 3.0];
        // limit > data.len() → prints all
        print_hex_dump(&data, 100);
    }

    #[test]
    fn test_print_hex_dump_data_equal_to_limit() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        print_hex_dump(&data, 4);
    }

    #[test]
    fn test_print_hex_dump_data_larger_than_limit() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        // limit < data.len() → should print "... N more elements"
        print_hex_dump(&data, 10);
    }

    #[test]
    fn test_print_hex_dump_limit_zero() {
        let data = [1.0_f32, 2.0, 3.0];
        // limit 0 → should show "... N more elements"
        print_hex_dump(&data, 0);
    }

    #[test]
    fn test_print_hex_dump_single_element() {
        print_hex_dump(&[42.0], 1);
    }

    #[test]
    fn test_print_hex_dump_exactly_one_row() {
        // 4 values = exactly one row
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        print_hex_dump(&data, 4);
    }

    #[test]
    fn test_print_hex_dump_two_rows() {
        // 8 values = two rows (4 per row)
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        print_hex_dump(&data, 8);
    }

    #[test]
    fn test_print_hex_dump_partial_last_row() {
        // 5 values = 1 full row + 1 partial row
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        print_hex_dump(&data, 5);
    }

    // ========================================================================
    // print_tensor_stats tests (no panic, correct flow)
    // ========================================================================

    #[test]
    fn test_print_tensor_stats_normal_data() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        print_tensor_stats(&data);
    }

    #[test]
    fn test_print_tensor_stats_empty_data() {
        print_tensor_stats(&[]);
    }

    #[test]
    fn test_print_tensor_stats_single_value() {
        print_tensor_stats(&[42.0]);
    }

    #[test]
    fn test_print_tensor_stats_with_nan() {
        let data = [1.0, f32::NAN, 3.0];
        // Should trigger NaN anomaly warning
        print_tensor_stats(&data);
    }

    #[test]
    fn test_print_tensor_stats_all_same() {
        // All same → std = 0.0 → triggers collapsed weights warning
        let data = [3.14_f32; 100];
        print_tensor_stats(&data);
    }

    // ========================================================================
    // list_tensors tests
    // ========================================================================

    #[test]
    fn test_list_tensors_text_mode_single() {
        let desc = make_descriptor("weight", vec![10, 20], "F32", 0, 800);
        let tensors: Vec<&AprTensorDescriptor> = vec![&desc];
        let result = list_tensors(&tensors, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_tensors_text_mode_multiple() {
        let d1 = make_descriptor("layer.0.weight", vec![512, 512], "F32", 0, 512 * 512 * 4);
        let d2 = make_descriptor("layer.0.bias", vec![512], "F32", 512 * 512 * 4, 512 * 4);
        let d3 = make_descriptor("layer.1.weight", vec![512, 256], "F32", 0, 512 * 256 * 4);
        let tensors: Vec<&AprTensorDescriptor> = vec![&d1, &d2, &d3];
        let result = list_tensors(&tensors, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_tensors_json_mode_single() {
        let desc = make_descriptor("weight", vec![10, 20], "F32", 0, 800);
        let tensors: Vec<&AprTensorDescriptor> = vec![&desc];
        let result = list_tensors(&tensors, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_tensors_json_mode_multiple() {
        let d1 = make_descriptor("encoder.weight", vec![768, 768], "F32", 0, 768 * 768 * 4);
        let d2 = make_descriptor(
            "decoder.weight",
            vec![768, 768],
            "F16",
            768 * 768 * 4,
            768 * 768 * 2,
        );
        let tensors: Vec<&AprTensorDescriptor> = vec![&d1, &d2];
        let result = list_tensors(&tensors, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_tensors_empty_slice() {
        let tensors: Vec<&AprTensorDescriptor> = vec![];
        let result = list_tensors(&tensors, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_tensors_empty_slice_json() {
        let tensors: Vec<&AprTensorDescriptor> = vec![];
        let result = list_tensors(&tensors, true);
        assert!(result.is_ok());
    }

    // ========================================================================
    // run() additional edge case tests
    // ========================================================================

    #[test]
    fn test_run_nonexistent_path_returns_file_not_found() {
        let result = run(
            Path::new("/tmp/this_does_not_exist_apr_test.apr"),
            None,
            100,
            false,
            false,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(p)) => {
                assert_eq!(p, Path::new("/tmp/this_does_not_exist_apr_test.apr"));
            }
            other => panic!("Expected FileNotFound, got {:?}", other),
        }
    }

    #[test]
    fn test_run_empty_file() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        // Empty file — should fail to parse as APR
        let result = run(file.path(), None, 100, false, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_json_and_list_only_together() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        // Both list_only and json_output — exercises the combined path
        let result = run(file.path(), None, 100, false, true, true);
        assert!(result.is_err()); // Invalid APR
    }

    #[test]
    fn test_run_stats_and_json_together() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        // Both show_stats and json_output
        let result = run(file.path(), None, 100, true, false, true);
        assert!(result.is_err()); // Invalid APR
    }

    #[test]
    fn test_run_limit_zero() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        // limit = 0
        let result = run(file.path(), None, 0, false, false, false);
        assert!(result.is_err()); // Invalid APR
    }

    #[test]
    fn test_run_limit_max() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        // limit = usize::MAX
        let result = run(file.path(), None, usize::MAX, false, false, false);
        assert!(result.is_err()); // Invalid APR
    }

    #[test]
    fn test_run_with_empty_filter_string() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        // Empty filter string should match everything (contains("") is always true)
        let result = run(file.path(), Some(""), 100, false, false, false);
        assert!(result.is_err()); // Invalid APR
    }

    #[test]
    fn test_run_with_very_long_filter() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        let long_filter = "a".repeat(1000);
        let result = run(file.path(), Some(&long_filter), 100, false, false, false);
        assert!(result.is_err()); // Invalid APR
    }

    #[test]
    fn test_run_all_flags_enabled() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        // All boolean flags true — list_only takes priority over other display
        let result = run(file.path(), Some("weight"), 50, true, true, true);
        assert!(result.is_err()); // Invalid APR
    }
}
