
/// List tensors from SafeTensors via mmap (efficient for large files)
fn list_tensors_safetensors_path(
    path: &Path,
    options: TensorListOptions,
) -> Result<TensorListResult> {
    use crate::serialization::safetensors::MappedSafeTensors;

    let mapped = MappedSafeTensors::open(path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to open SafeTensors: {e}"),
    })?;

    let mut tensors = Vec::new();
    let mut total_size = 0usize;
    let mut total_matching = 0usize;

    let mut names: Vec<&str> = mapped.tensor_names();
    names.sort_unstable();

    for name in names {
        if let Some(ref pattern) = options.filter {
            if !name.contains(pattern.as_str()) {
                continue;
            }
        }

        if let Some(meta) = mapped.get_metadata(name) {
            let size_bytes = meta.data_offsets[1] - meta.data_offsets[0];

            total_size += size_bytes;
            total_matching += 1;

            // Only collect details up to the limit
            if tensors.len() < options.limit {
                let mut info = TensorInfo {
                    name: name.to_string(),
                    shape: meta.shape.clone(),
                    dtype: meta.dtype.clone(),
                    size_bytes,
                    mean: None,
                    std: None,
                    min: None,
                    max: None,
                    nan_count: None,
                    inf_count: None,
                };

                if options.compute_stats {
                    if let Ok(f32_data) = mapped.get_tensor(name) {
                        compute_tensor_stats(&mut info, &f32_data);
                    }
                }

                tensors.push(info);
            }
        }
    }

    Ok(TensorListResult {
        file: String::new(),
        format_version: "SafeTensors".to_string(),
        tensor_count: total_matching,
        total_size_bytes: total_size,
        tensors,
    })
}

// ============================================================================
// Statistics Computation
// ============================================================================

/// Compute tensor statistics (mean, std, min, max, nan/inf count)
fn compute_tensor_stats(info: &mut TensorInfo, data: &[f32]) {
    if data.is_empty() {
        return;
    }

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut valid_count = 0usize;

    for &val in data {
        if val.is_nan() {
            nan_count += 1;
            continue;
        }
        if val.is_infinite() {
            inf_count += 1;
            continue;
        }

        valid_count += 1;
        let val_f64 = f64::from(val);
        sum += val_f64;
        sum_sq += val_f64 * val_f64;

        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }

    info.nan_count = Some(nan_count);
    info.inf_count = Some(inf_count);

    if valid_count > 0 {
        let n = valid_count as f64;
        let mean = sum / n;
        let variance = (sum_sq / n) - (mean * mean);
        let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };

        info.mean = Some(mean as f32);
        info.std = Some(std as f32);
        info.min = Some(min);
        info.max = Some(max);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Format size in human-readable form
#[must_use]
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
#[path = "tensors_tests.rs"]
mod tests;
