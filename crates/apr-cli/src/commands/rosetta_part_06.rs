
/// Compute statistics for a tensor
#[allow(clippy::type_complexity)]
fn compute_tensor_stats(
    values: &[f32],
) -> (
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    u32,
    u32,
    f32,
    u32,
) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0);
    }

    let mut nan_count = 0u32;
    let mut inf_count = 0u32;
    let mut zero_count = 0u32;
    let mut sum = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut checksum = 0u32;

    // Collect valid values for percentile calculation
    let mut valid_values: Vec<f32> = Vec::with_capacity(values.len());

    for &v in values {
        // Update checksum (simple CRC-like)
        checksum = checksum.wrapping_add(v.to_bits());

        if v.is_nan() {
            nan_count += 1;
        } else if v.is_infinite() {
            inf_count += 1;
        } else {
            valid_values.push(v);
            sum += v as f64;
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            if v == 0.0 {
                zero_count += 1;
            }
        }
    }

    let n = valid_values.len();
    if n == 0 {
        return (
            0.0, 0.0, min, max, 0.0, 0.0, 0.0, 0.0, 0.0, nan_count, inf_count, 0.0, checksum,
        );
    }

    let mean = (sum / n as f64) as f32;

    // Compute std
    let variance: f64 = valid_values
        .iter()
        .map(|&v| {
            let diff = v as f64 - sum / n as f64;
            diff * diff
        })
        .sum::<f64>()
        / n as f64;
    let std = variance.sqrt() as f32;

    // Compute percentiles (sort for percentile calculation)
    valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile = |p: f32| -> f32 {
        let idx = ((p / 100.0) * (n - 1) as f32) as usize;
        valid_values[idx.min(n - 1)]
    };

    let p5 = percentile(5.0);
    let p25 = percentile(25.0);
    let p50 = percentile(50.0);
    let p75 = percentile(75.0);
    let p95 = percentile(95.0);

    let zero_fraction = zero_count as f32 / values.len() as f32;

    (
        mean,
        std,
        min,
        max,
        p5,
        p25,
        p50,
        p75,
        p95,
        nan_count,
        inf_count,
        zero_fraction,
        checksum,
    )
}

/// Print fingerprints
fn print_fingerprints(fingerprints: &[TensorFingerprint], verbose: bool, json: bool) -> Result<()> {
    if json {
        println!("{}", fingerprints_to_json(fingerprints));
        return Ok(());
    }

    for fp in fingerprints {
        println!("║ {:<74} ║", truncate_path(fp.name.clone(), 74));
        println!("║   shape={:?} dtype={:<10} ║", fp.shape, fp.dtype);
        if verbose {
            println!(
                "║   mean={:>10.6} std={:>10.6} min={:>10.6} max={:>10.6} ║",
                fp.mean, fp.std, fp.min, fp.max
            );
            println!(
                "║   p5={:>10.6} p25={:>10.6} p50={:>10.6} p75={:>10.6} p95={:>10.6} ║",
                fp.p5, fp.p25, fp.p50, fp.p75, fp.p95
            );
            println!(
                "║   nan={} inf={} zero_frac={:.4} checksum=0x{:08X} ║",
                fp.nan_count, fp.inf_count, fp.zero_fraction, fp.checksum
            );
        } else {
            println!(
                "║   mean={:>10.6} std={:>10.6} nan={} inf={} ║",
                fp.mean, fp.std, fp.nan_count, fp.inf_count
            );
        }
        println!(
            "{}",
            "╠──────────────────────────────────────────────────────────────────────────────╣"
                .cyan()
        );
    }

    println!("║ Total tensors: {:<61} ║", fingerprints.len());

    Ok(())
}

/// Print fingerprint diff between two models
/// Compute normalized mean diff and detect anomalies between two fingerprints.
fn fingerprint_anomaly(fp_a: &TensorFingerprint, fp_b: &TensorFingerprint) -> (f32, bool) {
    let mean_diff = if fp_a.std > 1e-10 {
        (fp_a.mean - fp_b.mean).abs() / fp_a.std
    } else {
        (fp_a.mean - fp_b.mean).abs()
    };
    let has_anomaly =
        mean_diff > 3.0 || fp_a.nan_count != fp_b.nan_count || fp_a.inf_count != fp_b.inf_count;
    (mean_diff, has_anomaly)
}

/// Print a single tensor comparison row (text mode).
fn print_diff_row(
    fp_a: &TensorFingerprint,
    fp_b: &TensorFingerprint,
    mean_diff: f32,
    has_anomaly: bool,
) {
    let status = if has_anomaly {
        "⚠️".yellow()
    } else {
        "✓".green()
    };
    println!(
        "║ {} {:<72} ║",
        status,
        truncate_path(fp_a.name.clone(), 72)
    );
    println!(
        "║   A: mean={:>10.6} std={:>10.6} nan={} inf={} ║",
        fp_a.mean, fp_a.std, fp_a.nan_count, fp_a.inf_count
    );
    println!(
        "║   B: mean={:>10.6} std={:>10.6} nan={} inf={} ║",
        fp_b.mean, fp_b.std, fp_b.nan_count, fp_b.inf_count
    );
    if has_anomaly {
        println!(
            "║   {} mean_diff={:.2}σ ║",
            "ANOMALY:".red().bold(),
            mean_diff
        );
    }
    println!(
        "{}",
        "╠──────────────────────────────────────────────────────────────────────────────╣".cyan()
    );
}

/// Print the diff summary (JSON or text).
fn print_diff_summary(total: usize, anomalies: &[StatisticalAnomaly], json: bool) {
    if json {
        println!("{{");
        println!("  \"total_tensors\": {},", total);
        println!("  \"anomalies\": {},", anomalies.len());
        println!("  \"passed\": {}", anomalies.is_empty());
        println!("}}");
    } else if anomalies.is_empty() {
        println!(
            "║ {} ║",
            "✓ No statistical anomalies detected".green().bold()
        );
    } else {
        println!(
            "║ {} ║",
            format!("✗ {} ANOMALIES DETECTED", anomalies.len())
                .red()
                .bold()
        );
    }
}

fn print_fingerprint_diff(
    fps_a: &[TensorFingerprint],
    fps_b: &[TensorFingerprint],
    verbose: bool,
    json: bool,
) -> Result<()> {
    // GH-202: Use normalized names for cross-format matching
    let map_b: std::collections::HashMap<_, _> = fps_b
        .iter()
        .map(|fp| (normalize_tensor_name(&fp.name), fp))
        .collect();

    let mut anomalies = Vec::new();

    if !json {
        println!(
            "{}",
            "║                              FINGERPRINT DIFF                                ║"
                .yellow()
        );
        println!(
            "{}",
            "╠──────────────────────────────────────────────────────────────────────────────╣"
                .cyan()
        );
    }

    for fp_a in fps_a {
        let norm_name_a = normalize_tensor_name(&fp_a.name);
        let Some(fp_b) = map_b.get(&norm_name_a) else {
            if !json {
                println!(
                    "║ {} {:<72} ║",
                    "−".red(),
                    truncate_path(fp_a.name.clone(), 72)
                );
                println!("║   Missing in Model B ║");
            }
            continue;
        };

        let (mean_diff, has_anomaly) = fingerprint_anomaly(fp_a, fp_b);

        if has_anomaly || verbose {
            if !json {
                print_diff_row(fp_a, fp_b, mean_diff, has_anomaly);
            }
            if has_anomaly {
                anomalies.push(StatisticalAnomaly {
                    tensor: fp_a.name.clone(),
                    field: "mean".to_string(),
                    expected: fp_a.mean,
                    actual: fp_b.mean,
                    deviation_sigma: mean_diff,
                });
            }
        }
    }

    print_diff_summary(fps_a.len(), &anomalies, json);
    Ok(())
}

/// Convert fingerprints to JSON
fn fingerprints_to_json(fingerprints: &[TensorFingerprint]) -> String {
    let mut json = String::from("{\n  \"fingerprints\": [\n");

    for (i, fp) in fingerprints.iter().enumerate() {
        let comma = if i < fingerprints.len() - 1 { "," } else { "" };
        write!(
            json,
            "    {{\n      \"name\": \"{}\",\n      \"shape\": {:?},\n      \"dtype\": \"{}\",\n      \"mean\": {},\n      \"std\": {},\n      \"min\": {},\n      \"max\": {},\n      \"p5\": {},\n      \"p25\": {},\n      \"p50\": {},\n      \"p75\": {},\n      \"p95\": {},\n      \"nan_count\": {},\n      \"inf_count\": {},\n      \"zero_fraction\": {},\n      \"checksum\": {}\n    }}{}\n",
            fp.name, fp.shape, fp.dtype, fp.mean, fp.std, fp.min, fp.max,
            fp.p5, fp.p25, fp.p50, fp.p75, fp.p95,
            fp.nan_count, fp.inf_count, fp.zero_fraction, fp.checksum, comma
        )
        .expect("write to String should not fail");
    }

    json.push_str("  ]\n}");
    json
}

/// Load fingerprints from JSON file
fn load_fingerprints_from_json(path: &Path) -> Result<Vec<TensorFingerprint>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read fingerprints: {e}")))?;

    // Simple JSON parsing - in production would use serde
    let mut fingerprints = Vec::new();

    // Parse JSON manually (simplified)
    for line in content.lines() {
        if line.contains("\"name\":") {
            // Extract tensor info from JSON
            // This is a placeholder - proper implementation would use serde_json
            let name = line
                .split("\"name\": \"")
                .nth(1)
                .and_then(|s| s.split('"').next())
                .unwrap_or("unknown")
                .to_string();

            fingerprints.push(TensorFingerprint {
                name,
                shape: vec![],
                dtype: "unknown".to_string(),
                mean: 0.0,
                std: 1.0,
                min: 0.0,
                max: 0.0,
                p5: 0.0,
                p25: 0.0,
                p50: 0.0,
                p75: 0.0,
                p95: 0.0,
                nan_count: 0,
                inf_count: 0,
                zero_fraction: 0.0,
                checksum: 0,
            });
        }
    }

    Ok(fingerprints)
}

/// Validate fingerprints against reference
fn validate_fingerprints(
    actual: &[TensorFingerprint],
    reference: &[TensorFingerprint],
    threshold: f32,
    strict: bool,
) -> Vec<StatisticalAnomaly> {
    let ref_map: std::collections::HashMap<_, _> = reference
        .iter()
        .map(|fp| (normalize_tensor_name(&fp.name), fp))
        .collect();

    let mut anomalies = Vec::new();

    for actual_fp in actual {
        let norm_name = normalize_tensor_name(&actual_fp.name);
        if let Some(ref_fp) = ref_map.get(&norm_name) {
            // Get role-specific threshold
            let role_threshold = if strict {
                get_role_threshold(&actual_fp.name)
            } else {
                threshold
            };

            // Check mean
            let mean_deviation = if ref_fp.std > 1e-10 {
                (actual_fp.mean - ref_fp.mean).abs() / ref_fp.std
            } else {
                (actual_fp.mean - ref_fp.mean).abs() * 1000.0 // Scale up small differences
            };

            if mean_deviation > role_threshold {
                anomalies.push(StatisticalAnomaly {
                    tensor: actual_fp.name.clone(),
                    field: "mean".to_string(),
                    expected: ref_fp.mean,
                    actual: actual_fp.mean,
                    deviation_sigma: mean_deviation,
                });
            }

            // Check for NaN/Inf (always anomalous if reference doesn't have them)
            if actual_fp.nan_count > 0 && ref_fp.nan_count == 0 {
                anomalies.push(StatisticalAnomaly {
                    tensor: actual_fp.name.clone(),
                    field: "nan_count".to_string(),
                    expected: ref_fp.nan_count as f32,
                    actual: actual_fp.nan_count as f32,
                    deviation_sigma: f32::INFINITY,
                });
            }

            if actual_fp.inf_count > 0 && ref_fp.inf_count == 0 {
                anomalies.push(StatisticalAnomaly {
                    tensor: actual_fp.name.clone(),
                    field: "inf_count".to_string(),
                    expected: ref_fp.inf_count as f32,
                    actual: actual_fp.inf_count as f32,
                    deviation_sigma: f32::INFINITY,
                });
            }
        }
    }

    anomalies
}

/// Get role-specific threshold based on tensor name
fn get_role_threshold(tensor_name: &str) -> f32 {
    let name_lower = tensor_name.to_lowercase();

    if name_lower.contains("layernorm")
        || name_lower.contains("layer_norm")
        || name_lower.contains("ln_")
    {
        // LayerNorm weights should be very close to 1.0 - tight threshold
        2.0
    } else if name_lower.contains("embed") {
        // Embeddings can have more variance
        5.0
    } else if name_lower.contains("lm_head") || name_lower.contains("output") {
        // Output layers - moderate threshold
        3.0
    } else {
        // Default threshold for other weights
        3.0
    }
}
