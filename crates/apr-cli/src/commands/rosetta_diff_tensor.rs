
fn diff_tensor_pair(
    name: &str,
    tensor_a: Option<&TensorInfo>,
    tensor_b: Option<&TensorInfo>,
    mismatches_only: bool,
    json: bool,
    layout_mismatches: &mut Vec<(String, Vec<usize>, Vec<usize>)>,
    missing_in_a: &mut Vec<(String, Vec<usize>)>,
    missing_in_b: &mut Vec<(String, Vec<usize>)>,
) {
    let separator =
        "╠──────────────────────────────────────────────────────────────────────────────╣".cyan();
    match (tensor_a, tensor_b) {
        (Some(a), Some(b)) => {
            let dims_match = a.shape == b.shape;
            let is_transposed = is_transposed_dims(&a.shape, &b.shape);
            if !dims_match || !mismatches_only {
                if !json {
                    print_both_present(name, a, b, dims_match, is_transposed);
                }
                if is_transposed {
                    layout_mismatches.push((name.to_string(), a.shape.clone(), b.shape.clone()));
                }
            }
        }
        (Some(a), None) => {
            missing_in_b.push((name.to_string(), a.shape.clone()));
            if !mismatches_only && !json {
                println!("║ {} {:<72} ║", "−".red(), name);
                println!("║   A: {:?} (missing in B){}║", a.shape, " ".repeat(40));
                println!("{separator}");
            }
        }
        (None, Some(b)) => {
            missing_in_a.push((name.to_string(), b.shape.clone()));
            if !mismatches_only && !json {
                println!("║ {} {:<72} ║", "+".green(), name);
                println!("║   B: {:?} (missing in A){}║", b.shape, " ".repeat(40));
                println!("{separator}");
            }
        }
        (None, None) => {}
    }
}

pub fn run_diff_tensors(
    model_a: &Path,
    model_b: &Path,
    mismatches_only: bool,
    show_values: usize,
    filter: Option<&str>,
    json: bool,
) -> Result<()> {
    if !model_a.exists() {
        return Err(CliError::FileNotFound(model_a.to_path_buf()));
    }
    if !model_b.exists() {
        return Err(CliError::FileNotFound(model_b.to_path_buf()));
    }

    let rosetta = RosettaStone::new();

    // F-GT-002: Check for mixed quantization levels (R3 violation)
    if let Some(warning) = check_mixed_quant_warning(model_a, model_b) {
        if !json {
            println!("{}", warning.yellow());
            println!();
        }
    }

    // Inspect both models
    let report_a = rosetta
        .inspect(model_a)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model A: {e}")))?;
    let report_b = rosetta
        .inspect(model_b)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model B: {e}")))?;

    // Build tensor maps by normalized name (GH-202: cross-format tensor matching)
    let tensors_a: std::collections::HashMap<String, _> = report_a
        .tensors
        .iter()
        .map(|t| (normalize_tensor_name(&t.name), t))
        .collect();
    let tensors_b: std::collections::HashMap<String, _> = report_b
        .tensors
        .iter()
        .map(|t| (normalize_tensor_name(&t.name), t))
        .collect();

    // Collect all unique tensor names
    let mut all_names: Vec<_> = tensors_a.keys().chain(tensors_b.keys()).collect();
    all_names.sort();
    all_names.dedup();

    // Apply filter
    let filtered_names: Vec<_> = if let Some(pattern) = filter {
        all_names
            .into_iter()
            .filter(|n| n.contains(pattern))
            .collect()
    } else {
        all_names
    };

    let mut layout_mismatches = Vec::new();
    let mut missing_in_a = Vec::new();
    let mut missing_in_b = Vec::new();

    if !json {
        print_diff_header(
            model_a,
            model_b,
            report_a.tensors.len(),
            report_b.tensors.len(),
        );
    }

    for name in &filtered_names {
        let tensor_a = tensors_a.get(*name);
        let tensor_b = tensors_b.get(*name);
        diff_tensor_pair(
            name,
            tensor_a.copied(),
            tensor_b.copied(),
            mismatches_only,
            json,
            &mut layout_mismatches,
            &mut missing_in_a,
            &mut missing_in_b,
        );
    }

    // Summary
    if json {
        print_diff_json_summary(
            model_a,
            model_b,
            tensors_a.len(),
            tensors_b.len(),
            &layout_mismatches,
            &missing_in_a,
            &missing_in_b,
        );
    } else {
        print_diff_text_summary(
            tensors_a.len(),
            tensors_b.len(),
            &layout_mismatches,
            &missing_in_a,
            &missing_in_b,
        );
    }

    // Return error if mismatches found (for CI assertion)
    let count_a = report_a.tensors.len();
    let count_b = report_b.tensors.len();

    // GH-188: Tensor count mismatch is CRITICAL - fail immediately
    if count_a != count_b {
        return Err(CliError::ValidationFailed(format!(
            "TENSOR COUNT MISMATCH: Model A has {} tensors, Model B has {} ({} missing!)",
            count_a,
            count_b,
            (count_a as i64 - count_b as i64).abs()
        )));
    }

    if !layout_mismatches.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "Layout mismatch: {} tensors have transposed dimensions",
            layout_mismatches.len()
        )));
    }

    // PMAT-GLASS-HOUSE: show_values feature deferred (P2)
    // When show_values > 0, user expects to see tensor value samples.
    // Currently not implemented - inform user rather than silently ignore.
    if show_values > 0 {
        eprintln!(
            "Note: --show-values {} requested but value comparison not yet implemented. \
             Use 'apr rosetta fingerprint' for tensor statistics.",
            show_values
        );
    }

    Ok(())
}

/// Run the rosetta fingerprint subcommand (PMAT-201)
///
/// Computes statistical fingerprints for all tensors in a model.
/// Fingerprints include: mean, std, min, max, percentiles, nan/inf counts.
/// Print the fingerprint banner header (text mode).
fn print_fingerprint_banner(model: &Path) {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════════╗".cyan()
    );
    println!(
        "{}",
        "║           TENSOR STATISTICAL FINGERPRINTS (PMAT-201, JAX-STAT-001)          ║".cyan()
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
    );
    println!(
        "║ Model: {:<69} ║",
        truncate_path(model.display().to_string(), 69)
    );
}

/// Run fingerprint comparison or single-model output.
fn run_fingerprint_body(
    fingerprints_a: &[TensorFingerprint],
    model_b: Option<&Path>,
    filter: Option<&str>,
    verbose: bool,
    json: bool,
) -> Result<()> {
    let Some(model_b_path) = model_b else {
        if !json {
            println!(
                "{}",
                "╠══════════════════════════════════════════════════════════════════════════════╣"
                    .cyan()
            );
        }
        return print_fingerprints(fingerprints_a, verbose, json);
    };

    if !model_b_path.exists() {
        return Err(CliError::FileNotFound(model_b_path.to_path_buf()));
    }
    if !json {
        println!(
            "║ Compare: {:<67} ║",
            truncate_path(model_b_path.display().to_string(), 67)
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
    }
    let fingerprints_b = compute_fingerprints(model_b_path, filter)?;
    print_fingerprint_diff(fingerprints_a, &fingerprints_b, verbose, json)
}

pub fn run_fingerprint(
    model: &Path,
    model_b: Option<&Path>,
    output: Option<&Path>,
    filter: Option<&str>,
    verbose: bool,
    json: bool,
) -> Result<()> {
    if !model.exists() {
        return Err(CliError::FileNotFound(model.to_path_buf()));
    }

    if !json {
        print_fingerprint_banner(model);
    }

    let fingerprints_a = compute_fingerprints(model, filter)?;
    run_fingerprint_body(&fingerprints_a, model_b, filter, verbose, json)?;

    if let Some(output_path) = output {
        let json_content = fingerprints_to_json(&fingerprints_a);
        std::fs::write(output_path, json_content).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to write fingerprints: {e}"))
        })?;
        if !json {
            println!("║ Saved fingerprints to: {:<53} ║", output_path.display());
        }
    }

    if !json {
        println!(
            "{}",
            "╚══════════════════════════════════════════════════════════════════════════════╝"
                .cyan()
        );
    }

    Ok(())
}

/// Run the rosetta validate-stats subcommand (PMAT-202)
///
/// Validates tensor statistics against a reference model or stored fingerprints.
/// Resolve reference fingerprints from either a reference model or a fingerprints file.
fn resolve_reference_fingerprints(
    reference: Option<&Path>,
    fingerprints_file: Option<&Path>,
    json: bool,
) -> Result<Vec<TensorFingerprint>> {
    if let Some(ref_path) = reference {
        if !ref_path.exists() {
            return Err(CliError::FileNotFound(ref_path.to_path_buf()));
        }
        if !json {
            println!(
                "║ Reference: {:<65} ║",
                truncate_path(ref_path.display().to_string(), 65)
            );
        }
        compute_fingerprints(ref_path, None)
    } else if let Some(fp_path) = fingerprints_file {
        if !fp_path.exists() {
            return Err(CliError::FileNotFound(fp_path.to_path_buf()));
        }
        if !json {
            println!(
                "║ Fingerprints: {:<62} ║",
                truncate_path(fp_path.display().to_string(), 62)
            );
        }
        load_fingerprints_from_json(fp_path)
    } else {
        unreachable!()
    }
}

/// Print validation anomalies as JSON.
fn print_validate_stats_json(
    model: &Path,
    threshold: f32,
    strict: bool,
    total_tensors: usize,
    anomalies: &[StatisticalAnomaly],
) {
    println!("{{");
    println!("  \"model\": \"{}\",", model.display());
    println!("  \"threshold\": {},", threshold);
    println!("  \"strict\": {},", strict);
    println!("  \"total_tensors\": {},", total_tensors);
    println!("  \"anomalies\": {},", anomalies.len());
    if !anomalies.is_empty() {
        println!("  \"anomaly_details\": [");
        for (i, anomaly) in anomalies.iter().enumerate() {
            let comma = if i < anomalies.len() - 1 { "," } else { "" };
            println!(
                "    {{\"tensor\": \"{}\", \"field\": \"{}\", \"expected\": {:.6}, \"actual\": {:.6}, \"deviation\": {:.2}}}{}",
                anomaly.tensor, anomaly.field, anomaly.expected, anomaly.actual, anomaly.deviation_sigma, comma
            );
        }
        println!("  ],");
    }
    println!("  \"passed\": {}", anomalies.is_empty());
    println!("}}");
}

/// Print validation anomalies as formatted text.
fn print_validate_stats_text(anomalies: &[StatisticalAnomaly]) {
    if anomalies.is_empty() {
        println!(
            "║ {} ║",
            "✓ All tensors within expected statistical bounds"
                .green()
                .bold()
        );
    } else {
        println!(
            "║ {} ║",
            format!("✗ {} STATISTICAL ANOMALIES DETECTED", anomalies.len())
                .red()
                .bold()
        );
        println!(
            "{}",
            "╠──────────────────────────────────────────────────────────────────────────────╣"
                .cyan()
        );

        for anomaly in anomalies {
            let severity = if anomaly.deviation_sigma > 10.0 {
                "CRITICAL".red().bold()
            } else if anomaly.deviation_sigma > 5.0 {
                "WARNING".yellow()
            } else {
                "INFO".white()
            };

            println!("║ {} {} ║", severity, anomaly.tensor);
            println!(
                "║   {}: expected={:.6}, actual={:.6}, deviation={:.1}σ ║",
                anomaly.field, anomaly.expected, anomaly.actual, anomaly.deviation_sigma
            );
        }
    }
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════════╝".cyan()
    );
}
