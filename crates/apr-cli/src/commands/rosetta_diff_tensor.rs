
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

fn ensure_model_paths_exist(model_a: &Path, model_b: &Path) -> Result<()> {
    if !model_a.exists() {
        return Err(CliError::FileNotFound(model_a.to_path_buf()));
    }
    if !model_b.exists() {
        return Err(CliError::FileNotFound(model_b.to_path_buf()));
    }
    Ok(())
}

fn emit_mixed_quant_warning(model_a: &Path, model_b: &Path, json: bool) {
    if json {
        return;
    }
    if let Some(warning) = check_mixed_quant_warning(model_a, model_b) {
        println!("{}", warning.yellow());
        println!();
    }
}

fn inspect_model_report(rosetta: &RosettaStone, path: &Path, label: &str) -> Result<InspectionReport> {
    rosetta
        .inspect(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model {label}: {e}")))
}

fn sorted_unique_filtered_tensor_names<'a>(
    tensors_a: &'a std::collections::HashMap<String, &'a TensorInfo>,
    tensors_b: &'a std::collections::HashMap<String, &'a TensorInfo>,
    filter: Option<&str>,
) -> Vec<&'a String> {
    let mut all_names: Vec<_> = tensors_a.keys().chain(tensors_b.keys()).collect();
    all_names.sort();
    all_names.dedup();
    match filter {
        Some(pattern) => all_names.into_iter().filter(|n| n.contains(pattern)).collect(),
        None => all_names,
    }
}

#[derive(Default)]
struct DiffBuckets {
    layout_mismatches: Vec<(String, Vec<usize>, Vec<usize>)>,
    missing_in_a: Vec<(String, Vec<usize>)>,
    missing_in_b: Vec<(String, Vec<usize>)>,
}

fn emit_diff_summary(
    model_a: &Path,
    model_b: &Path,
    count_a: usize,
    count_b: usize,
    buckets: &DiffBuckets,
    json: bool,
) {
    if json {
        print_diff_json_summary(
            model_a,
            model_b,
            count_a,
            count_b,
            &buckets.layout_mismatches,
            &buckets.missing_in_a,
            &buckets.missing_in_b,
        );
    } else {
        print_diff_text_summary(
            count_a,
            count_b,
            &buckets.layout_mismatches,
            &buckets.missing_in_a,
            &buckets.missing_in_b,
        );
    }
}

fn check_diff_errors(count_a: usize, count_b: usize, buckets: &DiffBuckets) -> Result<()> {
    if count_a != count_b {
        return Err(CliError::ValidationFailed(format!(
            "TENSOR COUNT MISMATCH: Model A has {} tensors, Model B has {} ({} missing!)",
            count_a,
            count_b,
            (count_a as i64 - count_b as i64).abs()
        )));
    }
    if !buckets.layout_mismatches.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "Layout mismatch: {} tensors have transposed dimensions",
            buckets.layout_mismatches.len()
        )));
    }
    Ok(())
}

fn warn_show_values_unimplemented(show_values: usize) {
    if show_values == 0 {
        return;
    }
    eprintln!(
        "Note: --show-values {show_values} requested but value comparison not yet implemented. \
         Use 'apr rosetta fingerprint' for tensor statistics.",
    );
}

#[provable_contracts_macros::contract("apr-cli-command-safety-v1", equation = "read_only_no_side_effects")]
pub fn run_diff_tensors(
    model_a: &Path,
    model_b: &Path,
    mismatches_only: bool,
    show_values: usize,
    filter: Option<&str>,
    json: bool,
) -> Result<()> {
    ensure_model_paths_exist(model_a, model_b)?;

    let rosetta = RosettaStone::new();
    emit_mixed_quant_warning(model_a, model_b, json);

    let report_a = inspect_model_report(&rosetta, model_a, "A")?;
    let report_b = inspect_model_report(&rosetta, model_b, "B")?;

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

    let filtered_names = sorted_unique_filtered_tensor_names(&tensors_a, &tensors_b, filter);

    if !json {
        print_diff_header(
            model_a,
            model_b,
            report_a.tensors.len(),
            report_b.tensors.len(),
        );
    }

    let mut buckets = DiffBuckets::default();
    for name in &filtered_names {
        diff_tensor_pair(
            name,
            tensors_a.get(*name).copied(),
            tensors_b.get(*name).copied(),
            mismatches_only,
            json,
            &mut buckets.layout_mismatches,
            &mut buckets.missing_in_a,
            &mut buckets.missing_in_b,
        );
    }

    emit_diff_summary(
        model_a,
        model_b,
        tensors_a.len(),
        tensors_b.len(),
        &buckets,
        json,
    );

    check_diff_errors(report_a.tensors.len(), report_b.tensors.len(), &buckets)?;
    warn_show_values_unimplemented(show_values);

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
    // Deliberately not `?`: a failing diff must still write --output and close the
    // report box, but it must still be the exit status of the command.
    let diff_result = run_fingerprint_body(&fingerprints_a, model_b, filter, verbose, json);

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

    diff_result
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
///
/// Built with serde rather than hand-rolled `println!`: `deviation` is `f32::INFINITY`
/// for nan/inf-count anomalies, and a bare `inf` literal is not valid JSON.
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn validate_stats_json(
    model: &Path,
    threshold: f32,
    strict: bool,
    total_tensors: usize,
    anomalies: &[StatisticalAnomaly],
) -> serde_json::Value {
    let details: Vec<serde_json::Value> = anomalies
        .iter()
        .map(|a| {
            serde_json::json!({
                "tensor": a.tensor,
                "field": a.field,
                "expected": json_number(a.expected),
                "actual": json_number(a.actual),
                "deviation": json_number(a.deviation_sigma),
            })
        })
        .collect();

    serde_json::json!({
        "model": model.display().to_string(),
        "threshold": json_number(threshold),
        "strict": strict,
        "total_tensors": total_tensors,
        "anomalies": anomalies.len(),
        "anomaly_details": details,
        "passed": anomalies.is_empty(),
    })
}

fn print_validate_stats_json(
    model: &Path,
    threshold: f32,
    strict: bool,
    total_tensors: usize,
    anomalies: &[StatisticalAnomaly],
) {
    println!(
        "{:#}",
        validate_stats_json(model, threshold, strict, total_tensors, anomalies)
    );
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
