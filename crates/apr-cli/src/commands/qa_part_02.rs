
/// Check if model is GGUF format (for Ollama parity gate).
fn is_gguf_format(path: &Path) -> bool {
    #[cfg(feature = "inference")]
    {
        use realizar::format::{detect_format, ModelFormat};
        let magic = std::fs::read(path).ok().and_then(|b| {
            if b.len() >= 8 {
                Some(b[..8].to_vec())
            } else {
                None
            }
        });
        magic.and_then(|m| detect_format(&m).ok()) == Some(ModelFormat::Gguf)
    }
    #[cfg(not(feature = "inference"))]
    {
        let _ = path;
        false
    }
}

fn run_qa(path: &Path, config: &QaConfig) -> Result<QaReport> {
    let start = Instant::now();
    let mut gates = Vec::new();

    if !config.json {
        output::header("APR Quality Assurance");
        let config_pairs = vec![
            ("Model", path.display().to_string()),
            ("Min TPS", format!("{:.0} tok/s", config.min_tps)),
            ("Min Speedup", format!("{:.1}x Ollama", config.min_speedup)),
        ];
        println!("{}", output::kv_table(&config_pairs));
    }

    dispatch_gate(
        &mut gates,
        config.json,
        config.skip_contract,
        "tensor_contract",
        "Skipped by --skip-contract",
        || run_tensor_contract_gate(path, config),
    )?;
    dispatch_gate(
        &mut gates,
        config.json,
        config.skip_metadata,
        "metadata_plausibility",
        "Skipped by --skip-metadata",
        || run_metadata_plausibility_gate(path, config),
    )?;
    dispatch_gate(
        &mut gates,
        config.json,
        config.skip_golden,
        "golden_output",
        "Skipped by --skip-golden",
        || run_golden_output_gate(path, config),
    )?;
    dispatch_gate(
        &mut gates,
        config.json,
        config.skip_throughput,
        "throughput",
        "Skipped by --skip-throughput",
        || run_throughput_gate(path, config),
    )?;
    dispatch_gate(
        &mut gates,
        config.json,
        config.skip_ollama,
        "ollama_parity",
        "Skipped by --skip-ollama",
        || {
            if is_gguf_format(path) {
                run_ollama_parity_gate(path, config)
            } else {
                Ok(GateResult::skipped(
                    "ollama_parity",
                    "Non-GGUF format (F32/F16 lacks fused kernels for Ollama parity)",
                ))
            }
        },
    )?;
    dispatch_gate(
        &mut gates,
        config.json,
        config.skip_gpu_speedup,
        "gpu_speedup",
        "Skipped by --skip-gpu-speedup",
        || run_gpu_speedup_gate(path, config),
    )?;

    let (skip_format, format_skip_reason) = format_parity_skip_status(config);
    dispatch_gate(
        &mut gates,
        config.json,
        skip_format,
        "format_parity",
        format_skip_reason,
        || run_format_parity_gate(path, config),
    )?;
    dispatch_gate(
        &mut gates,
        config.json,
        config.skip_ptx_parity,
        "ptx_parity",
        "Skipped by --skip-ptx-parity",
        || run_ptx_parity_gate(path, config),
    )?;
    dispatch_gate(
        &mut gates,
        config.json,
        config.skip_gpu_state,
        "gpu_state_isolation",
        "Skipped by --skip-gpu-state",
        || run_gpu_state_isolation_gate(path, config),
    )?;

    // Gate 9: Performance regression detection (auto-discovers previous report)
    dispatch_regression_gate(path, &mut gates, config)?;

    let report = finalize_qa_report(path, &start, gates, config)?;

    // P0-QA-001: Save report to cache for future regression comparison
    save_qa_report_to_cache(path, &report);

    Ok(report)
}

/// Determine skip status for format parity gate.
/// P0-QA-001: Gates must never silently skip. Only explicit --skip flags skip.
fn format_parity_skip_status(config: &QaConfig) -> (bool, &str) {
    if config.skip_format_parity {
        (true, "Skipped by --skip-format-parity")
    } else {
        (false, "")
    }
}

/// P0-QA-001: Auto-discover previous QA report for regression comparison.
///
/// Reports are saved to `~/.cache/apr/qa-reports/{model-basename}.json` after each run.
/// On subsequent runs, the previous report is auto-loaded for comparison.
fn auto_discover_previous_report(model_path: &Path) -> Option<std::path::PathBuf> {
    let cache_dir = dirs::home_dir()?.join(".cache/apr/qa-reports");
    let basename = model_path.file_stem()?.to_str()?;
    let report_path = cache_dir.join(format!("{basename}.json"));
    if report_path.exists() {
        Some(report_path)
    } else {
        None
    }
}

/// Save QA report to cache for future regression comparison.
fn save_qa_report_to_cache(model_path: &Path, report: &QaReport) {
    let Some(home) = dirs::home_dir() else {
        return;
    };
    let cache_dir = home.join(".cache/apr/qa-reports");
    if std::fs::create_dir_all(&cache_dir).is_err() {
        return;
    }
    let Some(basename) = model_path.file_stem().and_then(|s| s.to_str()) else {
        return;
    };
    let report_path = cache_dir.join(format!("{basename}.json"));
    if let Ok(json) = serde_json::to_string_pretty(report) {
        let _ = std::fs::write(&report_path, json);
    }
}

/// Run the performance regression gate (needs read access to existing gates).
/// P0-QA-001: Never skip — auto-discover previous report or establish baseline.
fn dispatch_regression_gate(
    model_path: &Path,
    gates: &mut Vec<GateResult>,
    config: &QaConfig,
) -> Result<()> {
    let regression_result = if let Some(ref _prev) = config.previous_report {
        run_performance_regression_gate(gates, config)?
    } else {
        // Auto-discover previous report from cache
        match auto_discover_previous_report(model_path) {
            Some(prev_path) => {
                if !config.json {
                    println!(
                        "  {} Auto-discovered previous report: {}",
                        "INFO".cyan(),
                        prev_path.display()
                    );
                }
                let auto_config = QaConfig {
                    previous_report: Some(prev_path),
                    ..config.clone()
                };
                run_performance_regression_gate(gates, &auto_config)?
            }
            None => {
                // First run — establish baseline (PASS, not skip)
                GateResult::passed(
                    "performance_regression",
                    "First run — baseline established (saved for future comparison)",
                    Some(0.0),
                    Some(config.regression_threshold),
                    Duration::from_millis(0),
                )
            }
        }
    };
    if !config.json {
        print_gate_result(&regression_result);
    }
    gates.push(regression_result);
    Ok(())
}

/// Finalize the QA report: count gates, check thresholds, print summary.
fn finalize_qa_report(
    path: &Path,
    start: &Instant,
    gates: Vec<GateResult>,
    config: &QaConfig,
) -> Result<QaReport> {
    let total_duration = start.elapsed();
    let gates_executed = gates.iter().filter(|g| !g.skipped).count();
    let gates_skipped = gates.iter().filter(|g| g.skipped).count();

    warn_excessive_skips(config.json, gates_executed, gates_skipped);

    let mut passed = gates.iter().all(|g| g.passed);
    if !check_min_executed(config, gates_executed, &mut passed) && !config.json {
        println!(
            "  {} Only {} gates executed, minimum required: {}",
            "FAIL".red().bold(),
            gates_executed,
            config.min_executed.unwrap_or(0),
        );
    }

    let summary = build_qa_summary(&gates, passed, gates_executed, gates_skipped, config);
    if !config.json {
        print_qa_summary(&gates, passed, total_duration);
    }

    Ok(QaReport {
        model: path.display().to_string(),
        passed,
        gates,
        gates_executed,
        gates_skipped,
        total_duration_ms: total_duration.as_millis() as u64,
        timestamp: chrono::Utc::now().to_rfc3339(),
        summary,
        system_info: Some(SystemInfo::capture()),
    })
}

/// JIDOKA: Warn when >50% gates are skipped.
fn warn_excessive_skips(json: bool, executed: usize, skipped: usize) {
    if !json && skipped > executed {
        println!(
            "  {} {} of {} gates SKIPPED — QA not rigorous",
            "WARN".yellow().bold(),
            skipped,
            skipped + executed
        );
    }
}

/// Check --min-executed constraint. Returns true if constraint is satisfied.
fn check_min_executed(config: &QaConfig, gates_executed: usize, passed: &mut bool) -> bool {
    let Some(min) = config.min_executed else {
        return true;
    };
    if gates_executed >= min {
        return true;
    }
    *passed = false;
    false
}

/// Build the QA summary message.
fn build_qa_summary(
    gates: &[GateResult],
    passed: bool,
    gates_executed: usize,
    gates_skipped: usize,
    config: &QaConfig,
) -> String {
    if passed {
        return format!(
            "All QA gates passed ({} executed, {} skipped)",
            gates_executed, gates_skipped
        );
    }

    let names: Vec<_> = gates
        .iter()
        .filter(|g| !g.passed && !g.skipped)
        .map(|g| g.name.as_str())
        .collect();

    if names.is_empty() {
        format!(
            "Insufficient gate execution: {} < {} minimum",
            gates_executed,
            config.min_executed.unwrap_or(0)
        )
    } else {
        format!("Failed gates: {}", names.join(", "))
    }
}

/// Gate 0: Tensor Contract Validation (PMAT-235)
///
/// Validates model tensors against the PMAT-235 data quality contract BEFORE
/// running any inference. This catches bad models early (density, NaN/Inf,
/// degenerate distributions) without expensive forward passes.
///
/// Toyota Way: Jidoka - Stop the line before producing defective output.
/// Poka-Yoke: Invalid tensor data is rejected before it can cause garbage inference.
fn run_tensor_contract_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!(
            "{}",
            "Running tensor contract validation (PMAT-235)...".yellow()
        );
    }

    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let report = match rosetta.validate(path) {
        Ok(r) => r,
        Err(e) => {
            let duration = start.elapsed();
            return Ok(GateResult::failed(
                "tensor_contract",
                &format!("Failed to validate: {e}"),
                None,
                None,
                duration,
            ));
        }
    };

    let duration = start.elapsed();

    // Collect all contract violations (F-DATA-QUALITY-* rules)
    let contract_failures: Vec<String> = report
        .tensors
        .iter()
        .flat_map(|t| t.failures.iter().map(|f| format!("{}: {}", t.name, f)))
        .collect();

    if contract_failures.is_empty() {
        Ok(GateResult::passed(
            "tensor_contract",
            &format!(
                "{} tensors passed all PMAT-235 contract gates",
                report.tensor_count
            ),
            Some(report.tensor_count as f64),
            Some(0.0),
            duration,
        ))
    } else {
        let summary = if contract_failures.len() <= 3 {
            contract_failures.join("; ")
        } else {
            format!(
                "{}; ... and {} more",
                contract_failures[..3].join("; "),
                contract_failures.len() - 3
            )
        };
        Ok(GateResult::failed(
            "tensor_contract",
            &format!(
                "{} contract violations in {} tensors: {}",
                contract_failures.len(),
                report.failed_tensor_count,
                summary
            ),
            Some(contract_failures.len() as f64),
            Some(0.0),
            duration,
        ))
    }
}
