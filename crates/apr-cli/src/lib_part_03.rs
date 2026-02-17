
/// PMAT-237: Extract model file paths from an `ExtendedCommands` variant.
///
/// Helper for `extract_model_paths` — handles the 19 variants that moved to
/// `ExtendedCommands` (Chat, Bench, Eval, Profile, Qa, Parity, PtxMap, Ptx,
/// Tune, Cbtop, Probar, CompareHf, Hex, Tree, Flow, Showcase, Rosetta,
/// Publish, Oracle).
fn extract_extended_model_paths(command: &ExtendedCommands) -> Vec<PathBuf> {
    match command {
        // === ACTION COMMANDS (gated) ===
        ExtendedCommands::Probar { file, .. }
        | ExtendedCommands::CompareHf { file, .. }
        | ExtendedCommands::Chat { file, .. }
        | ExtendedCommands::Bench { file, .. }
        | ExtendedCommands::Eval { file, .. }
        | ExtendedCommands::Profile { file, .. } => vec![file.clone()],

        ExtendedCommands::Cbtop { model_path, .. } => model_path.iter().cloned().collect(),

        // Rosetta action subcommands
        ExtendedCommands::Rosetta { action } => match action {
            RosettaCommands::Convert { source, .. }
            | RosettaCommands::Chain { source, .. }
            | RosettaCommands::Verify { source, .. } => vec![source.clone()],
            RosettaCommands::CompareInference {
                model_a, model_b, ..
            } => {
                vec![model_a.clone(), model_b.clone()]
            }
            // Diagnostic rosetta commands — exempt
            _ => vec![],
        },

        // === DIAGNOSTIC COMMANDS (exempt) ===
        // Qa, Parity, PtxMap, Ptx, Tune, Hex, Tree, Flow, Showcase, Publish, Oracle
        _ => vec![],
    }
}

/// PMAT-237: Extract model file paths from a command variant.
///
/// Returns paths for action commands (run, serve, bench, etc.) that should be
/// validated against the tensor contract. Returns empty vec for diagnostic
/// commands (qa, validate, inspect, debug, etc.) that must work on corrupt models.
fn extract_model_paths(command: &Commands) -> Vec<PathBuf> {
    match command {
        // === ACTION COMMANDS (gated) ===
        Commands::Run { source, .. } => {
            // Only validate local files, not hf:// or URLs
            let path = PathBuf::from(source);
            if path.exists() {
                vec![path]
            } else {
                vec![]
            }
        }
        Commands::Export { file, .. } => file.iter().cloned().collect(),
        Commands::Serve { file, .. }
        | Commands::Trace { file, .. }
        | Commands::Convert { file, .. }
        | Commands::Check { file, .. } => vec![file.clone()],

        Commands::Merge { files, .. } => files.clone(),

        Commands::Quantize { file, .. } | Commands::Prune { file, .. } => vec![file.clone()],
        Commands::Distill { teacher, .. } => vec![teacher.clone()],
        Commands::Finetune { file, .. } => file.iter().cloned().collect(),

        Commands::Tui { file, .. } => file.iter().cloned().collect(),
        Commands::Import { source, .. } => {
            let path = PathBuf::from(source);
            if path.exists() {
                vec![path]
            } else {
                vec![]
            }
        }

        // Delegate to ExtendedCommands helper
        Commands::Extended(ref ext) => extract_extended_model_paths(ext),

        // === DIAGNOSTIC COMMANDS (exempt) ===
        // validate, inspect, debug, tensors, diff, lint,
        // explain, list, rm, pull, canary
        _ => vec![],
    }
}

/// PMAT-237: Validate model files against tensor contract before dispatch.
///
/// Uses `RosettaStone::validate()` to check for NaN, Inf, all-zeros, density,
/// and other contract violations. Returns `CliError::ValidationFailed` (exit 5)
/// if any violations are found.
///
/// GH-213: For sharded SafeTensors models (index.json), validates shard integrity
/// via `.apr-manifest.json` checksums instead of RosettaStone (which can't parse
/// index files). This catches truncated downloads before inference.
fn validate_model_contract(paths: &[PathBuf]) -> Result<(), CliError> {
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    for path in paths {
        if !path.exists() {
            continue; // Let the subcommand handle FileNotFound
        }
        if path.to_string_lossy().ends_with(".safetensors.index.json") {
            validate_shard_index(path)?;
            continue;
        }
        // GH-238: Skip rosetta validation for non-native formats (ONNX, NeMo, etc.)
        // Rosetta only validates GGUF/APR/SafeTensors tensor layout contracts.
        // Import pipeline handles format-specific validation for these formats.
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if !matches!(ext.as_str(), "gguf" | "safetensors" | "apr") {
            continue;
        }
        validate_single_model(&rosetta, path)?;
    }
    Ok(())
}

/// GH-213: For sharded index.json, validate shard integrity via manifest.
fn validate_shard_index(path: &Path) -> Result<(), CliError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    let manifest_path = parent.join(".apr-manifest.json");
    if manifest_path.exists() {
        validate_shard_manifest(&manifest_path, parent)?;
    }
    Ok(())
}

/// Validate a single model file against the tensor layout contract.
fn validate_single_model(
    rosetta: &aprender::format::rosetta::RosettaStone,
    path: &Path,
) -> Result<(), CliError> {
    let report = rosetta.validate(path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Contract validation failed for {}: {e}",
            path.display()
        ))
    })?;
    if !report.is_valid {
        let violation_count: usize = report.tensors.iter().map(|t| t.failures.len()).sum();
        return Err(CliError::ValidationFailed(format!(
            "PMAT-237 CONTRACT VIOLATION: {} has {} violations in {} tensors. \
             Use 'apr qa {}' for details. Use --skip-contract to bypass.",
            path.display(),
            violation_count,
            report.failed_tensor_count,
            path.display(),
        )));
    }
    Ok(())
}

/// GH-213: Validate sharded model integrity by checking file sizes against manifest.
///
/// This is an O(1)-per-file check (stat syscall only, no hashing) that catches
/// truncated downloads before they cause cryptic "tensor not found" errors.
fn validate_shard_manifest(
    manifest_path: &std::path::Path,
    cache_dir: &std::path::Path,
) -> Result<(), CliError> {
    let manifest_str = std::fs::read_to_string(manifest_path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Failed to read manifest {}: {e}",
            manifest_path.display()
        ))
    })?;
    let manifest: commands::pull::ShardManifest =
        serde_json::from_str(&manifest_str).map_err(|e| {
            CliError::ValidationFailed(format!(
                "Failed to parse manifest {}: {e}",
                manifest_path.display()
            ))
        })?;

    for (filename, checksum) in &manifest.files {
        let file_path = cache_dir.join(filename);
        if !file_path.exists() {
            return Err(CliError::ValidationFailed(format!(
                "Shard '{}' is missing. Re-run 'apr pull --force' to re-download.",
                filename
            )));
        }
        let actual_size = std::fs::metadata(&file_path)
            .map(|m| m.len())
            .map_err(|e| {
                CliError::ValidationFailed(format!("Failed to stat shard '{}': {e}", filename))
            })?;
        if actual_size != checksum.size {
            return Err(CliError::ValidationFailed(format!(
                "Shard '{}' size mismatch: expected {} bytes, got {} bytes \
                 (file may be truncated). Re-run 'apr pull --force' to re-download.",
                filename, checksum.size, actual_size
            )));
        }
    }
    Ok(())
}

/// Dispatch `apr cbtop` — extracted to reduce cognitive complexity of `execute_command`
#[allow(clippy::too_many_arguments)]
fn dispatch_cbtop(
    model: Option<&str>,
    attach: Option<&str>,
    model_path: Option<&Path>,
    headless: bool,
    json: bool,
    output: Option<&Path>,
    ci: bool,
    throughput: Option<f64>,
    brick_score: Option<u32>,
    warmup: usize,
    iterations: usize,
    speculative: bool,
    speculation_k: usize,
    draft_model: Option<&Path>,
    concurrent: usize,
    simulated: bool,
) -> Result<(), CliError> {
    let (resolved_model, resolved_model_path) = if let Some(m) = model {
        let path = std::path::Path::new(m);
        let is_gguf = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
        if is_gguf || path.exists() {
            (
                Some(
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or(m)
                        .to_string(),
                ),
                Some(PathBuf::from(m)),
            )
        } else {
            (Some(m.to_string()), model_path.map(PathBuf::from))
        }
    } else {
        (None, model_path.map(PathBuf::from))
    };

    cbtop::run(cbtop::CbtopConfig {
        model: resolved_model,
        attach: attach.map(String::from),
        model_path: resolved_model_path,
        headless,
        json,
        output: output.map(PathBuf::from),
        ci,
        throughput_threshold: throughput,
        brick_score_threshold: brick_score,
        warmup,
        iterations,
        speculative,
        speculation_k,
        draft_model_path: draft_model.map(PathBuf::from),
        concurrent,
        simulated,
    })
}

/// Dispatch `apr showcase` — extracted to reduce cognitive complexity of `execute_command`
#[allow(clippy::too_many_arguments)]
fn dispatch_showcase(
    auto_verify: bool,
    step: Option<&str>,
    tier: &str,
    model_dir: &Path,
    baseline: &str,
    zram: bool,
    runs: usize,
    gpu: bool,
    json: bool,
    verbose: bool,
    quiet: bool,
) -> Result<(), CliError> {
    let step = step.and_then(|s| match s {
        "import" => Some(showcase::ShowcaseStep::Import),
        "gguf" => Some(showcase::ShowcaseStep::GgufInference),
        "convert" => Some(showcase::ShowcaseStep::Convert),
        "apr" => Some(showcase::ShowcaseStep::AprInference),
        "bench" => Some(showcase::ShowcaseStep::Benchmark),
        "chat" => Some(showcase::ShowcaseStep::Chat),
        "visualize" => Some(showcase::ShowcaseStep::Visualize),
        "zram" => Some(showcase::ShowcaseStep::ZramDemo),
        "cuda" => Some(showcase::ShowcaseStep::CudaDemo),
        "brick" => Some(showcase::ShowcaseStep::BrickDemo),
        "all" => Some(showcase::ShowcaseStep::All),
        _ => None,
    });

    let tier = match tier {
        "tiny" => showcase::ModelTier::Tiny,
        "small" => showcase::ModelTier::Small,
        "medium" => showcase::ModelTier::Medium,
        "large" => showcase::ModelTier::Large,
        _ => showcase::ModelTier::Small,
    };

    let baselines: Vec<showcase::Baseline> = baseline
        .split(',')
        .filter_map(|b| match b.trim() {
            "llama-cpp" => Some(showcase::Baseline::LlamaCpp),
            "ollama" => Some(showcase::Baseline::Ollama),
            _ => None,
        })
        .collect();

    let export_format = if json {
        showcase::ExportFormat::Json
    } else {
        showcase::ExportFormat::None
    };

    let config = showcase::ShowcaseConfig {
        tier,
        model: tier.model_path().to_string(),
        quant: "Q4_K_M".to_string(),
        model_dir: model_dir.to_path_buf(),
        auto_verify,
        step,
        baselines,
        zram,
        bench_runs: runs,
        export_format,
        export_path: None,
        gpu,
        verbose,
        quiet,
    };

    showcase::run(&config)
}

/// Dispatch `apr profile` — extracted to reduce cognitive complexity of `execute_command`
#[allow(clippy::too_many_arguments)]
fn dispatch_profile(
    file: &Path,
    granular: bool,
    format: &str,
    focus: Option<&str>,
    detect_naive: bool,
    threshold: f64,
    compare_hf: Option<&str>,
    energy: bool,
    perf_grade: bool,
    callgraph: bool,
    fail_on_naive: bool,
    output: Option<&Path>,
    ci: bool,
    assert_throughput: Option<f64>,
    assert_p99: Option<f64>,
    assert_p50: Option<f64>,
    warmup: usize,
    measure: usize,
    tokens: usize,
    ollama: bool,
    no_gpu: bool,
    compare: Option<&Path>,
) -> Result<(), CliError> {
    let output_format = format.parse().unwrap_or(profile::OutputFormat::Human);

    // PMAT-192: CI mode takes precedence
    if ci || assert_throughput.is_some() || assert_p99.is_some() || assert_p50.is_some() {
        let assertions = profile::CiAssertions {
            min_throughput: assert_throughput,
            max_p99_ms: assert_p99,
            max_p50_ms: assert_p50,
            max_memory_mb: None,
        };
        match profile::run_ci(file, output_format, &assertions, warmup, measure) {
            Ok(true) => Ok(()),
            Ok(false) => {
                std::process::exit(1);
            }
            Err(e) => Err(e),
        }
    } else if let Some(compare_path) = compare {
        // F-PROFILE-011: Cross-format performance comparison
        profile::run_cross_format_comparison(file, compare_path, warmup, measure, tokens, no_gpu)
    } else {
        let profile_focus = focus
            .and_then(|f| f.parse().ok())
            .unwrap_or(profile::ProfileFocus::All);
        profile::run(
            file,
            granular,
            output_format,
            profile_focus,
            detect_naive,
            threshold,
            compare_hf,
            energy,
            perf_grade,
            callgraph,
            fail_on_naive,
            output,
            tokens,
            ollama,
            no_gpu,
        )
    }
}
