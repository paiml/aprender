
/// Setup SIGINT handler for Jidoka cleanup
///
/// Toyota Way: Stop the line, clean up, never leave orphan processes.
fn setup_signal_handler() {
    if let Err(e) = ctrlc::set_handler(move || {
        let count = aprender_qa_runner::process::kill_all_registered();
        eprintln!("\n[JIDOKA] SIGINT received. Reaping {count} child process(es)...");
        eprintln!("[JIDOKA] Toyota Way: Stop the line, clean up, exit.");
        std::process::exit(130); // 128 + SIGINT(2)
    }) {
        eprintln!("Warning: Failed to set signal handler: {e}");
    }
}

/// Print playbook execution status including model, workers, and test configuration
#[allow(clippy::fn_params_excessive_bools)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn print_run_status(
    playbook: &aprender_qa_runner::Playbook,
    effective_workers: usize,
    model_path: Option<&str>,
    dry_run: bool,
    timeout: u64,
    skip_conversion_tests: bool,
    hf_parity: bool,
    hf_corpus_path: &str,
    hf_model_family: Option<&str>,
) {
    println!(
        "{} {}",
        "Running playbook:".bold(),
        playbook.name.bold().cyan()
    );
    println!("  {} {}", "Total tests:".dimmed(), playbook.total_tests());
    println!("  {} {dry_run}", "Dry run:".dimmed());
    println!(
        "  {} {:?}",
        "Model size:".dimmed(),
        playbook.size_category()
    );
    if let Some(path) = model_path {
        println!("  {} {path}", "Model path:".dimmed());
    }
    println!(
        "  {} {} (max for size: {})",
        "Workers:".dimmed(),
        effective_workers,
        playbook.model.size_category.max_workers()
    );
    println!("  {} {timeout}ms", "Timeout:".dimmed());

    // Conversion test status (P0 CRITICAL)
    if !skip_conversion_tests && model_path.is_some() {
        println!(
            "  {} {}",
            "Conversion tests:".dimmed(),
            "ENABLED (P0 CRITICAL)".bold().green()
        );
    } else if skip_conversion_tests {
        println!(
            "  {} {}",
            "Conversion tests:".dimmed(),
            "DISABLED (WARNING: P0 tests skipped)".bold().yellow()
        );
    }

    // HF parity status
    if hf_parity {
        println!("  {} {}", "HF parity:".dimmed(), "ENABLED".green());
        println!("    {} {hf_corpus_path}", "Corpus:".dimmed());
        if let Some(family) = hf_model_family {
            println!("    {} {family}", "Model family:".dimmed());
        } else {
            println!(
                "    {} {}",
                "Model family:".dimmed(),
                "NOT SET (required for parity tests)".yellow()
            );
        }
    }
}

/// Load, validate, and execute a playbook with the given configuration
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
fn run_playbook(
    playbook_path: &PathBuf,
    output_dir: &PathBuf,
    failure_policy: &str,
    dry_run: bool,
    workers: usize,
    model_path: Option<String>,
    timeout: u64,
    no_gpu: bool,
    skip_conversion_tests: bool,
    run_tool_tests_flag: bool,
    profile_ci: bool,
    hf_parity: bool,
    hf_corpus_path: &str,
    hf_model_family: Option<String>,
    no_integrity_check: bool,
    metadata_only: bool,
) {
    if failure_policy == "fail-fast" {
        log_environment();
    }

    let playbook = load_playbook_or_exit(playbook_path);

    if !no_integrity_check {
        verify_playbook_lock_or_exit(playbook_path, &playbook.name);
    }

    validate_failure_policy_or_exit(failure_policy);

    let effective_workers = resolve_effective_workers(&playbook, workers);

    print_run_status(
        &playbook,
        effective_workers,
        model_path.as_deref(),
        dry_run,
        timeout,
        skip_conversion_tests,
        hf_parity,
        hf_corpus_path,
        hf_model_family.as_deref(),
    );

    let run_config = build_run_config(
        failure_policy,
        dry_run,
        effective_workers,
        model_path.clone(),
        timeout,
        no_gpu,
        skip_conversion_tests,
        run_tool_tests_flag,
        profile_ci,
        hf_parity,
        hf_corpus_path,
        hf_model_family,
        metadata_only,
    );

    let config = match build_execution_config(&run_config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    if run_tool_tests_flag && !dry_run {
        run_tool_tests_or_exit(model_path.as_deref(), no_gpu, timeout);
    }

    let result = match execute_playbook(&playbook, config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    print_playbook_results(&result);
    persist_and_exit(dry_run, &result, output_dir);
}

fn load_playbook_or_exit(playbook_path: &PathBuf) -> aprender_qa_runner::Playbook {
    println!(
        "{} {}",
        "Loading playbook:".bold().cyan(),
        playbook_path.display().to_string().cyan()
    );
    match load_playbook(playbook_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e.red());
            std::process::exit(1);
        }
    }
}

fn validate_failure_policy_or_exit(failure_policy: &str) {
    if parse_failure_policy(failure_policy).is_err() {
        eprintln!("Unknown failure policy: {failure_policy}");
        std::process::exit(1);
    }
}

/// §3.4: Resource-aware scheduling — enforce worker limits based on model size.
fn resolve_effective_workers(
    playbook: &aprender_qa_runner::Playbook,
    requested: usize,
) -> usize {
    let effective = playbook.effective_max_workers(requested);
    if effective < requested {
        eprintln!(
            "{} Model size {:?} caps workers at {} (requested {})",
            "[RESOURCE]".yellow(),
            playbook.size_category(),
            effective,
            requested
        );
    }
    effective
}

#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
fn build_run_config(
    failure_policy: &str,
    dry_run: bool,
    workers: usize,
    model_path: Option<String>,
    timeout: u64,
    no_gpu: bool,
    skip_conversion_tests: bool,
    run_tool_tests: bool,
    run_profile_ci: bool,
    hf_parity: bool,
    hf_corpus_path: &str,
    hf_model_family: Option<String>,
    metadata_only: bool,
) -> PlaybookRunConfig {
    PlaybookRunConfig {
        failure_policy: failure_policy.to_string(),
        dry_run,
        workers,
        model_path,
        timeout,
        no_gpu,
        skip_conversion_tests,
        run_tool_tests,
        run_profile_ci,
        run_hf_parity: hf_parity,
        hf_parity_corpus_path: if hf_parity {
            Some(hf_corpus_path.to_string())
        } else {
            None
        },
        hf_parity_model_family: hf_model_family,
        metadata_only,
    }
}

fn run_tool_tests_or_exit(model_path: Option<&str>, no_gpu: bool, timeout: u64) {
    let Some(mp) = model_path else {
        eprintln!("Error: --run-tool-tests requires --model-path to be set");
        std::process::exit(1);
    };
    println!("\n{}", "=== Running APR Tool Tests ===".bold().cyan());
    let tool_executor = ToolExecutor::new(mp.to_string(), no_gpu, timeout);
    let tool_results = tool_executor.execute_all();
    let tool_passed = tool_results.iter().filter(|r| r.passed).count();
    let tool_failed = tool_results.len() - tool_passed;
    println!(
        "  Tool tests: {} passed, {} failed",
        tool_passed.to_string().green(),
        if tool_failed > 0 {
            tool_failed.to_string().red()
        } else {
            tool_failed.to_string().dimmed()
        }
    );
    if tool_failed > 0 {
        eprintln!("Aborting: {tool_failed} tool test(s) failed (Jidoka: stop the line)");
        std::process::exit(1);
    }
}

/// Persist evidence (unless dry-run) and exit non-zero on any failure.
///
/// Dry-run skips evidence persistence: output directory may not be writable and
/// the purpose of dry-run is inspection, not artifact production. Jidoka: stop
/// the line on any failed scenario or gateway blow.
fn persist_and_exit(
    dry_run: bool,
    result: &aprender_qa_runner::ExecutionResult,
    output_dir: &PathBuf,
) {
    if !dry_run && !save_playbook_evidence(result, output_dir) {
        eprintln!("Error: evidence save failed — results may be lost");
        std::process::exit(1);
    }

    if !dry_run && (result.failed > 0 || result.gateway_failed.is_some()) {
        std::process::exit(1);
    }
}
