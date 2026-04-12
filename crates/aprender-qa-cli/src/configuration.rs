
/// Setup SIGINT handler for Jidoka cleanup
///
/// Toyota Way: Stop the line, clean up, never leave orphan processes.
fn setup_signal_handler() {
    if let Err(e) = ctrlc::set_handler(move || {
        let count = apr_qa_runner::process::kill_all_registered();
        eprintln!("\n[JIDOKA] SIGINT received. Reaping {count} child process(es)...");
        eprintln!("[JIDOKA] Toyota Way: Stop the line, clean up, exit.");
        std::process::exit(130); // 128 + SIGINT(2)
    }) {
        eprintln!("Warning: Failed to set signal handler: {e}");
    }
}

/// Entry point that dispatches CLI subcommands to their handlers
#[allow(clippy::too_many_lines)]
fn main() {
    setup_signal_handler();

    let cli = Cli::parse();

    match cli.command {
        Commands::Certify {
            all,
            family,
            tier,
            kernel_class,
            models,
            output,
            dry_run,
            model_cache,
            apr_binary,
            auto_ticket,
            ticket_repo,
            no_integrity_check,
            fail_fast,
            oracle_enhance,
        } => {
            run_certification(
                all,
                family,
                &tier,
                kernel_class,
                &models,
                &output,
                dry_run,
                model_cache,
                &apr_binary,
                auto_ticket,
                &ticket_repo,
                no_integrity_check,
                fail_fast,
                oracle_enhance,
            );
        }
        Commands::Run {
            playbook,
            output,
            failure_policy,
            fail_fast,
            dry_run,
            workers,
            model_path,
            timeout,
            no_gpu,
            skip_conversion_tests,
            run_tool_tests,
            profile_ci,
            hf_parity,
            hf_corpus_path,
            hf_model_family,
            no_integrity_check,
            metadata_only,
        } => {
            // --fail-fast flag overrides --failure-policy
            let effective_policy = if fail_fast {
                "fail-fast".to_string()
            } else {
                failure_policy
            };
            run_playbook(
                &playbook,
                &output,
                &effective_policy,
                dry_run,
                workers,
                model_path,
                timeout,
                no_gpu,
                skip_conversion_tests,
                run_tool_tests,
                profile_ci,
                hf_parity,
                &hf_corpus_path,
                hf_model_family,
                no_integrity_check,
                metadata_only,
            );
        }
        Commands::Tools {
            model_path,
            no_gpu,
            output,
            include_serve,
        } => {
            run_tool_tests(&model_path, no_gpu, &output, include_serve);
        }
        Commands::Generate {
            model,
            count,
            format,
        } => {
            generate_scenarios(&model, count, &format);
        }
        Commands::Score { evidence, model } => {
            calculate_score(&evidence, &model);
        }
        Commands::Report {
            evidence,
            output,
            formats,
            model,
        } => {
            generate_report(&evidence, &output, &formats, &model);
        }
        Commands::List { size } => {
            list_models(size.as_deref());
        }
        Commands::LockPlaybooks { dir, output } => match generate_lock_file(&dir, &output) {
            Ok(0) => {
                eprintln!("Error: No playbook files found in {}", dir.display());
                std::process::exit(1);
            }
            Ok(count) => println!("Locked {count} playbook(s) → {}", output.display()),
            Err(e) => {
                eprintln!("Error generating lock file: {e}");
                std::process::exit(1);
            }
        },
        Commands::Tickets {
            evidence,
            repo,
            black_swans_only,
            min_occurrences,
            ticket_mode,
        } => {
            generate_tickets(
                &evidence,
                &repo,
                black_swans_only,
                min_occurrences,
                &ticket_mode,
            );
        }
        Commands::Parity {
            model_family,
            corpus_path,
            logits_file,
            prompt,
            tolerance,
            list,
            self_check,
        } => {
            run_parity_check(
                &model_family,
                &corpus_path,
                logits_file.as_deref(),
                prompt.as_deref(),
                &tolerance,
                list,
                self_check,
            );
        }
        Commands::ExportCsv {
            evidence_dir,
            output,
            append,
        } => {
            export_csv(&evidence_dir, &output, append);
        }
        Commands::ExportEvidence {
            source,
            output_dir,
            model,
            family,
            size,
            playbook_name,
            tier,
        } => {
            export_evidence(
                &source,
                &output_dir,
                &model,
                &family,
                &size,
                &playbook_name,
                &tier,
            );
        }
        Commands::Bootstrap {
            family,
            size,
            hf_repo,
            tier,
            output,
            contracts_path,
            dry_run,
        } => {
            run_bootstrap(
                &family,
                &size,
                &hf_repo,
                &tier,
                output.as_deref(),
                &contracts_path,
                dry_run,
            );
        }
        Commands::ValidateContract {
            model_path,
            contract_path,
            format,
            critical_only,
        } => {
            validate_contract_command(
                &model_path,
                contract_path.as_deref(),
                &format,
                critical_only,
            );
        }
        Commands::KernelCoverage {
            architecture,
            all,
            models,
            verify,
            file_tickets,
            output_dir,
            trueno_path,
            realizar_path,
            format,
            contracts_path,
            bindings_path,
        } => {
            kernel_coverage_command(
                architecture.as_deref(),
                all,
                models,
                verify,
                file_tickets,
                &output_dir,
                &trueno_path,
                &realizar_path,
                &format,
                &contracts_path,
                &bindings_path,
            );
        }
    }
}

/// Print playbook execution status including model, workers, and test configuration
#[allow(clippy::fn_params_excessive_bools)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn print_run_status(
    playbook: &apr_qa_runner::Playbook,
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
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_lines)]
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

    println!(
        "{} {}",
        "Loading playbook:".bold().cyan(),
        playbook_path.display().to_string().cyan()
    );

    let playbook = match load_playbook(playbook_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e.red());
            std::process::exit(1);
        }
    };

    if !no_integrity_check {
        verify_playbook_lock_or_exit(playbook_path, &playbook.name);
    }

    if parse_failure_policy(failure_policy).is_err() {
        eprintln!("Unknown failure policy: {failure_policy}");
        std::process::exit(1);
    }

    // §3.4: Resource-aware scheduling - enforce worker limits based on model size
    let effective_workers = playbook.effective_max_workers(workers);
    if effective_workers < workers {
        eprintln!(
            "{} Model size {:?} caps workers at {} (requested {})",
            "[RESOURCE]".yellow(),
            playbook.size_category(),
            effective_workers,
            workers
        );
    }

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

    let run_config = PlaybookRunConfig {
        failure_policy: failure_policy.to_string(),
        dry_run,
        workers: effective_workers,
        model_path: model_path.clone(),
        timeout,
        no_gpu,
        skip_conversion_tests,
        run_tool_tests: run_tool_tests_flag,
        run_profile_ci: profile_ci,
        run_hf_parity: hf_parity,
        hf_parity_corpus_path: if hf_parity {
            Some(hf_corpus_path.to_string())
        } else {
            None
        },
        hf_parity_model_family: hf_model_family,
        metadata_only,
    };

    let config = match build_execution_config(&run_config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // Run tool tests if enabled (skip during dry-run)
    if run_tool_tests_flag && !dry_run {
        let Some(ref mp) = model_path else {
            eprintln!("Error: --run-tool-tests requires --model-path to be set");
            std::process::exit(1);
        };
        println!("\n{}", "=== Running APR Tool Tests ===".bold().cyan());
        let tool_executor = ToolExecutor::new(mp.clone(), no_gpu, timeout);
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

    let result = match execute_playbook(&playbook, config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    print_playbook_results(&result);

    // Dry-run: skip evidence persistence — output directory may not be writable
    // and the purpose of dry-run is inspection, not artifact production.
    if !dry_run && !save_playbook_evidence(&result, output_dir) {
        eprintln!("Error: evidence save failed — results may be lost");
        std::process::exit(1);
    }

    // Jidoka: non-zero exit on any failure or gateway blow (not in dry-run)
    if !dry_run && (result.failed > 0 || result.gateway_failed.is_some()) {
        std::process::exit(1);
    }
}
