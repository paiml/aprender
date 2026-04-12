
/// Print color-coded playbook execution results including pass rate and gateway status
fn print_playbook_results(result: &apr_qa_runner::ExecutionResult) {
    println!("\n{}", "=== Execution Results ===".bold().cyan());
    println!(
        "  {} {}",
        "Total scenarios:".dimmed(),
        result.total_scenarios
    );
    println!(
        "  {} {}",
        "Passed:".dimmed(),
        result.passed.to_string().bold().green()
    );
    println!(
        "  {} {}",
        "Failed:".dimmed(),
        if result.failed > 0 {
            result.failed.to_string().bold().red()
        } else {
            result.failed.to_string().dimmed()
        }
    );
    println!(
        "  {} {}",
        "Skipped:".dimmed(),
        if result.skipped > 0 {
            result.skipped.to_string().yellow()
        } else {
            result.skipped.to_string().dimmed()
        }
    );
    println!("  {} {}ms", "Duration:".dimmed(), result.duration_ms);
    let pass_rate = result.pass_rate();
    let rate_str = format!("{pass_rate:.1}%");
    let colored_rate = if pass_rate >= 90.0 {
        rate_str.green()
    } else if pass_rate >= 70.0 {
        rate_str.yellow()
    } else {
        rate_str.red()
    };
    println!("  {} {colored_rate}", "Pass rate:".dimmed());

    if let Some(ref gateway_fail) = result.gateway_failed {
        println!("  {} {gateway_fail}", "Gateway FAILED:".bold().red());
    }
}

/// Serialize and write execution evidence to a JSON file or directory.
///
/// Returns `true` on success, `false` on any I/O or serialization failure.
/// Callers should exit non-zero when this returns false to avoid silent data loss.
fn save_playbook_evidence(result: &apr_qa_runner::ExecutionResult, output_dir: &PathBuf) -> bool {
    // GH-212: If --output ends with .json, treat as file path, not directory
    let evidence_path = if output_dir
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    {
        let parent = output_dir.parent().unwrap_or_else(|| Path::new("."));
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("Error creating output directory: {e}");
            return false;
        }
        output_dir.clone()
    } else {
        if let Err(e) = std::fs::create_dir_all(output_dir) {
            eprintln!("Error creating output directory: {e}");
            return false;
        }
        output_dir.join("evidence.json")
    };
    match result.evidence.to_json() {
        Ok(json) => {
            if let Err(e) = std::fs::write(&evidence_path, json) {
                eprintln!("Error writing evidence: {e}");
                return false;
            }
            println!(
                "\n{} {}",
                "Evidence saved to:".green(),
                evidence_path.display().to_string().cyan()
            );
            true
        }
        Err(e) => {
            eprintln!("Error serializing evidence: {e}");
            false
        }
    }
}

/// Log environment information for fail-fast diagnostics (§12.5.3)
fn log_environment() {
    let tag = "[ENVIRONMENT]".dimmed().cyan();
    eprintln!("\n{tag} {}", "=== Diagnostic Context ===".dimmed());
    eprintln!(
        "{tag} {} {} {}",
        "OS:".dimmed(),
        std::env::consts::OS.dimmed(),
        std::env::consts::ARCH.dimmed()
    );
    eprintln!(
        "{tag} {} {}",
        "apr-qa version:".dimmed(),
        env!("CARGO_PKG_VERSION").dimmed()
    );

    // Git context
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        if output.status.success() {
            let commit = String::from_utf8_lossy(&output.stdout);
            eprintln!(
                "{tag} {} {}",
                "Git commit:".dimmed(),
                commit.trim().dimmed()
            );
        }
    }

    if let Ok(output) = std::process::Command::new("git")
        .args(["branch", "--show-current"])
        .output()
    {
        if output.status.success() {
            let branch = String::from_utf8_lossy(&output.stdout);
            eprintln!(
                "{tag} {} {}",
                "Git branch:".dimmed(),
                branch.trim().dimmed()
            );
        }
    }

    // Check for dirty files
    if let Ok(output) = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
    {
        if output.status.success() {
            let status = String::from_utf8_lossy(&output.stdout);
            let dirty_count = status.lines().count();
            if dirty_count > 0 {
                eprintln!(
                    "{tag} {} {}",
                    "Git dirty:".dimmed(),
                    format!("{dirty_count} file(s) modified").dimmed()
                );
            }
        }
    }

    // apr CLI version
    if let Ok(output) = std::process::Command::new("apr").arg("--version").output() {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            eprintln!("{tag} {} {}", "apr-cli:".dimmed(), version.trim().dimmed());
        }
    }

    // Rust version
    if let Ok(output) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            eprintln!("{tag} {}", version.trim().dimmed());
        }
    }

    eprintln!("{tag} {}\n", "===========================".dimmed());
}

/// Generate QA scenarios for a model and print them in the requested format
fn generate_scenarios(model_id: &str, count: usize, format: &str) {
    // Validate format before expensive generation
    if !matches!(format, "yaml" | "json") {
        eprintln!("Unknown format: {format}");
        std::process::exit(1);
    }

    if count == 0 {
        eprintln!("Error: --count must be at least 1");
        std::process::exit(1);
    }

    let scenarios = generate_model_scenarios(model_id, count);

    eprintln!("Generated {} scenarios for {model_id}", scenarios.len());

    match format {
        "yaml" => match scenarios_to_yaml(&scenarios) {
            Ok(yaml) => println!("{yaml}"),
            Err(e) => {
                eprintln!("Error serializing scenarios to YAML: {e}");
                std::process::exit(1);
            }
        },
        "json" => match scenarios_to_json(&scenarios) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("Error serializing scenarios to JSON: {e}");
                std::process::exit(1);
            }
        },
        _ => unreachable!(),
    }
}

/// Load evidence from file and compute the full MQS score breakdown
fn calculate_score(evidence_path: &PathBuf, model_id: &str) {
    let evidence_json = match std::fs::read_to_string(evidence_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading evidence file: {e}");
            std::process::exit(1);
        }
    };

    let evidence = match parse_evidence(&evidence_json) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    if evidence.is_empty() {
        eprintln!("Error: Evidence file contains no test results");
        eprintln!("  Popperian: untested hypotheses cannot earn a score");
        std::process::exit(1);
    }

    let collector = collect_evidence(evidence);

    match calculate_mqs_score(model_id, &collector) {
        Ok(score) => {
            println!("=== Model Qualification Score (MQS) ===");
            println!("Model: {}", score.model_id);
            println!("Raw Score: {}/1000", score.raw_score);
            println!("Normalized Score: {:.1}/100", score.normalized_score);
            println!("Grade: {}", score.grade);
            println!("Gateways Passed: {}", score.gateways_passed);
            println!("Qualifies: {}", score.qualifies());
            println!("Production Ready: {}", score.is_production_ready());

            println!("\n--- Category Breakdown ---");
            let breakdown = score.categories.breakdown();
            for (cat, (pts, max)) in &breakdown {
                println!("  {cat}: {pts}/{max}");
            }

            if !score.penalties.is_empty() {
                println!("\n--- Penalties ---");
                for penalty in &score.penalties {
                    println!(
                        "  {}: {} (-{} pts)",
                        penalty.code, penalty.description, penalty.points
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

/// Generate HTML and/or JUnit reports from evidence and write them to the output directory
fn generate_report(evidence_path: &PathBuf, output_dir: &PathBuf, formats: &str, model_id: &str) {
    // Parse comma-separated formats (e.g., "html,junit" or "all")
    let format_list: Vec<&str> = formats.split(',').map(str::trim).collect();
    let valid_formats = ["all", "html", "junit", "markdown"];
    for f in &format_list {
        if !valid_formats.contains(f) {
            eprintln!("Error: Unknown report format: {f}");
            eprintln!("  Valid formats: all, html, junit, markdown");
            std::process::exit(1);
        }
    }

    let evidence_json = read_file_or_exit(evidence_path, "evidence file");
    let evidence = parse_evidence_or_exit(&evidence_json);
    let collector = collect_evidence(evidence);
    let mqs_score = calculate_mqs_or_exit(model_id, &collector);
    let popperian_score = calculate_popperian_score(model_id, &collector);

    create_dir_or_exit(output_dir);
    if !write_report_formats(
        output_dir,
        &format_list,
        model_id,
        &mqs_score,
        &popperian_score,
        &collector,
    ) {
        eprintln!("Error: one or more report files failed to write");
        std::process::exit(1);
    }
}

/// Read a file to string or exit with an error message
fn read_file_or_exit(path: &PathBuf, desc: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading {desc}: {e}");
        std::process::exit(1);
    })
}

/// Parse evidence JSON or exit with an error message
fn parse_evidence_or_exit(json: &str) -> Vec<Evidence> {
    parse_evidence(json).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    })
}

/// Calculate MQS score or exit with an error message
fn calculate_mqs_or_exit(model_id: &str, collector: &EvidenceCollector) -> MqsScore {
    calculate_mqs_score(model_id, collector).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    })
}

/// Create directory tree or exit with an error message
fn create_dir_or_exit(dir: &PathBuf) {
    // Detect broken symlinks early — create_dir_all returns EEXIST on broken symlinks
    // without a useful message (rust-lang/rust#86442 wontfix).
    if let Ok(target) = std::fs::read_link(dir) {
        if !target.exists() {
            eprintln!(
                "Error: output path '{}' is a broken symlink → '{}'",
                dir.display(),
                target.display()
            );
            eprintln!("  Fix: remove the symlink or point it to an existing directory.");
            std::process::exit(1);
        }
    }
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("Error creating output directory: {e}");
        std::process::exit(1);
    }
}

/// Dispatch report generation to the requested format writers (HTML, JUnit, Markdown, MQS JSON)
///
/// Returns false if any write fails. Callers should exit non-zero on false.
fn write_report_formats(
    output_dir: &PathBuf,
    formats: &[&str],
    model_id: &str,
    mqs_score: &MqsScore,
    popperian_score: &PopperianScore,
    collector: &EvidenceCollector,
) -> bool {
    let is_all = formats.contains(&"all");
    let gen_html = is_all || formats.contains(&"html");
    let gen_junit = is_all || formats.contains(&"junit");
    let gen_markdown = is_all || formats.contains(&"markdown");
    let mut ok = true;

    if gen_html {
        ok &= write_html_report(output_dir, model_id, mqs_score, popperian_score, collector);
    }
    if gen_junit {
        ok &= write_junit_report(output_dir, model_id, collector, mqs_score);
    }
    if gen_markdown {
        ok &= write_markdown_report(output_dir, mqs_score, popperian_score, collector);
    }
    ok &= write_mqs_json(output_dir, mqs_score);
    ok
}

/// Generate and write the HTML report file
fn write_html_report(
    output_dir: &PathBuf,
    model_id: &str,
    mqs_score: &MqsScore,
    popperian_score: &PopperianScore,
    collector: &EvidenceCollector,
) -> bool {
    let result = generate_html_report(
        &format!("MQS Report: {model_id}"),
        mqs_score,
        popperian_score,
        collector,
    );
    write_report_file(output_dir, "report.html", "HTML report", result)
}

/// Generate and write the RAG-optimized markdown report file
fn write_markdown_report(
    output_dir: &PathBuf,
    mqs_score: &MqsScore,
    popperian_score: &PopperianScore,
    collector: &EvidenceCollector,
) -> bool {
    let markdown = apr_qa_report::markdown::generate_rag_markdown(mqs_score, popperian_score, collector);
    let path = output_dir.join("report.md");
    match std::fs::write(&path, markdown) {
        Ok(()) => {
            println!("Markdown report: {}", path.display());
            true
        }
        Err(e) => {
            eprintln!("Error writing markdown report: {e}");
            false
        }
    }
}

/// Generate and write the JUnit XML report file
fn write_junit_report(
    output_dir: &PathBuf,
    model_id: &str,
    collector: &EvidenceCollector,
    mqs_score: &MqsScore,
) -> bool {
    let result = generate_junit_report(model_id, collector, mqs_score);
    write_report_file(output_dir, "junit.xml", "JUnit report", result)
}

/// Write a generated report string to the output directory, handling errors
fn write_report_file<E: std::fmt::Display>(
    output_dir: &PathBuf,
    filename: &str,
    desc: &str,
    result: Result<String, E>,
) -> bool {
    match result {
        Ok(content) => {
            let path = output_dir.join(filename);
            match std::fs::write(&path, content) {
                Ok(()) => {
                    println!("{desc}: {}", path.display());
                    true
                }
                Err(e) => {
                    eprintln!("Error writing {desc}: {e}");
                    false
                }
            }
        }
        Err(e) => {
            eprintln!("{e}");
            false
        }
    }
}

/// Serialize and write the MQS score as pretty-printed JSON
fn write_mqs_json(output_dir: &PathBuf, mqs_score: &MqsScore) -> bool {
    let score_path = output_dir.join("mqs.json");
    match serde_json::to_string_pretty(mqs_score) {
        Ok(json) => match std::fs::write(&score_path, json) {
            Ok(()) => {
                println!("MQS score: {}", score_path.display());
                true
            }
            Err(e) => {
                eprintln!("Error writing MQS JSON: {e}");
                false
            }
        },
        Err(e) => {
            eprintln!("Error serializing MQS: {e}");
            false
        }
    }
}

/// List all available models, optionally filtered by size category
fn list_models(size_filter: Option<&str>) {
    let models = list_all_models();

    println!("=== Available Models ===\n");

    let filtered_models = if let Some(filter) = size_filter {
        let filtered = filter_models_by_size(&models, filter);
        if filtered.is_empty() {
            eprintln!("Error: No models matched size filter '{filter}'.");
            eprintln!("  Valid sizes: tiny, small, medium, large, xlarge, huge");
            std::process::exit(1);
        }
        filtered
    } else {
        models
    };

    for model in &filtered_models {
        println!("  {} ({:?})", model.id.hf_repo(), model.size);
    }

    println!("\n  Total: {} models", filtered_models.len());
}
