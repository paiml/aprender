
/// Generate GitHub tickets from evidence, in draft or create mode
fn generate_tickets(
    evidence_path: &PathBuf,
    repo: &str,
    black_swans_only: bool,
    min_occurrences: usize,
    ticket_mode: &str,
) {
    validate_ticket_mode_or_exit(ticket_mode);
    let is_draft = ticket_mode == "draft";

    let evidence_json = read_file_or_exit(evidence_path, "evidence file");
    let evidence = parse_evidence_or_exit(&evidence_json);
    let tickets =
        generate_tickets_from_evidence(&evidence, repo, black_swans_only, min_occurrences);

    if tickets.is_empty() {
        eprintln!("No tickets generated from evidence — nothing to file");
        eprintln!("  Popperian: ticket generation that produces nothing is vacuous");
        std::process::exit(1);
    }

    if is_draft {
        print_ticket_drafts(&tickets, repo);
    } else {
        create_tickets_or_exit(&tickets, repo);
    }
}

fn validate_ticket_mode_or_exit(ticket_mode: &str) {
    if !matches!(ticket_mode, "draft" | "create") {
        eprintln!("Error: Unknown ticket mode: {ticket_mode}");
        eprintln!("  Valid modes: draft, create");
        std::process::exit(1);
    }
}

/// F-TICKET-004: Draft mode — only print, don't create files.
fn print_ticket_drafts(tickets: &[aprender_qa_report::ticket::UpstreamTicket], repo: &str) {
    println!("=== Ticket Drafts ({}) ===", tickets.len());
    println!("(Draft mode: No files created)\n");

    for ticket in tickets {
        println!("--- {} ---", ticket.title);
        println!("Priority: {}", ticket.priority);
        println!("Category: {}", ticket.category);
        println!("Labels: {}", ticket.labels.join(", "));
        println!();
        println!("Body:");
        println!("{}", ticket.body);
        println!();
        println!("gh command (would run):");
        println!("  {}\n", ticket.to_gh_command(repo));
        println!("{}", "=".repeat(60));
    }
}

/// Create mode — actually run gh commands to file issues.
fn create_tickets_or_exit(tickets: &[aprender_qa_report::ticket::UpstreamTicket], repo: &str) {
    println!("=== Creating Tickets ({}) ===\n", tickets.len());
    let mut created = 0usize;
    let mut failed = 0usize;

    for ticket in tickets {
        println!("--- {} ---", ticket.title);
        let gh_cmd = ticket.to_gh_command(repo);
        println!("  Running: {gh_cmd}");
        if create_single_ticket(ticket, repo) {
            created += 1;
        } else {
            failed += 1;
        }
    }

    println!("\nCreated: {created}, Failed: {failed}");
    if failed > 0 {
        std::process::exit(1);
    }
}

/// Run `gh issue create` for a single ticket; return true on success.
fn create_single_ticket(ticket: &aprender_qa_report::ticket::UpstreamTicket, repo: &str) -> bool {
    let mut args = vec![
        "issue", "create",
        "--repo", repo,
        "--title", &ticket.title,
        "--body", &ticket.body,
    ];
    for label in &ticket.labels {
        args.push("--label");
        args.push(label);
    }
    match std::process::Command::new("gh").args(&args).output() {
        Ok(output) if output.status.success() => {
            let url = String::from_utf8_lossy(&output.stdout);
            println!("  Created: {}", url.trim());
            true
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("  Failed: {}", stderr.trim());
            false
        }
        Err(e) => {
            eprintln!("  Failed to run gh: {e}");
            false
        }
    }
}

/// Run HF Parity Oracle verification
///
/// Implements Popperian falsification: any divergence beyond tolerance
/// falsifies the hypothesis that the implementation is equivalent to HuggingFace.
#[allow(clippy::fn_params_excessive_bools)]
#[allow(clippy::too_many_lines)]
fn run_parity_check(
    model_family: &str,
    corpus_path: &std::path::Path,
    logits_file: Option<&std::path::Path>,
    prompt: Option<&str>,
    tolerance_str: &str,
    list: bool,
    self_check: bool,
) {
    use aprender_qa_gen::{HfParityOracle, Tolerance};

    println!("=== HuggingFace Parity Oracle ===\n");
    println!("Model family: {model_family}");
    println!("Corpus path: {}", corpus_path.display());

    // Parse tolerance
    let tolerance = match tolerance_str.to_lowercase().as_str() {
        "fp32" => Tolerance::fp32(),
        "fp16" => Tolerance::fp16(),
        "int8" => Tolerance::int8(),
        "int4" => Tolerance::int4(),
        _ => {
            eprintln!("Unknown tolerance level: {tolerance_str}");
            eprintln!("Valid options: fp32, fp16, int8, int4");
            std::process::exit(1);
        }
    };
    println!("Tolerance: {tolerance_str}");

    // Create oracle
    let oracle = HfParityOracle::new(corpus_path, model_family).with_tolerance(tolerance);

    // Check corpus exists
    let corpus_dir = corpus_path.join(model_family);
    if !corpus_dir.exists() {
        eprintln!(
            "\nError: Corpus directory not found: {}",
            corpus_dir.display()
        );
        eprintln!("Available models:");
        if let Ok(entries) = std::fs::read_dir(corpus_path) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    eprintln!("  - {}", entry.file_name().to_string_lossy());
                }
            }
        }
        std::process::exit(1);
    }

    if list {
        parity_list_golden(&corpus_dir);
        return;
    }

    if self_check {
        parity_self_check(&oracle, &corpus_dir);
        return;
    }

    parity_verify(&oracle, logits_file, prompt, tolerance_str);
}

/// List available golden outputs in the corpus directory
fn parity_list_golden(corpus_dir: &std::path::Path) {
    println!("\n=== Available Golden Outputs ===\n");
    let manifest_path = corpus_dir.join("manifest.json");
    if !manifest_path.exists() {
        eprintln!("No manifest.json found in {}", corpus_dir.display());
        std::process::exit(1);
    }
    let Ok(content) = std::fs::read_to_string(&manifest_path) else {
        eprintln!("Error reading manifest.json: {}", manifest_path.display());
        std::process::exit(1);
    };
    let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) else {
        eprintln!("Error parsing manifest.json: invalid JSON");
        std::process::exit(1);
    };
    let Some(prompts) = manifest.get("prompts").and_then(|p| p.as_array()) else {
        eprintln!("manifest.json missing 'prompts' array");
        std::process::exit(1);
    };
    println!("Found {} golden outputs:\n", prompts.len());

    for entry in std::fs::read_dir(corpus_dir)
        .into_iter()
        .flatten()
        .flatten()
    {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "json")
            || path.file_stem().is_none_or(|s| s == "manifest")
        {
            continue;
        }
        if let Some((hash, prompt_str)) = read_golden_prompt(&path) {
            let truncated = truncate_str(&prompt_str, 50);
            println!("  [{hash}] {truncated}");
        }
    }
}

/// Read the prompt from a golden output JSON file, returning (hash, prompt)
fn read_golden_prompt(path: &std::path::Path) -> Option<(String, String)> {
    let json = std::fs::read_to_string(path).ok()?;
    let meta: serde_json::Value = serde_json::from_str(&json).ok()?;
    let prompt = meta.get("prompt")?.as_str()?.to_string();
    let hash = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    Some((hash, prompt))
}

/// Truncate a string to max_len bytes, appending "..." if truncated.
/// UTF-8 safe: finds the last valid char boundary at or before max_len.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    // Find last char boundary at or before max_len
    let end = s
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= max_len)
        .last()
        .unwrap_or(0);
    format!("{}...", &s[..end])
}

/// Self-check mode: verify golden outputs match themselves
fn parity_self_check(oracle: &aprender_qa_gen::HfParityOracle, corpus_dir: &std::path::Path) {
    println!("\n=== Self-Check Mode ===");
    println!("Verifying golden outputs match themselves (sanity check)...\n");

    let mut passed = 0;
    let mut failed = 0;

    for entry in std::fs::read_dir(corpus_dir)
        .into_iter()
        .flatten()
        .flatten()
    {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "json")
            || path.file_stem().is_none_or(|s| s == "manifest")
        {
            continue;
        }
        let Some((_, prompt_str)) = read_golden_prompt(&path) else {
            continue;
        };
        match oracle.load_golden(&prompt_str) {
            Ok(golden) => match oracle.tensors_close(&golden.logits, &golden.logits) {
                Ok(()) => {
                    passed += 1;
                    println!("  ✓ {}", truncate_str(&prompt_str, 40));
                }
                Err(diff) => {
                    failed += 1;
                    eprintln!("  ✗ {prompt_str}: {diff}");
                }
            },
            Err(e) => {
                failed += 1;
                eprintln!("  ✗ Failed to load {prompt_str}: {e}");
            }
        }
    }

    println!("\n=== Self-Check Results ===");
    println!("Passed: {passed}");
    println!("Failed: {failed}");

    if passed == 0 && failed == 0 {
        eprintln!("Error: No golden files found — self-check is vacuous");
        eprintln!("  Popperian: a check that checks nothing cannot pass");
        std::process::exit(1);
    }

    if failed > 0 {
        std::process::exit(1);
    }
}

/// Verification mode: compare a logits file against golden reference
fn parity_verify(
    oracle: &aprender_qa_gen::HfParityOracle,
    logits_file: Option<&std::path::Path>,
    prompt: Option<&str>,
    tolerance_str: &str,
) {
    use aprender_qa_gen::hash_prompt;

    let Some(logits_path) = logits_file else {
        eprintln!("\nError: --logits-file is required for verification");
        eprintln!("Use --list to see available golden outputs");
        eprintln!("Use --self-check to verify corpus integrity");
        std::process::exit(1);
    };

    let Some(prompt_str) = prompt else {
        eprintln!("\nError: --prompt is required for verification");
        std::process::exit(1);
    };

    println!("\n=== Verification Mode ===");
    println!("Prompt: {prompt_str}");
    println!("Logits file: {}", logits_path.display());

    let logits = load_logits_from_file(logits_path);

    match oracle.load_golden(prompt_str) {
        Ok(golden) => {
            println!("\nGolden output found:");
            println!("  Model: {}", golden.model_id);
            println!("  Transformers version: {}", golden.transformers_version);
            println!("  Shape: {:?}", golden.shape);
            println!("  Input hash: {}", hash_prompt(prompt_str));
            println!(
                "\nComparing logits ({} vs {} elements)...",
                logits.len(),
                golden.logits.len()
            );

            match oracle.tensors_close(&logits, &golden.logits) {
                Ok(()) => {
                    println!("\n✓ PARITY VERIFIED");
                    println!("  Logits are within tolerance ({tolerance_str})");
                    println!("  Hypothesis corroborated: implementation matches HuggingFace");
                }
                Err(diff) => {
                    eprintln!("\n✗ PARITY FALSIFIED");
                    eprintln!("  {diff}");
                    eprintln!("\n  Interpretation (Popper, 1959):");
                    eprintln!("  The hypothesis that this implementation produces");
                    eprintln!("  equivalent outputs to HuggingFace has been falsified.");
                    eprintln!("  Investigation required before certification can proceed.");
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("Error loading golden output: {e}");
            eprintln!("\nHint: Use --list to see available golden outputs");
            std::process::exit(1);
        }
    }
}

/// Load logits tensor from a SafeTensors file
fn load_logits_from_file(logits_path: &std::path::Path) -> Vec<f32> {
    let logits_data = match std::fs::read(logits_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error reading logits file: {e}");
            std::process::exit(1);
        }
    };

    let tensors = match safetensors::SafeTensors::deserialize(&logits_data) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error parsing SafeTensors: {e}");
            std::process::exit(1);
        }
    };

    let logits_view = match tensors.tensor("logits") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: 'logits' tensor not found: {e}");
            std::process::exit(1);
        }
    };

    let data = logits_view.data();
    if data.len() % 4 != 0 {
        eprintln!(
            "Error: logits tensor byte length {} is not a multiple of 4 (corrupt or non-f32 data)",
            data.len()
        );
        std::process::exit(1);
    }
    data.as_chunks::<4>().0.iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Execute APR tool coverage tests and save results as JSON
fn run_tool_tests(
    model_path: &std::path::Path,
    no_gpu: bool,
    output_dir: &std::path::Path,
    include_serve: bool,
) {
    use aprender_qa_runner::ToolExecutor;

    print_tool_tests_banner(model_path, no_gpu, include_serve);

    let executor = ToolExecutor::new(model_path.to_string_lossy().to_string(), no_gpu, 120_000);
    let results = executor.execute_all_with_serve(include_serve);

    let (passed, failed) = print_tool_results_table(&results);
    println!("{}", "-".repeat(60));
    println!("Total: {passed} passed, {failed} failed\n");

    save_tool_results_json_or_exit(&results, output_dir);

    if failed > 0 {
        std::process::exit(1);
    }
}

fn print_tool_tests_banner(model_path: &std::path::Path, no_gpu: bool, include_serve: bool) {
    println!("=== APR Tool Coverage Tests ===\n");
    println!("Model: {}", model_path.display());
    println!("GPU: {}", if no_gpu { "disabled" } else { "enabled" });
    println!(
        "Serve test: {}\n",
        if include_serve { "enabled" } else { "disabled" }
    );
}

/// Print per-tool results table, return (passed, failed) counts.
fn print_tool_results_table(results: &[aprender_qa_runner::ToolTestResult]) -> (usize, usize) {
    println!("{:<20} {:<10} {:<10} Duration", "Tool", "Status", "Exit");
    println!("{}", "-".repeat(60));

    let mut passed = 0usize;
    let mut failed = 0usize;
    for result in results {
        let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
        println!(
            "{:<20} {:<10} {:<10} {}ms",
            result.tool, status, result.exit_code, result.duration_ms
        );
        if result.passed {
            passed += 1;
        } else {
            failed += 1;
        }
    }
    (passed, failed)
}

fn save_tool_results_json_or_exit(
    results: &[aprender_qa_runner::ToolTestResult],
    output_dir: &std::path::Path,
) {
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error creating output directory: {e}");
        std::process::exit(1);
    }

    // Built from a Serialize struct rather than serde_json::json!, because that
    // macro expands to Result::unwrap internally and this repo bans unwrap via
    // .clippy.toml disallowed-methods (GH-41). The ban fired here only after
    // this file moved from the bin target into the lib -- the diagnostic was
    // real the whole time, just charged to a target nothing linted.
    #[derive(serde::Serialize)]
    struct ToolResultJson<'a> {
        tool: &'a str,
        passed: bool,
        exit_code: i32,
        duration_ms: u64,
        gate_id: &'a str,
        stderr: &'a str,
    }

    let results_json = serde_json::to_string_pretty(
        &results
            .iter()
            .map(|r| ToolResultJson {
                tool: &r.tool,
                passed: r.passed,
                exit_code: r.exit_code,
                duration_ms: r.duration_ms,
                gate_id: &r.gate_id,
                stderr: &r.stderr,
            })
            .collect::<Vec<_>>(),
    )
    .unwrap_or_else(|e| {
        eprintln!("Error serializing tool results: {e}");
        std::process::exit(1);
    });

    let results_path = output_dir.join("tool_tests.json");
    if let Err(e) = std::fs::write(&results_path, results_json) {
        eprintln!("Error saving tool test results: {e}");
        std::process::exit(1);
    }
    println!("Results saved to: {}", results_path.display());
}

// Certification workflow — see certification.rs
include!("certification.rs");

// Export and cache — see export.rs
include!("export.rs");
