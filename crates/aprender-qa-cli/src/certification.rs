/// Run the full certification pipeline for selected models
#[allow(clippy::too_many_lines)]
#[allow(clippy::fn_params_excessive_bools)]
fn run_certification(
    all: bool,
    family: Option<String>,
    tier_str: &str,
    kernel_class: Option<String>,
    model_ids: &[String],
    output_dir: &PathBuf,
    dry_run: bool,
    model_cache: Option<PathBuf>,
    apr_binary: &str,
    auto_ticket: bool,
    ticket_repo: &str,
    no_integrity_check: bool,
    fail_fast: bool,
    oracle_enhance: bool,
) {
    use apr_qa_certify::write_csv;

    let tier: CertTier = match tier_str.parse() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let model_cache = resolve_default_model_cache(model_cache);

    if fail_fast {
        log_environment();
    }

    print_certification_header(tier_str, dry_run, fail_fast, model_cache.as_ref());

    let (csv_path, mut certifications) = load_certification_csv();
    let models_to_certify = resolve_models_for_certification(
        all,
        family.as_deref(),
        model_ids,
        kernel_class.as_deref(),
        &certifications,
    );

    println!("Models to certify: {}\n", models_to_certify.len());

    if models_to_certify.is_empty() {
        eprintln!("Error: No models matched the given filter.");
        if let Some(ref fam) = family {
            eprintln!("  Family '{fam}' not found in models.csv.");
        }
        eprintln!("  Use --all or specify valid model IDs.");
        std::process::exit(1);
    }

    if dry_run {
        for model_id in &models_to_certify {
            let playbook_name = playbook_path_for_model(model_id, tier);
            println!("  Would certify: {model_id}");
            println!("    Playbook: {playbook_name}");
        }
        return;
    }

    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error creating output directory: {e}");
        std::process::exit(1);
    }

    let (certified_count, failed_count) = certify_model_loop(
        &models_to_certify,
        tier,
        tier_str,
        model_cache.as_ref(),
        apr_binary,
        no_integrity_check,
        fail_fast,
        oracle_enhance,
        output_dir,
        &mut certifications,
    );

    let csv_output = write_csv(&certifications);
    if let Err(e) = std::fs::write(&csv_path, &csv_output) {
        eprintln!("Error writing models.csv: {e}");
        // Jidoka: CSV persistence failure is a P0 defect — certification
        // results are lost. Stop the line.
        std::process::exit(1);
    }
    println!(
        "{} {}",
        "Updated:".green(),
        csv_path.display().to_string().cyan()
    );

    warn_missing_lock_file(no_integrity_check);

    if auto_ticket {
        run_auto_ticket_generation(&models_to_certify, output_dir, ticket_repo);
    }

    println!("\n{}", "=== Certification Summary ===".bold().cyan());
    println!(
        "{} {}",
        "Processed:".dimmed(),
        certified_count.to_string().bold().green()
    );
    println!(
        "{} {}",
        "Failed:".dimmed(),
        if failed_count > 0 {
            failed_count.to_string().bold().red()
        } else {
            failed_count.to_string().dimmed()
        }
    );
    println!("{} {}", "Total:".dimmed(), models_to_certify.len());

    if failed_count > 0 {
        std::process::exit(1);
    }
}

/// Resolve the model cache path.
///
/// Returns `None` when no explicit cache is provided, letting the executor
/// resolve model paths via `resolve_hf_repo_to_cache` and `apr pull`.
/// The old default `~/.cache/apr-models` was a phantom directory that didn't
/// match `apr pull`'s actual storage (`~/.apr/cache/hf/`), causing dim-smoke
/// to fail with "no config.json" for every model.
const fn resolve_default_model_cache(model_cache: Option<PathBuf>) -> Option<PathBuf> {
    model_cache
}

/// Print the certification run header with tier, dry-run, and cache info
fn print_certification_header(
    tier_str: &str,
    dry_run: bool,
    fail_fast: bool,
    model_cache: Option<&PathBuf>,
) {
    println!("{}\n", "=== APR Model Certification ===".bold().cyan());
    println!("{} {}", "Tier:".dimmed(), tier_str.bold().magenta());
    if dry_run {
        println!("{} {}", "Dry run:".dimmed(), "true".yellow());
    } else {
        println!("{} {}", "Dry run:".dimmed(), "false".dimmed());
    }
    println!("{} {fail_fast}", "Fail-fast:".dimmed());
    if let Some(cache) = model_cache {
        println!(
            "{} {}",
            "Model cache:".dimmed(),
            cache.display().to_string().cyan()
        );
    }
    println!();
}

/// Load and parse the models.csv certification tracking file
fn load_certification_csv() -> (PathBuf, Vec<apr_qa_certify::ModelCertification>) {
    use apr_qa_certify::parse_csv;
    let csv_path = PathBuf::from("docs/certifications/models.csv");
    let csv_content = match std::fs::read_to_string(&csv_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading models.csv: {e}");
            std::process::exit(1);
        }
    };
    let certifications = match parse_csv(&csv_content) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error parsing models.csv: {e}");
            std::process::exit(1);
        }
    };
    (csv_path, certifications)
}

/// Select models to certify based on --all, --family, or explicit model IDs
fn determine_models_to_certify(
    all: bool,
    family: Option<&str>,
    model_ids: &[String],
    certifications: &[apr_qa_certify::ModelCertification],
) -> Vec<String> {
    if all {
        certifications.iter().map(|c| c.model_id.clone()).collect()
    } else if let Some(fam) = family {
        certifications
            .iter()
            .filter(|c| c.family == fam)
            .map(|c| c.model_id.clone())
            .collect()
    } else if !model_ids.is_empty() {
        model_ids.to_vec()
    } else {
        eprintln!("Error: Specify --all, --family, or model IDs");
        std::process::exit(1);
    }
}

/// Resolve the final model list, optionally filtering by kernel class
fn resolve_models_for_certification(
    all: bool,
    family: Option<&str>,
    model_ids: &[String],
    kernel_class: Option<&str>,
    certifications: &[apr_qa_certify::ModelCertification],
) -> Vec<String> {
    kernel_class.map_or_else(
        || determine_models_to_certify(all, family, model_ids, certifications),
        |kc_str| {
            let kc: apr_qa_gen::KernelClass = match kc_str.parse() {
                Ok(k) => k,
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            };
            let families_and_ids: Vec<(String, String)> = certifications
                .iter()
                .map(|c| (c.family.clone(), c.model_id.clone()))
                .collect();
            let models = apr_qa_gen::models_in_class(kc, &families_and_ids);
            if models.is_empty() {
                eprintln!("No models found for kernel class {kc}");
                std::process::exit(1);
            }
            println!(
                "Kernel class {kc} ({}) — {} models, proof: {}",
                kc.label(),
                models.len(),
                kc.representative_model(),
            );
            models
        },
    )
}

/// Iterate over models, run certification for each, and tally results
#[allow(clippy::too_many_arguments)]
fn certify_model_loop(
    models_to_certify: &[String],
    tier: CertTier,
    tier_str: &str,
    model_cache: Option<&PathBuf>,
    apr_binary: &str,
    no_integrity_check: bool,
    fail_fast: bool,
    oracle_enhance: bool,
    output_dir: &PathBuf,
    certifications: &mut [apr_qa_certify::ModelCertification],
) -> (usize, usize) {
    let mut certified_count = 0;
    let mut failed_count = 0;

    for model_id in models_to_certify {
        let short: &str = model_id.split('/').next_back().unwrap_or(model_id);
        let playbook_name = playbook_path_for_model(model_id, tier);

        println!(
            "{} {} {}",
            "---".bold(),
            format!("Certifying: {model_id}").bold(),
            "---".bold()
        );
        println!("  {} {playbook_name}", "Playbook:".dimmed());

        if let Some(cache) = model_cache {
            let model_dir = cache.join(short.to_lowercase().replace('.', "-"));
            auto_populate_model_cache(model_id, &model_dir, apr_binary);
        }

        let playbook_path = std::path::Path::new(&playbook_name);
        if !playbook_path.exists() {
            eprintln!("  Playbook not found, skipping");
            failed_count += 1;
            continue;
        }

        let playbook = match load_playbook(playbook_path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  Error loading playbook: {e}");
                failed_count += 1;
                continue;
            }
        };

        if !verify_playbook_lock(playbook_path, &playbook.name, no_integrity_check) {
            failed_count += 1;
            continue;
        }

        let model_cache_path = model_cache.map(|cache| {
            cache
                .join(short.to_lowercase().replace('.', "-"))
                .to_string_lossy()
                .to_string()
        });

        let config = build_certification_config_with_policy(tier, model_cache_path, fail_fast);
        let should_break = process_certification_result(
            model_id,
            &playbook,
            config,
            tier,
            tier_str,
            model_cache,
            apr_binary,
            fail_fast,
            oracle_enhance,
            output_dir,
            certifications,
            short,
            &mut certified_count,
            &mut failed_count,
        );

        if should_break {
            break;
        }
    }

    (certified_count, failed_count)
}

/// Verify playbook lock or exit (for `run` subcommand)
fn verify_playbook_lock_or_exit(playbook_path: &std::path::Path, playbook_name: &str) {
    let lock_path = std::path::Path::new("playbooks/playbook.lock.yaml");
    if lock_path.exists() {
        match apr_qa_runner::load_lock_file(lock_path) {
            Ok(lock_file) => {
                if let Err(e) = apr_qa_runner::verify_playbook_integrity(
                    playbook_path,
                    &lock_file,
                    playbook_name,
                ) {
                    eprintln!("[INTEGRITY] {e}");
                    eprintln!("[INTEGRITY] Playbook hash does not match lock file.");
                    eprintln!("[INTEGRITY] Either:");
                    eprintln!("  1. Run `apr-qa lock-playbooks` to regenerate the lock file");
                    eprintln!("  2. Use --no-integrity-check to bypass (NOT RECOMMENDED)");
                    std::process::exit(1);
                }
                println!("  Integrity check: PASSED");
            }
            Err(e) => {
                eprintln!("[WARN] Could not load lock file: {e}");
            }
        }
    } else {
        eprintln!(
            "[WARN] No playbook lock file found. Run `apr-qa lock-playbooks` to generate one."
        );
    }
}

/// Returns true if playbook integrity is verified (or check skipped), false if blocked
fn verify_playbook_lock(
    playbook_path: &std::path::Path,
    playbook_name: &str,
    no_integrity_check: bool,
) -> bool {
    if no_integrity_check {
        return true;
    }
    let lock_path = std::path::Path::new("playbooks/playbook.lock.yaml");
    if !lock_path.exists() {
        return true;
    }
    match apr_qa_runner::load_lock_file(lock_path) {
        Ok(lock_file) => {
            if let Err(e) =
                apr_qa_runner::verify_playbook_integrity(playbook_path, &lock_file, playbook_name)
            {
                eprintln!("  [INTEGRITY] {e}");
                eprintln!(
                    "  [INTEGRITY] CERTIFICATION BLOCKED: Playbook modified without updating lock file."
                );
                eprintln!(
                    "  [INTEGRITY] Run `apr-qa lock-playbooks` first or use --no-integrity-check"
                );
                return false;
            }
            println!("  Integrity check: PASSED");
            true
        }
        Err(e) => {
            eprintln!("  [WARN] Could not load lock file: {e}");
            true
        }
    }
}

include!("certification_processing.rs");
