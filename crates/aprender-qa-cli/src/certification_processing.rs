
/// Process a single model's certification result. Returns true if the loop should break.
#[allow(clippy::too_many_arguments)]
fn process_certification_result(
    model_id: &str,
    playbook: &apr_qa_runner::Playbook,
    config: apr_qa_runner::ExecutionConfig,
    tier: CertTier,
    tier_str: &str,
    model_cache: Option<&PathBuf>,
    apr_binary: &str,
    fail_fast: bool,
    oracle_enhance: bool,
    output_dir: &PathBuf,
    certifications: &mut [apr_qa_certify::ModelCertification],
    short: &str,
    certified_count: &mut usize,
    failed_count: &mut usize,
) -> bool {
    match execute_playbook(playbook, config) {
        Ok(result) => {
            print_execution_summary(&result);

            let Some((raw_score, status, grade, mqs)) =
                compute_certification_scores(model_id, &result, tier)
            else {
                *failed_count += 1;
                return false;
            };

            print_certification_scores(tier_str, raw_score, &grade, status);

            let profile = if matches!(tier, CertTier::DimensionalSmoke) {
                apr_qa_runner::SixColumnProfile::default()
            } else {
                run_profiling_phase(&result, playbook, model_cache, short, apr_binary, fail_fast)
            };

            update_certification_record(
                certifications,
                model_id,
                raw_score,
                &grade,
                status,
                tier_str,
                &mqs,
                &profile,
            );

            let model_output = output_dir.join(short.to_lowercase().replace('.', "-"));
            if !save_evidence(&model_output, &result) {
                eprintln!("  [ERROR] Evidence save failed — Jidoka: stop the line");
                std::process::exit(1);
            }

            if oracle_enhance && result.failed > 0 {
                run_oracle_enhancement(model_id, &result, &model_output);
            }

            *certified_count += 1;
            println!();

            if fail_fast && (result.failed > 0 || result.gateway_failed.is_some()) {
                eprintln!("[FAIL-FAST] Stopping certification after {model_id} (had failures)");
                return true;
            }
            false
        }
        Err(e) => {
            eprintln!("  Execution failed: {e}");
            *failed_count += 1;
            if fail_fast {
                eprintln!("[FAIL-FAST] Stopping certification after {model_id} (execution error)");
                return true;
            }
            false
        }
    }
}

/// Threshold-based RAG (Red/Amber/Green) tier classification.
/// Returns 0 for green (>= high), 1 for yellow (>= low), 2 for red (below low).
fn rag_tier(value: f64, high: f64, low: f64) -> usize {
    if value >= high { 0 } else if value >= low { 1 } else { 2 }
}

/// Score thresholds: green >= 700, yellow >= 400.
const SCORE_HIGH: f64 = 700.0;
const SCORE_LOW: f64 = 400.0;

/// Pass rate thresholds: green >= 90%, yellow >= 70%.
const RATE_HIGH: f64 = 90.0;
const RATE_LOW: f64 = 70.0;

/// Display color-coded certification tier, score, grade, and status
fn print_certification_scores(
    tier_str: &str,
    raw_score: u32,
    grade: &str,
    status: apr_qa_certify::CertificationStatus,
) {
    println!("  {} {tier_str}", "Tier:".dimmed());

    let score_str = format!("{raw_score}/1000");
    let colored_score = match rag_tier(f64::from(raw_score), SCORE_HIGH, SCORE_LOW) {
        0 => score_str.bold().green(),
        1 => score_str.bold().yellow(),
        _ => score_str.bold().red(),
    };
    println!("  {} {colored_score}", "MQS Score:".dimmed());
    let colored_grade = match grade {
        "A" | "B" => grade.green(),
        "C" | "D" => grade.yellow(),
        _ => grade.red(),
    };
    println!("  {} {colored_grade}", "Grade:".dimmed());
    let status_str = format!("{status}");
    let colored_status = if status_str.contains("Certified") || status_str.contains("Passed") {
        status_str.green()
    } else {
        status_str.red()
    };
    println!("  {} {colored_status}", "Status:".dimmed());
}

/// Print scenario counts, pass/fail tallies, and color-coded pass rate
fn print_execution_summary(result: &apr_qa_runner::ExecutionResult) {
    println!("  {} {}", "Scenarios:".dimmed(), result.total_scenarios);
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
    let pass_rate = result.pass_rate();
    let rate_str = format!("{pass_rate:.1}%");
    let colored_rate = match rag_tier(pass_rate, RATE_HIGH, RATE_LOW) {
        0 => rate_str.green(),
        1 => rate_str.yellow(),
        _ => rate_str.red(),
    };
    println!("  {} {colored_rate}", "Pass rate:".dimmed());
}

/// Compute MQS raw score, certification status, grade, and score breakdown
fn compute_certification_scores(
    model_id: &str,
    result: &apr_qa_runner::ExecutionResult,
    tier: CertTier,
) -> Option<(
    u32,
    apr_qa_certify::CertificationStatus,
    String,
    apr_qa_report::MqsScore,
)> {
    use apr_qa_certify::{CertificationTier, grade_from_tier, score_from_tier, status_from_tier};

    let evidence_vec: Vec<_> = result.evidence.all().to_vec();
    let collector = collect_evidence(evidence_vec);
    let mqs = match calculate_mqs_score(model_id, &collector) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  Error calculating MQS: {e}");
            return None;
        }
    };

    let cert_tier = match tier {
        CertTier::Mvp | CertTier::DimensionalSmoke | CertTier::Smoke => CertificationTier::Mvp,
        CertTier::Quick | CertTier::Standard | CertTier::Deep => CertificationTier::Full,
    };

    let pass_rate = result.pass_rate() / 100.0;
    let has_p0 = result.gateway_failed.is_some();
    let raw_score = score_from_tier(cert_tier, pass_rate, has_p0);
    let status = status_from_tier(cert_tier, pass_rate, has_p0);
    let grade = grade_from_tier(cert_tier, pass_rate, has_p0);

    Some((raw_score, status, grade.to_string(), mqs))
}

/// Execute the 6-column profiling phase and return the throughput profile
fn run_profiling_phase(
    result: &apr_qa_runner::ExecutionResult,
    playbook: &apr_qa_runner::Playbook,
    model_cache: Option<&PathBuf>,
    short: &str,
    apr_binary: &str,
    fail_fast: bool,
) -> apr_qa_runner::SixColumnProfile {
    let has_failures = result.failed > 0 || result.gateway_failed.is_some();
    let mut profile = apr_qa_runner::SixColumnProfile::default();

    if fail_fast && has_failures {
        eprintln!("\n[FAIL-FAST] Skipping profiling - failures detected");
        eprintln!("[FAIL-FAST] Use evidence above for GitHub ticket\n");
        return profile;
    }

    let Some(cache) = model_cache else {
        return profile;
    };

    let model_dir = cache.join(short.to_lowercase().replace('.', "-"));
    if !model_dir.exists() {
        return profile;
    }

    println!("  Running 6-column profiling...");
    match apr_qa_runner::run_six_column_profile(apr_binary, &model_dir, 1, 2) {
        Ok(p) => {
            profile = p;
            print_profiling_results(&profile);
            check_profiling_assertions(&mut profile, playbook);
        }
        Err(e) => {
            eprintln!("  Profiling failed: {e}");
        }
    }

    profile
}

/// Print conversion statuses and throughput metrics from the profiling run
fn print_profiling_results(profile: &apr_qa_runner::SixColumnProfile) {
    for conv in &profile.conversions {
        let status = if conv.cached {
            "cached"
        } else if conv.success {
            "ok"
        } else {
            "FAILED"
        };
        println!(
            "    {} → {}: {} ({}ms)",
            conv.source_format, conv.target_format, status, conv.duration_ms
        );
        if let Some(ref err) = conv.error {
            if let Some(line) = err.lines().last() {
                println!("      {line}");
            }
        }
    }
    println!("    Throughput (tok/s):");
    for (label, tps) in [
        ("GGUF CPU", profile.tps_gguf_cpu),
        ("GGUF GPU", profile.tps_gguf_gpu),
        ("APR CPU ", profile.tps_apr_cpu),
        ("APR GPU ", profile.tps_apr_gpu),
        ("ST CPU  ", profile.tps_st_cpu),
        ("ST GPU  ", profile.tps_st_gpu),
    ] {
        if let Some(tps) = tps {
            println!("      {label}: {tps:.1}");
        }
    }
    println!("    Total profiling time: {}ms", profile.total_duration_ms);
}

/// Validate profiling throughput against CI assertion thresholds
fn check_profiling_assertions(
    profile: &mut apr_qa_runner::SixColumnProfile,
    playbook: &apr_qa_runner::Playbook,
) {
    let Some(ref profile_ci) = playbook.profile_ci else {
        return;
    };
    let cpu_threshold = profile_ci
        .assertions
        .min_throughput_cpu
        .or(profile_ci.assertions.min_throughput)
        .unwrap_or(5.0);
    let gpu_threshold = profile_ci
        .assertions
        .min_throughput_gpu
        .or(profile_ci.assertions.min_throughput)
        .unwrap_or(50.0);

    profile.check_assertions(cpu_threshold, gpu_threshold);

    if !profile.failed_assertions.is_empty() {
        println!("    ⚠️  Assertion failures:");
        for fail in &profile.failed_assertions {
            println!(
                "      {} {}: {:.1} tok/s < {:.1} min",
                fail.format.to_uppercase(),
                fail.backend.to_uppercase(),
                fail.actual_tps,
                fail.min_threshold
            );
        }
    }
}

/// Write final score, grade, status, gateways, and throughput into the certification record
#[allow(clippy::too_many_arguments)]
fn update_certification_record(
    certifications: &mut [apr_qa_certify::ModelCertification],
    model_id: &str,
    raw_score: u32,
    grade: &str,
    status: apr_qa_certify::CertificationStatus,
    tier_str: &str,
    mqs: &apr_qa_report::MqsScore,
    profile: &apr_qa_runner::SixColumnProfile,
) {
    use apr_qa_certify::CertificationStatus;
    use chrono::Utc;

    let Some(cert) = certifications.iter_mut().find(|c| c.model_id == model_id) else {
        eprintln!("  [WARN] Model {model_id} not found in models.csv — certification result not recorded");
        return;
    };

    let (final_status, final_grade, final_tier) = if profile.failed_assertions.is_empty() {
        (status, grade.to_string(), tier_str.to_string())
    } else {
        println!("  ❌ Certification BLOCKED by throughput assertions");
        (
            CertificationStatus::Blocked,
            "-".to_string(),
            "none".to_string(),
        )
    };

    cert.mqs_score = raw_score;
    cert.grade = final_grade;
    cert.status = final_status;
    cert.certified_tier = final_tier;
    cert.last_certified = Some(Utc::now());

    let gw = &mqs.gateways;
    cert.g1 = gw.iter().any(|g| g.id == "G1" && g.passed);
    cert.g2 = gw.iter().any(|g| g.id == "G2" && g.passed);
    cert.g3 = gw.iter().any(|g| g.id == "G3" && g.passed);
    cert.g4 = gw.iter().any(|g| g.id == "G4" && g.passed);

    cert.tps_gguf_cpu = profile.tps_gguf_cpu;
    cert.tps_gguf_gpu = profile.tps_gguf_gpu;
    cert.tps_apr_cpu = profile.tps_apr_cpu;
    cert.tps_apr_gpu = profile.tps_apr_gpu;
    cert.tps_st_cpu = profile.tps_st_cpu;
    cert.tps_st_gpu = profile.tps_st_gpu;
}

/// Persist execution evidence as JSON to the model output directory.
/// Returns false on any I/O or serialization error.
fn save_evidence(model_output: &std::path::Path, result: &apr_qa_runner::ExecutionResult) -> bool {
    if let Err(e) = std::fs::create_dir_all(model_output) {
        eprintln!("  Error creating model output dir: {e}");
        return false;
    }
    let evidence_path = model_output.join("evidence.json");
    match result.evidence.to_json() {
        Ok(json) => match std::fs::write(&evidence_path, json) {
            Ok(()) => {
                println!("  Evidence: {}", evidence_path.display());
                true
            }
            Err(e) => {
                eprintln!("  Error writing evidence: {e}");
                false
            }
        },
        Err(e) => {
            eprintln!("  Error serializing evidence: {e}");
            false
        }
    }
}

/// Generate oracle-enhanced failure checklists from failed evidence
fn run_oracle_enhancement(
    model_id: &str,
    result: &apr_qa_runner::ExecutionResult,
    model_output: &std::path::Path,
) {
    use apr_qa_runner::{OracleEnhancer, generate_checklist_markdown};

    let enhancer = OracleEnhancer::new();
    let failed_evidence = result.evidence.failures();

    if failed_evidence.is_empty() {
        return;
    }

    let context = enhancer.enhance_failure(failed_evidence[0]);

    let total = result.passed + result.failed;
    #[allow(clippy::cast_precision_loss)]
    let pass_rate = if total > 0 {
        (result.passed as f64 / total as f64) * 1000.0
    } else {
        0.0
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mqs = pass_rate as u32;
    let grade = apr_qa_certify::grade_from_score(mqs);

    let checklist_md =
        generate_checklist_markdown(model_id, mqs, grade, total, result.failed, &context);

    let checklist_path = model_output.join("checklist.md");
    if let Err(e) = std::fs::write(&checklist_path, &checklist_md) {
        eprintln!("  Error writing checklist: {e}");
    } else {
        println!("  Checklist: {}", checklist_path.display());
    }

    if context.oracle_available {
        println!(
            "  Oracle: {} hypotheses, {} cross-refs ({}ms)",
            context.hypotheses.len(),
            context.cross_references.len(),
            context.query_latency_ms
        );
    } else {
        println!("  Oracle: unavailable (using static checklist)");
    }
}

/// Emit a warning if the playbook lock file is missing and integrity checks are enabled
fn warn_missing_lock_file(no_integrity_check: bool) {
    if no_integrity_check {
        return;
    }
    let lock_path = "playbooks/playbook.lock.yaml";
    if !std::path::Path::new(lock_path).exists() {
        eprintln!(
            "[WARN] No playbook lock file found at {lock_path}. Run `apr-qa lock-playbooks` to generate one."
        );
    }
}

/// Collect evidence from all models and auto-generate structured GitHub tickets
fn run_auto_ticket_generation(
    models_to_certify: &[String],
    output_dir: &PathBuf,
    ticket_repo: &str,
) {
    let mut all_evidence: Vec<apr_qa_runner::Evidence> = Vec::new();
    for model_id in models_to_certify {
        let short: &str = model_id.split('/').next_back().unwrap_or(model_id);
        let evidence_path = output_dir
            .join(short.to_lowercase().replace('.', "-"))
            .join("evidence.json");
        match std::fs::read_to_string(&evidence_path) {
            Ok(json) => match parse_evidence(&json) {
                Ok(ev) => all_evidence.extend(ev),
                Err(e) => eprintln!("  [WARN] Failed to parse evidence for {model_id}: {e}"),
            },
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                eprintln!("  [WARN] Failed to read evidence for {model_id}: {e}");
            }
            Err(_) => {} // NotFound is expected for skipped models
        }
    }

    if all_evidence.is_empty() {
        return;
    }

    let tickets = execute_auto_tickets(&all_evidence, ticket_repo);
    if tickets.is_empty() {
        println!("\n[AUTO-TICKET] No structured tickets generated (no classified failures).");
    } else {
        println!("\n=== Auto-Generated Tickets ({}) ===", tickets.len());
        for ticket in &tickets {
            println!("  {} [{}]", ticket.title, ticket.priority);
            if let Some(ref fixture) = ticket.upstream_fixture {
                println!("    Fixture: {fixture}");
            }
        }
    }
}
