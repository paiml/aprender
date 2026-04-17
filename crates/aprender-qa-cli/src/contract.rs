
/// Export evidence to schema-compliant JSON (PMAT-265)
///
/// Converts test run results to the EvidenceExport format for oracle consumption.
#[allow(clippy::too_many_arguments)]
fn export_evidence(
    source: &Path,
    output_dir: &Path,
    model: &str,
    family: &str,
    size: &str,
    playbook_name: &str,
    tier: &str,
) {
    println!("Exporting evidence to schema-compliant JSON...");
    println!("  Source: {}", source.display());
    println!("  Output dir: {}", output_dir.display());
    println!("  Model: {model}");

    let json_value = read_source_json_or_exit(source);
    let (evidence_array, meta) = extract_evidence_and_meta(&json_value);

    if evidence_array.is_empty() {
        eprintln!("Error: Source file contains no evidence entries");
        eprintln!("  Popperian: untested hypotheses cannot be exported as evidence");
        std::process::exit(1);
    }

    let summary = build_export_summary(&evidence_array, meta);
    let (mqs_score, gateway_passed, grade) = compute_mqs_triple(model, &evidence_array);
    let gates = collect_gates_from_evidence(&evidence_array);
    let export = build_evidence_export(
        ExportIdentity { model, family, size, playbook_name, tier },
        summary.clone(),
        mqs_score,
        &grade,
        gateway_passed,
        gates,
        evidence_array,
    );

    let output_path = write_export_file_or_exit(output_dir, model, &export);
    print_export_summary(&output_path, model, mqs_score, &grade, &summary);
}

fn read_source_json_or_exit(source: &Path) -> serde_json::Value {
    let content = match std::fs::read_to_string(source) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: Cannot read source file: {e}");
            std::process::exit(1);
        }
    };
    match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: Invalid JSON in source file: {e}");
            std::process::exit(1);
        }
    }
}

/// Extract evidence array and optional meta object from source JSON.
///
/// Supports two source formats:
///   1. Plain array (`certifications/*/evidence.json`): `[{...}, ...]`
///   2. Execution result object (apr-qa run output): `{"evidence": [...], ...}`
fn extract_evidence_and_meta(
    json_value: &serde_json::Value,
) -> (Vec<serde_json::Value>, Option<&serde_json::Value>) {
    if json_value.is_array() {
        (json_value.as_array().cloned().unwrap_or_default(), None)
    } else {
        let arr = json_value
            .get("evidence")
            .and_then(|e| e.as_array())
            .cloned()
            .unwrap_or_default();
        (arr, Some(json_value))
    }
}

/// Tally evidence outcomes into (passed, failed, skipped, total_duration_ms).
fn count_evidence_outcomes(arr: &[serde_json::Value]) -> (usize, usize, usize, u64) {
    arr.iter().fold((0usize, 0usize, 0usize, 0u64), |(p, f, s, d), ev| {
        let outcome = ev.get("outcome").and_then(|o| o.as_str()).unwrap_or("");
        let dur = ev
            .get("duration_ms")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        match outcome {
            "Corroborated" => (p + 1, f, s, d + dur),
            "Falsified" | "Timeout" | "Crashed" => (p, f + 1, s, d + dur),
            "Skipped" => (p, f, s + 1, d + dur),
            _ => (p, f, s, d + dur),
        }
    })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn build_export_summary(
    evidence_array: &[serde_json::Value],
    meta: Option<&serde_json::Value>,
) -> aprender_qa_report::ExportSummary {
    use chrono::Utc;

    let (ev_passed, ev_failed, ev_skipped, ev_duration_ms) =
        count_evidence_outcomes(evidence_array);

    let meta_usize = |key: &str, fallback: usize| {
        meta.and_then(|v| v.get(key))
            .and_then(serde_json::Value::as_u64)
            .map_or(fallback, |v| v as usize)
    };

    let total_scenarios = meta_usize("total_scenarios", evidence_array.len());
    let passed = meta_usize("passed", ev_passed);
    let failed = meta_usize("failed", ev_failed);
    let skipped = meta_usize("skipped", ev_skipped);
    let duration_ms = meta
        .and_then(|v| v.get("duration_ms"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(ev_duration_ms);

    let pass_rate = if total_scenarios > 0 {
        passed as f64 / total_scenarios as f64
    } else {
        0.0
    };

    aprender_qa_report::ExportSummary {
        total_scenarios,
        passed,
        failed,
        skipped,
        pass_rate,
        duration_ms,
        timestamp: Utc::now(),
    }
}

/// Compute canonical (MQS score, gateway_passed, grade) triple.
///
/// Uses the same calculator as `score` and `report` commands for consistency.
fn compute_mqs_triple(model: &str, evidence_array: &[serde_json::Value]) -> (u32, bool, String) {
    let evidence_json =
        serde_json::to_string(&serde_json::Value::Array(evidence_array.to_vec()))
            .unwrap_or_default();
    match crate::parse_evidence(&evidence_json).and_then(|ev| {
        let collector = crate::collect_evidence(ev);
        crate::calculate_mqs_score(model, &collector)
    }) {
        Ok(mqs) => (mqs.raw_score, mqs.gateways_passed, mqs.grade),
        Err(_) => (0, false, "F".to_string()),
    }
}

/// Collect gateway-level gate results from evidence using pessimistic merge
/// (any failure overrides a prior pass — Jidoka).
fn collect_gates_from_evidence(
    evidence_array: &[serde_json::Value],
) -> std::collections::HashMap<String, aprender_qa_report::GateResult> {
    use aprender_qa_report::GateResult;
    use std::collections::HashMap;

    let mut gates: HashMap<String, GateResult> = HashMap::new();
    for ev in evidence_array {
        let Some(gate_id) = ev.get("gate_id").and_then(|g| g.as_str()) else {
            continue;
        };
        if !gate_id.starts_with('G') {
            continue;
        }
        let passed = ev
            .get("outcome")
            .and_then(|o| o.as_str())
            .is_some_and(|o| o == "Corroborated" || o == "Skipped");
        let reason = ev
            .get("reason")
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .to_string();

        if let Some(existing) = gates.get(gate_id) {
            if existing.passed && !passed {
                gates.insert(gate_id.to_string(), GateResult { passed, reason });
            }
        } else {
            gates.insert(gate_id.to_string(), GateResult { passed, reason });
        }
    }
    gates
}

struct ExportIdentity<'a> {
    model: &'a str,
    family: &'a str,
    size: &'a str,
    playbook_name: &'a str,
    tier: &'a str,
}

#[allow(clippy::too_many_arguments)]
fn build_evidence_export(
    identity: ExportIdentity<'_>,
    summary: aprender_qa_report::ExportSummary,
    mqs_score: u32,
    grade: &str,
    gateway_passed: bool,
    gates: std::collections::HashMap<String, aprender_qa_report::GateResult>,
    evidence: Vec<serde_json::Value>,
) -> aprender_qa_report::EvidenceExport {
    use aprender_qa_report::{EvidenceExport, ModelMeta, MqsExport, PlaybookMeta};
    use std::collections::HashMap;

    EvidenceExport {
        schema: "https://paiml.com/schemas/apr-qa-evidence.schema.json".to_string(),
        version: "1.0.0".to_string(),
        model: ModelMeta {
            hf_repo: identity.model.to_string(),
            family: identity.family.to_string(),
            size: identity.size.to_string(),
            format: "safetensors".to_string(),
        },
        playbook: PlaybookMeta {
            name: identity.playbook_name.to_string(),
            version: "1.0.0".to_string(),
            tier: identity.tier.to_string(),
        },
        summary,
        mqs: MqsExport {
            score: mqs_score,
            grade: grade.to_string(),
            gateway_passed,
            category_scores: HashMap::new(),
        },
        gates,
        evidence,
    }
}

fn write_export_file_or_exit(
    output_dir: &Path,
    model: &str,
    export: &aprender_qa_report::EvidenceExport,
) -> std::path::PathBuf {
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error: Cannot create output directory: {e}");
        std::process::exit(1);
    }

    let safe_name = model.replace('/', "-").to_lowercase();
    let output_path = output_dir.join(format!("{safe_name}.json"));

    let json = match export.to_json() {
        Ok(j) => j,
        Err(e) => {
            eprintln!("Error: Failed to serialize export: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = std::fs::write(&output_path, &json) {
        eprintln!("Error: Failed to write output: {e}");
        std::process::exit(1);
    }

    output_path
}

fn print_export_summary(
    output_path: &Path,
    model: &str,
    mqs_score: u32,
    grade: &str,
    summary: &aprender_qa_report::ExportSummary,
) {
    println!("\nExported evidence to: {}", output_path.display());
    println!("  Model: {model}");
    println!("  MQS Score: {mqs_score}");
    println!("  Grade: {grade}");
    println!("  Pass Rate: {:.1}%", summary.pass_rate * 100.0);
    println!("  Total Scenarios: {}", summary.total_scenarios);
}

/// Validate a model against the tensor layout contract (Issue #4)
///
/// Checks that tensor shapes in the APR model match the contract expectations.
/// This prevents GH-202 style bugs where wrong shapes cause garbage output.
fn validate_contract_command(
    model_path: &Path,
    contract_path: Option<&Path>,
    format: &str,
    critical_only: bool,
) {
    use aprender_qa_runner::{get_critical_tensors, get_validation_rules, validate_model};

    // Validate format first — before any output (Bug #85: JSON output was polluted)
    if !matches!(format, "text" | "json") {
        eprintln!("Error: Unknown format: {format}");
        eprintln!("  Valid formats: text, json");
        std::process::exit(1);
    }

    let text_mode = format != "json";

    if text_mode {
        println!("Validating model against tensor layout contract...");
        println!("  Model: {}", model_path.display());
    }

    let contract = load_layout_contract_with_format(contract_path, text_mode);

    if text_mode {
        println!("  Contract version: {}", contract.metadata.version);
        print_validation_rules(get_validation_rules(&contract));
        if critical_only {
            print_critical_tensors(get_critical_tensors(&contract));
        }
        println!("\n=== Running Validation ===");
    }

    let result = match validate_model(model_path, &contract) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: Validation failed: {e}");
            std::process::exit(1);
        }
    };

    output_validation_result(&result, format);
    exit_with_validation_status(result.passed);
}

/// Load the tensor layout contract, exiting on failure.
/// `text_mode`: when false (JSON mode), suppress human-readable path prints.
fn load_layout_contract_with_format(
    contract_path: Option<&Path>,
    text_mode: bool,
) -> aprender_qa_runner::TensorLayoutContract {
    use aprender_qa_runner::{load_contract, load_contract_from};

    let contract = contract_path.map_or_else(
        || {
            if text_mode {
                println!(
                    "  Contract: {} (default)",
                    aprender_qa_runner::layout_contract::DEFAULT_CONTRACT_PATH
                );
            }
            load_contract()
        },
        |path| {
            if text_mode {
                println!("  Contract: {}", path.display());
            }
            load_contract_from(path)
        },
    );

    match contract {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: Failed to load contract: {e}");
            eprintln!("\nHint: Ensure aprender is cloned as a sibling directory:");
            eprintln!("  ../aprender/contracts/tensor-layout-v1.yaml");
            std::process::exit(1);
        }
    }
}

/// Print validation rules from the contract.
fn print_validation_rules(rules: &[aprender_qa_runner::ValidationRule]) {
    println!("\n=== Validation Rules ({}) ===", rules.len());
    for rule in rules {
        let critical_marker = if rule.critical { " [CRITICAL]" } else { "" };
        println!("  {}: {}{}", rule.id, rule.name, critical_marker);
    }
}

/// Print critical tensors from the contract.
fn print_critical_tensors(tensors: Vec<&aprender_qa_runner::TensorSpec>) {
    println!("\n=== Critical Tensors ({}) ===", tensors.len());
    for tensor in &tensors {
        println!(
            "  {} -> {} (transpose: {})",
            tensor.gguf_name, tensor.apr_name, tensor.transpose
        );
    }
}

/// Output validation result in text or JSON format.
fn output_validation_result(result: &aprender_qa_runner::ModelValidationResult, format: &str) {
    if format == "json" {
        match serde_json::to_string_pretty(result) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("Error serializing result: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    println!("\n=== Validation Results ===");
    println!(
        "  Status: {}",
        if result.passed { "PASSED" } else { "FAILED" }
    );
    println!("  Rules Checked: {}", result.rules_checked);
    println!("  Rules Passed: {}", result.rules_passed);
    println!("  Rules Failed: {}", result.rules_failed);

    print_tensor_results(&result.tensor_results);
    print_critical_failures(&result.critical_failures);
}

/// Print per-tensor validation results.
fn print_tensor_results(tensor_results: &[aprender_qa_runner::TensorValidationResult]) {
    if tensor_results.is_empty() {
        return;
    }
    println!("\n  Per-Tensor Results:");
    for tr in tensor_results {
        let status = if tr.passed { "✓" } else { "✗" };
        println!("    {} [{}] {}", status, tr.rule_id, tr.tensor_name);
        if !tr.passed {
            println!("      Details: {}", tr.details);
            if let Some(ref expected) = tr.expected {
                println!("      Expected: {expected}");
            }
            if let Some(ref actual) = tr.actual {
                println!("      Actual: {actual}");
            }
        }
    }
}

/// Print critical failures if any.
fn print_critical_failures(failures: &[String]) {
    if failures.is_empty() {
        return;
    }
    println!("\n  CRITICAL FAILURES:");
    for failure in failures {
        println!("    ✗ {failure}");
    }
}

/// Exit with appropriate status code based on validation result.
fn exit_with_validation_status(passed: bool) -> ! {
    if passed {
        println!("\n✓ Model conforms to tensor layout contract");
        std::process::exit(0);
    } else {
        println!("\n✗ Model DOES NOT conform to tensor layout contract");
        std::process::exit(1);
    }
}

/// Verify kernel coverage across HuggingFace architectures (Spec §20).
///
/// Checks which kernel operations each model architecture requires, verifies
/// implementation status in the sovereign stack (trueno/realizar), and
/// optionally generates upstream tickets for gaps.
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_lines)]
fn kernel_coverage_command(
    architecture: Option<&str>,
    all: bool,
    models: bool,
    verify: bool,
    file_tickets: bool,
    output_dir: &Path,
    trueno_path: &Path,
    realizar_path: &Path,
    format: &str,
    contracts_path: &Path,
    bindings_path: &Path,
) {
    use aprender_qa_gen::CoverageContext;

    if !matches!(format, "json" | "text") {
        eprintln!("Error: Unknown format: {format}");
        eprintln!("  Valid formats: json, text");
        std::process::exit(1);
    }

    // Jidoka: reject mutually exclusive flag combinations (Bug #86)
    let mode_count = [verify, models, all, architecture.is_some()]
        .iter()
        .filter(|&&b| b)
        .count();
    if mode_count > 1 {
        eprintln!(
            "Error: --verify, --models, --all, and --architecture are mutually exclusive"
        );
        eprintln!("  Specify exactly one mode at a time.");
        std::process::exit(1);
    }

    let ctx = match CoverageContext::load(contracts_path, bindings_path) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Error loading coverage data: {e}");
            eprintln!("  contracts: {}", contracts_path.display());
            eprintln!("  bindings:  {}", bindings_path.display());
            eprintln!("\nHint: Ensure provable-contracts is cloned as a sibling directory");
            std::process::exit(1);
        }
    };

    // --verify mode: check binding claims against source code
    if verify {
        if let Some(report) = ctx.verify_bindings_against_source(trueno_path, realizar_path) {
            if format == "json" {
                match serde_json::to_string_pretty(&report) {
                    Ok(json) => println!("{json}"),
                    Err(e) => {
                        eprintln!("Error serializing report: {e}");
                        std::process::exit(1);
                    }
                }
            } else {
                print_binding_verification(&report);
            }
            if report.drift_count > 0 {
                std::process::exit(1);
            }
        } else {
            eprintln!("Error: Neither trueno nor realizar repos found");
            eprintln!("  trueno:   {}", trueno_path.display());
            eprintln!("  realizar: {}", realizar_path.display());
            eprintln!("\nHint: Use --trueno-path and --realizar-path to specify locations");
            std::process::exit(1);
        }
        return;
    }

    // --models mode: walk all registry models
    if models {
        let summary = ctx.verify_all_registry_models();
        if format == "json" {
            match serde_json::to_string_pretty(&summary) {
                Ok(json) => println!("{json}"),
                Err(e) => {
                    eprintln!("Error serializing summary: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            print_model_coverage_summary(&summary);
        }
        if file_tickets {
            let arch_report = ctx.verify_all_architectures();
            if !arch_report.gaps.is_empty() {
                write_gap_tickets(&arch_report, output_dir);
            }
        }
        if summary.gap_count > 0 {
            std::process::exit(1);
        }
        return;
    }

    if !all && architecture.is_none() {
        eprintln!("Error: Specify --architecture <name>, --all, or --models");
        std::process::exit(1);
    }

    let report = architecture.map_or_else(
        || ctx.verify_all_architectures(),
        |arch| {
            ctx.verify_by_name(arch).unwrap_or_else(|| {
                eprintln!("Error: Unknown architecture '{arch}'");
                eprintln!("Known architectures:");
                for name in ctx.architecture_names() {
                    eprintln!("  - {name}");
                }
                std::process::exit(1);
            })
        },
    );

    if format == "json" {
        match serde_json::to_string_pretty(&report) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("Error serializing report: {e}");
                std::process::exit(1);
            }
        }
    } else {
        print_coverage_report(&report);
    }

    if file_tickets && !report.gaps.is_empty() {
        write_gap_tickets(&report, output_dir);
    }

    // Exit with failure if missing gaps found (Jidoka: stop the line)
    if report.missing_count > 0 {
        std::process::exit(1);
    }
}

/// Print a human-readable kernel coverage matrix.
fn print_coverage_report(report: &aprender_qa_gen::CoverageReport) {
    println!(
        "\n{} Kernel Coverage Report",
        "===".bold().cyan()
    );
    println!(
        "  {} {} fused, {} fallback, {} missing (of {} total)",
        "Summary:".dimmed(),
        report.fused_count.to_string().green(),
        report.fallback_count.to_string().yellow(),
        if report.missing_count > 0 {
            report.missing_count.to_string().red().to_string()
        } else {
            report.missing_count.to_string()
        },
        report.total_ops,
    );

    for arch in &report.architectures {
        let class_label = arch
            .kernel_class
            .as_deref()
            .unwrap_or("?");
        println!(
            "\n  {} [Class {}]",
            arch.architecture.bold(),
            class_label
        );
        for op in &arch.ops {
            let symbol = op.status.symbol();
            let status_colored = match op.status {
                aprender_qa_gen::ImplementationStatus::Fused => symbol.green().to_string(),
                aprender_qa_gen::ImplementationStatus::Fallback => symbol.yellow().to_string(),
                aprender_qa_gen::ImplementationStatus::Missing => symbol.red().to_string(),
            };
            println!(
                "    {status_colored} {:30} trueno={:30} realizar={}",
                op.op.description(),
                op.trueno_fn.as_deref().unwrap_or("—"),
                op.realizar_fn.as_deref().unwrap_or("—"),
            );
        }
    }

    if report.gaps.is_empty() {
        println!(
            "\n{}",
            "All required kernel ops are implemented.".green().bold()
        );
    } else {
        println!(
            "\n{} {} gap(s) found:",
            "GAPS:".red().bold(),
            report.gaps.len()
        );
        for gap in &report.gaps {
            println!(
                "  {} [{}] {} — affects: {}",
                gap.status.ticket_priority().red(),
                gap.status,
                gap.op.description(),
                gap.affected_architectures.join(", ")
            );
        }
    }
}

/// Write gap tickets as markdown files to the output directory.
fn write_gap_tickets(report: &aprender_qa_gen::CoverageReport, output_dir: &Path) {
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        eprintln!("Error creating ticket directory: {e}");
        std::process::exit(1);
    }

    let mut written = 0;
    for gap in &report.gaps {
        let safe_name = gap
            .op
            .description()
            .to_lowercase()
            .replace([' ', '/'], "-");
        let filename = format!("KERNEL-GAP-{safe_name}.md");
        let path = output_dir.join(&filename);

        match std::fs::write(&path, &gap.ticket_body) {
            Ok(()) => {
                println!(
                    "  {} {}",
                    "Ticket:".dimmed(),
                    path.display()
                );
                written += 1;
            }
            Err(e) => {
                eprintln!("Error writing ticket {filename}: {e}");
            }
        }
    }

    println!(
        "\n{} {written} ticket(s) written to {}",
        "Filed:".bold().green(),
        output_dir.display()
    );
}

/// Print binding verification results.
fn print_binding_verification(report: &aprender_qa_gen::BindingVerificationReport) {
    println!(
        "\n{} Binding Verification Against Source",
        "===".bold().cyan()
    );
    if let Some(ref p) = report.trueno_path {
        println!("  {} {p}", "trueno:".dimmed());
    }
    if let Some(ref p) = report.realizar_path {
        println!("  {} {p}", "realizar:".dimmed());
    }
    println!(
        "  {} {}/{} claims verified, {} drift",
        "Summary:".dimmed(),
        report.verified_count.to_string().green(),
        report.total_claims,
        if report.drift_count > 0 {
            report.drift_count.to_string().red().to_string()
        } else {
            report.drift_count.to_string()
        },
    );

    println!("\n  {}", "Per-Binding Results:".bold());
    for bv in &report.bindings {
        let op_name = bv.op.description();

        // trueno column
        let trueno_status = bv.trueno_claim.as_ref().map_or_else(
            || "—".dimmed().to_string(),
            |claim| {
                if bv.trueno_found {
                    format!("{} {claim}", "✓".green())
                } else {
                    format!("{} {claim}", "✗".red())
                }
            },
        );

        // realizar column
        let realizar_status = bv.realizar_claim.as_ref().map_or_else(
            || "—".dimmed().to_string(),
            |claim| {
                if bv.realizar_found {
                    format!("{} {claim}", "✓".green())
                } else {
                    format!("{} {claim}", "✗".red())
                }
            },
        );

        println!(
            "    {op_name:30} trueno={trueno_status:40} realizar={realizar_status}"
        );

        // Show file paths for verified bindings
        if let Some(ref file) = bv.trueno_file {
            println!("    {:30} {}", "", file.dimmed());
        }
        if let Some(ref file) = bv.realizar_file {
            println!("    {:30} {}", "", file.dimmed());
        }
    }

    if report.drift_count == 0 {
        println!(
            "\n{}",
            "All binding claims verified against source code."
                .green()
                .bold()
        );
    } else {
        println!(
            "\n{} {} claim(s) not found in source — binding drift detected!",
            "DRIFT:".red().bold(),
            report.drift_count
        );
    }
}

/// Print per-model kernel coverage summary for the full registry.
#[allow(clippy::too_many_lines)]
fn print_model_coverage_summary(summary: &aprender_qa_gen::ModelCoverageSummary) {
    println!(
        "\n{} Model Kernel Coverage ({} models)",
        "===".bold().cyan(),
        summary.models.len()
    );
    println!(
        "  {} {} covered, {} with gaps",
        "Summary:".dimmed(),
        summary.covered_count.to_string().green(),
        if summary.gap_count > 0 {
            summary.gap_count.to_string().red().to_string()
        } else {
            summary.gap_count.to_string()
        },
    );
    if summary.defaults_count > 0 {
        println!(
            "  {} {} model(s) using default constraints (arch not in contracts YAML)",
            "WARNING:".yellow().bold(),
            summary.defaults_count,
        );
    }

    // Class summary table
    println!("\n  {}", "By Kernel Class:".bold());
    for cs in &summary.class_summary {
        let status = if cs.fully_covered {
            "✓".green().to_string()
        } else {
            "✗".red().to_string()
        };
        println!(
            "    {status} Class {} ({}) — {} model(s)",
            cs.class, cs.label, cs.model_count
        );
        if !cs.missing_ops.is_empty() {
            for op in &cs.missing_ops {
                println!("      {} {op}", "└".dimmed());
            }
        }
    }

    // Per-model table
    println!("\n  {}", "Per-Model Coverage:".bold());
    let mut current_arch = String::new();
    for model in &summary.models {
        if model.architecture != current_arch {
            current_arch.clone_from(&model.architecture);
            let class_str = model
                .kernel_class
                .as_deref()
                .unwrap_or("?");
            println!(
                "\n    {} [Class {}]",
                current_arch.bold().cyan(),
                class_str
            );
        }

        let status = if model.using_defaults {
            "?".yellow().to_string()
        } else if model.fully_covered {
            "✓".green().to_string()
        } else if model.missing_ops > 0 {
            "✗".red().to_string()
        } else {
            "~".yellow().to_string()
        };

        let gap_info = if model.using_defaults {
            " — arch not in contracts YAML (using defaults)".to_string()
        } else if model.gap_ops.is_empty() {
            String::new()
        } else {
            format!(" — {}", model.gap_ops.join(", "))
        };

        println!("      {status} {}{gap_info}", model.model_id);
    }

    // Final verdict
    if summary.gap_count == 0 && summary.defaults_count == 0 {
        println!(
            "\n{}",
            "All registered models have full kernel coverage."
                .green()
                .bold()
        );
    } else {
        if summary.gap_count > 0 {
            println!(
                "\n{} {}/{} models have kernel gaps",
                "BLOCKED:".red().bold(),
                summary.gap_count,
                summary.models.len()
            );
        }
        if summary.defaults_count > 0 {
            println!(
                "\n{} {}/{} models use default constraints (arch missing from contracts YAML)",
                "UNVERIFIED:".yellow().bold(),
                summary.defaults_count,
                summary.models.len()
            );
            // Collect unique unknown architectures
            let mut unknown_archs: Vec<&str> = summary
                .models
                .iter()
                .filter(|m| m.using_defaults)
                .map(|m| m.architecture.as_str())
                .collect();
            unknown_archs.sort_unstable();
            unknown_archs.dedup();
            println!(
                "  Add to arch-constraints-v1.yaml: {}",
                unknown_archs.join(", ")
            );
        }
    }
}

/// Bootstrap an architecture-aware playbook from a family contract.
fn run_bootstrap(
    family: &str,
    size: &str,
    hf_repo: &str,
    tier: &str,
    output: Option<&Path>,
    contracts_path: &Path,
    dry_run: bool,
) {
    println!(
        "{} {}",
        "Bootstrapping playbook:".bold().cyan(),
        format!("{family}-{size}-{tier}").bold()
    );
    println!("  {} {hf_repo}", "HF Repo:".dimmed());
    println!("  {} {}", "Contracts:".dimmed(), contracts_path.display());

    match bootstrap_playbook_from_contract(family, size, hf_repo, tier, contracts_path) {
        Ok(yaml) => {
            if dry_run {
                println!("\n{yaml}");
            } else {
                let out_path = output.map_or_else(
                    || {
                        PathBuf::from(format!(
                            "playbooks/models/{family}-{size}-{tier}.playbook.yaml"
                        ))
                    },
                    PathBuf::from,
                );
                if let Some(parent) = out_path.parent() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        eprintln!("Error creating directory: {e}");
                        std::process::exit(1);
                    }
                }
                if out_path.exists() {
                    eprintln!(
                        "{} Playbook already exists: {}",
                        "Warning:".bold().yellow(),
                        out_path.display()
                    );
                    eprintln!("  Use --dry-run to preview, or delete the file first");
                    std::process::exit(1);
                }
                match std::fs::write(&out_path, &yaml) {
                    Ok(()) => {
                        println!("\n{} {}", "Written:".bold().green(), out_path.display());
                    }
                    Err(e) => {
                        eprintln!("Error writing playbook: {e}");
                        std::process::exit(1);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("{} {e}", "Error:".bold().red());
            std::process::exit(1);
        }
    }
}
