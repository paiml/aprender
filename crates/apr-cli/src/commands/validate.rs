//! Validate command implementation
//!
//! Toyota Way: Jidoka - Build quality in, stop on issues.
//! Validates model integrity using the 100-point QA checklist.

use crate::error::CliError;
use crate::output;
use aprender::error::AprenderError;
use aprender::format::rosetta::{
    FormatType, RosettaStone, ValidationReport as RosettaValidationReport,
};
use aprender::format::validation::{AprValidator, Category, CheckStatus, ValidationReport};
use colored::Colorize;
use std::fs;
use std::path::Path;

/// Run the validate command
#[provable_contracts_macros::contract("apr-cli-safety-v1", equation = "validate_exit_code")]
pub(crate) fn run(
    path: &Path,
    quality: bool,
    strict: bool,
    min_score: Option<u8>,
    json: bool,
    skip_contract: bool,
) -> Result<(), CliError> {
    contract_pre_validate_exit_code_consistency!();
    // BUG-VALIDATE-001 FIX: Validate min_score is in valid range [0, 100]
    if let Some(score) = min_score {
        if score > 100 {
            return Err(CliError::ValidationFailed(format!(
                "Invalid --min-score value: {}. Must be in range 0-100.",
                score
            )));
        }
    }

    validate_path(path)?;
    if !json {
        println!("Validating {}...\n", path.display());
    }

    // Detect format via magic bytes (Rosetta Stone dispatch)
    let format = FormatType::from_magic(path)
        .or_else(|_| FormatType::from_extension(path))
        .map_err(|e| CliError::InvalidFormat(format!("Cannot detect format: {e}")))?;

    // #2394 finding 17: `--min-score 100` on a GGUF exited 0. The dispatcher
    // simply did not pass `min_score` to the Rosetta branch, which computes no
    // score at all — the strictest possible threshold could never fail, and
    // nothing said so. A threshold against a number that is never computed is
    // a gate that cannot fail; refuse it instead of honouring it silently.
    if let Some(min) = min_score {
        if !produces_qa_score(format) {
            return Err(CliError::ValidationFailed(format!(
                "--min-score {min} does not apply to {format}: the 100-point QA checklist is the .apr validator, and {format} validation reports per-tensor pass/fail with no score. Re-run without --min-score, or use --strict to fail on NaN/Inf/all-zero findings."
            )));
        }
    }

    let result = match format {
        FormatType::Apr => {
            run_apr_validation(path, quality, strict, min_score, json, skip_contract)
        }
        FormatType::Gguf | FormatType::SafeTensors => {
            run_rosetta_validation(path, format, quality, strict, json, skip_contract)
        }
    };
    if let Ok(ref r) = result {
        contract_post_validate_exit_code_consistency!(r);
    }
    result
}

/// Does this format's validator produce a 0-100 QA score for `--min-score` to
/// threshold against?
///
/// Exhaustive on purpose, with no wildcard arm: a format added to
/// [`FormatType`] cannot compile until someone decides whether `--min-score`
/// means anything for it. That is the poka-yoke for #2394 finding 17 — the
/// GGUF path did not "handle min_score wrongly", it never received the
/// argument, and nothing forced anyone to notice.
fn produces_qa_score(format: FormatType) -> bool {
    match format {
        // `AprValidator` runs the 100-point checklist.
        FormatType::Apr => true,
        // RosettaStone reports per-tensor pass/fail; there is no score.
        FormatType::Gguf | FormatType::SafeTensors => false,
    }
}

/// APR validation via 100-point QA checklist + fail-closed content gates.
///
/// PMAT-926: the 100-point structural report (`AprValidator`) only covers
/// magic / header / version / flags on the `.apr` path — every Section-A
/// structural check 5-25 and every Section-B physics check is a
/// `Skip("Not implemented")` stub, and `--strict` was a no-op. The REAL
/// fail-closed content gates (F-DATA-QUALITY-001..007: all-zero, NaN/Inf,
/// L2~0, constant, density, dead output row) already exist in
/// `RosettaStone::validate_apr` but were UNREACHABLE from the CLI.
///
/// We now run BOTH:
///   1. the structural 100-point report (for the human-readable table), and
///   2. the Rosetta content gates on the dequantized `.apr` tensors,
/// and gate the exit code on the content gates so a content-broken `.apr`
/// (e.g. an all-zero `lm_head.weight`, or a NaN/Inf tensor) is REJECTED at
/// parity with the GGUF / SafeTensors path. `--strict` is now honored:
/// any NaN / Inf / all-zero finding escalates to a hard non-zero exit.
fn run_apr_validation(
    path: &Path,
    quality: bool,
    strict: bool,
    min_score: Option<u8>,
    json: bool,
    skip_contract: bool,
) -> Result<(), CliError> {
    let data = fs::read(path)?;
    let mut validator = AprValidator::new();
    let report = validator.validate_bytes(&data);

    // PMAT-926: run the real fail-closed content gates on the .apr tensors.
    // If the structural parse fails (bad magic / truncated / checksum
    // mismatch), `content` is the parse error — surfaced below so the
    // existing "invalid file" behavior is preserved.
    let content = RosettaStone::new().validate(path);

    if json {
        return print_apr_validation_json(path, report, &content, strict, min_score, skip_contract);
    }

    print_check_results(report);
    print_summary(report)?;

    if quality {
        print_quality_assessment(report);
    }

    check_min_score(report, min_score)?;

    // GH-647: Exit non-zero when validation shows contract violations
    // GH-642: --skip-contract bypasses the contract score threshold gate
    // #1866: gate on percentage of *implemented* checks (Pass/Fail/Warn),
    //        not the full 100-point denominator. Stubbed "Pending" checks
    //        scored as Skip — counting them against the model produced
    //        Grade F on every valid APR file until every stub was filled in.
    //        See apr-validate-quality-threshold-v1.yaml.
    if !skip_contract {
        let score = report.implemented_score();
        if let Some(pct) = score.pct() {
            if pct < 50.0 {
                return Err(CliError::ValidationFailed(format!(
                    "{score} ({pct:.0}%) — below 50% threshold"
                )));
            }
        }
        // implemented_score_pct() == None: entire QA suite is stubbed.
        // Treat as informational, not a hard fail. (apr qa remains the
        // canonical pass/fail gate per CLAUDE.md.)
    }

    // PMAT-926: fail-closed content gate (F-DATA-QUALITY-001..007) +
    // --strict wiring, applied identically to the Rosetta GGUF/ST path.
    gate_apr_content(&content, strict, skip_contract)
}

/// `--min-score N` against the checks that RAN, never against the aspirational
/// 100 (#1866).
///
/// This used to read `report.total_score < min`, where `total_score` is a raw
/// count of awarded points and the `.apr` checklist can award at most 4 of
/// them (22 of 26 checks are `Skip("Not implemented")` stubs). `--min-score 50`
/// therefore failed a model that the very same run printed `✓ VALID` for and
/// exited 0 on without the flag. Measured on
/// `qwen2.5-coder-0.5b-instruct.apr`: `error: Validation failed: Score 3/100
/// below minimum 50`, exit 5.
///
/// When nothing ran there is no score to threshold, so the flag is REFUSED
/// rather than silently satisfied — same rule as `produces_qa_score` above:
/// a threshold against a number that is never computed is a gate that cannot
/// fail.
fn check_min_score(report: &ValidationReport, min_score: Option<u8>) -> Result<(), CliError> {
    let Some(min) = min_score else {
        return Ok(());
    };
    let score = report.implemented_score();
    let Some(pct) = score.pct() else {
        return Err(CliError::ValidationFailed(format!(
            "--min-score {min} cannot be evaluated: none of the {} declared QA checks ran, so there is no score to threshold.",
            score.declared
        )));
    };
    if pct < f64::from(min) {
        return Err(CliError::ValidationFailed(format!(
            "Score {pct:.0}/100 below minimum {min} ({score})"
        )));
    }
    Ok(())
}

/// PMAT-926: gate the `.apr` exit code on the Rosetta content gates.
///
/// `--skip-contract` bypasses the content gate entirely (matches the
/// GGUF/SafeTensors path). Otherwise:
///   * `--strict` escalates any NaN / Inf / all-zero / L2~0 finding to a
///     hard non-zero exit (F-VALIDATE-STRICT-001), and
///   * any tensor that fails a data-quality gate (constant weight, density,
///     dead output row, NaN/Inf) fails closed (F-VALIDATE-APR-DISPATCH-001).
///
/// A clean, valid model produces an empty failure set → `Ok(())` (no false
/// positives). A structural parse error (bad magic / truncated / checksum
/// mismatch) surfaces as `ValidationFailed`.
fn gate_apr_content(
    content: &Result<RosettaValidationReport, AprenderError>,
    strict: bool,
    skip_contract: bool,
) -> Result<(), CliError> {
    if skip_contract {
        return Ok(());
    }

    let report = match content {
        Ok(report) => report,
        Err(e) => {
            // Structural parse failure (magic / header / checksum / truncated).
            return Err(CliError::ValidationFailed(format!(
                "APR content validation failed: {e}"
            )));
        }
    };

    if strict {
        if let Some(issues) = strict_blocking_issues(report) {
            return Err(CliError::ValidationFailed(format!("Strict mode: {issues}")));
        }
    }

    if report.is_valid {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(format!(
            "{} tensors failed data-quality validation (F-DATA-QUALITY)",
            report.failed_tensor_count
        )))
    }
}

/// Summarize the strict-blocking findings (NaN / Inf / all-zero) for a
/// Rosetta report, or `None` if there are none. Shared by the APR and the
/// GGUF/SafeTensors `--strict` gates so both paths behave identically
/// (F-VALIDATE-STRICT-001).
fn strict_blocking_issues(report: &RosettaValidationReport) -> Option<String> {
    if report.total_nan_count == 0
        && report.total_inf_count == 0
        && report.all_zero_tensors.is_empty()
    {
        return None;
    }
    let mut issues = Vec::new();
    if report.total_nan_count > 0 {
        issues.push(format!("{} NaN values", report.total_nan_count));
    }
    if report.total_inf_count > 0 {
        issues.push(format!("{} Inf values", report.total_inf_count));
    }
    if !report.all_zero_tensors.is_empty() {
        issues.push(format!(
            "{} all-zero tensors",
            report.all_zero_tensors.len()
        ));
    }
    Some(issues.join(", "))
}

/// GGUF/SafeTensors validation via RosettaStone (physics constraints)
fn run_rosetta_validation(
    path: &Path,
    format: FormatType,
    quality: bool,
    strict: bool,
    json: bool,
    skip_contract: bool,
) -> Result<(), CliError> {
    let rosetta = RosettaStone::new();
    let report = rosetta
        .validate(path)
        .map_err(|e| CliError::ValidationFailed(format!("Validation failed: {e}")))?;

    if json {
        // GH-610: Apply strict checks before JSON output (was previously skipped)
        if strict && !skip_contract {
            if let Some(issues) = strict_blocking_issues(&report) {
                // Still print the JSON report before returning error
                let _ = print_rosetta_validation_json(path, &report, format, quality);
                return Err(CliError::ValidationFailed(format!("Strict mode: {issues}")));
            }
        }
        return print_rosetta_validation_json(path, &report, format, quality);
    }

    output::header(&format!("Validate: {} (Rosetta Stone)", format));

    // Print per-tensor results as table
    let mut rows: Vec<Vec<String>> = Vec::new();
    for tv in &report.tensors {
        let badge = if tv.is_valid {
            output::badge_pass("PASS")
        } else {
            output::badge_fail("FAIL")
        };
        let failures_str = if tv.failures.is_empty() {
            String::new()
        } else {
            tv.failures.join("; ")
        };
        rows.push(vec![tv.name.clone(), badge, failures_str]);
    }
    if !rows.is_empty() {
        println!(
            "{}",
            output::table(&["Tensor", "Status", "Failures"], &rows)
        );
    }

    println!();
    println!("{}", report.summary());

    if quality {
        print_quality_constraints(&report);
    }

    // GH-507: --strict fails on warnings (NaN, Inf, all-zero tensors)
    // GH-642: --skip-contract bypasses strict contract checks
    if strict && !skip_contract {
        if let Some(issues) = strict_blocking_issues(&report) {
            return Err(CliError::ValidationFailed(format!("Strict mode: {issues}")));
        }
    }

    // GH-658: A model with 0 tensors is invalid (truncated/corrupt).
    if report.tensors.is_empty() {
        return Err(CliError::ValidationFailed(
            "Model contains 0 tensors (truncated or corrupt file)".to_string(),
        ));
    }

    // GH-642: --skip-contract bypasses tensor validation failure gate
    if skip_contract || report.is_valid {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(format!(
            "{} tensors failed validation",
            report.failed_tensor_count
        )))
    }
}

/// Print physics constraints and PMAT-235 contract gate breakdown.
fn print_quality_constraints(report: &RosettaValidationReport) {
    println!();
    println!(
        "{}",
        "=== Physics Constraints (APR-SPEC 10.9) ===".cyan().bold()
    );
    println!("  Total NaN:  {}", report.total_nan_count);
    println!("  Total Inf:  {}", report.total_inf_count);
    println!("  All-zeros:  {}", report.all_zero_tensors.len());
    println!("  Duration:   {} ms", report.duration_ms);

    let all_failures: Vec<(&str, &str)> = report
        .tensors
        .iter()
        .flat_map(|t| {
            t.failures
                .iter()
                .map(move |f| (t.name.as_str(), f.as_str()))
        })
        .collect();

    if all_failures.is_empty() {
        println!();
        println!(
            "  {} All tensors pass PMAT-235 contract gates",
            "[OK]".green()
        );
    } else {
        print_contract_violations(&all_failures);
    }
}

/// Print PMAT-235 contract violations grouped by rule ID.
fn print_contract_violations(failures: &[(&str, &str)]) {
    println!();
    println!("{}", "=== PMAT-235 Contract Violations ===".red().bold());
    let mut by_rule: std::collections::BTreeMap<&str, Vec<&str>> =
        std::collections::BTreeMap::new();
    for (tensor_name, failure) in failures {
        let rule_id = if failure.starts_with('[') {
            failure.find(']').map_or("UNKNOWN", |end| &failure[1..end])
        } else {
            "UNKNOWN"
        };
        by_rule.entry(rule_id).or_default().push(tensor_name);
    }
    for (rule, tensors) in &by_rule {
        println!("  {} {} tensor(s) failed", rule.red(), tensors.len());
        for name in tensors.iter().take(5) {
            println!("    - {}", name);
        }
        if tensors.len() > 5 {
            println!("    ... and {} more", tensors.len() - 5);
        }
    }
}

/// The exact `apr validate --json` document for a `.apr` file, and whether it
/// passed.
///
/// Kept separate from printing so a unit test can assert on the document a
/// machine consumer actually parses, rather than on a side effect of stdout.
/// #1866 shipped for 81 days because nothing ever looked at all three fields
/// at once: the same document said `"grade": "F"`, `"failed": 0` and
/// `"passed": true`.
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn apr_validation_json(
    path: &Path,
    report: &ValidationReport,
    content: &Result<RosettaValidationReport, AprenderError>,
    min_score: Option<u8>,
) -> (serde_json::Value, bool) {
    // #1866: `passed` is now the same predicate as the human VALID badge and
    // the grade band — `report.is_valid()` — and `--min-score` thresholds the
    // measured percentage, not the raw awarded points.
    let structurally_passed = report.is_valid() && check_min_score(report, min_score).is_ok();
    // GH-251: Only include executed checks (PASS/FAIL) — SKIP/WARN are not actionable
    // and cause parity checker false positives
    let checks_json: Vec<serde_json::Value> = report
        .checks
        .iter()
        .filter(|c| matches!(&c.status, CheckStatus::Pass | CheckStatus::Fail(_)))
        .map(|c| {
            let (status, detail) = match &c.status {
                CheckStatus::Pass => ("PASS", String::new()),
                CheckStatus::Fail(r) => ("FAIL", r.clone()),
                CheckStatus::Warn(r) => ("WARN", r.clone()),
                CheckStatus::Skip(r) => ("SKIP", r.clone()),
            };
            serde_json::json!({
                "id": c.id,
                "name": c.name,
                "status": status,
                "detail": detail,
                "points": c.points,
            })
        })
        .collect();
    // PMAT-926: surface the fail-closed content-gate summary in the JSON so
    // machine consumers can see WHY the .apr was rejected (parity with the
    // human-readable path).
    let (content_passed, content_nan, content_inf, content_zero, content_failed) = match content {
        Ok(r) => (
            r.is_valid,
            r.total_nan_count,
            r.total_inf_count,
            r.all_zero_tensors.len(),
            r.failed_tensor_count,
        ),
        Err(_) => (false, 0, 0, 0, 0),
    };
    let passed = structurally_passed && content_passed;
    let score = report.implemented_score();
    let output = serde_json::json!({
        "model": path.display().to_string(),
        "format": "apr",
        // #1866: `total_score` now means what its name says — a score out of
        // 100 — computed against the checks that RAN. It used to be the raw
        // count of awarded points printed as "3/100" for a healthy model.
        // `points_earned` / `checks_ran` keep the raw numbers available.
        "total_score": score.pct().map(|p| p.round() as u8),
        "grade": report.grade(),
        "verdict": if report.is_valid() { "VALID" } else { "INVALID" },
        "points_earned": score.passed,
        "checks_ran": score.ran,
        "checks_not_implemented": score.not_implemented(),
        "checks": checks_json,
        "total_checks": report.checks.len(),
        "failed": report.failed_checks().len(),
        "passed": passed,
        "content_passed": content_passed,
        "content_total_nan": content_nan,
        "content_total_inf": content_inf,
        "content_all_zero_tensors": content_zero,
        "content_failed_tensors": content_failed,
    });
    (output, passed)
}

/// Print APR validation report as JSON (GH-240/GH-251: machine-parseable output).
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn print_apr_validation_json(
    path: &Path,
    report: &ValidationReport,
    content: &Result<RosettaValidationReport, AprenderError>,
    strict: bool,
    min_score: Option<u8>,
    skip_contract: bool,
) -> Result<(), CliError> {
    // PMAT-926: --strict is now honored on the APR JSON path. The structural
    // report still drives `passed`, but the fail-closed content gate
    // (F-DATA-QUALITY) is applied AFTER the JSON is printed so machine
    // consumers always get a report, and the exit code fails closed.
    let (output, passed) = apr_validation_json(path, report, content, min_score);
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
    if !passed {
        // Report the reason the caller can act on: the threshold it missed, a
        // failed structural check, or — falling through — the content gate.
        check_min_score(report, min_score)?;
        let failed = report.failed_checks().len();
        if failed > 0 {
            // #2612: the JSON consumer reads the exit code AND stderr; give it
            // the same named reason the human path prints.
            return Err(CliError::ValidationFailed(format!(
                "{failed} validation checks failed ({}) — {}",
                report.implemented_score(),
                failed_check_details(report)
            )));
        }
    }
    // PMAT-926: fail-closed content gate + --strict, after the JSON is printed.
    gate_apr_content(content, strict, skip_contract)
}

/// Print Rosetta validation report as JSON (GH-240/GH-251: machine-parseable output).
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn print_rosetta_validation_json(
    path: &Path,
    report: &RosettaValidationReport,
    format: FormatType,
    quality: bool,
) -> Result<(), CliError> {
    // GH-251: Include individual tensor checks as a list (same schema as APR path)
    let checks_json: Vec<serde_json::Value> = report
        .tensors
        .iter()
        .map(|tv| {
            let status = if tv.is_valid { "PASS" } else { "FAIL" };
            let detail = if tv.failures.is_empty() {
                String::new()
            } else {
                tv.failures.join("; ")
            };
            serde_json::json!({
                "name": tv.name,
                "status": status,
                "detail": detail,
            })
        })
        .collect();

    let format_str = match format {
        FormatType::SafeTensors => "safetensors",
        FormatType::Gguf => "gguf",
        FormatType::Apr => "apr",
    };
    let mut output = serde_json::json!({
        "model": path.display().to_string(),
        "format": format_str,
        "total_tensors": report.tensor_count,
        "failed_tensors": report.failed_tensor_count,
        "total_nan": report.total_nan_count,
        "total_inf": report.total_inf_count,
        "duration_ms": report.duration_ms,
        "checks": checks_json,
        "total_checks": report.tensor_count,
        "failed": report.failed_tensor_count,
        "passed": report.is_valid,
    });

    // GH-508: Include quality details when --quality flag is set
    if quality {
        let all_zero_names: Vec<&str> =
            report.all_zero_tensors.iter().map(|s| s.as_str()).collect();
        output["quality"] = serde_json::json!({
            "total_nan": report.total_nan_count,
            "total_inf": report.total_inf_count,
            "all_zero_tensors": all_zero_names,
            "all_zero_count": report.all_zero_tensors.len(),
            "physics_pass": report.total_nan_count == 0
                && report.total_inf_count == 0
                && report.all_zero_tensors.is_empty(),
        });
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
    if !report.is_valid {
        return Err(CliError::ValidationFailed(format!(
            "{} tensors failed validation",
            report.failed_tensor_count
        )));
    }
    Ok(())
}

fn validate_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    Ok(())
}

fn print_check_results(report: &ValidationReport) {
    let mut rows: Vec<Vec<String>> = Vec::new();
    for check in &report.checks {
        let (badge, detail) = match &check.status {
            CheckStatus::Pass => (output::badge_pass("PASS"), String::new()),
            CheckStatus::Fail(reason) => (output::badge_fail("FAIL"), reason.clone()),
            CheckStatus::Warn(reason) => (output::badge_warn("WARN"), reason.clone()),
            CheckStatus::Skip(reason) => (output::badge_skip("SKIP"), reason.clone()),
        };
        rows.push(vec![
            format!("{}", check.id),
            check.name.to_string(),
            badge,
            detail,
        ]);
    }
    println!(
        "{}",
        output::table(&["#", "Check", "Status", "Detail"], &rows)
    );
}

/// The verdict line a human reads, with the denominator the score was
/// measured against.
///
/// #2394 finding 12: this used to read `✓ VALID 3/100 points` on a healthy
/// model — a green badge next to what looks like 3%. Most of the checklist is
/// `Skip("Not implemented")` stubs that never ran; on
/// `qwen2.5-coder-0.5b-instruct.apr` the report declares 26 checks of which 4
/// ran, so "100" was never the denominator anything was measured against. It
/// now reads `✓ VALID 3/4 checks that ran (22 of 26 not implemented — not
/// evidence of health)`.
fn summary_line(report: &ValidationReport) -> String {
    let failed = report.failed_checks().len();
    if failed == 0 {
        format!(
            "  {} {}",
            output::badge_pass("VALID"),
            report.implemented_score()
        )
    } else {
        format!("  {} {failed} checks failed", output::badge_fail("INVALID"))
    }
}

fn print_summary(report: &ValidationReport) -> Result<(), CliError> {
    // PMAT-926: --strict is now honored via the fail-closed content gate
    // (`gate_apr_content`), not ignored here. This function prints only the
    // structural 100-point summary table.
    println!();
    println!("{}", summary_line(report));

    let failed_checks = report.failed_checks();

    if failed_checks.is_empty() {
        Ok(())
    } else {
        // #2612: name the failures. A truncated `.apr` now fails check 5
        // ("Data section within file"), and the caller that reads only stderr
        // — every machine consumer does — needs the reason, not a count. The
        // "N validation checks failed" prefix is preserved verbatim.
        Err(CliError::ValidationFailed(format!(
            "{} validation checks failed — {}",
            failed_checks.len(),
            failed_check_details(report)
        )))
    }
}

/// `[id] name: reason` for every failed check, semicolon-separated (#2612).
///
/// Shared by the human and `--json` paths so both name the same defect. A count
/// alone ("1 validation checks failed") tells a machine consumer nothing it can
/// act on, and the count was the only thing either path emitted.
fn failed_check_details(report: &ValidationReport) -> String {
    report
        .failed_checks()
        .iter()
        .map(|c| match &c.status {
            CheckStatus::Fail(reason) => format!("[{}] {}: {reason}", c.id, c.name),
            _ => format!("[{}] {}", c.id, c.name),
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The `--quality` category table and TOTAL line, as a string.
///
/// Returned rather than printed so the falsifier can read the exact text a
/// user sees. #1866: this block printed
///
/// ```text
/// │ B. Tensor Physics & Statistics   │  0/25 │ ░░░░░░░░░░░░░░░░░░░░ │
///   TOTAL: 3/100  Grade: F
/// ```
///
/// on a model `apr qa` passes and `apr run` answers correctly from. Both
/// numbers were fiction: category B declares no checks at all on the `.apr`
/// path (so `0/25` is a zero for something never measured, not a zero score),
/// and `3/100` banded 3 awarded points against a ceiling of 4.
fn quality_assessment_body(report: &ValidationReport) -> String {
    let categories = [
        (Category::Structure, "A. Format & Structural Integrity"),
        (Category::Physics, "B. Tensor Physics & Statistics"),
        (Category::Tooling, "C. Tooling & Operations"),
        (Category::Conversion, "D. Conversion & Interoperability"),
    ];

    let mut rows: Vec<Vec<String>> = Vec::new();
    for (cat, name) in &categories {
        let score = report.category_score(*cat);
        let (cell, bar) = match score.pct() {
            Some(_) => (
                format!("{}/{}", score.passed, score.ran),
                output::progress_bar(score.passed as usize, score.ran as usize, 20),
            ),
            // Nothing in this category ran. Say so; do not draw a zero.
            None => ("not implemented".to_string(), String::new()),
        };
        rows.push(vec![(*name).to_string(), cell, bar]);
    }

    let mut out = output::table(&["Category", "Checks passed", "Progress"], &rows);
    let score = report.implemented_score();
    let grade = output::grade_color(report.grade());
    match score.pct() {
        Some(pct) => out.push_str(&format!(
            "\n  TOTAL: {}\n  SCORE: {pct:.0}% of the checks that ran   Grade: {grade}\n",
            format!("{score}").white().bold(),
        )),
        None => out.push_str(&format!(
            "\n  TOTAL: none of the {} declared checks ran — nothing was measured\n  SCORE: unavailable   Grade: {grade}\n",
            score.declared,
        )),
    }
    out
}

fn print_quality_assessment(report: &ValidationReport) {
    output::header("Quality Assessment");
    println!("{}", quality_assessment_body(report));

    // Print failed checks summary
    let failed = report.failed_checks();
    if !failed.is_empty() {
        output::subheader("Failed Checks");
        for check in failed {
            if let CheckStatus::Fail(reason) = &check.status {
                println!(
                    "  {} #{}: {} - {}",
                    "✗".red().bold(),
                    check.id,
                    check.name,
                    reason.dimmed()
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "validate_tests.rs"]
mod tests;
