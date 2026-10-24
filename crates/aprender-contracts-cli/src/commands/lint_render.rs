//! Lint output rendering (text, JSON, SARIF, GitHub).
//! Split from lint.rs to stay under the 500-line limit.

use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};

use provable_contracts::lint::finding::LintFinding;
use provable_contracts::lint::rules::RuleSeverity;
use provable_contracts::lint::sarif::{findings_to_sarif, sarif_to_json};
use provable_contracts::lint::{GateDetail, GateExtra, LintReport};
use provable_contracts::ontology::verdict::Verdict;

/// Whether to use ANSI colors (auto-detected or forced via --color).
fn use_color() -> bool {
    // Check NO_COLOR env (https://no-color.org/)
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }
    // Check if stdout is a terminal
    std::io::stdout().is_terminal()
}

// ANSI escape helpers
fn red(s: &str) -> String {
    if use_color() {
        format!("\x1b[31m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
fn yellow(s: &str) -> String {
    if use_color() {
        format!("\x1b[33m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
fn cyan(s: &str) -> String {
    if use_color() {
        format!("\x1b[36m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
fn bold(s: &str) -> String {
    if use_color() {
        format!("\x1b[1m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
fn green(s: &str) -> String {
    if use_color() {
        format!("\x1b[32m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

fn severity_colored(sev: RuleSeverity) -> String {
    match sev {
        RuleSeverity::Error => red("ERROR"),
        RuleSeverity::Warning => yellow("WARN"),
        RuleSeverity::Info => cyan("INFO"),
        RuleSeverity::Off => "OFF".to_string(),
    }
}

pub fn print_text(report: &LintReport) {
    println!("{}", bold("pv lint — contract quality gate"));
    println!("{}\n", bold("================================"));
    for (i, gate) in report.gates.iter().enumerate() {
        print_gate(i + 1, gate, !report.not_armed.contains(&gate.name));
    }
    println!();
    print_findings_grouped(report);
    print_contract_timings(report);
    print_summary(report);
}

/// One gate line. ONT-001 §3.9: an unarmed gate prints `Unknown(NotArmed)` (its own verdict in brackets) and is
/// excluded from the meet.
pub fn print_gate(num: usize, gate: &provable_contracts::lint::GateResult, armed: bool) {
    let icon = if gate.skipped {
        "⏭"
    } else if gate.passed {
        &green("✓")
    } else {
        &red("✗")
    };
    let summary =
        gate_extra_summary(gate.extra.as_ref()).unwrap_or_else(|| gate_summary(&gate.detail));
    let verdict = if armed {
        gate.verdict.to_string()
    } else {
        format!("Unknown(NotArmed) [gate: {}]", gate.verdict)
    };
    println!(
        "  Gate {num}: {:<20} {icon}  ({summary}) [{:.0}ms] → {verdict}",
        bold(&gate.name),
        gate.duration_ms
    );
    print_duplicate_stem_detail(gate.extra.as_ref());
}

/// Summary line for a gate whose shape post-dates the frozen `GateDetail` vocabulary.
///
/// `None` means the gate has no out-of-band payload and `gate_summary` applies.
///
/// `GateExtra` is `#[non_exhaustive]`, so this MUST stay a fallible `match` with a
/// `_` arm even while it has one variant: from outside `aprender-contracts` a
/// single-variant `let` is a refutable pattern (E0005). That is the point of the
/// attribute — the next post-0.3.1 gate adds a variant here and this renderer keeps
/// compiling, which is exactly what `GateDetail` cannot do.
fn gate_extra_summary(extra: Option<&GateExtra>) -> Option<String> {
    match extra? {
        GateExtra::DuplicateStems {
            divergent,
            baselined,
            unbaselined,
            stale,
            ..
        } => Some(format!(
            "{divergent} ambiguous stems, {baselined} baselined, {} unbaselined, {} stale",
            unbaselined.len(),
            stale.len()
        )),
        _ => None,
    }
}

/// Ambiguous stems are printed in full on EVERY run. A count alone would let 48
/// unresolvable contracts sit behind a green tick, which is how they got here.
fn print_duplicate_stem_detail(extra: Option<&GateExtra>) {
    let Some(GateExtra::DuplicateStems {
        unbaselined,
        stale,
        divergent_stems,
        ..
    }) = extra
    else {
        return;
    };
    for line in divergent_stems {
        println!("      {} {line}", yellow("ambiguous:"));
    }
    for stem in unbaselined {
        println!(
            "      {} `{stem}` diverges and is NOT in the baseline — deduplicate it",
            red("NEW:")
        );
    }
    for stem in stale {
        println!(
            "      {} `{stem}` no longer diverges — remove it from the baseline",
            red("STALE:")
        );
    }
}

pub fn gate_summary(detail: &GateDetail) -> String {
    match detail {
        GateDetail::Validate {
            contracts,
            errors,
            warnings,
            ..
        } => format!("{contracts} contracts, {errors} errors, {warnings} warnings"),
        GateDetail::Audit {
            contracts,
            findings,
            ..
        } => format!("{contracts} contracts, {findings} findings"),
        GateDetail::Score {
            contracts,
            mean_score,
            threshold,
            ..
        } => format!("{contracts} contracts, mean={mean_score:.2}, threshold={threshold:.2}"),
        GateDetail::Verify {
            total_refs,
            existing,
            missing,
        } => format!("{total_refs} refs, {existing} found, {missing} missing"),
        GateDetail::Enforce {
            equations_total,
            equations_with_pre,
            equations_with_post,
            ..
        } => format!("{equations_total} eqs, {equations_with_pre} pre, {equations_with_post} post"),
        GateDetail::ReverseCoverage {
            total_pub_fns,
            bound_fns,
            coverage_pct,
            ..
        } => format!("{bound_fns}/{total_pub_fns} bound ({coverage_pct:.1}%)"),
        GateDetail::Composition {
            edges_checked,
            edges_satisfied,
            edges_broken,
        } => format!("{edges_checked} edges, {edges_satisfied} satisfied, {edges_broken} broken"),
        GateDetail::Skipped { reason } => format!("skipped: {reason}"),
    }
}

/// Print findings grouped by contract file, then by rule within each contract.
fn print_findings_grouped(report: &LintReport) {
    let active: Vec<&LintFinding> = report.findings.iter().filter(|f| !f.suppressed).collect();
    if active.is_empty() {
        return;
    }

    // Group by file
    let mut by_file: BTreeMap<&str, Vec<&LintFinding>> = BTreeMap::new();
    for f in &active {
        let key = if f.file.is_empty() {
            "(global)"
        } else {
            &f.file
        };
        by_file.entry(key).or_default().push(f);
    }

    println!("{}:", bold("Findings"));
    for (file, findings) in &by_file {
        println!("\n  {} ({})", cyan(file), severity_counts(findings));
        for f in findings {
            print_finding(f);
        }
    }
    println!();
}

/// `2 errors, 1 warning, 3 info` for one file's findings; zero counts are omitted.
fn severity_counts(findings: &[&LintFinding]) -> String {
    let errors = findings
        .iter()
        .filter(|f| f.severity == RuleSeverity::Error)
        .count();
    let warnings = findings
        .iter()
        .filter(|f| f.severity == RuleSeverity::Warning)
        .count();
    let mut parts = Vec::new();
    if errors > 0 {
        parts.push(red(&format!(
            "{errors} error{}",
            if errors == 1 { "" } else { "s" }
        )));
    }
    if warnings > 0 {
        parts.push(yellow(&format!(
            "{warnings} warning{}",
            if warnings == 1 { "" } else { "s" }
        )));
    }
    let infos = findings.len() - errors - warnings;
    if infos > 0 {
        parts.push(format!("{infos} info"));
    }
    parts.join(", ")
}

fn print_finding(f: &LintFinding) {
    let sev = severity_colored(f.severity);
    let new_badge = if f.is_new {
        format!("  {}", yellow("[NEW]"))
    } else {
        String::new()
    };
    println!(
        "    [{sev}] {} — {}{new_badge}",
        bold(&f.rule_id),
        f.message
    );
    // Feature 5: Show source snippet if available
    if let Some(ref snippet) = f.snippet {
        println!("           | {snippet}");
    }
    // Feature 7: Show evidence if available
    if let Some(ref evidence) = f.evidence {
        println!("            evidence: {evidence}");
    }
    // Feature 10: Show fix suggestion if available
    if let Some(ref suggestion) = f.suggestion {
        for (i, line) in suggestion.lines().enumerate() {
            if i == 0 {
                println!("      fix: {line}");
            } else {
                println!("           {line}");
            }
        }
    }
}

pub fn print_summary(report: &LintReport) {
    use provable_contracts::lint::rules::find_rule;

    let total = report.findings.len();
    let active = report.findings.iter().filter(|f| !f.suppressed).count();
    let suppressed = total - active;
    let errors = report
        .findings
        .iter()
        .filter(|f| !f.suppressed && f.severity == RuleSeverity::Error)
        .count();
    let warnings = report
        .findings
        .iter()
        .filter(|f| !f.suppressed && f.severity == RuleSeverity::Warning)
        .count();
    let new_count = report
        .findings
        .iter()
        .filter(|f| !f.suppressed && f.is_new)
        .count();
    let new_part = if new_count > 0 {
        format!(", {} new", yellow(&new_count.to_string()))
    } else {
        String::new()
    };
    println!(
        "Summary: {} errors, {} warnings, {suppressed} suppressed{new_part}",
        if errors > 0 {
            red(&errors.to_string())
        } else {
            "0".to_string()
        },
        if warnings > 0 {
            yellow(&warnings.to_string())
        } else {
            "0".to_string()
        },
    );

    // Feature 8: Remediation effort estimation
    let total_effort: u32 = report
        .findings
        .iter()
        .filter(|f| !f.suppressed)
        .map(|f| find_rule(&f.rule_id).map_or(10, |r| r.effort_minutes))
        .sum();
    if total_effort > 0 {
        let hours = total_effort / 60;
        let minutes = total_effort % 60;
        let effort_str = if hours > 0 && minutes > 0 {
            format!("~{hours}h {minutes}m")
        } else if hours > 0 {
            format!("~{hours}h")
        } else {
            format!("~{minutes}m")
        };
        println!("Estimated remediation: {effort_str}");
    }

    print_armed_meet(report);
}

/// ONT-001 §3.9: the meet over the armed gates, the monotone check, and a `Result:` that agrees with the exit.
fn print_armed_meet(report: &LintReport) {
    let not_armed = if report.not_armed.is_empty() {
        String::new()
    } else {
        format!("; not armed: {}", report.not_armed.join(", "))
    };
    println!(
        "armed meet: {} ({} armed{not_armed})",
        report.verdict,
        report.armed_gates.len()
    );
    if let Some(monotone) = &report.armed_monotone {
        println!("armed_gates monotone: {monotone}");
    }
    if let Some(monotone) = &report.armed_shapes_monotone {
        println!("armed_shapes monotone: {monotone}");
    }
    let result = match report.verdict {
        Verdict::Pass => green("PASS"),
        Verdict::Fail => red("FAIL"),
        Verdict::Unknown(reason) => yellow(&format!("DECLINE ({reason})")),
    };
    println!("Result: {result}");
}

/// Print the slowest 5 contracts by processing time (Feature 11).
fn print_contract_timings(report: &LintReport) {
    if report.contract_timings.is_empty() {
        return;
    }
    println!("{}:", bold("Slowest contracts"));
    for (stem, ms) in report.contract_timings.iter().take(5) {
        println!("  {stem:<40} {ms}ms");
    }
    println!();
}

pub fn print_json(report: &LintReport) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(report)?;
    println!("{json}");
    Ok(())
}

pub fn print_sarif(report: &LintReport) {
    let sarif = findings_to_sarif(&report.findings, env!("CARGO_PKG_VERSION"));
    println!("{}", sarif_to_json(&sarif, true));
}

pub fn print_github(report: &LintReport) {
    for f in &report.findings {
        if f.suppressed {
            continue;
        }
        let level = match f.severity {
            RuleSeverity::Error => "error",
            RuleSeverity::Warning => "warning",
            _ => "notice",
        };
        println!(
            "::{level} file={},title={}::{}",
            f.file, f.rule_id, f.message
        );
    }
}

/// `println!` that targets `out`; a failed write (closed pipe) is dropped.
macro_rules! wl {
    ($w:expr) => {{
        let _ = writeln!($w);
    }};
    ($w:expr, $($a:tt)*) => {{
        let _ = writeln!($w, $($a)*);
    }};
}

/// Print long-form explanation for a lint rule.
pub fn print_explain(rule_id: &str) {
    if !write_explain(&mut std::io::stdout().lock(), rule_id) {
        eprintln!("Unknown rule: {rule_id}");
        eprintln!("Use `pv lint` to see active rules, or check docs/specifications/sub/lint.md");
        std::process::exit(1);
    }
}

/// Write the explanation for `rule_id` to `out`; `false` when the rule is unknown
/// (nothing is written in that case).
fn write_explain(out: &mut impl Write, rule_id: &str) -> bool {
    use provable_contracts::lint::rules::{find_rule, RuleCategory};

    let Some(rule) = find_rule(rule_id) else {
        return false;
    };

    wl!(
        out,
        "{}\n",
        bold(&format!("{}: {}", rule.id, rule.description))
    );
    wl!(
        out,
        "Category:         {}",
        format!("{:?}", rule.category).to_lowercase()
    );
    wl!(
        out,
        "Default severity: {}\n",
        rule.default_severity.as_str()
    );

    // Long-form guidance per rule category
    let (why, how) = match rule.category {
        RuleCategory::Validate => (
            "Schema validation ensures contracts are machine-parseable and complete.\n\
             Without valid schema, no downstream analysis (audit, score, proof) can run.",
            "Fix the YAML syntax or add the missing required field.\n\
             Run `pv validate <contract>` for detailed parse errors.",
        ),
        RuleCategory::Audit => (
            "Audit rules verify the traceability chain from paper to proof.\n\
             Gaps in traceability mean the contract's claims are not substantiated.",
            "Add the missing element (test, reference, domain, tolerance).\n\
             Run `pv audit <contract>` for the full traceability report.",
        ),
        RuleCategory::Score => (
            "Score rules flag contracts below quality thresholds.\n\
             Low scores indicate incomplete verification coverage.",
            "Improve the weakest dimension: add falsification tests, Kani\n\
             harnesses, Lean theorems, or binding entries as needed.\n\
             Run `pv score <contract>` to see per-dimension breakdown.",
        ),
        RuleCategory::Provability => (
            "Provability rules enforce the core invariant: kernel contracts\n\
             MUST have proof_obligations, falsification_tests, and kani_harnesses.\n\
             This is non-negotiable — it's why the project exists.",
            "Add the missing proof infrastructure to the contract YAML.\n\
             Use `pv scaffold <contract>` to generate test stubs.\n\
             Use `pv kani <contract>` to generate Kani harnesses.",
        ),
        RuleCategory::Trend => (
            "Trend rules detect quality regression over time.\n\
             A declining score means verification coverage is eroding.",
            "Review recent changes that reduced coverage.\n\
             Run `pv lint --show-trend` to see historical data.",
        ),
        RuleCategory::Suppression => (
            "Stale suppressions accumulate when findings are fixed but\n\
             the suppression entry remains. They mask future issues.",
            "Remove the stale suppression from --suppress-rule or .pv.toml.\n\
             The finding it suppressed no longer fires.",
        ),
        RuleCategory::Enforcement => (
            "Enforcement rules check per-contract verification levels and\n\
             prevent regression below locked levels (irreversible gates).",
            "Either improve the contract to meet its declared enforcement_level,\n\
             or use `pv unlock <contract> --reason \"...\"` to release the lock.",
        ),
        RuleCategory::Verify => (
            "Verify rules cross-check contract claims against the source tree.\n\
             A contract that cites a non-existent test function is unfalsifiable\n\
             — operators reading it think the falsifier is bound when it isn't.",
            "Either rename the test to match the contract, or update the contract\n\
             `test:` field to cite the real function name. Run `pv lint\n\
             --strict-test-binding contracts/` to see all dangling references.",
        ),
    };

    wl!(out, "WHY IT MATTERS:");
    for line in why.lines() {
        wl!(out, "  {line}");
    }
    wl!(out, "\nHOW TO FIX:");
    for line in how.lines() {
        wl!(out, "  {line}");
    }
    wl!(out, "\nREFERENCES:");
    wl!(out, "  - docs/specifications/sub/lint.md");
    wl!(
        out,
        "  - docs/specifications/pv-spec.md Section 5 (CLI Reference)"
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use provable_contracts::lint::rules::{rules_for_category, RuleCategory};

    fn explain(rule_id: &str) -> Option<String> {
        let mut buf: Vec<u8> = Vec::new();
        let known = write_explain(&mut buf, rule_id);
        let text = String::from_utf8(buf).expect("explanation is UTF-8");
        known.then_some(text)
    }

    #[test]
    fn unknown_rule_writes_nothing_and_reports_false() {
        let mut buf: Vec<u8> = Vec::new();
        assert!(!write_explain(&mut buf, "PV-NOPE-999"));
        assert!(buf.is_empty(), "nothing is written for an unknown rule");
    }

    #[test]
    fn explanation_has_header_metadata_and_all_sections_in_order() {
        let rule = &rules_for_category(RuleCategory::Validate)[0];
        let text = explain(rule.id).expect("catalog rule is known");

        assert!(text.contains(&format!("{}: {}", rule.id, rule.description)));
        assert!(text.contains("Category:         validate\n"));
        assert!(text.contains(&format!(
            "Default severity: {}\n",
            rule.default_severity.as_str()
        )));
        let at = |needle: &str| {
            text.find(needle)
                .unwrap_or_else(|| panic!("missing {needle}"))
        };
        assert!(at("Category:") < at("WHY IT MATTERS:"));
        assert!(at("WHY IT MATTERS:") < at("HOW TO FIX:"));
        assert!(at("HOW TO FIX:") < at("REFERENCES:"));
        assert!(text.contains("  - docs/specifications/sub/lint.md\n"));
    }

    #[test]
    fn guidance_lines_are_indented_two_spaces() {
        let rule = &rules_for_category(RuleCategory::Audit)[0];
        let text = explain(rule.id).expect("catalog rule is known");
        let why = text
            .split("WHY IT MATTERS:\n")
            .nth(1)
            .and_then(|t| t.split("\nHOW TO FIX:").next())
            .expect("WHY section present");
        assert!(why.lines().count() >= 2, "multi-line guidance is split");
        assert!(why.lines().all(|l| l.starts_with("  ")), "{why}");
    }

    /// Each category must carry ITS OWN guidance: a swapped or collapsed
    /// match arm shows up as the wrong marker phrase.
    #[test]
    fn every_category_gets_its_own_guidance() {
        let cases = [
            (
                RuleCategory::Validate,
                "validate",
                "Schema validation ensures",
                "pv validate <contract>",
            ),
            (
                RuleCategory::Audit,
                "audit",
                "traceability chain from paper to proof",
                "pv audit <contract>",
            ),
            (
                RuleCategory::Score,
                "score",
                "Score rules flag contracts below",
                "pv score <contract>",
            ),
            (
                RuleCategory::Provability,
                "provability",
                "core invariant",
                "pv scaffold <contract>",
            ),
            (
                RuleCategory::Trend,
                "trend",
                "detect quality regression",
                "pv lint --show-trend",
            ),
            (
                RuleCategory::Suppression,
                "suppression",
                "Stale suppressions accumulate",
                "--suppress-rule",
            ),
            (
                RuleCategory::Enforcement,
                "enforcement",
                "per-contract verification levels",
                "pv unlock <contract>",
            ),
            (
                RuleCategory::Verify,
                "verify",
                "cross-check contract claims",
                "--strict-test-binding",
            ),
        ];
        for (cat, label, why, how) in cases {
            let rules = rules_for_category(cat);
            let Some(rule) = rules.first() else {
                continue; // category with no catalog rule has nothing to explain
            };
            let text = explain(rule.id).expect("catalog rule is known");
            assert!(
                text.contains(&format!("Category:         {label}\n")),
                "{label}: {text}"
            );
            assert!(text.contains(why), "{label} WHY marker missing: {text}");
            assert!(text.contains(how), "{label} HOW marker missing: {text}");
            for (other, _, other_why, _) in cases {
                if other != cat {
                    assert!(!text.contains(other_why), "{label} leaked {other_why}");
                }
            }
        }
    }
}
