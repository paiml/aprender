//! Bug Hunter CLI output formatting and subcommands.

use crate::ansi_colors::Colorize;
use crate::bug_hunter::{
    hunt, DefectCategory, Finding, FindingSeverity, HuntConfig, HuntMode, HuntResult, HuntStats,
    PhaseTimings,
};
use std::path::PathBuf;

use super::BugHunterOutputFormat;

/// Stats for a single crate in the stack analysis.
#[derive(Debug)]
pub(super) struct CrateStats {
    pub(super) name: String,
    pub(super) total: usize,
    pub(super) critical: usize,
    pub(super) high: usize,
    pub(super) gpu: usize,
    pub(super) debt: usize,
    pub(super) test: usize,
    pub(super) silent: usize,
    pub(super) memory: usize,
    pub(super) contract: usize,
    pub(super) parity: usize,
}

impl CrateStats {
    /// Compute aggregate totals across a slice of crate stats.
    pub(super) fn sum_all(stats: &[CrateStats]) -> Self {
        macro_rules! sum {
            ($field:ident) => {
                stats.iter().map(|s| s.$field).sum()
            };
        }
        Self {
            name: "**TOTAL**".to_string(),
            total: sum!(total),
            critical: sum!(critical),
            high: sum!(high),
            gpu: sum!(gpu),
            debt: sum!(debt),
            test: sum!(test),
            silent: sum!(silent),
            memory: sum!(memory),
            contract: sum!(contract),
            parity: sum!(parity),
        }
    }
}

/// Output stack analysis as text.
pub(super) fn output_stack_text(
    crate_stats: &[CrateStats],
    total_findings: usize,
    total_critical: usize,
    total_high: usize,
    by_category: &std::collections::HashMap<DefectCategory, usize>,
) {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
            .bright_cyan()
    );
    println!(
        "{}",
        "║           CROSS-STACK BUG ANALYSIS - SOVEREIGN AI STACK               ║"
            .bright_cyan()
            .bold()
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝"
            .bright_cyan()
    );
    println!();

    println!(
        "{}",
        "┌─────────────────────────────────────────────────────────────────────────┐".dimmed()
    );
    println!(
        "{}",
        "│ STACK DEPENDENCY CHAIN: trueno → aprender → realizar → entrenar        │".dimmed()
    );
    println!(
        "{}",
        "└─────────────────────────────────────────────────────────────────────────┘".dimmed()
    );
    println!();

    println!("{}", "SUMMARY BY CRATE:".bold());
    println!("┌──────────────┬────────┬──────────┬──────┬────────┬──────┬────────┬──────┬────────┬────────┐");
    println!("│ Crate        │ Total  │ Critical │ High │ GPU    │ Debt │ Test   │ Mem  │ Ctrct  │ Parity │");
    println!("├──────────────┼────────┼──────────┼──────┼────────┼──────┼────────┼──────┼────────┼────────┤");

    for stats in crate_stats {
        println!(
            "│ {:<12} │ {:>6} │ {:>8} │ {:>4} │ {:>6} │ {:>4} │ {:>6} │ {:>4} │ {:>6} │ {:>6} │",
            stats.name,
            stats.total,
            stats.critical,
            stats.high,
            stats.gpu,
            stats.debt,
            stats.test,
            stats.memory,
            stats.contract,
            stats.parity
        );
    }

    println!("├──────────────┼────────┼──────────┼──────┼────────┼──────┼────────┼──────┼────────┼────────┤");
    println!(
        "│ {:<12} │ {:>6} │ {:>8} │ {:>4} │ {:>6} │ {:>4} │ {:>6} │ {:>4} │ {:>6} │ {:>6} │",
        "TOTAL".bold(),
        total_findings,
        total_critical,
        total_high,
        by_category.get(&DefectCategory::GpuKernelBugs).unwrap_or(&0),
        by_category.get(&DefectCategory::HiddenDebt).unwrap_or(&0),
        by_category.get(&DefectCategory::TestDebt).unwrap_or(&0),
        by_category.get(&DefectCategory::MemorySafety).unwrap_or(&0),
        by_category.get(&DefectCategory::ContractGap).unwrap_or(&0),
        by_category.get(&DefectCategory::ModelParityGap).unwrap_or(&0)
    );
    println!("└──────────────┴────────┴──────────┴──────┴────────┴──────┴────────┴──────┴────────┴────────┘");
    println!();

    // Risk summary
    println!("{}", "CROSS-STACK INTEGRATION RISKS:".bold());
    println!();

    let gpu_total = by_category.get(&DefectCategory::GpuKernelBugs).unwrap_or(&0);
    if *gpu_total > 0 {
        println!("  {} GPU Kernel Chain (trueno SIMD → realizar CUDA):", "1.".yellow());
        println!("     • {} GPU kernel bugs detected", gpu_total);
        println!("     • Impact: Potential performance degradation or kernel failures");
        println!();
    }

    let debt_total = by_category.get(&DefectCategory::HiddenDebt).unwrap_or(&0);
    if *debt_total > 0 {
        println!("  {} Hidden Technical Debt:", "2.".yellow());
        println!("     • {} euphemism patterns (placeholder, stub, etc.)", debt_total);
        println!("     • Impact: Incomplete implementations may cause failures");
        println!();
    }

    let test_total = by_category.get(&DefectCategory::TestDebt).unwrap_or(&0);
    if *test_total > 0 {
        println!("  {} Test Debt:", "3.".yellow());
        println!("     • {} tests ignored or removed", test_total);
        println!("     • Impact: Known bugs not being caught by CI");
        println!();
    }

    let contract_total = by_category.get(&DefectCategory::ContractGap).unwrap_or(&0);
    if *contract_total > 0 {
        println!("  {} Contract Verification Gaps:", "4.".yellow());
        println!("     • {} contract gaps (unbound, partial, missing proofs)", contract_total);
        println!("     • Impact: Kernel correctness claims lack formal verification");
        println!();
    }

    let parity_total = by_category.get(&DefectCategory::ModelParityGap).unwrap_or(&0);
    if *parity_total > 0 {
        println!("  {} Model Parity Gaps:", "5.".yellow());
        println!("     • {} parity gaps (missing oracles, failed claims)", parity_total);
        println!("     • Impact: Model conversion pipeline may produce incorrect results");
        println!();
    }
}

/// Output stack analysis as JSON.
pub(super) fn output_stack_json(crate_stats: &[CrateStats], results: &[(String, HuntResult)]) {
    use serde_json::json;

    let crates: Vec<_> = crate_stats
        .iter()
        .map(|s| {
            json!({
                "name": s.name,
                "total": s.total,
                "critical": s.critical,
                "high": s.high,
                "gpu": s.gpu,
                "debt": s.debt,
                "test": s.test,
                "silent": s.silent,
                "memory": s.memory,
                "contract": s.contract,
                "parity": s.parity
            })
        })
        .collect();

    let all_findings: Vec<_> = results
        .iter()
        .flat_map(|(name, r)| {
            r.findings.iter().map(move |f| {
                json!({
                    "crate": name,
                    "file": f.file,
                    "line": f.line,
                    "severity": format!("{:?}", f.severity),
                    "category": format!("{:?}", f.category),
                    "title": f.title,
                    "suspiciousness": f.suspiciousness
                })
            })
        })
        .collect();

    let totals = CrateStats::sum_all(crate_stats);
    let output = json!({
        "crates": crates,
        "findings": all_findings,
        "totals": {
            "findings": totals.total,
            "critical": totals.critical,
            "high": totals.high
        }
    });

    match serde_json::to_string_pretty(&output) {
        Ok(json) => println!("{json}"),
        Err(e) => eprintln!("Warning: JSON serialization failed: {e}"),
    }
}

/// Output GitHub issue body for cross-stack report.
pub(super) fn output_stack_issue(crate_stats: &[CrateStats], results: &[(String, HuntResult)]) {
    println!("{}", "--- GITHUB ISSUE BODY ---".dimmed());
    println!();
    println!("## Cross-Stack Bug Analysis - Sovereign AI Stack");
    println!();
    println!("### Summary by Crate");
    println!();
    println!("| Crate | Total | Critical | High | GPU | Debt | Test | Mem | Contract | Parity |");
    println!("|-------|-------|----------|------|-----|------|------|-----|----------|--------|");

    for s in crate_stats {
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            s.name,
            s.total,
            s.critical,
            s.high,
            s.gpu,
            s.debt,
            s.test,
            s.memory,
            s.contract,
            s.parity
        );
    }

    let totals = CrateStats::sum_all(crate_stats);
    println!(
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
        totals.name,
        totals.total,
        totals.critical,
        totals.high,
        totals.gpu,
        totals.debt,
        totals.test,
        totals.memory,
        totals.contract,
        totals.parity
    );

    println!();
    println!("### Critical Findings");
    println!();
    println!("```");
    for (crate_name, result) in results {
        for f in result.findings.iter().filter(|f| matches!(f.severity, FindingSeverity::Critical))
        {
            let file_name = f.file.file_name().map(|s| s.to_string_lossy()).unwrap_or_default();
            println!("{}: {}:{} - {}", crate_name, file_name, f.line, f.title);
        }
    }
    println!("```");
    println!();
    println!("*Generated by `batuta bug-hunter stack`*");
}

/// Output result in the specified format.
pub(super) fn output_result(result: &HuntResult, format: BugHunterOutputFormat) {
    match format {
        BugHunterOutputFormat::Text => output_text(result),
        BugHunterOutputFormat::Json => output_json(result),
        BugHunterOutputFormat::Sarif => output_sarif(result),
        BugHunterOutputFormat::Markdown => output_markdown(result),
    }
}

/// Severity badge: [C] bright_red bold, [H] red, [M] yellow, [L] blue, [I] dimmed.
pub(super) fn severity_badge(severity: &FindingSeverity) -> String {
    match severity {
        FindingSeverity::Critical => format!("{}", "[C]".bright_red().bold()),
        FindingSeverity::High => format!("{}", "[H]".red()),
        FindingSeverity::Medium => format!("{}", "[M]".yellow()),
        FindingSeverity::Low => format!("{}", "[L]".blue()),
        FindingSeverity::Info => format!("{}", "[I]".dimmed()),
    }
}

/// Suspiciousness bar: filled/empty blocks proportional to score.
fn suspiciousness_bar(score: f64, width: usize) -> String {
    let filled = (score * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("{}{} {:.2}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty), score)
}

/// Compact severity one-liner: "2C 5H 8M 0L 0I".
fn severity_summary_line(stats: &HuntStats) -> String {
    let c = stats.by_severity.get(&FindingSeverity::Critical).unwrap_or(&0);
    let h = stats.by_severity.get(&FindingSeverity::High).unwrap_or(&0);
    let m = stats.by_severity.get(&FindingSeverity::Medium).unwrap_or(&0);
    let l = stats.by_severity.get(&FindingSeverity::Low).unwrap_or(&0);
    let i = stats.by_severity.get(&FindingSeverity::Info).unwrap_or(&0);
    format!(
        "{}C {}H {}M {}L {}I",
        format!("{}", c).bright_red(),
        format!("{}", h).red(),
        format!("{}", m).yellow(),
        format!("{}", l).blue(),
        format!("{}", i).dimmed(),
    )
}

/// Print category distribution with horizontal bar chart.
fn print_category_distribution(stats: &HuntStats) {
    if stats.by_category.is_empty() {
        return;
    }
    println!("{}", "Category Distribution:".bold());
    let max_count = stats.by_category.values().max().copied().unwrap_or(1);
    let bar_width = 20;
    let mut sorted: Vec<_> = stats.by_category.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (cat, count) in &sorted {
        let bar_len = if max_count > 0 {
            (**count as f64 / max_count as f64 * bar_width as f64).round() as usize
        } else {
            0
        };
        println!("  {:<22} {} {}", format!("{}", cat), "\u{2588}".repeat(bar_len), count);
    }
    println!();
}

/// Print top-N hotspot files by finding count.
fn print_hotspot_files(findings: &[Finding], n: usize) {
    if findings.is_empty() {
        return;
    }
    let mut file_counts: std::collections::HashMap<&std::path::Path, usize> =
        std::collections::HashMap::new();
    for f in findings {
        *file_counts.entry(&f.file).or_default() += 1;
    }
    let mut sorted: Vec<_> = file_counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    let top = &sorted[..sorted.len().min(n)];
    if top.is_empty() {
        return;
    }
    let max_count = top[0].1;
    let bar_width = 15;

    println!("{}", "Hotspot Files:".bold());
    for (file, count) in top {
        let bar_len = if max_count > 0 {
            (*count as f64 / max_count as f64 * bar_width as f64).round() as usize
        } else {
            0
        };
        println!(
            "  {:<40} {} {}",
            format!("{}", file.display()).dimmed(),
            "\u{2588}".repeat(bar_len),
            count
        );
    }
    println!();
}

/// Print phase timing breakdown if any timings are non-zero.
fn print_phase_timings(pt: &PhaseTimings) {
    if pt.mode_dispatch_ms == 0
        && pt.pmat_index_ms == 0
        && pt.finalize_ms == 0
        && pt.contract_gap_ms == 0
        && pt.model_parity_ms == 0
    {
        return;
    }
    let mut parts = Vec::new();
    if pt.mode_dispatch_ms > 0 {
        parts.push(format!("scan={}ms", pt.mode_dispatch_ms));
    }
    if pt.pmat_index_ms > 0 {
        parts.push(format!("pmat-index={}ms", pt.pmat_index_ms));
    }
    if pt.pmat_weights_ms > 0 {
        parts.push(format!("weights={}ms", pt.pmat_weights_ms));
    }
    if pt.contract_gap_ms > 0 {
        parts.push(format!("contracts={}ms", pt.contract_gap_ms));
    }
    if pt.model_parity_ms > 0 {
        parts.push(format!("parity={}ms", pt.model_parity_ms));
    }
    if pt.finalize_ms > 0 {
        parts.push(format!("finalize={}ms", pt.finalize_ms));
    }
    println!("{}", parts.join("  ").dimmed());
}

/// Output as human-readable text.
fn output_text(result: &HuntResult) {
    let sep = "\u{2500}".repeat(74);

    // Header
    println!("\n{}", "Bug Hunter Report".bright_cyan().bold());
    println!("{}", sep.dimmed());
    println!(
        "Mode: {}  Findings: {}  Duration: {}ms",
        format!("{}", result.mode).bright_yellow(),
        result.findings.len(),
        result.duration_ms,
    );

    print_phase_timings(&result.phase_timings);

    // Severity summary

    println!("Severity: {}", severity_summary_line(&result.stats),);
    println!();

    // Category distribution chart
    print_category_distribution(&result.stats);

    // Hotspot files (top 5)
    print_hotspot_files(&result.findings, 5);

    // Findings list
    let top = result.top_findings(20);
    if top.is_empty() {
        println!("{}", "No findings discovered.".green());
    } else {
        println!("{}", "Findings:".bold());
        println!("{}", sep.dimmed());

        for finding in top {
            let badge = severity_badge(&finding.severity);
            let bar = suspiciousness_bar(finding.suspiciousness, 10);
            println!("{} {} {} {}", badge, finding.id.dimmed(), bar, finding.location());
            println!("    {}", finding.title.bright_white());
            if !finding.description.is_empty() {
                println!("    {}", finding.description.dimmed());
            }
            if let Some(risk) = finding.regression_risk {
                println!("    {}", format!("Regression Risk: {:.2}", risk).yellow());
            }
            // Display git blame info if available
            if let (Some(author), Some(commit), Some(date)) =
                (&finding.blame_author, &finding.blame_commit, &finding.blame_date)
            {
                println!("    {}", format!("Blame: {} ({}) {}", author, commit, date).dimmed());
            }
        }
    }

    // Summary
    println!("\n{}", sep.dimmed());
    println!("{}", result.summary().bold());
}

/// Output as JSON.
fn output_json(result: &HuntResult) {
    match serde_json::to_string_pretty(result) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("Error serializing to JSON: {}", e),
    }
}

/// Output as SARIF (Static Analysis Results Interchange Format).
pub(super) fn output_sarif(result: &HuntResult) {
    let sarif = build_sarif(result);
    match serde_json::to_string_pretty(&sarif) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("Error serializing SARIF: {}", e),
    }
}

/// Build SARIF structure.
fn build_sarif(result: &HuntResult) -> serde_json::Value {
    let results: Vec<serde_json::Value> = result
        .findings
        .iter()
        .map(|f| {
            serde_json::json!({
                "ruleId": f.id,
                "level": match f.severity {
                    FindingSeverity::Critical | FindingSeverity::High => "error",
                    FindingSeverity::Medium => "warning",
                    FindingSeverity::Low | FindingSeverity::Info => "note",
                },
                "message": {
                    "text": f.title
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": f.file.display().to_string()
                        },
                        "region": {
                            "startLine": f.line,
                            "startColumn": f.column.unwrap_or(1)
                        }
                    }
                }],
                "properties": {
                    "suspiciousness": f.suspiciousness,
                    "category": format!("{:?}", f.category),
                    "discoveredBy": format!("{}", f.discovered_by)
                }
            })
        })
        .collect();

    serde_json::json!({
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "batuta bug-hunter",
                    "version": env!("CARGO_PKG_VERSION"),
                    "informationUri": "https://github.com/paiml/batuta"
                }
            },
            "results": results
        }]
    })
}

/// Output as Markdown.
fn output_markdown(result: &HuntResult) {
    println!("# Bug Hunter Report\n");
    println!(
        "**Mode:** {} | **Duration:** {}ms | **Findings:** {}\n",
        result.mode,
        result.duration_ms,
        result.findings.len()
    );

    println!("## Statistics\n");
    println!("| Severity | Count |");
    println!("|----------|-------|");
    for (severity, count) in &result.stats.by_severity {
        println!("| {:?} | {} |", severity, count);
    }

    println!("\n## Top Findings\n");
    println!("| ID | Severity | Category | Score | Risk | Location |");
    println!("|-----|----------|----------|-------|------|----------|");

    for finding in result.top_findings(20) {
        let risk =
            finding.regression_risk.map(|r| format!("{:.2}", r)).unwrap_or_else(|| "-".to_string());
        println!(
            "| {} | {:?} | {:?} | {:.2} | {} | `{}` |",
            finding.id,
            finding.severity,
            finding.category,
            finding.suspiciousness,
            risk,
            finding.location()
        );
    }

    println!("\n---\n");
    println!("*{}*", result.summary());
}

/// Handle the diff command - show only new findings.
pub(super) fn handle_diff_command(
    path: PathBuf,
    base: Option<String>,
    since: Option<String>,
    min_suspiciousness: f64,
    format: BugHunterOutputFormat,
    save_baseline: bool,
) -> Result<(), String> {
    use crate::bug_hunter::diff::{Baseline, DiffResult};

    // Run current analysis
    let config = HuntConfig { mode: HuntMode::Analyze, min_suspiciousness, ..Default::default() };
    let result = hunt(&path, config);

    // Save baseline if requested
    if save_baseline {
        let baseline = Baseline::from_findings(&result.findings);
        baseline.save(&path)?;
        println!("{}", format!("Saved baseline with {} findings", result.findings.len()).green());
        return Ok(());
    }

    // Load existing baseline
    let baseline = Baseline::load(&path);

    if baseline.is_none() && base.is_none() && since.is_none() {
        println!(
            "{}",
            "No baseline found. Run with --save-baseline first, or use --base/--since.".yellow()
        );
        output_result(&result, format);
        return Ok(());
    }

    // Compute diff
    let base_ref = base.as_deref().or(since.as_deref()).unwrap_or("baseline");

    if let Some(baseline) = baseline {
        let diff = DiffResult::compute(&result, &baseline, base_ref);
        output_diff_result(&diff, format);
    } else {
        // Use git diff for changed files
        let changed =
            crate::bug_hunter::diff::get_changed_files(&path, base.as_deref(), since.as_deref());
        let filtered = crate::bug_hunter::diff::filter_changed_files(&result.findings, &changed);

        println!(
            "{}",
            format!(
                "Showing {} findings in {} changed files ({})",
                filtered.len(),
                changed.len(),
                base_ref
            )
            .bright_cyan()
            .bold()
        );
        println!();

        for f in &filtered {
            println!(
                "{} {} {}:{}",
                severity_badge(&f.severity),
                f.id.dimmed(),
                f.file.display(),
                f.line
            );
            println!("    {}", f.title.white());
        }
    }

    Ok(())
}

/// Output diff result.
fn output_diff_result(diff: &crate::bug_hunter::diff::DiffResult, format: BugHunterOutputFormat) {
    match format {
        BugHunterOutputFormat::Json => {
            use serde_json::json;
            let output = json!({
                "new_findings": diff.new_findings.len(),
                "resolved": diff.resolved_count,
                "total_current": diff.total_current,
                "total_baseline": diff.total_baseline,
                "base": diff.base_reference,
                "findings": diff.new_findings.iter().map(|f| json!({
                    "file": f.file,
                    "line": f.line,
                    "severity": format!("{:?}", f.severity),
                    "title": f.title
                })).collect::<Vec<_>>()
            });
            match serde_json::to_string_pretty(&output) {
                Ok(json) => println!("{json}"),
                Err(e) => eprintln!("Warning: JSON serialization failed: {e}"),
            }
        }
        _ => {
            println!(
                "{}",
                "╔══════════════════════════════════════════════════════════════════════════╗"
                    .bright_cyan()
            );
            println!(
                "{}",
                format!("║                    DIFF vs {} {:>30}║", diff.base_reference, "")
                    .bright_cyan()
                    .bold()
            );
            println!(
                "{}",
                "╚══════════════════════════════════════════════════════════════════════════╝"
                    .bright_cyan()
            );
            println!();

            // Summary
            let new_color = if diff.new_findings.is_empty() {
                "0".green()
            } else {
                format!("{}", diff.new_findings.len()).red().bold()
            };
            let resolved_color = if diff.resolved_count == 0 {
                "0".dimmed()
            } else {
                format!("{}", diff.resolved_count).green()
            };

            println!("  {} new findings, {} resolved", new_color, resolved_color);
            println!(
                "  {} total (was {})",
                diff.total_current.to_string().white(),
                diff.total_baseline.to_string().dimmed()
            );
            println!();

            if diff.new_findings.is_empty() {
                println!("{}", "  No new findings! 🎉".green());
            } else {
                println!("{}", "NEW FINDINGS:".bold());
                println!();
                for f in &diff.new_findings {
                    println!(
                        "{} {} {}:{}",
                        severity_badge(&f.severity),
                        f.id.dimmed(),
                        f.file.display(),
                        f.line
                    );
                    println!("    {}", f.title.white());
                }
            }
        }
    }
}

/// Handle trend command - show tech debt over time.
pub(super) fn handle_trend_command(
    path: PathBuf,
    weeks: usize,
    format: BugHunterOutputFormat,
) -> Result<(), String> {
    let trend_path = path.join(".pmat").join("bug-hunter-trend.json");

    // Load existing trend data
    let trend_data: Vec<TrendSnapshot> = if trend_path.exists() {
        let content = std::fs::read_to_string(&trend_path)
            .map_err(|e| format!("Failed to read trend data: {}", e))?;
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        Vec::new()
    };

    // Run current analysis and add snapshot
    let config =
        HuntConfig { mode: HuntMode::Analyze, min_suspiciousness: 0.3, ..Default::default() };
    let result = hunt(&path, config);

    let current = TrendSnapshot {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        total: result.findings.len(),
        critical: result.stats.by_severity.get(&FindingSeverity::Critical).copied().unwrap_or(0),
        high: result.stats.by_severity.get(&FindingSeverity::High).copied().unwrap_or(0),
        debt: result.stats.by_category.get(&DefectCategory::HiddenDebt).copied().unwrap_or(0),
    };

    // Save updated trend
    let mut updated_trend = trend_data.clone();
    updated_trend.push(current.clone());
    // Keep only recent snapshots
    if updated_trend.len() > weeks {
        updated_trend = updated_trend.split_off(updated_trend.len() - weeks);
    }
    let pmat_dir = path.join(".pmat");
    std::fs::create_dir_all(&pmat_dir).ok();
    match serde_json::to_string_pretty(&updated_trend) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&trend_path, &json) {
                eprintln!("Warning: failed to write trend file: {e}");
            }
        }
        Err(e) => eprintln!("Warning: JSON serialization failed for trend: {e}"),
    }

    // Output trend
    match format {
        BugHunterOutputFormat::Json => match serde_json::to_string_pretty(&updated_trend) {
            Ok(json) => println!("{json}"),
            Err(e) => eprintln!("Warning: JSON serialization failed: {e}"),
        },
        _ => {
            println!("{}", "TECH DEBT TREND".bold());
            println!();
            println!(
                "{:12} {:>8} {:>8} {:>8} {:>8}  {}",
                "Date".dimmed(),
                "Total".dimmed(),
                "Critical".dimmed(),
                "High".dimmed(),
                "Debt".dimmed(),
                "Trend".dimmed()
            );

            let max_total = updated_trend.iter().map(|s| s.total).max().unwrap_or(1);
            for snapshot in &updated_trend {
                let date = chrono::DateTime::from_timestamp(snapshot.timestamp as i64, 0)
                    .map(|d| d.format("%Y-%m-%d").to_string())
                    .unwrap_or_else(|| "unknown".to_string());

                let bar_len = (snapshot.total * 20 / max_total.max(1)).min(20);
                let bar = "━".repeat(bar_len);

                println!(
                    "{:12} {:>8} {:>8} {:>8} {:>8}  {}",
                    date,
                    snapshot.total,
                    snapshot.critical,
                    snapshot.high,
                    snapshot.debt,
                    bar.yellow()
                );
            }
        }
    }

    Ok(())
}

/// Trend snapshot for tracking over time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TrendSnapshot {
    timestamp: u64,
    total: usize,
    critical: usize,
    high: usize,
    debt: usize,
}

/// Handle triage command - group related findings.
pub(super) fn handle_triage_command(
    path: PathBuf,
    min_suspiciousness: f64,
    format: BugHunterOutputFormat,
) -> Result<(), String> {
    use std::collections::HashMap;

    let config = HuntConfig { mode: HuntMode::Analyze, min_suspiciousness, ..Default::default() };
    let result = hunt(&path, config);

    // Group by directory + pattern
    let mut groups: HashMap<String, Vec<&Finding>> = HashMap::new();
    for f in &result.findings {
        let dir = f
            .file
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        // Extract pattern from title
        let pattern = f
            .title
            .strip_prefix("Pattern: ")
            .or_else(|| f.title.strip_prefix("Custom: "))
            .unwrap_or(&f.title);
        let key = format!("{}:{}", dir, pattern);
        groups.entry(key).or_default().push(f);
    }

    // Sort groups by count
    let mut sorted_groups: Vec<_> = groups.into_iter().collect();
    sorted_groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    match format {
        BugHunterOutputFormat::Json => {
            use serde_json::json;
            let output: Vec<_> = sorted_groups
                .iter()
                .map(|(key, findings)| {
                    json!({
                        "root_cause": key,
                        "count": findings.len(),
                        "findings": findings.iter().map(|f| json!({
                            "file": f.file,
                            "line": f.line,
                        })).collect::<Vec<_>>()
                    })
                })
                .collect();
            match serde_json::to_string_pretty(&output) {
                Ok(json) => println!("{json}"),
                Err(e) => eprintln!("Warning: JSON serialization failed: {e}"),
            }
        }
        _ => {
            println!("{}", "AUTO-TRIAGE: Findings by Root Cause".bold());
            println!();

            for (key, findings) in sorted_groups.iter().take(20) {
                let parts: Vec<&str> = key.splitn(2, ':').collect();
                let (dir, pattern) =
                    if parts.len() == 2 { (parts[0], parts[1]) } else { (".", key.as_str()) };

                println!(
                    "{} ({} findings)",
                    format!("{}:{}", dir, pattern).white().bold(),
                    findings.len().to_string().yellow()
                );

                for f in findings.iter().take(3) {
                    let file_name =
                        f.file.file_name().map(|s| s.to_string_lossy()).unwrap_or_default();
                    println!("  {} {}:{}", severity_badge(&f.severity), file_name.dimmed(), f.line);
                }
                if findings.len() > 3 {
                    println!("  {} more...", format!("... +{}", findings.len() - 3).dimmed());
                }
                println!();
            }
        }
    }

    Ok(())
}
