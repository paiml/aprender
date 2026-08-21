use std::path::Path;

use provable_contracts::lint::config::{find_config, load_config};
use provable_contracts::lint::rules::RuleSeverity;
use provable_contracts::lint::trend;
use provable_contracts::lint::{run_lint, GateDetail, LintConfig, LintReport};

#[path = "lint_render.rs"]
mod lint_render;

#[path = "lint_html.rs"]
mod lint_html;

/// Print long-form explanation for a lint rule.
pub fn explain_rule(rule_id: &str) {
    lint_render::print_explain(rule_id);
}

#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub fn run(
    contract_dir: &Path,
    binding_path: Option<&Path>,
    min_score: f64,
    format: Option<&str>,
    severity: Option<&str>,
    strict: bool,
    suppress: Option<&str>,
    suppress_rule: Option<&str>,
    suppress_file: Option<&str>,
    rule_overrides: &[String],
    config_path: Option<&Path>,
    diff_ref: Option<&str>,
    do_trend: bool,
    show_trend: bool,
    no_cache: bool,
    cache_stats: bool,
    coverage: bool,
    min_coverage: Option<f64>,
    crate_dir: Option<&Path>,
    min_level: Option<&str>,
    watch: bool,
    strict_test_binding: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if watch {
        return run_watch(
            contract_dir,
            binding_path,
            min_score,
            format,
            severity,
            strict,
            suppress,
            suppress_rule,
            suppress_file,
            rule_overrides,
            config_path,
            no_cache,
            cache_stats,
            crate_dir,
            min_level,
            strict_test_binding,
        );
    }

    if show_trend {
        show_trend_history(contract_dir);
        return Ok(());
    }

    if let Some(base) = diff_ref {
        if let Some(result) = run_diff_check(contract_dir, base) {
            return result;
        }
    }

    let config = build_config(
        contract_dir,
        binding_path,
        min_score,
        format,
        severity,
        strict,
        suppress,
        suppress_rule,
        suppress_file,
        rule_overrides,
        config_path,
        no_cache,
        cache_stats,
        crate_dir,
        min_level,
        strict_test_binding,
    );

    let report = run_lint(&config);

    if cache_stats {
        print_cache_stats(&report);
    }
    if do_trend {
        record_trend(contract_dir, &report);
    }

    let effective_format = resolve_format(format, config_path, contract_dir);
    print_report(&effective_format, &report)?;

    // --coverage: compute and print aggregate contract coverage metric
    if coverage {
        let coverage_result = compute_contract_coverage(contract_dir);
        println!(
            "\nContract Coverage: {}/{} at Standard+ ({:.1}%)",
            coverage_result.standard_plus, coverage_result.total, coverage_result.percentage,
        );
        if let Some(threshold) = min_coverage {
            if coverage_result.percentage < threshold {
                return Err(format!(
                    "contract coverage {:.1}% is below minimum {:.1}%",
                    coverage_result.percentage, threshold,
                )
                .into());
            }
        }
    }

    if report.passed {
        Ok(())
    } else {
        let passed_count = report.gates.iter().filter(|g| g.passed).count();
        Err(format!(
            "lint failed ({}/{} gates passed)",
            passed_count,
            report.gates.len()
        )
        .into())
    }
}

struct CoverageResult {
    standard_plus: usize,
    total: usize,
    percentage: f64,
}

/// Count contracts at `Standard+` level (have both `falsification_tests` AND `kani_harnesses`).
fn compute_contract_coverage(contract_dir: &Path) -> CoverageResult {
    let mut total = 0usize;
    let mut standard_plus = 0usize;

    let mut yaml_paths = Vec::new();
    collect_yaml_files_lint(contract_dir, &mut yaml_paths);

    for path in &yaml_paths {
        let Ok(contract) = provable_contracts::schema::parse_contract(path) else {
            continue;
        };
        // Coverage is a kernel-contract metric. Skip registries,
        // model-family schemas, pattern contracts, and reference documents.
        if !contract.requires_proofs() {
            continue;
        }
        total += 1;
        if !contract.falsification_tests.is_empty() && !contract.kani_harnesses.is_empty() {
            standard_plus += 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let percentage = if total > 0 {
        (standard_plus as f64 / total as f64) * 100.0
    } else {
        100.0
    };

    CoverageResult {
        standard_plus,
        total,
        percentage,
    }
}

fn show_trend_history(contract_dir: &Path) {
    let trend_root = trend::trend_dir(contract_dir);
    let snapshots = trend::load_snapshots(&trend_root);
    if snapshots.is_empty() {
        println!("No trend data. Run `pv lint --trend` to record snapshots.");
    } else {
        println!("{}", trend::format_trend(&snapshots, 30));
    }
}

/// Returns `Some(Ok(()))` to short-circuit when no contracts changed,
/// or `None` to continue with full lint.
fn run_diff_check(
    contract_dir: &Path,
    base: &str,
) -> Option<Result<(), Box<dyn std::error::Error>>> {
    match provable_contracts::lint::diff::changed_contracts(contract_dir, base) {
        Ok(changed) if changed.is_empty() => {
            println!("No contracts changed since {base}. Nothing to lint.");
            Some(Ok(()))
        }
        Ok(changed) => {
            println!(
                "Diff-aware: {} contracts changed since {base}",
                changed.len()
            );
            for stem in &changed {
                println!("  {stem}");
            }
            println!();
            None
        }
        Err(e) => {
            eprintln!("Warning: diff-aware mode failed ({e}), linting all contracts");
            None
        }
    }
}

fn print_cache_stats(report: &LintReport) {
    eprintln!(
        "Cache: {} total, {} hits, {} misses ({:.0}% hit rate)",
        report.cache_stats.total,
        report.cache_stats.hits,
        report.cache_stats.misses,
        report.cache_stats.hit_rate() * 100.0,
    );
}

fn record_trend(contract_dir: &Path, report: &LintReport) {
    let trend_root = trend::trend_dir(contract_dir);
    let contracts_count = count_contracts(report);
    match trend::record_snapshot(&trend_root, report, contracts_count) {
        Ok(path) => eprintln!("Trend snapshot saved: {}", path.display()),
        Err(e) => eprintln!("Warning: failed to save trend snapshot: {e}"),
    }
    let snapshots = trend::load_snapshots(&trend_root);
    if let Some(drop) = trend::detect_drift(&snapshots, 0.05) {
        eprintln!("Warning: quality drift detected (score dropped {drop:.3})");
    }
}

fn print_report(format: &str, report: &LintReport) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        "json" => lint_render::print_json(report)?,
        "sarif" => lint_render::print_sarif(report),
        "github" => lint_render::print_github(report),
        "html" => println!("{}", lint_html::render_html(report)),
        _ => lint_render::print_text(report),
    }
    Ok(())
}

fn count_contracts(report: &LintReport) -> usize {
    for gate in &report.gates {
        match &gate.detail {
            GateDetail::Validate { contracts, .. }
            | GateDetail::Audit { contracts, .. }
            | GateDetail::Score { contracts, .. } => return *contracts,
            GateDetail::Verify { .. }
            | GateDetail::Enforce { .. }
            | GateDetail::ReverseCoverage { .. }
            | GateDetail::Composition { .. }
            | GateDetail::DuplicateStems { .. }
            | GateDetail::Skipped { .. } => {}
        }
    }
    0
}

/// Watch mode: polling-based re-lint every 5 seconds.
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn run_watch(
    contract_dir: &Path,
    binding_path: Option<&Path>,
    min_score: f64,
    format: Option<&str>,
    severity: Option<&str>,
    strict: bool,
    suppress: Option<&str>,
    suppress_rule: Option<&str>,
    suppress_file: Option<&str>,
    rule_overrides: &[String],
    config_path: Option<&Path>,
    no_cache: bool,
    cache_stats: bool,
    crate_dir: Option<&Path>,
    min_level: Option<&str>,
    strict_test_binding: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let config = build_config(
            contract_dir,
            binding_path,
            min_score,
            format,
            severity,
            strict,
            suppress,
            suppress_rule,
            suppress_file,
            rule_overrides,
            config_path,
            no_cache,
            cache_stats,
            crate_dir,
            min_level,
            strict_test_binding,
        );

        let report = run_lint(&config);

        if cache_stats {
            print_cache_stats(&report);
        }

        let effective_format = resolve_format(format, config_path, contract_dir);
        print_report(&effective_format, &report)?;

        println!("\n--- Watching for changes (Ctrl+C to stop) ---\n");
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}

fn resolve_format(format: Option<&str>, config_path: Option<&Path>, contract_dir: &Path) -> String {
    // CLI flag takes precedence when explicitly set
    if let Some(f) = format {
        return f.to_string();
    }
    // Fall back to config file
    let pv_config = config_path
        .and_then(|cp| load_config(cp).ok())
        .or_else(|| find_config(contract_dir).and_then(|p| load_config(&p).ok()))
        .unwrap_or_default();
    pv_config.output.format.unwrap_or_else(|| "text".into())
}

#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn build_config<'a>(
    contract_dir: &'a Path,
    binding_path: Option<&'a Path>,
    min_score: f64,
    _format: Option<&str>,
    severity: Option<&str>,
    strict: bool,
    suppress: Option<&str>,
    suppress_rule: Option<&str>,
    suppress_file: Option<&str>,
    rule_overrides: &[String],
    config_path: Option<&Path>,
    no_cache: bool,
    cache_stats: bool,
    crate_dir: Option<&'a Path>,
    min_level: Option<&str>,
    strict_test_binding: bool,
) -> LintConfig<'a> {
    let pv_config = config_path
        .and_then(|cp| match load_config(cp) {
            Ok(c) => {
                if c.lint.min_score.is_none()
                    && !c.lint.strict
                    && c.lint.severity.is_none()
                    && c.lint.rules.is_empty()
                    && c.output.format.is_none()
                {
                    eprintln!(
                        "Warning: config {} parsed but no lint settings found",
                        cp.display()
                    );
                }
                Some(c)
            }
            Err(e) => {
                eprintln!("Warning: failed to load config {}: {e}", cp.display());
                None
            }
        })
        .or_else(|| find_config(contract_dir).and_then(|p| load_config(&p).ok()))
        .unwrap_or_default();

    let effective_min_score = if min_score > 0.0 {
        min_score
    } else {
        pv_config.lint.min_score.unwrap_or(0.0)
    };

    let severity_filter = severity
        .or(pv_config.lint.severity.as_deref())
        .and_then(RuleSeverity::from_str_opt);

    let effective_strict = strict || pv_config.lint.strict;

    let suppressed_findings: Vec<String> = parse_csv(suppress)
        .into_iter()
        .chain(pv_config.lint.suppress.findings.iter().cloned())
        .collect();
    let suppressed_rules: Vec<String> = parse_csv(suppress_rule)
        .into_iter()
        .chain(pv_config.lint.suppress.rules.iter().cloned())
        .collect();
    let suppressed_files: Vec<String> = parse_csv(suppress_file)
        .into_iter()
        .chain(pv_config.lint.suppress.files.iter().cloned())
        .collect();

    let mut severity_overrides = std::collections::HashMap::new();
    for entry in &pv_config.lint.rules {
        if let Some(sev) = RuleSeverity::from_str_opt(entry.1) {
            severity_overrides.insert(entry.0.clone(), sev);
        }
    }
    for r in rule_overrides {
        if let Some((id, sev_str)) = r.split_once('=') {
            if let Some(sev) = RuleSeverity::from_str_opt(sev_str) {
                severity_overrides.insert(id.to_string(), sev);
            }
        }
    }

    LintConfig {
        contract_dir,
        binding_path,
        min_score: effective_min_score,
        severity_filter,
        severity_overrides,
        suppressed_findings,
        suppressed_rules,
        suppressed_files,
        strict: effective_strict,
        no_cache,
        cache_stats,
        crate_dir,
        min_level: min_level.and_then(parse_enforcement_level),
        strict_test_binding,
    }
}

fn parse_enforcement_level(s: &str) -> Option<provable_contracts::schema::EnforcementLevel> {
    use provable_contracts::schema::EnforcementLevel;
    match s.to_lowercase().as_str() {
        "basic" => Some(EnforcementLevel::Basic),
        "standard" => Some(EnforcementLevel::Standard),
        "strict" => Some(EnforcementLevel::Strict),
        "proven" => Some(EnforcementLevel::Proven),
        _ => None,
    }
}

fn parse_csv(s: Option<&str>) -> Vec<String> {
    s.map(|v| {
        v.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    })
    .unwrap_or_default()
}

/// Recursively collect `.yaml` contract files, skipping non-contract directories.
fn collect_yaml_files_lint(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let dirname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if dirname == "kaizen" || dirname == "legacy" || dirname == "pipelines" {
                continue;
            }
            collect_yaml_files_lint(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("yaml")
            && !matches!(
                path.file_name().and_then(|n| n.to_str()),
                Some("binding.yaml" | "binding.yml")
            )
        {
            out.push(path);
        }
    }
}
