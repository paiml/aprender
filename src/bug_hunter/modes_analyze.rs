//! BH-03: LLM-augmented static analysis (LLIFT pattern).

use super::config;
use super::defect_patterns;
use super::pmat_quality;
use super::types::*;
use std::path::Path;

/// BH-03: LLM-augmented static analysis (LLIFT pattern)
pub(super) fn run_analyze_mode(
    project_path: &Path,
    config_val: &HuntConfig,
    result: &mut HuntResult,
) {
    let clippy_output = std::process::Command::new("cargo")
        .args(["clippy", "--all-targets", "--message-format=json"])
        .current_dir(project_path)
        .output();

    let clippy_json = match clippy_output {
        Ok(output) => String::from_utf8_lossy(&output.stdout).to_string(),
        Err(_) => {
            result.add_finding(
                Finding::new(
                    "BH-ANALYZE-NOCLIPPY",
                    project_path.join("Cargo.toml"),
                    1,
                    "Clippy not available",
                )
                .with_severity(FindingSeverity::Info)
                .with_discovered_by(HuntMode::Analyze),
            );
            return;
        }
    };

    let mut finding_id = 0;
    for line in clippy_json.lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(finding) = extract_clippy_finding(&msg, config_val, &mut finding_id) {
                result.add_finding(finding);
            }
        }
    }

    analyze_common_patterns(project_path, config_val, result);
}

/// Extract a finding from a single clippy JSON message, if applicable.
pub(super) fn extract_clippy_finding(
    msg: &serde_json::Value,
    config_val: &HuntConfig,
    finding_id: &mut usize,
) -> Option<Finding> {
    if msg.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
        return None;
    }
    let message = msg.get("message")?;
    let level = message.get("level").and_then(|l| l.as_str()).unwrap_or("");
    if level != "warning" && level != "error" {
        return None;
    }
    let spans = message.get("spans").and_then(|s| s.as_array())?;
    let span = spans.first()?;
    let file = span.get("file_name").and_then(|f| f.as_str()).unwrap_or("unknown");
    let line_start = span.get("line_start").and_then(|l| l.as_u64()).unwrap_or(1) as usize;
    let msg_text = message.get("message").and_then(|m| m.as_str()).unwrap_or("Unknown warning");
    let code = message
        .get("code")
        .and_then(|c| c.get("code"))
        .and_then(|c| c.as_str())
        .unwrap_or("unknown");

    if code == "dead_code" || code == "unused_imports" {
        return None;
    }

    let (category, severity) = categorize_clippy_warning(code, msg_text);
    let suspiciousness = match severity {
        FindingSeverity::Critical => 0.95,
        FindingSeverity::High => 0.8,
        FindingSeverity::Medium => 0.6,
        FindingSeverity::Low => 0.4,
        FindingSeverity::Info => 0.2,
    };

    if suspiciousness < config_val.min_suspiciousness {
        return None;
    }

    *finding_id += 1;
    Some(
        Finding::new(format!("BH-CLIP-{:04}", finding_id), file, line_start, msg_text)
            .with_severity(severity)
            .with_category(category)
            .with_suspiciousness(suspiciousness)
            .with_discovered_by(HuntMode::Analyze)
            .with_evidence(FindingEvidence::static_analysis("clippy", code)),
    )
}

/// Categorize clippy warning by code.
pub(super) fn categorize_clippy_warning(
    code: &str,
    _message: &str,
) -> (DefectCategory, FindingSeverity) {
    match code {
        // Memory safety
        c if c.contains("ptr") || c.contains("mem") || c.contains("uninit") => {
            (DefectCategory::MemorySafety, FindingSeverity::High)
        }
        // Concurrency
        c if c.contains("mutex")
            || c.contains("arc")
            || c.contains("send")
            || c.contains("sync") =>
        {
            (DefectCategory::ConcurrencyBugs, FindingSeverity::High)
        }
        // Security
        c if c.contains("unsafe") || c.contains("transmute") => {
            (DefectCategory::SecurityVulnerabilities, FindingSeverity::High)
        }
        // Logic errors
        c if c.contains("unwrap") || c.contains("expect") || c.contains("panic") => {
            (DefectCategory::LogicErrors, FindingSeverity::Medium)
        }
        // Type errors
        c if c.contains("cast") || c.contains("as_") || c.contains("into") => {
            (DefectCategory::TypeErrors, FindingSeverity::Medium)
        }
        // Default
        _ => (DefectCategory::Unknown, FindingSeverity::Low),
    }
}

/// Parse defect category from string (for custom config patterns).
pub(super) fn parse_defect_category(s: &str) -> DefectCategory {
    match s.to_lowercase().as_str() {
        "logicerrors" | "logic" => DefectCategory::LogicErrors,
        "memorysafety" | "memory" => DefectCategory::MemorySafety,
        "concurrency" | "concurrencybugs" => DefectCategory::ConcurrencyBugs,
        "gpukernelbugs" | "gpu" => DefectCategory::GpuKernelBugs,
        "silentdegradation" | "silent" => DefectCategory::SilentDegradation,
        "testdebt" | "test" => DefectCategory::TestDebt,
        "hiddendebt" | "debt" => DefectCategory::HiddenDebt,
        "performanceissues" | "performance" => DefectCategory::PerformanceIssues,
        "securityvulnerabilities" | "security" => DefectCategory::SecurityVulnerabilities,
        "contractgap" | "contract" => DefectCategory::ContractGap,
        "modelparitygap" | "modelparity" | "parity" => DefectCategory::ModelParityGap,
        _ => DefectCategory::LogicErrors,
    }
}

/// Parse finding severity from string (for custom config patterns).
pub(super) fn parse_finding_severity(s: &str) -> FindingSeverity {
    match s.to_lowercase().as_str() {
        "critical" => FindingSeverity::Critical,
        "high" => FindingSeverity::High,
        "medium" => FindingSeverity::Medium,
        "low" => FindingSeverity::Low,
        "info" => FindingSeverity::Info,
        _ => FindingSeverity::Medium,
    }
}

/// Context for pattern matching on a single source line.
pub(super) struct PatternMatchContext<'a> {
    pub(super) line: &'a str,
    pub(super) line_num: usize,
    pub(super) entry: &'a Path,
    pub(super) in_test_code: bool,
    pub(super) is_bug_hunter_file: bool,
    pub(super) bh_config: &'a config::BugHunterConfig,
    pub(super) min_susp: f64,
}

/// Check a single line against a language pattern, returning a finding if matched.
pub(super) fn match_lang_pattern(
    ctx: &PatternMatchContext<'_>,
    pattern: &str,
    category: DefectCategory,
    severity: FindingSeverity,
    suspiciousness: f64,
) -> Option<Finding> {
    use super::patterns::{is_real_pattern, should_suppress_finding};

    if ctx.in_test_code
        && category != DefectCategory::TestDebt
        && category != DefectCategory::GpuKernelBugs
        && category != DefectCategory::HiddenDebt
    {
        return None;
    }
    if ctx.is_bug_hunter_file && category == DefectCategory::HiddenDebt {
        return None;
    }
    if ctx.bh_config.is_allowed(ctx.entry, pattern, ctx.line_num) {
        return None;
    }
    if !ctx.line.contains(pattern)
        || !is_real_pattern(ctx.line, pattern)
        || suspiciousness < ctx.min_susp
    {
        return None;
    }
    let finding =
        Finding::new(String::new(), ctx.entry, ctx.line_num, format!("Pattern: {}", pattern))
            .with_description(ctx.line.trim().to_string())
            .with_severity(severity)
            .with_category(category)
            .with_suspiciousness(suspiciousness)
            .with_discovered_by(HuntMode::Analyze)
            .with_evidence(FindingEvidence::static_analysis("pattern", pattern));
    if should_suppress_finding(&finding, ctx.line) {
        None
    } else {
        Some(finding)
    }
}

/// Check a single line against a custom pattern, returning a finding if matched.
pub(super) fn match_custom_pattern(
    ctx: &PatternMatchContext<'_>,
    pattern: &str,
    category: DefectCategory,
    severity: FindingSeverity,
    suspiciousness: f64,
) -> Option<Finding> {
    use super::patterns::should_suppress_finding;

    if suspiciousness < ctx.min_susp || ctx.bh_config.is_allowed(ctx.entry, pattern, ctx.line_num) {
        return None;
    }
    if !ctx.line.contains(pattern) {
        return None;
    }
    let finding =
        Finding::new(String::new(), ctx.entry, ctx.line_num, format!("Custom: {}", pattern))
            .with_description(ctx.line.trim().to_string())
            .with_severity(severity)
            .with_category(category)
            .with_suspiciousness(suspiciousness)
            .with_discovered_by(HuntMode::Analyze)
            .with_evidence(FindingEvidence::static_analysis("custom_pattern", pattern));
    if should_suppress_finding(&finding, ctx.line) {
        None
    } else {
        Some(finding)
    }
}

/// Scan a single file for pattern matches.
pub(super) fn scan_file_for_patterns(
    entry: &std::path::Path,
    patterns: &[(&str, DefectCategory, FindingSeverity, f64)],
    custom_patterns: &[(String, DefectCategory, FindingSeverity, f64)],
    bh_config: &config::BugHunterConfig,
    min_susp: f64,
    findings: &mut Vec<Finding>,
) {
    use super::languages;
    use super::patterns::compute_test_lines;

    let Ok(content) = std::fs::read_to_string(entry) else {
        return;
    };
    let test_lines = compute_test_lines(&content);
    let lang =
        entry.extension().and_then(|e| e.to_str()).and_then(languages::Language::from_extension);
    let lang_patterns = lang
        .map(languages::patterns_for_language)
        .unwrap_or_else(|| patterns.iter().map(|&(p, c, s, su)| (p, c, s, su)).collect());
    let is_bug_hunter_file = entry.to_str().map(|p| p.contains("bug_hunter")).unwrap_or(false);

    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1;
        let ctx = PatternMatchContext {
            line,
            line_num,
            entry,
            in_test_code: test_lines.contains(&line_num),
            is_bug_hunter_file,
            bh_config,
            min_susp,
        };

        for &(pattern, category, severity, suspiciousness) in &lang_patterns {
            if let Some(f) = match_lang_pattern(&ctx, pattern, category, severity, suspiciousness) {
                findings.push(f);
            }
        }

        for (pattern, category, severity, suspiciousness) in custom_patterns {
            if let Some(f) =
                match_custom_pattern(&ctx, pattern.as_str(), *category, *severity, *suspiciousness)
            {
                findings.push(f);
            }
        }
    }
}

/// BH-23 helper: Generate SATD findings from PMAT quality index.
fn run_pmat_satd_phase(
    pmat_satd_active: bool,
    project_path: &Path,
    config_val: &HuntConfig,
    result: &mut HuntResult,
) {
    if !pmat_satd_active {
        return;
    }
    let query = config_val.pmat_query.as_deref().unwrap_or("*");
    if let Some(index) = pmat_quality::build_quality_index(project_path, query, 200) {
        let satd_findings = pmat_quality::generate_satd_findings(project_path, &index);
        for f in satd_findings {
            result.add_finding(f);
        }
    }
}

/// Analyze common bug patterns across source files.
pub(super) fn analyze_common_patterns(
    project_path: &Path,
    config_val: &HuntConfig,
    result: &mut HuntResult,
) {
    use super::blame;
    use super::languages;

    // Load bug-hunter config for allowlist and custom patterns
    let bh_config = config::BugHunterConfig::load(project_path);

    // BH-23: If PMAT SATD is enabled and pmat is available, generate SATD findings
    // and skip the manual TODO/FIXME/HACK/XXX pattern matching
    let pmat_satd_active = config_val.pmat_satd && pmat_quality::pmat_available();
    run_pmat_satd_phase(pmat_satd_active, project_path, config_val, result);

    let mut patterns = defect_patterns::base_defect_patterns(pmat_satd_active);
    patterns.extend(defect_patterns::gpu_and_crosscutting_patterns());

    // Convert custom patterns from config to owned patterns
    let custom_patterns: Vec<(String, DefectCategory, FindingSeverity, f64)> = bh_config
        .patterns
        .iter()
        .map(|p| {
            let category = parse_defect_category(&p.category);
            let severity = parse_finding_severity(&p.severity);
            (p.pattern.clone(), category, severity, p.suspiciousness)
        })
        .collect();

    // Collect all file paths first (multi-language support)
    let mut all_files: Vec<std::path::PathBuf> = Vec::new();
    for target in &config_val.targets {
        let target_path = project_path.join(target);
        // Scan all supported languages
        for glob_pattern in languages::all_language_globs() {
            if let Ok(entries) = glob::glob(&format!("{}/{}", target_path.display(), glob_pattern))
            {
                all_files.extend(entries.flatten());
            }
        }
    }

    // Parallel file scanning via std::thread::scope
    let min_susp = config_val.min_suspiciousness;
    let chunk_size = (all_files.len() / 4).max(1);
    let chunks: Vec<&[std::path::PathBuf]> = all_files.chunks(chunk_size).collect();

    let all_chunk_findings: Vec<Vec<Finding>> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|chunk| {
                let patterns = &patterns;
                let custom_patterns = &custom_patterns;
                let bh_config = &bh_config;
                s.spawn(move || {
                    let mut chunk_findings = Vec::new();
                    for entry in *chunk {
                        scan_file_for_patterns(
                            entry,
                            patterns,
                            custom_patterns,
                            bh_config,
                            min_susp,
                            &mut chunk_findings,
                        );
                    }
                    chunk_findings
                })
            })
            .collect();

        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });

    // Merge all findings and assign globally unique IDs
    // Also fetch git blame info for each finding
    let mut blame_cache = blame::BlameCache::new();
    let mut finding_id = 0u32;
    for chunk_findings in all_chunk_findings {
        for mut finding in chunk_findings {
            finding_id += 1;
            finding.id = format!("BH-PAT-{:04}", finding_id);

            // Fetch git blame info
            if let Some(blame_info) =
                blame_cache.get_blame(project_path, &finding.file, finding.line)
            {
                finding.blame_author = Some(blame_info.author);
                finding.blame_commit = Some(blame_info.commit);
                finding.blame_date = Some(blame_info.date);
            }

            result.add_finding(finding);
        }
    }
}
