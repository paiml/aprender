//! BH-02: SBFL without failing tests (SBEST pattern).

use super::types::*;
use std::path::Path;

/// BH-02: SBFL without failing tests (SBEST pattern)
pub(super) fn run_hunt_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    // Look for crash logs, stack traces, or error reports
    let crash_patterns = ["crash.log", "panic.log", "*.crash", "stack_trace.txt", "core.*"];

    let mut stack_traces_found = Vec::new();

    for pattern in crash_patterns {
        if let Ok(entries) = glob::glob(&format!("{}/**/{}", project_path.display(), pattern)) {
            for entry in entries.flatten() {
                stack_traces_found.push(entry);
            }
        }
    }

    // Also check for recent panic messages in test output
    let test_output = project_path.join("target/nextest/ci/junit.xml");
    if test_output.exists() {
        if let Ok(content) = std::fs::read_to_string(&test_output) {
            if content.contains("panicked") || content.contains("FAILED") {
                stack_traces_found.push(test_output);
            }
        }
    }

    if stack_traces_found.is_empty() {
        // No crash data available, fall back to coverage-based analysis
        analyze_coverage_hotspots(project_path, config, result);
    } else {
        for trace_file in stack_traces_found {
            analyze_stack_trace(&trace_file, project_path, config, result);
        }
    }
}

/// Analyze coverage data for suspicious hotspots.
pub(super) fn analyze_coverage_hotspots(
    project_path: &Path,
    config: &HuntConfig,
    result: &mut HuntResult,
) {
    // Check custom coverage path first
    if let Some(ref custom_path) = config.coverage_path {
        if custom_path.exists() {
            if let Ok(content) = std::fs::read_to_string(custom_path) {
                parse_lcov_for_hotspots(&content, project_path, result);
                return;
            }
        }
    }

    // Build list of paths to search
    let mut lcov_paths: Vec<std::path::PathBuf> = vec![
        // Project root (common location for cargo llvm-cov output)
        project_path.join("lcov.info"),
        // Standard target locations
        project_path.join("target/coverage/lcov.info"),
        project_path.join("target/llvm-cov/lcov.info"),
        project_path.join("coverage/lcov.info"),
    ];

    // Check CARGO_TARGET_DIR for custom target locations
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let target_path = std::path::Path::new(&target_dir);
        lcov_paths.push(target_path.join("coverage/lcov.info"));
        lcov_paths.push(target_path.join("llvm-cov/lcov.info"));
    }

    for lcov_path in &lcov_paths {
        if lcov_path.exists() {
            if let Ok(content) = std::fs::read_to_string(lcov_path) {
                parse_lcov_for_hotspots(&content, project_path, result);
                return;
            }
        }
    }

    // No coverage data available
    let searched =
        lcov_paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ");
    result.add_finding(
        Finding::new(
            "BH-HUNT-NOCOV",
            project_path.join("target"),
            1,
            "No coverage data available",
        )
        .with_description(format!(
            "Run `cargo llvm-cov --lcov --output-path lcov.info` or use --coverage-path. Searched: {}",
            searched
        ))
        .with_severity(FindingSeverity::Info)
        .with_category(DefectCategory::ConfigurationErrors)
        .with_suspiciousness(0.1)
        .with_discovered_by(HuntMode::Hunt),
    );
}

/// Parse a single DA line from LCOV data, recording uncovered lines.
pub(super) fn parse_lcov_da_line(
    da: &str,
    file: &str,
    file_uncovered: &mut std::collections::HashMap<String, Vec<usize>>,
) {
    let Some((line_str, hits_str)) = da.split_once(',') else {
        return;
    };
    let Ok(line_num) = line_str.parse::<usize>() else {
        return;
    };
    let Ok(hits) = hits_str.parse::<usize>() else {
        return;
    };
    if hits == 0 {
        file_uncovered.entry(file.to_string()).or_default().push(line_num);
    }
}

/// Report files with many uncovered lines as suspicious findings.
pub(super) fn report_uncovered_hotspots(
    file_uncovered: std::collections::HashMap<String, Vec<usize>>,
    project_path: &Path,
    result: &mut HuntResult,
) {
    let mut finding_id = 0;
    for (file, lines) in file_uncovered {
        if lines.len() <= 5 {
            continue;
        }
        finding_id += 1;
        let suspiciousness = (lines.len() as f64 / 100.0).min(0.8);
        result.add_finding(
            Finding::new(
                format!("BH-COV-{:04}", finding_id),
                project_path.join(&file),
                lines[0],
                format!("Low coverage region ({} uncovered lines)", lines.len()),
            )
            .with_description(format!(
                "Lines {} are never executed; potential dead code or missing tests",
                lines.iter().take(5).map(|l| l.to_string()).collect::<Vec<_>>().join(", ")
            ))
            .with_severity(FindingSeverity::Low)
            .with_category(DefectCategory::LogicErrors)
            .with_suspiciousness(suspiciousness)
            .with_discovered_by(HuntMode::Hunt)
            .with_evidence(FindingEvidence::sbfl("Coverage", suspiciousness)),
        );
    }
}

/// Parse LCOV data for coverage hotspots.
pub(super) fn parse_lcov_for_hotspots(content: &str, project_path: &Path, result: &mut HuntResult) {
    let mut current_file: Option<String> = None;
    let mut file_uncovered: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();

    for line in content.lines() {
        if let Some(file) = line.strip_prefix("SF:") {
            current_file = Some(file.to_string());
        } else if let Some(da) = line.strip_prefix("DA:") {
            if let Some(ref file) = current_file {
                parse_lcov_da_line(da, file, &mut file_uncovered);
            }
        } else if line == "end_of_record" {
            current_file = None;
        }
    }

    report_uncovered_hotspots(file_uncovered, project_path, result);
}

/// Analyze a stack trace file.
pub(super) fn analyze_stack_trace(
    trace_file: &Path,
    _project_path: &Path,
    _config: &HuntConfig,
    result: &mut HuntResult,
) {
    let content = match std::fs::read_to_string(trace_file) {
        Ok(c) => c,
        Err(_) => return,
    };

    let mut finding_id = 0;

    // Look for panic locations
    for line in content.lines() {
        // Pattern: "at src/file.rs:123"
        if let Some(at_pos) = line.find(" at ") {
            let location = &line[at_pos + 4..];
            if let Some(colon_pos) = location.rfind(':') {
                let file = &location[..colon_pos];
                if let Ok(line_num) = location[colon_pos + 1..].trim().parse::<usize>() {
                    if file.ends_with(".rs") && !file.contains("/.cargo/") {
                        finding_id += 1;
                        result.add_finding(
                            Finding::new(
                                format!("BH-STACK-{:04}", finding_id),
                                file,
                                line_num,
                                "Stack trace location",
                            )
                            .with_description(format!(
                                "Found in stack trace: {}",
                                trace_file.display()
                            ))
                            .with_severity(FindingSeverity::High)
                            .with_category(DefectCategory::LogicErrors)
                            .with_suspiciousness(0.85)
                            .with_discovered_by(HuntMode::Hunt)
                            .with_evidence(FindingEvidence::sbfl("StackTrace", 0.85)),
                        );
                    }
                }
            }
        }
    }
}
