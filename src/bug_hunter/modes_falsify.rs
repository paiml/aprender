//! BH-01: Mutation-based invariant falsification (FDV pattern).

use super::types::*;
use std::path::Path;

/// BH-01: Mutation-based invariant falsification (FDV pattern)
pub(super) fn run_falsify_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    // Check for cargo-mutants availability
    let mutants_available = std::process::Command::new("cargo")
        .args(["mutants", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !mutants_available {
        result.add_finding(
            Finding::new(
                "BH-FALSIFY-UNAVAIL",
                project_path.join("Cargo.toml"),
                1,
                "cargo-mutants not installed",
            )
            .with_description("Install with: cargo install cargo-mutants")
            .with_severity(FindingSeverity::Info)
            .with_category(DefectCategory::ConfigurationErrors)
            .with_suspiciousness(0.1)
            .with_discovered_by(HuntMode::Falsify),
        );
        return;
    }

    // Analyze Rust files for potential mutation targets
    for target in &config.targets {
        let target_path = project_path.join(target);
        // Match .rs files both directly in the target dir and in subdirectories
        for pattern in &[
            format!("{}/*.rs", target_path.display()),
            format!("{}/**/*.rs", target_path.display()),
        ] {
            if let Ok(entries) = glob::glob(pattern) {
                for entry in entries.flatten() {
                    analyze_file_for_mutations(&entry, config, result);
                }
            }
        }
    }
}

/// Detected mutation target with metadata.
pub(super) struct MutationMatch {
    pub(super) title: &'static str,
    pub(super) description: &'static str,
    pub(super) severity: FindingSeverity,
    pub(super) suspiciousness: f64,
    pub(super) prefix: &'static str,
}

/// Detect mutation targets in a single line of code.
pub(super) fn detect_mutation_targets(line: &str) -> Vec<MutationMatch> {
    let mut matches = Vec::new();

    let has_comparison =
        line.contains("< ") || line.contains("> ") || line.contains("<= ") || line.contains(">= ");
    let has_len = line.contains("len()") || line.contains("size()") || line.contains(".len");
    if has_comparison && has_len {
        matches.push(MutationMatch {
            title: "Boundary condition mutation target",
            description: "Off-by-one errors are common; this comparison should be mutation-tested",
            severity: FindingSeverity::Medium,
            suspiciousness: 0.6,
            prefix: "boundary",
        });
    }

    let has_arith = line.contains(" + ") || line.contains(" - ") || line.contains(" * ");
    let no_safe =
        !line.contains("saturating_") && !line.contains("checked_") && !line.contains("wrapping_");
    let has_cast = line.contains("as usize") || line.contains("as u") || line.contains("as i");
    if has_arith && no_safe && has_cast {
        matches.push(MutationMatch {
            title: "Arithmetic operation mutation target",
            description:
                "Unchecked arithmetic with type cast; consider checked_* or saturating_* operations",
            severity: FindingSeverity::Medium,
            suspiciousness: 0.55,
            prefix: "arith",
        });
    }

    let has_logic = line.contains(" && ") || line.contains(" || ");
    let has_predicate = line.contains('!') || line.contains("is_") || line.contains("has_");
    if has_logic && has_predicate {
        matches.push(MutationMatch {
            title: "Boolean logic mutation target",
            description:
                "Complex boolean expression; verify test coverage catches negation mutations",
            severity: FindingSeverity::Low,
            suspiciousness: 0.4,
            prefix: "bool",
        });
    }

    matches
}

/// Analyze a file for mutation testing targets.
pub(super) fn analyze_file_for_mutations(
    file_path: &Path,
    _config: &HuntConfig,
    result: &mut HuntResult,
) {
    let Ok(content) = std::fs::read_to_string(file_path) else {
        return;
    };

    let mut finding_id = 0;

    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1;
        for m in detect_mutation_targets(line) {
            finding_id += 1;
            result.add_finding(
                Finding::new(format!("BH-MUT-{:04}", finding_id), file_path, line_num, m.title)
                    .with_description(m.description)
                    .with_severity(m.severity)
                    .with_category(DefectCategory::LogicErrors)
                    .with_suspiciousness(m.suspiciousness)
                    .with_discovered_by(HuntMode::Falsify)
                    .with_evidence(FindingEvidence::mutation(
                        format!("{}_{}", m.prefix, finding_id),
                        true,
                    )),
            );
        }
    }
}
