//! BH-04/05: Unsafe fuzzing and deep hunt modes.

use super::types::*;
use std::path::Path;

/// Check if a source file contains #![forbid(unsafe_code)] in its first 50 lines.
// SAFETY: no actual unsafe code -- checks if target crate forbids unsafe
pub(super) fn source_forbids_unsafe(path: &Path) -> bool {
    let Ok(content) = std::fs::read_to_string(path) else {
        return false;
    };
    content.lines().take(50).any(|line| {
        let t = line.trim();
        t.starts_with("#![") && t.contains("forbid") && t.contains("unsafe_code")
    })
}

/// Check if the crate forbids unsafe code (BH-19 fix).
// SAFETY: no actual unsafe code -- checks if target crate forbids unsafe via lint attrs
pub(super) fn crate_forbids_unsafe(project_path: &Path) -> bool {
    for entry in ["src/lib.rs", "src/main.rs"] {
        // SAFETY: no actual unsafe code -- delegating to source-level check
        if source_forbids_unsafe(&project_path.join(entry)) {
            return true;
        }
    }
    if let Ok(content) = std::fs::read_to_string(project_path.join("Cargo.toml")) {
        // SAFETY: no actual unsafe code -- checking Cargo.toml for unsafe_code lint config
        if content.contains("unsafe_code") && content.contains("forbid") {
            return true;
        }
    }
    false
}

/// Scan a single file for unsafe blocks and dangerous operations within them.
pub(super) fn scan_file_for_unsafe_blocks(
    entry: &Path,
    finding_id: &mut usize,
    unsafe_inventory: &mut Vec<(std::path::PathBuf, usize)>,
    result: &mut HuntResult,
) {
    let Ok(content) = std::fs::read_to_string(entry) else {
        return;
    };
    let mut in_unsafe = false;
    let mut unsafe_start = 0;

    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1;

        // SAFETY: no actual unsafe code -- scanning target file for unsafe blocks
        if line.contains("unsafe ") && line.contains('{') {
            in_unsafe = true;
            unsafe_start = line_num;
        }

        // SAFETY: no actual unsafe code -- tracking state for unsafe block content analysis
        if in_unsafe {
            if line.contains('*') && (line.contains("ptr") || line.contains("as *")) {
                *finding_id += 1;
                unsafe_inventory.push((entry.to_path_buf(), line_num));
                result.add_finding(
                    Finding::new(
                        format!("BH-UNSAFE-{:04}", finding_id),
                        entry,
                        line_num,
                        "Pointer dereference in unsafe block",
                    )
                    .with_description(format!(
                        "Unsafe block starting at line {}; potential fuzzing target",
                        unsafe_start
                    ))
                    .with_severity(FindingSeverity::High)
                    .with_category(DefectCategory::MemorySafety)
                    .with_suspiciousness(0.75)
                    .with_discovered_by(HuntMode::Fuzz)
                    .with_evidence(FindingEvidence::fuzzing("N/A", "pointer_deref")),
                );
            }

            if line.contains("transmute") {
                *finding_id += 1;
                result.add_finding(
                    Finding::new(
                        format!("BH-UNSAFE-{:04}", finding_id),
                        entry,
                        line_num,
                        "Transmute in unsafe block",
                    )
                    .with_description(
                        "std::mem::transmute bypasses type safety; high-priority fuzzing target",
                    )
                    .with_severity(FindingSeverity::Critical)
                    .with_category(DefectCategory::MemorySafety)
                    .with_suspiciousness(0.9)
                    .with_discovered_by(HuntMode::Fuzz)
                    .with_evidence(FindingEvidence::fuzzing("N/A", "transmute")),
                );
            }
        }

        // SAFETY: no actual unsafe code -- detecting end of unsafe block being scanned
        if line.contains('}') && in_unsafe {
            in_unsafe = false;
        }
    }
}

/// BH-04: Targeted unsafe Rust fuzzing (FourFuzz pattern)
pub(super) fn run_fuzz_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    // SAFETY: no actual unsafe code -- checking if fuzz targets needed based on crate policy
    if crate_forbids_unsafe(project_path) {
        result.add_finding(
            Finding::new(
                "BH-FUZZ-SKIPPED",
                project_path.join("src/lib.rs"),
                1,
                "Fuzz targets not needed - crate forbids unsafe code",
            )
            .with_description("Crate uses #![forbid(unsafe_code)], no unsafe blocks to fuzz")
            .with_severity(FindingSeverity::Info)
            .with_category(DefectCategory::ConfigurationErrors)
            .with_suspiciousness(0.0)
            .with_discovered_by(HuntMode::Fuzz),
        );
        return;
    }

    let mut unsafe_inventory = Vec::new();
    let mut finding_id = 0;

    for target in &config.targets {
        let target_path = project_path.join(target);
        for pattern in &[
            format!("{}/*.rs", target_path.display()),
            format!("{}/**/*.rs", target_path.display()),
        ] {
            if let Ok(entries) = glob::glob(pattern) {
                for entry in entries.flatten() {
                    scan_file_for_unsafe_blocks(
                        &entry,
                        &mut finding_id,
                        &mut unsafe_inventory,
                        result,
                    );
                }
            }
        }
    }

    let fuzz_dir = project_path.join("fuzz");
    if !fuzz_dir.exists() {
        result.add_finding(
            Finding::new(
                "BH-FUZZ-NOTARGETS",
                project_path.join("Cargo.toml"),
                1,
                "No fuzz directory found",
            )
            .with_description(format!(
                // SAFETY: no actual unsafe code -- format string referencing unsafe block count
                "Create fuzz targets for {} identified unsafe blocks",
                unsafe_inventory.len()
            ))
            .with_severity(FindingSeverity::Medium)
            .with_category(DefectCategory::ConfigurationErrors)
            .with_suspiciousness(0.4)
            .with_discovered_by(HuntMode::Fuzz),
        );
    }

    // SAFETY: no actual unsafe code -- computing fuzz coverage from unsafe inventory
    result.stats.mode_stats.fuzz_coverage = if unsafe_inventory.is_empty() { 100.0 } else { 0.0 };
}

/// Scan a single file for deeply nested conditionals and complex boolean guards.
pub(super) fn scan_file_for_deep_conditionals(
    entry: &Path,
    finding_id: &mut usize,
    result: &mut HuntResult,
) {
    let Ok(content) = std::fs::read_to_string(entry) else {
        return;
    };
    let mut complexity: usize = 0;
    let mut complex_start: usize = 0;

    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1;

        if line.contains("if ") || line.contains("match ") {
            complexity += 1;
            if complexity == 1 {
                complex_start = line_num;
            }
        }

        if complexity >= 3 && line.contains("if ") {
            *finding_id += 1;
            result.add_finding(
                Finding::new(
                    format!("BH-DEEP-{:04}", *finding_id),
                    entry,
                    line_num,
                    "Deeply nested conditional",
                )
                .with_description(format!(
                    "Complexity {} starting at line {}; concolic execution recommended",
                    complexity, complex_start
                ))
                .with_severity(FindingSeverity::Medium)
                .with_category(DefectCategory::LogicErrors)
                .with_suspiciousness(0.6)
                .with_discovered_by(HuntMode::DeepHunt)
                .with_evidence(FindingEvidence::concolic(format!("depth={}", complexity))),
            );
        }

        if line.contains(" && ") && line.contains(" || ") {
            *finding_id += 1;
            result.add_finding(
                Finding::new(
                    format!("BH-DEEP-{:04}", *finding_id),
                    entry,
                    line_num,
                    "Complex boolean guard",
                )
                .with_description("Mixed AND/OR logic; path explosion potential")
                .with_severity(FindingSeverity::Medium)
                .with_category(DefectCategory::LogicErrors)
                .with_suspiciousness(0.55)
                .with_discovered_by(HuntMode::DeepHunt)
                .with_evidence(FindingEvidence::concolic("complex_guard")),
            );
        }

        if line.contains('}') && complexity > 0 {
            complexity -= 1;
        }
    }
}

/// BH-05: Hybrid concolic + SBFL (COTTONTAIL pattern)
pub(super) fn run_deep_hunt_mode(
    project_path: &Path,
    config: &HuntConfig,
    result: &mut HuntResult,
) {
    let mut finding_id = 0;

    for target in &config.targets {
        let target_path = project_path.join(target);
        for pattern in &[
            format!("{}/*.rs", target_path.display()),
            format!("{}/**/*.rs", target_path.display()),
        ] {
            if let Ok(entries) = glob::glob(pattern) {
                for entry in entries.flatten() {
                    scan_file_for_deep_conditionals(&entry, &mut finding_id, result);
                }
            }
        }
    }

    super::modes_hunt::run_hunt_mode(project_path, config, result);
}
