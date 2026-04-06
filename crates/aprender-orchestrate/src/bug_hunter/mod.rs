//! Proactive Bug Hunting Module
//!
//! Implements Section 11 of the Popperian Falsification Checklist (BH-01 to BH-15).
//! Provides falsification-driven defect discovery using multiple hunting modes.
//!
//! # Philosophy
//!
//! "A theory that explains everything, explains nothing." — Karl Popper
//!
//! Bug hunting operationalizes falsification: we systematically attempt to break
//! code, not merely verify it works. Each mode represents a different strategy
//! for falsifying the implicit claim "this code is correct."
//!
//! # Modes
//!
//! - **Falsify**: Mutation-based invariant falsification (FDV pattern)
//! - **Hunt**: SBFL without failing tests (SBEST pattern)
//! - **Analyze**: LLM-augmented static analysis (LLIFT pattern)
//! - **Fuzz**: Targeted unsafe Rust fuzzing (FourFuzz pattern)
//! - **DeepHunt**: Hybrid concolic + SBFL (COTTONTAIL pattern)
//!
//! # Advanced Features (BH-11 to BH-15)
//!
//! - **Spec-Driven**: Hunt bugs guided by specification files
//! - **Ticket-Scoped**: Focus on areas defined by PMAT work tickets
//! - **Scoped Analysis**: --lib, --bin, --path for targeted hunting
//! - **Bidirectional Linking**: Update specs with findings
//! - **False Positive Suppression**: Filter known false positive patterns
//!
//! # Integration with OIP
//!
//! These modes leverage OIP's SBFL (Tarantula/Ochiai/DStar), defect classification,
//! and RAG enhancement to proactively identify bugs before they reach production.

pub mod blame;
pub mod cache;
pub mod config;
#[cfg(feature = "native")]
pub mod contracts;
pub mod coverage;
mod defect_patterns;
pub mod diff;
pub mod languages;
pub mod localization;
#[cfg(feature = "native")]
pub mod model_parity;
mod modes_analyze;
mod modes_falsify;
mod modes_fuzz;
mod modes_hunt;
pub mod patterns;
pub mod pmat_quality;
pub mod spec;
pub mod ticket;
mod types;

#[allow(unused_imports)]
pub use localization::{CrashBucketer, MultiChannelLocalizer, ScoredLocation};
pub use patterns::{compute_test_lines, is_real_pattern, should_suppress_finding};
pub use spec::ParsedSpec;
pub use ticket::PmatTicket;
pub use types::*;

use std::path::Path;
use std::time::Instant;

// Re-exports for test access (functions moved to submodules in QA-002 split)
#[cfg(test)]
use modes_analyze::{
    analyze_common_patterns, categorize_clippy_warning, extract_clippy_finding,
    match_custom_pattern, match_lang_pattern, parse_defect_category, parse_finding_severity,
    scan_file_for_patterns, PatternMatchContext,
};
#[cfg(test)]
use modes_falsify::{analyze_file_for_mutations, detect_mutation_targets, run_falsify_mode};
#[cfg(test)]
use modes_fuzz::{
    crate_forbids_unsafe, run_deep_hunt_mode, run_fuzz_mode, scan_file_for_deep_conditionals,
    scan_file_for_unsafe_blocks, source_forbids_unsafe,
};
#[cfg(test)]
use modes_hunt::{
    analyze_coverage_hotspots, analyze_stack_trace, parse_lcov_da_line, parse_lcov_for_hotspots,
    report_uncovered_hotspots, run_hunt_mode,
};

/// Print a phase progress indicator to stderr (native builds only).
#[cfg(feature = "native")]
fn eprint_phase(phase: &str, mode: &HuntMode) {
    use crate::ansi_colors::Colorize;
    eprintln!("  {} {}", format!("[{:>8}]", mode).dimmed(), phase);
}

/// Run bug hunting with the specified configuration.
pub fn hunt(project_path: &Path, config: HuntConfig) -> HuntResult {
    let start = Instant::now();

    // Check cache first
    if let Some(cached) = cache::load_cached(project_path, &config) {
        #[cfg(feature = "native")]
        {
            use crate::ansi_colors::Colorize;
            eprintln!("  {} hit — using cached findings", "[  cache]".dimmed());
        }
        let mut result = HuntResult::new(project_path, cached.mode, config);
        result.findings = cached.findings;
        result.duration_ms = start.elapsed().as_millis() as u64;
        result.finalize();
        return result;
    }

    let mut result = HuntResult::new(project_path, config.mode, config.clone());

    // Phase 1: Mode dispatch (scanning)
    #[cfg(feature = "native")]
    eprint_phase("Scanning...", &config.mode);
    let phase_start = Instant::now();

    match config.mode {
        HuntMode::Falsify => modes_falsify::run_falsify_mode(project_path, &config, &mut result),
        HuntMode::Hunt => modes_hunt::run_hunt_mode(project_path, &config, &mut result),
        HuntMode::Analyze => modes_analyze::run_analyze_mode(project_path, &config, &mut result),
        HuntMode::Fuzz => modes_fuzz::run_fuzz_mode(project_path, &config, &mut result),
        HuntMode::DeepHunt => modes_fuzz::run_deep_hunt_mode(project_path, &config, &mut result),
        HuntMode::Quick => run_quick_mode(project_path, &config, &mut result),
    }
    result.phase_timings.mode_dispatch_ms = phase_start.elapsed().as_millis() as u64;

    // Phase 2: BH-21 to BH-24: PMAT quality integration
    #[cfg(feature = "native")]
    if config.use_pmat_quality {
        eprint_phase("Quality index...", &config.mode);
        let pmat_start = Instant::now();
        let query = config.pmat_query.as_deref().unwrap_or("*");
        if let Some(index) = pmat_quality::build_quality_index(project_path, query, 200) {
            result.phase_timings.pmat_index_ms = pmat_start.elapsed().as_millis() as u64;

            let weights_start = Instant::now();
            eprint_phase("Applying weights...", &config.mode);
            pmat_quality::apply_quality_weights(
                &mut result.findings,
                &index,
                config.quality_weight,
            );
            pmat_quality::apply_regression_risk(&mut result.findings, &index);
            result.phase_timings.pmat_weights_ms = weights_start.elapsed().as_millis() as u64;
        }
    }

    // Phase 2b: Coverage-based hotpath weighting
    #[cfg(feature = "native")]
    if config.coverage_weight > 0.0 {
        // Try to find coverage file
        let cov_path =
            config.coverage_path.clone().or_else(|| coverage::find_coverage_file(project_path));

        if let Some(cov_path) = cov_path {
            if let Some(cov_index) = coverage::load_coverage_index(&cov_path) {
                eprint_phase("Coverage weights...", &config.mode);
                coverage::apply_coverage_weights(
                    &mut result.findings,
                    &cov_index,
                    config.coverage_weight,
                );
            }
        }
    }

    // Phase 2c: Contract verification gaps (BH-26)
    #[cfg(feature = "native")]
    run_contract_gap_phase(project_path, &config, &mut result);

    // Phase 2d: Model parity gaps (BH-27)
    #[cfg(feature = "native")]
    run_model_parity_phase(project_path, &config, &mut result);

    // Phase 3: Finalize
    #[cfg(feature = "native")]
    eprint_phase("Finalizing...", &config.mode);
    let finalize_start = Instant::now();

    result.duration_ms = start.elapsed().as_millis() as u64;
    result.finalize();
    result.phase_timings.finalize_ms = finalize_start.elapsed().as_millis() as u64;

    // Save to cache
    cache::save_cache(project_path, &config, &result.findings, result.mode);

    result
}

/// Phase 2c helper: Contract verification gap analysis (BH-26).
#[cfg(feature = "native")]
fn run_contract_gap_phase(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    if config.contracts_path.is_none() && !config.contracts_auto {
        return;
    }
    let Some(dir) =
        contracts::discover_contracts_dir(project_path, config.contracts_path.as_deref())
    else {
        return;
    };
    eprint_phase("Contract gaps...", &config.mode);
    let contract_start = Instant::now();
    for f in contracts::analyze_contract_gaps(&dir, project_path) {
        if f.suspiciousness >= config.min_suspiciousness {
            result.add_finding(f);
        }
    }
    result.phase_timings.contract_gap_ms = contract_start.elapsed().as_millis() as u64;
}

/// Phase 2d helper: Model parity gap analysis (BH-27).
#[cfg(feature = "native")]
fn run_model_parity_phase(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    if config.model_parity_path.is_none() && !config.model_parity_auto {
        return;
    }
    let Some(dir) =
        model_parity::discover_model_parity_dir(project_path, config.model_parity_path.as_deref())
    else {
        return;
    };
    eprint_phase("Model parity...", &config.mode);
    let parity_start = Instant::now();
    for f in model_parity::analyze_model_parity_gaps(&dir, project_path) {
        if f.suspiciousness >= config.min_suspiciousness {
            result.add_finding(f);
        }
    }
    result.phase_timings.model_parity_ms = parity_start.elapsed().as_millis() as u64;
}

/// Run all modes and combine results (ensemble approach).
pub fn hunt_ensemble(project_path: &Path, base_config: HuntConfig) -> HuntResult {
    let start = Instant::now();
    let mut combined = HuntResult::new(project_path, HuntMode::Analyze, base_config.clone());

    // Run each mode and collect findings
    for mode in [HuntMode::Analyze, HuntMode::Hunt, HuntMode::Falsify] {
        let mut config = base_config.clone();
        config.mode = mode;
        let mode_result = hunt(project_path, config);

        for finding in mode_result.findings {
            // Avoid duplicates by checking location + category.
            // Category is included so distinct finding types at the same
            // location (e.g., multiple contract gaps in one binding.yaml)
            // are preserved.
            let exists = combined.findings.iter().any(|f| {
                f.file == finding.file
                    && f.line == finding.line
                    && f.category == finding.category
                    && f.title == finding.title
            });
            if !exists {
                combined.add_finding(finding);
            }
        }
    }

    combined.duration_ms = start.elapsed().as_millis() as u64;
    combined.finalize();
    combined
}

/// Run spec-driven bug hunting (BH-11).
///
/// Parses a specification file, extracts claims, finds implementing code,
/// and hunts bugs specifically in those areas.
pub fn hunt_with_spec(
    project_path: &Path,
    spec_path: &Path,
    section_filter: Option<&str>,
    mut config: HuntConfig,
) -> Result<(HuntResult, ParsedSpec), String> {
    let start = Instant::now();

    // Parse the specification
    let mut parsed_spec = ParsedSpec::parse(spec_path)?;

    // Get claim IDs to hunt (filtered by section if specified)
    let claim_ids: Vec<String> = if let Some(section) = section_filter {
        parsed_spec.claims_for_section(section).iter().map(|c| c.id.clone()).collect()
    } else {
        parsed_spec.claims.iter().map(|c| c.id.clone()).collect()
    };

    // Find implementations for each claim
    for claim in &mut parsed_spec.claims {
        claim.implementations = spec::find_implementations(claim, project_path);
    }

    // Collect target paths from implementations (for claims in our filter)
    let mut target_paths: Vec<std::path::PathBuf> = parsed_spec
        .claims
        .iter()
        .filter(|c| claim_ids.contains(&c.id))
        .flat_map(|c| c.implementations.iter().map(|i| i.file.clone()))
        .collect();

    // Deduplicate
    target_paths.sort();
    target_paths.dedup();

    // If no implementations found, use default targets
    if target_paths.is_empty() {
        target_paths = config.targets.clone();
    }

    // Update config with discovered targets
    config.targets = target_paths
        .iter()
        .map(|p| p.parent().unwrap_or(Path::new("src")).to_path_buf())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    if config.targets.is_empty() {
        config.targets = vec![std::path::PathBuf::from("src")];
    }

    // Capture PMAT config before hunt() consumes config
    let use_pmat_quality = config.use_pmat_quality;
    let pmat_query_str = config.pmat_query.clone();

    // Run the hunt
    let mut result = hunt(project_path, config);

    // Map findings to claims
    let mapping = spec::map_findings_to_claims(&parsed_spec.claims, &result.findings, project_path);

    // BH-25: Quality gate — if PMAT quality is enabled, check implementing functions
    if use_pmat_quality {
        let query = pmat_query_str.as_deref().unwrap_or("*");
        apply_spec_quality_gate(&mut parsed_spec, project_path, &mut result, query);
    }

    // Update spec claims with findings
    let findings_by_claim: Vec<(String, Vec<Finding>)> = mapping.into_iter().collect();
    if let Ok(updated_content) = parsed_spec.update_with_findings(&findings_by_claim) {
        parsed_spec.original_content = updated_content;
    }

    result.duration_ms = start.elapsed().as_millis() as u64;

    Ok((result, parsed_spec))
}

/// Apply quality gate checks to spec claims using PMAT quality index (BH-25).
fn apply_spec_quality_gate(
    parsed_spec: &mut ParsedSpec,
    project_path: &Path,
    result: &mut HuntResult,
    query: &str,
) {
    let Some(index) = pmat_quality::build_quality_index(project_path, query, 200) else {
        return;
    };
    for claim in &mut parsed_spec.claims {
        for imp in &claim.implementations {
            let Some(pmat) = pmat_quality::lookup_quality(&index, &imp.file, imp.line) else {
                continue;
            };
            let is_low_quality =
                pmat.tdg_grade == "D" || pmat.tdg_grade == "F" || pmat.complexity > 20;
            if !is_low_quality {
                continue;
            }
            result.add_finding(
                Finding::new(
                    format!("BH-QGATE-{}", claim.id),
                    &imp.file,
                    imp.line,
                    format!(
                        "Quality gate: claim `{}` implemented by low-quality code",
                        claim.id
                    ),
                )
                .with_description(format!(
                    "Function `{}` (grade {}, complexity {}) implements spec claim `{}`; consider refactoring",
                    pmat.function_name, pmat.tdg_grade, pmat.complexity, claim.id
                ))
                .with_severity(FindingSeverity::Medium)
                .with_category(DefectCategory::LogicErrors)
                .with_suspiciousness(0.6)
                .with_discovered_by(HuntMode::Analyze)
                .with_evidence(FindingEvidence::quality_metrics(
                    &pmat.tdg_grade,
                    pmat.tdg_score,
                    pmat.complexity,
                )),
            );
        }
    }
}

/// Run ticket-scoped bug hunting (BH-12).
///
/// Parses a PMAT ticket and focuses analysis on affected paths.
pub fn hunt_with_ticket(
    project_path: &Path,
    ticket_ref: &str,
    mut config: HuntConfig,
) -> Result<HuntResult, String> {
    // Parse the ticket
    let ticket = PmatTicket::parse(ticket_ref, project_path)?;

    // Update targets from ticket
    config.targets = ticket.target_paths();

    // Run the hunt with scoped targets
    Ok(hunt(project_path, config))
}

/// Quick mode: pattern matching only, no clippy, no coverage analysis.
/// Fastest mode for quick scans.
fn run_quick_mode(project_path: &Path, config: &HuntConfig, result: &mut HuntResult) {
    // Only run pattern analysis (from analyze mode)
    modes_analyze::analyze_common_patterns(project_path, config, result);
}

#[cfg(test)]
#[path = "tests_mod.rs"]
mod tests;
