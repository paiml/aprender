//! Contract quality gate: validate + audit + score in one pass.
//!
//! Runs three sequential gates across all contracts in a directory:
//! 1. **validate** — schema completeness (SCHEMA-001..013, PROVABILITY-001)
//! 2. **audit** — traceability chain (paper→equation→obligation→test→proof)
//! 3. **score** — 5-dimension quality score vs threshold
//!
//! Extended with SARIF output, rule catalog, config file, and findings.
//! Spec: `docs/specifications/sub/lint.md`

pub mod cache;
mod composition_gate;
pub mod config;
pub mod diff;
pub mod duplicate_stems;
pub mod finding;
mod gates;
mod gates_extended;
pub mod rules;
pub mod sarif;
mod strict_test_binding;
pub mod trend;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use self::finding::LintFinding;
use self::gates::{
    load_binding, load_contracts, run_audit_gate, run_score_gate, run_validate_gate,
};
use self::gates_extended::{
    check_stale_suppressions, run_enforce_gate, run_enforcement_level_gate,
    run_reverse_coverage_gate, run_verify_gate,
};
use self::rules::RuleSeverity;

/// Result of a single gate execution.
#[derive(Debug, Clone, Serialize)]
pub struct GateResult {
    pub name: String,
    pub passed: bool,
    pub skipped: bool,
    pub duration_ms: u64,
    pub detail: GateDetail,
}

/// Gate-specific detail payload.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum GateDetail {
    #[serde(rename = "validate")]
    Validate {
        contracts: usize,
        errors: usize,
        warnings: usize,
        error_messages: Vec<String>,
    },
    #[serde(rename = "audit")]
    Audit {
        contracts: usize,
        findings: usize,
        finding_messages: Vec<String>,
    },
    #[serde(rename = "score")]
    Score {
        contracts: usize,
        min_score: f64,
        mean_score: f64,
        threshold: f64,
        below_threshold: Vec<String>,
    },
    #[serde(rename = "verify")]
    Verify {
        total_refs: usize,
        existing: usize,
        missing: usize,
    },
    #[serde(rename = "enforce")]
    Enforce {
        equations_total: usize,
        equations_with_pre: usize,
        equations_with_post: usize,
        equations_with_lean: usize,
    },
    #[serde(rename = "reverse_coverage")]
    ReverseCoverage {
        total_pub_fns: usize,
        bound_fns: usize,
        unbound_fns: usize,
        coverage_pct: f64,
        threshold_pct: f64,
    },
    #[serde(rename = "composition")]
    Composition {
        edges_checked: usize,
        edges_satisfied: usize,
        edges_broken: usize,
    },
    /// PV-DUP-001: contract stems claimed by several files with divergent content.
    #[serde(rename = "duplicate_stems")]
    DuplicateStems {
        /// Total ambiguous stems found in the tree.
        divergent: usize,
        /// How many of those are recorded in the ratchet baseline.
        baselined: usize,
        /// Ambiguous stems NOT in the baseline — these fail the gate.
        unbaselined: Vec<String>,
        /// Baseline entries that no longer diverge — these fail the gate too.
        stale: Vec<String>,
        /// Every ambiguous stem, with its variant count and paths, for the report.
        divergent_stems: Vec<String>,
    },
    #[serde(rename = "skipped")]
    Skipped { reason: String },
}

/// Overall lint report.
#[derive(Debug, Clone, Serialize)]
pub struct LintReport {
    pub passed: bool,
    pub gates: Vec<GateResult>,
    pub total_duration_ms: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub findings: Vec<LintFinding>,
    #[serde(skip)]
    pub cache_stats: cache::CacheStats,
    /// Per-contract processing times: `(contract_stem, duration_ms)`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contract_timings: Vec<(String, u64)>,
}

/// Configuration for `pv lint`.
pub struct LintConfig<'a> {
    pub contract_dir: &'a Path,
    pub binding_path: Option<&'a Path>,
    pub min_score: f64,
    pub severity_filter: Option<RuleSeverity>,
    pub severity_overrides: HashMap<String, RuleSeverity>,
    pub suppressed_findings: Vec<String>,
    pub suppressed_rules: Vec<String>,
    pub suppressed_files: Vec<String>,
    pub strict: bool,
    pub no_cache: bool,
    pub cache_stats: bool,
    /// Optional crate directory for reverse coverage gate (Gate 7).
    pub crate_dir: Option<&'a Path>,
    /// Minimum enforcement level for Gate 6 (from `--min-level`).
    pub min_level: Option<crate::schema::EnforcementLevel>,
    /// Enable Gate 9 (strict test-binding, PV-VER-002). Issue #1510.
    pub strict_test_binding: bool,
}

impl<'a> LintConfig<'a> {
    /// Create a basic config (backward compatible).
    pub fn new(contract_dir: &'a Path, binding_path: Option<&'a Path>, min_score: f64) -> Self {
        Self {
            contract_dir,
            binding_path,
            min_score,
            severity_filter: None,
            severity_overrides: HashMap::new(),
            suppressed_findings: Vec::new(),
            suppressed_rules: Vec::new(),
            suppressed_files: Vec::new(),
            strict: false,
            no_cache: false,
            cache_stats: false,
            crate_dir: None,
            min_level: None,
            strict_test_binding: false,
        }
    }
}

/// Run all lint gates across a contract directory.
#[allow(clippy::too_many_lines)]
pub fn run_lint(config: &LintConfig) -> LintReport {
    let overall_start = Instant::now();
    let mut gates = Vec::with_capacity(3);
    let mut all_findings = Vec::new();
    let mut stats = cache::CacheStats::default();
    let mut contract_timings: Vec<(String, u64)> = Vec::new();

    let cache_root = if config.no_cache {
        None
    } else {
        Some(cache::cache_dir(config.contract_dir))
    };

    let (contracts, parse_errors) = load_contracts(config.contract_dir);
    let binding = load_binding(config.binding_path);

    // Gate 1: validate
    let (validate_result, mut validate_findings) = run_validate_gate(&contracts, &parse_errors);
    let validation_passed = validate_result.passed;
    gates.push(validate_result);

    // Gate 2: audit (skip if validation failed)
    if validation_passed {
        let (audit_result, mut audit_findings) = run_audit_gate(&contracts);
        gates.push(audit_result);
        all_findings.append(&mut audit_findings);
    } else {
        gates.push(skipped_gate("audit", "validation failed"));
    }

    // Gate 3: score (skip if validation failed)
    if validation_passed {
        let (score_result, mut score_findings) =
            run_score_gate(&contracts, binding.as_ref(), config.min_score);
        gates.push(score_result);
        all_findings.append(&mut score_findings);
    } else {
        gates.push(skipped_gate("score", "validation failed"));
    }

    // Gate 4: verify (source code fulfillment)
    if validation_passed {
        let project_root = config.contract_dir.parent().unwrap_or(config.contract_dir);
        let (verify_result, mut verify_findings) = run_verify_gate(&contracts, project_root);
        gates.push(verify_result);
        all_findings.append(&mut verify_findings);
    } else {
        gates.push(skipped_gate("verify", "validation failed"));
    }

    // Gate 5: enforce (equations must have preconditions/postconditions)
    if validation_passed {
        let (enforce_result, mut enforce_findings) = run_enforce_gate(&contracts);
        gates.push(enforce_result);
        all_findings.append(&mut enforce_findings);
    } else {
        gates.push(skipped_gate("enforce", "validation failed"));
    }

    // Gate 6: enforcement level (Section 17, Gap 1 + Gap 5 level lock)
    if validation_passed {
        let min_level = config
            .min_level
            .unwrap_or(crate::schema::EnforcementLevel::Standard);
        let (level_result, mut level_findings) = run_enforcement_level_gate(&contracts, min_level);
        gates.push(level_result);
        all_findings.append(&mut level_findings);
    } else {
        gates.push(skipped_gate("enforcement-level", "validation failed"));
    }

    // Gate 7: reverse coverage (optional — skip if no binding or crate dir)
    if validation_passed {
        if let (Some(bp), Some(cd)) = (config.binding_path, config.crate_dir) {
            let (rev_result, mut rev_findings) = run_reverse_coverage_gate(bp, cd);
            gates.push(rev_result);
            all_findings.append(&mut rev_findings);
        } else {
            gates.push(skipped_gate(
                "reverse-coverage",
                "no --binding or --crate-dir provided",
            ));
        }
    } else {
        gates.push(skipped_gate("reverse-coverage", "validation failed"));
    }

    // Gate 8: duplicate stems (PV-DUP-001). Must run BEFORE composition — it tells
    // the composition gate which stems are unresolvable, which is the difference
    // between a defined verdict and one decided by `read_dir` order.
    let duplicates = duplicate_stems::scan_duplicate_stems(config.contract_dir);
    let ambiguous = duplicate_stems::ambiguous_stems(&duplicates);
    if validation_passed {
        let project_root = config.contract_dir.parent().unwrap_or(config.contract_dir);
        let baseline = duplicate_stems::read_baseline(project_root);
        let (dup_result, mut dup_findings) =
            duplicate_stems::run_duplicate_stem_gate(&duplicates, &baseline);
        gates.push(dup_result);
        all_findings.append(&mut dup_findings);
    } else {
        gates.push(skipped_gate("duplicate-stems", "validation failed"));
    }

    // Gate 9: composition (assumes/guarantees chain verification)
    if validation_passed {
        let (comp_result, mut comp_findings) =
            composition_gate::run_composition_gate(&contracts, &ambiguous);
        gates.push(comp_result);
        all_findings.append(&mut comp_findings);
    } else {
        gates.push(skipped_gate("composition", "validation failed"));
    }

    // Gate 9: strict test-binding (Issue #1510, opt-in via --strict-test-binding)
    if config.strict_test_binding {
        if validation_passed {
            let project_root = config.contract_dir.parent().unwrap_or(config.contract_dir);
            let (binding_result, mut binding_findings) =
                strict_test_binding::run_strict_test_binding_gate(
                    &contracts,
                    project_root,
                    config.strict,
                );
            gates.push(binding_result);
            all_findings.append(&mut binding_findings);
        } else {
            gates.push(skipped_gate("strict-test-binding", "validation failed"));
        }
    }

    all_findings.append(&mut validate_findings);

    // Per-contract timing: measure how long each contract's findings take to process
    if validation_passed {
        for (stem, contract) in &contracts {
            let ct_start = Instant::now();
            // Validate
            let _ = crate::schema::validate_contract(contract);
            // Audit
            let _ = crate::audit::audit_contract(contract);
            // Score
            let _ = crate::scoring::score_contract(contract, binding.as_ref(), stem);
            let ct_ms = u64::try_from(ct_start.elapsed().as_micros() / 1000).unwrap_or(0);
            contract_timings.push((format!("{stem}.yaml"), ct_ms));
        }
        // Sort by duration descending
        contract_timings.sort_by_key(|b| std::cmp::Reverse(b.1));
    }

    // Stale suppression detection (PV-SUP-001, Section 17 Gap 2)
    let mut stale_findings = check_stale_suppressions(
        &all_findings,
        &config.suppressed_rules,
        &config.suppressed_findings,
    );
    all_findings.append(&mut stale_findings);

    // Issue lifecycle: mark each finding as new or pre-existing
    mark_new_findings(&mut all_findings, config.contract_dir);

    // Cache: store findings per-contract for future runs
    if let Some(ref root) = cache_root {
        let rule_cfg = format!("{:?}{:?}", config.severity_overrides, config.strict);
        for (stem, _) in &contracts {
            stats.total += 1;
            let yaml_path = config.contract_dir.join(format!("{stem}.yaml"));
            let yaml_content = std::fs::read_to_string(&yaml_path).unwrap_or_default();
            let hash = cache::content_hash(&yaml_content, &rule_cfg);
            if cache::cache_get(root, &hash).is_some() {
                stats.hits += 1;
            } else {
                stats.misses += 1;
                let contract_findings: Vec<_> = all_findings
                    .iter()
                    .filter(|f| f.contract_stem.as_deref() == Some(stem.as_str()))
                    .cloned()
                    .collect();
                let _ = cache::cache_put(root, &hash, &contract_findings);
            }
        }
    }

    // Apply suppressions, severity overrides, strict mode, and severity filter
    apply_suppressions(&mut all_findings, config);
    apply_severity_overrides(&mut all_findings, config);
    if let Some(min_sev) = config.severity_filter {
        all_findings.retain(|f| f.severity >= min_sev);
    }

    let passed = gates.iter().all(|g| g.passed || g.skipped);

    LintReport {
        passed,
        gates,
        total_duration_ms: u64::try_from(overall_start.elapsed().as_millis()).unwrap_or(u64::MAX),
        findings: all_findings,
        cache_stats: stats,
        contract_timings,
    }
}

fn skipped_gate(name: &str, reason: &str) -> GateResult {
    GateResult {
        name: name.into(),
        passed: false,
        skipped: true,
        duration_ms: 0,
        detail: GateDetail::Skipped {
            reason: reason.into(),
        },
    }
}

fn apply_suppressions(findings: &mut [LintFinding], config: &LintConfig) {
    for f in findings.iter_mut() {
        if config.suppressed_rules.iter().any(|r| r == &f.rule_id) {
            f.suppressed = true;
            f.suppression_reason = Some("Suppressed by --suppress-rule".into());
        }
        if let Some(ref stem) = f.contract_stem {
            if config.suppressed_findings.iter().any(|s| s == stem) {
                f.suppressed = true;
                f.suppression_reason = Some("Suppressed by --suppress".into());
            }
        }
        if config.suppressed_files.iter().any(|p| f.file.contains(p)) {
            f.suppressed = true;
            f.suppression_reason = Some("Suppressed by --suppress-file".into());
        }
    }
}

/// Resolve the `.pv/` state directory relative to the contract directory's parent.
fn pv_state_dir(contract_dir: &Path) -> std::path::PathBuf {
    contract_dir.parent().unwrap_or(contract_dir).join(".pv")
}

/// Load previous fingerprints, compare with current findings, mark new ones,
/// and persist the current fingerprint set for the next run.
fn mark_new_findings(findings: &mut [LintFinding], contract_dir: &Path) {
    let state_dir = pv_state_dir(contract_dir);
    let previous_path = state_dir.join("lint-previous.json");

    // Load previous fingerprints (empty set if file missing or unreadable)
    let previous: HashSet<String> = std::fs::read_to_string(&previous_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    // Compute current fingerprints and mark new findings
    let mut current = HashSet::new();
    for f in findings.iter_mut() {
        let fp = f.fingerprint();
        if !previous.contains(&fp) {
            f.is_new = true;
        }
        current.insert(fp);
    }

    // Persist current fingerprints for the next run
    if let Err(e) = std::fs::create_dir_all(&state_dir) {
        eprintln!("pv lint: cannot create {}: {e}", state_dir.display());
        return;
    }
    if let Ok(json) = serde_json::to_string(&current) {
        let _ = std::fs::write(&previous_path, json);
    }
}

fn apply_severity_overrides(findings: &mut [LintFinding], config: &LintConfig) {
    for f in findings.iter_mut() {
        if let Some(&sev) = config.severity_overrides.get(&f.rule_id) {
            f.severity = sev;
        }
    }
    if config.strict {
        for f in findings.iter_mut() {
            if f.severity == RuleSeverity::Warning {
                f.severity = RuleSeverity::Error;
            }
        }
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
