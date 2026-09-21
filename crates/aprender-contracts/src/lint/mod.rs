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
pub use gates::collect_yaml_files;
mod gates_extended;
pub mod relations_gate;
pub mod rules;
pub mod sarif;
pub mod shapes_gate;
pub mod sigma_gate;
pub mod sigma_symbols;
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
use crate::ontology::arming::{meet_armed, ArmedGates};
use crate::ontology::verdict::Verdict;

/// Result of a single gate execution.
#[derive(Debug, Clone, Serialize)]
pub struct GateResult {
    pub name: String,
    pub passed: bool,
    pub skipped: bool,
    /// ONT-001 §3.4: this gate's element of the one verdict lattice (`Verdict::from_gate(passed, skipped)`).
    pub verdict: Verdict,
    pub duration_ms: u64,
    pub detail: GateDetail,
    /// Structured payload for gates invented AFTER `GateDetail` was frozen.
    ///
    /// See [`GateExtra`] for why this second channel exists. Adding a field to a
    /// struct is invisible to a `match` on `GateDetail`, which is the property
    /// the 0.3.1 compatibility corpus depends on; adding a *variant* is not.
    /// Serialised only when present, so existing JSON output is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<GateExtra>,
}

/// Gate-specific detail payload.
///
/// # This enum is FROZEN at the eight variants published in `provable-contracts`
/// 0.3.1 — do not add a ninth
///
/// `GateDetail` is public, is not `#[non_exhaustive]`, and 0.3.1 shipped 28 example
/// programs that `match` it exhaustively. Those programs are vendored verbatim under
/// `crates/facades/provable-contracts/compat/0.3.1/`, sha256-verified against their
/// published checksums by `scripts/check_facade_compat.sh` (row R6), and compiled in
/// the `ci` job that the required `gate` check depends on. A ninth variant is
/// `error[E0004]` in that corpus, and the corpus cannot be edited to accommodate it —
/// being uneditable is the whole point of a compatibility contract. Marking the enum
/// `#[non_exhaustive]` does not help either: it makes the *existing* exhaustive
/// matches non-exhaustive, which is the same compile error for the same reason.
///
/// So the vocabulary is closed. A gate added after 0.3.1 picks the truest of these
/// eight for `detail` and carries its own shape in [`GateResult::extra`], which is a
/// type 0.3.1 never names and is therefore free to grow.
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
    #[serde(rename = "skipped")]
    Skipped { reason: String },
}

/// Out-of-band structured detail for gates that post-date the frozen [`GateDetail`]
/// vocabulary.
///
/// This type did not exist in `provable-contracts` 0.3.1, so no 0.3.1 program names
/// it and none can `match` it. That is what makes it extensible where `GateDetail` is
/// not, and it is `#[non_exhaustive]` from birth so the *next* post-0.3.1 gate does
/// not have to repeat this exercise.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum GateExtra {
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
    /// ONT-2b: what Σ declares and what the corpus was checked against.
    #[serde(rename = "sigma")]
    Sigma {
        /// `entity_types` Σ declares.
        entity_types: usize,
        /// `roles` Σ declares.
        roles: usize,
        /// `symbols` Σ declares.
        symbols: usize,
        /// Contract files read (never counting `ontology.yaml` itself).
        contracts_checked: usize,
        /// Findings: undeclared entity types, undeclared roles, undeclared glyphs.
        violations: usize,
        /// `formal:` expressions read.
        formal_total: usize,
        /// Of those, how many carry NO symbol Σ declares — the `formal_prose` debt, shrink-only.
        formal_prose: usize,
    },
    /// ONT-4: the corpus's typed relations, checked against Σ's roles.
    #[serde(rename = "relations")]
    Relations {
        /// Typed edges after symmetric closure.
        relations_n: usize,
        /// Contracts carrying a `relations:` block.
        contracts_with_relations: usize,
        /// `role=count` per role used.
        roles_used: Vec<String>,
        /// Cycles through acyclic roles, as `role: a -> b -> a`.
        cycles: Vec<String>,
        /// `metadata.depends_on` edges, read and counted, never rewritten (R-5).
        legacy_depends_on: usize,
        /// Of those, how many name no contract — ratcheted shrink-only in the baseline.
        legacy_unresolved_depends_on: usize,
        /// Findings.
        violations: usize,
    },
    /// ONT-4b: the shapes gate — every `shape:` block over the extracted graph, with the plant.
    #[serde(rename = "shapes")]
    Shapes {
        /// `shape:` blocks read.
        shapes_n: usize,
        /// Distinct focus nodes across the shapes (the plant excluded).
        focus_nodes_n: usize,
        /// `fired` — the plant drew a violation; the gate never reaches `Ran` otherwise.
        pc_shape: String,
        /// How many violations the plant drew.
        plant_violations: usize,
        /// Corpus violations (the plant excluded).
        violations: usize,
        /// Corpus warnings.
        warnings: usize,
        /// `shape=focus-count` per shape.
        by_shape: Vec<String>,
        /// Triples in the extracted graph.
        triples: usize,
        /// ONT-4c1 (§3.9 per-shape arming): shapes that feed the verdict, in corpus order.
        armed_shapes: Vec<String>,
        /// Shapes computed and reported but not armed — their violations are in `unarmed_violations`.
        not_armed_shapes: Vec<String>,
        /// Violations from unarmed shapes (named in the findings as warnings; never in the meet).
        unarmed_violations: usize,
        /// Focus nodes each extractor produced: `pv-contract`, `gguf`, `apr-model`.
        by_entity_type: std::collections::BTreeMap<String, usize>,
        /// The extractor positive controls: `gguf` (corrupt magic refused), `apr-model` (lying header refused).
        pc_extract: std::collections::BTreeMap<String, String>,
        /// Ladder receipt files read under `evidence/dogfood/models/`.
        receipts: usize,
        /// Receipt rows whose `sha256` equals a rung's.
        witnesses: usize,
        /// Receipt rows whose `sha256` differs from the rung's.
        hex_mismatches: usize,
        /// Receipt rows without a `sha256` (never a witness).
        unmeasured_rows: usize,
        /// ONT-4b2: vendored W3C SHACL-Core cases that passed this run, and how many are vendored.
        w3c_cases_passed: usize,
        w3c_cases_n: usize,
        /// ONT-4b2: bound Rust symbols the `syn` walk resolved / could not resolve.
        symbols_resolved: usize,
        symbols_unresolved: usize,
        /// ONT-4b2: Lean theorems extracted, and contract `lean_theorem:` references naming none of them.
        lean_statements: usize,
        lean_refs_unresolved: usize,
        /// aprender#3715: what `extract:release-evidence` derived — absent unless a release subject was given.
        #[serde(skip_serializing_if = "Option::is_none")]
        release: Option<Box<crate::ontology::extract::release_evidence::ReleaseStats>>,
    },
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
    /// ONT-001 §3.9: the meet over the armed gates — what the exit code reports. `passed` is the legacy
    /// every-gate-passed-or-skipped flag and is kept unchanged for existing readers.
    pub verdict: Verdict,
    /// Each armed gate, in armed order, with the verdict it contributed (`Unknown(NotRun)` if it did not run).
    pub armed_gates: Vec<ArmedGateVerdict>,
    /// Gates that ran but are not armed: printed as `Unknown(NotArmed)` and excluded from the meet.
    pub not_armed: Vec<String>,
    /// The `armed_gates` monotone check, as the CLI measured it; `None` when nothing checked it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub armed_monotone: Option<String>,
    /// ONT-4c1: the `armed_shapes` monotone check, as the CLI measured it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub armed_shapes_monotone: Option<String>,
}

/// One armed gate's contribution to [`LintReport::verdict`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArmedGateVerdict {
    pub name: String,
    pub verdict: Verdict,
}

impl LintReport {
    /// Recompute the meet over `armed`. Gates that ran but are not armed are listed in `not_armed`.
    pub fn arm(&mut self, armed: &ArmedGates) {
        let results: Vec<(String, Verdict)> = self
            .gates
            .iter()
            .map(|g| (g.name.clone(), g.verdict))
            .collect();
        let meet = meet_armed(&results, armed);
        self.verdict = meet.verdict;
        self.armed_gates = meet
            .armed
            .into_iter()
            .map(|(name, verdict)| ArmedGateVerdict { name, verdict })
            .collect();
        self.not_armed = meet.not_armed;
    }
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

/// Run one gate and record it, or record it as skipped when gate 1 (validate) failed.
///
/// Every gate after validate repeats the same `if validation_passed { run } else
/// { skipped_gate(name, "validation failed") }` shape; this is that shape, once.
fn push_gate<F>(
    gates: &mut Vec<GateResult>,
    all_findings: &mut Vec<LintFinding>,
    run: F,
    name: &str,
    validation_passed: bool,
) where
    F: FnOnce() -> (GateResult, Vec<LintFinding>),
{
    let (result, mut findings) = if validation_passed {
        run()
    } else {
        (skipped_gate(name, "validation failed"), Vec::new())
    };
    gates.push(result);
    all_findings.append(&mut findings);
}

/// Per-contract timing: measure how long each contract's findings take to process,
/// sorted by duration descending.
fn per_contract_timings(
    contracts: &[(String, crate::schema::Contract)],
    binding: Option<&crate::binding::BindingRegistry>,
) -> Vec<(String, u64)> {
    let mut contract_timings: Vec<(String, u64)> = Vec::with_capacity(contracts.len());
    for (stem, contract) in contracts {
        let ct_start = Instant::now();
        // Validate
        let _ = crate::schema::validate_contract(contract);
        // Audit
        let _ = crate::audit::audit_contract(contract);
        // Score
        let _ = crate::scoring::score_contract(contract, binding, stem);
        let ct_ms = u64::try_from(ct_start.elapsed().as_micros() / 1000).unwrap_or(0);
        contract_timings.push((format!("{stem}.yaml"), ct_ms));
    }
    // Sort by duration descending
    contract_timings.sort_by_key(|b| std::cmp::Reverse(b.1));
    contract_timings
}

/// Cache: store findings per-contract for future runs.
fn store_findings_in_cache(
    root: &std::path::Path,
    config: &LintConfig,
    contracts: &[(String, crate::schema::Contract)],
    all_findings: &[LintFinding],
    stats: &mut cache::CacheStats,
) {
    let rule_cfg = format!("{:?}{:?}", config.severity_overrides, config.strict);
    for (stem, _) in contracts {
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
    push_gate(
        &mut gates,
        &mut all_findings,
        || run_audit_gate(&contracts),
        "audit",
        validation_passed,
    );

    // Gate 3: score (skip if validation failed)
    push_gate(
        &mut gates,
        &mut all_findings,
        || run_score_gate(&contracts, binding.as_ref(), config.min_score),
        "score",
        validation_passed,
    );

    // Gate 4: verify (source code fulfillment)
    push_gate(
        &mut gates,
        &mut all_findings,
        || {
            let project_root = config.contract_dir.parent().unwrap_or(config.contract_dir);
            run_verify_gate(&contracts, project_root)
        },
        "verify",
        validation_passed,
    );

    // Gate 5: enforce (equations must have preconditions/postconditions)
    push_gate(
        &mut gates,
        &mut all_findings,
        || run_enforce_gate(&contracts),
        "enforce",
        validation_passed,
    );

    // Gate 6: enforcement level (Section 17, Gap 1 + Gap 5 level lock)
    push_gate(
        &mut gates,
        &mut all_findings,
        || {
            let min_level = config
                .min_level
                .unwrap_or(crate::schema::EnforcementLevel::Standard);
            run_enforcement_level_gate(&contracts, min_level)
        },
        "enforcement-level",
        validation_passed,
    );

    // Gate 7: reverse coverage (optional — skip if no binding or crate dir)
    push_gate(
        &mut gates,
        &mut all_findings,
        || match (config.binding_path, config.crate_dir) {
            (Some(bp), Some(cd)) => run_reverse_coverage_gate(bp, cd),
            _ => (
                skipped_gate("reverse-coverage", "no --binding or --crate-dir provided"),
                Vec::new(),
            ),
        },
        "reverse-coverage",
        validation_passed,
    );

    // Gate 8: duplicate stems (PV-DUP-001). Must run BEFORE composition — it tells
    // the composition gate which stems are unresolvable, which is the difference
    // between a defined verdict and one decided by `read_dir` order.
    let duplicates = duplicate_stems::scan_duplicate_stems(config.contract_dir);
    let ambiguous = duplicate_stems::ambiguous_stems(&duplicates);
    push_gate(
        &mut gates,
        &mut all_findings,
        || {
            let project_root = config.contract_dir.parent().unwrap_or(config.contract_dir);
            let baseline = duplicate_stems::read_baseline(project_root);
            duplicate_stems::run_duplicate_stem_gate(&duplicates, &baseline)
        },
        "duplicate-stems",
        validation_passed,
    );

    // Gate 9: composition (assumes/guarantees chain verification)
    push_gate(
        &mut gates,
        &mut all_findings,
        || composition_gate::run_composition_gate(&contracts, &ambiguous),
        "composition",
        validation_passed,
    );

    // Gate 10: sigma (ONT-2b). R-8: a new gate is COMPUTED everywhere and armed per repo — so it runs here as
    // well as under `--gate sigma`, or `armed_gates` could name a gate no run ever computes.
    let (sigma_gate_result, mut sigma_findings) =
        sigma_result(config.contract_dir, validation_passed);
    gates.push(sigma_gate_result);
    all_findings.append(&mut sigma_findings);

    // Gate 11: relations (ONT-4). Same R-8 shape as sigma: computed in every run, armed per repo.
    let (relations_gate_result, mut relations_findings) =
        relations_result(config.contract_dir, validation_passed);
    gates.push(relations_gate_result);
    all_findings.append(&mut relations_findings);

    // Gate 12: shapes (ONT-4b). Same R-8 shape: computed in every run, armed per repo.
    let (shapes_gate_result, mut shapes_findings) =
        shapes_result(config.contract_dir, validation_passed);
    gates.push(shapes_gate_result);
    all_findings.append(&mut shapes_findings);

    // Gate 9: strict test-binding (Issue #1510, opt-in via --strict-test-binding)
    if config.strict_test_binding {
        push_gate(
            &mut gates,
            &mut all_findings,
            || {
                let project_root = config.contract_dir.parent().unwrap_or(config.contract_dir);
                strict_test_binding::run_strict_test_binding_gate(
                    &contracts,
                    project_root,
                    config.strict,
                )
            },
            "strict-test-binding",
            validation_passed,
        );
    }

    all_findings.append(&mut validate_findings);

    // Per-contract timing: measure how long each contract's findings take to process
    if validation_passed {
        contract_timings = per_contract_timings(&contracts, binding.as_ref());
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
        store_findings_in_cache(root, config, &contracts, &all_findings, &mut stats);
    }

    // Apply suppressions, severity overrides, strict mode, and severity filter
    apply_suppressions(&mut all_findings, config);
    apply_severity_overrides(&mut all_findings, config);
    if let Some(min_sev) = config.severity_filter {
        all_findings.retain(|f| f.severity >= min_sev);
    }

    let passed = gates.iter().all(|g| g.passed || g.skipped);

    let mut report = LintReport {
        passed,
        gates,
        total_duration_ms: u64::try_from(overall_start.elapsed().as_millis()).unwrap_or(u64::MAX),
        findings: all_findings,
        cache_stats: stats,
        contract_timings,
        verdict: Verdict::Unknown(crate::ontology::verdict::Reason::NotArmed),
        armed_gates: Vec::new(),
        not_armed: Vec::new(),
        armed_monotone: None,
        armed_shapes_monotone: None,
    };
    // The default set. `pv lint` re-arms from the corpus's `lint-baseline.json` (ONT-001 §3.9): a gate a flag
    // ran but the declaration does not arm is printed and excluded, like every other unarmed gate.
    report.arm(&ArmedGates::default_set());
    report
}

/// ONT-2b: what `--gate <name>` answered. Only `Ran` and `Sigma(Ran)` are verdicts about the corpus.
pub enum NamedGateOutcome {
    /// `--gate` was given a name this build does not compute alone.
    UnknownGate,
    /// The `sigma` gate, which has two non-verdict answers of its own (no Σ, malformed Σ).
    Sigma(sigma_gate::SigmaOutcome),
    /// The `relations` gate (ONT-4), with three non-verdict answers (no Σ, malformed Σ, no typed relations).
    Relations(relations_gate::RelationsOutcome),
    /// The `shapes` gate (ONT-4b), with four non-verdict answers (unsupported shape, no shapes, no focus, control failed).
    Shapes(shapes_gate::ShapesOutcome),
    /// A gate that ran and judged the corpus.
    Ran {
        result: Box<GateResult>,
        findings: Vec<LintFinding>,
    },
}

/// Run ONE named gate and nothing else (ONT-001 §5 ONT-2b).
///
/// Only the gates that are meaningful ALONE are offered: `sigma` reads Σ and the corpus, `validate` parses the
/// corpus. The rest of `run_lint`'s gates are skipped when validation fails, so running one of them by itself would
/// report a verdict whose precondition nobody checked — `UnknownGate` is the honest answer, not a silent pass.
#[must_use]
pub fn run_named_gate(contract_dir: &Path, name: &str) -> NamedGateOutcome {
    run_named_gate_with(contract_dir, name, &shapes_gate::ShapesOptions::default())
}

/// [`run_named_gate`] with the shapes gate's `--shape` / `--release-*` options (aprender#3715).
pub fn run_named_gate_with(
    contract_dir: &Path,
    name: &str,
    shapes_opts: &shapes_gate::ShapesOptions,
) -> NamedGateOutcome {
    match name {
        "relations" => {
            NamedGateOutcome::Relations(relations_gate::run_relations_gate(contract_dir))
        }
        "shapes" => {
            NamedGateOutcome::Shapes(shapes_gate::run_shapes_gate_with(contract_dir, shapes_opts))
        }
        "sigma" => NamedGateOutcome::Sigma(sigma_gate::run_sigma_gate(contract_dir)),
        "validate" => {
            let (contracts, parse_errors) = load_contracts(contract_dir);
            let (result, findings) = run_validate_gate(&contracts, &parse_errors);
            NamedGateOutcome::Ran {
                result: Box::new(result),
                findings,
            }
        }
        _ => NamedGateOutcome::UnknownGate,
    }
}

/// The gate names `--gate` computes alone, for the refusal message.
pub const NAMED_GATES: [&str; 4] = ["relations", "shapes", "sigma", "validate"];

/// The `sigma` gate as `run_lint` reports it. Σ's two non-verdict answers become SKIPPED gates here — under
/// `--gate sigma` they are an exit of their own (decline / error), but inside a full run "skipped" is how the
/// lattice already says "not measured".
fn sigma_result(contract_dir: &Path, validation_passed: bool) -> (GateResult, Vec<LintFinding>) {
    if !validation_passed {
        return (skipped_gate("sigma", "validation failed"), Vec::new());
    }
    match sigma_gate::run_sigma_gate(contract_dir) {
        sigma_gate::SigmaOutcome::Ran { result, findings } => (*result, findings),
        sigma_gate::SigmaOutcome::NoSigma => (
            skipped_gate("sigma", "no contracts/ontology.yaml"),
            Vec::new(),
        ),
        sigma_gate::SigmaOutcome::Malformed(e) => (
            skipped_gate("sigma", &format!("Σ is malformed: {e}")),
            Vec::new(),
        ),
    }
}

/// The `relations` gate as `run_lint` reports it. Its three non-verdict answers become SKIPPED gates here, as
/// sigma's do — under `--gate relations` they are exits of their own (decline / error).
fn relations_result(
    contract_dir: &Path,
    validation_passed: bool,
) -> (GateResult, Vec<LintFinding>) {
    if !validation_passed {
        return (skipped_gate("relations", "validation failed"), Vec::new());
    }
    match relations_gate::run_relations_gate(contract_dir) {
        relations_gate::RelationsOutcome::Ran { result, findings } => (*result, findings),
        relations_gate::RelationsOutcome::NoSigma => (
            skipped_gate("relations", "no contracts/ontology.yaml"),
            Vec::new(),
        ),
        relations_gate::RelationsOutcome::Malformed(e) => (
            skipped_gate("relations", &format!("Σ is malformed: {e}")),
            Vec::new(),
        ),
        relations_gate::RelationsOutcome::NoRelations {
            contracts_checked,
            legacy_depends_on,
        } => (
            skipped_gate(
                "relations",
                &format!("no typed relations in {contracts_checked} contracts ({legacy_depends_on} legacy metadata.depends_on edges) — R-2: zero is a decline"),
            ),
            Vec::new(),
        ),
    }
}

/// The `shapes` gate as `run_lint` reports it. Its four non-verdict answers become SKIPPED gates here (under
/// `--gate shapes` they are exits of their own); an `Unknown{Warn}` run is reported as the gate returned it.
fn shapes_result(contract_dir: &Path, validation_passed: bool) -> (GateResult, Vec<LintFinding>) {
    if !validation_passed {
        return (skipped_gate("shapes", "validation failed"), Vec::new());
    }
    match shapes_gate::run_shapes_gate(contract_dir) {
        shapes_gate::ShapesOutcome::Ran { result, findings } => (*result, findings),
        shapes_gate::ShapesOutcome::Unsupported(e) => (skipped_gate("shapes", &format!("{e}")), Vec::new()),
        shapes_gate::ShapesOutcome::ExtractFailed(e) => (skipped_gate("shapes", &format!("{e}")), Vec::new()),
        shapes_gate::ShapesOutcome::NoShapes { contracts_checked } => (
            skipped_gate("shapes", &format!("no `shape:` block in {contracts_checked} contracts — R-2: zero is a decline")),
            Vec::new(),
        ),
        shapes_gate::ShapesOutcome::NoFocus { shapes_n } => (
            skipped_gate("shapes", &format!("{shapes_n} shape(s), no focus node")),
            Vec::new(),
        ),
        shapes_gate::ShapesOutcome::WrongCorpus { shapes_n, expected, found, refused } => (
            skipped_gate("shapes", &format!("extract:parity-receipt matched {found} focus node(s) and evidence/parity/EXPECTED_RECEIPTS says {expected} ({shapes_n} shape(s)){} — an extractor that saw the wrong corpus reports the same \"no violations\" as one that saw all of it", if refused.is_empty() { String::new() } else { format!("; refused: {}", refused.join("; ")) })),
            Vec::new(),
        ),
        shapes_gate::ShapesOutcome::HarnessBroken { causes } => (
            skipped_gate("shapes", &format!("the CRUX harness measured itself, not apr: {}", causes.join("; "))),
            Vec::new(),
        ),
        shapes_gate::ShapesOutcome::NoReceipts { shapes_n, dir } => (
            skipped_gate("shapes", &format!("{shapes_n} shape(s) resolve receipts and the tree holds none under {dir}/ — R-2: unmeasured is a decline")),
            Vec::new(),
        ),
        shapes_gate::ShapesOutcome::PositiveControlFailed { shapes_n, focus_nodes_n, which } => (
            skipped_gate("shapes", &format!("positive control {which} did not fire ({shapes_n} shape(s), {focus_nodes_n} focus node(s)) — the gate cannot reject")),
            Vec::new(),
        ),
        shapes_gate::ShapesOutcome::Differential { shapes_n, focus_nodes_n, passed, n, failed } => {
            let mut g = skipped_gate(
                "shapes",
                &format!(
                    "W3C SHACL-Core differential: {passed} of {n} vendored case(s) pass; failed: {} ({shapes_n} shape(s), {focus_nodes_n} focus node(s)) — the validator disagrees with the standard, so no corpus verdict",
                    failed.join(" | ")
                ),
            );
            g.verdict = Verdict::Unknown(crate::ontology::verdict::Reason::Differential);
            (g, Vec::new())
        }
    }
}

fn skipped_gate(name: &str, reason: &str) -> GateResult {
    GateResult {
        name: name.into(),
        passed: false,
        skipped: true,
        verdict: Verdict::from_gate(false, true),
        duration_ms: 0,
        detail: GateDetail::Skipped {
            reason: reason.into(),
        },
        extra: None,
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
