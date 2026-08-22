//! Extended gate implementations: verify, enforce, enforcement-level, level-lock.
//!
//! Split from `gates.rs` to keep file sizes under the 500-line limit.

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use crate::schema::Contract;
use crate::schema::EnforcementLevel;

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::{GateDetail, GateResult};

/// Gate 4: Source verification — do referenced test functions exist in source?
#[allow(clippy::too_many_lines)]
pub(crate) fn run_verify_gate(
    contracts: &[(String, Contract)],
    project_root: &Path,
) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let mut findings = Vec::new();
    let mut total_refs = 0usize;
    let mut missing = 0usize;

    let mut source_tests = HashSet::new();
    // Scan project directories: src/, crates/, generated/, tests/
    for sub in &["src", "crates", "generated", "tests"] {
        let d = project_root.join(sub);
        if d.exists() {
            collect_test_fns(&d, &mut source_tests);
        }
    }
    // Scan workspace member directories (e.g., trueno-gpu/src/)
    // Workspace members have their own src/ and tests/ dirs at root level
    let effective_root = if project_root.as_os_str().is_empty() {
        Path::new(".")
    } else {
        project_root
    };
    let root_canon = effective_root
        .canonicalize()
        .unwrap_or_else(|_| effective_root.to_path_buf());
    if let Ok(entries) = std::fs::read_dir(&root_canon) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.join("Cargo.toml").exists() {
                // Skip dirs already covered (src/, crates/, etc.)
                let name = path.file_name().unwrap_or_default();
                if name == "src" || name == "crates" || name == "tests" {
                    continue;
                }
                for sub in &["src", "tests"] {
                    let d = path.join(sub);
                    if d.exists() {
                        collect_test_fns(&d, &mut source_tests);
                    }
                }
            }
        }
    }
    // Scan sibling repos for downstream tests (e.g., ../entrenar/src/)
    if let Some(parent) = project_root.parent() {
        for (stem, _) in contracts {
            if let Some(repo) = stem.split('/').next() {
                for sub in &["src", "crates"] {
                    let d = parent.join(repo).join(sub);
                    if d.exists() {
                        collect_test_fns(&d, &mut source_tests);
                    }
                }
            }
        }
    }

    for (stem, contract) in contracts {
        for ft in &contract.falsification_tests {
            if let Some(ref test_name) = ft.test {
                let raw = test_name.trim().trim_matches('"');
                let name = raw.rsplit("::").next().unwrap_or(raw);
                if name.starts_with("test_") || name.starts_with("prop_") {
                    total_refs += 1;
                    if !source_tests.contains(name) {
                        missing += 1;
                        let is_gpu = name.contains("gpu")
                            || name.contains("wgpu")
                            || name.contains("cuda")
                            || stem.contains("wgpu")
                            || stem.contains("gpu");
                        let sev = if is_gpu {
                            RuleSeverity::Warning
                        } else {
                            RuleSeverity::Error
                        };
                        let note = if is_gpu {
                            " (may require --features gpu)"
                        } else {
                            ""
                        };
                        let mut f = LintFinding::new(
                            "PV-VER-001",
                            sev,
                            format!("Unfalsifiable: test `{name}` not found in src/{note}"),
                            format!("contracts/{stem}.yaml"),
                        );
                        f.contract_stem = Some(stem.clone());
                        findings.push(f);
                    }
                }
            }
        }
    }

    // Only errors fail the gate; warnings (feature-gated tests) are tolerated
    let error_count = findings
        .iter()
        .filter(|f| f.rule_id == "PV-VER-001" && f.severity == RuleSeverity::Error)
        .count();
    let passed = error_count == 0;
    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    (
        GateResult {
            name: "verify".into(),
            passed,
            skipped: false,
            duration_ms: duration,
            detail: GateDetail::Verify {
                total_refs,
                existing: total_refs - missing,
                missing,
            },
            extra: None,
        },
        findings,
    )
}

fn collect_test_fns(dir: &Path, tests: &mut HashSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_test_fns(&path, tests);
        } else if path.extension().is_some_and(|e| e == "rs") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines() {
                    if let Some(pos) = line.find("fn test_").or_else(|| line.find("fn prop_")) {
                        let rest = &line[pos + 3..];
                        let name: String = rest
                            .chars()
                            .take_while(|c| c.is_alphanumeric() || *c == '_')
                            .collect();
                        if !name.is_empty() {
                            tests.insert(name);
                        }
                    }
                }
            }
        }
    }
}

/// Gate 5: Enforce — equations MUST have preconditions, postconditions, and `lean_theorem`.
/// An equation without enforceable contracts is an unverifiable claim.
pub(crate) fn run_enforce_gate(contracts: &[(String, Contract)]) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let mut findings = Vec::new();
    let mut total_eqs = 0usize;
    let mut with_pre = 0usize;
    let mut with_post = 0usize;
    let mut with_lean = 0usize;

    for (stem, contract) in contracts {
        if !contract.requires_proofs() {
            continue; // non-kernel kinds exempt
        }
        for (eq_name, eq) in &contract.equations {
            total_eqs += 1;
            if eq.preconditions.is_empty() {
                findings.push(LintFinding {
                    rule_id: "PV-ENF-001".into(),
                    severity: RuleSeverity::Warning,
                    message: format!("Equation `{eq_name}` has no preconditions"),
                    file: format!("contracts/{stem}.yaml"),
                    line: None,
                    contract_stem: Some(stem.clone()),
                    suppressed: false,
                    suppression_reason: None,
                    is_new: false,
                    snippet: None,
                    suggestion: Some(format!(
                        "Add to equations.{eq_name}:\n  preconditions:\n    - \"!input.is_empty()\""
                    )),
                    evidence: None,
                });
            } else {
                with_pre += 1;
            }
            if eq.postconditions.is_empty() {
                findings.push(LintFinding {
                    rule_id: "PV-ENF-001".into(),
                    severity: RuleSeverity::Warning,
                    message: format!("Equation `{eq_name}` has no postconditions"),
                    file: format!("contracts/{stem}.yaml"),
                    line: None,
                    contract_stem: Some(stem.clone()),
                    suppressed: false,
                    suppression_reason: None,
                    is_new: false,
                    snippet: None,
                    suggestion: Some(format!(
                        "Add to equations.{eq_name}:\n  postconditions:\n    - \"result.len() > 0\""
                    )),
                    evidence: None,
                });
            } else {
                with_post += 1;
            }
            if eq.lean_theorem.is_some() {
                with_lean += 1;
            } else {
                // Capitalize first letter of equation name for Lean theorem name
                let lean_name = capitalize_first(eq_name);
                findings.push(LintFinding {
                    rule_id: "PV-ENF-002".into(),
                    severity: RuleSeverity::Warning,
                    message: format!(
                        "Equation `{eq_name}` has no lean_theorem — proof recommended"
                    ),
                    file: format!("contracts/{stem}.yaml"),
                    line: None,
                    contract_stem: Some(stem.clone()),
                    suppressed: false,
                    suppression_reason: None,
                    is_new: false,
                    snippet: None,
                    suggestion: Some(format!(
                        "Add to equations.{eq_name}:\n  lean_theorem: \"Theorems.{lean_name}\""
                    )),
                    evidence: None,
                });
            }
        }
    }

    let has_errors = findings.iter().any(|f| f.severity == RuleSeverity::Error);
    let passed = !has_errors || total_eqs == 0;
    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    (
        GateResult {
            name: "enforce".into(),
            passed,
            skipped: false,
            duration_ms: duration,
            detail: GateDetail::Enforce {
                equations_total: total_eqs,
                equations_with_pre: with_pre,
                equations_with_post: with_post,
                equations_with_lean: with_lean,
            },
            extra: None,
        },
        findings,
    )
}

/// Capitalize the first character of a string.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Gate 6: Enforcement level — check contracts meet minimum level.
///
/// `min_level` defaults to `Standard` if not specified. A contract's actual
/// level is derived from its content: Basic (schema only), Standard
/// (has falsification + kani), Strict (all bindings implemented), Proven (Lean).
pub(crate) fn run_enforcement_level_gate(
    contracts: &[(String, Contract)],
    min_level: EnforcementLevel,
) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let mut findings = Vec::new();
    let mut below = 0usize;

    for (stem, contract) in contracts {
        if !contract.requires_proofs() {
            continue; // non-kernel kinds exempt
        }
        let declared = contract
            .metadata
            .enforcement_level
            .unwrap_or(EnforcementLevel::Standard);
        let actual = compute_actual_level(contract);

        // Check: actual level must meet declared level
        if actual < declared {
            findings.push(LintFinding {
                rule_id: "PV-ENF-001".into(),
                severity: RuleSeverity::Warning,
                message: format!(
                    "Contract `{stem}` declares enforcement_level={declared:?} but only achieves {actual:?}"
                ),
                file: format!("contracts/{stem}.yaml"),
                line: None,
                contract_stem: Some(stem.clone()),
                suppressed: false,
                suppression_reason: None,
                is_new: false,
                snippet: None,
                suggestion: None,
                evidence: None,
            });
            below += 1;
        }

        // Check: actual level must meet --min-level
        if actual < min_level {
            findings.push(LintFinding {
                rule_id: "PV-ENF-001".into(),
                severity: RuleSeverity::Warning,
                message: format!(
                    "Contract `{stem}` at level {actual:?}, below required {min_level:?}"
                ),
                file: format!("contracts/{stem}.yaml"),
                line: None,
                contract_stem: Some(stem.clone()),
                suppressed: false,
                suppression_reason: None,
                is_new: false,
                snippet: None,
                suggestion: None,
                evidence: None,
            });
            below += 1;
        }

        // Gate 7: Level lock — cannot regress below locked_level
        if let Some(ref locked) = contract.metadata.locked_level {
            let locked_level = match locked.to_lowercase().as_str() {
                "basic" | "l1" => EnforcementLevel::Basic,
                "strict" | "l4" => EnforcementLevel::Strict,
                "proven" | "l5" => EnforcementLevel::Proven,
                // "standard", "l2", "l3", or anything unrecognized
                _ => EnforcementLevel::Standard,
            };
            if actual < locked_level {
                findings.push(LintFinding {
                    rule_id: "PV-LCK-001".into(),
                    severity: RuleSeverity::Error,
                    message: format!(
                        "Contract `{stem}` locked at {locked} but regressed to {actual:?}. Use `pv unlock` to release."
                    ),
                    file: format!("contracts/{stem}.yaml"),
                    line: None,
                    contract_stem: Some(stem.clone()),
                    suppressed: false,
                    suppression_reason: None,
                    is_new: false,
                    snippet: None,
                    suggestion: None,
                    evidence: None,
                });
            }
        }
    }

    let has_errors = findings.iter().any(|f| f.severity == RuleSeverity::Error);
    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    (
        GateResult {
            name: "enforcement-level".into(),
            passed: !has_errors,
            skipped: false,
            duration_ms: duration,
            detail: GateDetail::Skipped {
                reason: format!("{} contracts, {} below level", contracts.len(), below),
            },
            extra: None,
        },
        findings,
    )
}

/// Derive a contract's actual enforcement level from its content.
fn compute_actual_level(contract: &Contract) -> EnforcementLevel {
    let has_falsification = !contract.falsification_tests.is_empty();
    let has_kani = !contract.kani_harnesses.is_empty();
    let has_lean = contract
        .verification_summary
        .as_ref()
        .is_some_and(|v| v.l4_lean_proved > 0 && v.l4_sorry_count == 0);

    if has_lean {
        EnforcementLevel::Proven
    } else if has_falsification && has_kani {
        EnforcementLevel::Standard
    } else {
        EnforcementLevel::Basic
    }
}

/// Gate 7: Reverse coverage — detect public functions without contract bindings.
///
/// Calls `crate::reverse_coverage::reverse_coverage()` and emits findings
/// for unbound public functions. Fails if coverage falls below the default
/// threshold of 50%.
pub(crate) fn run_reverse_coverage_gate(
    binding_path: &Path,
    crate_dir: &Path,
) -> (GateResult, Vec<LintFinding>) {
    const DEFAULT_THRESHOLD: f64 = 50.0;
    let start = Instant::now();
    let mut findings = Vec::new();

    let report = crate::reverse_coverage::reverse_coverage(crate_dir, binding_path);

    for uf in &report.unbound {
        findings.push(LintFinding {
            rule_id: "PV-RCV-001".into(),
            severity: RuleSeverity::Warning,
            message: format!(
                "Public function `{}` has no contract binding ({}:{})",
                uf.path, uf.file, uf.line,
            ),
            file: uf.file.clone(),
            line: Some(u32::try_from(uf.line).unwrap_or(0)),
            contract_stem: None,
            suppressed: false,
            suppression_reason: None,
            is_new: false,
            snippet: None,
            suggestion: None,
            evidence: None,
        });
    }

    let passed = report.coverage_pct >= DEFAULT_THRESHOLD;
    if !passed {
        findings.push(LintFinding {
            rule_id: "PV-RCV-002".into(),
            severity: RuleSeverity::Error,
            message: format!(
                "Reverse coverage {:.1}% is below threshold {DEFAULT_THRESHOLD:.1}%",
                report.coverage_pct,
            ),
            file: String::new(),
            line: None,
            contract_stem: None,
            suppressed: false,
            suppression_reason: None,
            is_new: false,
            snippet: None,
            suggestion: None,
            evidence: None,
        });
    }

    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    (
        GateResult {
            name: "reverse-coverage".into(),
            passed,
            skipped: false,
            duration_ms: duration,
            detail: GateDetail::ReverseCoverage {
                total_pub_fns: report.total_pub_fns,
                bound_fns: report.bound_fns,
                unbound_fns: report.unbound.len(),
                coverage_pct: report.coverage_pct,
                threshold_pct: DEFAULT_THRESHOLD,
            },
            extra: None,
        },
        findings,
    )
}

#[cfg(test)]
#[allow(clippy::all)]
#[path = "gates_extended_tests.rs"]
mod tests;

/// Detect stale suppressions: rules/findings that were suppressed but no
/// longer fire. Returns PV-SUP-001 findings for each stale suppression.
pub(crate) fn check_stale_suppressions(
    findings: &[LintFinding],
    suppressed_rules: &[String],
    suppressed_findings: &[String],
) -> Vec<LintFinding> {
    let mut stale = Vec::new();

    // Check rule suppressions: does the suppressed rule still fire?
    let active_rules: HashSet<&str> = findings.iter().map(|f| f.rule_id.as_str()).collect();
    for rule in suppressed_rules {
        if !active_rules.contains(rule.as_str()) {
            stale.push(LintFinding {
                rule_id: "PV-SUP-001".into(),
                severity: RuleSeverity::Warning,
                message: format!(
                    "Suppression for rule `{rule}` is stale — the rule no longer fires. Remove the suppression."
                ),
                file: String::new(),
                line: None,
                contract_stem: None,
                suppressed: false,
                suppression_reason: None,
                is_new: false,
                snippet: None,
                suggestion: None,
                evidence: None,
            });
        }
    }

    // Check finding suppressions: does the suppressed contract still have findings?
    let active_stems: HashSet<&str> = findings
        .iter()
        .filter_map(|f| f.contract_stem.as_deref())
        .collect();
    for stem in suppressed_findings {
        if !active_stems.contains(stem.as_str()) {
            stale.push(LintFinding {
                rule_id: "PV-SUP-001".into(),
                severity: RuleSeverity::Warning,
                message: format!(
                    "Suppression for `{stem}` is stale — no findings exist for this contract. Remove the suppression."
                ),
                file: String::new(),
                line: None,
                contract_stem: Some(stem.clone()),
                suppressed: false,
                suppression_reason: None,
                is_new: false,
                snippet: None,
                suggestion: None,
                evidence: None,
            });
        }
    }

    stale
}
