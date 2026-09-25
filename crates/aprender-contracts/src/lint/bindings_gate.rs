//! ONT-001 ONT-3a — the `bindings` gate: every symbol a `binding.yaml` states `implemented` or `partial` resolves,
//! through the `syn` walk of [`crate::ontology::extract::code`], to an item in a workspace member. Fail-closed: a
//! binding that does not resolve is a rejection unless `binding-allowlist.json` names it, with a reason and a ticket.
//!
//! Rules (`reject:`, exit 1):
//!
//! - PV-ONT-028 — an `implemented`/`partial` binding does not resolve, and the allowlist does not name it;
//! - PV-ONT-029 — an allowlist entry names a symbol that now resolves, or that no registry binds. The allowlist is
//!   shrink-only: a fixed ghost must leave it, or the next ghost at that path would pass unseen;
//! - PV-ONT-030 — `binding-allowlist.json` is not `{"entries": [{symbol, reason, ticket}]}`, or names a symbol twice.
//!
//! `not_implemented` and `pending` bindings claim no code and are not resolved. Declines (exit 2, `NoCheckable`): the
//! corpus's parent has no `[workspace]` manifest (the fixture corpora), no registry binds an `implemented`/`partial`
//! symbol, or the resolver's positive control did not fire — a walker that resolved everything would pass every
//! ghost.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::binding::ImplStatus;
use crate::ontology::extract::code::{positive_control, resolve_all};
use crate::ontology::verdict::Verdict;

use super::finding::LintFinding;
use super::ratchet_gates::RatchetOutcome;
use super::rules::RuleSeverity;
use super::{GateDetail, GateExtra, GateResult};

/// The gate's name, under `--gate` and in a full run.
pub const GATE: &str = "bindings";

/// The allowlist, beside the registries. JSON, so no YAML contract walker mistakes it for a contract.
pub const ALLOWLIST: &str = "binding-allowlist.json";

/// One known ghost: a bound symbol that does not resolve today, and the ticket that owns fixing it.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AllowEntry {
    pub symbol: String,
    pub reason: String,
    pub ticket: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AllowFile {
    entries: Vec<AllowEntry>,
}

/// The counters `--gate bindings` reports, flattened into [`GateExtra::Bindings`]'s JSON.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BindingsCounters {
    /// `binding.yaml` registries read.
    pub registries: usize,
    /// `implemented`/`partial` bindings checked (one per registry row, so a symbol bound twice counts twice).
    pub checked: usize,
    pub resolved: usize,
    /// Unresolved and named by the allowlist.
    pub allowlisted: usize,
    /// Unresolved and not allowlisted (PV-ONT-028).
    pub ghosts: usize,
    /// Allowlist entries that no longer name an unresolved binding (PV-ONT-029).
    pub stale_allowlist: usize,
    /// Distinct crate roots the walk entered: one would mean the workspace was never walked.
    pub crates_scanned: usize,
    pub files_parsed: usize,
    /// `fired` — the resolver found a present item and refused an absent one this run.
    pub pc_resolver: String,
    /// ONT-001 row ONT-3a's probe names the extractor's positive control `pc_extract` (R-3's `pc_*` set); it is the
    /// same control as `pc_resolver`, reported under the spec's name so the row's probe reads it (#4072).
    pub pc_extract: String,
    /// The bindings ONT-3a's probe counts as unresolved: bound, not resolving, NOT allowlisted — `ghosts` under the
    /// spec's name. An allowlisted ghost is accounted for (`allowlisted`), so it is not unresolved here (#4072).
    pub unresolved: usize,
    pub violations: usize,
}

/// Read `<contract_dir>/binding-allowlist.json`. Absent is an empty allowlist; malformed is a finding.
fn read_allowlist(contract_dir: &Path) -> Result<Vec<AllowEntry>, String> {
    let path = contract_dir.join(ALLOWLIST);
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return Ok(Vec::new());
    };
    let file: AllowFile = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    let mut seen = BTreeSet::new();
    for e in &file.entries {
        if e.symbol.trim().is_empty() || e.reason.trim().is_empty() || e.ticket.trim().is_empty() {
            return Err(format!(
                "entry `{}` has an empty symbol, reason or ticket",
                e.symbol
            ));
        }
        if !seen.insert(e.symbol.as_str()) {
            return Err(format!("`{}` is listed twice", e.symbol));
        }
    }
    Ok(file.entries)
}

/// Run the gate over `contract_dir`.
#[must_use]
pub fn run_bindings_gate(contract_dir: &Path) -> RatchetOutcome {
    let start = Instant::now();
    if !positive_control() {
        return RatchetOutcome::Declined(
            "positive control pc_resolver did not fire: the resolver cannot tell a present item from an absent one"
                .into(),
        );
    }
    let r = resolve_all(contract_dir);
    if !r.at_workspace_root {
        return RatchetOutcome::Declined(format!(
            "not at a workspace root: `{}` has no [workspace] Cargo.toml beside it, so no binding can be resolved",
            contract_dir.display()
        ));
    }
    let claimed: Vec<_> = r
        .symbols
        .iter()
        .filter(|(b, _)| matches!(b.status, ImplStatus::Implemented | ImplStatus::Partial))
        .collect();
    if claimed.is_empty() {
        return RatchetOutcome::Declined(
            "no binding.yaml binds an implemented or partial symbol — nothing was measured".into(),
        );
    }

    let mut findings = Vec::new();
    let allow = match read_allowlist(contract_dir) {
        Ok(a) => a,
        Err(why) => {
            findings.push(LintFinding::new(
                "PV-ONT-030",
                RuleSeverity::Error,
                format!("{ALLOWLIST} is malformed: {why}"),
                format!("contracts/{ALLOWLIST}"),
            ));
            Vec::new()
        }
    };
    // Keyed by the symbol path alone, not (contract, equation): resolution is a function of the path, so every row
    // binding the same path is the same ghost. A NEW path is never covered by an old entry.
    let allowed: BTreeMap<&str, &AllowEntry> =
        allow.iter().map(|e| (e.symbol.as_str(), e)).collect();

    let mut c = BindingsCounters {
        registries: r.stats.registries,
        checked: claimed.len(),
        crates_scanned: r.stats.crates_scanned,
        files_parsed: r.stats.files_parsed,
        pc_resolver: crate::ontology::witness::FIRED.to_string(),
        pc_extract: crate::ontology::witness::FIRED.to_string(),
        ..BindingsCounters::default()
    };
    let mut unresolved: BTreeSet<String> = BTreeSet::new();
    let mut reported: BTreeSet<String> = BTreeSet::new();
    for (b, found) in claimed {
        let symbol = format!("{}::{}", b.module_path, b.function);
        match found {
            Ok(_) => c.resolved += 1,
            Err(_) if allowed.contains_key(symbol.as_str()) => {
                c.allowlisted += 1;
                unresolved.insert(symbol);
            }
            Err(u) => {
                c.ghosts += 1;
                if reported.insert(symbol.clone()) {
                    findings.push(LintFinding::new(
                        "PV-ONT-028",
                        RuleSeverity::Error,
                        format!(
                            "`{symbol}` ({} {}) is bound `{}` and does not resolve: {}",
                            b.contract,
                            b.equation,
                            format!("{:?}", b.status).to_lowercase(),
                            u.reason
                        ),
                        b.contract.clone(),
                    ));
                }
                unresolved.insert(symbol);
            }
        }
    }
    for e in &allow {
        if !unresolved.contains(&e.symbol) {
            c.stale_allowlist += 1;
            findings.push(LintFinding::new(
                "PV-ONT-029",
                RuleSeverity::Error,
                format!(
                    "{ALLOWLIST} names `{}` ({}), which no implemented/partial binding leaves unresolved — remove the entry",
                    e.symbol, e.ticket
                ),
                format!("contracts/{ALLOWLIST}"),
            ));
        }
    }

    c.unresolved = c.ghosts;
    c.violations = findings.len();
    let verdict = if findings.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Fail
    };
    let result = GateResult {
        name: GATE.into(),
        passed: verdict == Verdict::Pass,
        skipped: false,
        verdict,
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        detail: GateDetail::Validate {
            contracts: c.checked,
            errors: findings.len(),
            warnings: 0,
            error_messages: findings.iter().map(|f| f.message.clone()).collect(),
        },
        extra: Some(GateExtra::Bindings(Box::new(c))),
    };
    RatchetOutcome::Ran {
        result: Box::new(result),
        findings,
    }
}

#[cfg(test)]
#[path = "bindings_gate_tests.rs"]
mod tests;
