//! PV-RAT-004 (aprender#6397) — `references-resolve`: every `metadata.references` entry that is a repo path
//! must name a file or directory that exists.
//!
//! Classification is deliberately conservative, so a hit is never a guess. An entry is a REPO PATH only when its
//! first whitespace-delimited token (minus a trailing `,;.)` and any `::symbol` / `:line` suffix) has a first
//! segment that is a top-level entry of the repo root, contains no glob/brace character, and ends in a file
//! extension. URLs, `../` paths, arXiv/DOI ids and author-year citations are free text here and never fail.
//! The debt (`dead_references`) is shrink-only against `contracts/lint-baseline.json`, like the EV-11 ratchets.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

use crate::schema::parse_contract;

use super::finding::LintFinding;
use super::ratchet_gates::{baseline_of, ratchet_finding, repo_root, result_of, RatchetOutcome};
use super::rules::RuleSeverity;
use super::GateExtra;

/// Gate name, under `--gate` and in a full run.
pub const GATE: &str = "references-resolve";
/// Top-level `lint-baseline.json` key for the dead-reference debt.
pub const DEAD_KEY: &str = "dead_references";
/// Top-level `lint-baseline.json` key for the SET of accepted dead references (`"<stem>: <path>"`). A swap of one
/// dead path for another keeps the count and must still fail, so the ratchet compares members, not just size.
pub const DEAD_SET_KEY: &str = "dead_reference_set";

/// The repo-relative path an entry names, or `None` when the entry is not a repo path.
#[must_use]
pub fn repo_path_of(entry: &str, root: &Path) -> Option<String> {
    let tok = entry.split_whitespace().next()?;
    let tok = tok.trim_matches(|c| c == '"' || c == '\'' || c == '`');
    let tok = tok.trim_end_matches([',', ';', '.', ')']);
    let tok = tok.split("::").next()?;
    let tok = match tok.rfind(':') {
        Some(i)
            if tok[i + 1..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit()) =>
        {
            &tok[..i]
        }
        _ => tok,
    };
    if tok.contains(['{', '}', '*', '?', '[', '<', ':'])
        || tok.contains("..")
        || tok.starts_with("../")
        || tok.starts_with('/')
    {
        return None;
    }
    let (first, rest) = tok.split_once('/')?;
    let ext = rest.rsplit_once('.').map(|(_, e)| e)?;
    if ext.is_empty() || ext.len() > 5 || !ext.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    root.join(first).exists().then(|| tok.to_string())
}

/// The accepted dead-reference set from `lint-baseline.json`, or `None` when the key is absent.
fn baseline_set(contract_dir: &Path) -> Option<BTreeSet<String>> {
    let raw = std::fs::read_to_string(contract_dir.join("lint-baseline.json")).ok()?;
    let doc: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let arr = doc.get(DEAD_SET_KEY)?.as_array()?;
    Some(
        arr.iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
    )
}

/// Findings: with a baseline set, one per dead reference NOT in it (so a swap fails); otherwise the count ratchet.
fn dead_findings(
    contract_dir: &Path,
    baseline: Option<usize>,
    dead: &[String],
) -> Vec<LintFinding> {
    const FIX: &str = "a `metadata.references` repo path that does not exist was added — fix the path or drop the reference";
    let Some(set) = baseline_set(contract_dir) else {
        return ratchet_finding("PV-RAT-004", DEAD_KEY, baseline, dead.len(), FIX)
            .into_iter()
            .collect();
    };
    dead.iter()
        .filter(|d| !set.contains(*d))
        .map(|d| {
            let mut f = LintFinding::new(
                "PV-RAT-004",
                RuleSeverity::Error,
                format!("new dead reference `{d}` is not in the {DEAD_SET_KEY} baseline: {FIX}. The baseline is shrink-only; `make lint-ratchet` only removes members"),
                "contracts/lint-baseline.json".to_string(),
            );
            f.contract_stem = Some("lint-baseline".to_string());
            f
        })
        .collect()
}

/// The `references-resolve` gate over `contract_dir`.
#[must_use]
pub fn run_references_gate(contract_dir: &Path) -> RatchetOutcome {
    let start = Instant::now();
    let root = repo_root(contract_dir);
    let mut files = Vec::new();
    super::collect_yaml_files(contract_dir, &mut files);
    files.sort();
    let (mut checked, mut paths) = (0usize, 0usize);
    let mut dead: Vec<String> = Vec::new();
    for file in &files {
        let Ok(contract) = parse_contract(file) else {
            continue;
        };
        checked += 1;
        let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        for r in &contract.metadata.references {
            if let Some(p) = repo_path_of(r, &root) {
                paths += 1;
                if !root.join(&p).exists() {
                    dead.push(format!("{stem}: {p}"));
                }
            }
        }
    }
    if paths == 0 {
        return RatchetOutcome::Declined(format!(
            "no repo-path reference in {checked} contract(s): nothing was measured"
        ));
    }
    let baseline = baseline_of(contract_dir, DEAD_KEY);
    let findings = dead_findings(contract_dir, baseline, &dead);
    let extra = GateExtra::ReferencesResolve {
        contracts_checked: checked,
        path_references: paths,
        dead_references: dead.len(),
        baseline,
        dead,
        violations: findings.len(),
    };
    RatchetOutcome::Ran {
        result: result_of(GATE, checked, &findings, baseline, start, extra),
        findings,
    }
}

#[cfg(test)]
#[path = "references_gate_tests.rs"]
mod tests;
