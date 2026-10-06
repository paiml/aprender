//! PV-VER-003: the legacy `falsification:` rows that PV-VER-002 does not read. Report-only, against a ratchet.
//!
//! `run_strict_test_binding_gate` walks `falsification_tests[]` only. About 400 contracts carry their rows under
//! the legacy top-level `falsification:` block instead (13 under `falsification_conditions:`), which the schema keeps
//! as a raw [`serde_yaml::Value`]. Those rows were never read: binding `lora-target-selection-v1` row 002 to a real
//! test left the gate at 752 refs / 27 missing, and a dangling name there would have done the same (2026-10-06).
//!
//! Each legacy row is read with PV-VER-002's own rules ([`binding_sources`], [`classify_binding`]) and sorted into
//! one of four classes. A row that is `Unbound` (prose such as "Related tests in crates/…", or no binding field) or
//! `Dangling` (it cites a test that no source defines) is a hole. Today's holes are listed in
//! [`BASELINE_REL_PATH`]. A hole outside that list is reported as an Info finding, and a baseline line that is no
//! longer a hole is reported as stale. Neither changes what any gate accepts: the counts ride in
//! [`GateExtra::LegacyBinding`] and the strict-test-binding gate's `passed` is computed exactly as before. Making
//! the ratchet block is a later change, after three green nights.

use std::collections::BTreeSet;
use std::path::Path;

use serde_yaml::{Mapping, Value};

use crate::schema::{Contract, FalsificationTest};

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::strict_test_binding::{
    binding_sources, cited_names, classify_binding, BindingKind, SourceIndex,
};
use super::GateExtra;

/// The ratchet baseline, relative to the project root: one `<stem>\t<row id>` per line, `#` comments allowed.
pub(crate) const BASELINE_REL_PATH: &str = "scripts/contract_legacy_binding_baseline.txt";

/// The legacy blocks, in the order they are read.
const LEGACY_KEYS: [&str; 2] = ["falsification", "falsification_conditions"];

/// What one legacy row binds to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LegacyClass {
    /// Every cited test resolves in source.
    Bound,
    /// The row declares a shell harness; PV-VER-002 never reads those as Rust tests either.
    Shell,
    /// No test name can be read from the row: prose, `LIVE-PENDING`, or no binding field.
    Unbound,
    /// The row cites these names and no source defines them.
    Dangling(Vec<String>),
}

/// The legacy rows of a contract as `(row id, entry)`. A row with no string `id` is named by its place,
/// `falsification[3]`; a row that does not read as a [`FalsificationTest`] keeps only that id, so it classifies
/// as `Unbound` rather than vanishing. A mapping block (not a list) is one row.
pub(crate) fn legacy_rows(contract: &Contract) -> Vec<(String, FalsificationTest)> {
    let blocks = [&contract.falsification, &contract.falsification_conditions];
    let mut out = Vec::new();
    for (key, block) in LEGACY_KEYS.iter().zip(blocks) {
        let items: Vec<&Value> = match block {
            Some(Value::Sequence(s)) => s.iter().collect(),
            Some(m @ Value::Mapping(_)) => vec![m],
            _ => Vec::new(),
        };
        for (i, item) in items.into_iter().enumerate() {
            let mut m = match item {
                Value::Mapping(m) => m.clone(),
                _ => Mapping::new(),
            };
            let id = match m.get("id").and_then(Value::as_str) {
                Some(id) => id.to_string(),
                None => format!("{key}[{i}]"),
            };
            m.insert("id".into(), id.clone().into());
            let ft =
                serde_yaml::from_value(Value::Mapping(m)).unwrap_or_else(|_| FalsificationTest {
                    id: id.clone(),
                    ..FalsificationTest::default()
                });
            out.push((id, ft));
        }
    }
    out
}

/// Classify one legacy row with PV-VER-002's rules. A shell harness ends the reading, as it does there.
pub(crate) fn classify_row(ft: &FalsificationTest, index: &SourceIndex) -> LegacyClass {
    let mut cited = Vec::new();
    for (_, raw) in binding_sources(ft) {
        let kind = classify_binding(&raw);
        if kind == BindingKind::ShellHarness {
            return LegacyClass::Shell;
        }
        cited.extend(cited_names(kind, &raw));
    }
    if cited.is_empty() {
        return LegacyClass::Unbound;
    }
    let missing: Vec<String> = cited.into_iter().filter(|c| !index.resolves(c)).collect();
    if missing.is_empty() {
        LegacyClass::Bound
    } else {
        LegacyClass::Dangling(missing)
    }
}

/// Read the baseline. A missing file is an empty baseline, so every hole reads as new and nothing is hidden.
pub(crate) fn read_baseline(project_root: &Path) -> BTreeSet<String> {
    std::fs::read_to_string(project_root.join(BASELINE_REL_PATH))
        .unwrap_or_default()
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// Measure every legacy row and compare the holes to `baseline`. Returns the payload and one Info finding per hole
/// outside the baseline; the caller's gate verdict does not read either.
pub(crate) fn measure(
    contracts: &[(String, Contract)],
    index: &SourceIndex,
    baseline: &BTreeSet<String>,
) -> (GateExtra, Vec<LintFinding>) {
    let (mut rows, mut bound, mut shell, mut unbound, mut dangling) = (0, 0, 0, 0, 0);
    let mut holes: Vec<(String, String, String)> = Vec::new();
    for (stem, contract) in contracts {
        for (id, ft) in legacy_rows(contract) {
            rows += 1;
            let why = match classify_row(&ft, index) {
                LegacyClass::Bound => {
                    bound += 1;
                    continue;
                }
                LegacyClass::Shell => {
                    shell += 1;
                    continue;
                }
                LegacyClass::Unbound => {
                    unbound += 1;
                    "names no test PV-VER-002 can resolve".to_string()
                }
                LegacyClass::Dangling(names) => {
                    dangling += 1;
                    format!("cites `{}`, which no source defines", names.join("`, `"))
                }
            };
            holes.push((stem.clone(), id, why));
        }
    }
    let found: BTreeSet<String> = holes
        .iter()
        .map(|(s, id, _)| format!("{s}\t{id}"))
        .collect();
    let mut findings = Vec::new();
    let mut unbaselined = Vec::new();
    for (stem, id, why) in &holes {
        let key = format!("{stem}\t{id}");
        if baseline.contains(&key) {
            continue;
        }
        let mut f = LintFinding::new(
            "PV-VER-003",
            RuleSeverity::Info,
            format!("Legacy falsification row {id} {why}, and it is not in {BASELINE_REL_PATH}"),
            format!("contracts/{stem}.yaml"),
        );
        f.contract_stem = Some(stem.clone());
        f.suggestion = Some(format!(
            "Move {id} into `falsification_tests:` with a `test:` that names a real test fn, so PV-VER-002 reads it."
        ));
        findings.push(f);
        unbaselined.push(key);
    }
    let stale: Vec<String> = baseline.difference(&found).cloned().collect();
    let baselined = found.len() - unbaselined.len();
    (
        GateExtra::LegacyBinding {
            rows,
            bound,
            shell,
            unbound,
            dangling,
            baselined,
            unbaselined,
            stale,
        },
        findings,
    )
}

#[cfg(test)]
#[path = "legacy_binding_tests.rs"]
mod tests;
