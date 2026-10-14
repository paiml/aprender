//! #5291: the no-test debt, as a list that only shrinks and a cap per release.
//!
//! On d97936f470, 763 of the 1913 contracts pv walked carried ZERO falsification
//! tests and every pv gate passed them. Walking the 46 kaizen contracts too
//! (operator ruling via the cop) made it 803 of 1959. SCHEMA-024/025 refuse the written-empty
//! shapes; this module owns the rest:
//!
//! - **SCHEMA-026**: a contract under `contracts/` with no tests is refused
//!   unless its path is on [`NO_TEST_DEBT`]. No new contract joins the debt.
//!   Scoped to the census tree ([`in_census_tree`]): the list names paths in
//!   this repo's `contracts/`, not in every directory of that name.
//! - **stale lines**: a listed contract that now has tests, or no longer
//!   exists, fails [`measure`], so a fix must delete its line.
//! - **the cap**: the list's length may not exceed [`debt_cap`] for the
//!   release being built (operator's four-part split: 803 → 602 → 401 → 200
//!   → 0, one part per release 0.72 to 0.75).
//! - **unwalked**: `.yaml`/`.yml` files under `contracts/` that the walker
//!   never reads ([`crate::lint::UNWALKED_DIRS`], the binding/ontology/corpora
//!   files) are counted and capped at [`UNWALKED_CAP`], so nothing leaves the
//!   count by moving out of the walker's sight. All 15 are NOT contracts,
//!   pending a move out of `contracts/` (#5291); the number never rises.

use std::collections::BTreeSet;
use std::path::{Component, Path};

use super::{is_contract_yaml, parse_contract, Contract};
use crate::error::{Severity, Violation};
use crate::lint::{collect_yaml_files, UNWALKED_DIRS};

/// One path per line, relative to `contracts/`; `#` lines are comments.
pub const NO_TEST_DEBT: &str = include_str!("no_test_debt.txt");

/// `.yaml` files under `contracts/` outside the walker's universe once kaizen
/// is walked: 10 `binding.yaml`, `ontology.yaml`, `external-corpora.yaml` and
/// 3 `publish-manifests/` (`quarantine/` excluded: the census counts it as
/// `quarantined_n`). Lower it with each one moved out.
pub const UNWALKED_CAP: usize = 15;

/// The listed paths.
#[must_use]
pub fn debt_entries() -> BTreeSet<&'static str> {
    NO_TEST_DEBT
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

/// The most entries [`NO_TEST_DEBT`] may hold when building `version`
/// (`major.minor[.patch…]`); `None` when the version does not parse.
#[must_use]
pub fn debt_cap(version: &str) -> Option<usize> {
    let mut parts = version.split('.');
    let major: u64 = parts.next()?.parse().ok()?;
    let minor: u64 = parts.next()?.parse().ok()?;
    if major > 0 {
        return Some(0);
    }
    Some(match minor {
        0..=71 => 803,
        72 => 602,
        73 => 401,
        74 => 200,
        _ => 0,
    })
}

/// `path` relative to its last `contracts` component, `/`-joined; `None` when
/// the path is not inside a `contracts/` tree, is a file the walker does not
/// read as a contract (`ontology.yaml`, `binding.yaml`, …), or sits in a
/// directory the walker skips.
#[must_use]
pub fn rel_to_contracts(path: &Path) -> Option<String> {
    if !is_contract_yaml(path) {
        return None;
    }
    let parts: Vec<&str> = path
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => s.to_str(),
            _ => None,
        })
        .collect();
    let at = parts.iter().rposition(|p| *p == "contracts")?;
    let rel = &parts[at + 1..];
    let (_, dirs) = rel.split_last()?;
    if dirs.iter().any(|d| UNWALKED_DIRS.contains(d)) {
        return None;
    }
    Some(rel.join("/"))
}

/// The census marker: the `contracts/` root the debt list describes is the one
/// the release train writes `census.json` into. Another project's `contracts/`
/// (pmat's corpus fixture) or a scratch `contracts/` has none, and the list
/// says nothing about it.
pub const CENSUS_MARKER: &str = "census.json";

/// Is `path` inside the `contracts/` tree [`NO_TEST_DEBT`] describes? Its
/// deepest `contracts` ancestor must hold [`CENSUS_MARKER`].
#[must_use]
pub fn in_census_tree(path: &Path) -> bool {
    path.ancestors()
        .skip(1)
        .find(|a| a.file_name().and_then(|n| n.to_str()) == Some("contracts"))
        .is_some_and(|root| root.join(CENSUS_MARKER).is_file())
}

/// SCHEMA-026 for one contract read from `path`: [`check_in_tree`] when `path`
/// is [`in_census_tree`], otherwise nothing.
#[must_use]
pub fn check(path: &Path, contract: &Contract) -> Option<Violation> {
    if !in_census_tree(path) {
        return None;
    }
    check_in_tree(path, contract)
}

/// SCHEMA-026 on the path alone, for a `path` already known to be in the
/// census tree.
#[must_use]
pub fn check_in_tree(path: &Path, contract: &Contract) -> Option<Violation> {
    if !contract.falsification_tests.is_empty() {
        return None;
    }
    let rel = rel_to_contracts(path)?;
    if debt_entries().contains(rel.as_str()) {
        return None;
    }
    Some(Violation {
        severity: Severity::Error,
        rule: "SCHEMA-026".to_string(),
        message: format!(
            "contracts/{rel} has no falsification_tests and is not on the #5291 no-test debt \
             list; a new contract must ship at least one test"
        ),
        location: Some("falsification_tests".to_string()),
    })
}

/// The debt as measured over one `contracts/` directory.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct DebtReport {
    /// Walked contracts with zero falsification tests.
    pub no_falsification_tests: usize,
    /// Zero-test contracts the list does not name (SCHEMA-026 refuses each).
    pub unlisted: Vec<String>,
    /// Listed paths that now have tests or are gone: delete the line.
    pub stale: Vec<String>,
    /// Contracts the walker reaches that do not parse (other gates refuse them;
    /// listed here so the count cannot skip them silently).
    pub unparsed: Vec<String>,
    /// `.yaml` files under the directory the walker never reads.
    pub unwalked: usize,
}

/// Measure `dir` against [`NO_TEST_DEBT`].
#[must_use]
pub fn measure(dir: &Path) -> DebtReport {
    let mut files = Vec::new();
    collect_yaml_files(dir, &mut files);
    files.sort();
    let listed = debt_entries();
    let mut report = DebtReport::default();
    let mut zero = BTreeSet::new();
    for file in &files {
        let rel = file
            .strip_prefix(dir)
            .map_or_else(|_| file.display().to_string(), |r| r.display().to_string());
        match parse_contract(file) {
            Ok(c) if c.falsification_tests.is_empty() => {
                report.no_falsification_tests += 1;
                if !listed.contains(rel.as_str()) {
                    report.unlisted.push(rel.clone());
                }
                zero.insert(rel);
            }
            Ok(_) => {}
            Err(_) => report.unparsed.push(rel),
        }
    }
    report.stale = listed
        .iter()
        .filter(|l| !zero.contains(**l))
        .map(|l| (*l).to_string())
        .collect();
    report.unwalked = count_yaml(dir).saturating_sub(files.len());
    report
}

/// `.yaml` files under `dir` that [`collect_yaml_files`] never reads
/// (`quarantine/` excluded).
#[must_use]
pub fn unwalked(dir: &Path) -> usize {
    let mut files = Vec::new();
    collect_yaml_files(dir, &mut files);
    count_yaml(dir).saturating_sub(files.len())
}

/// Every `.yaml` / `.yml` under `dir` except `quarantine/`.
fn count_yaml(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut n = 0;
    for path in entries.flatten().map(|e| e.path()) {
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) != Some("quarantine") {
                n += count_yaml(&path);
            }
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("yaml" | "yml")
        ) {
            // `.yml` too: the walker reads only `.yaml`, so a `.yml` contract
            // is unwalked by definition and must not be invisible here.
            n += 1;
        }
    }
    n
}

#[cfg(test)]
#[path = "no_test_debt_tests.rs"]
mod tests;
