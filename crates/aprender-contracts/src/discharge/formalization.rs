//! `pv discharge check --validate-formalization` (PVL-001 EV-8b, #4082): `formalization.yaml` is the tree's record
//! in the mathlib-initiative v0.4 shape, and every field that states a MEASURABLE fact is measured, not trusted.
//!
//! - `main_results` — each must be a theorem `discharge-summary.json` lists (a claimed headline result the
//!   discharge never saw is RED);
//! - `status.axioms` — must be the pinned kernel set [`DEFAULT_AXIOMS`], as a set;
//! - `sorry_count` — must equal the `sorry` tokens the escape scan finds in the tree.
//!
//! `scope`, `review.status` and `automation.methods` are prose: they must be present and non-empty. A missing or
//! unparseable file is RED, never a default: this arm is the one that says the record exists.

use std::collections::BTreeSet;
use std::path::Path;

use serde_yaml::Value;

use super::summary::{self, Summary};
use super::{escapes, Report, Tree, DEFAULT_AXIOMS};

pub const FILE: &str = "formalization.yaml";

fn strings(v: Option<&Value>) -> Option<Vec<String>> {
    v?.as_sequence()?
        .iter()
        .map(|x| x.as_str().map(str::to_string))
        .collect()
}

fn text<'a>(doc: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut v = doc;
    for k in path {
        v = v.get(k)?;
    }
    v.as_str().filter(|s| !s.trim().is_empty())
}

/// The `sorry` tokens the escape scan finds (comments and strings are blanked first).
#[must_use]
pub fn measured_sorry_count(tree: &Tree) -> usize {
    escapes(tree).iter().filter(|e| e.kind == "sorry").count()
}

/// Judge `<lean_dir>/formalization.yaml` against the tree and the summary beside it.
pub fn judge(lean_dir: &Path, tree: &Tree, r: &mut Report) {
    let p = lean_dir.join(FILE);
    let doc: Value = match std::fs::read_to_string(&p)
        .map_err(|e| e.to_string())
        .and_then(|t| serde_yaml::from_str(&t).map_err(|e| e.to_string()))
    {
        Ok(d) => d,
        Err(e) => return r.fail(format!("FORMALIZATION {}: {e}", p.display())),
    };
    let summary = summary::load(&summary::summary_path(lean_dir));
    judge_doc(&doc, summary.as_ref().ok(), measured_sorry_count(tree), r);
    if let Err(e) = summary {
        r.fail(format!("FORMALIZATION main_results cannot be checked: {e}"));
    }
}

/// The pure half of [`judge`]: `summary` is `None` when it could not be read (the caller reports why).
pub fn judge_doc(doc: &Value, summary: Option<&Summary>, sorry_measured: usize, r: &mut Report) {
    let before = r.lines.len();
    judge_prose(doc, r);
    judge_main_results(doc, summary, r);
    judge_axioms(doc, r);
    match doc.get("sorry_count").and_then(Value::as_u64) {
        Some(n) if usize::try_from(n) == Ok(sorry_measured) => {}
        Some(n) => r.fail(format!(
            "FORMALIZATION sorry_count {n} != {sorry_measured} measured by the escape scan"
        )),
        None => r.fail("FORMALIZATION sorry_count is missing or not a count".into()),
    }
    if r.lines.len() == before {
        r.lines.push(format!(
            "ok    {FILE}: main_results in the summary, axioms pinned, sorry_count {sorry_measured}"
        ));
    }
}

fn judge_prose(doc: &Value, r: &mut Report) {
    for path in [&["scope"][..], &["review", "status"]] {
        if text(doc, path).is_none() {
            r.fail(format!(
                "FORMALIZATION {} is missing or empty",
                path.join(".")
            ));
        }
    }
    for (key, v) in [
        ("capstones", doc.get("capstones")),
        (
            "automation.methods",
            doc.get("automation").and_then(|a| a.get("methods")),
        ),
    ] {
        if strings(v).is_none() {
            r.fail(format!(
                "FORMALIZATION {key} is missing or not a list of strings"
            ));
        }
    }
}

fn judge_main_results(doc: &Value, summary: Option<&Summary>, r: &mut Report) {
    let Some(main) = strings(doc.get("main_results")).filter(|m| !m.is_empty()) else {
        return r.fail("FORMALIZATION main_results is missing or empty".into());
    };
    let Some(s) = summary else { return };
    let listed: BTreeSet<&str> = s
        .modules
        .iter()
        .flat_map(|m| m.theorems.iter().map(String::as_str))
        .collect();
    for m in main.iter().filter(|m| !listed.contains(m.as_str())) {
        r.fail(format!(
            "FORMALIZATION main_results {m} is not a theorem {} lists",
            summary::SUMMARY_FILE
        ));
    }
}

fn judge_axioms(doc: &Value, r: &mut Report) {
    let Some(got) = strings(doc.get("status").and_then(|s| s.get("axioms"))) else {
        return r.fail("FORMALIZATION status.axioms is missing or not a list of strings".into());
    };
    let got: BTreeSet<&str> = got.iter().map(String::as_str).collect();
    let pinned: BTreeSet<&str> = DEFAULT_AXIOMS.iter().copied().collect();
    if got != pinned {
        r.fail(format!(
            "FORMALIZATION status.axioms {got:?} != the pinned set {pinned:?}"
        ));
    }
}

#[cfg(test)]
#[path = "formalization_tests.rs"]
mod tests;
