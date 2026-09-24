//! PVL-001 EV-7b (#4201): the comparator — does each solution close EXACTLY the statement its Challenge pins?
//!
//! EV-7a writes `<lean-dir>/Challenge/<contract-stem>.lean`: one `theorem PvlChallenge.F <statement> := sorry` per
//! bound root F, the statement lifted from F itself. `scripts/Comparator.lean` (run by `pv discharge check
//! --comparator` as `lake env lean --run scripts/Comparator.lean Challenge/…`) elaborates each Challenge file
//! against the built tree and prints one NDJSON [`Row`] per challenge. This module parses and JUDGES those rows;
//! the Lean side only measures.
//!
//! | row | verdict |
//! |---|---|
//! | `solution_type_hash: null` | FAIL `MISSING-ROOT` — the challenge pins a theorem that does not exist |
//! | hashes differ, `defeq_instances: true` | MATCH — the same statement through a different instance path |
//! | hashes differ otherwise (`false` or absent) | FAIL `MISMATCH` — a different statement (a weakened one, say) |
//! | `sorryAx` among the solution's axioms | FAIL `SORRY` — it closes nothing |
//! | a name twice | FAIL `DUPLICATE` |
//! | no `challenge_type_hash`, or a solution with no `axioms` | FAIL `MALFORMED` — unmeasured is never closed |
//! | otherwise | closed |
//!
//! MATCH = hashes equal OR `isDefEq` at `.instances` transparency (cop ruling on #4237). The hash stays the
//! fingerprint: `defeq_instances` is measured only when the hashes differ, and an absent one is `MISMATCH`.
//!
//! Zero rows is not a pass: [`judge_rows`] declines (rc 2).
//!
//! Rows are cross-checked against the roots the Challenge files DECLARE (#4240): `Comparator.lean`'s `rowsOf`
//! drops any name that trips `isInternal`, and a dropped root emits no row, so `n/m` would understate `m` with
//! nothing noticing. [`cross_check`] counts every `theorem _root_.PvlChallenge.F` line ([`expected_roots`]) and
//! fails `MISSING-ROW` for a root with no row and `UNEXPECTED-ROW` for a row with no root; `m` is the union.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::Report;

/// The comparator script, relative to the Lean dir.
pub const COMPARATOR: &str = "scripts/Comparator.lean";
/// EV-7a's Challenge directory, relative to the Lean dir.
pub const CHALLENGE_DIR: &str = "Challenge";
/// The namespace every challenge declaration lives under.
pub const CHALLENGE_NS: &str = "PvlChallenge";
/// The axiom a `sorry` elaborates to.
pub const SORRY_AXIOM: &str = "sorryAx";

/// THE naming rule: solution `F` is pinned by the challenge `PvlChallenge.F`. `Comparator.lean`'s `solutionOf?`
/// is its inverse; `solution_of` here is that inverse too, and the two are tested to round-trip.
#[must_use]
pub fn challenge_decl(solution: &str) -> String {
    format!("{CHALLENGE_NS}.{solution}")
}

/// `PvlChallenge.F` → `F`; anything not under the namespace (or the bare namespace) → `None`.
#[must_use]
pub fn solution_of(challenge: &str) -> Option<&str> {
    challenge
        .strip_prefix(CHALLENGE_NS)
        .and_then(|s| s.strip_prefix('.'))
        .filter(|s| !s.is_empty())
}

/// One comparator row: a challenge and the solution it names.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct Row {
    /// The SOLUTION's full name.
    pub name: String,
    pub challenge_type_hash: Option<String>,
    pub solution_type_hash: Option<String>,
    /// Measured only when the hashes differ: the two types are defeq at `.instances` transparency.
    #[serde(default)]
    pub defeq_instances: Option<bool>,
    /// The solution's axioms; `None` when there is no solution.
    pub axioms: Option<Vec<String>>,
}

/// How many challenges the rows close: EV-8a's `challenges_closed: "n/m"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Closure {
    pub closed: usize,
    pub total: usize,
}

/// Parse the comparator's stdout. Blank lines are skipped; any other line that is not a [`Row`] is an error
/// naming its line number — a half-read stream is never judged.
pub fn parse_rows(ndjson: &str) -> Result<Vec<Row>, String> {
    ndjson
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str::<Row>(l)
                .map_err(|e| format!("comparator output line {}: {e}: {l}", i + 1))
        })
        .collect()
}

/// `Challenge/*.lean` under `lean_dir`, relative to it, sorted. A missing directory is an empty list.
#[must_use]
pub fn challenge_files(lean_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(lean_dir.join(CHALLENGE_DIR)) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("lean"))
        .filter_map(|p| p.file_name().map(|n| Path::new(CHALLENGE_DIR).join(n)))
        .collect();
    out.sort();
    out
}

/// The prefix EV-7a's `render` writes on every challenge declaration.
const ROOT_DECL: &str = "theorem _root_.";

/// The solution names the Challenge files declare roots for: `theorem _root_.PvlChallenge.F[.{u}] …` ↦ `F`.
/// Scans the text, never the elaborated environment, so it cannot share a filter with `rowsOf`.
#[must_use]
pub fn roots_in(text: &str) -> BTreeSet<String> {
    text.lines()
        .filter_map(|l| l.trim_start().strip_prefix(ROOT_DECL))
        .filter_map(|rest| {
            let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
            let decl = &rest[..end];
            let decl = decl.find(".{").map_or(decl, |i| &decl[..i]);
            solution_of(decl).map(str::to_string)
        })
        .collect()
}

/// [`roots_in`] over `files` (relative to `lean_dir`). An unreadable file is an error: its roots were never counted.
pub fn expected_roots(lean_dir: &Path, files: &[PathBuf]) -> Result<BTreeSet<String>, String> {
    let mut out = BTreeSet::new();
    for f in files {
        let p = lean_dir.join(f);
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        out.extend(roots_in(&text));
    }
    Ok(out)
}

/// Rows vs declared roots (#4240): a root with no row is `MISSING-ROW`, a row with no root `UNEXPECTED-ROW`, both
/// rc 1. `c.total` becomes the size of the union, so a dropped row can never shrink `m`.
pub fn cross_check(rows: &[Row], roots: &BTreeSet<String>, c: &mut Closure, r: &mut Report) {
    let names: BTreeSet<&str> = rows.iter().map(|row| row.name.as_str()).collect();
    let mut missing = 0;
    for root in roots.iter().filter(|n| !names.contains(n.as_str())) {
        missing += 1;
        r.fail(format!(
            "MISSING-ROW {root} -- {} is declared in a Challenge file but the comparator emitted no row for it",
            challenge_decl(root)
        ));
    }
    for n in names.iter().filter(|n| !roots.contains(**n)) {
        r.fail(format!(
            "UNEXPECTED-ROW {n} -- the comparator reported {} but no Challenge file declares it",
            challenge_decl(n)
        ));
    }
    c.total = names.len() + missing;
    r.lines.push(format!(
        "COMPARATOR rows cross-checked: {} row(s), {} declared root(s)",
        names.len(),
        roots.len()
    ));
}

/// Judge the rows into `r`. Failures are judged before the zero-row decline.
pub fn judge_rows(rows: &[Row], r: &mut Report) -> Closure {
    let mut seen = BTreeSet::new();
    let mut closed = 0;
    for row in rows {
        let n = &row.name;
        let ch = challenge_decl(n);
        if !seen.insert(n.as_str()) {
            r.fail(format!(
                "DUPLICATE {ch} -- the comparator reported it twice"
            ));
            continue;
        }
        match judge_row(row, &ch) {
            Err(fail) => {
                r.fail(fail);
                continue;
            }
            Ok(Some(line)) => r.lines.push(line),
            Ok(None) => {}
        }
        closed += 1;
    }
    let c = Closure {
        closed,
        total: rows.len(),
    };
    r.lines.push(format!(
        "COMPARATOR {}/{} challenge(s) closed",
        c.closed, c.total
    ));
    if rows.is_empty() && !r.reject {
        r.decline = Some("comparator: 0 challenge rows -- nothing was compared".to_string());
    }
    c
}

/// One row's verdict: `Err(FAIL line)`, or `Ok` with the `MATCH(instances)` line a defeq-only close adds.
fn judge_row(row: &Row, ch: &str) -> Result<Option<String>, String> {
    let n = &row.name;
    let (Some(c), sol) = (&row.challenge_type_hash, &row.solution_type_hash) else {
        return Err(format!("MALFORMED {ch} -- no challenge_type_hash"));
    };
    let Some(s) = sol else {
        return Err(format!(
            "MISSING-ROOT {n} -- {ch} pins a theorem that does not exist"
        ));
    };
    let Some(axioms) = &row.axioms else {
        return Err(format!(
            "MALFORMED {ch} -- a solution with no axioms list: its sorry-freedom was never measured"
        ));
    };
    if c != s && row.defeq_instances != Some(true) {
        return Err(format!(
            "MISMATCH {n} -- it proves a different statement than {ch} pins (challenge {c}, solution {s})"
        ));
    }
    if axioms.iter().any(|a| a == SORRY_AXIOM) {
        return Err(format!(
            "SORRY {n} -- the solution rests on {SORRY_AXIOM}: it closes nothing"
        ));
    }
    Ok((c != s).then(|| {
        format!(
            "MATCH(instances) {n} -- defeq to {ch} at .instances transparency (challenge {c}, solution {s})"
        )
    }))
}

#[cfg(test)]
#[path = "comparator_tests.rs"]
mod tests;
