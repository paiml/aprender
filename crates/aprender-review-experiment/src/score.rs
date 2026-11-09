//! REX-03 scorer: §2.3 metrics and the §3 hypothesis inputs, over receipts.
//!
//! The denominator is always the EXPECTED item set for a (cell, arm), never
//! the receipts that happen to exist: an item with no admissible receipt is
//! scored as `NotRun{Inadmissible}`, so dropping or corrupting a row can only
//! lower a score. The statistics themselves are the frozen `stats.rs`.

use crate::corpus::{Class, Item};
use crate::receipt::{admissible, localized, parse_verdict, Expect, NotRun, Receipt, Verdict};
use crate::stats::{self, Paired, VsVoters};
use serde::Serialize;
use std::collections::BTreeMap;

/// A proportion with its Wilson 95 % interval (`None` when n = 0).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Ratio {
    pub k: u64,
    pub n: u64,
    pub ci: Option<(f64, f64)>,
}

impl Ratio {
    #[must_use]
    pub fn new(k: u64, n: u64) -> Self {
        Self {
            k,
            n,
            ci: stats::wilson(k, n, stats::Z95),
        }
    }

    /// Point estimate (`None` when n = 0).
    #[must_use]
    pub fn value(&self) -> Option<f64> {
        (self.n > 0).then(|| self.k as f64 / self.n as f64)
    }
}

/// One expected item's outcome for one (cell, arm).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scored {
    pub id: String,
    /// Corpus class; H5 reads class R only (spec v2 §3).
    pub class: Class,
    pub defect: bool,
    pub verdict: Verdict,
    /// A finding names the defect file (only set on a FAIL of a defect).
    pub localized: bool,
    /// sha of the raw output; `None` when nothing ran.
    pub output_sha: Option<String>,
}

/// §2.3 metrics for one (cell, arm).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LaneScore {
    pub items: u64,
    pub parse_rate: Ratio,
    pub recall: Ratio,
    pub precision: Ratio,
    pub false_refute: Ratio,
    pub localization: Ratio,
    pub correct: Ratio,
    pub unparsed: u64,
    pub not_run: BTreeMap<String, u64>,
}

/// Score a lane. Recall's denominator is every defect item, run or not;
/// precision's is every parsed FAIL; `Unparsed` is attempted but never a
/// FAIL or a PASS; `NotRun` is not attempted and never correct.
#[must_use]
pub fn score(rows: &[Scored]) -> LaneScore {
    let n = |f: &dyn Fn(&Scored) -> bool| rows.iter().filter(|r| f(r)).count() as u64;
    let fail = |r: &Scored| r.verdict == Verdict::Fail;
    let mut not_run = BTreeMap::new();
    for r in rows {
        if let Verdict::NotRun(why) = r.verdict {
            *not_run.entry(format!("{why:?}")).or_insert(0) += 1;
        }
    }
    LaneScore {
        items: rows.len() as u64,
        parse_rate: Ratio::new(
            n(&|r| matches!(r.verdict, Verdict::Pass | Verdict::Fail)),
            n(&|r| r.verdict.executed()),
        ),
        recall: Ratio::new(n(&|r| r.defect && fail(r)), n(&|r| r.defect)),
        precision: Ratio::new(n(&|r| r.defect && fail(r)), n(&fail)),
        false_refute: Ratio::new(n(&|r| !r.defect && fail(r)), n(&|r| !r.defect)),
        localization: Ratio::new(
            n(&|r| r.defect && fail(r) && r.localized),
            n(&|r| r.defect && fail(r)),
        ),
        correct: Ratio::new(n(&|r| r.verdict.correct(r.defect)), rows.len() as u64),
        unparsed: n(&|r| r.verdict == Verdict::Unparsed),
        not_run,
    }
}

/// Why a receipt line was not used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejected {
    pub line: usize,
    pub reasons: Vec<String>,
}

/// Turn receipt lines into one [`Scored`] per expected item for (cell, arm,
/// rerun). A row is used only if it is admissible, belongs to the selection,
/// names an expected item with the right diff sha and class, is the only row
/// for that item, and (when it ran) its raw output exists with the recorded
/// sha. The verdict is RE-PARSED from that raw output; the row's own verdict
/// must agree with it.
pub fn collect<F>(
    lines: &str,
    expect: Expect<'_>,
    expected: &[&Item],
    select: impl Fn(&Receipt) -> bool,
    read_raw: F,
) -> (Vec<Scored>, Vec<Rejected>)
where
    F: Fn(&str) -> Option<String>,
{
    let by_id: BTreeMap<&str, &Item> = expected.iter().map(|i| (i.id.as_str(), *i)).collect();
    let mut got: BTreeMap<String, Scored> = BTreeMap::new();
    let mut dup: Vec<String> = Vec::new();
    let mut rejected = Vec::new();
    for (ix, line) in lines
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let r = match admissible(line, expect) {
            Ok(r) if select(&r) => r,
            Ok(_) => continue,
            Err(reasons) => {
                rejected.push(Rejected {
                    line: ix + 1,
                    reasons,
                });
                continue;
            }
        };
        match check(&r, &by_id, &read_raw) {
            Ok(s) => {
                if got.insert(s.id.clone(), s).is_some() {
                    dup.push(r.item_id.clone());
                }
            }
            Err(reasons) => rejected.push(Rejected {
                line: ix + 1,
                reasons,
            }),
        }
    }
    for id in &dup {
        got.remove(id);
        rejected.push(Rejected {
            line: 0,
            reasons: vec![format!("{id}: more than one row")],
        });
    }
    let rows = expected
        .iter()
        .map(|i| {
            got.remove(&i.id).unwrap_or_else(|| Scored {
                id: i.id.clone(),
                class: i.class,
                defect: i.class.is_defect(),
                verdict: Verdict::NotRun(NotRun::Inadmissible),
                localized: false,
                output_sha: None,
            })
        })
        .collect();
    (rows, rejected)
}

fn check<F>(r: &Receipt, by_id: &BTreeMap<&str, &Item>, read_raw: &F) -> Result<Scored, Vec<String>>
where
    F: Fn(&str) -> Option<String>,
{
    let item = by_id
        .get(r.item_id.as_str())
        .ok_or_else(|| vec![format!("{}: not an expected item", r.item_id)])?;
    if item.diff_sha256 != r.item_sha256 || item.class != r.class {
        return Err(vec![format!(
            "{}: item sha/class differs from the corpus",
            r.item_id
        )]);
    }
    let defect = item.class.is_defect();
    let Some(out) = &r.output else {
        return Ok(Scored {
            id: r.item_id.clone(),
            class: item.class,
            defect,
            verdict: r.verdict,
            localized: false,
            output_sha: None,
        });
    };
    let text =
        read_raw(&out.path).ok_or_else(|| vec![format!("{}: raw output missing", r.item_id)])?;
    if crate::corpus::sha256_hex(text.as_bytes()) != out.sha256 {
        return Err(vec![format!("{}: raw output sha mismatch", r.item_id)]);
    }
    let verdict = if r.verdict.executed() {
        let v = parse_verdict(&text);
        if v != r.verdict {
            return Err(vec![format!(
                "{}: recorded {:?}, raw parses {v:?}",
                r.item_id, r.verdict
            )]);
        }
        v
    } else {
        r.verdict
    };
    Ok(Scored {
        id: r.item_id.clone(),
        class: item.class,
        defect,
        verdict,
        localized: verdict == Verdict::Fail && defect && localized(&text, &item.defect),
        output_sha: Some(out.sha256.clone()),
    })
}

fn divergence_by(a: &[Scored], b: &[Scored], bytes: bool) -> (u64, u64) {
    let bm: BTreeMap<&str, &Scored> = b.iter().map(|s| (s.id.as_str(), s)).collect();
    let (mut divergent, mut missing) = (0, 0);
    for s in a {
        match bm.get(s.id.as_str()) {
            Some(t) if s.verdict.executed() && t.verdict.executed() => {
                let differs = s.verdict != t.verdict || (bytes && s.output_sha != t.output_sha);
                divergent += u64::from(differs);
            }
            _ => missing += 1,
        }
    }
    (divergent, missing)
}

/// H2 input (and the descriptive H1 byte count): items whose verdict or
/// output bytes differ between two runs, and items that did not run on both
/// (reported, not silently dropped).
#[must_use]
pub fn divergence(a: &[Scored], b: &[Scored]) -> (u64, u64) {
    divergence_by(a, b, true)
}

/// H1 input (spec v2 §3): items whose VERDICT differs across cells. Output
/// bytes are descriptive for H1 (`divergence`); a byte-only difference does
/// not reject it.
#[must_use]
pub fn verdict_divergence(a: &[Scored], b: &[Scored]) -> (u64, u64) {
    divergence_by(a, b, false)
}

/// H4/H5 input: items where BOTH arms parsed a verdict, paired; the rest are
/// dropped pairwise and counted (analysis plan).
#[must_use]
pub fn paired_parsed(a: &[Scored], b: &[Scored]) -> (Vec<Paired>, u64) {
    let bm: BTreeMap<&str, &Scored> = b.iter().map(|s| (s.id.as_str(), s)).collect();
    let parsed = |v: Verdict| matches!(v, Verdict::Pass | Verdict::Fail);
    let mut out = Vec::new();
    let mut dropped = 0;
    for s in a {
        match bm.get(s.id.as_str()) {
            Some(t) if parsed(s.verdict) && parsed(t.verdict) => out.push(Paired {
                defect: s.defect,
                a_fail: s.verdict == Verdict::Fail,
                b_fail: t.verdict == Verdict::Fail,
            }),
            _ => dropped += 1,
        }
    }
    (out, dropped)
}

/// H5 input (spec v2 §3): `paired_parsed` restricted to items of `class` in
/// `a`. H5 decides on class R; class P is reported beside it (mutant confound).
#[must_use]
pub fn paired_parsed_class(a: &[Scored], b: &[Scored], class: Class) -> (Vec<Paired>, u64) {
    let only: Vec<Scored> = a.iter().filter(|s| s.class == class).cloned().collect();
    paired_parsed(&only, b)
}

/// H4 input (spec v2 §3): items where apr AND both voters parsed a verdict;
/// the rest are dropped and counted.
#[must_use]
pub fn vs_voters_parsed(apr: &[Scored], voters: [&[Scored]; 2]) -> (Vec<VsVoters>, u64) {
    let maps = voters.map(|v| {
        v.iter()
            .map(|s| (s.id.as_str(), s))
            .collect::<BTreeMap<_, _>>()
    });
    let parsed = |v: Verdict| matches!(v, Verdict::Pass | Verdict::Fail);
    let mut out = Vec::new();
    let mut dropped = 0;
    for s in apr {
        match (maps[0].get(s.id.as_str()), maps[1].get(s.id.as_str())) {
            (Some(h), Some(g)) if parsed(s.verdict) && parsed(h.verdict) && parsed(g.verdict) => {
                out.push(VsVoters {
                    defect: s.defect,
                    apr_fail: s.verdict == Verdict::Fail,
                    voter_fail: [h.verdict == Verdict::Fail, g.verdict == Verdict::Fail],
                });
            }
            _ => dropped += 1,
        }
    }
    (out, dropped)
}

/// κ input (spec v2 §3, descriptive): per-item error indicators of (a, b)
/// over items both lanes parsed. Error = not correct.
#[must_use]
pub fn error_pairs(a: &[Scored], b: &[Scored]) -> (Vec<bool>, Vec<bool>) {
    paired_parsed(a, b)
        .0
        .iter()
        .map(|p| (p.a_fail != p.defect, p.b_fail != p.defect))
        .unzip()
}

/// H3 input: per-item correctness of (challenger, champion) over the
/// champion's items; a missing challenger row is incorrect.
#[must_use]
pub fn correctness_pairs(challenger: &[Scored], champion: &[Scored]) -> Vec<(bool, bool)> {
    let cm: BTreeMap<&str, &Scored> = challenger.iter().map(|s| (s.id.as_str(), s)).collect();
    champion
        .iter()
        .map(|c| {
            let ch = cm
                .get(c.id.as_str())
                .is_some_and(|s| s.verdict.correct(s.defect));
            (ch, c.verdict.correct(c.defect))
        })
        .collect()
}

/// H3: one-sided McNemar exact p that the challenger (9B) beats the champion.
#[must_use]
pub fn h3_p(challenger: &[Scored], champion: &[Scored]) -> f64 {
    let (b, c) = stats::discordant(&correctness_pairs(challenger, champion));
    stats::mcnemar_exact_one_sided(b, c)
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod tests;
