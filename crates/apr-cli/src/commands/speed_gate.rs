//! EXT-19 (aprender#4401): the release-phase consumer of the speed ledger
//! (EXT-001 §3.7, goal G3).
//!
//! One verdict over the whole ledger: every (tag, cell) pair has a row, and
//! every cell's newest measured record clears its shrink-only floor. A hole or
//! a RED cell fails the gate; an unarmed cell (fewer than three prior records)
//! passes, because the ratchet has nothing to hold it to yet.
//!
//! The rendered report is a published surface, so it names verdicts and never
//! a number derived from the arms: T28 makes a comparator ratio illegal in
//! any output, cited or not. The ratio stays inside [`Ratchet`].

use super::speed_ledger::{parse_ledger, ratchet, uncovered, Ratchet};

/// The gate's verdict over one ledger.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GateReport {
    /// (tag, cell) pairs with no ledger row.
    pub holes: Vec<(String, String)>,
    /// Each cell's ratchet verdict, in `cells` order.
    pub cells: Vec<(String, Ratchet)>,
    /// How many (tag, cell) pairs the gate required.
    pub pairs: usize,
}

impl GateReport {
    pub(crate) fn passed(&self) -> bool {
        self.holes.is_empty() && !self.cells.iter().any(|(_, r)| matches!(r, Ratchet::Red { .. }))
    }

    /// The report as printed. Verdicts and tags only — never a ratio or floor.
    pub(crate) fn render(&self) -> String {
        let covered = self.pairs - self.holes.len();
        let mut out = format!("G3 coverage: {covered}/{} (tag, cell) pairs\n", self.pairs);
        for (t, c) in &self.holes {
            out.push_str(&format!("HOLE     {c} @ {t}: no ledger row\n"));
        }
        for (c, r) in &self.cells {
            let line = match r {
                Ratchet::Unarmed { records } => {
                    format!("UNARMED  {c}: {records} measured record(s), arms after 3 prior")
                }
                Ratchet::Green { tag, .. } => format!("GREEN    {c} @ {tag}: at or above its floor"),
                Ratchet::Red { tag, .. } => {
                    format!("RED      {c} @ {tag}: below its shrink-only floor")
                }
            };
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str(if self.passed() { "speed gate: PASS\n" } else { "speed gate: FAIL\n" });
        out
    }
}

/// Judge a JSONL ledger against the tags and cells the release requires.
/// A malformed ledger is an error, never a verdict.
pub(crate) fn gate(ledger_jsonl: &str, tags: &[String], cells: &[String]) -> Result<GateReport, String> {
    if tags.is_empty() || cells.is_empty() {
        return Err("speed gate needs at least one tag and one cell".into());
    }
    let rows = parse_ledger(ledger_jsonl)?;
    let holes = uncovered(&rows, tags, cells)?;
    let cells_v = cells.iter().map(|c| (c.clone(), ratchet(&rows, tags, c))).collect();
    Ok(GateReport { holes, cells: cells_v, pairs: tags.len() * cells.len() })
}

#[cfg(test)]
#[path = "speed_gate_tests.rs"]
mod tests;
