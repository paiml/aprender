//! REX-04 cell admission (`rex-cell-admission-v1`).
//!
//! Every §2.1 cell gets exactly one row: `Admitted` with a parity receipt,
//! `Refused{removed_by}`, or `NotRun{…}`. A cell with no row is silent, and a
//! silent cell makes the whole admission file inadmissible. When no cell is
//! `Admitted`, the summary raises S-7.

use crate::receipt::{is_hex64, NotRun};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEME: &str = "rex-cell-admission-v1";

/// The §7 REX-04 oracle `[X]`. The released `apr parity` (v0.69.3) compares
/// GPU with CPU only; a llama.cpp comparison is recorded under this name.
pub const LLAMA_CPP: &str = "llama.cpp@d1d3c3396";

/// One §2.1 device cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub cell: &'static str,
    pub host: &'static str,
    pub backend: &'static str,
}

/// The six §2.1 cells, in table order.
pub const CELLS: [Cell; 6] = [
    Cell {
        cell: "C1",
        host: "intel",
        backend: "wgpu",
    },
    Cell {
        cell: "C2",
        host: "intel",
        backend: "cpu",
    },
    Cell {
        cell: "C3",
        host: "lambda-labs",
        backend: "cpu",
    },
    Cell {
        cell: "C4",
        host: "gx10",
        backend: "cuda",
    },
    Cell {
        cell: "C5a",
        host: "mini",
        backend: "cpu",
    },
    Cell {
        cell: "C5b",
        host: "mini",
        backend: "metal",
    },
];

/// A parity measurement against the oracle on the exact weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parity {
    pub oracle: String,
    pub cosine: f64,
    pub threshold: f64,
    /// Where the threshold comes from (R-7: no invented thresholds).
    pub threshold_basis: String,
    pub receipt_sha256: String,
}

/// A cell's admission state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state")]
pub enum Status {
    Admitted { parity: Parity },
    Refused { removed_by: String },
    NotRun { reason: NotRun },
}

/// One admission row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub schema: String,
    pub cell: String,
    pub host: String,
    pub backend: String,
    pub apr_tag: String,
    pub apr_sha256: String,
    pub model_id: String,
    pub weights_sha256: String,
    pub prereg_sha: String,
    pub at: String,
    pub status: Status,
}

/// What an admission file resolves to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Summary {
    pub admitted: Vec<String>,
    pub refused: Vec<String>,
    pub not_run: Vec<String>,
    /// S-7: no cell is admissible for (A).
    pub s7: bool,
}

fn row_errors(r: &Row, prereg_sha: &str) -> Vec<String> {
    let mut e = Vec::new();
    let id = &r.cell;
    if r.schema != SCHEME {
        e.push(format!("{id}: schema {}", r.schema));
    }
    match CELLS.iter().find(|c| c.cell == r.cell) {
        None => e.push(format!("{id}: not a §2.1 cell")),
        Some(c) if c.host != r.host || c.backend != r.backend => e.push(format!(
            "{id}: {}/{} is not the declared {}/{}",
            r.host, r.backend, c.host, c.backend
        )),
        Some(_) => {}
    }
    for (k, v) in [
        ("apr_tag", &r.apr_tag),
        ("model_id", &r.model_id),
        ("at", &r.at),
    ] {
        if v.trim().is_empty() || v == "unknown" {
            e.push(format!("{id}: {k} missing"));
        }
    }
    for (k, v) in [
        ("apr_sha256", &r.apr_sha256),
        ("weights_sha256", &r.weights_sha256),
    ] {
        if !is_hex64(v) {
            e.push(format!("{id}: {k} is not a sha256"));
        }
    }
    if r.prereg_sha != prereg_sha {
        e.push(format!("{id}: prereg_sha differs from the lock"));
    }
    match &r.status {
        Status::Admitted { parity: p } => {
            if p.oracle.trim().is_empty() || p.threshold_basis.trim().is_empty() {
                e.push(format!("{id}: Admitted without oracle or threshold basis"));
            }
            if !is_hex64(&p.receipt_sha256) {
                e.push(format!("{id}: Admitted without a parity receipt sha"));
            }
            if !(p.cosine.is_finite() && p.threshold.is_finite() && p.cosine >= p.threshold) {
                e.push(format!(
                    "{id}: Admitted at cosine {} below threshold {}",
                    p.cosine, p.threshold
                ));
            }
        }
        Status::Refused { removed_by } if removed_by.trim().is_empty() => {
            e.push(format!("{id}: Refused without removed_by"));
        }
        _ => {}
    }
    e
}

/// Check an admission file (JSON lines). Every §2.1 cell must appear exactly
/// once with a well-formed row; otherwise the file is rejected as a whole.
///
/// # Errors
/// Every reason the file is inadmissible (a silent cell is one of them).
pub fn check(lines: &str, prereg_sha: &str) -> Result<Summary, Vec<String>> {
    let mut errors = Vec::new();
    let mut rows: BTreeMap<String, Vec<Row>> = BTreeMap::new();
    for (ix, line) in lines
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        match serde_json::from_str::<Row>(line) {
            Ok(r) => {
                errors.extend(row_errors(&r, prereg_sha));
                rows.entry(r.cell.clone()).or_default().push(r);
            }
            Err(err) => errors.push(format!("line {}: {err}", ix + 1)),
        }
    }
    let mut s = Summary {
        admitted: Vec::new(),
        refused: Vec::new(),
        not_run: Vec::new(),
        s7: true,
    };
    for c in CELLS {
        match rows.get(c.cell).map(Vec::as_slice) {
            None | Some([]) => errors.push(format!("{}: silent (no admission row)", c.cell)),
            Some([r]) => match r.status {
                Status::Admitted { .. } => s.admitted.push(c.cell.into()),
                Status::Refused { .. } => s.refused.push(c.cell.into()),
                Status::NotRun { .. } => s.not_run.push(c.cell.into()),
            },
            Some(_) => errors.push(format!("{}: more than one admission row", c.cell)),
        }
    }
    s.s7 = s.admitted.is_empty();
    if errors.is_empty() {
        Ok(s)
    } else {
        Err(errors)
    }
}

#[cfg(test)]
#[path = "admission_tests.rs"]
mod tests;
