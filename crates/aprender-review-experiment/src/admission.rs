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

/// The interim oracle of §4.1 rule 1 until llama.cpp is a declared input: the
/// released `apr parity` GPU-vs-CPU comparison (plus H1 in the analysis).
pub const APR_GPU_CPU: &str = "apr-parity-gpu-cpu";

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

/// What a parity receipt must match before it admits a cell: the row's own
/// binary and weights, the declared threshold and its basis, and the evidence
/// floor (`min_positions` of `evidence/parity/thresholds.yaml`).
#[derive(Debug, Clone, Copy)]
pub struct Expect<'a> {
    pub apr_sha256: &'a str,
    pub weights_sha256: &'a str,
    pub threshold: f64,
    pub threshold_basis: &'a str,
    pub min_positions: usize,
}

/// Derive an `Admitted` parity block from the bytes of a parity receipt.
///
/// # Errors
/// Every reason the receipt cannot admit the cell.
pub fn parity_from_receipt(receipt: &[u8], x: &Expect<'_>) -> Result<Parity, Vec<String>> {
    use serde_json::Value;
    let v: Value =
        serde_json::from_slice(receipt).map_err(|e| vec![format!("receipt is not JSON: {e}")])?;
    let mut e = Vec::new();
    // (oracle, cosines, weights sha, binary sha) per receipt shape
    let (oracle, cos, weights, binary): (String, Vec<Option<f64>>, Option<&str>, Option<&str>) =
        if v["schema"].as_str() == Some("apr-parity-oracle/v1") {
            if v["verdict"].as_str() != Some("GREEN") {
                e.push(format!("verdict {} is not GREEN", v["verdict"]));
            }
            let per = v["per_position"].as_array().map_or(&[][..], Vec::as_slice);
            (
                v["oracle"].as_str().unwrap_or_default().to_string(),
                per.iter().map(|p| p["cosine"].as_f64()).collect(),
                v["subject"]["producer"]["model_sha256"].as_str(),
                None,
            )
        } else if v.get("parity").is_some() && v.get("comparator").is_some() {
            if v["exit"].as_i64() != Some(0) {
                e.push(format!("exit {} is not 0", v["exit"]));
            }
            if v["verdict"].as_str() != Some("pass") {
                e.push(format!("verdict {} is not pass", v["verdict"]));
            }
            let p = &v["parity"];
            if p["failed"].as_u64() != Some(0) || p["parity"].as_bool() != Some(true) {
                e.push(format!(
                    "parity failed {} (parity {})",
                    p["failed"], p["parity"]
                ));
            }
            let comparator = v["comparator"].as_str().unwrap_or_default();
            let oracle = if comparator.starts_with("apr-cpu") {
                APR_GPU_CPU.to_string()
            } else {
                comparator.to_string()
            };
            let metrics = p["metrics"].as_array().map_or(&[][..], Vec::as_slice);
            (
                oracle,
                metrics
                    .iter()
                    .map(|m| m["cosine_similarity"].as_f64())
                    .collect(),
                v["weights_sha256"].as_str(),
                Some(v["binary_sha256"].as_str().unwrap_or_default()),
            )
        } else {
            return Err(vec![
                "not a parity receipt (apr-parity-oracle/v1 or apr-review-serve parity)".into(),
            ]);
        };
    if oracle != LLAMA_CPP && oracle != APR_GPU_CPU {
        e.push(format!("oracle {oracle:?} is not a declared parity.oracle"));
    }
    if weights != Some(x.weights_sha256) {
        e.push(format!(
            "weights sha {weights:?} is not the row's {}",
            x.weights_sha256
        ));
    }
    if binary.is_some_and(|b| b != x.apr_sha256) {
        e.push(format!(
            "binary sha {binary:?} is not the row's {}",
            x.apr_sha256
        ));
    }
    if cos.len() < x.min_positions {
        e.push(format!(
            "{} positions < min_positions {}",
            cos.len(),
            x.min_positions
        ));
    }
    if !x.threshold.is_finite() {
        e.push(format!("threshold {} is not finite", x.threshold));
    }
    if x.threshold_basis.trim().is_empty() {
        e.push("no threshold basis declared".into());
    }
    let mut min = f64::INFINITY;
    for (i, c) in cos.iter().enumerate() {
        match c {
            // JSON carries no NaN or infinity: a number is finite
            Some(c) => min = min.min(*c),
            None => e.push(format!("position {i}: no numeric cosine")),
        }
    }
    if min < x.threshold {
        e.push(format!("min cosine {min} below threshold {}", x.threshold));
    }
    if e.is_empty() {
        Ok(Parity {
            oracle,
            cosine: min,
            threshold: x.threshold,
            threshold_basis: x.threshold_basis.to_string(),
            receipt_sha256: crate::corpus::sha256_hex(receipt),
        })
    } else {
        Err(e)
    }
}

#[cfg(test)]
#[path = "admission_tests.rs"]
mod tests;
