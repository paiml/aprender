//! TRACE-001 TR-10: bind every tracing surface to a CRUX contract (§2.4).
//!
//! The CRUX contracts are enumerated through the contract loader
//! ([`crate::schema::parse_contract`]), the same parser `pv lint` uses, not by
//! grepping YAML. Three fields TR-10 needs are NOT in the typed schema:
//! `metadata.status`, `metadata.category` and `falsification_tests[].discharge_status`.
//! `Contract` is not `deny_unknown_fields`, so the loader drops them silently.
//! They are read from the raw YAML here and named in [`Receipt::loader_gap`],
//! so every receipt records the gap instead of hiding it.
//!
//! G14: a contract that is `status: active` with no `discharge_status` on any
//! falsifier is a claim nothing discharged. [`g14_violations`] lists them.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_yaml::Value;

use crate::schema::{is_contract_yaml, parse_contract};

/// Schema id of the receipt TR-10 emits.
pub const RECEIPT_SCHEMA: &str = "crux-trace-bind-receipt-v1";

/// Fields TR-10 reads that the typed loader does not deserialize.
pub const LOADER_GAP: [&str; 3] = [
    "metadata.status",
    "metadata.category",
    "falsification_tests[].discharge_status",
];

/// What §2.4 says to do with a surface's contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    /// The existing contract covers the surface as is.
    Reuse,
    /// The existing contract is extended (re-discharge or status change).
    Extend,
    /// The existing contract is blocked on another row.
    Unblock,
    /// No contract exists yet; TR-11 defines it.
    New,
    /// An instrument, not a product claim; no CRUX contract applies.
    Instrument,
}

/// One row of the §2.4 coverage matrix.
#[derive(Debug, Clone, Copy)]
pub struct Surface {
    /// The tracing surface as a user reaches it.
    pub surface: &'static str,
    /// Stem of the CRUX contract that binds it; `None` until TR-11 defines one.
    pub contract: Option<&'static str>,
    /// What §2.4 says to do with the contract.
    pub disposition: Disposition,
    /// Competitor arms the metric is measured against.
    pub arms: &'static [&'static str],
    /// The metric, measured with the same tool for every arm.
    pub metric: &'static str,
}

/// TRACE-001 v1.2 §2.4, row for row.
pub const SURFACES: [Surface; 10] = [
    Surface {
        surface: "apr trace --layers, apr run --trace-level",
        contract: Some("crux-F-02-v1"),
        disposition: Disposition::Extend,
        arms: &["pytorch-profiler"],
        metric: "per-layer set equality; sum of layers <= wall",
    },
    Surface {
        surface: "apr profile --roofline",
        contract: Some("crux-F-05-v1"),
        disposition: Disposition::Extend,
        arms: &["pytorch-profiler", "ncu-roofline"],
        metric: "arithmetic intensity per op vs ncu on the same kernel",
    },
    Surface {
        surface: "chrome trace export",
        contract: Some("crux-F-07-v1"),
        disposition: Disposition::Reuse,
        arms: &["pytorch-profiler"],
        metric: "schema validity; no negative or NaN durations",
    },
    Surface {
        surface: "/metrics",
        contract: Some("crux-K-07-v1"),
        disposition: Disposition::Reuse,
        arms: &["vllm"],
        metric: "metric-name set; histogram counts = requests",
    },
    Surface {
        surface: "OTLP spans from apr serve",
        contract: Some("crux-K-08-v1"),
        disposition: Disposition::Unblock,
        arms: &["vllm", "tgi"],
        metric: "apr otlp-lint --require-apr-span --require-genai-attrs on a captured export",
    },
    Surface {
        surface: "renacer syscall tracing",
        contract: None,
        disposition: Disposition::New,
        arms: &["strace", "perf-trace", "bpftrace"],
        metric: "E-S1 syscall parity, E-S2 overhead",
    },
    Surface {
        surface: "renacer validate golden traces",
        contract: None,
        disposition: Disposition::New,
        arms: &["strace-c"],
        metric: "planted-regression detection rate; false-RED rate on unchanged builds",
    },
    Surface {
        surface: "cgp profile (cuda, simd, wgpu)",
        contract: None,
        disposition: Disposition::New,
        arms: &["nsys", "ncu"],
        metric: "E-S3 kernel profile agreement",
    },
    Surface {
        surface: "BrickTracer per-op attribution",
        contract: None,
        disposition: Disposition::New,
        arms: &["pytorch-profiler", "perf-record"],
        metric: "sum of components <= wall; NotInstrumented when unmeasured",
    },
    Surface {
        surface: "external serve witness (TR-07)",
        contract: None,
        disposition: Disposition::Instrument,
        arms: &[],
        metric: "governed by APR-OBS OBS-04",
    },
];

/// A competitor arm. No arm is pinned yet, so `pin` is `null` with the reason.
#[derive(Debug, Clone, Serialize)]
pub struct Arm {
    /// The competitor tool.
    pub name: &'static str,
    /// The pinned build, `None` until TR-11/TR-13 pin it.
    pub pin: Option<String>,
    /// Why `pin` is `None`.
    pub unpinned_reason: Option<&'static str>,
}

/// One falsifier's discharge record, as the contract states it.
#[derive(Debug, Clone, Serialize)]
pub struct Discharge {
    /// Falsifier id.
    pub id: String,
    /// `discharge_status` as recorded; `None` when absent.
    pub discharge_status: Option<String>,
}

/// What the receipt knows about a surface.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Binding {
    /// A contract exists and the loader parsed it.
    Bound {
        /// `metadata.status` (raw YAML; see [`LOADER_GAP`]).
        status: Option<String>,
        /// `metadata.category` (raw YAML).
        category: Option<String>,
        /// `metadata.competitor` (typed loader).
        competitor: Option<String>,
        /// One entry per falsifier (ids from the typed loader).
        falsifiers: Vec<Discharge>,
    },
    /// No verdict: the row cannot be bound, and the reason says why.
    NotRun {
        /// Why the row is not bound.
        reason: String,
    },
}

/// One row of the emitted matrix.
#[derive(Debug, Clone, Serialize)]
pub struct Row {
    /// The tracing surface.
    pub surface: &'static str,
    /// The contract stem, when one exists.
    pub contract: Option<&'static str>,
    /// What §2.4 says to do with it.
    pub disposition: Disposition,
    /// Competitor arms.
    pub arms: Vec<Arm>,
    /// The metric.
    pub metric: &'static str,
    /// The binding verdict.
    pub binding: Binding,
}

/// `crux-trace-bind-receipt.json`.
#[derive(Debug, Clone, Serialize)]
pub struct Receipt {
    /// Always [`RECEIPT_SCHEMA`].
    pub schema: &'static str,
    /// The spec section the matrix comes from.
    pub spec: &'static str,
    /// CRUX contracts the loader parsed.
    pub crux_parsed: usize,
    /// CRUX files the loader refused, as `(path, error)`.
    pub crux_parse_errors: Vec<(String, String)>,
    /// Fields read from raw YAML because the loader drops them.
    pub loader_gap: [&'static str; 3],
    /// The §2.4 matrix.
    pub rows: Vec<Row>,
    /// Stems that are `active` with no discharge record (G14).
    pub g14_active_without_discharge: Vec<String>,
}

fn crux_paths(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let is_crux = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("crux-"));
        if is_crux && is_contract_yaml(&path) {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn raw_metadata(raw: &Value, key: &str) -> Option<String> {
    raw.get("metadata")?.get(key)?.as_str().map(str::to_owned)
}

fn raw_discharge(raw: &Value, id: &str) -> Option<String> {
    raw.get("falsification_tests")?
        .as_sequence()?
        .iter()
        .find(|t| t.get("id").and_then(Value::as_str) == Some(id))?
        .get("discharge_status")?
        .as_str()
        .map(str::to_owned)
}

fn bind(dir: &Path, stem: &str) -> Binding {
    let path = dir.join(format!("{stem}.yaml"));
    let contract = match parse_contract(&path) {
        Ok(c) => c,
        Err(e) => {
            return Binding::NotRun {
                reason: format!("loader refused {}: {e}", path.display()),
            }
        }
    };
    let raw: Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_yaml::from_str(&s).ok())
        .unwrap_or(Value::Null);
    let falsifiers = contract
        .falsification_tests
        .iter()
        .map(|t| Discharge {
            id: t.id.clone(),
            discharge_status: raw_discharge(&raw, &t.id),
        })
        .collect();
    Binding::Bound {
        status: raw_metadata(&raw, "status"),
        category: raw_metadata(&raw, "category"),
        competitor: contract.metadata.competitor.clone(),
        falsifiers,
    }
}

fn row(dir: &Path, s: &Surface) -> Row {
    let binding = match (s.disposition, s.contract) {
        (Disposition::Instrument, _) => Binding::NotRun {
            reason: "instrument, not a product claim".to_owned(),
        },
        (_, None) => Binding::NotRun {
            reason: "contract not yet defined (TR-11)".to_owned(),
        },
        (_, Some(stem)) => bind(dir, stem),
    };
    let arms = s
        .arms
        .iter()
        .map(|name| Arm {
            name,
            pin: None,
            unpinned_reason: Some("no pinned competitor build yet (TR-11/TR-13)"),
        })
        .collect();
    Row {
        surface: s.surface,
        contract: s.contract,
        disposition: s.disposition,
        arms,
        metric: s.metric,
        binding,
    }
}

/// Stems whose row is `active` with no `discharge_status` on any falsifier.
#[must_use]
pub fn g14_violations(rows: &[Row]) -> Vec<String> {
    rows.iter()
        .filter_map(|r| match &r.binding {
            Binding::Bound {
                status, falsifiers, ..
            } if status.as_deref() == Some("active")
                && falsifiers.iter().all(|f| f.discharge_status.is_none()) =>
            {
                r.contract.map(str::to_owned)
            }
            _ => None,
        })
        .collect()
}

/// Build the receipt from a contracts directory.
///
/// # Errors
/// Returns the I/O error when `dir` cannot be listed.
pub fn build(dir: &Path) -> std::io::Result<Receipt> {
    let mut crux_parsed = 0;
    let mut crux_parse_errors = Vec::new();
    for path in crux_paths(dir)? {
        match parse_contract(&path) {
            Ok(_) => crux_parsed += 1,
            Err(e) => crux_parse_errors.push((path.display().to_string(), e.to_string())),
        }
    }
    let rows: Vec<Row> = SURFACES.iter().map(|s| row(dir, s)).collect();
    let g14_active_without_discharge = g14_violations(&rows);
    Ok(Receipt {
        schema: RECEIPT_SCHEMA,
        spec: "TRACE-001 v1.2 §2.4",
        crux_parsed,
        crux_parse_errors,
        loader_gap: LOADER_GAP,
        rows,
        g14_active_without_discharge,
    })
}

#[cfg(test)]
#[path = "crux_trace_bind_tests.rs"]
mod tests;
