//! OBS-15 kernel path (APR-OBS-001 v1.3 §2.10, §3; contract `apr-kernel-path-v1`).
//!
//! `kernel_path = {source: "trace" | "kreg", entries: [{op, kernel_id, qtype, layout, arch,
//! shape_class, precision}]}`. It names the kernels a perf row ran, so a night-over-night
//! ratio change can be attributed to a kernel change instead of being read as apr getting
//! slower. Three checks:
//!
//! - once the recorder's apr binary carries the OBS-09 trace, a GPU row with a null
//!   `kernel_path` is refused; before that cut it is admitted only with a reason (OBS-05);
//! - a non-null `kernel_path` has a known source and every entry carries all seven fields;
//! - a KREG entry is admitted only with a passing parity receipt from a host of its arch,
//!   for its backend and kernel.

use crate::obs_backfill::known;
use crate::obs_ledger::{night_of, GPU_BACKENDS};
use chrono::NaiveDate;
use serde_json::Value;
use std::collections::BTreeMap;

/// Where a `kernel_path` came from: the OBS-09 per-layer trace, then KREG-001 (#4539).
pub const SOURCES: [&str; 2] = ["trace", "kreg"];

/// The fields every `kernel_path` entry carries (§2.10).
pub const ENTRY_FIELDS: [&str; 7] = [
    "op",
    "kernel_id",
    "qtype",
    "layout",
    "arch",
    "shape_class",
    "precision",
];

/// Why a row's `kernel_path` is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelPathError {
    /// A GPU row on or after the trace cut with `kernel_path: null`.
    NullAfterTrace,
    /// `kernel_path` is neither null nor an object.
    NotAnObject,
    /// `source` is not one of [`SOURCES`].
    BadSource(String),
    /// `entries` is absent or empty: a path that names no kernel attributes nothing.
    NoEntries,
    /// An entry lacks fields from [`ENTRY_FIELDS`].
    Entry {
        /// Position of the entry in `entries`.
        index: usize,
        /// The fields it lacks, in [`ENTRY_FIELDS`] order.
        missing: Vec<&'static str>,
    },
}

/// Check a perf row's `kernel_path`. `trace_cut` is the first night whose recorder runs an
/// apr binary with the OBS-09 trace; `None` until OBS-09 merges, and then only the shape of
/// a non-null path is checked.
pub fn check_kernel_path(row: &Value, trace_cut: Option<NaiveDate>) -> Result<(), KernelPathError> {
    let path = row.get("kernel_path").unwrap_or(&Value::Null);
    if path.is_null() {
        let gpu = GPU_BACKENDS.contains(&row["backend"].as_str().unwrap_or_default());
        let traced = match (trace_cut, night_of(row)) {
            (Some(cut), Some(night)) => night >= cut,
            // A row whose night cannot be read is not given the pre-trace allowance.
            (Some(_), None) => true,
            (None, _) => false,
        };
        return if gpu && traced {
            Err(KernelPathError::NullAfterTrace)
        } else {
            Ok(())
        };
    }
    let Value::Object(p) = path else {
        return Err(KernelPathError::NotAnObject);
    };
    let source = p.get("source").and_then(Value::as_str).unwrap_or_default();
    if !SOURCES.contains(&source) {
        return Err(KernelPathError::BadSource(source.to_string()));
    }
    let entries = match p.get("entries") {
        Some(Value::Array(e)) if !e.is_empty() => e,
        _ => return Err(KernelPathError::NoEntries),
    };
    for (index, e) in entries.iter().enumerate() {
        let missing: Vec<&'static str> = ENTRY_FIELDS
            .iter()
            .copied()
            .filter(|f| !known(e.get(*f)))
            .collect();
        if !missing.is_empty() {
            return Err(KernelPathError::Entry { index, missing });
        }
    }
    Ok(())
}

/// One op whose kernel changed between two nights.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelChange {
    /// `(op, shape_class)`: the slot the kernel fills.
    pub slot: (String, String),
    /// The entry the earlier night ran there, if any.
    pub before: Option<Value>,
    /// The entry the later night ran there, if any.
    pub after: Option<Value>,
}

fn slots(path: &Value) -> Option<BTreeMap<(String, String), &Value>> {
    let entries = path.get("entries")?.as_array()?;
    let mut m = BTreeMap::new();
    for e in entries {
        let key = |f: &str| {
            e.get(f)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        m.insert((key("op"), key("shape_class")), e);
    }
    Some(m)
}

/// Attribute a night-over-night change: the slots whose kernel entry differs. `None` when
/// either row has no `kernel_path`, so an unattributable change is never reported as "no
/// kernel changed".
pub fn kernel_diff(prev: &Value, cur: &Value) -> Option<Vec<KernelChange>> {
    let (a, b) = (
        slots(prev.get("kernel_path")?)?,
        slots(cur.get("kernel_path")?)?,
    );
    let mut keys: Vec<&(String, String)> = a.keys().chain(b.keys()).collect();
    keys.sort();
    keys.dedup();
    Some(
        keys.into_iter()
            .filter(|k| a.get(*k) != b.get(*k))
            .map(|k| KernelChange {
                slot: k.clone(),
                before: a.get(k).map(|v| (*v).clone()),
                after: b.get(k).map(|v| (*v).clone()),
            })
            .collect(),
    )
}

/// The KREG-001 key dimensions §2.10 requires: arch-generic, no x86- or CUDA-only fields.
pub const KREG_KEY: [&str; 3] = ["arch", "isa_features", "backend"];

/// Admit a KREG entry only with a passing parity receipt from a host of its arch, for its
/// backend and `kernel_id`. Returns the missing key dimensions or `"no_host_receipt"`.
pub fn admit_kreg_entry(entry: &Value, receipts: &[Value]) -> Result<(), Vec<&'static str>> {
    let missing: Vec<&'static str> = KREG_KEY
        .iter()
        .chain(["kernel_id"].iter())
        .copied()
        .filter(|f| !known(entry.get(*f)))
        .collect();
    if !missing.is_empty() {
        return Err(missing);
    }
    let same = |r: &Value, f: &str| r.get(f).is_some() && r.get(f) == entry.get(f);
    let receipted = receipts.iter().any(|r| {
        same(r, "arch")
            && same(r, "backend")
            && same(r, "kernel_id")
            && known(r.get("host"))
            && r.get("parity").and_then(Value::as_str) == Some("pass")
    });
    if receipted {
        Ok(())
    } else {
        Err(vec!["no_host_receipt"])
    }
}

#[cfg(test)]
#[path = "obs_kernel_path_tests.rs"]
mod tests;
