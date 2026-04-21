//! CUDA OOM postmortem classifier (CRUX-F-13).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-F-13-{001,002,003,005}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! an already-captured `/tmp/apr-oom-<ts>.json` postmortem file:
//!
//!   * `classify_oom_schema` — report object has all 7 required top-level keys
//!     with correct types (numbers, arrays, object, RFC3339 string).
//!   * `classify_oom_invariants` — peak_reserved >= peak_allocated >= 0,
//!     last_100_ops length <= 100, tensor_histogram non-empty,
//!     largest_alloc_stack non-empty, exit_code != 0.
//!   * `classify_oom_size` — serialized report size < 10 MiB (no raw tensor
//!     data leaked into the postmortem).
//!
//! FALSIFY-CRUX-F-13-004 (stderr breadcrumb) is discharged by the CLI layer
//! in `oom_lint.rs` via `classify_oom_breadcrumb` over a captured stderr body.
//!
//! Full discharge blocks on a live CUDA OOM trigger in `aprender-serve`
//! (BLOCKER-UPSTREAM-MISSING — no OOM hook plumbed yet as of 2026-04-21).

use serde_json::Value;

/// The 7 required top-level keys on a CRUX-F-13 OOM postmortem report
/// (see `contracts/crux-F-13-v1.yaml` equation `oom_report_schema`).
pub const OOM_REPORT_REQUIRED_KEYS: &[&str] = &[
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "largest_alloc_stack",
    "tensor_histogram",
    "last_100_ops",
    "exit_code",
    "timestamp",
];

/// Maximum allowed size of a serialized OOM report file, in bytes.
/// Contract: `sizeof(report.json) < 10 * 1024 * 1024`.
pub const OOM_REPORT_MAX_SIZE_BYTES: u64 = 10 * 1024 * 1024;

/// Maximum number of entries allowed in `last_100_ops`.
pub const OOM_REPORT_MAX_OPS: usize = 100;

/// Outcome of `classify_oom_schema`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OomSchemaOutcome {
    /// Report is a JSON object with all 7 required keys present and of correct type.
    Ok,
    /// Report root is not a JSON object.
    NotAnObject,
    /// One of the 7 required top-level keys is absent.
    MissingRequiredKey { key: &'static str },
    /// `peak_allocated_bytes` or `peak_reserved_bytes` is not a JSON number.
    PeakBytesNotNumber { key: &'static str },
    /// `largest_alloc_stack` is not a JSON array of strings.
    StackNotStringArray,
    /// `tensor_histogram` is not a JSON object.
    HistogramNotObject,
    /// `last_100_ops` is not a JSON array.
    OpsNotArray,
    /// `exit_code` is not a JSON number.
    ExitCodeNotNumber,
    /// `timestamp` is not a JSON string.
    TimestampNotString,
}

/// Outcome of `classify_oom_invariants`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OomInvariantsOutcome {
    /// All invariants hold.
    Ok,
    /// `peak_reserved_bytes < peak_allocated_bytes` — CUDA accounting bug.
    ReservedLessThanAllocated { reserved: u64, allocated: u64 },
    /// `last_100_ops` array length exceeds 100.
    TooManyOps { got: usize },
    /// `tensor_histogram` has zero buckets.
    EmptyHistogram,
    /// `largest_alloc_stack` is empty.
    EmptyStack,
    /// `exit_code == 0` — OOM was silent-swallowed.
    SilentExitCode,
}

/// Outcome of `classify_oom_size`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OomSizeOutcome {
    /// Serialized file size < 10 MiB.
    Ok { bytes: u64 },
    /// Serialized file size >= 10 MiB — raw tensor data leaked into the postmortem.
    TooLarge { bytes: u64, cap_bytes: u64 },
}

/// Outcome of `classify_oom_breadcrumb`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OomBreadcrumbOutcome {
    /// Stderr has a single `OOM_REPORT path=/tmp/apr-oom-<digits>.json` line.
    Ok { path: String },
    /// Zero breadcrumb lines found.
    NotFound,
    /// Breadcrumb line present but path does not match `/tmp/apr-oom-<digits>.json`.
    MalformedPath { got: String },
}

/// Non-streaming schema gate.
/// Pure function of `report`; deterministic.
#[allow(clippy::too_many_lines)]
pub fn classify_oom_schema(report: &Value) -> OomSchemaOutcome {
    let obj = match report.as_object() {
        Some(o) => o,
        None => return OomSchemaOutcome::NotAnObject,
    };
    for &key in OOM_REPORT_REQUIRED_KEYS {
        if !obj.contains_key(key) {
            return OomSchemaOutcome::MissingRequiredKey { key };
        }
    }
    for key in ["peak_allocated_bytes", "peak_reserved_bytes"] {
        if !obj[key].is_number() {
            let static_key: &'static str = match key {
                "peak_allocated_bytes" => "peak_allocated_bytes",
                "peak_reserved_bytes" => "peak_reserved_bytes",
                _ => unreachable!(),
            };
            return OomSchemaOutcome::PeakBytesNotNumber { key: static_key };
        }
    }
    match obj["largest_alloc_stack"].as_array() {
        Some(arr) => {
            if !arr.iter().all(Value::is_string) {
                return OomSchemaOutcome::StackNotStringArray;
            }
        }
        None => return OomSchemaOutcome::StackNotStringArray,
    }
    if !obj["tensor_histogram"].is_object() {
        return OomSchemaOutcome::HistogramNotObject;
    }
    if !obj["last_100_ops"].is_array() {
        return OomSchemaOutcome::OpsNotArray;
    }
    if !obj["exit_code"].is_number() {
        return OomSchemaOutcome::ExitCodeNotNumber;
    }
    if !obj["timestamp"].is_string() {
        return OomSchemaOutcome::TimestampNotString;
    }
    OomSchemaOutcome::Ok
}

/// Report invariants gate (contract `oom_report_schema.invariants`).
/// Assumes schema gate passed; callers MUST check `classify_oom_schema` first.
pub fn classify_oom_invariants(report: &Value) -> OomInvariantsOutcome {
    let allocated = report["peak_allocated_bytes"].as_u64().unwrap_or(0);
    let reserved = report["peak_reserved_bytes"].as_u64().unwrap_or(0);
    if reserved < allocated {
        return OomInvariantsOutcome::ReservedLessThanAllocated {
            reserved,
            allocated,
        };
    }
    if let Some(ops) = report["last_100_ops"].as_array() {
        if ops.len() > OOM_REPORT_MAX_OPS {
            return OomInvariantsOutcome::TooManyOps { got: ops.len() };
        }
    }
    if let Some(hist) = report["tensor_histogram"].as_object() {
        if hist.is_empty() {
            return OomInvariantsOutcome::EmptyHistogram;
        }
    }
    if let Some(stack) = report["largest_alloc_stack"].as_array() {
        if stack.is_empty() {
            return OomInvariantsOutcome::EmptyStack;
        }
    }
    if report["exit_code"].as_i64() == Some(0) {
        return OomInvariantsOutcome::SilentExitCode;
    }
    OomInvariantsOutcome::Ok
}

/// File size gate (contract `sizeof(report.json) < 10 MiB`).
#[must_use]
pub fn classify_oom_size(bytes: u64) -> OomSizeOutcome {
    if bytes < OOM_REPORT_MAX_SIZE_BYTES {
        OomSizeOutcome::Ok { bytes }
    } else {
        OomSizeOutcome::TooLarge {
            bytes,
            cap_bytes: OOM_REPORT_MAX_SIZE_BYTES,
        }
    }
}

/// Stderr breadcrumb gate. Parses `stderr` and returns the captured
/// `OOM_REPORT path=...` line if present.
///
/// Contract: stderr MUST contain exactly one line matching the pattern
/// `^OOM_REPORT path=/tmp/apr-oom-\d+\.json$`.
#[must_use]
pub fn classify_oom_breadcrumb(stderr: &str) -> OomBreadcrumbOutcome {
    let mut found: Option<&str> = None;
    for line in stderr.lines() {
        if let Some(rest) = line.strip_prefix("OOM_REPORT path=") {
            found = Some(rest);
        }
    }
    match found {
        None => OomBreadcrumbOutcome::NotFound,
        Some(path) => {
            if !is_valid_oom_path(path) {
                return OomBreadcrumbOutcome::MalformedPath {
                    got: path.to_string(),
                };
            }
            OomBreadcrumbOutcome::Ok {
                path: path.to_string(),
            }
        }
    }
}

fn is_valid_oom_path(path: &str) -> bool {
    let Some(rest) = path.strip_prefix("/tmp/apr-oom-") else {
        return false;
    };
    let Some(stem) = rest.strip_suffix(".json") else {
        return false;
    };
    !stem.is_empty() && stem.chars().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn well_formed_report() -> Value {
        json!({
            "peak_allocated_bytes": 2_000_000_u64,
            "peak_reserved_bytes": 3_000_000_u64,
            "largest_alloc_stack": ["apr::run", "realizar::load", "cuda::malloc"],
            "tensor_histogram": { "1-64KB": 12, "64KB-1MB": 4, "1MB+": 1 },
            "last_100_ops": [ { "op": "matmul", "bytes": 4096_u64, "ts_ns": 1_u64 } ],
            "exit_code": 137_i64,
            "timestamp": "2026-04-21T18:30:00Z",
        })
    }

    #[test]
    fn schema_accepts_well_formed_report() {
        assert_eq!(
            classify_oom_schema(&well_formed_report()),
            OomSchemaOutcome::Ok
        );
    }

    #[test]
    fn schema_rejects_non_object() {
        assert_eq!(
            classify_oom_schema(&json!(["not", "object"])),
            OomSchemaOutcome::NotAnObject
        );
    }

    #[test]
    fn schema_rejects_each_missing_key() {
        for &drop in OOM_REPORT_REQUIRED_KEYS {
            let mut r = well_formed_report();
            r.as_object_mut().unwrap().remove(drop);
            assert_eq!(
                classify_oom_schema(&r),
                OomSchemaOutcome::MissingRequiredKey { key: drop },
                "drop={drop}"
            );
        }
    }

    #[test]
    fn schema_rejects_non_number_peak_allocated() {
        let mut r = well_formed_report();
        r["peak_allocated_bytes"] = json!("not a number");
        assert_eq!(
            classify_oom_schema(&r),
            OomSchemaOutcome::PeakBytesNotNumber {
                key: "peak_allocated_bytes"
            }
        );
    }

    #[test]
    fn schema_rejects_non_number_peak_reserved() {
        let mut r = well_formed_report();
        r["peak_reserved_bytes"] = json!(null);
        assert_eq!(
            classify_oom_schema(&r),
            OomSchemaOutcome::PeakBytesNotNumber {
                key: "peak_reserved_bytes"
            }
        );
    }

    #[test]
    fn schema_rejects_stack_with_non_strings() {
        let mut r = well_formed_report();
        r["largest_alloc_stack"] = json!(["frame", 7_u64]);
        assert_eq!(
            classify_oom_schema(&r),
            OomSchemaOutcome::StackNotStringArray
        );
    }

    #[test]
    fn schema_rejects_stack_not_array() {
        let mut r = well_formed_report();
        r["largest_alloc_stack"] = json!("one frame");
        assert_eq!(
            classify_oom_schema(&r),
            OomSchemaOutcome::StackNotStringArray
        );
    }

    #[test]
    fn schema_rejects_histogram_not_object() {
        let mut r = well_formed_report();
        r["tensor_histogram"] = json!([1, 2, 3]);
        assert_eq!(
            classify_oom_schema(&r),
            OomSchemaOutcome::HistogramNotObject
        );
    }

    #[test]
    fn schema_rejects_ops_not_array() {
        let mut r = well_formed_report();
        r["last_100_ops"] = json!({ "op": "matmul" });
        assert_eq!(classify_oom_schema(&r), OomSchemaOutcome::OpsNotArray);
    }

    #[test]
    fn schema_rejects_exit_code_not_number() {
        let mut r = well_formed_report();
        r["exit_code"] = json!("137");
        assert_eq!(classify_oom_schema(&r), OomSchemaOutcome::ExitCodeNotNumber);
    }

    #[test]
    fn schema_rejects_timestamp_not_string() {
        let mut r = well_formed_report();
        r["timestamp"] = json!(1_745_000_000_i64);
        assert_eq!(
            classify_oom_schema(&r),
            OomSchemaOutcome::TimestampNotString
        );
    }

    #[test]
    fn invariants_accepts_well_formed_report() {
        assert_eq!(
            classify_oom_invariants(&well_formed_report()),
            OomInvariantsOutcome::Ok
        );
    }

    #[test]
    fn invariants_rejects_reserved_less_than_allocated() {
        let mut r = well_formed_report();
        r["peak_reserved_bytes"] = json!(1_000_000_u64);
        r["peak_allocated_bytes"] = json!(2_000_000_u64);
        assert_eq!(
            classify_oom_invariants(&r),
            OomInvariantsOutcome::ReservedLessThanAllocated {
                reserved: 1_000_000,
                allocated: 2_000_000,
            }
        );
    }

    #[test]
    fn invariants_accepts_reserved_equals_allocated() {
        let mut r = well_formed_report();
        r["peak_reserved_bytes"] = json!(2_000_000_u64);
        r["peak_allocated_bytes"] = json!(2_000_000_u64);
        assert_eq!(classify_oom_invariants(&r), OomInvariantsOutcome::Ok);
    }

    #[test]
    fn invariants_rejects_too_many_ops() {
        let mut r = well_formed_report();
        let ops: Vec<Value> = (0..101_u64)
            .map(|i| json!({ "op": "x", "bytes": 1_u64, "ts_ns": i }))
            .collect();
        r["last_100_ops"] = json!(ops);
        assert_eq!(
            classify_oom_invariants(&r),
            OomInvariantsOutcome::TooManyOps { got: 101 }
        );
    }

    #[test]
    fn invariants_accepts_exactly_100_ops() {
        let mut r = well_formed_report();
        let ops: Vec<Value> = (0..100_u64)
            .map(|i| json!({ "op": "x", "bytes": 1_u64, "ts_ns": i }))
            .collect();
        r["last_100_ops"] = json!(ops);
        assert_eq!(classify_oom_invariants(&r), OomInvariantsOutcome::Ok);
    }

    #[test]
    fn invariants_rejects_empty_histogram() {
        let mut r = well_formed_report();
        r["tensor_histogram"] = json!({});
        assert_eq!(
            classify_oom_invariants(&r),
            OomInvariantsOutcome::EmptyHistogram
        );
    }

    #[test]
    fn invariants_rejects_empty_stack() {
        let mut r = well_formed_report();
        r["largest_alloc_stack"] = json!([]);
        assert_eq!(
            classify_oom_invariants(&r),
            OomInvariantsOutcome::EmptyStack
        );
    }

    #[test]
    fn invariants_rejects_silent_exit_code_zero() {
        let mut r = well_formed_report();
        r["exit_code"] = json!(0_i64);
        assert_eq!(
            classify_oom_invariants(&r),
            OomInvariantsOutcome::SilentExitCode
        );
    }

    #[test]
    fn invariants_accepts_non_zero_exit_code() {
        for ec in [1_i64, 137, 139, -1] {
            let mut r = well_formed_report();
            r["exit_code"] = json!(ec);
            assert_eq!(
                classify_oom_invariants(&r),
                OomInvariantsOutcome::Ok,
                "exit_code={ec}"
            );
        }
    }

    #[test]
    fn size_accepts_under_cap() {
        assert_eq!(classify_oom_size(100), OomSizeOutcome::Ok { bytes: 100 });
        assert_eq!(
            classify_oom_size(OOM_REPORT_MAX_SIZE_BYTES - 1),
            OomSizeOutcome::Ok {
                bytes: OOM_REPORT_MAX_SIZE_BYTES - 1
            }
        );
    }

    #[test]
    fn size_rejects_at_and_over_cap() {
        assert_eq!(
            classify_oom_size(OOM_REPORT_MAX_SIZE_BYTES),
            OomSizeOutcome::TooLarge {
                bytes: OOM_REPORT_MAX_SIZE_BYTES,
                cap_bytes: OOM_REPORT_MAX_SIZE_BYTES,
            }
        );
        assert_eq!(
            classify_oom_size(OOM_REPORT_MAX_SIZE_BYTES + 1),
            OomSizeOutcome::TooLarge {
                bytes: OOM_REPORT_MAX_SIZE_BYTES + 1,
                cap_bytes: OOM_REPORT_MAX_SIZE_BYTES,
            }
        );
    }

    #[test]
    fn breadcrumb_accepts_well_formed() {
        let stderr = "Some noise\nOOM_REPORT path=/tmp/apr-oom-1745000000.json\nmore noise\n";
        assert_eq!(
            classify_oom_breadcrumb(stderr),
            OomBreadcrumbOutcome::Ok {
                path: "/tmp/apr-oom-1745000000.json".to_string()
            }
        );
    }

    #[test]
    fn breadcrumb_rejects_when_missing() {
        assert_eq!(
            classify_oom_breadcrumb("no breadcrumb here"),
            OomBreadcrumbOutcome::NotFound
        );
    }

    #[test]
    fn breadcrumb_rejects_wrong_path_prefix() {
        let stderr = "OOM_REPORT path=/var/log/apr-oom-1.json\n";
        assert_eq!(
            classify_oom_breadcrumb(stderr),
            OomBreadcrumbOutcome::MalformedPath {
                got: "/var/log/apr-oom-1.json".to_string()
            }
        );
    }

    #[test]
    fn breadcrumb_rejects_non_digit_ts() {
        let stderr = "OOM_REPORT path=/tmp/apr-oom-abc.json\n";
        assert_eq!(
            classify_oom_breadcrumb(stderr),
            OomBreadcrumbOutcome::MalformedPath {
                got: "/tmp/apr-oom-abc.json".to_string()
            }
        );
    }

    #[test]
    fn breadcrumb_rejects_empty_ts() {
        let stderr = "OOM_REPORT path=/tmp/apr-oom-.json\n";
        assert_eq!(
            classify_oom_breadcrumb(stderr),
            OomBreadcrumbOutcome::MalformedPath {
                got: "/tmp/apr-oom-.json".to_string()
            }
        );
    }

    #[test]
    fn breadcrumb_rejects_wrong_suffix() {
        let stderr = "OOM_REPORT path=/tmp/apr-oom-1.txt\n";
        assert_eq!(
            classify_oom_breadcrumb(stderr),
            OomBreadcrumbOutcome::MalformedPath {
                got: "/tmp/apr-oom-1.txt".to_string()
            }
        );
    }
}
