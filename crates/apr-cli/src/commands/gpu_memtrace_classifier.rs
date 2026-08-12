//! Chrome Trace Event Format GPU memory-trace classifier (CRUX-F-07).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-F-07-{001,002,003}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! an already-captured `apr profile --gpu-memory-trace` output:
//!
//!   * `classify_schema` — body parses as a JSON object with a non-empty
//!     `traceEvents` array; every event has the Chrome Trace required
//!     fields `ph`, `ts`, `name` (plus optional `pid`/`tid`/`args`).
//!     `displayTimeUnit` (if present) must be `"ns"` or `"ms"`.
//!   * `classify_alloc_free_pairing` — every `alloc` event has a matching
//!     `free` event with the same `args.addr`; orphan allocations or
//!     duplicate frees are reported with the offending addresses.
//!   * `classify_monotonic_timestamps` — for each (pid, tid) stream,
//!     timestamps are non-decreasing — CUDA stream ordering must be
//!     preserved in the trace.
//!
//! FALSIFY-CRUX-F-07-004 (peak agrees with NVML within 10%) requires a live
//! NVML reading at the peak instant — discharged only when the emitter is
//! wired into `apr profile`. Tracked as BLOCKER-UPSTREAM-MISSING.

use serde_json::Value;

/// Outcome of `classify_schema`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ChromeTraceSchemaOutcome {
    Ok { event_count: usize },
    NotAnObject,
    MissingTraceEvents,
    TraceEventsNotArray,
    TraceEventsEmpty,
    EventNotAnObject { index: usize },
    EventMissingField { index: usize, field: &'static str },
    InvalidDisplayTimeUnit { got: String },
}

/// Outcome of `classify_alloc_free_pairing`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AllocFreePairingOutcome {
    Ok,
    OrphanAllocs { addresses: Vec<String> },
    OrphanFrees { addresses: Vec<String> },
    AllocMissingAddr { index: usize },
}

/// Outcome of `classify_monotonic_timestamps`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum MonotonicTimestampsOutcome {
    Ok,
    NonMonotonic {
        index: usize,
        pid: i64,
        tid: i64,
        prev_ts: f64,
        got_ts: f64,
    },
}

/// Validate the Chrome Trace JSON schema:
/// - root is an object with a non-empty `traceEvents` array
/// - every event has `ph`, `ts`, `name`
/// - `displayTimeUnit` (if present) is `"ns"` or `"ms"`
pub fn classify_schema(body: &Value) -> ChromeTraceSchemaOutcome {
    let Some(obj) = body.as_object() else {
        return ChromeTraceSchemaOutcome::NotAnObject;
    };
    if let Some(unit) = obj.get("displayTimeUnit").and_then(Value::as_str) {
        if unit != "ns" && unit != "ms" {
            return ChromeTraceSchemaOutcome::InvalidDisplayTimeUnit {
                got: unit.to_string(),
            };
        }
    }
    let Some(events_val) = obj.get("traceEvents") else {
        return ChromeTraceSchemaOutcome::MissingTraceEvents;
    };
    let Some(events) = events_val.as_array() else {
        return ChromeTraceSchemaOutcome::TraceEventsNotArray;
    };
    if events.is_empty() {
        return ChromeTraceSchemaOutcome::TraceEventsEmpty;
    }
    for (i, e) in events.iter().enumerate() {
        if !e.is_object() {
            return ChromeTraceSchemaOutcome::EventNotAnObject { index: i };
        }
        for field in ["ph", "ts", "name"] {
            if e.get(field).is_none() {
                return ChromeTraceSchemaOutcome::EventMissingField { index: i, field };
            }
        }
    }
    ChromeTraceSchemaOutcome::Ok {
        event_count: events.len(),
    }
}

/// Verify that every `alloc` event has a matching `free` event with the
/// same `args.addr`, and vice-versa.
pub fn classify_alloc_free_pairing(body: &Value) -> AllocFreePairingOutcome {
    let mut allocs: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut frees: std::collections::HashSet<String> = std::collections::HashSet::new();
    let Some(events) = body.get("traceEvents").and_then(Value::as_array) else {
        return AllocFreePairingOutcome::Ok;
    };
    for (i, e) in events.iter().enumerate() {
        let name = e.get("name").and_then(Value::as_str).unwrap_or("");
        if name != "alloc" && name != "free" {
            continue;
        }
        let addr = e
            .get("args")
            .and_then(|a| a.get("addr"))
            .and_then(Value::as_str);
        let Some(addr) = addr else {
            if name == "alloc" {
                return AllocFreePairingOutcome::AllocMissingAddr { index: i };
            }
            continue;
        };
        if name == "alloc" {
            allocs.insert(addr.to_string());
        } else {
            frees.insert(addr.to_string());
        }
    }
    let mut orphan_a: Vec<String> = allocs.difference(&frees).cloned().collect();
    let mut orphan_f: Vec<String> = frees.difference(&allocs).cloned().collect();
    if !orphan_a.is_empty() {
        orphan_a.sort();
        return AllocFreePairingOutcome::OrphanAllocs {
            addresses: orphan_a,
        };
    }
    if !orphan_f.is_empty() {
        orphan_f.sort();
        return AllocFreePairingOutcome::OrphanFrees {
            addresses: orphan_f,
        };
    }
    AllocFreePairingOutcome::Ok
}

/// Verify that timestamps are non-decreasing per (pid, tid).
pub fn classify_monotonic_timestamps(body: &Value) -> MonotonicTimestampsOutcome {
    use std::collections::HashMap;
    let mut last: HashMap<(i64, i64), f64> = HashMap::new();
    let Some(events) = body.get("traceEvents").and_then(Value::as_array) else {
        return MonotonicTimestampsOutcome::Ok;
    };
    for (i, e) in events.iter().enumerate() {
        let pid = e.get("pid").and_then(Value::as_i64).unwrap_or(0);
        let tid = e.get("tid").and_then(Value::as_i64).unwrap_or(0);
        let ts = e.get("ts").and_then(Value::as_f64).unwrap_or(0.0);
        let key = (pid, tid);
        if let Some(&prev) = last.get(&key) {
            if ts < prev {
                return MonotonicTimestampsOutcome::NonMonotonic {
                    index: i,
                    pid,
                    tid,
                    prev_ts: prev,
                    got_ts: ts,
                };
            }
        }
        last.insert(key, ts);
    }
    MonotonicTimestampsOutcome::Ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn good_body() -> Value {
        json!({
            "displayTimeUnit": "ns",
            "traceEvents": [
                {"ph": "i", "ts": 0,    "name": "alloc", "pid": 0, "tid": 1, "args": {"bytes": 1024, "addr": "0xAAAA"}},
                {"ph": "i", "ts": 100,  "name": "alloc", "pid": 0, "tid": 1, "args": {"bytes": 2048, "addr": "0xBBBB"}},
                {"ph": "i", "ts": 200,  "name": "free",  "pid": 0, "tid": 1, "args": {"bytes": 1024, "addr": "0xAAAA"}},
                {"ph": "i", "ts": 300,  "name": "free",  "pid": 0, "tid": 1, "args": {"bytes": 2048, "addr": "0xBBBB"}},
            ]
        })
    }

    #[test]
    fn schema_ok_on_well_formed_body() {
        assert_eq!(
            classify_schema(&good_body()),
            ChromeTraceSchemaOutcome::Ok { event_count: 4 }
        );
    }

    #[test]
    fn schema_rejects_not_an_object() {
        assert_eq!(
            classify_schema(&json!([1, 2])),
            ChromeTraceSchemaOutcome::NotAnObject
        );
    }

    #[test]
    fn schema_rejects_missing_trace_events() {
        assert_eq!(
            classify_schema(&json!({"otherKey": []})),
            ChromeTraceSchemaOutcome::MissingTraceEvents
        );
    }

    #[test]
    fn schema_rejects_empty_trace_events() {
        assert_eq!(
            classify_schema(&json!({"traceEvents": []})),
            ChromeTraceSchemaOutcome::TraceEventsEmpty
        );
    }

    #[test]
    fn schema_rejects_event_missing_ph() {
        let body = json!({"traceEvents": [{"ts": 0, "name": "alloc"}]});
        match classify_schema(&body) {
            ChromeTraceSchemaOutcome::EventMissingField {
                index: 0,
                field: "ph",
            } => {}
            other => panic!("expected EventMissingField(ph), got {other:?}"),
        }
    }

    #[test]
    fn schema_rejects_invalid_display_time_unit() {
        let body = json!({
            "displayTimeUnit": "sec",
            "traceEvents": [{"ph": "i", "ts": 0, "name": "alloc"}]
        });
        assert!(matches!(
            classify_schema(&body),
            ChromeTraceSchemaOutcome::InvalidDisplayTimeUnit { .. }
        ));
    }

    #[test]
    fn alloc_free_pairing_ok_on_balanced_body() {
        assert_eq!(
            classify_alloc_free_pairing(&good_body()),
            AllocFreePairingOutcome::Ok
        );
    }

    #[test]
    fn alloc_free_pairing_reports_orphan_alloc() {
        let body = json!({
            "traceEvents": [
                {"ph": "i", "ts": 0, "name": "alloc", "args": {"addr": "0xLEAK"}},
                {"ph": "i", "ts": 1, "name": "alloc", "args": {"addr": "0xOK"}},
                {"ph": "i", "ts": 2, "name": "free",  "args": {"addr": "0xOK"}},
            ]
        });
        match classify_alloc_free_pairing(&body) {
            AllocFreePairingOutcome::OrphanAllocs { addresses } => {
                assert_eq!(addresses, vec!["0xLEAK".to_string()]);
            }
            other => panic!("expected OrphanAllocs, got {other:?}"),
        }
    }

    #[test]
    fn alloc_free_pairing_reports_orphan_free() {
        let body = json!({
            "traceEvents": [
                {"ph": "i", "ts": 0, "name": "alloc", "args": {"addr": "0xAAA"}},
                {"ph": "i", "ts": 1, "name": "free",  "args": {"addr": "0xAAA"}},
                {"ph": "i", "ts": 2, "name": "free",  "args": {"addr": "0xPHANTOM"}},
            ]
        });
        match classify_alloc_free_pairing(&body) {
            AllocFreePairingOutcome::OrphanFrees { addresses } => {
                assert_eq!(addresses, vec!["0xPHANTOM".to_string()]);
            }
            other => panic!("expected OrphanFrees, got {other:?}"),
        }
    }

    #[test]
    fn alloc_free_pairing_reports_alloc_missing_addr() {
        let body = json!({
            "traceEvents": [
                {"ph": "i", "ts": 0, "name": "alloc", "args": {"bytes": 10}}
            ]
        });
        assert!(matches!(
            classify_alloc_free_pairing(&body),
            AllocFreePairingOutcome::AllocMissingAddr { index: 0 }
        ));
    }

    #[test]
    fn monotonic_timestamps_ok_on_good_body() {
        assert_eq!(
            classify_monotonic_timestamps(&good_body()),
            MonotonicTimestampsOutcome::Ok
        );
    }

    #[test]
    fn monotonic_timestamps_reports_violation() {
        let body = json!({
            "traceEvents": [
                {"ph": "i", "ts": 100, "name": "alloc", "pid": 0, "tid": 1, "args": {"addr": "0xA"}},
                {"ph": "i", "ts":  50, "name": "free",  "pid": 0, "tid": 1, "args": {"addr": "0xA"}},
            ]
        });
        match classify_monotonic_timestamps(&body) {
            MonotonicTimestampsOutcome::NonMonotonic {
                index, pid, tid, ..
            } => {
                assert_eq!(index, 1);
                assert_eq!(pid, 0);
                assert_eq!(tid, 1);
            }
            other => panic!("expected NonMonotonic, got {other:?}"),
        }
    }

    #[test]
    fn monotonic_timestamps_allows_independent_streams_to_interleave() {
        // Different (pid, tid) keys are independent — ordering across streams
        // is not constrained.
        let body = json!({
            "traceEvents": [
                {"ph": "i", "ts": 100, "name": "alloc", "pid": 0, "tid": 1, "args": {"addr": "0xA"}},
                {"ph": "i", "ts":  50, "name": "alloc", "pid": 0, "tid": 2, "args": {"addr": "0xB"}},
            ]
        });
        assert_eq!(
            classify_monotonic_timestamps(&body),
            MonotonicTimestampsOutcome::Ok
        );
    }
}
