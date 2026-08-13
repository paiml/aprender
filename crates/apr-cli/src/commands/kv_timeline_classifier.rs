//! KV-cache utilization timeline classifier (CRUX-F-06).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-F-06-{001..004}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! an already-captured KV-timeline JSON emitted by
//! `apr profile --kv-timeline --json`:
//!
//!   * `classify_schema` — top-level keys (`timeline`, `block_size_tokens`,
//!     `total_blocks`, `peak_used_pct`, `preemption_count`) present and of
//!     correct types; each step entry has all 7 required fields.
//!   * `classify_block_conservation` — for every step
//!     `used_blocks + free_blocks == total_blocks`.
//!   * `classify_used_pct_arithmetic` — for every step
//!     `used_pct == used_blocks / total_blocks` within 1e-9.
//!   * `classify_peak_consistency` — `peak_used_pct == max(timeline[*].used_pct)`
//!     within 1e-9 and `preemption_count == sum(timeline[*].preempted_seqs)`.
//!   * `classify_preemption_trigger` — every step with `preempted_seqs > 0`
//!     has `used_pct >= threshold` (vllm default 0.95).
//!
//! Full discharge blocks on a live `apr profile --kv-timeline` emitter —
//! tracked as BLOCKER-UPSTREAM-MISSING.

use serde_json::Value;

/// Default vLLM preemption threshold; preemptions fired below this used_pct
/// indicate a scheduler bug.
pub const F06_DEFAULT_PREEMPT_THRESHOLD: f64 = 0.95;

/// The 5 required top-level keys on a CRUX-F-06 KV-timeline JSON
/// (see `contracts/crux-F-06-v1.yaml` equation `kv_timeline_schema`).
pub const KV_TIMELINE_TOP_KEYS: &[&str] = &[
    "timeline",
    "block_size_tokens",
    "total_blocks",
    "peak_used_pct",
    "preemption_count",
];

/// The 7 required per-step keys on each `timeline[i]` entry.
pub const KV_TIMELINE_STEP_KEYS: &[&str] = &[
    "step",
    "t_ms",
    "used_blocks",
    "free_blocks",
    "used_pct",
    "active_seqs",
    "preempted_seqs",
];

/// Outcome of `classify_schema`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum KvSchemaOutcome {
    Ok,
    NotAnObject,
    MissingTopKey { key: &'static str },
    TimelineNotArray,
    StepNotAnObject { index: usize },
    StepMissingKey { index: usize, key: &'static str },
    BlockSizeNotPositive { got: i64 },
    TotalBlocksNotPositive { got: i64 },
}

/// Outcome of `classify_block_conservation`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum KvBlockConservationOutcome {
    Ok,
    Violation {
        index: usize,
        used_blocks: u64,
        free_blocks: u64,
        total_blocks: u64,
    },
}

/// Outcome of `classify_used_pct_arithmetic`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum KvUsedPctOutcome {
    Ok,
    Mismatch {
        index: usize,
        expected: f64,
        got: f64,
    },
}

/// Outcome of `classify_peak_consistency`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum KvPeakOutcome {
    Ok,
    PeakMismatch { expected: f64, got: f64 },
    PreemptionCountMismatch { expected: u64, got: u64 },
}

/// Outcome of `classify_preemption_trigger`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum KvPreemptionOutcome {
    Ok,
    PreemptionBelowThreshold {
        index: usize,
        used_pct: f64,
        threshold: f64,
        preempted_seqs: u64,
    },
}

/// Validate that the JSON body matches the F-06 timeline schema.
pub fn classify_schema(body: &Value) -> KvSchemaOutcome {
    let Some(obj) = body.as_object() else {
        return KvSchemaOutcome::NotAnObject;
    };
    for k in KV_TIMELINE_TOP_KEYS {
        if !obj.contains_key(*k) {
            return KvSchemaOutcome::MissingTopKey { key: k };
        }
    }
    let Some(timeline) = obj.get("timeline").and_then(Value::as_array) else {
        return KvSchemaOutcome::TimelineNotArray;
    };
    let block_size = obj
        .get("block_size_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    if block_size <= 0 {
        return KvSchemaOutcome::BlockSizeNotPositive { got: block_size };
    }
    let total = obj.get("total_blocks").and_then(Value::as_i64).unwrap_or(0);
    if total <= 0 {
        return KvSchemaOutcome::TotalBlocksNotPositive { got: total };
    }
    for (i, step) in timeline.iter().enumerate() {
        let Some(step_obj) = step.as_object() else {
            return KvSchemaOutcome::StepNotAnObject { index: i };
        };
        for k in KV_TIMELINE_STEP_KEYS {
            if !step_obj.contains_key(*k) {
                return KvSchemaOutcome::StepMissingKey { index: i, key: k };
            }
        }
    }
    KvSchemaOutcome::Ok
}

/// Verify `used_blocks + free_blocks == total_blocks` for every step.
pub fn classify_block_conservation(body: &Value) -> KvBlockConservationOutcome {
    let total = body
        .get("total_blocks")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let Some(timeline) = body.get("timeline").and_then(Value::as_array) else {
        return KvBlockConservationOutcome::Ok;
    };
    for (i, step) in timeline.iter().enumerate() {
        let used = step.get("used_blocks").and_then(Value::as_u64).unwrap_or(0);
        let free = step.get("free_blocks").and_then(Value::as_u64).unwrap_or(0);
        if used.saturating_add(free) != total {
            return KvBlockConservationOutcome::Violation {
                index: i,
                used_blocks: used,
                free_blocks: free,
                total_blocks: total,
            };
        }
    }
    KvBlockConservationOutcome::Ok
}

/// Verify `used_pct == used_blocks / total_blocks` for every step (within 1e-9).
pub fn classify_used_pct_arithmetic(body: &Value) -> KvUsedPctOutcome {
    let total = body
        .get("total_blocks")
        .and_then(Value::as_u64)
        .unwrap_or(0) as f64;
    let Some(timeline) = body.get("timeline").and_then(Value::as_array) else {
        return KvUsedPctOutcome::Ok;
    };
    for (i, step) in timeline.iter().enumerate() {
        let used = step.get("used_blocks").and_then(Value::as_u64).unwrap_or(0) as f64;
        let got = step.get("used_pct").and_then(Value::as_f64).unwrap_or(0.0);
        let expected = if total > 0.0 { used / total } else { 0.0 };
        if (got - expected).abs() > 1e-9 {
            return KvUsedPctOutcome::Mismatch {
                index: i,
                expected,
                got,
            };
        }
    }
    KvUsedPctOutcome::Ok
}

/// Verify `peak_used_pct == max(timeline.used_pct)` and
/// `preemption_count == sum(timeline.preempted_seqs)`.
pub fn classify_peak_consistency(body: &Value) -> KvPeakOutcome {
    let peak_got = body
        .get("peak_used_pct")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let pc_got = body
        .get("preemption_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let timeline = body.get("timeline").and_then(Value::as_array);

    let (peak_expected, pc_expected) = match timeline {
        Some(t) => {
            let mut peak = f64::NEG_INFINITY;
            let mut total_preempt: u64 = 0;
            for step in t {
                let up = step.get("used_pct").and_then(Value::as_f64).unwrap_or(0.0);
                if up > peak {
                    peak = up;
                }
                total_preempt = total_preempt.saturating_add(
                    step.get("preempted_seqs")
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                );
            }
            if peak == f64::NEG_INFINITY {
                peak = 0.0;
            }
            (peak, total_preempt)
        }
        None => (0.0, 0),
    };

    if (peak_got - peak_expected).abs() > 1e-9 {
        return KvPeakOutcome::PeakMismatch {
            expected: peak_expected,
            got: peak_got,
        };
    }
    if pc_got != pc_expected {
        return KvPeakOutcome::PreemptionCountMismatch {
            expected: pc_expected,
            got: pc_got,
        };
    }
    KvPeakOutcome::Ok
}

/// Verify `preempted_seqs > 0 IMPLIES used_pct >= threshold` for every step.
pub fn classify_preemption_trigger(body: &Value, threshold: f64) -> KvPreemptionOutcome {
    let Some(timeline) = body.get("timeline").and_then(Value::as_array) else {
        return KvPreemptionOutcome::Ok;
    };
    for (i, step) in timeline.iter().enumerate() {
        let preempted = step
            .get("preempted_seqs")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if preempted == 0 {
            continue;
        }
        let up = step.get("used_pct").and_then(Value::as_f64).unwrap_or(0.0);
        if up < threshold {
            return KvPreemptionOutcome::PreemptionBelowThreshold {
                index: i,
                used_pct: up,
                threshold,
                preempted_seqs: preempted,
            };
        }
    }
    KvPreemptionOutcome::Ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn good_body() -> Value {
        json!({
            "timeline": [
                {"step": 0, "t_ms": 0.0,  "used_blocks":  10, "free_blocks":  90, "used_pct": 0.10, "active_seqs": 1, "preempted_seqs": 0},
                {"step": 1, "t_ms": 8.0,  "used_blocks":  50, "free_blocks":  50, "used_pct": 0.50, "active_seqs": 1, "preempted_seqs": 0},
                {"step": 2, "t_ms": 16.0, "used_blocks":  96, "free_blocks":   4, "used_pct": 0.96, "active_seqs": 2, "preempted_seqs": 1},
            ],
            "block_size_tokens": 16,
            "total_blocks": 100,
            "peak_used_pct": 0.96,
            "preemption_count": 1,
        })
    }

    #[test]
    fn schema_ok_on_well_formed_body() {
        assert_eq!(classify_schema(&good_body()), KvSchemaOutcome::Ok);
    }

    #[test]
    fn schema_rejects_not_an_object() {
        assert_eq!(
            classify_schema(&json!([1, 2])),
            KvSchemaOutcome::NotAnObject
        );
    }

    #[test]
    fn schema_reports_missing_top_key() {
        let body = json!({"timeline": [], "block_size_tokens": 16, "total_blocks": 100, "peak_used_pct": 0.0});
        assert_eq!(
            classify_schema(&body),
            KvSchemaOutcome::MissingTopKey {
                key: "preemption_count"
            }
        );
    }

    #[test]
    fn schema_reports_missing_step_key() {
        let body = json!({
            "timeline": [{"step": 0, "t_ms": 0.0}],
            "block_size_tokens": 16,
            "total_blocks": 100,
            "peak_used_pct": 0.0,
            "preemption_count": 0,
        });
        match classify_schema(&body) {
            KvSchemaOutcome::StepMissingKey { index, .. } => assert_eq!(index, 0),
            other => panic!("expected StepMissingKey, got {other:?}"),
        }
    }

    #[test]
    fn schema_rejects_nonpositive_block_size() {
        let body = json!({
            "timeline": [],
            "block_size_tokens": 0,
            "total_blocks": 100,
            "peak_used_pct": 0.0,
            "preemption_count": 0,
        });
        assert!(matches!(
            classify_schema(&body),
            KvSchemaOutcome::BlockSizeNotPositive { got: 0 }
        ));
    }

    #[test]
    fn block_conservation_ok_on_good_body() {
        assert_eq!(
            classify_block_conservation(&good_body()),
            KvBlockConservationOutcome::Ok
        );
    }

    #[test]
    fn block_conservation_reports_violation() {
        let body = json!({
            "timeline": [{"step": 0, "t_ms": 0.0, "used_blocks": 30, "free_blocks": 30, "used_pct": 0.30, "active_seqs": 1, "preempted_seqs": 0}],
            "block_size_tokens": 16,
            "total_blocks": 100,
            "peak_used_pct": 0.30,
            "preemption_count": 0,
        });
        match classify_block_conservation(&body) {
            KvBlockConservationOutcome::Violation {
                index,
                used_blocks,
                free_blocks,
                total_blocks,
            } => {
                assert_eq!(index, 0);
                assert_eq!(used_blocks, 30);
                assert_eq!(free_blocks, 30);
                assert_eq!(total_blocks, 100);
            }
            other => panic!("expected Violation, got {other:?}"),
        }
    }

    #[test]
    fn used_pct_arithmetic_ok_on_good_body() {
        assert_eq!(
            classify_used_pct_arithmetic(&good_body()),
            KvUsedPctOutcome::Ok
        );
    }

    #[test]
    fn used_pct_arithmetic_reports_mismatch() {
        let body = json!({
            "timeline": [{"step": 0, "t_ms": 0.0, "used_blocks": 50, "free_blocks": 50, "used_pct": 0.10, "active_seqs": 1, "preempted_seqs": 0}],
            "block_size_tokens": 16,
            "total_blocks": 100,
            "peak_used_pct": 0.10,
            "preemption_count": 0,
        });
        assert!(matches!(
            classify_used_pct_arithmetic(&body),
            KvUsedPctOutcome::Mismatch { index: 0, .. }
        ));
    }

    #[test]
    fn peak_consistency_ok_on_good_body() {
        assert_eq!(classify_peak_consistency(&good_body()), KvPeakOutcome::Ok);
    }

    #[test]
    fn peak_consistency_reports_peak_mismatch() {
        let mut body = good_body();
        body["peak_used_pct"] = json!(0.50);
        assert!(matches!(
            classify_peak_consistency(&body),
            KvPeakOutcome::PeakMismatch { .. }
        ));
    }

    #[test]
    fn peak_consistency_reports_preemption_count_mismatch() {
        let mut body = good_body();
        body["preemption_count"] = json!(99);
        assert!(matches!(
            classify_peak_consistency(&body),
            KvPeakOutcome::PreemptionCountMismatch {
                expected: 1,
                got: 99
            }
        ));
    }

    #[test]
    fn preemption_trigger_ok_on_good_body() {
        assert_eq!(
            classify_preemption_trigger(&good_body(), F06_DEFAULT_PREEMPT_THRESHOLD),
            KvPreemptionOutcome::Ok
        );
    }

    #[test]
    fn preemption_trigger_reports_below_threshold() {
        let body = json!({
            "timeline": [{"step": 0, "t_ms": 0.0, "used_blocks": 50, "free_blocks": 50, "used_pct": 0.50, "active_seqs": 1, "preempted_seqs": 1}],
            "block_size_tokens": 16,
            "total_blocks": 100,
            "peak_used_pct": 0.50,
            "preemption_count": 1,
        });
        assert!(matches!(
            classify_preemption_trigger(&body, F06_DEFAULT_PREEMPT_THRESHOLD),
            KvPreemptionOutcome::PreemptionBelowThreshold { index: 0, .. }
        ));
    }
}
