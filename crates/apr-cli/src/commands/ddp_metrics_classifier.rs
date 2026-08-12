//! DDP multi-GPU metrics classifier (CRUX-D-11).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-D-11-{001,002,003}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! captured `apr finetune --parallel ddp --json` outputs at N=1 and N>=2 GPUs:
//!
//!   * `classify_scaling_efficiency` — `T_N / (N * T_1) >= threshold`
//!     (default 0.85, vLLM/PyTorch DDP convention). T_X = `tokens_per_sec`.
//!   * `classify_loss_parity` — `|L_N - L_1| / L_1 <= tolerance` (default
//!     0.01). Verifies the all-reduce reduction + deterministic sampler
//!     preserve loss invariance at identical seed/total-samples.
//!   * `classify_allreduce_bandwidth` — every per-step bandwidth entry in
//!     `ddp_metrics.allreduce_bandwidth_gbps[]` is strictly positive.
//!
//! Full discharge of -001/-002/-003 requires a live multi-GPU `apr finetune`
//! actually emitting the JSON metrics — tracked as BLOCKER-UPSTREAM-MISSING.

use serde_json::Value;

/// Default DDP scaling-efficiency floor (PyTorch DDP convention).
pub const D11_DEFAULT_SCALING_FLOOR: f64 = 0.85;

/// Default DDP loss-parity tolerance (1%).
pub const D11_DEFAULT_LOSS_TOLERANCE: f64 = 0.01;

/// Outcome of `classify_scaling_efficiency`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DdpScalingOutcome {
    Ok {
        efficiency: f64,
    },
    MissingTokensPerSec {
        which: &'static str,
    },
    NonPositiveTokensPerSec {
        which: &'static str,
        got: f64,
    },
    InvalidWorldSize {
        got: i64,
    },
    BelowThreshold {
        world_size: i64,
        t1: f64,
        tn: f64,
        efficiency: f64,
        threshold: f64,
    },
}

/// Outcome of `classify_loss_parity`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DdpLossParityOutcome {
    Ok {
        rel_diff: f64,
    },
    MissingFinalLoss {
        which: &'static str,
    },
    NonPositiveBaselineLoss {
        got: f64,
    },
    Divergence {
        l1: f64,
        ln: f64,
        rel_diff: f64,
        tolerance: f64,
    },
}

/// Outcome of `classify_allreduce_bandwidth`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DdpAllreduceOutcome {
    Ok { steps: usize },
    MissingDdpMetrics,
    MissingBandwidthArray,
    EmptyBandwidthArray,
    NonPositiveBandwidth { step_index: usize, got: f64 },
}

/// Verify `T_N / (N * T_1) >= threshold` (default 0.85).
pub fn classify_scaling_efficiency(
    metrics_1gpu: &Value,
    metrics_ngpu: &Value,
    world_size: i64,
    threshold: f64,
) -> DdpScalingOutcome {
    if world_size <= 1 {
        return DdpScalingOutcome::InvalidWorldSize { got: world_size };
    }
    let Some(t1) = metrics_1gpu.get("tokens_per_sec").and_then(Value::as_f64) else {
        return DdpScalingOutcome::MissingTokensPerSec {
            which: "metrics_1gpu",
        };
    };
    let Some(tn) = metrics_ngpu.get("tokens_per_sec").and_then(Value::as_f64) else {
        return DdpScalingOutcome::MissingTokensPerSec {
            which: "metrics_ngpu",
        };
    };
    if t1 <= 0.0 {
        return DdpScalingOutcome::NonPositiveTokensPerSec {
            which: "metrics_1gpu",
            got: t1,
        };
    }
    if tn <= 0.0 {
        return DdpScalingOutcome::NonPositiveTokensPerSec {
            which: "metrics_ngpu",
            got: tn,
        };
    }
    let efficiency = tn / (world_size as f64 * t1);
    if efficiency < threshold {
        return DdpScalingOutcome::BelowThreshold {
            world_size,
            t1,
            tn,
            efficiency,
            threshold,
        };
    }
    DdpScalingOutcome::Ok { efficiency }
}

/// Verify `|L_N - L_1| / L_1 <= tolerance` (default 0.01).
pub fn classify_loss_parity(
    metrics_1gpu: &Value,
    metrics_ngpu: &Value,
    tolerance: f64,
) -> DdpLossParityOutcome {
    let Some(l1) = metrics_1gpu.get("final_loss").and_then(Value::as_f64) else {
        return DdpLossParityOutcome::MissingFinalLoss {
            which: "metrics_1gpu",
        };
    };
    let Some(ln) = metrics_ngpu.get("final_loss").and_then(Value::as_f64) else {
        return DdpLossParityOutcome::MissingFinalLoss {
            which: "metrics_ngpu",
        };
    };
    if l1 <= 0.0 {
        return DdpLossParityOutcome::NonPositiveBaselineLoss { got: l1 };
    }
    let rel_diff = (ln - l1).abs() / l1;
    if rel_diff > tolerance {
        return DdpLossParityOutcome::Divergence {
            l1,
            ln,
            rel_diff,
            tolerance,
        };
    }
    DdpLossParityOutcome::Ok { rel_diff }
}

/// Verify every per-step entry in `ddp_metrics.allreduce_bandwidth_gbps`
/// is strictly positive.
pub fn classify_allreduce_bandwidth(metrics: &Value) -> DdpAllreduceOutcome {
    let Some(ddp) = metrics.get("ddp_metrics") else {
        return DdpAllreduceOutcome::MissingDdpMetrics;
    };
    let Some(arr) = ddp
        .get("allreduce_bandwidth_gbps")
        .and_then(Value::as_array)
    else {
        return DdpAllreduceOutcome::MissingBandwidthArray;
    };
    if arr.is_empty() {
        return DdpAllreduceOutcome::EmptyBandwidthArray;
    }
    for (i, v) in arr.iter().enumerate() {
        let bw = v.as_f64().unwrap_or(0.0);
        if bw <= 0.0 {
            return DdpAllreduceOutcome::NonPositiveBandwidth {
                step_index: i,
                got: bw,
            };
        }
    }
    DdpAllreduceOutcome::Ok { steps: arr.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn t1_body() -> Value {
        json!({
            "tokens_per_sec": 1000.0,
            "final_loss": 2.5,
            "ddp_metrics": {"allreduce_bandwidth_gbps": [120.0, 118.0, 121.0]}
        })
    }

    fn tn_body_efficient() -> Value {
        // N=4: T4 = 3500 → eff = 3500 / 4000 = 0.875 (≥0.85)
        json!({
            "tokens_per_sec": 3500.0,
            "final_loss": 2.51,
            "ddp_metrics": {"allreduce_bandwidth_gbps": [80.0, 82.0, 79.0]}
        })
    }

    fn tn_body_inefficient() -> Value {
        // N=4: T4 = 2000 → eff = 2000 / 4000 = 0.5
        json!({
            "tokens_per_sec": 2000.0,
            "final_loss": 2.51,
            "ddp_metrics": {"allreduce_bandwidth_gbps": [40.0]}
        })
    }

    #[test]
    fn scaling_ok_at_0_875() {
        match classify_scaling_efficiency(&t1_body(), &tn_body_efficient(), 4, 0.85) {
            DdpScalingOutcome::Ok { efficiency } => {
                assert!((efficiency - 0.875).abs() < 1e-6, "got {efficiency}");
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn scaling_rejects_below_threshold() {
        match classify_scaling_efficiency(&t1_body(), &tn_body_inefficient(), 4, 0.85) {
            DdpScalingOutcome::BelowThreshold { efficiency, .. } => {
                assert!((efficiency - 0.5).abs() < 1e-6, "got {efficiency}");
            }
            other => panic!("expected BelowThreshold, got {other:?}"),
        }
    }

    #[test]
    fn scaling_rejects_invalid_world_size() {
        let out = classify_scaling_efficiency(&t1_body(), &tn_body_efficient(), 1, 0.85);
        assert!(matches!(
            out,
            DdpScalingOutcome::InvalidWorldSize { got: 1 }
        ));
    }

    #[test]
    fn scaling_rejects_missing_tokens_per_sec() {
        let empty = json!({});
        assert!(matches!(
            classify_scaling_efficiency(&empty, &tn_body_efficient(), 4, 0.85),
            DdpScalingOutcome::MissingTokensPerSec {
                which: "metrics_1gpu"
            }
        ));
    }

    #[test]
    fn scaling_rejects_zero_tokens_per_sec() {
        let zero = json!({"tokens_per_sec": 0.0});
        assert!(matches!(
            classify_scaling_efficiency(&t1_body(), &zero, 4, 0.85),
            DdpScalingOutcome::NonPositiveTokensPerSec {
                which: "metrics_ngpu",
                ..
            }
        ));
    }

    #[test]
    fn loss_parity_ok_within_tolerance() {
        let out = classify_loss_parity(&t1_body(), &tn_body_efficient(), 0.01);
        match out {
            DdpLossParityOutcome::Ok { rel_diff } => {
                // |2.51 - 2.5| / 2.5 = 0.004
                assert!((rel_diff - 0.004).abs() < 1e-6, "got {rel_diff}");
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn loss_parity_rejects_divergence() {
        let diverged = json!({"final_loss": 5.0, "tokens_per_sec": 1.0});
        // |5.0 - 2.5| / 2.5 = 1.0 > 0.01
        assert!(matches!(
            classify_loss_parity(&t1_body(), &diverged, 0.01),
            DdpLossParityOutcome::Divergence { .. }
        ));
    }

    #[test]
    fn loss_parity_rejects_missing_final_loss() {
        let empty = json!({});
        assert!(matches!(
            classify_loss_parity(&empty, &tn_body_efficient(), 0.01),
            DdpLossParityOutcome::MissingFinalLoss {
                which: "metrics_1gpu"
            }
        ));
    }

    #[test]
    fn allreduce_ok_on_positive_array() {
        match classify_allreduce_bandwidth(&tn_body_efficient()) {
            DdpAllreduceOutcome::Ok { steps } => assert_eq!(steps, 3),
            other => panic!("expected Ok(3), got {other:?}"),
        }
    }

    #[test]
    fn allreduce_rejects_missing_ddp_metrics() {
        let empty = json!({});
        assert_eq!(
            classify_allreduce_bandwidth(&empty),
            DdpAllreduceOutcome::MissingDdpMetrics
        );
    }

    #[test]
    fn allreduce_rejects_empty_array() {
        let body = json!({"ddp_metrics": {"allreduce_bandwidth_gbps": []}});
        assert_eq!(
            classify_allreduce_bandwidth(&body),
            DdpAllreduceOutcome::EmptyBandwidthArray
        );
    }

    #[test]
    fn allreduce_rejects_zero_bandwidth() {
        let body = json!({"ddp_metrics": {"allreduce_bandwidth_gbps": [120.0, 0.0, 121.0]}});
        match classify_allreduce_bandwidth(&body) {
            DdpAllreduceOutcome::NonPositiveBandwidth { step_index: 1, .. } => {}
            other => panic!("expected NonPositiveBandwidth(step 1), got {other:?}"),
        }
    }
}
