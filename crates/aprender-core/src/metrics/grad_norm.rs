//! Gradient-norm telemetry classifier (CRUX-F-09).
//!
//! Pure-function analysis of per-step gradient-norm training telemetry.
//! Mirrors PyTorch `torch.nn.utils.clip_grad_norm_` convention: pre-clip L2
//! norm over parameter gradients, with optional clipping and spike detection
//! against a rolling median.
//!
//! Contract: `contracts/crux-F-09-v1.yaml`.
//!
//! This module takes pre-collected per-step grad-norm values as input rather
//! than running training itself — it's the pure analysis half. Wiring into
//! `apr finetune` / `apr pretrain` as a live telemetry hook is discharged
//! as PARTIAL_ALGORITHM_LEVEL under BLOCKER-UPSTREAM-MISSING until a stable
//! training-loop hook lands.

/// Per-step gradient telemetry record (JSON-parseable).
#[derive(Debug, Clone, PartialEq)]
pub struct StepRecord {
    pub step: u64,
    pub grad_norm: f64,
    pub grad_norm_clipped: Option<f64>,
    pub loss: Option<f64>,
}

/// Outcome of `compute_grad_norm_l2` — pure L2 norm over a flat gradient vector.
#[derive(Debug, Clone, PartialEq)]
pub enum GradNormOutcome {
    Ok(f64),
    EmptyGradients,
    NonFiniteGradient,
}

/// Computes the L2 norm of a flat gradient vector: `sqrt(sum g_i^2)`.
///
/// Invariants:
/// - `empty -> EmptyGradients` (distinct variant; no silent pass)
/// - `NaN/±∞ -> NonFiniteGradient`
/// - `all finite -> Ok(v)` with `v >= 0.0` and finite.
pub fn compute_grad_norm_l2(gradients: &[f64]) -> GradNormOutcome {
    if gradients.is_empty() {
        return GradNormOutcome::EmptyGradients;
    }
    for &g in gradients {
        if !g.is_finite() {
            return GradNormOutcome::NonFiniteGradient;
        }
    }
    let sum_sq: f64 = gradients.iter().map(|g| g * g).sum();
    GradNormOutcome::Ok(sum_sq.sqrt())
}

/// Outcome of `clip_grad_norm` — PyTorch-style in-place L2 clipping.
#[derive(Debug, Clone, PartialEq)]
pub enum ClipOutcome {
    /// Reports pre-clip norm and the post-clip norm after scaling.
    Ok {
        pre_norm: f64,
        post_norm: f64,
    },
    EmptyGradients,
    NonFiniteGradient,
    NonFiniteMaxNorm,
    NonPositiveMaxNorm(f64),
}

/// Clips gradients to `max_norm` L2 in place; returns pre and post norms.
///
/// Invariants (see contract `crux-F-09-v1` §equations):
/// - `post_norm <= pre_norm` (clipping non-expansive)
/// - `post_norm <= max_norm + 1e-6`
/// - when `pre_norm <= max_norm`, gradients unchanged and `post_norm == pre_norm`.
pub fn clip_grad_norm(gradients: &mut [f64], max_norm: f64) -> ClipOutcome {
    if !max_norm.is_finite() {
        return ClipOutcome::NonFiniteMaxNorm;
    }
    if max_norm <= 0.0 {
        return ClipOutcome::NonPositiveMaxNorm(max_norm);
    }
    let pre = match compute_grad_norm_l2(gradients) {
        GradNormOutcome::Ok(v) => v,
        GradNormOutcome::EmptyGradients => return ClipOutcome::EmptyGradients,
        GradNormOutcome::NonFiniteGradient => return ClipOutcome::NonFiniteGradient,
    };
    if pre <= max_norm {
        return ClipOutcome::Ok {
            pre_norm: pre,
            post_norm: pre,
        };
    }
    let scale = max_norm / pre;
    for g in gradients.iter_mut() {
        *g *= scale;
    }
    let post = match compute_grad_norm_l2(gradients) {
        GradNormOutcome::Ok(v) => v,
        _ => unreachable!("rescaling finite gradients cannot introduce NaN"),
    };
    ClipOutcome::Ok {
        pre_norm: pre,
        post_norm: post,
    }
}

/// Spike detection result over the history of grad-norms up to (and including) step `k`.
#[derive(Debug, Clone, PartialEq)]
pub enum SpikeOutcome {
    /// Not enough history yet — skipping (first `window` steps).
    NotEnoughHistory,
    /// No spike detected at this step.
    NoSpike { median: f64, ratio: f64 },
    /// Spike detected: `grad_norm > multiplier * rolling_median_{window}`.
    Spike { median: f64, ratio: f64 },
}

/// Detects a grad-norm spike at index `k` using a rolling median over the
/// previous `window` steps and a `multiplier` threshold.
///
/// Returns `NotEnoughHistory` when `k < window`. Otherwise computes the
/// median of `history[k-window..k]` and flags a `Spike` iff
/// `history[k] > multiplier * median`. Requires `multiplier > 0` and all
/// history values finite and ≥ 0.
///
/// Invariants:
/// - `Spike` iff `history[k] > multiplier * rolling_median_{window}(k)`.
/// - `NoSpike` otherwise.
///
/// # Panics
/// Panics if `window == 0`, `multiplier <= 0.0`, or `k >= history.len()`.
pub fn detect_grad_spike(
    history: &[f64],
    k: usize,
    window: usize,
    multiplier: f64,
) -> SpikeOutcome {
    assert!(window > 0, "window must be > 0");
    assert!(multiplier > 0.0, "multiplier must be > 0");
    assert!(
        k < history.len(),
        "k={k} out of range len={}",
        history.len()
    );

    if k < window {
        return SpikeOutcome::NotEnoughHistory;
    }
    let mut win: Vec<f64> = history[k - window..k].to_vec();
    win.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if win.len() % 2 == 1 {
        win[win.len() / 2]
    } else {
        0.5 * (win[win.len() / 2 - 1] + win[win.len() / 2])
    };
    let ratio = if median == 0.0 {
        f64::INFINITY
    } else {
        history[k] / median
    };
    if history[k] > multiplier * median {
        SpikeOutcome::Spike { median, ratio }
    } else {
        SpikeOutcome::NoSpike { median, ratio }
    }
}

/// Aggregated report over a full history of per-step grad-norms.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryReport {
    pub num_steps: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub num_spikes: usize,
    pub all_non_negative: bool,
    pub clipping_non_expansive: bool,
    pub max_exceeds_cap: bool,
}

/// Analyzes a full history, checking the three grad-norm invariants
/// (non-negative, clipping non-expansive, post-clip ≤ cap+eps) and
/// counting spikes against a rolling-median threshold.
pub fn analyze_history(
    records: &[StepRecord],
    max_grad_norm: Option<f64>,
    spike_window: usize,
    spike_multiplier: f64,
) -> HistoryReport {
    let n = records.len();
    if n == 0 {
        return HistoryReport {
            num_steps: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            num_spikes: 0,
            all_non_negative: true,
            clipping_non_expansive: true,
            max_exceeds_cap: false,
        };
    }
    let norms: Vec<f64> = records.iter().map(|r| r.grad_norm).collect();
    let min = norms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean: f64 = norms.iter().sum::<f64>() / (n as f64);
    let all_non_negative = norms.iter().all(|&v| v >= 0.0 && v.is_finite());
    let clipping_non_expansive = records.iter().all(|r| {
        r.grad_norm_clipped.map_or(true, |c| {
            c.is_finite() && c >= 0.0 && c <= r.grad_norm + 1e-9
        })
    });
    let max_exceeds_cap = match max_grad_norm {
        Some(cap) => records
            .iter()
            .any(|r| r.grad_norm_clipped.map_or(false, |c| c > cap + 1e-6)),
        None => false,
    };

    let mut num_spikes = 0usize;
    if spike_window > 0 && spike_multiplier > 0.0 {
        for k in spike_window..n {
            if let SpikeOutcome::Spike { .. } =
                detect_grad_spike(&norms, k, spike_window, spike_multiplier)
            {
                num_spikes += 1;
            }
        }
    }

    HistoryReport {
        num_steps: n,
        min,
        max,
        mean,
        num_spikes,
        all_non_negative,
        clipping_non_expansive,
        max_exceeds_cap,
    }
}

// ─── Classifier one-liners for falsification-test dispatch ─────────────

pub fn classify_empty_distinct() -> bool {
    matches!(compute_grad_norm_l2(&[]), GradNormOutcome::EmptyGradients)
}

pub fn classify_l2_matches_formula() -> bool {
    // Known: ||(3, 4)||_2 == 5
    matches!(compute_grad_norm_l2(&[3.0, 4.0]), GradNormOutcome::Ok(v) if (v - 5.0).abs() < 1e-12)
}

pub fn classify_clip_non_expansive() -> bool {
    let mut grads = [3.0_f64, 4.0]; // pre-norm 5.0
    match clip_grad_norm(&mut grads, 1.0) {
        ClipOutcome::Ok {
            pre_norm,
            post_norm,
        } => pre_norm > post_norm && (post_norm - 1.0).abs() < 1e-6,
        _ => false,
    }
}

pub fn classify_clip_identity_below_cap() -> bool {
    let mut grads = [0.3_f64, 0.4]; // pre-norm 0.5
    match clip_grad_norm(&mut grads, 1.0) {
        ClipOutcome::Ok {
            pre_norm,
            post_norm,
        } => (pre_norm - post_norm).abs() < 1e-12,
        _ => false,
    }
}

pub fn classify_spike_detected() -> bool {
    let hist = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0];
    matches!(
        detect_grad_spike(&hist, 8, 8, 10.0),
        SpikeOutcome::Spike { .. }
    )
}

pub fn classify_no_spike_in_stable_run() -> bool {
    let hist = vec![1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 1.08, 0.92, 1.01];
    matches!(
        detect_grad_spike(&hist, 8, 8, 10.0),
        SpikeOutcome::NoSpike { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── compute_grad_norm_l2 ──────────────────────────────────────────

    #[test]
    fn empty_gradients_distinct_outcome() {
        assert!(matches!(
            compute_grad_norm_l2(&[]),
            GradNormOutcome::EmptyGradients
        ));
    }

    #[test]
    fn l2_norm_of_3_4_is_5() {
        assert!(matches!(
            compute_grad_norm_l2(&[3.0, 4.0]),
            GradNormOutcome::Ok(v) if (v - 5.0).abs() < 1e-12
        ));
    }

    #[test]
    fn zero_gradients_give_zero_norm() {
        assert!(matches!(
            compute_grad_norm_l2(&[0.0, 0.0, 0.0]),
            GradNormOutcome::Ok(v) if v == 0.0
        ));
    }

    #[test]
    fn nan_gradient_rejected() {
        assert!(matches!(
            compute_grad_norm_l2(&[1.0, f64::NAN, 2.0]),
            GradNormOutcome::NonFiniteGradient
        ));
    }

    #[test]
    fn infinite_gradient_rejected() {
        assert!(matches!(
            compute_grad_norm_l2(&[1.0, f64::INFINITY, 2.0]),
            GradNormOutcome::NonFiniteGradient
        ));
    }

    #[test]
    fn negative_gradients_still_give_non_negative_norm() {
        // L2 norm of signed gradients is always >= 0
        match compute_grad_norm_l2(&[-3.0, -4.0]) {
            GradNormOutcome::Ok(v) => assert!((v - 5.0).abs() < 1e-12),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    // ── clip_grad_norm ────────────────────────────────────────────────

    #[test]
    fn clip_scales_down_to_cap() {
        let mut grads = vec![3.0_f64, 4.0];
        match clip_grad_norm(&mut grads, 1.0) {
            ClipOutcome::Ok {
                pre_norm,
                post_norm,
            } => {
                assert!((pre_norm - 5.0).abs() < 1e-12);
                assert!((post_norm - 1.0).abs() < 1e-6);
                assert!(post_norm <= pre_norm);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
        // Actual gradients should have been rescaled.
        let resulting = compute_grad_norm_l2(&grads);
        assert!(matches!(resulting, GradNormOutcome::Ok(v) if (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn clip_identity_when_below_cap() {
        let mut grads = vec![0.3_f64, 0.4];
        let copy = grads.clone();
        match clip_grad_norm(&mut grads, 1.0) {
            ClipOutcome::Ok {
                pre_norm,
                post_norm,
            } => {
                assert!((pre_norm - 0.5).abs() < 1e-12);
                assert!((post_norm - 0.5).abs() < 1e-12);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
        // No rescaling should have occurred.
        assert_eq!(grads, copy);
    }

    #[test]
    fn clip_rejects_non_finite_max_norm() {
        let mut grads = vec![1.0, 2.0];
        assert!(matches!(
            clip_grad_norm(&mut grads, f64::NAN),
            ClipOutcome::NonFiniteMaxNorm
        ));
    }

    #[test]
    fn clip_rejects_non_positive_max_norm() {
        let mut grads = vec![1.0, 2.0];
        assert!(matches!(
            clip_grad_norm(&mut grads, 0.0),
            ClipOutcome::NonPositiveMaxNorm(_)
        ));
        assert!(matches!(
            clip_grad_norm(&mut grads, -1.0),
            ClipOutcome::NonPositiveMaxNorm(_)
        ));
    }

    #[test]
    fn clip_rejects_non_finite_gradient() {
        let mut grads = vec![1.0, f64::NAN];
        assert!(matches!(
            clip_grad_norm(&mut grads, 1.0),
            ClipOutcome::NonFiniteGradient
        ));
    }

    // ── spike detection ───────────────────────────────────────────────

    #[test]
    fn spike_detected_on_10x_spike() {
        let hist = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0];
        assert!(matches!(
            detect_grad_spike(&hist, 8, 8, 10.0),
            SpikeOutcome::Spike { .. }
        ));
    }

    #[test]
    fn no_spike_when_within_tolerance() {
        let hist = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0];
        assert!(matches!(
            detect_grad_spike(&hist, 8, 8, 10.0),
            SpikeOutcome::NoSpike { .. }
        ));
    }

    #[test]
    fn not_enough_history_before_window() {
        let hist = vec![1.0, 1.0, 1.0];
        assert!(matches!(
            detect_grad_spike(&hist, 2, 8, 10.0),
            SpikeOutcome::NotEnoughHistory
        ));
    }

    // ── aggregated report ─────────────────────────────────────────────

    #[test]
    fn analyze_history_detects_spike_and_checks_invariants() {
        let mut recs = vec![];
        for step in 0..8 {
            recs.push(StepRecord {
                step,
                grad_norm: 1.0,
                grad_norm_clipped: Some(0.9),
                loss: Some(2.0),
            });
        }
        recs.push(StepRecord {
            step: 8,
            grad_norm: 100.0,
            grad_norm_clipped: Some(1.0),
            loss: Some(10.0),
        });
        let report = analyze_history(&recs, Some(1.0), 8, 10.0);
        assert_eq!(report.num_steps, 9);
        assert_eq!(report.num_spikes, 1);
        assert!(report.all_non_negative);
        assert!(report.clipping_non_expansive);
        assert!(!report.max_exceeds_cap);
    }

    #[test]
    fn analyze_history_flags_clip_cap_violation() {
        let recs = vec![StepRecord {
            step: 0,
            grad_norm: 5.0,
            grad_norm_clipped: Some(2.5), // > cap=1.0
            loss: None,
        }];
        let report = analyze_history(&recs, Some(1.0), 0, 10.0);
        assert!(report.max_exceeds_cap);
    }

    #[test]
    fn analyze_history_flags_expansive_clipping() {
        let recs = vec![StepRecord {
            step: 0,
            grad_norm: 1.0,
            grad_norm_clipped: Some(2.0),
            loss: None,
        }];
        let report = analyze_history(&recs, None, 0, 10.0);
        assert!(!report.clipping_non_expansive);
    }

    #[test]
    fn all_classifier_stubs_pass() {
        assert!(classify_empty_distinct());
        assert!(classify_l2_matches_formula());
        assert!(classify_clip_non_expansive());
        assert!(classify_clip_identity_below_cap());
        assert!(classify_spike_detected());
        assert!(classify_no_spike_in_stable_run());
    }
}
