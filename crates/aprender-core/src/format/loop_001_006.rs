// `training-loop-v1` algorithm-level PARTIAL discharge for
// FALSIFY-LOOP-001..006.
//
// Contract: `contracts/training-loop-v1.yaml`.
//
// LOOP-001: EMA loss decreasing (last-5 < first-5)
// LOOP-002: validation metrics present + finite + accuracy in [0, 1] per epoch
// LOOP-003: checkpoint restored loss within 1e-2 of original
// LOOP-004: LR follows warmup (increasing) + cosine (decreasing post-warmup)
// LOOP-005: train/val splits disjoint (zero overlap, sum == total)
// LOOP-006: data shuffled — every epoch order differs from previous

use std::collections::HashSet;

/// LOOP-003: tolerance for restored val loss vs checkpoint.
pub const AC_LOOP_RESTORE_TOLERANCE: f32 = 0.01;
/// LOOP-002: validation accuracy must be in [0.0, 1.0].
pub const AC_LOOP_ACCURACY_MIN: f32 = 0.0;
pub const AC_LOOP_ACCURACY_MAX: f32 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopVerdict {
    Pass,
    Fail,
}

/// Reference EMA: y_t = alpha * x_t + (1 - alpha) * y_{t-1}.
/// Standard alpha = 2 / (n + 1).
#[must_use]
pub fn ema(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    let alpha = 2.0 / (values.len() as f32 + 1.0);
    let mut out = values[0];
    for &v in values.iter().skip(1) {
        out = alpha * v + (1.0 - alpha) * out;
    }
    out
}

/// LOOP-001: EMA(losses[..5]) > EMA(losses[len-5..]) — losses decreasing.
#[must_use]
pub fn verdict_from_loss_decreasing(losses: &[f32]) -> LoopVerdict {
    if losses.len() < 10 {
        return LoopVerdict::Fail;
    }
    if losses.iter().any(|x| !x.is_finite()) {
        return LoopVerdict::Fail;
    }
    let n = losses.len();
    let first_5 = ema(&losses[..5]);
    let last_5 = ema(&losses[n - 5..]);
    if last_5 < first_5 {
        LoopVerdict::Pass
    } else {
        LoopVerdict::Fail
    }
}

/// LOOP-002: every epoch's val_loss is finite AND val_accuracy ∈ [0, 1].
#[must_use]
pub fn verdict_from_validation_per_epoch(
    val_losses: &[f32],
    val_accuracies: &[f32],
) -> LoopVerdict {
    if val_losses.is_empty() || val_accuracies.is_empty() {
        return LoopVerdict::Fail;
    }
    if val_losses.len() != val_accuracies.len() {
        return LoopVerdict::Fail;
    }
    for (l, a) in val_losses.iter().zip(val_accuracies.iter()) {
        if !l.is_finite() {
            return LoopVerdict::Fail;
        }
        if !(AC_LOOP_ACCURACY_MIN..=AC_LOOP_ACCURACY_MAX).contains(a) || !a.is_finite() {
            return LoopVerdict::Fail;
        }
    }
    LoopVerdict::Pass
}

/// LOOP-003: |restored - checkpoint| < tolerance.
#[must_use]
pub fn verdict_from_checkpoint_restorable(
    checkpoint_loss: f32,
    restored_loss: f32,
) -> LoopVerdict {
    if !checkpoint_loss.is_finite() || !restored_loss.is_finite() {
        return LoopVerdict::Fail;
    }
    if (checkpoint_loss - restored_loss).abs() < AC_LOOP_RESTORE_TOLERANCE {
        LoopVerdict::Pass
    } else {
        LoopVerdict::Fail
    }
}

/// LOOP-004: LR schedule — warmup (increasing) + cosine (decreasing).
///
/// `lrs` is the per-step LR sequence. `warmup_steps` is N.
/// Pass iff:
///   - `lrs[2] > lrs[0]` (warmup phase climbing)
///   - `lrs[N-1] < lrs[N+1]` (post-warmup decay)
///
/// Empty / too short → Fail.
#[must_use]
pub fn verdict_from_lr_schedule(lrs: &[f32], warmup_steps: usize) -> LoopVerdict {
    let need = warmup_steps + 5;
    if lrs.len() < need {
        return LoopVerdict::Fail;
    }
    if lrs.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return LoopVerdict::Fail;
    }
    // Warmup: LR increases from step 0 to step 2.
    // Use `<=` rather than `!(>)` so NaN (incomparable) maps to Fail explicitly.
    if lrs[2] <= lrs[0] {
        return LoopVerdict::Fail;
    }
    // Post-warmup: LR decreases (cosine).
    let after = warmup_steps + 4;
    let before = warmup_steps;
    if lrs[after] >= lrs[before] {
        return LoopVerdict::Fail;
    }
    LoopVerdict::Pass
}

/// LOOP-005: train/val splits disjoint AND `train.len() + val.len() == total`.
#[must_use]
pub fn verdict_from_split_disjoint(
    train_ids: &[u32],
    val_ids: &[u32],
    expected_total: usize,
) -> LoopVerdict {
    if train_ids.is_empty() || val_ids.is_empty() {
        return LoopVerdict::Fail;
    }
    if train_ids.len() + val_ids.len() != expected_total {
        return LoopVerdict::Fail;
    }
    let train: HashSet<u32> = train_ids.iter().copied().collect();
    let val: HashSet<u32> = val_ids.iter().copied().collect();
    if train.intersection(&val).next().is_some() {
        return LoopVerdict::Fail;
    }
    LoopVerdict::Pass
}

/// LOOP-006: data shuffled — every adjacent epoch pair differs.
#[must_use]
pub fn verdict_from_data_shuffled(epoch_orders: &[Vec<u32>]) -> LoopVerdict {
    if epoch_orders.len() < 2 {
        return LoopVerdict::Fail;
    }
    for window in epoch_orders.windows(2) {
        if window[0] == window[1] {
            return LoopVerdict::Fail;
        }
    }
    LoopVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_constants() {
        assert_eq!(AC_LOOP_RESTORE_TOLERANCE, 0.01);
        assert_eq!(AC_LOOP_ACCURACY_MIN, 0.0);
        assert_eq!(AC_LOOP_ACCURACY_MAX, 1.0);
    }

    // -----------------------------------------------------------------
    // Section 2: LOOP-001 loss decreasing.
    // -----------------------------------------------------------------
    #[test]
    fn floop001_pass_decreasing() {
        let losses: Vec<f32> = (0..20).map(|i| 5.0 - i as f32 * 0.2).collect();
        let v = verdict_from_loss_decreasing(&losses);
        assert_eq!(v, LoopVerdict::Pass);
    }

    #[test]
    fn floop001_fail_increasing() {
        let losses: Vec<f32> = (0..20).map(|i| 1.0 + i as f32 * 0.1).collect();
        let v = verdict_from_loss_decreasing(&losses);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop001_fail_flat() {
        let losses = vec![5.0_f32; 20];
        let v = verdict_from_loss_decreasing(&losses);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop001_fail_too_short() {
        let losses: Vec<f32> = (0..5).map(|i| 5.0 - i as f32 * 0.1).collect();
        let v = verdict_from_loss_decreasing(&losses);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop001_fail_nan() {
        let mut losses: Vec<f32> = (0..20).map(|i| 5.0 - i as f32 * 0.2).collect();
        losses[10] = f32::NAN;
        let v = verdict_from_loss_decreasing(&losses);
        assert_eq!(v, LoopVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: LOOP-002 validation.
    // -----------------------------------------------------------------
    #[test]
    fn floop002_pass_typical() {
        let losses = vec![1.5_f32, 1.2, 1.0, 0.9, 0.8];
        let accs = vec![0.6_f32, 0.7, 0.78, 0.82, 0.85];
        let v = verdict_from_validation_per_epoch(&losses, &accs);
        assert_eq!(v, LoopVerdict::Pass);
    }

    #[test]
    fn floop002_fail_nan_loss() {
        let losses = vec![1.5_f32, f32::NAN];
        let accs = vec![0.6_f32, 0.7];
        let v = verdict_from_validation_per_epoch(&losses, &accs);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop002_fail_accuracy_above_one() {
        let losses = vec![1.5_f32];
        let accs = vec![1.5_f32];
        let v = verdict_from_validation_per_epoch(&losses, &accs);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop002_fail_length_mismatch() {
        let losses = vec![1.5_f32, 1.2];
        let accs = vec![0.6_f32];
        let v = verdict_from_validation_per_epoch(&losses, &accs);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop002_fail_empty() {
        let v = verdict_from_validation_per_epoch(&[], &[]);
        assert_eq!(v, LoopVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: LOOP-003 checkpoint restorable.
    // -----------------------------------------------------------------
    #[test]
    fn floop003_pass_exact_match() {
        let v = verdict_from_checkpoint_restorable(1.234, 1.234);
        assert_eq!(v, LoopVerdict::Pass);
    }

    #[test]
    fn floop003_pass_within_tolerance() {
        let v = verdict_from_checkpoint_restorable(1.234, 1.239);
        assert_eq!(v, LoopVerdict::Pass);
    }

    #[test]
    fn floop003_fail_out_of_tolerance() {
        let v = verdict_from_checkpoint_restorable(1.234, 1.5);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop003_fail_nan() {
        let v = verdict_from_checkpoint_restorable(1.234, f32::NAN);
        assert_eq!(v, LoopVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: LOOP-004 LR schedule.
    // -----------------------------------------------------------------
    #[test]
    fn floop004_pass_warmup_then_cosine() {
        // 5 warmup steps + 5 post-warmup → LR climbs then drops
        let lrs: Vec<f32> = vec![
            0.0001, 0.0003, 0.0006, 0.0010, 0.0014, // warmup increasing
            0.0020, // peak
            0.0019, 0.0017, 0.0014, 0.0010, // cosine decay
        ];
        let v = verdict_from_lr_schedule(&lrs, 5);
        assert_eq!(v, LoopVerdict::Pass);
    }

    #[test]
    fn floop004_fail_constant() {
        let lrs = vec![0.001_f32; 12];
        let v = verdict_from_lr_schedule(&lrs, 5);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop004_fail_too_short() {
        let lrs = vec![0.001_f32; 3];
        let v = verdict_from_lr_schedule(&lrs, 5);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop004_fail_negative_lr() {
        let mut lrs: Vec<f32> = vec![0.0001, 0.0003, 0.0006, 0.0010, 0.0014, 0.0020, 0.0019, 0.0017, 0.0014, 0.0010];
        lrs[3] = -1.0;
        let v = verdict_from_lr_schedule(&lrs, 5);
        assert_eq!(v, LoopVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: LOOP-005 split disjoint, LOOP-006 shuffled.
    // -----------------------------------------------------------------
    #[test]
    fn floop005_pass_disjoint_complete() {
        let train: Vec<u32> = (0..80).collect();
        let val: Vec<u32> = (80..100).collect();
        let v = verdict_from_split_disjoint(&train, &val, 100);
        assert_eq!(v, LoopVerdict::Pass);
    }

    #[test]
    fn floop005_fail_overlap() {
        let train: Vec<u32> = (0..80).collect();
        let val: Vec<u32> = (75..95).collect();
        let v = verdict_from_split_disjoint(&train, &val, 100);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop005_fail_count_mismatch() {
        let train: Vec<u32> = (0..70).collect();
        let val: Vec<u32> = (80..100).collect();
        let v = verdict_from_split_disjoint(&train, &val, 100);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop005_fail_empty_train() {
        let val: Vec<u32> = (0..20).collect();
        let v = verdict_from_split_disjoint(&[], &val, 20);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop006_pass_three_distinct_orders() {
        let orders = vec![
            vec![1_u32, 2, 3, 4],
            vec![3_u32, 1, 4, 2],
            vec![2_u32, 4, 1, 3],
        ];
        let v = verdict_from_data_shuffled(&orders);
        assert_eq!(v, LoopVerdict::Pass);
    }

    #[test]
    fn floop006_fail_two_consecutive_same() {
        let orders = vec![
            vec![1_u32, 2, 3, 4],
            vec![3_u32, 1, 4, 2],
            vec![3_u32, 1, 4, 2], // same as previous
        ];
        let v = verdict_from_data_shuffled(&orders);
        assert_eq!(v, LoopVerdict::Fail);
    }

    #[test]
    fn floop006_fail_only_one_epoch() {
        let orders = vec![vec![1_u32, 2, 3]];
        let v = verdict_from_data_shuffled(&orders);
        assert_eq!(v, LoopVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 7: Realistic + EMA reference.
    // -----------------------------------------------------------------
    #[test]
    fn ema_constant_returns_constant() {
        assert!((ema(&[1.0_f32; 5]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ema_decreasing_returns_decreasing() {
        let v = ema(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        assert!(v < 5.0);
        assert!(v > 0.0);
    }

    #[test]
    fn realistic_healthy_passes_all_6() {
        let losses: Vec<f32> = (0..20).map(|i| 5.0 - i as f32 * 0.2).collect();
        let val_losses: Vec<f32> = vec![1.5, 1.2, 1.0, 0.9, 0.8];
        let val_accs: Vec<f32> = vec![0.6, 0.7, 0.78, 0.82, 0.85];
        let lrs: Vec<f32> = vec![
            0.0001, 0.0003, 0.0006, 0.0010, 0.0014,
            0.0020,
            0.0019, 0.0017, 0.0014, 0.0010,
        ];
        let train: Vec<u32> = (0..80).collect();
        let val: Vec<u32> = (80..100).collect();
        let orders = vec![vec![1_u32, 2, 3], vec![3_u32, 2, 1], vec![2_u32, 1, 3]];

        let v1 = verdict_from_loss_decreasing(&losses);
        let v2 = verdict_from_validation_per_epoch(&val_losses, &val_accs);
        let v3 = verdict_from_checkpoint_restorable(0.95, 0.953);
        let v4 = verdict_from_lr_schedule(&lrs, 5);
        let v5 = verdict_from_split_disjoint(&train, &val, 100);
        let v6 = verdict_from_data_shuffled(&orders);
        for v in [v1, v2, v3, v4, v5, v6] {
            assert_eq!(v, LoopVerdict::Pass);
        }
    }

    #[test]
    fn realistic_pre_fix_all_6_failures() {
        // Pre-fix regression class:
        //  1: model not learning — losses flat
        //  2: NaN val_loss
        //  3: checkpoint missing optimizer state → 0.5 drift
        //  4: LR scheduler not connected — constant LR
        //  5: data leakage — overlap between train/val
        //  6: shuffle not applied — same order every epoch
        let losses = vec![5.0_f32; 20];
        let val_losses = vec![f32::NAN, 1.2];
        let val_accs = vec![0.6_f32, 0.7];
        let lrs = vec![0.001_f32; 12];
        let train: Vec<u32> = (0..80).collect();
        let val: Vec<u32> = (75..95).collect(); // overlap
        let orders = vec![vec![1_u32, 2, 3], vec![1_u32, 2, 3]];

        let v1 = verdict_from_loss_decreasing(&losses);
        let v2 = verdict_from_validation_per_epoch(&val_losses, &val_accs);
        let v3 = verdict_from_checkpoint_restorable(0.95, 0.5);
        let v4 = verdict_from_lr_schedule(&lrs, 5);
        let v5 = verdict_from_split_disjoint(&train, &val, 100);
        let v6 = verdict_from_data_shuffled(&orders);
        for v in [v1, v2, v3, v4, v5, v6] {
            assert_eq!(v, LoopVerdict::Fail);
        }
    }
}
