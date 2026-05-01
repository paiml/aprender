// SHIP-TWO-001 — `training-loop-pretrain-v1` algorithm-level
// PARTIAL discharge for INV-TRAIN-001..010 (closes 10/10).
//
// Contract: `contracts/training-loop-pretrain-v1.yaml`.
// Spec: training-loop pretrain invariants for MODEL-2 + finetune.

// ===========================================================================
// INV-TRAIN-001 — every step emits 6 required metrics
// ===========================================================================

pub const AC_INV_TRAIN_001_REQUIRED_FIELDS: u32 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvTrain001Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `INV-TRAIN-001`. Pass iff
/// `present_field_count == 6`.
#[must_use]
pub fn verdict_from_step_metric_count(present_field_count: u32) -> InvTrain001Verdict {
    if present_field_count == AC_INV_TRAIN_001_REQUIRED_FIELDS {
        InvTrain001Verdict::Pass
    } else {
        InvTrain001Verdict::Fail
    }
}

// ===========================================================================
// INV-TRAIN-002 — every epoch produces ckpt + 9-field metadata.json
// ===========================================================================

pub const AC_INV_TRAIN_002_REQUIRED_META_FIELDS: u32 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvTrain002Verdict {
    Pass,
    Fail,
}

/// Pass iff `ckpt_exists AND metadata_field_count == 9`.
#[must_use]
pub fn verdict_from_ckpt_meta(ckpt_exists: bool, metadata_field_count: u32) -> InvTrain002Verdict {
    if ckpt_exists && metadata_field_count == AC_INV_TRAIN_002_REQUIRED_META_FIELDS {
        InvTrain002Verdict::Pass
    } else {
        InvTrain002Verdict::Fail
    }
}

// ===========================================================================
// INV-TRAIN-003 — optimizer_state_sha matches recomputed sha256
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvTrain003Verdict {
    Pass,
    Fail,
}

/// Pass iff both digests are 32 bytes AND byte-identical.
#[must_use]
pub fn verdict_from_optimizer_sha(recorded: &[u8], recomputed: &[u8]) -> InvTrain003Verdict {
    if recorded.len() != 32 || recomputed.len() != 32 {
        return InvTrain003Verdict::Fail;
    }
    if recorded == recomputed {
        InvTrain003Verdict::Pass
    } else {
        InvTrain003Verdict::Fail
    }
}

// ===========================================================================
// INV-TRAIN-004 — convergence: val_loss[N] ≤ val_loss[N-1] OR patience++
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvTrain004Verdict {
    Pass,
    Fail,
}

/// Pass iff `val_loss_curr <= val_loss_prev OR patience_incremented`.
#[must_use]
pub fn verdict_from_convergence_step(
    val_loss_prev: f64,
    val_loss_curr: f64,
    patience_incremented: bool,
) -> InvTrain004Verdict {
    if !val_loss_prev.is_finite() || !val_loss_curr.is_finite() {
        return InvTrain004Verdict::Fail;
    }
    if val_loss_curr <= val_loss_prev || patience_incremented {
        InvTrain004Verdict::Pass
    } else {
        InvTrain004Verdict::Fail
    }
}

// ===========================================================================
// INV-TRAIN-005 — non-divergence: val_loss[N] ≤ 2× val_loss[N-1]
// ===========================================================================

pub const AC_INV_TRAIN_005_DIVERGENCE_FACTOR: f64 = 2.0;
pub const AC_INV_TRAIN_005_FINETUNE_INIT_CAP: f64 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvTrain005Verdict {
    Pass,
    Fail,
}

/// Pass iff `val_loss_curr <= 2.0 * val_loss_prev` (and both finite).
#[must_use]
pub fn verdict_from_non_divergence(
    val_loss_prev: f64,
    val_loss_curr: f64,
) -> InvTrain005Verdict {
    if !val_loss_prev.is_finite() || !val_loss_curr.is_finite() {
        return InvTrain005Verdict::Fail;
    }
    if val_loss_prev < 0.0 || val_loss_curr < 0.0 {
        return InvTrain005Verdict::Fail;
    }
    if val_loss_curr <= AC_INV_TRAIN_005_DIVERGENCE_FACTOR * val_loss_prev {
        InvTrain005Verdict::Pass
    } else {
        InvTrain005Verdict::Fail
    }
}

// ===========================================================================
// INV-TRAIN-006 — reproducibility (byte-identical first-100 metrics)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvTrain006Verdict {
    Pass,
    Fail,
}

/// Pass iff metric sequences are byte-identical for the first 100
/// steps. (`metrics_a == metrics_b`)
#[must_use]
pub fn verdict_from_reproducibility(metrics_a: &[u8], metrics_b: &[u8]) -> InvTrain006Verdict {
    if metrics_a.is_empty() || metrics_b.is_empty() {
        return InvTrain006Verdict::Fail;
    }
    if metrics_a == metrics_b {
        InvTrain006Verdict::Pass
    } else {
        InvTrain006Verdict::Fail
    }
}

// ===========================================================================
// INV-TRAIN-007 — no NaN/Inf in train_loss / grad_norm
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvTrain007Verdict {
    Pass,
    Fail,
}

/// Pass iff both loss and grad_norm are finite.
#[must_use]
pub fn verdict_from_no_nan_inf(train_loss: f64, grad_norm: f64) -> InvTrain007Verdict {
    if train_loss.is_finite() && grad_norm.is_finite() {
        InvTrain007Verdict::Pass
    } else {
        InvTrain007Verdict::Fail
    }
}

// ===========================================================================
// INV-TRAIN-008 — tokens_per_sec / gpu_util_pct non-negative + finite
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvTrain008Verdict {
    Pass,
    Fail,
}

/// Pass iff both throughput metrics are finite AND >= 0.0.
#[must_use]
pub fn verdict_from_throughput_validity(
    tokens_per_sec: f64,
    gpu_util_pct: f64,
) -> InvTrain008Verdict {
    if !tokens_per_sec.is_finite() || !gpu_util_pct.is_finite() {
        return InvTrain008Verdict::Fail;
    }
    if tokens_per_sec < 0.0 || gpu_util_pct < 0.0 {
        return InvTrain008Verdict::Fail;
    }
    InvTrain008Verdict::Pass
}

// ===========================================================================
// INV-TRAIN-009 — atomic 4-tuple HP defaults per regime
// ===========================================================================

pub const AC_INV_TRAIN_009_TUPLE_FIELDS: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvTrain009Verdict {
    Pass,
    Fail,
}

/// Pass iff `populated_field_count == 4` (regime, lr_max,
/// warmup_steps, target_val_loss all set).
#[must_use]
pub fn verdict_from_hp_atomic_tuple(populated_field_count: u32) -> InvTrain009Verdict {
    if populated_field_count == AC_INV_TRAIN_009_TUPLE_FIELDS {
        InvTrain009Verdict::Pass
    } else {
        InvTrain009Verdict::Fail
    }
}

// ===========================================================================
// INV-TRAIN-010 — drive_real default routing (synthetic == false)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvTrain010Verdict {
    Pass,
    Fail,
}

/// Pass iff `synthetic_flag == false` (default routes to drive_real).
#[must_use]
pub fn verdict_from_drive_real_default(synthetic_flag: bool) -> InvTrain010Verdict {
    if !synthetic_flag {
        InvTrain010Verdict::Pass
    } else {
        InvTrain010Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // INV-TRAIN-001 -------------------------------------------------------------
    #[test]
    fn t001_pass_six_fields() { assert_eq!(verdict_from_step_metric_count(6), InvTrain001Verdict::Pass); }
    #[test]
    fn t001_fail_five_fields() { assert_eq!(verdict_from_step_metric_count(5), InvTrain001Verdict::Fail); }
    #[test]
    fn t001_fail_seven_fields() { assert_eq!(verdict_from_step_metric_count(7), InvTrain001Verdict::Fail); }

    // INV-TRAIN-002 -------------------------------------------------------------
    #[test]
    fn t002_pass_ckpt_and_9_meta() { assert_eq!(verdict_from_ckpt_meta(true, 9), InvTrain002Verdict::Pass); }
    #[test]
    fn t002_fail_no_ckpt() { assert_eq!(verdict_from_ckpt_meta(false, 9), InvTrain002Verdict::Fail); }
    #[test]
    fn t002_fail_8_meta_fields() { assert_eq!(verdict_from_ckpt_meta(true, 8), InvTrain002Verdict::Fail); }

    // INV-TRAIN-003 -------------------------------------------------------------
    #[test]
    fn t003_pass_matching_sha() {
        let s = [0xab_u8; 32];
        assert_eq!(verdict_from_optimizer_sha(&s, &s), InvTrain003Verdict::Pass);
    }
    #[test]
    fn t003_fail_drift() {
        let a = [0xab_u8; 32];
        let mut b = [0xab_u8; 32]; b[0] = 0xac;
        assert_eq!(verdict_from_optimizer_sha(&a, &b), InvTrain003Verdict::Fail);
    }
    #[test]
    fn t003_fail_wrong_length() {
        assert_eq!(verdict_from_optimizer_sha(&[0u8; 16], &[0u8; 16]), InvTrain003Verdict::Fail);
    }

    // INV-TRAIN-004 -------------------------------------------------------------
    #[test]
    fn t004_pass_decreasing_loss() {
        assert_eq!(verdict_from_convergence_step(9.0, 8.0, false), InvTrain004Verdict::Pass);
    }
    #[test]
    fn t004_pass_increasing_with_patience() {
        assert_eq!(verdict_from_convergence_step(8.0, 9.0, true), InvTrain004Verdict::Pass);
    }
    #[test]
    fn t004_fail_increasing_without_patience() {
        assert_eq!(verdict_from_convergence_step(8.0, 9.0, false), InvTrain004Verdict::Fail);
    }
    #[test]
    fn t004_fail_nan() {
        assert_eq!(verdict_from_convergence_step(f64::NAN, 8.0, false), InvTrain004Verdict::Fail);
    }

    // INV-TRAIN-005 -------------------------------------------------------------
    #[test]
    fn t005_pass_decreasing() {
        assert_eq!(verdict_from_non_divergence(9.0, 8.0), InvTrain005Verdict::Pass);
    }
    #[test]
    fn t005_pass_at_2x_boundary() {
        assert_eq!(verdict_from_non_divergence(5.0, 10.0), InvTrain005Verdict::Pass);
    }
    #[test]
    fn t005_fail_above_2x() {
        assert_eq!(verdict_from_non_divergence(5.0, 10.001), InvTrain005Verdict::Fail);
    }
    #[test]
    fn t005_fail_negative_loss() {
        assert_eq!(verdict_from_non_divergence(-1.0, 5.0), InvTrain005Verdict::Fail);
    }
    #[test]
    fn t005_fail_nan() {
        assert_eq!(verdict_from_non_divergence(f64::NAN, 5.0), InvTrain005Verdict::Fail);
    }

    // INV-TRAIN-006 -------------------------------------------------------------
    #[test]
    fn t006_pass_byte_identical() {
        assert_eq!(verdict_from_reproducibility(b"abc", b"abc"), InvTrain006Verdict::Pass);
    }
    #[test]
    fn t006_fail_drift() {
        assert_eq!(verdict_from_reproducibility(b"abc", b"abd"), InvTrain006Verdict::Fail);
    }
    #[test]
    fn t006_fail_empty() {
        assert_eq!(verdict_from_reproducibility(&[], &[]), InvTrain006Verdict::Fail);
    }

    // INV-TRAIN-007 -------------------------------------------------------------
    #[test]
    fn t007_pass_finite_values() { assert_eq!(verdict_from_no_nan_inf(2.5, 1.2), InvTrain007Verdict::Pass); }
    #[test]
    fn t007_fail_nan_loss() { assert_eq!(verdict_from_no_nan_inf(f64::NAN, 1.0), InvTrain007Verdict::Fail); }
    #[test]
    fn t007_fail_inf_grad() { assert_eq!(verdict_from_no_nan_inf(2.0, f64::INFINITY), InvTrain007Verdict::Fail); }

    // INV-TRAIN-008 -------------------------------------------------------------
    #[test]
    fn t008_pass_finite_nonneg() { assert_eq!(verdict_from_throughput_validity(150.0, 95.5), InvTrain008Verdict::Pass); }
    #[test]
    fn t008_fail_negative_tps() { assert_eq!(verdict_from_throughput_validity(-1.0, 95.0), InvTrain008Verdict::Fail); }
    #[test]
    fn t008_fail_nan_gpu() { assert_eq!(verdict_from_throughput_validity(150.0, f64::NAN), InvTrain008Verdict::Fail); }
    #[test]
    fn t008_pass_zero() { assert_eq!(verdict_from_throughput_validity(0.0, 0.0), InvTrain008Verdict::Pass); }

    // INV-TRAIN-009 -------------------------------------------------------------
    #[test]
    fn t009_pass_4_tuple() { assert_eq!(verdict_from_hp_atomic_tuple(4), InvTrain009Verdict::Pass); }
    #[test]
    fn t009_fail_3_fields() { assert_eq!(verdict_from_hp_atomic_tuple(3), InvTrain009Verdict::Fail); }
    #[test]
    fn t009_fail_5_fields() { assert_eq!(verdict_from_hp_atomic_tuple(5), InvTrain009Verdict::Fail); }

    // INV-TRAIN-010 -------------------------------------------------------------
    #[test]
    fn t010_pass_synthetic_false() { assert_eq!(verdict_from_drive_real_default(false), InvTrain010Verdict::Pass); }
    #[test]
    fn t010_fail_synthetic_true() { assert_eq!(verdict_from_drive_real_default(true), InvTrain010Verdict::Fail); }

    // Provenance pins -----------------------------------------------------------
    #[test]
    fn provenance_pins() {
        assert_eq!(AC_INV_TRAIN_001_REQUIRED_FIELDS, 6);
        assert_eq!(AC_INV_TRAIN_002_REQUIRED_META_FIELDS, 9);
        assert_eq!(AC_INV_TRAIN_005_DIVERGENCE_FACTOR, 2.0);
        assert_eq!(AC_INV_TRAIN_005_FINETUNE_INIT_CAP, 10.0);
        assert_eq!(AC_INV_TRAIN_009_TUPLE_FIELDS, 4);
    }
}
