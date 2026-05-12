// SHIP-TWO-001 — `lora-target-selection-v1` + `lora-gradient-flow-v1`
// algorithm-level PARTIAL discharge for FALSIFY-LTSEL-001..002 AND
// FALSIFY-LGF-001..002 (closes 4/4 across both LoRA family contracts).
//
// Contracts:
// - `contracts/lora-target-selection-v1.yaml`
// - `contracts/lora-gradient-flow-v1.yaml`
// Spec: Hu et al. (2022) LoRA — target module selection and gradient
// flow through the frozen base.

// ===========================================================================
// LTSEL-001 — ∀ target ∈ selected: target exists in base model weights
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ltsel001Verdict { Pass, Fail }

/// Pass iff every entry of `selected` is present in `base_modules`.
/// Both inputs are slices of module name strings.
#[must_use]
pub fn verdict_from_targets_exist(
    selected: &[&str],
    base_modules: &[&str],
) -> Ltsel001Verdict {
    if selected.is_empty() { return Ltsel001Verdict::Fail; }
    if base_modules.is_empty() { return Ltsel001Verdict::Fail; }
    for &t in selected {
        if t.is_empty() { return Ltsel001Verdict::Fail; }
        if !base_modules.iter().any(|&m| m == t) {
            return Ltsel001Verdict::Fail;
        }
    }
    Ltsel001Verdict::Pass
}

// ===========================================================================
// LTSEL-002 — Default selection = {q_proj, v_proj} for decoder-only LLMs
// ===========================================================================

pub const AC_LTSEL_002_DEFAULT_TARGETS: [&str; 2] = ["q_proj", "v_proj"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ltsel002Verdict { Pass, Fail }

/// Pass iff `defaults` (as a set) equals `{"q_proj", "v_proj"}`. Order
/// is irrelevant; duplicates within `defaults` are not allowed (the
/// verdict requires len == 2 AND both expected entries present).
#[must_use]
pub fn verdict_from_default_targets(defaults: &[&str]) -> Ltsel002Verdict {
    if defaults.len() != AC_LTSEL_002_DEFAULT_TARGETS.len() {
        return Ltsel002Verdict::Fail;
    }
    let mut seen_q = false;
    let mut seen_v = false;
    for &t in defaults {
        match t {
            "q_proj" => {
                if seen_q { return Ltsel002Verdict::Fail; } // duplicate
                seen_q = true;
            }
            "v_proj" => {
                if seen_v { return Ltsel002Verdict::Fail; } // duplicate
                seen_v = true;
            }
            _ => return Ltsel002Verdict::Fail, // unknown default
        }
    }
    if seen_q && seen_v { Ltsel002Verdict::Pass } else { Ltsel002Verdict::Fail }
}

// ===========================================================================
// LGF-001 — Frozen base: ∇W_base == 0 during LoRA training
// ===========================================================================

pub const AC_LGF_001_BASE_GRAD_TOLERANCE: f32 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lgf001Verdict { Pass, Fail }

/// Pass iff every base-weight gradient component is zero (within a
/// tight numerical tolerance — base must be FROZEN, not just small).
#[must_use]
pub fn verdict_from_frozen_base(base_grads: &[f32]) -> Lgf001Verdict {
    if base_grads.is_empty() { return Lgf001Verdict::Fail; }
    for &g in base_grads {
        if !g.is_finite() { return Lgf001Verdict::Fail; }
        if g.abs() > AC_LGF_001_BASE_GRAD_TOLERANCE { return Lgf001Verdict::Fail; }
    }
    Lgf001Verdict::Pass
}

// ===========================================================================
// LGF-002 — Adapter gradients: ∇A, ∇B non-zero for non-zero loss
// ===========================================================================

pub const AC_LGF_002_NONZERO_TOLERANCE: f32 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lgf002Verdict { Pass, Fail }

/// Pass iff:
/// 1. `loss > tolerance` (non-zero loss precondition)
/// 2. At least ONE component of `grad_a` is non-zero (above tolerance)
/// 3. At least ONE component of `grad_b` is non-zero (above tolerance)
///
/// Both must show signal — a single zero-gradient adapter is a sign
/// of broken backward (often: A initialized to zero AND B initialized
/// to zero produces identically-zero gradients — both must come alive).
#[must_use]
pub fn verdict_from_adapter_gradients(
    loss: f32,
    grad_a: &[f32],
    grad_b: &[f32],
) -> Lgf002Verdict {
    if !loss.is_finite() { return Lgf002Verdict::Fail; }
    if loss.abs() <= AC_LGF_002_NONZERO_TOLERANCE { return Lgf002Verdict::Fail; }
    if grad_a.is_empty() || grad_b.is_empty() { return Lgf002Verdict::Fail; }
    if !grad_a.iter().all(|v| v.is_finite()) { return Lgf002Verdict::Fail; }
    if !grad_b.iter().all(|v| v.is_finite()) { return Lgf002Verdict::Fail; }
    let any_a = grad_a.iter().any(|&g| g.abs() > AC_LGF_002_NONZERO_TOLERANCE);
    let any_b = grad_b.iter().any(|&g| g.abs() > AC_LGF_002_NONZERO_TOLERANCE);
    if any_a && any_b { Lgf002Verdict::Pass } else { Lgf002Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // LTSEL-001 (target exists)
    #[test] fn ltsel001_pass_canonical() {
        let selected = ["q_proj", "v_proj"];
        let base = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"];
        assert_eq!(verdict_from_targets_exist(&selected, &base), Ltsel001Verdict::Pass);
    }
    #[test] fn ltsel001_fail_missing_target() {
        // Operator typo: `qrproj` doesn't exist in the base.
        let selected = ["qrproj"];
        let base = ["q_proj", "v_proj"];
        assert_eq!(verdict_from_targets_exist(&selected, &base), Ltsel001Verdict::Fail);
    }
    #[test] fn ltsel001_fail_empty_selected() {
        let base = ["q_proj"];
        assert_eq!(verdict_from_targets_exist(&[], &base), Ltsel001Verdict::Fail);
    }
    #[test] fn ltsel001_fail_empty_base() {
        let selected = ["q_proj"];
        assert_eq!(verdict_from_targets_exist(&selected, &[]), Ltsel001Verdict::Fail);
    }
    #[test] fn ltsel001_fail_empty_string_target() {
        let selected = [""];
        let base = ["q_proj", "v_proj"];
        assert_eq!(verdict_from_targets_exist(&selected, &base), Ltsel001Verdict::Fail);
    }

    // LTSEL-002 (default targets)
    #[test] fn ltsel002_pass_canonical_order() {
        let defaults = ["q_proj", "v_proj"];
        assert_eq!(verdict_from_default_targets(&defaults), Ltsel002Verdict::Pass);
    }
    #[test] fn ltsel002_pass_reverse_order() {
        // Order shouldn't matter — set semantics.
        let defaults = ["v_proj", "q_proj"];
        assert_eq!(verdict_from_default_targets(&defaults), Ltsel002Verdict::Pass);
    }
    #[test] fn ltsel002_fail_only_q_proj() {
        let defaults = ["q_proj"];
        assert_eq!(verdict_from_default_targets(&defaults), Ltsel002Verdict::Fail);
    }
    #[test] fn ltsel002_fail_extra_target() {
        // Defaults must be exactly {q_proj, v_proj} — adding k_proj fails.
        let defaults = ["q_proj", "k_proj", "v_proj"];
        assert_eq!(verdict_from_default_targets(&defaults), Ltsel002Verdict::Fail);
    }
    #[test] fn ltsel002_fail_unknown_target() {
        let defaults = ["q_proj", "ffn_proj"];
        assert_eq!(verdict_from_default_targets(&defaults), Ltsel002Verdict::Fail);
    }
    #[test] fn ltsel002_fail_duplicates() {
        let defaults = ["q_proj", "q_proj"];
        assert_eq!(verdict_from_default_targets(&defaults), Ltsel002Verdict::Fail);
    }
    #[test] fn ltsel002_fail_empty() {
        let defaults: [&str; 0] = [];
        assert_eq!(verdict_from_default_targets(&defaults), Ltsel002Verdict::Fail);
    }

    // LGF-001 (frozen base)
    #[test] fn lgf001_pass_all_zero() {
        let grads = [0.0_f32; 100];
        assert_eq!(verdict_from_frozen_base(&grads), Lgf001Verdict::Pass);
    }
    #[test] fn lgf001_pass_within_tolerance() {
        let grads = [1e-10_f32, -1e-10, 0.0]; // all < 1e-9
        assert_eq!(verdict_from_frozen_base(&grads), Lgf001Verdict::Pass);
    }
    #[test] fn lgf001_fail_nonzero_grad() {
        // Even a small non-zero gradient indicates backward leaked into base.
        let grads = [0.0_f32, 0.0, 1e-3, 0.0];
        assert_eq!(verdict_from_frozen_base(&grads), Lgf001Verdict::Fail);
    }
    #[test] fn lgf001_fail_nan() {
        let grads = [0.0_f32, f32::NAN];
        assert_eq!(verdict_from_frozen_base(&grads), Lgf001Verdict::Fail);
    }
    #[test] fn lgf001_fail_empty() {
        assert_eq!(verdict_from_frozen_base(&[]), Lgf001Verdict::Fail);
    }

    // LGF-002 (adapter gradients)
    #[test] fn lgf002_pass_canonical() {
        // Loss > 0, both A and B have at least some non-zero gradient.
        let grad_a = [0.0_f32, 0.1, -0.05, 0.0];
        let grad_b = [0.2_f32, 0.0, 0.0, -0.3];
        assert_eq!(verdict_from_adapter_gradients(0.5, &grad_a, &grad_b), Lgf002Verdict::Pass);
    }
    #[test] fn lgf002_fail_zero_loss() {
        // The contract says "non-zero loss" precondition; zero loss
        // should never produce non-zero gradients (vacuous), and the
        // verdict explicitly rejects this case to surface "loss = 0
        // but training keeps going" as a regression class.
        let grad_a = [0.1_f32];
        let grad_b = [0.1_f32];
        assert_eq!(verdict_from_adapter_gradients(0.0, &grad_a, &grad_b), Lgf002Verdict::Fail);
    }
    #[test] fn lgf002_fail_zero_a_grad() {
        // A had zero gradient — means update path through A is broken.
        let grad_a = [0.0_f32, 0.0, 0.0];
        let grad_b = [0.1_f32, 0.2, 0.3];
        assert_eq!(verdict_from_adapter_gradients(0.5, &grad_a, &grad_b), Lgf002Verdict::Fail);
    }
    #[test] fn lgf002_fail_zero_b_grad() {
        let grad_a = [0.1_f32, 0.2];
        let grad_b = [0.0_f32, 0.0];
        assert_eq!(verdict_from_adapter_gradients(0.5, &grad_a, &grad_b), Lgf002Verdict::Fail);
    }
    #[test] fn lgf002_fail_both_zero() {
        let grad_a = [0.0_f32];
        let grad_b = [0.0_f32];
        assert_eq!(verdict_from_adapter_gradients(0.5, &grad_a, &grad_b), Lgf002Verdict::Fail);
    }
    #[test] fn lgf002_fail_nan_loss() {
        let grad_a = [0.1_f32];
        let grad_b = [0.1_f32];
        assert_eq!(
            verdict_from_adapter_gradients(f32::NAN, &grad_a, &grad_b),
            Lgf002Verdict::Fail
        );
    }
    #[test] fn lgf002_fail_nan_grad() {
        let grad_a = [f32::NAN];
        let grad_b = [0.1_f32];
        assert_eq!(verdict_from_adapter_gradients(0.5, &grad_a, &grad_b), Lgf002Verdict::Fail);
    }
    #[test] fn lgf002_fail_empty() {
        assert_eq!(verdict_from_adapter_gradients(0.5, &[], &[0.1_f32]), Lgf002Verdict::Fail);
        assert_eq!(verdict_from_adapter_gradients(0.5, &[0.1_f32], &[]), Lgf002Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert_eq!(AC_LTSEL_002_DEFAULT_TARGETS, ["q_proj", "v_proj"]);
        assert!((AC_LGF_001_BASE_GRAD_TOLERANCE - 1e-9).abs() < 1e-15);
        assert!((AC_LGF_002_NONZERO_TOLERANCE - 1e-9).abs() < 1e-15);
    }
}
