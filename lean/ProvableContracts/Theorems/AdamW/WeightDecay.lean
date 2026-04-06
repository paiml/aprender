import ProvableContracts.Defs.AdamW

/-!
# AdamW Weight Decay Reduces Magnitude

Proves that the weight decay term in AdamW reduces the magnitude
of weights: |decay_update(θ)| ≤ |θ| when 0 < lr·λ < 1.

## Obligation

`AW-INV-001`: |θ - lr·λ·θ| ≤ |θ| when 0 < lr·λ < 1

The decay update is θ·(1 - lr·λ). When 0 < lr·λ < 1, we have
0 < 1 - lr·λ < 1, so |θ·(1 - lr·λ)| = |θ|·|1 - lr·λ| ≤ |θ|.
-/

namespace ProvableContracts.AdamW

-- Status: proved
/-- Decay update equals (1 - lr·wd) · θ. -/
theorem decay_update_eq (theta lr wd : ℝ) :
    decay_update theta lr wd = (1 - lr * wd) * theta := by
  unfold decay_update
  ring

-- Status: proved
/-- Weight decay reduces magnitude when 0 < lr·wd < 1.
    |θ - lr·λ·θ| = |1 - lr·λ| · |θ| ≤ |θ|. -/
theorem weight_decay_reduces_magnitude (theta lr wd : ℝ)
    (h_pos : 0 < lr * wd) (h_lt : lr * wd < 1) :
    |decay_update theta lr wd| ≤ |theta| := by
  rw [decay_update_eq, abs_mul]
  have h1 : |1 - lr * wd| ≤ 1 := by
    rw [abs_le]
    constructor
    · linarith
    · linarith
  calc |1 - lr * wd| * |theta|
      ≤ 1 * |theta| := by exact mul_le_mul_of_nonneg_right h1 (abs_nonneg theta)
    _ = |theta| := by ring

-- Status: proved
/-- Weight decay strictly reduces magnitude for non-zero weights. -/
theorem weight_decay_strict (theta lr wd : ℝ) (htheta : theta ≠ 0)
    (h_pos : 0 < lr * wd) (h_lt : lr * wd < 1) :
    |decay_update theta lr wd| < |theta| := by
  rw [decay_update_eq, abs_mul]
  have h1 : |1 - lr * wd| < 1 := by
    rw [abs_lt]
    constructor
    · linarith
    · linarith
  exact mul_lt_of_lt_one_left (abs_pos.mpr htheta) h1

-- Tests
#check @decay_update_eq
#check @weight_decay_reduces_magnitude
#check @weight_decay_strict

example : decay_update 5.0 0.1 0.01 = (1 - 0.1 * 0.01) * 5.0 :=
  decay_update_eq 5.0 0.1 0.01

end ProvableContracts.AdamW
