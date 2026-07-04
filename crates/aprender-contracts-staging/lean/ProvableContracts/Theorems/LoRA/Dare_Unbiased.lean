import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# DARE — unbiased drop-and-rescale

Contract: `lora-algebra-v1`, equation `dare_unbiased`.

DARE drops each delta entry with probability `p` and rescales the survivors by
`1/(1-p)`. The per-entry expectation is the two-point mean
`E[·] = (1-p)·(δ/(1-p)) + p·0`, which equals `δ` for `p ≠ 1` — i.e. DARE is an
unbiased estimator of `δ` (contract invariant "Unbiased estimator of delta").
-/

namespace ProvableContracts.LoRA.Dare

/-- Per-entry DARE expectation: kept w.p. `(1-p)` rescaled by `1/(1-p)`,
    dropped to `0` w.p. `p`. -/
noncomputable def dare_expectation (delta p : ℝ) : ℝ :=
  (1 - p) * (delta / (1 - p)) + p * 0

-- Status: proved (core algebraic; two-point expectation)
/-- Unbiasedness: `E[DARE(δ,p)] = δ` for dropout probability `p ≠ 1`. -/
theorem dare_unbiased (delta p : ℝ) (hp : p ≠ 1) :
    dare_expectation delta p = delta := by
  unfold dare_expectation
  have h : 1 - p ≠ 0 := by
    intro hh; apply hp; linarith
  rw [mul_zero, add_zero]
  field_simp

#check @dare_unbiased

end ProvableContracts.LoRA.Dare
