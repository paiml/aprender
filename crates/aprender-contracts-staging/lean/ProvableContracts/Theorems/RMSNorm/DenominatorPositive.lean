import ProvableContracts.Defs.RMSNorm
import Mathlib.Data.Real.Sqrt

/-!
# RMSNorm Denominator Positivity

Proves that the RMS denominator √(mean(x²) + ε) is strictly positive
when ε > 0.

## Obligation

`RN-BND-001`: RMS(x) > 0 when ε > 0.

Since x² ≥ 0 for all x, mean(x²) ≥ 0. Adding ε > 0 gives a strictly
positive argument to √, and √ of a positive real is positive.
-/

namespace ProvableContracts.RMSNorm

open Finset

-- Status: proved
/-- mean_sq is non-negative: a sum of squares divided by a positive
    natural number is non-negative. -/
theorem mean_sq_nonneg {n : ℕ} (x : RVec (n + 1)) :
    mean_sq x ≥ 0 := by
  unfold mean_sq
  apply div_nonneg
  · apply Finset.sum_nonneg
    intro i _
    exact sq_nonneg (x i)
  · positivity

-- Status: proved
/-- The RMS denominator is strictly positive when ε > 0. -/
theorem rms_pos {n : ℕ} (x : RVec (n + 1)) (eps : ℝ) (heps : eps > 0) :
    rms x eps > 0 := by
  unfold rms
  apply Real.sqrt_pos_of_pos
  linarith [mean_sq_nonneg x]

-- Tests
#check @mean_sq_nonneg
#check @rms_pos

end ProvableContracts.RMSNorm
