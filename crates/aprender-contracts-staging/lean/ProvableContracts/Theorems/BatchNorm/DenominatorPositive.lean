import ProvableContracts.Defs.BatchNorm
import Mathlib.Data.Real.Sqrt

/-!
# BatchNorm Denominator Positivity

Proves that √(σ²_B + ε) > 0 when ε > 0.

## Obligation

`Denominator strictly positive`: √(batchVar(x) + ε) > 0 when ε > 0.

Batch variance is a sum of squares ÷ N, hence ≥ 0. Adding ε > 0 gives a
strictly positive argument to √.
-/

namespace ProvableContracts.BatchNorm

open Finset

-- Status: proved
/-- Batch variance is non-negative: a sum of squares divided by N. -/
theorem batchVar_nonneg {n : ℕ} (x : RVec (n + 1)) :
    batchVar x ≥ 0 := by
  unfold batchVar
  apply div_nonneg
  · apply Finset.sum_nonneg
    intro i _
    exact sq_nonneg _
  · positivity

-- Status: proved
/-- The BatchNorm denominator is strictly positive when ε > 0. -/
theorem bn_denom_pos {n : ℕ} (x : RVec (n + 1)) (eps : ℝ) (heps : eps > 0) :
    bn_denom x eps > 0 := by
  unfold bn_denom
  apply Real.sqrt_pos_of_pos
  linarith [batchVar_nonneg x]

-- Tests
#check @batchVar_nonneg
#check @bn_denom_pos

end ProvableContracts.BatchNorm
