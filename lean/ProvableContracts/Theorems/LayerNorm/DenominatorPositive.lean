import ProvableContracts.Defs.LayerNorm
import Mathlib.Data.Real.Sqrt

/-!
# LayerNorm Denominator Positivity

Proves that √(σ² + ε) > 0 when ε > 0.

## Obligation

`LN-BND-001`: √(variance(x) + ε) > 0 when ε > 0.

Variance is a sum of squares ÷ n, hence ≥ 0. Adding ε > 0 gives
a strictly positive argument to √.
-/

namespace ProvableContracts.LayerNorm

open Finset

-- Status: proved
/-- Variance is non-negative: a sum of squares divided by n+1. -/
theorem variance_nonneg {n : ℕ} (x : RVec (n + 1)) :
    variance x ≥ 0 := by
  unfold variance
  apply div_nonneg
  · apply Finset.sum_nonneg
    intro i _
    exact sq_nonneg _
  · positivity

-- Status: proved
/-- The LayerNorm denominator is strictly positive when ε > 0. -/
theorem ln_denom_pos {n : ℕ} (x : RVec (n + 1)) (eps : ℝ) (heps : eps > 0) :
    ln_denom x eps > 0 := by
  unfold ln_denom
  apply Real.sqrt_pos_of_pos
  linarith [variance_nonneg x]

-- Tests
#check @variance_nonneg
#check @ln_denom_pos

end ProvableContracts.LayerNorm
