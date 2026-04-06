import ProvableContracts.Defs.Softmax
import ProvableContracts.Theorems.Softmax.NonNegativity
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Softmax Monotonicity

Proves that softmax preserves input ordering.

## Obligation

`SM-INV-003`: ∀ x ∈ ℝⁿ⁺¹, x_i > x_j → softmax(x)_i > softmax(x)_j
-/

namespace ProvableContracts.Softmax

open Real Finset

-- Status: proved
/-- Softmax preserves input ordering: larger input → larger output.
    Since both terms share the same positive denominator Z, we reduce
    to showing exp(xᵢ) > exp(xⱼ), which follows from exp being monotone. -/
theorem monotone {n : ℕ} (x : RVec (n + 1)) (i j : Fin (n + 1))
    (h : x i > x j) :
    softmax x i > softmax x j := by
  unfold softmax
  exact div_lt_div_of_pos_right (Real.exp_strictMono h) (sum_exp_pos x)

-- Tests
#check @monotone

end ProvableContracts.Softmax
