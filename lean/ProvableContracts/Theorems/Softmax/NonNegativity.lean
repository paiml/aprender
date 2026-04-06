import ProvableContracts.Defs.Softmax
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Softmax Non-Negativity

Proves that softmax outputs are strictly positive for all real inputs.

## Obligation

`SM-INV-002`: ∀ x ∈ ℝⁿ⁺¹, softmax(x)_i > 0

This is the starter proof — it follows directly from `Real.exp_pos`.
-/

namespace ProvableContracts.Softmax

open Real Finset

-- Status: proved
/-- The partition function Z = Σⱼ exp(xⱼ) is strictly positive. -/
theorem sum_exp_pos {n : ℕ} (x : RVec (n + 1)) :
    0 < ∑ j : Fin (n + 1), Real.exp (x j) :=
  Finset.sum_pos (fun j _ => Real.exp_pos (x j)) Finset.univ_nonempty

-- Status: proved
/-- Softmax outputs are strictly positive.
    Follows from `exp > 0` and the denominator being a sum of positives. -/
theorem softmax_pos {n : ℕ} (x : RVec (n + 1)) (i : Fin (n + 1)) :
    softmax x i > 0 := by
  unfold softmax
  exact div_pos (Real.exp_pos _) (sum_exp_pos x)

-- Tests
#check @sum_exp_pos
#check @softmax_pos

end ProvableContracts.Softmax
