import ProvableContracts.Defs.Softmax
import ProvableContracts.Theorems.Softmax.NonNegativity
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field

/-!
# Softmax Partition of Unity

Proves that softmax outputs sum to 1 for all real inputs.

## Obligation

`SM-INV-001`: ∀ x ∈ ℝⁿ⁺¹, Σᵢ softmax(xᵢ) = 1

This is the flagship Phase 7 proof. The key insight is that
the sum telescopes: Σᵢ exp(xᵢ)/Z = Z/Z = 1 where Z = Σⱼ exp(xⱼ).
-/

namespace ProvableContracts.Softmax

open Real Finset

-- Status: proved
/-- Softmax outputs sum to 1: Σᵢ exp(xᵢ)/Z = Z/Z = 1. -/
theorem partition_of_unity {n : ℕ} (x : RVec (n + 1)) :
    ∑ i : Fin (n + 1), softmax x i = 1 := by
  simp only [softmax]
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (sum_exp_pos x))

-- Tests
#check @partition_of_unity

end ProvableContracts.Softmax
