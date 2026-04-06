import ProvableContracts.Defs.Softmax
import ProvableContracts.Theorems.Softmax.PartitionOfUnity
import ProvableContracts.Theorems.Softmax.NonNegativity
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Softmax Bounded

Proves that softmax outputs lie strictly in (0, 1).

## Obligation

`SM-BND-001`: ∀ x ∈ ℝⁿ⁺¹, 0 < softmax(x)ᵢ < 1

The lower bound follows from `softmax_pos`. The upper bound follows
from `partition_of_unity`: since all terms are positive and sum to 1,
each term must be strictly less than 1.
-/

namespace ProvableContracts.Softmax

open Real Finset

-- Status: proved
/-- Each softmax output is strictly less than 1.
    Since n+1 ≥ 2, there exists j ≠ i with softmax(x)_j > 0,
    so softmax(x)_i = 1 - Σ_{k≠i} softmax(x)_k < 1. -/
theorem softmax_lt_one {n : ℕ} (x : RVec (n + 2)) (i : Fin (n + 2)) :
    softmax x i < 1 := by
  have hsum := partition_of_unity x
  have hpos := softmax_pos x
  -- n+2 ≥ 2, so there exists j ≠ i
  obtain ⟨j, hji⟩ : ∃ j : Fin (n + 2), j ≠ i := exists_ne i
  rw [show (1 : ℝ) = ∑ k : Fin (n + 2), softmax x k from hsum.symm]
  exact Finset.single_lt_sum hji (Finset.mem_univ i) (Finset.mem_univ j)
    (hpos j) (fun k _ _ => le_of_lt (hpos k))

/-- Softmax outputs are strictly bounded in (0, 1). -/
theorem softmax_bounded {n : ℕ} (x : RVec (n + 2)) (i : Fin (n + 2)) :
    0 < softmax x i ∧ softmax x i < 1 :=
  ⟨softmax_pos x i, softmax_lt_one x i⟩

-- Tests
#check @softmax_lt_one
#check @softmax_bounded

end ProvableContracts.Softmax
