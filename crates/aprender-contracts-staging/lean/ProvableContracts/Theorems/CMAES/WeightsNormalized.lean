import ProvableContracts.Defs.CMAES
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# CMA-ES Weight Normalization

Proves the analytic core of obligation **CMA-INV-001** (`|Σ wᵢ − 1| < ε`):
the recombination weights `wᵢ = raw i / Σⱼ raw j` sum **exactly** to 1
whenever the raw weight total is nonzero. This is an algebraic identity, so
the tolerance `ε` in the runtime falsifier is only floating-point slack — the
exact value is `1`.

We additionally show each weight is a genuine convex coefficient
(`0 ≤ wᵢ`) when the raw weights are nonnegative, establishing that the mean
update is a convex combination.
-/

namespace ProvableContracts.CMAES

open Finset

-- Status: proved
/-- The normalized recombination weights sum exactly to 1. -/
theorem weights_sum_one {n : ℕ} (raw : Fin n → ℝ)
    (h : (∑ j, raw j) ≠ 0) :
    ∑ i, normalizedWeights raw i = 1 := by
  unfold normalizedWeights
  rw [← Finset.sum_div, div_self h]

-- Status: proved
/-- Each normalized weight is nonnegative when raw weights are nonnegative and
    the total is positive — so the mean update is a genuine convex combination. -/
theorem weights_nonneg {n : ℕ} (raw : Fin n → ℝ) (i : Fin n)
    (hraw : ∀ j, 0 ≤ raw j) (hpos : 0 < ∑ j, raw j) :
    0 ≤ normalizedWeights raw i := by
  unfold normalizedWeights
  exact div_nonneg (hraw i) (le_of_lt hpos)

-- Tests
#check @weights_sum_one
#check @weights_nonneg

end ProvableContracts.CMAES
