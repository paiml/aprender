import ProvableContracts.Defs.Softmax
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# Softmax Shift Invariance

Proves that softmax is invariant under translation: σ(x + c·1) = σ(x).

## Obligation

`SM-INV-006`: ∀ x ∈ ℝⁿ⁺¹, ∀ c ∈ ℝ, softmax(x + c) = softmax(x)

This is the key property for numerical stability: subtracting max(x)
does not change the output. The proof is algebraic: exp(xᵢ+c)/Σexp(xⱼ+c)
= exp(xᵢ)·exp(c) / (exp(c)·Σexp(xⱼ)) = exp(xᵢ)/Σexp(xⱼ).
-/

namespace ProvableContracts.Softmax

open Real Finset

/-- Shifted input vector: adds scalar c to every component. -/
noncomputable def shift {n : ℕ} (x : RVec n) (c : ℝ) : RVec n :=
  fun i => x i + c

-- Status: proved
/-- Softmax is invariant under uniform translation.
    exp(xᵢ+c) / Σⱼ exp(xⱼ+c) = exp(xᵢ)·e^c / (e^c · Σⱼ exp(xⱼ))
    = exp(xᵢ) / Σⱼ exp(xⱼ). -/
theorem shift_invariance {n : ℕ} (x : RVec (n + 1)) (c : ℝ) (i : Fin (n + 1)) :
    softmax (shift x c) i = softmax x i := by
  simp only [softmax, shift, Real.exp_add]
  rw [← Finset.sum_mul]
  exact mul_div_mul_right _ _ (ne_of_gt (Real.exp_pos c))

-- Tests
#check @shift_invariance

end ProvableContracts.Softmax
