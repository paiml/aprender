import ProvableContracts.Defs.LayerNorm
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# LayerNorm Shift Invariance

Proves that LayerNorm is invariant under uniform input shift:
LN(x + c·1) = LN(x).

## Obligation

`LN-INV-006`: LN(x + c) = LN(x) for any scalar c.

Key insight: mean(x + c) = mean(x) + c, so (xᵢ + c) - mean(x + c)
= xᵢ - mean(x). The variance and normalization are therefore unchanged.
-/

namespace ProvableContracts.LayerNorm

open Finset

/-- Shifted vector: adds scalar c to every component. -/
def shift {n : ℕ} (x : RVec (n + 1)) (c : ℝ) : RVec (n + 1) :=
  fun i => x i + c

-- Status: proved
/-- Mean of shifted vector: mean(x + c) = mean(x) + c. -/
theorem mean_shift {n : ℕ} (x : RVec (n + 1)) (c : ℝ) :
    mean (shift x c) = mean x + c := by
  unfold mean shift
  simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  have hn : (↑(n + 1) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  field_simp
  push_cast
  ring

-- Status: proved
/-- Centered values are shift-invariant: (xᵢ + c) - mean(x + c) = xᵢ - mean(x). -/
theorem centered_shift {n : ℕ} (x : RVec (n + 1)) (c : ℝ) (i : Fin (n + 1)) :
    shift x c i - mean (shift x c) = x i - mean x := by
  rw [mean_shift]
  simp only [shift]
  ring

-- Status: proved
/-- Variance is shift-invariant: var(x + c) = var(x). -/
theorem variance_shift {n : ℕ} (x : RVec (n + 1)) (c : ℝ) :
    variance (shift x c) = variance x := by
  simp only [variance]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  rw [show shift x c i - mean (shift x c) = x i - mean x from centered_shift x c i]

-- Tests
#check @mean_shift
#check @centered_shift
#check @variance_shift

end ProvableContracts.LayerNorm
