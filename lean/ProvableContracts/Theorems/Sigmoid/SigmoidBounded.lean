import ProvableContracts.Defs.Sigmoid
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Sigmoid Bounded

Proves that sigmoid outputs lie strictly in (0, 1).

## Obligation

σ(x) ∈ (0, 1) for all x ∈ ℝ.

Since exp(-x) > 0, the denominator 1 + exp(-x) > 1, so
1/(1 + exp(-x)) < 1. And since the denominator is positive,
the whole expression is positive.
-/

namespace ProvableContracts.Sigmoid

open Real

-- Status: proved
/-- Sigmoid is strictly positive: 1/(1 + exp(-x)) > 0. -/
theorem sigmoid_pos (x : ℝ) : sigmoid x > 0 := by
  unfold sigmoid
  apply div_pos one_pos
  linarith [Real.exp_pos (-x)]

-- Status: proved
/-- Sigmoid is strictly less than 1: 1/(1 + exp(-x)) < 1.
    The denominator exceeds 1 because exp(-x) > 0. -/
theorem sigmoid_lt_one (x : ℝ) : sigmoid x < 1 := by
  unfold sigmoid
  rw [div_lt_one (by linarith [Real.exp_pos (-x)])]
  linarith [Real.exp_pos (-x)]

-- Status: proved
/-- Sigmoid outputs are strictly bounded in (0, 1). -/
theorem sigmoid_bounded (x : ℝ) : 0 < sigmoid x ∧ sigmoid x < 1 :=
  ⟨sigmoid_pos x, sigmoid_lt_one x⟩

-- Tests
#check @sigmoid_pos
#check @sigmoid_lt_one
#check @sigmoid_bounded

example : sigmoid 0 > 0 := sigmoid_pos 0
example : sigmoid 0 < 1 := sigmoid_lt_one 0
example : sigmoid 100 > 0 := sigmoid_pos 100
example : sigmoid (-100) < 1 := sigmoid_lt_one (-100)

end ProvableContracts.Sigmoid
