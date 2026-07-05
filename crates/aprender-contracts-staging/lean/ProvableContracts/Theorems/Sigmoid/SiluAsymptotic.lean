import ProvableContracts.Defs.Sigmoid
import ProvableContracts.Theorems.Sigmoid.SigmoidBounded
import ProvableContracts.Theorems.Sigmoid.SigmoidSymmetry
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# SiLU Asymptotic Linearity

Proves the analytic core of "SiLU(x) → x as x → +∞": for positive `x`,
SiLU stays below the identity and the gap decays at least as fast as
`x · exp(-x)`.

## Obligation

`SI-ASY-001`: for x > 0,  0 < x - SiLU(x) < x · exp(-x)

Since `x - SiLU(x) = x · (1 - σ(x)) = x · σ(-x)` and `σ(-x) < exp(-x)`,
the gap is squeezed by `x · exp(-x) → 0`. The tight numeric instance
`|SiLU(x) - x| < 0.01 for x > 10` is retained as a runtime falsification
test (FALSIFY-SI-005).
-/

namespace ProvableContracts.Sigmoid

open Real

-- Status: proved
/-- Sigmoid is dominated by `exp`: `σ(x) < exp(x)` for all x, because
    `exp(x) · (1 + exp(-x)) = exp(x) + 1 > 1`. -/
theorem sigmoid_lt_exp (x : ℝ) : sigmoid x < Real.exp x := by
  unfold sigmoid
  have hpos : (0:ℝ) < 1 + Real.exp (-x) := by linarith [Real.exp_pos (-x)]
  rw [div_lt_iff₀ hpos]
  have hmul : Real.exp x * (1 + Real.exp (-x)) = Real.exp x + 1 := by
    rw [mul_add, mul_one, ← Real.exp_add, add_neg_cancel, Real.exp_zero]
  rw [hmul]
  linarith [Real.exp_pos x]

-- Status: proved
/-- For positive inputs SiLU lies strictly below the identity: `SiLU(x) < x`. -/
theorem silu_lt_self {x : ℝ} (hx : 0 < x) : silu x < x := by
  unfold silu
  exact mul_lt_of_lt_one_right hx (sigmoid_lt_one x)

-- Status: proved
/-- Asymptotic linearity gap bound: for `x > 0`, the gap `x - SiLU(x)` is
    positive and strictly below `x · exp(-x)`, which vanishes as `x → +∞`. -/
theorem silu_gap_bound {x : ℝ} (hx : 0 < x) :
    0 < x - silu x ∧ x - silu x < x * Real.exp (-x) := by
  refine ⟨by linarith [silu_lt_self hx], ?_⟩
  unfold silu
  have h1 : 1 - sigmoid x < Real.exp (-x) := by
    rw [← sigmoid_symmetry x]
    exact sigmoid_lt_exp (-x)
  have hrw : x - x * sigmoid x = x * (1 - sigmoid x) := by ring
  rw [hrw]
  exact mul_lt_mul_of_pos_left h1 hx

-- Tests
#check @sigmoid_lt_exp
#check @silu_lt_self
#check @silu_gap_bound

end ProvableContracts.Sigmoid
