import ProvableContracts.Defs.Sigmoid
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# SiLU Global Lower Bound

Proves the analytic lower bound `SiLU(x) > -1` for all `x ∈ ℝ`.

## Obligation

`SI-BND-001`: SiLU(x) > -1 for all x.

The true global minimum of SiLU is ≈ -0.2784 at x ≈ -1.278, whose exact
value requires solving a transcendental stationarity equation. The clean,
elementary analytic bound proved here is `SiLU(x) > -1`, obtained from the
convexity witness `x + 1 + exp(-x) ≥ 2 > 0`:

  SiLU(x) = x / (1 + exp(-x)),  and  -1 < x/(1+exp(-x)) ⟺ x + 1 + exp(-x) > 0.

By `add_one_le_exp` we have `exp(-x) ≥ 1 - x`, hence `x + 1 + exp(-x) ≥ 2`.
The tight empirical bound `> -0.279` is retained as a runtime falsification
test (FALSIFY-SI-002).
-/

namespace ProvableContracts.Sigmoid

open Real

-- Status: proved
/-- SiLU is bounded below by -1 everywhere: `SiLU(x) > -1`. -/
theorem silu_gt_neg_one (x : ℝ) : silu x > -1 := by
  unfold silu sigmoid
  have hpos : (0:ℝ) < 1 + Real.exp (-x) := by linarith [Real.exp_pos (-x)]
  have ht : -x + 1 ≤ Real.exp (-x) := Real.add_one_le_exp (-x)
  rw [gt_iff_lt, mul_one_div, lt_div_iff₀ hpos]
  linarith

-- Tests
#check @silu_gt_neg_one

example : silu 0 > -1 := silu_gt_neg_one 0
example : silu (-1.278) > -1 := silu_gt_neg_one (-1.278)

end ProvableContracts.Sigmoid
