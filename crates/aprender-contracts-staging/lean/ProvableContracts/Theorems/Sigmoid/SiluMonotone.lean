import ProvableContracts.Defs.Sigmoid
import ProvableContracts.Theorems.Sigmoid.SigmoidBounded
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# SiLU Positive Monotonicity

Proves that SiLU is strictly increasing on the positive reals:
`0 < y < x → SiLU(y) < SiLU(x)`.

## Obligation

`SI-MON-001`: 0 < y < x → SiLU(y) < SiLU(x)

No derivatives are needed. The sigmoid is globally strictly increasing
(smaller `1 + exp(-·)` denominator for larger argument), and for positive
arguments both factors of `SiLU(x) = x · σ(x)` are positive and increasing,
so the product is strictly increasing (`mul_lt_mul''`).
-/

namespace ProvableContracts.Sigmoid

open Real

-- Status: proved
/-- Sigmoid is globally strictly increasing. -/
theorem sigmoid_strictMono {x y : ℝ} (h : x < y) : sigmoid x < sigmoid y := by
  unfold sigmoid
  have hy : (0:ℝ) < 1 + Real.exp (-y) := by linarith [Real.exp_pos (-y)]
  have hexp : Real.exp (-y) < Real.exp (-x) := by
    apply Real.exp_lt_exp.mpr; linarith
  exact one_div_lt_one_div_of_lt hy (by linarith)

-- Status: proved
/-- SiLU is strictly increasing on the positive reals. -/
theorem silu_strictMono_pos {x y : ℝ} (hx : 0 < x) (hxy : x < y) :
    silu x < silu y := by
  unfold silu
  exact mul_lt_mul'' hxy (sigmoid_strictMono hxy) (le_of_lt hx)
    (le_of_lt (sigmoid_pos x))

-- Tests
#check @sigmoid_strictMono
#check @silu_strictMono_pos

end ProvableContracts.Sigmoid
