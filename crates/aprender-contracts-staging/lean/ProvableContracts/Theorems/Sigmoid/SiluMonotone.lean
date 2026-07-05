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

/-!
# Sigmoid & SiLU Monotonicity

Proves the monotonicity invariants underlying the SwiGLU gate:

- `σ` is monotone nondecreasing on all of ℝ.
- `SiLU(x) = x·σ(x)` is strictly increasing on `[0, ∞)`.

## Obligation

`SG-MONO-001`: "SiLU is monotonic for `x > 0`" (contract equation `silu`,
`kernel_structure.silu_activation`). Increasing the gate pre-activation
(when already nonnegative) increases the gate output, matching the
falsification test `FALSIFY-SG-006`.

## Proof sketch

`σ(x) = 1/(1 + exp(-x))`; for `a ≤ b`, `exp(-b) ≤ exp(-a)` so the
denominator shrinks and the reciprocal grows — hence `σ` is monotone.
For `0 ≤ a < b`, both factors of `SiLU` are nonneg/positive and increasing:
`a·σ(a) ≤ a·σ(b) < b·σ(b)`.
-/

namespace ProvableContracts.Sigmoid

open Real

-- Status: proved
/-- Sigmoid is monotone nondecreasing: `a ≤ b → σ(a) ≤ σ(b)`. -/
theorem sigmoid_mono {a b : ℝ} (h : a ≤ b) : sigmoid a ≤ sigmoid b := by
  unfold sigmoid
  have hdb : (0:ℝ) < 1 + Real.exp (-b) := by positivity
  have hexp : Real.exp (-b) ≤ Real.exp (-a) := Real.exp_le_exp.mpr (by linarith)
  have hle : 1 + Real.exp (-b) ≤ 1 + Real.exp (-a) := by linarith
  exact one_div_le_one_div_of_le hdb hle

-- Status: proved
/-- SiLU is strictly increasing on the nonnegative reals:
    `0 ≤ a → a < b → SiLU(a) < SiLU(b)`. -/
theorem silu_mono_nonneg {a b : ℝ} (ha : 0 ≤ a) (hab : a < b) :
    silu a < silu b := by
  unfold silu
  have hsb : 0 < sigmoid b := sigmoid_pos b
  have hmono : sigmoid a ≤ sigmoid b := sigmoid_mono (le_of_lt hab)
  have step1 : a * sigmoid a ≤ a * sigmoid b := mul_le_mul_of_nonneg_left hmono ha
  have step2 : a * sigmoid b < b * sigmoid b := mul_lt_mul_of_pos_right hab hsb
  linarith

-- Tests
#check @sigmoid_mono
#check @silu_mono_nonneg

example : sigmoid 0 ≤ sigmoid 1 := sigmoid_mono (by norm_num)
example : silu 1 < silu 2 := silu_mono_nonneg (by norm_num) (by norm_num)

end ProvableContracts.Sigmoid
