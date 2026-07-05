import ProvableContracts.Defs.Gelu
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# GELU Bounds on Non-Negative Inputs

For the tanh approximation
`GELU(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`,
the `tanh` factor is bounded (`-1 < tanh a < 1`), so on non-negative inputs
GELU is squeezed between `0` and the identity:

  0 ≤ GELU(x) ≤ x    for x ≥ 0.

## Obligations

`GELU-BND-001`: ∀ x ≥ 0, 0 ≤ GELU(x)
`GELU-BND-002`: ∀ x ≥ 0, GELU(x) ≤ x

Both are algebraic consequences of the Mathlib bounds
`Real.neg_one_lt_tanh` and `Real.tanh_lt_one`; no numerical/interval reasoning
about the *approximation error* is involved.

## References

- Hendrycks & Gimpel (2016) Gaussian Error Linear Units (GELUs)
-/

namespace ProvableContracts.Gelu

open Real

-- Status: proved
/-- GELU is non-negative on non-negative inputs: `0 ≤ x ⟹ 0 ≤ gelu x`.
    `0.5·x ≥ 0` and the factor `1 + tanh a > 0` since `tanh a > -1`. -/
theorem gelu_nonneg_of_nonneg (x : ℝ) (hx : 0 ≤ x) : 0 ≤ gelu x := by
  unfold gelu
  set a := Real.sqrt (2 / Real.pi) * (x + 0.044715 * x ^ 3) with ha
  have hfac : (0 : ℝ) < 1 + Real.tanh a := by
    have := Real.neg_one_lt_tanh a
    linarith
  have hhalf : (0 : ℝ) ≤ 0.5 * x := by linarith
  calc (0 : ℝ) = 0.5 * x * 0 := by ring
    _ ≤ 0.5 * x * (1 + Real.tanh a) := by
        exact mul_le_mul_of_nonneg_left (le_of_lt hfac) hhalf

-- Status: proved
/-- GELU is bounded above by the identity on non-negative inputs:
    `0 ≤ x ⟹ gelu x ≤ x`. Since `tanh a < 1`, the factor `1 + tanh a < 2`,
    and `0.5·x·2 = x`. -/
theorem gelu_le_self_of_nonneg (x : ℝ) (hx : 0 ≤ x) : gelu x ≤ x := by
  unfold gelu
  set a := Real.sqrt (2 / Real.pi) * (x + 0.044715 * x ^ 3) with ha
  have hfac : 1 + Real.tanh a ≤ 2 := by
    have := Real.tanh_lt_one a
    linarith
  have hhalf : (0 : ℝ) ≤ 0.5 * x := by linarith
  calc 0.5 * x * (1 + Real.tanh a)
      ≤ 0.5 * x * 2 := mul_le_mul_of_nonneg_left hfac hhalf
    _ = x := by ring

-- Tests
#check @gelu_nonneg_of_nonneg
#check @gelu_le_self_of_nonneg

end ProvableContracts.Gelu
