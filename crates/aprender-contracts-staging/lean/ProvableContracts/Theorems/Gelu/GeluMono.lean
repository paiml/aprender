import ProvableContracts.Defs.Gelu
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

/-!
# GELU Monotonicity for Positive Inputs

Proves the `monotonicity` obligation of `gelu-kernel-v1.yaml`:

  `GE-MON-001`: `x > y > 0 → GELU(x) > GELU(y)`.

## Mechanism

Write the tanh-approximation GELU as a product of two factors

  `GELU(t) = (0.5·t) · (1 + tanh g(t))`,  `g(t) = √(2/π)·(t + 0.044715·t³)`.

For `0 < y < x` **both** factors are positive and strictly increase:

* left factor `0.5·t` : strictly increasing, and `> 0` since `t > 0`;
* inner `g` : strictly increasing because `√(2/π) > 0` and `t + 0.044715·t³`
  is strictly increasing (`t ↦ t³` is strictly monotone via `Odd.pow_lt_pow`);
* `tanh` is strictly increasing (`tanh_strictMono`, proved below from
  `sinh (β−α) > 0`), so the right factor `1 + tanh g(t)` strictly increases and
  is `> 0` (as `tanh > −1`).

A product of two positive, strictly-increasing factors strictly increases:
`(0.5·y)·R_y < (0.5·x)·R_y < (0.5·x)·R_x`.
-/

namespace ProvableContracts.Gelu

open Real

/-- `Real.tanh` is strictly monotone.

Derived from `tanh = sinh / cosh` (positive denominator) and
`sinh (β − α) = sinh β · cosh α − cosh β · sinh α > 0` for `α < β`. -/
theorem tanh_strictMono : StrictMono Real.tanh := by
  intro a b hab
  rw [Real.tanh_eq_sinh_div_cosh, Real.tanh_eq_sinh_div_cosh,
    div_lt_div_iff₀ (Real.cosh_pos a) (Real.cosh_pos b)]
  -- goal: sinh a * cosh b < sinh b * cosh a
  have hpos : 0 < Real.sinh (b - a) := Real.sinh_pos_iff.mpr (by linarith)
  rw [Real.sinh_sub] at hpos
  nlinarith [hpos]

-- Status: proved
/-- **GE-MON-001 / positive-input monotonicity.**
    `GELU(y) < GELU(x)` whenever `0 < y < x`. -/
theorem gelu_strictMono_of_pos {x y : ℝ} (hy : 0 < y) (hyx : y < x) :
    gelu y < gelu x := by
  -- inner monotonicity: g y < g x
  have hsqrt : 0 < Real.sqrt (2 / Real.pi) := Real.sqrt_pos.mpr (by positivity)
  have hcube : y ^ 3 < x ^ 3 := (Odd.pow_lt_pow (by decide : Odd 3)).mpr hyx
  have hinner : y + 0.044715 * y ^ 3 < x + 0.044715 * x ^ 3 := by nlinarith [hcube, hyx]
  have hg : Real.sqrt (2 / Real.pi) * (y + 0.044715 * y ^ 3)
          < Real.sqrt (2 / Real.pi) * (x + 0.044715 * x ^ 3) :=
    mul_lt_mul_of_pos_left hinner hsqrt
  -- right factor: strictly increasing and positive
  have htanh : Real.tanh (Real.sqrt (2 / Real.pi) * (y + 0.044715 * y ^ 3))
             < Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x ^ 3)) :=
    tanh_strictMono hg
  have hRy_pos : 0 < 1 + Real.tanh (Real.sqrt (2 / Real.pi) * (y + 0.044715 * y ^ 3)) := by
    have := Real.neg_one_lt_tanh (Real.sqrt (2 / Real.pi) * (y + 0.044715 * y ^ 3)); linarith
  have hR : 1 + Real.tanh (Real.sqrt (2 / Real.pi) * (y + 0.044715 * y ^ 3))
          < 1 + Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x ^ 3)) := by linarith
  -- left factor: strictly increasing and positive
  have hLx_pos : 0 < 0.5 * x := by linarith
  have hLyx : 0.5 * y < 0.5 * x := by linarith
  unfold gelu
  -- (0.5·y)·R_y < (0.5·x)·R_y < (0.5·x)·R_x
  have step1 : 0.5 * y * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (y + 0.044715 * y ^ 3)))
             < 0.5 * x * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (y + 0.044715 * y ^ 3))) :=
    mul_lt_mul_of_pos_right hLyx hRy_pos
  have step2 : 0.5 * x * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (y + 0.044715 * y ^ 3)))
             < 0.5 * x * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x ^ 3))) :=
    mul_lt_mul_of_pos_left hR hLx_pos
  linarith [step1, step2]

-- Tests
#check @tanh_strictMono
#check @gelu_strictMono_of_pos

example {x y : ℝ} (hy : 0 < y) (hyx : y < x) : gelu y < gelu x :=
  gelu_strictMono_of_pos hy hyx

end ProvableContracts.Gelu
