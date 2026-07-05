import ProvableContracts.Defs.Sigmoid
import ProvableContracts.Theorems.Sigmoid.SigmoidBounded
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# SiLU / Swish Lower Bound

Proves an elementary, fully-analytic global lower bound for the SiLU
(a.k.a. Swish-1) activation `SiLU(x) = x · σ(x)`:

`SiLU(x) > -1/e   for all x ∈ ℝ`   (`e = exp 1 ≈ 2.71828`).

## Obligation

`SG-BND-001` (Gate output bounded below): the SiLU gate output is bounded
below by a finite constant, so the SwiGLU gate can never diverge to `-∞`.

## Note on the constant

The *tight* global minimum of SiLU sits at the transcendental critical
point `z* ≈ -1.2784` (root of `eᶻ(z-1) = 1`), with value `≈ -0.2785`.
That exact constant is NOT elementarily provable. This file instead
proves the clean, fully-analytic bound `-1/e ≈ -0.3679`, which is a true
lower bound of the same order and requires only `exp x ≥ x + 1`
(`Real.add_one_le_exp`) plus `σ(x) < exp x`. This is the honest analytic
core of the "bounded below" obligation.

## Proof sketch

1. `σ(x) < exp x` for all `x`  (since `exp x · (1 + exp(-x)) = exp x + 1 > 1`).
2. `t · exp t ≥ -1/e` for all `t`  (from `exp(-t-1) ≥ -t`, multiply by `exp(t+1)`).
3. For `x ≥ 0`: `SiLU(x) = x·σ(x) ≥ 0 > -1/e`.
   For `x < 0`: `SiLU(x) = x·σ(x) > x·exp x ≥ -1/e` (multiplying `σ(x)<exp x`
   by the negative `x` flips the inequality).
-/

namespace ProvableContracts.Sigmoid

open Real

-- Status: proved
/-- Sigmoid is dominated by the exponential: `σ(x) < exp x` for all `x`.
    Equivalent to `1 < exp x · (1 + exp(-x)) = exp x + 1`, i.e. `0 < exp x`. -/
theorem sigmoid_lt_exp (x : ℝ) : sigmoid x < Real.exp x := by
  unfold sigmoid
  rw [div_lt_iff₀ (by positivity)]
  have hcollapse : Real.exp x * (1 + Real.exp (-x)) = Real.exp x + 1 := by
    rw [mul_add, mul_one, ← Real.exp_add, add_neg_cancel, Real.exp_zero]
  rw [hcollapse]
  linarith [Real.exp_pos x]

-- Status: proved
/-- The map `t ↦ t·exp t` has global lower bound `-1/e`:
    `-(1 / exp 1) ≤ t · exp t` for all `t`.
    Uses only the tangent-line bound `exp y ≥ y + 1`. -/
theorem mul_exp_ge_neg_inv_e (t : ℝ) : -(1 / Real.exp 1) ≤ t * Real.exp t := by
  have he1 : (0:ℝ) < Real.exp 1 := Real.exp_pos 1
  have key : -t ≤ Real.exp (-t - 1) := by
    have h := Real.add_one_le_exp (-t - 1)
    linarith
  have hpos : (0:ℝ) < Real.exp (t + 1) := Real.exp_pos _
  have hmul : -t * Real.exp (t + 1) ≤ Real.exp (-t - 1) * Real.exp (t + 1) :=
    mul_le_mul_of_nonneg_right key (le_of_lt hpos)
  have hc : Real.exp (-t - 1) * Real.exp (t + 1) = 1 := by
    rw [← Real.exp_add]
    have harg : -t - 1 + (t + 1) = 0 := by ring
    rw [harg, Real.exp_zero]
  rw [hc] at hmul
  have hexpand : t * Real.exp t * Real.exp 1 = t * Real.exp (t + 1) := by
    rw [Real.exp_add]; ring
  have hge : (-1 : ℝ) ≤ t * Real.exp t * Real.exp 1 := by
    rw [hexpand]; linarith
  have hdiv : (-1 : ℝ) / Real.exp 1 ≤ t * Real.exp t := by
    rw [div_le_iff₀ he1]; linarith
  simpa [neg_div] using hdiv

-- Status: proved
/-- SiLU global lower bound: `SiLU(x) > -1/e` for all `x ∈ ℝ`.
    This is the analytic core of the SwiGLU "gate bounded below" obligation. -/
theorem silu_gt_neg_inv_e (x : ℝ) : silu x > -(1 / Real.exp 1) := by
  have hpos : (0:ℝ) < 1 / Real.exp 1 := by positivity
  by_cases hx : 0 ≤ x
  · -- x ≥ 0 : SiLU(x) = x·σ(x) ≥ 0 > -(1/e)
    have hnn : silu x ≥ 0 := by
      unfold silu
      exact mul_nonneg hx (le_of_lt (sigmoid_pos x))
    linarith
  · -- x < 0 : SiLU(x) = x·σ(x) > x·exp x ≥ -(1/e)
    have hx : x < 0 := not_le.mp hx
    have hsig : sigmoid x < Real.exp x := sigmoid_lt_exp x
    have h1 : x * Real.exp x < x * sigmoid x := mul_lt_mul_of_neg_left hsig hx
    have h2 : -(1 / Real.exp 1) ≤ x * Real.exp x := mul_exp_ge_neg_inv_e x
    unfold silu
    linarith

-- Tests
#check @sigmoid_lt_exp
#check @mul_exp_ge_neg_inv_e
#check @silu_gt_neg_inv_e

example (x : ℝ) : silu x > -(1 / Real.exp 1) := silu_gt_neg_inv_e x
example : silu 0 > -(1 / Real.exp 1) := silu_gt_neg_inv_e 0

end ProvableContracts.Sigmoid
