import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Inverted Dropout — analytic obligations

Contract: `dropout-v1`, equations `dropout_train` / `dropout_eval`.

Inverted ("train-time scaled") dropout multiplies each unit by a Bernoulli mask
and rescales survivors by `1/(1-p)`:

  `y_i = mask_i * x_i / (1 - p)`,  `mask_i ~ Bernoulli(1 - p)`.

The four proof obligations of `dropout-v1` are all analytic (algebraic identity /
inequality / structural cardinality); the only genuinely stochastic ingredient —
*which* elements a particular RNG draw zeroes — is NOT a proof obligation. This
file discharges all four analytic obligations over ℝ, fully proved (no axioms).

* DO-1 `dropout_eval_identity`      — eval mode is the identity (`y = x`).
* DO-2 `dropout_train_unbiased`     — inverted dropout is unbiased: `E[y] = x`
                                      via `E[mask]·x = x` (the headline).
* DO-3 `dropout_shape_preserved`    — an element-wise map preserves length
                                      (output shape = input shape).
* DO-4 `dropout_prob_welldef`       — `p ∈ [0,1)` makes the `1/(1-p)` scale
                                      well-defined (`1 - p > 0`, `1/(1-p) ≥ 1`),
                                      i.e. `p = 1` (division by zero) is excluded.
-/

namespace ProvableContracts.Dropout

/-! ## DO-1 — eval mode is the identity -/

/-- Eval-mode dropout applies no mask and no scaling. -/
def dropout_eval (x : ℝ) : ℝ := x

/-- Eval mode returns its input unchanged. -/
theorem dropout_eval_identity (x : ℝ) : dropout_eval x = x := rfl

/-! ## DO-2 — inverted dropout is unbiased: `E[y] = x`

Per entry the scaled mask takes value `1/(1-p)` with probability `(1-p)` (kept)
and `0` with probability `p` (dropped). Its expectation is therefore
`E[mask_scaled] = (1-p)·(1/(1-p)) + p·0 = 1`, so `E[y] = E[mask_scaled]·x = x`. -/

/-- Expectation of the *scaled* Bernoulli mask value:
    `(1-p)·(1/(1-p)) + p·0`. -/
noncomputable def mask_expectation (p : ℝ) : ℝ :=
  (1 - p) * (1 / (1 - p)) + p * 0

/-- The scaled mask is unbiased: `E[mask] = 1` for `p ≠ 1`. -/
theorem mask_expectation_eq_one (p : ℝ) (hp : p ≠ 1) : mask_expectation p = 1 := by
  have h : 1 - p ≠ 0 := fun hh => hp (by linarith)
  unfold mask_expectation
  rw [mul_zero, add_zero]
  field_simp

/-- **Unbiasedness (headline).** Since `E[mask] = 1`, the per-entry output
    expectation `E[y] = E[mask]·x` equals `x` for every `x` and every valid
    dropout probability `p ≠ 1`. -/
theorem dropout_train_unbiased (x p : ℝ) (hp : p ≠ 1) :
    mask_expectation p * x = x := by
  rw [mask_expectation_eq_one p hp, one_mul]

/-- `p = 0` is the identity limit: no unit is dropped and the scale is `1`. -/
theorem dropout_train_p_zero_identity (x : ℝ) : mask_expectation 0 * x = x := by
  simpa using dropout_train_unbiased x 0 (by norm_num)

/-! ## DO-3 — output shape preserved

Dropout is applied element-wise, i.e. it is a `List.map` of the per-entry
transform over the input. `List.map` preserves length, so the output shape
(here modelled as the vector length) equals the input shape. -/

/-- An element-wise dropout pass over a vector, modelled as `List.map`. -/
def dropout_apply (f : ℝ → ℝ) (xs : List ℝ) : List ℝ := xs.map f

/-- Output shape equals input shape (length is preserved by the element-wise
    map — holds identically in train and eval modes). -/
theorem dropout_shape_preserved (f : ℝ → ℝ) (xs : List ℝ) :
    (dropout_apply f xs).length = xs.length := by
  unfold dropout_apply
  simp

/-! ## DO-4 — drop probability is in the valid range `[0,1)`

The obligation excludes `p = 1` (which would divide by zero). We prove the
positive analytic content: for `p ∈ [0,1)` the denominator `1 - p` is strictly
positive (scale factor well-defined) and the inverted-dropout scale
`1/(1-p) ≥ 1`. -/

/-- Denominator positivity: `p < 1 ⟹ 0 < 1 - p`, so `1/(1-p)` is well-defined
    (no division by zero). -/
theorem dropout_denominator_pos (p : ℝ) (h1 : p < 1) : 0 < 1 - p := by linarith

/-- The inverted-dropout scale factor is at least `1` on the valid range
    `0 ≤ p < 1`. -/
theorem dropout_scale_ge_one (p : ℝ) (h0 : 0 ≤ p) (h1 : p < 1) :
    1 ≤ 1 / (1 - p) := by
  have hpos : 0 < 1 - p := by linarith
  rw [le_div_iff₀ hpos]
  linarith

#check @dropout_eval_identity
#check @dropout_train_unbiased
#check @dropout_shape_preserved
#check @dropout_denominator_pos
#check @dropout_scale_ge_one

end ProvableContracts.Dropout
