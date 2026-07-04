import ProvableContracts.Theorems.LayerNorm.Centering
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Data.Real.Sqrt

/-!
# LayerNorm Standardization

Proves that with `gamma = 1, beta = 0` the LayerNorm output has

    var(LN(x)) = variance(x) / (variance(x) + ε)          (exact ratio)

and, in the ideal (ε = 0) regime with non-constant input,

    var(LN(x)) = 1.                                        (standardization)

## Obligation

`LN-INV-002` (Standardization): `|var(LN(x)) - 1.0| < eps when gamma = 1,
beta = 0`.

Key insight: the output has mean 0 (Centering with β = 0), so its
variance is `(1/d)·Σ((xᵢ-μ)/denom)²  = variance(x)/denom²`, and
`denom² = variance(x) + ε`. When `ε = 0` and `variance(x) > 0`, the ratio
is exactly 1.
-/

namespace ProvableContracts.LayerNorm

open Finset

-- Status: proved
/-- Exact variance ratio of the normalized output (γ = 1, β = 0):
`var(LN(x)) = variance(x) / (variance(x) + ε)`. -/
theorem variance_layernorm_ratio {n : ℕ} (x : RVec (n + 1)) (eps : ℝ)
    (h : variance x + eps ≥ 0) :
    variance (layernorm x (fun _ => 1) (fun _ => 0) eps)
      = variance x / (variance x + eps) := by
  -- Output mean is 0 (centering with β = 0).
  have hmean : mean (layernorm x (fun _ => 1) (fun _ => 0) eps) = 0 := by
    rw [mean_layernorm_centering]
    unfold mean
    simp
  -- denom² = variance x + ε.
  have hdenom : (ln_denom x eps) ^ 2 = variance x + eps := by
    unfold ln_denom
    exact Real.sq_sqrt h
  -- Pointwise form of the normalized output.
  have hLi : ∀ i, layernorm x (fun _ => 1) (fun _ => 0) eps i
      = (x i - mean x) / ln_denom x eps := by
    intro i
    simp [layernorm]
  -- Expand the variance of the output.
  unfold variance
  rw [hmean]
  simp only [sub_zero, hLi, div_pow]
  rw [← hdenom, ← Finset.sum_div]
  -- goal: (Σ (xᵢ-μ)² / denom²) / (n+1) = ((Σ (xᵢ-μ)²)/(n+1)) / denom²
  rw [div_div, div_div, mul_comm ((ln_denom x eps) ^ 2) (↑(n + 1) : ℝ)]

-- Status: proved
/-- Standardization: in the ideal ε = 0 regime with non-constant input
(`variance(x) > 0`), the normalized output has unit variance. -/
theorem variance_layernorm_standardized {n : ℕ} (x : RVec (n + 1))
    (hx : variance x > 0) :
    variance (layernorm x (fun _ => 1) (fun _ => 0) 0) = 1 := by
  rw [variance_layernorm_ratio x 0 (by linarith)]
  rw [add_zero, div_self (ne_of_gt hx)]

-- Tests
#check @variance_layernorm_ratio
#check @variance_layernorm_standardized

end ProvableContracts.LayerNorm
