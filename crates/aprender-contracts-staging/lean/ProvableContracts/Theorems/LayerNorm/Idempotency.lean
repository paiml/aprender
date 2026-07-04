import ProvableContracts.Theorems.LayerNorm.Standardization
import Mathlib.Data.Real.Sqrt

/-!
# LayerNorm Idempotency

Proves that with `gamma = 1, beta = 0` in the ideal (ε = 0) regime,
LayerNorm is idempotent on non-constant input:

    LN(LN(x)) = LN(x)   when variance(x) > 0.

## Obligation

`LN-INV-005` (Idempotency): `|LN(LN(x)) - LN(x)| < eps when gamma = 1,
beta = 0`.

Key insight: `y = LN(x)` already has mean 0 (Centering, β = 0) and unit
variance (Standardization, ε = 0). Re-normalizing a mean-0 unit-variance
vector with `γ = 1, β = 0, ε = 0` subtracts 0 and divides by
`√(1) = 1`, so it acts as the identity.
-/

namespace ProvableContracts.LayerNorm

open Finset

-- Status: proved
/-- A mean-0, unit-variance vector is a fixed point of LayerNorm
(`γ = 1, β = 0, ε = 0`). -/
theorem layernorm_id_of_normalized {n : ℕ} (y : RVec (n + 1))
    (hm : mean y = 0) (hv : variance y = 1) :
    layernorm y (fun _ => 1) (fun _ => 0) 0 = y := by
  have hd : ln_denom y 0 = 1 := by
    unfold ln_denom
    rw [hv, add_zero, Real.sqrt_one]
  funext i
  simp only [layernorm, one_mul, add_zero]
  rw [hm, sub_zero, hd, div_one]

-- Status: proved
/-- Idempotency: with `γ = 1, β = 0, ε = 0` and non-constant input
(`variance(x) > 0`), `LN(LN(x)) = LN(x)`. -/
theorem layernorm_idempotent {n : ℕ} (x : RVec (n + 1)) (hx : variance x > 0) :
    layernorm (layernorm x (fun _ => 1) (fun _ => 0) 0)
      (fun _ => 1) (fun _ => 0) 0
      = layernorm x (fun _ => 1) (fun _ => 0) 0 := by
  apply layernorm_id_of_normalized
  · rw [mean_layernorm_centering]
    unfold mean
    simp
  · exact variance_layernorm_standardized x hx

-- Tests
#check @layernorm_id_of_normalized
#check @layernorm_idempotent

end ProvableContracts.LayerNorm
