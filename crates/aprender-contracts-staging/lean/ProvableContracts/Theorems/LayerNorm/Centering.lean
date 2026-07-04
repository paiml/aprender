import ProvableContracts.Defs.LayerNorm
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# LayerNorm Centering

Proves that with `gamma = 1`, the mean of the LayerNorm output equals
`mean(beta)`:

    mean(LN(x)) = mean(beta)  when gamma = 1.

## Obligation

`LN-INV-001` (Centering): `|mean(LN(x)) - mean(beta)| < eps when gamma = 1`.

Key insight: the centered values `xᵢ - μ` sum to zero, so the
normalized contribution to the mean vanishes exactly, leaving only
`mean(beta)`. This holds for every ε (the ε only affects the common
denominator, which is annihilated by the zero numerator).
-/

namespace ProvableContracts.LayerNorm

open Finset

-- Status: proved
/-- The centered values sum to zero: `Σᵢ (xᵢ - μ) = 0`. -/
theorem sum_sub_mean_zero {n : ℕ} (x : RVec (n + 1)) :
    univ.sum (fun i => x i - mean x) = 0 := by
  rw [Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  unfold mean
  have hn : (↑(n + 1) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  field_simp
  push_cast
  ring

-- Status: proved
/-- Centering: with `gamma = 1`, `mean(LN(x)) = mean(beta)`. -/
theorem mean_layernorm_centering {n : ℕ} (x beta : RVec (n + 1)) (eps : ℝ) :
    mean (layernorm x (fun _ => 1) beta eps) = mean beta := by
  unfold mean layernorm
  simp only [one_mul]
  congr 1
  rw [Finset.sum_add_distrib]
  have h0 : univ.sum (fun i => (x i - mean x) / ln_denom x eps) = 0 := by
    simp only [div_eq_mul_inv, ← Finset.sum_mul, sum_sub_mean_zero, zero_mul]
  rw [h0, zero_add]

-- Tests
#check @sum_sub_mean_zero
#check @mean_layernorm_centering

end ProvableContracts.LayerNorm
