import ProvableContracts.Defs.CMAES
import ProvableContracts.Theorems.Cholesky.SPD
import Mathlib.LinearAlgebra.Matrix.DotProduct

/-!
# CMA-ES Covariance Positive-Definiteness

Proves the analytic core of obligation **CMA-INV-003** (`eigenvalues(C) > 0 at
every generation`) at the quadratic-form level, which is equivalent to positive
definiteness: for every `v ≠ 0`,

    vᵀ C_{t+1} v = a·(vᵀ C_t v) + b·(vᵀ P Pᵀ v) + c·(vᵀ Q Qᵀ v) > 0

whenever `a = 1 − c₁ − c_μ > 0`, `b = c₁ ≥ 0`, `c = c_μ ≥ 0`, and `C_t` is
positive definite (so `vᵀ C_t v > 0`). The two Gram terms `P Pᵀ`, `Q Qᵀ`
contribute `vᵀ(·)v ≥ 0` (reusing the Cholesky PSD lemma), so a strictly
positive `a`-weighted PD term plus nonnegative rank-one / rank-mu terms is
strictly positive. This is the exact convexity argument that keeps `C`
positive definite across generations.
-/

namespace ProvableContracts.CMAES

open Matrix

-- Status: proved
/-- The covariance-update quadratic form is strictly positive for a fixed test
    vector `v` on which `C_t` is positive: a convex-combination of a PD term
    (coefficient `a > 0`) and two PSD Gram terms (coefficients `b, c ≥ 0`)
    stays strictly positive. -/
theorem covUpdate_quadForm_pos {n : ℕ} (a b c : ℝ)
    (ha : 0 < a) (hb : 0 ≤ b) (hc : 0 ≤ c)
    (C P Q : Matrix (Fin n) (Fin n) ℝ) (v : Fin n → ℝ)
    (hC : 0 < dotProduct v (C.mulVec v)) :
    0 < dotProduct v ((covUpdate a b c C P Q).mulVec v) := by
  have hP : 0 ≤ dotProduct v ((P * Pᵀ).mulVec v) :=
    ProvableContracts.Cholesky.cholesky_product_psd P v
  have hQ : 0 ≤ dotProduct v ((Q * Qᵀ).mulVec v) :=
    ProvableContracts.Cholesky.cholesky_product_psd Q v
  simp only [covUpdate, Matrix.add_mulVec, Matrix.smul_mulVec,
    dotProduct_add, dotProduct_smul, smul_eq_mul]
  have h1 : 0 < a * dotProduct v (C.mulVec v) := mul_pos ha hC
  have h2 : 0 ≤ b * dotProduct v ((P * Pᵀ).mulVec v) := mul_nonneg hb hP
  have h3 : 0 ≤ c * dotProduct v ((Q * Qᵀ).mulVec v) := mul_nonneg hc hQ
  exact add_pos_of_pos_of_nonneg (add_pos_of_pos_of_nonneg h1 h2) h3

-- Tests
#check @covUpdate_quadForm_pos

end ProvableContracts.CMAES
