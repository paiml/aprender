import ProvableContracts.Defs.Cholesky
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.DotProduct
import Mathlib.Data.Real.StarOrdered

/-!
# Cholesky Decomposition: SPD Structural Properties

We prove the key structural properties that justify Cholesky:
1. L * Lᵀ is always symmetric for any matrix L.
2. The quadratic form xᵀ(LLᵀ)x = ‖Lᵀx‖² ≥ 0 (positive semidefiniteness).

## Obligation

`CHOL-SYM-001`: L * Lᵀ is symmetric.
`CHOL-PSD-001`: xᵀ (L Lᵀ) x ≥ 0 for all x.
-/

namespace ProvableContracts.Cholesky

open Matrix

-- Status: proved
/-- L * Lᵀ is always symmetric, for any matrix L. -/
theorem cholesky_product_symmetric {n : ℕ}
    (L : Matrix (Fin n) (Fin n) ℝ) :
    (L * Lᵀ)ᵀ = L * Lᵀ := by
  simp [Matrix.transpose_mul, Matrix.transpose_transpose]

-- Status: proved
/-- The quadratic form xᵀ(LLᵀ)x = ‖Lᵀx‖² ≥ 0.
    This proves LLᵀ is positive semidefinite. -/
theorem cholesky_product_psd {n : ℕ}
    (L : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) :
    0 ≤ dotProduct x ((L * Lᵀ).mulVec x) := by
  rw [← mulVec_mulVec, dotProduct_mulVec, ← mulVec_transpose]
  exact dotProduct_self_star_nonneg _

-- Tests
#check @cholesky_product_symmetric
#check @cholesky_product_psd

end ProvableContracts.Cholesky
