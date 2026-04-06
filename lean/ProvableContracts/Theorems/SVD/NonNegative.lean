import ProvableContracts.Defs.SVD
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.DotProduct
import Mathlib.Data.Real.StarOrdered

/-!
# SVD Non-Negative Singular Values

Proves that singular values — defined as square roots of eigenvalues
of AᵀA — are non-negative. The key insight is that AᵀA is positive
semidefinite (xᵀAᵀAx = ‖Ax‖² ≥ 0), so its eigenvalues are non-negative,
and square roots of non-negative numbers are non-negative.

## Obligation

`SVD-NN-001`: xᵀ(AᵀA)x ≥ 0 for all x (AᵀA is PSD).
`SVD-NN-002`: AᵀA is symmetric.
-/

namespace ProvableContracts.SVD

open Matrix

-- Status: proved
/-- AᵀA is symmetric: (AᵀA)ᵀ = AᵀA. -/
theorem ata_symmetric {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) :
    (Aᵀ * A)ᵀ = Aᵀ * A := by
  simp [Matrix.transpose_mul, Matrix.transpose_transpose]

-- Status: proved
/-- The Gram matrix AᵀA is positive semidefinite:
    xᵀ(AᵀA)x = ‖Ax‖² ≥ 0. -/
theorem ata_psd {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ) :
    0 ≤ dotProduct x ((Aᵀ * A).mulVec x) := by
  rw [← mulVec_mulVec, dotProduct_mulVec, vecMul_transpose]
  exact dotProduct_self_star_nonneg _

-- Tests
#check @ata_symmetric
#check @ata_psd

end ProvableContracts.SVD
