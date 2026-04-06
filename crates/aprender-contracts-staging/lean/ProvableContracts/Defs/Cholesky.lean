import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.LinearAlgebra.Matrix.PosDef

/-!
# Cholesky Definitions

Definitions for symmetric positive definite (SPD) matrices and
the existence claim of Cholesky decomposition.

## References

- Golub & Van Loan, "Matrix Computations," 4th ed., Section 4.2.
-/

namespace ProvableContracts.Cholesky

open Matrix

/-- A matrix is lower triangular if all entries above the diagonal are zero. -/
def IsLowerTriangular {n : ℕ} (L : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  ∀ i j : Fin n, j.val > i.val → L i j = 0

/-- Cholesky factorization: A = L * Lᵀ where L is lower triangular
    with positive diagonal. -/
def IsCholeskyOf {n : ℕ} (L A : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  IsLowerTriangular L ∧ (∀ i : Fin n, L i i > 0) ∧ L * Lᵀ = A

end ProvableContracts.Cholesky
