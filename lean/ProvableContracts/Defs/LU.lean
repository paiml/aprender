import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

/-!
# LU Factorization Definitions

Definitions for LU decomposition of square matrices.

## References

- Golub & Van Loan, "Matrix Computations," 4th ed., Section 3.2.
-/

namespace ProvableContracts.LU

open Matrix

/-- A matrix is lower triangular with unit diagonal. -/
def IsUnitLowerTriangular {n : ℕ} (L : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  (∀ i j : Fin n, j.val > i.val → L i j = 0) ∧ (∀ i : Fin n, L i i = 1)

/-- A matrix is upper triangular. -/
def IsUpperTriangular {n : ℕ} (U : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  ∀ i j : Fin n, i.val > j.val → U i j = 0

/-- LU factorization: A = L * U where L is unit lower triangular
    and U is upper triangular. -/
def IsLUOf {n : ℕ} (L U A : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  IsUnitLowerTriangular L ∧ IsUpperTriangular U ∧ L * U = A

end ProvableContracts.LU
