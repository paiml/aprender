import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

/-!
# SVD Definitions

Definitions for Singular Value Decomposition.

## References

- Golub & Van Loan, "Matrix Computations," 4th ed., Section 2.5.
-/

namespace ProvableContracts.SVD

open Matrix

/-- The singular values of A are the square roots of eigenvalues of AᵀA.
    We model them as a vector of non-negative reals. -/
noncomputable def singularValues {m n : ℕ} (_A : Matrix (Fin m) (Fin n) ℝ) :
    Type := { σ : Fin (min m n) → ℝ // ∀ i, σ i ≥ 0 }

end ProvableContracts.SVD
