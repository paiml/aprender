import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

/-!
# QR Factorization Definitions

Definitions for QR decomposition.

## References

- Golub & Van Loan, "Matrix Computations," 4th ed., Section 5.2.
-/

namespace ProvableContracts.QR

open Matrix

/-- A square matrix Q is orthogonal: Qᵀ * Q = I. -/
def IsOrthogonal {n : ℕ} (Q : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  Qᵀ * Q = (1 : Matrix (Fin n) (Fin n) ℝ)

end ProvableContracts.QR
