import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

/-!
# Transpose Definitions

Mathematical definition of matrix transpose, matching the
`transpose-kernel-v1.yaml` contract equations.

## References

- Standard linear algebra: B[j,i] = A[i,j]
-/

namespace ProvableContracts.Transpose

open Matrix

/-- Matrix transpose: B[j][i] = A[i][j].
    We use Mathlib's built-in `Matrix.transpose`. -/
noncomputable def transpose_mat {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
    Matrix (Fin n) (Fin m) ℝ :=
  Aᵀ

end ProvableContracts.Transpose
