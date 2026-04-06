import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import ProvableContracts.Basic

/-!
# Tensor Contraction Definitions

Einstein summation (einsum) as generalized contraction.

## References

- Einstein, A. "The Foundation of the General Theory of Relativity." 1916.
-/

namespace ProvableContracts.Tensor

open Matrix Finset

/-- Tensor contraction over a single index is equivalent to matrix multiplication.
    For rank-2 tensors: einsum("ik,kj->ij", A, B) = A @ B. -/
noncomputable def contract {m n k : ℕ}
    (A : Matrix (Fin m) (Fin k) ℝ) (B : Matrix (Fin k) (Fin n) ℝ) :
    Matrix (Fin m) (Fin n) ℝ :=
  A * B

end ProvableContracts.Tensor
