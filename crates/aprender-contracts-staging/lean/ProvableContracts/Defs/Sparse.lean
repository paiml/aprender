import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import ProvableContracts.Basic

/-!
# Sparse Matrix-Vector Multiply Definitions

Sparse matrix-vector multiplication (SpMV) is mathematically identical to
dense matrix-vector multiplication. The "sparse" property is a storage
optimization; the mathematical operation is the same linear map.

## References

- Saad, Y. "Iterative Methods for Sparse Linear Systems," 2nd ed., 2003.
-/

namespace ProvableContracts.Sparse

open Matrix

/-- SpMV is mathematically identical to dense matrix-vector multiplication.
    The sparsity is a storage concern, not a mathematical one.
    We model it as `Matrix.mulVec`. -/
noncomputable def spmv {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ) :
    Fin m → ℝ :=
  A.mulVec x

end ProvableContracts.Sparse
