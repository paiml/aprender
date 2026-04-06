import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import ProvableContracts.Basic

/-!
# GEMV Definitions

Mathematical definition of General Matrix-Vector Multiplication (GEMV),
matching the BLAS Level 2 operation: y = A * x.

## References

- Lawson et al. "Basic Linear Algebra Subprograms for Fortran Usage." 1979.
-/

namespace ProvableContracts.GEMV

open Matrix

/-- GEMV: y = A * x where A is m×n and x is n-dimensional.
    Result is an m-dimensional vector. We use Mathlib's `Matrix.mulVec`. -/
noncomputable def gemv {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ) :
    Fin m → ℝ :=
  A.mulVec x

end ProvableContracts.GEMV
