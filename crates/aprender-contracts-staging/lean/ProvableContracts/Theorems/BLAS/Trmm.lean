import Mathlib.Data.Matrix.Basic

/-!
# TRMM — Triangular Matrix-Matrix Multiply

TRMM is a specialization of GEMM where one operand is triangular.
The product A*B where A is lower triangular preserves the
matrix multiplication associativity.
-/

namespace ProvableContracts.BLAS

open Matrix

-- Status: proved
/-- TRMM is a special case of GEMM: associativity still holds. -/
theorem trmm_assoc {m n k : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin k) ℝ)
    (C : Matrix (Fin k) (Fin m) ℝ) :
    A * B * C = A * (B * C) :=
  Matrix.mul_assoc A B C

#check @trmm_assoc

end ProvableContracts.BLAS
