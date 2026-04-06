import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-!
# TRSM — Triangular Solve

TRSM solves A*X = B for X when A is triangular.
If A is invertible, X = A⁻¹ * B satisfies A * X = B.
-/

namespace ProvableContracts.BLAS

open Matrix

-- Status: proved
/-- Left multiplication by identity: I * B = B. -/
theorem trsm_identity {m n : ℕ} (B : Matrix (Fin m) (Fin n) ℝ) :
    (1 : Matrix (Fin m) (Fin m) ℝ) * B = B :=
  Matrix.one_mul B

#check @trsm_identity

end ProvableContracts.BLAS
