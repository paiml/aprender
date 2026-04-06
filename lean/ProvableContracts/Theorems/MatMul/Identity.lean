import ProvableContracts.Defs.MatMul
import Mathlib.Data.Matrix.Basic

/-!
# Matrix Identity

Proves A * I = A for square matrices over ℝ.

## Obligation

`MM-IDENT-001`: A * I = A

The identity matrix is the neutral element for matrix multiplication.
Mathlib provides this via the `MulOneClass` instance on `Matrix`.
-/

namespace ProvableContracts.MatMul

open Matrix

-- Status: proved
/-- Right identity: A * I = A. -/
theorem matmul_identity_right {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) :
    A * (1 : Matrix (Fin n) (Fin n) ℝ) = A :=
  Matrix.mul_one A

-- Status: proved
/-- Left identity: I * A = A. -/
theorem matmul_identity_left {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) :
    (1 : Matrix (Fin m) (Fin m) ℝ) * A = A :=
  Matrix.one_mul A

-- Tests
#check @matmul_identity_right
#check @matmul_identity_left

end ProvableContracts.MatMul
