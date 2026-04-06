import ProvableContracts.Defs.MatMul
import Mathlib.Data.Matrix.Basic

/-!
# Matrix Multiplication Associativity

Proves (AB)C = A(BC) for conformable real matrices.

## Obligation

`MM-ASSOC-001`: (A * B) * C = A * (B * C)

This is a fundamental property of matrix multiplication.
Mathlib provides `Matrix.mul_assoc` via the `Semigroup` instance.
-/

namespace ProvableContracts.MatMul

open Matrix

-- Status: proved
/-- Matrix multiplication is associative: (AB)C = A(BC). -/
theorem matmul_assoc {m n p q : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin p) ℝ)
    (C : Matrix (Fin p) (Fin q) ℝ) :
    (A * B) * C = A * (B * C) :=
  Matrix.mul_assoc A B C

-- Tests
#check @matmul_assoc

end ProvableContracts.MatMul
