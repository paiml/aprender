import ProvableContracts.Defs.MatMul
import Mathlib.Data.Matrix.Basic

/-!
# Transpose of a Product

Contract: `matmul-kernel-v1` — the fundamental matmul-algebra identity
`(AB)ᵀ = Bᵀ Aᵀ`. This is the correctness backbone of the row-major/col-major
transpose-at-import boundary (LAYOUT-001/002): a matmul followed by transpose
equals the reversed matmul of the transposes.

Over ℝ this holds exactly; Mathlib provides `Matrix.transpose_mul`.
-/

namespace ProvableContracts.MatMul

open Matrix

-- Status: proved
/-- Transpose of a product reverses order: `(A * B)ᵀ = Bᵀ * Aᵀ`. -/
theorem matmul_transpose {m n p : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin p) ℝ) :
    (A * B)ᵀ = Bᵀ * Aᵀ :=
  Matrix.transpose_mul A B

#check @matmul_transpose

end ProvableContracts.MatMul
