import ProvableContracts.Defs.Transpose
import Mathlib.Data.Matrix.Basic

/-!
# Transpose Involution

Proves that transpose is an involution: (Aᵀ)ᵀ = A.

## Obligation

`TP-IDEMP-001`: transpose(transpose(A)) = A (bitwise exact)

This is a standard Mathlib result: `Matrix.transpose_transpose`.
-/

namespace ProvableContracts.Transpose

open Matrix

-- Status: proved
/-- Transpose is an involution: (Aᵀ)ᵀ = A. -/
theorem transpose_involution {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
    (Aᵀ)ᵀ = A :=
  Matrix.transpose_transpose A

-- Status: proved
/-- Element-level correctness: transpose(A)[j][i] = A[i][j]. -/
theorem transpose_element {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ)
    (i : Fin m) (j : Fin n) :
    Aᵀ j i = A i j :=
  rfl

-- Tests
#check @transpose_involution
#check @transpose_element

end ProvableContracts.Transpose
