import ProvableContracts.Defs.Sparse
import Mathlib.Data.Matrix.Basic

/-!
# SpMV Linearity

Proves that sparse matrix-vector multiplication is a linear operation:
- SpMV(A, x + y) = SpMV(A, x) + SpMV(A, y)     (additivity)
- SpMV(A, c • x) = c • SpMV(A, x)               (homogeneity)

These are the defining properties of a linear map and hold regardless
of sparsity structure.

## Obligation

`SPMV-LIN-001`: SpMV distributes over vector addition.
`SPMV-LIN-002`: SpMV is compatible with scalar multiplication.
-/

namespace ProvableContracts.Sparse

open Matrix

-- Status: proved
/-- SpMV distributes over vector addition: A(x + y) = Ax + Ay. -/
theorem spmv_add {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (x y : Fin n → ℝ) :
    spmv A (x + y) = spmv A x + spmv A y := by
  unfold spmv
  exact Matrix.mulVec_add A x y

-- Status: proved
/-- SpMV is compatible with scalar multiplication: A(cx) = c(Ax). -/
theorem spmv_smul {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (c : ℝ) (x : Fin n → ℝ) :
    spmv A (c • x) = c • spmv A x := by
  unfold spmv
  exact Matrix.mulVec_smul A c x

-- Status: proved
/-- SpMV preserves the zero vector: A * 0 = 0. -/
theorem spmv_zero {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) :
    spmv A 0 = 0 := by
  unfold spmv
  exact Matrix.mulVec_zero A

-- Tests
#check @spmv_add
#check @spmv_smul
#check @spmv_zero

example : spmv (0 : Matrix (Fin 2) (Fin 3) ℝ) 0 = 0 :=
  spmv_zero (0 : Matrix (Fin 2) (Fin 3) ℝ)

end ProvableContracts.Sparse
