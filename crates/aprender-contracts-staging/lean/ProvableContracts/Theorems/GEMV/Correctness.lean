import ProvableContracts.Defs.GEMV
import Mathlib.Data.Matrix.Basic

/-!
# GEMV Output Length Correctness

Proves that GEMV (y = Ax) maps an n-vector to an m-vector when
A is m×n. In type-theoretic terms, the output type is `Fin m → ℝ`,
which is guaranteed by the type signature. We additionally prove
the element-level definition is correct.

## Obligation

`GEMV-DIM-001`: output of gemv(A, x) has dimension m when A is m×n.
`GEMV-ELEM-001`: gemv(A, x)_i = Σⱼ A_ij * x_j.
-/

namespace ProvableContracts.GEMV

open Matrix Finset

-- Status: proved
/-- GEMV element-level correctness: (Ax)ᵢ = Σⱼ Aᵢⱼ xⱼ.
    This unfolds the definition and shows it equals the dot product
    of row i of A with x. -/
theorem gemv_element {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ)
    (i : Fin m) :
    gemv A x i = ∑ j : Fin n, A i j * x j :=
  rfl

-- Status: proved
/-- Dimensionality is encoded in the type system: gemv on an m×n matrix
    and n-vector produces an m-vector. We prove this trivially via
    type equality (the function type `Fin m → ℝ` has exactly m outputs). -/
theorem gemv_output_type {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ) :
    gemv A x = A.mulVec x :=
  rfl

-- Tests
#check @gemv_element
#check @gemv_output_type

end ProvableContracts.GEMV
