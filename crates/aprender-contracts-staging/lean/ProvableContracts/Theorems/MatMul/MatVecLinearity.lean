import ProvableContracts.Defs.MatMul
import Mathlib.Data.Matrix.Basic

/-!
# Matrix-Vector Product Linearity

Contract: `matmul-kernel-v1` — matvec linearity, the vector specialization of
the *Matmul distributes* obligation (`A(x+y) = Ax + Ay`, `A(c·x) = c·(Ax)`).
This is the algebra underlying GEMV (`aprender-contracts-staging` `GEMV.lean`)
and every attention / FFN projection.

Over ℝ these hold exactly; Mathlib provides `Matrix.mulVec_add` and
`Matrix.mulVec_smul`.
-/

namespace ProvableContracts.MatMul

open Matrix

-- Status: proved
/-- Additivity of matvec: `A(x + y) = Ax + Ay`. -/
theorem matvec_add {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (x y : Fin n → ℝ) :
    A.mulVec (x + y) = A.mulVec x + A.mulVec y :=
  Matrix.mulVec_add A x y

-- Status: proved
/-- Homogeneity of matvec: `A(c · x) = c · (A x)`. -/
theorem matvec_smul {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (c : ℝ) (x : Fin n → ℝ) :
    A.mulVec (c • x) = c • A.mulVec x :=
  Matrix.mulVec_smul A c x

#check @matvec_add
#check @matvec_smul

end ProvableContracts.MatMul
