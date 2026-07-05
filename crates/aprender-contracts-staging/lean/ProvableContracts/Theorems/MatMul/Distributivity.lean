import ProvableContracts.Defs.MatMul
import Mathlib.Data.Matrix.Basic

/-!
# Matrix Multiplication Distributes Over Addition

Contract: `matmul-kernel-v1`, proof-obligation *Matmul distributes*
(`formal: |A(B+C) - AB - AC| < ε`) and equation invariant
"Matmul distributes over addition: A(B+C) = AB + AC".

Over ℝ this holds exactly (the `< ε` tolerance is the floating-point shadow of
the exact algebraic identity). Mathlib provides both sides via the `Ring`/
`NonUnitalNonAssocSemiring` structure on matrices:
- left  distributivity `Matrix.mul_add`: `A(B+C) = AB + AC`
- right distributivity `Matrix.add_mul`: `(A+B)C = AC + BC`
-/

namespace ProvableContracts.MatMul

open Matrix

-- Status: proved
/-- Left distributivity: `A(B+C) = AB + AC`. -/
theorem matmul_left_distrib {m n p : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ)
    (B C : Matrix (Fin n) (Fin p) ℝ) :
    A * (B + C) = A * B + A * C :=
  Matrix.mul_add A B C

-- Status: proved
/-- Right distributivity: `(A+B)C = AC + BC`. -/
theorem matmul_right_distrib {m n p : ℕ}
    (A B : Matrix (Fin m) (Fin n) ℝ)
    (C : Matrix (Fin n) (Fin p) ℝ) :
    (A + B) * C = A * C + B * C :=
  Matrix.add_mul A B C

#check @matmul_left_distrib
#check @matmul_right_distrib

end ProvableContracts.MatMul
