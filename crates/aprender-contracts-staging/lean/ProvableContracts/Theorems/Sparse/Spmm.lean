import Mathlib.Data.Matrix.Basic

/-!
# SpMM — Sparse Matrix-Matrix Multiply Distributes Over Addition

SpMM: A * (B + C) = A*B + A*C (left distributivity).
-/

namespace ProvableContracts.Sparse

open Matrix

-- Status: proved
/-- SpMM left-distributes over addition. -/
theorem spmm_left_distrib {m n k : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ)
    (B C : Matrix (Fin n) (Fin k) ℝ) :
    A * (B + C) = A * B + A * C :=
  Matrix.mul_add A B C

#check @spmm_left_distrib

end ProvableContracts.Sparse
