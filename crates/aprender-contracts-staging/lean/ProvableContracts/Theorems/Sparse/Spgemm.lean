import Mathlib.Data.Matrix.Basic

/-!
# SpGEMM — Sparse Matrix-Matrix Multiply Associativity

SpGEMM is mathematically identical to dense GEMM. Associativity holds.
-/

namespace ProvableContracts.Sparse

open Matrix

-- Status: proved
/-- SpGEMM is associative: (A * B) * C = A * (B * C). -/
theorem spgemm_assoc {m n k l : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin k) ℝ)
    (C : Matrix (Fin k) (Fin l) ℝ) :
    A * B * C = A * (B * C) :=
  Matrix.mul_assoc A B C

#check @spgemm_assoc

end ProvableContracts.Sparse
