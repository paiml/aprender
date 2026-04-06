import ProvableContracts.Defs.Tensor
import Mathlib.Data.Matrix.Basic

/-!
# Einsum — Contraction as Matrix Multiply

For rank-2 tensors, einsum("ik,kj->ij", A, B) = A * B.
Matrix multiplication is the fundamental contraction operation.
-/

namespace ProvableContracts.Tensor

open Matrix

-- Status: proved
/-- Rank-2 contraction is associative (matrix multiplication). -/
theorem einsum_assoc {m n k l : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin k) ℝ)
    (C : Matrix (Fin k) (Fin l) ℝ) :
    contract (contract A B) C = contract A (contract B C) := by
  unfold contract; exact Matrix.mul_assoc A B C

-- Status: proved
/-- Contraction with identity is identity. -/
theorem einsum_identity {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) :
    contract (1 : Matrix (Fin m) (Fin m) ℝ) A = A := by
  unfold contract; exact Matrix.one_mul A

#check @einsum_assoc
#check @einsum_identity

end ProvableContracts.Tensor
