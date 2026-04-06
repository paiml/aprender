import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

/-!
# Matrix Multiplication Definitions

Standard matrix multiplication using Mathlib's `Matrix.mul`.

## References

- Standard linear algebra: (AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
-/

namespace ProvableContracts.MatMul

open Matrix

/-- Matrix multiplication: C = A * B.
    We use Mathlib's built-in `Matrix.mul`. -/
noncomputable def matmul {m n p : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin p) ℝ) :
    Matrix (Fin m) (Fin p) ℝ :=
  A * B

end ProvableContracts.MatMul
