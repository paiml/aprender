import ProvableContracts.Defs.QR
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.SemiringInverse

/-!
# QR Orthogonality

Proves structural properties of orthogonal matrices: if QᵀQ = I then
QQᵀ = I (orthogonal matrices are invertible with inverse = transpose),
and the transpose of an orthogonal matrix is orthogonal.

## Obligation

`QR-ORTH-001`: QᵀQ = I ⇒ QQᵀ = I
`QR-ORTH-002`: Transpose of orthogonal is orthogonal.
`QR-ORTH-003`: (QᵀQ)ᵢⱼ = δᵢⱼ.
-/

namespace ProvableContracts.QR

open Matrix

-- Status: proved
/-- If QᵀQ = I then QQᵀ = I (left inverse equals right inverse for
    square matrices over ℝ). -/
theorem orthogonal_left_inverse {n : ℕ}
    (Q : Matrix (Fin n) (Fin n) ℝ) (h : Qᵀ * Q = 1) :
    Q * Qᵀ = 1 :=
  (Matrix.mul_eq_one_comm_of_equiv (Equiv.refl _)).mp h

-- Status: proved
/-- The transpose of an orthogonal matrix is orthogonal. -/
theorem orthogonal_transpose {n : ℕ}
    (Q : Matrix (Fin n) (Fin n) ℝ) (h : IsOrthogonal Q) :
    IsOrthogonal Qᵀ := by
  unfold IsOrthogonal at *
  rw [Matrix.transpose_transpose]
  exact (Matrix.mul_eq_one_comm_of_equiv (Equiv.refl _)).mp h

-- Status: proved
/-- Orthogonal columns: (QᵀQ)ᵢⱼ = δᵢⱼ directly from the definition. -/
theorem orthogonal_columns {n : ℕ}
    (Q : Matrix (Fin n) (Fin n) ℝ) (h : IsOrthogonal Q) (i j : Fin n) :
    (Qᵀ * Q) i j = (1 : Matrix (Fin n) (Fin n) ℝ) i j := by
  rw [h]

-- Tests
#check @orthogonal_left_inverse
#check @orthogonal_transpose
#check @orthogonal_columns

end ProvableContracts.QR
