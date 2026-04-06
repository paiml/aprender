import ProvableContracts.Defs.BLAS
import Mathlib.Data.Matrix.Basic

/-!
# SYRK Symmetry

Proves that the SYRK operation C = A * Aᵀ produces a symmetric matrix.

## Obligation

`SYRK-SYM-001`: (A * Aᵀ)ᵀ = A * Aᵀ

This is the fundamental symmetry property that justifies storing only
the upper (or lower) triangle of the result in BLAS.
-/

namespace ProvableContracts.BLAS

open Matrix

-- Status: proved
/-- SYRK produces a symmetric matrix: (AAᵀ)ᵀ = AAᵀ. -/
theorem syrk_symmetric {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) :
    (syrk A)ᵀ = syrk A := by
  unfold syrk
  simp [Matrix.transpose_mul, Matrix.transpose_transpose]

-- Status: proved
/-- Element-level symmetry: (AAᵀ)ᵢⱼ = (AAᵀ)ⱼᵢ. -/
theorem syrk_symmetric_elem {m n : ℕ}
    (A : Matrix (Fin m) (Fin n) ℝ) (i j : Fin m) :
    syrk A i j = syrk A j i := by
  have h := syrk_symmetric A
  have h2 : (syrk A)ᵀ j i = syrk A i j := Matrix.transpose_apply (syrk A) j i
  rw [h] at h2
  exact h2.symm

-- Tests
#check @syrk_symmetric
#check @syrk_symmetric_elem

end ProvableContracts.BLAS
