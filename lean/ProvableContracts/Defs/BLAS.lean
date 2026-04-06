import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

/-!
# BLAS Definitions

Definitions for BLAS Level 3 operations.

## References

- Dongarra et al. "A set of level 3 basic linear algebra subprograms." 1990.
-/

namespace ProvableContracts.BLAS

open Matrix

/-- SYRK: C = A * Aᵀ (symmetric rank-k update, beta=0, alpha=1). -/
noncomputable def syrk {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
    Matrix (Fin m) (Fin m) ℝ :=
  A * Aᵀ

end ProvableContracts.BLAS
