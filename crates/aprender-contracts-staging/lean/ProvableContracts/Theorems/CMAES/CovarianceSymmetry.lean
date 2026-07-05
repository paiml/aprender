import ProvableContracts.Defs.CMAES
import Mathlib.Data.Matrix.Basic

/-!
# CMA-ES Covariance Symmetry

Proves the analytic core of obligation **CMA-INV-002** (`C = Cᵀ at every
generation`): the rank-one + rank-mu covariance update

    `C_{t+1} = a·C_t + b·(P Pᵀ) + c·(Q Qᵀ)`

preserves symmetry. Given `C_t` symmetric, every summand is symmetric
(`(P Pᵀ)ᵀ = P Pᵀ`, scalar multiples and sums of symmetric matrices are
symmetric), hence `C_{t+1}` is symmetric. This holds for **all** scalars and
matrices — no positivity hypothesis needed — so it is an exact algebraic
identity, not a tolerance-bounded numerical claim.
-/

namespace ProvableContracts.CMAES

open Matrix

-- Status: proved
/-- The covariance update preserves symmetry: if `Cᵀ = C` then
    `(covUpdate a b c C P Q)ᵀ = covUpdate a b c C P Q`. -/
theorem covUpdate_symmetric {n : ℕ} (a b c : ℝ)
    (C P Q : Matrix (Fin n) (Fin n) ℝ) (hC : Cᵀ = C) :
    (covUpdate a b c C P Q)ᵀ = covUpdate a b c C P Q := by
  unfold covUpdate
  simp only [Matrix.transpose_add, Matrix.transpose_smul, hC,
    Matrix.transpose_mul, Matrix.transpose_transpose]

-- Status: proved
/-- Element-level symmetry: `C_{t+1} i j = C_{t+1} j i`. -/
theorem covUpdate_symmetric_elem {n : ℕ} (a b c : ℝ)
    (C P Q : Matrix (Fin n) (Fin n) ℝ) (hC : Cᵀ = C) (i j : Fin n) :
    covUpdate a b c C P Q i j = covUpdate a b c C P Q j i := by
  have h := covUpdate_symmetric a b c C P Q hC
  have h2 : (covUpdate a b c C P Q)ᵀ j i = covUpdate a b c C P Q i j :=
    Matrix.transpose_apply (covUpdate a b c C P Q) j i
  rw [h] at h2
  exact h2.symm

-- Tests
#check @covUpdate_symmetric
#check @covUpdate_symmetric_elem

end ProvableContracts.CMAES
