import ProvableContracts.Defs.BatchNorm
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# BatchNorm Training-Output Centering (Standardization)

Proves that the batch mean of the training output equals β. When γ = 1
this is exactly the contract's "training output standardized" obligation:
the normalized values have zero mean over the batch, so the affine shift
leaves the output batch mean equal to β.

## Obligation

`Training output standardized`:
`|mean(BN(x)[:, c]) - β_c| < ε per channel c when γ = 1`.

We prove the exact identity `mean(BN(x)) = β` (for any γ), of which the
tolerance statement with γ = 1 is an immediate consequence.

Key insight: Σₙ (xₙ - μ_B) = 0, so the centered term contributes nothing
to the output mean and only β remains.
-/

namespace ProvableContracts.BatchNorm

open Finset

-- Status: proved
/-- The centered batch sums to zero: Σₙ (xₙ - μ_B) = 0. -/
theorem sum_centered_zero {n : ℕ} (x : RVec (n + 1)) :
    univ.sum (fun i => x i - batchMean x) = 0 := by
  rw [Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  unfold batchMean
  have hn : (↑(n + 1) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  field_simp
  push_cast
  ring

-- Status: proved
/-- The batch mean of the BatchNorm training output equals β (for any γ).
    Since μ = batchMean x factors through the zero-sum centered term, the
    output mean collapses to β. -/
theorem batchnorm_output_mean {n : ℕ} (x : RVec (n + 1))
    (gamma beta eps : ℝ) :
    batchMean (batchnorm x gamma beta eps) = beta := by
  unfold batchMean batchnorm
  have hn : (↑(n + 1) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  -- Split the sum:  Σ [γ·(xᵢ-μ)/d + β] = (γ/d)·Σ(xᵢ-μ) + (n+1)·β
  have hsplit :
      univ.sum (fun i => gamma * (x i - batchMean x) / bn_denom x eps + beta)
        = gamma / bn_denom x eps * univ.sum (fun i => x i - batchMean x)
          + (n + 1 : ℝ) * beta := by
    rw [Finset.sum_add_distrib]
    congr 1
    · rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      ring
    · simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      push_cast; ring
  rw [hsplit, sum_centered_zero]
  field_simp
  ring

-- Status: proved
/-- Contract form: with γ = 1 the output batch mean is exactly β, so the
    standardization tolerance `|mean(BN(x)) - β| < ε` holds for every ε > 0. -/
theorem batchnorm_centering_unit_gamma {n : ℕ} (x : RVec (n + 1))
    (beta eps tol : ℝ) (htol : tol > 0) :
    |batchMean (batchnorm x 1 beta eps) - beta| < tol := by
  rw [batchnorm_output_mean]
  simpa using htol

-- Tests
#check @sum_centered_zero
#check @batchnorm_output_mean
#check @batchnorm_centering_unit_gamma

end ProvableContracts.BatchNorm
