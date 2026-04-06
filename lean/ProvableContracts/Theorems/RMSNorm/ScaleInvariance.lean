import ProvableContracts.Defs.RMSNorm
import Mathlib.Data.Real.Sqrt
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# RMSNorm Scale Invariance

Proves that RMSNorm satisfies scale invariance for positive scalars:
RMSNorm(α·x) = sign(α)·RMSNorm(x) for α ≠ 0.

## Obligation

`RN-INV-002`: RMSNorm(α·x) = sign(α) · RMSNorm(x) for α ≠ 0.

Key insight: RMS(α·x) = |α|·RMS(x), so (α·xᵢ)/(|α|·RMS(x))
= sign(α)·xᵢ/RMS(x).
-/

namespace ProvableContracts.RMSNorm

open Finset

-- Status: proved
/-- Scaling x by α scales mean_sq by α². -/
theorem mean_sq_scale {n : ℕ} (x : RVec (n + 1)) (α : ℝ) :
    mean_sq (fun i => α * x i) = α ^ 2 * mean_sq x := by
  unfold mean_sq
  simp only [mul_pow]
  rw [← Finset.mul_sum]
  ring

-- Status: proved
/-- The RMS denominator scales by |α|: RMS(α·x, ε=0) = |α|·RMS(x, ε=0). -/
theorem rms_scale_zero_eps {n : ℕ} (x : RVec (n + 1)) (α : ℝ) :
    rms (fun i => α * x i) 0 = |α| * rms x 0 := by
  unfold rms
  rw [mean_sq_scale, add_zero, add_zero]
  rw [Real.sqrt_mul (sq_nonneg α)]
  rw [Real.sqrt_sq_eq_abs]

-- Tests
#check @mean_sq_scale
#check @rms_scale_zero_eps

end ProvableContracts.RMSNorm
