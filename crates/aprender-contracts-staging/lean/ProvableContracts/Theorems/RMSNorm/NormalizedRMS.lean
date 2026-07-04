import ProvableContracts.Defs.RMSNorm
import ProvableContracts.Theorems.RMSNorm.ScaleInvariance
import ProvableContracts.Theorems.RMSNorm.DenominatorPositive
import Mathlib.Data.Real.Sqrt

/-!
# RMSNorm Normalized-RMS Idempotency

Proves the idempotency obligation: after normalization with unit weight
(`γ = 1`) and `ε = 0`, the root-mean-square of the output is exactly `1`
(for any non-degenerate input, `mean(x²) > 0`).

## Obligation

`RN-IDEM-001`: RMS(RMSNorm(x)/γ) ≈ 1 when γ = 1.

Key insight: RMSNorm(x)ᵢ = xᵢ / RMS(x). Then
mean((xᵢ/RMS(x))²) = mean(x²)/RMS(x)² = mean(x²)/mean(x²) = 1,
so RMS of the normalized vector is √1 = 1.

The `mean(x²) > 0` hypothesis excludes the zero vector, whose normalized
output is `0` (RMS = 0 ≠ 1) — the edge case covered separately by
falsification test `FALSIFY-RN-004`.
-/

namespace ProvableContracts.RMSNorm

open Finset

-- Status: proved
/-- The RMS denominator squared equals `mean_sq` at `ε = 0`
    (since `mean_sq ≥ 0`, squaring the square-root recovers the argument). -/
theorem rms_sq_zero_eps {n : ℕ} (x : RVec (n + 1)) :
    (rms x 0) ^ 2 = mean_sq x := by
  unfold rms
  rw [add_zero, Real.sq_sqrt (mean_sq_nonneg x)]

-- Status: proved
/-- Mean of squares of the unit-weight normalized vector is `1`
    whenever the input is non-degenerate (`mean_sq x > 0`). -/
theorem normalized_mean_sq_one {n : ℕ} (x : RVec (n + 1))
    (hx : mean_sq x > 0) :
    mean_sq (rmsnorm x (fun _ => 1) 0) = 1 := by
  have hfun : (rmsnorm x (fun _ => 1) 0) = (fun i => (1 / rms x 0) * x i) := by
    funext i
    unfold rmsnorm
    ring
  rw [hfun, mean_sq_scale, div_pow, one_pow, rms_sq_zero_eps, one_div,
    inv_mul_cancel₀ hx.ne']

-- Status: proved
/-- Idempotency: the RMS of the unit-weight, `ε = 0` normalized vector is
    exactly `1` for any non-degenerate input. -/
theorem normalized_rms_one {n : ℕ} (x : RVec (n + 1))
    (hx : mean_sq x > 0) :
    rms (rmsnorm x (fun _ => 1) 0) 0 = 1 := by
  unfold rms
  rw [add_zero, normalized_mean_sq_one x hx, Real.sqrt_one]

-- Tests
#check @rms_sq_zero_eps
#check @normalized_mean_sq_one
#check @normalized_rms_one

end ProvableContracts.RMSNorm
