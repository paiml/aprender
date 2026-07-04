import ProvableContracts.Defs.Sampling
import Mathlib.Algebra.Order.Field.Basic

/-!
# Temperature Scaling Theorems

Proves the analytic content of the `temperature_bounds` equation of
`apr-cli-sampling-v1.yaml`: for a valid (positive) temperature, dividing the
logits by `t` is an order-preserving rescaling, so it preserves both the
pairwise ordering of logits and the argmax. This is precisely why
`temperature → 0` recovers greedy (argmax) decoding and why any `t > 0`
leaves the ranking — and hence the set the sampler draws from — intact.

## Obligation

`apr-cli-sampling-v1 / temperature_bounds`: temperature must be non-negative;
temperature scaling preserves argmax / is monotone (`temperature >= 0.0`).
-/

namespace ProvableContracts.Sampling

open ProvableContracts

/-- Temperature scaling with a positive temperature is strictly monotone:
    it preserves the strict ordering of logits. -/
theorem tempScale_monotone {n : ℕ} (x : RVec n) (t : ℝ) (ht : 0 < t)
    (i j : Fin n) (h : x j < x i) :
    tempScale x t j < tempScale x t i := by
  unfold tempScale
  gcongr

/-- Temperature scaling with a positive temperature preserves the non-strict
    ordering of logits. -/
theorem tempScale_mono_le {n : ℕ} (x : RVec n) (t : ℝ) (ht : 0 < t)
    (i j : Fin n) (h : x j ≤ x i) :
    tempScale x t j ≤ tempScale x t i := by
  unfold tempScale
  gcongr

/-- **Argmax preservation.** If `m` is an argmax of the raw logits (its logit
    dominates every other), then it remains an argmax after temperature
    scaling by any `t > 0`. Hence greedy decoding (`argmax`) is invariant to
    the temperature, which is exactly the `t → 0⁺` limit behaviour. -/
theorem tempScale_preserves_argmax {n : ℕ} (x : RVec n) (t : ℝ) (ht : 0 < t)
    (m : Fin n) (hm : ∀ j, x j ≤ x m) :
    ∀ j, tempScale x t j ≤ tempScale x t m := by
  intro j
  unfold tempScale
  gcongr
  exact hm j

/-- A valid temperature is non-negative; a positive temperature (the strict
    case used for scaling, since division by `0` is the greedy sentinel) is in
    particular non-negative. This records the guard invariant
    `temperature >= 0.0`. -/
theorem valid_temperature_nonneg (t : ℝ) (ht : 0 < t) : 0 ≤ t := le_of_lt ht

-- Tests
#check @tempScale_monotone
#check @tempScale_mono_le
#check @tempScale_preserves_argmax
#check @valid_temperature_nonneg

end ProvableContracts.Sampling
