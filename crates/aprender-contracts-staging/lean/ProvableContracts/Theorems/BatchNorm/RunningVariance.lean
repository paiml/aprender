import ProvableContracts.Defs.BatchNorm

/-!
# BatchNorm Running-Variance Non-Negativity

Proves that the EMA-updated running variance stays non-negative after any
number of updates. Each update is a convex combination
`σ_run' = (1-m)·σ_run + m·σ_B` with momentum `m ∈ [0,1]`; a convex
combination of non-negatives is non-negative, and non-negativity is
preserved by iteration (induction over the update sequence).

## Obligation

`Running variance non-negative`: `σ_run ≥ 0 after any number of updates`.
-/

namespace ProvableContracts.BatchNorm

open Finset

-- Status: proved
/-- One EMA step preserves non-negativity:
    a convex combination of non-negative values is non-negative. -/
theorem ema_step_nonneg (prev batch m : ℝ)
    (hp : 0 ≤ prev) (hb : 0 ≤ batch) (hm0 : 0 ≤ m) (hm1 : m ≤ 1) :
    0 ≤ ema_step prev batch m := by
  unfold ema_step
  have h1 : 0 ≤ (1 - m) * prev := mul_nonneg (by linarith) hp
  have h2 : 0 ≤ m * batch := mul_nonneg hm0 hb
  linarith

-- Status: proved
/-- Iterated EMA preserves non-negativity: starting from `init ≥ 0` and
    folding over batch variances that are all `≥ 0`, the running variance
    stays `≥ 0` after any number of updates. -/
theorem ema_fold_nonneg (init m : ℝ) (batches : List ℝ)
    (hinit : 0 ≤ init) (hm0 : 0 ≤ m) (hm1 : m ≤ 1)
    (hbatches : ∀ b ∈ batches, 0 ≤ b) :
    0 ≤ ema_fold init m batches := by
  unfold ema_fold
  induction batches generalizing init with
  | nil => simpa using hinit
  | cons b bs ih =>
    simp only [List.foldl_cons]
    apply ih
    · exact ema_step_nonneg init b m hinit (hbatches b (by simp)) hm0 hm1
    · intro c hc
      exact hbatches c (by simp [hc])

-- Tests
#check @ema_step_nonneg
#check @ema_fold_nonneg

end ProvableContracts.BatchNorm
