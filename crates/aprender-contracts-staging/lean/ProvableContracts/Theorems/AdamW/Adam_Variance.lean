import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# AdamW Second Moment (EMA) — non-negativity

Contract: `adamw-kernel-v1`, equation `adam_variance`.

`v_t = β₂·v_{t-1} + (1-β₂)·g_t²`. Since `g_t² ≥ 0` and `v_0 = 0`, the second
moment stays non-negative for all steps when `β₂ ∈ [0,1]` — the contract's
`v_t ≥ 0` invariant / bound obligation.
-/

namespace ProvableContracts.AdamW.Variance

/-- Second-moment EMA update. -/
noncomputable def adam_variance_update (beta2 v_prev g : ℝ) : ℝ :=
  beta2 * v_prev + (1 - beta2) * g ^ 2

-- Status: proved (core algebraic)
/-- The second moment is non-negative when the previous value is non-negative
    and `β₂ ∈ [0,1]` (base case `v_0 = 0 ≥ 0`, then induction over steps). -/
theorem adam_variance (beta2 v_prev g : ℝ)
    (hv : 0 ≤ v_prev) (h0 : 0 ≤ beta2) (h1 : beta2 ≤ 1) :
    0 ≤ adam_variance_update beta2 v_prev g := by
  unfold adam_variance_update
  have t1 : 0 ≤ beta2 * v_prev := mul_nonneg h0 hv
  have t2 : 0 ≤ (1 - beta2) * g ^ 2 := mul_nonneg (by linarith) (sq_nonneg g)
  linarith

#check @adam_variance

end ProvableContracts.AdamW.Variance
