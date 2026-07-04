import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# AdamW First Moment (EMA) — identity and magnitude bound

Contract: `adamw-kernel-v1`, equation `adam_moments`.

`m_t = β₁·m_{t-1} + (1-β₁)·g_t` is an exponential moving average — a convex
combination of the previous moment and the current gradient. This file proves
the defining identity and the magnitude bound stated in the contract invariant
("|m_t| bounded by max(|g_1|, …, |g_t|) when β₁ < 1").
-/

namespace ProvableContracts.AdamW.Moments

/-- First-moment EMA update. -/
noncomputable def adam_moment (beta1 m_prev g : ℝ) : ℝ :=
  beta1 * m_prev + (1 - beta1) * g

-- Status: proved (core algebraic)
/-- Defining identity: the update is exactly `β₁·m_{t-1} + (1-β₁)·g_t`. -/
theorem adam_moments (beta1 m_prev g : ℝ) :
    adam_moment beta1 m_prev g = beta1 * m_prev + (1 - beta1) * g := rfl

-- Status: proved (core algebraic)
/-- Convex-combination magnitude bound: for `β₁ ∈ [0,1]`, if the previous moment
    and the gradient are bounded by `B`, so is the updated moment. -/
theorem adam_moment_bounded (beta1 m_prev g B : ℝ)
    (h0 : 0 ≤ beta1) (h1 : beta1 ≤ 1)
    (hm : |m_prev| ≤ B) (hg : |g| ≤ B) :
    |adam_moment beta1 m_prev g| ≤ B := by
  unfold adam_moment
  have h1' : (0:ℝ) ≤ 1 - beta1 := by linarith
  rw [abs_le] at hm hg ⊢
  constructor
  · nlinarith [mul_le_mul_of_nonneg_left hm.1 h0, mul_le_mul_of_nonneg_left hg.1 h1']
  · nlinarith [mul_le_mul_of_nonneg_left hm.2 h0, mul_le_mul_of_nonneg_left hg.2 h1']

#check @adam_moments
#check @adam_moment_bounded

end ProvableContracts.AdamW.Moments
