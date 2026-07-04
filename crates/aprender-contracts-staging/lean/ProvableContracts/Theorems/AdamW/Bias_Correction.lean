import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# AdamW Bias Correction — factor exceeds one

Contract: `adamw-kernel-v1`, equation `bias_correction`.

`m̂_t = m_t / (1 - β₁ᵗ)`. For `β ∈ (0,1)` and `t ≥ 1` the denominator lies in
`(0,1)`, so the correction factor `1/(1-βᵗ) > 1` (contract invariant "Correction
factor > 1 for all t ≥ 1").
-/

namespace ProvableContracts.AdamW.BiasCorrection

/-- Bias-corrected moment: `m̂ = m / (1 - βᵗ)`. -/
noncomputable def bias_correct (m beta : ℝ) (t : ℕ) : ℝ :=
  m / (1 - beta ^ t)

-- Status: proved (core algebraic)
/-- The bias-correction factor exceeds 1 for `β ∈ (0,1)` and `t ≥ 1`. -/
theorem bias_correction (beta : ℝ) (t : ℕ)
    (hpos : 0 < beta) (hlt : beta < 1) (ht : 1 ≤ t) :
    1 < 1 / (1 - beta ^ t) := by
  have hbt_pos : 0 < beta ^ t := pow_pos hpos t
  have hbt_lt : beta ^ t < 1 := pow_lt_one₀ hpos.le hlt (by omega)
  have hden_pos : 0 < 1 - beta ^ t := by linarith
  rw [one_lt_div hden_pos]
  linarith

#check @bias_correction

end ProvableContracts.AdamW.BiasCorrection
