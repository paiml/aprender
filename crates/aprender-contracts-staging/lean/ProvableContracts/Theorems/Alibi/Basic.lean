import ProvableContracts.Defs.Alibi

/-!
# ALiBi (Attention with Linear Biases) — Theorems

Proves the four **analytic** proof obligations of `alibi-kernel-v1.yaml`:

* `AL-BND-001` Negative bias      → `alibi_bias_nonpos`
* `AL-BND-002` Slope positivity   → `alibi_slope_pos`
* `causal-consistency`            → `alibi_causal_masks_future`
* `head-monotonic slopes`         → `alibi_slope_antitone`

Supporting structural facts (`bias(i,i) = 0`, linearity, monotone-in-distance)
are proved too. The fifth obligation (SIMD-vs-scalar within 8 ULP) is an
empirical floating-point property, not an analytic one, and is intentionally
NOT proved here (marked `l4_not_applicable` in the contract's
`verification_summary`).
-/

namespace ProvableContracts.Alibi

open Real

/-- The integer distance is non-negative. -/
theorem dist_nonneg (i j : ℤ) : 0 ≤ dist i j := by
  unfold dist; exact abs_nonneg _

/-- `dist` is symmetric. -/
theorem dist_comm (i j : ℤ) : dist i j = dist j i := by
  unfold dist; rw [abs_sub_comm]

/-- Self-position has zero bias: `bias(i,i) = 0`. -/
theorem alibi_bias_self (m : ℝ) (i : ℤ) : alibiBias m i i = 0 := by
  unfold alibiBias dist; simp

/-- **Linearity in distance**: `bias(i,j) = -m · |i-j|`. -/
theorem alibi_bias_linear (m : ℝ) (i j : ℤ) :
    alibiBias m i j = -m * (dist i j : ℝ) := rfl

/-- **Obligation AL-BND-001 (Negative bias)**: with a non-negative slope,
    every ALiBi bias is `≤ 0` (attention scores only ever decrease). -/
theorem alibi_bias_nonpos {m : ℝ} (hm : 0 ≤ m) (i j : ℤ) :
    alibiBias m i j ≤ 0 := by
  unfold alibiBias
  have hd : (0 : ℝ) ≤ (dist i j : ℝ) := by exact_mod_cast dist_nonneg i j
  nlinarith [mul_nonneg hm hd]

/-- **Monotone-decreasing in distance**: a strictly larger distance yields a
    smaller (more negative) bias. -/
theorem alibi_bias_antitone {m : ℝ} (hm : 0 ≤ m) {i₁ j₁ i₂ j₂ : ℤ}
    (h : dist i₁ j₁ ≤ dist i₂ j₂) :
    alibiBias m i₂ j₂ ≤ alibiBias m i₁ j₁ := by
  unfold alibiBias
  have hc : (dist i₁ j₁ : ℝ) ≤ (dist i₂ j₂ : ℝ) := by exact_mod_cast h
  nlinarith [mul_nonneg hm (by linarith : (0 : ℝ) ≤ (dist i₂ j₂ : ℝ) - (dist i₁ j₁ : ℝ))]

/-- **Obligation AL-BND-002 (Slope positivity)**: `m_h = 2^(-8h/H) > 0` for
    every head — `2 > 0`, so any real power of it is strictly positive. -/
theorem alibi_slope_pos (H h : ℕ) : 0 < slope H h := by
  unfold slope
  exact Real.rpow_pos_of_pos (by norm_num) _

/-- **Obligation (Head-monotonic slopes)**: with `H > 0`, the slope schedule
    is strictly decreasing in the head index — `h₁ < h₂ → m_{h₁} > m_{h₂}`. -/
theorem alibi_slope_antitone {H h₁ h₂ : ℕ} (hH : 0 < H) (h : h₁ < h₂) :
    slope H h₂ < slope H h₁ := by
  unfold slope
  apply Real.rpow_lt_rpow_of_exponent_lt (by norm_num : (1 : ℝ) < 2)
  have hHpos : (0 : ℝ) < (H : ℝ) := by exact_mod_cast hH
  have hinv : (0 : ℝ) < (H : ℝ)⁻¹ := inv_pos.mpr hHpos
  have hlt : (h₁ : ℝ) < (h₂ : ℝ) := by exact_mod_cast h
  have key : -(8 * (h₂ : ℝ)) * (H : ℝ)⁻¹ < -(8 * (h₁ : ℝ)) * (H : ℝ)⁻¹ := by
    apply mul_lt_mul_of_pos_right _ hinv
    linarith
  simpa [div_eq_mul_inv] using key

/-- **Obligation (Causal consistency)**: in causal mode a future position
    (`j > i`) is masked to `-∞` (`⊥` in `EReal`); after softmax `exp(-∞) = 0`,
    so future tokens receive zero attention weight. -/
theorem alibi_causal_masks_future (m : ℝ) {i j : ℤ} (h : j > i) :
    causalBias m i j = ⊥ := by
  unfold causalBias; rw [if_pos h]

/-- Complementary fact: a non-future position keeps its finite linear bias. -/
theorem alibi_causal_keeps_past (m : ℝ) {i j : ℤ} (h : ¬ j > i) :
    causalBias m i j = ((alibiBias m i j : ℝ) : EReal) := by
  unfold causalBias; rw [if_neg h]

-- Regression checks
#check @alibi_bias_nonpos
#check @alibi_slope_pos
#check @alibi_slope_antitone
#check @alibi_causal_masks_future

end ProvableContracts.Alibi
