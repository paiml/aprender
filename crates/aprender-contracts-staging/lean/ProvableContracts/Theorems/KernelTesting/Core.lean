/-
  KernelTesting.lean — machine-checked lemmas behind kernel-testing-gpu-cpu-hardware.md (KTEST-001)
  Lean 4 core only (no Mathlib). Check with:  lean KernelTesting.lean
-/
namespace KernelTesting

/-! ## K3 — race freedom from an injective store map
  A kernel where thread `t` writes element `j < s` of its own stride-`s` slice
  (index = t*s + j) never has two (thread, lane) pairs write the same address. -/

theorem store_index_injective (s t₁ t₂ j₁ j₂ : Nat)
    (h₁ : j₁ < s) (h₂ : j₂ < s) (h : t₁ * s + j₁ = t₂ * s + j₂) :
    t₁ = t₂ ∧ j₁ = j₂ := by
  have hs : 0 < s := by omega
  have d₁ : (t₁ * s + j₁) / s = t₁ := by
    rw [Nat.add_comm (t₁ * s) j₁, Nat.add_mul_div_right _ _ hs, Nat.div_eq_of_lt h₁, Nat.zero_add]
  have d₂ : (t₂ * s + j₂) / s = t₂ := by
    rw [Nat.add_comm (t₂ * s) j₂, Nat.add_mul_div_right _ _ hs, Nat.div_eq_of_lt h₂, Nat.zero_add]
  have m₁ : (t₁ * s + j₁) % s = j₁ := by
    rw [Nat.add_comm (t₁ * s) j₁, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt h₁]
  have m₂ : (t₂ * s + j₂) % s = j₂ := by
    rw [Nat.add_comm (t₂ * s) j₂, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt h₂]
  constructor
  · rw [← d₁, ← d₂, h]
  · rw [← m₁, ← m₂, h]

/-- The same map stays in bounds: with `n` threads and stride `s`, every index is `< n*s`. -/
theorem store_index_in_bounds (n s t j : Nat) (ht : t < n) (hj : j < s) :
    t * s + j < n * s := by
  have : (t + 1) * s ≤ n * s := Nat.mul_le_mul_right s ht
  rw [Nat.succ_mul] at this
  omega

/-! ## K1 — when the γ_K error bound is vacuous
  Higham: |fl(xᵀy) − xᵀy| ≤ γ_K |x|ᵀ|y| with γ_K = K·u/(1 − K·u), valid only when K·u < 1.
  With u = 2^(−p): K·u < 1 ⇔ K < 2^p. -/

/-- f16 accumulation (p = 11): a 4096-long dot product has NO γ bound (K·u = 2). -/
theorem f16_acc_k4096_vacuous : ¬ (4096 < 2 ^ 11) := by decide
/-- bf16 accumulation (p = 8): vacuous already at K = 256. -/
theorem bf16_acc_k256_vacuous : ¬ (256 < 2 ^ 8) := by decide
/-- f32 accumulation (p = 24): K up to 2^20 keeps K·u ≤ 1/16, so γ_K ≤ 1/15 (meaningful). -/
theorem f32_acc_k2p20_ok : 16 * 2 ^ 20 ≤ 2 ^ 24 := by decide
/-- Therefore the rule "accumulate in f32" is forced for the hidden sizes we ship (≤ 2^20). -/
theorem f32_needed_for_4096 : ¬ (4096 < 2 ^ 11) ∧ 4096 < 2 ^ 24 := by decide

/-! ## K2 — shape-class coverage for a tiled loop
  A loop over `d` elements in tiles of `T` executes `d / T` full tiles and one tail of `d % T`.
  Its control flow is determined by the pair (d / T = 0?, d % T = 0?). The four classes are
  covered by the representatives {T - 1 (tail only), T (one full, no tail), T + 1 (full + tail),
  2T (multi full, no tail)} for T ≥ 2. -/

def cls (T d : Nat) : Bool × Bool := (d / T == 0, d % T == 0)

theorem classes_covered (T : Nat) (hT : 2 ≤ T) :
    cls T (T - 1) = (true, false) ∧
    cls T T = (false, true) ∧
    cls T (T + 1) = (false, false) := by
  refine ⟨?_, ?_, ?_⟩
  · have h1 : (T - 1) / T = 0 := Nat.div_eq_of_lt (by omega)
    have h2 : (T - 1) % T = T - 1 := Nat.mod_eq_of_lt (by omega)
    have h3 : T - 1 ≠ 0 := by omega
    unfold cls; rw [h1, h2, beq_false_of_ne h3]; rfl
  · have hT0 : 0 < T := by omega
    unfold cls; rw [Nat.div_self hT0, Nat.mod_self]; rfl
  · have hT0 : 0 < T := by omega
    have h1 : (T + 1) / T = 1 := by
      rw [Nat.add_comm T 1, Nat.add_div_right 1 hT0, Nat.div_eq_of_lt (by omega)]
    have h2 : (T + 1) % T = 1 := by
      rw [Nat.add_mod_left, Nat.mod_eq_of_lt (by omega)]
    simp [cls, h1, h2]

/-- Every `d` falls into exactly one of the reachable classes; (true, true) means d = 0. -/
theorem zero_class (T d : Nat) (hT : 0 < T) (h : cls T d = (true, true)) : d = 0 := by
  simp [cls] at h
  obtain ⟨h1, h2⟩ := h
  rcases h1 with h1 | h1
  · omega
  · rw [Nat.mod_eq_of_lt h1] at h2
    exact h2


/-- v1.1 (review 1B): the 4-class model distinguishes 1 full tile from ≥ 2 full tiles
    (accumulator/unroll bugs), so the representatives {T−1, T, T+1, 2T} map to 4 DISTINCT classes. -/
def cls4 (T d : Nat) : Nat × Bool := (min (d / T) 2, d % T == 0)

theorem classes4_distinct (T : Nat) (hT : 2 ≤ T) :
    cls4 T (T - 1) = (0, false) ∧
    cls4 T T = (1, true) ∧
    cls4 T (T + 1) = (1, false) ∧
    cls4 T (2 * T) = (2, true) := by
  have hT0 : 0 < T := by omega
  refine ⟨?_, ?_, ?_, ?_⟩
  · have h1 : (T - 1) / T = 0 := Nat.div_eq_of_lt (by omega)
    have h2 : (T - 1) % T = T - 1 := Nat.mod_eq_of_lt (by omega)
    have h3 : T - 1 ≠ 0 := by omega
    unfold cls4; rw [h1, h2, beq_false_of_ne h3]; rfl
  · unfold cls4; rw [Nat.div_self hT0, Nat.mod_self]; rfl
  · have h1 : (T + 1) / T = 1 := by
      rw [Nat.add_comm T 1, Nat.add_div_right 1 hT0, Nat.div_eq_of_lt (by omega)]
    have h2 : (T + 1) % T = 1 := by
      rw [Nat.add_mod_left, Nat.mod_eq_of_lt (by omega)]
    unfold cls4; rw [h1, h2]; rfl
  · have h1 : (2 * T) / T = 2 := Nat.mul_div_cancel 2 hT0
    have h2 : (2 * T) % T = 0 := Nat.mul_mod_left 2 T
    unfold cls4; rw [h1, h2]; rfl

/-- FTZ drift for f16 accumulation at K = 4096: 4096 · 2^-14 = 1/4 (review 2B), i.e. 4096 = 2^14 / 4. -/
theorem f16_ftz_drift_quarter : 4096 * 4 = 2 ^ 14 := by decide

/-! ## K4 — cost: kernel receipts vs model cells (T8-style additivity)
  Retest cost scales with changed kernels, not with models × verbs × contexts. -/

/-- In half-second units: 60 kernels × 8 device-backends × 10 shape classes × 1 (0.5 s each)
    = 4800 half-seconds (40 min of device time, split across hosts), against 1008 cells × 48
    (24 s each, the measured 6.7 h / 1008) = 48384 half-seconds. Kernel receipts are ~10× cheaper
    before any reuse. -/
theorem kernel_receipts_cheaper : 60 * 8 * 10 * 1 < 1008 * 48 := by decide

end KernelTesting
