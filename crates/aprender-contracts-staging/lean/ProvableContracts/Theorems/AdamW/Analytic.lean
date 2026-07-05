import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic

/-!
# AdamW — remaining analytic proof obligations

Contract: `adamw-kernel-v1`.

The four equation invariants (`adam_moments`, `adam_variance`, `bias_correction`,
`weight_update`) are proved in their own files. This file discharges the
remaining *analytic* `proof_obligations` of the contract that are algebraic /
arithmetic facts over ℝ (and ℕ), leaving only the genuinely runtime obligations
(imperative frame/aliasing, SIMD-ULP equivalence) for L2/L3 coverage.

Obligations proved here:

* `precondition` — the hyperparameter domain is satisfiable (a witness exists).
* `postcondition` — the Adam denominator `√v̂ + ε` is strictly positive when
  `v̂ ≥ 0` and `ε > 0`, so the update is well-defined (no division by zero);
  paired with second-moment non-negativity.
* `loop_invariant` — the second moment stays `≥ 0` across *all* training steps
  (induction over the step sequence with arbitrary gradients).
* `loop_variant` — the termination measure `V = max_steps − t` strictly
  decreases as the step counter advances.
* `bound` (bias-corrected finite) — `1 − βᵗ > 0` for `β ∈ (0,1)`, `t ≥ 1`, so
  `m̂ = m/(1−βᵗ)` and `v̂ = v/(1−βᵗ)` are well-defined, and `v̂ ≥ 0`.
-/

namespace ProvableContracts.AdamW.Analytic

/-! ## Precondition: hyperparameter domain is satisfiable -/

-- Status: proved (core algebraic)
/-- The AdamW precondition domain
    `lr>0 ∧ β₁∈(0,1) ∧ β₂∈(0,1) ∧ ε>0 ∧ λ≥0 ∧ t≥1` is non-empty: it is
    satisfied by a concrete witness, so the assumed preconditions are consistent
    (not vacuously unsatisfiable). -/
theorem precondition_satisfiable :
    ∃ (lr beta1 beta2 eps lam : ℝ) (t : ℕ),
      0 < lr ∧ 0 < beta1 ∧ beta1 < 1 ∧ 0 < beta2 ∧ beta2 < 1 ∧
      0 < eps ∧ 0 ≤ lam ∧ 1 ≤ t :=
  ⟨1, 1/2, 1/2, 1, 0, 1, by norm_num⟩

/-! ## Postcondition: Adam denominator is strictly positive (update well-defined) -/

-- Status: proved (core algebraic)
/-- The Adam denominator `√v̂ + ε` is strictly positive when `v̂ ≥ 0` and
    `ε > 0`; hence the Adam step `m̂ / (√v̂ + ε)` is well-defined (no division by
    zero) and the resulting weight update is a finite real. -/
theorem denom_pos (vhat eps : ℝ) (_hv : 0 ≤ vhat) (he : 0 < eps) :
    0 < Real.sqrt vhat + eps := by
  have hs : 0 ≤ Real.sqrt vhat := Real.sqrt_nonneg vhat
  linarith

-- Status: proved (core algebraic)
/-- Companion postcondition fact: the second moment update stays non-negative
    (finite denominator + non-negative variance = well-posed update). -/
theorem variance_update_nonneg (beta2 v_prev g : ℝ)
    (hv : 0 ≤ v_prev) (h0 : 0 ≤ beta2) (h1 : beta2 ≤ 1) :
    0 ≤ beta2 * v_prev + (1 - beta2) * g ^ 2 := by
  have t1 : 0 ≤ beta2 * v_prev := mul_nonneg h0 hv
  have t2 : 0 ≤ (1 - beta2) * g ^ 2 := mul_nonneg (by linarith) (sq_nonneg g)
  linarith

/-! ## Loop invariant: second moment non-negative across ALL steps -/

/-- The second-moment sequence over training steps, driven by an arbitrary
    gradient stream `g : ℕ → ℝ`, starting from `v₀ = 0`. -/
noncomputable def v_seq (beta2 : ℝ) (g : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | (n + 1) => beta2 * v_seq beta2 g n + (1 - beta2) * (g n) ^ 2

-- Status: proved (core algebraic — induction over steps)
/-- Loop invariant: for `β₂ ∈ [0,1]` the second moment is non-negative at
    *every* training step `n` (base case `v₀ = 0`, inductive step reuses the
    single-step non-negativity). -/
theorem v_seq_nonneg (beta2 : ℝ) (g : ℕ → ℝ)
    (h0 : 0 ≤ beta2) (h1 : beta2 ≤ 1) :
    ∀ n, 0 ≤ v_seq beta2 g n := by
  intro n
  induction n with
  | zero => simp [v_seq]
  | succ k ih =>
    show 0 ≤ beta2 * v_seq beta2 g k + (1 - beta2) * (g k) ^ 2
    exact variance_update_nonneg beta2 (v_seq beta2 g k) (g k) ih h0 h1

/-! ## Loop variant: termination measure strictly decreases -/

-- Status: proved (core arithmetic over ℕ)
/-- Loop variant: the measure `V = max_steps − t` strictly decreases as the step
    counter advances (while `t < max_steps`), and is bounded below by `0`. This
    is the well-founded termination measure for the training loop. -/
theorem loop_variant_decreasing (max_steps t : ℕ) (h : t < max_steps) :
    max_steps - (t + 1) < max_steps - t := by omega

-- Status: proved (core arithmetic over ℕ)
/-- The measure is always non-negative (trivially, over ℕ) — the `V ≥ 0` half of
    the loop-variant obligation. -/
theorem loop_variant_nonneg (max_steps t : ℕ) : 0 ≤ max_steps - t := Nat.zero_le _

/-! ## Bound: bias-corrected moments are finite (well-defined) -/

-- Status: proved (core algebraic)
/-- The bias-correction denominator `1 − βᵗ` is strictly positive for
    `β ∈ (0,1)` and `t ≥ 1`; hence `m̂ = m/(1−βᵗ)` and `v̂ = v/(1−βᵗ)` are
    well-defined (finite) whenever `m`, `v` are finite. -/
theorem bias_denom_pos (beta : ℝ) (t : ℕ)
    (hpos : 0 < beta) (hlt : beta < 1) (ht : 1 ≤ t) :
    0 < 1 - beta ^ t := by
  have hbt_lt : beta ^ t < 1 := pow_lt_one₀ hpos.le hlt (by omega)
  linarith

-- Status: proved (core algebraic)
/-- Bias-corrected second moment is non-negative: `v̂ = v/(1−βᵗ) ≥ 0` when
    `v ≥ 0` and the denominator is positive. -/
theorem vhat_nonneg (v beta : ℝ) (t : ℕ) (hv : 0 ≤ v)
    (hpos : 0 < beta) (hlt : beta < 1) (ht : 1 ≤ t) :
    0 ≤ v / (1 - beta ^ t) :=
  div_nonneg hv (bias_denom_pos beta t hpos hlt ht).le

#check @precondition_satisfiable
#check @denom_pos
#check @variance_update_nonneg
#check @v_seq_nonneg
#check @loop_variant_decreasing
#check @loop_variant_nonneg
#check @bias_denom_pos
#check @vhat_nonneg

end ProvableContracts.AdamW.Analytic
