import ProvableContracts.Defs.KLDivergence
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Forward-KL KD Loss — Gradient / Loss Consistency (`KD-INV-002`)

Proves the loss / gradient consistency obligation of
`kd-loss-forward-kl-v1.yaml`: the `T²`-scaled forward-KL soft-target loss
is the antiderivative of the training gradient `T·(p_s − p_t)`.

For a fixed teacher distribution `p_t` (with `Σ p_t = 1`) and temperature
`T > 0`, the student soft-target loss as a function of the student logits
`s` is

  `L(s) = T² · Σ_i p_t_i · (log p_t_i − logSoftmaxT(s)_i)`,
  `logSoftmaxT(s)_i = s_i/T − log(Σ_k exp(s_k/T))`.

We prove the partial derivative along coordinate `j`, taken honestly as
the one-dimensional slice `t ↦ L(update s j t)` at the base point
`t = s_j`, equals

  `∂L/∂s_j = T·(softmaxT(s)_j − p_t_j) = T·(p_s_j − p_t_j)`,

exactly the KD-term gradient `kd_logit_gradient` trains the student with.
This is the mean-seeking forward-KL gradient; the reverse KL
`KL(p_s ‖ p_t)` does NOT have this gradient.

## References

- Hinton, Vinyals & Dean (2015), §2 (soft targets, `T²` scaling).
- Bishop (2006) PRML eq. 4.108 (softmax/log-partition derivative).
-/

namespace ProvableContracts.KLDivergence

open Real Finset

/-- Student softmax at temperature `T`:
    `softmaxT(s)_i = exp(s_i/T) / Σ_k exp(s_k/T)`. -/
noncomputable def softmaxT {n : ℕ} (T : ℝ) (s : RVec n) (i : Fin n) : ℝ :=
  Real.exp (s i / T) / ∑ k : Fin n, Real.exp (s k / T)

/-- Log-softmax at temperature `T`:
    `logSoftmaxT(s)_i = s_i/T − log(Σ_k exp(s_k/T))`. -/
noncomputable def logSoftmaxT {n : ℕ} (T : ℝ) (s : RVec n) (i : Fin n) : ℝ :=
  s i / T - Real.log (∑ k : Fin n, Real.exp (s k / T))

/-- The `T²`-scaled forward-KL soft-target loss as a function of the
    student logits `s` (teacher distribution `pt` and temperature `T`
    fixed):
    `L(s) = T² · Σ_i pt_i · (log pt_i − logSoftmaxT(s)_i)`. -/
noncomputable def kdSoftLoss {n : ℕ} (T : ℝ) (pt s : RVec n) : ℝ :=
  T ^ 2 * ∑ i : Fin n, pt i * (Real.log (pt i) - logSoftmaxT T s i)

/-- Derivative of `t ↦ log(exp(t/T) + rest)` for `rest ≥ 0`, `T ≠ 0`:
    `(1/T)·exp(t₀/T) / (exp(t₀/T) + rest)`. -/
theorem hasDerivAt_log_exp_div_add_const (T rest t₀ : ℝ)
    (_hT : T ≠ 0) (hrest : 0 ≤ rest) :
    HasDerivAt (fun t => Real.log (Real.exp (t / T) + rest))
      ((1 / T) * Real.exp (t₀ / T) / (Real.exp (t₀ / T) + rest)) t₀ := by
  have hZpos : 0 < Real.exp (t₀ / T) + rest :=
    add_pos_of_pos_of_nonneg (Real.exp_pos _) hrest
  -- inner: t ↦ t / T, derivative 1/T
  have hdiv : HasDerivAt (fun t => t / T) (1 / T) t₀ := by
    simpa using (hasDerivAt_id t₀).div_const T
  -- t ↦ exp(t/T), derivative (1/T)·exp(t₀/T)
  have hexp : HasDerivAt (fun t => Real.exp (t / T))
      (Real.exp (t₀ / T) * (1 / T)) t₀ := by
    simpa [Function.comp] using (Real.hasDerivAt_exp (t₀ / T)).comp t₀ hdiv
  -- t ↦ exp(t/T) + rest
  have hsum : HasDerivAt (fun t => Real.exp (t / T) + rest)
      (Real.exp (t₀ / T) * (1 / T)) t₀ := hexp.add_const rest
  -- log ∘ (exp(·/T) + rest)
  have hlog : HasDerivAt Real.log (Real.exp (t₀ / T) + rest)⁻¹
      (Real.exp (t₀ / T) + rest) := Real.hasDerivAt_log (ne_of_gt hZpos)
  have hcomp := hlog.comp t₀ hsum
  -- rewrite the goal's derivative value into `hcomp`'s form; `log ∘ f` is
  -- definitionally `fun t => log (f t)`.
  rw [show (1 / T) * Real.exp (t₀ / T) / (Real.exp (t₀ / T) + rest)
        = (Real.exp (t₀ / T) + rest)⁻¹ * (Real.exp (t₀ / T) * (1 / T)) from by ring]
  exact hcomp

/-- **`KD-INV-002` — Loss / gradient consistency.**

    The partial derivative of the `T²`-scaled forward-KL soft-target loss
    in coordinate `j`, along the honest slice `t ↦ kdSoftLoss(update s j t)`
    at the base point `t = s_j`, equals `T·(softmaxT(s)_j − pt_j)`.

    Hence the logged loss `T²·KL(p_t ‖ p_s)` is the antiderivative of the
    training gradient `T·(p_s − p_t)`: telemetry matches optimization. -/
theorem kdSoftLoss_backward {n : ℕ} (T : ℝ) (hT : 0 < T)
    (pt s : RVec (n + 1)) (j : Fin (n + 1)) (hpt : ∑ i : Fin (n + 1), pt i = 1) :
    HasDerivAt (fun t => kdSoftLoss T pt (Function.update s j t))
      (T * (softmaxT T s j - pt j)) (s j) := by
  classical
  have hTne : T ≠ 0 := ne_of_gt hT
  set rest : ℝ := ∑ k ∈ univ \ {j}, Real.exp (s k / T) with hrest_def
  have hrest : 0 ≤ rest :=
    Finset.sum_nonneg (fun k _ => le_of_lt (Real.exp_pos (s k / T)))
  set linRest : ℝ := ∑ i ∈ univ \ {j}, pt i * s i with hlin_def
  set C : ℝ := ∑ i : Fin (n + 1), pt i * Real.log (pt i) with hC_def
  -- Partition function under the coordinate update.
  have hZ : ∀ t : ℝ,
      (∑ k : Fin (n + 1), Real.exp ((Function.update s j t) k / T))
        = Real.exp (t / T) + rest := by
    intro t
    have happ : (fun k : Fin (n + 1) => Real.exp ((Function.update s j t) k / T))
              = Function.update (fun k => Real.exp (s k / T)) j (Real.exp (t / T)) := by
      funext k
      by_cases hk : k = j
      · subst hk; simp
      · simp [Function.update, hk]
    calc (∑ k : Fin (n + 1), Real.exp ((Function.update s j t) k / T))
        = ∑ k : Fin (n + 1), Function.update (fun k => Real.exp (s k / T)) j
            (Real.exp (t / T)) k := by rw [happ]
      _ = Real.exp (t / T) + rest := by
          rw [Finset.sum_update_of_mem (Finset.mem_univ j)]
  -- Weighted logit sum under the coordinate update.
  have hlin : ∀ t : ℝ,
      (∑ i : Fin (n + 1), pt i * (Function.update s j t) i) = pt j * t + linRest := by
    intro t
    have happ : (fun i : Fin (n + 1) => pt i * (Function.update s j t) i)
              = Function.update (fun i => pt i * s i) j (pt j * t) := by
      funext i
      by_cases hi : i = j
      · subst hi; simp
      · simp [Function.update, hi]
    calc (∑ i : Fin (n + 1), pt i * (Function.update s j t) i)
        = ∑ i : Fin (n + 1), Function.update (fun i => pt i * s i) j (pt j * t) i := by
          rw [happ]
      _ = pt j * t + linRest := by
          rw [Finset.sum_update_of_mem (Finset.mem_univ j)]
  -- Closed form of the loss slice.
  have hsplit : ∀ t : ℝ,
      kdSoftLoss T pt (Function.update s j t)
        = T ^ 2 * C - T * (pt j * t + linRest)
          + T ^ 2 * Real.log (Real.exp (t / T) + rest) := by
    intro t
    unfold kdSoftLoss logSoftmaxT
    rw [hZ t]
    have expand : (∑ i : Fin (n + 1), pt i *
          (Real.log (pt i) - ((Function.update s j t) i / T
            - Real.log (Real.exp (t / T) + rest))))
        = (∑ i : Fin (n + 1), pt i * Real.log (pt i))
          - (1 / T) * (∑ i : Fin (n + 1), pt i * (Function.update s j t) i)
          + (∑ i : Fin (n + 1), pt i) * Real.log (Real.exp (t / T) + rest) := by
      rw [Finset.mul_sum, Finset.sum_mul, ← Finset.sum_sub_distrib,
        ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl (fun i _ => ?_)
      ring
    rw [expand, hlin t, hpt, ← hC_def]
    field_simp
  -- Differentiate the closed form.
  have h1 : HasDerivAt (fun _ : ℝ => T ^ 2 * C) 0 (s j) := hasDerivAt_const (s j) _
  have h2a : HasDerivAt (fun t => pt j * t + linRest) (pt j) (s j) := by
    have := ((hasDerivAt_id (s j)).const_mul (pt j)).add_const linRest
    simpa using this
  have h2 : HasDerivAt (fun t => T * (pt j * t + linRest)) (T * pt j) (s j) :=
    h2a.const_mul T
  have h3a := hasDerivAt_log_exp_div_add_const T rest (s j) hTne hrest
  have h3 : HasDerivAt (fun t => T ^ 2 * Real.log (Real.exp (t / T) + rest))
      (T ^ 2 * ((1 / T) * Real.exp (s j / T) / (Real.exp (s j / T) + rest))) (s j) :=
    h3a.const_mul (T ^ 2)
  have hcomb : HasDerivAt
      (fun t => T ^ 2 * C - T * (pt j * t + linRest)
        + T ^ 2 * Real.log (Real.exp (t / T) + rest))
      (0 - T * pt j
        + T ^ 2 * ((1 / T) * Real.exp (s j / T) / (Real.exp (s j / T) + rest))) (s j) :=
    (h1.sub h2).add h3
  -- Rewrite into the loss slice and simplify the derivative value.
  have hfun : (fun t => kdSoftLoss T pt (Function.update s j t))
            = (fun t => T ^ 2 * C - T * (pt j * t + linRest)
                + T ^ 2 * Real.log (Real.exp (t / T) + rest)) := funext hsplit
  rw [hfun]
  -- softmaxT(s)_j = exp(s_j/T) / (exp(s_j/T) + rest)
  have hZs : (∑ k : Fin (n + 1), Real.exp (s k / T)) = Real.exp (s j / T) + rest := by
    have h := hZ (s j)
    rwa [Function.update_eq_self] at h
  have hsm : softmaxT T s j = Real.exp (s j / T) / (Real.exp (s j / T) + rest) := by
    unfold softmaxT; rw [hZs]
  have hZpos : 0 < Real.exp (s j / T) + rest :=
    add_pos_of_pos_of_nonneg (Real.exp_pos _) hrest
  have hval : (0 - T * pt j
        + T ^ 2 * ((1 / T) * Real.exp (s j / T) / (Real.exp (s j / T) + rest)))
      = T * (softmaxT T s j - pt j) := by
    rw [hsm]
    field_simp
    ring
  rw [hval] at hcomb
  exact hcomb

-- Tests
#check @hasDerivAt_log_exp_div_add_const
#check @kdSoftLoss_backward

end ProvableContracts.KLDivergence
