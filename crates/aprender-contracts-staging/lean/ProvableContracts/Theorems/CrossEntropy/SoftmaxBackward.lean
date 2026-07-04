import ProvableContracts.Defs.CrossEntropy
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Cross-Entropy Backward Gradient (softmax − onehot)

Proves the headline autograd obligation of `cross-entropy-kernel-v1.yaml`:
the closed-form gradient of the softmax→onehot cross-entropy loss w.r.t. the
logits is `softmax(z) − onehot(k)`.

## Obligation `F-AUTOGRAD-CE-REDUCTION-001` (analytic core)

For a one-hot target at class `k`, the cross-entropy as a function of the
logit vector `z` is

  `CE(onehot k, z) = log(Σⱼ exp(zⱼ)) − z_k`.

Its partial derivative in coordinate `i` is

  `∂/∂z_i CE = exp(z_i)/Σⱼ exp(zⱼ) − [i = k] = softmax(z)_i − onehot(k)_i.`

This is the `s = 1` (Reduction::Sum) case; the Mean (÷batch) and None
(per-sample upstream) reductions are the scalar multiples `s · (softmax−onehot)`
required by the contract, obtained by multiplying this Jacobian by the upstream
scale (an algebraic corollary of `HasDerivAt.const_mul`).

We phrase the partial derivative honestly via `Function.update z i t`, i.e. the
genuine one-dimensional slice of the loss along coordinate `i`, evaluated at the
base point `t = z i`, and we tie it back to the contract's `cross_entropy`
definition with a one-hot target vector.

## References

- Bishop (2006) Pattern Recognition and Machine Learning, eq. 4.108
-/

namespace ProvableContracts.CrossEntropy

open Real Finset

/-- Softmax component `softmax(z)_i = exp(z_i) / Σⱼ exp(zⱼ)`. -/
noncomputable def softmax {n : ℕ} (z : RVec n) (i : Fin n) : ℝ :=
  Real.exp (z i) / ∑ j : Fin n, Real.exp (z j)

/-- One-hot indicator `onehot(k)_i = [i = k]`. -/
def onehot {n : ℕ} (k i : Fin n) : ℝ := if i = k then 1 else 0

/-- Cross-entropy with a one-hot target at class `k`, as a function of the
    logits: `CE(onehot k, z) = log(Σⱼ exp(zⱼ)) − z_k`. -/
noncomputable def ce_onehot {n : ℕ} (z : RVec n) (k : Fin n) : ℝ :=
  Real.log (∑ j : Fin n, Real.exp (z j)) - z k

/-- The one-hot target vector `onehotVec k = fun i => [i = k]`. -/
noncomputable def onehotVec {n : ℕ} (k : Fin n) : RVec n :=
  fun i => if i = k then 1 else 0

/-- The contract's `cross_entropy` with a one-hot target reduces to `ce_onehot`,
    i.e. `−Σᵢ [i=k]·log_softmax(z)_i = log(Σ exp) − z_k`. -/
theorem cross_entropy_onehot_eq {n : ℕ} (z : RVec n) (k : Fin n) :
    cross_entropy (onehotVec k) z = ce_onehot z k := by
  unfold cross_entropy ce_onehot onehotVec log_softmax
  rw [show (∑ i : Fin n, (if i = k then (1 : ℝ) else 0) *
        (z i - Real.log (∑ j : Fin n, Real.exp (z j))))
      = ∑ i : Fin n, (if i = k then
          (z i - Real.log (∑ j : Fin n, Real.exp (z j))) else 0) from by
        refine Finset.sum_congr rfl (fun i _ => ?_)
        by_cases hik : i = k <;> simp [hik]]
  rw [Finset.sum_ite_eq' Finset.univ k
        (fun i => z i - Real.log (∑ j : Fin n, Real.exp (z j)))]
  simp

/-- The definitional decomposition obligation
    (`LogSoftmax + NLL equals CrossEntropy`): the fused cross-entropy equals
    the separate `−Σᵢ tᵢ · log_softmax(z)_i`.  In exact real arithmetic this is
    an identity (the `< 1e-6` in the contract is the residual float rounding). -/
theorem cross_entropy_eq_nll {n : ℕ} (t z : RVec n) :
    cross_entropy t z = -(∑ i : Fin n, t i * log_softmax z i) := rfl

/-- Key analytic lemma: the derivative of `t ↦ log(exp t + rest)` for a
    non-negative constant `rest` is `exp t / (exp t + rest)`. -/
theorem hasDerivAt_log_exp_add_const (rest t : ℝ) (hrest : 0 ≤ rest) :
    HasDerivAt (fun s => Real.log (Real.exp s + rest))
      (Real.exp t / (Real.exp t + rest)) t := by
  have hpos : 0 < Real.exp t + rest := add_pos_of_pos_of_nonneg (Real.exp_pos t) hrest
  have hu : HasDerivAt (fun s => Real.exp s + rest) (Real.exp t) t :=
    (Real.hasDerivAt_exp t).add_const rest
  have hlog : HasDerivAt Real.log (Real.exp t + rest)⁻¹ (Real.exp t + rest) :=
    Real.hasDerivAt_log (ne_of_gt hpos)
  have hcomp := hlog.comp t hu
  simpa [Function.comp, div_eq_mul_inv, mul_comm] using hcomp

/-- **Cross-entropy backward gradient** (analytic core of
    `F-AUTOGRAD-CE-REDUCTION-001`).

    The partial derivative of the one-hot cross-entropy in coordinate `i`,
    taken along the slice `t ↦ CE(update z i t)` at the base point `t = z i`,
    equals `softmax(z)_i − onehot(k)_i`. -/
theorem ce_onehot_backward {n : ℕ} (z : RVec (n + 1)) (k i : Fin (n + 1)) :
    HasDerivAt (fun t => ce_onehot (Function.update z i t) k)
      (softmax z i - onehot k i) (z i) := by
  classical
  set rest : ℝ := ∑ x ∈ univ \ {i}, Real.exp (z x) with hrest_def
  have hrest : 0 ≤ rest := Finset.sum_nonneg (fun j _ => le_of_lt (Real.exp_pos (z j)))
  -- Sum over the updated logits, as a function of the free coordinate `t`.
  have key : ∀ t : ℝ,
      (∑ j : Fin (n + 1), Real.exp ((Function.update z i t) j)) = Real.exp t + rest := by
    intro t
    have happ : (fun j : Fin (n + 1) => Real.exp ((Function.update z i t) j))
              = Function.update (fun j => Real.exp (z j)) i (Real.exp t) := by
      funext j
      by_cases hj : j = i
      · subst hj; simp
      · simp [Function.update, hj]
    calc (∑ j : Fin (n + 1), Real.exp ((Function.update z i t) j))
        = ∑ j : Fin (n + 1), Function.update (fun j => Real.exp (z j)) i (Real.exp t) j := by
          rw [happ]
      _ = Real.exp t + rest := by
          rw [Finset.sum_update_of_mem (Finset.mem_univ i)]
  -- Base-point partition function `Z = exp(z i) + rest`, obtained from `key`
  -- at `t = z i` where `update z i (z i) = z`.
  have hZ : (∑ j : Fin (n + 1), Real.exp (z j)) = Real.exp (z i) + rest := by
    have h := key (z i)
    rwa [Function.update_eq_self] at h
  have hsm : softmax z i = Real.exp (z i) / (Real.exp (z i) + rest) := by
    unfold softmax; rw [hZ]
  -- Rewrite the loss slice into `log(exp t + rest) − (update z i t) k`.
  have hfun : (fun t => ce_onehot (Function.update z i t) k)
            = (fun t => Real.log (Real.exp t + rest) - (Function.update z i t) k) := by
    funext t; unfold ce_onehot; rw [key t]
  rw [hfun]
  -- Derivative of the log-partition term.
  have hlog := hasDerivAt_log_exp_add_const rest (z i) hrest
  rw [← hsm] at hlog
  -- Derivative of the `−(update z i t) k` term is `onehot k i`.
  have hf2 : HasDerivAt (fun t => (Function.update z i t) k) (onehot k i) (z i) := by
    unfold onehot
    by_cases hik : i = k
    · have hfeq : (fun t : ℝ => (Function.update z i t) k) = (fun t : ℝ => t) := by
        funext t; rw [Function.update_apply, if_pos hik.symm]
      rw [hfeq, if_pos hik]; exact hasDerivAt_id (z i)
    · have hfeq : (fun t : ℝ => (Function.update z i t) k) = (fun _ : ℝ => z k) := by
        funext t; rw [Function.update_apply, if_neg (Ne.symm hik)]
      rw [hfeq, if_neg hik]; exact hasDerivAt_const (z i) (z k)
  exact hlog.sub hf2

/-- **Cross-entropy backward gradient, contract form.**  The partial derivative
    of the contract's `cross_entropy` with a one-hot target, along coordinate
    `i`, is `softmax(z)_i − onehot(k)_i`. -/
theorem cross_entropy_onehot_backward {n : ℕ} (z : RVec (n + 1)) (k i : Fin (n + 1)) :
    HasDerivAt (fun t => cross_entropy (onehotVec k) (Function.update z i t))
      (softmax z i - onehot k i) (z i) := by
  have h := ce_onehot_backward z k i
  have hfun : (fun t => cross_entropy (onehotVec k) (Function.update z i t))
            = (fun t => ce_onehot (Function.update z i t) k) := by
    funext t; exact cross_entropy_onehot_eq _ k
  rw [hfun]; exact h

-- Tests
#check @cross_entropy_onehot_eq
#check @cross_entropy_eq_nll
#check @hasDerivAt_log_exp_add_const
#check @ce_onehot_backward
#check @cross_entropy_onehot_backward

end ProvableContracts.CrossEntropy
