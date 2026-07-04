import ProvableContracts.Defs.BatchNorm
import Mathlib.Data.Real.Sqrt

/-!
# BatchNorm Eval Uses Running Statistics

Proves that the eval-mode output genuinely depends on the running mean:
holding the variance/denominator fixed at the batch value, if the running
mean differs from the batch mean (and γ ≠ 0), then the eval output differs
from the train output at every index. This is the analytic core of the
FALSIFY-BN-004 obligation — a mode flag that "ignored running stats and
always used batch stats" would make the two functions equal, which we
refute algebraically.

## Obligation

`Eval mode uses running stats`:
`BN_eval(x) uses μ_run/σ_run, not batch statistics`
(FALSIFY-BN-004: `BN_eval(x) ≠ BN_train(x) when running stats differ`).
-/

namespace ProvableContracts.BatchNorm

open Finset

-- Status: proved
/-- With the eval variance pinned to the batch variance (so both modes share
    the denominator √(σ²_B + ε) > 0), a running mean that differs from the
    batch mean produces a strictly different output at every index, provided
    γ ≠ 0. Hence eval mode genuinely consumes the running mean rather than
    silently falling back to batch statistics. -/
theorem batchnorm_eval_ne_train {n : ℕ} (x : RVec (n + 1))
    (mu_run gamma beta eps : ℝ) (i : Fin (n + 1))
    (hg : gamma ≠ 0) (heps : eps > 0) (hmu : mu_run ≠ batchMean x) :
    batchnorm_eval x mu_run (batchVar x) gamma beta eps i
      ≠ batchnorm x gamma beta eps i := by
  unfold batchnorm_eval batchnorm bn_denom
  set d : ℝ := Real.sqrt (batchVar x + eps) with hd_def
  have hv : batchVar x ≥ 0 := by
    unfold batchVar
    apply div_nonneg
    · apply Finset.sum_nonneg; intro i _; exact sq_nonneg _
    · positivity
  have hd_pos : d > 0 := by
    rw [hd_def]; apply Real.sqrt_pos_of_pos; linarith
  have hd : d ≠ 0 := ne_of_gt hd_pos
  intro h
  -- cancel + β, multiply by d, cancel γ  ⇒  mu_run = batchMean x
  have h2 : gamma * (x i - mu_run) / d = gamma * (x i - batchMean x) / d := by
    linarith
  rw [div_eq_div_iff hd hd] at h2
  have h3 : gamma * (x i - mu_run) = gamma * (x i - batchMean x) := by
    have := mul_right_cancel₀ hd h2
    exact this
  have h4 : x i - mu_run = x i - batchMean x := by
    exact mul_left_cancel₀ hg h3
  apply hmu
  linarith

-- Tests
#check @batchnorm_eval_ne_train

end ProvableContracts.BatchNorm
