import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# AdamW Weight Update — decoupled weight decay

Contract: `adamw-kernel-v1`, equation `weight_update`.

`θ_t = θ_{t-1} - lr·(m̂/(√v̂+ε) + λ·θ_{t-1})`. This file proves the *decoupling*
identity that defines AdamW (vs. L2-regularised Adam): the update splits
additively into the pure Adam gradient step `θ - lr·adam_step` and a separate
weight-decay term `lr·λ·θ`. The decay acts on `θ` directly, not through the
gradient — the contract's "Weight decay applied AFTER Adam update (decoupled)"
invariant.
-/

namespace ProvableContracts.AdamW.WeightUpdate

/-- Full AdamW step. `adam_step` abstracts `m̂/(√v̂+ε)`. -/
noncomputable def adamw_update (theta adam_step lr wd : ℝ) : ℝ :=
  theta - lr * (adam_step + wd * theta)

-- Status: proved (core algebraic)
/-- Decoupling identity: the AdamW update equals the pure Adam gradient step
    minus an independent weight-decay term `lr·λ·θ`. -/
theorem weight_update (theta adam_step lr wd : ℝ) :
    adamw_update theta adam_step lr wd
      = (theta - lr * adam_step) - lr * wd * theta := by
  unfold adamw_update; ring

#check @weight_update

end ProvableContracts.AdamW.WeightUpdate
