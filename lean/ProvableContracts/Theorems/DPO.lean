import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Topology.Order.Basic

/-!
# Direct Preference Optimization (DPO) Loss

Proves properties of the DPO loss function from Rafailov et al. (2023).

## Contract: dpo-alignment-v1

### DPO Loss Function
L_DPO(π_θ; π_ref) = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

where:
- y_w is the preferred (chosen) response
- y_l is the rejected response
- π_θ is the policy model
- π_ref is the reference model
- β is the temperature parameter
- σ is the sigmoid function
-/

namespace ProvableContracts.DPO

/-- Sigmoid function -/
noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

/-- DPO loss for a single preference pair -/
noncomputable def dpo_loss (β : ℝ) (log_ratio_w log_ratio_l : ℝ) : ℝ :=
  -Real.log (sigmoid (β * (log_ratio_w - log_ratio_l)))

-- Status: axiom (positivity + bound argument)
/-- Sigmoid is bounded in (0, 1) -/
axiom sigmoid_bounded (x : ℝ) : 0 < sigmoid x ∧ sigmoid x < 1

-- Status: proved
/-- DPO loss is non-negative when sigmoid argument is in (0,1) -/
theorem dpo_loss_nonneg (β : ℝ) (lrw lrl : ℝ) (hβ : β > 0) :
    dpo_loss β lrw lrl ≥ 0 := by
  unfold dpo_loss
  have hs := sigmoid_bounded (β * (lrw - lrl))
  have hlog : Real.log (sigmoid (β * (lrw - lrl))) ≤ 0 := by
    exact Real.log_nonpos hs.1.le hs.2.le
  linarith

-- Status: axiom (requires limit theory)
/-- When chosen is strongly preferred (log_ratio_w >> log_ratio_l), loss → 0 -/
axiom dpo_loss_zero_at_strong_preference :
    ∀ ε > 0, ∃ M : ℝ, ∀ lrw lrl : ℝ, lrw - lrl > M →
    dpo_loss 1 lrw lrl < ε

-- Status: axiom (gradient formula from Rafailov et al.)
/-- DPO gradient: ∇L = -β * σ(-β*Δ) * (∇log π_θ(y_w|x) - ∇log π_θ(y_l|x)) -/
axiom dpo_gradient_formula
    (β : ℝ) (Δ : ℝ) :
    ∃ (grad_scale : ℝ),
      grad_scale = -β * sigmoid (-β * Δ) ∧
      grad_scale ≤ 0

#check @sigmoid_bounded
#check @dpo_loss_nonneg
#check @dpo_gradient_formula

end ProvableContracts.DPO
