import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
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

-- Status: proved
/-- Sigmoid is bounded in (0, 1): the denominator `1 + exp(-x)` exceeds 1 because `exp(-x) > 0`. -/
theorem sigmoid_bounded (x : ℝ) : 0 < sigmoid x ∧ sigmoid x < 1 := by
  have he := Real.exp_pos (-x)
  unfold sigmoid
  refine ⟨div_pos one_pos (by linarith), ?_⟩
  rw [div_lt_one (by linarith)]
  linarith

-- Status: proved
/-- DPO loss is non-negative when sigmoid argument is in (0,1) -/
theorem dpo_loss_nonneg (β : ℝ) (lrw lrl : ℝ) (_hβ : β > 0) :
    dpo_loss β lrw lrl ≥ 0 := by
  unfold dpo_loss
  have hs := sigmoid_bounded (β * (lrw - lrl))
  have hlog : Real.log (sigmoid (β * (lrw - lrl))) ≤ 0 := by
    exact Real.log_nonpos hs.1.le hs.2.le
  linarith

/-- The loss in closed form: `-log σ(z) = log (1 + exp (-z))`. -/
theorem dpo_loss_eq_log (β lrw lrl : ℝ) :
    dpo_loss β lrw lrl = Real.log (1 + Real.exp (-(β * (lrw - lrl)))) := by
  unfold dpo_loss sigmoid
  rw [one_div, Real.log_inv, neg_neg]

-- Status: proved
/-- When chosen is strongly preferred (log_ratio_w >> log_ratio_l), loss → 0.
    Witness: `M = -log (exp ε - 1)`; past it `1 + exp(-Δ) < exp ε`, so the loss is below `ε`. -/
theorem dpo_loss_zero_at_strong_preference :
    ∀ ε > 0, ∃ M : ℝ, ∀ lrw lrl : ℝ, lrw - lrl > M →
    dpo_loss 1 lrw lrl < ε := by
  intro ε hε
  have hpos : 0 < Real.exp ε - 1 := by
    have := Real.add_one_lt_exp (ne_of_gt hε)
    linarith
  refine ⟨-Real.log (Real.exp ε - 1), fun lrw lrl h => ?_⟩
  rw [dpo_loss_eq_log, one_mul]
  have hlt : Real.exp (-(lrw - lrl)) < Real.exp ε - 1 := by
    rw [← Real.exp_log hpos]
    exact Real.exp_lt_exp.mpr (by linarith)
  have h1 : 0 < 1 + Real.exp (-(lrw - lrl)) := by linarith [Real.exp_pos (-(lrw - lrl))]
  rw [Real.log_lt_iff_lt_exp h1]
  linarith

open Filter Topology in
-- Status: proved
/-- The same limit as a `Tendsto`: with β = 1, the loss tends to 0 as the preference margin grows. -/
theorem dpo_loss_tendsto_zero :
    Tendsto (fun Δ : ℝ => dpo_loss 1 Δ 0) atTop (𝓝 0) := by
  rw [Metric.tendsto_atTop]
  intro ε hε
  obtain ⟨M, hM⟩ := dpo_loss_zero_at_strong_preference ε hε
  refine ⟨M + 1, fun Δ hΔ => ?_⟩
  have hlt := hM Δ 0 (by linarith)
  have hnn := dpo_loss_nonneg 1 Δ 0 one_pos
  rw [Real.dist_eq, sub_zero, abs_of_nonneg hnn]
  exact hlt

-- Status: proved
/-- DPO gradient (Rafailov et al.): `d/dΔ [-log σ(βΔ)] = -β * σ(-βΔ)`. The chain rule through
    `log (1 + exp (-βΔ))`, then `exp(-a) / (1 + exp(-a)) = σ(-a)`. -/
theorem dpo_gradient_formula (β Δ : ℝ) :
    HasDerivAt (fun d => dpo_loss β d 0) (-β * sigmoid (-β * Δ)) Δ := by
  have hfun : (fun d => dpo_loss β d 0) = fun d => Real.log (1 + Real.exp (-(β * d))) := by
    funext d
    rw [dpo_loss_eq_log, sub_zero]
  rw [hfun]
  have hpos : 0 < 1 + Real.exp (-(β * Δ)) := by linarith [Real.exp_pos (-(β * Δ))]
  have hd := (((hasDerivAt_id Δ).const_mul β).neg.exp.const_add 1).log hpos.ne'
  convert hd using 1
  simp only [Pi.neg_apply, id]
  unfold sigmoid
  have hE := Real.exp_pos (β * Δ)
  rw [show -(-β * Δ) = β * Δ by ring, Real.exp_neg]
  field_simp
  ring

-- Status: proved
/-- For β ≥ 0 the gradient scale is non-positive. (The old axiom asserted this for every β, which is false
    for β < 0: the scale `-β * σ(-βΔ)` is then positive.) -/
theorem dpo_gradient_scale_nonpos (β Δ : ℝ) (hβ : 0 ≤ β) : -β * sigmoid (-β * Δ) ≤ 0 := by
  have := (sigmoid_bounded (-β * Δ)).1
  nlinarith

#check @sigmoid_bounded
#check @dpo_loss_nonneg
#check @dpo_gradient_formula
#check @dpo_loss_tendsto_zero
#check @dpo_gradient_scale_nonpos

end ProvableContracts.DPO
