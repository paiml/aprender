import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt

/-!
# AdamW Optimizer Definitions

Mathematical definition of the AdamW weight update rule with
decoupled weight decay.

## References

- Loshchilov & Hutter (2019) Decoupled Weight Decay Regularization
- Kingma & Ba (2015) Adam: A Method for Stochastic Optimization
-/

namespace ProvableContracts.AdamW

/-- AdamW update rule (simplified single-parameter form).
    θ_{t+1} = θ_t - lr * (m_hat / (√v_hat + ε) + λ * θ_t)
    where the weight decay term λ·θ is decoupled from the gradient. -/
noncomputable def adamw_update (theta : ℝ) (m_hat : ℝ) (v_hat : ℝ)
    (lr : ℝ) (eps : ℝ) (wd : ℝ) : ℝ :=
  theta - lr * (m_hat / (Real.sqrt v_hat + eps) + wd * theta)

/-- The weight-decay-only update (ignoring gradient term).
    θ_{t+1} = θ_t - lr * λ * θ_t = (1 - lr·λ) · θ_t. -/
noncomputable def decay_update (theta : ℝ) (lr : ℝ) (wd : ℝ) : ℝ :=
  theta - lr * wd * theta

end ProvableContracts.AdamW
