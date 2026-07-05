import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import ProvableContracts.Basic

/-!
# Forward-KL Knowledge-Distillation Loss Definitions

Mathematical definitions of the forward Kullback–Leibler divergence
`KL(p ‖ q) = Σ_i p_i · (log p_i − log q_i)` used as the soft-target
knowledge-distillation objective, matching the
`kd-loss-forward-kl-v1.yaml` contract equations.

The *forward* direction — with the teacher distribution `p_t` as the
outer measure — is the Hinton (2015) / PyTorch `nn.KLDivLoss` objective
`Σ p_t·(log p_t − log p_s)`.

## References

- Hinton, Vinyals & Dean (2015) *Distilling the Knowledge in a Neural
  Network* (arXiv:1503.02531).
- Kullback & Leibler (1951) *On Information and Sufficiency*.
- Gibbs' inequality: `KL(p ‖ q) ≥ 0` for probability distributions.
-/

namespace ProvableContracts.KLDivergence

open Real Finset

/-- Forward KL divergence `KL(p ‖ q) = Σ_i p_i · (log p_i − log q_i)`.

    In the distillation setting `p = p_t = softmax(teacher/T)` is the
    teacher distribution (the outer measure) and `q = p_s =
    softmax(student/T)` is the student distribution.  This is FORWARD KL
    `KL(teacher ‖ student)`, not reverse KL. -/
noncomputable def kl {n : ℕ} (p q : RVec n) : ℝ :=
  ∑ i : Fin n, p i * (Real.log (p i) - Real.log (q i))

end ProvableContracts.KLDivergence
