import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import ProvableContracts.Basic

/-!
# Softmax Definitions

Mathematical definition of softmax over real-valued vectors,
matching the `softmax-kernel-v1.yaml` contract equations.

## References

- Bridle, J.S. "Training Stochastic Model Recognition Algorithms
  as Networks can Lead to Maximum Mutual Information Estimation
  of Parameters." NeurIPS, 1990.
- Vaswani et al. "Attention Is All You Need." NeurIPS, 2017. Eq. 3.
-/

namespace ProvableContracts.Softmax

open Real Finset

/-- The softmax function: `softmax(x)_i = exp(x_i) / Σ_j exp(x_j)`. -/
noncomputable def softmax {n : ℕ} (x : RVec n) (i : Fin n) : ℝ :=
  Real.exp (x i) / ∑ j : Fin n, Real.exp (x j)

/-- Log-softmax: `log_softmax(x)_i = x_i - log(Σ_j exp(x_j))`. -/
noncomputable def log_softmax {n : ℕ} (x : RVec n) (i : Fin n) : ℝ :=
  x i - Real.log (∑ j : Fin n, Real.exp (x j))

end ProvableContracts.Softmax
