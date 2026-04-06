import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Sigmoid and SiLU Definitions

Mathematical definitions of sigmoid and SiLU (Sigmoid Linear Unit),
matching the `silu-kernel-v1.yaml` contract equations.

## References

- Ramachandran et al. (2017) Searching for Activation Functions
- Elfwing et al. (2018) Sigmoid-Weighted Linear Units
-/

namespace ProvableContracts.Sigmoid

open Real

/-- The sigmoid function: σ(x) = 1 / (1 + exp(-x)). -/
noncomputable def sigmoid (x : ℝ) : ℝ :=
  1 / (1 + Real.exp (-x))

/-- The SiLU activation: SiLU(x) = x · σ(x). -/
noncomputable def silu (x : ℝ) : ℝ :=
  x * sigmoid x

end ProvableContracts.Sigmoid
