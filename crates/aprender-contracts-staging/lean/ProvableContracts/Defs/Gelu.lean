import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# GELU Definition

Mathematical definition of the GELU (Gaussian Error Linear Unit) activation
using the tanh approximation, matching the `activation-kernel-v1.yaml`
contract equation:

  GELU(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))

## References

- Hendrycks & Gimpel (2016) Gaussian Error Linear Units (GELUs)
-/

namespace ProvableContracts.Gelu

open Real

/-- GELU tanh approximation:
    `GELU(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`. -/
noncomputable def gelu (x : ℝ) : ℝ :=
  0.5 * x * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x ^ 3)))

end ProvableContracts.Gelu
