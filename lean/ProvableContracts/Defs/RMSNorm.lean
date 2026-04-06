import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import ProvableContracts.Basic

/-!
# RMSNorm Definitions

Mathematical definition of Root Mean Square Layer Normalization,
matching the `rmsnorm-kernel-v1.yaml` contract equations.

## References

- Zhang & Sennrich (2019) Root Mean Square Layer Normalization
- Touvron et al. (2023) Llama 2: Open Foundation and Fine-Tuned Chat Models
-/

namespace ProvableContracts.RMSNorm

open Finset

/-- Mean of squares: (1/n) · Σᵢ xᵢ². -/
noncomputable def mean_sq {n : ℕ} (x : RVec (n + 1)) : ℝ :=
  univ.sum (fun i => x i ^ 2) / (n + 1 : ℝ)

/-- RMS denominator: √(mean(x²) + ε). -/
noncomputable def rms {n : ℕ} (x : RVec (n + 1)) (eps : ℝ) : ℝ :=
  Real.sqrt (mean_sq x + eps)

/-- RMSNorm: xᵢ / RMS(x) · γᵢ. -/
noncomputable def rmsnorm {n : ℕ} (x : RVec (n + 1)) (gamma : RVec (n + 1))
    (eps : ℝ) : RVec (n + 1) :=
  fun i => x i / rms x eps * gamma i

end ProvableContracts.RMSNorm
