import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import ProvableContracts.Basic

/-!
# LayerNorm Definitions

Mathematical definition of Layer Normalization, matching the
`layernorm-kernel-v1.yaml` contract equations.

## References

- Ba, Kiros & Hinton (2016) Layer Normalization
-/

namespace ProvableContracts.LayerNorm

open Finset

/-- Mean of a vector: μ = (1/d) · Σᵢ xᵢ. -/
noncomputable def mean {n : ℕ} (x : RVec (n + 1)) : ℝ :=
  univ.sum x / (n + 1 : ℝ)

/-- Variance: σ² = (1/d) · Σᵢ (xᵢ - μ)². -/
noncomputable def variance {n : ℕ} (x : RVec (n + 1)) : ℝ :=
  let mu := mean x
  univ.sum (fun i => (x i - mu) ^ 2) / (n + 1 : ℝ)

/-- LayerNorm denominator: √(σ² + ε). -/
noncomputable def ln_denom {n : ℕ} (x : RVec (n + 1)) (eps : ℝ) : ℝ :=
  Real.sqrt (variance x + eps)

/-- LayerNorm: γᵢ · (xᵢ - μ) / √(σ² + ε) + βᵢ. -/
noncomputable def layernorm {n : ℕ} (x : RVec (n + 1))
    (gamma beta : RVec (n + 1)) (eps : ℝ) : RVec (n + 1) :=
  let mu := mean x
  let denom := ln_denom x eps
  fun i => gamma i * (x i - mu) / denom + beta i

end ProvableContracts.LayerNorm
