import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# CMA-ES Definitions

Definitions for the Covariance Matrix Adaptation Evolution Strategy
(Hansen 2016, *The CMA Evolution Strategy: A Tutorial*).

We capture the **analytic** core of the algorithm — the algebraic identities
and inequalities that hold at every generation independent of any RNG draw:

* `stepSizeUpdate`  — the CSA multiplicative step-size update `σ · exp(f)`.
* `normalizedWeights` — recombination weights normalized to a convex combination.
* `covUpdate` — the rank-one + rank-mu covariance update as a linear combination
  of a symmetric-PD matrix and two Gram (PSD) matrices.

The Gram representation `P * Pᵀ` for the rank-one term `p_c p_cᵀ` (take `P` a single
column) and for the rank-mu term `Σ wᵢ yᵢ yᵢᵀ = (Y√W)(Y√W)ᵀ` (with `wᵢ ≥ 0`) is
faithful: every positive-semidefinite update term arising in CMA-ES is a Gram matrix.
-/

namespace ProvableContracts.CMAES

open Matrix

/-- CSA step-size update: `σ_{t+1} = σ_t · exp(factor)` where
    `factor = (c_σ / d_σ)·(‖p_σ‖/E‖N(0,I)‖ − 1)`. The multiplicative
    `exp` form is what guarantees the step size never leaves `ℝ_{>0}`. -/
noncomputable def stepSizeUpdate (sigma factor : ℝ) : ℝ :=
  sigma * Real.exp factor

/-- Iterate the step-size update over a list of adaptation factors
    (one per generation). Models "σ after `factors.length` generations". -/
noncomputable def stepSizeIterate (sigma : ℝ) : List ℝ → ℝ
  | [] => sigma
  | f :: fs => stepSizeIterate (stepSizeUpdate sigma f) fs

/-- Normalized recombination weight `wᵢ = raw i / Σⱼ raw j`. -/
noncomputable def normalizedWeights {n : ℕ} (raw : Fin n → ℝ) (i : Fin n) : ℝ :=
  raw i / ∑ j, raw j

/-- The CMA-ES covariance update as a linear combination:

    `C_{t+1} = a · C_t + b · (P Pᵀ) + c · (Q Qᵀ)`

    with `a = 1 − c₁ − c_μ`, `b = c₁`, `c = c_μ`. `P Pᵀ` is the rank-one
    evolution-path term and `Q Qᵀ` the rank-mu term, both Gram (⇒ PSD). -/
noncomputable def covUpdate {n : ℕ} (a b c : ℝ)
    (C P Q : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  a • C + b • (P * Pᵀ) + c • (Q * Qᵀ)

end ProvableContracts.CMAES
