import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import ProvableContracts.Basic

/-!
# BatchNorm Definitions

Mathematical definition of Batch Normalization (per channel, across the
batch dimension), matching the `batchnorm-kernel-v1.yaml` contract
equations.

A single channel over a batch of `N = n + 1` samples is an `RVec (n+1)`.
The per-channel batch statistics and normalization are structurally the
same reduction as LayerNorm, but taken over the batch axis rather than
the feature axis.

## References

- Ioffe & Szegedy (2015) Batch Normalization: Accelerating Deep Network
  Training by Reducing Internal Covariate Shift
-/

namespace ProvableContracts.BatchNorm

open Finset

/-- Batch mean for a single channel: μ_B = (1/N)·Σₙ xₙ. -/
noncomputable def batchMean {n : ℕ} (x : RVec (n + 1)) : ℝ :=
  univ.sum x / (n + 1 : ℝ)

/-- Batch variance: σ²_B = (1/N)·Σₙ (xₙ - μ_B)². -/
noncomputable def batchVar {n : ℕ} (x : RVec (n + 1)) : ℝ :=
  let mu := batchMean x
  univ.sum (fun i => (x i - mu) ^ 2) / (n + 1 : ℝ)

/-- BatchNorm denominator: √(σ²_B + ε). -/
noncomputable def bn_denom {n : ℕ} (x : RVec (n + 1)) (eps : ℝ) : ℝ :=
  Real.sqrt (batchVar x + eps)

/-- BatchNorm (training) per channel over the batch:
    BN(x)ₙ = γ·(xₙ - μ_B)/√(σ²_B + ε) + β. -/
noncomputable def batchnorm {n : ℕ} (x : RVec (n + 1))
    (gamma beta eps : ℝ) : RVec (n + 1) :=
  fun i => gamma * (x i - batchMean x) / bn_denom x eps + beta

/-- BatchNorm (eval) per channel, using running statistics rather than
    batch statistics: BN_eval(x)ₙ = γ·(xₙ - μ_run)/√(σ_run + ε) + β. -/
noncomputable def batchnorm_eval {n : ℕ} (x : RVec (n + 1))
    (mu_run sigma_run gamma beta eps : ℝ) : RVec (n + 1) :=
  fun i => gamma * (x i - mu_run) / Real.sqrt (sigma_run + eps) + beta

/-- One exponential-moving-average update step:
    s' = (1 - m)·s_prev + m·s_batch. -/
noncomputable def ema_step (prev batch m : ℝ) : ℝ :=
  (1 - m) * prev + m * batch

/-- Iterated EMA update over a sequence of batch statistics. -/
noncomputable def ema_fold (init m : ℝ) (batches : List ℝ) : ℝ :=
  batches.foldl (fun acc b => ema_step acc b m) init

end ProvableContracts.BatchNorm
