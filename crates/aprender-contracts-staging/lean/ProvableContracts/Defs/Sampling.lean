import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import ProvableContracts.Basic

/-!
# Token Sampling Definitions

Mathematical definitions for the CLI token-sampling pipeline, matching the
`apr-cli-sampling-v1.yaml` contract equations: temperature scaling, top-k
filtering (structural), and the repeat-penalty transform.

These are the ANALYTIC core of the sampling contract. Runtime/empirical
obligations (RNG bit-determinism across process runs, IEEE-754 NaN/inf
rejection, the f32-exact `[0,1)` unit-draw construction, and CLI process
exit codes) are NOT modelled here — Lean's `ℝ` has no NaN/inf and no f32
rounding, so those obligations are genuinely not-applicable to a real-number
proof and stay outside the Lean layer.

## References

- Fan et al. "Hierarchical Neural Story Generation." ACL 2018 (top-k).
- Holtzman et al. "The Curious Case of Neural Text Degeneration." ICLR 2020.
-/

namespace ProvableContracts.Sampling

open Finset

/-- Temperature scaling of a logit vector: `logits_t(i) = x(i) / t`.
    For `t > 0` this is a strictly-increasing rescaling, so it preserves the
    relative order of logits (and hence the argmax). -/
noncomputable def tempScale {n : ℕ} (x : RVec n) (t : ℝ) : RVec n :=
  fun i => x i / t

/-- Top-k "kept" membership, modelled structurally by a cutoff threshold `τ`:
    a token `i` survives the filter iff its logit is at least the cutoff.
    The k-th largest logit plays the role of `τ`; `top_k = 0` corresponds to a
    cutoff at or below the minimum logit (everything survives), and `top_k = 1`
    to a cutoff at the maximum (only the argmax survives). -/
def kept {n : ℕ} (x : RVec n) (τ : ℝ) (i : Fin n) : Prop :=
  τ ≤ x i

/-- The repeat-penalty transform on a single logit `l` with factor `p`
    (`llama.cpp` convention): positive logits are divided by `p`, negative
    logits are multiplied by `p`, zero is left unchanged. -/
noncomputable def applyPenalty (l p : ℝ) : ℝ :=
  if 0 < l then l / p else if l < 0 then l * p else l

end ProvableContracts.Sampling
