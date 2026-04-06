import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import ProvableContracts.Basic
import ProvableContracts.Defs.Softmax

/-!
# Cross-Entropy Definitions

Mathematical definitions of log-softmax and cross-entropy loss,
matching the `cross-entropy-kernel-v1.yaml` contract equations.

## References

- Shannon (1948) A Mathematical Theory of Communication
- Milakov & Gimelshein (2018) Online normalizer calculation for softmax
-/

namespace ProvableContracts.CrossEntropy

open Real Finset

/-- Log-softmax: log(exp(xᵢ)/Z) = xᵢ - log(Z). -/
noncomputable def log_softmax {n : ℕ} (x : RVec n) (i : Fin n) : ℝ :=
  x i - Real.log (univ.sum (fun j => Real.exp (x j)))

/-- Cross-entropy loss: CE(t, x) = -Σᵢ tᵢ · log_softmax(x)ᵢ. -/
noncomputable def cross_entropy {n : ℕ} (targets : RVec n) (logits : RVec n) : ℝ :=
  -(univ.sum (fun i => targets i * log_softmax logits i))

end ProvableContracts.CrossEntropy
