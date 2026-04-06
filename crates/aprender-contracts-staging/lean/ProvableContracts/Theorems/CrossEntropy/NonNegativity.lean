import ProvableContracts.Defs.CrossEntropy
import ProvableContracts.Theorems.CrossEntropy.LogSoftmaxBound

/-!
# Cross-Entropy Non-Negativity

Proves that cross-entropy CE(t, x) ≥ 0 when t is a probability
distribution (tᵢ ≥ 0).

## Obligation

`CE-INV-001`: CE(targets, logits) ≥ 0

Since log_softmax(x)ᵢ ≤ 0 and tᵢ ≥ 0, each product tᵢ·log_softmax ≤ 0,
so the sum is ≤ 0, and negation gives ≥ 0.
-/

namespace ProvableContracts.CrossEntropy

open Finset

-- Status: proved
/-- Cross-entropy is non-negative when targets are non-negative.
    Since each tᵢ ≥ 0 and log_softmax ≤ 0, the sum is ≤ 0,
    so -sum ≥ 0. -/
theorem cross_entropy_nonneg {n : ℕ} (targets : RVec (n + 1)) (logits : RVec (n + 1))
    (ht : ∀ i, targets i ≥ 0) :
    cross_entropy targets logits ≥ 0 := by
  simp only [cross_entropy, neg_nonneg]
  apply Finset.sum_nonpos
  intro i _
  exact mul_nonpos_of_nonneg_of_nonpos (ht i) (log_softmax_le_zero logits i)

-- Tests
#check @cross_entropy_nonneg

end ProvableContracts.CrossEntropy
