import ProvableContracts.Defs.Conv1D
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.MeanInequalities

/-!
# Conv1d Output Bound

Proves the contract's magnitude bound for 1-D convolution output:

  `|y[n]| ≤ K · max|w| · max|x| (+ |bias|)`.

## Obligation

`CV-BND-001` (contract `conv1d-kernel-v1.yaml`, proof obligation
"Output bounded by input and kernel"):
`|y[n]| ≤ C_in · K · max(|w|) · max(|x|) + |bias|`.

We prove the single-channel (`C_in = 1`), zero-bias case:
`|conv w x n| ≤ w.length · Wmax · Xmax`, given uniform bounds
`|w[k]| ≤ Wmax` and `|x[m]| ≤ Xmax`. The multi-channel form is a finite sum
of `C_in` such terms, so the bound lifts with the extra `C_in` factor, and a
bias term adds `|bias|` by the triangle inequality.
-/

namespace ProvableContracts.Conv1D

open Finset

-- Status: proved
/-- **Output bound.** With `|w[k]| ≤ Wmax` for every tap and `|x[m]| ≤ Xmax`
for every sample, the convolution output at any position is bounded by
`w.length · Wmax · Xmax`. -/
theorem conv_abs_le (w : List ℝ) (x : Signal) (n : ℕ) (Wmax Xmax : ℝ)
    (hW : ∀ k, |w.getD k 0| ≤ Wmax) (hX : ∀ m, |x m| ≤ Xmax) :
    |conv w x n| ≤ w.length * Wmax * Xmax := by
  unfold conv
  calc |∑ k ∈ Finset.range w.length, w.getD k 0 * x (n + k)|
      ≤ ∑ k ∈ Finset.range w.length, |w.getD k 0 * x (n + k)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _k ∈ Finset.range w.length, Wmax * Xmax := by
        apply Finset.sum_le_sum
        intro k _
        rw [abs_mul]
        have h1 : |w.getD k 0| ≤ Wmax := hW k
        have h2 : |x (n + k)| ≤ Xmax := hX (n + k)
        exact mul_le_mul h1 h2 (abs_nonneg _) (le_trans (abs_nonneg _) h1)
    _ = w.length * (Wmax * Xmax) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
    _ = w.length * Wmax * Xmax := by ring

#check @conv_abs_le

end ProvableContracts.Conv1D
