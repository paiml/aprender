import ProvableContracts.Defs.Softmax
import ProvableContracts.Theorems.Softmax.PartitionOfUnity
import ProvableContracts.Theorems.Softmax.NonNegativity
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# GQA Output Convex-Combination Bound

The attention output for a query position is the weighted sum
`Σ_j softmax(scores)_j · V_j` of the (shared) KV-head value rows `V_j`. Because
the softmax weights are non-negative and sum to 1, the output is a convex
combination of the value rows and is therefore bounded, coordinate-wise, by the
range of `V`.

Discharges `GQ-BND-001` (output is a convex combination of V):
`min(V) ≤ output_i ≤ max(V)` per head.
-/

namespace ProvableContracts.Gqa

open ProvableContracts Finset

/-- **General convex-combination bound.** If weights `w` are non-negative and
    sum to 1, then any weighted average `Σ w_j v_j` lies between a lower bound
    `lo ≤ v_j` and an upper bound `v_j ≤ hi`. -/
theorem convex_combination_bounds {s : ℕ} (w v : RVec s)
    (hw_nonneg : ∀ j, 0 ≤ w j) (hw_sum : ∑ j, w j = 1)
    (lo hi : ℝ) (hlo : ∀ j, lo ≤ v j) (hhi : ∀ j, v j ≤ hi) :
    lo ≤ ∑ j, w j * v j ∧ ∑ j, w j * v j ≤ hi := by
  have hlo_eq : ∑ j, w j * lo = lo := by
    rw [← Finset.sum_mul, hw_sum, one_mul]
  have hhi_eq : ∑ j, w j * hi = hi := by
    rw [← Finset.sum_mul, hw_sum, one_mul]
  refine ⟨?_, ?_⟩
  · rw [← hlo_eq]
    exact Finset.sum_le_sum fun j _ =>
      mul_le_mul_of_nonneg_left (hlo j) (hw_nonneg j)
  · rw [← hhi_eq]
    exact Finset.sum_le_sum fun j _ =>
      mul_le_mul_of_nonneg_left (hhi j) (hw_nonneg j)

/-- **GQA output bound.** The softmax-weighted sum of value rows is bounded by
    the value range `[lo, hi]` — the attention output is a convex combination
    of `V`. -/
theorem gqa_output_convex {s : ℕ} (scores v : RVec (s + 1)) (lo hi : ℝ)
    (hlo : ∀ j, lo ≤ v j) (hhi : ∀ j, v j ≤ hi) :
    lo ≤ ∑ j, Softmax.softmax scores j * v j ∧
      ∑ j, Softmax.softmax scores j * v j ≤ hi :=
  convex_combination_bounds (Softmax.softmax scores) v
    (fun j => le_of_lt (Softmax.softmax_pos scores j))
    (Softmax.partition_of_unity scores) lo hi hlo hhi

#check @convex_combination_bounds
#check @gqa_output_convex

end ProvableContracts.Gqa
