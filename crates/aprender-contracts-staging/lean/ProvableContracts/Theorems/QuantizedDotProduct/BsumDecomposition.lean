import Mathlib.Algebra.BigOperators.Group.List.Basic

/-!
# Bsum Precomputation Decomposition

Models the `bsum` (sub-block activation sum) precomputation used by the
K-quant fused dot-product kernels (Q4_K / Q5_K, `has_dmin = true`).

A super-block's activations are partitioned into contiguous sub-blocks.
The offset term of the dot product needs, for each sub-block `j`, the integer
sum `bsumⱼ = Σ actᵢ` over that sub-block.  The kernel may either

* **precompute** all sub-block sums up front (once, reused across weight rows), or
* **inline** each sub-block sum on the fly inside the super-block loop.

The contract obligation `QDOT bsum decomposition` states these two strategies
agree exactly (integer arithmetic, no rounding).  We model a partition as a
`List (List ℤ)` (`chunks`) and prove:

1. `precompute` (index into the map of per-chunk sums) equals `inline`
   (sum the indexed chunk) at every sub-block — the literal
   `precompute == inline` obligation; and
2. summing the precomputed bsums equals summing the whole flattened activation
   vector — the decomposition is sound and order-independent.

Both are `analytic` (exact integer identities), no runtime measurement.
-/

namespace ProvableContracts.QuantizedDotProduct

/-- Per-sub-block agreement: the `j`-th **precomputed** bsum (obtained by first
mapping `List.sum` over every chunk, then indexing) equals the `j`-th
**inline** bsum (obtained by indexing the chunk, then summing).  This is the
literal `precompute_bsums == inline_bsums` obligation, per element. -/
theorem bsum_precompute_eq_inline
    (chunks : List (List ℤ)) (j : ℕ) (hj : j < chunks.length) :
    (chunks.map List.sum)[j]'(by simpa using hj) = (chunks[j]).sum := by
  simp

/-- Decomposition soundness: the sum of the precomputed sub-block bsums equals
the sum over the whole flattened activation vector.  This certifies that
partitioning the activations into sub-blocks and pre-summing each one loses
nothing — the offset term is exact and independent of sub-block boundaries. -/
theorem bsum_decomposition_total (chunks : List (List ℤ)) :
    (chunks.map List.sum).sum = chunks.flatten.sum := by
  induction chunks with
  | nil => simp
  | cons c cs ih =>
      simp only [List.map_cons, List.sum_cons, List.flatten_cons, List.sum_append, ih]

#check @bsum_precompute_eq_inline
#check @bsum_decomposition_total

end ProvableContracts.QuantizedDotProduct
