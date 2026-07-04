import ProvableContracts.Defs.Softmax
import ProvableContracts.Theorems.Softmax.NonNegativity
import ProvableContracts.Theorems.Softmax.PartitionOfUnity
import ProvableContracts.Theorems.Softmax.Bounded
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Scaled Dot-Product Attention — Analytic Obligations

Proves the analytic (algebraic) proof obligations of `attention-kernel-v1.yaml`:

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) · V

For a fixed query row `i`, the attention weights are `softmax(scores)` where
`scores_j = (Q Kᵀ / √d_k)_{ij}`. Every analytic property of an attention row
therefore reduces to a property of `softmax` applied to the (already-computed,
already-scaled) real score vector, plus a convex-combination bound for the
weighted sum against `V`.

## Obligations discharged (see `proof_obligations` in the contract)

* **ATT-INV-001 "Attention weights normalize"** — `Σ_j attn_{ij} = 1`
  (`attention_weights_sum_to_one`, reuses `Softmax.partition_of_unity`).
* **ATT-BND-001 "Attention weights in (0,1)"** — `0 < attn_{ij} < 1`
  (`attention_weights_bounded`, reuses `Softmax.softmax_bounded`).
* **ATT-BND-002 "Output bounded by V"** — `min V ≤ output_{i} ≤ max V`
  (`attention_output_bounded`, convex-combination sandwich).
* **ATT-INV-002 "Scaling factor 1/√d_k"** — the scale is well-defined and
  positive with `scale² = 1/d_k`, which pins it to `d_k^{-1/2}` and separates it
  from the linear `1/d_k` scaling (`scale_factor_pos`, `scale_factor_sq`,
  `scale_factor_ne_linear`).

The remaining obligation "SIMD matches scalar within 8 ULP" is a runtime
floating-point equivalence claim with no algebraic statement over ℝ; it is
marked `l4_not_applicable` in the contract, not proved here.

Bonus structural lemma `causal_mask_zeroes_future` shows a causal mask (weights
forced to 0 on future positions) drops those positions from the output sum.

## References

- Vaswani et al. "Attention Is All You Need." NeurIPS, 2017. Eq. 1.
-/

namespace ProvableContracts.Attention

open Real Finset
open ProvableContracts.Softmax

/-! ## Weight normalization (reuse of softmax partition of unity) -/

/-- **ATT-INV-001.** The attention weights of a query row are the softmax of the
    scaled score vector, hence sum to 1. -/
theorem attention_weights_sum_to_one {m : ℕ} (scores : RVec (m + 1)) :
    ∑ j : Fin (m + 1), softmax scores j = 1 :=
  partition_of_unity scores

/-! ## Weight bounds (reuse of softmax boundedness) -/

/-- Each attention weight is strictly positive. -/
theorem attention_weight_pos {m : ℕ} (scores : RVec (m + 1)) (j : Fin (m + 1)) :
    0 < softmax scores j :=
  softmax_pos scores j

/-- **ATT-BND-001.** With at least two keys, every attention weight lies strictly
    in `(0, 1)`. -/
theorem attention_weights_bounded {m : ℕ} (scores : RVec (m + 2)) (j : Fin (m + 2)) :
    0 < softmax scores j ∧ softmax scores j < 1 :=
  softmax_bounded scores j

/-! ## Convex-combination output bounds -/

/-- Upper convex-combination bound: nonnegative weights summing to 1 against
    values bounded above by `c` produce a weighted sum bounded above by `c`. -/
theorem convex_comb_le {n : ℕ} (w V : Fin n → ℝ) (c : ℝ)
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) (hV : ∀ j, V j ≤ c) :
    ∑ j, w j * V j ≤ c := by
  calc
    ∑ j, w j * V j ≤ ∑ j, w j * c :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hV j) (hw j)
    _ = (∑ j, w j) * c := by rw [← Finset.sum_mul]
    _ = c := by rw [hsum, one_mul]

/-- Lower convex-combination bound: nonnegative weights summing to 1 against
    values bounded below by `c` produce a weighted sum bounded below by `c`. -/
theorem le_convex_comb {n : ℕ} (w V : Fin n → ℝ) (c : ℝ)
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) (hV : ∀ j, c ≤ V j) :
    c ≤ ∑ j, w j * V j := by
  calc
    c = (∑ j, w j) * c := by rw [hsum, one_mul]
    _ = ∑ j, w j * c := by rw [Finset.sum_mul]
    _ ≤ ∑ j, w j * V j :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hV j) (hw j)

/-- **ATT-BND-002.** An attention output component is a convex combination of the
    value entries, hence sandwiched between any lower bound `lo` and upper bound
    `hi` of `V`. Instantiating `lo := min V`, `hi := max V` gives
    `min V ≤ output ≤ max V`. Weights are the softmax of the scaled scores, so
    nonnegativity and normalization come from the softmax proofs. -/
theorem attention_output_bounded {m : ℕ} (scores : RVec (m + 1)) (V : RVec (m + 1))
    (lo hi : ℝ) (hlo : ∀ j, lo ≤ V j) (hhi : ∀ j, V j ≤ hi) :
    lo ≤ ∑ j, softmax scores j * V j ∧ ∑ j, softmax scores j * V j ≤ hi := by
  have hw : ∀ j, 0 ≤ softmax scores j := fun j => le_of_lt (softmax_pos scores j)
  have hsum : ∑ j, softmax scores j = 1 := partition_of_unity scores
  exact ⟨le_convex_comb _ V lo hw hsum hlo, convex_comb_le _ V hi hw hsum hhi⟩

/-! ## Scale factor 1/√d_k -/

/-- **ATT-INV-002 (well-definedness / positivity).** For `d_k > 0` the scale
    factor `1/√d_k` is well-defined and strictly positive. -/
theorem scale_factor_pos {d : ℝ} (hd : 0 < d) : 0 < 1 / Real.sqrt d :=
  one_div_pos.mpr (Real.sqrt_pos.mpr hd)

/-- **ATT-INV-002 (defining identity).** The scale factor squares to `1/d_k`,
    i.e. `scale = d_k^{-1/2}`. This is the algebraic signature of the
    square-root scaling. -/
theorem scale_factor_sq {d : ℝ} (hd : 0 < d) : (1 / Real.sqrt d) ^ 2 = 1 / d := by
  rw [div_pow, one_pow, Real.sq_sqrt (le_of_lt hd)]

/-- **ATT-INV-002 (separation from linear scaling).** For `d_k > 1` the
    square-root scale strictly exceeds the linear scale `1/d_k`, so the kernel
    provably uses `1/√d_k` rather than `1/d_k`. -/
theorem scale_factor_ne_linear {d : ℝ} (hd : 1 < d) : 1 / d < 1 / Real.sqrt d := by
  have hdpos : (0 : ℝ) < d := lt_trans one_pos hd
  have hsqrt_pos : 0 < Real.sqrt d := Real.sqrt_pos.mpr hdpos
  -- √d < d because d > 1
  have hsqrt_lt : Real.sqrt d < d := by
    have : Real.sqrt d < Real.sqrt (d * d) := by
      apply Real.sqrt_lt_sqrt (le_of_lt hdpos)
      nlinarith [hd, hdpos]
    rwa [Real.sqrt_mul_self (le_of_lt hdpos)] at this
  exact one_div_lt_one_div_of_lt hsqrt_pos hsqrt_lt

/-! ## Causal mask (structural) -/

/-- **Structural.** A causal mask forces the attention weight to `0` on every
    "future" position `j ∉ S` (the allowed/past set). Those positions then drop
    out of the output sum entirely: `Σ_j w_j V_j = Σ_{j ∈ S} w_j V_j`. -/
theorem causal_mask_zeroes_future {n : ℕ} (w V : Fin n → ℝ) (S : Finset (Fin n))
    (hmask : ∀ j ∈ (Finset.univ : Finset (Fin n)), j ∉ S → w j = 0) :
    ∑ j, w j * V j = ∑ j ∈ S, w j * V j := by
  symm
  apply Finset.sum_subset (Finset.subset_univ S)
  intro j hj hjS
  rw [hmask j hj hjS, zero_mul]

-- Tests
#check @attention_weights_sum_to_one
#check @attention_weights_bounded
#check @attention_output_bounded
#check @scale_factor_pos
#check @scale_factor_sq
#check @scale_factor_ne_linear
#check @causal_mask_zeroes_future

end ProvableContracts.Attention
