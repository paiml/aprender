import ProvableContracts.Defs.Sampling

/-!
# Top-K Filtering Theorems (Structural)

Proves the analytic/structural content of the `top_k_top_p_interaction`
equation of `apr-cli-sampling-v1.yaml`: top-k keeps the k largest logits. We
model the filter by its defining cutoff threshold `τ` (the k-th largest
logit); the key structural facts are then threshold-free consequences:

* every kept token dominates every dropped token (`topk_separates`);
* `top_k = 0` (cutoff at/below the minimum) keeps every token — i.e. disables
  filtering (`topk_zero_keeps_all`);
* `top_k = 1` (cutoff at the maximum) keeps exactly the argmax, so it is
  equivalent to greedy decoding (`topk_one_is_argmax`).

## Obligation

`apr-cli-sampling-v1 / top_k_top_p_interaction`: top-k selects the k largest
logits; `top_k = 0` disables filtering; `top_k = 1` is equivalent to argmax.
-/

namespace ProvableContracts.Sampling

open ProvableContracts

/-- **Separation.** Any kept token has a strictly larger logit than any
    dropped token. This is the defining structural property of top-k: the
    survivors are exactly the largest logits. -/
theorem topk_separates {n : ℕ} (x : RVec n) (τ : ℝ) (i j : Fin n)
    (hi : kept x τ i) (hj : ¬ kept x τ j) :
    x j < x i := by
  unfold kept at hi hj
  push_neg at hj
  exact lt_of_lt_of_le hj hi

/-- **`top_k = 0` disables filtering.** If the cutoff `b` is at or below every
    logit (the `top_k = 0` / no-filter sentinel), then every token is kept, so
    the full vocabulary is considered. -/
theorem topk_zero_keeps_all {n : ℕ} (x : RVec n) (b : ℝ)
    (hb : ∀ i, b ≤ x i) (i : Fin n) :
    kept x b i := hb i

/-- **`top_k = 1` is argmax (greedy).** With the cutoff placed at the maximum
    logit `x m`, a token is kept iff its logit equals the maximum — i.e. the
    kept set is exactly the argmax set. Hence `top_k = 1` decoding coincides
    with greedy/argmax decoding. -/
theorem topk_one_is_argmax {n : ℕ} (x : RVec n) (m : Fin n)
    (hm : ∀ j, x j ≤ x m) (i : Fin n) :
    kept x (x m) i ↔ x i = x m := by
  unfold kept
  constructor
  · intro h; exact le_antisymm (hm i) h
  · intro h; exact h.ge

-- Tests
#check @topk_separates
#check @topk_zero_keeps_all
#check @topk_one_is_argmax

end ProvableContracts.Sampling
