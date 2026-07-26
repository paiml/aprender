import ProvableContracts.Defs.Softmax
import ProvableContracts.Theorems.Softmax.PartitionOfUnity
import ProvableContracts.Theorems.Softmax.NonNegativity

/-!
# GQA Attention-Weight Distribution

The per-query attention weights of grouped-query attention are, per key
position, the softmax of the scaled score row `Q_g · K_hᵀ / √d_k`. GQA changes
only *which* KV head (`K_h`, `V_h`) a query attends — the normalisation over
key positions is exactly softmax. Hence the distribution obligations discharge
directly from the shared softmax proofs.

Discharges `GQ-INV-001` (attention weight normalization): the attention
weights are a probability distribution — non-negative and summing to 1 per
query position.
-/

namespace ProvableContracts.Gqa

open ProvableContracts Finset

/-- **Normalization.** For any score row over `s+1` key positions, the GQA
    attention weights (softmax of the scores) sum to 1. Reuses the softmax
    partition-of-unity proof. -/
theorem attn_weights_sum_one {s : ℕ} (scores : RVec (s + 1)) :
    ∑ j : Fin (s + 1), Softmax.softmax scores j = 1 :=
  Softmax.partition_of_unity scores

/-- **Non-negativity.** Each GQA attention weight is strictly positive, so the
    weights form a genuine distribution together with `attn_weights_sum_one`. -/
theorem attn_weights_pos {s : ℕ} (scores : RVec (s + 1)) (j : Fin (s + 1)) :
    0 < Softmax.softmax scores j :=
  Softmax.softmax_pos scores j

#check @attn_weights_sum_one
#check @attn_weights_pos

end ProvableContracts.Gqa
