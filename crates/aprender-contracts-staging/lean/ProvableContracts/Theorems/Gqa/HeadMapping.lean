import Mathlib
import ProvableContracts.Basic

/-!
# GQA Head-Mapping Theorems

Analytic (integer-arithmetic) content of grouped-query attention routing,
backing `gqa-kernel-v1.yaml`.

GQA partitions the `num_heads` query heads into `num_kv_heads` contiguous
groups of size `r = num_heads / num_kv_heads` (the *group size*). Every query
head in group `g` attends the single shared KV head `g`. The routing function
is integer division `kvHead r q = q / r`, with
`num_heads = num_kv_heads * r` guaranteed by the divisibility precondition
(`num_heads % num_kv_heads == 0`).

Discharges:

- `GQ-INV-002` (KV head broadcasting correctness): query heads
  `[g*r, (g+1)*r)` share KV head `g`, and the map is surjective onto the KV
  heads — head grouping is a well-defined surjection.
- `GQ-INV-003` (GPU head-mapping identity): `kvHead r q = q * num_kv_heads /
  num_heads`, the closed form the kernel's integer division realises.
- `GQ-SUB-001` / `GQ-EQV-001` (refines / degenerates to MHA): when
  `num_kv_heads = num_heads` (group size `r = 1`) routing is the identity, so
  GQA reduces to standard multi-head attention.

## References

- Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints." EMNLP, 2023.
-/

namespace ProvableContracts.Gqa

open ProvableContracts

/-- KV-head routing: query head `q` attends KV head `q / r`, where `r` is the
    group size (`num_heads / num_kv_heads`). -/
def kvHead (r q : ℕ) : ℕ := q / r

/-- **Broadcasting within a group.** Every query head in group `g`, i.e. the
    `r` contiguous indices `g*r, g*r+1, …, g*r+(r-1)`, routes to KV head `g`. -/
theorem kvHead_group (r g i : ℕ) (hr : 0 < r) (hi : i < r) :
    kvHead r (g * r + i) = g := by
  unfold kvHead
  rw [Nat.mul_comm g r, Nat.mul_add_div hr, Nat.div_eq_of_lt hi, Nat.add_zero]

/-- **Surjectivity.** For divisibility `num_heads = num_kv_heads * r`, every KV
    head `h < num_kv_heads` is the image of some query head `q < num_heads`
    (namely `q = h*r`). Head grouping is a surjection onto the KV heads. -/
theorem kvHead_surjective (r kvHeads h : ℕ) (hr : 0 < r) (hh : h < kvHeads) :
    ∃ q, q < kvHeads * r ∧ kvHead r q = h := by
  refine ⟨h * r, ?_, ?_⟩
  · exact (Nat.mul_lt_mul_right hr).mpr hh
  · simpa using kvHead_group r h 0 hr hr

/-- **GPU head-mapping identity.** The routing map equals the closed form
    `q * num_kv_heads / num_heads` used by the kernel's integer division, given
    `num_heads = num_kv_heads * r` with `num_kv_heads > 0`. -/
theorem kvHead_eq_mul_div (r kvHeads q : ℕ) (hk : 0 < kvHeads) :
    kvHead r q = q * kvHeads / (kvHeads * r) := by
  unfold kvHead
  rw [Nat.mul_comm q kvHeads, Nat.mul_div_mul_left q r hk]

/-- **Degeneration to MHA (identity routing).** With group size `r = 1`
    (`num_kv_heads = num_heads`) every query head attends its own KV head, so
    the routing map is the identity and GQA coincides with standard MHA. -/
theorem kvHead_group_one (q : ℕ) : kvHead 1 q = q := by
  unfold kvHead
  exact Nat.div_one q

#check @kvHead_group
#check @kvHead_surjective
#check @kvHead_eq_mul_div
#check @kvHead_group_one

end ProvableContracts.Gqa
