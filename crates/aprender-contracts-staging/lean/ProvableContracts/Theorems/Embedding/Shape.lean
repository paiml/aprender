import ProvableContracts.Defs.Embedding

/-!
# Embedding Lookup — Output Shape Correctness

Contract obligation: *Output shape correctness* —
`output.shape = (seq_len, d_model)` for `token_ids.len() = seq_len`.

The gather output has exactly one row per token ID, so its length equals the
length of the ID sequence (`seq_len`). Each row is copied verbatim from the
table, so every output row has the same width (`d_model`) as the table rows.
This file proves the length component; the width component is definitional
(rows are copied unchanged — see `Rows.lean`).
-/

namespace ProvableContracts.Embedding

/-- **Length preservation.** The gathered output has exactly `ids.length` rows,
    i.e. `len(out) = len(ids) = seq_len`. -/
theorem gather_length {α : Type _} (table : List α) (ids : List ℕ) (dflt : α) :
    (gather table ids dflt).length = ids.length := by
  simp [gather]

/-- Corollary in seq_len form: if `ids.length = seq_len` then the output also
    has `seq_len` rows. -/
theorem gather_length_seq {α : Type _} (table : List α) (ids : List ℕ) (dflt : α)
    (seq_len : ℕ) (h : ids.length = seq_len) :
    (gather table ids dflt).length = seq_len := by
  rw [gather_length, h]

end ProvableContracts.Embedding
