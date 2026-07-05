import ProvableContracts.Defs.Embedding

/-!
# Embedding Lookup — Per-Row Correctness and Gather∘Scatter Roundtrip

Contract obligation (supporting *Deterministic output* and *Finite output*):
each output row equals the indexed table row, `output[i] = table[ids[i]]`.

We also prove the **gather∘scatter identity**: gathering a table at the identity
index sequence `range(len(table))` reconstructs the table exactly. This is the
canonical structural roundtrip for a gather primitive.
-/

namespace ProvableContracts.Embedding

/-- **Per-row correctness.** The `j`-th gathered row is the table lookup of the
    `j`-th token id: `output[j] = table[ids[j]]` (option form, total in `j`). -/
theorem gather_row? {α : Type _} (table : List α) (ids : List ℕ) (dflt : α)
    (j : ℕ) :
    (gather table ids dflt)[j]? = (ids[j]?).map (fun i => table.getD i dflt) := by
  simp [gather, List.getElem?_map]

/-- Point form: when `j` is in range, the `j`-th output row is exactly
    `gatherAt table ids[j] = table[ids[j]]`. -/
theorem gather_row {α : Type _} (table : List α) (ids : List ℕ) (dflt : α)
    (j : ℕ) (hj : j < ids.length) :
    (gather table ids dflt)[j]? = some (gatherAt table (ids[j]) dflt) := by
  simp [gather, gatherAt, List.getElem?_map, List.getElem?_eq_getElem hj]

/-- **Gather∘scatter identity.** Gathering the table at the identity index
    sequence returns the table unchanged: `gather W (range |W|) = W`. -/
theorem gather_scatter_id {α : Type _} (table : List α) (dflt : α) :
    gather table (List.range table.length) dflt = table := by
  apply List.ext_getElem?
  intro j
  rw [gather_row?]
  by_cases hj : j < table.length
  · rw [List.getElem?_range hj]
    simp [List.getElem?_eq_getElem hj]
  · have h1 : (List.range table.length)[j]? = none :=
      List.getElem?_eq_none_iff.mpr (by simpa using Nat.le_of_not_lt hj)
    have h2 : table[j]? = none := List.getElem?_eq_none_iff.mpr (Nat.le_of_not_lt hj)
    rw [h1, h2]; rfl

end ProvableContracts.Embedding
