import ProvableContracts.Defs.Embedding
import ProvableContracts.Theorems.Embedding.Rows

/-!
# Embedding Lookup — In-Bounds Index Invariant (Out-of-Bounds Panic Freedom)

Contract obligation: *Out-of-bounds panic freedom* —
`token_ids[i] < vocab_size for all i implies no panic`.

Modelling `vocab_size = table.length`, the precondition `∀ i ∈ ids, i < |table|`
guarantees that every lookup hits a genuine table row (the option access is
`isSome`), so the total-function default (`dflt`) is never observed. A real
row access can never be out of bounds — hence no panic.
-/

namespace ProvableContracts.Embedding

/-- **In-bounds invariant.** If every token id is `< table.length` (`< vocab_size`),
    then the `j`-th lookup index is in bounds, so `table[ids[j]]?` resolves to an
    actual row (`isSome`) — the access can never panic. -/
theorem gather_inbounds {α : Type _} (table : List α) (ids : List ℕ)
    (hb : ∀ i ∈ ids, i < table.length) (j : ℕ) (hj : j < ids.length) :
    (table[ids[j]]?).isSome = true := by
  have hlt : ids[j] < table.length := hb _ (List.getElem_mem hj)
  rw [List.getElem?_eq_getElem hlt]
  rfl

/-- Corollary: under the in-bounds precondition the gathered row is a genuine
    member of the table — the default padding is never reached, so no out-of-bounds
    behaviour occurs. -/
theorem gather_hits_table {α : Type _} (table : List α) (ids : List ℕ) (dflt : α)
    (hb : ∀ i ∈ ids, i < table.length) (j : ℕ) (hj : j < ids.length) :
    (gather table ids dflt)[j]? = some (table[ids[j]]'(hb _ (List.getElem_mem hj))) := by
  have hlt : ids[j] < table.length := hb _ (List.getElem_mem hj)
  rw [gather_row?, List.getElem?_eq_getElem hj]
  simp only [Option.map_some, List.getD_eq_getElem table dflt hlt]

end ProvableContracts.Embedding
