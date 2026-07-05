import ProvableContracts.Defs.Embedding

/-!
# Embedding Lookup — Finite Output (Value Preservation)

Contract obligation: *Finite output* —
`W[j][k] is finite implies output[i][k] is finite`.

The analytic core is **value preservation**: a gather copies rows verbatim and
introduces no new values beyond the table rows and the padding default. Hence
any per-row predicate `P` that holds on every table row (and on the default)
holds on every output row. Finiteness is exactly such a predicate `P`, so
finite-in ⇒ finite-out; the copy introduces neither `NaN` nor `Inf`.

(The IEEE bit-level meaning of `is_finite()` is a runtime property checked by
the falsification tests; what is *provable* — and what actually guarantees the
implication — is that gather is a value-preserving copy, established here.)
-/

namespace ProvableContracts.Embedding

/-- **Value preservation.** Every gathered row is either a genuine table row or
    the default; equivalently, gather introduces no value outside `table ∪ {dflt}`. -/
theorem gather_preserves {α : Type _} (P : α → Prop) (table : List α)
    (ids : List ℕ) (dflt : α) (hdflt : P dflt) (htab : ∀ x ∈ table, P x) :
    ∀ y ∈ gather table ids dflt, P y := by
  intro y hy
  simp only [gather, List.mem_map] at hy
  obtain ⟨i, _, rfl⟩ := hy
  rcases lt_or_ge i table.length with h | h
  · rw [List.getD_eq_getElem table dflt h]
    exact htab _ (List.getElem_mem h)
  · rw [List.getD_eq_default table dflt h]
    exact hdflt

/-- **Finite output corollary.** Instantiating value preservation with the
    finiteness predicate: if the padding default is finite and every table row is
    finite, then every gathered output row is finite. No `NaN`/`Inf` is created. -/
theorem gather_finite {α : Type _} (isFinite : α → Prop) (table : List α)
    (ids : List ℕ) (dflt : α) (hdflt : isFinite dflt)
    (htab : ∀ x ∈ table, isFinite x) :
    ∀ y ∈ gather table ids dflt, isFinite y :=
  gather_preserves isFinite table ids dflt hdflt htab

end ProvableContracts.Embedding
