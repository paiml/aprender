import Mathlib.Data.Real.Basic
import Mathlib.Data.List.GetD
import ProvableContracts.Basic

/-!
# Embedding Lookup Definitions

An embedding lookup maps a sequence of token IDs to dense vectors by gathering
rows from an embedding table `W ∈ R^{vocab_size × d_model}`:

    output[i] = W[token_ids[i]]   for i in 0 .. seq_len

Mathematically this is a pure **gather** over the rows of a table. We model the
table as a `List` of rows (each row an arbitrary type `α`, e.g. a `d_model`-vector)
and the token IDs as a `List ℕ`. A default row `dflt` (the padding / zero vector)
is used for the total function `List.getD`; the in-bounds theorems establish that
the default is never actually reached when the precondition holds.

The list model captures exactly the structural content of the contract —
length preservation, per-row correctness, the in-bounds index invariant, and
predicate preservation (finiteness) — without committing to a floating-point
representation, which is the runtime concern handled by falsification tests.

## References

- Mikolov et al. (2013) Efficient Estimation of Word Representations in Vector Space
- Vaswani et al. (2017) Attention Is All You Need
-/

namespace ProvableContracts.Embedding

/-- Gather the rows of `table` selected by `ids`, using `dflt` for the total
    `List.getD`. This is the pure functional model of the embedding lookup
    `output[i] = table[ids[i]]`. -/
def gather {α : Type _} (table : List α) (ids : List ℕ) (dflt : α) : List α :=
  ids.map (fun i => table.getD i dflt)

/-- Single-row lookup helper `table[i]`, exposed so the per-row correctness and
    roundtrip lemmas can be stated cleanly. -/
def gatherAt {α : Type _} (table : List α) (i : ℕ) (dflt : α) : α :=
  table.getD i dflt

end ProvableContracts.Embedding
