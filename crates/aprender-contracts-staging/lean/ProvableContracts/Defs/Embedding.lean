import Mathlib.Data.Real.Basic
import Mathlib.Data.List.GetD
import ProvableContracts.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

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

/-!
# Embedding Algebra Definitions

Mathematical definitions backing the `embedding-algebra-v1.yaml` contract:
token embedding lookup as row selection, one-hot projection, unembedding
via the (tied) weight matrix, temperature scaling of logits, and
sum/mean pooling of embedding vectors.

## References

- Vaswani et al. (2017) "Attention Is All You Need" — shared embeddings.
- Press & Wolf (2017) "Using the Output Embedding to Improve Language Models."
-/

namespace ProvableContracts.Embedding

open Matrix Finset

/-- Embedding table: `V` rows (vocabulary), `d` columns (model dimension). -/
abbrev EmbTable (V d : ℕ) := Matrix (Fin V) (Fin d) ℝ

/-- Embedding lookup: select row `t` of the table, giving a `d`-dimensional
    vector. The result type `RVec d` *is* the `[d_model]` shape. -/
def embed {V d : ℕ} (W : EmbTable V d) (t : Fin V) : RVec d :=
  fun j => W t j

/-- One-hot row vector for token `t`: `1` at position `t`, else `0`. -/
noncomputable def onehot {V : ℕ} (t : Fin V) : Fin V → ℝ :=
  fun k => if k = t then 1 else 0

/-- Temperature scaling of a logit vector: `temp_scale T z = z / T`. -/
noncomputable def temp_scale {n : ℕ} (T : ℝ) (z : RVec n) : RVec n :=
  fun i => z i / T

/-- Sum pooling over a family of `m` embedding vectors. -/
def sum_pool {m d : ℕ} (E : Fin m → RVec d) : RVec d :=
  fun j => ∑ i : Fin m, E i j

/-- Mean pooling over a family of `m` embedding vectors. -/
noncomputable def mean_pool {m d : ℕ} (E : Fin m → RVec d) : RVec d :=
  fun j => (∑ i : Fin m, E i j) / (m : ℝ)

end ProvableContracts.Embedding
