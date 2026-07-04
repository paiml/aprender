import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import ProvableContracts.Basic

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
