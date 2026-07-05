import ProvableContracts.Defs.Embedding
import Mathlib.Data.Matrix.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Embedding Algebra Theorems

Analytic obligations of `embedding-algebra-v1.yaml` discharged over exact
reals / dependent `Fin` dimensions.

| Contract obligation                | Theorem                    |
|------------------------------------|----------------------------|
| Embedding lookup shape / selection | `onehot_select`, `embed_dim` |
| Unembedding output shape           | `unembed_apply`, `unembed_shape` |
| Tied weight identity               | `tied_row_eq`, `tied_apply` |
| Token ID bounds                    | `token_lt_vocab`, `token_nonneg` |
| Temperature identity               | `temp_identity`            |

Supporting analytic core (linearity of pooling, additive composition, scale):
`sum_pool_add`, `sum_pool_smul`, `mean_pool_smul`.

Two obligations are intentionally NOT discharged here:
- **Embedding non-degeneracy** (`‖embed t‖₂ > 0`) is a property of the *loaded
  weight values*, not an algebraic identity — a zero row is representable for an
  arbitrary table. Runtime/actual-weight class (contract N/A).
- **Temperature entropy monotonicity** (`T₁<T₂ → H(softmax(z/T₁)) < H(softmax(z/T₂))`)
  is a genuine analytic inequality but requires the derivative identity
  `dH/dβ = -β·Varₚ(z)`; left UNCOVERED (analytic-but-unproven), so the contract
  honestly remains below full L4.
-/

namespace ProvableContracts.Embedding

open Matrix Finset

/-! ## Obligation 1 — embedding lookup shape / one-hot row selection -/

/-- `one-hot(t) @ W = W[t]`: the one-hot projection selects exactly row `t`.
    This is the algebraic content of an embedding lookup. -/
theorem onehot_select {V d : ℕ} (W : EmbTable V d) (t : Fin V) (j : Fin d) :
    (∑ k : Fin V, onehot t k * W k j) = embed W t j := by
  simp only [onehot, embed, ite_mul, one_mul, zero_mul]
  simp [Finset.sum_ite_eq']

/-- Embedding output is `d`-dimensional: its index type has cardinality `d`. -/
theorem embed_dim {V d : ℕ} (_W : EmbTable V d) (_t : Fin V) :
    Fintype.card (Fin d) = d := by
  simp

/-! ## Obligation 2 — unembedding output shape / matmul entry -/

/-- Unembedding entry: `(h · Wuᵀ)[s,v] = Σₖ h[s,k]·Wu[v,k]`, i.e. the dot
    product of hidden row `s` with embedding row `v`. -/
theorem unembed_apply {seq d V : ℕ} (h : Matrix (Fin seq) (Fin d) ℝ)
    (Wu : EmbTable V d) (s : Fin seq) (v : Fin V) :
    (h * Wuᵀ) s v = ∑ k : Fin d, h s k * Wu v k := by
  simp [Matrix.mul_apply, Matrix.transpose_apply]

/-- Unembedding output shape is `[seq_len, V]`: the row index has cardinality
    `seq_len` and the column index has cardinality `V`. -/
theorem unembed_shape (seq _d V : ℕ) :
    Fintype.card (Fin seq) = seq ∧ Fintype.card (Fin V) = V := by
  constructor <;> simp

/-! ## Obligation 3 — tied weights -/

/-- Tied weights: if the unembedding table equals the embedding table, every
    embedding row coincides. No independent parameters. -/
theorem tied_row_eq {V d : ℕ} (We Wu : EmbTable V d) (hty : Wu = We) (t : Fin V) :
    embed Wu t = embed We t := by
  rw [hty]

/-- Entrywise tied-weight identity. -/
theorem tied_apply {V d : ℕ} (We Wu : EmbTable V d) (hty : Wu = We)
    (i : Fin V) (j : Fin d) : Wu i j = We i j := by
  rw [hty]

/-! ## Obligation 4 — token id bounds -/

/-- Every token index is strictly below the vocabulary size. -/
theorem token_lt_vocab {V : ℕ} (t : Fin V) : (t : ℕ) < V := t.isLt

/-- Every token index is non-negative (no negative ids). -/
theorem token_nonneg {V : ℕ} (t : Fin V) : 0 ≤ (t : ℕ) := Nat.zero_le _

/-! ## Obligation 6 — temperature identity -/

/-- `T = 1` is the identity temperature: `z / 1 = z`. -/
theorem temp_identity {n : ℕ} (z : RVec n) : temp_scale 1 z = z := by
  funext i
  simp [temp_scale]

/-! ## Supporting analytic core — pooling linearity / additive composition / scale -/

/-- Sum pooling is additive: pooling `A + B` = pooling `A` + pooling `B`. -/
theorem sum_pool_add {m d : ℕ} (A B : Fin m → RVec d) (j : Fin d) :
    sum_pool (fun i => fun c => A i c + B i c) j = sum_pool A j + sum_pool B j := by
  simp [sum_pool, Finset.sum_add_distrib]

/-- Sum pooling is homogeneous (scale): pooling `a·A` = `a·` pooling `A`. -/
theorem sum_pool_smul {m d : ℕ} (a : ℝ) (A : Fin m → RVec d) (j : Fin d) :
    sum_pool (fun i => fun c => a * A i c) j = a * sum_pool A j := by
  simp [sum_pool, Finset.mul_sum]

/-- Mean pooling is homogeneous (scale). -/
theorem mean_pool_smul {m d : ℕ} (a : ℝ) (A : Fin m → RVec d) (j : Fin d) :
    mean_pool (fun i => fun c => a * A i c) j = a * mean_pool A j := by
  simp only [mean_pool]
  rw [← Finset.mul_sum]
  ring

-- Tests
#check @onehot_select
#check @unembed_apply
#check @tied_row_eq
#check @token_lt_vocab
#check @temp_identity
#check @sum_pool_add

end ProvableContracts.Embedding
