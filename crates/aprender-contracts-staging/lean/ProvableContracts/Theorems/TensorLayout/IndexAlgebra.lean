/-!
# Row-Major Tensor Index Algebra (core-only)

Pillar-4 (BEAT Ollama) data-layer correctness for `contracts/tensor-layout-v1.yaml`,
obligation **"Transpose shape correctness"** and the row-major layout algebra it
rests on (LAYOUT-001/002).

APR and realizar are EXCLUSIVELY row-major: the element `(i, j)` of an
`nrows × ncols` tensor lives at the linear byte-slot `idx = i * ncols + j`. This
file proves the purely-algebraic backbone that the importer / kernels depend on:

  * `idx-BIJ-001` — on the valid rectangle `i < nrows`, `j < ncols`, the map
    `idx i j = i*ncols + j` lands in range `< nrows*ncols` and
    `unidx k = (k / ncols, k % ncols)` is a TWO-SIDED inverse ⇒ `idx` is a
    bijection onto `[0, nrows*ncols)` (⇒ injective, ⇒ every slot hit once).
  * `idx-TRANSPOSE-001` — the row-major transpose reindex
    `out[r*cols + c] = in[c*rows + r]` used by `transpose_q{4,6}k_for_matmul`
    is exactly `idx`/`unidx` with the axes swapped, and is itself a bijection
    between the two rectangles (element preserving — nothing dropped/duplicated).
  * `idx-SHAPE-001` — 2-D transpose swaps the shape pair exactly
    (`apr_shape[0] = gguf_shape[1]`, `apr_shape[1] = gguf_shape[0]`), 1-D tensors
    are identity, and the element count (byte size) is preserved.
  * `idx-STRIDE-001` — contiguous strides: row stride `= ncols`, col stride `= 1`,
    and a full row occupies the contiguous block `[i*ncols, i*ncols + ncols)`.

Core-only (no imports) ⇒ verifies cleanly (no axioms, no holes) with the standalone `lean <file>`
binary — no Mathlib, no `import`.

Reference: `contracts/tensor-layout-v1.yaml`, `src/format/converter/mod.rs`
(`transpose_q4k_for_matmul`, `transpose_q6k_for_matmul`).
-/

namespace ProvableContracts.TensorLayout

/-! ## Row-major linear index and its inverse -/

/-- Row-major linear index into an `nrows × ncols` tensor: `data[i * ncols + j]`. -/
def idx (ncols i j : Nat) : Nat := i * ncols + j

/-- Recover the row from a linear index. -/
def unrow (ncols k : Nat) : Nat := k / ncols

/-- Recover the column from a linear index. -/
def uncol (ncols k : Nat) : Nat := k % ncols

/-- The index formula, definitionally. -/
@[simp] theorem idx_def (ncols i j : Nat) : idx ncols i j = i * ncols + j := rfl

/-! ## `idx-BIJ-001` — bijection on the valid rectangle -/

/-- Range: a valid `(i, j)` maps into `[0, nrows * ncols)`. -/
theorem idx_lt {nrows ncols i j : Nat} (hi : i < nrows) (hj : j < ncols) :
    idx ncols i j < nrows * ncols := by
  have hstep : (i + 1) * ncols = i * ncols + ncols := Nat.succ_mul i ncols
  have hmono : (i + 1) * ncols ≤ nrows * ncols := Nat.mul_le_mul_right ncols hi
  unfold idx
  omega

/-- Inverse recovers the row (needs `j < ncols`). -/
theorem unrow_idx {ncols i j : Nat} (hj : j < ncols) :
    unrow ncols (idx ncols i j) = i := by
  have hpos : 0 < ncols := Nat.lt_of_le_of_lt (Nat.zero_le j) hj
  unfold unrow idx
  rw [Nat.mul_comm i ncols, Nat.add_comm, Nat.add_mul_div_left j i hpos,
    Nat.div_eq_of_lt hj, Nat.zero_add]

/-- Inverse recovers the column (needs `j < ncols`). -/
theorem uncol_idx {ncols i j : Nat} (hj : j < ncols) :
    uncol ncols (idx ncols i j) = j := by
  unfold uncol idx
  rw [Nat.mul_comm i ncols, Nat.add_comm, Nat.add_mul_mod_self_left]
  exact Nat.mod_eq_of_lt hj

/-- Other direction: for ANY linear index, `idx (unrow k) (uncol k) = k`
    (surjectivity onto `[0, nrows*ncols)`; here even unbounded). -/
theorem idx_unrow_uncol (ncols k : Nat) :
    idx ncols (unrow ncols k) (uncol ncols k) = k := by
  unfold idx unrow uncol
  rw [Nat.mul_comm (k / ncols) ncols]
  exact Nat.div_add_mod k ncols

/-- Injectivity on the rectangle: equal linear indices ⇒ equal `(i, j)`. -/
theorem idx_injective {ncols i j i' j' : Nat} (hj : j < ncols) (hj' : j' < ncols)
    (h : idx ncols i j = idx ncols i' j') : i = i' ∧ j = j' := by
  refine ⟨?_, ?_⟩
  · have e := congrArg (unrow ncols) h
    rwa [unrow_idx hj, unrow_idx hj'] at e
  · have e := congrArg (uncol ncols) h
    rwa [uncol_idx hj, uncol_idx hj'] at e

/-! ## `idx-TRANSPOSE-001` — row-major transpose reindex

The APR importer turns a GGUF `[cols, rows]` weight into an APR `[rows, cols]`
weight via `out[r*cols + c] = in[c*rows + r]`. In `idx` terms the destination
linear index `idx cols r c` reads the source linear index `idx rows c r`. -/

/-- The transpose reindex is the swap of the two `idx` forms. -/
theorem transpose_reindex (rows cols r c : Nat) :
    idx cols r c = r * cols + c ∧ idx rows c r = c * rows + r :=
  ⟨rfl, rfl⟩

/-- The transpose reindex is a bijection between the two rectangles. Decoding the
    destination address `idx cols r c` under the destination stride `cols` recovers
    `(r, c)`; decoding the source address `idx rows c r` under the source stride
    `rows` recovers the swapped `(c, r)`. Both halves are exact ⇒ the reindex is an
    element-preserving bijection (nothing dropped or duplicated). -/
theorem transpose_reindex_bijective {rows cols r c : Nat}
    (hr : r < rows) (hc : c < cols) :
    (unrow cols (idx cols r c), uncol cols (idx cols r c)) = (r, c) ∧
    (unrow rows (idx rows c r), uncol rows (idx rows c r)) = (c, r) := by
  refine ⟨?_, ?_⟩
  · rw [unrow_idx hc, uncol_idx hc]
  · rw [unrow_idx hr, uncol_idx hr]

/-! ## `idx-SHAPE-001` — shape swap, 1-D identity, size preservation -/

/-- A 2-D tensor shape as an ordered `(d0, d1)` pair. -/
structure Shape2 where
  d0 : Nat
  d1 : Nat
deriving DecidableEq

/-- Shape transpose swaps the two extents. -/
def Shape2.transpose (s : Shape2) : Shape2 := ⟨s.d1, s.d0⟩

/-- Obligation **"Transpose shape correctness"**:
    `apr_shape[0] = gguf_shape[1]` and `apr_shape[1] = gguf_shape[0]`. -/
theorem transpose_shape_swap (s : Shape2) :
    s.transpose.d0 = s.d1 ∧ s.transpose.d1 = s.d0 :=
  ⟨rfl, rfl⟩

/-- The shape swap is an involution: `(sᵀ)ᵀ = s`. -/
@[simp] theorem transpose_shape_involution (s : Shape2) :
    s.transpose.transpose = s :=
  rfl

/-- 1-D tensors are identity: `should_transpose_gguf` returns `false` for a
    single-extent shape, so the importer leaves it unchanged. -/
def transpose1d (n : Nat) : Nat := n

theorem transpose_shape_1d_identity (n : Nat) : transpose1d n = n := rfl

/-- Byte size (element count) is preserved across transpose. -/
theorem transpose_size_preserved (s : Shape2) :
    s.transpose.d0 * s.transpose.d1 = s.d0 * s.d1 :=
  Nat.mul_comm s.d1 s.d0

/-! ## `idx-STRIDE-001` — contiguous strides -/

/-- Row stride is `ncols`: advancing one row adds `ncols` to the linear index. -/
theorem row_stride (ncols i j : Nat) :
    idx ncols (i + 1) j = idx ncols i j + ncols := by
  have hstep : (i + 1) * ncols = i * ncols + ncols := Nat.succ_mul i ncols
  unfold idx
  omega

/-- Column stride is `1`: advancing one column adds `1`. -/
theorem col_stride (ncols i j : Nat) :
    idx ncols i (j + 1) = idx ncols i j + 1 := by
  unfold idx
  omega

/-- A full row is contiguous: its slots occupy `[i*ncols, i*ncols + ncols)`. -/
theorem row_block_lower (ncols i j : Nat) : i * ncols ≤ idx ncols i j := by
  unfold idx; omega

theorem row_block_upper {ncols i j : Nat} (hj : j < ncols) :
    idx ncols i j < i * ncols + ncols := by
  unfold idx; omega

-- Checks
#check @idx_lt
#check @unrow_idx
#check @uncol_idx
#check @idx_unrow_uncol
#check @idx_injective
#check @transpose_reindex_bijective
#check @transpose_shape_swap
#check @transpose_size_preserved
#check @row_stride
#check @col_stride

end ProvableContracts.TensorLayout
