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

/-! ## `idx-MODEL-001` — simulation of the Rust `transpose_row_major`

`transposeRowMajor` transcribes `crates/aprender-quant/src/transpose.rs:17-21`
(`trueno_quant::transpose::transpose_row_major`, the reindex every
`transpose_q{4,5,6}k_for_matmul` runs between dequantize and requantize):

    (0..rows * cols).map(|k| src[(k % cols) * rows + k / cols]).collect()

`List.range (rows * cols)` is `0..rows * cols`, `List.map` is `.map(..).collect()`, and
`(k % cols) * rows + k / cols` is `idx rows (uncol cols k) (unrow cols k)`. The element type
is generic in both. Rust's `src[i]` panics out of range where the model reads `getD i default`;
`transposeRowMajor_reads_in_range` shows no read is out of range when `src.len() = rows * cols`,
the only way the importer calls it, so the two agree there.

Witness: `crates/aprender-quant/src/transpose_row_major_witness_tests.rs` checks the Rust fn
against this def's `#eval` on every shape `rows, cols ≤ 8` (golden table beside it). -/

/-- Transcription of `transpose_row_major` (`crates/aprender-quant/src/transpose.rs:17`). -/
def transposeRowMajor {α : Type} [Inhabited α] (src : List α) (rows cols : Nat) : List α :=
  (List.range (rows * cols)).map fun k => src.getD (idx rows (uncol cols k) (unrow cols k)) default

/-- The output has exactly `rows * cols` elements (the Rust `Vec` length). -/
theorem transposeRowMajor_length {α : Type} [Inhabited α] (src : List α) (rows cols : Nat) :
    (transposeRowMajor src rows cols).length = rows * cols := by
  simp [transposeRowMajor]

/-- Every slot the Rust body reads, `(k % cols) * rows + k / cols` for `k < rows * cols`, is in
    range of a `rows * cols` source: the Rust indexing never panics on a well-sized input. -/
theorem transposeRowMajor_reads_in_range {rows cols k : Nat} (hk : k < rows * cols) :
    idx rows (uncol cols k) (unrow cols k) < rows * cols := by
  have hc : 0 < cols := Nat.pos_of_ne_zero (fun h => by simp [h] at hk)
  have hu : uncol cols k < cols := Nat.mod_lt k hc
  have hr : unrow cols k < rows := (Nat.div_lt_iff_lt_mul hc).2 hk
  have := idx_lt (nrows := cols) hu hr
  rwa [Nat.mul_comm cols rows] at this

/-- LAYOUT-002 on the model: APR element `(r, c)` (slot `r * cols + c`) is GGUF element
    `(c, r)` (slot `c * rows + r`). -/
theorem transposeRowMajor_get {α : Type} [Inhabited α] {src : List α} {rows cols r c : Nat}
    (hr : r < rows) (hc : c < cols) :
    (transposeRowMajor src rows cols)[idx cols r c]? = some (src.getD (idx rows c r) default) := by
  have hk : idx cols r c < rows * cols := idx_lt hr hc
  simp only [transposeRowMajor, List.getElem?_map, List.getElem?_range hk, Option.map_some,
    unrow_idx hc, uncol_idx hc]

/-- Transposing back with the axes swapped returns the input exactly: the reindex is a
    permutation, nothing dropped or duplicated. -/
theorem transposeRowMajor_involution {α : Type} [Inhabited α] {src : List α} {rows cols : Nat}
    (hlen : src.length = rows * cols) :
    transposeRowMajor (transposeRowMajor src rows cols) cols rows = src := by
  apply List.ext_getElem
  · rw [transposeRowMajor_length, hlen, Nat.mul_comm]
  · intro k h1 h2
    have hk : k < rows * cols := hlen ▸ h2
    have hrp : 0 < rows := Nat.pos_of_ne_zero (fun h => by simp [h] at hk)
    have ha : uncol rows k < rows := Nat.mod_lt k hrp
    have hb : unrow rows k < cols := (Nat.div_lt_iff_lt_mul hrp).2 (by rwa [Nat.mul_comm] at hk)
    have hinner := transposeRowMajor_get (src := src) ha hb
    simp only [transposeRowMajor, List.getElem_map, List.getElem_range]
    rw [List.getD_eq_getElem?_getD]
    have hL : (List.map (fun k => src.getD (idx rows (uncol cols k) (unrow cols k)) default)
        (List.range (rows * cols)))[idx cols (uncol rows k) (unrow rows k)]? =
        some (src.getD (idx rows (unrow rows k) (uncol rows k)) default) := hinner
    rw [hL, Option.getD_some, idx_unrow_uncol, List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem h2, Option.getD_some]

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
#check @transposeRowMajor_reads_in_range
#check @transposeRowMajor_get
#check @transposeRowMajor_involution

end ProvableContracts.TensorLayout
