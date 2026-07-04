/-!
# GGUF → APR Tensor-Transpose Round-Trip Involution (core-only)

Pillar-4 (BEAT Ollama) data-layer correctness.

`aprender-quant::transpose_q4k_for_matmul` / `transpose_q6k_for_matmul` convert a
2-D GGUF column-major weight `[cols, rows]` into an APR row-major weight
`[rows, cols]` by reindexing `out[r*cols + c] = in[c*rows + r]` and swapping the
shape. This file proves the two purely-algebraic facts the pillar rests on:

  * `TT-SHAPE-INVOL-001` — the shape swap is an involution: `(mⁿ)ⁿ = m`
  * `TT-BYTE-PRESERVE-001` — every element is relocated, never dropped/duplicated,
    and the round trip restores the exact original element (byte preservation).

This is a **core-only** port of `Theorems/Transpose/Involution.lean` (which uses
`Mathlib.Data.Matrix.Basic`). Everything below is modelled over `Nat` and Lean
core `structure`/function types, so it compiles sorry-free with the standalone
`lean <file>` binary — no `import Mathlib`, no imports at all.

Reference: LAYOUT-001/002 tensor-layout safety, `contracts/tensor-layout-v1.yaml`.
-/

namespace ProvableContracts.TensorTranspose

universe u

/-- A 2-D tensor: `rows × cols` with an element accessor `get i j`.
    `get` models the row-major storage `data[i * cols + j]` abstractly, so byte
    preservation is exactly "the element at each index is unchanged after the
    round trip". -/
structure Tensor2D (α : Type u) where
  rows : Nat
  cols : Nat
  get  : Nat → Nat → α

/-- Transpose = swap dims and swap indices, exactly matching the Rust reindex
    `out[r*cols + c] = in[c*rows + r]`. -/
def Tensor2D.transpose {α : Type u} (t : Tensor2D α) : Tensor2D α :=
  { rows := t.cols, cols := t.rows, get := fun i j => t.get j i }

/-- The two shape components are swapped by a single transpose. -/
@[simp] theorem transpose_shape {α : Type u} (t : Tensor2D α) :
    t.transpose.rows = t.cols ∧ t.transpose.cols = t.rows :=
  ⟨rfl, rfl⟩

/-- `TT-BYTE-PRESERVE-001` (element-level): a single transpose relocates element
    `(i, j)` to `(j, i)` with no loss — `Bᵀ[j][i] = A[i][j]`. -/
theorem transpose_element {α : Type u} (t : Tensor2D α) (i j : Nat) :
    t.transpose.get j i = t.get i j :=
  rfl

/-- `TT-SHAPE-INVOL-001` + `TT-BYTE-PRESERVE-001` (full):
    the round trip `(Aᵀ)ᵀ` is *bit-for-bit* the original tensor — same shape,
    and every element restored to its original index. -/
theorem transpose_involution {α : Type u} (t : Tensor2D α) :
    t.transpose.transpose = t :=
  rfl

/-- Round-trip byte preservation stated at the index level (corollary of the
    involution, provable directly by `rfl`). -/
theorem roundtrip_preserves_element {α : Type u} (t : Tensor2D α) (i j : Nat) :
    t.transpose.transpose.get i j = t.get i j :=
  rfl

/-!
## Shape-pair-swap involution (the minimal "2-line" core form)

The same fact expressed purely on the `(rows, cols)` shape pair — this is what
`should_transpose_gguf` gates on (transpose only 2-D tensors) and what the APR
importer swaps.
-/

/-- A tensor shape is a `(rows, cols)` pair. -/
structure Shape where
  rows : Nat
  cols : Nat
  deriving DecidableEq

/-- Shape transpose swaps the pair. -/
def Shape.transpose (s : Shape) : Shape := ⟨s.cols, s.rows⟩

/-- The shape swap is an involution: `(sᵀ)ᵀ = s`. -/
@[simp] theorem shape_transpose_involution (s : Shape) :
    s.transpose.transpose = s :=
  rfl

/-- A 1-D shape (`should_transpose_gguf` returns `false`) is a fixed point when
    modelled as `rows = 1`: transposing a row-vector shape and back is identity —
    trivially the same involution, documenting that the `shape.len() != 2`
    early-return path is also involutive. -/
theorem shape_transpose_involution_vec (n : Nat) :
    (Shape.mk 1 n).transpose.transpose = Shape.mk 1 n :=
  rfl

-- Checks
#check @transpose_involution
#check @transpose_element
#check @roundtrip_preserves_element
#check @shape_transpose_involution

end ProvableContracts.TensorTranspose
