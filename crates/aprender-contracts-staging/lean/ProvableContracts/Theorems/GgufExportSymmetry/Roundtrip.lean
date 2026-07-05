/-!
# APR ↔ GGUF Export/Import Dtype-Symmetry Round-Trip (core-only)

Pillar-4 (BEAT Ollama) serialization correctness.

`aprender-core::format::converter::fusion::apr_dtype_to_ggml` maps an APR tensor
dtype to a GGML type ONLY for the four layout-identical dtypes
(`F32`, `F16`, `Q4K`, `Q6K` — whose numeric ids `0, 1, 12, 14` are shared byte
for byte with GGML) and returns `None` for every APR-native / non-GGML dtype
(`AprQ8`, `AprQ4`, `BF16`, `F64`, `I32`, `I64`, `I8`, `U8`). Relabeling an
APR-native quant as a byte-incompatible GGML type would emit a corrupt GGUF, so
the export path REJECTS it — mirroring the import-side refusal of GGUF `Q8_0`
(which APR cannot represent exactly).

This file proves the purely-algebraic facts the export/import symmetry rests on:

  * `GES-DTYPE-ROUNDTRIP-001` — `importDtype ∘ exportDtype = id` on the
    compatible subset: every dtype the exporter accepts is restored exactly by
    the importer (dtype preserved).
  * `GES-REJECT-SYM-001` — `AprQ8` / `AprQ4` export to `none` (rejected, never
    relabeled), and `Q8_0` imports to `none` (symmetric refusal).
  * `GES-SHAPE-PRESERVE-001` — a successful export copies the tensor shape
    verbatim (shape preserved).
  * `GES-PAYLOAD-INVOL-001` — the full-tensor round trip `importTensor ∘
    exportTensor = id`: dtype, shape AND the raw byte payload are restored
    bit-for-bit on the exportable subset (the export/import involution on the
    tensor payload).

**Core-only**: everything is modelled over Lean-core `inductive`/`structure`
types, so it compiles with zero unproved goals under the standalone `lean <file>`
binary — no `import Mathlib`, no imports at all.

Reference: `contracts/apr-gguf-export-symmetry-v1.yaml`,
`crates/aprender-core/src/format/converter/fusion.rs` (`apr_dtype_to_ggml`),
`crates/apr-format/src/v2/tensor_index_impl.rs` (`TensorDType`),
`crates/aprender-core/src/format/gguf/types.rs` (`GgmlType`).
-/

namespace ProvableContracts.GgufExportSymmetry

/-- APR tensor dtypes (mirrors `TensorDType`, `tensor_index_impl.rs`). The four
    layout-identical members carry the same numeric id as their GGML twin
    (`F32=0`, `F16=1`, `Q4K=12`, `Q6K=14`); `AprQ4=128`/`AprQ8=129` are placed
    OUTSIDE the GGML range precisely to prevent a relabel collision. -/
inductive AprDType
  | F32 | F16 | BF16 | F64 | I32 | I64 | I8 | U8 | AprQ4 | AprQ8 | Q4K | Q6K
  deriving DecidableEq

/-- GGML tensor types (subset of `GgmlType`, `gguf/types.rs`). `Q8_0` is the type
    the corrupt relabel would have targeted; APR cannot represent it exactly. -/
inductive GgmlType
  | F32 | F16 | Q4_0 | Q4_1 | Q8_0 | Q4K | Q6K
  deriving DecidableEq

/-- APR→GGUF export dtype map — mirrors `apr_dtype_to_ggml` (fusion.rs). `some`
    only for the four layout-identical dtypes; `none` (reject) for everything
    else, incl. `AprQ8`/`AprQ4`. -/
def exportDtype : AprDType → Option GgmlType
  | .F32 => some .F32
  | .F16 => some .F16
  | .Q4K => some .Q4K
  | .Q6K => some .Q6K
  | _    => none

/-- GGUF→APR import dtype map — the inverse on the compatible subset. `Q8_0`
    (and the other non-representable GGML types) → `none`, mirroring the
    import-side rejection in `write_model_config.rs`. -/
def importDtype : GgmlType → Option AprDType
  | .F32 => some .F32
  | .F16 => some .F16
  | .Q4K => some .Q4K
  | .Q6K => some .Q6K
  | _    => none

/-!
## `GES-REJECT-SYM-001` — APR-native quants are rejected, symmetrically

The bug this contract closed was `exportDtype AprQ8 = some Q8_0` (silent
relabel). The fix makes it `none`, and the importer already refuses `Q8_0`.
-/

/-- `AprQ8` export is rejected — NOT relabeled as `Q8_0`. -/
@[simp] theorem export_rejects_aprq8 : exportDtype .AprQ8 = none := rfl

/-- `AprQ4` export is rejected (unchanged behaviour). -/
@[simp] theorem export_rejects_aprq4 : exportDtype .AprQ4 = none := rfl

/-- Symmetric import-side refusal: GGUF `Q8_0` has no exact APR dtype. -/
@[simp] theorem import_rejects_q8_0 : importDtype .Q8_0 = none := rfl

/-!
## `GES-DTYPE-ROUNDTRIP-001` — `importDtype ∘ exportDtype = id` (dtype preserved)
-/

/-- Every dtype the exporter accepts is restored exactly by the importer:
    `exportDtype d = some g → importDtype g = some d`. -/
theorem dtype_roundtrip (d : AprDType) (g : GgmlType)
    (h : exportDtype d = some g) : importDtype g = some d := by
  cases d <;> simp only [exportDtype] at h <;>
    first
      | (injection h with h; subst h; rfl)
      | simp at h

/-- The other direction on the compatible subset:
    `importDtype g = some d → exportDtype d = some g`. -/
theorem dtype_roundtrip_section (g : GgmlType) (d : AprDType)
    (h : importDtype g = some d) : exportDtype d = some g := by
  cases g <;> simp only [importDtype] at h <;>
    first
      | (injection h with h; subst h; rfl)
      | simp at h

/-!
## Full-tensor payload round trip

A tensor payload is `(dtype, shape, bytes)`; the exporter/importer copy the
shape and the opaque byte payload verbatim, gated only by the dtype map. This is
the exact structure of `export_apr_to_gguf_raw` (it emits raw APR bytes under
the mapped GGML label).
-/

/-- An APR tensor: dtype + shape + opaque byte payload. -/
structure AprTensor where
  dtype : AprDType
  shape : List Nat
  bytes : List UInt8

/-- A GGUF tensor: GGML type + shape + opaque byte payload. -/
structure GgufTensor where
  ggml  : GgmlType
  shape : List Nat
  bytes : List UInt8

/-- Export: reject non-compatible dtypes, else copy shape + bytes verbatim. -/
def exportTensor (t : AprTensor) : Option GgufTensor :=
  match exportDtype t.dtype with
  | some g => some { ggml := g, shape := t.shape, bytes := t.bytes }
  | none   => none

/-- Import: reject non-representable GGML types, else copy shape + bytes. -/
def importTensor (g : GgufTensor) : Option AprTensor :=
  match importDtype g.ggml with
  | some d => some { dtype := d, shape := g.shape, bytes := g.bytes }
  | none   => none

/-- `GES-SHAPE-PRESERVE-001` — a successful export copies the shape verbatim. -/
theorem export_preserves_shape (t : AprTensor) (gt : GgufTensor)
    (h : exportTensor t = some gt) : gt.shape = t.shape := by
  unfold exportTensor at h
  cases hd : exportDtype t.dtype with
  | none => rw [hd] at h; simp at h
  | some g =>
    rw [hd] at h
    injection h with h'
    subst h'
    rfl

/-- `GES-PAYLOAD-INVOL-001` — the full round trip `importTensor ∘ exportTensor`
    is the identity on the exportable subset: dtype, shape AND the raw byte
    payload are restored bit-for-bit. -/
theorem export_import_roundtrip (t : AprTensor) (gt : GgufTensor)
    (h : exportTensor t = some gt) : importTensor gt = some t := by
  unfold exportTensor at h
  cases hd : exportDtype t.dtype with
  | none => rw [hd] at h; simp at h
  | some g =>
    rw [hd] at h
    injection h with h'
    subst h'
    unfold importTensor
    have hg : importDtype g = some t.dtype := dtype_roundtrip t.dtype g hd
    rw [hg]

-- Checks
#check @dtype_roundtrip
#check @dtype_roundtrip_section
#check @export_rejects_aprq8
#check @export_rejects_aprq4
#check @import_rejects_q8_0
#check @export_preserves_shape
#check @export_import_roundtrip

end ProvableContracts.GgufExportSymmetry
