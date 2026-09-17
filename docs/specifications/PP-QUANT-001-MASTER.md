# PP-QUANT-001 v1.0 — MASTER — Quant-type dispatch consolidation

**Status:** DRAFT · ticket #3418 · train **0.69.0** (per operator decision 2026-09-17 —
overrides #3418's own request for a dedicated, non-cadence release; see §6.1 for the
risk this decision accepts) · authored 2026-09-17 against `origin/main` @ `425e84888`
**Companions:** `contracts/quant-dispatch-completeness-v1.yaml` (the type universe, the
Phase 0 inventory, and the gate's obligations; `pv validate` green) ·
`crates/aprender-serve/src/quantize/format_trait.rs` (existing partial trait, extended
not replaced) · epic #3421 · `docs/specifications/06x-release-schedule.md` (the
train this rides; not amended by this document — see §6.2 for what would need to change
there)
**Downstream:** `docs/specifications/PP-ARCH-001-MASTER.md` (#3422, epic #3423,
milestone PP-ARCH-001) — the shared attention/FFN forward-path consolidation this
document is a prerequisite for. Closing PP-QUANT-001 alone does not deliver "new
architectures cost hundreds of lines, not thousands": of #3418's own 1,678-line Qwen3-MoE
count, only ~85 lines are quant dispatch; the rest is forward-pass logic PP-ARCH-001
governs. PP-ARCH-001 needs this document's dispatch table to make its shared
`attn_block`/`moe_ffn_block` layer quant-agnostic.
**Governs:** how `crates/aprender-serve/` maps a GGUF/APR `qtype` to `{byte_size,
dequant_to_f32, family}`, and where that mapping is looked up from.
**Owner:** unassigned (§7)

---

## §0 Conventions

Same marks as `PP-LLAMA-001-MASTER.md` §0.3: `[V]` verified by a cited command in this
session · `[C]` calculated from `[V]` inputs · `[A]` asserted, re-verify at implementing
HEAD · `[U]` unmeasured, row names who measures it.

This document states rules and phases. `contracts/quant-dispatch-completeness-v1.yaml`
declares the actual enumerated type list and is what a CI gate reads — a claim that a
type is "covered" is true only if that contract's obligation for it is `ARMED` (§0.4 of
PP-LLAMA-001's convention, reused here). This document cites the contract; it does not
duplicate its content.

---

## §1 Problem, as evidenced in #3418

Quant-type dispatch (`qtype → {byte_size, dequant_to_f32, family}`) is reimplemented at
**30 call sites** across `crates/aprender-serve/src/` instead of living in one table
(llama.cpp's `ggml_type_traits[]` equivalent). `crates/aprender-serve/src/quantize/format_trait.rs`
already has `QuantBlockFormat` — a real, working trait — but:

- Only ~5 of the 30 call sites route through it `[A]` (#3418 investigation).
- It models **affine** quantization only (`x = d*s*q - dmin*m`): Q4_0/Q8_0/Q4_K/Q5_K/Q6_K.
  It has no representation for **codebook/lattice** types (IQ2_XXS/XS, IQ3_XXS/S,
  IQ4_NL/XS) — those need a second trait shape, not an extension of the first, because
  each block indexes a fixed grid table rather than carrying a per-block scale.
- The crate names 13 `GGUF_TYPE_*` tensor-type constants (§4 — #3418's "16" included
  three metadata value types); ggml defines 35 live tensor types. Every type absent
  from a given call site's own match arms is a live
  `RealizarError::UnsupportedOperation` waiting on the next real download.
- Five closed tickets (#1749, #1789, #2535, #3341, #3091) are the same defect —
  "this file's qtype list was incomplete" — recurring in five different files. Each fix
  was scoped to the one file that crashed; none closed the class.

Full evidence, grep commands, and the file list are in #3418 — not reproduced here to
avoid this document drifting from the ticket as the tree changes; §0 rule: this
document does not duplicate content that ages, it cites where the aging content lives.

---

## §2 Target architecture

### §2.1 Two traits, one dispatch entry point

1. **`QuantBlockFormat`** (exists, `format_trait.rs`) — affine block formats. Extend its
   impls to the full affine subset of the ggml universe (Q1_0, Q2_0, Q4_0/1, Q5_0/1,
   Q8_0, Q2_K through Q6_K, TQ1_0/TQ2_0) rather than the current 5. Q8_1 and Q8_K get
   a *size* entry only — ggml itself has no `to_float` for them (they are activation
   layouts), so the table records `dequant_row: None` rather than pretending.
2. **`QuantCodebookFormat`** (new) — lattice/codebook formats (IQ1–IQ4 family). Grid
   tables are ported data from llama.cpp's `ggml-quants.c` (`iq2xs_grid`, `iq3xs_grid`,
   etc.) — reference data, not something to re-derive.
3. **One dispatch table**, `static QUANT_TYPE_TRAITS: [Option<QuantTypeTraits>; N]`
   plus a lookup fn `quant_type_traits(qtype: u32) -> Option<&'static QuantTypeTraits>`
   (names TBD at implementation — analogous to `ggml_type_traits[]`). `QuantTypeTraits`
   is a **plain data struct** — `{byte_size, block_size, family, dequant_to_f32: fn(&[u8],
   &mut [f32]) -> Result<()>}` — not an enum over trait objects (§2.1a). This is the
   single new public surface every call site is migrated to use.

### §2.1a Why a data table, not `dyn QuantBlockFormat` (correction, 2026-09-17)

An earlier draft of this section described the table's entries as
`Affine(dyn QuantBlockFormat) | Codebook(dyn QuantCodebookFormat)`. That is not
buildable: `QuantBlockFormat` carries associated `const`s (`FORMAT_ID`,
`SUPERBLOCK_BYTES`, …) and its methods take no `self` receiver (`read_d(superblock:
&[u8])`, not `read_d(&self, ...)`) — both independently make a trait non-object-safe,
so `dyn QuantBlockFormat` fails to compile. This was the point of `format_trait.rs`'s
existing design: monomorphized generics, zero vtable overhead — the opposite of what a
trait object gives you.

`ggml_type_traits[]` itself was never trait-object-shaped either — it is a C struct of
plain fields and **function pointers**, stored in a static array indexed by the type
enum. The direct Rust port of that is a struct whose `dequant_to_f32` field is an
ordinary `fn` pointer, not a `dyn Trait`. Each table entry's function pointer is a
concrete, non-generic wrapper (e.g. `dequant_q4k_to_f32`) that calls the existing
monomorphized generic (`dequant_generic::<Q4K>(...)`) internally; the generic parameter
is fully resolved before the function is placed in the array, so no object-safety rule
is ever in play. `QuantBlockFormat`/`QuantCodebookFormat` are unchanged by this
correction — they remain the const-based, compile-time-specialized traits the hot-path
kernels use directly; the dispatch table is an additional, thin runtime layer over them,
not a replacement.

### §2.2 The completeness gate (the mechanical backstop, not the honor system)

`contracts/quant-dispatch-completeness-v1.yaml` (exists; `pv validate` green `[V]`)
is the enumeration. It carries, per live ggml type id: `name`, `family`, `block`,
`bytes`, whether ggml itself can dequantize it, and aprender's status at `425e84888`;
plus `removed_ids` (the eight ids ggml.h marks removed), `type_count` (43), the ten
codebook `grids` with their element type and length, and the Phase 0 `dispatch_sites`
inventory. Every number is transcribed from `ggml-common.h`'s `static_assert(sizeof(
block_X) == …)` lines at llama.cpp `3173a5647` and self-checked: 35 rows + 8 removed
ids = 43 = `type_count`, no gaps, no duplicates `[V]`.

The gate is a Rust test (owed, Phase 1) that reads the contract and asserts the table
in §2.4 has `Some` for every row, `None` for every removed id and for `type_count`, and
that every `QuantBlockFormat`/`QuantCodebookFormat` impl's byte constants equal its row.
A missing type is then a CI failure at merge time — what #1749/#1789/#2535/#3341/#3091
needed and did not have; each was an honor-system list, five times over. The contract's
`FALSIFY-QDC-001..006` name the mutation that must turn each assertion RED.

### §2.3 Per-backend fan-out is additive, not multiplicative

CPU SIMD, CUDA, and wgpu each implement the trait once per type. The ~30-call-site
migration happens once, regardless of backend count, because call sites stop matching
on `qtype` themselves and instead look up the one dispatch table and call the returned
`QuantTypeTraits` struct's function-pointer fields.

### §2.4 Rust design (Phase 1 blueprint)

**§2.4.1 There are already three registries; the table is built on one of them, not
beside them.** At `425e84888` `[V]`:

| Registry | Where | Covers | Carries |
|---|---|---|---|
| `GgmlQuantType` (`#[repr(u32)]`, `from_id`, `as_str`) | `crates/aprender-serve/src/gguf/types.rs` | 16 ids | name ↔ id only |
| `WeightQuantType` (PMAT-232: exhaustive matches, no `Default`) | `crates/aprender-serve/src/cuda/types.rs` | 8 | `bytes_per_superblock`, `bytes_per_block`, CUDA kernel choice |
| `QuantBlockFormat` impls (`Q4K`, `Q5K`, `Q6K`, `Q4_0`, `Q8_0`) | `crates/aprender-serve/src/quantize/format_trait.rs` | 5 | full block algebra, compile-time |
| `tensor_byte_size` (#3091's site) | `crates/aprender-serve/src/gguf/transformer.rs:435` | 11 | bytes only, its own arms |

A fourth enum would be the two-lists defect again. `GgmlQuantType` is the key: it is
already `#[repr(u32)]` on the ggml id, already has `from_id`, and is what the loader
holds. Phase 1 extends it to all 35 live types and makes `from_id` reject the eight
removed ids by name. `WeightQuantType` stays as the *CUDA kernel selector* (its
`match`es choose kernels, which is legitimately backend-specific) but loses its byte
methods: they become `quant_type_traits(self.into()).bytes` so there is one number.

**§2.4.2 The table entry.**

```rust
// crates/aprender-serve/src/quantize/dispatch.rs (owed)
pub type DequantRowFn = fn(src: &[u8], dst: &mut [f32]) -> Result<()>;

pub struct QuantTypeTraits {
    pub id: u32,                       // ggml id; == index in the table
    pub name: &'static str,            // "Q4_K" — the GGUF spelling
    pub family: QuantFamilyKind,       // Scalar | AffineBlock | AffineKQuant | Codebook | Ternary | Fp4 | Activation
    pub block: usize,                  // elements per (super-)block; 1 for scalars
    pub bytes: usize,                  // bytes per (super-)block
    pub dequant_row: Option<DequantRowFn>, // None ⇔ ggml has no to_float (Q8_1, Q8_K, I*)
}

impl QuantTypeTraits {
    pub const fn bytes_for(&self, n_elements: usize) -> usize { n_elements.div_ceil(self.block) * self.bytes }
}

static QUANT_TYPE_TRAITS: [Option<QuantTypeTraits>; GGML_TYPE_COUNT] = [ /* index == id */ ];

pub fn quant_type_traits(id: u32) -> Option<&'static QuantTypeTraits> {
    QUANT_TYPE_TRAITS.get(id as usize).and_then(Option::as_ref)
}
```

`dequant_row` is an `fn` pointer (§2.1a), filled with a non-generic wrapper per type:

```rust
fn dequant_row_q4k(src: &[u8], dst: &mut [f32]) -> Result<()> { dequant_row::<Q4K>(src, dst) }
```

where `dequant_row<F: QuantBlockFormat>` is generic and monomorphized. The hot kernels
(`fused_q4k_parallel_matvec`, `generic_fused_gate_up_matvec_into<F>`, the CUDA GEMV
paths) are **not** routed through the fn pointer — they keep calling the generics
directly with a concrete `F`. The table serves the cold paths that today hold the
duplicated arms: sizing, validation, loading, conversion, metadata, "can this backend
take this tensor" decisions. That is where all five prior tickets crashed; none crashed
inside a kernel inner loop.

`GGML_TYPE_COUNT` (43) is a `const` in `gguf/types.rs`; the array is indexed by id so a
removed id is a literal `None` slot, and an id ≥ 43 falls off the end — both are the
contract's `rejection` equation with no `_ =>` anywhere.

**§2.4.3 `QuantCodebookFormat`** (`crates/aprender-serve/src/quantize/codebook_trait.rs`,
owed) mirrors `QuantBlockFormat`'s const-based shape so the same kind of generic
kernel can be written over it, but the algebra differs: a block holds a scale, a run
of grid *indices*, sign bits, and (for the `_S`/`_XS` variants) sub-block scales.

```rust
pub trait QuantCodebookFormat: Send + Sync + 'static {
    const FORMAT_ID: &'static str;
    const ELEMENTS_PER_SUPERBLOCK: usize;   // 256, or 32 for IQ4_NL
    const SUPERBLOCK_BYTES: usize;
    const GRID_POINT_LEN: usize;            // elements one grid entry expands to (8 for IQ2/IQ1, 4 for IQ3, 1 for IQ4)
    type GridElem: Copy + 'static;          // u64 / u32 / i8
    const GRID: &'static [Self::GridElem];  // the ported table, by reference
    fn read_d(superblock: &[u8]) -> f32;
    fn grid_index(superblock: &[u8], point: usize) -> usize;
    fn signs(superblock: &[u8], point: usize) -> u8;
    fn subblock_scale(superblock: &[u8], idx: usize) -> f32;
    fn dequant_point(superblock: &[u8], point: usize, out: &mut [f32]);
}
```

MXFP4/NVFP4 are `QuantCodebookFormat` too: their "grid" is `kvalues_fp4` (16 E2M1
values) and the scale is a shared exponent; forcing them into the affine trait would
need a fake `dmin`. TQ1_0/TQ2_0 *are* affine (`{-1,0,1}·d`) and go on
`QuantBlockFormat` with `ZERO_OFFSET = 1`, `QUANT_BITS = 2`.

**§2.4.4 Grid tables are generated, never typed.** `scripts/gen_iq_grids.sh` (owed)
reads `ggml-common.h` at the pinned commit and emits
`crates/aprender-serve/src/quantize/iq_grids.rs`: ten `pub const` arrays, 25.6 KiB of
`.rodata` in total (the contract's `grids` section lists each with its length). The
generated file carries the source commit in its header, and PO-QDC-004's test compares
each array's length, first and last element against the contract. A hand edit to a
grid is therefore a RED test, not a silent lattice corruption.

**§2.4.5 Migration shape (Phase 2, one site per PR).** `tensor_byte_size` is the
template — it is #3091's exact crash site and the simplest:

```rust
// before: eleven arms and a `_ => Err(UnsupportedOperation)`
// after:
fn tensor_byte_size(qtype: u32, num_elements: usize, dims: &[u64]) -> Result<usize> {
    let t = quant_type_traits(qtype).ok_or_else(|| unsupported_qtype("tensor_byte_size", qtype))?;
    Ok(match (t.family, dims) {
        (QuantFamilyKind::AffineKQuant | QuantFamilyKind::Codebook | QuantFamilyKind::Ternary, [rows, cols]) =>
            (*rows as usize) * t.bytes_for(*cols as usize),   // row-padded, LAYOUT-001
        _ => t.bytes_for(num_elements),
    })
}
```

The row-padding rule (K-quant rows pad to super-block boundaries) is the one piece of
logic that stays at the site because it is about tensor *shape*, not type. Every other
site follows the same pattern: look up, branch on `family` if the site genuinely
differs per family, never on the id.

**§2.4.6 Naming.** `quant_type_traits` / `QuantTypeTraits` / `QUANT_TYPE_TRAITS` —
deliberately the ggml name so a reader coming from llama.cpp finds it by grep;
`QuantFamilyKind` rather than reusing `QuantFamily` (which is the trait's two-valued
`KQuant | Simple` and stays as is). Module: `crate::quantize::dispatch`, re-exported
from `crate::quantize`.

---

## §3 Phases

Phase boundaries exist so the completeness gate (Phase 1) can go green and be checked
in *before* the 30-file migration (Phase 2) starts touching the CPU/CUDA/wgpu backends
simultaneously — the ordering #3418 asks for, independent of which train it rides on.

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **0 — Inventory** — **DONE in this PR** | The 30 call sites (`dispatch_sites`) and the 35-type universe (`types`, `removed_ids`, `type_count`) live in `contracts/quant-dispatch-completeness-v1.yaml`, not in a script: the contract is the one list, and the `pv`-dogfooding rule forbids a bash re-implementation. The oracle that produced `dispatch_sites` is written into the contract beside it | `pv validate contracts/quant-dispatch-completeness-v1.yaml` exit 0 `[V]`; 35 + 8 = 43 = `type_count` `[V]`; 30 sites = #3418's count `[V]` |
| **1 — Trait + gate** | Per §2.4: `GgmlQuantType` → 35 variants; `QuantCodebookFormat` + generated `iq_grids.rs`; `QuantBlockFormat` impls for the remaining affine/ternary types; `dispatch.rs` with `QUANT_TYPE_TRAITS` and `quant_type_traits()`; the five PO-QDC tests reading the contract. Additive — no call site migrated | All five `quant_dispatch_*` tests green, each shown RED first by its FALSIFY-QDC mutation; `WeightQuantType::bytes_per_*` deleted in favour of the table (the first consumer, proves the table is load-bearing) |
| **2 — Migration** | Every `dispatch_sites` row flipped to `migrated: true`, one site (or tightly related group) per PR, `transformer.rs::tensor_byte_size` first (§2.4.5) | PO-QDC-005's oracle finds only the `keep` rows; the contract's `dispatch_sites` list and the tree agree |
| **3 — Parity re-proof** | Full parity re-run of the existing Qwen2.5-Coder GPU-beat measurement (README/BEATS.md's demonstrated model) against pre-migration baseline, per quant type touched, per backend | cosine ≥ 0.98 vs. pre-migration baseline (the bar #3091's own history already holds this codebase to); no regression on `contracts/beat-ollama-decode-throughput-speed-v1.yaml` |

Phase 2 is the ~30-file, multi-backend blast radius #3418 warns about. Phases 0/1 are
additive and low-risk; they can land first inside 0.69 without touching a live call
site, so Qwen2.5-Coder's proven path is untouched until Phase 3 is ready to re-verify it.

---

## §4 The universe, measured (2026-09-17, `425e84888` vs llama.cpp `3173a5647`)

| Fact | Value | Measured by |
|---|---|---|
| Live ggml tensor types | **35** (ids 0–42 minus 8 removed; `GGML_TYPE_COUNT` = 43) | `awk '/enum ggml_type \{/,/\};/' ggml/include/ggml.h` in the llama.cpp checkout `[V]` |
| `GGUF_TYPE_*` tensor-type consts in aprender-serve | **13** (#3418's "16" counted `STRING`/`UINT32`/`ARRAY`/`INT32`, which are metadata value types, not tensor types) | `grep -cE 'pub const GGUF_TYPE_' crates/aprender-serve/src/gguf/types.rs` `[V]` |
| `GgmlQuantType` variants | 16 | `gguf/types.rs` `[V]` |
| `WeightQuantType` variants | 8 | `cuda/types.rs` `[V]` |
| `QuantBlockFormat` impls | 5 | `format_trait.rs` `[V]` |
| Status by contract row | `dequant` 8 · `sized` 3 · `named` 5 · **`missing` 19** | `contracts/quant-dispatch-completeness-v1.yaml` `types[].status` `[C]` |

Of the 19 `missing`: 9 are codebook (IQ1_S, IQ1_M, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL,
IQ4_XS, MXFP4, NVFP4 — #3091's IQ2_XS/IQ4_XS crash class), 2 ternary (TQ1_0, TQ2_0),
2 new affine (Q1_0, Q2_0 — added to ggml after #3418 was written), 5 scalar (I8–I64,
F64), and Q8_K (activation-only; needs a size entry, no dequant). Two (Q8_1, Q8_K)
legitimately have **no** dequant in ggml either; the table encodes that as
`dequant_row: None` rather than as an "unsupported" error, so a caller can tell
"cannot" from "not yet".

---

## §5 Owed artifacts (do not exist at `425e84888`, written without backticks per the
`PP-LLAMA-001` convention so the drift gate is never asked to check a file this document
is asking someone to create)

Delivered by this document's PR:

- `contracts/quant-dispatch-completeness-v1.yaml` — Phase 0 inventory + Phase 1 gate
  definition, `pv validate` green
- `crates/aprender-serve/src/quantize/dispatch_contract_tests.rs` — the five `qdc_*`
  tests that bind the contract to the tree today (PO-QDC-002/003/005 plus the
  contract's own gaplessness and name agreement); each shown RED by its FALSIFY-QDC
  mutation before landing (receipt in the PR). Registered as a tree reader in
  `scripts/tree_reader_tests.txt` so the quick CI tier always runs it
- `serde_yaml_ng` as an aprender-serve dev-dependency (it was build-only)

Still owed (Phase 1 unless marked):

- crates/aprender-serve/src/quantize/dispatch.rs — `QuantTypeTraits`, `QuantFamilyKind`,
  `QUANT_TYPE_TRAITS`, `quant_type_traits()`, the per-type `dequant_row_*` wrappers (§2.4.2)
- crates/aprender-serve/src/quantize/codebook_trait.rs — `QuantCodebookFormat` + impls (§2.4.3)
- crates/aprender-serve/src/quantize/iq_grids.rs — GENERATED, ten `const` grids (§2.4.4)
- scripts/gen_iq_grids.sh — the generator; pins the llama.cpp commit; `--check` diffs
- scripts/gen_quant_dispatch_rows.sh — regenerates the contract's `types` rows from
  ggml.h + ggml-common.h; `--check` is FALSIFY-QDC-006
- `qdc_table_covers_every_row` (PO-QDC-001) and `qdc_grids_match_reference`
  (PO-QDC-004), added to the existing `dispatch_contract_tests.rs` once the table and
  grids exist; the contract's FALSIFY-QDC-001/004 then gain their `test:` citation
- a `ci/explicit-test-commands.d/NNN-quant-dispatch.cmd` fragment if the tests land as
  an integration target rather than `--lib` (a `tests/*.rs` file is dark until named)
- docs/audits/impl-PP-QUANT-001-phaseN-receipt.md — one per phase, RED→GREEN proof (each phase)

---

## §6 Risks accepted by the 0.69 placement decision

**§6.1** #3418 explicitly requested this ride its own dedicated release cycle, arguing
the ~30-file, 3-backend (CPU/CUDA/wgpu) blast radius of Phase 2 competes for the same
verification attention as the model-specific tickets riding the normal cadence, and
risks regressing Qwen2.5-Coder — the one model line with a real GPU-parity receipt
(`docs/BEATS.md`). The operator decided (2026-09-17) to place this on **0.69** instead
of reserving 0.70 or a new milestone. This document's phase split (§3) is the mitigation
available within that constraint: **only Phase 0/1 are committed to the 0.69 window**;
Phase 2 (the actual multi-file migration) and Phase 3 (parity re-proof) are **not**
bound to 0.69's `2026-09-18T06:00Z` train-leaves-by instant and may slip forward under
the train rule (`06x-release-schedule.md` §1.1) without blocking the tag, carrying a
`slipped_from:` note on the epic.

**§6.2** `docs/specifications/06x-release-schedule.md` is not amended by this document.
0.69's stated theme ("declarative fine-tune lands; batching bands turn VALID") and its
114 open milestone issues are unaffected; this work is additive to that train, not a
replacement of its scope. If Phase 2/3 need dedicated verification attention that the
schedule doc's priorities A–G don't already carry, that is a separate amendment to
`06x-release-schedule.md`, proposed once Phase 1 lands and Phase 2's real size is known
from the Phase 0 inventory — not asserted here in advance.

---

## §7 Next steps (tracking, not yet done by this document)

1. ~~Open a GitHub epic under milestone `0.69.0`, linking #3418~~ — done: epic #3421.
2. Assign an owner (currently unassigned).
3. ~~Phase 0 inventory as the first PR~~ — done: `contracts/quant-dispatch-completeness-v1.yaml`
   plus the `qdc_*` tests binding it to the tree.
4. Phase 1: the table, `QuantCodebookFormat`, generated grids, and the two remaining
   tests (§2.4, §5).
5. See PP-ARCH-001-MASTER.md for the downstream consolidation this one is a
   prerequisite for.
