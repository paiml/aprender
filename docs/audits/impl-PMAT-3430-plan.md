# PMAT-3430 plan — PP-QUANT-001 M1: one `GgmlType` + `TRAITS[43]`

Status: **PLAN, not reviewed.** Operator rule (#3421, 2026-09-17): no code until the plan quorum is 3/3 on Q1 and X1. Inputs: Tables S / E / B / G of `docs/audits/impl-PMAT-3427-receipt.md` on draft PR #3447 (`d76b4a417`), measured on `origin/main` @ `7eb81a531`. Everything marked `[V]` below was re-measured for this plan on `origin/main` @ `ee684e94c`; `[U]` is unverified and says so.

## 1. What is measured `[V]`

### 1.1 Home crate — derived from `cargo metadata --locked`, normal-dependency closure, workspace members only

| consumer | workspace crates in its closure |
|---|---|
| aprender-core | apr-format, aprender-common, aprender-compute, aprender-contracts-macros, aprender-gemm-codegen, aprender-quant |
| aprender-compute | aprender-contracts-macros, aprender-gemm-codegen, aprender-quant |
| aprender-serve | aprender-compute, aprender-contracts-macros, aprender-gemm-codegen, aprender-present-core, aprender-present-terminal, aprender-profile-core, aprender-quant |

Intersection (self included): aprender-compute, aprender-contracts-macros, aprender-gemm-codegen, aprender-quant. Of those, three are leaves (no workspace deps): a proc-macro crate, a codegen crate, and **aprender-quant** (`[lib] name = "trueno_quant"`, one external dep `half`, 1641 lines, "K-quantization formats for GGUF/APR model weights", already exports `Q4_K_BLOCK_SIZE`/`Q4_K_BLOCK_BYTES`-style constants).

**Proposed home: `aprender-quant`, new module `ggml_type`.** No new crate, so clean-room membership and publish order do not change. aprender-serve does not depend on aprender-core (normal deps), so core cannot be the home; compute could, but it is not the lowest and pulls 162 packages.

### 1.2 The three enums disagree in more than their id sets

| | compute `GgmlType` | core `GgmlType` | serve `GgmlQuantType` |
|---|---|---|---|
| file | aprender-compute/src/inference/gguf.rs:27 | aprender-core/src/format/gguf/types.rs:55 | aprender-serve/src/gguf/types.rs:101 |
| ids | 15 | 12 | 16 |
| k-quant variant spelling | `Q4K` | `Q4K` | `Q4K` |
| bf16 spelling | `Bf16` | — | `BF16` |
| only here | `Q8K` | `I8 I16 I32 I64 F64` | `IQ2XXS IQ2XS` |
| id → enum | `fn from_u32` (private) | — | `pub const fn from_id` |
| methods | `block_bytes block_size tensor_bytes` | none | `as_id as_byte as_str from_str_lossy Display` |

**A size disagreement already exists:** compute's `block_bytes` gives `Q8_1 => 36`; Table S (gguf-py `GGML_QUANT_SIZES`) gives 40. `[U]`: ggml's C `block_q8_1` is two f16 + 32 bytes = 36, and gguf-py has carried `4 + 4 + 32` from the older f32 layout. The ticket's falsifier says "equals gguf-py for every shared id", which would bake in 40. Phase 2 must read both upstream files at `3173a5647` and the quorum is asked to rule (§2, Q1-d).

## 2. Design (Q1)

```rust
// crates/aprender-quant/src/ggml_type.rs   (lib name: trueno_quant)
#[repr(u32)] #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType { F32 = 0, F16 = 1, Q4_0 = 2, /* … 35 live variants, upstream ids … */ Q2_0 = 42 }

pub enum GgmlFamily { Float, Int, Affine, KQuant, Codebook, Ternary, MicroscaleFp, Removed }
pub struct QuantTraits { pub name: &'static str, pub blck_size: u32, pub type_size: u32, pub family: GgmlFamily }
pub const TRAITS: [QuantTraits; 43] = [ /* indexed BY ID; the 8 removed ids are rows with family = Removed, sizes 0 */ ];

pub enum GgmlTypeError { Removed { id: u32, name: &'static str }, Unknown { id: u32 } }
impl GgmlType {
    pub const fn from_id(id: u32) -> Result<Self, GgmlTypeError>;   // match, never transmute (unsafe_code = forbid)
    pub const fn as_id(self) -> u32;   pub const fn traits(self) -> &'static QuantTraits;
    pub const fn name(self) -> &'static str;   pub const fn blck_size(self) -> usize;   pub const fn type_size(self) -> usize;
    pub const fn tensor_bytes(self, n_elements: usize) -> Option<usize>;   // checked: None on overflow or n % blck_size != 0
}
```

Decisions put to the quorum:

- **Q1-a. 35 variants, 43 table rows.** A removed id is not constructible; `from_id(4)` is `Err(Removed{..})`, distinct from `Err(Unknown{..})`. The table stays dense so `TRAITS[id]` is the lookup and a `const` assertion proves `TRAITS[v.as_id()].name == v.name()` for every variant.
- **Q1-b. Variant spelling.** One spelling. Proposal: keep the tree's existing `Q4K`-style (all three enums agree on it; 151 arm/ref sites in Table G use it) and `BF16`; compute's `Bf16` becomes an associated const alias `pub const Bf16: Self = Self::BF16` inside the quant crate — usable in expressions and patterns — so M1 touches no dispatch arm. Aliases are deleted in Q2.
- **Q1-c. M1 is behaviour-preserving: no crate accepts an id it refused before.** Going from 12/15/16 to 35 variants would silently widen every `match` that ends in `_ =>`. So each crate keeps a thin, named admission function (`supported(GgmlType) -> bool`, one each in compute, core and serve) that reproduces its former id set exactly, and its parser calls it. Matches without a wildcard stop compiling — that is the compiler enumerating the sites, and each gets an explicit refusal arm, not a `_`. Widening admission is M4 (#3432) / Q2, not this ticket.
- **Q1-d. Q8_1 type_size** — 36 (ggml C struct, what compute has today) or 40 (gguf-py, what the ticket's falsifier names). Proposal: the truth source for `type_size` is `sizeof` in ggml's C headers at the pinned sha; gguf-py is the cross-check; a disagreement between the two upstream sources is recorded in the contract as a named exception, never silently resolved.
- **Q1-e. Re-exports.** compute: `pub use trueno_quant::GgmlType;`. core: same. serve: `pub use trueno_quant::GgmlType as GgmlQuantType;` plus `pub use … GgmlType`. serve's `as_str`/`from_str_lossy`/`as_byte`/`Display` move into the quant crate (inherent impls cannot live downstream of a re-export); every id ≤ 42 fits a `u8`, so `as_byte` stays infallible.
- **Q1-f. #3405's private `QTYPE_LABELS` (13 rows)** is replaced by `TRAITS[id].name` in this ticket — it is a fourth table of the same fact.

### Table E — the 24 "keep?" rows resolved

Rule applied: `merge` iff the enum's variants are ggml tensor types and nothing else (a subset of Table S by meaning, not by spelling); `keep(concept)` names the different concept. **Execution of every `merge` below is Q2 (#3423 Phase 2), not M1** — M1 lands the target they merge into.

| enum | file | verdict |
|---|---|---|
| `QuantKernel` | aprender-cgp profilers/quant.rs:9 | keep(kernel identity — `Nf4Gemv` is not a ggml type) |
| `QuantType` | aprender-core format/quantize.rs:39 | keep(APR's own on-disk id space: `0x01/0x02/0x10/0xFF`, has `Q8Tensor`, `Custom`) |
| `GgufValue` | aprender-cbtop quantize/gguf.rs:24 | keep(metadata value, not tensor type) |
| `AprQuantizationType` | aprender-serve apr_transformer/loader.rs:303 | merge (F32/Q4_K/Q8_0 — a ggml subset used as a loader selector) |
| `Quantization` | aprender-train-inspect convert.rs:113 | merge (Q4_0/Q8_0/F16) |
| `KvQuantType` | aprender-serve paged_kv:254 | keep(KV-cache precision: FP32/Q8/Q4 are cache layouts, not ggml block formats) |
| `QuantizedKvData` | aprender-serve paged_kv:410 | keep(data carrier, not a type id) |
| `GgufQuantization` | aprender-train gguf_writer.rs:13 | merge → `Option<GgmlType>` (`None` variant is "don't quantize") |
| `GGUFQuantType` | aprender-train quant_type.rs:5 | merge (Q4_0/Q8_0) |
| `QuantScheme` | apr-cli quantize.rs:27 | keep(user-facing scheme: Int8/Int4/Fp16/Q4K — maps onto types, is not one) |
| `QuantizationType` | aprender-core converter_types_expectations.rs:149 | keep(same scheme concept as the row above; these two duplicate EACH OTHER — file for Q2) |
| `QuantizationType` | aprender-registry lineage/mod.rs:53 | keep(lineage metadata: has `Dynamic`) |
| `AutoQuantError` · `QuantizationError` · `QuantPublishError` | apr-cli · orchestrate · train | keep(error types) ×3 |
| `QuantizeArgvVerdict` | apr-cli quantize_flag_parity.rs:49 | keep(verdict) |
| `QuantScheme` | aprender-cbtop grammar/transform.rs:7 | keep(grammar) |
| `QuantizationType` | aprender-core demo/mod.rs:344 · stack/mod.rs:223 | keep(demo / stack descriptor) ×2 |
| `GgufValue` | aprender-core format/gguf/types.rs:84 | keep(metadata value) |
| `QuantFamily` | aprender-serve quantize/format_trait.rs:31 | keep(kernel-format family, 2 variants) — which is why §2 names the new enum `GgmlFamily`, not `QuantFamily`; revisit in Q2 |
| `QuantMethod` · `QuantGranularity` · `QuantMode` | aprender-train | keep(training-time quantization config) ×3 |

Count: 4 merge · 20 keep(concept).

## 3. Ordering (X1)

Q1 (this ticket) → T1/T2 (#3433/#3434) → Q2 → Q3. Evidence: 14 of 34 serve raw-integer qtype sites (41 %, file-level) sit in a file that uses a tensor carrier; the falsifier for running T1 ∥ Q2 was < 20 %. Nothing in §1–§2 changes that: M1 retypes no `qtype: u32` field (Table B's 53 rows are untouched), so it does not collide with T1.

## 4. Phases and acceptance commands

| # | phase | scope_paths | `A_i` |
|---|---|---|---|
| 1 | RED: falsifier guard + size fixture | scripts/check_one_ggml_type_enum.sh, contracts/ggml-type-v1.yaml, crates/aprender-quant/tests/ | `bash scripts/check_one_ggml_type_enum.sh` exits 1 naming 3 definitions; its case table (must-match / must-not-match, incl. `GgufValueType`) passes |
| 2 | `GgmlType` + `TRAITS[43]` in aprender-quant | crates/aprender-quant/ | `cargo test -p aprender-quant --lib ggml_type` — every live id round-trips, 8 removed ids are `Err(Removed)`, `TRAITS` equals the pinned upstream size fixture row by row |
| 3 | compute + core re-export, admission functions | crates/aprender-compute/src/inference/, crates/aprender-core/src/format/gguf/ | `cargo test -p aprender-compute --lib gguf && cargo test -p aprender-core --lib format::gguf` + an admission test per crate: former id set in, everything else refused |
| 4 | serve re-export, `QTYPE_LABELS` removed | crates/aprender-serve/src/gguf/, the #3405 file | `cargo test -p aprender-serve --lib gguf::types` + `bash scripts/check_one_ggml_type_enum.sh` exits **0** |
| 5 | DoD: `pv validate`, mutation (re-add a second enum → guard RED), diff quorum, receipt, PR | docs/audits/ | `make gate` |

Phases 3 and 4 have disjoint scopes and may run together. The guard is wired into the same PR's CI selection line or it is dark.

## 5. Not in this ticket

Retyping `qtype: u32` fields (Table B) — T1/Q2. The 558 bare-integer lines — #3431. `tensor_byte_size` single refusal — #3432. Executing the 4 Table E merges and the 10 merge-candidates — #3423 Phase 2. Widening any crate's admitted id set.
