# PMAT-3430 plan — PP-QUANT-001 M1: one `GgmlType` + `TRAITS[43]`

Status: **PLAN v2 — v1 was grilled 3/3 FAIL (§6); this revision answers every finding; not yet re-reviewed.** Operator rule (#3421, 2026-09-17): no code until the plan quorum is 3/3 on Q1 and X1. Inputs: Tables S / E / B / G of `docs/audits/impl-PMAT-3427-receipt.md` on draft PR #3447 (`d76b4a417`), measured on `origin/main` @ `7eb81a531`. Everything marked `[V]` below was re-measured for this plan on `origin/main` @ `ee684e94c`; `[U]` is unverified and says so.

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
| file | crates/aprender-compute/src/inference/gguf.rs:27 | crates/aprender-core/src/format/gguf/types.rs:55 | crates/aprender-serve/src/gguf/types.rs:101 |
| ids | 15 | 12 | 16 |
| k-quant variant spelling | `Q4K` | `Q4K` | `Q4K` |
| bf16 spelling | `Bf16` | — | `BF16` |
| only here | `Q8K` | `I8 I16 I32 I64 F64` | `IQ2XXS IQ2XS` |
| id → enum | `fn from_u32` (private) | — | `pub const fn from_id` |
| methods | `block_bytes block_size tensor_bytes` | none | `as_id as_byte as_str from_str_lossy Display` |

**A size disagreement already exists:** compute's `block_bytes` gives `Q8_1 => 36`; Table S (gguf-py `GGML_QUANT_SIZES`) gives 40. Both upstream sources were read at the pinned sha — see Q1-d.

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
    pub const fn from_id(id: u32) -> Option<Self>;                   // same signature serve's callers use today; match, never transmute
    pub const fn try_from_id(id: u32) -> Result<Self, GgmlTypeError>; // names WHY: Removed vs Unknown
    pub const fn as_id(self) -> u32;   pub const fn traits(self) -> &'static QuantTraits;
    pub const fn name(self) -> &'static str;   pub const fn blck_size(self) -> usize;   pub const fn type_size(self) -> usize;
    pub const fn tensor_bytes(self, n_elements: usize) -> usize;               // compute's existing signature and rounding, unchanged
    pub const fn checked_tensor_bytes(self, n_elements: usize) -> Option<usize>; // None on overflow or n % blck_size != 0; M4 (#3432) adopts it
}
```

Decisions put to the quorum:

- **Q1-a. 35 variants, 43 table rows.** A removed id is not constructible; `try_from_id(4)` is `Err(Removed{..})`, distinct from `Err(Unknown{..})`; `from_id(4)` is `None`. The table stays dense so `TRAITS[id]` is the lookup and a `const` assertion proves `TRAITS[v.as_id()].name == v.name()` for every variant.
- **Q1-b. Variant spelling — one spelling, NO aliases.** Keep the tree's `Q4K`-style (all three enums agree on it) and `BF16`. v1 proposed an associated-const alias `Bf16`; that is withdrawn: measured with rustc, `match self { Self::Bf16 => … }` on `&self` is **E0308** (a const pattern gets no default-binding-mode deref, unlike a variant pattern), it needs `#[allow(non_upper_case_globals)]`, and a glob import does not carry it. Instead compute's `GgmlType::Bf16` sites are renamed to `BF16` in Phase 3 (2 files: `crates/aprender-compute/src/inference/gguf.rs`, `…/inference/model.rs`; the `Bf16` in `brick/tracing/quant_type.rs` belongs to a different enum and is not touched).
- **Q1-c. M1 is behaviour-preserving: no crate accepts an id or a name it refused before — enforced at every id→enum and name→enum BOUNDARY, not only at the GGUF parser.** v1 gated the parser only; the grill found boundaries outside it. The measured list (serve): `gguf/dtype.rs:17` and `:54`, `infer/mod.rs:39`, `apr/mod.rs:324` (APR dtype byte), `apr/special_tokens.rs:210`, `apr/dequant.rs:235`, test `gguf/tests/loader_tests_apr.rs:311`. compute: private `from_u32` in `inference/gguf.rs`. core: `format/gguf/reader.rs`. Mechanism: each crate gets ONE admission function (`admitted_from_id(u32) -> Option<GgmlType>`, and in serve `admitted_from_name(&str)`) that is `GgmlType::from_id` filtered by a `const ADMITTED: [GgmlType; N]` reproducing the former set exactly (compute 15, core 12, serve 16), and every boundary above calls it instead of `from_id`/`from_str_lossy`. **Proof is a characterization test written in Phase 1, before any change:** for each boundary function, its output for every id 0..=255 (and every name in `TRAITS` plus case variants) is snapshotted on `origin/main`, and the same test must pass unchanged after Phases 3–4. Wildcard arms (`_ =>`; ~22 across 8 files that name these enums, e.g. `crates/aprender-compute/src/inference/model.rs:734` "quantized norms … return zeros") are therefore not widened in practice: they are already reached today by admitted quantized ids, and a non-admitted variant can only arrive by a new literal in new code. Phase 3/4 receipts still list every wildcard site with its disposition. Widening admission is M4 (#3432) / Q2.
- **Q1-d. Q8_1 `type_size` = 36. Measured at llama.cpp `3173a56471c1753650cd806694145ffd6dcace67`:** `ggml/src/ggml-common.h:269` is `static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1, "wrong q8_1 block size/padding")` = 36 — the assert rules out padding, so `sizeof` IS the on-disk block — while `gguf-py/gguf/constants.py:5678` says `(32, 4 + 4 + 32)` = 40 (the pre-f16 layout). Truth source for `type_size` is the `static_assert`ed C struct; gguf-py is the cross-check. The falsifier becomes: `TRAITS` equals gguf-py `GGML_QUANT_SIZES` for every live id **except a named exception list, today exactly `[Q8_1]`**, each exception carrying both upstream file:line citations in `contracts/ggml-type-v1.yaml`. The test fails if an exception stops being a disagreement (so the list cannot rot). This amends the ticket's falsifier wording; the amendment goes on #3430 with the verdict grid. (Grill: lanes 2, 3 ruled 36; lane 1 ruled 40 on a padding argument the `static_assert` refutes.)
- **Q1-e. Re-exports.** compute: `pub use trueno_quant::GgmlType;`. core: same. serve: `pub use trueno_quant::GgmlType as GgmlQuantType;` plus `GgmlType`. serve's `from_id`/`as_id`/`as_byte`/`as_str`/`from_str_lossy`/`Display` and compute's `block_bytes`/`block_size`/`tensor_bytes` move into the leaf with **unchanged names and signatures** (all three lanes: no body depends on a crate-local type; `TruenoError` is used by the parser, not the enum). `from_id` stays `Option` so `apr/mod.rs:325`'s zero-arg `map_or_else` and `loader_tests_apr.rs:311`'s `.is_some()` compile untouched; `block_bytes` stays as a name alongside `type_size`.
- **Q1-f. #3405's private `QTYPE_LABELS` (13 rows, `crates/aprender-serve/src/gguf/qwen3_moe_load.rs:90`)** is replaced by `TRAITS[id].name` in this ticket — it is a fourth table of the same fact.

### Table E — the 24 "keep?" rows resolved

Rule applied: `merge` iff the enum's variants are ggml tensor types and nothing else (a subset of Table S by meaning, not by spelling); `keep(concept)` names the different concept. **Execution of every `merge` below is Q2 (#3423 Phase 2), not M1** — M1 lands the target they merge into.

| enum | file | verdict |
|---|---|---|
| `QuantKernel` | crates/aprender-cgp/src/profilers/quant.rs:9 | keep(kernel identity — `Nf4Gemv` is not a ggml type) |
| `QuantType` | crates/aprender-core/src/format/quantize.rs:39 | keep(APR's own on-disk id space: `0x01/0x02/0x10/0xFF`, has `Q8Tensor`, `Custom`) |
| `GgufValue` | crates/aprender-cbtop/src/quantize/gguf.rs:24 | keep(metadata value, not tensor type) |
| `AprQuantizationType` | crates/aprender-serve/src/apr_transformer/loader.rs:303 | merge (F32/Q4_K/Q8_0 — a ggml subset used as a loader selector) |
| `Quantization` | crates/aprender-train-inspect/src/convert.rs:113 | merge (Q4_0/Q8_0/F16) |
| `KvQuantType` | crates/aprender-serve/src/paged_kv/mod_compute_prefix.rs:254 | keep(KV-cache precision: FP32/Q8/Q4 are cache layouts, not ggml block formats) — **dissent:** lane 1 would merge, lanes 2 and 3 keep; Q2 re-rules after reading the cache's block layout |
| `QuantizedKvData` | crates/aprender-serve/src/paged_kv/mod_compute_prefix.rs:410 | keep(data carrier, not a type id) |
| `GgufQuantization` | crates/aprender-train/src/hf_pipeline/export/gguf_writer.rs:13 | merge → `Option<GgmlType>` (`None` variant is "don't quantize") |
| `GGUFQuantType` | crates/aprender-train/src/quant/gguf_quant/quant_type.rs:5 | merge (Q4_0/Q8_0) |
| `QuantScheme` | crates/apr-cli/src/commands/quantize.rs:27 | keep(user-facing scheme: Int8/Int4/Fp16/Q4K — maps onto types, is not one) |
| `QuantizationType` | crates/aprender-core/src/format/converter_types_expectations.rs:149 | keep(same scheme concept as the row above; these two duplicate EACH OTHER — file for Q2) |
| `QuantizationType` | crates/aprender-registry/src/lineage/mod.rs:53 | keep(lineage metadata: has `Dynamic`) |
| `AutoQuantError` · `QuantizationError` · `QuantPublishError` | crates/apr-cli/src/commands/auto_quant.rs:100 · crates/aprender-orchestrate/src/oracle/rag/quantization/error.rs:7 · crates/aprender-train/src/hf_pipeline/export/publish_pipeline.rs:66 | keep(error types) ×3 |
| `QuantizeArgvVerdict` | crates/apr-cli/src/commands/quantize_flag_parity.rs:49 | keep(verdict) |
| `QuantScheme` | crates/aprender-cbtop/src/grammar/transform.rs:7 | keep(grammar) |
| `QuantizationType` | crates/aprender-core/src/demo/mod.rs:344 · crates/aprender-core/src/stack/mod.rs:223 | keep(demo / stack descriptor) ×2 |
| `GgufValue` | crates/aprender-core/src/format/gguf/types.rs:84 | keep(metadata value) |
| `QuantFamily` | crates/aprender-serve/src/quantize/format_trait.rs:31 | keep(kernel-format family, 2 variants) — which is why §2 names the new enum `GgmlFamily`, not `QuantFamily`; revisit in Q2 |
| `QuantMethod` · `QuantGranularity` · `QuantMode` | crates/aprender-train/src/config/cli/quant_merge.rs:76 · crates/aprender-train/src/quant/granularity/types.rs:7, :19 | keep(training-time quantization config) ×3 |

Count: 4 merge · 20 keep(concept).

## 3. Ordering (X1)

Q1 (this ticket) → T1/T2 (#3433/#3434) → Q2 → Q3. Evidence: 14 of 34 serve raw-integer qtype sites (41 %, file-level) sit in a file that uses a tensor carrier; the falsifier for running T1 ∥ Q2 was < 20 %. Nothing in §1–§2 changes that: M1 retypes no `qtype: u32` field (Table B's 53 rows are untouched), so it does not collide with T1.

## 4. Phases and acceptance commands

| # | phase | scope_paths | `A_i` |
|---|---|---|---|
| 1 | RED: falsifier guard, upstream size fixture, boundary characterization tests (snapshot taken on unchanged code) | scripts/check_one_ggml_type_enum.sh, contracts/ggml-type-v1.yaml, crates/aprender-quant/tests/, crates/aprender-serve/src/gguf/tests/, crates/aprender-compute/src/inference/, crates/aprender-core/src/format/gguf/tests/ | `bash scripts/check_one_ggml_type_enum.sh` exits 1 naming exactly 3 definitions; its case table passes (must-match: the 3; must-not-match: `GgufValueType`, `GgufValue`, `QuantType`, a commented-out enum); characterization tests GREEN on unchanged code |
| 2 | `GgmlType` + `TRAITS[43]` in aprender-quant | crates/aprender-quant/ | `cargo test -p aprender-quant` (lib AND tests/): every live id round-trips, 8 removed ids are `None` / `Err(Removed)`, `TRAITS` vs the pinned fixture with the exception list exactly `[Q8_1]` |
| 3 | compute + core re-export, admission, `Bf16`→`BF16` | crates/aprender-compute/src/inference/, crates/aprender-core/src/format/ | `cargo test -p aprender-compute --lib -- inference:: && cargo test -p aprender-core --lib -- format::` — includes the Phase 1 characterization tests, unchanged |
| 4 | serve re-export, admission at all 7 boundaries, `QTYPE_LABELS` removed | crates/aprender-serve/src/gguf/, crates/aprender-serve/src/apr/, crates/aprender-serve/src/infer/, crates/aprender-serve/tests/apr_coverage.rs | `cargo test -p aprender-serve --lib -- gguf:: apr:: infer:: && cargo test -p aprender-serve --test apr_coverage` + `bash scripts/check_one_ggml_type_enum.sh` exits **0** + `cargo check --workspace --all-targets` (apr-cli's `tests_trace_dispatch.rs` uses core's re-export) |
| 5 | DoD: `pv validate`, mutation (re-add a second enum → guard RED; change one `ADMITTED` set → characterization RED; set `TRAITS[9].type_size = 40` → fixture RED), diff quorum, receipt, PR | docs/audits/, .github CI selection line (needs the operator's web-UI merge click) | `make gate` |

Phases 3 and 4 have disjoint scopes and may run together. The guard is wired into the same PR's CI selection line or it is dark.

## 5. Not in this ticket

Retyping `qtype: u32` fields (Table B) — T1/Q2. The 558 bare-integer lines — #3431. `tensor_byte_size` single refusal — #3432. Executing the 4 Table E merges and the 10 merge-candidates — #3423 Phase 2. Widening any crate's admitted id set.

## 6. Grill record — v1, 2026-09-17, 3 lanes (gemini-3.1-pro-high, gemini-3.8-flash-high, gemini-3.7-flash-high; author fable-5-1): **0/3 PASS**

Reducer: `agreed=false`, `partial=true` (each lane's `.err` had 2 lines beyond workspace narration). Every finding below was re-executed by the orchestrator before being accepted.

| finding | lanes | re-run | disposition in v2 |
|---|---|---|---|
| assoc-const alias `Bf16` fails in `match self` on `&self` | 2 (E0308), 3 (lint, glob) ; 1 said it works | rustc: **E0308 reproduced** — lane 1 was wrong | Q1-b: aliases withdrawn, rename 2 files |
| admission at the GGUF parser only; `apr/mod.rs:324` and 5 more boundaries bypass it | 1, 2, 3 | grep: 7 serve boundaries confirmed | Q1-c rewritten; characterization tests |
| `from_id → Result` and `tensor_bytes → Option` break callers | 2, 3 | `apr/mod.rs:325`, `loader_tests_apr.rs:311` confirmed | signatures kept; `try_from_id` / `checked_tensor_bytes` added |
| Q8_1: 36 vs 40 | 2, 3 → 36 ; 1 → 40 | upstream read at the full sha: C `static_assert` = 36, gguf-py = 40 | Q1-d: 36, named exception |
| `KvQuantType` should merge | 1 ; 3 agrees with keep | not re-run (Q2 executes it) | recorded as dissent in Table E |
| Phase 2 `--lib` filter skips `tests/`; Phase 4 `gguf::types` exercises 0 `GgmlQuantType` sites | 2, 3 | — | §4 acceptance commands widened |
| scope_paths miss `serve/src/apr/`, `serve/src/infer/`, core `format/` siblings, the `QTYPE_LABELS` file; Table E paths not repo-relative | 1, 2, 3 | usage footprint measured: core 22 files under `format/`, serve 9 files in 4 dirs + `tests/` | §4 and Table E corrected |
| home crate `aprender-quant`: sound, no cycle | 1, 2, 3 | cargo metadata (§1.1) | unchanged |
| X1 ordering | no lane contradicted it | — | unchanged |

