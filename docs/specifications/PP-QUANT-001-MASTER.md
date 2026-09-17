# PP-QUANT-001 v1.0 — MASTER — Quant-type dispatch consolidation

**Status:** DRAFT · ticket #3418 · train **0.69.0** (per operator decision 2026-09-17 —
overrides #3418's own request for a dedicated, non-cadence release; see §6.1 for the
risk this decision accepts) · authored 2026-09-17 against `origin/main` @ `425e84888`
**Companions:** `contracts/quant-dispatch-completeness-v1.yaml` (the gate; owed, §5) ·
`crates/aprender-serve/src/quantize/format_trait.rs` (existing partial trait, extended
not replaced) · epic (owed, §7) · `docs/specifications/06x-release-schedule.md` (the
train this rides; not amended by this document — see §6.2 for what would need to change
there)
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
- The crate names 16 `GGUF_TYPE_*` constants total; the GGUF spec defines 30+. Every
  type absent from a given call site's own match arms is a live
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
   registry to the full affine subset of the GGUF spec (Q4_0/1, Q5_0/1, Q8_0/1, Q2_K
   through Q6_K, Q8_K, TQ1_0/TQ2_0) rather than the current 5.
2. **`QuantCodebookFormat`** (new) — lattice/codebook formats (IQ1–IQ4 family). Grid
   tables are ported data from llama.cpp's `ggml-quants.c` (`iq2xs_grid`, `iq3xs_grid`,
   etc.) — reference data, not something to re-derive.
3. **One dispatch function**, `quant_type_traits(qtype: u32) -> QuantTypeTraits`
   (name TBD at implementation — analogous to `ggml_type_traits[]`), returning
   `{byte_size, dequant_to_f32, family: Affine(dyn QuantBlockFormat) | Codebook(dyn
   QuantCodebookFormat)}` in one place. This is the single new public surface every
   call site is migrated to use.

### §2.2 The completeness gate (the mechanical backstop, not the honor system)

`contracts/quant-dispatch-completeness-v1.yaml` (owed) enumerates every `GGUF_TYPE_*`
the format spec defines and asserts `quant_type_traits()` has a non-panicking entry for
each. A missing type is a CI failure at merge time. This is what #1749/#1789/#2535/
#3341/#3091 needed and didn't have — each was an honor-system list, five times over.

### §2.3 Per-backend fan-out is additive, not multiplicative

CPU SIMD, CUDA, and wgpu each implement the trait once per type. The ~30-call-site
migration happens once, regardless of backend count, because call sites stop matching
on `qtype` themselves and instead call the one dispatch function and operate on its
returned trait object / enum.

---

## §3 Phases

Phase boundaries exist so the completeness gate (Phase 1) can go green and be checked
in *before* the 30-file migration (Phase 2) starts touching the CPU/CUDA/wgpu backends
simultaneously — the ordering #3418 asks for, independent of which train it rides on.

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **0 — Inventory** | Enumerate every call site (owed: a checked-in list, not a one-time grep — `scripts/find_qtype_dispatch_sites.sh` producing the same 30+ paths #3418 found, kept current) and every `GGUF_TYPE_*` the spec defines vs. the 16 named today | `scripts/find_qtype_dispatch_sites.sh --format json` diffs cleanly against a checked-in baseline; new sites fail CI until triaged |
| **1 — Trait + gate** | `QuantCodebookFormat` written; `QuantBlockFormat` registry extended to the full affine set; `quant_type_traits()` dispatch fn added (additive — no call site migrated yet); `contracts/quant-dispatch-completeness-v1.yaml` created and passing | `pv validate contracts/quant-dispatch-completeness-v1.yaml` green; the gate itself mutation-tested RED→GREEN (remove one type's entry, gate fails) |
| **2 — Migration** | All ~30 call sites refactored to call `quant_type_traits()` instead of maintaining their own match arms, one call site (or tightly related group) per PR | `grep -rln "GGUF_TYPE_Q4_K\|GGUF_TYPE_Q4_0\|match qtype\|match.*\.qtype" crates/aprender-serve/src/ --include="*.rs" \| grep -v test \| wc -l` → 0 (or only the dispatch function itself) |
| **3 — Parity re-proof** | Full parity re-run of the existing Qwen2.5-Coder GPU-beat measurement (README/BEATS.md's demonstrated model) against pre-migration baseline, per quant type touched, per backend | cosine ≥ 0.98 vs. pre-migration baseline (the bar #3091's own history already holds this codebase to); no regression on `contracts/beat-ollama-decode-throughput-speed-v1.yaml` |

Phase 2 is the ~30-file, multi-backend blast radius #3418 warns about. Phases 0/1 are
additive and low-risk; they can land first inside 0.69 without touching a live call
site, so Qwen2.5-Coder's proven path is untouched until Phase 3 is ready to re-verify it.

---

## §4 Missing quant types this closes (relative to today's 16)

Per #3418 evidence: `GGUF_TYPE_Q8_1`, `GGUF_TYPE_Q8_K`, `GGUF_TYPE_TQ1_0`,
`GGUF_TYPE_TQ2_0`, and the full IQ family (`IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`,
`IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`) — 12 types, none of which have a dequant path
anywhere in `crates/aprender-serve/src/` today `[A]` (re-verify at implementing HEAD:
`grep -rohE "GGUF_TYPE_[A-Z0-9_]+" crates/aprender-serve/src/ | sort -u`).

---

## §5 Owed artifacts (do not exist at `425e84888`, written without backticks per the
`PP-LLAMA-001` convention so the drift gate is never asked to check a file this document
is asking someone to create)

- contracts/quant-dispatch-completeness-v1.yaml — Phase 1 exit gate
- scripts/find_qtype_dispatch_sites.sh — Phase 0 inventory, checked-in baseline + diff
- crates/aprender-serve/src/quantize/codebook_trait.rs — QuantCodebookFormat + grid tables
- crates/aprender-serve/src/quantize/dispatch.rs — the single `quant_type_traits()` entry point
- docs/audits/impl-PP-QUANT-001-phaseN-receipt.md — one per phase, RED→GREEN proof

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

1. Open a GitHub epic under milestone `0.69.0`, linking #3418, for Phase 0/1 work.
2. Assign an owner (currently unassigned).
3. Phase 0 inventory as the first PR — no trait/gate code, just the checked-in call-site
   list — so Phase 1's contract has a real enumeration to validate against instead of a
   re-transcription of #3418's evidence section.
