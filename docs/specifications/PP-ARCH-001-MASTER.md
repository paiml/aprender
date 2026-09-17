# PP-ARCH-001 v1.0 — MASTER — Shared-block forward-path consolidation

**Status:** DRAFT · ticket #3422 · epic #3423 · milestone **PP-ARCH-001** (dedicated,
not the 06x 2-3 day cadence — see §6) · authored 2026-09-17 against `origin/main` @
`1d7dcc5e9`
**Prerequisite:** PP-QUANT-001 (`docs/specifications/PP-QUANT-001-MASTER.md`, #3418,
epic #3421). A shared attention/FFN composition layer must be quant-agnostic to be
reusable across architectures; that needs PP-QUANT-001's dispatch table (its §2.4).
This document assumes PP-QUANT-001 Phase 1 is done and does not re-derive it.
**Companions:** `contracts/shared-block-adoption-v1.yaml` (owed, §5) ·
`crates/aprender-serve/src/gguf/inference/forward/attention.rs`,
`ffn_block.rs` (the existing, under-adopted shared layer this builds on)
**Governs:** how a model's forward pass in `crates/aprender-serve/` is composed —
what must come from a shared module, what may be architecture-specific, and how new
architecture-specific operators get a place to live without becoming a 31st
private reimplementation of attention or RMSNorm.
**Owner:** unassigned (§7)

---

## §0 Conventions

Same marks as `PP-QUANT-001-MASTER.md` §0 / `PP-LLAMA-001-MASTER.md` §0.3: `[V]`
verified by a cited command in this session · `[C]` calculated from `[V]` inputs ·
`[A]` asserted, re-verify at implementing HEAD · `[U]` unmeasured, row names who
measures it.

---

## §1 Problem, and why it is not the same problem as PP-QUANT-001

#3418 (PP-QUANT-001) is real and necessary, but closing it does not deliver "add a
new architecture the way llama.cpp does" — of the ~1,678 lines #3418 counted for
Qwen3-MoE, only ~85 are quant-type dispatch (`[V]` §4 below); the other ~1,600 are
forward-pass logic PP-QUANT-001 never touches. This document is about that ~1,600.

**The infrastructure already exists.** `crates/aprender-serve/src/gguf/inference/
forward/attention.rs` has 25 functions — `parallel_multihead_attention_gpu`,
`standard_softmax`, `online_softmax`, `tiled_single_head_attention`,
`reshape_for_parallel_heads`, `parallel_batched_qk_scores`, and more — and
`ffn_block.rs` has 24. This is aprender's version of llama.cpp's `build_attn` /
`build_moe_ffn`. It is not a from-scratch design problem.

**Adoption is zero on the two newest, largest architectures**, measured `[V]`:

```
$ grep -cE 'standard_single_head_attention|parallel_multihead_attention|parallel_batched_qk_scores|standard_softmax|online_softmax|tiled_single_head_attention|reshape_for_parallel_heads' \
    forward_qwen35.rs forward_qwen3_moe.rs
forward_qwen35.rs:0
forward_qwen3_moe.rs:0
$ grep -cE 'ffn_block::|adaptive_ffn::' forward_qwen35.rs forward_qwen3_moe.rs
forward_qwen35.rs:0
forward_qwen3_moe.rs:0
```

`forward_qwen35.rs` (939 lines) has its own `forward_attention` (line 896) with its
own softmax, scaling, and head-splitting — none of it calling the shared module sitting
one directory away. `forward_qwen3_moe.rs` (321 lines) is the same.

**Not all of the 939/321 lines are avoidable.** `forward_qwen35.rs` also implements
genuinely novel operators for the Qwen3.5/Next hybrid architecture —
`delta_rule_recurrence`, `causal_conv1d`, `gated_rmsnorm`, `apply_partial_neox_rope`
(gated DeltaNet linear attention interleaved with full attention) — with no
equivalent anywhere in the shared modules, because nothing else in the tree needs
them yet `[V]`. llama.cpp had to add graph ops for the same thing; its shortness
comes from having a *place to plug a new op into*, not from every architecture being
math aprender already has. The claim here is narrower and falsifiable per-file
(§3 Phase 0): some fraction of each forward is reimplementation of attention/
softmax/head-reshaping that already exists elsewhere in the same crate (avoidable,
currently unenforced); some fraction is new math (not avoidable, needs a slot).

**Duplication is broader than these two files.** RoPE is implemented independently
in ~25 files, attention math in 20+, RMSNorm in 38 `[V]` (§4). Quant dispatch
(#3418) is one axis; forward-pass logic is the larger one, and it is the axis that
sets how many lines a new architecture costs.

**Per-backend duplication compounds it.** `forward_qwen3_moe.rs` (CPU, 321 lines)
and `forward_qwen3_moe_gpu.rs` (GPU, 66 lines) are separate entry points per
architecture, not one graph two backends walk — the same multiplication #3418
described for quant dispatch, one layer up (`architectures × backends`, now on
forward logic instead of just dequant kernels).

---

## §2 Target: what "adding an architecture" should cost

**§2.1 Definition of done, stated as a falsifiable exit criterion (not an LOC target
chosen in advance):** a new dense or MoE architecture that needs no operator absent
from the shared layer lands as (a) a loader/tensor-name mapping file, and (b) a
forward function whose body is calls into shared blocks plus architecture-specific
config (layer count, head dims, MoE routing parameters) — with **zero** new
reimplementations of attention, softmax, RMSNorm, or RoPE. The measure is
`contracts/shared-block-adoption-v1.yaml`'s adoption gate (§5), not a line count,
because line count alone can be gamed (one giant shared function is not the goal;
the goal is that the *specific* operators named in §2.1 are never rewritten).

**§2.2 A slot for genuinely new operators.** Not every future architecture reuses
existing blocks — Qwen3.5/Next's DeltaNet did not, and the next hybrid architecture
after it may add something else again. The composition layer must let an
architecture register a new operator (like `delta_rule_recurrence`) as a
first-class, independently testable unit *without* that registration becoming
license to also reimplement attention/softmax/RoPE that already exist. §2.1's gate
is scoped to the operators that already have a shared implementation; it does not
forbid new operators, it forbids *re*implementing old ones.

**§2.3 Composition, not necessarily a graph.** llama.cpp's mechanism is a compute
graph (`ggml_cgraph`) built once per forward and walked generically. Aprender does
not need to adopt a graph IR to get the same reuse property — Rust's existing
`attention.rs`/`ffn_block.rs` functions already are the "graph nodes"; what's
missing is a *convention that architecture forwards call them* plus a *gate that
enforces it*, which is far smaller than introducing a graph builder/executor.
Phase 1 (§3) evaluates both shapes and states which one PP-ARCH-001 adopts, with
the plain-function-composition shape as the default hypothesis given how much of
the target already exists.

---

## §3 Phases

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **0 — Classification** | Every function in every `gguf/inference/forward/*.rs` file, and every independent RoPE/attention/RMSNorm implementation elsewhere in `crates/aprender-serve/src/` (§4's file lists), classified as `shared` (already in `attention.rs`/`ffn_block.rs`, or a match for one), `duplicate` (reimplements something `shared` already has), or `novel` (a real new operator, no shared equivalent). Checked into a contract (`contracts/shared-block-adoption-v1.yaml`, owed), mirroring PP-QUANT-001's Phase-0-as-contract pattern | The classification is re-derivable by a script/oracle, not hand-maintained; totals match §4's counts |
| **1 — Composition design** | Decide §2.3's open question (function-composition convention vs. a graph IR) and specify it; design the "new operator slot" (§2.2) so `delta_rule_recurrence`-class additions have a defined home instead of living inside a monolithic forward file | Design section written and reviewed; no code changes yet |
| **2 — Migration** | Every `duplicate`-classified site replaced with a call into the shared module, starting with the proven baseline (Qwen2.5-Coder) so regressions are caught against the model with an actual parity receipt, then Qwen3-MoE, then Qwen3.5 | `contracts/shared-block-adoption-v1.yaml`'s gate shows zero `duplicate` rows remaining; parity re-run (§3 row 3 below) shows no regression on `docs/BEATS.md`'s Qwen2.5-Coder measurement |
| **3 — Proof on a real new architecture** | Land Qwen3-Next / Qwen3.8-Flash (already sitting untested per #3418's own note) using only the Phase 1 composition convention plus whatever new operators Phase 3 itself discovers it needs | The new architecture's own forward file's `duplicate` count is zero from the day it lands (never entered the debt in the first place); GPU-parity re-proof against Qwen2.5-Coder still holds |

Phase 0/1 are additive — no existing forward is touched, so they can start
immediately without regression risk. Phase 2 is the large-blast-radius work (it
touches the proven Qwen2.5-Coder path) and is exactly why this milestone exists
outside the 06x cadence (§6). Phase 3 is the actual "new model support like
llama.cpp" proof the whole effort is for.

---

## §4 The universe, measured (2026-09-17, `1d7dcc5e9`)

| Fact | Value | Measured by |
|---|---|---|
| Qwen3-MoE lines that are quant dispatch (of #3418's cited 1,678) | ~85, all in `qwen3_moe_load.rs`; `qwen3_moe_generate.rs` has 0 | `grep -cE 'qtype\|GGUF_TYPE_\|WeightQuantType\|Q4_K\|Q6_K\|Q4_0\|Q8_0'` on both files `[V]` |
| `gguf/inference/forward/*.rs` line counts (non-blank, non-comment) | `forward_qwen35.rs` 939 · `ffn_block.rs` 633 · `forward_qwen3_moe_traced.rs` 328 · `forward_qwen3_moe.rs` 321 · `forward_cached.rs` 314 · `batched.rs` 284 · `encoder_decoder.rs` 167 · `forward_qwen3_moe_gpu.rs` 66 · `gemma_dispatch.rs` 32 | `grep -vc '^\s*//\|^\s*$'` per file `[V]` |
| Shared attention functions that exist | 25 (`attention.rs`) | `grep -cE '^\s*(pub )?fn '` `[V]` |
| Shared FFN functions that exist | 24 (`ffn_block.rs`) | same `[V]` |
| Shared-function calls from `forward_qwen35.rs` / `forward_qwen3_moe.rs` | **0 / 0**, both modules | grep listed §1 `[V]` |
| Files independently implementing RoPE | ~25 | `grep -rl 'fn (rope\|apply_rope\|rotate_half)'` over `crates/aprender-serve/src` `[V]` |
| Files independently implementing attention math | 20+ | `grep -rl 'fn.*(attention\|scaled_dot_product\|causal_attn)'` `[V]` |
| Files independently implementing RMSNorm | 38 | `grep -rl 'fn.*rmsnorm'` `[V]` |
| Novel (non-shared-equivalent) operators found in `forward_qwen35.rs` | `silu`, `softplus`, `l2_norm`, `l2_norm_per_head`, `apply_sigmoid_gate`, `gated_rmsnorm`, `causal_conv1d`, `delta_rule_recurrence`, `apply_partial_neox_rope` | function listing `[V]`; classification into novel vs. duplicate is Phase 0's job, this row is the raw candidate list, not the verdict |

---

## §5 Owed artifacts

- `contracts/shared-block-adoption-v1.yaml` — Phase 0 classification + the adoption gate, in `pv`'s registry-contract shape (mirrors `contracts/quant-dispatch-completeness-v1.yaml`'s `dispatch_sites` pattern: one row per forward file/function, `class: shared|duplicate|novel`, re-derivable by oracle)
- `scripts/gen_shared_block_classification.sh` (or the Rust test that plays the oracle's role directly, per PP-QUANT-001's `qdc_dispatch_sites_match_oracle` precedent) — Phase 0
- The composition-layer design write-up itself (Phase 1's actual deliverable is a decision, recorded in this document as a §2.3 amendment, not a separate file)
- `crates/aprender-serve/src/gguf/inference/forward/` new-operator-slot mechanism (name TBD at Phase 1) — Phase 1
- Per-Phase-2-PR: the migrated file, plus a parity receipt against the Qwen2.5-Coder GPU-beat measurement for anything touching a live inference path
- `docs/audits/impl-PP-ARCH-001-phaseN-receipt.md` — one per phase

---

## §6 Why this is its own milestone, not a 06x train

Per `docs/specifications/06x-release-schedule.md` §1.1, a train leaves in a fixed
48-72h window regardless of scope. Phase 2 of this document touches the same
CPU/CUDA forward paths that carry aprender's one proven parity receipt
(Qwen2.5-Coder, `docs/BEATS.md`) — a strictly larger blast radius than
PP-QUANT-001's Phase 2 (which #3418 itself asked to be dedicated, and which the
operator instead placed on 0.69, accepting that risk explicitly — PP-QUANT-001 §6).
This document's Phase 2/3 is not offered for a 0.69/0.70-style window: milestone
**PP-ARCH-001** (created 2026-09-17, distinct from the 06x milestones) holds it,
with no train-leaves-by deadline. Phase 0/1 are additive and low-risk and could in
principle ride a 06x train if that's ever wanted, but nothing requires it — this
milestone has no clock, which is the point.

---

## §7 Next steps

1. Assign an owner (currently unassigned).
2. Phase 0 as the first PR: the classification contract, built the same way
   PP-QUANT-001's Phase 0 was — enumerate, cite the oracle, `pv validate` green,
   no code changes to any forward file yet.
3. Phase 1 as the second PR: the composition-layer decision (§2.3), written as an
   amendment to this document's §2.3, reviewed before any migration starts.
