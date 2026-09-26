# PP-ARCH-001 v1.0 — MASTER — Shared-block forward-path consolidation

**Status:** DRAFT · ticket #3422, a sub-issue of the umbrella #3418 (see #3418's
2026-09-17 comment for how the two relate) · epic #3423, which also tracks
PP-QUANT-001 Phase 2/3 · milestone **`Inference dispatch & architecture
consolidation`** (dedicated, not the 06x 2-3 day cadence — see §6) · authored
2026-09-17 against `origin/main` @ `1d7dcc5e9`
**Prerequisite:** PP-QUANT-001 (`docs/specifications/PP-QUANT-001-MASTER.md`, #3418,
epic #3421 for its Phase 0/1). A shared attention/FFN composition layer must be
quant-agnostic to be reusable across architectures; that needs PP-QUANT-001's
dispatch table (its §2.4). This document assumes PP-QUANT-001 Phase 1 is done and
does not re-derive it. This document's own Phase 2/3, and PP-QUANT-001's Phase 2/3,
share the same milestone and epic (#3423) — they are the two sub-projects of the one
dedicated cycle, not independent scope.
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

**Decided 2026-09-26 (Phase 1, §9): plain-function composition, no graph IR.**

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
PP-QUANT-001's Phase 0/1 (which rode 0.69; PP-QUANT-001 §6). This document's Phase
2/3 is not offered for a 0.69/0.70-style window: milestone `Inference dispatch &
architecture consolidation` (created 2026-09-17, distinct from the 06x milestones,
renamed from its original working title `PP-ARCH-001` once it absorbed
PP-QUANT-001's own Phase 2/3 — see #3423) holds it, with no train-leaves-by
deadline. Phase 0/1 of this document are additive and low-risk and could in
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

---

## §8 Phase 0 delivered (2026-09-26, stage/0.70.1 @ `b8edaf177`, #3422)

The classification is a re-derivable oracle, not a hand list:
`scripts/classify_shared_block_adoption.py` (contract
`contracts/shared-block-adoption-v1.yaml`, CI fragment
`ci/explicit-test-commands.d/490-shared-block-ratchet.cmd`, baseline
`docs/audits/shared-block-duplicates-baseline.tsv`).

**Rule.** A non-test fn in `crates/aprender-serve/src` whose name is in a §2.1
family is `shared` if it lives in a shared home (`gguf/ops.rs`,
`forward/attention.rs`, `forward/ffn_block.rs`); otherwise `duplicate` if its
body carries the family's math (rmsnorm: sqrt/rsqrt; rope: sin/cos/powf; softmax:
exp; attention: an inline exp, or a softmax call plus a sqrt scale), else
`wrapper` (launcher, dispatch, getter). String literals and comments are
blanked first, so PTX/WGSL source is never read as CPU math. `novel` is a
reviewed OVERRIDE with a reason — at creation the four fused RMSNorm+Q8_0
quantize kernels.

| family | shared | duplicate | wrapper | novel |
|---|---|---|---|---|
| rmsnorm | 6 | 15 | 41 | 4 |
| rope | 1 | 16 | 52 | 0 |
| softmax | 3 | 15 | 4 | 0 |
| attention | 5 | 26 | 133 | 0 |

**72 duplicates across 40 files.** This replaces §4's grep-by-name counts (~25
RoPE files, 20+ attention, 38 RMSNorm), which counted launchers and wrappers as
implementations. Adoption: 18 of 33 non-test forward files call `gguf/ops.rs`;
**4 of 33** call `attention.rs`/`ffn_block.rs`. Qwen3.5 calls 0 of the blocks
and Qwen3-MoE calls 1 (`single_cache_final_output`). `--summary` prints the
per-file table.

**Gate.** The baseline only shrinks. A new duplicate is RED. So is a baselined
row whose fn is gone: a Phase-2 migration must delete its row, so the gain it
freed cannot be spent later. A stale override is also RED. Falsified
2026-09-26 by planting each of the three: rc=1 for each, with the planted
fn/row/override named, and rc=0 after restore.

**Known under-count (conservative by design).** An attention fn that calls
softmax but computes its scale without `sqrt`, for example a precomputed
`scale` argument, reads as `wrapper`. The ratchet can miss such a new
duplicate, but it never flags a launcher. Tightening it is a Phase 2 entry
item (§9.5).

**Open for Phase 1:** whether `gated_rmsnorm`/`rms_norm_gated` and
`apply_partial_neox_rope` are `novel` (§2.2) or compositions of shared ops.
They stay `duplicate` until that is decided.

---

## §9 Phase 1 — composition decision (2026-09-26, #3422)

Design only; no forward file changes. Phase 2 implements it.

### §9.1 Decision: plain functions, not a graph IR

PP-ARCH-001 adopts **plain-function composition**. It does not adopt a
`ggml_cgraph`-style builder and executor. The deciding fact comes from the code,
not from taste. The shared blocks already exist, but they are **inherent methods
on `OwnedQuantizedModel`**. `attention.rs` and `ffn_block.rs` are each one
`impl OwnedQuantizedModel` block. Examples are `standard_softmax(&self, ..)`
and `standard_single_head_attention(&self, ..)`. `forward_qwen35.rs` defines its
own `Qwen35Model`, and `apr_transformer/` defines another model type again.
Neither *can* call a method on a type it does not hold. That is why §8 measures
4 of 33 forward files calling the blocks and Qwen3.5 calling 0. The missing
reuse is a **type-coupling** defect. It is not a missing-graph defect, and a
graph IR would not fix it: its nodes would still need a callable,
model-independent operator underneath.

A graph IR is also rejected on cost:
- It adds a runtime indirection on the CPU decode hot path, which §BEATS
  measures at parity.
- The GPU path already has its own executor (`cuda/executor/`), so a second
  graph layer would duplicate scheduling.
- The §2.1 gate is function-granular, which a graph would obscure.

A graph is revisited only if Phase 3 finds an architecture whose forward cannot
be written as a straight-line sequence of calls. Qwen3.5's hybrid layer schedule
can be written that way: it is a per-layer `match` on layer kind.

### §9.2 The convention (what Phase 2 migrates to)

1. **A shared op is a free `pub fn` over slices plus a small `Copy` config
   struct.** It never takes `&self` of a model type. Example:
   `fn rope_neox_into(x: &mut [f32], shape: HeadShape, n_rot: usize, pos: usize, base: f32)`.
   A model method may remain as a one-line delegating wrapper. The oracle
   already classifies that as `wrapper`, not `duplicate`.
2. **Canonical homes, one per family:**

   | family | home | canonical fn(s) |
   |---|---|---|
   | rmsnorm | `gguf/ops.rs` (exists) | `rms_norm[_into]`, `rms_norm_unit_offset[_into]`, `apply_per_head_rms_norm`; **new** `rms_norm_gated_into` (§9.4) |
   | softmax | `gguf/ops.rs` (exists) | `softmax(&mut [f32])` — `attention.rs::standard_softmax` becomes a wrapper over it |
   | rope | `gguf/ops.rs` (**new**; today RoPE has no shared home, so §8 counts only 1 shared rope fn against 16 duplicates) | `rope_neox_into`, `rope_normal_into`, both with `n_rot <= head_dim` |
   | attention | `forward/attention.rs`, hoisted to free fns | `single_head_attention(q, k, v, shape, scale)`, `multihead_attention(..)`, tiled/online variants |

3. **A forward file is config plus calls.** An architecture's forward body
   contains projections through the quantized-matmul API (PP-QUANT-001's
   territory), calls into the homes above, and its own control flow: layer
   schedule, MoE routing, residual wiring. The §2.1 gate enforces the negative
   half of this rule.

### §9.3 The novel-operator slot (§2.2)

A novel operator lives in **`crates/aprender-serve/src/gguf/novel_ops/<operator>.rs`**.
Each file holds one operator, as a free fn under the §9.2.1 signature rule, with
its own unit tests and a contract falsifier. DeltaNet's
`delta_rule_recurrence`, `causal_conv1d` and `l2_norm`, which today sit inside
`forward_qwen35.rs`, are the first tenants.

What the slot does **not** grant:
- Names outside the four §2.1 families need no oracle row at all.
- A §2.1-family name in `novel_ops/` is **not** exempt by location. It needs an
  `OVERRIDES` row with a reason, exactly like the four fused RMSNorm+Q8_0
  quantize kernels. Otherwise the slot becomes the loophole §2.2 forbids.
- A novel op must call shared ops for any §2.1 sub-step it contains.

### §9.4 Rulings on the three open rows (§8)

- **`gated_rmsnorm` (forward_qwen35.rs) and `rms_norm_gated`
  (gpu/scheduler/linear_attn.rs): composition, so `duplicate`.**
  - Both compute per-`head_v_dim` chunk RMSNorm × weight × `silu(gate)`.
  - They are the *same* op written twice, differing only in allocation (`&mut
    out` versus returning a `Vec`). Both are `rms_norm_into` per chunk followed by
    an elementwise `silu` multiply, and `ops.rs` already has both.
  - Phase 2 adds `ops::rms_norm_gated_into` and turns both sites into calls.
    That removes 2 baseline rows.
- **`apply_partial_neox_rope` (forward_qwen35.rs): composition, so
  `duplicate`.**
  - It is NeoX RoPE restricted to the first `n_rot` dims of each head.
    `gguf/inference/rope.rs::apply_rope` is the `n_rot == head_dim` case.
  - Once `ops::rope_neox_into` takes `n_rot`, both become calls. **Numeric
    caveat:** the partial variant advances theta iteratively (`theta *=
    theta_scale`, as `ggml_rope_multi` does), while `apply_rope` precomputes
    per-pair powers. Those differ in the last ulps.
  - The shared fn keeps the iterative form, which matches the llama.cpp
    oracle. Each migrated site must re-pass its own parity receipt. The
    Qwen2.5-Coder BEATS row is first, per §3 row 2.

### §9.5 Phase 2 entry items

- **Tighten the attention under-count (§8).** Add a rule that treats a `softmax`
  call plus a `*`-by-`scale` identifier as attention math. Land it with a
  self-test row and a re-baseline in the same commit, so the ratchet never
  loosens.
- **Order of migration**, one commit per family, each deleting its baseline
  rows:
  1. Hoist `attention.rs`/`ffn_block.rs` to free fns, with the model methods
     kept as wrappers. The row count is unchanged, but this unblocks
     everything after it.
  2. RoPE home: `ops::rope_*_into` and its 16 rows.
  3. rmsnorm, including the §9.4 gated pair: 15 rows.
  4. softmax: 15 rows.
  5. attention: 26 rows.

  Qwen2.5-Coder sites go first within each family (§3 row 2).

### §9.6 Step 1 delivered (#3422)

This step preserves behaviour: every hoisted body moved byte for byte, and
each model method became a one-line delegate.
- **`forward/attention.rs`** gained free fns: `standard_softmax`,
  `online_softmax`, `standard_single_head_attention`,
  `tiled_single_head_attention`, and, under `gpu`, `reshape_for_parallel_heads`
  and `parallel_batched_qk_scores`.
- **`forward/ffn_block.rs`** gained free fns: `ffn_gated_activate` (`use_gelu`
  replaces the `&self` GeGLU lookup), `first_token_attention` and
  `post_norm_in_place`.
- All are re-exported `pub(crate)` from `gguf::inference::forward`.

Not hoisted in this step:
- **`parallel_multihead_attention_gpu`**: it calls
  `acceleration.rs::apply_causal_mask_softmax`, which is still a method.
- **The weight-bound `single_cache_*` blocks.** They read `self.layers[i]` and
  dispatch through the quantized matmul. Their free form needs a borrowed
  per-layer weight view, and that is PP-QUANT-001's API to shape, not this one.

Folding `standard_softmax` into `ops::softmax` changes numerics: `/ sum`
becomes `* (1/sum)`. So it belongs to the softmax-family step, with its parity
receipt.


### 9.7 Phase 2 step 2 delivered: RoPE shared home (2026-09-26)

The shared home is `gguf::ops::rope_into(x, num_heads, head_dim, position, theta,
RopeStyle)`, where `RopeStyle::{Norm, Neox}` and `RopeStyle::from_rope_type(2) == Neox`.
It builds the sin/cos table once per call instead of once per head.

**Proof of equivalence.** `ops::rope_into_equivalence_tests` keeps the old per-site
loop frozen and compares `to_bits()` across 360 cases:
- head_dim {2, 64, 80, 128, 256, 512}
- heads {1, 3, 8}
- positions {0, 1, 17, 4095, 131071}
- theta {1e4, 1e6}
- both styles
- a trailing partial head

Every migrated site computed the same `freq = 1/theta^(2i/d)` and `(pos*freq).sin_cos()`,
so the change is bit-identical and needs no parity receipt.

**Migrated (11 baseline rows deleted, 72 → 61).** Each of these now calls the shared home:
- `apr/helpers.rs::apply_rope_norm`
- `apr_transformer/helpers.rs::apply_rope_f32`
- `apr_transformer/attention_kernels.rs::apply_rope`. Its private
  `apply_rope_to_head` and `apply_rope_quad` are deleted.
- both `cuda/executor/*::apply_rope_to_buffer`
- `gpu/adapters/apr_q4_apply_rope_gpu.rs::apply_rope_inplace`
- `gpu/scheduler/kv.rs::apply_rope`
- `gpu/scheduler/ops.rs::apply_rope_inline`. Its contract macros are kept.
- `gpu/simd_ops.rs::scalar_rope`
- `inference/norm.rs::apply_rope`

**Behaviour change on malformed input only.** If a head does not fit in `x`, it is now
skipped. Before, some sites rotated a partial pair and others panicked on the index.

**Deferred, because each one's numerics differ and needs a parity receipt:**
- `gguf/inference/rope.rs::apply_rope`: NEOX runs the AVX2/AVX-512 FMA kernel. This
  is the Qwen2.5-Coder parity path.
- `forward_qwen35.rs::apply_partial_neox_rope`: iterative theta. It is a composition
  under §9.3.
- `gpu/adapters/apr_q4k.rs::apply_rope_neox`: computed in f64.
- `gpu/simd_ops.rs` frequency and trig tables: trueno vectors.

### 9.8 Phase 2 step 3 delivered: scalar RMSNorm shared home (2026-09-26)

**Why a second home.** `ops::rms_norm` / `rms_norm_into` sum with trueno SIMD, which
reorders the adds. Every duplicate site summed left to right, so moving them onto the
SIMD home would change numerics. They move instead onto three new scalar functions,
each bit-identical to the loop it replaces:
- `ops::rms_scalar(x, eps)` returns `sqrt(sum(x^2)/n + eps)`.
- `ops::rms_norm_scalar_into(x, w, eps, RmsScale, out)`
- `ops::rms_norm_scalar_in_place(x, w, eps, RmsScale)`

**`RmsScale` names the rounding each site used:**
- `Divide`: `(x / rms) * w`
- `ScaleThenWeight`: `(x * inv) * w`
- `WeightedScale`: `x * (inv * w)`

`the_three_forms_are_distinct_so_the_enum_is_load_bearing` proves the three forms
differ in bits. So collapsing them to one form, or onto the SIMD home, needs a parity
receipt and is a later step.

**Proof of equivalence.** `rms_norm_scalar_equivalence_tests` keeps the old loops frozen
and compares `to_bits()`, for both the `into` and `in_place` paths:
- n in {1, 2, 7, 64, 128, 896, 1536, 4096}
- two eps values
- three magnitudes
- all three forms

That is 144 cases.

**Migrated (13 baseline rows deleted, 61 → 48):**
- `apply.rs` and `gamma.rs`: `apply_rms_norm_cpu` and `apply_rms_norm_layer_cpu` (Divide)
- `apr/helpers.rs::rms_norm`, the non-gpu branch (Divide)
- `apr_transformer/helpers.rs::rms_norm` (Divide, then bias)
- `q4_simd_activations_cache.rs`: `rms_norm_weighted` (Divide), and `rms_norm_batched`,
  which now calls it per row
- `cuda/executor/layer_norm_gpu.rs::rmsnorm_into`, the `CPU_RMSNORM=1` diagnostic
  bypass (Divide)
- `gpu/adapters/apr_q4k.rs`: `rms_norm` (ScaleThenWeight) and `per_head_rms_norm`
  (WeightedScale)
- `gpu/adapters/using.rs::rms_norm_inplace`, via `rms_scalar`. It keeps its
  weight-fallback-1.0 loop.
- `inference/norm.rs::simd_rms_norm` (ScaleThenWeight)

**Behaviour change on malformed input only.** Output now covers the shortest of `x`,
`weight` and `out`. Sites that used to index `weight[i]` past its end no longer panic.

**Left as duplicates:**
- `forward_qwen35.rs::gated_rmsnorm` and `linear_attn.rs::rms_norm_gated`. These are
  compositions under §9.3, and their home is `ops::rms_norm_gated_into` in a later step.

### 9.9 Phase 2 step 4 delivered: scalar softmax shared home (2026-09-26)

**One home, two roundings.** Every scalar softmax site did the same three things: took
the max with `f32::max` folded from `-inf`, exponentiated in place, and summed left to
right. `iter().sum()` adds in that same order, and none of the exponentials is `-0.0`,
so it counts as the same. The sites differ only in the final division, so the home
splits at that point:
- `ops::softmax_exp_in_place(x) -> sum`
- `ops::softmax_normalize(x, sum, SoftmaxNorm)`, where `Divide` is `e / sum` and
  `MulInv` is `e * (1.0 / sum)`
- `ops::softmax_scalar_in_place(x, SoftmaxNorm)`, which does both. `ops::softmax` now
  calls it with `MulInv`, keeping its contract precondition.

Two sites normalised only when `sum > 0.0`. They call the two halves themselves and
keep that check, because it changes the output when the sum is NaN.

**Proof of equivalence.** `softmax_scalar_equivalence_tests` keeps four frozen copies of
the old loops: divide, mul-inv, collect-then-`iter().sum()`, and guarded mul-inv. It
compares them by `to_bits()`:
- n in {1, 2, 7, 64, 151, 1024, 32000}
- four magnitudes
- rows with and without `-inf` causal masks

That is 224 cases. `the_two_forms_are_distinct_so_the_enum_is_load_bearing` proves
`Divide` and `MulInv` differ in bits.

**Migrated (12 baseline rows deleted, 48 → 36):**
- `apr/helpers.rs::softmax_causal`
- `apr_transformer/attention_kernels.rs::softmax_inplace`
- `gguf/inference/cached/attention.rs::batched_causal_softmax` (guarded; the causal row
  is copied, then normalised in place)
- `gguf/inference/forward/acceleration.rs::apply_causal_mask_softmax`
- `gpu/scheduler/attention.rs`: `apply_causal_softmax` and `softmax_inplace`
- `gpu/simd_ops.rs::scalar_softmax`
- `inference/simd.rs::simd_softmax` (guarded, MulInv)
- `layers/mod.rs::softmax` (per row; its postcondition is unchanged)
- `quantize/quantize_rmsnorm_into.rs::softmax_scalar` (MulInv)

All of these use `Divide` unless noted.

**Dead file removed.** `quantize/avx2.rs` was compiled nowhere: nothing `include!`s it
and no `mod` declares it. It is a stale copy of `quantize_rmsnorm_into.rs` from before
PMAT-780, and it carried two of the softmax rows. Its `OVERRIDES` entry is deleted with
it, because a stale override is RED.

**Left as duplicates, each needing a parity receipt:**
- `bench/…itl_metrics.rs::softmax`: computes in f64, so it is a different operator.
- `gpu/simd_ops.rs::simd_softmax`: sums with trueno SIMD, so its add order differs.
- `quantize_rmsnorm_into.rs::softmax_avx2`: the live AVX2 kernel. `_mm256_max_ps`
  handles NaN differently from `f32::max`.

### 9.10 Phase 2 step 5a delivered: tiled attention home and dead copies (2026-09-26)

Step 5 is the attention family, which had 26 baseline rows. A read-only survey split
them into four groups:
- **Dead** (4 rows). Nothing compiles these files: no `mod` declares them and no live
  file `include!`s them.
- **Family B** (3 rows). Online-softmax tiled attention.
- **Family A** (11 rows). Scalar one-row attention. These are left for step 5b.
- **Different operators.** SIMD dot/axpy, trueno, GPU matmul, and the
  `flash_attention_dispatch.rs` rescale form. These stay as duplicates, each needing a
  parity receipt.

**Family B home.** `ops::attend_row_online_tiled(q_i, k, v, head_dim, n_keys, scale,
tile_size, out)` holds the per-row body of `tiled_causal_attention`,
`tiled_bidirectional_attention` and `tiled_cross_attention`, moved byte for byte. The
three bodies were identical except for the key count, which is `i + 1`, `seq_len` or
`encoder_len`. Each method is now a row loop that calls the home. `tile_size.max(1)`
moved into the home, and `tile_size_zero_is_clamped_to_one_like_the_callers_did` pins
that.

**Proof of equivalence.** `attend_row_online_tiled_equivalence_tests` keeps a frozen
copy of the old body and compares it with the home by `to_bits()` over 480 cases:
- head_dim in {1, 8, 64, 128}
- n_keys in {0, 1, 3, 17, 64, 130}
- tile in {1, 4, 16, 64, 256}
- four magnitudes

**Dead files removed (4 rows):**
- `gguf/inference/cache.rs`: three rows. It was an older copy of `attention_gqa.rs`,
  and the only file that referenced it was the next one.
- `gguf/inference/flash_attention_tiled.rs`: one row. It was the same as
  `flash_attention_dispatch.rs` apart from its include line.
- `apr_transformer/compute_attention.rs`: its row was already a byte-identical copy of
  `cache_attention.rs`. It was not included anywhere.

The build proves the deletion: `cargo clippy` with and without `cuda` still compiles.

**Rows: 36 → 28, all 8 deleted.** Also removed the `complexity_baseline.txt` row for
the deleted `flash_attention_tiled.rs`.

**Step 5b, next: family A.** It needs `ops::attend_row_scalar` with two enum parameters:
- the score scale: `* scale` or `/ sqrt(hd)`, for qwen35 `forward_attention`
- `SoftmaxNorm`

Key access goes through closures, so the four layouts (packed, `&[&[f32]]`, cache plus
current, head-major) share one body. The dot product must stay a plain `0.0f32` loop:
`zip().map().sum()` starts from `-0.0` on some toolchains, which changes the sign of an
all-zero dot. Each migration needs its own frozen-copy test.

### 9.11 Phase 2 step 5b delivered: scalar one-row attention home, Divide group (2026-09-26)

**Home:** `gguf/ops.rs::attend_row_scalar(q, n_keys, key, value, scale, RowSoftmax, scores, out)`.
- `score_j = dot(q, key(j)) * scale`, each dot a plain `0.0f32` loop.
- Then `softmax_exp_in_place` + `softmax_normalize`.
- Then `out[d] += w_j * value(j)[d]` in key order. `out` must be zeroed by the caller.
- `RowSoftmax { norm, guard_positive_sum }`. The guard mirrors the one guarded site.
  It cannot change the output: the max term contributes `exp(0) = 1`, so the sum is
  `>= 1` or NaN.

**Sites migrated (6 rows, all unguarded `Divide` except pmat-260):**
- `apr_transformer/pmat-260.rs::compute_causal_gqa_attention` (guarded)
- `gpu/adapters/apr_q4k.rs::gqa_attention`
- `gpu/scheduler/kv_forward_block.rs::gqa_attention_with_kv`
- `gpu/scheduler/kv_forward_block.rs::gqa_incremental_attention`
- `apr_transformer/attention_kernels.rs::causal_attention_cached`: its three helpers
  collapsed into one `attend_row`, which also serves `causal_attention`
- `gpu/scheduler/attention.rs::simplified_attention`: its dim-outer V loop has the same
  per-element order

**Bit-exact:** `attend_row_scalar_equivalence_tests` checks 300 cases with `to_bits`
against frozen copies of the three body forms.
- The forms: plain loop, the apr_q4k variant, and the iterator dot/sum.
- The grid: hd in {1, 8, 64, 128}, n in {1, 2, 7, 64, 300}, and magnitudes
  {0, 1e-3, 1, 8, 60}.
- Magnitude 0 is the all-zero dot, where the iterator sum can give `-0.0`. That case
  is proven equal, not assumed.

**Kept:** `apr/helpers.rs::simple_attention`. Its `get().unwrap_or(0.0)` reads turn
out-of-bounds into zeros rather than a panic, so moving it would change behaviour.

**Rows: 28 → 22.**

**Next: step 5c, the MulInv family.**
- rope.rs `causal_attention`
- forward/batched.rs `compute_attention_output`
- qwen35 `forward_attention`, which divides by `sqrt(hd)` and so needs a `ScoreScale` enum
