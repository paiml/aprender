# Specification: Ship Two Models — Sovereign AI Stack Proof

**Document ID:** SPEC-SHIP-TWO-001
**Version:** 2.29.4

**Current status** (machine-parseable; source of truth for CI gates and
`pmat work audit-ship-two`):

```yaml
status:
  model_1_teacher:
    state: RELEASED
    date: 2026-04-18
    artifact: paiml/qwen2.5-coder-7b-apache-q4k-v1
    formats: [apr, gguf, safetensors]
    tag: SHIP-TWO-001-MODEL-1-TEACHER
  model_1_distilled:
    state: DEFERRED
    tracker: task #86 retry plan (docs/specifications/aprender-train/model-1-qlora-retry-plan.md)
    blocking_decision: PMAT-684
  model_1_ship1_gates:
    on_main:
      count: 6
      of: 10
      ids: [SHIP-002, SHIP-005, SHIP-006, SHIP-007, SHIP-008, SHIP-010]
    pending_open_pr:
      - { id: SHIP-009, pr: 1009, branch: feat/falsify-ship-009-partial-discharge, commit: 90598277a }
    pending_stacked_branch:
      branch: feat/falsify-ship-001-partial-discharge
      ahead_of_main: 3
      ids: [SHIP-001, SHIP-003, SHIP-004]
  model_2_sovereign:
    state: BLOCKED
    blocker: task #132 Phase 3 evidence (lambda-labs RTX 4090 dispatch)
    phase_complete: [0, 1, 2]
    phase_open: [3, 4]
  model_2_ship2_gates:
    total: 13  # 12 original + AC-SHIP2-013 added v2.29.4
    on_main_discharged:
      count: 3
      of: 13
      ids: [AC-SHIP2-001, AC-SHIP2-011, AC-SHIP2-012]
    on_main_partial:
      count: 3
      of: 13
      ids: [AC-SHIP2-002, AC-SHIP2-005, AC-SHIP2-009]
    pending_open_pr:
      - { id: SHIP-016, pr: 1008 }
      - { id: SHIP-017, pr: 1004 }
      - { id: SHIP-018, pr: 1006 }
      - { id: SHIP-020, pr: 1005 }
    blocked_on_task_132_phase_3:
      ids: [AC-SHIP2-003, AC-SHIP2-004]
    blocked_on_trueno_203:
      ids: [AC-SHIP2-013]  # gx10 backend parity + pretrain residency (PMAT-696)
  spec_document:
    version_on_main: 2.29.1
    pending_amendment: null  # PR #1024 was for v2.30.0; superseded by this branch's v2.29.1/.2 corrections
    last_audit: 2026-04-23 (docs/specifications/aprender-train/ship-two-models-spec-audit.md @ 601c0740f)
  parallel_training_lab:
    name: albor
    path: ~/src/albor (separate repo, not a monorepo crate)
    state: v28 STOPPED @ step 11K (2026-04-05); v29 SUPERSEDED by v30 per PMAT-685 Option B decision
    relationship: uses the monorepo `apr` binary via `bin/apr-train`; owns corpus + HPO configs (NOT arch config)
    config_divergence_vs_monorepo: decided — albor aligns to monorepo (Option B, 2026-04-23)
    standing_policy: monorepo is single source of truth; all downstream repos MUST sync
    decision: PMAT-685 CLOSED (Option B chosen); execution tracked as PMAT-687..694
    v30_implementation_chain:
      - PMAT-688 retrain tokenizer at vocab 50257 (step 1 — serial)
      - PMAT-689 re-pretokenize corpus (step 2 — serial)
      - PMAT-690 replace v29 config with monorepo-compliant v30 config (step 3 — serial)
      - PMAT-687 dispatch v30 on lambda-labs (step 4, doubles as task #132 Phase 3)
      - PMAT-691 albor README/CLAUDE.md text updates (parallel post-687)
      - PMAT-692 cross-repo contract parity audit (parallel post-687)
      - PMAT-693 `cargo xtask audit-ship-two --include-albor` enforcement CI gate (parallel post-687)
      - PMAT-694 aprender-train/CLAUDE.md "spec phase" staleness fix (parallel post-687)
```

**Author:** PAIML Engineering
**Reviewer:** Noah Gift
**Date:** 2026-04-17 (initial v1.0.0); see Appendix A for full amendment timeline through v2.29.1.

**v2.29.1 correction amendment (2026-04-23):** Spec-vs-main audit
(`docs/specifications/aprender-train/ship-two-models-spec-audit.md`)
found that the running "MODEL-1 coverage N/10" counter in amendments
v2.24.0 through v2.29.0 silently assumed AC-SHIP1-009 (the apr-provenance
`license` / `data_source` / `data_license` gate, bound via
`GATE-APR-PROV-004` in `contracts/apr-provenance-v1.yaml` v1.1.0) was
live on main as its 1/10 baseline. **On main (`601c0740f`) that baseline
is not live.** `contracts/apr-provenance-v1.yaml` is still at v1.0.0
with 3 gates (GATE-APR-PROV-001/002/003 discharging AC-SHIP2-012 only);
the SHIP-009 multi-bind addition lives only in open PR #1009
(`feat/falsify-ship-009-partial-discharge`, commit `90598277a`, not a
main ancestor).

Corrected counts as of main `601c0740f` 2026-04-23:

| Count                           | Prior spec claim (v2.29.0)       | On-main reality (`601c0740f`)   | Delta                 |
|---------------------------------|----------------------------------|---------------------------------|-----------------------|
| MODEL-1 AC-SHIP1-* PARTIAL      | 7/10                             | **6/10** (SHIP-002/005/006/007/008/010) | −1 (SHIP-009 pending #1009) |
| MODEL-2 AC-SHIP2-* touched      | "10/12 touched" (v2.23.x running) | **6/12** (011 D, 012 D, 021 D + 002 P, 005 P, 009 P) | −4 (SHIP-016/017/018/020 pending #1008/1004/1006/1005) |

Root cause (Five Whys): the per-amendment "MODEL-1 coverage N/10 → (N+1)/10"
increment pattern was computed against the *expected* merge order, not the
*actual* ancestry of main. Any amendment written after a PR was merged to
`main` was correct; any amendment written speculatively ahead of its PR
landing — or any amendment whose upstream PR silently stalled — drifted.
**Rule:** spec counts are main-ancestry assertions; treat every `N/M on
main` claim as a check against `git merge-base --is-ancestor <evidence
commit> main`. PMAT-683 (cargo xtask audit-ship-two) closes the loop
structurally.

Prior amendment text (v2.24.0..v2.29.0) is retained unchanged below this
correction to preserve audit history. Amendments after this correction
MUST cite both the on-main count and the across-branch count when they
differ. MEMORY.md session-wrap (2026-04-23) line reading "MODEL-1 7/10"
is superseded by the 6/10 figure above.

---

**v2.21.0 amendment (2026-04-19):** Three MODEL-2 architecture + tokenizer
gates landed in the same post-v2.19 evidence window, on branch
`chore/post-v2.19-evidence`:

1. **FALSIFY-SHIP-011 (AC-SHIP2-001) — DISCHARGED** at commit `338c6eb3c`
   (task #114). `contracts/model-families/llama-370m-sovereign-v1.yaml`
   promoted v1.0.0 PROPOSED → v1.1.0 ACTIVE. Rust scaffold
   `Llama370MConfig` (crates/aprender-train/src/models/llama_370m.rs) now
   binds **byte-equally** to the YAML contract via the harness test
   `falsify_ship_011_rust_scaffold_matches_yaml_contract`, which uses
   `include_str!` to embed the contract at compile time and
   `serde_yaml::Value` to parse-and-compare every architecture.* and
   constraints.* field against the corresponding `Llama370MConfig::*`
   const. Any edit to either side that diverges fails
   `cargo test -p aprender-train --lib llama_370m` before a single step
   of compute runs. INV-ARCH-370M-002..008 remain enforced at compile
   time via `const _: () = Llama370MConfig::validate();`, so the
   compile-time tier is intact even without the new YAML-binding test.
   The deliberate *sibling* approach over amending `llama.yaml` with a
   `370m` entry is recorded in the discharge memo: albor's
   `tied_embeddings=true` and `rope_theta=10000.0` conflict with
   Meta Llama-3's family-wide `tied_embeddings=false` /
   `rope_theta=500000.0`, and GATE-ARCH-370M-001's
   "llama.yaml (or this sibling contract)" language explicitly permits
   it.

2. **FALSIFY-SHIP-012 (AC-SHIP2-002) — PARTIAL_ALGORITHM_LEVEL** at
   commit `2e8b8b8e2` (task #115). `contracts/tokenizer-bpe-v1.yaml`
   bumped v1.0.0 → v1.1.0, **status intentionally stays PROPOSED**.
   GATE-BPE-003 gains `evidence_discharged_by` pointing at 3 harness
   tests in
   `crates/apr-cli/tests/falsify_ship_012_tokenizer_roundtrip.rs`:
   byte-exact round-trip on a 20-doc Python-like holdout (ASCII
   keywords + Unicode identifiers + docstrings + emoji + combining
   marks) under `aprender::text::tokenize::BpeTokenizer`, standalone
   NFC idempotence (INV-BPE-005), and train/holdout disjointness. The
   gate's `evidence_required` explicitly asks for **10K** docs; the
   current harness runs 20 on a synthetic fixture, so the gate lands
   with `discharge_status: PARTIAL_ALGORITHM_LEVEL` and
   `full_discharge_blocks_on: "task #91 (10K The Stack v2 Python
   holdout)"`. The harness module doc-comment locks in the zero-rewrite
   swap path: when task #91's 10K corpus materializes, replacing
   `HOLDOUT_CORPUS` + `TRAIN_CORPUS` with shard readers is a data-only
   change, then the contract can bump to 2.0.0 and promote to ACTIVE.
   This is the first spec-level use of a PARTIAL gate inside a
   PROPOSED contract — the pattern is: if the algorithm is provable
   today but the production-scale evidence is deferred, wire the
   algorithm proof and surface the data gap as first-class contract
   state rather than leaving the `evidence_discharged_by` slot blank.

3. **FALSIFY-SHIP-015 (AC-SHIP2-005) — PARTIAL_ALGORITHM_LEVEL** at
   commit `bfb883199` (task #116). Sovereign contract v1.1.0 → v1.2.0,
   stays ACTIVE. GATE-ARCH-370M-003 gains `evidence_discharged_by`
   pointing at the pre-existing `estimated_param_count_within_contract_band`
   unit test plus the `estimated_param_count` /
   `estimated_stored_param_count` const fns in
   `crates/aprender-train/src/models/llama_370m.rs`. The gate's
   `evidence_required` asks for `apr inspect --json model.apr |
   jq '.param_count'` to yield an integer in [366_000_000,
   374_000_000]; no on-disk `.apr` exists pre-compute, so the gate
   lands with `discharge_status: PARTIAL_ALGORITHM_LEVEL` and
   `full_discharge_blocks_on: "real 370M .apr checkpoint from
   pretraining compute-dispatch (AC-SHIP2-003/004)"`. The unit test
   hard-asserts p ∈ [366M, 374M], |p − 370M|/370M < 5%, and that
   embedding tying reduces stored params by exactly
   VOCAB_SIZE × HIDDEN_DIM; any edit to `Llama370MConfig` that moves
   the count out of the INV-ARCH-370M-001 band fails
   `cargo test -p aprender-train --lib llama_370m` before any compute
   runs. Contract remains ACTIVE because SHIP-011 (not SHIP-015) is
   what gates the sovereign contract's ACTIVE promotion — a gate-level
   PARTIAL nested inside an ACTIVE contract is a valid shape.

**Pattern codified by v2.21.0 (PARTIAL_ALGORITHM_LEVEL):** when a gate's
`evidence_required` text describes a production-scale check (10K docs,
on-disk artifact, benchmark run) that is not yet runnable, but the
underlying invariant is provable at algorithm / compile / unit-test
level today, emit the gate with `evidence_discharged_by` listing the
algorithm proofs + `discharge_status: PARTIAL_ALGORITHM_LEVEL` +
`partial_discharge_note:` + `full_discharge_blocks_on:` +
`ship_blocking: true`. The last field is load-bearing: PARTIAL gates
MUST still block `apr publish` until full discharge lands. Downstream
auditors must treat `evidence_discharged_by` alone (without checking
`discharge_status`) as **not** sufficient green — the two fields
together are the authoritative read.

**v2.22.0 amendment (2026-04-19):** One additional MODEL-2 ship gate
attained PARTIAL_ALGORITHM_LEVEL in the same post-v2.19 evidence window,
on branch `chore/post-v2.19-evidence`:

4. **FALSIFY-SHIP-019 (AC-SHIP2-009) — PARTIAL_ALGORITHM_LEVEL** at
   commit `846cc1dbb` (task #117). Sovereign contract v1.2.0 → v1.3.0,
   stays ACTIVE. GATE-ARCH-370M-004 gains `evidence_discharged_by`
   pointing at two new harness tests + an enumerator helper in
   `crates/aprender-train/src/models/llama_370m.rs` plus three
   cross-referenced assets (`LayoutContract`, `validate_apr_shape`,
   `contracts/tensor-layout-v1.yaml`). The gate's `evidence_required`
   asks for GGUF-exported 370M first-token cosine similarity ≤ 1e-3 vs
   APR on 100 canary prompts — that runner is blocked on
   AC-SHIP2-003/004 pretraining compute plus GATE-SHIP-006 harness
   invocation, so the gate lands with `discharge_status:
   PARTIAL_ALGORITHM_LEVEL` + `full_discharge_blocks_on: "real 370M .apr
   checkpoint from pretraining compute-dispatch (AC-SHIP2-003/004) +
   harness invocation of GATE-SHIP-006 cosine-parity runner"`. The
   algorithm-level proofs collectively establish the conditional: *if*
   GGUF export invokes `LayoutContract::validate_apr_shape` on every
   tensor, *then* row-major layout and GH-202 regression rejection are
   mathematically enforced. The enumerator counts
   **3 + 9 × NUM_LAYERS = 219** tensors and cross-checks each with
   `LayoutContract::get_apr_contract`; adding a tensor to
   `Llama370MConfig` without a matching entry in
   `layout_contract_specs.rs` now fails
   `cargo test -p aprender-train --lib llama_370m` before any compute
   runs. Spec §9 Risk #2's explicit instruction to "reuse
   `layout_contract.rs` validator" was the load-bearing hint that
   pointed at a non-compute, algorithm-level asset.

**Pattern lesson codified by v2.22.0 (counter-example hunting):** the
v2.21.0 cycle declared all non-compute PARTIAL levers for MODEL-2
"exhausted". Re-running the 7-gate FALSIFY-SHIP survey (013/014/016/017/
018/019/020) with explicit counter-example hunting found exactly one
genuine lever (SHIP-019); SHIP-017/018/020 truly need compute,
SHIP-013/014/016 collapse into SHIP-011's wiring. Prior verdict was ~86%
correct. **Rule: before declaring a search space exhausted, re-run the
survey with explicit counter-example hunting — the spec's own Risk
mitigations are the highest-leverage hint source.**

Combined MODEL-2 ship-gate status after v2.22.0: **3/12 AC-SHIP2 gates
fully ACTIVE** (001, 011, 012) + **3/12 PARTIAL_ALGORITHM_LEVEL** (002
via SHIP-012, 005 via SHIP-015, 009 via SHIP-019) = **6/12 touched**
(50%). The remaining 6 (003/004/006/007/008/010) all require either
real 370M training compute, a trained on-disk `.apr` with evaluation
harness, or a wall-clock benchmark on RTX 4090, and will remain
untouched until compute-dispatch lands — the pretrain loop driver + CLI
from v2.19.0 are ready for them. Genuine algorithm-level PARTIAL
harvesting is now exhausted for MODEL-2.

**v2.20.0 amendment (2026-04-19):** Two MODEL-2 ship gates **DISCHARGED**
in the post-v2.19 evidence window on branch `chore/post-v2.19-evidence`:

1. **FALSIFY-SHIP-021 (AC-SHIP2-011) — DISCHARGED** at commit `0b8ca8c84`
   (task #112). `falsify_ship_021_seed_0_100_step_reproducibility` proves
   two seed=0 × 100-step training runs produce |Δloss| ≤ 1e-6 at every
   step and bit-identical AdamW-state sha256; a counter-test
   `falsify_ship_021_different_seeds_do_diverge` proves seed=0 vs seed=1
   diverge > 1e-4 within 10 steps. Root cause of the original green-run
   flake (step-0 6.854 vs 6.928 under parallel cargo test) was a sibling
   test racing on the global `INIT_SEED` atomic; fix landed as
   `transformer::init::lock_init_seed(seed) -> MutexGuard` which any
   future caller doing concurrent weight init under a set-before-read
   global MUST hold across the full init work. Contract
   `training-loop-pretrain-v1.yaml` bumped 1.0.0 → 1.1.0,
   status PROPOSED → ACTIVE, INV-TRAIN-006 + GATE-TRAIN-006 got
   harness/evidence_discharged_by blocks.

2. **FALSIFY-SHIP-022 (AC-SHIP2-012) — DISCHARGED** at commit `8f0607d42`
   (task #113). `apr inspect` now surfaces the three provenance keys —
   `license`, `data_source`, `data_license` — from every .apr binary,
   rendering absent values as the literal `(missing)` in text mode and
   `null` in JSON mode. Key design: `AprV2Metadata` gained
   `data_source` + `data_license` as NAMED Option<String> fields (not
   buried in `custom: HashMap`); no `skip_serializing_if` is allowed on
   any provenance field on either `AprV2Metadata` or `MetadataInfo`,
   because silent-skip via serde is the exact failure mode
   (`FM-APR-PROV-SILENT-SKIP`) the contract guards against. Text
   rendering goes through a pure helper `format_provenance_block()` so
   tests assert on a returned `String` rather than capturing stdout
   (`gag::BufferRedirect` is NOT parallel-test-safe — recorded as a
   reusable pattern). New schema contract `apr-provenance-v1.yaml`
   (C-APR-PROVENANCE v1.0.0 ACTIVE, `kind: schema`) declares 3
   invariants (round-trip, always-emit, publish-gate-rejects), 3 gates,
   and 3 failure modes, all bound to AC-SHIP2-012. `pv validate` PASS
   (0 errors). Live smoke test on
   `qwen2.5-coder-1.5b-instruct-q4k.apr` (no provenance stored)
   correctly prints the Provenance block with `(missing)` on all three
   rows. Together with PM-003/PM-008/PM-009/PM-007 pre-flight gates,
   any operator can now answer "what data trained this, under what
   license?" from a `.apr` alone — the sidecar-manifest dependency is
   severed.

Combined MODEL-2 ship-gate status after v2.20.0: 2/12 AC-SHIP2 gates
DISCHARGED (011, 012). The remaining 10 (001–010) all block on the
actual 370M checkpoint, which is the compute-dispatch long-pole; the
pretrain loop driver from v2.19.0 is ready to exercise them once
compute-dispatch for real weights lands.

**v2.17.0 amendment (2026-04-18):** Task #101 contracts schema
harmonization **SHIPPED** on `feat/pm-007-preflight-poka-yoke` at commit
`4fc453d57`. Closes the last parser barrier preventing `pv validate`
from serving as the canonical dogfooded gate across all SHIP-TWO-001
contract work. `crates/aprender-contracts/src/schema/types.rs` now
(a) accepts legacy ProofObligation field spellings (`statement`/
`verification`) via `#[serde(alias)]`, (b) accepts both map
`{id: Equation}` and list `[{id, ...}]` equation forms via a custom
polymorphic `deserialize_equations`, (c) adds `Safety` + `Liveness` to
`ObligationType` (28 variants now, up from 26), (d) uplifts 6 legacy
contracts (decode-gpu-resident-sampling, decode-hot-path-*,
eval-harness-humaneval, eval-sharding, profile-graph-vs-per-op-
methodology, publish-manifest) to the metadata-block form. Target
tests `load_contracts_real` + `parse_missing_metadata_returns_error`
both green; 1368/1371 aprender-contracts lib tests pass. Remaining
3 failures (lint_passes_on_real_contracts, validate_gate_passes,
lint_findings_on_failure) are downstream content checks — empty
`formula:` bodies, missing `kani_harnesses`, falsifications <
proof_obligations on the same 6 legacy contracts — dispatched as
task #102 follow-up content-authoring lane. This amendment ties the
"pv not bash for contracts" MEMORY.md policy (2026-04-18) to
concrete unblocked state: no more adhoc bash/grep workarounds when
the dogfood tool covers the workflow.

**v2.18.0 amendment (2026-04-18):** Parallel dispatch lanes #102/#103/#104
**ALL CLOSED** in a single concurrent compute window against
non-overlapping surfaces, demonstrating the monorepo's sub-agent
workflow scales. Results:
(a) **#102 contract backfill CLOSED** — 8 legacy contracts
(`decode-gpu-resident-sampling`, `decode-hot-path-{first-tokens,
prefix-cache,zero-syscalls}`, `eval-harness-humaneval`, `eval-sharding`,
`profile-graph-vs-per-op-methodology`, `publish-manifest`) received
metadata references, formula bodies, kani_harnesses, and falsification
parity. 22 ERROR findings → 0, `lint_passes_on_real_contracts` green.
Verified live via `pv validate` dogfood: 8/8 contracts parse clean
(1 advisory SCHEMA-013 qa_gate-missing on eval-sharding kept as
forward work). No bash/grep workaround needed.
(b) **#103 MODEL-2 `--min-frequency` plumbing CLOSED** — `apr-cli`
tokenize call-site swapped from `aprender::text::tokenize::BpeTokenizer`
→ `entrenar::tokenizer::BPETokenizer::train` via `train_bpe_via_entrenar`
helper; `TokenizerConfig::bpe().with_min_frequency(..).with_normalization(..)`
now threads user-provided `--min-frequency` + `--normalization` into
merge pruning. Public read-only `vocab()`/`merges()` accessors added to
`aprender-train::tokenizer::BPETokenizer`. 17 `apr-cli` tokenize tests
pass including new `run_train_honors_min_frequency_pruning` which
asserts singleton byte-pairs ("xyz" single occurrence) are pruned
from `merges.txt`/`vocab.json` at threshold 2. Closes v2.15.0 §1
"Known gap". Redundant `build_normalizer()` call-site removed since
`aprender-train`'s BPE applies NFC internally (no double-normalization).
(c) **#104 gx10 capacity gate PASS** — llama.cpp (b1-b0f0dd3 CUDA)
on teacher GGUF (sha256 `e6cac5d6…7981`) measured **38.0 tok/s decode**
(prompt eval 509.0 tok/s, 7.7 GiB VRAM, 5 s wall) vs 30 tok/s gate
threshold = PASS 26.7% margin. 2.45× the forbidden 15.5 tok/s fused
NF4 steady-state fallback — Zero-Tolerance §3 row #8 "no perf
regression" clause preserved. Two follow-ups flagged: (1) decode drift
from memory's 46 tok/s → 38.0 tok/s on current build; (2) gx10 disk
95% full (44 GB free) needs cleanup before MODEL-2 7B training lands.
Evidence: `evidence/ship-two-001/gx10-capacity-baseline-20260418-213928.json`.

With #102+#103 closed, **task #105** (370M MODEL-2 pretraining loop
wiring per `training-loop-pretrain-v1` GATE-TRAIN-005) is now the sole
long-pole item. Expected surface: `aprender-train/src/train/pretrain.rs`
loop driver calling the `llama_370m` forward pass with AdamW optimizer
and gradient accumulation, gated by the dataset ingest binary shipped
in v2.15.0.

**v2.19.0 amendment (2026-04-18):** Task #105 **CLOSED** via background
sub-agent `ac479445bcd722bf7` — commit `9a5af3ac2` on
`feat/pm-007-preflight-poka-yoke`. Surface landed (6 files, +1379 LOC):
(a) `crates/aprender-train/src/train/pretrain.rs` (963 LOC) — PretrainConfig
with `model_2_defaults()` baking LR=5e-5 + rank=32 + seed=42 remedies
from MODEL-1 v2 QLoRA divergence post-mortem;
(b) `crates/apr-cli/src/commands/pretrain.rs` (332 LOC) — CLI entrypoint
gated behind `training` cargo feature;
(c) extended_commands.rs + dispatch_analysis.rs — wired `apr pretrain`
into the apr-cli dispatch table.
Contract compliance verified: `contracts/training-loop-pretrain-v1.yaml`
passes `pv validate` with 0 errors; GATE-TRAIN-005 (val_loss[N] ≤ 2.0×
val_loss[N-1]) wired in `check_non_divergence`; INV-TRAIN-007 NaN/Inf
guard wired in `check_numerical_stability` before metric logging;
GATE-TRAIN-008 throughput bounds wired via `PretrainAbort::ThroughputOutOfRange`.
`per_step_metrics.required` and `per_epoch_artifacts.required_fields`
enforced as struct invariants. Checkpoint path template
`{run_dir}/ckpt/epoch-{N:03d}.apr` frozen in `EpochArtifact::new`.
Synthetic drive via injected `StepFn`/`ValFn` traits allows exercising
the full gate surface today while the real 370M forward pass wiring
(llama_370m.rs) completes. Test verification: 15/15 pretrain unit tests
pass, 3/3 CLI tests pass, 947/947 aprender-train lib tests no
regressions. Abort errors map 1:1 to contract gate IDs so operators
see the tripped gate via shell `$?`.

Concurrent with #105, **task #108** closed the 32-way workspace-test
regression discovered by CI run 24614757928. Root cause: five directory
iterators in `aprender-core/src/format/` were treating
`contracts/model-families/llama-370m-sovereign-v1.yaml` (a
ModelFamilyVariant CONTRACT starting with `contract_id:`) as a
ModelFamily REGISTRY entry. Fix (commit `21d43bd7a`): all iterators
now skip files whose first top-level key is `contract_id:` (family
registry YAMLs all begin with `metadata:` — a clean discriminator,
verified by corpus scan). `cargo test -p aprender-core --lib format::`
re-green at 13031 passed / 0 failed.

The ci/lint workspace package-ambiguity blocker (`aprender@0.27.8` vs
`aprender@0.31.0` — caused by transitive deps on published
`realizar ^0.7/^0.8`, `renacer ^0.9/^0.10`, `trueno ^0.15/^0.16/^0.17`,
`entrenar ^0.7`, `bashrs ^6.35/^6.65`, `pacha ^0.2` that all re-export
old `aprender@0.27`) was split into task #109 for a separate
path-dependency migration pass. This is orthogonal to SHIP-TWO-001 and
pre-dated the branch; the monorepo's `[patch.crates-io]` block was
removed during RC4 cc-cleanup and was never restored for the
aprender/realizar/renacer/trueno/trueno-gpu chain documented in
CLAUDE.md.

**Parallel dispatch state (2026-04-18 post-v2.17.0 — preserved for audit
trail):** three lanes ran concurrently against non-overlapping surfaces —
(a) task #102 contract backfill (content-authoring, contracts/*.yaml),
(b) task #103 MODEL-2 CLI `--min-frequency` plumbing (swap apr-cli
tokenize call-site from aprender-core BPE to aprender-train BPE;
closes v2.15.0 §1 "Known gap" — 0.5 day), (c) task #104 gx10
third-party framework capacity gate (llama.cpp on teacher GGUF,
enforces Zero-Tolerance §3 row #8). Tasks #105 (370M pretraining
loop wiring per training-loop-pretrain-v1 GATE-TRAIN-005) remains
the long-pole item awaiting #102+#103 closure. Compute pool utilization
is deliberately heterogeneous: lambda-labs (x86_64 RTX 4090) does
contract+code surgery, gx10 (aarch64 GB10 Blackwell) does remote
bench, yoga (x86_64 RTX 4060 Laptop) stays idle pending apr 0.31.0
upgrade per Zero-Tolerance §3 row #8. Jetson remains blocked per
`project_ship_two_001_jetson_blocked.md`.

**v2.16.0 amendment (2026-04-18):** Codified **Zero-Tolerance** as §3 row
#8. The operationalization, verbatim: "We never accept bugs or poor
performance. Defects and perf regressions are both blockers, not trade-
offs. All work improves or holds the line; never degrades it. No 'pre-
existing' carve-outs. No `#[ignore]` as a release valve." Why now: the
SHIP-TWO-001 compute-pool reality (lambda-labs x86_64 RTX 4090 +
yoga x86_64 RTX 4090 Laptop + gx10 aarch64 GB10 Blackwell + jetson
aarch64) surfaces cases where it is tempting to accept a regression
("gx10 is Blackwell — 15.5 tok/s fused NF4 is fine") or a bug ("yoga's
apr 0.4.11 is stale but it works for small models"). The Zero-Tolerance
principle writes the refusal explicitly: when a host drops to a slower
path OR runs stale software, that is a blocker on the host, not a
baseline to ship against. Concrete application to in-flight work:
(a) yoga stays blocked from SHIP-TWO-001 eval dispatch until apr
binary is upgraded to 0.31.0 AND cuBLAS smoke passes (no "it works
with the old binary" ship), (b) gx10 must run a non-fused third-party
framework (llama.cpp / PyTorch nightly cu128 / vllm) at ≥ reference
tok/s before counting as GPU capacity for MODEL-2 parity training —
the 15.5 tok/s fused fallback is FORBIDDEN as a steady-state (see
`project_pmat_587_*` memos for prior perf discipline). Ties to the
existing Toyota Way feedback memory: "all defects are your defects;
never 'pre-existing'" now extends to performance.

**v2.15.0 amendment (2026-04-18):** MODEL-2 pretraining scaffold **LANDED**
on `feat/pm-007-preflight-poka-yoke`. Three commits close the three
P0 blockers identified in the v2.14.0 readiness audit:

1. **Task #89 — BPE NFC patch SHIPPED (commit `b0e0a280b`):**
   `crates/aprender-train/src/tokenizer/{config,bpe}.rs` now enforce
   C-TOK-BPE-001 INV-TOK-003 (NFC before optional lowercase).
   `TokenizerConfig::normalization` defaults to `None` (`#[serde(default)]`
   for backward compat); set `Normalization::NFC` via
   `.with_normalization()` builder. Two falsification tests locked:
   (a) `test_bpe_nfc_composed_decomposed_parity` — composed `café`
   U+00E9 and decomposed `cafe\u{0301}` encode to identical token IDs
   under NFC; (b) `test_bpe_without_nfc_composed_decomposed_diverge` —
   live falsification witness: without NFC the two forms MUST diverge.
   If the witness test starts passing under `Normalization::None`, the
   invariant is no longer load-bearing and the contract should be
   revisited. `preprocess()` doc-comment records **why NFC before
   lowercase**: `char::to_lowercase()` is not closed over non-NFC input
   for every grapheme — normalizing first keeps the pipeline
   deterministic for composed/decomposed variants.

2. **Task #90 — `apr tokenize train` subcommand SHIPPED (commit
   `512ea51a6`):** new `TokenizeCommands::Train { corpus, vocab_size,
   min_frequency, output, normalization }` variant. Walks `.jsonl`
   files (file or directory), extracts `content` field per line,
   applies NFC via `unicode-normalization::UnicodeNormalization::nfc`
   when `--normalization nfc` (default), calls the BPE trainer, emits
   `vocab.json` + `merges.txt`. `--json` mode round-trips all
   parameters. 3 unit tests pass (happy-path JSONL, directory walk,
   unknown-normalization rejection). **Known gap** (follow-up, NOT a
   ship blocker): `--min-frequency` is accepted for contract parity
   but NOT threaded through — the CLI currently calls
   `aprender::text::tokenize::BpeTokenizer::train(corpus, vocab_size)`
   (aprender-core) which has no public `min_frequency` parameter.
   Strategic fix: switch the CLI to
   `aprender-train::tokenizer::BPETokenizer` (which both honors
   `with_min_frequency()` AND has the NFC plumbing task #89 added).
   Documented in memory `project_ship_two_001_nfc_bpe_patch.md`.

3. **Task #91 — `apr-corpus-ingest` binary SHIPPED (commit
   `512ea51a6`):** new `crates/apr-cli/src/bin/apr-corpus-ingest.rs`
   (+517 LOC) with `plan` and `validate-contract` subcommands over
   `C-DATA-THESTACK-PYTHON` v1.0.0. `plan` reads the contract,
   asserts the 6 required top-level keys (source, license_whitelist,
   pii_scrub, deduplication, split, budget), validates 7
   `INV-DATA-*` + 5 `FALSIFY-DATA-*` + 5 `GATE-DATA-*` prefixes, and
   emits `./output/dry-run-manifest.yaml` with TODO placeholders + UTC
   timestamp. `validate-contract` is exit-code-only. **Hard constraints
   honored:** NO network, NO writes outside `./output/`, deps limited
   to workspace `serde`/`serde_yaml`/`anyhow`/`clap`. Does NOT touch
   `aprender-train/` or `aprender-core/`. 2 unit tests pass.

**MODEL-2 training readiness estimate (post-v2.15.0):** 4 contracts +
3 scaffolding commits shipped. Remaining work to first pretraining
loss curve:
- Thread `--min-frequency` through CLI (switch call to aprender-train
  BPE) — 0.5 day, follow-up ticket.
- Actual corpus download + validated ingest into train/val split
  honoring the 6 C-DATA-THESTACK-PYTHON gates (MinHash-LSH dedup,
  PII scrub, license whitelist, deterministic hash-by-sha256 split,
  corpus_sha256 merkle gate yoga vs gx10) — 2-3 days.
- 370M Llama architecture implementation + pretraining loop wiring
  honoring `training-loop-pretrain-v1.yaml` GATE-TRAIN-005 (val_loss
  divergence abort) — 5-7 days.
- First pretraining smoke run on gx10 — 1-2 days.

**Total: ~10-14 days to first loss curve** (revised up from
v2.14.0's 5-7d estimate now that the scaffold is concrete and the
370M arch implementation is clearly the gating path). Post-v2.15.0,
MODEL-2 moves from contract+scaffold into execution.

**v2.14.0 amendment (2026-04-18):** MODEL-2 pretraining readiness audit
closed two gaps in contract + impl surface:

1. **Dataset contract drafted:** `contracts/dataset-thestack-python-v1.yaml`
   (C-DATA-THESTACK-PYTHON v1.0.0 PROPOSED). 7 invariants + 5
   falsification tests + 5 compound gates covering (a) upstream
   revision pin + raw_tar_sha256 reproducibility, (b) permissive-
   license whitelist (Apache/MIT/BSD/ISC/Unlicense/CC0/0BSD) with
   unknown→reject policy, (c) PII scrub (AWS/PEM/GH PAT/Slack/Google),
   (d) MinHash-LSH near-duplicate removal (seed=42, Jaccard ≥0.85 →
   drop), (e) deterministic hash-by-file-sha256 split (train=0.98,
   val=0.02, assertion: same seed → byte-identical split across
   hosts), (f) corpus_sha256 merkle-style parity gate (FALSIFY-DATA-003
   yoga vs gx10), (g) UTF-8 + NFC round-trip encoding hygiene
   (INV-DATA-007). Closes the P0 blocker identified by the 2026-04-18
   MODEL-2 training-readiness audit: `training-loop-pretrain-v1.yaml`
   line 22 referenced this peer contract, but the file did not exist.

2. **BPE NFC gap identified (IMPLEMENTATION BLOCKER):** The BPE
   tokenizer at `crates/aprender-train/src/tokenizer/bpe.rs` does NOT
   implement NFC normalization, despite `contracts/tokenizer-bpe-v1.yaml`
   INV-TOK-003 / `dataset-thestack-python-v1.yaml` INV-DATA-007
   requiring it. No HF `tokenizers` dep to defer to.
   `TokenizerConfig`/`BpeConfig` have no normalizer field. Fix surface:
   (a) add `normalization: Option<Normalization>` to BpeConfig, (b)
   apply `unicode_normalization::nfc()` at `encode()` entry, (c) add a
   round-trip property test on `café` (composed vs decomposed) + emoji.
   Without this, MODEL-2 tokenizer will drift between train-time and
   inference-time on non-ASCII code and GATE-DATA-005 will ship-block.

**MODEL-2 training readiness estimate (post-v2.14.0):** contract surface
is complete (4 contracts: llama arch, BPE tokenizer, pretrain loop,
dataset). Remaining code work: BPE NFC patch (~1 day), tokenizer
trainer CLI wiring (~3 days), corpus ingest harness honoring the
dataset contract (~2 days). **5-7 days to first pretraining run**
modulo Blackwell JIT warm-up and corpus download time.

**v2.13.0 amendment (2026-04-18):** FALSIFY-SHARD-003 DISCHARGED. Live
probe run yoga (RTX 4090, x86_64) vs gx10 (GB10 aarch64) on the released
teacher GGUF (`paiml/qwen2.5-coder-7b-apache-q4k-v1`, sha
`e6cac5d6…7981`) returned **16/16 byte-identical completions** on
HumanEval/0..15 at temperature=0.0, top-k=1, max_tokens=512. Evidence:
`evidence/ship-two-001/shard-003-determinism/probe_20260418_143041.json`.
Contract `contracts/eval-sharding-v1.yaml` bumped 1.0.0 → 1.1.0 and
flipped **DRAFT → ACTIVE**; `discharged:` block recorded on
FALSIFY-SHARD-003 mirroring the SHARD-004 pattern. Combined with the
prior SHARD-004 discharge (Δ=0.0039 pp merged-score identity), both
correctness gates for AC-EX-007 are green. The parallel eval-shard lane
(yoga+gx10) is now a legitimate accelerator for any future SHIP-TWO-001
re-audit that respects the contract prerequisites (temp=0.0, top-k=1).
Task #79 closed.


**v2.12.0 amendment (2026-04-18):** Post-ship artifacts landed (commit
`cc52e7bfc`) while the teacher is live on HF. All of these are
**out-of-scope for the current ship** but advance the next-wave deliverables:

1. **MODEL-2 Phase 1-B contracts** (task #81) — three new YAMLs:
   - `contracts/model-families/llama-370m-sovereign-v1.yaml` (9 invariants,
     4 gates, sovereign 370M arch with frozen intermediate_dim=2816)
   - `contracts/tokenizer-bpe-v1.yaml` (7 inv, 7 gates; vocab bounds,
     special tokens, byte-exact round-trip, NFC normalization)
   - `contracts/training-loop-pretrain-v1.yaml` (8 inv, 8 gates;
     GATE-TRAIN-005 ship-blocking: `val_loss[N] > 2.0 × val_loss[N-1]`
     → ABORT — encodes the MODEL-1 v2 divergence lesson)
2. **MODEL-1 QLoRA retry plan** (task #86) —
   `docs/specifications/aprender-train/model-1-qlora-retry-plan.md`,
   6 falsification gates, hyperparameter deltas from v2 (LR 2e-4 →
   5e-5, rank 16 → 32, temperature 4.0 → 2.0).
3. **FALSIFY-SHARD-003 determinism probe** (task #88) —
   `scripts/ship-two-001/eval-shard-determinism-probe.sh` (239 lines).
   Closes the one blocking gap for AC-EX-007 found by the eval-shard
   audit (contract `eval-sharding-v1.yaml` line 151 referenced a script
   that did not exist). DRY_RUN=1 validates the JSONL builder without
   dispatch. Full `--hosts yoga,gx10 --model <gguf> --probe-tasks 0-15`
   run requires teacher GGUF pre-cached on both hosts.

**Compute-pool reality check (2026-04-18):** yoga RTX 4090 + gx10 GB10
aarch64 are today's effective parallel pool. Jetson remains blocked by
the 5 blockers documented in memory `project_ship_two_001_jetson_blocked.md`.
Lambda-labs is referenced in spec docs but **not provisioned** — no SSH
alias, no memory file, no credentials surfaced; treat as aspirational
until provisioning is in place.

**v2.11.0 amendment (2026-04-18):** SHIP-TWO-001-MODEL-1-TEACHER **RELEASED**.
EX-05, EX-06, EX-07 all DISCHARGED on the teacher artifact (`paiml/qwen2.5-coder-7b-apache-q4k-v1`):

1. **EX-05 verify-manifest (live, 3 formats)**:
   `apr validate-manifest <m> --live --json` PASS for `.apr` (8.0 GiB, sha
   `0a854098…c73666`), `.safetensors` (15.2 GiB, sha `c1058ce7…d8954`),
   `.gguf` (7.5 GiB, sha `e6cac5d6…7981`). All five gates fire green:
   PM-001 (schema), PM-003 (HEAD content-length), PM-002-live
   (streaming sha256), PM-004 (SPDX), PM-005 (recipe_sha256), PM-006
   (parent chain). Evidence:
   `evidence/ship-two-001/ex-05-manifest-verify-*.json` (3 per-format +
   1 summary).

2. **EX-06 apr pull + re-inference**: `apr pull
   paiml/qwen2.5-coder-7b-apache-q4k-v1` → cached GGUF at
   `~/.cache/pacha/models/7bcabb852fedb36b.gguf`; sha256 of pulled file
   exactly matches the declared GGUF manifest sha (harness v3 auto-
   detects pulled format from file extension, fixes v2 bug that hard-
   coded the APR manifest and produced a spurious format-mismatch FAIL);
   `apr run <pulled> --prompt 'def fib(n):' --max-tokens 64 --temp 0 --top-k 1`
   produces output whose longest parseable prefix contains ≥1 non-trivial
   Python statement (spec §12.3 AC-EX-006 literal: "emits syntactically
   valid Python"). Both **AC-EX-005 (sha256 roundtrip)** and **AC-EX-006
   (Python validity)** PASS. Evidence:
   `evidence/ship-two-001/ex-06-pull-rerun.json` → `overall: PASS`.

3. **EX-07 tag release**: Git tag `SHIP-TWO-001-MODEL-1-TEACHER` created
   at HEAD of the ship branch; announcement blurb embedded in this
   amendment. The teacher artifact is live on HF Hub at
   https://huggingface.co/paiml/qwen2.5-coder-7b-apache-q4k-v1 (3 formats),
   and downloadable via `apr pull paiml/qwen2.5-coder-7b-apache-q4k-v1`.

**Announcement (v2.11.0):** Aprender ships its first sovereign model:
Qwen2.5-Coder-7B-Instruct Q4_K (Apache-2.0), 85.98% HumanEval pass@1
(141/164, confirmed via `apr eval --benchmark humaneval` on 2026-03-28),
8.0 GiB APR / 7.5 GiB GGUF / 15.2 GiB SafeTensors. Runs end-to-end on
`apr run` / `apr serve`. MODEL-1 v2 (distilled student) is falsified
at the adapter (non-converged QLoRA, task #86 holds the retry plan);
MODEL-2 (albor sovereign) follows in a separate ship per spec §12.4.

**v2.10.0 amendment (2026-04-18):** MODEL-1 v2 root cause is **DEFINITIVE**:
non-converged QLoRA adapter. Deep-probe sub-agent (memory:
`project_ship_two_001_model1_qlora_divergence.md`) found the smoking gun in
`instruct-qlora-7b/best/metadata.json` — `train_loss=15.41`,
`val_loss=31.99`, `train_perplexity=1e6`, `val_perplexity=1e6`,
`epoch=0` (of planned 3). The `best/` and `epoch-0/` adapter safetensors
are byte-identical; training halted at epoch 0 with both losses
diverging and perplexity saturated at the 1M cap. Merging this
non-converged adapter into Qwen2.5-Coder-7B produced the mode-collapsed
`ylkoylkoylko…` output observed by AC-SHIP1-005. **Hypotheses all
FALSIFIED**: tokenizer (embedded BPE loads cleanly, `embed_tokens`
byte-identical to teacher), tensor layout (`apr qa` Tensor Contract PASS,
339 tensors pass PMAT-235), quantization (Q4K lm_head stats match
teacher f32 within quant noise). Probable failure mode: LR=2e-4 too
hot for rank-16 actual (recipe specified rank=32) × soft-label
temperature=4.0. **Ship decision**: TEACHER-ONLY
(`qwen2.5-coder-7b-instruct-q4k.apr`, 85.98% pass@1 confirmed via
`/home/noah/src/apr-leaderboard/results/humaneval_20260328_121327.json`
— 141/164 pass). AC-SHIP1-005 (distilled student ≥30% HumanEval)
blocked by MODEL-1 retry (task #86, out of scope for current ship).
EX-05/06/07 proceeds with teacher artifacts only. Reduced-gate ship per
§ Failure Protocol (Hansei).

**v2.9.0 amendment (2026-04-18):** EX-04 **DISCHARGED**. Two falsifications
of the v2.8.1 code motivated two fixes:
1. v1.1.2 — `upload_via_xet` was early-returning on the Xet branch,
   skipping the LFS-pointer commit entirely (bytes in CAS but invisible
   on repo tree). Evidence: `evidence/ship-two-001/ex-04-xet-clobber-falsification.json`.
2. v1.1.3 — after the v1.1.2 fix, `commit_lfs_pointer` was using
   `application/json` with an `{operations:[{op:addOrUpdate,...}]}`
   schema that HF Hub accepts with HTTP 200 + `success:true` but
   silently no-ops (produces empty commits identical to parent tree).
   Evidence: `evidence/ship-two-001/ex-04-xet-postfix-still-falsified.json`.
   Fix: NDJSON body (newline-delimited JSON) with `Content-Type: application/x-ndjson`
   and `{"key":"header",...}` + `{"key":"lfsFile","value":{...}}` line
   schema. New gate **FALSIFY-PUB-LFS-011** (NDJSON schema) with source-
   invariant test `commit_lfs_pointer_uses_ndjson_lfsFile_schema`.
   Live discharge: `evidence/ship-two-001/ex-04-xet-postfix-v1.1.3-discharged.json`
   — all three formats (8.0 GiB .apr, 8.0 GiB .gguf, 15.2 GiB .safetensors)
   now present on `/tree/main` with sha256 oids matching staging; GGUF
   idempotent re-upload completed in 16.9s (CAS cache-hit). Contract
   `contracts/apr-publish-hf-large-file-v1.yaml` bumped to v1.1.3,
   `status` → `DISCHARGED`. **FALSIFY-PUB-LFS-009/010/011** all DISCHARGED.
   Next: EX-05 (verify-manifest live), EX-06 (apr pull + re-inference),
   EX-07 (tag release SHIP-TWO-001-MODEL-1-TEACHER).

**v2.8.1 amendment (2026-04-18):** Phase 2 of F-PUB-LFS-001 shipped in
commit `18fd9536e` (PR #882). The `xet` sub-feature wires `hf-xet`
1.5.1 (HF's Apache-2.0 reference impl) into `apr publish`. The
`reject_oversized_file` hard-abort is deleted; files > 5 GiB now
dispatch through `crates/aprender-core/src/hf_hub/xet.rs::XetUploader`,
which uses the `hf-xet` blocking API (`XetSessionBuilder` → token-
refresh URL → `upload_from_path_blocking` → `commit_blocking`). The
client-side surface is 178 lines because phases 3–7 of the Xet
protocol (chunking, dedup, xorb/shard CAS upload, hash encoding) are
delegated wholesale to the reference impl. **FALSIFY-PUB-LFS-001**
(file-size dispatch) and **-002** (token-refresh URL shape) are
deterministically discharged by 4 unit tests; **-003..-009** are
inherited from `hf-xet`; **-010** (three-format dogfood) still
pending HF_TOKEN in the ship environment. Contract
`contracts/apr-publish-hf-large-file-v1.yaml` bumped to v1.1.0 with
status `IMPLEMENTED`.

**v2.8.0 amendment (2026-04-18):** EX-04 discovered that `apr publish`
aborts on every SHIP-TWO-001 teacher artifact because all three formats
exceed the 5 GiB HTTP preupload threshold (.apr 8.0 GiB / .gguf 8.0 GiB /
.safetensors 15.2 GiB). The fix is NOT sharding (workaround) and NOT a
self-hosted S3 mirror (not sovereign — AWS-dependent). The fix is to
implement HF Hub's actual current large-file protocol: **Xet**
(huggingface.co/docs/xet/index v1.0.0, reference Rust impl
github.com/huggingface/xet-core Apache-2.0 v1.4.3). New contract
`contracts/apr-publish-hf-large-file-v1.yaml` v1.0.0 codifies the
10-gate falsification set **FALSIFY-PUB-LFS-001..010** (file-size
dispatch, token acquisition, chunk/xorb invariants, shard ordering,
idempotency, retry policy, hash-string encoding, LFS pointer commit,
three-format dogfood). See §12.8 for the full protocol amendment.

**v2.7.0 amendment (2026-04-18):** the pre-flight gate set grows to nine.
**FALSIFY-PM-009** (APR magic-bytes Poka-Yoke, contract
`publish-manifest-v1.yaml` v1.3.0) closes the three-format ship symmetry
— every shipped format (`.safetensors`, `.gguf`, `.apr`) now has a
pre-flight gate that aborts BEFORE any network I/O when the staged file
disagrees with the manifest. v1.0 scope for PM-009 is magic-bytes only:
first 4 bytes must be one of `APR\0`, `APRN`, `APR1`, `APR2`. The exact
class it catches is "wrong file staged under format=apr manifest" (e.g.
a GGUF renamed `.apr`, or a stray `.safetensors`). Tensor-index quant
validation deferred to v1.1. 45 unit tests on every push; real-artifact
dogfood evidence in §12.7.

**v2.6.0 amendment (2026-04-18):** the pre-flight gate set grew from seven
to eight. **FALSIFY-PM-008** (GGUF tensor-type Poka-Yoke, contract
`publish-manifest-v1.yaml` v1.2.1) closes the same ship-blocker class as
PM-007 but for the `.gguf` format. Evidence surfaced during the discharge
run that `general.file_type` is advisory: our own 8 GiB teacher GGUF ships
with stale `file_type = 0` (ALL_F32) despite fully Q4_K tensors, so PM-008
treats the **predominant GGML tensor type** as authoritative and the
metadata field as a fallback. Real-artifact verification at
`evidence/ship-two-001/ex-04-preflight-gate-smoketest.json`.

**v2.5.0 amendment (2026-04-18):** all seven ship manifest gates (PM-001..007)
now run inside `scripts/ship-two-001/ex-04-upload-hf.sh` as a pre-flight
Poka-Yoke. Any manifest-vs-artifact divergence aborts with non-zero exit
BEFORE any network I/O (contract `apr-cli-publish-extra-v1.yaml` v1.2.0,
`publish-manifest-v1.yaml` v1.1.0). Local validation shows all three ship
artifacts (`.apr`, `.safetensors`-fp16, `.gguf`) PASS every gate — the ship
is unblocked on `HF_TOKEN` alone.

---

## 1. Abstract

This specification defines the contract-first, falsification-driven plan to ship **production models**
through the aprender monorepo, proving end-to-end sovereignty (training → format → inference → eval) of the
Sovereign AI Stack.

**v2.0.0 scope change:** the original distilled-student artifact failed the 2026-04-17 contract-first
audit (see §1.5). The spec now pivots to an **expedited teacher-first ship** (see §12) while defering
distillation and sovereign training to follow-on releases. Either artifact reaching SHIP status
falsifies the null hypothesis "the stack cannot produce shippable weights"; the teacher-first ship
alone satisfies that falsification.

Original (v1.0.0) scope, retained for reference:
1. **MODEL-1 (apr-leaderboard):** A distilled Qwen2.5-Coder-7B student targeting **87.20% HumanEval pass@1**,
   shippable in **~36 engineering hours** from the current trained checkpoint.
2. **MODEL-2 (albor):** A sovereign, from-scratch **370M Python code-completion model** targeting **≥30%
   HumanEval pass@1**, shippable in **3–4 weeks** of compute + engineering.

All shipped artifacts must load via `apr run` (realizar backend), pass `apr qa` Golden Output gates, and
carry a contract-conforming manifest (`contracts/publish-manifest-v1.yaml`).

---

## 1.5. Audit Findings (2026-04-17)

**Verdict:** v1.0.0 MODEL-1 SHIP PATH IS BLOCKED. AC-SHIP1-005 falsified; teacher-first pivot in progress.

### 1.5.1 What was audited

Under contract `F-EVAL-HUMANEVAL-AUDIT-001` (`contracts/eval-harness-humaneval-v1.yaml` v1.1.0):
- Primary student checkpoint: `qwen2.5-coder-7b-distilled-v2-q4k.apr` (5.8 GB, Apr-3)
- Audit tool: `apr qa` + partial `apr eval --benchmark humaneval` via apr-leaderboard harness
- Binary: `/mnt/nvme-raid0/targets/aprender/release/apr` 0.31.0 (commit 9217e9c8a), RTX 4090

### 1.5.2 What we found

| Gate                 | Result            | Measured                                                           |
|----------------------|-------------------|--------------------------------------------------------------------|
| Capability Match     | ✓ PASS            | —                                                                  |
| Tensor Contract      | ✓ PASS            | 339 tensors pass PMAT-235                                          |
| Metadata Plausibility| ✓ PASS            | arch=qwen2, rope_theta=1000000                                     |
| **Golden Output**    | **✗ FAIL**        | For "2+2=" expected "4"; got `xxx9,x,x,,,,,,,,,,,,,999`            |
| Throughput           | ✓ PASS            | 9.7 tok/s (threshold=1)                                            |
| HumanEval pass@1     | **~0 (inferred)** | Batch output was incoherent BPE ("uardsylkoylkoiaÅĤ...") on 2/164  |
| Teacher pass@1       | 85.98 (prior run) | `results/humaneval_20260328_121327.json` — pipeline is sound       |

**The distilled student cannot generate coherent text.** Its tensors are structurally valid (all pass
shape, dtype, and non-finite checks) but its weights do not represent a working model.

### 1.5.3 Five-Whys (recorded in contract `validation_result_v1_1`)

1. **Why did the audit fail?** Student emits garbage BPE sequences on every prompt.
2. **Why is output garbage if weights load?** Tensor Contract validates structure, not semantics —
   weights with legal dtype+shape+finite values can still be a broken model.
3. **Why might weights be broken?** Three candidates, in decreasing likelihood:
   (a) distillation diverged and the run was saved without a sanity gate;
   (b) `apr convert --quantize q4_k_m` introduced a LAYOUT-001-class transpose bug;
   (c) BPE tokenizer / chat-template drift so generation samples from wrong token space.
4. **Why can't we tell which?** Diagnostics (`apr diff`, merged-checkpoint run, tokenizer round-trip)
   were not required gates — they remain diagnostic follow-ups in the contract.
5. **Why did no earlier gate catch this?** `apr qa` Tensor Contract exits PASS before Golden Output
   runs; Golden Output failure does NOT block publish in the current gate matrix. This is the
   root contract gap, now promoted to the expedited plan's first action (§12.1).

### 1.5.4 Notable gap surfaced

The 87.20% figure traces back to recipe-h-32b-distill.yaml's comment labelling the *base 7B-Instruct
few-shot* HumanEval — not a distilled-student zero-shot run. No `apr eval` result file for a
distilled student exists in `apr-leaderboard/results/`; all 17 archived HumanEval runs measure the
teacher. The headline number in v1.0.0 §4.1 was therefore never a reproducible claim.

---

## 2. Motivation

### 2.1 Why These Two Models

| Criterion                    | MODEL-1 (apr-leaderboard) | MODEL-2 (albor)        |
|------------------------------|---------------------------|------------------------|
| Current state                | trained, needs packaging  | architecture designed, pretraining required |
| Engineering distance to SHIP | 36 h                      | 3–4 weeks              |
| Proves distillation path     | yes                       | no                     |
| Proves sovereign path        | partial (uses HF teacher) | **yes (end-to-end)**   |
| Proves eval harness          | yes (HumanEval)           | yes (HumanEval)        |
| Risk profile                 | LOW                       | MEDIUM-HIGH (training) |

Shipping both gives orthogonal proof: one demonstrates the stack can finish what PyTorch started;
the other demonstrates the stack can start AND finish without PyTorch in the loop.

### 2.2 Explicit Non-Goals (v1)

- Not shipping: `entrenar-rl` (POC), `entrenar-rlhf` (POC), `verificar-agent` (research).
- Not targeting: chat tuning, multimodal, tool use, >10B params.
- Not blocking on: full leaderboard automation (post-SHIP), wandb integration, distributed training.

### 2.3 Research Catalog — Reference Implementations (v2.29.4)

**The purpose of SHIP-TWO-001 is to find gaps in the Sovereign AI Stack
and fix them.** Every unexpected-slow path, crash, numerical drift, or
missing kernel surfaced during MODEL-2 work is a **bug to fix at root**,
not a reason to route around. When a fix requires understanding how
established projects solved the same problem, check these four reference
implementations first, cite the relevant file/commit in the fix, and
record the insight in the contract that governs the fix.

| Repo            | Path             | Primary value for SHIP-TWO-001                                                                          |
|-----------------|------------------|---------------------------------------------------------------------------------------------------------|
| **unsloth**     | `~/src/unsloth`  | Triton / CUDA kernels for LoRA / QLoRA / 4-bit quantized training; optimized RoPE + Swiglu + RMSNorm    |
| **vllm**        | `~/src/vllm`     | PagedAttention + paged KV cache; efficient continuous batching; prefix caching; tensor-parallel serving |
| **pytorch**     | `~/src/pytorch`  | Reference autograd semantics; numerical precision ground truth for backward pass parity                 |
| **candle**      | `~/src/candle`   | Rust-native reference: compare our API ergonomics + perf; validate our tensor abstraction choices       |

Usage rules:

1. **Read before you write a custom kernel.** If our path is slow or
   crashes, a reference impl has probably solved it. Cite
   `~/src/<repo>/path/to/file.py:LINE` (or `.cu`, `.cpp`, `.rs`) in
   the commit message AND in the provable contract's `references:`
   block.
2. **Translate, don't copy.** License / attribution matters; each
   reference repo has its own license. Use reference impls as
   algorithmic guides, not source for verbatim copy.
3. **Regression tests parity-check against the reference.** When a
   kernel is rewritten for correctness or speed, add a test that
   asserts `|ours(x) − reference(x)| < ε` on a canonical input set.
   For pytorch this is a direct numerical anchor; for candle it's a
   Rust-to-Rust sanity check; for unsloth it's a CUDA-to-CUDA check.
4. **The reference repos do NOT constrain SHIP-TWO-001 scope.** We
   don't have to match their API; we use them as oracles for what's
   possible and what's correct, not as targets.

---

## 3. Design Principles

| #  | Principle                   | Operationalization                                                    |
|----|-----------------------------|-----------------------------------------------------------------------|
| 1  | Contract-first              | Every weight file, config, and eval path has a YAML contract BEFORE code |
| 2  | Falsification-driven        | Every acceptance criterion has a named, executable FALSIFY-* test     |
| 3  | Sovereign                   | No PyTorch in the production path; GGUF/APR/SafeTensors only          |
| 4  | Lean on existing artifacts  | Reuse `contracts/model-families/qwen2.yaml` and `llama.yaml` — do not fork |
| 5  | Dogfood tooling             | `apr qa`, `apr bench`, `apr trace`, `apr eval` — never bespoke scripts |
| 6  | Binary gates                | Every GATE-SHIP-* is pass/fail; no partial credit                     |
| 7  | Five-Whys on failure        | Any FALSIFY-* failure triggers documented Hansei (§10) before retry   |
| 8  | Zero tolerance              | We never accept bugs or poor performance. Defects and perf regressions are both blockers, not trade-offs. All work improves or holds the line; never degrades it. No "pre-existing" carve-outs. No `#[ignore]` as a release valve. |
| 9  | Monorepo single source of truth | The monorepo is canonical for architecture, contracts, and tooling conventions. All downstream repos (albor, apr-leaderboard, etc.) MUST stay in sync with the monorepo. Divergence is a defect, not a parallel track. Downstream repos may own corpus, HPO, configs, and evidence — they MAY NOT fork the architectural contract. Enforcement: PMAT-693 `cargo xtask audit-ship-two --include-albor` CI gate. Ratified 2026-04-23 per PMAT-685 Option B decision. |
| 10 | Fix root causes, never route around | The purpose of SHIP-TWO-001 is to **find and fix** bugs / perf gaps in the Sovereign AI Stack. Any crash, numerical drift, silent slowdown, or missing kernel surfaced during MODEL-2 work is a bug that MUST be fixed at root, not worked around by host-skipping, feature-disabling, or "good enough" fallback paths. Each fix gets (a) a five-whys commit analysis, (b) a provable contract binding the invariant, (c) a regression test (parity-check against ../unsloth /vllm /pytorch /candle where applicable). Ratified 2026-04-23 per user directive: "the entire point of spec is to find bugs/performance gaps and fix along the way." |

---

## 4. Model 1 — apr-leaderboard (Distilled Qwen2.5-Coder-7B)

> **⚠ 2026-04-17 audit (v2.0.0):** The student checkpoint that was the subject of this section
> produces garbage tokens (see §1.5). MODEL-1 v1.0.0 as specified cannot ship. This section is
> retained unchanged as historical scope; the path forward is in §12 (teacher-first expedited ship).

### 4.1 Current State

- Teacher: `Qwen/Qwen2.5-Coder-7B-Instruct` (matches `contracts/model-families/qwen2.yaml` 7B variant).
- Student: same architecture, distilled on 20K code-instruction pairs.
- ~~Measured: **87.20% HumanEval pass@1** (source: POC notebook, pre-audit).~~
  **[v2.0.0] Falsified 2026-04-17:** this figure was a pre-distillation *few-shot* teacher score
  mis-attributed to the distilled student. The distilled checkpoint's actual pass@1 under `apr eval`
  is ~0 (garbage output). See §1.5 and `contracts/eval-harness-humaneval-v1.yaml` v1.1.0.
- Format: SafeTensors (HF-native), not yet exported to GGUF or APR.
- Eval: ran on reference Python harness; `apr eval` run terminated after garbage output detected.

### 4.2 Acceptance Criteria

**On-main status** (last audited 2026-04-23 vs `601c0740f`): 6/10
PARTIAL on main; 1 pending in open PR #1009 (SHIP-009); 3 pending in
stacked branch `feat/falsify-ship-001-partial-discharge` (SHIP-001
WIP, SHIP-003 PARTIAL, SHIP-004 PARTIAL). See
`ship-two-models-spec-audit.md` §1.1.

| ID            | Criterion                                                                 | Verification            | On-main status (2026-04-23) |
|---------------|---------------------------------------------------------------------------|-------------------------|-----------------------------|
| AC-SHIP1-001  | Student weights load via `realizar::Model::load_safetensors`              | FALSIFY-SHIP-001        | ✗ stacked branch (d4c6b6141 WIP) |
| AC-SHIP1-002  | `apr run <model>.safetensors --prompt "def fib(n):"` emits valid Python   | FALSIFY-SHIP-002 **(PARTIAL_ALGORITHM_LEVEL v2.26.0)** | ✓ on main (PR #1017, `qa/ship_002.rs`) |
| AC-SHIP1-003  | Convert to APR via `apr convert --quantize q4_k_m`; round-trip weights match (cos ≥ 0.999) | FALSIFY-SHIP-003 | ✗ stacked branch (f9c2d4753 PARTIAL) |
| AC-SHIP1-004  | Export to GGUF via `apr export --format gguf`; loads in llama.cpp         | FALSIFY-SHIP-004        | ✗ stacked branch (5f1db6ab7 PARTIAL) |
| AC-SHIP1-005  | `apr eval --benchmark humaneval` reproduces ≥86.00% pass@1 (allow 1.2% noise) | FALSIFY-SHIP-005 **(PARTIAL_ALGORITHM_LEVEL v2.27.0)** | ✓ on main (PR #1021, `metrics/ship_005.rs`) |
| AC-SHIP1-006  | `apr qa <model>` — all 8 gates PASS (Golden Output, layout, tensor stats, etc.) | FALSIFY-SHIP-006 **(PARTIAL_ALGORITHM_LEVEL v2.25.0)** | ✓ on main (PR #1013, `qa/ship_006.rs`) |
| AC-SHIP1-007  | `apr bench` decode throughput ≥30 tok/s on RTX 4090 (7B Q4_K target)      | FALSIFY-SHIP-007 **(PARTIAL_ALGORITHM_LEVEL v2.29.0)** | ✓ on main (PR #1019, `bench/ship_007.rs`) |
| AC-SHIP1-008  | Chat template (`contracts/chat-template-v1.yaml`) applies cleanly        | FALSIFY-SHIP-008 **(PARTIAL_ALGORITHM_LEVEL v2.24.0)** | ✓ on main (PR #1012, `text/chat_template/ship_008.rs`) |
| AC-SHIP1-009  | License & provenance recorded in `model.apr` metadata (Qwen2 Apache-2.0) | FALSIFY-SHIP-009 **(claimed PARTIAL_ALGORITHM_LEVEL v2.23.0 — NOT on main; GATE-APR-PROV-004 unmerged)** | ✗ PR #1009 OPEN (`feat/falsify-ship-009-partial-discharge` @ 90598277a) |
| AC-SHIP1-010  | Published artifact URL resolves; SHA-256 matches manifest                 | FALSIFY-SHIP-010 **(PARTIAL_ALGORITHM_LEVEL v2.28.0)** | ✓ on main (PR #1022, `format/ship_010.rs`) |

### 4.3 Critical Path (MODEL-1)

```
[checkpoint.safetensors] ──► AC-001 load ──► AC-002 run ──► AC-005 eval (baseline)
                                                 │                    │
                                                 ▼                    ▼
                                        AC-008 chat-template   AC-006 qa gates
                                                 │                    │
                                                 ▼                    ▼
                                         AC-003 convert ──► AC-007 bench
                                                 │
                                                 ▼
                                         AC-004 export gguf
                                                 │
                                                 ▼
                                         AC-009 metadata ──► AC-010 publish
```

### 4.4 Contract Registry (MODEL-1)

Leverages 28 existing contracts from the apr-leaderboard POC, promoted into the monorepo:

| Kind             | Contract                                              | Status      |
|------------------|-------------------------------------------------------|-------------|
| model-family     | `contracts/model-families/qwen2.yaml`                 | EXISTS      |
| tensor-layout    | `contracts/tensor-layout-v1.yaml`                     | EXISTS      |
| chat-template    | `contracts/chat-templates-v1.yaml` (qwen2 variant)    | EXISTS      |
| eval-harness     | `contracts/eval-harness-humaneval-v1.yaml`            | **NEW**     |
| distillation     | `contracts/distillation-pipeline-v1.yaml`             | **NEW**     |
| publish-manifest | `contracts/publish-manifest-v1.yaml`                  | **NEW**     |

---

## 5. Model 2 — albor (Sovereign 370M Python Code Completion)

### 5.1 Current State

- Architecture: LLaMA-family decoder, 370M params (hidden=1024, layers=24, heads=16, kv_heads=4).
  Slot: registered as a new variant under `contracts/model-families/llama.yaml` `370m`.
- Tokenizer: BPE over 50K vocab, Python-biased corpus.
- Training data: 60GB deduplicated Python (The Stack v2 filtered subset).
- Target: ≥30% HumanEval pass@1 (baseline reference: CodeParrot 1.1B ≈ 4%, StarCoderBase 1B ≈ 15.4%).
- Current blocker: pretraining run not yet executed end-to-end via `entrenar` CUDA path.

### 5.2 Acceptance Criteria

| ID            | Criterion                                                                 | Verification           |
|---------------|---------------------------------------------------------------------------|------------------------|
**On-main status** (last audited 2026-04-23 vs `601c0740f`): 6/13
touched on main (3 DISCHARGED + 3 PARTIAL); 4 pending in open PRs
(#1004/1005/1006/1008); 2 blocked on task #132 Phase 3 RTX 4090
compute dispatch; AC-SHIP2-013 (added v2.29.4) blocked on trueno#203
pre-compiled sm_121 cubins. See `ship-two-models-spec-audit.md` §1.2.

| ID            | Criterion                                                                 | Verification           | On-main status (2026-04-23)  |
|---------------|---------------------------------------------------------------------------|------------------------|------------------------------|
| AC-SHIP2-001  | Architecture registered in `contracts/model-families/llama.yaml` 370m     | FALSIFY-SHIP-011       | ✓ DISCHARGED on main (PR #898) |
| AC-SHIP2-002  | Tokenizer trained; `apr tokenize` round-trip exact on 10K held-out docs   | FALSIFY-SHIP-012       | ✓ PARTIAL on main (PR #898)    |
| AC-SHIP2-003  | `entrenar` pretraining loop reaches target loss (CE ≤ 2.2 on val)         | FALSIFY-SHIP-013       | ⏸ blocked on task #132 Phase 3 |
| AC-SHIP2-004  | Training on RTX 4090 completes within 21 days (hardware budget)           | FALSIFY-SHIP-014       | ⏸ blocked on task #132 Phase 3 |
| AC-SHIP2-005  | Checkpoint weights saved as `.apr` (native format, no PyTorch)            | FALSIFY-SHIP-015       | ✓ PARTIAL on main (PR #898)    |
| AC-SHIP2-006  | `apr qa <model>.apr` — all 8 gates PASS                                   | FALSIFY-SHIP-016       | ✗ PR #1008 OPEN (`feat/falsify-ship-016-partial-discharge`) |
| AC-SHIP2-007  | `apr run` produces syntactically valid Python on 100 held-out prompts     | FALSIFY-SHIP-017       | ✗ PR #1004 OPEN (`feat/falsify-ship-017-partial-discharge`) |
| AC-SHIP2-008  | `apr eval --benchmark humaneval` ≥30.0% pass@1                            | FALSIFY-SHIP-018       | ✗ PR #1006 OPEN (`feat/falsify-ship-018-partial-discharge`) |
| AC-SHIP2-009  | GGUF export loads in llama.cpp AND produces matching tokens (tol ≤ 1e-3)  | FALSIFY-SHIP-019       | ✓ PARTIAL on main (PR #898)    |
| AC-SHIP2-010  | `apr bench` decode ≥100 tok/s on RTX 4090 (370M target)                   | FALSIFY-SHIP-020       | ✗ PR #1005 OPEN (`feat/falsify-ship-020-partial-discharge`) |
| AC-SHIP2-011  | Training reproducible: seed fixed, two runs produce identical first 100 steps | FALSIFY-SHIP-021   | ✓ DISCHARGED on main (PR #898) |
| AC-SHIP2-012  | Weights + tokenizer + config published with CC-BY-4.0 data provenance     | FALSIFY-SHIP-022       | ✓ DISCHARGED on main (PR #898) |
| AC-SHIP2-013  | Backend parity + gx10 pretrain residency (§5.6 multi-backend policy)      | FALSIFY-SHIP-025/026   | ⏸ blocked on trueno#203 (PMAT-696) |

### 5.3 Critical Path (MODEL-2)

```
[llama.yaml 370m entry] ──► AC-001 ──► AC-002 tokenizer
                                               │
                                               ▼
                                      AC-011 reproducibility check (dry run, 100 steps)
                                               │
                                               ▼
                                      AC-003 pretraining loop
                                               │
                                               ▼
                                      AC-004 hardware budget ── (MONITOR) ──► AC-005 save .apr
                                                                                    │
                                                              ┌─────────────────────┼─────────────────────┐
                                                              ▼                     ▼                     ▼
                                                       AC-006 qa gates      AC-007 run valid     AC-008 humaneval
                                                                                                         │
                                                                             ┌───────────────────────────┤
                                                                             ▼                           ▼
                                                                      AC-009 gguf export         AC-010 bench
                                                                             │
                                                                             ▼
                                                                      AC-012 publish
```

### 5.4 Contract Registry (MODEL-2) + monorepo crate layout

Originally 54 contracts from the albor POC were planned for promotion into the
monorepo. Audit 2026-04-23 shows that promotion is partial (see §5.5). The
contracts active for the on-main training path and the crates that implement
them:

| Kind             | Contract (monorepo path)                                   | Status on main    | Implementing crate(s)                                                       |
|------------------|------------------------------------------------------------|-------------------|------------------------------------------------------------------------------|
| model-family     | `contracts/model-families/llama-370m-sovereign-v1.yaml`    | ACTIVE v1.5.0     | `crates/aprender-train/src/models/llama_370m.rs` (byte-equal to contract)    |
| tokenizer        | `contracts/tokenizer-bpe-v1.yaml`                          | PROPOSED v1.2.0   | `crates/aprender-train/src/tokenizer/bpe.rs`, CLI `apr tokenize train/encode-corpus` |
| dataset          | `contracts/dataset-thestack-python-v1.yaml`                | PROPOSED          | `crates/apr-cli/src/bin/apr-corpus-ingest.rs`                                |
| pretokenize-bin  | `contracts/pretokenize-bin-v1.yaml`                        | PROPOSED v1.0.0   | `crates/apr-cli/src/commands/tokenize_commands.rs` (`encode-corpus`), `entrenar::train::shard_reader::ShardBatchIter` |
| training-loop    | `contracts/training-loop-pretrain-v1.yaml`                 | ACTIVE v1.1.0     | `crates/aprender-train/src/train/pretrain.rs` + `.../transformer_trainer/{trainer,cuda_trainer}.rs` |
| gpu-backend      | `contracts/entrenar/gpu-training-backend-v1.yaml`          | PROPOSED v1.0.0   | `crates/aprender-train/src/train/device.rs`, `apr-cli::commands::pretrain::drive_real_{cpu,cuda}` |
| checkpoint       | `contracts/apr-provenance-v1.yaml` (APR v2 inline fields)  | ACTIVE v1.0.0     | `crates/aprender-core/src/format/{write,metadata}.rs`, `AprCheckpointFn`     |
| eval-harness     | `contracts/eval-harness-humaneval-v1.yaml` (shared)        | SHARED            | `crates/apr-cli/src/commands/eval/mod.rs`, inference via `crates/aprender-serve/` |
| publish-manifest | `contracts/publish-manifest-v1.yaml` (shared)              | DRAFT v1.4.0      | `crates/aprender-core/src/hf_hub/` + `crates/apr-cli/src/commands/publish.rs`|

**Runtime call graph** when `apr pretrain --mode from-scratch --device cuda:0` runs:

```
apr (root binary, cargo install aprender)
 └─ apr-cli::commands::pretrain::execute                     [crates/apr-cli]
     ├─ resolve_device() + preflight_tokenizer_vocab_matches_model()
     └─ drive_real() → drive_real_cuda()
         ├─ entrenar::train::pretrain_real_cuda::build_shared_cuda_trainer
         │   └─ CudaTransformerTrainer::new                  [crates/aprender-train]
         │       └─ trueno CUDA kernels (cuBLAS GEMM, NF4 dequant, fused CE)
         │                                                    [crates/aprender-compute]
         ├─ ShardBatchIter over *.bin shards                  [crates/aprender-train]
         └─ PretrainLoop::run() (GATE-TRAIN-005 divergence abort)
             └─ AprCheckpointFn writes .apr per epoch         [crates/aprender-core]
```

Legacy naming: `crates/aprender-train/` was once the standalone `entrenar`
repo; `crates/aprender-compute/` was `trueno`; `crates/aprender-serve/` was
`realizar`. The `[lib] name` fields preserve the old identifiers for
backward-compat — e.g. `use entrenar::...`. See
`docs/specifications/aprender-monorepo-consolidation.md` for the full
70-crate layout.

### 5.5 Relationship to the albor parallel training lab

**Status note (added v2.29.2, 2026-04-23):** the spec until this amendment
treated MODEL-2 as an artifact *to be built inside the monorepo*. Audit
(`ship-two-models-spec-audit.md`) and direct inspection of `~/src/albor`
found that albor is a **live, independent training lab** that has been
running MODEL-2-class pretraining experiments for weeks and is the
authoritative source of trained checkpoints — not the monorepo scaffold.

**Albor today (`~/src/albor`, last commit `be23737` 2026-04-05):**

- 350M-parameter decoder, Python code completion target
- 29 pretrain configs on disk (`configs/train/pretrain-350m-v01..v29.yaml`)
- v28 ran to step 11K on gx10 / RTX 4090, peaked 38.53 HumanEval, diverged
  to 75.65 val loss → STOPPED
- v29 planned on filtered data (2.04B clean tokens, codeparrot-quality-filtered)
- 54 contracts in `~/src/albor/contracts/`, 129+ gaps in the Sovereign AI stack
  identified during dogfooding
- Drives the monorepo `apr` binary via `bin/apr-train` — albor uses the
  consolidated CLI, not its own training code. The actual training logic
  sits in `crates/aprender-train/` (this repo).

**Material config divergence** between albor v29 (live) and the monorepo's
`Llama370MConfig` scaffold + `llama-370m-sovereign-v1.yaml` contract:

| Field                   | Monorepo `Llama370MConfig` + contract | Albor v29 live config            | Impact                                                            |
|-------------------------|----------------------------------------|----------------------------------|-------------------------------------------------------------------|
| `hidden_size`           | 1024                                   | 1024                             | same ✓                                                             |
| `num_hidden_layers`     | 24                                     | 24                               | same ✓                                                             |
| `num_attention_heads`   | 16                                     | 16                               | same ✓                                                             |
| `num_key_value_heads`   | 4 (GQA)                                | 4 (GQA)                          | same ✓                                                             |
| `intermediate_size`     | **2816** (2.75 × hidden)               | **4096** (4.0 × hidden)          | **DIFFERENT** — changes per-layer FFN params and compute ratio    |
| `vocab_size`            | **50,257** (GPT-2 aligned, task #131)  | **32,768** (albor-tokenizer-v2)  | **DIFFERENT** — incompatible embedding + lm_head shape            |
| `max_position_embeddings` | **4096**                             | **1024**                         | **DIFFERENT** — albor trained on seq_len=1024 only                |
| `rope_theta`            | 10000                                  | 10000                            | same ✓                                                             |
| `rms_norm_eps`          | 1e-5                                   | 1e-5                             | same ✓                                                             |
| Target parameter count  | 370M                                   | 350M                             | **DIFFERENT** — follows from vocab + FFN deltas                   |
| Tokenizer file          | 50,257-BPE Python (monorepo-trained)   | `models/albor-tokenizer-v2/…`    | **DIFFERENT** — incompatible vocab spaces                         |

**Conclusion: they are NOT the same model.** Any future dispatch of
`apr pretrain` that loads a monorepo `.apr` checkpoint into albor's
inference path (or vice versa) will fail the tokenizer vocab preflight
gate (`preflight_tokenizer_vocab_matches_model`) immediately.

### 5.5.1 Decision — Option B (DECIDED 2026-04-23)

**Standing policy (ratified 2026-04-23):**

> **The monorepo is the single source of truth. All downstream repos
> — including albor — MUST align to monorepo conventions, contracts,
> and architecture. Divergence is a defect, not a parallel track.
> Keep all repos in sync with the monorepo at all times.**

This is now a durable rule; it outranks any ad-hoc local choice in
a downstream repo. Memory pointer: `feedback_monorepo_single_source_of_truth.md`.

**Chosen reconciliation: Option B — albor aligns to monorepo.**

| Field                   | Monorepo (canonical)            | Albor v30 (required)             | Change vs albor v29                     |
|-------------------------|----------------------------------|----------------------------------|-----------------------------------------|
| `hidden_size`           | 1024                             | 1024                             | no change                               |
| `num_hidden_layers`     | 24                               | 24                               | no change                               |
| `num_attention_heads`   | 16                               | 16                               | no change                               |
| `num_key_value_heads`   | 4                                | 4                                | no change                               |
| `intermediate_size`     | **2816**                         | **2816**                         | 4096 → 2816 (FFN slimmed)               |
| `vocab_size`            | **50,257**                       | **50,257**                       | 32,768 → 50,257 (GPT-2 aligned)         |
| `max_position_embeddings` | **4096**                       | **4096**                         | 1024 → 4096 (4× context)                |
| `rope_theta`            | 10,000                           | 10,000                           | no change                               |
| `rms_norm_eps`          | 1e-5                             | 1e-5                             | no change                               |
| Tokenizer               | monorepo 50,257-BPE (per `tokenizer-bpe-v1.yaml`) | monorepo 50,257-BPE | `albor-tokenizer-v2` DEPRECATED         |
| Target parameter count  | 370M                             | 370M                             | 350M → 370M (follows from vocab + FFN)  |

Rationale (why Option B over A/C):

- **Sovereignty integrity.** Option A would have forced the monorepo to
  absorb a downstream fork's arithmetic choices; that breaks the "monorepo
  is canonical" property at exactly the moment the stack is trying to
  prove sovereignty. Option B preserves that property.
- **No multi-variant drift.** Option C's MODEL-2a / MODEL-2b split would
  require maintaining two contract registries, two ship gate matrices,
  and two publish pipelines for an arbitrary pair of architectural
  choices that differ only in FFN ratio + vocab. That is ongoing tax
  for zero product value.
- **Standing policy prevents recurrence.** The monorepo-single-source
  rule above means that any future downstream repo (not just albor)
  inheriting divergence is a defect the monorepo-sync audit catches, not
  a decision to be relitigated per incident.

Cost accepted: ~3–5 days of compute re-dispatch on lambda-labs (retrain
the albor tokenizer at vocab=50,257 via `apr tokenize train`, re-pretokenize
the filtered 2.04 B-token corpus with `apr tokenize encode-corpus`,
launch v30 pretraining against `Llama370MConfig`). v28's 5.08 B-token
run and v29's planned run are effectively sunk; the compute artifacts
survive as training-dynamics data but not as ship-path evidence.

### 5.5.2 Albor action list (Option B implementation)

Tracked as follow-up tickets; must complete in order because each step
depends on the previous artifact:

1. **[PMAT-688] Align `~/src/albor` tokenizer to monorepo.** Delete
   `models/albor-tokenizer-v2/`. Run `apr tokenize train
   --corpus data/filtered-codeparrot --vocab-size 50257
   --normalization nfc -o models/monorepo-bpe-v1/` (the
   monorepo command, dogfooded). Output must pass `pv validate
   contracts/tokenizer-bpe-v1.yaml` (the monorepo contract).
2. **[PMAT-689] Re-pretokenize the filtered corpus at vocab 50,257.**
   `apr tokenize encode-corpus --tokenizer models/monorepo-bpe-v1/
   --input data/filtered-codeparrot/ -o data/pretokenized-1024-v5/`.
   Result is `.bin` shards (u32 LE per `pretokenize-bin-v1.yaml`);
   `apr-corpus-ingest validate-contract` against
   `dataset-thestack-python-v1.yaml` must PASS.
3. **[PMAT-690] Update `configs/train/pretrain-350m-v30.yaml`** to
   mirror monorepo `Llama370MConfig` exactly (vocab 50257, intermediate
   2816, max_pos 4096). Drop the "350m" label — the resulting model is
   370M; rename the config `pretrain-370m-v30.yaml` and all downstream
   references. File name is now a monorepo-compliant identifier.
4. **[PMAT-687] Dispatch albor v30 pretraining** via `apr pretrain
   --mode from-scratch --device cuda:0 --config configs/train/pretrain-370m-v30.yaml`
   on lambda-labs RTX 4090. Concurrent with task #132 Phase 3 smoke — v30's
   first 50 steps ARE the Phase 3 residency proof. (Note: ticket ID
   out-of-order at 687 due to parallel-create race when tickets were
   first filed; scope unchanged.)
5. **[PMAT-691] Albor repo hygiene.** Update `~/src/albor/README.md`
   "350M" → "370M" and any spec-book chapter that cites the old
   arithmetic. Update `~/src/albor/CLAUDE.md` to include the
   monorepo-single-source rule at the top.
6. **[PMAT-692] Cross-repo contract parity audit.** `pv validate
   ~/src/albor/contracts/*.yaml` and cross-reference every albor
   contract ID against the monorepo's `contracts/` tree. Any albor
   contract with a monorepo counterpart must be pinned to the
   monorepo's version; any orphan albor contract either gets promoted
   into the monorepo or retired. Result: one canonical contract per
   responsibility, no duplicates.
7. **[PMAT-693] Bidirectional sync CI gate.** Extend the
   `cargo xtask audit-ship-two` gate from PMAT-683 with an
   `--include-albor` mode that reads `~/src/albor/configs/train/*.yaml`
   and `~/src/albor/contracts/*.yaml` and fails CI on any field that
   disagrees with the monorepo contract of the same `contract_id`.
   Structurally enforces the standing policy.
8. **[PMAT-694] Update `aprender-train/CLAUDE.md`** — current text says
   "Status: Specification phase - implementation not yet started"
   which is obviously stale given 3,432 LOC of `CudaTransformerTrainer`
   and the live `apr pretrain` CLI path.

Ticket dependencies: 688 → 689 → 690 → 687 (dispatch); 691/692/693/694
run in parallel once 687 closes. Step 4 (PMAT-687) discharges
task #132 Phase 3 + AC-SHIP2-003 target-loss gate in one dispatch.

### 5.5.3 Enforcement (how the policy survives staff turnover)

The standing monorepo-single-source rule is only durable if CI enforces
it. PMAT-693 is the enforcement mechanism; until it lands, the policy
rests on honor. Two interim stopgaps:

- `scripts/ship-two-001/check-monorepo-sync.sh` (to be added) — a bash
  harness that `git -C ~/src/albor log -1 --format=%H` + diffs the
  current monorepo-relevant albor configs against the monorepo contracts.
  Run manually before every MODEL-2 compute dispatch.
- The v2.29.2 audit doc (`ship-two-models-spec-audit.md`) recorded the
  drift; PMAT-685 memorialized the decision. Future contributors who
  read either will see the reasoning without re-litigating the choice.

### 5.6 Backend selection policy (v2.29.4 amendment, 2026-04-23)

The Sovereign AI Stack has **three NVIDIA backends + one cross-platform
backend** available for training/inference kernels, each shipping today
in `crates/aprender-compute/`. Backend selection is **"fastest wins
initially"** with hard contract gates on correctness and an explicit
"no host skipping" discipline per §3 row #10.

#### 5.6.1 Backend matrix

| Backend | Scope                 | Supported hardware                                         | Current status on `main`                                                                                         |
|---------|-----------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **PTX** (custom CUDA kernels) | Inference forward + backward training | NVIDIA sm_80 (A100), sm_86, sm_89 (RTX 4090) — pre-compiled cubins. sm_121 (Blackwell GB10) BLOCKED by trueno#200 at JIT compile during backward. | Forward path works on all; backward crashes on Blackwell. Pre-compiled kernel-bank fix tracked as **trueno#203**. |
| **cuBLAS** (NVIDIA vendor)    | GEMM forward + training-backward GEMM | All NVIDIA, all architectures (no JIT). 4 of 5 parity tests PASS vs fused PTX kernels.   | Production-ready. Already the fallback when trueno#200 fires. 298 tok/s forward. Preferred initial choice on Blackwell. |
| **WGPU** (WGSL shaders)       | Forward + backward training           | AMD, Intel, NVIDIA — anything with a WebGPU driver. Tested on AMD Radeon Pro W5700X.     | Phase 2 COMPLETE — forward/backward works via trueno's WGSL backward shaders; FALSIFY-WGPU-002 PASS. Phase 3 (full LoRA training loop integration) pending. Cross-platform sovereignty. |
| **SIMD CPU**                  | Forward only (too slow for training)  | AVX2 / AVX512 / NEON / scalar                                                             | Production. Used by `apr pretrain --device cpu` for deterministic / debugging runs. |

Reference: `crates/aprender-train/CLAUDE.md` §"cuBLAS Training
Integration Status" + §"WGPU Training Support".

#### 5.6.2 Selection policy — fastest wins initially

At the start of every compute dispatch `apr pretrain --device X`:

1. **Measure, don't assume.** On a new host or after any kernel PR,
   run `apr bench --backends all --json` to benchmark every backend
   that compiles on that host.
2. **Pick the fastest that passes correctness.** The backend with
   highest tok/s that ALSO passes `apr qa --parity-check` against a
   reference backend (typically cuBLAS) is the default for the host
   for that session.
3. **Record the choice in the contract.** The training run's
   `evidence/<run-id>/backend-selection.json` names the backend chosen
   plus the benchmark results that justified it. Reviewers audit by
   reading this file.

No backend is intrinsically "better" — Blackwell may favor cuBLAS today
and PTX tomorrow when trueno#203 lands; AMD hosts use WGPU. The policy
is about **measurement + evidence**, not brand loyalty.

#### 5.6.3 No host skipping — root-cause fix discipline

**Per §3 row #10:** a crashed backend on a host is a bug to fix, not
a reason to exclude the host. Concrete ongoing work:

| Bug / gap                                              | Host affected    | Fix owner (contract / ticket)                                                         |
|--------------------------------------------------------|------------------|---------------------------------------------------------------------------------------|
| **trueno#200** — Blackwell PTX JIT crash on backward   | gx10 GB10 sm_121 | Fix via trueno#203 (pre-compiled sm_121 cubins for all backward kernels); **NOT a host-exclusion** |
| **trueno#200 workaround** — fused NF4 fallback 15.5 tok/s | gx10 steady state | §3 row #8 Zero-Tolerance rejects this as a ship baseline; must converge to sm_121 parity with sm_89 |
| **aprender-train CUDA trainer device plumbing**        | any NVIDIA       | task #132 Phase 1+2 SHIPPED; Phase 3 evidence is the next dispatch (see §14)          |
| **WGPU LoRA training-loop integration (Phase 3)**      | AMD / Intel      | trueno WGSL backward shaders already work; wiring into `CudaTransformerTrainer` analog pending |

Research anchors (§2.3) for each class:

- **PTX/sm_121 JIT issue** — study `~/src/unsloth/unsloth/kernels/*.py`
  for how unsloth handles multi-architecture Triton cubins; compare
  with `~/src/pytorch/torch/cuda/` JIT policy.
- **cuBLAS GEMM parity** — `~/src/candle/candle-core/src/cuda_backend/`
  has Rust-native cuBLAS bindings with a clean API surface; parity-test
  the SGEMM + GEMM_EX paths we use.
- **WGPU backward shaders** — no large-scale reference; this is stack
  novel. Cite trueno's own `src/backends/gpu/*.wgsl` in all fix
  commits and bind to `wgpu-production-training-v1.yaml`.
- **PagedAttention inference** (post-training serving) —
  `~/src/vllm/vllm/attention/backends/` is canonical; we have
  `realizar` KV cache code to parity-test.

#### 5.6.4 gx10 role — training target, not eval-only

Prior spec language (and prior drafts of §5.5) implicitly treated gx10
as eval-only because of trueno#200. That framing is **retracted**:
gx10 IS a training target, blocked today by trueno#200/#203, and the
fix is on the critical path — not a "later, maybe" concern. Parallel
eval lane (§12.6) is one use; once trueno#203 lands, gx10 joins
lambda-labs as a pretraining host for AC-SHIP2-003/004 dispatches,
doubling the compute pool.

Acceptance criterion (new AC-SHIP2-013, added v2.29.4):

> `apr pretrain --mode from-scratch --device cuda:0` on gx10 produces
> `evidence/gx10-pretrain-50step.json` with median step-wall <= 2× the
> lambda-labs RTX 4090 result AND GATE-GPUTRAIN-003 residency proof
> PASS. Discharges trueno#200/#203 end-to-end.

Tracking: **PMAT-696** (gx10 backend convergence smoke). Blocked by
trueno#203 (pre-compiled sm_121 kernel bank).

---

## 6. Compound Ship Gates

All gates are binary; any failure blocks publish.

| Gate             | Description                                                    | Blocks          |
|------------------|----------------------------------------------------------------|-----------------|
| GATE-SHIP-001    | MODEL-1: all 10 AC-SHIP1-* PASS                                | MODEL-1 publish |
| GATE-SHIP-002    | MODEL-2: all 12 AC-SHIP2-* PASS                                | MODEL-2 publish |
| GATE-SHIP-003    | Both models: `apr qa` Golden Output never regresses post-quantize | publish     |
| GATE-SHIP-004    | HumanEval harness produces identical score on two consecutive runs (seed=0) | AC-005, AC-008 |
| GATE-SHIP-005    | License metadata is present AND matches upstream declaration   | publish         |
| GATE-SHIP-006    | GGUF round-trip: APR → GGUF → load in llama.cpp → first-token match (tol ≤ 1e-3) | AC-004, AC-009 |
| GATE-SHIP-007    | No unwrap() in new code (enforced by `.clippy.toml`)           | merge           |
| GATE-SHIP-008    | Contract density: every new public fn has `#[contract]`        | merge           |
| GATE-SHIP-009    | CI green: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test --workspace` | merge |
| GATE-SHIP-010    | `cargo deny check advisories` — zero vulnerabilities in weight/tokenizer dependencies | merge |
| GATE-SHIP-011    | PMAT quality score ≥ A- (project), TDG ≥ 90                    | merge           |
| GATE-SHIP-012    | Coverage ≥ 95% line on new modules (per `.pmat-gates.toml`)    | merge           |
| GATE-SHIP-013    | **Multi-backend selection parity** (§5.6): `apr bench --backends all --json` produces a row per available backend; winning backend passes `apr qa --parity-check` against cuBLAS reference | AC-SHIP2-013, dispatch  |
| GATE-SHIP-014    | **gx10 pretrain residency** (AC-SHIP2-013): `apr pretrain --device cuda:0` on gx10 GB10 produces `evidence/gx10-pretrain-50step.json` with step-wall ≤ 2× lambda-labs baseline AND GATE-GPUTRAIN-003 PASS (discharges trueno#200/#203) | MODEL-2 ship on non-sm_89 |

---

## 7. Falsification Tests

Each test is named, executable, and has a defined failure signal.

### 7.1 MODEL-1 Falsification (12 tests)

| ID                 | Test                                                              | Failure Signal                         |
|--------------------|-------------------------------------------------------------------|----------------------------------------|
| FALSIFY-SHIP-001   | `realizar::Model::load_safetensors(path)` returns Ok              | Err(_) returned                        |
| FALSIFY-SHIP-002   | Run `apr run ... --prompt "def fib(n):"`; parse output as Python AST (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-002` in `contracts/qwen2-e2e-verification-v1.yaml` v1.1.0; `cargo test -p aprender-core --lib falsify_ship_002_python_syntax_error_threshold_logic`) | SyntaxError (> 0 errors) |
| FALSIFY-SHIP-003   | Convert then compare per-layer cosine similarity                  | any layer cos < 0.999                  |
| FALSIFY-SHIP-004   | Shell out to `llama-cli` on exported GGUF; prompt → logits        | llama.cpp exit ≠ 0                     |
| FALSIFY-SHIP-005   | Run HumanEval 164× via `apr eval`; pass@1 computed (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-005` in `contracts/qwen2-e2e-verification-v1.yaml` v1.2.0; `cargo test -p aprender-core --lib falsify_ship_005_humaneval_pass_at_1_threshold_logic`) | pass@1 < 86.00% (or < 84.80% under 1.2 pp noise allowance) |
| FALSIFY-SHIP-006   | `apr qa <model>` exit code = 0 (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QA-SHIP-006` in `contracts/apr-model-qa-v1.yaml` v1.2.0; `cargo test -p aprender-core --lib falsify_ship_006_apr_qa_eight_gates_aggregate`) | any gate reports FAIL |
| FALSIFY-SHIP-007   | `apr bench --iterations 5 --max-tokens 128`; median tok/s         | median < 30                            |
| FALSIFY-SHIP-008   | Render chat template on canonical system+user; diff vs golden (**PARTIAL_ALGORITHM_LEVEL** — `GATE-CHAT-SHIP-008` in `contracts/chat-template-v1.yaml` v1.1.0; `cargo test -p aprender-core --lib falsify_ship_008_chat_template_render_bind`) | diff ≠ 0 |
| FALSIFY-SHIP-009   | `apr inspect <model>.apr`; grep for `license: apache-2.0`         | missing or mismatched                  |
| FALSIFY-SHIP-010   | curl + sha256sum against manifest                                 | hash mismatch or 404                   |
| FALSIFY-SHIP-023   | Re-run AC-005 on second day; score drift                          | drift > 1.2 pp                         |
| FALSIFY-SHIP-024   | Prompt-injection torture suite (50 adversarial inputs)            | any panic or NaN in logits             |

### 7.2 MODEL-2 Falsification (10 tests)

| ID                 | Test                                                              | Failure Signal                         |
|--------------------|-------------------------------------------------------------------|----------------------------------------|
| FALSIFY-SHIP-011   | `llama.yaml` `370m` entry validates against `_schema.yaml`        | schema error                           |
| FALSIFY-SHIP-012   | Tokenize 10K docs, detokenize, byte-compare                       | any byte mismatch                      |
| FALSIFY-SHIP-013   | Training val CE at final step                                     | CE > 2.2                               |
| FALSIFY-SHIP-014   | Wall-clock from train start to final checkpoint                   | > 21 days                              |
| FALSIFY-SHIP-015   | Load checkpoint via `apr inspect`; count params                   | params ≠ 370M ± 1%                     |
| FALSIFY-SHIP-016   | `apr qa <model>.apr` exit code                                    | any gate FAIL                          |
| FALSIFY-SHIP-017   | 100 prompts → Python AST parse                                    | ≥2 SyntaxError (tolerate ≤1)           |
| FALSIFY-SHIP-018   | `apr eval --benchmark humaneval` pass@1                           | < 30.0%                                |
| FALSIFY-SHIP-019   | GGUF export; first-token probability vs APR                       | |Δp| > 1e-3 on top-1                   |
| FALSIFY-SHIP-020   | `apr bench` median tok/s on RTX 4090                              | < 100                                  |
| FALSIFY-SHIP-021   | Run training 100 steps × 2 with seed=0; diff loss trajectories    | any step diff > 1e-6                   |
| FALSIFY-SHIP-022   | `apr inspect`; check `license`, `data_source`, `data_license`     | any field missing                      |
| FALSIFY-SHIP-025   | **Backend parity** (§5.6): `apr qa --parity-check` for every backend pair (PTX↔cuBLAS, cuBLAS↔WGPU) on canonical 100-prompt set | any pair ULP-relative diff > 1e-3        |
| FALSIFY-SHIP-026   | **gx10 pretrain smoke** (AC-SHIP2-013, PMAT-696): `apr pretrain --device cuda:0` on gx10 produces 50 steps, step-wall P50 ≤ 2× lambda-labs, `nvidia-smi` residency > 0 MiB within 5 s | any guard FAIL, crash, or wall > 2× lambda-labs |

*Note:* FALSIFY-SHIP-023/024 already appear in §7.1 for MODEL-1
(post-ship drift, prompt injection). SHIP-025/026 are MODEL-2 extensions
introduced by v2.29.4 §5.6 multi-backend policy.

---

## 8. Execution Plan

### 8.1 Phase DAG

```
                    ┌─────────────────────────────────────┐
                    │          Phase 0: Scaffold          │
                    │  (contracts, schema, test harness)  │
                    └──────────────────┬──────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      │                                 │
                      ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │   Phase 1-A (MODEL-1)   │       │   Phase 1-B (MODEL-2)   │
         │  36h — packaging path   │       │  Week 1: tokenizer + dry │
         │                         │       │  run (AC-001,002,011)    │
         │  AC-001..004 load/conv  │       └───────────┬─────────────┘
         └───────────┬─────────────┘                   │
                     ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │  Phase 2-A: eval + qa   │       │  Phase 2-B: pretraining │
         │  AC-005..008            │       │  Weeks 2-3: AC-003,004  │
         └───────────┬─────────────┘       └───────────┬─────────────┘
                     ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │  Phase 3-A: publish     │       │  Phase 3-B: eval + qa   │
         │  AC-009,010  — SHIP-1   │       │  Week 4: AC-005..012    │
         └─────────────────────────┘       └───────────┬─────────────┘
                                                       ▼
                                           ┌─────────────────────────┐
                                           │  Phase 4-B: publish     │
                                           │  SHIP-2                 │
                                           └─────────────────────────┘
```

Phase 1-A and Phase 1-B are independent and run in parallel.

### 8.2 Effort Budget

| Phase | Model    | Effort       | Calendar  | Owner        |
|-------|----------|--------------|-----------|--------------|
| 0     | shared   | 6 h          | day 0     | eng          |
| 1-A   | MODEL-1  | 10 h         | day 1     | eng          |
| 2-A   | MODEL-1  | 12 h         | day 2     | eng          |
| 3-A   | MODEL-1  | 4 h          | day 3     | eng          |
| 1-B   | MODEL-2  | 40 h         | week 1    | eng          |
| 2-B   | MODEL-2  | compute-bound | weeks 2-3 | GPU node    |
| 3-B   | MODEL-2  | 16 h         | week 4    | eng          |
| 4-B   | MODEL-2  | 4 h          | week 4    | eng          |
| **Σ** |          | 92 h + 2 wk  | ~4 weeks  |              |

### 8.3 Integration with `apr run`

After SHIP, both models must satisfy:

```bash
apr run ./model-1.apr --prompt "def quicksort(arr):"       # MODEL-1
apr run ./model-2.apr --prompt "def binary_search(xs, t):" # MODEL-2
```

Both paths resolve through `realizar::Model` (see `crates/aprender-serve/CLAUDE.md` Realizar-First Architecture).
No code path in `aprender-core` may invoke generation.

---

## 9. Risk Matrix

| # | Risk                                           | Probability | Impact | Mitigation                                                           |
|---|------------------------------------------------|-------------|--------|----------------------------------------------------------------------|
| 1 | HumanEval eval non-deterministic               | MED         | HIGH   | seed=0, greedy; GATE-SHIP-004 enforces two-run identity              |
| 2 | GGUF export has tensor-layout bug (LAYOUT-001) | HIGH        | HIGH   | FALSIFY-SHIP-019 parity check; reuse `layout_contract.rs` validator  |
| 3 | MODEL-2 training diverges                      | MED         | HIGH   | AC-SHIP2-003 loss gate; fallback = reduce LR, resume from ckpt       |
| 4 | RTX 4090 insufficient for 370M in 21 days      | LOW         | HIGH   | AC-SHIP2-004 budget; overflow → rent 2× H100 week 3                  |
| 5 | Teacher (Qwen2.5-Coder-7B) license ambiguity   | LOW         | MED    | AC-SHIP1-009; confirm Apache-2.0 in `config.json`                    |
| 6 | `apr convert` quantize drops accuracy > 1pp    | MED         | MED    | FALSIFY-SHIP-003 cos-sim gate; fallback = Q5_K_M                     |
| 7 | Tokenizer round-trip bytes mismatch            | LOW         | HIGH   | FALSIFY-SHIP-012 on 10K corpus                                       |
| 8 | `cargo install aprender` breaks during release | MED         | HIGH   | CI `cargo install --path .` smoke test (GATE-SHIP-009)               |
| 9 | HF artifact hosting outage                     | LOW         | LOW    | dual-publish to HF + self-hosted bucket                              |
| 10| CUDA JIT regression (trueno#200/203)           | MED         | MED    | Pin trueno version per memory note 2026-03-22                        |

---

## 10. Failure Protocol (Hansei)

Any FALSIFY-SHIP-* failure triggers the following sequence. No retry is permitted before completion.

### 10.1 Five Whys

1. What check failed? (name the FALSIFY-*)
2. What invariant did it violate?
3. What code path was responsible?
4. Why was the code path wrong? (bug class: layout, numeric, eval harness, toolchain)
5. Why did no earlier gate catch it? (contract gap → file follow-up)

### 10.2 Decision Gate

| Condition                                              | Action                         |
|--------------------------------------------------------|--------------------------------|
| Single test failed, root cause known, fix ≤ 2 h        | Fix + re-run full gate         |
| Multiple tests failed OR root cause unknown            | Escalate to design review      |
| AC breaks but SHIP is blocking deadline                | Reduced-gate ship (below)      |
| Hardware / compute budget breached                     | Full failure escalation        |

### 10.3 Reduced-Gate Ship (emergency only)

Acceptable only with written Noah Gift approval. A reduced ship may drop:
- AC-SHIP1-007 / AC-SHIP2-010 (bench speed) — ship with "beta performance" label
- AC-SHIP1-010 / AC-SHIP2-012 (artifact publication) — ship to internal bucket only

All 8 `apr qa` gates (Golden Output, layout, tensor stats, license, etc.) MUST still pass.

### 10.4 Full Failure Escalation

If neither model ships within the budget, this specification is void. Retrospective must
answer: (a) which assumption was wrong, (b) what contract should have caught it earlier,
(c) what gets deleted from scope before restart.

---

## 12. Expedited Ship Plan (v2.0.0 — teacher-first)

**Goal:** publish ONE artifact within **10 engineering hours** of 2026-04-17 to falsify the null
hypothesis "the stack cannot produce shippable weights."

**Strategy:** ship the **teacher** (`qwen2.5-coder-7b-instruct-q4k.apr`) under a new artifact ID
`paiml/qwen2.5-coder-7b-apache-q4k-v1`. Defer distillation proof to v1.1. Defer MODEL-2 to v2.0.

### 12.1 Pre-requisite: plug the Golden Output gate gap

Before any publish, `apr qa` must be configured so that **Golden Output failure blocks publish**.
Today it is reported but non-fatal — exactly the hole that let the v1.0.0 plan rely on a
garbage checkpoint for 14 days before audit. Track as contract amendment to `apr-qa-v1.yaml`
(or equivalent); must land before §12.2.

### 12.2 Teacher-first critical path (10h budget)

```
[qwen2.5-coder-7b-instruct-q4k.apr]      # already in apr-leaderboard/checkpoints/, 7.5 GB
         │
         ▼
  EX-01  apr qa --require-golden-output   # must PASS after §12.1 gate fix  (1 h)
         │
         ▼
  EX-02  apr eval --benchmark humaneval   # reproduces ≥84.5 pass@1 (noise-band of 85.98)  (2 h)
         │
         ▼
  EX-03  Write contracts/publish-manifest-v1.yaml entry    (1 h)
           - sha256, size_bytes, license=Apache-2.0
           - provenance.pipeline=finetune
           - provenance.parent=Qwen/Qwen2.5-Coder-7B-Instruct
           - provenance.recipe=contracts/model-families/qwen2.yaml
         │
         ▼
  EX-04  Upload artifact to HF Hub AND self-hosted bucket  (2 h)
         │
         ▼
  EX-05  Verify manifest: sha256 match, URL 200, SPDX valid  (1 h)
         │
         ▼
  EX-06  apr pull <published_id> → local file; re-run EX-02 from downloaded artifact  (2 h)
         │
         ▼
  EX-07  Tag release in spec + announce  (1 h)
```

### 12.3 Expedited Acceptance Criteria

| ID            | Criterion                                                                        | Verification        |
|---------------|----------------------------------------------------------------------------------|---------------------|
| AC-EX-001     | Golden Output gate is a HARD BLOCKER in `apr qa`                                 | FALSIFY-EX-001      |
| AC-EX-002     | Teacher passes all 8 `apr qa` gates including Golden Output                      | FALSIFY-EX-002      |
| AC-EX-003     | `apr eval --benchmark humaneval` on teacher ≥84.5% pass@1 (85.98 − 1.5 noise)   | FALSIFY-EX-003      |
| AC-EX-004     | `publish-manifest-v1.yaml` instance for artifact passes `apr validate-manifest`  | FALSIFY-EX-004      |
| AC-EX-005     | `apr pull paiml/qwen2.5-coder-7b-apache-q4k-v1` resolves + SHA-256 matches       | FALSIFY-EX-005      |
| AC-EX-006     | `apr run <published>.apr --prompt "def fib(n):"` emits syntactically valid Python | FALSIFY-EX-006     |
| AC-EX-007     | Parallel eval lane: N-shard run on ≥2 hosts matches single-host pass@1 (Δ ≤ 0.01 pp) and completes in ≤ `single_host_wall_time / N × 1.25` | FALSIFY-SHARD-001..004 |

### 12.4 Explicit Scope Cut (v2.0.0)

Moved out of v1 ship:
- **Distilled student artifact** → v1.1 (requires diagnosis per `validation_result_v1_1` ACT-01..03,
  then re-distillation with contract-gated Golden Output at each epoch).
- **MODEL-2 (albor sovereign)** → v2.0 (3+ weeks of compute; no reason to couple to MODEL-1 ship).
- **GGUF round-trip export** (AC-SHIP1-004) → v1.1 (teacher already has GGUF on HF).

### 12.5 What falsifies the expedited plan

| Condition                                         | Action                                              |
|---------------------------------------------------|-----------------------------------------------------|
| AC-EX-002 FAIL on teacher                         | Pipeline regressed — block ship, investigate realizar |
| AC-EX-003 pass@1 < 84.5                           | Harness drift since 2026-03-28; do not ship until resolved |
| AC-EX-004 manifest invalid                        | Fix manifest schema compliance, retry                |
| AC-EX-005 SHA-256 mismatch                        | Re-upload; investigate CDN/transit corruption        |
| Any EX-* step takes > 2× budget                   | Escalate; triggers §13.2 retrospective update        |
| Shard merged pass@1 differs from single-host by > 0.01 pp | Parity FAIL — block ship, investigate shard determinism (FALSIFY-SHARD-003) |
| Any shard reports missing / duplicate task_ids    | Completeness or disjointness FAIL — block ship (FALSIFY-SHARD-001/002) |

### 12.6 Parallel Eval Lane (post-hoc lesson, 2026-04-17)

**Problem surfaced during v2.0.0 ship.** EX-02 (single-host HumanEval on yoga) ran
serially for ~2 hours while `gx10` (Blackwell GB10, `apr-cli` inference unaffected
by PMAT-587 JIT issues) and any Lambda-Labs GPU instance sat idle. 5-Whys (recorded
in contract `eval-sharding-v1.yaml::five_whys`):

1. **Why only yoga?** The orchestration script accepted a single `MODEL_PATH` and
   invoked `apr run --batch-jsonl` once, consuming all 164 tasks serially.
2. **Why serial batch?** `eval-pass-at-k.sh` (inherited from apr-leaderboard) has
   no shard dimension; it assumes one GPU.
3. **Why wasn't sharding added?** EX-02 was treated as a monolithic §12.2 step;
   decomposing "generate N completions" into `generate N/k × k hosts` was not
   considered because the 10h budget was written assuming yoga alone.
4. **Why was the budget yoga-alone?** `gx10` was mentally categorized as "training,
   blocked on JIT bug" without separating the inference path, which works today.
5. **Root cause.** Spec optimized for *matching existing tooling* (one-host eval
   harness) instead of *minimizing critical path*. A 2-way shard cuts EX-02 from
   ~2h → ~1h; a 3-way shard (yoga+gx10+Lambda) to ~40 min.

#### 12.6.1 Scope

This lane is **post-hoc for v2.0.0** (sunk cost on the in-flight serial run) and
**pre-requisite for v1.1 / v2.0** future evals (distilled student, MODEL-2 sovereign,
multi-seed reproducibility runs per FALSIFY-PUBLISH-RECIPE-001).

#### 12.6.2 Architecture

```
benchmark.jsonl  ──(round-robin split, stride N)──►  shard_0.jsonl … shard_{N-1}.jsonl
                                                           │ │ … │
                                                           ▼ ▼   ▼
                                                       host_0 host_1 … host_{N-1}
                                                       (yoga) (gx10) … (lambda)
                                                           │ │ … │
                                                           ▼ ▼   ▼
                                                   humaneval_shard_i.json (each host)
                                                           │ │ … │
                                                           └─┴───┘
                                                              ▼
                                          eval-shard-merge.py: concat problems[],
                                          recompute Chen pass@1 → humaneval_merged.json
```

- **Shard algorithm.** Round-robin stride: task `i` goes to host `i mod N`.
  Evens out per-task cost variance (long prompts, long generations) without
  needing a pre-estimated cost model.
- **Model sync.** `rsync -c` (content-checksum) pushes the .apr + tokenizer to
  each host once; subsequent runs are no-ops.
- **Merge.** Per-shard result JSONs share the `eval-pass-at-k.sh` schema
  (`problems[]`, `results.passed`, `results.total`). Merge = concat `problems`,
  sum totals, recompute pass@1 using Chen et al. unbiased estimator on merged
  array.

#### 12.6.3 Acceptance (AC-EX-007 discharge)

Run the 4 FALSIFY-SHARD tests in `contracts/eval-sharding-v1.yaml`:

- **FALSIFY-SHARD-001 (completeness):** `sum(shard_i.total) == benchmark.total`
  and every benchmark task_id appears in exactly one shard result.
- **FALSIFY-SHARD-002 (disjointness):** no task_id appears in two shards.
- **FALSIFY-SHARD-003 (determinism parity):** at temperature=0.0, completions for
  task T on host A == completions for task T on host B for a 16-task probe set.
- **FALSIFY-SHARD-004 (merged-score identity):** reshard an existing single-host
  humaneval_*.json result by task_id; merged pass@1 matches within 0.01 pp of the
  original.

Evidence location: `evidence/ship-two-001/shard-eval/`.

#### 12.6.4 Non-goals for this lane

- **Dynamic load-balancing.** Static stride-N is sufficient for ≤5 hosts and
  benchmarks under a few thousand tasks.
- **Remote-managed model caches.** `rsync -c` on each invocation is <2 min on
  gigabit for a 7.5 GB .apr; optimizing further is premature.
- **Fault-tolerant shard retry.** If one host dies mid-run, operator re-runs the
  missing shard manually — no automatic reassignment. (Revisit for v1.1 if
  experienced in practice.)

### 12.7 Dogfood Gate + Three-Format Ship (2026-04-18 amendment)

**Problem surfaced during EX-04.** The first-cut `ex-04-upload-hf.sh` called
`uv run --with huggingface-hub python3` instead of our own product. That is the
same failure class as §13.2 cause 7 ("Tooling investment vs tooling usage"):
we have `apr publish`, and we should be shipping through it.

Two product gaps had to be closed before EX-04 could run through `apr publish`:

1. `apr publish` did not natively consume `publish-manifest-v1.yaml` or upload
   arbitrary sidecar files (tokenizer.json, per-format manifests).
2. No contract stated that ships must be published in multiple ecosystem
   formats, and no contract stated the required safetensors dtype.

Both gaps are now closed by **contract `contracts/apr-cli-publish-extra-v1.yaml`
(F-PUBLISH-EXTRA-001)**, a peer of `publish-manifest-v1.yaml` that adds:

- `manifest_upload_roundtrip` — `apr publish --manifest <yaml>` validates, hashes
  the declared artifact locally, and aborts before network I/O on mismatch.
- `extra_file_passthrough` — `apr publish --extra-file <path>` (repeatable) uploads
  sidecars verbatim in CLI-argument order.
- `no_readme_when_manifest` — when `--manifest` is passed, the auto-generated
  `README.md` is suppressed; the manifest is the provenance document.
- `dogfood_shell_script` — `scripts/ship-two-001/ex-04-upload-hf.sh` MUST invoke
  `apr publish`; `uv run`, `huggingface_hub`, `huggingface-cli`, and `pip install`
  are forbidden in the ship script.
- `three_format_preference` — every SHIP-TWO-* release publishes `.apr`,
  `.safetensors`, and `.gguf` side-by-side in the same HF repo.
- `safetensors_dtype_fp16` — ship-bound `.safetensors` MUST be exported via
  `apr export --format safetensors --quantize fp16`. Default-fp32 export doubles
  disk/network cost; the `transformers` / `candle` / HF ecosystem reads fp16
  natively. Expected 7B sizes: `.apr` ≈ 7.5 GB, `.safetensors-fp16` ≈ 14 GB,
  `.gguf` ≈ 7.5 GB (a fp32 safetensors at ≈ 29 GB is forbidden for ships).

Discharged by falsification tests **FALSIFY-PUB-EXTRA-001 through -010**
(contract `apr-cli-publish-extra-v1.yaml` v1.2.0):

- **-001..-004** covered by `apr publish` unit tests
- **-005** dogfood gate (no Python in ship scripts)
- **-006** post-upload sha256 round-trip (discharged by EX-05)
- **-007** three-format HF repo (discharged by EX-05 + list-repo-files)
- **-008** no Python in `ex-05-verify-manifest.sh` (discharged)
- **-009** corrupt-manifest pre-flight abort (shows exit code 5 blocking upload)
- **-010** `preflight_validate_manifest` function present + invoked before any `publish_format`

Additionally, **FALSIFY-PM-007** (safetensors header dtype Poka-Yoke, contract
`publish-manifest-v1.yaml` v1.1.0) fires automatically inside every pre-flight
invocation for `.safetensors` format. Eight unit tests cover both the happy path
and the exact §12.7.2 ship-blocker scenario (`pm007_f32_weight_when_fp16_declared_fails`).

**FALSIFY-PM-008** (GGUF tensor-type Poka-Yoke, contract `publish-manifest-v1.yaml`
v1.2.1, added 2026-04-18) closes the same class for `.gguf` ships. Design pivot
made mid-discharge: the teacher GGUF that had to pass this gate ships with
`general.file_type = 0` (ALL_F32) despite fully Q4_K tensors — a known llama.cpp
quantize-tool bug. PM-008 therefore treats the **predominant non-float GGML
tensor type** from the tensor_metadata section as authoritative and the
metadata_kv ftype as an advisory fallback (used only when tensor metadata is
absent, e.g. for synthetic fixtures). 15 unit tests, including the real-teacher
scenario (`pm008_q4_k_tensors_override_stale_ftype_zero`) and the "wrong file
pointed at" scenario (`pm008_tensor_type_mismatch_fails`).

**FALSIFY-PM-009** (APR magic-bytes Poka-Yoke, contract `publish-manifest-v1.yaml`
v1.3.0, added 2026-04-18) closes the three-format ship symmetry. With PM-007
covering `.safetensors` and PM-008 covering `.gguf`, PM-009 ensures `.apr` ships
can't pass pre-flight with a mis-staged artifact. v1.0 scope = first 4 bytes
match one of `APR\0` / `APRN` / `APR1` / `APR2` (the four APR magic variants
recognised by `crates/aprender-registry/src/format.rs::parse_apr_header`). The
exact ship-blocker this catches is "GGUF file renamed `.apr` and staged under
format=apr manifest" — covered explicitly by
`pm009_gguf_magic_staged_as_apr_fails`. Dogfooded against the real 8 GiB
teacher APR: verdict PASS (`apr magic = APR\0 (v2) (valid)`). Expansion to
tensor-index quant validation is deferred to v1.1 until a real-world FAIL
demonstrates need.

The unit test matrix (`cargo test -p apr-cli validate_manifest`) runs 45 tests on
every push; the end-to-end pre-flight gate runs against real 8–15 GiB artifacts
only at ship time. All three staged teacher artifacts (`.apr` 8.0 GiB,
`.safetensors` 15.2 GiB, `.gguf` 8.0 GiB) discharged every applicable gate on
2026-04-18, with overall verdict **PASS** per format. Evidence:
`evidence/ship-two-001/ex-04-preflight-gate-smoketest-v2.json` (9-gate
coverage; supersedes v1 which only captured PM-001..007).

#### 12.7.1 Revised EX-04 invocation

EX-04 is now **one command per format**, pointed at a per-format manifest in
`contracts/publish-manifests/`:

```
apr publish /mnt/nvme-raid0/models/ship-two-001/ \
    paiml/qwen2.5-coder-7b-apache-q4k-v1 \
    --manifest contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1-apr.yaml \
    --extra-file /mnt/nvme-raid0/models/ship-two-001/tokenizer.json
```

and repeats for `-safetensors.yaml` and `-gguf.yaml`. Each invocation runs the
pre-flight sha256 guard *before* opening any network socket, then uploads the
artifact + tokenizer + manifest.yaml.

#### 12.7.2 What falsifies the dogfood gate

| Condition                                                                                   | Action                                                  |
|---------------------------------------------------------------------------------------------|---------------------------------------------------------|
| `ex-04-upload-hf.sh` contains `uv run` / `huggingface_hub` / `huggingface-cli` / `pip install` | FALSIFY-PUB-EXTRA-005 FAIL — fix script, rerun           |
| `ex-05-verify-manifest.sh` contains `uv run` / `python3` / `pip` / `huggingface_hub`        | FALSIFY-PUB-EXTRA-008 FAIL — ex-05 must use `apr validate-manifest --live` |
| HF repo missing any of `.apr` / `.safetensors` / `.gguf` after EX-04                        | FALSIFY-PUB-EXTRA-007 FAIL — re-upload missing format    |
| Staged `.safetensors` header declares F32 for weight tensors when manifest says fp16        | **FALSIFY-PM-007 FAIL — pre-flight gate aborts with exit 2 BEFORE any network I/O; re-export with `--quantize fp16`** |
| Staged `.gguf`'s predominant GGML tensor type disagrees with manifest quantization (e.g. manifest says `q4_k` but tensors are predominantly `Q6_K`) | **FALSIFY-PM-008 FAIL — pre-flight gate aborts with exit 2 BEFORE any network I/O; correct the manifest or re-quantize.** (Note: stale `general.file_type=0` does NOT trigger FAIL — it is surfaced as a diagnostic note.) |
| Staged `.apr` file's first 4 magic bytes are not one of `APR\0` / `APRN` / `APR1` / `APR2` (e.g. a GGUF file renamed `.apr`, or a stray `.safetensors`) | **FALSIFY-PM-009 FAIL — pre-flight gate aborts with exit 2 BEFORE any network I/O; restage the correct `.apr` artifact.** |
| Staged artifact's local sha256 ≠ per-format manifest sha256 at ship time                    | **FALSIFY-PUB-EXTRA-009 FAIL — pre-flight gate aborts with exit 5 BEFORE any network I/O** |
| `preflight_validate_manifest` removed or reordered after `publish_format`                   | FALSIFY-PUB-EXTRA-010 FAIL — Poka-Yoke bypassed; re-sequence |
| Any uploaded artifact's CDN-served sha256 ≠ per-format manifest sha256                      | FALSIFY-PUB-EXTRA-006 FAIL (post-upload) — investigate transit corruption |

**Ship-time Poka-Yoke:** prior to contract v1.2.0 (2026-04-18), the dtype mismatch
row above required post-hoc detection and a deprecation cycle on HF Hub. With
PM-007 + the pre-flight gate, it is structurally unreachable: an ex-04 invocation
with divergent bytes exits non-zero before the first HTTP connection opens.

---

### 12.8 Large-File Upload via Xet (2026-04-18 amendment — v2.8.0)

**Trigger:** a real EX-04 upload run with live `HF_TOKEN` (commit
`ec60b5c9e`, `--features cuda`) surfaced that every SHIP-TWO-001 teacher
artifact exceeds HF Hub's 5 GiB HTTP preupload threshold:

| Format         | Size     |
|----------------|----------|
| `.apr`         | 8.0 GiB  |
| `.gguf`        | 8.0 GiB  |
| `.safetensors` | 15.2 GiB |

HF Hub's `preupload/main` endpoint returned `200 OK` with `uploadMode:
"lfs"` but **both** `upload_url` and `chunk_urls` empty. Our upload
path (`crates/aprender-core/src/hf_hub/upload.rs:283 —
reject_oversized_file`) hard-aborts in that state. Five Whys evidence
at `evidence/ship-two-001/ex-04-five-whys-lfs-5gb-blocker.md`.

#### 12.8.1 Rejected paths (for the record)

| Option                                    | Why rejected                                                   |
|-------------------------------------------|----------------------------------------------------------------|
| A) `apr export --max-shard-size` sharding | **Workaround**, not a fix. Only helps `.safetensors`; `.apr` and `.gguf` lack native sharding conventions; loses single-file UX. |
| B) LFS batch API only                     | Pulls git-lfs subprocess / reimplements legacy protocol. HF has moved to Xet; LFS batch is legacy/fallback, not the current path. |
| C) Self-hosted S3 bucket                  | **Not sovereign** — still AWS-dependent. Decouples us from HF Hub discovery and breaks AC-SHIP1-006 (`apr pull` from HF). |
| D) Respec to a smaller parent model       | Q4_K of 7 B is already near the practical floor for coder-quality; changing parent is out of scope for SHIP-TWO-001. |
| E) Ship fewer formats                     | Violates `three_format_preference` equation in `apr-cli-publish-extra-v1.yaml`. |

The real fix is the real protocol: **Xet**, HF Hub's current
content-addressable storage backend for large files.

#### 12.8.2 The Xet protocol (normative summary)

Source of truth: [huggingface.co/docs/xet/index v1.0.0](https://huggingface.co/docs/xet/index).
Reference Rust impl: [github.com/huggingface/xet-core](https://github.com/huggingface/xet-core)
(Apache-2.0, v1.4.3 as of 2026-03-31). Crates on crates.io: `hf-xet`,
`xet-client`, `xet-data`, `xet-core-structures`, `xet-runtime`.

**Upload lifecycle** (MUST be performed in order):

1. **Token acquisition** —
   `GET https://huggingface.co/api/models/{repo_id}/xet-write-token/{revision}`
   with `Authorization: Bearer ${HF_TOKEN}`. Response:
   `{ accessToken, exp (unix seconds), casUrl }`. Refresh at
   `exp - 30s`.
2. **Chunking** — content-defined (gearhash) with 8 KiB min /
   ~64 KiB avg / 128 KiB max. Exception: last chunk of a file MAY
   be smaller than min.
3. **Deduplication** (OPTIONAL) —
   `GET ${casUrl}/v1/chunks/default-merkledb/{chunk_hash_hex}`.
4. **Xorb formation** — group chunks into xorbs, each ≤ 64 MiB
   serialized, avg ~1024 chunks. Hash via xet-core
   `xorb_hashing` procedure.
5. **Xorb upload** —
   `POST ${casUrl}/v1/xorbs/default/{xorb_hash_hex}` with
   `Authorization: Bearer ${accessToken}`, body
   `application/octet-stream`. Response: `{ was_inserted: bool }`.
   `was_inserted:false` is SUCCESS (idempotent replay).
6. **Shard assembly** — one shard references one or more xorbs
   plus file reconstructions. Shard ≤ 64 MiB. All referenced xorbs
   MUST already be uploaded (strict happens-before).
7. **Shard upload** — `POST ${casUrl}/v1/shards`. Response
   `{ result: 0|1 }`; both values are SUCCESS.
8. **LFS pointer commit** — `POST https://huggingface.co/api/models/{repo_id}/commit/{revision}`
   with an LFS pointer file (oid sha256 = sha256(file), size =
   bytes). Without this step the bytes are safe in CAS but the
   repo file tree does not show them.

**Hash-string encoding rule (CRITICAL)** — URLs embed 32-byte hashes
as 64 hex chars, but NOT naive hex. For each 8-byte block, reverse
bytes within the block, then concatenate hex. Equivalent to reading
each 8-byte block as a little-endian u64 and printing as 16 hex
chars. Naive hex triggers 400 Bad Request. `MerkleHash::to_string()`
in xet-core does this correctly; direct `hex::encode` is FORBIDDEN.

**Retry taxonomy:**
- RETRYABLE (exp. backoff, Retry-After on 429): 429, 500, 503, 504,
  connection-level errors.
- NON-RETRYABLE (abort immediately): 400, 403, 404, 416.
- 401 = refresh token once, then abort.

#### 12.8.3 Contract and Falsification Set

Contract file: `contracts/apr-publish-hf-large-file-v1.yaml` v1.1.1
(status `IMPLEMENTED` as of 2026-04-18, commit `18fd9536e`; evidence
fields added in v1.1.1 at commit `671535b44`). Ten falsifiable gates:

| Gate                      | What it falsifies                                                              |
|---------------------------|--------------------------------------------------------------------------------|
| FALSIFY-PUB-LFS-001       | File-size dispatch: > 5 GiB routes to Xet, not `reject_oversized_file()`.     |
| FALSIFY-PUB-LFS-002       | Xet token acquisition URL template + header + JSON response parsing.          |
| FALSIFY-PUB-LFS-003       | Chunk size bounds (8 KiB ≤ len ≤ 128 KiB) except last chunk.                 |
| FALSIFY-PUB-LFS-004       | Xorb size ≤ 64 MiB serialized.                                                |
| FALSIFY-PUB-LFS-005       | Strict shard-after-xorbs ordering (all referenced xorbs 2xx before shard).    |
| FALSIFY-PUB-LFS-006       | Content-addressable idempotency (`was_inserted:false` and `result:0` = OK).   |
| FALSIFY-PUB-LFS-007       | Retry policy matches Xet error taxonomy.                                      |
| FALSIFY-PUB-LFS-008       | Hash-in-URL uses 8-byte-reversed hex, not naive hex.                          |
| FALSIFY-PUB-LFS-009       | LFS pointer git commit uses one-pass sha256 + size from the Xet upload.       |
| FALSIFY-PUB-LFS-010       | Three-format real dogfood (8-15 GiB each) round-trips via `apr publish` only. |

#### 12.8.4 Implementation (shipped 2026-04-18, commit `18fd9536e`)

Actual wiring diverged from the v1.0.0 plan in two ways: (i) `hf-xet`
1.5.1 exposes a *blocking* API (`build_blocking`,
`upload_from_path_blocking`, `commit_blocking`), which obviates the
planned tokio↔sync bridge (step 3 below, deleted); (ii) phases 3–7
of the Xet protocol are fully internal to `hf-xet`, so the four-file
`xet/` module tree anticipated in v1.0.0 collapses to a single
178-line `xet.rs`. See
`contracts/apr-publish-hf-large-file-v1.yaml` v1.1.0 changelog for
the v1.0.0→v1.1.0 delta.

1. **Dependency surface** — ADDED `hf-xet = "1.5.1"` (Apache-2.0) to
   `[workspace.dependencies]` plus
   `hf-xet = { workspace = true, optional = true }` in
   `crates/aprender-core/Cargo.toml`. NEW `xet` sub-feature:
   `xet = ["hf-hub-integration", "hf-xet"]`. `apr-cli` forwards it
   via `xet = ["hf-hub", "aprender/xet"]`. Default `cargo install
   aprender` footprint unchanged (xet off by default; adds ~4 MB
   when enabled).
2. **Dispatch site** — DELETED
   `crates/aprender-core/src/hf_hub/upload.rs::reject_oversized_file`.
   ADDED `upload_via_xet` (tempfile materialize + `XetUploader`
   invoke) and `reject_needs_xet_feature` (clear error when built
   without `--features xet`). Dispatch gate in `upload_via_lfs`
   routes files > 5 GiB through `super::super::xet::should_use_xet`.
   The < 5 GiB HTTP-LFS path is untouched.
3. **Sync call surface** — `hf-xet` provides `*_blocking` variants,
   so we call them directly from the sync CLI path. No tokio
   runtime spawned in `apr publish`.
4. **Error surface** — ADDED `HfHubError::XetUpload(String)` and
   `HfHubError::PartialUpload { cas_success: bool,
   commit_success: bool, detail: String }`. Partial-upload splits
   "CAS xorbs landed but LFS pointer commit failed" from "nothing
   happened" — consumed by retry UX.
5. **Dogfood** — live upload still pending `HF_TOKEN`. Gate evidence
   paths for the live upload remain:
   `evidence/ship-two-001/ex-04-xet-upload.log` +
   `evidence/ship-two-001/ex-04-xet-verify.json`. Pre-live evidence
   already captured in two files:
   (a) Static wiring proof at
   `evidence/ship-two-001/ex-04-xet-phase2-wiring.json` (commit
   `ee6382803`) — `strings(apr)` confirms the full `hf-xet` 1.5.1
   runtime is linked into the canonical binary.
   (b) Live-on-teacher dry-run at
   `evidence/ship-two-001/ex-04-xet-dryrun-teacher.{json,txt}`
   (commit `18f8b5604`) — all three real SHIP-TWO-001 teacher
   artifacts (.apr 8.0 GiB / .gguf 8.0 GiB / .safetensors 15.2 GiB)
   route to the Xet CAS path under the canonical
   `/mnt/nvme-raid0/targets/aprender/release/apr` (features
   `cuda,xet`). This discharges FALSIFY-PUB-LFS-001 against real
   teacher sizes, not synthetic fixtures.

Actual edit sites (see `contracts/apr-publish-hf-large-file-v1.yaml`
`implementation_plan.edit_sites` for the authoritative list):

```
Cargo.toml                                      (+ hf-xet = "1.5.1")
crates/aprender-core/
├── Cargo.toml                                  (+ optional hf-xet dep, + xet feature)
└── src/hf_hub/
    ├── mod.rs                                  (+ pub mod xet; + XetUpload / PartialUpload variants)
    ├── upload.rs                               (- reject_oversized_file
    │                                            + upload_via_xet
    │                                            + reject_needs_xet_feature
    │                                            ~ upload_via_lfs dispatch)
    └── xet.rs                                  (NEW, 178 lines)
crates/apr-cli/
└── Cargo.toml                                  (+ xet feature forwarder; + xet in `full`)
```

Known Phase 3 follow-up (non-blocking): `push_to_hub` still takes
`&[u8]`, so `upload_via_xet` materializes bytes to a tempfile
before invoking `upload_from_path_blocking`. Threading `&Path`
through the upload stack eliminates the round-trip; tracked for a
follow-up contract amendment.

#### 12.8.5 Sovereignty position

The Sovereign AI Stack ships models **through** HF Hub (discovery
convenience) without **depending on** HF Hub (bytes are also
mirrored via `artifact_url_mirror` in every manifest, per
`publish-manifest-v1.yaml` §4.3). Xet-based upload does not
compromise sovereignty: we publish to the Hub via the Hub's own
public protocol, and the manifest links to an independent mirror
whose bytes match by sha256. Loss of HF Hub availability degrades
discovery, not operation.

#### 12.8.6 What falsifies the v2.8 amendment (v2.8.0 + v2.8.1)

| Event                                                                                   | Falsification verdict                                                        |
|-----------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| EX-04 succeeds via any path **other than** `apr publish`'s Xet code (e.g., `hf upload`) | §12.8 failed: we took a workaround, not the contract-mandated path.          |
| Any one of the 3 real 8-15 GiB artifacts does not round-trip by sha256                  | FALSIFY-PUB-LFS-010 FAIL — ship blocked; investigate CAS corruption or LFS pointer drift. |
| `reject_oversized_file` remains reachable in production code                            | FALSIFY-PUB-LFS-001 FAIL — code delete incomplete. (Already verified deleted at `18fd9536e`.) |
| Default `cargo install aprender` binary size regresses > 20 %                           | Feature gating broken; re-architect to push xet into a separate crate. (xet is off by default — `cargo install aprender` does NOT pull `hf-xet`.) |
| `cargo test -p aprender-core --features xet --lib hf_hub` fails on any of the 4 PUB-LFS-001/002 unit tests | Regression in dispatch-gate or token-URL builder. Phase 2 static proof void. |

Failure here is recoverable and distinct from §12.5/§12.7 failures:
a bug in the Xet path can be fixed by shipping an aprender patch
release without redoing training or re-evaluating the teacher.

---

## 13. Why Did This Take So Long? (Retrospective)

### 13.1 Timeline

- **2026-01-01 → 2026-04-17:** 3.5 months of work on the Sovereign AI Stack.
- 2141 commits to aprender (of which 181 = 8.5% are perf-path-to-1.5× commits).
- apr-leaderboard ran **12 distillation recipes** (a → l); multiple had broken checkpoint output.
- Commit `0fc5436 fix: LoRA merge was element-wise, not matrix multiply — root cause of distilled
  model garbage` on apr-leaderboard — **this exact failure class has been fixed before**.
- Commit `a20f234 docs: document Q4K roundtrip corruption blocker` — **also previously known**.
- Current distilled-v2 checkpoints are dated 2026-04-03; they sat **14 days** before any
  `apr qa`-driven audit.
- Spec SPEC-SHIP-TWO-001 v1.0.0 was written 2026-04-17, *after* the broken checkpoint had been
  sitting for 2 weeks and cited as "trained, needs packaging."

### 13.2 Root causes

1. **Contract came after POC, not before.** The 87.20% number lived in a recipe comment since
   Q1 and was promoted to a spec headline without a falsification test run. Design Principle 1
   (Contract-first) was violated by its own spec.
2. **`apr qa` gate matrix lets Golden Output fail silently.** A model that cannot generate "4"
   for "2+2=" passed `apr qa` overall because Golden Output was reported but non-blocking. Tensor
   Contract PASS became a false confidence signal. This is the structural defect that allowed
   broken weights to persist for 14 days.
3. **Perf work crowded out ship work.** 181 perf commits toward 1.5× Ollama parity between
   January and April; **zero** commits on publishing a *model* artifact (as opposed to a *crate*).
   Perf gains are visible in benchmarks; ship state was not gated anywhere.
4. **Monorepo reorg (APR-MONO) consumed weeks.** Phases 1–11 moved 70 crates, introduced shim
   layers, debugged publishing cycles — necessary ceremony but directly competed with model-ship
   bandwidth. Commits like "Phase 10d+10e done" / "Phase 11 (CI fix + publish babysit)" show
   sustained multi-week focus.
5. **Distillation recipe churn without shipping discipline.** Recipes a/b/c/d/e/f/g/h/i/j/k/l —
   twelve experiments, each generating a checkpoint — but no contract defining what makes a
   recipe's output "ship-quality." Each recipe was treated as a new chance; no recipe was ever
   contractually retired. `contracts/distillation-pipeline-v1.yaml` (per v1.0.0 §4.4) is listed as
   "NEW" but has never existed — the proof of this is that broken checkpoints sat unaudited for weeks.
6. **Two-model scope from the start.** v1.0.0 bundled distilled (quick) with sovereign
   (multi-week). This guaranteed the multi-week item would block the quick item from any
   expedited ship path. The fix is to ship MODEL-1 alone as a v1 and move MODEL-2 to v2.
7. **Tooling investment vs tooling usage.** `apr qa`, `apr trace`, `apr profile`, `apr diff` all
   exist and are well-built. They were not being *dogfooded on the shipping artifact* until
   2026-04-17. The audit that exposed this was a 10-minute `apr qa` invocation.
8. **"87.20%" was never in a results JSON.** 17 HumanEval result files exist in
   `apr-leaderboard/results/`; all 17 are teacher runs. The student has no recorded result. A
   spec claim not traceable to a results file is an unfalsified claim.

### 13.3 Lessons codified as contracts

| Contract (new)                                  | Prevents future occurrence of                                |
|-------------------------------------------------|--------------------------------------------------------------|
| `contracts/eval-harness-humaneval-v1.yaml` v1.1 | Headline pass@1 numbers without a results-JSON trail         |
| `contracts/publish-manifest-v1.yaml` v1.0       | Artifacts shipping without sha256 / license / provenance     |
| `contracts/publish-manifest-v1.yaml` v1.1 (PM-007) | Uploading `.safetensors` whose header dtype contradicts the manifest |
| `contracts/publish-manifest-v1.yaml` v1.2 (PM-008) | Trusting stale `general.file_type` over per-tensor `ggml_type` histogram for GGUF |
| `contracts/publish-manifest-v1.yaml` v1.3 (PM-009) | Uploading a renamed `.gguf`/`.safetensors` under a `.apr` manifest |
| **APR-QA GATE AMENDMENT (§12.1)**               | Tensor Contract masking a Golden Output failure              |
| `contracts/distillation-pipeline-v1.yaml` (TBD) | Recipes run without per-epoch Golden Output gating           |

### 13.4 Publish-fastest rules going forward

1. **No claim without a JSON.** Every pass@1 number in any spec must be a `jq`-extractable field
   in a file under version control. If it isn't, it doesn't exist.
2. **Golden Output is a ship-blocking gate.** `apr qa` must exit non-zero if Golden Output fails,
   even when all structural gates pass.
3. **Ship-first, proof-second.** Ship the teacher in 10 hours; use the published artifact as the
   reference against which distillation is measured. Do not wait to ship because distillation
   isn't done.
4. **One artifact per release.** v1 = teacher. v1.1 = distilled (if/when it works). v2 =
   sovereign. Coupling them couples their risks.
5. **Dogfood before declaring "trained."** A checkpoint is not "trained" until `apr qa` and
   `apr eval` agree it is. Until then it is "saved."

---

## 11. References

### 11.1 Existing Contracts

- `contracts/model-families/qwen2.yaml` — Qwen2 architecture descriptor
- `contracts/model-families/llama.yaml` — LLaMA architecture descriptor
- `contracts/model-families/_schema.yaml` — family schema validator
- `contracts/tensor-layout-v1.yaml` — row-major APR invariant (LAYOUT-001/002)
- `contracts/chat-templates-v1.yaml` — chat-template engine spec
- `contracts/apr-cli-commands-v1.yaml` — 57-command CLI contract
- `contracts/publish-manifest-v1.yaml` v1.3.0 — artifact-shipping schema (sha256, provenance, license) + **FALSIFY-PM-007** safetensors header dtype Poka-Yoke + **FALSIFY-PM-008** GGUF tensor-type Poka-Yoke (tensor-authoritative; `general.file_type` is advisory fallback) + **FALSIFY-PM-009** APR magic-bytes Poka-Yoke (three-format ship symmetry)
- `contracts/apr-cli-publish-extra-v1.yaml` v1.2.0 — **F-PUBLISH-EXTRA-001** (§12.7): manifest consumption, `--extra-file` passthrough, three-format ship, safetensors fp16 dtype, **preflight_validate_manifest** (FALSIFY-PUB-EXTRA-009/-010)
- `contracts/eval-harness-humaneval-v1.yaml` — pass@1 harness / AC-EX-003 floor
- `contracts/apr-model-qa-v1.yaml` — `apr qa` gate matrix / AC-EX-001/-002 (Golden Output hard-block)
- `contracts/training-loop-pretrain-v1.yaml` v1.4.0 — MODEL-2 training loop (GATE-TRAIN-001..010), peer of the new GPU backend contract below
- `contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 PROPOSED — **§14 (v2.23.0)** task #132 GPU training backend dispatch (INV-GPUTRAIN-001..007, GATE-GPUTRAIN-001..006, FALSIFY-GPUTRAIN-001..007)

### 11.2 Related Specifications

- `docs/specifications/aprender-train/hugging-face-distill-learn-pipeline-spec.md`
- `docs/specifications/aprender-train/comprehensive-qa-falsification.md`
- `docs/specifications/aprender-train/model-eval-framework-spec.md`
- `docs/specifications/aprender-monorepo-consolidation.md`

### 11.3 External

- HumanEval: Chen et al. 2021, *Evaluating Large Language Models Trained on Code*
- Qwen2.5-Coder: Hui et al. 2024, *Qwen2.5-Coder Technical Report*
- The Stack v2: BigCode, CC-BY-4.0

---

## 14. Task #132 — CUDA training backend gap (v2.23.0 amendment, 2026-04-21)

### 14.1 Surface (what broke)

First MODEL-2 from-scratch real-compute dispatch on lambda-labs RTX 4090
at commit `f7ad11408` (post-task-#131 vocab alignment):

- `apr pretrain --mode from-scratch --dataset … --tokenizer …`
- 14 minutes observed runtime
- 114% CPU (single-thread), 0 MiB GPU memory per `nvidia-smi`
- Empty run dir; no step logging; no checkpoints
- Killed after observing no GPU activity

The dispatch accepted flags, printed startup banner, and silently ran on
CPU. No error surfaced because there was no contract binding "operator
asked for GPU" to "training ran on GPU."

### 14.2 Root cause

`crates/aprender-train/src/train/transformer_trainer/trainer.rs:42`:

```rust
impl TransformerTrainer {
    pub fn new(config: TransformerTrainConfig) -> Self {
        let seed_guard = crate::transformer::init::lock_init_seed(config.seed);
        let model = Transformer::new(&config.model_config);
        drop(seed_guard);
        Self::build(model, config)
    }
}
```

`TransformerTrainer::new` takes no `Device`. Everything downstream —
`Transformer`, `AdamW`, autograd tape, `GradScaler` — uses CPU-backed
`aprender::Tensor` (trueno SIMD). The `--features cuda` flag gates
`realizar` inference kernels, **not** `aprender-train` training.

Why this was not caught before task #126:

1. `apr pretrain --synthetic` passes — the synthetic drive path never
   instantiates the real model, so GPU residency was never exercised.
2. Unit tests of the training path explicitly avoid the 370M scale
   (allocating ~5 GB of parameters is too expensive per test). CPU is
   tractable at toy scale, which masks the CPU-only dispatch.
3. Task #119's "real-compute smoke test PASS" on lambda-labs used the
   synthetic drive (or a toy config), not a 370M cold start.

Scale math: 370M × CPU forward+backward ≈ 30–60 s/step → 10 k steps ≈
100 + hours. Impractical. This is what task #126 actually dispatched,
which is why the run sat at 114% CPU with no log output.

### 14.3 Plan agent finding — existing GPU infrastructure

Phase 0 input (Plan agent survey, 2026-04-21):

| Artifact                                                             | Status        | LOC   |
|----------------------------------------------------------------------|---------------|-------|
| `crates/aprender-train/src/train/transformer_trainer/cuda_trainer.rs` | EXISTS        | 3,432 |
| `CudaTransformerTrainer` AdamW + fused CE + gradient clip + pre-warmed kernels | EXISTS | — |
| YAML training-config loader `loader/mod.rs:227`                      | EXISTS — HAS `if use_cuda { CudaTransformerTrainer::… → train_loop_cuda } else { CPU fallback }` | — |
| `apr pretrain` CLI `drive_real` path (`pretrain.rs:230`)              | MISSING — unconditionally calls `TransformerTrainer::new` (CPU) | — |

**The gap is wiring, not kernels.** The YAML-config path dispatches
correctly; the CLI-flag path does not. Task #132 converges them.

### 14.4 Contract (Phase 0 deliverable)

`contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 PROPOSED,
kind: `training-loop`, peer of `training-loop-pretrain-v1.yaml`.

**Invariants:**

| ID                | Rule                                                                       |
|-------------------|----------------------------------------------------------------------------|
| INV-GPUTRAIN-001  | `--device` grammar: `^(cpu\|cuda(:[0-9]\|:1[0-5])?\|auto)$`, reject others |
| INV-GPUTRAIN-002  | No silent CPU fallback when CUDA was explicitly requested                   |
| INV-GPUTRAIN-003  | GPU residency proof: `nvidia-smi` shows `pid == training_pid AND used_memory > 0` within 5 s of step 0 |
| INV-GPUTRAIN-004  | CPU fallback path remains fully functional (peer GATE-TRAIN-001..010 still PASS) |
| INV-GPUTRAIN-005  | 370M step time < 500 ms on RTX 4090 (seq_len=2048, batch=1, sm_89 pre-compiled) |
| INV-GPUTRAIN-006  | Same-device seed reproducibility holds (two `cuda:0` runs at seed=0, `\|Δloss[k]\| ≤ 1e-5`) |
| INV-GPUTRAIN-007  | `apr --version --json` reports `{cuda_feature, cuda_runtime_available, visible_devices[]}` |

**Ship-blocking gates:** GATE-GPUTRAIN-002 (no-silent-fallback) and
GATE-GPUTRAIN-003 (residency proof). Both must land before task #126
re-dispatches.

### 14.5 Implementation plan (5 phases)

| Phase | Deliverables                                                                                                   | Status      |
|-------|----------------------------------------------------------------------------------------------------------------|-------------|
| 0     | `contracts/entrenar/gpu-training-backend-v1.yaml` + this §14 amendment (PROPOSED status)                       | **SHIPPED** (PR #989 `a5fee06b6`) |
| 1     | `Device` enum + `resolve_device()` in `crates/aprender-train/src/train/device.rs` + `--device` CLI flag + SharedTrainer enum extended with `CudaVariant` (NotImplemented stub) + FALSIFY-GPUTRAIN-001/002 | **SHIPPED** — `crates/aprender-train/src/train/device.rs` on main; `--device` parsed by `apr pretrain` through `resolve_device()`; see `apr-cli/src/commands/pretrain.rs` (search `// parse --device BEFORE any trainer allocation`) |
| 2     | Wire `SharedTrainer::CudaVariant` → existing `CudaTransformerTrainer`; mirror `loader/mod.rs:227` dispatch in `drive_real`; `nvidia-smi` residency probe + FALSIFY-GPUTRAIN-003 | **SHIPPED** — `drive_real` now branches `if device.is_cuda() { drive_real_cuda(…) } else { drive_real_cpu(…) }` (`apr-cli/src/commands/pretrain.rs`); `drive_real_cuda` builds `CudaTransformerTrainer` via `build_shared_cuda_trainer`; `#[cfg(not(feature = "cuda"))]` stub surfaces `GATE-GPUTRAIN-002` byte-for-byte as the contract mandates. `CudaTransformerTrainer` is 3 432 LOC on main. `nvidia-smi` residency probe (FALSIFY-GPUTRAIN-003) remains an evidence-file gate discharged in Phase 3 |
| 3     | Lambda-labs re-dispatch: `apr pretrain --mode from-scratch --device cuda:0 --num-steps 50 --json` produces `evidence/task-132/rtx4090-370m-step-budget.json` with median step-wall < 500 ms; GATE-GPUTRAIN-001..006 all `verdict: pass` | **OPEN** — `evidence/task-132/` does not yet exist; this is the sole remaining code-path-touching blocker for AC-SHIP2-003/004 and for full discharge of SHIP-016/017/018/020 |
| 4     | Promote `gpu-training-backend-v1.yaml` PROPOSED → ACTIVE; spec v2.23.0 → next records promotion; MEMORY.md pointer for task #132 flipped to CLOSED | **OPEN** — trivial once Phase 3 evidence lands |

Total estimate: **~2 days remaining** (Phase 3 dispatch + Phase 4
promotion), down from initial multi-week scope because Phases 0–2
have landed on main. The `CudaTransformerTrainer` existed already
and the CLI dispatch gap is now closed. Audited 2026-04-23 against
commit `601c0740f` (main).

### 14.6 Critical path DAG

Task #131 (vocab bump) CLOSED at `f7ad11408`. Previous DAG claimed
task #126 was ready; the lambda-labs dispatch falsified that claim.
Updated DAG:

```
#118 BPE train 50_257  ──► #131 vocab align  ──► ( #126 blocked by #132 Phase 3 )
                                                        │
                                                        ▼
                                            #132 Phase 0 ✓ SHIPPED (contract + spec)
                                                        │
                                                        ▼
                                            #132 Phase 1 ✓ SHIPPED (device enum + CLI flag)
                                                        │
                                                        ▼
                                            #132 Phase 2 ✓ SHIPPED (wire CudaTransformerTrainer)
                                                        │
                                                        ▼
                                            #132 Phase 3 (RTX 4090 evidence — OPEN)
                                                        │
                                                        ▼
                                            #132 Phase 4 (PROPOSED → ACTIVE — OPEN)
                                                        │
                                                        ▼
                                                    #126 re-dispatches
                                                        │
                                                        ▼
                                                AC-SHIP2-003 (target_val_loss ≤ 3.0)
```

### 14.7 Risks + mitigations

| Risk                                                                 | Mitigation                                                                                   |
|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| CudaTransformerTrainer API drift since last exercise                 | Phase 1 adds FALSIFY-GPUTRAIN-006 same-device seed-reproducibility test — exercises full forward/backward/AdamW cycle before Phase 2 wires drive_real |
| `--features cuda` footgun (memory/feedback_cuda_feature_footgun.md)  | INV-GPUTRAIN-007 + GATE-GPUTRAIN-006 — `apr --version --json` must distinguish build-time feature from runtime availability |
| Seed plumbing broken across device-dispatch layer                    | INV-GPUTRAIN-006 explicit counter-test; `lock_init_seed` mutex stays in place                |
| Test cost for 370M × CUDA in unit tests                              | Keep INV-GPUTRAIN-005 as an evidence-file gate (JSONL from lambda-labs), not a unit test     |
| CPU path regression during refactor                                  | INV-GPUTRAIN-004 + GATE-GPUTRAIN-005 — peer-contract GATE-TRAIN-001..010 must still PASS on `--device cpu` |

### 14.8 Toyota Way — Five Whys

1. **Why** did task #126 burn 14 minutes of compute? — The run was CPU-only.
2. **Why** was the run CPU-only when the operator wanted GPU? — The CLI
   path never selected CUDA.
3. **Why** didn't the CLI select CUDA? — `TransformerTrainer::new` takes
   no `Device` and `drive_real` unconditionally constructs it.
4. **Why** was a CPU-only constructor accepted for a training CLI that
   advertises `--features cuda`? — No contract bound "requested device"
   to "actual device" at ship time.
5. **Why** was there no such contract? — The YAML-config loader has
   correct dispatch; no one noticed the CLI-flag path diverged. This
   contract (§14.4) closes that loop so the two paths converge on the
   same invariants.

**Lesson codified:** `contracts/entrenar/gpu-training-backend-v1.yaml`
GATE-GPUTRAIN-002 (ship-blocking: no silent CPU fallback when CUDA
requested) — prevents future occurrence.

---

## Appendix A: Amendment timeline

This appendix consolidates the per-version amendment dates and
one-line summaries. For full amendment prose, see the individual
version blocks in §4–§14; the YAML Status block at the top of this
spec is the machine-parseable current state. All claims here are
descriptive, not normative.

| Version  | Date       | Summary                                                                                      |
|----------|------------|----------------------------------------------------------------------------------------------|
| v1.0.0   | 2026-04-17 | Initial two-model ship plan (MODEL-1 distilled + MODEL-2 sovereign)                          |
| v2.0.0   | 2026-04-17 | Audit + pivot: distilled student falsified on Golden Output; teacher-first ship              |
| v2.5.0   | 2026-04-18 | Pre-flight Poka-Yoke (PM-001..007) wired into `apr publish`                                  |
| v2.6.0   | 2026-04-18 | PM-008 GGUF tensor-type Poka-Yoke (`general.file_type` advisory)                             |
| v2.7.0   | 2026-04-18 | PM-009 APR magic-bytes Poka-Yoke (three-format ship symmetry)                                |
| v2.8.0   | 2026-04-18 | HF Hub Xet large-file upload contract (`.apr`/`.gguf` > 5 GiB)                               |
| v2.8.1   | 2026-04-18 | Xet impl landed (`hf-xet = "1.5.1"` behind `xet` feature)                                    |
| v2.9.0   | 2026-04-18 | EX-04 DISCHARGED via NDJSON `lfsFile` schema (clobber + silent-no-op fix)                    |
| v2.10.0  | 2026-04-18 | MODEL-1 v2 QLoRA divergence root-caused — teacher-only ship                                  |
| v2.11.0  | 2026-04-18 | EX-05/06/07 DISCHARGED — teacher tagged SHIP-TWO-001-MODEL-1-TEACHER                         |
| v2.12.0  | 2026-04-18 | Post-ship artifacts — MODEL-2 contracts + MODEL-1 retry plan + SHARD-003 probe               |
| v2.13.0  | 2026-04-18 | FALSIFY-SHARD-003 DISCHARGED live yoga vs gx10                                               |
| v2.14.0  | 2026-04-18 | MODEL-2 dataset contract drafted + BPE NFC gap identified                                    |
| v2.15.0  | 2026-04-18 | MODEL-2 scaffold LANDED — BPE NFC + tokenizer CLI + corpus ingest binary                     |
| v2.16.0  | 2026-04-18 | Zero-Tolerance design principle codified (§3 row #8)                                         |
| v2.17.0  | 2026-04-18 | Contracts schema harmonization shipped — `pv validate` works across all 760 contracts        |
| v2.18.0  | 2026-04-18 | Parallel dispatch lanes #102/#103/#104 all closed                                            |
| v2.19.0  | 2026-04-18 | MODEL-2 pretrain loop driver landed via task #105; loader hardened via task #108             |
| v2.20.0  | 2026-04-19 | FALSIFY-SHIP-021 + FALSIFY-SHIP-022 DISCHARGED — seed-repro + apr inspect provenance         |
| v2.21.0  | 2026-04-19 | FALSIFY-SHIP-011 DISCHARGED + FALSIFY-SHIP-012/015 PARTIAL_ALGORITHM_LEVEL                   |
| v2.22.0  | 2026-04-19 | FALSIFY-SHIP-019 PARTIAL_ALGORITHM_LEVEL via `layout_contract.rs` reuse                      |
| v2.23.0  | 2026-04-21 | Task #132 CUDA training backend gap surfaced; gpu-training-backend-v1 PROPOSED               |
| v2.24.0  | 2026-04-22 | FALSIFY-SHIP-008 PARTIAL_ALGORITHM_LEVEL (chat-template render gate)                         |
| v2.25.0  | 2026-04-22 | FALSIFY-SHIP-006 PARTIAL_ALGORITHM_LEVEL (apr qa 8-gate aggregate)                           |
| v2.26.0  | 2026-04-22 | FALSIFY-SHIP-002 PARTIAL_ALGORITHM_LEVEL (`def fib(n):` Python syntax)                       |
| v2.27.0  | 2026-04-22 | FALSIFY-SHIP-005 PARTIAL_ALGORITHM_LEVEL (HumanEval pass@1 ≥86%)                             |
| v2.28.0  | 2026-04-23 | FALSIFY-SHIP-010 PARTIAL_ALGORITHM_LEVEL (artifact URL + SHA-256)                            |
| v2.29.0  | 2026-04-23 | FALSIFY-SHIP-007 PARTIAL_ALGORITHM_LEVEL (apr bench ≥30 tok/s)                               |
| v2.29.1  | 2026-04-23 | Spec-vs-main audit correction — MODEL-1 6/10 on main (not 7/10); on-main/PR/stacked columns  |
| v2.29.2  | 2026-04-23 | §5.4 gains crate/call-graph; new §5.5 documents material divergence between monorepo `Llama370MConfig` and live albor v29 training config; PMAT-685 reconciliation decision required |
| v2.29.3  | 2026-04-23 | PMAT-685 CLOSED with Option B (albor aligns to monorepo); **monorepo-single-source-of-truth policy ratified**; §5.5.1/.2/.3 add decision + albor action list (PMAT-687..694) + enforcement plan |
| v2.29.4  | 2026-04-23 | §2.3 research catalog (../unsloth /vllm /pytorch /candle) + §3 row #10 **fix root causes, never route around** + §5.6 multi-backend selection policy (PTX/cuBLAS/WGPU, fastest wins) + new AC-SHIP2-013 gx10 pretrain gate (PMAT-696); trueno#200 elevated from workaround to fix-required bug |

**Note on amendment density:** 28 amendments over 6 days (2026-04-17
to 2026-04-23) is a high rate. Most amendments record discharge of
a single PARTIAL or DISCHARGED gate. The audit at
`docs/specifications/aprender-train/ship-two-models-spec-audit.md`
identifies the structural debt the amendment cadence accrued and
proposes moving per-AC status into a single machine-parseable source
of truth (PMAT-683 / `cargo xtask audit-ship-two`).

---

**END OF SPECIFICATION**
