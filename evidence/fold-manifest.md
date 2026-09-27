# Post-rc.1 drain manifest (2026-09-27, aprender-fb)

The queue drains right after the rc.1 tag. Each fold is at most two READY branches (k≤2)
from the same workstream: one PR, one quorum, one CI run. The survey was taken at 10:40Z against
origin/main `761d6247de` and car/0.70.0. "Merged?" was checked two ways, by ancestry and by
the branch's own files differing from main, because squash merges hide ancestry.

Order: 0.70.0 rows first, then 0.70.1, then 0.71+. Within each fold, land the **first**
branch first. Where the pair shares a file, the second branch is rebased onto the first
before the fold is pushed. No main-merges and no force-pushes: re-cut the branch fresh from
main and cherry-pick.

## Folds (pairs)

| # | Workstream | First | Second | Base fix before fold | Shared files |
|---|---|---|---|---|---|
| F1 | release gates (0.70.0) | d1/4275-4256-on-main `d26a085cf5` (#4275/#4256) | e6/rc-carry-forward-gate `da88d81d17` (rc P0) | e6 is on car, so land it after car merges | none. d1 touches binary-release.yml, a **workflow** that needs the 3/3 agy quorum |
| F2 | ladder | c1/3846-into-b2 `a34b910dea` (#3846) | 98/4520-ladder-box `e0f5f33a62` (#4520) | c1 sits on the pre-merge B2 head: cherry-pick 087c8d7a30 + a34b910dea onto main | scripts/model_ladder.sh: 4520 is rebased on 3846 |
| F3 | prefill perf #4313 (0.70.0) | fix/4313-f2-prob-metric `63f36159ef` | docs/4313-gb10-f2-prefill-finding `c24200634b` | fix/4313 sits on fold/4425-onto-stage-0701: replay onto main | none |
| F4 | APR-OBS-001 | cb/4493-obs06-red-rules `93462c65c6` (#4493) | bb/4491-obs04-selfreport `093b4d6421` (#4491) | both are stacked on fold/88-obs00-on-b3 (31 ahead, **no PR**), which must land first or be folded in as a third car. Ask 88 | README.md, contracts/census.json, contracts/contracts.nt (generated: regenerate, don't hand-merge). cb/4493 also touches ci/sections.yml, and so does #4517 |
| F5 | FLOW-003 | 89/4513-qm01-inputs `69d0ac2c24` (#4513/#4519) | a2/flake0-retries0 `812fde40b1` (#4515/#4516) | none (both on main) | none |
| F6 | gpu-correctness A (m0694) | batch/gpu-correctness-071 `25839c996e` (#3850) | fix/3483-fp8-prefill-parity `4e893a3999` (#3483) | none | none |
| F7 | gpu-correctness B | fix/3111-q5k-gemm `232c198d4f` (#3111, 60) | fix/3759-cuda-module-key-gate `00c2ec721a` (#3759, 41) | none | root Cargo.toml (small) |
| F8 | SRV perf (76) | perf/gdn-recurrence-cuda `d856be9e56` (SRV-TIM-001 #6) | perf/4486-qwen35-decode-graph `49fb4bb4cf` (#4486) | 4486 is on car | gdn_ops.rs. 4486 is findings-only (graph 1–2% slower than eager). 59 owns the fold call |
| F9 | serve epic #2706 (60) | fix/2817-stream-usage `b6b8c9809a` | fix/2815-im-end-after-punct `9adae252d2` | none | none. 2815 is only the guard test (the fix is already on main) |
| F10 | contracts / pv hardening | fix/2530-kani-bounded-array `b7997e69de` (#2530) | df/2556-ratchet-070 `af7c5b92c1` (#2556, closed) | df sits on batch/0.70.0 (PR #4372 closed): replay onto main | aprender-contracts: check on replay |
| F11 | E9 linfa gap (5d) | feat/3149-neighbor-index `d16279dbe2` (#3149) | docs/findings-5d-e9-kmeans-d2 `2d72e0f553` | docs branch is stacked on n_init (12 ahead) | aprender-core: check |
| F12 | #2378 apr run/serve refusals (epic #3997) | fix/2378-apr-run-refusals `ed0d546e6e` (m0694) | df's serve/finetune dequant branch (not pushed yet; pair it when it lands) | 2378 **conflicts with main: re-cut it fresh from main and cherry-pick** (no main-merge). Paired per cop 77 | forward_qwen35.rs with F3: land after F3 |

## Singles (no same-workstream partner; fold alone or ride an open batch)

| Branch | Issue | Note |
|---|---|---|
| 0d/3559-ghosts `a0219a25a5` | #3559 ONT (0.70.0) | on car. contracts.nt overlaps with F4, so land it before F4 regenerates |
| 91/3761-on-main `9ffc76bef2` | #3761 | B1r / #4520 step 2. Could ride #4428 |
| 76/3558-pareto-select `4369a21f01` | #3558 dogfood | apr-cli + scripts/release |
| feat/4496-obs09-trace-serve `8eb949fc82` | #4496 OBS | on car. Shares forward_qwen35_cuda.rs with 4486 (F8), and session.rs with fix/4325 |
| fix/4325-cancel-poll-before-forward `dd9168f4bd` | #4325 | session.rs vs feat/4496 |
| a2/4153-clippy-slice1 `25dd398e6a` | #4153 | 212 files: keep it alone. Shares serve falsification_tests.rs with F5 a2/flake0 |
| 1c/2730-cbtop-vacuous-pass `19ec64f7d9` | #2730 (0.73.0) | |
| 1c/3552-facet-coord-example `c885bc00bb` | #3552 (0.72.0) | root Cargo.toml |
| 49/x86-slow-2 `88449ce44e` | X86-SLOW-2 | on car. 84 owns the speed fold |
| fix/2661-apr-gpu-unified-macos `6911a8e324` | #2661 | |
| fix/3174-canary-not-dark `01da5f2702` | #3174 | **adds a workflow**: 3/3 agy quorum |
| fix/cb200-measured-count `7e7d0783df` | CB-200 | **guards-nightly.yml**: 3/3 agy quorum |
| df/4191-hermetic-root `18039c0fbd` | #4191 | on closed batch/0.70.0: replay |
| fb/4518-tier-base-race | #4518 tier race | ci/sections.yml (CI definition): 3/3 agy quorum. Fixes the "quick quick full" Σ split that would hit rc CI |

## Held / not in the drain

- feat/4435-merge-queue-bk `1c58fcd0d3`: held on a cop ruling.
- 98/ont4g-extract-binary `cb8da39415`: already rides #4502 (batch/ont-10).
- fix/car-069-backports `6be797c204`: 71 ahead, but only 7 files still differ from car. The owner (98) needs to reduce it to the residue before any fold.
- d1/E8-4002-recipe: blocked by its owner (no trainer path).
- Already in car, no action: 49/4507-fat-ws-aside, d1/ont10-s12-trueno-rag-g13, fix/fat-signer-secret-b64, e6/x86-slow-tests.
- Superseded: PMAT-3850-resolve-qtype-refuses (by batch/gpu-correctness-071), fix/3761-header-only-reads-on-main (by 91/3761-on-main).

## Pre-built fold branches (cop 77, 2026-09-27)

Every fold is built off origin/main `761d6247de` and has no PR. CI is started by
`gh workflow run ci.yml --ref <branch>`, three at a time. A dispatched run takes the full tier.
One test, `thread_config::test_auto_config_at_least_half_for_decode`, fails on the gx10 serve lib
suite for main `761d6247de` itself. It is marked PRE-EXISTING below and is not caused by any fold.

| Fold | Branch | Head | Local result | CI run |
|---|---|---|---|---|
| F2a | fold/rc1-F2a-ladder-c1 | `0dcad701ab` | c1/3846 alone: ladder self-test 153/0 | – |
| F3 | fold/rc1-F3-4313-prefill | `2c54d9844d` | fmt fixed. Serve lib: PRE-EXISTING thread_config only. 63f36159ef left out (patches aprender-gpu/src/ptx_patch.rs, which is car-only) | – |
| F5 | fold/rc1-F5-flow003 | `adc8356100` | green. The serve lib test was killed by the memcap, but serve/src is unchanged | 36322092872 |
| F6 | fold/rc1-F6-gpu-correctness-a | `8a45a50e6c` | green except PRE-EXISTING thread_config | – |
| F9 | fold/rc1-F9-serve-2706 | `4897c0890f` | fmt and clippy fixed (`;` at tokenizer.rs:338); 2815 tokenizer test ok | – |
| F11 | fold/rc1-F11-e9 | `75e79f0733` | green (core lib 14326/0) | – |
| F12 | fold/rc1-F12-2378 | `9ba41b3f18` | fix/2378 cherry-picked + cb/2378 merged; PRE-EXISTING thread_config only | – |
| S1 | fold/rc1-S1-3761 | `0c6aec0da2` | green except PRE-EXISTING thread_config | – |
| S2 | fold/rc1-S2-4153 | `13746adedb` | clippy clean on core/serve/cli; PRE-EXISTING thread_config only | – |
| S3 | fold/rc1-S3-parity-receipt | `fa6289e784` | green (contracts.nt regenerated) | 36322094539: determinism RED on gx10-build (13:50→14:51Z, determinism-compare at `timeout (-1803s)`). Same budget exhaustion as S5; runner time, not the fold |
| S4 | fold/rc1-S4-rex-001 | `89832a3cbf` | green on the touched modules (the full serve and cli suites were memcap-killed) | – |
| S5 | fold/rc1-S5-tier-base-race | `f8a1d940ca` | tier self-test 94/0. **CI definition: needs 3/3 agy quorum** | 36322091100: determinism RED on gx10-pool3. The job used its whole budget (13:28→14:05Z) and determinism-compare started with `timeout (-400s)`. That is runner time, not the fold. Other jobs still running at 14:15Z |

**Not pushed. Each needs its owner or a base to land first:**

- **F1** (d1 + e6): e6's commits change scripts/release/rc_cut.sh and rc_fleet_stage.sh, which exist only on car. They apply after car lands. d1 alone merged clean after a binary-release.yml resolution. **workflow**
- **F2 part b** (98/4520 `e0f5f33a62`): with the box re-exec, `check_model_ladder.sh --self-test` goes 3 bad: `producer mutant no-lock / no-choom / unbounded SURVIVED the lock checks`. Owner 98.
- **F4** (OBS pair): blocked on fold/88-obs00-on-b3, which has the witness and contract files. Owner 88.
- **F7** (3111 + 3759): `scripts/cuda_module_key_gate.sh:213 DET002` (`started=$(date -u …)`) fails bashrs-gate. Owner 41.
- **F8** (gdn + 4486): 49fb4bb4cf depends on #4273 split-decode, which is car-only. It applies after car lands.
- **F10** (2530 + df/2556): the ratchet ceiling was measured on batch/0.70.0: `kernel contracts self-exempted by registry: true rose to 513 (ceiling 512)`. Owner df re-measures on main.
- **F13** (0d/3559 + 98 ont4g): the binding.yaml files are car-only and ont4g needs batch/ont-10 files. Both apply after car and ont-10 land.
