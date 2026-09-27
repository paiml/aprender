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
| F5 | fold/rc1-F5-flow003 | `adc8356100` | green. The serve lib test was killed by the memcap, but serve/src is unchanged | 36322092872 **RED, fold defects**: check_nextest_ci_profile self-test rows 22/23 rc=2 (table-form `retries = { count = N }`, a2); check_release_scripts_derive_identity R3 on scripts/release/queue_inputs.sh, a literal count (89); bashrs DET002/SEC010. Also workspace-test not done within 12000s (shard starvation) |
| F6 | fold/rc1-F6-gpu-correctness-a | `8a45a50e6c` | green except PRE-EXISTING thread_config | – |
| F9 | fold/rc1-F9-serve-2706 | `4897c0890f` | fmt and clippy fixed (`;` at tokenizer.rs:338); 2815 tokenizer test ok | – |
| F11 | fold/rc1-F11-e9 | `75e79f0733` | green (core lib 14326/0) | – |
| F12 | fold/rc1-F12-2378 | `9ba41b3f18` | fix/2378 cherry-picked + cb/2378 merged; PRE-EXISTING thread_config only | – |
| S1 | fold/rc1-S1-3761 | `0c6aec0da2` | green except PRE-EXISTING thread_config | – |
| S2 | fold/rc1-S2-4153 | `13746adedb` | clippy clean on core/serve/cli; PRE-EXISTING thread_config only | – |
| S3 | fold/rc1-S3-parity-receipt | `fa6289e784` | green (contracts.nt regenerated) | 36322094539: determinism RED on gx10-build (13:50→14:51Z, determinism-compare at `timeout (-1803s)`). Same budget exhaustion as S5; runner time, not the fold. Also x86-main: roadmap_diff_additive + roadmap_fragment_required [run] FAIL (a dispatched run has no PR base; check this on the PR) and workspace-test >12000s |
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

---

# Part 2 — post-LIFT fold plan for the OPEN PRs (aprender-3d, 2026-09-27 17:16Z)

Scope: the 21 open PRs against a cap of 10. Part 1 above drains branches that have **no** PR; this part
drains the PR list itself. Nothing here is executed before LIFT: no new PRs, no closes, no pushes to
anyone's PR branch. Snapshot: origin/main `aca6f2d7f6`, car/0.70.0 `f2f6f8c965`. Every PR head was
fetched as `refs/pull/N/head`.

**Method (re-run these, do not trust the numbers):**
- Patch-id dup check: for each PR, `git rev-list --no-merges origin/main..HEAD`, then per commit
  `git show C | git patch-id --stable`, then count the patch-ids shared between PRs. Also
  `git cherry origin/main HEAD` and `git cherry origin/car/0.70.0 HEAD` (a `-` means the patch is already there).
- Quorum: a ticket quorum is a `docs/audits/quorum-*.json` added by the PR's own diff. A PR receipt is
  `evidence/pr-review/N/<sha>/` on the head. **No open PR has a receipt for its current head.**
  `present` is RED on 20/21: A2 "no receipt directory", or A3/Arm-4 "binds a DIFFERENT diff / no
  diff_patch_id" (signed before #4421). So every PR, folded or not, needs a re-signed receipt on the
  head that actually merges.
- File overlap: `git diff --name-only $(git merge-base origin/main HEAD) HEAD`, then pairwise intersection.

## Excluded (not folded)

| PR | Why |
|---|---|
| #4429 car/0.70.0 | the release car |
| #4502 batch/ont-10 | ONT-10. Carries #4122 EV-5a, which closes on its merge with a receipt (owner 3d, no auto-Closes) |
| #4448 readme-tweak (spudnic) | a contributor PR in its own slot. `present` is GREEN; it is BEHIND, so update the branch, then land |
| #4428 B1r serve/perf | an already-quorumed batch; 59 folds into it. 3 of its commits already have patch-ids on car (`git cherry origin/car/0.70.0` shows `-`); drop them when it is re-cut after car |
| #4431 fold/b3 | **Cop ruling 17:2xZ: not folded or MOVEd, and 59 does not close it.** It lands as its own single after #4502, with no workflow changes (conleche split out, see 57). Keeps the #4197 Kani files (D1) |
| #4459 rex/001 | held under PRM v3 (owner 84). It shares session.rs/session_tests.rs with #4577, and infer/{mod,inference_result}.rs with #4506: rebase it after those land |
| already-quorumed singles | #4506, #4532, #4534, #4535, #4550, #4554, #4576: each carries its own ticket quorum. Folding voids it, so they land alone (section 3) |

## Duplicate / move findings

- **D1 (RESOLVED by cop ruling: #4431 lands as a single after #4502, never closed as moved). Finding kept for the record: #4431 could not be closed as "moved into #4502".** Of its 732 files, 152 differ at #4502 head
  and 21 are absent there. 13 of the absent files are Lean `Challenge/*.lean` files that #4502 removed
  on purpose (66's cross-contract dedup, 16:1xZ). That is forward. **The real residue is #4197 EV-6c
  (Kani assume baseline, 0d):** `crates/aprender-contracts-cli/src/commands/discharge_kani.rs`,
  `crates/aprender-contracts-cli/tests/pvl_kani_assume.rs`, `contracts/kani-assume-baseline.json`,
  `ci/explicit-test-commands.d/464-aprender-contracts-cli-pvl-kani-assume.cmd`, `contracts/witness/bb2825…json`.
  These exist only on #4431, which is why it must land and not close. When #4431 is
  rebased onto #4502 after that lands, the 131 other differing files resolve in the rebase. Take #4502's side for the 13 deduped Challenge/*.lean files. Command:
  `for f in $(git diff --name-only $(git merge-base origin/main A) A); do git cat-file -e B:$f || echo $f; done`
- **D2: #4506 is 50/53 car commits.** It is based on an old car `11bcf6c2ba`. Only 3 commits are its own:
  `36fc26df7f`, `fd5961ad8a`, `4fffd71d7c` (3/3 quorum on `f1810100d8`). After #4429 merges, re-cut it
  fresh from main with only those 3, or it re-carries car.
- **D3: #4429 x #4502 share 199 patch-ids, and #4431 x #4502 share 33.** Expected (stacked batches), so no action.
  No other pair of open PRs shares a patch. `docs/roadmaps/roadmap.yaml` is the only file shared across
  unrelated PRs (#4501/#4503/#4506/#4512/#4534), so resolve it as the union.

## Landing order at LIFT

**1. CI fixes (first; each unblocks the rest)**

| Step | PR(s) | k | Note |
|---|---|---|---|
| 1a | #4512 signing-secret name (4 files) | 1 | Shares ci.yml with #4457, so it lands first. Workflow change: 3/3 agy quorum |
| 1b | FOLD **CI-PR-REVIEW**: #4503 (diff classifier + docs tier) + #4517 (fork attest label) | 2 | Same workstream: they share 7 files (pr-review SKILL.md, ci/sections.yml, contracts/binding.yaml, pr-review-skill-v2.yaml, PR-REVIEW-SKILL-002-v2.md…). **#4503 first**, #4517 rebased onto it. Both unquorumed, so one 3/3 agy quorum covers the fold |
| 1c | #4457 guards run-all (12 files) | 1 | Rebase it after 1a (ci.yml) and give it its own quorum. Kept apart from 1b because it is a different workstream (guards, not pr-review) |

**2. #4448** (spudnic README): update the branch, and it lands on its own green.

**3. Singles, already quorumed.** Order: smallest diff first, and 4554 before 4414 because they share present-cli main.rs.
Each needs a re-signed receipt on the head that merges.

| Order | PR | Files | Note |
|---|---|---|---|
| 3a | #4535 pv proof_status Lean scan | 4 | contracts-only |
| 3b | #4550 serve over-length refusal | 5 | |
| 3c | #4532 qwen35 per-arch logits budget | 5 | touches cuda-nightly.yml (workflow): its quorum must be 3/3 agy |
| 3d | #4576 inspect header-only read | 34 | BEHIND: update the branch |
| 3e | #4554 build reads in-tree contracts | 39 | before #4414 |
| 3f | #4534 apr code tools / Qwen3.5 | 162 | biggest single, last |
| 3g | #4506 F2 dense receipt | 11 | only after #4429 merges, and only the re-cut (D2) |

**4. Folds and singles, not yet quorumed**

| Fold | PRs | k | Note |
|---|---|---|---|
| F-OBS | #4501 OBS-10 loop admission + Part 1 F4 (cb/4493-obs06, bb/4491-obs04; no PR) | ≤3 | same epic, APR-OBS-001. F4's base fold/88-obs00-on-b3 must land or be folded first (Part 1) |
| F-SERVE-BENCH | #4577 qwen35 bench session prefix | 1 | no like-kind open partner; it could ride #4428 B1r (serve/perf). Land it before #4459 (session.rs) |
| S-UPDATE | #4414 update check (58 files) | 1 | has a quorum on the old head `3a75d9b93b`, stale; re-quorum on its head. After 3e |
| S-DOCS | #4533 FLOW-003 v2.1 (1 file) | 1 | docs tier: land it after 1b so the docs-tier quorum applies |

**Projected count:** steps 1–3 close 11 PRs: 1a, 1b ×2, 1c, #4448 and 3a–3f. 3g is a re-cut (close 1, open 1).
That leaves 10: car, #4502, #4428, #4431, #4459, #4506', #4501, #4577, #4414, #4533, which is at the cap, not over it.
Order for #4431: after #4502 merges, rebase it onto main and land it as a single (step 3, before 3g). Section 4 then
adds at most 1 new fold PR (F-OBS) and closes 3 (#4501, #4577, #4414 as they land).
