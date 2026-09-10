# 06x Release Schedule — 0.67.0 · 0.68.0 · 0.69.0 · 0.70.0 at a 2–3 day cadence, 24/7

**Status:** PROPOSED · ticket `PMAT-1097` · authored 2026-09-10 against `origin/main` @ `2c584a168` (version `0.66.0`, tag `v0.66.0` published 2026-09-10T05:39Z) · contract `contracts/release-schedule-06x-v1.yaml` · drift gate: every backticked repo path in this file is checked by `crates/aprender-core/tests/readme_contract.rs` (FALSIFY-DOCS-CLAUDE-001, extended to this document in the same PR)
**Epics (one per release, so Alfredo can comment on each):** 0.67.0 → #3078 · 0.68.0 → #3079 · 0.69.0 → #3080 · 0.70.0 → #3081. Topic epics that ride these trains: #3062 (NVIDIA CUDA Rust, 0.67), #2873 (PP-066 rows, now 0.68).
**Review chain:** `paiml-implement` (Phase 0 discovery → plan → this document) → `agy /teamwork-preview` review (§10 records every finding, accepted or refuted) → PR.

**Operator's words (verbatim, 2026-09-10):** *"We need to assume a tight release schedule of every 2-3 days working 24/7. i.e. .67.0 .68.0 .69.0. Each release needs to be a GitHub epic so Alfredo can comment."* Priorities, verbatim:

> A. always building binary for CUDA and CPU targets: ARM, x86 with each tagged release (i.e. 4).
> B. supporting the NVIDIA Rust approach 100% and default
> C. Ensure we use gx10 more heavily, it should never be idle and we need to use it more often to unclog the build system and increase CUDA quality and default model story.
> D. We need to use yoga more often for CUDA testing
> E. We need to speed up PR iterations by limiting the testing to at MOST 20 minutes, i.e. reducing the amount of tests run, then forcing the FULL test runs for pre-release dogfood ONLY.
> F. Parity of Qwen 2.5 Coder 7B with ollama and llama.cpp is a priority
> G. in .69 and .70 we offer the SAME easy story for Qwen 2.5 Coder 7B to declarative fine-tune model.

Three P0 five-whys tickets were handed over in the same session with *"(add these to priorities)"*; they are rows 67-A1, 67-C2 and 67-E1 below, filed as #3082, #3083 and #3084 under the 0.67.0 epic #3078 (§5).

**Marks.** `[V]` measured in this session, the command beside it · `[C]` computed from `[V]` inputs, arithmetic shown · `[A]` taken from a receipt, issue or spec as written — re-verify at the implementing HEAD · `[U]` unmeasured; the row names who measures it and how. **The command is the fact; the value is a dated sample.** Comparator ratios and throughput figures are quoted only beside the `evidence/` receipt that carries them (PP-12; `scripts/check_no_claim_literals.sh` scans this directory).

---

## §0. Measured ground truth (2026-09-10, `origin/main` @ `2c584a168`)

| # | Fact | Measured by |
|---|------|-------------|
| G1 | **Cadence history.** Tags: `v0.66.0` 09-10 · `v0.65.2` 09-05 · `v0.65.1` 09-04 · `v0.65.0` 09-04 · `v0.64.0` 08-24 · `v0.63.0` 08-01. The 2–3 day cadence has been sustained before: `v0.56.0` 07-01 → `v0.60.0` 07-06 is four releases in five days | `git for-each-ref --sort=-creatordate --format '%(refname:short) %(creatordate:short)' refs/tags` `[V]` |
| G2 | **PR-time CI is 66 minutes wall.** Run `34449608126` (pull_request, 07:22→08:28Z): `workspace-test` 59 min, of which the step *"Quick tier: every test target that reads the tree (BSE-17)"* is **55 min**; `guard-cargo` 44 min; `guard-tree` 27 min; `mutants` 7 min (starts after workspace-test); `ci / gate` 12 min end-to-end | `gh run view 34449608126 --json jobs` and `gh api repos/paiml/aprender/actions/runs/34449608126/jobs` (step timestamps) `[V]` |
| G3 | **Push-to-main FULL tier is 86 minutes** (`workspace-test`, run `34448893051`); `guard-cargo` 41 min, `guard-tree` 24 min on the same run | `gh run view 34448893051 --json jobs` `[V]` |
| G4 | **The quick tier is serial by construction.** `scripts/tree_reader_tests.txt` lists 41 targets across 26 crates; `.github/workflows/ci.yml` part 2 chains **one `cargo nextest run -p <crate> …` per crate with `&&`** — 26 builds, no shared graph, inside one 60-minute step. #3070 (open) records the 60-minute timeout under fleet load | `git show origin/main:scripts/tree_reader_tests.txt \| grep -vc '^#'` = 41; ci.yml lines 554–600 `[V]` |
| G5 | **Selection cap is 3 crates.** `scripts/gate_touched_crates.sh` `CAP=3`: touching `aprender-core` (or > 3 crates incl. reverse dependents, or a root manifest) sends a PR to the FULL tier. A `merge_group` run goes FULL whenever main moved under the PR (`ci_test_tier.sh`: *"queue tree differs from PR head tree"*) — the second full trip of P0-3 | `grep -n 'CAP=' scripts/gate_touched_crates.sh`; `scripts/ci_test_tier.sh` case table `[V]` |
| G6 | **The clog, in runs per day:** ci.yml runs created 2026-09-08 = **77** (72 pull_request, 3 merge_group); 09-09 = 69 (55 / 8); 09-10 (to 09:00Z) = 19 (8 / 6). Merge throughput = merge_group runs/day = **3–8** | `gh run list --workflow ci.yml --created <day> --json event` `[V]` |
| G7 | **Fleet at 09:00Z:** 16 `intel-clean-room` runners (15 busy, `intel-clean-room-16` is `perf-solo`), `gx10-blackwell` **idle**, `yoga-gpu` **idle**. No lambda runner is registered | `gh api orgs/paiml/actions/runners` `[V]` |
| G8 | **gx10's scheduled load is about one hour a day, and two of its three nightlies are red.** `cuda-nightly.yml` 01:30Z: 10 min, **failure** on *"PP-26 - batch-invariance witness (perf041)"* and the BSE-11 byte-compare (09-09, 09-10); `qwen-story-daily.yml` 04:17Z: 9 min, success; `silicon-nightly.yml` 03:30Z: 8–90 min, **failure** five nights running. **Nothing runs on gx10 at PR time** | `gh run list --workflow <wf> --limit 5`; `gh run view <id> --json jobs` `[V]` |
| G9 | **yoga runs nothing but the release lane.** Its only jobs are `binary-release.yml` `build-apr-cuda` (x86_64) and `smoke-cuda`. Labels `self-hosted,Linux,X64,gpu,cuda,yoga,ada`; runner group `gpu-x86` (id 5) has `allows_public_repositories=false`, `restricted_to_workflows=false`; `paiml/aprender` is public | `grep -n runs-on .github/workflows/*.yml`; `gh api orgs/paiml/actions/runner-groups` `[V]` |
| G10 | **`v0.66.0` carries eight `pv` tarballs and zero `apr` binaries.** The CUDA backfill (dispatch run `34448908554`) built on gx10 (green) and **failed at "Upload assets to release": no `gh` on the box**; `verify-cuda-assets` went RED as designed; the yoga leg was cancelled. PR #3074 (REST upload) is open and red on a `present` check | `gh release view v0.66.0 --json assets`; `gh run view 34448908554 --json jobs` `[V]` |
| G11 | **CPU `apr` binaries exist only in `nightly.yml`**, built `--no-default-features --features inference` on **hosted** `ubuntu-latest`/`macos-latest`/`windows-latest` runners for five targets — never attached to a tagged release (Alfredo's #2869, milestone 0.68.0). The `pv` lane of `binary-release.yml` also runs on `ubuntu-latest`. Both violate the operator's 2026-09-10 rule (no hosted runners) | `git show origin/main:.github/workflows/nightly.yml \| grep -n 'runs-on\|features'` `[V]` |
| G12 | **NVIDIA CUDA Rust (#3062, milestone 0.67.0): the code landed, the paper did not.** T0 → #3063 merged (pulled into 0.66); O2 → #3064 merged 09-09 22:20Z; T3 → #3065 merged (cutile RMSNorm on GB10: parity-correct, **1.03–1.31× slower** than the hand-PTX kernel `[A]` #3062 comment 3). Open and **red**: #3061 (spec, `workspace-test` FAILURE), #3068 (PMAT-1095 fleet readiness, `guard-tree` FAILURE), #3066 (PTX emitter non-determinism: the cubin cache has never hit). T1/T2 deferred to 0.68 by the spec | `gh pr view 3064 3065 --json mergedAt`; `gh pr checks 3061 3068 3074` `[V]` |
| G13 | **cuda-core floor:** builds on stable Rust; refuses at runtime below driver **R580**; `cuda-bindings` build.rs hard-fails without a **CUDA 13.0+ toolkit** on the *build* host. Fleet: gx10 driver 590.48.01 / toolkit **13.3** (default since 2026-09-09, alternatives link); yoga driver **595.91.07** (≥ R580, so cuda-core can *run*) but **nvcc 12.4** (so cuda-bindings cannot *build* there); lambda driver 570 (below the floor) | NVIDIA spec §0 G7–G11, G21–G22 `[A]`; `reference_cuda_ci_runners` probe 2026-09-08 `[A]` |
| G14 | **Milestones:** `0.66.0` open 1 / closed 6 · `0.67.0` open 5 / closed 2 · `0.68.0` open 15 / closed 1 (the PP-066 rows R-0..R-8, C0-*, G-*, KEY, #2869 moved there 2026-09-09) · **no `0.69.0` or `0.70.0` milestone exists**. Open PRs: 33 | `gh api repos/paiml/aprender/milestones`; `gh pr list --state open` `[V]` |
| G15 | **Qwen2.5-Coder-7B correctness parity is proven; throughput parity is not.** GPU-vs-CPU cosine on the 7B ≥ 0.9984 on lambda and gx10 (`evidence/parity/thresholds.yaml`, n=5 each). Against llama.cpp at c=1 on the 4090: decode ratio **0.650**, prefill ratio **0.275** (receipt: `evidence/parity-http/findings.json`, 2026-08-24, apr `53062e7f3`). Every `c>1` band is INVALID-CORRECTNESS (#2753 closed, #2776 merged; the PP-26 witness fails nightly on gx10 at c=16). **No CONFORMANT ledger row exists** (`evidence/parity/LEDGER.md`: every row RECORDED). **No Ollama-vs-7B receipt exists at all** | files named `[V]`; `docs/specifications/0.66-performance-parity-report.md` §1 `[A]` |
| G16 | **Fine-tune surface today:** `apr finetune` has 30 flags in `crates/apr-cli/src/model_ops_commands.rs` (method, rank, data, epochs, gpu-backend, adapters-config, …) and **no recipe/config file**; `apr tune --plan` does memory planning; `apr modelfile parse` (CRUX-K-11) already turns an Ollama Modelfile into apr config JSON — the declarative precedent. Contracts: `contracts/apr-finetune-metrics-v1.yaml`, `contracts/apr-lora-merge-equivalence-beat-v1.yaml`, `contracts/apr-page-cli-finetune-v1.yaml` | `git grep` over `crates/apr-cli/src`; `contracts/apr-cli-commands-v1.yaml` `[V]` |
| G17 | **The daily FULL run is red.** `coverage-nightly.yml` (a full-workspace `cargo llvm-cov` run on the clean-room pool) has **failed five consecutive nights** (2026-09-06..10, 17–58 min, step *"Coverage (cargo llvm-cov via make)"* exit 2). Priority E moves the FULL tier out of PRs; today there is no green full run anywhere but the pre-publish dogfood | `gh run list --workflow coverage-nightly.yml --limit 5` `[V]` |
| G18 | **Required checks:** branch protection `["ci / gate","workspace-test"]`; rulesets *Green Main*, *workspace-test*, *Merge Queue (main)*. The bare `gate` job needs `[ci, workspace-test, mutants, guard-tree, guard-cargo]` | `gh api repos/paiml/aprender/branches/main/protection`; ci.yml `[V]` |
| G19 | **Release mechanics of record:** `scripts/bump-version.sh <v>` (39 files incl. the excluded facades), `CHANGELOG.md`, `scripts/dogfood.sh --phase pre-publish` (FULL gates), tag, `gh release create`, `binary-release.yml` on `release: published`, cascade behind `scripts/check_publish_preflight.sh`, post-publish dogfood. Receipt pattern: `docs/audits/impl-PMAT-1096-receipt.md` | files named `[V]` |
| G20 | **Prior art for F on yoga:** `~/src/qwen-coder-deploy` (outside this repo) deployed realizar, Ollama and llama.cpp side by side on yoga via forjar (`forjar-yoga-{realizr,ollama,llamacpp}.yaml`) with a c=1 / c=4 serial protocol; yoga was its PRIMARY benchmark target | `ls ~/src/qwen-coder-deploy` `[V]` |
| G21 | **Alfredo** is `@alfredodeza`, an admin collaborator and the author of #2869 | `gh api repos/paiml/aprender/collaborators` `[V]` |

---

## §1. The cadence, mechanically

### §1.1 The train rule

A release is a **train**. It leaves in a fixed window; scope that is not on the platform rides the next train. Dates never slip for scope. The only thing that holds a train is a **red release gate** (jidoka), and it holds it for at most **12 hours** while the root cause is fixed — after that the train leaves without the item and the item's row moves to the next release with a `slipped_from:` note in the epic.

| Train | Window opens (T+48h) | Must leave by (T+72h) | Planned tag instant | Theme |
|---|---|---|---|---|
| **0.67.0** | 2026-09-12T06:00Z | 2026-09-13T06:00Z | **2026-09-12T18:00Z** | the release that ships its own binaries, and a 20-minute PR |
| **0.68.0** | 2026-09-14T18:00Z | 2026-09-15T18:00Z | **2026-09-15T06:00Z** | NVIDIA Rust runtime by default; yoga is a CUDA host; signed assets |
| **0.69.0** | 2026-09-17T06:00Z | 2026-09-18T06:00Z | **2026-09-17T18:00Z** | declarative fine-tune lands; batching bands turn VALID |
| **0.70.0** | 2026-09-19T18:00Z | 2026-09-20T18:00Z | **2026-09-20T06:00Z** | the story closes: pull → run → finetune --recipe → run, with published parity |

`T` is the previous train's *actual* tag instant (`gh release view vX.Y.Z --json publishedAt`); the table assumes each train leaves at its planned instant (+60h). A train that leaves early or late shifts every later window by the same amount — recompute, do not re-plan.

### §1.2 Capacity arithmetic (why E is the enabler, not a nice-to-have)

- Merge throughput today is **3–8 merges/day** (G6) → **8–20 PRs per 60-hour train** `[C]`.
- Each PR costs **66 min wall** and roughly **2.2 runner-hours** (G2: 59 + 44 + 27 + 7 min of runner time) `[C]`; 72 PR runs/day (G6) ≈ **158 runner-hours/day** on a 16-runner pool that has 384 — **41 % of the fleet on PR-time tests alone**, before reruns and the merge queue `[C]`.
- At a 20-minute cap with the same job set, the same day costs **≈ 48 runner-hours** `[C]` — the difference is the "unclog" in priority C, and it is what makes 8–20 merges per train achievable with headroom for reruns.
- Release day itself consumes **≈ 4 h** of a 60-hour window: pre-publish dogfood (FULL tier, ≈ 90 min, G3) + tag + `binary-release.yml` (≈ 60 min for six binaries `[U]`, measured on the 0.67.0 run) + cascade + post-publish dogfood on two GPU hosts.

### §1.3 What rides every train (invariants, not scope)

1. `scripts/bump-version.sh X.Y.0` · `CHANGELOG.md` section · pre-publish dogfood **FULL** (`scripts/dogfood.sh --phase pre-publish`, the only place the FULL tier is mandatory — priority E) · tag · `gh release create` · `binary-release.yml` · asset verification (§2 A) · cascade · post-publish dogfood on **gx10 and yoga** (+ intel) with receipts under `evidence/dogfood/X.Y.0/` · a closing comment on the release epic with the receipt path · milestone closed.
2. Every ticket on the train carries `docs/audits/impl-<ticket>-receipt.md`; a release contract row per new gate; every gate mutation-verified RED→GREEN in the same PR (`[[feedback_contracts_ratchet_not_radar]]`).
3. **No hosted runner** in any workflow touched by the train (operator rule 2026-09-10); every `runs-on` names one box.

---

## §2. Priorities A–G — what each one means, measured, and how it lands per train

Every row below has an **acceptance command** (a command, not a sentence), a **host**, and a **gate** that can go RED. Rows are numbered `<train>-<priority><n>` and are the rows of the release epics (§5) and the obligation DAG (§8). *Owed* artifacts (scripts, contracts, workflows that do not exist at `2c584a168`) are written without backticks so the drift gate cannot be asked to check a file this document is asking someone to create.

### A. Four `apr` binaries on every tag — CUDA and CPU × x86_64 and aarch64

**State (G10, G11):** `v0.66.0` has zero `apr` binaries. `binary-release.yml` builds `apr-<tag>-{x86_64,aarch64}-unknown-linux-gnu-cuda.tar.gz` on yoga/gx10 (#3072, merged 07:13Z) and `verify-cuda-assets` fails the workflow when either is missing — the gate works (it went RED on the backfill, G10). CPU builds exist only in the hosted nightly. The operator's D-10 ruling (2026-09-10) makes CUDA assets a hard requirement; this priority makes the set **four**: `{cuda, cpu} × {x86_64, aarch64}`, each with `.sha256`.

| Row | Train | Deliverable | Host | Acceptance command | Gate |
|---|---|---|---|---|---|
| **67-A1** (P0-1, #3082) | 0.67 | `build-apr-cpu` matrix in `.github/workflows/binary-release.yml`: default features, `apr-<tag>-<target>-cpu.tar.gz` + `.sha256`, glibc floor printed to the job summary; `verify-cuda-assets` becomes **verify-apr-assets** and requires all **four** apr assets; a **pre-publish dogfood row** and a **release-criteria row** (C13 of `docs/specifications/PP-066-release-spec.md` §4, narrowed to Linux) so a tag cannot be promoted without them. Owed: scripts/check_release_assets.sh `<tag>` (exit 1 unless 4 apr + 4 sha256 + 8 pv assets; `--selftest` case table with a removed-asset RED row) | x86_64: yoga (or `intel-clean-room`); aarch64: gx10 | `gh release view v0.67.0 --json assets --jq '[.assets[].name] \| map(select(test("^apr-v0.67.0-(x86_64\|aarch64)-unknown-linux-gnu-(cuda\|cpu).tar.gz(.sha256)?$")))\| length'` = 8 | verify-apr-assets RED on a missing asset (observed once on the 0.66.0 backfill); dogfood row |
| 67-A1b | 0.67 | Backfill `v0.66.0` with all four apr assets by `workflow_dispatch -f tag=v0.66.0` once #3074 (REST upload; the gx10/yoga boxes have no `gh`) is merged | gx10, yoga | same query with `v0.66.0` = 8 | — |
| 68-A1 | 0.68 | Signed manifest (R-5 #2908, KEY #3045, S0-19), one-line installer (R-6 #2909), README leads with it (R-7 #2910) — already on milestone 0.68.0; the `pv` lane and the summary/verify jobs move off `ubuntu-latest` onto the clean-room pool; `nightly.yml`'s hosted matrix is deleted or moved (G11) | intel pool | `grep -c 'ubuntu-latest\|macos-latest\|windows-latest' .github/workflows/*.yml` = 0; `scripts/check_multiplatform_dogfood.sh --require-resolved-backend cuda` exit 0 | owed: scripts/check_no_hosted_runners.sh in guard-tree |
| 69-A1 | 0.69 | Post-publish dogfood on **gx10, yoga, intel** installs the release through the installer and records `apr devices --json` (C4) | three hosts | receipts under `evidence/dogfood/0.69.0/` for all three, `backend: cuda` on the two GPU hosts | dogfood C4 row |
| 70-A1 | 0.70 | Asset set frozen as a contract row: `contracts/release-schedule-06x-v1.yaml` obligation `four_apr_assets_per_tag` is checked by the release workflow, not by a human | — | `pv validate contracts/release-schedule-06x-v1.yaml` | contract |

**Not in A:** darwin and windows `apr` binaries. They were only ever built on hosted runners (G11); the fleet has `mini` (Apple M4, cowork-first) and no windows box. They return under D-3 of PP-066 once a fleet builder exists — recorded, not promised.

### B. NVIDIA CUDA Rust — 100 % supported, and the default

**State (G12, G13):** the 0.67 scope of #3062 is *landed in code* (T0/O2/T3) with the spec PR itself still red. The measurements that govern "default" are: cutile RMSNorm on GB10 is parity-correct and 1.03–1.31× slower than the hand-PTX kernel (#3062, comment 3 `[A]`); cuda-oxide ports were retracted as ~4× slower than production `HwDp4a` (NVIDIA spec §0 G15 `[A]`); `cuda-core` (the host runtime) is stable-Rust and runs wherever the driver is ≥ R580 — gx10 and yoga, not lambda (G13).

**What "default" means here, layer by layer** (decision D-2, §6):

- **Host runtime (loader, launch, occupancy, memory):** `cuda-core`/`cuda-bindings` become the **default** path in the `--features cuda` binary from **0.68**, selected by the BackendRegistry (R-0, #2904/#3002/#3004 on milestone 0.68.0) when the driver meets the floor, with the hand FFI as the registry's fallback and a three-way differential gate (T1, O1a/O1b/O1c) proving the two agree with the CPU reference. This is 100 % of the runtime surface on every host that can run it, and an honest `unavailable reason=DriverBelowR580` where it cannot.
- **Kernels:** a cutile or oxide kernel becomes the default **per kernel, per architecture** when it is parity-correct **and ≥ 1.0× the production kernel** for three consecutive nightlies on the host that runs it (§8 exit criterion 1 of the NVIDIA spec). A slower kernel does not ship as the default; P-4 ("correct before fast") and the withdrawn-claims doctrine forbid it. The ladder is measured every train (69-B1), so the day a port crosses 1.0× it is promoted on the next train without a spec amendment.

| Row | Train | Deliverable | Host | Acceptance command | Gate |
|---|---|---|---|---|---|
| 67-B1 | 0.67 | #3061 (spec) and #3068 (PMAT-1095: the two wrong-host GPU tests, fleet readiness) merged green; #3066 (PTX emitter determinism — the cubin cache has never hit) merged; `cargo test -p aprender-gpu --features cuda --lib --release` green on gx10 (CUDA 13.3) **and** on yoga (sm_89), receipts in `evidence/dogfood/0.67.0/` | gx10, yoga | `gh pr view 3061 3068 3066 --json state --jq .state` = MERGED ×3; the two test logs cite `test result: ok` | cuda-nightly, 67-D1 |
| 68-B1 | 0.68 | **T1** loader-differential oracle: experiments/cuda-core-oracle/ (own `[workspace]`, never a dependency of a published crate), O1a (is the GH-480 rewriter still necessary?), O1b (three-way patched/unpatched/CPU), O1c (cubin cache key case table); owed contract contracts/nvidia-cuda-rs-loader-differential-v1.yaml; `cargo deny check --manifest-path experiments/cuda-core-oracle/Cargo.toml` explicit | gx10 (yoga after 68-D1) | the oracle's case table prints `attempted= killed=` with 0 survivors; O1a/O1b/O1c rows recorded in the contract's evidence | nightly on gx10 |
| 68-B2 | 0.68 | **cuda-core is the default host runtime** behind the registry: feature `cuda-rs` on by default inside `cuda`; `apr devices --json` prints `runtime: cuda-core` on gx10 and yoga and `runtime: hand-ffi reason=DriverBelowR580` elsewhere; the 3-way differential runs in cuda-nightly on both GPU hosts | gx10, yoga | `apr devices --json \| jq -r '.[] \| select(.kind=="cuda") \| .runtime'` = `cuda-core` on both hosts | differential RED on any disagreement with the CPU reference |
| 68-B3 | 0.68 | **T2** `cargo oxide sanitize --tool memcheck\|racecheck\|synccheck` over the five `experiments/cuda-oxide/` ports and `ptx-schedule` perturbation fuzzing, nightly on gx10, pinned `nightly-2026-08-28` (not the blog's stale date) | gx10 | the nightly job exists, ran, and its summary lists 5 ports × 3 tools with 0 findings or a filed issue per finding | nightly |
| 69-B1 | 0.69 | **Kernel ladder:** cutile ports of the decode hot path (Q4_K matvec, attention, SwiGLU, RoPE; RMSNorm exists) A/B'd nightly against the production kernel on gx10 (sm_121) and yoga (sm_89; cutile needs CUDA ≥ 13.2 on sm_8x → 68-D1); promotion rule above; results land in the NVIDIA spec's §8 table | gx10, yoga | owed: scripts/kernel_ladder.sh prints one row per (kernel, arch) with `parity=ok ratio=<r>` and `default=<cutile\|hand-ptx>` | nightly; a promoted kernel also runs the PP-26 witness (kernel-family divergence, `[[project_pp26_kernel_family_divergence]]`) |
| 70-B1 | 0.70 | **100 %:** every GPU path (loader, launch, occupancy, memory, cuBLAS handle) goes through cuda-core where the driver allows; the hand FFI remains only as the registry fallback; O1a's verdict decides the GH-480 rewriter (a decision row, D-8); NVIDIA spec §8 exit criteria re-measured and published in `docs/BEATS.md` Pillar-4 notes with receipts | gx10, yoga | `grep -rn 'load_sym!' crates/aprender-gpu/src/driver/sys/mod.rs \| wc -l` shrinks to the fallback set named in D-8; `apr devices --json` shows `runtime: cuda-core` on both GPU hosts | contract obligation `nvidia_rust_default_runtime` |

### C. gx10 never idle — unclog the build system, raise CUDA quality, own the default model story

**State (G7, G8):** gx10 is idle at PR time and runs ≈ 1 h/day of scheduled work, two thirds of it red. "Never idle" is a queue that is never empty, measured in job-minutes per day from the Actions API — not a load average (`[[feedback_high_load_is_intended_headroom]]`).

| Row | Train | Deliverable | Acceptance command | Gate |
|---|---|---|---|---|
| 67-C1 | 0.67 | **`gpu-quick` PR job on gx10** for PRs whose touched-crate selection intersects `{aprender-gpu, aprender-cuda-edge, aprender-serve, aprender-train, aprender-compute}`: `nice -n 19 cargo test -p aprender-gpu --features cuda --lib --release` + the perf053 filter, `timeout-minutes: 15`, label `[self-hosted, Linux, ARM64, cuda, gx10]`, the cuda-nightly *yield-to-training* step reused. **Advisory** in 0.67 (not in `gate.needs`) | `gh run view <run> --json jobs --jq '.jobs[] \| select(.name=="gpu-quick") \| .conclusion'` = success on a GPU-touching PR; `scripts/check_runner_labels.sh` exit 0 | 68-C3 makes it required |
| 67-C2 (P0-2, #3083) | 0.67 | **Self-hosted preflight:** a first step on every self-hosted job that fails fast on a missing tool (`gh`, `jq`, `curl`, `rustup`, `cargo`, `nvidia-smi` on cuda labels) — owed: scripts/ci_self_hosted_preflight.sh with a case table; plus owed workflow fleet-toolset.yml that records each runner's toolset daily under `evidence/fleet/` (owed directory) so "what does this box have" is a file, not a memory | the preflight RED on a box missing one named tool (mutation: hide `jq` from PATH); `evidence/fleet/<runner>.json` exists for gx10, yoga and one intel runner | guard-tree |
| 67-C3 | 0.67 | **The default model story on GB10, daily:** `qwen-story-daily.yml` extended from the 1.5B to **Qwen2.5-Coder-7B-Instruct Q4_K_M**: `apr pull` → `apr run` → `apr serve` + the PP-26 witness at c=1/4 (9 min today, G8) | the run's summary shows the 7B story green with the witness `intra_agree_to` recorded | nightly |
| 68-C1 | 0.68 | **gx10 takes arch-neutral fleet work:** `guard-tree` (cargo-free, 18–27 min, G2/G3) and `mutants` (diff-scoped) may run on `[self-hosted, Linux, ARM64, gx10]` under a `concurrency` group of width 2 and `nice -n 19`; the intel pool keeps `workspace-test` | `gh run view <run> --json jobs --jq '.jobs[] \| select(.name=="guard-tree") \| .runnerName'` = `gx10-blackwell` on ≥ 5 PR runs; intel queue depth (busy/16 at three daily samples) falls — recorded, not asserted | `scripts/check_runner_labels.sh` |
| 68-C2 | 0.68 | **Utilization instrument:** owed scripts/fleet_utilization.sh — job-minutes per runner per day from `gh api …/actions/runs?created=<day>` + `jobs`, printed as a table; a dogfood row `gx10_minutes_per_day >= 480` (8 h) from 0.68, `>= 720` from 0.69, `>= 960` from 0.70 — labels: **Target** | the script's `--selftest` case table; the dogfood row RED below threshold | dogfood |
| 68-C3 | 0.68 | `gpu-quick` (67-C1) becomes **required** (added to `gate.needs`) after **5 consecutive green nightlies** on gx10 — measured, not asserted | `gh api …/check-runs?check_name=gpu-quick` on 5 consecutive nightly commits = success | ci.yml (web-UI merge click) |
| 69-C1 | 0.69 | gx10 runs the kernel ladder (69-B1), the 7B parity cells (69-F1/F2), the QLoRA recipe smoke (69-G2) and the aarch64 release builds (A) — the queue is never empty; the backlog lane (the ladder) fills idle windows | `fleet_utilization.sh` ≥ 720 min/day for 3 consecutive days | dogfood |
| 70-C1 | 0.70 | ≥ 960 min/day for the whole train; `[self-hosted, Linux, ARM64, cuda, gx10]` appears in ≥ 6 workflows | `fleet_utilization.sh`; `grep -l 'gx10' .github/workflows/*.yml \| wc -l` ≥ 6 | dogfood |

### D. yoga for CUDA testing

**State (G9, G13, G20):** yoga is rack-mounted and permanent (operator 2026-09-09), driver 595 (≥ R580), 8 GB VRAM, no CUDA 13 toolkit, and runs nothing but the release lane. Its runner group `gpu-x86` does not allow public repositories and is not restricted to named workflows. The intermittency objection is dead; the security one is not (`[[reference_cuda_ci_runners]]`).

| Row | Train | Deliverable | Acceptance command | Gate |
|---|---|---|---|---|
| **67-D0** | 0.67 | **Security precondition before PR code executes on yoga:** runner group `gpu-x86` restricted to the named workflows (`ci.yml`'s `cuda-unit`, `cuda-nightly.yml`, `binary-release.yml`, `qwen-story-daily.yml`) with `allows_public_repositories` set as required for a public repo, and `pr-gate.yml`'s `authorize` job in front of any `pull_request` job that lands there; `[self-hosted, Linux, X64, cuda, yoga]` names one box | `gh api orgs/paiml/actions/runner-groups/5 --jq '{restricted_to_workflows, selected_workflows, allows_public_repositories}'` matches the decision in D-3; a dispatch smoke on yoga from this repo succeeds (today `[U]` — G9) | infra |
| 67-D1 | 0.67 | **`cuda-unit` job on yoga** (advisory): `cargo test -p aprender-gpu --features cuda --lib --release` and `cargo test -p aprender-serve --features cuda --lib --release <cuda filter>`, `timeout-minutes: 15`, for GPU-touching PRs (same selection as 67-C1); plus **YOGA-NIGHTLY-001** (#3060: the sm_89 nightly axis) merged so yoga has a nightly of its own | job green on a GPU-touching PR; `gh pr view 3060 --json state` = MERGED | 68-D2 |
| 67-D2 | 0.67 | **yoga as the sm_89 parity host** (lambda has no runner, G7): `~/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf` pulled once; llama.cpp and Ollama pinned builds deployed from the prior-art forjar files (G20); the first `RECORDED` 7B c=1 row on yoga **with both comparators** appended to `evidence/parity/LEDGER.md` | the ledger row exists with `host=yoga`, both comparator pins, and `what_it_lacks[]` filled honestly | ledger (PP-9) |
| 68-D1 | 0.68 | **CUDA 13.3 toolkit on yoga**, side-by-side via alternatives exactly as on gx10 (driver untouched; rollback is `update-alternatives --set cuda /usr/local/cuda-12.4`); verified by `cargo test -p aprender-gpu --features cuda --lib --release` under 13.3 and by building `cuda-bindings` (G13) | `update-alternatives --query cuda`; the two commands' logs in `evidence/dogfood/0.68.0/yoga/` (owed dir) | 68-B1/B2 on yoga |
| 68-D2 | 0.68 | `cuda-unit` becomes **required** after 5 consecutive green yoga nightlies | as 68-C3, `check_name=cuda-unit` | ci.yml (web-UI click) |
| 69-D1 | 0.69 | yoga runs the sm_89 leg of the kernel ladder (69-B1) and the 7B parity bands c=1/4/8 (69-F1) nightly; utilization ≥ 240 min/day — **Target** | `fleet_utilization.sh` yoga ≥ 240 | dogfood |
| 70-D1 | 0.70 | yoga ≥ 480 min/day; the two GPU hosts together carry every CUDA gate this document names | `fleet_utilization.sh`; `grep -l 'yoga' .github/workflows/*.yml \| wc -l` ≥ 4 | dogfood |

### E. PR iterations ≤ 20 minutes; the FULL tier for pre-release dogfood only

**State (G2–G6, G17):** a PR costs 66 min wall today and the critical path is one serial step; the merge queue re-runs FULL whenever main moved; the daily full run is red. The operator's rule is two-sided: **cap PR testing at 20 minutes by running fewer tests**, and **force the FULL run for pre-release dogfood only**. This document adds one safety net the rule does not forbid: a **green nightly FULL run on main** (it already exists as `coverage-nightly.yml`; it must be green, G17) so a full-tier-only defect is caught within 24 h instead of on release day.

**Budget (hard, per PR):** every job in `.github/workflows/ci.yml` that runs on `pull_request` gets `timeout-minutes: 20`, and the PR's wall clock from first job start to `gate` completion is ≤ 20 min. A job that cannot fit is **scoped down or moved to nightly**, never given a longer timeout.

| Row | Train | Deliverable | Acceptance command | Gate |
|---|---|---|---|---|
| **67-E0** | 0.67 | **The nightly FULL run is green** (`coverage-nightly.yml`, G17) or a dedicated `full-nightly` job runs the exact `workspace-test` FULL tier on main at 03:00Z on the clean-room pool. Precondition for moving anything out of PRs | `gh run list --workflow coverage-nightly.yml --limit 3 --json conclusion` = success ×3 | nightly |
| **67-E1** (P0-3, #3084) | 0.67 | **One build graph for the tree-reader targets:** replace the 26-crate `&&` chain (G4) with a single `cargo nextest run --profile ci` over a nextest filterset derived from `scripts/tree_reader_tests.txt` (`package(x) & kind(lib)` ∪ `binary_id(x::t)`), so the union builds once and runs in parallel; `timeout-minutes: 20` on the quick steps; the registry guard (`scripts/check_tree_reader_tests.sh`) unchanged | the step's measured duration on three PR runs ≤ 12 min (**Target**; the 55-min baseline is G2); `bash scripts/ci_test_tier.sh --self-test` green | workspace-test |
| 67-E2 (P0-3, #3084) | 0.67 | **The merge queue never runs FULL:** `merge_group` runs `reuse` (same tree) or the **quick** tier on the queue ref (touched ∪ tree-readers), never `full`; `push` to main runs quick + guards; **FULL runs only in `coverage-nightly`/`full-nightly` and `scripts/dogfood.sh --phase pre-publish`** — decision D-1 | `bash scripts/ci_test_tier.sh --self-test` rows: `merge_group different tree -> quick`, `push -> quick`; a merge_group run's `workspace-test` ≤ 20 min | ci_test_tier case table |
| 67-E3 | 0.67 | **guard-cargo and guard-tree under the budget:** nightly-class steps leave the PR path (book examples 3 min, wasm32 1 min, model-tests 4 min, the 9-min tier case table, G2) into owed workflow guards-nightly.yml; `scripts/guard_tree.sh --no-cargo` runs its guards with `xargs -P` (18 min → **Target** ≤ 8); path-scoped guards stay at PR time | both jobs ≤ 20 min on three consecutive PR runs | guard-tree |
| 68-E1 | 0.68 | **PR budget guard:** owed scripts/check_pr_budget.sh — reads the last 10 `pull_request` runs, asserts every non-skipped job ≤ 20 min and wall ≤ 20 min, `--selftest` with a synthetic 21-minute job RED; wired into the pre-publish dogfood as a row | `bash scripts/check_pr_budget.sh` exit 0 over 10 consecutive runs | dogfood |
| 68-E2 | 0.68 | `mutants` starts in parallel with `workspace-test` (it only needs the diff), so the 7 min no longer sit on the critical path | the two jobs' `startedAt` within 60 s on a PR run | ci.yml |
| 69-E1 | 0.69 | The budget guard is **required** (`gate.needs`); E is declared done at **10 consecutive PR runs ≤ 20 min** | `check_pr_budget.sh` in `gate.needs`; 10 green | ci.yml (web-UI click) |
| 70-E1 | 0.70 | Sustained: 20 consecutive ≤ 20 min; the freed runner-hours are recorded in the 0.70 epic (fleet_utilization before/after) | `check_pr_budget.sh --window 20` exit 0 | dogfood |

**What the FULL tier still guards, and where:** feature-gated suites (`model-tests`, `setfit`, `--features cuda` on the GPU hosts), the whole-workspace `nextest` line, the GPU crates' per-package default-feature run, `aprender-compute`, and the explicit integration list (ci.yml line 470). They run **nightly on main** and **on every release candidate before publish**. A defect they catch on release day holds the train under §1.1 (≤ 12 h) — the price of a 20-minute PR, paid at most once per train and never silently.

### F. Qwen2.5-Coder-7B parity with Ollama and llama.cpp

**State (G15):** correctness parity is proven and gated (cosine ≥ 0.98 floor, both GPU hosts). Throughput at c=1 is short of llama.cpp (decode ratio 0.650, prefill 0.275 — receipt: `evidence/parity-http/findings.json`); every `c>1` band is INVALID until the PP-26 witness passes; there is no Ollama-vs-7B receipt and no CONFORMANT ledger row. The mechanisms and levers are already named in `docs/specifications/0.66-performance-parity-report.md` §4–§5 (W-A host work, W-B multi-warp GEMV, W-C prefill copies, W-D f16 KV, W-E scheduler, W-F resident `apr run`, W-G GB10 NEON/prefill, W-H batch-invariant reduction, W-I GPU install). This document **schedules** them; it does not restate them. Parity is *published* only from CONFORMANT rows (PP-12) — everything before that is recorded.

| Row | Train | Deliverable | Host | Acceptance command | Gate |
|---|---|---|---|---|---|
| **67-F1** | 0.67 | **PP-26 witness green on gx10** (cuda-nightly is red on it today, G8): the logprob-margin instrument (master §12 row 22) classifies the c=16 parting as near-tie or defect; the witness is re-baselined at the classified value; BSE-11 byte-compare green | gx10 | `gh run list --workflow cuda-nightly.yml --limit 2 --json conclusion` = success ×2 | cuda-nightly |
| 67-F2 | 0.67 | **First Ollama receipt on the 7B** and the yoga parity host (67-D2): c=1 RECORDED rows for apr vs llama.cpp *and* apr vs Ollama on yoga (sm_89) and gx10 (sm_121), interleaved, n ≥ 5, comparator pins in `scripts/llama_pin.toml` (Ollama pin added) | yoga, gx10 | two new `RECORDED` rows in `evidence/parity/LEDGER.md` naming both comparators and `what_it_lacks[]` | ledger |
| 68-F1 | 0.68 | **W-A + W-B** (host work out of the token loop; multi-warp Q4_K GEMV at M ∈ {1,4,8}) on sm_89 and sm_121; **Target:** c=1 decode ratio ≥ 0.85 vs llama.cpp after W-A, recorded per host beside its receipt | yoga, gx10 | `scripts/perf_gate.sh` band c=1 rows with `comparator_status: MEASURED`; the PP-26 m=1 stream unchanged | perf gate (REPORT until P-5) |
| 68-F2 | 0.68 | **W-H** batch-invariant reduction: PP-26 witness PASS at c ∈ {4, 8, 16} for **3 consecutive nightlies on both GPU hosts** | yoga, gx10 | witness `intra_agree_to = 128` ×3 nights ×2 hosts | cuda-nightly |
| 69-F1 | 0.69 | **W-E** batched admission under one graph + **W-C** prefill without synchronous copies; c=4/8 bands **VALID**; **Target:** `agg(4) ≥ 3 × dec(1)` (the prior-art bar) on yoga, recorded | yoga, gx10 | perf_gate band rows c=4/8 `validity=VALID`; the aggregate row beside its receipt | perf gate |
| 69-F2 | 0.69 | **First CONFORMANT ledger rows** (PP-21 signature, PP-25 client sha256, PP-26 witness, PP-28 counter, PP-30 clock) for 7B vs llama.cpp and vs Ollama on yoga and gx10 | yoga, gx10 | `evidence/parity/LEDGER.md` rows with `conformance=CONFORMANT`; `bash scripts/spec_conformance.sh` exit 0 | spec_conformance |
| 70-F1 | 0.70 | **Publish** the 7B ratios from CONFORMANT rows only: `docs/BEATS.md` Pillar-4 rows with receipts; a 7B decode no-collapse floor contract per architecture (pattern: `contracts/beat-ollama-decode-throughput-speed-v1.yaml`) enforced in cuda-nightly on both hosts; **W-F** resident `apr run` for the one-shot wall-clock comparison | yoga, gx10 | `cargo test -p aprender-core --test readme_contract test_beats_md_publishes_every_contract_measurement`; the new beat contract `status: enforced` | readme_contract, cuda-nightly |

### G. The same easy story for Qwen2.5-Coder-7B, declaratively fine-tuned (0.69 and 0.70)

**State (G16):** inference is three commands (`apr pull`, `apr run`, `apr serve`). Fine-tuning is one command with 30 flags and no file that names a run. Ollama's story is a Modelfile, and `apr modelfile parse` already reads one. **The story this priority ships:**

```bash
apr pull qwen2.5-coder-7b-instruct-q4_k_m
apr finetune --recipe examples/recipes/qwen2.5-coder-7b-lora.yaml      # plan, train, merge — one file
apr run ./out/qwen2.5-coder-7b-mine --prompt "…"
```

| Row | Train | Deliverable | Host | Acceptance command | Gate |
|---|---|---|---|---|---|
| 68-G1 | 0.68 | **Recipe schema decided and contracted** (D-7): owed contracts/apr-finetune-recipe-v1.yaml (`kind: schema`) — `base`, `method` (lora/qlora), `rank`, `data`, `epochs`, `learning_rate`, `max_seq_len`, `output`, `gpu_backend`, `merge`, `checkpoint_format`; every key maps 1:1 to an existing `apr finetune` flag (G16) — no second vocabulary | — | `pv validate contracts/apr-finetune-recipe-v1.yaml` | pv |
| **69-G1** | 0.69 | **`apr finetune --recipe <file>`**: one resolver turns the recipe into the same argument struct the flags fill (flags override recipe keys; a conflict is printed, never silent); `--plan` reuses `apr tune --plan`'s memory planner; owed examples/recipes/qwen2.5-coder-7b-lora.yaml and a 1.5B sibling; `contracts/apr-page-cli-finetune-v1.yaml` and the book page updated in the same PR | CPU box (plan), gx10 (train) | `apr finetune --recipe examples/recipes/qwen2.5-coder-7b-lora.yaml --plan` exit 0 on a CPU box; the registry contract row for `finetune` lists `--recipe` | `cli_commands` + contract falsifiers |
| 69-G2 | 0.69 | **7B QLoRA smoke on gx10 via the recipe**, nightly: train N steps under the existing `cuda_loss_window` falsifier, merge, `apr run` the merged model — the loop closes end to end on the default model | gx10 | the nightly summary shows loss-window PASS and a generated completion from the merged model | cuda-nightly |
| 70-G1 | 0.70 | **The story is THE story:** README's training row leads with `apr finetune --recipe`; `docs/BEATS.md` Pillar-3 row for the 7B recipe against the Unsloth comparator (the 0.66 parity report's T-rows) **recorded, not claimed**; post-publish dogfood runs the three commands above on gx10 | gx10 | `grep -c 'apr finetune --recipe' README.md` ≥ 1; dogfood receipt `evidence/dogfood/0.70.0/gx10.json` carries the recipe row | dogfood, readme_contract |
| 70-G2 | 0.70 | **Modelfile import:** `apr modelfile parse --to-recipe` emits a recipe from an Ollama Modelfile (CRUX-K-11's parser, one more output format), so an Ollama user's declarative file becomes an apr fine-tune in one step | — | round-trip case table: Modelfile → recipe → `--plan` exit 0 | cli_commands |

---

## §3. The four trains — scope as it stands at `2c584a168`

A train's scope is the union of (1) the rows above assigned to it, (2) what the milestone already holds, (3) the invariants of §1.3. Sizes are bounded by §1.2 (8–20 PRs per train); a row that cannot land in its window slips **forward**, never backward, and never holds the train past 12 h.

### 0.67.0 — leaves 2026-09-12T18:00Z (window 09-12T06:00Z → 09-13T06:00Z) · epic #3078

| Row | Item | Source | Host | State at HEAD |
|---|---|---|---|---|
| 67-A1 | **P0-1** (#3082) four `apr` assets per tag, verify-apr-assets, dogfood + release-criteria rows | §2 A | yoga, gx10, intel | `v0.66.0`: 0 of 4 `[V]` |
| 67-A1b | backfill `v0.66.0` after #3074 | §2 A | gx10, yoga | #3074 open, `present` check red `[V]` |
| 67-C2 | **P0-2** (#3083) self-hosted preflight + fleet toolset probe | §2 C | all self-hosted | `gh` missing on gx10 measured on run `34448908554` `[V]` |
| 67-E0 | nightly FULL run green | §2 E | intel | red ×5 `[V]` |
| 67-E1 | **P0-3** (#3084) one build graph for tree-reader targets; `timeout-minutes: 20` | §2 E | intel | 55-min step `[V]` |
| 67-E2 | merge queue never FULL; push quick + guards; FULL only nightly + pre-publish | §2 E | intel | merge_group FULL on moved main `[V]` |
| 67-E3 | guard-cargo / guard-tree ≤ 20 min | §2 E | intel (gx10 from 0.68) | 44 / 27 min `[V]` |
| 67-B1 | #3061 · #3068 · #3066 merged; `--features cuda` suite green on gx10 (13.3) and yoga | #3062 | gx10, yoga | all three open and red `[V]` |
| 67-C1 | `gpu-quick` advisory job on gx10 | §2 C | gx10 | gx10 idle at PR time `[V]` |
| 67-C3 | 7B default-model story daily on GB10 | §2 C | gx10 | 1.5B today `[A]` |
| 67-D0 | yoga runner-group security + label + dispatch smoke | §2 D | infra | group not restricted `[V]`; smoke `[U]` |
| 67-D1 | `cuda-unit` advisory on yoga; #3060 merged | §2 D | yoga | #3060 open `[V]` |
| 67-D2 | yoga parity host: model + two comparators + first RECORDED 7B rows | §2 D/F | yoga | no model on yoga `[A]` |
| 67-F1 | PP-26 witness green on gx10 | §2 F | gx10 | cuda-nightly red on it `[V]` |
| 67-F2 | first Ollama-vs-7B receipts (yoga, gx10) | §2 F | yoga, gx10 | none exists `[V]` |
| #3067 | `aprender-gpu` `#[cfg(feature = "cuda")]` tests run only in the gx10 perf053 filter | milestone | gx10 | open `[V]`; covered by 67-B1's full `--features cuda` run |
| 67-R | bump `0.67.0`, CHANGELOG, pre-publish dogfood FULL, tag, release (4 apr + 8 pv assets), cascade, post-publish dogfood on gx10 + yoga, epic closed | §1.3 | — | — |

Size: 16 rows + release, of which 6 are CI/workflow files (web-UI merge click each, `[[feedback_stacked_prs_get_no_ci]]`). This is the upper bound of §1.2; 67-C3, 67-D2 and 67-F2 are the first to slip if the window closes.

### 0.68.0 — leaves 2026-09-15T06:00Z (window 09-14T18:00Z → 09-15T18:00Z) · epic #3079

| Row | Item | Source | Host |
|---|---|---|---|
| 68-B1 | T1 loader-differential oracle (O1a/O1b/O1c) + contract | #3062 §5 | gx10 |
| 68-B2 | cuda-core default host runtime behind the registry; 3-way differential nightly | §2 B | gx10, yoga |
| 68-B3 | T2 oxide sanitizers + ptx-schedule nightly | #3062 §6 | gx10 |
| 68-D1 | CUDA 13.3 toolkit on yoga (alternatives, side-by-side) | §2 D | yoga |
| 68-D2 | `cuda-unit` required after 5 green nightlies | §2 D | yoga |
| 68-C1 | guard-tree + mutants may run on gx10 (width 2, nice 19) | §2 C | gx10 |
| 68-C2 | fleet_utilization instrument + dogfood row (≥ 480 min/day gx10) | §2 C | — |
| 68-C3 | `gpu-quick` required after 5 green nightlies | §2 C | gx10 |
| 68-A1 | R-5 signed manifest + KEY, R-6 installer, R-7 README; `pv` lane and `nightly.yml` off hosted runners | milestone (#2908 #3045 #2909 #2910) | intel |
| 68-E1 | PR budget guard (dogfood row) | §2 E | — |
| 68-E2 | mutants in parallel with workspace-test | §2 E | intel |
| 68-F1 | W-A + W-B; **Target** c=1 decode ratio ≥ 0.85 recorded per host | §2 F | yoga, gx10 |
| 68-F2 | W-H; PP-26 witness PASS c=4/8/16 ×3 nights ×2 hosts | §2 F | yoga, gx10 |
| 68-G1 | recipe schema contract (D-7) | §2 G | — |
| milestone | R-0/R-0b registry (#2904 #3002; R-0a #3004 open), R-2 cuda default features (#2905), R-3 training banner (#2906), C0-1/C0-2/C0-4 gates (#2890 #2891 #2893), G-10b/G-11b (#3013 #3018), R-8 nightly-latest-dogfood (#3019), #2869 (closed by 67-A1's asset set, kept for the installer half) | milestone 0.68.0 (15 open) | — |
| 68-R | release | §1.3 | — |

The milestone already holds 15 issues; with the 14 rows above the train is over the §1.2 bound. **Order of boarding:** 68-B2/68-D1/68-A1/68-E1 first (they gate later trains), then the R-0 registry rows (68-B2 depends on them), then F, then C0/G rows; whatever is left rides 0.69 with a `slipped_from: 0.68.0` note on the epic.

### 0.69.0 — leaves 2026-09-17T18:00Z (window 09-17T06:00Z → 09-18T06:00Z) · epic #3080

| Row | Item | Source | Host |
|---|---|---|---|
| **69-G1** | `apr finetune --recipe` (resolver, `--plan`, example recipes, book + contract) | §2 G | CPU + gx10 |
| 69-G2 | 7B QLoRA recipe smoke nightly on gx10 (loss window, merge, run) | §2 G | gx10 |
| 69-F1 | W-E scheduler + W-C prefill; c=4/8 VALID; **Target** `agg(4) ≥ 3 × dec(1)` recorded | §2 F | yoga, gx10 |
| 69-F2 | first CONFORMANT ledger rows, both comparators, both hosts | §2 F | yoga, gx10 |
| 69-B1 | kernel ladder nightly (cutile ports vs production) on both archs; per-kernel promotion rule | §2 B | gx10, yoga |
| 69-D1 | yoga runs the sm_89 ladder leg + parity bands; ≥ 240 min/day | §2 D | yoga |
| 69-C1 | gx10 ≥ 720 min/day for 3 consecutive days | §2 C | gx10 |
| 69-A1 | post-publish dogfood installs via the installer on gx10, yoga, intel (C4) | §2 A | three hosts |
| 69-E1 | PR budget guard required; 10 consecutive PRs ≤ 20 min | §2 E | — |
| 69-R | release | §1.3 | — |

### 0.70.0 — leaves 2026-09-20T06:00Z (window 09-19T18:00Z → 09-20T18:00Z) · epic #3081

| Row | Item | Source | Host |
|---|---|---|---|
| **70-G1** | the fine-tune story is THE story: README, BEATS Pillar-3 row vs Unsloth (recorded), dogfood runs pull → finetune --recipe → run on gx10 | §2 G | gx10 |
| 70-G2 | `apr modelfile parse --to-recipe` | §2 G | — |
| 70-F1 | publish 7B ratios from CONFORMANT rows only; per-arch decode floor contract enforced nightly; W-F resident `apr run` | §2 F | yoga, gx10 |
| 70-B1 | NVIDIA Rust 100 %: every GPU path through cuda-core where the driver allows; hand FFI = fallback only; O1a decides GH-480 (D-8); §8 exit criteria re-measured | §2 B | gx10, yoga |
| 70-C1 / 70-D1 | gx10 ≥ 960, yoga ≥ 480 min/day; gx10 in ≥ 6 workflows, yoga in ≥ 4 | §2 C/D | both |
| 70-A1 | asset set frozen as a contract obligation checked by the workflow | §2 A | — |
| 70-E1 | 20 consecutive PRs ≤ 20 min; freed runner-hours recorded | §2 E | — |
| 70-R | release | §1.3 | — |

---

## §4. Release-day protocol (every train; the FULL tier's only mandatory home)

Numbered so an autopilot (`/mnt/nvme-raid0/agent-wt/rel-066-autopilot/` is the 0.66 precedent) can run it fail-closed, with a `STOP <step>` line naming the refusing gate.

1. **Freeze the platform.** `gh pr list --state open --label <train>`: rows still open move to the next epic with `slipped_from:`. No new pushes while the queue drains (`[[feedback_push_cadence_starves_the_merge_queue]]`).
2. **Bump.** `bash scripts/bump-version.sh X.Y.0`; `CHANGELOG.md` section from the epic's rows; the bump PR carries the receipt(s).
3. **Pre-publish dogfood — FULL.** `bash scripts/dogfood.sh --phase pre-publish` on the intel pool: the whole-workspace nextest line, GPU crates per-package, `aprender-compute`, the integration list, `make coverage-check` (`COV_FLOOR := 88`), `pv lint contracts/`, the PR-budget row (68-E1), the fleet-utilization row (68-C2), and the **release-assets row** (67-A1, on the *previous* tag until this one exists). NO-GO stops the train (§1.1).
4. **Merge the bump, tag, release.** `gh release create vX.Y.0 --target <main tip>`; `binary-release.yml` fires on `release: published`: 4 `apr` + 4 `pv` builds, verify-apr-assets, smoke-cuda on yoga and gx10.
5. **Verify assets by command, never by eye.** scripts/check_release_assets.sh vX.Y.0 (owed by 67-A1; until it lands, the §2 A acceptance query) — 8 apr files, 8 pv files.
6. **Cascade.** `scripts/check_publish_preflight.sh` then the crates.io drain (multi-pass, `[[feedback_crates_io_cascade_auth_config_traps]]`).
7. **Post-publish dogfood** on gx10 and yoga (+ intel): install the release asset (from 0.68 through the installer), `apr devices --json`, the parity protocol at c=1, the 7B story (67-C3; from 0.69 the recipe smoke), receipts under `evidence/dogfood/X.Y.0/<host>.json`.
8. **Close.** Comment on the release epic with the receipt paths and the §1.1 window arithmetic for the next train; close the milestone; the next train's epic gets `T = <publishedAt>`.

---

## §5. Epics — one per release, so Alfredo can comment

One GitHub issue per train, label `epic`, milestone `X.Y.0` (milestones `0.69.0` and `0.70.0` are created by this ticket — G14). Conventions, copied from #2873 so the two read alike:

- **Body = the train's §3 table + the §6 decisions that apply**, and the sentence *"Decisions apply at their §6 defaults unless overridden in a comment here."* A comment on the epic is how a decision is overridden — by Alfredo or anyone — and the override is copied into §11 of this document with `decided_by` and the comment link.
- **One sub-issue per row** as rows start (`pmat work add` mints the ticket; the issue links the ticket and the row id). The three P0 five-whys tickets are #3082 (P0-1 → 67-A1), #3083 (P0-2 → 67-C2) and #3084 (P0-3 → 67-E1/E2), milestone 0.67.0, linked from #3078.
- **Topic epics ride trains:** #3062 is linked from the 0.67.0 epic; #2873's PP-066 rows from the 0.68.0 epic.
- **Closing comment** = §4 step 8.

| Release | Epic | Milestone | Created by |
|---|---|---|---|
| 0.67.0 | #3078 | `0.67.0` (exists, #4) | PMAT-1097 |
| 0.68.0 | #3079 | `0.68.0` (exists, #5) | PMAT-1097 |
| 0.69.0 | #3080 | `0.69.0` (created, #6) | PMAT-1097 |
| 0.70.0 | #3081 | `0.70.0` (created, #7) | PMAT-1097 |

---

## §6. Decisions — recommended defaults, with the evidence and the dissent

Each decision is applied below at its default. Overriding one is a comment on the release epic (§5), copied into §11. None is a question: a fork is surfaced as a decided recommendation with its dissent (`[[feedback_quorum_decides_toyota_way]]`).

| id | Decision | Default (applied) | Evidence | Dissent, stated fairly |
|---|---|---|---|---|
| **D-1** | Where the FULL tier lives | PR and merge queue: quick tier only, hard 20-min cap. FULL: nightly on main (`coverage-nightly`/`full-nightly`) **and** pre-publish dogfood. Never on `pull_request`, `merge_group` or `push` | G2–G6, §1.2; operator's E verbatim | main can carry a full-tier-only defect for up to 24 h and the train can be held on release day (≤ 12 h, §1.1). The nightly is the mitigation and it is **red today** (G17) — 67-E0 is therefore a precondition, not a nicety |
| **D-2** | What "NVIDIA Rust default" means | Runtime layer default in 0.68 behind the registry where driver ≥ R580; kernels default per kernel/arch only at parity **and ≥ 1.0×** production, measured nightly; a slower kernel never ships as default | G12/G13; T3's 1.03–1.31× slower; oxide ~4× slower `[A]` | the operator asked for "100 % and default". If the operator rules that slower NVIDIA kernels ship as the default anyway, §11 records it and 69-B1's promotion rule drops the ratio clause — P-4 is then overridden explicitly, not silently |
| **D-3** | yoga at PR time | Advisory `cuda-unit` in 0.67 → required in 0.68 after 5 green nightlies; **runner-group restriction to named workflows lands first** (67-D0) because the repo is public and a self-hosted runner executing PR code is arbitrary code execution on owned hardware | G9; `[[reference_cuda_ci_runners]]` security point | a required GPU check adds a single-box dependency on yoga to every GPU-touching PR; `infra#359` still says Option A. Permanence is grounds to reopen it; this document reopens it |
| **D-4** | gx10 as a general aarch64 runner | Yes for cargo-free guards and diff-scoped mutants, width 2, `nice -n 19`, yield-to-training; never `workspace-test` (aarch64 ≠ the x86 comparand) | G7/G8, §1.2 | gx10 is the only Blackwell host: a saturated queue slows the CUDA nightlies. The width cap and the utilization instrument (68-C2) are the guard; if the nightly's start slips > 30 min for 3 nights, width drops to 1 |
| **D-5** | Trains leave on time | Scope slips, dates do not; a red release gate holds a train ≤ 12 h | §1.1; G1 (the cadence was held in July) | with 33 open PRs and 3–8 merges/day some rows will slip every train; the epic's `slipped_from:` trail is the record, and a row slipping twice is escalated as its own five-whys |
| **D-6** | Parity proof hosts for F | **yoga (sm_89) and gx10 (sm_121)**; lambda receipts stay valid history, lambda takes no new rows until a runner is registered | G7 (no lambda runner), G20 (yoga was the prior-art primary) | yoga's 8 GB VRAM is tight for 7B Q4_K_M at c=16 (KV per slot per the parity report §4 G3); c=16 on yoga may be UNMEASURED with an owner — recorded, not hidden |
| **D-7** | Recipe format for G | YAML, one key per existing `apr finetune` flag, schema contract `kind: schema`, flags override the file, `--plan` before train, Modelfile import as a second input (0.70) | G16 | a second config dialect is a maintenance cost; the 1:1 key↔flag rule and the single resolver are the guard against drift |
| **D-8** | Fate of the GH-480 PTX rewriter | Decided by O1a's verdict (68-B1) in 0.70: unnecessary on the current driver → deleted with the hand FFI reduced to the fallback set; necessary → kept and its contract row names the driver range | NVIDIA spec §5.2 | deleting a rewriter that a future driver needs again is a Blackwell-class regression; the contract row's driver range is the tripwire |
| **D-9** | Non-Linux binaries | Out of A; return under PP-066 D-3 when a fleet builder exists | G11; operator's rule on hosted runners | users on darwin/windows lose the nightly binaries they had; the `cargo install` path remains documented |

---

## §7. Gates this document ships, and the ones it owes

**Ships in this PR (PMAT-1097), each mutation-verified RED→GREEN before merge:**

| Gate | Where | RED when | Mutation run |
|---|---|---|---|
| FALSIFY-DOCS-CLAUDE-001 extended to this file | `crates/aprender-core/tests/readme_contract.rs` `DOCS_WITH_PATHS` | a backticked repo path in this document does not exist | a bogus citation of crates/nowhere/x.rs → `cargo test -p aprender-core --test readme_contract test_documented_paths_exist` fails; revert → passes |
| `contracts/release-schedule-06x-v1.yaml` (`kind: pattern`) | `pv validate`, `pv lint contracts/` | the spec is missing from the tree or from `docs/specifications/TOC.md`; a train section lacks its epic link or its release row; the drift gate no longer names this file | each falsification test's `test:` is a command; FALSIFY-REL-06X-003 is run against a copy with a train's epic line deleted |
| `contracts/apr-docs-v1.yaml` FALSIFY-DOCS-CLAUDE-001 row | prediction text names this document | — (a description change; the test is the one above) | — |

**Owed by rows above (each lands with its own contract row and RED→GREEN proof, in the train named):**

| Owed artifact | Row | Train |
|---|---|---|
| scripts/check_release_assets.sh (`<tag>`, `--selftest`) + dogfood row + release-criteria row | 67-A1 | 0.67 |
| scripts/ci_self_hosted_preflight.sh + fleet-toolset.yml + `evidence/fleet/` | 67-C2 | 0.67 |
| `ci_test_tier.sh` case-table rows for `merge_group → quick`, `push → quick`; nextest filterset for tree-readers | 67-E1/E2 | 0.67 |
| guards-nightly.yml; `guard_tree.sh --no-cargo` parallel runner | 67-E3 | 0.67 |
| `gpu-quick` (gx10) and `cuda-unit` (yoga) jobs in `ci.yml` | 67-C1, 67-D1 | 0.67 |
| scripts/check_no_hosted_runners.sh | 68-A1 | 0.68 |
| scripts/fleet_utilization.sh + dogfood row | 68-C2 | 0.68 |
| scripts/check_pr_budget.sh + dogfood row | 68-E1 | 0.68 |
| contracts/nvidia-cuda-rs-loader-differential-v1.yaml; experiments/cuda-core-oracle/ | 68-B1 | 0.68 |
| contracts/apr-finetune-recipe-v1.yaml | 68-G1 | 0.68 |
| scripts/kernel_ladder.sh | 69-B1 | 0.69 |
| examples/recipes/qwen2.5-coder-7b-lora.yaml (+ 1.5B) | 69-G1 | 0.69 |
| per-arch 7B decode floor contract (`contracts/beat-*`) | 70-F1 | 0.70 |

---

## §8. Obligation DAG (data, not prose)

`blocked_by` is the only edge. A row with a live blocker carries no date of its own — its expiry is its latest blocker's train (the `spec_conformance.sh` rule for §12 of the master, applied here). Roots carry the train they board.

```yaml
# id            blocked_by                        train   host
67-E0:          []                                0.67    intel
67-E1:          []                                0.67    intel
67-E2:          [67-E1]                           0.67    intel
67-E3:          []                                0.67    intel
67-A1:          [3074]                            0.67    yoga,gx10,intel   # 3074 = PR #3074 (REST upload)
67-A1b:         [67-A1]                           0.67    gx10,yoga
67-C2:          []                                0.67    all
67-D0:          []                                0.67    infra
67-C1:          [67-C2]                           0.67    gx10
67-D1:          [67-D0, 67-C2]                    0.67    yoga
67-B1:          [67-D1]                           0.67    gx10,yoga
67-C3:          []                                0.67    gx10
67-D2:          [67-D0]                           0.67    yoga
67-F1:          []                                0.67    gx10
67-F2:          [67-D2]                           0.67    yoga,gx10
68-D1:          [67-D1]                           0.68    yoga
68-B1:          [67-B1]                           0.68    gx10
68-B2:          [68-B1, 68-D1, R-0]               0.68    gx10,yoga         # R-0 = #2904/#3002/#3004
68-B3:          [67-B1]                           0.68    gx10
68-C1:          [67-E3]                           0.68    gx10
68-C2:          []                                0.68    -
68-C3:          [67-C1]                           0.68    gx10
68-D2:          [67-D1]                           0.68    yoga
68-A1:          [67-A1]                           0.68    intel
68-E1:          [67-E1, 67-E2, 67-E3]             0.68    -
68-E2:          [67-E1]                           0.68    intel
68-F1:          [67-F2]                           0.68    yoga,gx10
68-F2:          [67-F1]                           0.68    yoga,gx10
68-G1:          []                                0.68    -
69-G1:          [68-G1]                           0.69    cpu,gx10
69-G2:          [69-G1]                           0.69    gx10
69-F1:          [68-F1, 68-F2]                    0.69    yoga,gx10
69-F2:          [69-F1]                           0.69    yoga,gx10
69-B1:          [68-B2, 68-D1]                    0.69    gx10,yoga
69-D1:          [69-B1, 69-F1]                    0.69    yoga
69-C1:          [68-C1, 68-C2]                    0.69    gx10
69-A1:          [68-A1]                           0.69    gx10,yoga,intel
69-E1:          [68-E1]                           0.69    -
70-G1:          [69-G2]                           0.70    gx10
70-G2:          [69-G1]                           0.70    -
70-F1:          [69-F2]                           0.70    yoga,gx10
70-B1:          [68-B1, 69-B1]                    0.70    gx10,yoga
70-C1:          [69-C1]                           0.70    gx10
70-D1:          [69-D1]                           0.70    yoga
70-A1:          [69-A1]                           0.70    -
70-E1:          [69-E1]                           0.70    -
```

Invariants (checked by the contract's `obligation_dag_is_acyclic_and_forward`): no cycle; every edge points to an equal or earlier train (a row never waits on a later train); every root names its train.

---

## §9. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Three trains in ten days on a queue that merges 3–8 PRs a day.** | §1.2 arithmetic; E first (67-E1/E2 are the first rows to board); D-5 slips scope, not dates |
| R2 | **The FULL tier leaves PRs while the nightly FULL run is red** (G17) | 67-E0 is a precondition; until it is green, D-1 is applied to the merge queue only and PRs keep today's tier — stated in the 0.67 epic, not assumed |
| R3 | **gx10 becomes a second SPOF as it takes fleet work** (D-4) | width 2, yield-to-training, the utilization instrument; the aarch64 release build has priority over guard work via `concurrency` ordering |
| R4 | **PR code on yoga before the runner group is restricted** | 67-D0 precedes 67-D1 in the DAG; `pr-gate.yml`'s `authorize` is already `pull_request_target` |
| R5 | **"Default NVIDIA Rust" read as "ship the slower kernel"** | D-2 defines default per layer; the operator can override in one epic comment and §11 records it |
| R6 | **Six `ci.yml` edits in one train** — each needs a web-UI merge click and only one PR may hold the integration-list line | batch the ci.yml rows (67-E1/E2/E3, 67-C1, 67-D1) into two PRs; stacked PRs get no CI, so run `gh workflow run ci.yml --ref <head>` on each |
| R7 | **Numbers in this document drift** | §0 carries commands; `check_no_claim_literals.sh` scans this directory; the drift gate checks every cited path; the next train's epic re-derives §0 rows it depends on |
| R8 | **Ratios quoted before a CONFORMANT row exists** | none is quoted without its `evidence/` path; F publishes only from CONFORMANT rows (70-F1) |
| R9 | **This review chain was one lane** (`/teamwork-preview`, §10) | every claim the lane makes is re-verified before it is accepted; the contract and drift gate do not depend on the review |

---

## §10. Review record — `agy /teamwork-preview`

*(Filled by PMAT-1097 Phase 3. Every finding is listed with its verdict: accepted-and-applied, refuted-with-evidence, or deferred-to-row. A lane's claim is a claim until re-run.)*

---

## §11. Amendments

| date | who (verbatim source) | change |
|---|---|---|
| 2026-09-10 | operator: *"override "kind:docs" block if needed"* | PMAT-1097 filed `kind:code` (contract + drift-gate extension ride with the spec) |
| 2026-09-10 | operator: *"(add these to priorities)"* — three P0 five-whys tickets | rows 67-A1 (P0-1 #3082), 67-C2 (P0-2 #3083), 67-E1/E2 (P0-3 #3084) |

