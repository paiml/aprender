# Branch triage — 2026-09-14

Standing operator rule: *"tickets, pull requests and branches that are not triaged are P0."*

**Universe, derived not listed.** `git ls-remote --heads origin` = 112 branches; branches that
have never had a PR of any state (`gh pr list --state all --limit 400`) = **35**. That set is
the subject here. It is recomputed by the command, never read from this file.

## Method — and why commit count was the wrong instrument

The obvious measure, `git rev-list --count origin/main..origin/<b>`, says 30 of these branches
carry unique commits. It is **wrong by construction**: this repo squash-merges from a merge
queue, so a branch whose content landed still shows every original commit as unique.

The measurement that decides is **path residue**: of the paths a branch changes against its
merge-base, how many do not exist on `main` at all.

```
tot=0; miss=0
for f in $(git diff --name-only origin/main...origin/<b>); do
    tot=$((tot+1))
    git cat-file -e origin/main:"$f" 2>/dev/null || miss=$((miss+1))
done
```

`miss = 0` ⟹ every file the branch touches already exists on `main` ⟹ the work landed by
another route and the branch is a duplicate. That reading is an upper bound on what is
genuinely unlanded, so a non-zero residue still has to be read, never assumed.

## Result

| disposition | n | branches |
|---|---|---|
| **DELETE** — merge-queue artifact | 4 | `gh-readonly-queue/main/pr-{3092,3099,3253,3267}-*` |
| **DELETE** — 0 commits ahead of main | 1 | `fix/apr-cli-doctests` |
| **DELETE** — 0 path residue (superseded) | 11 | `feat/prrev-{001,002,008,010,011}`, `docs/pr-review-skill-spec`, `prior-art/{2811,2836,2847}`, `PMAT-1098-67-pv-lane-fleet`, `split/2738-make-test-targets` |
| **DELETE** — residue is a fixture RENAME only | 6 | `feat/prrev-{003,004,005,006,007,009}` |
| **DELETE** — superseded by an OPEN PR | 2 | `agent/G-10c`, `agent/G-10-full` → **#3021** (`scripts/check_pmat_pinned.sh`) |
| **REVIEW** — carries work absent from main | 11 | below |

4 + 1 + 11 + 6 + 2 + 11 = **35**, the universe.

### The prrev stack: superseded, and the residue was a rename

`feat/prrev-001…011` + `docs/pr-review-skill-spec` are one stack, last touched 2026-08-30,
115 commits behind. Five rows show a residue of exactly one fixture directory,
`tests/fixtures/pr-review/row-07-honest-docs-only-all-not-triggered/`, which reads like a
discriminating test case that never landed.

It is not. Every one of those residues is a rename or a renumber:

```
branch                                        main
row-07-honest-docs-only-all-not-triggered  -> row-07-honest-docs-only-pmat-consulted
row-16-duplication-surface-...-pass        -> row-23-duplication-surface-...-pass
row-17-duplication-surface-...-degraded    -> row-24-duplication-surface-...-degraded
```

Counted properly:

```
fixture rows on main                113   (43 row-* + 70 q-*)
fixture rows on the branch tip       24
```

The PR-review subsystem shipped by another route and grew 4.7× more coverage than the stack
ever had. All 12 stack branches are safe to delete; the apparent gap was an artifact of
comparing directory NAMES instead of counting rows — the residue measure is sound, the
inference from a residue to a missing case is not.

### REVIEW — residue that is genuinely not on main

| branch | absent paths | what it is |
|---|---|---|
| `agent/R-5` | 9 | `.github/workflows/release-assets.yml`, `contracts/apr-publish-cascade-v1.yaml`, `contracts/apr-release-assets*` — **release-train surface**, and `main` has neither file. Relevant to APR-RELEASE-001 §4 T-3/T-4 |
| `prior-art/2803` | 13 | `crates/apr-cli/src/compute_latch.rs`, `crates/aprender-serve/src/infer/compute_resolution.rs` + PERF-062 evidence — the `--gpu is ignored` adoption killer |
| `prior-art/2821` | 4 | `crates/aprender-test-lib/src/perf_gate/tokenizer*.rs`, `scripts/check_band_client_tokenizer.sh` |
| `prior-art/2825` | 2 | `contracts/gpu-device-init-lock-liveness-v1.yaml`, `crates/aprender-compute/tests/gpu_cold_init_deadlock.rs` |
| `prior-art/2820` | 1 | `docs/specifications/APR-PERF-GATE-001-v2.2.md` |
| `agent/C0-3` | 11 | `contracts/work/GH-66{3..7}.cot.yaml` — chain-of-thought work contracts |
| `agent/G-11b` | 3 | `scripts/fleet_verify.sh`, one receipt |
| `agent/report-066` | 1 | `docs/reports/progress-report-066.md` |
| `split/2738-perf034-sampler-alloc` | 4 | `crates/aprender-serve/src/sampling_select*.rs` + PERF-034 tests |
| `feat/y1-7bgarbage` | 2 | `crates/aprender-serve/examples/perf059_pos_bisect.rs`, `scripts/perf059_band_ladder.sh` |
| `PMAT-1094-lint-json-one-shape` | 8 | **7 of 8 are throwaway**: `fix_derives{,2,3,4}.py`, `fix_dry{,_final}.py`, `fix_lint*.py` — scripts that were never meant to land. The only real file is `crates/apr-cli/src/commands/lint_json_outcome_shape_tests.rs` |

## What this does NOT claim

A path absent from `main` is **not** proof the work is wanted. `PMAT-1094` is the worked
example: 8 absent paths, 7 of them throwaway Python. Residue bounds the question; it does
not answer it. Every REVIEW row still needs a human-or-quorum read before anything is
deleted, and nothing in this pass deletes a branch — this is classification and linking only
(`kind:triage`: no diff outside `docs/audits/**`, `docs/roadmaps/roadmap.yaml`, `.quorum/**`).

## Ordering

The DELETE set is 24 branches and is safe to execute now except `agent/G-10{c,-full}`, which
should wait for **#3021** to land so the supersession is a fact rather than a forecast.

## Second bucket: 37 branches whose PR was CLOSED, never merged

Added 2026-09-15. **The universe above is built from the wrong side.** It asks *"which
branches never had a PR?"*, so a branch whose PR was opened and then **closed unmerged** is
excluded by construction — it *has* had a PR. That is precisely a branch carrying unlanded
work with no path forward, which is the question that matters.

Derived the same way, with the same instrument:

```
gh pr list --state all  --limit 500 --json headRefName -q '.[].headRefName' | sort -u  > ever
gh pr list --state open --limit 200 --json headRefName -q '.[].headRefName' | sort      > open
git ls-remote --heads origin | sed 's|.*refs/heads/||' | sort                           > all
comm -12 ever all | comm -13 open -     # had a PR, none open, branch still on origin
```

117 remote branches = 41 with an open PR + 38 that never had one + 37 here + `main`.
**Zero** branches are "merged with the branch left behind", so deletion-on-merge is working;
every one of these 37 is an abandoned PR. That is consistent with 29 of 146 PRs over the
last 10 days (**20% of production**) being closed unmerged.

| disposition | n |
|---|---|
| **DELETE** — 0 path residue or 0 commits ahead | 31 |
| **REVIEW** — carries paths absent from `main` | 6 |

31 + 6 = **37**, the universe.


### DELETE — content landed by another route

| branch | closed PR | age (d) | reason |
|---|---|---|---|
| `agent/R-0-amend` | #3003 | 8 | 0-residue |
| `feat/prrev-rung1-shadow-lane` | #2836 | 13 | 0-residue |
| `fix/prrev-control-cache` | #2847 | 12 | 0-residue |
| `fix/workspace-test-timeout-headroom` | #2811 | 14 | 0-residue |
| `PMAT-1096-cuda-asset-target-mountpoint` | #3098 | 3 | 0-ahead |
| `PMAT-1097-06x-release-schedule` | #3087 | 3 | 0-residue |
| `PMAT-1098-3126-mut-paths` | #3128 | 3 | 0-residue |
| `PMAT-1098-67-A1-four-apr-assets` | #3092 | 3 | 0-residue |
| `PMAT-1098-67-C1-D1-gpu-pr-jobs` | #3095 | 3 | 0-residue |
| `PMAT-1098-67-C2-self-hosted-preflight` | #3088 | 3 | 0-residue |
| `PMAT-1098-67-E3-guards-under-budget` | #3094 | 3 | 0-residue |
| `PMAT-1098-build-pool-any-of-three` | #3101 | 4 | 0-residue |
| `PMAT-1098-coverage-nightly-on-yoga` | #3123 | 3 | 0-residue |
| `PMAT-1098-gx10-arch-neutral-routing` | #3104 | 3 | 0-residue |
| `PMAT-1098-pp066-shallow-fetch` | #3108 | 4 | 0-residue |
| `PMAT-1098-qwen35-honest-refusal` | #3099 | 4 | 0-residue |
| `PMAT-1098-rustc-wrapper-hotfix` | #3107 | 4 | 0-residue |
| `PMAT-1098-spec-checklist-paths` | #3131 | 3 | 0-residue |
| `PMAT-1098-spec-conformance-locale` | #3109 | 3 | 0-residue |
| `PMAT-1100-triage-release-trains` | #3102 | 3 | 0-residue |
| `PMAT-1101-q5k-ggml-layout` | #3110 | 3 | 0-residue |
| `PMAT-1102-aarch64-lint` | #3112 | 3 | 0-residue |
| `PMAT-1104-cuda-q5k-gemv` | #3113 | 3 | 0-residue |
| `PMAT-1106-gpu-tests-skip` | #3116 | 3 | 0-residue |
| `PMAT-3121-examples-dogfood` | #3122 | 3 | 0-residue |
| `PMAT-3124-triage-labels` | #3125 | 3 | 0-residue |
| `PMAT-3228-always-latest-pin` | #3274 | 0 | 0-residue |
| `PMAT-3229-rustsec-2026-0285` | #3276 | 0 | 0-residue |
| `PMAT-952-wgpu-init-deadlock` | #2862 | 10 | 0-residue |
| `PMAT-953-stdio-write-error-clean-exit` | #2863 | 10 | 0-residue |
| `PMAT-955-sibling-devdeps-path-only` | #2864 | 10 | 0-residue |

### REVIEW — paths absent from `main`

| branch | closed PR | age (d) | residue |
|---|---|---|---|
| `bse/receipt-adhoc` | #3052 | 2 | 1/4 |
| `feat/client-tokenizer-counter` | #2821 | 14 | 4/19 |
| `feat/y6-provenance` | #2803 | 14 | 13/65 |
| `fix/wgpu-feature-graph-and-honest-compute-clas` | #2825 | 13 | 2/9 |
| `perf/tokenization-mismatch-fatal` | #2820 | 13 | 1/3 |
| `spec/performance-parity-llamacpp` | #2845 | 12 | 2/14 |

The same caveat applies as above: residue **bounds** the question, it does not answer it. A
closed PR is a decision someone made, so a REVIEW row here needs the close reason read before
anything is resurrected — a branch may be absent from `main` *because it was rejected*.

Nothing here deletes a branch either. Classification and linking only.

