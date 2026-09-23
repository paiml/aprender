# DEBT-RATCHET-001 — clearing 80% of measured debt across 0.70.0 → 0.74.0

**Status:** STEP 1 (quorum research): a plan for operator review. Nothing here is applied yet.
**Epic:** paiml/aprender#3997 · **Ticket:** PMAT-3997 · **kind:** docs

**Step 2 is out of scope** and waits for operator approval: filing child issues, moving milestones, closing
issues or PRs, and deleting branches or worktrees. This document **proposes** those actions. It performs none of them.

## 0. The operator's words (verbatim, from #3997)

> lets also declear a "rachet" to clean up technical debt lets assume 5 releaes to clear up 80% of debt. The issues; Code coverae needs to be closer to 95%, and we can use YOGA nightly CUDA coverage sharded, etc. B. we need actual "pv" contract enforcement at deepest level. C. We need to merge in the pv ontology spec fully. D. we need to purge the backlog of branches, tickets, pull requests and have all open tickets assigned to a release number. the unassigned queue is not allowed anymore, and open/stale pull requests are not allowed. First quorum research this, then apply in equal portions to .70, .71, .72, .73. .74

## 1. How to read this document

Each number here comes from a command in §7, run at the time stamped there. **The formulas are the plan; the numbers
are a sample of them.** When a re-measurement disagrees with a number here, the command wins, and the slice
thresholds are re-derived from the same formula, never edited by hand.

A ratchet has three parts, and every pillar below states all three:

- **Unit:** what one piece of debt is, as something you can count.
- **Baseline `B`:** the count today.
- **Floor `F_r`** for release `r ∈ {0.70, …, 0.74}`, with `k = 1..5`: `F_k = B − k · ⌈0.8·B / 5⌉` for counts that
  should fall, or `B + k · (0.8·gap / 5)` for percentages that should rise.

The gate at release `r` refuses **(a)** any value worse than `F_r`, and **(b)** any value worse than the value
measured at the previous release's tag, whichever is stricter. (b) makes it a ratchet. Without (b), overshooting in
0.70 would buy room to regress in 0.71.

## 2. Measured baselines (question 1)

Measured 2026-09-23 ~10:25–10:45Z. Trees: aprender `origin/main` @ `49fe19c28`; infra `origin/main` @ `29a84779`.

### A. Coverage

| Fact | Value | Source |
|---|---|---|
| Last **green** coverage measurement | **810,420 / 918,869 lines = 88.20%** (8,819 bp) | coverage-nightly run `33245815502`, `d1b7d1995`, 2026-08-29 |
| The figure #3997 quotes | 88.78% (2026-07-29) | CLAUDE.md. It is **stale**: coverage fell 0.58 pp since, and the integer floor `88` could not see the drop |
| Coverage-nightly since the last green | **38 failure / 10 cancelled / 12 success in the last 60 runs**. **0 green since 2026-08-29** (25 days) | `gh run list --workflow coverage-nightly.yml` |
| Why the latest run is red | `aprender-zram-core` `benchmark::tests::test_f060_no_performance_regression` panics (`benchmark.rs:470`). It is a timing assertion running under llvm-cov instrumentation, and it fails **before** `TOTAL:` is printed, so no number is produced | run `35800448700` |
| Enforced floor | `COV_FLOOR := 88`, **integer percent** (`LH*100/LF`, bash integer division) | `Makefile:508` |
| Denominator filter | `COVERAGE_EXCLUDE_REGEX` removes **415,335 physical lines** of 3,318,674 tracked `crates/*/src` lines (aprender-gpu excluded separately). **All of `apr-cli/` (258k lines)** is among them, plus `models/`, `format/converter`, `serialization/`, and others | `Makefile:492` + §7 A-4 |
| CUDA-dark code (not in the denominator at all) | `aprender-gpu` 113,786 lines (`--exclude aprender-gpu`), `aprender-cuda-edge` 5,244 lines, plus every `cfg(feature = "cuda")` path in serve/train/compute (the run has no `--features cuda`) | §7 A-5 |

**Implication:** "88% coverage" means 88% of a **filtered, CPU-only** population. The ratchet must pin that population,
or coverage can rise just by widening the exclude regex (a denominator ratchet: §3.A rule A-3).

### B. pv enforcement depth

| Fact | Value | Source |
|---|---|---|
| Contract files / parsed contracts | 1,882 yaml / 1,829 contracts, 3,250 equations, 3,849 obligations, 4,800 falsification tests, 1,967 Kani harnesses | `pv coverage` (pv 0.69.1) |
| Equations **bound to code** | **219 / 3,250 = 6.7%** (3,031 unbound) | `pv coverage --binding contracts/aprender/binding.yaml` |
| Contract call sites in source | **510**: **E0 = 267 (52%)**, E1 = 128, E2 = 115 | `pv coverage --binding … --enforcement crates/<c>`, summed over 22 crates |
| Meaning of E0/E1/E2 | E0 = generic `!is_empty` placeholder · E1 = domain pre-checks · E2 = pre + post checks (pv's own legend; quality weights 0.1/0.5/1.0) | pv output |
| Contracts with obligations but **zero falsifiers** | **19** (e.g. `ward-linkage-v1`, `tokenizer-v1`, `xtc-sampling-correctness-v1`) | `pv coverage`, rows `ob>0 ft=0` |
| `#[contract(…)]` attributes in source | 168 | literal count, §7 B-4 |
| Is `pv lint contracts/` a PR gate? | **No.** `make contracts` runs it, but no workflow invokes `make contracts`. CI runs `pv validate` only in `book.yml` (book completeness) | §7 B-5 |

### C. ONT-001 (paiml-ontology) rows bound

| Fact | Value | Source |
|---|---|---|
| Rows / bound / unbound | **27 / 13 / 14 (48%)** | infra `scripts/ont/precondition-lint.sh … --ledger …` at `29a84779` |
| Unbound rows by repo | aprender 12 · infra 1 (ONT-E, PR infra#909 unmerged) · paiml-mcp-agent-toolkit 1 (ONT-11) | §5 row headers |
| Unbound K̂ (the spec's own turn estimates, all `[U]`) | **1,230 turns** in total | row headers, summed |
| Blocked externally | ONT-4c4 ← aprender#3522 O-1 | the lint's `BLOCKED-EXTERNAL` line |

Unbound rows (K̂; depends_on): ONT-E 30 (P) · ONT-4c4 90 (4c1, 4b2) · ONT-4c 150 (4b, 2b) · ONT-4d 90 (2b, 4b) ·
ONT-2c 120 (2b, 0) · ONT-3a 90 (PVL EV-2, 6) · ONT-3b 90 (2b, 3a, PVL EV-8b) · ONT-5 120 (4, 6, PVL EV-9) ·
ONT-4e 120 (4, 5, 4d) · ONT-7 60 (1, 6) · ONT-8 60 (PVL EV-3, 1, 6) · ONT-9 90 (5, 6) · ONT-11 60 (6, PVL EV-15) ·
ONT-10 60 (0..9, D, 11, PVL EV-12).

### D. Backlog

| Fact | Measured | #3997 said | Source |
|---|---|---|---|
| Open issues | **723** | 715 | `gh issue list --state open` |
| No milestone | **128**, and **all** of them were created in the last 5 days (median age 0 d; 94 in the last 24 h) | 121 | same |
| In `backlog` milestone | **298** (median age 17 d, max 79 d, 35 older than 30 d) | 298 | same |
| **Not in a release milestone** (none, `backlog`, `Inference dispatch…`) | **428 / 723 (59%)** | — | same |
| Issue inflow | **487 issues created in the last 7 days** (~70/day) | — | `gh issue list --search created:>=2026-09-16` |
| Open PRs | **57** (9 draft, 32 with no milestone) | 57 | `gh pr list` |
| PRs by `updatedAt` idle | max 139 h; **0 idle > 7 d** | — | same |
| PRs by **creation age** | **22 older than 7 d** (oldest 10 d) | — | same |
| Live remote branches | **353** (`git ls-remote --heads origin`), **297 with no open PR** | **2,594** (not reproducible; the ls-remote count is 353) | §7 D-6 |
| Local branches (dev box, main checkout) | 3,334; **1,977 with upstream `[gone]`** | 3,328 | `git for-each-ref refs/heads` |
| Worktrees (dev box) | **651**, `.claude/worktrees` = **44 GB** | 646 | `git worktree list` |
| Release cadence | 0.70.0 due 09-26 · 0.71.0 09-29 · 0.72.0 10-02 · 0.73.0 / 0.74.0 **no due date** | — | milestones API |
| Milestone load | 0.70.0 has **176 open issues** and is due in 3 days | — | same |

**Two measured facts reshape pillar D:**

1. **The unmilestoned queue is inflow, not stock.** No unmilestoned issue is older than 5 days. Draining it once
   does nothing unless issues get a milestone as they are filed. So the gate belongs at **intake**.
2. **`updatedAt` cannot detect a stale PR.** Fleet sweeps (update-branch, labels, bots) keep touching PRs: 9 PRs
   share `updatedAt` = 68 h, and none are idle > 7 d, yet 22 are more than 7 days old. A stale-PR gate keyed on
   `updatedAt` could never fire. It must key on **creation age** or on the **head commit's author date**.

## 3. "80% of debt" as a countable unit, and the slices (questions 2 and 3)

Milestones: 0.70.0 = #7, 0.71.0 = #9, 0.72.0 = #10, 0.73.0 = #15, 0.74.0 = #16.

### A. Coverage: unit = basis points of line coverage over a **pinned** population

- **A-1 Unit.** `LH·10000 / LF` in basis points over the population `P₀`: `--workspace --exclude aprender-gpu --lib`,
  with `COVERAGE_EXCLUDE_REGEX` frozen at its `49fe19c28` value.
- **A-2 Debt.** `gap = 9500 − B`. With `B = 8819`, gap = 681 bp, 80% = 545 bp, slice = **109 bp per release**
  (≈ 10,000 newly covered lines at LF ≈ 918,869).
- **A-3 Denominator ratchet.** A PR that **adds** an alternative to `COVERAGE_EXCLUDE_REGEX` is refused. Removing one is
  allowed, and the removed population is reported as `P₀ ∪ Δ` alongside the gated figure. The gate stays on `P₀`, so
  honesty never turns the gate red.
- **A-4 Precision.** `COV_FLOOR` moves from integer percent to basis points. At integer precision a 0.58 pp regression
  (88.78 → 88.20) went unseen, which is 5.3 slices' worth of signal.
- **A-5 Precondition (0.70.0).** Coverage-nightly must be green again: 25 days with no number is a gate that cannot
  fail. Timing assertions such as `test_f060_no_performance_regression` must not run under instrumentation. They
  need a `cfg(not(coverage))` guard or a separate lane.
- **A-6 CUDA population `P_cuda`** (§4) is a **separate ratchet**. Its baseline is its first measurement, and it is
  never mixed into `P₀`. Folding 119k+ dark lines into `P₀` would lower the figure by several pp and turn the
  CPU ratchet red for reasons unrelated to any regression.

| Release | `P₀` floor (bp) | `P_cuda` | Gate |
|---|---|---|---|
| 0.70.0 | **8,928** (and coverage-nightly green ≥ 3 consecutive nights) | first sharded measurement recorded as `B_cuda`; report-only | `make coverage` with `COV_FLOOR_BP` |
| 0.71.0 | **9,037** | floor = `B_cuda` (no regression) | same + `P_cuda` merge report |
| 0.72.0 | **9,146** | `B_cuda + 0.8·(9500−B_cuda)·2/5`, armed | both |
| 0.73.0 | **9,255** | `+ 1 slice` | both |
| 0.74.0 | **9,364** | `+ 1 slice` | both |

If 0.70's re-measurement differs from 8,819, every row is recomputed by A-2 from the new `B`. **Feasibility flag:**
releases are 3 days apart, so each slice is about 10k covered lines in 3 days. §6 asks the operator.

### B. pv: unit = **enforced call site-grade**, plus bound equations

The operator asked for "actual pv contract enforcement at deepest level". pv already grades depth (E0/E1/E2).
Three countable units, all read from `pv coverage`, never hand-counted:

| Unit | Baseline | 80% target at 0.74 | Per release |
|---|---|---|---|
| **B-1** E0 call sites (placeholder `!is_empty`) upgraded to ≥ E1 or deleted | 267 | ≤ 52 left | −43 |
| **B-2** Contracts with obligations and zero falsifiers | 19 | 0 (the count is small, so this goes to 100%) | −4 (−3 at 0.74) |
| **B-3** Bound equations (`Binding implemented`) | 219 / 3,250 | **operator decision (§6)**: 80% of the 3,031 unbound is ~485 bindings per release, which is not credible in 3-day releases. The proposal: 80% of the equations whose contracts **name an in-tree function**, a denominator `pv coverage --reverse` can derive | derived |
| **B-4** `pv lint contracts/` is a **required PR check** | not wired | wired and green by 0.70.0 | one-time |

Gate: `pv coverage --binding contracts/aprender/binding.yaml --enforcement <crate>` per crate. The release gate refuses
an E0 count above the floor, a falsifier-less-contract count above the floor, or a bound-equation count below it.

### C. ONT-001: unit = **rows bound** (a ledger row with a non-null `merged_sha`)

- End state is **27 / 27**. The operator said "fully", so pillar C targets 100%, not 80%.
- 14 unbound rows = **1,230 K̂**, about **246 K̂ per release**. The order is the spec's own selector (R-24): each slice is
  the next rows whose `depends_on` are all bound, cut at ~246 K̂. The table below is a projection only. The selector
  picks each row live.

| Release | Rows (projected, the selector decides) | Bound floor |
|---|---|---|
| 0.70.0 | ONT-E (infra, 30), ONT-4c (150), ONT-7 (60) | ≥ 16 |
| 0.71.0 | ONT-4d (90), ONT-8 (60), ONT-3a (90) | ≥ 19 |
| 0.72.0 | ONT-2c (120), ONT-3b (90), ONT-11 (pmat, 60) | ≥ 22 |
| 0.73.0 | ONT-5 (120), ONT-4c4 (90, if aprender#3522 O-1 lands) | ≥ 24 |
| 0.74.0 | ONT-4e (120), ONT-9 (90), ONT-10 (60, the release row) | = 27 |

Gate: the infra lint's `bound=` field, read at the aprender release tag. infra-83 owns the spec. This plan
**consumes** its selector and does not reorder rows.

### D. Backlog: unit = **open issues outside a release milestone**, plus stale PRs and dead remote branches

The operator's words for D are absolute ("the unassigned queue is not allowed anymore", "open/stale pull requests
are not allowed"), so D targets **zero**, not 80%. The slices below drain the **stock**. The intake gates (§5) stop
the **inflow** from 0.70.0 on.

| Unit | Baseline | 0.70 | 0.71 | 0.72 | 0.73 | 0.74 |
|---|---|---|---|---|---|---|
| **D-1** open issues not in a release milestone (older than the 24 h grace) | 428 | ≤ 342 | ≤ 256 | ≤ 170 | ≤ 84 | **0** |
| **D-2** open PRs older than 7 d with no head commit in 72 h | 22 | ≤ 17 | ≤ 12 | ≤ 7 | ≤ 2 | **0** |
| **D-3** live remote branches with no open PR | 297 | ≤ 237 | ≤ 178 | ≤ 118 | ≤ 59 | **0** (except `main`, `release/*`) |
| **D-4** local branches `[gone]` + worktrees (host hygiene, not a repo gate) | 1,977 + 651 | reported per host; purged by a host janitor, not a release gate | | | | |

D-1 can be drained by **triage** (assign a release milestone, or close with a citation). Triage is a
`paiml-implement kind=triage` run, not code work.

## 4. Sharded CUDA coverage on yoga (+ lambda, gx10) (question 4)

**What exists today:** coverage-nightly runs on `[clean-room, yoga]` (CPU, 60–70 min, 150 min timeout). cuda-nightly
has a yoga job (`[gpu, yoga, X64, cuda, ada]`, 30 min) and a gx10 job (90 min). Per #3986's comment, yoga's RTX 4060
Laptop is **sm_89** (the same architecture as lambda's 4090) and was **idle** while lambda's lock queued 6 jobs.

**Design:**

1. **Build outside the lock, once per host.** `cargo llvm-cov nextest --no-report --features cuda` needs instrumented
   binaries. Build them with `cargo nextest archive` (or `llvm-cov show-env` + `cargo build --tests`) **before** taking
   the GPU lock. #3986 P1 measured 1.5 min of compiling inside a 25-min lock hold.
2. **Shard by nextest partition, and take one lock per shard.** `--partition hash:i/N` over
   `-p aprender-gpu -p aprender-cuda-edge -p aprender-serve -p aprender-train -p aprender-compute --features cuda`.
   Each shard is its own `gpu-q --prio 1 -- <nextest run --partition hash:i/N>` call, so the lock is held only while
   GPU tests run, and short shards benefit from shortest-job-first (P2, which cut waits from 25 min to 2 min).
   **Never** wrap a lock-taking script in gpu-q (flock is not re-entrant, so it self-deadlocks). The shard command is a
   plain nextest invocation.
3. **Shard count: derived, not guessed.** `N = ⌈T_instr / 10 min⌉`, where `T_instr` is the first **unsharded**
   instrumented run's wall time on yoga. `[U]` until measured. The one data point is uninstrumented: `aprender-serve
   --features cuda --lib` held the lock for 25 min on lambda. llvm-cov overhead is typically 1.5–3×, which gives
   `N ≈ 4–8` for serve alone. That is an estimate, not a measurement.
4. **Merge within a host by profdata, across hosts by LCOV.** `.profraw` merges only across **identical** instrumented
   binaries, so all shards on one host come from one archive → `llvm-profdata merge` → one LCOV. Across hosts (yoga
   x86 sm_89, lambda x86 sm_89, gx10 ARM64 sm_121) the binaries differ, so the merge is the **LCOV union at one pinned
   SHA**: per `(file, line)`, covered if any host covered it. The job refuses to merge LCOVs from different SHAs.
5. **Host roles.** yoga is the nightly home (idle card, already the coverage host). gx10 adds the ARM64 / sm_121
   (Blackwell) paths. lambda is **not** in the nightly rota: it is the release-evidence host, and #3986 measured it as
   the contended one. It joins only by operator call.
6. **Freeze rule.** Nothing here touches a GPU host's queue before the 0.69.1 freeze (13:00Z 2026-09-23). The first
   unsharded measurement run (item 3) is scheduled after the 0.69.1 tag.
7. **Output.** `P_cuda` = the lines of `aprender-gpu`, `aprender-cuda-edge`, and `cfg(feature = "cuda")` code, reported
   as its own figure (§3.A A-6), plus the union LCOV uploaded as an artifact keyed by SHA.

## 5. Pillar-D refusal gates, each with a first-green proof (question 5)

**A gate that has never been green on its target is not a gate.** Both gates therefore land in two stages:
**report-only with a ratchet on the count** (green on the real repo from day one, because the count must only fall),
then **armed at zero** once the stock reaches 0 at 0.74.0.

### G-D1: the unmilestoned-issue gate (`scripts/check_issue_milestones.sh`)

- **Universe:** `gh issue list --state open --json number,milestone,createdAt`, derived and never cached.
- **Violation:** an open issue older than **24 h** whose milestone is not a release (`^[0-9]+\.[0-9]+\.[0-9]+$`).
  The 24 h grace exists because 94 of the 128 unmilestoned issues are under a day old. The cop's triage duty assigns
  them within the day.
- **Where it refuses:** (i) the **release train's T-5 reconcile** (already a hard gate) fails when the count exceeds the
  slice's floor; (ii) an hourly scheduled run posts the list. GitHub cannot refuse the creation of an issue, so the
  decision surfaces are the release and the cop, not issue creation.
- **First-green proof:** run against paiml/aprender today in ratchet mode, with `--max 428`: PASS at 428 (the real
  target). Negative control: `--max 427`: FAIL, naming the issues. Case table: an issue in `0.70.0` passes; in `backlog`
  it fails; with no milestone at 23 h it passes; at 25 h it fails; a closed issue is ignored; a PR is ignored (the
  issues API returns PRs, and they must be filtered out).

### G-D2: the stale-PR gate (`scripts/check_stale_prs.sh`)

- **Violation:** an open PR more than **7 d** old (`createdAt`) whose **head commit author date** is more than **72 h**
  ago. `updatedAt` is not used (§2.D, fact 2).
- **Where it refuses:** the release's T-5 reconcile. PRs are never closed automatically. Closing is a step-2 action,
  quorum-or-operator only.
- **First-green proof:** ratchet mode with `--max 22` on the live PR list: PASS; `--max 21`: FAIL. Case table: a 10-day
  PR with a commit 2 h ago passes; a 10-day PR with a commit 4 days ago fails; a 5-day PR passes; a draft is counted
  the same (draft is not an exemption); a PR bumped only by `update-branch` still fails (the merge commit's author is
  the bot, which is excluded).

Both scripts are **step 2**. This document specifies them, and their case tables become the tests.

## 6. Decisions for the operator (not taken here)

1. **Pillar-A feasibility:** ~10k newly covered lines per 3-day release. Keep equal slices, or slice by time rather than
   by release?
2. **Pillar A, the exclude regex:** 415k physical lines (all of `apr-cli`) sit outside the denominator. Keep gating on
   `P₀` and report `P_full`, or make shrinking the exclude list its own unit?
3. **B-3 denominator:** all 3,031 unbound equations, or only those naming an in-tree function?
4. **D targets zero**, per the operator's words, rather than 80%. Confirm.
5. **What "refused" means for an issue or a PR:** blocking the release (proposed), a label, or closing.
   Closing is never automated without a quorum.
6. **0.70.0 is due 2026-09-26 with 176 open issues.** Should step 2's triage move issues forward out of 0.70.0, or only
   assign the unassigned ones?

## 7. Commands (re-run these; the numbers above are their output on 2026-09-23)

```bash
# A
gh run list --workflow coverage-nightly.yml --limit 60 --json conclusion,createdAt,headSha           # A-1 history
gh run view 33245815502 --log | grep -E 'TOTAL: [0-9]+/[0-9]+'                                        # A-2 last green
gh run view 35800448700 --log | grep -E 'panicked|FAILED'                                             # A-3 why red
git ls-files 'crates/**/src/**/*.rs' | grep -v ^crates/aprender-gpu/ | grep -E "$COVERAGE_EXCLUDE_REGEX" | xargs cat | wc -l   # A-4
for c in aprender-gpu aprender-cuda-edge; do find crates/$c -path '*src*' -name '*.rs' | xargs cat | wc -l; done              # A-5
# B
pv coverage                                                                                            # B-1 totals, ft=0 rows
pv coverage --binding contracts/aprender/binding.yaml --quiet | grep -A9 Totals                        # B-2 bound
for c in crates/*/; do pv coverage --binding contracts/aprender/binding.yaml --enforcement "$c" --quiet | grep -c '\[E0\]'; done   # B-3
git ls-files 'crates/**/*.rs' | xargs grep -hE '^\s*#\[(\w+::)?contract\(' | wc -l                   # B-4
grep -n 'make contracts' .github/workflows/*.yml                                                      # B-5 (no hits)
# C (in an infra worktree at origin/main)
bash scripts/ont/precondition-lint.sh docs/specifications/paiml-ontology.md --ledger docs/audits/ONT-001/ledger.jsonl
# D
gh issue list --state open --limit 2000 --json number,milestone,createdAt                             # D-1..3
gh pr list --state open --limit 500 --json number,createdAt,updatedAt,isDraft,milestone,headRefName   # D-4,5
git ls-remote --heads origin | wc -l                                                                  # D-6
git for-each-ref refs/heads --format='%(upstream:track)' | sort | uniq -c                             # D-7
git worktree list --porcelain | grep -c '^worktree '                                                  # D-8
gh api 'repos/paiml/aprender/milestones?state=all&per_page=100'                                       # D-9
```

## 8. Quorum

This plan is grilled by a width-3 agy quorum (`--mode grillme`) before it goes to the operator. The lanes' verdicts
and the changes they forced are recorded in §9.

## 9. Quorum record

_Filled after the quorum returns._
