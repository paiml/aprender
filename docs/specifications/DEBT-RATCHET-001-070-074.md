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
| Unbound K̂: **not a measurement.** These are the spec's own turn estimates, every one `[U]` | **1,230 turns `[U]`** in total | row headers, summed |
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
| PRs by **creation age** and **head commit** `max(authoredDate, committedDate)` | **20 older than 7 d; 19 of them with no head commit in 72 h** (oldest 23 d). **25 of 56 PR heads are 2-parent merge commits**, most committed by `GitHub` (update-branch) | — | GraphQL, §7 D-5b (re-measured after the quorum) |
| Live remote branches | **353** (`git ls-remote --heads origin`), **297 with no open PR**; **40** of those also have a tip older than 14 d, which is exactly `check_reconcile.sh` R3's definition | **2,594** (not reproducible; the ls-remote count is 353) | §7 D-6 |
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
- **A-3 Denominator ratchet (two holes, both closed).** (i) A PR that **adds** an alternative to
  `COVERAGE_EXCLUDE_REGEX` is refused. (ii) **Moving code into an already-excluded path** games the regex without
  touching it. So `P₀` is also pinned as a **file list**: the tracked `.rs` files that the regex excluded at the
  baseline SHA. An excluded file that is not on that list (new, or moved in) counts **in** the denominator. Removing a
  regex alternative is allowed.
- **A-3b The full figure is reported, not hidden.** Every coverage run also prints `P_full`, which is `P₀` with the
  regex emptied. The regex excludes 415k physical lines, all of `apr-cli` among them, so `P₀` alone overstates
  coverage. Whether shrinking the regex becomes a gated unit is operator decision 2 (§6). Until it is ruled on,
  `P_full` is recorded at every tag and never gated.
- **A-4 Precision (a step-2 change; not in the tree today).** `Makefile:508` gates on `COV_FLOOR := 88`, an integer
  percent. `COV_FLOOR_BP` **does not exist yet**, and adding it is the first 0.70 child issue. At integer precision a
  0.58 pp regression (88.78 → 88.20) went unseen, which is 5.3 slices' worth of signal. An empty or zero `LF` already
  fails closed (`Makefile:628` sets `COV_PCT=0`). The BP version must keep that.
- **A-5 Precondition (0.70.0).** Coverage-nightly must produce a number again: 25 days without one. The fix is to keep
  timing assertions such as `test_f060_no_performance_regression` (`aprender-zram-core/src/benchmark.rs:470`) out of
  instrumented runs, with `cfg(not(coverage))` or a separate lane. The 0.70 requirement is **one green
  coverage-nightly run on the release SHA**. Three consecutive nights cannot be met by 09-26.
- **A-6 CUDA population `P_cuda`** (§4) is a **separate ratchet**. Its baseline is its first measurement, and it is
  never mixed into `P₀`. Folding 119k+ dark lines into `P₀` would lower the figure by several pp and turn the
  CPU ratchet red for reasons unrelated to any regression.

| Release | `P₀` floor (bp) | `P_cuda` | Gate |
|---|---|---|---|
| 0.70.0 | **8,928** (plus one green coverage-nightly on the release SHA) | not measured yet | `make coverage` with `COV_FLOOR_BP` (to be built, A-4) |
| 0.71.0 | **9,037** | first sharded measurement = `B_cuda`; report-only | same + the `P_cuda` merge report |
| 0.72.0 | **9,146** | floor `B_cuda + 1·s_cuda` | both |
| 0.73.0 | **9,255** | `B_cuda + 2·s_cuda` | both |
| 0.74.0 | **9,364** | `B_cuda + 3·s_cuda` | both |

`s_cuda = 0.8·(9500 − B_cuda)/5`. **`P_cuda` reaches only 3 of its 5 slices by 0.74**, because nothing can be ratcheted
before a population has been measured, and the first GPU measurement cannot run before the 0.69.1 freeze ends. Either
its window runs to 0.76 (5 equal slices, 0.72–0.76), or the last three releases take 5/3 of a slice each. This is
operator decision 7 (§6). The plan does not pretend `P_cuda` fits equal slices inside 0.70–0.74.

If 0.70's re-measurement differs from 8,819, every row is recomputed by A-2 from the new `B`. **Feasibility flag:**
releases are 3 days apart, so each slice is about 10k covered lines in 3 days. §6 asks the operator.

### B. pv: unit = **enforced call site-grade**, plus bound equations

The operator asked for "actual pv contract enforcement at deepest level". pv already grades depth (E0/E1/E2).
Three countable units, all read from `pv coverage`, never hand-counted:

**Deepest means E2.** The unit is the E2 call site (pre + post), not "no longer E0". Upgrading only to E1, or
deleting a check, does not reduce the debt. Hence the second clause of B-1: the total number of call sites (pv's
penetration numerator) may never fall.

| Unit | Baseline | 0.70 | 0.71 | 0.72 | 0.73 | 0.74 |
|---|---|---|---|---|---|---|
| **B-1** E2 call sites (debt = sites below E2 = 267 E0 + 128 E1 = **395**; 80% = 316; ⌈316/5⌉ = 64 per release). **Total call sites never fall below 510** | 115 | ≥ 179 | ≥ 243 | ≥ 307 | ≥ 371 | ≥ 435 |
| **B-2** Contracts with obligations and zero falsifiers (small count, so this goes to 100%; ⌈19/5⌉ = 4) | 19 | ≤ 15 | ≤ 11 | ≤ 7 | ≤ 3 | 0 |
| **B-3** Bound equations, **full denominator** (unbound 3,031; 80% = 2,425; ⌈2,425/5⌉ = 485 per release). **Not credible in 3-day releases `[U]`: see operator decision 3** | 219 | ≥ 704 | ≥ 1,189 | ≥ 1,674 | ≥ 2,159 | ≥ 2,644 |
| **B-4** `pv lint contracts/` is a **required PR check** | not wired | wired and green | kept | kept | kept | kept |

B-3's alternative denominator, "equations whose contract names an in-tree function", **cannot be derived with today's
pv**. `pv coverage --reverse <crate>` lists unbound **pub fns** (1,358+ in `aprender-core` alone), which is the
code-side view, not the contract-side one. Deriving the alternative needs a pv change (a step-2 child issue), so the
table carries the full-denominator numbers until the operator rules.

Gate: `pv coverage --binding contracts/aprender/binding.yaml --enforcement <crate>`, summed over every crate with
`src/`. The release gate refuses an E2 count below the floor, a total call-site count below 510, a falsifier-less
contract count above the floor, or a bound-equation count below the floor.

### C. ONT-001: unit = **rows bound** (a ledger row with a non-null `merged_sha`)

- End state is **27 / 27**. The operator said "fully", so pillar C targets 100%, not 80%.
- 14 unbound rows = **1,230 K̂ `[U]`** (estimates, not measurements), about **246 K̂ `[U]` per release**. The **gate
  unit is rows**, not K̂: K̂ only balances the slices, and the floors below are row counts. The order is the spec's own selector (R-24): each slice is
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
Every floor is `B − k·⌈B/5⌉`, with the last slice clamped to 0.

| Unit | Baseline | 0.70 | 0.71 | 0.72 | 0.73 | 0.74 |
|---|---|---|---|---|---|---|
| **D-1** open issues not in an **open release milestone ≥ the current release** (24 h grace); ⌈428/5⌉ = 86 | 428 | ≤ 342 | ≤ 256 | ≤ 170 | ≤ 84 | **0** |
| **D-2** open PRs older than 7 d with no author activity in 72 h (G-D2 key); ⌈19/5⌉ = 4 | 19 | ≤ 15 | ≤ 11 | ≤ 7 | ≤ 3 | **0** |
| **D-3** live remote branches with no open PR (excluding `main`, `release/*`); ⌈297/5⌉ = 60 | 297 | ≤ 237 | ≤ 177 | ≤ 117 | ≤ 57 | **0** |
| **D-4** local branches `[gone]` + worktrees (host hygiene, not a repo gate) | 1,977 + 651 | reported per host; purged by a host janitor, not a release gate | | | | |

D-3 is deliberately wider than `check_reconcile.sh` R3 ("no open PR **and** tip older than 14 d", 40 today). R3 stays
as it is, and D-3 is the ratchet on the full stock.

D-1 can be drained by **triage** (assign a release milestone, or close with a citation). Triage is a
`paiml-implement kind=triage` run, not code work.

## 3.E The 0.75.0 slice: the ratchet does not stop at 0.74 (operator ruling, 2026-09-23)

Relayed on #3997 (comment 2026-09-23T10:46Z), operator's words: *"ALL releases in .7 have some rachet"*. The ≥80%-by-0.74
target stands. **0.75.0 carries a 6th slice: no regression, plus continued paydown.** Every later 0.7x release does the
same. The rule is the one in §1: each release refuses a level below the previous release's tag, and a pillar still
above zero keeps paying down at its 0.70–0.74 rate.

| Pillar | 0.74 floor | 0.75 floor | Basis |
|---|---|---|---|
| A: P₀ bp | 9,364 | **≥ 9,473** | one more 109 bp slice (reaches 100% of the gap only at 0.76: 9,500) |
| A: `P_cuda` | `B_cuda + 3·s_cuda` | **`B_cuda + 4·s_cuda`** | continues its own window (decision 7) |
| B-1: E2 call sites | 435 | **≥ 499** | one more 64-site slice |
| B-2: contracts with no falsifier | 0 | **0** | hold |
| B-3: bound equations | 2,644 `[U]` | **≥ 3,129 `[U]`** | pending decision 3 |
| C: ONT rows bound | 27 | **27** | hold (and every row added to the spec after 0.74 must bind in the release that adds it) |
| D-1 / D-2 / D-3 | 0 / 0 / 0 | **0 / 0 / 0** | hold, armed at zero |

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
   binaries, so all shards on one host come from one archive → `llvm-profdata merge` → one LCOV. The report step names
   its packages explicitly (`-p aprender-gpu -p aprender-cuda-edge -p …`): an unscoped two-phase `cargo llvm-cov report`
   reports on the root facade and prints 0/0 (`Makefile:521-532` documents this trap). Across hosts (yoga x86 sm_89,
   gx10 ARM64 sm_121, and lambda only by operator call) the binaries differ, so the merge is the **LCOV union at one
   pinned SHA**:
   - **LF** = the union of instrumented `(file, line)` over the hosts. A line that exists only on one architecture
     (`cfg(target_arch)`) counts once.
   - **LH** = the lines hit on **any** host.
   - The merge refuses LCOVs from different SHAs.
   - The merge also prints per-host LH/LF, so an architecture-only regression stays visible.

   No LCOV-union tool exists in the tree. It is a step-2 child issue with a case table (the same line hit on one host
   only; an x86-only line; an ARM-only line; a SHA mismatch is refused; an empty LCOV is refused).
5. **Host roles.** yoga is the nightly home (idle card, already the coverage host). gx10 adds the ARM64 / sm_121
   (Blackwell) paths. lambda is **not** in the nightly rota: it is the release-evidence host, and #3986 measured it as
   the contended one. It joins only by operator call.
6. **Freeze rule.** Nothing here touches a GPU host's queue before the 0.69.1 freeze (13:00Z 2026-09-23). The first
   unsharded measurement run (item 3) is scheduled after the 0.69.1 tag, so `B_cuda` first exists in the 0.71 window
   (§3.A).
7. **Output.** `P_cuda` = the lines of `aprender-gpu`, `aprender-cuda-edge`, and `cfg(feature = "cuda")` code, reported
   as its own figure (§3.A A-6), plus the union LCOV uploaded as an artifact keyed by SHA.

## 5. Pillar-D refusal gates, each with a first-green proof (question 5)

**A gate that has never been green on its target is not a gate.** Both gates therefore land in two stages:
**ratchet mode** (the count may only fall to the slice's floor, so the gate is green on the real repo on the day it
lands), then **armed at zero** at 0.74.0.

**Where they live:** they are **new predicates R6 and R7 in `scripts/check_reconcile.sh`**, the T-5 reconcile that is
already a hard release gate. They are not new scripts. They reuse its file-fed predicate functions and its `--self-test`
fixture pattern, so the self-test never touches the network. R3 (dead branches) and R4 (dirty stale PRs) already exist
there and are unchanged. GitHub cannot refuse the **creation** of an issue or a PR, so the decision surfaces are
(i) the release, via T-5, and (ii) the cop's triage duty, via an hourly report. Neither closes anything. Closing is a
step-2 action, quorum-or-operator only.

**What is and is not proven today.** Neither predicate exists yet, so **no first-green proof has been run**. What step 1
measured is the count each predicate will read on its first run (below). The first-green proof is the **acceptance
test of the step-2 child issue**, stated here so that it cannot be weakened later.

### R6: the unmilestoned-issue predicate (G-D1)

- **Universe:** `gh issue list -R paiml/aprender --state open --json number,milestone,createdAt` (the issues endpoint,
  which never returns PRs), derived on each run and never cached.
- **Violation:** an open issue older than **24 h** whose milestone is **not an open release milestone at or after the
  current release**. A closed or past release (0.66.0 … 0.69.1) does not satisfy it, so an issue cannot be parked
  there. Grace: 94 of the 128 unmilestoned issues are under a day old, and the cop's triage assigns them within the day.
- **Count on its first run** (step-1 measurement, not a proof): **428**.
- **Acceptance (first-green proof, step 2):** live run with `--max 428` (or the slice's floor at that moment): exit 0.
  Live run with `--max <count−1>`: exit 1, naming the issues. Self-test case table:
  - an issue in `0.70.0`: pass;
  - in `backlog`: fail;
  - in closed `0.68.0`: fail;
  - no milestone at 23 h: pass;
  - no milestone at 25 h: fail;
  - a closed issue: ignored.

### R7: the stale-PR predicate (G-D2)

- **Key:** PR age from `createdAt` > **7 d**, **and** no **author activity** in **72 h**. Author activity is
  `max(authoredDate, committedDate)` of the newest head commit that is **not** a 2-parent merge committed by `GitHub`.
  Measured: 25 of 56 PR heads are exactly such update-branch merges. `updatedAt` is not used (§2.D, fact 2). Taking
  the max of the two dates keeps a rebase or an amend from reading as stale, because a rebase rewrites the committer
  date.
- **Count on its first run** (step-1 measurement over head commits, before the merge-commit exclusion): **19**
  (of 20 PRs older than 7 d).
- **Acceptance (step 2):** live run with `--max 19`: exit 0; with `--max 18`: exit 1. Self-test case table:
  - a 10-day PR with a real commit 2 h ago: pass;
  - a 10-day PR whose only commit in 72 h is a `GitHub` update-branch merge: fail;
  - a 10-day PR rebased 2 h ago (old author date, new committer date): pass;
  - a 5-day PR: pass;
  - a draft: counted the same.

## 6. Decisions for the operator (not taken here)

1. **Pillar-A feasibility:** ~10k newly covered lines per 3-day release. Keep equal slices, or slice by time rather than
   by release?
2. **Pillar A, the exclude regex:** 415k physical lines (all of `apr-cli`) sit outside the denominator. Keep gating on
   `P₀` and report `P_full` (proposed), or make shrinking the exclude list its own gated unit?
3. **B-3 denominator:** all 3,031 unbound equations (485 bindings per release, which this plan judges not credible), or
   a subset whose denominator needs a pv change first?
4. **D targets zero**, per the operator's words, rather than 80%. Confirm.
5. **What "refused" means for an issue or a PR:** a T-5 reconcile failure that blocks the release (proposed), a label,
   or closing. Closing is never automated without a quorum.
6. **0.70.0 is due 2026-09-26 with 176 open issues.** Should step 2's triage move issues forward out of 0.70.0, or only
   assign the unassigned ones?
7. **`P_cuda` cannot fit 5 equal slices inside 0.70–0.74**, because its baseline first exists at 0.71. Extend its
   window to 0.76, or compress it into 3 larger slices?

## 7. Commands (re-run these; the numbers above are their output on 2026-09-23)

```bash
# A
gh run list -R paiml/aprender --workflow coverage-nightly.yml --limit 60 --json conclusion,createdAt,headSha   # A-1 history
gh run view -R paiml/aprender 33245815502 --log | grep -E 'TOTAL: [0-9]+/[0-9]+'                     # A-2 last green
gh run view -R paiml/aprender 35800448700 --log | grep -E 'panicked|FAILED'                          # A-3 why red
git ls-files 'crates/**/src/**/*.rs' | grep -v ^crates/aprender-gpu/ | grep -E "$COVERAGE_EXCLUDE_REGEX" | xargs cat | wc -l   # A-4
for c in aprender-gpu aprender-cuda-edge; do find crates/$c -path '*src*' -name '*.rs' | xargs cat | wc -l; done              # A-5
# B
pv coverage                                                                                            # B-1 totals, ft=0 rows
pv coverage --binding contracts/aprender/binding.yaml --quiet | grep -A9 Totals                        # B-2 bound
for c in crates/*/; do pv coverage --binding contracts/aprender/binding.yaml --enforcement "$c" --quiet | grep -cE '\[E[012]\]'; done   # B-3 E0/E1/E2
git ls-files 'crates/**/*.rs' | xargs grep -hE '^\s*#\[(\w+::)?contract\(' | wc -l                   # B-4
grep -n 'make contracts\|pv lint' .github/workflows/*.yml                                             # B-5 (comment lines only)
# C (in an infra worktree at origin/main)
bash scripts/ont/precondition-lint.sh docs/specifications/paiml-ontology.md --ledger docs/audits/ONT-001/ledger.jsonl
# D
gh issue list -R paiml/aprender --state open --limit 2000 --json number,milestone,createdAt           # D-1..3
gh pr list -R paiml/aprender --state open --limit 500 --json number,createdAt,updatedAt,isDraft,milestone,headRefName   # D-4,5
gh api graphql -f query='…pullRequests(states:OPEN){nodes{number createdAt commits(last:1){nodes{commit{authoredDate committedDate committer{name} parents{totalCount}}}}}}'   # D-5b
git ls-remote --heads origin | wc -l                                                                  # D-6
git for-each-ref refs/heads --format='%(upstream:track)' | sort | uniq -c                             # D-7
git worktree list --porcelain | grep -c '^worktree '                                                  # D-8
gh api 'repos/paiml/aprender/milestones?state=all&per_page=100'                                       # D-9
```

## 8. Quorum

This plan was grilled by a width-3 agy quorum (`--mode grillme`) before going to the operator. §9 records the result.

## 9. Quorum record

**Lanes:** `gemini-3.1-pro-high` → PASS-with-changes (8 must-fix) · `gemini-3.7-flash-high` → PASS ·
`gemini-3.8-flash-high` → PASS-with-changes (14 must-fix). **One family only.** The gpt-oss seat hit a 429 with a 95 h
reset (the openai/claude agy pool is out until about 2026-09-27) and fell back to gemini. Claude-family lanes are refused
for a Claude-authored plan. **Every lane exited 3**: other fleet sessions moved shared refs and rewrote the shared
`.git/config` during the window. The lanes were sandboxed, their clones came back byte-identical, and the tree witness
verified on all three. The verdicts are recorded as **advisory**, not as a quorum PASS. agy conversations:
`49c4b7a2-da9b-44fe-b37e-950b1740267c`, `01b9f5a5-dbff-4e75-9478-4a0bd89c3753`, `247bdc1b-5f20-46b9-a3f6-4ee7e344eb3e`.

**Changes the quorum forced** (each re-checked against the tree before it was applied):

| # | Finding | Lanes | Change |
|---|---|---|---|
| 1 | B-1 accepted E1 or deletion as "deepest" | 1, 3 | B-1 is now an **E2** ratchet, and the total call-site count may never fall |
| 2 | G-D2's author-date key calls a rebased PR stale, and update-branch merges look like activity | 1, 3 | key = `max(authoredDate, committedDate)`, excluding `GitHub` 2-parent merges; baseline re-measured by GraphQL: **19**, not 22 |
| 3 | The frozen regex still allows gaming (move code into an excluded directory) | 1, 3 | A-3 pins the excluded **file list** too; A-3b reports `P_full` |
| 4 | First-green proofs were written as done, but the scripts do not exist | 3 | §5 now says **no proof has been run**; they are step-2 acceptance tests, and the counts are labelled step-1 measurements |
| 5 | The gates ignored the existing `check_reconcile.sh` (R1–R5) | 3 | the gates are now predicates **R6/R7** in that script |
| 6 | `COV_FLOOR_BP` does not exist | 3 | A-4 says so; it is the first 0.70 child issue |
| 7 | 3 green nights by 09-26 fails on day one | 3 | 0.70 requires one green run on the release SHA |
| 8 | `P_cuda` reached only 4/5 of its target | 3 | the table is honest (3/5 by 0.74); operator decision 7 |
| 9 | D-3 floors broke the `⌈B/5⌉` rule | 1 | fixed to 237/177/117/57/0 |
| 10 | K̂ presented as measured | 1 | labelled `[U]`; the gate unit is rows |
| 11 | Report-step scoping (0/0 on the facade), and cross-arch LF in the LCOV union | 3 | §4 item 4 now says both |
| 12 | The milestone regex accepted closed past releases | 3 | R6 requires an **open** release milestone ≥ the current one |
| 13 | Bare `gh` fails without an origin remote | 3 | every `gh` call carries `-R paiml/aprender` |
| 14 | B-3 had no numbers | 3 | full-denominator floors given, marked not credible `[U]`; `pv --reverse` measures the code side, not the contract side |

**Refuted by the orchestrator:** lane 1's claim that an empty `COV_PCT` lets `make coverage` exit 0. `Makefile:628`
sets `COV_PCT=0` when `LF` is 0 or empty, which fails the floor.
**Not covered by any lane:** whether CI runs `pv lint`. The orchestrator's grep of `.github/workflows/*.yml` finds only
comment lines (`ci.yml:1321`, `:1940`), and `make contracts` is invoked by no workflow.
