---
phase: 05-benchmark-and-claims-gate
plan: 12
subsystem: benchmark-claims-evidence
status: complete
tags: [EVAL-02, EVAL-03, EVAL-05, evidence-set, compute-authorization, peak-rss-asymmetry, sequential-sweep]

requires:
  - "05-03 (the frozen epsilon basis; and the checkpoint that deferred this compute to wave 7)"
  - "05-09 (scripts/run_bench_cells.sh — the sequential driver, its single-writer lock and hash-based resume)"
  - "05-10 (verify_run and the fail-closed report)"
  - "05-11 (the 40-cell ACTIVE expectation set, and the single-cell verification door)"
provides:
  - "40 digest-committed SetFit benchmark rows covering the full 4 shot-level x 10 seed grid"
  - "40 selection manifests — EVAL-02's retained pairing key, one file per cell"
  - "40 committed lock records whose bytes the gate re-hashes"
  - "run-manifest.json closed at 40/40 complete, every entry carrying a row_sha256 equal to its row file's own digest"
  - "a measured cost law for SetFit cells on this host: train_wall = 5.93 s + 1.1289 s/step"
  - "the peak-RSS mechanism asymmetry, declared per row and tallied over all 40"
affects:
  - "05-13 (the complete, committed set the fail-closed report runs over)"

tech-stack:
  added: []
  patterns:
    - "authorize compute at a blocking checkpoint with the projection written down as a NUMBER first, so the outcome is comparable afterwards"
    - "two-half gate proof — a per-row door AND a whole-set refusal, because the set-level check refuses before it reads any row"
    - "assert status lines POSITIVELY (a zero match, a non-zero match), never by negation"
    - "hash-based resume as a behavioural equivalence check: the driver SKIPPING a hand-run cell proves the two invocations agree"
    - "tally a mechanism over the whole population, never quote one row — the 40-row tally found three sample intervals where one row showed one"

key-files:
  created:
    - benchmarks/tweeteval-stance/rows/ (40 rows)
    - benchmarks/tweeteval-stance/selections/ (40 selection manifests)
    - benchmarks/tweeteval-stance/locks/ (40 lock records)
    - benchmarks/tweeteval-stance/run-manifest.json
    - .planning/phases/05-benchmark-and-claims-gate/05-12-compute-projection.md
  modified:
    - .gitignore

key-decisions:
  - "Compute AUTHORIZED by a human at a blocking checkpoint as option-b, verbatim: 'Select: option-b — run the pilot cell only, then re-present the projection against its measured time before any of the remaining 39.' A second explicit approval, 'Select: approve-39 — run the remaining 39 cells, sequential, protocol intact', released the sweep. Auto-mode was off for both (workflow.auto_advance false, _auto_chain_active false)."
  - "The pre-pilot projection quoted TWO measured bases that disagreed ~2.2x rather than picking the convenient one. The pilot REFUTED Basis A (05-01's calibration tier) and CORROBORATED Basis B (05-07's CLI tier)."
  - "Artifacts, logs and probe text are gitignored with ROOT-ANCHORED patterns (CB-510); the evidence set — rows, selections, locks, run manifest — is committed."
  - "No aggregate statistic is computed here. `bench report` over the complete set is 05-13's act; this plan's only report run was the deliberate refusal against a COPY."

requirements-completed: []

coverage:
  - deliverable: "All 40 SetFit cells ran on CPU from reloaded production artifacts, one process per cell, each emitting one digest-committed row"
    verification:
      - kind: command
        ref: "bash scripts/run_bench_cells.sh setfit ... -> 'DONE setfit: 39 executed, 1 skipped, 40 of 40 cells covered', rc=0"
        status: pass
      - kind: command
        ref: "ls rows/setfit-*.json | wc -l == 40 && 40 manifest cells status=complete (the plan's Task 3 <automated> verify)"
        status: pass
      - kind: command
        ref: "all 40 manifest row_sha256 equal to the row file's own semantic_hash; 40 distinct"
        status: pass
    human_judgment: false
  - deliverable: "The compute was authorized at a blocking checkpoint BEFORE any cell ran, with the projection recorded as a number first"
    verification:
      - kind: command
        ref: "commit 0b62da726 (projection) precedes commit ea842b6b6 (first cell) in git history"
        status: pass
    human_judgment: true
    rationale: "That the authorization was a genuine human decision rather than an inferred one is a fact about the interaction, not something a test can assert. Recorded verbatim below; auto-mode was measured off."
  - deliverable: "One pilot cell driven through BOTH halves of the real gate before the remaining 39 were authorized"
    verification:
      - kind: command
        ref: "apr setfit bench verify-cell --shots 8 --seed 13 -> rc=0 with 39 pending / 1 complete (half a)"
        status: pass
      - kind: command
        ref: "apr setfit bench report --bench-dir /tmp/p12-bench-copy/tweeteval-stance -> rc=5, completeness refusal naming setfit/s8/seed17 (half b)"
        status: pass
      - kind: command
        ref: "the plan's Task 2 <automated> verify, positively asserted -> rc=0"
        status: pass
    human_judgment: false
  - deliverable: "The pilot row is the artifact the driver itself accepts — the sweep SKIPS it by digest rather than re-running it"
    verification:
      - kind: command
        ref: "grep SKIP /tmp/p12-sweep.log -> exactly one line: 'SKIP setfit s8 seed13 (row digest matches the manifest)'"
        status: pass
    human_judgment: false
  - deliverable: "The within-row peak-RSS mechanism asymmetry declared for all 40 rows and tallied over all 40"
    verification:
      - kind: command
        ref: "40/40 train mechanisms are sysinfo_sampled_* with peak_rss_sample_interval_hz present; 40/40 inference are child_max_rss_time_l; 40/40 cold_measured_in_child_process=true"
        status: pass
    human_judgment: false
  - deliverable: "Every row records the selection-manifest hash for its (shots, seed) cell — EVAL-02's retained pairing key"
    verification:
      - kind: command
        ref: "40 distinct selection_manifest_hash values across the 40 rows, one per cell"
        status: pass
    human_judgment: false
  - deliverable: "No committed benchmark artifact contains dataset text"
    verification:
      - kind: command
        ref: "24 real 50-char row prefixes from train/validation/test grepped against rows/, selections/, locks/, run-manifest.json -> 0 matches"
        status: pass
    human_judgment: false
  - deliverable: "Cells ran sequentially; the protocol was never thinned under wall-clock pressure"
    verification:
      - kind: command
        ref: "driver has no parallel-dispatch construct (the 3 'parallel' greps are comment lines 13/16/22); single-writer pid lock held; WARMUP_COUNT=3 and cold-probe-in-child true in all 40 rows"
        status: pass
    human_judgment: false

metrics:
  duration: "6h 56m"
  completed: 2026-09-09
  tasks: 3
  files: 126

actuals:
  tokens: 41000
  tasks: 3
  commits: 6
  plan_head_before: 00350eb603a5ba583436df297591193adff8b274
  # MEASURED with `git rev-list --count ${plan_head_before}..HEAD`, re-measured after
  # the SUMMARY and STATE/ROADMAP commits landed. It read 4 at SUMMARY-write time and
  # was corrected upward rather than left understated.
---

# Phase 05 Plan 12: The 40-Cell SetFit Benchmark Evidence Set Summary

Forty SetFit cells — the full 4 shot-level x 10 seed grid — generated sequentially on this CPU
host from reloaded production artifacts, fronted by a compute authorization the human gave in two
explicit steps, and by a pilot cell driven through both halves of the real gate before the
remaining thirty-nine were allowed to run.

---

## Task 1 — the compute authorization, and the projection that preceded it

**The compute was not pre-authorized.** 05-03's checkpoint recorded it verbatim — *"05-12 compute
pre-authorization: **Defer — ask me at wave 7**"* — and D-19 had already established there is no
pre-authorized host to fall back on (`ssh lambda-vector` → rc=255, `Could not resolve hostname`).

Automation first, then ask. The projection was computed, **written down as a number, and committed
(`0b62da726`) before any cell ran** — that commit precedes the first cell's commit in history, which
is the check that makes the comparison below meaningful rather than retrospective.

### Two measured bases that disagreed, both stated

The honest finding at projection time was that this host carried **two** measured bases and they
did not agree at s8. Reporting only the convenient one would have been the defect CLAUDE.md rule 1
exists to prevent.

| basis | source | s8 per-step | 40-cell total |
|---|---|---|---|
| **A** | 05-01 calibration tier: 2.033 s/step at s8, 2.113 at s64 (`wall_clock=3246.2s steps=1536`) | 2.033 | **12.59 h** |
| **B** | 05-07 CLI tier: a whole s8 chain measured at ~37 s | 0.925 – 1.542 | **5.52 – 8.74 h** |

> Recorded projection: **12.6 h conservative, honest range 5.5 – 12.6 h.** The cost law itself was
> derived rather than assumed: `budget = 6n²`, `steps = ceil(6n²/16)` → 24 / 96 / 384 / **1536**,
> so s64 is **64x** s8, not 8x, and the s64 tier alone was 75.6% of the projection.

Free disk 36.4 GiB at 96% capacity; projected peak write 3.40 GiB.

### The decision, verbatim

> **Select: option-b — run the pilot cell only, then re-present the projection against its
> measured time before any of the remaining 39.**

and, after the pilot was re-presented:

> **Select: approve-39 — run the remaining 39 cells, sequential, protocol intact.**

Auto-mode was measured off for both (`workflow.auto_advance: false`,
`workflow._auto_chain_active: false`), so neither was an auto-approval. Options c (a second host)
and d (halt) were presented and refused on the record rather than omitted.

---

## Task 2 — the pilot cell and the two-half gate proof

One s8/seed13 cell, run with the same `apr data select` and `apr setfit bench run` invocations the
driver issues, under the full protocol: **47 s wall clock**.

### Why two halves, and not one

`verify_run` checks the manifest digest, then the expectation set, then completeness, and only
**then** reads any row's bytes. A report run over a directory holding one row therefore refuses at
the completeness step and **never reaches the pilot row at all**. Running only that half would have
established that the gate refuses an incomplete set — worth having — while appearing to establish
per-row properties it never checked.

**Half (a) — the per-row half.** `apr setfit bench verify-cell --bench-dir <real> --method setfit
--shots 8 --seed 13`, status line **`rc=0`**, taken while the manifest held **39 `pending` / 1
`complete` of 40 declared** (counted from the manifest, not assumed). Its own output names its
scope:

> `scope: steps 1+4+6 of verify_run over ONE cell - manifest digest, this entry's own completeness,
> the row file/schema/envelope digest/manifest digest/slot, and provenance recomputed from committed
> bytes. NOT the set-level steps (expectation-set equality, the all-entries sweep, pairing,
> attestation), which is what lets this pass while other cells are still pending. No statistic is
> emitted; bench report is the only door that publishes numbers.`

Covered: steps **1, 4 and 6**. Excluded: step 2, step 3's sweep, step 5, step 7. The resource and
size fields are **not** separate checks — their presence and parse come from the schema parse, and
nothing here validates their values independently.

**Half (b) — the whole-set half.** The bench directory was copied to
`/tmp/p12-bench-copy/tweeteval-stance` and the real `bench report` run against the **copy**; the
real directory at `…/aprender/benchmarks/tweeteval-stance` was never the target, and the two paths
were asserted to differ. Status line **`rc=5`**. The refusal, verbatim:

> `cell setfit/s8/seed17 is not complete: the manifest still lists it as `pending`, so this run is
> not complete and no aggregate over it is publishable. The report has no partial-data mode — run it
> with `apr setfit bench run --method setfit --shots ... --seed ...`, or, if it ran on the other
> host, ingest its row with `--record <ROW_FILE>``

It names **seed17 — a pending cell** — and says nothing about the pilot row's digest, schema, slot
or lock, which is precisely the evidence that it refused *before* reading any row and therefore
that half (a) was necessary.

**Both asserted positively.** `grep -q '^rc=0$'` for half (a) and `grep -qE '^rc=[1-9][0-9]*$'` for
half (b) — never a negated line match, which succeeds whenever any other line differs and so would
pass on exactly the failure it exists to catch. **Every status in this plan was captured with the
`|| rc=$?` form on the command itself and appended to its log by a separate redirect; nothing was
read through a pipe.** The plan's Task 2 `<automated>` verify re-ran clean: **rc=0**.

### The pilot row, quoted

| field | value |
|---|---|
| `backend_identity` | `cpu:setfit-core:autograd-trueno-matmul` — read from execution, not echoed from config |
| `selection_manifest_hash` | `1d2dbbb37edea7aba85b251860ee9c92734bd3b09fa5f4d571bed62355fb7ad0` |
| `apr_artifact_sha256` | `f6931948e0c1030446f76090f3b6c5e406fd953d9c150ec0257b92e8d73163e7` |
| `lock.lock_record_path` | `locks/setfit-s8-seed13.lock.json` (role `written`, rule `max_metric_lowest_index_tie_break`) |
| `train_peak_rss_mechanism` | `sysinfo_sampled_19` + `peak_rss_sample_interval_hz: 19` — a **sampled lower bound** |
| `inference_peak_rss_mechanism` | `child_max_rss_time_l` — a **kernel high-water mark** |
| `cold_measured_in_child_process` | `true`, `cold_latency_ms = 877.06` |
| `artifact_bytes` / `deployable_total_bytes` | `90 777 156` / `90 777 156` (SetFit ships one file) |

### The re-projection the pilot bought

`train_wall = fixed_train + steps x per_step` is one equation in two unknowns, and one cell cannot
solve it. So only the **upper bound** was claimed as measured — all of `train_wall` attributed to
the steps:

```
per_step <= 30.713 / 24 = 1.2797 s/step
20 400 steps x 1.2797   = 26 106 s
+ 40 x 16.287 s         =    651 s
UPPER BOUND (measured)  = 26 757 s = 7.43 h      (pre-pilot upper edge was 12.59 h, -41%)
```

**Basis A was refuted, Basis B corroborated** — and this was stated rather than fitted. Basis A
predicted a 48.8 s s8 *tuning* phase; the whole measured `train_wall` is 30.7 s, less than Basis A's
tuning alone, so A is high by **≥1.59x** however that 30.7 s splits. The measured 1.2797 falls
**inside** Basis B's 0.925 – 1.542 bracket. The cell did not land outside both. The two CLI-tier
measurements are also mutually consistent: 47 s here versus 05-07's ~37 s ladder is +27%, which is
what adding an APR write, a reload, a cold-probe child and a throughput pass should cost.

---

## Task 3 — the remaining 39, and the closure

```
$ bash scripts/run_bench_cells.sh setfit <bench> <data> <encoder>
DONE setfit: 39 executed, 1 skipped, 40 of 40 cells covered
rc=0
```

**39 PASS, 1 SKIP, 0 FAIL, 0 HALT.** The rc was read from the driver's own log line, not from the
background wrapper's exit status. The driver's vacuity floor — which asserts the loop covered 40
cells rather than merely reporting a tally of zeroes — passed.

### The pilot skip, which is the point

```
SKIP setfit s8 seed13 (row digest matches the manifest)
```

Exactly one SKIP line, and it is the pilot's. The driver found the pilot's row, matched its digest
against the manifest, and did not re-run it. **That is a behavioural proof that the hand-run pilot
invocation and the driver's invocation agree** — a stronger check than comparing two spellings of a
command. A re-run would have been a finding to diagnose; it did not occur.

### Sequential, and never thinned

The driver has **no parallel-dispatch construct**. The three `parallel` matches in it are comment
lines 13, 16 and 22, quoted from the source rather than asserted. It held its single-writer pid lock
for the whole run, one `apr` process at a time. No warmup was reduced (`warmup_count = 3` in all 40
rows), no cold-probe child skipped (`cold_measured_in_child_process = true` in all 40), no seed or
shot dropped (the grid check found **40/40 covered, 0 missing, 0 extra**).

### Wall clock, against the number recorded before the run

| | |
|---|---|
| sweep start / end | `2026-09-08T21:12:17Z` → `2026-09-09T03:48:49Z` |
| sweep wall clock | **6 h 36 m** (39 cells + 39 selection manifests) |
| + the pilot | 47 s → **6.62 h for all 40** |
| re-projected upper bound | **7.43 h** |
| **ratio** | **0.89** — under the bound |
| against the original pre-pilot projection | 6.62 / 12.59 = **0.53** |

No overrun, so no escalation was needed. Had one occurred it would have gone back to the human as a
checkpoint; parallelism and protocol-thinning were never on the table.

### The cost law, now MEASURED rather than bounded

Four tier means over ten seeds each, least-squares against the step counts — this closes the
one-equation-two-unknowns gap the pilot alone could not:

```
train_wall = 5.93 s + 1.1289 s/step
```

| tier | steps | measured mean `train_wall` | model |
|---|---|---|---|
| s8 | 24 | 31.2 s (min 29.2, max 33.6) | 33.0 s |
| s16 | 96 | 113.0 s (min 105.3, max 118.4) | 114.3 s |
| s32 | 384 | 443.4 s (min 427.8, max 465.8) | 439.4 s |
| s64 | 1536 | 1739.1 s (min 1708.3, max 1781.9) | 1740.0 s |

The measured `fixed_train` of 5.93 s sits inside the sensitivity band presented at the checkpoint
(where 5 s → 6.31 h and 8 s → 5.63 h were offered as *labelled assumptions*). Feeding the measured
law forward predicts 6.64 h for the matrix against 6.62 h actual — **0.3% error**. The projection's
weakest input is now a measurement.

### The peak-RSS mechanism tally — over all 40 rows, and why one row would have lied

| mechanism | count | denominator |
|---|---|---|
| TRAIN `sysinfo_sampled_19` | 30 | 40 |
| TRAIN `sysinfo_sampled_18` | 8 | 40 |
| TRAIN `sysinfo_sampled_17` | 2 | 40 |
| **TRAIN sampled lower bound, any interval** | **40** | **40** |
| rows carrying `peak_rss_sample_interval_hz` | **40** | **40** |
| INFERENCE `child_max_rss_time_l` | **40** | **40** |
| rows with `cold_measured_in_child_process: true` | **40** | **40** |

**The tally earned its keep.** The pilot row reported `sysinfo_sampled_19`, and quoting it would
have described the whole run as a 19 Hz sample. Across 40 rows the sampler actually achieved three
different intervals — 19 Hz on 30 cells, 18 Hz on 8, 17 Hz on 2 — because the sampling rate degrades
under the heavier tiers' memory pressure. The direction of the error is unchanged (every one is a
**lower bound**, and can only understate), but the magnitude varies per row, which is exactly the
kind of thing an anecdote hides and a denominator surfaces.

The asymmetry itself is uniform and stark: **every** train peak is a sampled lower bound carrying
its interval; **every** inference peak is a true kernel high-water mark from a dedicated fresh
child. 05-13 cannot render them as one averaged figure without contradicting all 40 rows.

### Closure, and the dataset-text control

- **Run manifest:** 40 declared, **40 `complete`**, 40 `row_sha256` values — each **equal to its row
  file's own `semantic_hash` on disk**, 40 distinct, 0 missing, 0 mismatched. (The field is
  `row_sha256`; this was read off the schema rather than guessed — a first pass using guessed key
  names reported `0` and was corrected before any claim was made.)
- **Pairing key:** **40 distinct `selection_manifest_hash` values**, one per cell. This is what lets
  the deferred second arm (D-ITEM-05-15) be added later without re-running the SetFit half.
- **Dataset-text control**, run before the commit:
  ```
  24 real 50-char row prefixes from train/validation/test
    grepped -RF against rows/, selections/, locks/, run-manifest.json
  DATASET_TEXT_CONTROL: matches=0 over 24 probes
  ```
- **Staging control:** `git diff --cached --name-only | grep -cE 'artifacts/|logs/|probe/|\.apr$'`
  → **0**. The 3.46 GB of `.apr` artifacts never entered the index.

The full `bench report` was deliberately **not** run over the complete set — that is 05-13's act.
The only report invocation here was the pilot's refusal against a copy.

---

## Deviations from Plan

**1. [Rule 3 — blocking] Three forced release rebuilds, because the pin worked**

- **Found during:** Tasks 2 and 3.
- **Issue:** `scripts/apr_bin.sh` compares the binary's embedded SHA against HEAD **exactly**. Each
  of this plan's own commits advanced HEAD past the built binary, so the pin refused it —
  `STALE apr BINARY … reports: apr 0.63.0 (00350eb60) / HEAD: 0b62da726`. Separately, the first run
  failed with `unrecognized subcommand 'setfit'`: the `setfit` surface is feature-gated
  (`apr-cli/setfit`) and the existing release binary was built without it.
- **Fix:** rebuilt from HEAD with `--features setfit` each time — 74 s, 84 s, 79 s. **The pin was
  never weakened.** `APR_BIN` would not have helped: the override is freshness-checked too, by
  design, so it cannot smuggle a stale binary past the gate. Every one of the 40 cells ran against a
  binary whose embedded SHA equalled HEAD at its launch.
- **Verification:** `apr 0.63.0 (ec5874541)` = HEAD immediately before the sweep launched.

**2. [Rule 2 — missing critical] Root-anchored `.gitignore` rules for the bench working files**

- **Found during:** Task 2, before the first commit.
- **Issue:** `.gitignore:50` is `/*.apr` — **root-anchored**, so it does not match
  `benchmarks/tweeteval-stance/artifacts/*.apr`. Without a rule, **3.46 GB** of binaries would have
  entered the repository. This is the CB-510 failure mode in the opposite direction: there an
  unanchored pattern hid source; here an anchored one failed to cover a subdirectory.
- **Fix:** added root-anchored `/benchmarks/*/artifacts/`, `/logs/`, `/probe/`, `/.driver.lock`.
- **Verification:** two-sided. `git check-ignore` confirms the artifact and log are ignored **and**
  that the row, lock, selection manifest and run manifest are still tracked. **Files:** `.gitignore`.
  **Commit:** `ea842b6b6`.

**3. [Rule 3 — blocking] A shell-dialect artifact nearly produced a false red**

- **Found during:** Task 2. The plan's `<automated>` verify, pasted as a multi-line `&&` chain,
  returned `rc=2` with `(eval):test:2: too many arguments` — the Bash tool runs **zsh**, which parsed
  the continuation differently.
- **Fix:** re-ran the identical predicate under `bash -c`, giving **rc=0**. Recorded rather than
  quietly re-run, because a dialect artifact that looks like a verification failure is exactly the
  kind of thing that gets "fixed" by weakening the check.

**Total deviations:** 3 auto-fixed (2 blocking, 1 missing-critical). **Impact:** none on the
evidence. No plan behaviour changed, no protocol step altered, no gate relaxed.

---

## A measurement correction worth recording

Midway through the sweep, free space read **11.5 GiB**, down from 22.6 GiB at launch — an apparent
380 MiB per cell against a measured 86.6 MiB artifact, which under the standing instruction would
have justified stopping.

It was measured instead of believed. `du` on the bench directory gave **2.5 GB for 30 cells = 83
MB/cell**, matching the anchor; `target/` was unchanged at 39 GB; `tmutil listlocalsnapshots` found
none; and `df -H` a moment later read 19 GB where `df -k` had read 11.5 GiB. The swing was
**OS-level purgeable space**, not the sweep. Had the `df` reading been taken at face value the run
would have been halted at cell 30 on a number that described the operating system rather than the
work. *When a result looks bad, check how it was measured* — the same discipline as when it looks
good.

Final: 15.2 GiB free, `artifacts/` at 3.46 GB for 40 cells.

---

## Requirements: none marked complete, deliberately

`requirements-completed: []`.

- **EVAL-02** — the plan's own `<amended_requirements>` table is explicit: this delivers 40 of the
  80 contracted cells. The second method's arm is deferred as **D-ITEM-05-15**. EVAL-02 remains
  Pending and amended in scope. The pairing key is recorded per row precisely so that arm never
  requires re-running this half.
- **EVAL-03** — the rows exist and nothing was withheld, but the requirement's evidence is the
  *verified, reported* set; 05-13 runs the fail-closed report over it. Marking it here would file
  completion against an unrun gate.
- **EVAL-05** — within-SetFit resource comparison across shot levels and seeds is delivered;
  cross-method cost comparison is removed, not repurposed.

Filing a narrower deliverable against an unamended requirement is the false-completion defect this
project has hit repeatedly. `requirements.mark-complete` was not run.

## Known Stubs

None. Every one of the 40 rows was produced by a real training run against the pinned encoder and a
reloaded production artifact; no value in the evidence set is placeholder, hardcoded or mocked.

## Threat Flags

None. No new network endpoint, auth path or trust-boundary schema. The register is discharged:
**T-05-12-01** (DoS) — projection recorded and authorized before any cell, hash-based resume, disk
tracked; **T-05-12-02** (contended measurements) — no dispatch construct, pid lock held, no
concurrency, no thinning; **T-05-12-03** (dataset text) — attested directory kept outside the repo,
24-probe control clean, artifacts gitignored; **T-05-12-04** (backend identity) — read from
execution, quoted; **T-05-12-05** (late per-row defect) — the two-half proof ran before the 39;
**T-05-12-06** (averaged asymmetry) — tallied over 40 rows, three sample intervals surfaced;
**T-05-12-07** (divergent pilot) — the driver SKIPPED it by digest, recorded.

## Issues Encountered

The pre-existing failures named in the dispatch (21 `aprender-train gpu::{guard,ledger,wait}` tests,
1 `aprender-core setfit::artifact::determinism` test, and `crates/aprender-profile/examples/
validate_golden_trace.rs` blocking `make test` on macOS) were **not touched and did not block this
plan**. Targeted commands were used throughout; `make test` was never invoked. They remain
D-ITEM-05-16's business.

## Next Phase Readiness

**Ready for 05-13.** The complete, committed, digest-closed evidence set is on
`gsd/phase-2-contract-gate` at `86d3d4c34`. Three things 05-13 should carry rather than rediscover:

1. **The peak-RSS tally is 40/40 sampled train vs 40/40 child-measured inference, with THREE
   distinct sample intervals (19/18/17 Hz).** A renderer that prints one interval for the run would
   be wrong on 10 of 40 rows.
2. **`apr` is feature-gated.** `cargo build --release --bin apr --features setfit` (~80 s), and
   re-run it after any commit — the pin compares against HEAD exactly.
3. **The artifacts are gitignored and local.** `benchmarks/tweeteval-stance/artifacts/` holds 3.46 GB
   that no clone will have. Anything 05-13 needs must come from the rows, locks, selections and
   manifest, which is what the gate reads anyway.

## Self-Check: PASSED

| Claim | Check | Result |
|---|---|---|
| 40 row files | `ls rows/setfit-*.json \| wc -l` | **40** — FOUND |
| 40 lock records | `ls locks/*.json \| wc -l` | **40** — FOUND |
| 40 selection manifests | `ls selections/*/selection-manifest.json \| wc -l` | **40** — FOUND |
| run manifest closed | 40 declared / 40 `complete` / 40 `row_sha256` matching disk | FOUND |
| `05-12-compute-projection.md` | `test -f` | FOUND |
| commits `0b62da726`, `ea842b6b6`, `ec5874541`, `86d3d4c34` | `git log` | all FOUND |
| Task 2 `<automated>` verify | `bash -c …` | **rc=0** |
| Task 3 `<automated>` verify | `bash -c …` | **rc=0** |
| driver | `DONE … 40 of 40 cells covered`, `rc=0` | PASS |
| dataset-text control | 24 probes over rows/selections/locks/manifest | **0 matches** |
