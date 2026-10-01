---
phase: 05-benchmark-and-claims-gate
plan: 13
subsystem: benchmark-claims-report
status: complete
tags: [EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, D-11, claims-report, no-comparison-control, closing-audit]

requires:
  - "05-10 (verify_run, the fail-closed report, the six doctored negatives)"
  - "05-11 (the 40-cell ACTIVE expectation set, the single-method renderer, the rendering case table)"
  - "05-12 (the complete 40-row evidence set the report runs over)"
provides:
  - "benchmarks/tweeteval-stance/report.md — the human single-method claims report over all 40 cells"
  - "benchmarks/tweeteval-stance/report.json — the machine-readable detail, the only place a p-value appears"
  - "a two-run byte-identity demonstration of exact recomputability, with a comparator whose status is meaningful"
  - "three sub-section must-not-match literals the 05-11 case table did not cover, plus a token-level control"
  - "the D-11 encoder-only qa refusal that names the doors which DO cover an encoder or classifier APR"
  - "the closing evidence map, marking EVAL-02/04/05 met-narrowly with their deferred clauses named"
affects:
  - "the verifier (/gsd-verify-work) — this plan flips no requirement state and no checkbox"

tech-stack:
  added: []
  patterns:
    - "derive a renderer's scope FROM ITS DATA, not from a caller-supplied flag, when the function only receives the data"
    - "a section-level must-not-match table does not cover sub-section labels — the empirical run is what found the gap, not review"
    - "parse the must-not-match literals OUT OF THE SHIPPED SOURCE so the control cannot drift from the constant it gates"
    - "every absence assertion carries a positive half AND a proof that it can fire; a control that cannot fail is not a control"
    - "print a tally the closing audit must READ, so the number is an observation instead of a quotation"
    - "prove a KNOWN-RED is pre-existing by showing the phase's diff over the code under test is EMPTY, not by citing a dispatch note"

key-files:
  created:
    - benchmarks/tweeteval-stance/report.md
    - benchmarks/tweeteval-stance/report.json
  modified:
    - crates/apr-cli/src/commands/setfit_bench.rs
    - crates/apr-cli/src/commands/setfit_bench_tests.rs
    - crates/apr-cli/src/commands/output_verification.rs
    - crates/aprender-train/src/train/setfit/bench_gate.rs
    - crates/aprender-train/src/train/setfit/bench_gate_tests.rs
    - Makefile

key-decisions:
  - "The plan's own Task 1 <automated> verify FAILED on the first honest run, and it was right to. The 05-11 retarget removed the delta and comparison SECTIONS but left three sub-section strings naming the deferred method: a per-group row label printed on all four SetFit groups, the size table's four-sentence cross-method footnote, and the header's provenance clause crediting a candidate-ledger recomputation the active scope cannot perform. Fixed at the renderer, with the case table grown from six rows to nine and a token-level control added."
  - "The gate itself was never touched to make anything pass. No row, lock record or manifest entry was edited; the single refusal encountered was a deliberate control on a copy."
  - "The exclusion set for the byte-identity comparison is EMPTY, established by scanning the payload for time/date/version/duration keys rather than assumed. cmp was used rather than diff because diff's exit status is unreliable through this environment's tooling."
  - "requirements.mark-complete was NOT run. EVAL-02, EVAL-04 and EVAL-05 are met by a NARROWER deliverable under the 2026-09-07 amendment; EVAL-01 and EVAL-03 are met in full but the flip is the verifier's act, not this plan's."

requirements-completed: []

coverage:
  - deliverable: "The fail-closed report ran over the real 40-cell SetFit evidence set and passed on the first honest attempt"
    verification:
      - kind: command
        ref: "apr setfit bench report --bench-dir benchmarks/tweeteval-stance --out report.json > report.md -> GATE_RUN1_RC=0"
        status: pass
    human_judgment: false
  - deliverable: "Every number in the report is exactly recomputable — two runs over the same directory are byte-identical"
    verification:
      - kind: command
        ref: "sha256 2c1b15ef...848486 (json) and d7e3fd0b...cbd94a (md) equal across two runs; cmp rc=0 on both"
        status: pass
      - kind: command
        ref: "cmp NON-VACUITY: cmp against a one-byte-longer copy -> rc=1"
        status: pass
    human_judgment: false
  - deliverable: "The gate bites on real-shaped data: an omitted row is refused by name, with nothing rendered"
    verification:
      - kind: command
        ref: "omission control on /tmp/p13-omission-copy -> rc=5, names setfit/s32/seed37 and the missing file, 0 bytes rendered"
        status: pass
      - kind: command
        ref: "the real directory re-verified afterwards: 40/40/40 files, report reproduces byte-identically, 0 changed evidence files in git"
        status: pass
    human_judgment: false
  - deliverable: "No committed report artifact states or implies a second-method result"
    verification:
      - kind: command
        ref: "9 must-not-match literals PARSED FROM setfit_bench.rs + 6 bare tokens: 0 hits in report.md and report.json; 18 positive rows all present; CONTROL_FAILURES=0"
        status: pass
      - kind: command
        ref: "control NON-VACUITY: the same script over a doctored copy -> rc=1, CONTROL_FAILURES=3"
        status: pass
      - kind: tests
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#setfit_bench_report_active_scope_never_names_the_deferred_method"
        status: pass
    human_judgment: false
  - deliverable: "The two peak-RSS figures are presented separately with their mechanism strings, the sampled one labelled a lower bound"
    verification:
      - kind: command
        ref: "report.md lines 31-33 (s8) and 51-53 (s32): separate rows, mechanism strings, LOWER BOUND label, asymmetry note; s32/s64 print all three sample intervals"
        status: pass
    human_judgment: false
  - deliverable: "D-11: the encoder-only qa refusal names the quality-validation and evaluation doors; exit code unchanged"
    verification:
      - kind: command
        ref: "apr qa crates/aprender-core/tests/fixtures/setfit/slice_model.apr -> exit 5, message quoted below"
        status: pass
      - kind: tests
        ref: "crates/apr-cli/src/commands/output_verification.rs#qa_encoder_only_refusal_names_the_doors_and_never_impugns_the_artifact (+ its non-vacuity twin)"
        status: pass
      - kind: tests
        ref: "crates/apr-cli/tests/setfit_cli_lifecycle.rs#setfit_cli_tooling_generic_apr_commands_survive_the_schema_owned_entries -- --ignored tooling -> 1 passed"
        status: pass
    human_judgment: false
  - deliverable: "The phase's gates pass at the closing SHA 7c3a6b401"
    verification:
      - kind: command
        ref: "make setfit-bench-tests rc=0 (27/38/14/65); make contract-audit-phase5 rc=0 zero unbound; make contract-audit-phase4 rc=0 zero unbound; pv validate x3 rc=0"
        status: pass
      - kind: command
        ref: "17 of 18 setfit-all-tests suites rc=0; setfit-apr-tests red on one PRE-EXISTING test, proven pre-existing by an EMPTY phase-5 diff over its source and fixtures"
        status: pass
    human_judgment: false
  - deliverable: "The closing evidence map states, per requirement id, met-in-full versus met-narrowly with the deferred clause named"
    verification:
      - kind: command
        ref: "git diff --stat ec1bf58e8..7c3a6b401 -- .planning/REQUIREMENTS.md -> 0 bytes; requirements.mark-complete not run"
        status: pass
    human_judgment: true
    rationale: "Whether the map is HONEST about narrowness — rather than merely present — is a judgment the verifier must make against the amended requirement text. The mechanical half (no checkbox moved) is measured above."

metrics:
  duration: "4h 0m"
  completed: 2026-09-09
  tasks: 3
  files: 8

actuals:
  tokens: 21000
  tasks: 3
  commits: 4
  plan_head_before: ec1bf58e845eea34f7b5cf0f606ee20b4c28c9c3
  # MEASURED with `git rev-list --count ec1bf58e845eea34f7b5cf0f606ee20b4c28c9c3..HEAD`
  # at SUMMARY-write time. The SUMMARY/STATE commit that follows is not counted here.
---

# Phase 05 Plan 13: The 40-Cell SetFit Claims Report and the Closing Audit Summary

The milestone's claims deliverable: a fail-closed report over all forty committed SetFit cells,
proven byte-identical on re-run, gated three ways against implying a comparison that was never
run — and one of those three gates fired on the first honest attempt and found a real defect the
retarget had left behind.

---

## Task 1 — the report, the recompute demonstration, and the two controls

### The gate passed, and nothing was edited to make it

```
$ apr setfit bench report --bench-dir benchmarks/tweeteval-stance \
      --out benchmarks/tweeteval-stance/report.json > benchmarks/tweeteval-stance/report.md
GATE_RUN1_RC=0
```

Status captured on the command, appended by a separate `echo`, never read through a pipe
(CLAUDE.md rule 1). The precondition was checked first and held: the run manifest declares
**40 cells, all `complete`** — not the retired 80.

No row, lock record, selection manifest or manifest entry was touched at any point in this plan.
`git status` over `rows/`, `locks/`, `selections/` and `run-manifest.json` reports **0 changed
files** at the closing SHA.

### The headline table, quoted from the committed artifact

```
QUALITY - official F_avg, mean +/- (n-1) std [min, max]
method   shots       mean       std        min        max      95% CI (seeds)   (macro F1 mean / MCC mean)
setfit       8     0.4746    0.0456     0.4057     0.5318   [0.4420, 0.5072]       0.4643   0.2656
setfit      16     0.5115    0.0383     0.4437     0.5678   [0.4841, 0.5389]       0.4988   0.3059
setfit      32     0.5346    0.0208     0.4996     0.5678   [0.5197, 0.5495]       0.5133   0.3198
setfit      64     0.5607    0.0270     0.5185     0.6026   [0.5414, 0.5800]       0.5343   0.3397
95% CI over the ten contracted seeds at fixed data and protocol - a seed-dispersion interval, not a population interval.
```

Estimation-first, and it says so in its own last line: *"Estimation-first (D-08): point estimates,
dispersion and 95% seed-dispersion intervals only. No binary verdict is printed."* The `report.md`
contains **no p-value and no test statistic**; both live in `report.json` under `detail`, which is
where D-08 permits them.

### Exact recomputability, demonstrated rather than asserted

Two runs of the identical command over the identical directory, the second into a temporary path:

| artifact | run 1 (committed) | run 2 (temp) | `cmp` |
|---|---|---|---|
| `report.json` | `2c1b15ef485b7c36353ac5ff6bc273d6ca8e5bcb39972a6f0646304123848486` | identical | `CMP_JSON_RC=0` |
| `report.md` | `d7e3fd0bceb22e487f83492af436870cf4f3b117de0a7557cedcd0bf5c8bd94a` | identical | `CMP_MD_RC=0` |

**`cmp`, not `diff`** — this environment's tooling returns rc=0 for differing files through `diff`,
so `diff`'s status is not evidence. The comparator's own non-vacuity was proven: `cmp` against a
copy one byte longer returned **rc=1** with `cmp: EOF on …/report.json`.

**The exclusion set is empty, and that was established rather than assumed.** The plan permitted
excluding "whatever envelope section the schema itself declares volatile". A key scan over the
payload for `time|date|_at|stamp|version|host|duration` found exactly one hit — `detail.resource[].hosts`,
which is a measured field, not an envelope. `ReportPayload` is `{schema, contract_id, detail}` with
no timestamp and no tool version, unlike the run manifest, which does carry a
`volatile: {created_at, tool_version}` block. So the comparison above is over **the whole file**.

Re-run at the closing SHA `7c3a6b401` against a freshly built binary: `CMP_JSON_AT_CLOSING_SHA_RC=0`,
`CMP_MD_AT_CLOSING_SHA_RC=0`.

### Control 1 — omission, on a COPY, never on the real directory

```
real = /Users/guy/Development/machine-learning/aprender/benchmarks/tweeteval-stance
copy = /tmp/p13-omission-copy/tweeteval-stance          PATHS_DIFFER: yes
COPY_BASELINE_RC=0        <- the copy reports clean BEFORE the mutation
VICTIM_REMOVED: rows now 39   (rows/setfit-s32-seed37.json, 3151 bytes)
OMISSION_RC=5
```

The refusal, verbatim:

> `error: Validation failed: cell setfit/s32/seed37 is recorded complete in the manifest, but /tmp/p13-omission-copy/tweeteval-stance/rows/setfit-s32-seed37.json does not exist. A recorded digest with no bytes behind it is an omission the manifest cannot see; re-run the cell or restore the row file`
>
> `The report has no partial-data mode: a missing, substituted, unmatched or post-test-selected cell invalidates the whole run (EVAL-04).`

It names the cell **and** the file. **Rendered bytes: 0** — a refusal produces no partial table.
The baseline run on the copy matters: without it, a refusal would not distinguish "the omission was
detected" from "the copy was never valid".

**The real directory afterwards:** 40 rows / 40 locks / 40 selections, `REAL_POSTCONTROL_RC=0`, the
report reproduces byte-identically, and git sees **0 changed evidence files**.

### Control 2 — no second method, over the COMMITTED bytes

The must-not-match literals are **parsed out of `crates/apr-cli/src/commands/setfit_bench.rs` at
run time**, not retyped into the control, so the control cannot drift away from the constant it
gates. Six rows are 05-11's case table; three were added by this plan (see the deviation below).

```
== MUST NOT MATCH (literals parsed from crates/apr-cli/src/commands/setfit_bench.rs) ==
  delta table header                 md=0 json=0  PASS  'paired delta - f_avg(setfit) - f_avg(lora), same selecti'
  cross-method comparison header     md=0 json=0  PASS  'resource comparison - setfit beside lora, mechanism-labe'
  per-row comparison format          md=0 json=0  PASS  '  |  lora '
  two-method title line              md=0 json=0  PASS  'setfit vs lora - benchmark claims report'
  paired estimation-first note       md=0 json=0  PASS  'estimation-first (d-08): point estimates, dispersion and'
  two-host resource framing          md=0 json=0  PASS  'as-deployed method costs; hosts differ by design and are'
  two-method artifact-bytes label    md=0 json=0  PASS  'artifact bytes (adapter only for lora)'
  cross-method size footnote         md=0 json=0  PASS  'a cross-method size claim uses this column and no other.'
  two-method provenance sources      md=0 json=0  PASS  'the committed lock and ledger bytes'
== MUST NOT MATCH (bare token, any spelling) ==
  'lora' 'versus' ' vs ' 'compared to' 'paired delta' 'significant'   all md=0 json=0  PASS
== MUST MATCH (the positive half — this control cannot pass on an empty file) ==
  seed-dispersion interval label / single-method scope note / sampled lower-bound label /
  single-method title / estimation-first note (active) / active provenance sources /
  D-19 / D-ITEM-05-15 / sysinfo_sampled_ / child_max_rss_time_l /
  'train peak RSS (training process)' / 'inference peak RSS (cold child)'            all PASS
== JSON must carry the mechanism strings too ==
  sysinfo_sampled_17 / sysinfo_sampled_18 / sysinfo_sampled_19 / child_max_rss_time_l   all PASS
CONTROL_FAILURES=0
```

**Its non-vacuity half**, because a control that cannot fail proves nothing: the same script over a
copy of `report.md` with `SetFit vs LoRA - benchmark claims report` appended exits **rc=1** with
`CONTROL_FAILURES=3` — the title literal, the `lora` token and the ` vs ` connective all fire.

**The mandated note and this gate hold together, as 05-11 designed.** The single-method note is
present in the committed report and trips none of the nine literals or six tokens: it refers to the
deferred arm only as `D-19`, `D-ITEM-05-15` and "a second method".

### The two peak figures, rendered separately — quoted from the committed report

```
  train peak RSS (training process)       5560488755 B   mechanism: sysinfo_sampled_19 [sampled_lower_bound]  LOWER BOUND (sampled; can only understate)
  inference peak RSS (cold child)         1553154048 B   mechanism: child_max_rss_time_l [exact_kernel_high_water_mark]
  ^ The two figures above are TWO metrics from TWO processes, and here they come from two different mechanism CLASSES. They are never added, never averaged, and never reduced to a single figure: a kernel high-water mark is process-cumulative, so the training process's peak is not the inference peak.
```

and, at s32 and s64, **the three sample intervals 05-12's 40-row tally surfaced are all printed**
rather than collapsed to the one the pilot row showed:

```
  train peak RSS (training process)       5828229530 B   mechanism: sysinfo_sampled_17, sysinfo_sampled_18, sysinfo_sampled_19 [sampled_lower_bound]  LOWER BOUND (sampled; can only understate)
```

s8 and s16 print `sysinfo_sampled_19` alone because every cell in those tiers achieved 19 Hz; the
distribution is per-group and read off the rows, not a single constant asserted for the run.

---

## Task 2 — D-11: the encoder-only qa refusal names the doors that cover it

The refusal site is `crates/apr-cli/src/commands/output_verification.rs`, which `qa.rs`
`include!`s. The message was extended in place; **the detection logic, the exit code and the qa
command module are untouched**, and the diff is one file.

Run against Phase 4's plain-BERT control fixture:

```
$ apr qa crates/aprender-core/tests/fixtures/setfit/slice_model.apr --json
QA_EXIT=5
error: Validation failed: APR missing embedded tokenizer. `apr qa` is a GENERATIVE-model gate:
its first check loads an embedded BPE tokenizer and then generates with a token budget, so it has
nothing to gate on an encoder or classifier APR. This is a statement about the GATE's scope, not
about the model. Encoder and classifier APRs are covered by `apr validate --quality` for format
and quality validation, by `apr eval` for task metrics, and by `apr setfit bench run` /
`apr setfit bench report` for the benchmark door. Extending qa itself to cover encoder-only APRs
is deferred to its own ticket (D-11).
```

**It EXTENDS rather than replaces.** The leading phrase `APR missing embedded tokenizer` is
unchanged, which is the exact substring the Phase 4 lifecycle control asserts on **both** fixtures
— so this change cannot silently retire that control. Re-run to prove it:

```
$ cargo test -p apr-cli --features setfit --test setfit_cli_lifecycle -- --ignored tooling
LIFECYCLE_RC=0        1 passed, 5 filtered out
```

Exit code **5**, unchanged, and the Phase 4 comment records the same value.

### The must-match / must-not-match table, and why its negative half is the important one

`cargo test -p apr-cli --lib qa` → **rc=0, 266 passed** (264 before, plus two).

The must-not-match rows are all **invalid-artifact wording**: `invalid`, `corrupt`, `malformed`,
`unsupported`, `broken`, `damaged`, `bad `, `the model failed`, `not a valid`. The refusal is about
the gate's scope, not the file: an operator who reads it as a verdict on their model goes hunting
for a defect that is not there. The message therefore contains none of those words, including as
negations — writing "not malformed" would have satisfied the intent and failed the table.

A second test proves every one of those nine rows **can** fire, by matching them against a doctored
string that contains them all. Without it, a list of words no message would ever carry passes
forever.

---

## Task 3 — the closing audit at SHA `7c3a6b401`

Every status captured on its own command and reported on its own line.

| # | command | status |
|---|---|---|
| 1 | `make setfit-bench-tests` | **rc=0** — 27 / 38 / 14 / **65** |
| 2 | `make contract-audit-phase5` | **rc=0** — 10/10 equations bound, 9 obligations, **zero unbound**, zero BIND- findings |
| 3 | `make contract-audit-phase4` | **rc=0** — 15/15 equations bound, 18 obligations, **zero unbound** |
| 4a | `pv validate contracts/setfit-train-lifecycle-v1.yaml` | **rc=0** — 0 errors, 0 warnings |
| 4b | `pv validate contracts/setfit-benchmark-claims-v1.yaml` | **rc=0** — 0 errors, 0 warnings |
| 4c | `pv validate contracts/calibration-v1.yaml` | **rc=0** — 0 errors, 0 warnings |
| 5 | the full scoped SetFit suite | **17 of 18 targets rc=0**; `setfit-apr-tests` red on ONE pre-existing test (see below) |
| 6 | the doctored negatives, bare `cargo test` | **rc=0**, 38 passed |

`make setfit-all-tests` itself exits 2 because `make` stops at its first failing prerequisite. The
remaining seventeen targets were therefore each run individually so the audit is not truncated by a
single standing red: `setfit-classify/bundle/config/evaluate/codec/reload/lock/verify/lifecycle/ui-tests`
and `setfit-cli-train/predict/inspect/eval/io/serve-tests` and `setfit-serve-tests`, **all rc=0**.

### The refusal-variant count, READ FROM THE LOG

The plan forbids quoting this number from a plan document, and the test did not previously print
it — so it now does, and the count below is transcribed from a run:

```
$ ./target/debug/deps/entrenar-094a7a01ddf87b82 bench_gate --nocapture
[bench_gate] DOCTORED_NEGATIVES=6 DISTINCT_REFUSAL_VARIANTS=6 TAGS=["incomplete_cell",
  "row_schema_refused", "unpaired_selection", "row_digest_mismatch", "post_test_selection",
  "provenance_mismatch"]
test result: ok. 38 passed; 0 failed; 0 ignored; 0 measured; 8044 filtered out
```

**Six doctored shapes, six DISTINCT variant tags, named.** The test binary was invoked directly
rather than through `cargo test`, because this environment's tooling filters per-test lines out of
`cargo test` output and the tally line would have been invisible.

### The standing KNOWN-REDs, each named, classified, and given its reason

None of these is a Phase 5 regression, and each was measured here rather than recalled.

| red | status | why it is pre-existing |
|---|---|---|
| `setfit-apr-tests` — `setfit::artifact::determinism::the_fixture_artifact_hash_matches_the_committed_golden` | rc=101, **85 passed / 1 failed** | `git diff --stat 35cd7c210..HEAD -- crates/aprender-core/src/setfit/ crates/aprender-core/tests/fixtures/setfit/` is **EMPTY**. Phase 5 changed neither the code under test nor its fixtures, so the failure cannot be a phase-5 regression. Hash pair: left `cc17764d…aab675`, right `13e5c296…7000bf4`. Last touch of `artifact.rs` is the Phase 4 commit `fb7904bad`. D-ITEM-05-16's business |
| `cargo package -p apr-cli` | rc=101 | `failed to select a version for the requirement 'aprender-contrastive-data = "^0.63.0"'` — the crate is not on crates.io until the human-approved publish cascade. STATE.md line 425: red from Phase 2 wave 2 through phase exit, and `--no-verify` does **not** help because resolution, not the build, is what breaks |
| `make tier2` / strict clippy on arm64 | dependency crates fail under `-D warnings` | Every error is in `aprender-compute`, `aprender-zram-core` or `aprender-present-terminal` — arch-gated SIMD arms CI never lints (D-ITEM-02). **Neither touched crate contributes one**: `cargo clippy -p aprender-train --lib --features setfit` and `-p apr-cli` both report **0 findings in their own sources** |
| `cargo check --workspace` on Darwin | needs its documented exclusion | `cargo check --workspace --exclude aprender-profile` → **rc=0** here. `aprender-profile/src/main.rs:6` is a `compile_error!` under `cfg(not(linux))` (STATE.md line 426) |

`make test` was never invoked: `crates/aprender-profile/examples/validate_golden_trace.rs` imports a
linux-gated module and cannot compile on macOS. Targeted commands throughout.

---

## The closing evidence map — met in full versus met NARROWLY

**This map flips nothing.** `requirements.mark-complete` was not run;
`git diff --stat ec1bf58e845eea34f7b5cf0f606ee20b4c28c9c3..7c3a6b401348121e00c986bce3b238c5f1b2aa74 -- .planning/REQUIREMENTS.md`
is **0 bytes** — measured across this plan's OWN pinned commit range, both SHAs quoted, not across
an unpinned working-tree diff that for an already-committed change would report nothing and pass
vacuously.

| Id | Verdict | Evidence | Deferred clause |
|---|---|---|---|
| **EVAL-01** | **Met in full** | `report.md` QUALITY table: official `F_avg` with per-class-derived macro-F1 and MCC per group; `report.json` `detail` carries per-class metrics, the confusion matrix and validation-only calibration diagnostics bound to explicit ordered labels. Gate: `make contract-audit-phase5` rc=0; `bench_metrics` 14 passed | — |
| **EVAL-02** | **Met NARROWLY** | All **40** SetFit shot/seed cells, each bound to a recorded selection manifest (40 distinct `selection_manifest_hash`, 05-12). The report exists only over a complete, hash-verified, in-scope set | **The second arm.** Deferred as **`D-ITEM-05-15`**: no GPU host reachable, and the Qwen3.5-9B hybrid forward path is unimplemented. Amended 2026-09-07 (D-19). **Stays Pending** |
| **EVAL-03** | **Met in full** | One machine-readable row per cell with the contracted core and evidence block, 40 of 40 `complete` with `row_sha256` equal to each row's own `semantic_hash`; the *verified, reported* set now exists — `GATE_RUN1_RC=0`. `bench_row` 27 passed | — |
| **EVAL-04** | **Met NARROWLY** | Exact recomputation of means, dispersion and uncertainty from all 40 stored cells, demonstrated by **byte-identical re-run** (`cmp` rc=0 both artifacts, comparator proven able to fail). Invalidate-on-omission **UNWEAKENED**: rc=5 on a real-shaped omission, 0 bytes rendered; six distinct refusal variants read from the log | **The paired second-method delta clause.** The paired-t machinery is retained and contract-bound, unexercised — `bench_gate_deferred_scope_still_computes_the_paired_delta` still passes. Ticket **`D-ITEM-05-15`**. **Stays Pending** |
| **EVAL-05** | **Met NARROWLY** | The full cost picture *within* SetFit from the same reloaded production artifacts: train wall, cold/warm latency, throughput with batch and warmup boundaries, both peak-RSS figures with their mechanism strings and the sampled figure's lower-bound label, artifact bytes and `deployable_total_bytes`, across four shot levels x ten seeds | **Cross-method cost comparison.** Removed, not repurposed. Ticket **`D-ITEM-05-15`** |

**Verifier note — Phase 4 items whose user-facing tiers now have production evidence via 05-07.**
05-07 drove a user-produced `setfit-apr-v1` end to end at the **spawned-binary** tier —
`apr setfit train → inspect → eval(validation --lock-out) → eval(test --selection-lock) → predict`,
all exit 0, **one artifact SHA-256 read out of all five surfaces**. That retires the present-tense
F-10 claim that "no user-reachable path produces a setfit-apr-v1" (zero such claims remain in
`crates/`), and it restored `make setfit-lifecycle-tests` to green. Phase 4 items resting on the
absence of a user-reachable production path should be re-read against that evidence. **This is a
pointer, not a flip.**

---

## Deviations from Plan

### 1. [Rule 1 — Bug] The active report named the deferred method BELOW section level

- **Found during:** Task 1, by the plan's own `<automated>` verify failing on the first honest run.
- **Issue:** 05-11's retarget removed the delta and comparison **sections**, and its case table
  carried only **section-level** literals. Three sub-section strings survived and named the
  deferred method in the active report: the per-group row label `artifact bytes (adapter only for
  lora)`, printed on **all four** SetFit groups; the size table's four-sentence cross-method
  footnote (`A cross-method size claim uses THIS column and no other. LoRA's adapter-only …`); and
  the header's `verified:` clause, which credited a recomputation "from the committed lock and
  **ledger** bytes" when the active scope has no candidate ledger to recompute from. Six `lora`
  hits in `report.md`.
- **Why it matters:** T-05-13-04 exactly — a reader who sees the deferred method's name beside a
  SetFit number reads a comparison. The footnote is worse than a label: four sentences of
  cross-method claim language, including "would understate LoRA by the size of its base model",
  under a scope where LoRA was never run.
- **Fix:** three RETAINED-AND-DEFERRED constants and three ACTIVE replacements;
  `render_header`, `render_resource_detail` and `render_sizes` made scope-aware. The two renderers
  that receive only a resource slice **derive** the scope from the slice
  (`resource_is_cross_method`) rather than take a flag a caller can get wrong. The case table grew
  from six must-not-match rows to **nine**, each with its provenance recorded, and each asserted
  present in the two-method render so the absence assertions stay non-vacuous. A new
  **token-level** test refuses the deferred method's name by any spelling — the case table can only
  refuse literals someone thought to add, and this gap is proof of that limit.
- **Files:** `crates/apr-cli/src/commands/setfit_bench.rs`,
  `crates/apr-cli/src/commands/setfit_bench_tests.rs`, `Makefile` (floor 64 → **65**, raised to the
  measured count; banner corrected).
- **Verification:** `make setfit-bench-tests` rc=0; the control over the committed bytes reports
  `CONTROL_FAILURES=0` with a proven-failing negative.
- **Commit:** `f447d4946`.
- **Not a reason to weaken anything.** 05-11 anticipated the opposite tension — the mandated note
  tripping the gate — and pinned the note against the same table. That held; this was a different
  gap, in the renderer rather than in the note.

### 2. [Rule 3 — Blocking] The refusal-variant count could not be read from any log

- **Found during:** Task 3. The acceptance criterion requires the count be READ FROM THE LOG, and
  `bench_gate_the_six_doctored_negatives_are_six_distinct_variants` asserted six without printing
  anything. Reading "six" off the test's *name* would be quoting a document.
- **Fix:** the test now prints `DOCTORED_NEGATIVES`, `DISTINCT_REFUSAL_VARIANTS` and the six tags
  under `--nocapture`. Assertions unchanged.
- **Commit:** `7c3a6b401`.

### 3. [Rule 3 — Blocking] A dead import in Phase 5's own work

- **Found during:** Task 3's clippy pass. 05-11 left `ACTIVE_METHODS` imported and unused in
  `bench_gate.rs` — the phase's only clippy finding in either touched crate, and one that
  `-D warnings` would turn into a build failure.
- **Fix:** removed. Both touched crates now report **0 clippy findings in their own sources**.
- **Commit:** `7c3a6b401`.

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking). **Impact on the evidence: none.** No gate
was relaxed, no floor lowered, no row/lock/manifest byte edited. Two floors and one guard were
**strengthened**.

## A finding recorded, not fixed

`apr setfit bench report --help` still describes the two-method design: *"Point estimates,
dispersion and paired 95% CIs … a LoRA cell's no-selection attestation"*. It is a CLI doc comment,
not a report artifact, so it is outside this plan's prohibition and outside its `files_modified` —
but it is the same class of staleness Deviation 1 fixed, and it will read as a promise the active
scope does not keep. Left for the verifier to route rather than silently widened into this plan.

## Known Stubs

None. Every number in the committed report is computed by the shipped gate from the 40 committed
rows and their lock records; no value is placeholder, hardcoded or mocked. The deferred two-method
renderer is retained and unexercised by design (D-19), not stubbed — its tests still pass against
the two-method fixture.

## Threat Flags

None. No new network endpoint, auth path or trust-boundary schema; no new registry package
(T-05-13-SC discharged as `accept (n/a)`). Register discharged: **T-05-13-01** (repudiation) —
two-run byte identity on the real set with `cmp`, comparator proven able to fail, exclusion set
empty by inspection; **T-05-13-02** (tampering) — both destructive controls on copies, real
directory re-verified with 0 changed files; **T-05-13-03** (elevation) — evidence map only,
REQUIREMENTS.md 0-byte diff across the pinned range, `mark-complete` not run; **T-05-13-04**
(implied comparison) — three enforcements, and the render-time one **fired and found a real
defect**; **T-05-13-05** (misleading resource figure) — both peaks separate with mechanism strings
and the lower-bound label, verified in the COMMITTED report; **T-05-13-06** (KNOWN-RED misread) —
four standing reds named, classified, each with a measured reason.

## Issues Encountered

The pre-existing failures named in the dispatch did not block this plan and were not touched. One
was measured rather than accepted on report: the `setfit::artifact::determinism` red is proven
pre-existing by an **empty** phase-5 diff over its source and fixtures, which is stronger evidence
than the dispatch note it corroborates. The 21 `aprender-train gpu::{guard,ledger,wait}` tests were
never in any suite this plan ran.

## Next Phase Readiness

**Phase 5's deliverables are complete and its gates are green at `7c3a6b401`.** Marking the phase
complete and flipping any requirement state is the verifier's act. Three things the verifier should
carry:

1. **EVAL-02, EVAL-04 and EVAL-05 are met NARROWLY.** Each deferred clause and its ticket
   (`D-ITEM-05-15`) is named in the map above. Filing them against the unamended text would be the
   false-completion defect this project has hit six times.
2. **`apr` is feature-gated and pin-checked.** `cargo build --release --bin apr --features setfit`
   (~80 s), re-run after any commit; `scripts/apr_bin.sh` compares the embedded SHA against HEAD
   exactly. It refused four times during this plan, correctly, and was never weakened.
3. **The `bench report --help` doc comment is stale** (see the finding above) — a one-line routing
   decision, not a defect in any committed artifact.


## Self-Check: PASSED

| Claim | Check | Result |
|---|---|---|
| `benchmarks/tweeteval-stance/report.md` | `test -f` | FOUND |
| `benchmarks/tweeteval-stance/report.json` | `test -f` | FOUND |
| `05-13-SUMMARY.md` | `test -f` | FOUND |
| commits `f447d4946`, `3968d98a7`, `e52dd4bff`, `7c3a6b401` | `git log --oneline --all` | all FOUND |
| commit count | `git rev-list --count ec1bf58e8..7c3a6b401` | **4**, MEASURED |
| REQUIREMENTS.md untouched | `git diff --stat ec1bf58e8..7c3a6b401 -- .planning/REQUIREMENTS.md` | **0 bytes** |
| Task 1 `<automated>` verify | `bash -c '…'` | **rc=0** |
| Task 2 `<automated>` verify | `cargo test -p apr-cli --lib qa` | **rc=0**, 266 passed |
| Task 3 `<automated>` verify | the three make gates chained | **rc=0 / rc=0 / rc=0** |
| no-second-method control | 9 literals + 6 tokens + 18 positives | `CONTROL_FAILURES=0`, negative proven at rc=1 |
| evidence set untouched | `git status --short` over rows/locks/selections/manifest | **0 changed files** |
