---
phase: 05-benchmark-and-claims-gate
plan: 10
subsystem: library+cli
tags: [claims-gate, eval-04, setfit, lora, fail-closed, provenance, statistics, makefile, binding]

requires:
  - phase: 05-benchmark-and-claims-gate
    provides: "05-04 closed-form f64 statistics (mean/std/min-max, paired_ci95_df9, ttest_rel_f64, T_CRIT_975_DF9, ZeroVarianceDifferences); 05-05 BenchRow/RunManifest + the 80-cell expectation set; 05-08 assemble_quality_block; 05-09 `apr setfit bench run`, the committed lock records and candidate ledgers this gate recomputes from"
provides:
  - "`entrenar::train::setfit::bench_gate::verify_run` — the fail-closed boundary guard over a whole benchmark directory"
  - "`VerifiedRunSet` — no public constructor, so `aggregate` cannot run on unverified data"
  - "`bench_gate::aggregate` — closed-form, deterministic, bit-reproducible recomputation"
  - "`Ci95` with `null_reason: zero_variance` — a degenerate interval that is visible rather than `null`"
  - "`mechanism_class` / `mechanisms_are_comparable` — the resource-comparability predicate the renderer labels with"
  - "`apr setfit bench report [--json] [--out FILE]` — verified data only, estimation-first"
  - "`ROWS_DIR` / `LOCKS_DIR` / `LEDGER_DIR` / `RUN_MANIFEST_FILE` / `row_file_name` — one spelling, shared by the writer and the gate"
  - "`make setfit-bench-tests` — four suites with measured assert_tests_ran floors, wired into tier3"
  - "`make contract-audit-phase5` — now requires ZERO BIND- lines of any severity"
affects: [05-11 the 40 remote 9B cells, 05-12 the 40 local SetFit cells, 05-13 the report over real data]

actuals:
  tokens: 118000
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "A typestate whose guarantee is load-bearing enough to make the consumer INFALLIBLE: `aggregate` takes `VerifiedRunSet`, returns `RunAggregate` with no Result, and resolves the helpers' `Option`s with `expect` on a stated invariant — if the type cannot carry that, the typestate is decoration"
    - "Two failure modes with the same symptom get two variants: a trimmed row is an omission, a bit-flipped one is tampering, and `variant_tag()` is what lets a test assert six mutations produce SIX distinct refusals rather than one catch-all"
    - "`skip_serializing_if = \"Option::is_none\"` as the anti-`null` mechanism, checked by a STRUCTURAL walk for `Value::Null` rather than a substring grep for NaN/Infinity — the grep would find neither and would collide with the `null_reason` key"
    - "A two-sided control on every conditional label: the incomparability note is proven present on a mixed-mechanism row AND absent on a same-mechanism one, because a note that always fires is boilerplate a reader learns to skip"
    - "Test fixture constants chosen so no assertion can be decided by digit coincidence (40_000_000 is a substring of 18_040_000_000)"
    - "Recording a control that did NOT fire, when the reason is a real limit of the tooling"

key-files:
  created:
    - crates/aprender-train/src/train/setfit/bench_gate.rs
    - crates/aprender-train/src/train/setfit/bench_gate_tests.rs
  modified:
    - crates/aprender-train/src/train/setfit/mod.rs
    - crates/apr-cli/src/commands/setfit_bench.rs
    - crates/apr-cli/src/commands/setfit_bench_tests.rs
    - crates/apr-cli/src/setfit_commands.rs
    - crates/apr-cli/src/dispatch_analysis.rs
    - Makefile
    - contracts/aprender/binding.yaml

key-decisions:
  - "`verify_run` takes the BENCH DIRECTORY, not the rows directory the plan named. A row's `lock_record_path` and `candidate_ledger_path` are relative to the benchmark directory, so a rows-only parameter could not resolve the very files the provenance recomputation reads — the plan's own headline requirement would have been unimplementable as specified."
  - "The manifest's recorded digest is compared to the row's `semantic_hash`, NOT to a hash of the row FILE's bytes. `emit_row` records the semantic hash, the file is pretty and the digest is over canonical compact bytes (bench_row_schema's own invariant), so a file-bytes comparison would call a reformatted-but-identical row tampering. The check is not weakened: `from_bytes` has already proven the two agree."
  - "`aggregate` is INFALLIBLE. A fallible signature would have been an admission that `VerifiedRunSet` does not actually guarantee ten seeds per group, which is the one thing it exists to guarantee."
  - "The layout constants and `row_file_name` moved into the library and the CLI now delegates. Two spellings of a filename are two filenames, and a drift would have made every cell look un-run while reporting a missing-file error naming a path the writer never used."
  - "The `--json` payload nests the aggregate under a `detail` key. D-08 permits p-values in the machine-readable detail and forbids them in claim language; a key named `detail` states that boundary structurally rather than leaving it to a convention a future renderer can forget."
  - "The estimation-first note avoids the verdict word entirely rather than using it in a negation. `\"significance\"` does not contain `\"significant\"`, so the scan would have passed — but a report that types the word at all invites a reader to take it away."
  - "EVAL-04 is NOT marked complete. The mechanism is done and falsified six ways; the requirement says a user can recompute from all 40 stored comparison cells, and those cells do not exist until 05-11 and 05-12 run them."

requirements-completed: []

coverage:
  - id: D1
    description: "`verify_run` refuses each of the six doctored dishonesty shapes BEFORE any arithmetic, and each refusal is its own typed variant naming the cell"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_refuses_a_missing_cell_naming_it"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_refuses_a_trimmed_row_whose_evidence_block_was_removed"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_refuses_a_pair_measured_on_different_selection_manifests"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_refuses_a_row_whose_payload_bytes_were_edited"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_refuses_a_setfit_row_whose_lock_rule_is_not_the_committed_one"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_refuses_a_lora_row_that_completed_fewer_epochs_than_it_requested"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_refuses_a_ledger_carrying_a_second_candidate_the_row_does_not_declare"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_refuses_a_setfit_row_whose_committed_lock_file_was_edited"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_the_six_doctored_negatives_are_six_distinct_variants"
        status: pass
    human_judgment: false
  - id: D2
    description: "Provenance is RECOMPUTED from committed bytes, not read off the row — proven by an induced failure that replaced the recomputation with the row's own field"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_verify_run_reads_the_lock_and_ledger_bytes_from_disk"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_refuses_a_ledger_transplanted_from_another_cell"
        status: pass
      - kind: command
        ref: "induced RED: `let recomputed = evidence.lock.lock_hash.clone()` -> cargo test bench_gate rc=101, 2 failed (the edited-lock negative and the six-distinct-variants test); reverted -> rc=0, 31 passed"
        status: pass
    human_judgment: false
  - id: D3
    description: "Aggregation is closed-form, deterministic to the BIT, ordered by the contract, and cross-pinned to the contract's frozen t literal"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_two_aggregations_of_one_run_are_bit_identical"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_aggregate_emits_the_pinned_key_sequence"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_aggregate_recomputes_the_closed_form_summary_from_the_rows"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_frozen_t_constant_matches_the_contract_by_bits"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_source_carries_no_rng_or_resampling_vocabulary"
        status: pass
    human_judgment: false
  - id: D4
    description: "A zero-variance paired delta set surfaces a point estimate plus an explicit no-CI marker; nothing in the aggregate serializes to `null` and every f64 is finite"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_a_zero_variance_delta_set_reports_a_point_estimate_and_no_interval"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#report_render::setfit_bench_report_renders_a_zero_variance_delta_as_a_named_absence"
        status: pass
    human_judgment: false
  - id: D5
    description: "`aggregate` is unreachable on an unverified set: `VerifiedRunSet` has no public constructor and no public field"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/bench_gate_tests.rs#bench_gate_verified_run_set_has_no_public_constructor"
        status: pass
      - kind: command
        ref: "grep -c 'pub fn new\\|pub const fn new' bench_gate.rs -> 0"
        status: pass
    human_judgment: false
  - id: D6
    description: "The report prints no verdict word, carries the D-09 per-host framing, and labels a mixed-mechanism comparison two-sidedly"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#report_render::setfit_bench_report_prints_no_verdict_word_and_names_the_interval"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#report_render::setfit_bench_report_human_output_carries_the_per_host_framing_line"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#report_render::setfit_bench_report_labels_a_mixed_mechanism_comparison_and_leaves_a_matched_one_alone"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#report_render::setfit_bench_report_resource_detail_names_every_measurement_scope"
        status: pass
    human_judgment: false
  - id: D7
    description: "Cross-method size claims use `deployable_total_bytes`; LoRA's adapter-only figure appears only in the labelled per-method detail"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#report_render::setfit_bench_report_size_table_uses_deployable_and_never_the_adapter_only_figure"
        status: pass
    human_judgment: false
  - id: D8
    description: "A refused run set produces exit-nonzero, a cell-naming message with a remedy, and ZERO table output"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#report_render::setfit_bench_report_refuses_an_incomplete_run_naming_the_cell_and_renders_nothing"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/setfit_bench_tests.rs#report_render::setfit_bench_report_refuses_a_directory_with_no_run_manifest"
        status: pass
    human_judgment: false
  - id: D9
    description: "The Make floors and the binding audit are non-vacuous, proven by induced failures in both"
    requirement: "EVAL-04"
    verification:
      - kind: command
        ref: "floor RED: filter `bench_gate` -> `bench_gaate`, make setfit-bench-tests rc=2, 'reported 0 test(s) passed, expected at least 29'; reverted rc=0"
        status: pass
      - kind: command
        ref: "binding RED: one row -> `status: pending`, make contract-audit-phase5 rc=2, one BIND-004, 'Implemented: 9'; reverted rc=0, 'Implemented: 10', zero BIND- lines"
        status: pass
    human_judgment: false
  - id: D10
    description: "The gate has NOT been run against real benchmark rows — none exist yet"
    verification:
      - kind: manual
        ref: "05-11 (40 LoRA cells, lambda-vector GPU) and 05-12 (40 SetFit cells, local CPU) produce them; 05-13 renders the report over real data"
        status: deferred
    human_judgment: true
    rationale: "Every test above runs against a programmatically generated synthetic 80-row set. That is the right fixture for a GATE — it is what lets each dishonesty shape be introduced one at a time — but it is not evidence that the gate accepts the rows the real pipeline writes. The synthetic builder was written against `bench_row`'s types and `setfit_bench.rs`'s committed lock/ledger layout, so the shapes should agree; `should` is not `does`, and 05-12 is the first run that will find out."

duration: 195min
completed: 2026-08-17
status: complete
---

# Phase 5 Plan 10: The Claims Gate and `apr setfit bench report` Summary

**`verify_run` walks `setfit-benchmark-claims-v1`'s rules in the contract's order and returns on the first failure, recomputing every lock and ledger digest from the committed bytes rather than reading the row's claim about them; `aggregate` takes a `VerifiedRunSet` that only `verify_run` can construct, so there is no partial-data path to express; and `apr setfit bench report` renders what was verified — estimation-first, mechanism-labelled at the point of every cross-method comparison, and refusing outright rather than printing a shrunken table.**

## Performance

- **Duration:** ~195 min
- **Tasks:** 3
- **Commits:** 3
- **Files:** 9 (2 created, 7 modified)

## Task Commits

1. **Task 1: `bench_gate` — fail-closed verification + closed-form aggregation + six negatives** — `37f0793e7` (feat)
2. **Task 2: `bench report` CLI — human + `--json`, estimation-first** — `dc66a47b1` (feat)
3. **Task 3: Make gates + binding rows, both non-vacuous** — `073892ae3` (chore)

## The one property this plan exists for, and how it is actually enforced

EVAL-04 says missing, selectively omitted, unmatched or post-test-selected cells invalidate the report. That is a structural claim, so it is enforced structurally rather than asserted:

| the claim | the mechanism | what makes it more than a convention |
|---|---|---|
| no partial-data mode | `aggregate(&VerifiedRunSet)` | `VerifiedRunSet` has no public constructor and no public field; a source guard scans for `pub fn new`, `pub fn from_rows` and `pub rows:` and requires zero |
| refuse before arithmetic | the seven-step walk, returning on the first failure | the two vacuity backstops run before ANY row byte is read |
| selection safety is evidence, not a boolean | the lock and ledger bytes are read and re-digested; the ledger's lines are COUNTED | an induced mutation replacing the recomputation with the row's own field turned two tests RED |
| the arithmetic is recomputable | only 05-04's closed-form f64 helpers, no local mean or std | a source scan for `rand::`, `thread_rng`, `bootstrap`, `resample`, `shuffle` requires zero, with a non-vacuity guard on the haystack |
| the report cannot mislead by construction | mechanism strings + scopes per column, a two-sided incomparability note, `deployable_total_bytes` for size | each is a test, and each absence assertion is paired with a presence one |

## Verification order is the property, and one step of it moved

The plan's order is implemented as written with one exception worth naming. The plan says `from_bytes` should be reached after "digest-green, schema-green"; `BenchRow::from_bytes` internally parses first and checks the envelope digest second, and that order belongs to `bench_row` (05-05) and is documented there. The gate therefore uses `from_bytes` as the ONE door and splits its outcome by variant tag:

- `semantic_hash_mismatch` → `RowDigestMismatch` (the row contradicts itself — tampering)
- anything else → `RowSchemaRefused` (a trimmed block, an unknown field, a foreign schema — omission)

That split is what makes doctored negatives 2 and 4 distinguishable at all. Without it they are one catch-all, and "a missing cell is an omission, a hash failure is tampering, and an unpaired hash is an incomparable comparison" — the contract's own words — would be three defects with one symptom.

## The residual, stated in three places rather than hidden

Recomputing the lock and ledger digests raises selection safety from a self-asserted boolean to append-only, recomputable evidence. **It does not reach a cryptographic train-then-seal credential, and a producer that controls both the rows and those files can still emit a mutually consistent forgery.** The six negatives prove detection of INCONSISTENT evidence; they prove nothing about truthful provenance.

That sentence is in `bench_gate.rs`'s module doc, in `bench_gate_tests.rs`'s header (so a reader of the test list does not conclude more from a green suite than it supports), and in the rendered report's own header block (so a reader of the OUTPUT sees it without reading source). It matches `selection_safety_evidence.residual_risk`'s wording in the contract.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 3 — Blocking] `verify_run(manifest, rows_dir)` cannot resolve the files it is required to recompute from**

- **Found during:** Task 1, writing the signature.
- **Issue:** The plan specifies `verify_run(manifest, rows_dir)`. A row's `lock.lock_record_path` and `candidate_ledger_path` are RELATIVE TO THE BENCHMARK DIRECTORY (`locks/…`, `ledger/…`) — 05-09 committed them that way and 05-09-SUMMARY records the layout. Given only `{bench}/rows`, the gate would have had to walk up a level by string manipulation to reach them, or trust the row fields instead of recomputing. The second is the plan's own headline requirement inverted.
- **Fix:** the parameter is `bench_dir`, and `rows_dir` is derived as `bench_dir.join(ROWS_DIR)`. Documented on the function with the reason, so the next reader does not "simplify" it back.
- **Files:** `crates/aprender-train/src/train/setfit/bench_gate.rs`
- **Committed in:** `37f0793e7`

**2. [Rule 1 — Bug] The manifest-digest check as first written would have refused a byte-identical row**

- **Found during:** Task 1, wiring the manifest comparison.
- **Issue:** the contract's `completeness_rule` formula reads `sha256(bytes(row_file(c))) == manifest.row_sha256(c)`, but `emit_row` records `row.semantic_hash` — the digest over the payload's CANONICAL COMPACT bytes — and writes the file PRETTY. Comparing the file's bytes would therefore have called a reformatted-but-identical row tampering, and would have been red against every row the shipped writer produces.
- **Fix:** the comparison is against `row.semantic_hash`. It is not weakened: `from_bytes` has already proven `semantic_hash == sha256(canonical(payload))`, so any payload difference at all changes the value being compared. The reasoning is recorded at the check.
- **Note:** this is a looseness in the CONTRACT's prose, not in the code — `bench_row_schema`'s own invariant says explicitly that the digest is over canonical compact bytes and the file is pretty. No contract edit was made; a schema amendment is a deliberate change with its own `pv diff` bump, not a drive-by.
- **Committed in:** `37f0793e7`

**3. [Rule 2 — Missing critical] The row filename grammar and the directory layout had two copies**

- **Found during:** Task 2.
- **Issue:** `setfit_bench.rs` defined `ROWS_DIR`/`LOCKS_DIR`/`LEDGER_DIR`/`RUN_MANIFEST_FILE` and `row_file_name`. The gate needs all five. Two spellings of a filename are two filenames — and the failure mode is nasty: a drift makes every cell look un-run while the error names a path the writer never used.
- **Fix:** all five moved into `bench_gate` and the CLI now `pub(crate) use`s them. A test asserts the delegation is present in source and that each constant equals the library's.
- **Committed in:** `dc66a47b1`

**4. [Rule 2 — Missing critical] `expect_err` on `verify_run` dumps eighty rows**

- **Found during:** Task 1, reading the induced-RED output.
- **Issue:** `Result::expect_err` prints the `Ok` value's `Debug`, and a `VerifiedRunSet`'s `Debug` is 80 nested structs — 182 KB of failure output for a one-line assertion. A diagnostic nobody can read is a diagnostic that does not exist.
- **Fix:** a `refuse(manifest, root, doctored)` helper that panics with the doctored shape's name and the verified COUNT.
- **Committed in:** `37f0793e7`

### Deviations that are NOT auto-fixes, and are recorded as findings

**5. The plan's binding-audit falsification cannot fire, and the Makefile now says so**

The plan asks: *"temporarily point one binding row at a nonexistent symbol and observe the audit refuse."*

MEASURED, status captured directly: `function: this_symbol_does_not_exist_anywhere` on the `pairing_rule` row → `make contract-audit-phase5` **rc=0**, `Implemented: 10`, zero BIND- lines. `pv audit` does not resolve symbols; it reads the `status` field — which `contracts/aprender/binding.yaml`'s own Phase 4 block states, and which is exactly why plan 04-10 declined to flip a status it could not verify.

So the requested control proves nothing, and crediting the gate with it would be worse than recording no control at all (CLAUDE.md rule 5's sibling: a gate credited with a failure it cannot detect). Two things were done instead:

- the control that DOES exercise the strengthened gate was run: one row flipped to `status: pending` → **rc=2**, one BIND-004 line, `Implemented: 9`, then reverted → **rc=0**, `Implemented: 10`, zero BIND- lines, obligations covered 90.
- the Makefile comment records the non-firing control and states plainly that neither this gate nor anything else in the repository can detect a `module_path`/`function` pair that names nothing — those columns are checked by review and by the resolution each plan performs before it flips a status.

**6. The plan's floor falsification, as literally specified, would also not have fired**

The plan says *"temporarily rename one bench test and observe the floor fail."* Renaming one test takes `bench_gate` from 31 to 30, and the floor is 29 — green, correctly. `assert_tests_ran` is a VACUITY guard with a deliberate margin, not an exact-count assertion. The mutation that exercises what the floor is for is a misspelled FILTER: `bench_gate` → `bench_gaate` → **rc=2**, `reported 0 test(s) passed, expected at least 29`, with the underlying `cargo test` exiting 0. Reverted → rc=0.

**7. EVAL-04 is not marked complete**

The plan's frontmatter lists `requirements: [EVAL-04]`. The requirement reads: *"A user can recompute headline means, dispersion, uncertainty, and paired SetFit-versus-LoRA deltas exactly from all 40 stored comparison cells, and missing or selectively omitted cells invalidate the report."*

The second clause is delivered and falsified six ways. The first is not: there are no stored comparison cells yet — 05-11 and 05-12 run them. Ticking the box now would put a green tick beside a claim nothing measured, which is what 05-09 refused for the same reason (its coverage D8). `requirements-completed` is empty and `REQUIREMENTS.md` is untouched.

---

**Total deviations:** 4 auto-fixed (1 bug, 2 missing critical, 1 blocking), 3 recorded findings. No Rule 4 architectural decision arose.

## Verification, as measured

Status captured directly off each command, never through a pipe (CLAUDE.md rule 1).

| command | result |
|---|---|
| `cargo test -p aprender-train --lib --features setfit bench_gate` | **31 passed**, 0 failed, rc=0 |
| `cargo test -p aprender-train --lib --features setfit bench_row` | **24 passed**, rc=0 |
| `cargo test -p aprender-train --lib --features setfit bench_metrics` | **14 passed**, rc=0 |
| `cargo test -p apr-cli --lib --features setfit setfit_bench` | **58 passed**, rc=0 (was 47 before this plan) |
| `cargo test -p apr-cli --lib --features setfit finetune` | **81 passed**, rc=0 — no regression |
| `cargo clippy -p aprender-train --features setfit --lib --tests` | rc=0, 0 findings in changed files |
| `cargo clippy -p apr-cli --features setfit --lib --tests` | rc=0, 0 findings in changed files |
| `make setfit-bench-tests` | rc=0 |
| `make contract-audit-phase5` | rc=0, `Implemented: 10`, zero BIND- lines, obligations covered 90 |
| `make -n tier3` | rc=0, `setfit-bench-tests` present with all four floors |

### The four induced-control runs (red, green, red, green)

| control | red | green |
|---|---|---|
| test-count floor (`bench_gate` → `bench_gaate`) | rc=**2**, "reported 0 test(s) passed, expected at least 29" | rc=**0**, 24/31/14/58 passed |
| binding audit (one row → `status: pending`) | rc=**2**, one BIND-004, `Implemented: 9` | rc=**0**, `Implemented: 10`, zero BIND- lines |

Plus a fifth, on the plan's sharpest claim, run before either gate existed:

| control | red | green |
|---|---|---|
| provenance recomputation (`sha256_hex(&bytes)` → `evidence.lock.lock_hash.clone()`) | rc=**101**, 2 failed — `..._committed_lock_file_was_edited` and `..._six_distinct_variants` | rc=**0**, 31 passed |

And a sixth that did NOT fire, recorded because it is the more useful finding:

| control | result |
|---|---|
| binding row → `function: this_symbol_does_not_exist_anywhere` | rc=**0**, `Implemented: 10`, zero BIND- lines. `pv audit` does not resolve symbols. |

### Acceptance greps

| grep | result |
|---|---|
| `pub fn new\|pub const fn new` in `bench_gate.rs` | **0** (`VerifiedRunSet` is unconstructible from outside `verify_run`) |
| `read_evidence(cell` in `bench_gate.rs` | 4 (both provenance branches read bytes from disk) |
| `setfit-bench-tests` in `grep -v '^#' Makefile` | **7** (non-zero, as the criterion requires) |
| `assert_tests_ran` floors in `make -n tier3` | 4, values 22 / 29 / 12 / 55 |

## Tier placement, and why the first estimate was wrong by twenty-fold

The plan allowed a tier2 subset if a fast leg measured under 5 s. The comment first written claimed ~11 s wall; MEASURED, two consecutive runs on an already-built tree gave **166 s** and **236 s** wall (417 s user — the box compiles in parallel). Test execution inside that is 1.7 s total.

The cause is a property of the target's SHAPE, not a cold cache: `cargo test -p aprender-train --features setfit` and `cargo test -p apr-cli --features setfit` unify features differently across their shared dependency graphs, so each alternation re-links the other's artifacts. `setfit-tests` has the same shape for the same reason, and a single invocation is unavailable because `cargo test` accepts at most one positional filter.

tier3 only. No leg is tier2-shaped — the cheapest one still pays the whole cross-crate re-link, so splitting the gate would buy a second re-link for no earlier signal. The measured numbers are in the Makefile comment, replacing the guess.

## Known Stubs

None. `bench_gate.rs` and the report renderer are complete implementations with no placeholder returns, no `todo!`, and no constant standing in for a computation.

One thing that is NOT a stub but must not be read as coverage: **every test in this plan runs against a programmatically generated synthetic 80-row set.** That is the right fixture for a gate — it is what lets each dishonesty shape be introduced one at a time against an otherwise-perfect run — but it is not evidence that the gate accepts the rows the real pipeline writes. The synthetic builder was written against `bench_row`'s types and the lock/ledger layout 05-09 committed, so the shapes should agree. *Should* is not *does*; 05-12 is the first run that will find out. Recorded as coverage D10 with `status: deferred` rather than left to look like a gap nobody noticed.

## Threat Flags

None beyond the plan's `<threat_model>`. Each mitigated threat has a named passing test:

| threat | mitigation | test |
|---|---|---|
| T-05-10-01 doctored rows | digest + schema + manifest-hash + slot per row | `..._payload_bytes_were_edited`, `..._trimmed_row_whose_evidence_block_was_removed`, `..._substituted_for_the_one_the_manifest_recorded`, `..._filed_under_the_wrong_slot` |
| T-05-10-02 selective omission | the contract-derived 80 compared before any row is read; every cell must be complete | `..._missing_cell_naming_it`, `..._zero_cell_manifest...`, `..._expectation_set_is_not_the_contracted_eighty` |
| T-05-10-03 post-test selection | lock role + rule; all five LoRA conjuncts, each exercised separately | `..._lock_rule_is_not_the_committed_one`, `..._completed_fewer_epochs...`, `..._every_conjunct_of_the_lora_attestation_separately` |
| T-05-10-04 RNG in aggregation | `VerifiedRunSet` typestate; only 05-04 helpers; bit-identical determinism | `..._no_public_constructor`, `..._no_rng_or_resampling_vocabulary`, `..._bit_identical` |
| T-05-10-05 verdict laundering | no verdict vocabulary in the renderer; p-values under `detail` only | `..._prints_no_verdict_word_and_names_the_interval`, `..._json_payload_round_trips...` |
| T-05-10-06 forged provenance | digests recomputed from committed bytes; ledger lines counted; ledger's own manifest hash cross-checked | `..._ledger_carrying_a_second_candidate...`, `..._committed_lock_file_was_edited`, `..._ledger_transplanted_from_another_cell`; residual stated in three places |
| T-05-10-07 misleading like-for-like | mechanism + scope per column; two-sided incomparability note; `deployable_total_bytes` only | `..._labels_a_mixed_mechanism_comparison_and_leaves_a_matched_one_alone`, `..._size_table_uses_deployable_and_never_the_adapter_only_figure`, `bench_gate_mechanism_class_case_table` |
| T-05-10-SC package installs | none in this plan | n/a |

## Issues Encountered

### `bashrs make lint` on the Makefile

`rc=2, 1 error, 37 warnings` — against `rc=2, 1 error, 36 warnings` on the committed Makefile at `HEAD`. The error (`SC2168: 'local' is only valid in functions`, in `dev-setup`) is pre-existing and untouched. The one added warning is `MAKE012` (recursive make) on the new `$(MAKE) setfit-bench-tests` line in tier3, matching the 21 recursive invocations tier2 and tier3 already use — the house pattern, not a defect this plan introduced. Measured both ways rather than assumed: the warning categories were diffed between `HEAD`'s Makefile and the working one.

### `cargo fmt -p <crate>` reformats files this plan does not touch

Same finding 05-09 recorded, still true: `cargo fmt -p aprender-train` wants to reformat `apr_reload.rs`, and `cargo fmt -p apr-cli` touches unrelated files. `rustfmt` was therefore run on the six files this plan owns, individually. One incidental change did slip in — `rustfmt` stripped a leading blank line from `setfit_commands.rs` (an `include!`d fragment) — and was restored, so that file's diff is additions only.

## Next Phase Readiness

**Ready for 05-13 (the report over real data):**
- `apr setfit bench report --bench-dir <DIR> [--json] [--out FILE]` is the whole surface. It refuses unless all 80 cells verify.
- `bench_gate::verify_run` + `aggregate` are callable directly for a determinism check; the synthetic run builder in `bench_gate_tests.rs` is available as a fixture generator.
- The `--json` payload is `{schema, contract_id, detail: RunAggregate}`; `detail.deltas[].per_seed_deltas` and `detail.deltas[].p_value` are the machine-readable half.

**Blocking 05-13, and NOT closed by this plan:**
- **There are no rows.** 05-11 (40 LoRA cells on lambda-vector) and 05-12 (40 SetFit cells, local CPU) must both complete before `bench report` can produce anything but a refusal. That is the design working, not a gap.
- **05-06's blocker for 05-11 still stands**, carried forward unchanged from 05-09: on CPU, `apr finetune --task classify` never touches the LoRA adapters. Whether the GPU path shares that graph cut is a different question neither plan answers.

**A note for whoever runs 05-12 first:** the first real `bench report` invocation is also the first test of whether the synthetic fixture's shapes match the pipeline's. If it refuses, read the variant tag before assuming the rows are wrong — `row_schema_refused` on a freshly written row means the FIXTURE was wrong, not the run.

## Self-Check: PASSED

Files verified present:
- `crates/aprender-train/src/train/setfit/bench_gate.rs` — FOUND (63.9K)
- `crates/aprender-train/src/train/setfit/bench_gate_tests.rs` — FOUND (53.2K)
- `crates/apr-cli/src/commands/setfit_bench.rs` — FOUND (121.7K)
- `Makefile` — FOUND (172.6K)
- `contracts/aprender/binding.yaml` — FOUND (87.0K)

`must_haves.artifacts` contents verified:
- `bench_gate.rs` contains `verify_run` — 8 matches; `pub fn new` on `VerifiedRunSet` — 0 matches
- `setfit_bench.rs` contains the `report` module with `render_human` and `verified_aggregate`
- `Makefile` contains `setfit-bench-tests` — 7 non-comment matches, and `contract-audit-phase5` extended with the zero-BIND check

Commits verified in `git log`: `37f0793e7`, `dc66a47b1`, `073892ae3` — all 3 FOUND.

Working tree clean at the point of this check.

---
*Phase: 05-benchmark-and-claims-gate*
*Completed: 2026-08-17*
