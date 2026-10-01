---
phase: 05-benchmark-and-claims-gate
plan: 15
subsystem: testing
tags: [eval-04, claims-gate, path-traversal, symlink, provenance, bench_gate, setfit, rust]

requires:
  - phase: 05-benchmark-and-claims-gate (05-10)
    provides: "verify_run, the seven-step fail-closed claims gate, BenchGateError and the six doctored negatives"
  - phase: 05-benchmark-and-claims-gate (05-11)
    provides: "the D-19 ACTIVE 40-cell narrowing, the re-mutation rule, verify_cell and the two out-of-scope refusals"
  - phase: 05-benchmark-and-claims-gate (05-13)
    provides: "the phase-5 bench surface state and the apr-cli rendering case table"
provides:
  - "resolve_committed_evidence_path: the ONE door a row-supplied path may reach the filesystem through, with a syntactic stage and a canonical-containment stage, in that order"
  - "BenchGateError::EvidencePathEscape and ::EvidenceFileMissing, plus EvidenceKind { Row, Lock, Ledger }"
  - "an enumeration of the gate's ENTIRE row-supplied input surface, classed (i)/(ii)/(iii), which found a THIRD unvalidated field the two measured gaps were hiding"
  - "contract-derived comparisons for contract_id (row and manifest), calibration_split, warmup_count and cold_measured_in_child_process"
  - "scripts/setfit_bench_gate_door_probe.sh: verifier spot-check E replayed through the shipped apr door, with a positive control first"
  - "make setfit-bench-door-probe, and a setfit-bench-gate floor of 45 read off the suite's own log"
affects: [05-16, 05-17, D-ITEM-05-15, benchmark claims gate, EVAL-02, EVAL-04]

actuals:
  tokens: 28995
  tasks: 3
  commits: 4
plan_head_before: bfa6a25a33cd09d9236637d5f1a7dcdd5dc9c834

tech-stack:
  added: []
  patterns:
    - "One validation door for a whole field CLASS, not a patch per measured instance"
    - "Syntactic refusal BEFORE any filesystem call, so a 'does not exist' answer cannot mask an escape"
    - "Path::starts_with (component-wise) over canonicalized BOTH sides, never str::starts_with"
    - "Enumerate the input surface first, then fix — so a third unvalidated field cannot hide behind the two that were measured"
    - "A parameterized case table with ACCEPTANCE rows, not one test per incident"
    - "A door-level probe with a POSITIVE CONTROL that must pass before the attack means anything"

key-files:
  created:
    - ".planning/phases/05-benchmark-and-claims-gate/05-15-gate-input-surface.md"
    - "scripts/setfit_bench_gate_door_probe.sh"
    - "scripts/setfit_bench_gate_doctor.py"
  modified:
    - "crates/aprender-train/src/train/setfit/bench_gate.rs"
    - "crates/aprender-train/src/train/setfit/bench_gate_tests.rs"
    - "Makefile"

key-decisions:
  - "WR-06 taken as option (a): read_evidence takes an EvidenceKind and a non-row kind yields EvidenceFileMissing. Minting a correctly-typed path-escape variant beside a mis-typed missing-file variant inside one function would have been incoherent."
  - "resolve_committed_evidence_path returns the JOINED path, not the canonical one, so downstream refusals keep naming files by the spelling on disk. Canonicalization is used only for the containment CHECK."
  - "Three MORE contract-pinned constants closed beyond the plan's named contract_id (calibration_split, warmup_count, cold_measured_in_child_process), because Task 1 conditioned it on being a one-line comparison and it was. throughput_batch_size deliberately NOT closed: the contract pins no value for it."
  - "The variant_tag arm-count guard was raised 13 -> 15 with an argued rationale per minted variant, rather than deleted. The four new comparisons mint nothing; that asymmetry is the guard's point."
  - "The probe's Python was extracted to a sibling .py file because bashrs does not skip a quoted heredoc body and reported twelve phantom shell parse errors against the Python inside one."

patterns-established:
  - "Scope is part of what a negative proves: the same three shapes are proven at verify_run/ACTIVE-40 AND at the resolver helper, and neither proof discharges the other"
  - "A test-count floor is MEASURED by running the suite and reading its own `test result:` line, never computed from a plan"

requirements-completed: []

coverage:
  - id: D1
    description: "The gate's entire row-supplied input surface enumerated by class, finding contract_id as the third unvalidated field"
    requirement: "EVAL-04"
    verification:
      - kind: other
        ref: "test -s 05-15-gate-input-surface.md && grep -c '(iii)' -> 65 lines"
        status: pass
      - kind: other
        ref: "debt-marker scan (TODO/TBD/FIXME/XXX) -> 0"
        status: pass
      - kind: other
        ref: "python3 field-coverage check: every pub field of BenchRowPayload and its nested blocks appears in the table"
        status: pass
    human_judgment: true
    rationale: "Existence, debt-freedom and field coverage are checked automatically, but whether each field's CLASS and each (iii) entry's stated reason are correct is a reading of the source that only a human can confirm."
  - id: D2
    description: "One validation door (resolve_committed_evidence_path) both provenance arms resolve through, with two typed refusals"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_evidence_path_case_table_over_both_evidence_kinds"
        status: pass
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_the_variant_tag_table_gained_exactly_the_two_arms_this_round_authorised"
        status: pass
    human_judgment: false
  - id: D3
    description: "All three bounds gap 1 names (absolute, `..`, symlink) refused at the ACTIVE 40-cell scope through verify_run, each RED-before / GREEN-after"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_refuses_every_escaping_lock_path_shape_at_the_active_scope"
        status: pass
    human_judgment: false
  - id: D4
    description: "WR-06: a missing LOCK or LEDGER names its own kind, and the ROW remedy is unchanged"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_refuses_a_missing_lock_record_as_its_own_kind_not_as_a_missing_row"
        status: pass
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_evidence_reads_are_bounded_from_the_declared_length"
        status: pass
    human_judgment: false
  - id: D5
    description: "The third enumerated field closed: a row or manifest declaring a foreign contract_id is refused, plus three contract-pinned constants"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_refuses_a_row_declaring_a_foreign_contract_id"
        status: pass
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_refuses_a_manifest_declaring_a_foreign_contract_id"
        status: pass
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_refuses_each_contract_pinned_constant_a_row_may_not_choose"
        status: pass
    human_judgment: false
  - id: D6
    description: "Spot-check E replayed through the SHIPPED apr door: positive control rc=0 first, then the doctored tree rc=5 naming the escaping path"
    requirement: "EVAL-04"
    verification:
      - kind: e2e
        ref: "bash scripts/setfit_bench_gate_door_probe.sh -> rc=0, PASS (run twice, identical verdict)"
        status: pass
      - kind: e2e
        ref: "apr setfit bench report --bench-dir benchmarks/tweeteval-stance (UNDOCTORED) -> rc=0"
        status: pass
    human_judgment: false
  - id: D7
    description: "Thirteen path shapes crossed with both evidence kinds, including three acceptance rows and two behaviour-preserving rows"
    requirement: "EVAL-04"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_evidence_path_case_table_over_both_evidence_kinds (26 assertions)"
        status: pass
    human_judgment: false
  - id: D8
    description: "Deterministic refusal order: the FIRST offending cell in contract order, across repeated invocations"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_reports_the_first_offending_cell_in_contract_order_across_runs"
        status: pass
    human_judgment: false
  - id: D9
    description: "The setfit-bench-gate floor reads the suite's measured count (45), and a separate setfit-bench-door-probe target exists that is a prerequisite of nothing"
    verification:
      - kind: integration
        ref: "make setfit-bench-tests -> rc=0, bench_gate leg reports 45 passed against a floor of 45"
        status: pass
      - kind: integration
        ref: "make setfit-bench-door-probe -> rc=0"
        status: pass
      - kind: other
        ref: "grep -n '^\\.SHELLFLAGS' Makefile -> exactly 29 and 57, byte-unchanged"
        status: pass
    human_judgment: false
  - id: D10
    description: "The stale eighty-cell test-module header replaced with a fifteen-entry negative inventory naming the SCOPE each is mutated at"
    verification:
      - kind: other
        ref: "grep -c '80-row' -> 0; grep -c 'All six run in a default' -> 0"
        status: pass
    human_judgment: true
    rationale: "The greps prove the two false claims are gone; whether the replacement inventory accurately describes what the file now contains is a reading a human must confirm."

duration: 113min
completed: 2026-09-11
status: complete
---

# Phase 5 Plan 15: Close the Evidence-Path Escape at the Level of its Class Summary

**One validation door (`resolve_committed_evidence_path`) that both provenance arms resolve through, closing the absolute / `..` / symlink escape that let `apr setfit bench report` exit 0 while printing an attestation that was false — plus an enumeration of the gate's whole row-supplied input surface that found a THIRD unvalidated field (`contract_id`) and closed it in the same round.**

## Performance

- **Duration:** 113 min (includes one watchdog-killed run resumed from committed state)
- **Started:** 2026-09-11T20:26:48Z
- **Completed:** 2026-09-11T22:19:57Z
- **Tasks:** 3
- **Files created/modified:** 6

## Accomplishments

- **Verifier gap 1 (EVAL-04, graded FAILED) is closed at the level of its class.** `verify_provenance` built `bench_dir.join(row.evidence.setfit.lock.lock_record_path)` from a producer-written string with no validation. `Path::join` DISCARDS its base when the argument is absolute and never resolves `..`. Both provenance arms now resolve through one helper that refuses syntactically before any filesystem call and then by canonical containment.
- **The class, not just the probe.** Task 1 enumerated all 85 row- and manifest-supplied fields the gate consumes, classed each as recomputed / contract-compared / trusted-as-written, and found `payload.contract_id` and `RunManifestPayload.contract_id` — read, carried, never compared, while `aggregate` stamped the published payload with `setfit-benchmark-claims-v1` regardless. Closed in the same round, with three more contract-pinned constants.
- **All three bounds gap 1 names were observed RED at the ACTIVE 40-cell scope, then GREEN.** Not reasoned about — run.
- **The shipped door is proven, not just the library.** `scripts/setfit_bench_gate_door_probe.sh` replays spot-check E through `apr setfit bench report` with a positive control that must pass first.

## Task Commits

1. **Task 1: Enumerate the gate's entire row-supplied input surface** — `3e0efbcd3` (docs)
2. **Task 2: One validation door, wired end-to-end (tracer)** — `e3b878340` (feat)
3. **Formatting correction to Task 2's code** — `f0fcc5e7c` (style)
4. **Task 3: The parameterized case table + Make floors** — `b2cbf3279` (test)

**Commits:** 4, MEASURED as `git rev-list --count bfa6a25a3..HEAD`, not narrated.

## THE RED / GREEN EVIDENCE (plan `<output>` requirement)

### The ACTIVE 40-cell scope sweep — one line PER SHAPE

**How the RED was taken.** The sweep test was written FIRST, into `bench_gate_tests.rs`, while `bench_gate.rs` was still at commit `3e0efbcd3` — i.e. the production file was untouched, the pre-fix code. It uses only symbols that existed pre-fix (it compares `error.variant_tag()` against the string literal `"evidence_path_escape"`, which pre-fix simply never matches), so it compiles and runs against the unfixed gate. The sweep was deliberately built to record and print a result PER ROW and assert only at the end, so one run reports all three bounds rather than aborting on the first.

Command: `cargo test -p aprender-train --lib --features setfit bench_gate_refuses_every_escaping -- --nocapture`, **rc=101**.

| shape | declared string (runtime-built) | PRE-FIX (`3e0efbcd3`) | POST-FIX (`e3b878340`) |
|---|---|---|---|
| `absolute` | `/var/folders/.../T/.tmpU63b5y/anywhere.json` | **`Ok(40 rows verified)`** | `evidence_path_escape` |
| `parent_traversal` | `../.tmph419Mi/anywhere.json` | **`Ok(40 rows verified)`** | `evidence_path_escape` |
| `last_component_symlink` | `locks/escape-via-symlink.lock.json` | **`Ok(40 rows verified)`** | `evidence_path_escape` |

Verbatim RED output (`[bench_gate] ACTIVE_SCOPE_ESCAPE` lines, `--nocapture`):

```
shape=absolute declared=/var/folders/3s/xftgktnj6qs681vbh0tg5hmc0000gn/T/.tmpU63b5y/anywhere.json observed=Ok(40 rows verified)
shape=parent_traversal declared=../.tmph419Mi/anywhere.json observed=Ok(40 rows verified)
shape=last_component_symlink declared=locks/escape-via-symlink.lock.json observed=Ok(40 rows verified)
```

and the assertion that failed, which carries the same three observations independently of the printout:

```
every bound gap 1 names must be refused as `evidence_path_escape` through verify_run at the
ACTIVE 40-cell scope; these were not: [("absolute", "Ok(40 rows verified)"),
("parent_traversal", "Ok(40 rows verified)"), ("last_component_symlink", "Ok(40 rows verified)")]
```

**`Ok` on all three is the RIGHT red.** The plan required exactly this and not some other refusal: each iteration writes that cell's own `synthetic_lock_bytes` to a file in a SECOND temp dir outside the bench dir and DELETES the in-tree `locks/…lock.json`, so the escape target holds the very bytes the row attests. The pre-fix gate therefore hashed them, MATCHED, and returned `Ok` — the same false green the verifier observed through the shipped door. A red that was `provenance_mismatch` would have proved the escape was DETECTED rather than that it SUCCEEDED.

Verbatim GREEN output (post-fix, test binary run directly to bypass output filtering):

```
shape=absolute declared=/var/folders/.../T/.tmpgcdktz/anywhere.json observed=evidence_path_escape
shape=parent_traversal declared=../.tmpLKB2Cs/anywhere.json observed=evidence_path_escape
shape=last_component_symlink declared=locks/escape-via-symlink.lock.json observed=evidence_path_escape
```

### The other new negatives

These were authored after the fix, so their RED is a design-time property rather than an observed pre-fix run. Stated as such rather than dressed up:

| negative | RED basis | GREEN observation |
|---|---|---|
| missing lock record names its own kind | Pre-fix, `read_evidence` mapped EVERY `NotFound` to `RowFileMissing` unconditionally (`bench_gate.rs:540`, one arm, no kind parameter) — so the asserted tag `evidence_file_missing` did not exist and the asserted-absent literal `restore the row file` was necessarily present. Not separately run pre-fix. | `evidence_file_missing`, message names `lock record` and the file, and does NOT contain `restore the row file`. The ROW arm still returns `row_file_missing` WITH that literal. |
| row declares a foreign `contract_id` | Pre-fix, zero of the eight `contract_id` / `CLAIMS_CONTRACT_ID` sites compared a declared id against the constant (enumerated site-by-site in `05-15-gate-input-surface.md`). Not separately run pre-fix. | `row_schema_refused`, naming both `setfit-benchmark-claims-v99` and `setfit-benchmark-claims-v1`. |
| manifest declares a foreign `contract_id` | Same. | `row_schema_refused`, naming both ids. |
| three contract-pinned constants (`calibration_split`, `warmup_count`, `cold_measured_in_child_process`) | Pre-fix, each was set at EMISSION time only (`bench_metrics.rs:262`, `setfit_bench.rs:1488/1492`) and read by nothing on the gate path. Not separately run pre-fix. | `row_schema_refused` for each, each naming its own field. |
| 13-row path-shape table x 2 evidence kinds | The eight escape rows had no refusal to return pre-fix, since the variant did not exist. Not separately run pre-fix. | 26/26 assertions pass: 8 escapes, 3 acceptances, `evidence_file_missing` for an absent file, `evidence_read_failed` / `not a regular file` for a directory. |
| deterministic refusal order over two doctored cells | Same — depends on the new variant. | Names `setfit/s8/seed17` on all three invocations, with the cells doctored in REVERSE contract order. |

### The measured test-count floor, and where the number came from

The Makefile floor was raised **38 → 45**. The 45 was MEASURED, not computed from the plan: `cargo test -p aprender-train --lib --features setfit bench_gate` was run and its own log parsed with the `assert_tests_ran` macro's awk:

```
test result: ok. 45 passed; 0 failed; 0 ignored; 0 measured; 8044 filtered out; finished in 1.32s
MEASURED PASSED = 45
```

`make setfit-bench-tests` then reports `45 passed` against a floor of `45` — exact, and 45 > 38.

### The door probe's PASS output, including its positive control

```
CONTROL: undoctored slim copy of <repo>/benchmarks/tweeteval-stance verifies (rc=0)
DOCTORED: setfit-s8-seed13 now points at <tmp>/outside/anywhere.json, and the committed lock record is gone
ATTACK: rc=5, refused as a path escape naming <tmp>/outside/anywhere.json
PASS: <repo>/target/release/apr refuses a row-supplied evidence path that leaves the benchmark
      directory, having first verified the undoctored tree
```

Binary pin (CLAUDE.md rule 3): `apr 0.63.0 (b2cbf3279)` == `git rev-parse --short HEAD`. Run twice; the two runs differ only in the `mktemp` directory name and the verdict is identical, with zero scratch directories left behind and `benchmarks/` untouched — which discharges the `verification: backstop` idempotency truth.

## Files Created/Modified

- `.planning/phases/05-benchmark-and-claims-gate/05-15-gate-input-surface.md` — the class-(a) enumeration: 85 field rows across the row envelope, payload, quality, resource, both evidence arms and the manifest; a "Closed in this round" table; and a "Still trusted, with reason" section grouped A–I, in which every remaining (iii) entry carries the reason it is one.
- `crates/aprender-train/src/train/setfit/bench_gate.rs` — `EvidenceKind`, `resolve_committed_evidence_path`, `missing_evidence`, `verify_contracted_row_constants`, `verify_manifest_contract`; two new `BenchGateError` variants; `read_evidence` gained a kind; the module header and `verify_run`'s doc comment now state the enforcement instead of asserting the convention.
- `crates/aprender-train/src/train/setfit/bench_gate_tests.rs` — the ACTIVE-scope escape sweep, the 13-row path-shape table over both kinds, the WR-06 typing negative, two contract-id negatives, the contract-pinned-constants table, the deterministic-order negative, and a rewritten module header carrying a fifteen-entry inventory with a `scope` column.
- `scripts/setfit_bench_gate_door_probe.sh` — the door-level probe.
- `scripts/setfit_bench_gate_doctor.py` — the fixture doctoring the probe drives (see Deviation 1).
- `Makefile` — floor 38 → 45, banner prose rewritten, new `setfit-bench-door-probe` target.

## Decisions Made

- **WR-06 as option (a), as the plan directed.** `read_evidence` takes an `EvidenceKind`; the ROW kind keeps `RowFileMissing` and its message verbatim so spot-check A's output is unchanged.
- **`EvidenceKind` is deliberately not `#[non_exhaustive]`,** so 05-16's `SelectionManifest` variant forces every match arm to be revisited rather than falling into a `_` arm that gives the new kind somebody else's diagnosis.
- **The resolver returns the JOINED path, not the canonical one.** Returning the canonical form would have rewritten every downstream refusal's file path — on macOS `TempDir` lives under `/var/folders/...`, whose canonical form is `/private/var/...`, which would have broken the pre-existing `bench_gate_refuses_a_setfit_row_whose_committed_lock_file_was_edited` assertion and, more importantly, would show operators a path they did not type.
- **Canonicalizing BOTH sides is not optional,** and `Path::starts_with` (component-wise) is used rather than `str::starts_with`. The prefix-sibling table row (`bench_dir` = `<tmp>/bench`, target `<tmp>/bench-evil/x.json`) exists to go red on that exact mistake.
- **`throughput_batch_size` was NOT closed.** The contract pins no value ("the batch size that pass used"); the writer's `THROUGHPUT_BATCH_SIZE = 32` is a producer choice in `apr-cli`. Comparing against it would refuse a legitimately different batch size while calling it a contract violation. Recorded under "Still trusted, with reason" section D rather than silently skipped.
- **`requirements-completed` is deliberately empty.** All fourteen prior phase-5 plans left it empty on the grounds that flipping requirement state is the verifier's act, and `05-VERIFICATION.md` explicitly calls that "correct process". `requirements.mark-complete` was therefore not run for EVAL-04. This is consistency with the phase's established convention, not an omission.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] The probe's Python was extracted to `scripts/setfit_bench_gate_doctor.py`**
- **Found during:** Task 2 (the door probe)
- **Issue:** The plan specified "an inline `python3` block". `bashrs lint` — which CLAUDE.md mandates over shellcheck — does not skip a quoted heredoc body: it parsed the embedded Python as shell and reported **12 errors** (`SC1007`, `SC1035`, `SC1078`) against Python syntax. The plan's own acceptance criterion requires `bashrs lint` to be clean, so the inline form makes the criterion unsatisfiable.
- **Fix:** Moved the doctoring to a sibling file the probe invokes. The file is **load-bearing** — the probe cannot doctor the tree or repair digests without it — so it is not removable.
- **Scope note:** The file is NOT in the plan's `files_modified` frontmatter, and the plan's success criteria say "No out-of-scope item is silently touched". Recording it here is the not-silently half. Nothing else was touched: the diff across all four commits is exactly the six files listed above.
- **Verification:** `bashrs lint scripts/setfit_bench_gate_door_probe.sh` → **0 errors** (was 13).
- **Committed in:** `e3b878340`

**2. [Rule 3 — Blocking] The probe gained an explicit `setfit`-feature pin**
- **Found during:** Task 2
- **Issue:** `setfit` is not a default feature. A plain `cargo build --release --bin apr` produces a binary whose `apr setfit` is `unrecognized subcommand`, and the probe's positive control then failed with rc=2 — which reads as "the committed evidence is broken" when the evidence is fine and the binary is not.
- **Fix:** Added a feature precheck between the binary pin and the positive control, with its own remedy naming `cargo build --release --bin apr --features setfit`. The Make target's help text names the same command.
- **Verification:** Probe rc=0 with the feature-enabled binary; the precheck fires with the correct remedy without one.
- **Committed in:** `e3b878340`

**3. [Rule 1 — Bug] `bench_gate_the_variant_tag_table_gained_no_arm_in_this_plan` asserted 13 arms**
- **Found during:** Task 2
- **Issue:** 05-11 pinned the `variant_tag` arm count at 13. This plan's `<artifacts>` section mandates two new variants, so the guard necessarily went red.
- **Fix:** Renamed to `..._gained_exactly_the_two_arms_this_round_authorised` and raised to 15, with a per-variant argument for why each was minted (a producer-reachable path with no refusal at all; a mis-typed missing-file diagnosis) and an explicit note that the four contract-derived comparisons added in the same round mint NOTHING — they reuse `row_schema_refused`. The guard now also asserts that the two arms present are the two that were argued for.
- **Verification:** `bench_gate_the_variant_tag_table_gained_exactly_the_two_arms_this_round_authorised` passes.
- **Committed in:** `e3b878340`

**4. [Rule 2 — Missing Critical] Three more contract-pinned constants closed beyond `contract_id`**
- **Found during:** Task 1
- **Issue:** Task 1 instructed: "If closing them is a one-line comparison per field, close them in Task 2." Three of the four (`calibration_split`, `warmup_count`, `cold_measured_in_child_process`) have contract-derived constants already in the library and were checked only at emission time.
- **Fix:** `verify_contracted_row_constants`, reusing `RowSchemaRefused`.
- **The instrument was NOT tuned to the measurement.** Before adding the checks, all 40 committed rows were measured: `calibration_split` = `validation` ×40, `warmup_count` = 3 ×40, `cold_measured_in_child_process` = true ×40, `contract_id` = `setfit-benchmark-claims-v1` ×40 and on the manifest. No committed row was edited and no floor was lowered.
- **Verification:** `apr setfit bench report` on the UNDOCTORED committed tree still rc=0 — the regression half.
- **Committed in:** `e3b878340`

**5. [Rule 1 — Bug] `cargo fmt` reflowed two blocks the Task 2 commit left unformatted**
- **Fix:** `cargo fmt -p aprender-train`, in its own `style` commit so Task 3's commit stays about Task 3.
- **Verification:** `cargo fmt -p aprender-train -- --check` → rc=0; gate suite still 45 passed; clippy still clean.
- **Committed in:** `f0fcc5e7c`

---

**Total deviations:** 5 auto-fixed (2 blocking, 2 bugs, 1 missing-critical)
**Impact on plan:** All five were required to satisfy the plan's own acceptance criteria. No scope creep: the diff is exactly the five files the plan named plus the one extracted helper documented in Deviation 1.

## Issues Encountered

### 1. The plan's clippy verification line CANNOT PASS on this tree — stated plainly rather than claimed clean

The plan's `<verify>` says `cargo clippy -p aprender-train --lib --features setfit -- -D warnings` must be clean. **It exits 101.** This is reported as a finding, not smoothed over.

Attribution, measured rather than assumed:

- **Zero** of the findings are in `crates/aprender-train`. Every erroring file is in `crates/aprender-compute` and `crates/aprender-present-terminal` — unused imports, unreachable expressions, dead code — surfaced because `-D warnings` propagates to dependency crates compiled in the same session. Last touched by the APR-MONO Phase 2 subtree merge, long before this plan.
- `cargo clippy -p aprender-compute --lib --no-deps -- -D warnings` is **independently red at HEAD**, which proves the debt is the crate's own and not an interaction with this change.
- The in-scope signal, taken with `--no-deps` so only the selected package is linted: `cargo clippy -p aprender-train --lib --features setfit --no-deps -- -D warnings` → **rc=0, clean.**
- `cargo fmt --all -- --check` → rc=1, flagging only `crates/aprender-image/*` and `crates/aprender-mcp-chronos/*`. `cargo fmt -p aprender-train -- --check` → rc=0.

This is the CLAUDE.md **#2370 class** of defect the "Linting" section documents: findings from newer clippy releases accumulate invisibly because no gate the repo owns runs a whole-workspace `-D warnings`. **Not fixed here** — it is squarely out of the scope boundary (pre-existing lint failures in unrelated files), and CLAUDE.md assigns it to the `toolchain-ceiling.yml` gate. Recorded so a reader does not mistake `--no-deps` for the plan's literal command.

### 2. `bashrs lint` "no findings" is an unreachable bar in this repo — the achievable bar is 0 errors

The plan's criterion is "reports no findings" / "any reported finding" fails. Calibrated against the tree:

| script | errors | warnings | infos | rc |
|---|---|---|---|---|
| `scripts/run_bench_cells.sh` (phase 5's own) | 0 | 1 | 41 | 1 |
| `scripts/apr_bin.sh` | 0 | 20 | 38 | 1 |
| `scripts/check_msrv.sh` | 0 | 4 | 4 | 1 |
| **`scripts/setfit_bench_gate_door_probe.sh`** | **0** | 8 | 22 | 1 |

**No script in the repository is bashrs-clean,** and `bashrs` exits non-zero on any finding at any severity including `info`. The probe was held to the repo's de-facto bar — **0 errors** — which it meets. The 8 remaining warnings are all false positives, individually checked: 5 × `SC2154 'APR' is referenced but not assigned` (it is exported by the sourced `scripts/apr_bin.sh`, which bashrs does not follow), 2 × `SC2047` on variables that ARE quoted, and 1 × `SEC014` on a `cp` whose operands are both derived locally and carry a `# bashrs:allow SEC014` directive bashrs does not honour for that rule.

`bashrs make lint Makefile` → 1 error / 43 warnings, **byte-identical to the same lint run against `git show HEAD:Makefile`** before this plan's edit. The one error is at `dev-setup` (line 3280, `local` outside a function), pre-existing.

### 3. FINDING for a later round: `bench_row.rs`'s canonical-bytes rationale is stale

Reverse-engineering the digest scheme for the probe measured something the module doc contradicts. `bench_row.rs:37-43` states that `to_canonical_bytes` serializes through `serde_json::Value`, "whose `Map` is `BTreeMap`-backed (no workspace crate enables `preserve_order`)", concluding the digest is key-SORTED and "independent of Rust field-declaration order".

Measured against the committed tree:

- `sha256(compact JSON, **sort_keys=True**)` reproduces **neither** the row digest nor the manifest digest.
- `sha256(compact JSON, **file/serde order**)` reproduces **both exactly** — row `fbce5eef…` and manifest `5d38f2fc…`.

So `preserve_order` **is** enabled somewhere in this build's feature unification, and the digest is over declaration-ordered bytes. This is **not a correctness defect** — the scheme is deterministic and independently reproducible either way, which is what EVAL-04 needs, and the verifier reverse-engineered the same thing — but the doc's stated REASON is false, and the property it claims (adding a field in a different position cannot change historical digests) does **not** hold. `bench_row.rs` is outside this plan's `files_modified`, so it was not edited. Recommended for 05-16 or 05-17, both of which touch that module.

### 4. Transient, pre-existing: one nextest LEAK

One run reported `45 passed (1 leaky)`, on `bench_gate_accepts_a_complete_valid_run` — a test this plan did not touch. It did not reproduce on the next run. Not investigated; out of scope.

## Known Stubs

None. No file created or modified by this plan contains a hardcoded empty value flowing to output, a placeholder string, or an unwired data source.

## Threat Flags

None. Every file this plan touched is covered by the plan's own `<threat_model>`, and the mitigations it assigns (T-05-15-01, -02, -03, -04, -05) are implemented and proven above. No new network endpoint, auth path, file-access pattern or schema change at a trust boundary was introduced — the change strictly NARROWS the set of files the gate will open.

## Next Phase Readiness

- **Ready for 05-16 (EVAL-02, gap 2).** `resolve_committed_evidence_path` and `EvidenceKind` are the surface 05-16 extends: it adds a `SelectionManifest` variant, and because the enum is not `#[non_exhaustive]` the compiler will force every match arm to be revisited. The enumeration artifact already records `payload.selection_manifest_hash` as class (iii) with 05-16 named as its closer.
- **Ready for 05-17 (EVAL-01).** The artifact records exactly which quality fields 05-17 can move to class (i) by recomputing from `confusion_matrix`, and — more usefully — which it CANNOT: `ece_top_label_validation` and `brier_multiclass_validation` need per-row validation probabilities no committed file carries, and `confusion_matrix` itself is the unrecomputable base the whole recomputation rests on. 05-17's claim should be a REDUCTION of forgeable surface, not an elimination.
- **Both 05-16 and 05-17 must extend the fifteen-entry negative inventory** in `bench_gate_tests.rs`'s header, and must raise the `setfit-bench-gate` floor to their own MEASURED count.
- **Open, not closed by this round:** `payload.quality.ordered_labels` (a contract-derived comparison is reachable but needs a second contract in this module's compile-time surface) and `payload.resource.peak_rss_sample_interval_hz` (needs a two-field conditional rule, not a constant comparison). Both are recorded in "Still trusted, with reason".
- **Carried forward for a human:** the workspace clippy/fmt debt in Issue 1 and the stale digest rationale in Issue 3.

---
*Phase: 05-benchmark-and-claims-gate*
*Completed: 2026-09-11*
