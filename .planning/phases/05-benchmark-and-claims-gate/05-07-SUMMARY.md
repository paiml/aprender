---
phase: 05-benchmark-and-claims-gate
plan: 07
subsystem: cli
tags: [setfit, f-10-unblock, spawned-ladder, ops-01, ops-02, trn-07, blast-radius, eval-02]

requires:
  - phase: 05-benchmark-and-claims-gate
    provides: "05-03's production calibration regime at a63bb130b — without it rung 1 below is exit 6"
provides:
  - "A user-produced setfit-apr-v1 DEMONSTRATED end-to-end at the spawned-binary tier: apr setfit train -> inspect -> eval(validation --lock-out) -> eval(test --selection-lock) -> predict, all exit 0, one artifact SHA-256 across all five surfaces"
  - "setfit_cli_production_chain_completes_after_the_calibration_edit — the reusable, env-gated production ladder Phase 5's benchmark cells can cite"
  - "An honest F-10 blast radius: zero present-tense 'no user-reachable path produces a setfit-apr-v1' claims remain in crates/"
  - "make setfit-lifecycle-tests restored to GREEN — it was RED at HEAD, left by 05-03's calibrated.len() == 1 literal"
affects: [05-09, 05-11, 05-12, 05-13]

tech-stack:
  added: []
  patterns:
    - "a positive spawned rung is env-gated on the offline prerequisite, and a REQUIRED=1 toggle turns the skip into a failure so a skip cannot masquerade as a pass"
    - "cross-surface artifact identity is asserted as ONE 64-hex digest read out of four independent processes' JSON, with the hex shape checked first so two matching nothings cannot satisfy it"
    - "a rename is warranted when the NAME states a falsehood; the assertions do not move, and the rename is recorded with its Make-filter re-run"

key-files:
  created: []
  modified:
    - crates/apr-cli/tests/setfit_cli_lifecycle.rs
    - crates/aprender-train/tests/setfit_apr_lifecycle.rs
    - crates/aprender-train/src/train/setfit/verify.rs
    - crates/aprender-train/src/train/setfit/apr_evaluate_tests.rs
    - crates/apr-cli/tests/setfit_parity.rs
    - crates/apr-cli/src/commands/setfit_train.rs
    - crates/apr-cli/src/commands/predict_tests.rs
    - crates/apr-cli/src/commands/eval/setfit_tests.rs

key-decisions:
  - "The production ladder is DELIBERATELY not selected by `make setfit-cli-lifecycle`'s `-- --ignored lifecycle` filter. Its name carries no `lifecycle` substring, so that gate's 2-test floor and its wall clock are unchanged. Wiring a real 22M-parameter training run into a routine gate would add ~30 min of debug-profile tuning to tier3 (05-01: 1758.6 s debug vs 51.0 s release for the same s8 tuning). The trade-off is stated at the test, not hidden."
  - "04-15's rung 4 refusal was NEVER F-10's rung, and this plan says so on the record. The message names `config.json`, so it fires inside `from_pretrained_dir`: the conformance slice is a FIXTURE directory, not a pretrained checkout. That is why the calibration edit did not turn it green, and why leaving it asserting exit 6 is correct rather than stale."
  - "The `calibrated.len() == 1` assertion 05-03 left behind was MIGRATED, not deleted: it now asserts 2 and names BOTH entries, so the state change is recorded by an assertion rather than by an assertion's disappearance."
  - "Two tests were renamed and two were not. The bar applied: rename only when the NAME states a falsehood. `setfit_train_e2e_records_the_blocker_that_stops_short_of_an_artifact` still describes a real blocker (the slice `--model-dir`), so only its prose moved."

patterns-established:
  - "Env-gated positive rung + REQUIRED=1 failure toggle, with both control runs recorded: skip-without-REQUIRED exits 0 and prints SKIPPED; skip-with-REQUIRED exits 101"
  - "`apr --version` in the ladder's r0 prints the git SHA, so the transcript itself proves WHICH commit the binary under test was built from"

requirements-completed: [EVAL-02]

coverage:
  - id: D1
    description: "One user-produced setfit-apr-v1 exists: the spawned train -> inspect -> eval(validation --lock-out) -> eval(test --selection-lock) -> predict chain completes with exit 0 at every rung on the production encoder"
    requirement: "EVAL-02"
    verification:
      - kind: integration
        ref: "crates/apr-cli/tests/setfit_cli_lifecycle.rs#setfit_cli_production_chain_completes_after_the_calibration_edit"
        status: pass
    human_judgment: false
  - id: D2
    description: "The same artifact SHA-256 appears in inspect, eval and predict outputs — one artifact through every surface"
    requirement: "EVAL-02"
    verification:
      - kind: integration
        ref: "same test: hex64() refuses an absent/null/truncated digest, then asserts all four surfaces equal the training run's"
        status: pass
    human_judgment: false
  - id: D3
    description: "Every stale present-tense 'no user-reachable path produces a setfit-apr-v1' claim in the F-10 blast radius is re-worded; the slice-encoder REFUSAL tests stay green"
    verification:
      - kind: other
        ref: "grep -rn --include=*.rs 'no user-reachable' crates/ = 7 lines, ALL historically scoped (each names a63bb130b / 'Before Phase 5's 05-03 calibration edit')"
        status: pass
      - kind: unit
        ref: "8 affected Make gates re-run green; setfit:: 347 passed, apr-cli --lib 6781 passed"
        status: pass
    human_judgment: false
  - id: D4
    description: "The env-gate cannot let a skip masquerade as a pass"
    verification:
      - kind: integration
        ref: "APRENDER_MINILM_DIR=/tmp/nonexistent-checkout -> rc=0 + 'SKIPPED'; same with APRENDER_PRODUCTION_CHAIN_REQUIRED=1 -> rc=101"
        status: pass
    human_judgment: false

duration: ~2h30m
completed: 2026-08-17

actuals:
  tokens: 8800
  tasks: 2
  commits: 2

status: complete
---

# Phase 5 Plan 07: F-10 Unblock Proven End-to-End Summary

**`apr setfit train` on the production encoder now exits 0 and writes a 90,777,156-byte
`setfit-apr-v1`, and four further processes — `inspect`, `eval --split validation --lock-out`,
`eval --split test --selection-lock`, `predict` — each report the SAME SHA-256 for it. 04-15
measured that exact rung at exit 6; the flip is now a measurement, not an expectation.**

## Performance

- **Duration:** ~2h 30m
- **Tasks:** 2 of 2
- **Files modified:** 8 (0 created)

## Task Commits

1. **Task 1 — the positive production ladder** — `db12111a9`
   (`test(05-07): prove the F-10 flip end-to-end at the spawned-binary tier`)
2. **Task 2 — the blast-radius prose re-audit** — `7ae052f64`
   (`docs(05-07): re-audit the F-10 blast radius to the post-unblock truth`)

## The flip, measured

Run at the committed state, through `rtk proxy` so the log holds libtest's raw lines rather
than the hook's summary, status captured directly (`cmd > log 2>&1; rc=$?`), never through a
pipe:

```
APRENDER_PRODUCTION_CHAIN_REQUIRED=1 CARGO_INCREMENTAL=0 cargo test --release -p apr-cli \
  --test setfit_cli_lifecycle --features setfit -- --ignored --nocapture production_chain
```

```
[05-07] RAN — production checkout /Users/guy/.cache/aprender/minilm-l6-v2-1110a243 (config.json present)
[05-07] r0 --version: apr 0.63.0 (7ae052f64)
[05-07] r1 setfit train: exit 0 | model.apr = 90777156 bytes | artifact_sha256 = 30a98fd6...eb1a4a2
[05-07] r2 inspect: artifact_sha256 = 30a98fd6...eb1a4a2 | labels = ["none", "against", "favor"]
[05-07] r3 eval(validation): artifact_sha256 = 30a98fd6...eb1a4a2 | lock_hash = 6353ed03...31a61d00 | accuracy = 0.3181818181818182
[05-07] r4 eval(test): artifact_sha256 = 30a98fd6...eb1a4a2 | lock_hash = 6353ed03...31a61d00 | accuracy = 0.5892857142857143 over 280 rows
[05-07] r5 predict: artifact_sha256 = 30a98fd6...eb1a4a2 | results = 2
[05-07] ONE ARTIFACT: 30a98fd6...eb1a4a2 reported identically by train, inspect, eval(validation), eval(test) and predict
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out; finished in 36.72s
```

Full digest: `30a98fd60f4ad68ea28afc68b629d01cf1fda90a488ddf188a2e63959eb1a4a2`.
Lock hash: `6353ed035a386cbdc2d5672be3ca1f56b82f7b61d1eb58489e17067731a61d00`.

**The rung that flipped, quoted beside its historical record:**

| rung | 04-15 measured | 05-07 measures | how |
|---|---|---|---|
| `setfit train` (real) | **exit 6** `ModelLoadFailed` | **exit 0**, `model.apr` 90,777,156 bytes | `--model-dir` = the production checkout instead of the conformance slice |
| `inspect model.apr --json` | nonzero — nothing to read | **exit 0**, SetFit section, `document_schema_valid: true` | the file r1 wrote |
| `eval --split validation --lock-out` | unreachable | **exit 0**, `lock.json` on disk, `role: written` | — |
| `eval --split test --selection-lock` | unreachable (04-07 labelled its own proof IN-PROCESS) | **exit 0**, `role: consumed`, SAME lock hash | a SEPARATE process reading that FILE |
| `predict --input request.json --json` | unreachable | **exit 0**, ordered labels + probabilities summing to 1 | the FULL eight-rung load ladder, probe replay included |

**`r0` is the mechanism proof, not decoration.** `apr --version` prints `0.63.0 (7ae052f64)`
— the commit this SUMMARY describes. CLAUDE.md rule 3 says pin the binary;
`env!("CARGO_BIN_EXE_apr")` does that, and the printed SHA is what lets a reader confirm it
without re-running.

## The non-empty criterion, and why an existence check would not have done

The cross-AI review's 05-07 row asked for a non-empty, valid artifact. Both halves are
asserted separately, because they fail differently:

- **Non-empty:** `fs::metadata(...).len() > 0`, and the byte count is PRINTED so the SUMMARY
  transcribes it from a run. A truncated write leaves a zero-byte file that `is_file()` accepts.
- **Valid:** r2's `inspect --json` on that same path is the load proof — it parses the header,
  the metadata and the SetFit document out of those bytes and hashes the whole file. A byte
  length says nothing about whether the bytes parse; a green r2 carrying a 64-hex digest does.

`hex64()` refuses an absent, `null`, non-string, short or non-hexadecimal digest BEFORE any
comparison, so the four cross-surface equalities cannot be satisfied by two matching nothings.

## The env gate cannot let a skip masquerade as a pass

Recorded as two controls rather than asserted:

| run | `APRENDER_MINILM_DIR` | `..._REQUIRED` | rc | output |
|---|---|---|---|---|
| the real one | (default cache path) | `1` | **0** | `RAN — production checkout ...` |
| CI shape | `/tmp/nonexistent-checkout` | unset | **0** | `SKIPPED — no production checkout at ...` |
| the falsifier | `/tmp/nonexistent-checkout` | `1` | **101** | panic at `setfit_cli_lifecycle.rs:896` |

Without the third row, "1 passed" would be the same string whether the ladder ran or not.

## 04-15's rung 4 was never F-10's rung — recorded, with the evidence

The slice chain still asserts exit 6, and the calibration edit did not touch it. That is not
an oversight: the refusal message names `config.json`, so it fires inside
`SetFitMiniLm::from_pretrained_dir` — the conformance slice is a FIXTURE directory
(`slice_config.json` + `vocab_remap.json`, no `config.json`, no `model.safetensors`) and never
reaches the calibration regime gate at all. The old header called this rung "the rung F-10
removes"; the two chains now differ by `--model-dir` and nothing else, and they end
differently for a reason that is visible in the message.

The restore instruction was rewritten accordingly: `F10_CLOSED` became
`SLICE_TRAIN_SUCCEEDED`, and it now says explicitly that it is **not** the F-10 signal and
points at the production ladder.

## Where the production ladder is, and is not, wired

**Deliberate:** `setfit_cli_production_chain_completes_after_the_calibration_edit` carries no
`lifecycle` substring, so `make setfit-cli-lifecycle`'s `-- --ignored lifecycle` filter does
not select it. Measured after the rename: that gate still reports `3 passed` against its floor
of 2, and the `tooling` leg `1 passed` against 1.

The reason is stated at the test rather than left implicit: this rung TRAINS a 22M-parameter
encoder, and the Make gates run debug. 05-01 measured the same s8 tuning at **1758.6 s debug
vs 51.0 s release** — a 34.5x spread — so wiring it into tier3 would add roughly half an hour
to a routine gate. The explicit command is in the `#[ignore]` reason, and this is flagged
below as the one thing a later plan may want to revisit.

## Blast-radius audit: what was found and what was changed

Enumerated files audited (05-RESEARCH "F-10 flip blast radius"), plus a repo-wide sweep.

| file | present-tense claim found | change |
|---|---|---|
| `apr-cli/tests/setfit_cli_lifecycle.rs` | header: "That chain cannot close on this host"; `F10_CLOSED` | header split into two chains; constant renamed + re-scoped |
| `aprender-train/tests/setfit_apr_lifecycle.rs` | header facts 1+2; 2 test names; 3 panic messages | re-worded; 2 renames; **1 assertion migrated (see Deviations)** |
| `aprender-train/src/train/setfit/verify.rs` | "no test in this repository can produce one" | scoped to "no test in THIS crate's default suite"; cites the measured 90,777,156 bytes |
| `aprender-train/src/train/setfit/apr_evaluate_tests.rs` | "the only crate in which a setfit-apr-v1 can be produced at all" | "…WITHOUT the 86.7 MB production checkout"; names 05-07 as the spawned proof |
| `apr-cli/tests/setfit_parity.rs` | "That chain cannot close on this host" | historically scoped; states why the fixture is still correct HERE |
| `apr-cli/src/commands/predict_tests.rs` | "`apr-cli` cannot construct a setfit-apr-v1 …" | scoped to the UNIT suite; points at the positive rung |
| `apr-cli/src/commands/eval/setfit_tests.rs` | same, x3 | same |
| `apr-cli/src/commands/setfit_train.rs` | "BLOCKER 1 … is still true and still the reason" | both blockers now struck through; the real cause named as the `--model-dir` |

**Final state of the audit, measured:**

```
grep -rn --include="*.rs" "no user-reachable" crates/     -> 7 lines
```

All seven are historically scoped — every one names `a63bb130b` and reads "Before Phase 5's
05-03 calibration edit". **Zero present-tense refusal claims remain.**

A `pmat query "F-10 user-reachable setfit-apr-v1" --limit 10` was run as the plan asked. It
returned no relevant hit — semantic function search does not index doc-comment prose, and its
top result was `ExtendedCommands`. The targeted grep is the instrument that found every site;
this is recorded so a later reader does not re-run the query expecting it to work.
(It also wrote a 407 MB `./.pmat/context.idx`; `.gitignore:59` covers `**/.pmat/`, verified
with `git check-ignore -v`, and `git status --short` shows no untracked files.)

## Zero assertions weakened — shown, not claimed

`git diff -U0` over the six pure-prose files, with comment lines filtered out, leaves exactly
**two** non-comment changed hunks in the whole set:

1. an `#[ignore = "…"]` reason string on `setfit_train`'s e2e test, and
2. one `assert!` MESSAGE string (the "reduced dimensions F-10 names" sentence).

No condition, no `expect`, no matcher moved. The seventh file's changes are itemised under
Deviations.

## Renames, and their floor re-runs

| old name | new name | why | Make filter | floor after |
|---|---|---|---|---|
| `setfit_cli_lifecycle_the_binary_walks_ops_02_to_the_rung_f_10_removes` | `…_against_the_conformance_slice` | the rung is not F-10's | `--ignored lifecycle` | **3 passed / floor 2** |
| `lifecycle_the_apr_save_rung_is_refused_by_the_only_calibrated_encoder` | `…_by_the_conformance_slice_encoder` | there are two calibrated regimes now | `--test setfit_apr_life` | **5 passed / floor 5** |
| `lifecycle_no_second_encoder_can_reach_the_save_rung` | `lifecycle_an_out_of_envelope_run_cannot_reach_the_save_rung` | a second architecture IS calibrated | `--test setfit_apr_life` | same run, **5 / 5** |

Both new lifecycle-file names keep the `lifecycle` prefix, so the CLI filter's selection is
unchanged. The only external references to the old names are two Phase 4 SUMMARYs, which are
historical records of a past state and were deliberately left alone.

## Verification at the COMMITTED state

Every status captured directly; `rtk proxy` used wherever a `test result:` line had to survive.

| Check | Command | Result |
|---|---|---|
| production ladder | `cargo test --release … -- --ignored --nocapture production_chain` | **rc=0, 1 passed**, ran (not skipped), transcript above |
| plan's Task 1 verify (debug) | `CARGO_INCREMENTAL=0 cargo test -p apr-cli --test setfit_cli_lifecycle --features setfit` | rc=0, 1 passed / 5 ignored |
| slice ladder + TRN-07 + WR-10 | `… -- --ignored lifecycle` | rc=0, **3 passed** |
| tooling ladder | `… -- --ignored tooling` | rc=0, 1 passed |
| plan's Task 2 verify (a) | `cargo test -p aprender-train --lib --features setfit setfit::` | rc=0, **347 passed; 0 failed** |
| plan's Task 2 verify (b) | `cargo test -p apr-cli --lib --features setfit` | rc=0, **6781 passed; 0 failed** |
| `make setfit-lifecycle-tests` | floor 5 | rc=0, **5 passed** (was RED — see Deviations) |
| `make setfit-cli-lifecycle` | floors 2 + 1 | rc=0, 3 passed + 1 passed |
| `make setfit-verify-tests` | floor 16 | rc=0, 18 passed |
| `make setfit-evaluate-tests` | floor 12 | rc=0, 14 passed |
| `make setfit-cli-train-tests` | floors 13 + 1 | rc=0, 15 passed + 1 passed |
| `make setfit-cli-predict-tests` | floor 30 | rc=0, 35 passed |
| `make setfit-cli-eval-tests` | floor 18 | rc=0, 20 passed |
| `make setfit-parity` | floor 18 | rc=0, 20 passed |
| clippy (ladder file) | `cargo clippy --release -p apr-cli --features setfit --test setfit_cli_lifecycle` | rc=0, **0 findings in the file** |
| rustfmt | `rustfmt --check` on all 8 files | clean |
| migrated assertion BITES | induced mutation `2 -> 3` | **RED**, printing both regime ids; reverted, green again |

## Deviations from Plan

### 1. [Rule 1 — Bug] `make setfit-lifecycle-tests` was RED at HEAD, left by 05-03

- **Found during:** Task 2, running the affected Make gates.
- **Issue:** `crates/aprender-train/tests/setfit_apr_lifecycle.rs:503` asserted
  `calibrated.len() == 1` with the note "if it has grown, an APR-capable encoder may now be
  trainable and this whole file must be revisited". 05-03 grew it to 2 and did not revisit
  this file — it is an INTEGRATION test, outside the `--lib setfit::` filter 05-03 ran. The
  gate has been failing since `a63bb130b`:
  `test result: FAILED. 4 passed; 1 failed`, `assertion left == right failed`.
- **Why this plan owns it:** the file is in this plan's `files_modified`, and the assertion's
  own message names the revisit this plan is.
- **Fix:** MIGRATED, not deleted, per 05-03's own deviation-3 pattern ("assert the state change
  positively"). It now asserts `== 2`, finds the fixture entry by architecture prefix, asserts
  the PRODUCTION entry is present by its own prefix (so a revert of the unblock turns it red),
  and generalises the non-vacuity check from "differs from THE entry" to
  `!calibrated.contains(&observed)`.
- **Proven to bite:** induced mutation `2 -> 3` → RED, printing both regime ids verbatim.
  Reverted; 5 passed.
- **This is the ONE place an assertion changed**, against Task 2's "zero assertion changes"
  criterion. It is a repair of a broken gate, not a weakening: the test asserts strictly more
  than before (two entries by name, plus a set-wide non-vacuity check).
- **Commit:** `7ae052f64`.

### 2. [Rule 2 — Missing critical] The skip path needed a falsifier

- **Issue:** the plan asks that the log show the test RAN, not skipped. A printed message is
  not a mechanism: `1 passed` reads identically either way, and the acceptance criterion would
  have rested on a human reading `--nocapture` output.
- **Fix:** `APRENDER_PRODUCTION_CHAIN_REQUIRED=1` turns the skip into an `assert!` failure, and
  both controls are recorded above (rc=0 skipping, rc=101 with the toggle).
- **Commit:** `db12111a9`.

### 3. [Scope, deliberate] Test 1's prose and name were edited in Task 1's commit

- The plan says "leave every existing blocked-rung/refusal test untouched". Its ASSERTIONS are
  untouched — exit 6, the three `expect_mentions`, the empty-listing checks all stand. Its
  NAME and its restore instruction claimed the refusal is F-10's, which the same commit proves
  false, so leaving them would have shipped a measurement carrying a false label. The file is
  Task 1's, so the edit is in Task 1's commit.

### 4. [Measurement discipline] Two runs were re-taken because the first was rewritten

- The `rtk` hook rewrites `cargo test` into its summarised form, which does not contain
  libtest's `test result:` line or the `[05-07]` transcript. The first production-chain run
  therefore reported only `cargo test: 1 passed … (36.32s)` — a true statement that could not
  have supported any claim about the artifact. Re-run through `rtk proxy`; the Makefile's own
  comment block at line 2338 records the same trap.
- The ladder was also re-run at the COMMITTED state after `rustfmt` reflowed one `assert_eq!`,
  rather than citing the pre-format run. `--version` in that run prints `7ae052f64`.

---

**Total deviations:** 1 bug fix (a pre-existing RED gate), 1 missing-critical addition,
1 deliberate in-scope edit, 1 measurement re-take. No scope creep; nothing weakened.

## Known Stubs

None. The new test has no placeholder rung: every one of its eight spawns is a real
`fork`/`exec` with a reaped `ExitStatus`, and no value it asserts is defaulted.

## Deferred / out of scope (NOT fixed)

- **The production ladder is in no Make gate.** Deliberate and documented at the test (see
  above), but it does mean this evidence runs only when invoked. A later plan that wants it
  gated should add its own target with its own floor and a `--release` recipe — NOT append it
  to `setfit-cli-lifecycle`, whose debug recipe would pay ~30 min per run.
- **`cargo clippy … -- -D warnings` is RED on this tree** for a pre-existing, unrelated
  warning: `unreachable expression` at `crates/aprender-present-terminal/src/compute_block.rs:93`
  (a `return Self::Neon` on aarch64 above a `Self::Scalar` fallthrough). Untouched by this
  plan; scoped clippy over the ladder file reports 0 findings.
- **The r3/r4 accuracies (0.318 validation, 0.589 test) are NOT a quality claim.** The corpus
  is the lifecycle fixture's synthetic rows ("authored fixture … sample N"), whose text carries
  no class signal. They are recorded because a run that prints them is a run that computed
  them; they must never be quoted as SetFit performance.
- **`.planning/WINDOWS.md` was NOT appended to.** The one recordable item (deviation 1) is
  already FIXED rather than open, and the ledger is a shared cross-phase file that three
  wave-3 worktrees could conflict on. Flagged here for the orchestrator to record centrally if
  it wants the entry.

## Threat Flags

None. No network endpoint, auth path, file-access pattern or trust-boundary schema was
introduced. The register is discharged:

- **T-05-07-01** (wrong binary) — one `Command::new(APR_BIN)` site in the file, asserted by the
  default-suite source guard; `env!("CARGO_BIN_EXE_apr")` occurs exactly once; r0's
  `--version` prints `0.63.0 (7ae052f64)` before any later exit code is interpreted.
- **T-05-07-02** (vacuous gate via rename) — three renames, each recorded with its Make target
  re-run and its floor: 3/2, 5/5, 5/5.
- **T-05-07-03** (artifact identity) — one 64-hex digest asserted equal across four surfaces,
  shape-checked first so the equality cannot be satisfied by empty strings.
- **T-05-07-04** (refusal weakening) — `git diff -U0` minus comments leaves two message-string
  hunks across six files; the slice `probe_unicode` refusal suites and all CLI refusal ladders
  re-run green.
- **T-05-07-SC** (package installs) — none performed.

## Next Phase Readiness

**D-01 is discharged: a user-produced `setfit-apr-v1` exists and is demonstrated.** 05-09,
05-11, 05-12 and 05-13 can depend on the chain rather than on 05-03's expectation.

Three things the next plans should carry rather than rediscover:

1. **The production s8/seed-13 cell costs ~37 s wall in release**, whole chain included
   (train + inspect + two evals over 66 and 280 rows + predict). That is the measured unit for
   projecting benchmark-cell wall clock at the CLI tier — and it is a RELEASE figure; debug is
   ~34.5x on the tuning half alone.
2. **`prepare_phase2_inputs(root)`** in `setfit_cli_lifecycle.rs` is the reusable rungs-1-2
   helper (attested benchmark directory + selection manifest, both on disk) for any further
   spawned-tier evidence.
3. **05-12 must still ask for its SetFit compute separately** — 05-03 recorded it as explicitly
   NOT pre-authorized and deferred to wave 7. Nothing here changes that.

## Self-Check: PASSED

| Claim | Check | Result |
|---|---|---|
| commit `db12111a9` exists | `git log --oneline --all \| grep` | FOUND |
| commit `7ae052f64` exists | `git log --oneline --all \| grep` | FOUND |
| `crates/apr-cli/tests/setfit_cli_lifecycle.rs` | `test -f` | FOUND |
| `crates/aprender-train/tests/setfit_apr_lifecycle.rs` | `test -f` | FOUND |
| `crates/aprender-train/src/train/setfit/verify.rs` | `test -f` | FOUND |
| `crates/aprender-train/src/train/setfit/apr_evaluate_tests.rs` | `test -f` | FOUND |
| `crates/apr-cli/tests/setfit_parity.rs` | `test -f` | FOUND |
| `crates/apr-cli/src/commands/setfit_train.rs` | `test -f` | FOUND |
| `crates/apr-cli/src/commands/predict_tests.rs` | `test -f` | FOUND |
| `crates/apr-cli/src/commands/eval/setfit_tests.rs` | `test -f` | FOUND |
| exactly one top-level `status:` in this frontmatter | `grep -c '^status:'` | 1 (line 91) |
| no file deleted by any of the three commits | `git diff --diff-filter=D --name-only f02a8aadf..HEAD` | empty |
| working tree clean, no untracked files | `git status --short` | empty |
