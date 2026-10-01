---
phase: 04-apr-artifact-and-production-parity
plan: 21
subsystem: aprender-serve HTTP transport / mutation gate
tags: [mutation-testing, setfit, boundary, OPS-05, SAFE-01, gap-closure]
wave: 11
requires:
  - "crates/aprender-serve/src/api/setfit_handlers.rs (04-08's surface)"
  - "cargo-mutants 25.3.1"
provides:
  - "a MEASURED production-mutant survivor set for setfit_handlers.rs at HEAD"
  - "an at-the-bound acceptance test that kills both surviving batch-bound mutants"
  - "a non-vacuous setfit-serve-tests floor at the new measured count"
  - "D-04-21-A — the re-measurement, the varied-input diagnosis, and the D-04-11-A correction"
affects:
  - "crates/aprender-serve/src/api/setfit_handlers.rs"
  - "Makefile (setfit-serve-tests floor + its gate-table row ONLY)"
tech-stack:
  added: []
  patterns:
    - "boundary mutants under defense-in-depth need an ACCEPTANCE witness, not a refusal witness"
key-files:
  created:
    - .planning/phases/04-apr-artifact-and-production-parity/04-21-SUMMARY.md
  modified:
    - crates/aprender-serve/src/api/setfit_handlers.rs
    - Makefile
    - .planning/phases/04-apr-artifact-and-production-parity/deferred-items.md
decisions:
  - "D-04-11-A's Survivor A is moot by deletion, not fixed — no test was written for deleted code"
  - "Plan Test 3 was SKIPPED because its target mutant measured CAUGHT; a measurement was recorded instead of a test that kills nothing"
  - "Plan Test 2's premise was REFUTED by measurement — the one-over responses are byte-identical, so no transport-attributable property exists to assert without a production change"
  - "The newly-found :108 survivor is triaged EQUIVALENT BY CONSTRUCTION with evidence, not closed with a vacuous test"
metrics:
  duration_min: 59
  completed: 2026-08-16
  tasks: 2
  files_changed: 3
  mutants_run: 25
---

# Phase 04 Plan 21: Kill the Measured `setfit_handlers.rs` Mutation Survivors — Summary

Re-measured the production-mutant survivor set for `crates/aprender-serve/src/api/setfit_handlers.rs`
at HEAD instead of inheriting D-04-11-A's claim, found it wrong in three separate ways, diagnosed the
real survivor from three varied inputs, and killed it with an at-the-bound acceptance test confirmed
by a `-F`-scoped cargo-mutants re-run.

## What was measured, and how it differs from D-04-11-A

D-04-11-A's claim was **incomplete rather than wrong** — three commits landed after it and its own run
was interrupted at 68 min before reaching the whole set.

### The enumeration (`--list`, not estimated)

**99 mutants total for the file; 78 are `tests::` / `fixture::`; 21 are PRODUCTION.**

The 78 stay triaged out on D-04-11-A's own argument (a suite cannot detect the deletion of one of its
own tests). Both counts are recorded so the exclusion is auditable rather than a hand-wave. D-04-11-B
recorded 101 for this file; the drop to 99 is the `b47acc4fe` cleanup.

### The full scoped run — verbatim summary line

```
cargo mutants --no-times --timeout 180 --package aprender-serve \
  --features setfit --cargo-arg=--lib \
  -f crates/aprender-serve/src/api/setfit_handlers.rs \
  -E 'tests::' -E 'fixture::' -- setfit
```

Baseline `ok`. **COMPLETED in 593 s** — D-04-11-A's was interrupted at 4,094 s, which is exactly why it
never produced a score.

```
21 mutants tested: 3 missed, 5 caught, 13 unviable
```

**Mechanism proven engaged, not labelled by intent (CLAUDE.md rule 2):** baseline `ok` with a non-zero
caught count. `-- setfit` is load-bearing — the unfiltered `-p aprender-serve --lib` suite is 51-red
(D-04-08-A), so a baseline reported `ok` *is itself* the proof the filter engaged. The `setfit` feature
was genuinely ON: the whole file sits behind `#[cfg(feature = "setfit")]` and produced 21 mutants,
which a build with it compiled out cannot do (the F-04 vacuity check D-04-11-B's deviation 4 exists for).

### Three corrections

| # | D-04-11-A said | Measured at HEAD |
|---|----------------|------------------|
| 1 | Survivor A: `replace AppState::has_setfit_model -> bool with false` is a real gap | **MOOT BY DELETION.** `has_setfit_model` was deleted at `b47acc4fe` ("zero callers"). `grep -rn has_setfit_model crates/` returns nothing (rc=1) and the `--list` log has **zero** occurrences. Its successor mutant `AppState::setfit_model -> None` is **CAUGHT**. |
| 2 | Survivor B is ONE mutant at `:158` | **TWO** mutants at `:152` (`==` and `>=`). The third sibling `> with <` is CAUGHT. |
| 3 | (not reported) | A **THIRD** survivor at `:108`, which 04-11's interrupted run never reached. |

## The varied-input diagnosis (CLAUDE.md rule 6)

D-04-11-A explicitly declined to diagnose from one input, and that judgement stood. **Three inputs were
EXECUTED** through the real router via `oneshot`, under correct code and under each hand-applied mutant:

| batch size | correct `>` | mutant `==` | mutant `>=` |
|-----------|-------------|-------------|-------------|
| `MAX_BATCH_TEXTS - 1` = 255 | **200** | 200 | 200 |
| `MAX_BATCH_TEXTS` = 256 | **200** | **400** `the request carried 256 texts; the bound is 256` | **400** (identical) |
| `MAX_BATCH_TEXTS + 1` = 257 | **400** `the request carried 257 texts; the bound is 256` | **400** (byte-identical) | **400** (byte-identical) |

**The cause, named from the measurements.** At 257 all three implementations answer 400 with the same
body: the mutated transport check falls through, core re-checks the same bound at `classify.rs:624`,
returns `BatchTooLarge { max: 256, got: 257 }`, and `classify_error_response` maps it to `BAD_REQUEST`
with the same `Display` string the transport `refuse` produces. So
`setfit_classify_refuses_a_batch_one_over_the_contract_bound` (`:725`), which asserts 400 plus the
substrings `257` and `256`, **provably cannot distinguish the implementations**.

The diagnosis is therefore **neither of the two D-04-11-A offered**. The test is not mis-scoped and it is
not unreached — it reaches the branch and asserts truthfully. A redundant INNER check makes the outer
branch unobservable through the response *for that input*. That is the cost of defense in depth
(T-04-23), and it is worth paying; the boundary just needs a different witness.

**The distinguishing input is exactly `MAX_BATCH_TEXTS`.** `aprender-core` already had that acceptance
test (`classify.rs:1729`); `aprender-serve` had none — which is precisely why two mutants lived there.

## The kill, proven twice

`setfit_classify_admits_a_batch_at_exactly_the_contract_bound` added beside the existing ten. It reads
the bound from the exported constant (never a literal `256`) and asserts the serialized body sits under
half of `classify_body_limit_bytes()`, so it cannot silently become a body-limit test.

**Hand-applied probe at `:152` — the asymmetry IS the finding:**

| probe | `…admits_a_batch_at_exactly_the_contract_bound` | `…refuses_a_batch_one_over_the_contract_bound` |
|-------|--------------------------------------------------|--------------------------------------------------|
| `> → ==` | **FAILED** | ok |
| `> → >=` | **FAILED** | ok |

**Confirmed by cargo-mutants itself** (the criterion D-04-11-A's action item 1 asked for), baseline `ok`:

```
cargo mutants ... -F 'replace > with .* in setfit_classify_handler|delete match arm' -- setfit
4 mutants tested: 1 missed, 3 caught
```

`caught.txt` holds all three `:152` mutants (`==`, `>=`, `<`). **Zero missed among the mutants this plan
claims to have killed.**

## The one survivor left, and why no test was written for it

`:108:9 delete match arm ClassifyError::EmptyInput | BatchTooLarge{..} | UnsupportedSchemaVersion{..}`
— **MISSED, and triaged EQUIVALENT BY CONSTRUCTION.** `classify_error_response` has exactly one caller
(`:166`), so the arm is reachable only if `classify()` returns one of its three variants after the
transport pre-checks pass:

- `EmptyInput` — pre-empted at `:146`, unreachable;
- `BatchTooLarge` — pre-empted at `:152`, unreachable;
- `UnsupportedSchemaVersion` — **not producible on the request path at all.**
  `ClassifyRequestDocument` is `{ texts, include_logits }` with `deny_unknown_fields` and has no
  `schema_version`; that variant belongs to `ClassifyResponseWire`'s `TryFrom` (`classify.rs:559`), and
  `classify()` builds through `ClassifyResponse::new` (`:722`), never that `TryFrom`. **Measured, not
  inferred:** posting `{"schema_version":2,"texts":["ok"]}` answers **422** from axum's extractor, so the
  variant never reaches the mapper.

Every error `classify()` *can* return under those pre-conditions already lands on the
`_ => INTERNAL_SERVER_ERROR` wildcard, so deleting the arm changes no HTTP response. Same structural
cause as Survivor B — the transport pre-checks shadow core's equivalents — but here it makes the mutant
genuinely equivalent rather than merely hard to observe. A test would kill nothing and would dilute the
floor, so a measurement was recorded instead.

**Adjusted production score after this plan: 7 caught / 7 non-equivalent viable = 100%**
(21 − 13 unviable = 8 viable, less the 1 equivalent).

## The Makefile: exactly TWO lines

| | old | new |
|---|-----|-----|
| measured passed count | 10 | **11** |
| `assert_tests_ran` floor (recipe, `:2177`) | 9 | **10** |
| gate-table row (`:1920`) | `serve setfit  10 / 0   9` | `serve setfit  11 / 0  10` |

`git diff --numstat -- Makefile` = `2 2 Makefile`.

**Floor proven non-vacuous (the CR-02 class):** with the recipe's filter temporarily pointed at a
zero-match name, libtest exits 0 reporting `0 passed` while the gate correctly **fails rc=2** —
*"reported 0 test(s) passed, expected at least 10 … A name filter that matches nothing exits 0
(REVIEW CR-02)"*. Reverted.

**`setfit-cli-eval-tests` at `:1918` and its floor at `:2148` were deliberately NOT touched.** That row is
stale (records 15; 16 at HEAD), but sibling 04-20 adds tests to `eval::setfit` in THIS wave, so any value
written here would be stale by four the moment 04-20 landed — which it did, at `4b78c6f46`, during this
execution. The correction belongs to 04-22 in wave 12, which runs after both siblings. This is exactly
the decay `Makefile:1927-1929` warns about.

## Verification

Statuses captured with `cmd > log 2>&1; rc=$?`, never through a pipe.

| # | Command | rc | Result |
|---|---------|----|--------|
| 1 | `make setfit-serve-tests` | 0 | 11 passed / 0 failed, new floor 10 satisfied |
| 2 | `make setfit-serve-smoke` | 0 | 1 passed (spawned tier) |
| 3 | `make setfit-parity` | 0 | 20 passed / 0 failed / 1 ignored |
| 4 | `cargo test -p aprender-serve --features setfit --lib setfit` | 0 | 11 passed / 0 failed |
| 5 | `cargo mutants -F …` re-run | 2 | `4 mutants tested: 1 missed, 3 caught`; missed = the equivalent `:108` |
| 6 | `make setfit-feature-matrix` | 0 | `setfit-feature-matrix: PASSED` |
| 7 | `cargo fmt -p aprender-serve -- --check` | 0 | clean |

`-p aprender-serve --lib` whole-crate was deliberately NOT run: D-04-08-A records 51 standing failures
from one overflow at `contract_gate.rs:428:21` that this phase did not touch.

0 `unwrap()` in the addition (`expect()` with stated reasons only).

## Deviations from Plan

### [Rule 3 — blocking] Not a worktree: four agents shared ONE checkout

The prompt stated `isolation="worktree"`, but `git rev-parse --git-dir` returned `.git` as a **directory**
and `git worktree list` showed a single entry — this ran in the main checkout on `gsd/phase-2-contract-gate`
alongside all three siblings. Consequences and handling:

- The `<worktree_branch_check>` `worktree-agent-*` namespace assertion does not apply (its `#2924` intent —
  never commit on `main`/`master`/`develop`/`trunk`/`release/*` — is satisfied: this is a non-protected
  feature branch, which is the workflow CLAUDE.md prescribes). **No `git update-ref` or self-recovery was
  attempted.**
- The base assertion was checked BEFORE acting: HEAD was already exactly `91eba6f1c`, so the
  `git reset --hard` branch was never taken. Running it blindly in a shared checkout would have destroyed
  sibling work.
- Because `git commit` commits the shared index, **both commits used explicit pathspecs**
  (`git commit -m … -- <files>`) so sibling staged changes could not be swept in. Verified: each commit
  contains exactly its intended files.

### [Rule 3 — blocking] A sibling's in-flight probe broke the shared tree mid-verification

The first `> → >=` probe failed to **compile**, not on my code: a sibling had renamed
`ExecutionBackend::identity` → `identity_x` in `crates/aprender-core/src/setfit/encoder.rs` (their own
hand-applied mutation probe). Handled by **waiting ~20 s for them to revert** rather than touching their
file, then re-running. The retry produced the expected asymmetry. Recorded because a reader comparing logs
would otherwise see one inexplicable red run.

### Plan Test 2 — premise REFUTED by measurement, so not written

Test 2 was specified as asserting *"the one-over case is refused by the TRANSPORT check specifically,
distinguishably from core's inner refusal … by asserting a transport-attributable property"*. The
measurement shows the two refusals are **byte-identical** (same 400, same `Display` string). No
transport-attributable property exists to assert without adding a distinguishing marker to production
code — a behaviour change outside this plan's scope, and worse design, since the module deliberately
routes core's own message. Test 1 is what makes the outer branch observable, at n=256. The plan
anticipated this branch ("record that … instead of adding a test that kills nothing"); recorded here and
in D-04-21-A. `setfit_classify_refuses_a_batch_one_over_the_contract_bound` was neither weakened nor
renamed.

### Plan Test 3 — SKIPPED, exactly as the plan conditioned

Its target, `replace AppState::setfit_model -> … with None`, measured **CAUGHT** —
`setfit_readiness_is_200_and_reports_the_exact_artifact_hash` asserts the exact hash VALUE, so under
`-> None` the key is absent and the test fails. The readiness gap D-04-11-A's action item 2 asked to close
does not exist at HEAD. Measurement recorded instead of a test.

### [Rule 2 — scope-adjacent finding] A third survivor was found and triaged rather than silently omitted

`:108` was not in the plan's target set. Per the plan's own acceptance criterion ("Any mutant that remains
missed is recorded as still-open in `D-04-21-A`, not silently omitted") it is recorded with an evidence-backed
equivalence argument, including the measured 422 probe. No production code was changed for it.

## What remains OPEN — stated plainly

- **04-11's must-have 4 remains OPEN** — the four-crate per-crate cargo-mutants gate with four baselines and
  an explicitly computed aggregate adjusted score. D-04-11-B measured **≥ 10 h** wall clock for its 890
  mutants; this plan ran **25 mutants over 865 s against ONE file in ONE crate**. It stays a
  `human_verification` item, exactly as D-04-11-B left it. **This plan does not close it and does not claim to.**
- **SAFE-02's "in CI" clause remains OPEN.** No CI run was triggered, nothing was pushed, no PR was opened,
  and no file under `.github/` was touched. The 16 legs at `.github/workflows/ci.yml:378-397` remain never-executed.
- **F-10 remains OPEN. OPS-01 and OPS-02 are still NOT MET. OPS-05 remains PARTIAL** — its fixture is
  synthetic because no user-producible artifact exists. SC1, SC2, SC3 and the "over a produced artifact"
  halves of SC4/SC5 remain OPEN. **This plan closes neither OPS-01 nor OPS-02.**
- **IN-06 (D-04-14-B) is untouched** — the `unwrap()` ban is still not machine-enforced in `apr-cli`.
- `aprender-serve --lib` whole-crate remains a standing RED (D-04-08-A, 51 failures, one overflow).

## Commits

| Task | Commit | Files |
|------|--------|-------|
| 1 — measure & diagnose | `887f4d622` | `deferred-items.md` (+160, −0) |
| 2 — kill & raise the floor | `7ddb0ad98` | `setfit_handlers.rs` (+67), `Makefile` (2/2) |

## Self-Check: PASSED

- `crates/aprender-serve/src/api/setfit_handlers.rs` — FOUND, contains `setfit_classify_admits_a_batch_at_exactly_the_contract_bound`, `:152` intact as `>`, no probe residue.
- `Makefile` — FOUND, exactly 2 changed lines, `setfit-cli-eval-tests` row/floor untouched.
- `deferred-items.md` — FOUND, `## D-04-21-A` present, additions only, `## D-04-11-A` byte-unchanged.
- `04-21-SUMMARY.md` — FOUND.
- Commits `887f4d622`, `7ddb0ad98` — FOUND in `git log`.
- STATE.md and ROADMAP.md — NOT modified (orchestrator owns those writes).
