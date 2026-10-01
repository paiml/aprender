---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 22
subsystem: testing
tags: [gate-honesty, justfile, makefile, iam, mutation-testing, class-c, laya]
status: complete

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-12 laya-verify-suite and contract-audit-phase8; 08-17 stack-declared IAM grant and read-only laya-grant; 08-10 deploy recipes and laya-deploy-selftest"
provides:
  - "scripts/laya_gates.tsv: the class-C enumeration of every Phase 8 gate (27 rows, 2 external, 3 not-a-gate)"
  - "just laya-gates-selftest: verdict case table + drift check over every recipe (private included) BEFORE any case, then one must-fail / must-pass dispatcher per row; AWS CALLS: 0"
  - "_laya-leg-verdict: laya-verify-suite legs need positive MEASURED evidence; SKIP anywhere fails; the real ladder rung now runs in the suite"
  - "laya-grant / _laya-grant-check: every IAM read status-checked, attached managed policies read, wildcard / NotAction / any other S3 grant refused"
  - "laya-teardown NoSuchEntity classification; one shasum helper; _laya-resolver-parse (unanchored)"
  - "Makefile P8_*_ERE variables + contract-audit-phase8-selftest (23-case table), a prerequisite of contract-audit-phase8"
  - "bootstrap intent line: check_duplicate_bin_names.sh green"
affects: [08-23, 08-31, 08-32, phase-8-verification]

actuals:
  tokens: 18900
  tasks: 3
  commits: 3
plan_head_before: 9b013771ebaae6e39a035ad624368635ed0e8abe

tech-stack:
  added: []
  patterns:
    - "Gate table + sweep: every gate is a TSV row with a must-fail and a must-pass case, and a drift check refuses a gate recipe with no row"
    - "not-a-gate is admissible only for a recipe that prints no verdict token and does not exec another program"
    - "PATH-shimmed fakes (aws, rtk, uv, cargo-zigbuild) turn hidden-failure paths into must-fail cases with zero external calls"
    - "Heavy must-pass cases are a named FULL tier; the default OK line lists what it did not run"

key-files:
  created:
    - scripts/laya_gates.tsv
  modified:
    - justfile
    - Makefile
    - crates/aprender-decide/tests/laya_parity.rs
    - scripts/duplicate_bin_names_allowlist.txt
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/COVERAGE.md

key-decisions:
  - "Gate enumeration is fail-closed: exit 1/2/3, exit(N), FAIL, an exec, or a verdict token makes a laya recipe a gate (the plan's literal rule missed exec-delegating and python sys.exit recipes)"
  - "An exec-ing recipe (laya-verify, laya-pack, laya-inspect, laya-pack-fixture) counts as verdict-printing: the drift check cannot see the exec'd program's verdict, so it may not be exempted"
  - "Heavy must-pass cases (armed laya-verify-suite, real repack, DRY_RUN upload, real resolver proof) are the FULL tier (LAYA_GATES_FULL=1); the default sweep runs one real-weights leg (the default-armed deploy selftest)"
  - "_laya-grant-check refuses ANY S3 grant other than the stack-declared read, so its success line is literally true, and names what it does not inspect (boundary, SCPs, bucket policy)"

patterns-established:
  - "Class C gate honesty: a gate added without a row fails laya-gates-selftest"

requirements-completed: [D-07, D-15, D-17, D-18]

coverage:
  - id: D1
    description: "laya-verify-suite leg verdict needs positive evidence; the real ladder rung runs (V11-a, AL7, D3-1, A2-6)"
    requirement: "D-17"
    verification:
      - kind: integration
        ref: "heavy env LAYA_MODEL_DIR=<55cf4c4e> LAYA_LADDER_BIN=<main ladder> cargo test -p aprender-decide --release --test laya_parity -- --nocapture | just _laya-leg-verdict laya_parity -> LEG OK, MEASURED ids/probs/ladder (32 blocks)"
        status: pass
      - kind: unit
        ref: "crates/aprender-decide/tests/laya_parity.rs#nan_max_keeps_an_earlier_nan"
        status: pass
      - kind: integration
        ref: "LAYA_GATES_FULL=1 just laya-gates-selftest row verify-suite -> LAYA VERIFY SUITE OK (870 s)"
        status: pass
    human_judgment: false
  - id: D2
    description: "IAM read failures named honestly, attached policies and wildcards refused by the grant check, teardown NoSuchEntity classification (V11-b, WR-07)"
    requirement: "D-18"
    verification:
      - kind: integration
        ref: "just laya-gates-selftest rows grant-check, grant-listing-failure, teardown-classify (fake aws)"
        status: pass
      - kind: integration
        ref: "DRY_RUN=1 just laya-deploy-selftest GRANT table (12 rows)"
        status: pass
    human_judgment: false
  - id: D3
    description: "One shasum helper independent of rtk on PATH; resolver tokens read anywhere on a line (V11-c, D3-1)"
    requirement: "D-18"
    verification:
      - kind: integration
        ref: "just laya-gates-selftest rows sha256-helper (foreign rtk and rtk-absent PATH), resolver-proof-sed"
        status: pass
    human_judgment: false
  - id: D4
    description: "laya-deploy-selftest default-armed on the deployed artifact; a skipped positive run never prints the bare OK (AL7)"
    requirement: "D-07"
    verification:
      - kind: integration
        ref: "heavy just laya-gates-selftest row deploy-selftest-skip (synthetic-armed FAIL, disarmed qualified OK, default-armed DRY-RUN OK 24a44d7e)"
        status: pass
    human_judgment: false
  - id: D5
    description: "Resolver ERE case table run before contract-audit-phase8; bootstrap intent line (V14-a, V14-c share)"
    requirement: "D-15"
    verification:
      - kind: other
        ref: "make contract-audit-phase8 (23-case selftest first, 44 rows resolved); bash scripts/check_duplicate_bin_names.sh + --self-test"
        status: pass
    human_judgment: false
  - id: D6
    description: "Class C: every Phase 8 gate enumerated in scripts/laya_gates.tsv, swept by laya-gates-selftest, each fix mutation-proven"
    requirement: "D-18"
    verification:
      - kind: integration
        ref: "heavy just laya-gates-selftest -> LAYA GATES SELFTEST OK 27 rows; planted not-a-gate for laya-verify / laya-deploy-verify / _laya-iam-check each refused with DRIFT"
        status: pass
      - kind: integration
        ref: "14-mutation table (below): each RED in its row, restored byte-identical"
        status: pass
    human_judgment: false

duration: 102min
completed: 2026-09-28
---

# Phase 8 Plan 22: Class C Gate Honesty Summary

**Every Phase 8 gate is now a row of `scripts/laya_gates.tsv` with a case that fails it and a case that passes it, and `just laya-gates-selftest` runs them all.** The sweep first runs a drift check over every recipe, private ones included. A gate recipe with no row fails the sweep, and so does a `not-a-gate` exemption on a recipe that prints a verdict. The gates the reviews showed could not fail are fixed, and each fix is mutation-proven:

- the suite's SKIP check that missed the ladder rung;
- the IAM listing whose failure read as a missing grant;
- the grant check that said "nothing broader" without reading attached policies or wildcards;
- the hash helper that depended on `rtk`;
- the selftest that printed a bare OK after skipping its positive leg;
- the resolver EREs, which had no case table.

AWS CALLS: 0.

## Performance

- **Duration:** 102 min
- **Started:** 2026-09-28T17:14:08Z
- **Completed:** 2026-09-28T18:56:09Z
- **Tasks:** 3 (tracer, auto, auto)
- **Files modified:** 6 committed code/config files, plus COVERAGE.md (Task 2) and deferred-items.md (metadata commit)

## Accomplishments

- `_laya-leg-verdict`: a pure classifier. SKIP anywhere on a line fails it. So does a missing positive line: `MEASURED ids/probs/ladder`, `FAIL-CLOSED VECTORS REFUSED 2/2`, `DEMO OUTCOME`, or `NOISE` + `MEDIAN`, depending on the leg. An unknown leg also fails. `laya-verify-suite` routes every leg through it and exports `LAYA_LADDER_BIN` from the MAIN checkout. **The ladder rung now runs in the suite for the first time:** 32 blocks within their bars.
- laya_parity prints its MEASURED lines only after each rung has compared. Its probability rung now uses a NaN-visible running max and checks the row length first (A2-6).
- `laya-grant` is a READ-ONLY check:
  - every IAM read's status is checked;
  - attached managed policies are read too;
  - its header describes what it does.

  `_laya-grant-check` refuses NotAction/NotResource, wildcard S3 actions and resources, any other S3 grant, and any attached policy that reaches S3. Its success line names the counts it checked and what it did not inspect.
- `laya-teardown`: NoSuchEntity means absent. Any other delete failure exits 1 and says containment is already in place.
- `sha256()` calls `shasum` directly in all three deploy recipes. `_laya-resolver-parse` reads RESOLVED and CONTROL anywhere on a line.
- `laya-deploy-selftest` arms its positive dry run by default on the deployed artifact 24a44d7e. A skipped run ends `DEPLOY SELFTEST OK (positive dry run SKIPPED: <why>)`.
- Makefile: three `P8_*_ERE` variables and `contract-audit-phase8-selftest`, a 23-case table that is now a prerequisite of `contract-audit-phase8`. Comment text and just assignments no longer resolve a row. The audit still resolves all 44 rows.
- `bootstrap` intent line in the allowlist: `check_duplicate_bin_names.sh` is green (rc 0, self-test 9/9).
- COVERAGE.md's IAM rows are current: the IAM reads are INTEGRATE, `put_role_policy` is OPT-OUT (no longer used), and the ListBucket sentence is corrected (no ListBucket, so a missing key reads 403).

## Task Commits

1. **Task 1 (tracer): leg verdict + real ladder rung**: `778417a86` (feat)
2. **Task 2: IAM/teardown honesty, grant scope, hash helper, resolver parse**: `47494e1c6` (fix)
3. **Task 3: laya-gates-selftest, ERE case table, honest skip line, bootstrap intent**: `4e6894882` (feat)

**Plan metadata:** the docs commit after this SUMMARY.

## Files Created/Modified

- `scripts/laya_gates.tsv`: the class-C enumeration (created).
- `justfile`:
  - new recipes: `_laya-leg-verdict`, `_laya-resolver-parse` and `laya-gates-selftest`;
  - changed recipes: laya-verify-suite, laya-grant, _laya-grant-check, laya-teardown, the three sha256 helpers, laya-resolver-proof, and laya-deploy-selftest (default arming, honest skip line, extended GRANT table).
- `Makefile`: the ERE variables, `contract-audit-phase8-selftest`, and the audit's prerequisite.
- `crates/aprender-decide/tests/laya_parity.rs`: the MEASURED lines, `nan_max`, the length check, and `nan_max_keeps_an_earlier_nan`.
- `scripts/duplicate_bin_names_allowlist.txt`: the bootstrap line.
- `.planning/phases/08-.../COVERAGE.md`: the IAM rows.

## scripts/laya_gates.tsv (full enumeration)

| gate_id | target | must_fail | must_pass | finding |
|---|---|---|---|---|
| leg-verdict | `just _laya-leg-verdict` | canned SKIP-ladder log / libtest-prefixed SKIP log / missing MEASURED ladder / unknown leg name | canned full-evidence log for each of the four legs | V11-a,AL7,D3-1 |
| grant-check | `just _laya-grant-check` | an attached managed policy with s3:GetObject on arn:aws:s3:::* / s3:Get* on the wanted prefix / s3:GetObject on decide/* / an Allow with NotAction: each REFUSED grant | the stack-declared inline read plus a logs-only attached managed policy: grant ok naming the inline and attached counts | WR-07 |
| grant-listing-failure | `just laya-grant` | fake aws: list-role-policies exits 255 -> exit 1 "IAM read failed", never "REFUSED grant" | fake aws: the stack-declared inline read and a logs-only attached policy -> exit 0 "grant ok" | V11-b |
| teardown-classify | `just laya-teardown` | fake aws: delete-role-policy AccessDenied -> exit 1 naming the failure (containment already done) | fake aws: delete-role-policy NoSuchEntity -> exit 0 "no legacy policy" | V11-b |
| sha256-helper | `just laya-deploy-config` | laya-deploy on a config whose pin was mutated, foreign rtk first on PATH -> REFUSED sha-pin | foreign rtk first on PATH, and rtk absent from PATH: laya-deploy-config pins exactly the digest shasum -a 256 prints | V11-c |
| resolver-proof-sed | `just _laya-resolver-parse` | a canned log with no RESOLVED line -> exit 1 | a canned libtest-prefixed log (test <name> ... RESOLVED root=crates ...) -> RESOLVED=crates/aprender-mcp-decide-lambda | D3-1 |
| deploy-selftest-skip | `just laya-deploy-selftest` | armed with the synthetic fixture (LAYA_ELIGIBLE_* = the tiny artifact) -> exit 1 "FAIL positive dry run" | disarmed (LAYA_DEPLOY_SELFTEST_POSITIVE=0) -> exit 0 ending "DEPLOY SELFTEST OK (positive dry run SKIPPED: ...)", never the bare OK; default-armed on the deployed artifact (real weights) -> DRY-RUN OK and the bare OK | AL7 |
| resolver-ere | `make contract-audit-phase8-selftest` | the pre-08-22 P8_FN_ERE and P8_RECIPE_ERE overridden on the command line -> exit non-zero naming the /// comment and the name := value cases | the committed EREs -> exit 0 (a #[cfg(test)] helper still resolves: a grep cannot see cfg, a documented limit) | V14-a |
| dup-bin-names | `bash scripts/check_duplicate_bin_names.sh` | the guard engine (scripts/lib/bin_names.py) over a temp allowlist without the bootstrap line -> exit 1 naming bootstrap | the guard exit 0 and its --self-test exit 0 | V14-c |
| iam-check | `just _laya-iam-check` | a config granting s3:* / a second [[iam.statements]] -> REFUSED iam | the scoped single-statement config -> iam ok | 08-17 option 1 |
| parser-guard | external:08-23 | bash scripts/check_no_hand_rolled_parsers.sh --self-test | - | V14-c (Phase 8 offender aprender-mcp-decide; plan 08-23 fixes it) |
| cascade-guard | external:08-31 | - | - | V14-c (Phase 8 offender aprender-decide in the cascade TIERS; plan 08-31 fixes it, 08-32 laya-gap-regression runs the guard) |
| crates-root-swap | `just _laya-crates-root-swap` | a root that is not a directory -> exit 2 REFUSED swap; a leftover backup -> exit 2 REFUSED swap | a temp root with state: the command sees the decide config alone, exit 0, RESTORED byte-identical | D-ITEM-08-10 shared-crates-root |
| edge-health-url | `just _laya-edge-health-url` | https://<host>/health and http://<host>/mcp -> exit 2 REFUSED | https://<host>/mcp -> https://<host>/health | D-ITEM-08-17-A |
| edge-health-check | `just _laya-edge-health-check` | the chronos edge body and the bootstrap body -> REFUSED edge-health | the decide edge body -> edge-health ok | D-ITEM-08-17-A |
| bootstrap-build | `just laya-build-bootstrap` | a cargo-zigbuild shim that builds nothing -> exit 1 (the binary is not newer than the build) | the real arm64 cross-build -> BOOTSTRAP aarch64 OK | 08-07 stale bootstrap |
| deploy-refusals | `just laya-deploy` | the synthetic fixture: a placeholder config -> REFUSED placeholder; a consistent config -> REFUSED eligibility SyntheticNotDeployable | the default-armed deploy selftest positive dry run on the deployed artifact -> DRY-RUN OK | D-07 |
| upload-eligibility | `just laya-upload` | the synthetic fixture -> REFUSED eligibility SyntheticNotDeployable (no AWS call) | FULL tier: DRY_RUN upload of the deployed artifact -> DRY-RUN: would upload | D-07 |
| verify | `just laya-verify` | the synthetic fixture -> exit 2 REFUSED SyntheticNotDeployable | the deployed artifact (via the default-armed deploy selftest) -> deploy_eligible true with its sha256 | D-07 |
| pack | `just laya-pack` | the synthetic run -> exit 2 REFUSED SyntheticNotDeployable, nothing written | FULL tier: the deployed run repacked to a temp file -> PACKED | D-07 |
| pack-fixture | `just laya-pack-fixture` | a run dir with no task.json -> exit 2 REFUSED, nothing written | the tiny fixture -> PACKED-FIXTURE with the golden sha256 | D-07 |
| inspect | `just laya-inspect` | the tiny artifact truncated to 1000 bytes -> exit 2 REFUSED load ladder | the tiny artifact -> its sha256 and variant synthetic-fixture | D-07 |
| verify-suite | `just laya-verify-suite` | a model dir with no model.safetensors -> exit 2; the real model with LAYA_LADDER_BIN missing -> exit 2 | wiring: every leg verdict is _laya-leg-verdict and laya_parity gets LAYA_LADDER_BIN; FULL tier: the armed suite -> LAYA VERIFY SUITE OK with four LEG OK lines | V11-a,AL7,D3-1 |
| train-selftest | `just laya-train-selftest` | a uv shim failing gate.py --selftest -> exit non-zero, no LAYA TRAIN SELFTEST OK | the torch-free self-tests and the lifecycle -> LAYA TRAIN SELFTEST OK | D-07 |
| deploy-verify | `just laya-deploy-verify` | samples=1 -> exit 2; a missing artifact -> exit 2 | DRY_RUN on the deployed artifact -> DRY-RUN: stopped before the network | D-ITEM-08-11-A |
| resolver-proof | `just laya-resolver-proof` | an sdk path that is not a git checkout -> exit 2 | wiring: RESOLVED/CONTROL come from _laya-resolver-parse (no column-0 sed); FULL tier: the real proof at the recorded sdk_commit -> RESOLVER PROOF crates/aprender-mcp-decide-lambda | D3-1 |
| gates-selftest | `just laya-gates-selftest` | a temp table with an unknown gate_id -> exit 1 before any case; a temp table exempting laya-verify as not-a-gate -> exit 1 DRIFT | a one-row filtered sweep (LAYA_GATES_ONLY=edge-health-url) -> LAYA GATES ROWS OK | class C |

### not-a-gate rows (each with its reason)

| recipe | reason |
|---|---|
| `laya-prepare-stance` | data preparation: exits non-zero only when its local inputs are missing; the shot and row-count assertions are prepare_stance.py and the recipe renders no verdict |
| `laya-train` | a thin wrapper over train.py: its own exit is only the missing-task.json refusal; the gate decision is train.py, whose rules gate.py --selftest proves (row train-selftest) and whose one declared run is 08-GATE-RUN-EVIDENCE.json |
| `laya-weights-bucket` | provisioning (AWS writes): exits non-zero on a bad env name or a failing aws call; nothing downstream reads it as a check |

The drift check reported: 28 laya recipes enumerated (private included) and 26 gates. All 26 are covered: 23 by row targets and 3 by not-a-gate rows. The table has 27 rows. `laya-fixtures` and `laya-train-lifecycle` are not gates: they have no exit, no FAIL, no exec and no verdict token in their bodies.

## Mutation table (every fix reverted alone, its row run, then restored)

Each mutation was applied by `mutate.py` and run through its row. The row runs used `lockf -k /tmp/aprender-laya-real-weights.lock env LAYA_GATES_ONLY=<row> just laya-gates-selftest`. The file was then restored and byte-compared with the original. All 14 restores compared equal.

| # | Fix reverted | Row | Result | What went red |
|---|---|---|---|---|
| M1 | leg verdict back to a column-0 `^SKIP:` grep | leg-verdict | exit 1, RED | the libtest-prefixed SKIP log PASSED the verdict (exit 0); the SKIP-ladder log failed only for the missing ladder line |
| M2 | inline listing failure swallowed (the pre-08-22 empty-loop semantics) | grant-listing-failure | exit 1, RED | exit 1 but no "IAM read failed"; "a failed listing was reported as a grant verdict" |
| M3 | grant check ignores attached managed policies | grant-check | exit 1, RED | must-fail attached-s3-any exit 0 (accepted) |
| M4 | teardown back to the `2>/dev/null` branch | teardown-classify | exit 1, RED | must-fail access-denied exit 0 ("no legacy policy") |
| M5 | rtk proxy hash branch restored in laya-deploy-config | sha256-helper | exit 1, RED | with a foreign rtk first on PATH the pin is 'rtk', not shasum's 37d65159... |
| M6 | resolver token re-anchored at column 0 | resolver-proof-sed | exit 1, RED | the libtest-prefixed log yields no RESOLVED (exit 1); CONTROL not read |
| M7 | deploy selftest prints the bare OK after a skipped positive run | deploy-selftest-skip | exit 1, RED (508 s, real weights) | disarmed run printed the bare DEPLOY SELFTEST OK |
| M8 | P8_FN_ERE without the comment exclusion (the pre-08-22 form) | resolver-ere | exit 1, RED | `make contract-audit-phase8-selftest` exit 2 (4 comment cases match) |
| M9 | bootstrap intent line removed from the allowlist | dup-bin-names | exit 1, RED | no bootstrap line to remove; the guard itself exit 1 |
| M10 | drift enumeration back to `just --summary` | drift check | exit 1, RED | `DRIFT: the recipe enumeration does not contain the private recipe _laya-iam-check`; no case ran |
| M11 | the not-a-gate verdict constraint removed | Task-3 verify (planted exemptions) | exit 1, RED | "the selftest failed, but not with the not-a-gate refusal for laya-verify" |
| M12 | laya-verify-suite no longer arms the ladder rung | verify-suite | exit 1, RED | wiring: laya_parity does not get LAYA_LADDER_BIN |
| M13 | probability rung's NaN-masking fold restored (A2-6) | `cargo test --test laya_parity nan_max` | exit 101, RED | "an earlier NaN was dropped: 0.1" |
| M14 | laya-grant stops reading attached managed policies | grant-listing-failure | exit 1, RED | must-fail attached-fail exit 0; must-pass count "1 attached" absent |

## Evidence

- **Task 1 tracer (real weights, under the host lock).** Run: `heavy env CARGO_INCREMENTAL=0 LAYA_MODEL_DIR=<55cf4c4e> LAYA_LADDER_BIN=<MAIN ladder> cargo test -p aprender-decide --release --test laya_parity -- --nocapture`. It returned rc 0 in 23 s, and `_laya-leg-verdict` printed `LEG OK: laya_parity`. Measured lines:
  - `MEASURED ids 14/14 argmax 14/14 truncated 1`
  - `MEASURED probs max_abs 3.841e-6 bar 1e-5 logits max_abs 2.146e-5 bar 1e-4 ARCH aarch64`
  - `MEASURED ladder 32 blocks within bars` (worst layer rel_rms 2.145e-6)

  The three canned must-fail logs and an unknown leg each exited non-zero.
- **Task 2 verify.** `DRY_RUN=1 just laya-deploy-selftest` returned rc 0. The GRANT table ran 12 rows: 5 new, and every refusal named its intended reason. No `rtk proxy shasum` remains. `list-attached-role-policies` is read. All six TSV rows are present.
- **Task 3 verify (the plan's exact command).** VERIFY rc 0:
  - `heavy just laya-gates-selftest` printed `LAYA GATES SELFTEST OK 27 rows (FULL-tier must-pass cases SKIPPED: upload-eligibility, pack, verify-suite, resolver-proof; set LAYA_GATES_FULL=1)`.
  - The three planted not-a-gate tables each failed with `DRIFT: <recipe> prints a verdict`, before any case ran.
  - `make contract-audit-phase8`: 23-case selftest first, then 44 rows resolved, rc 0.
  - `check_duplicate_bin_names.sh` rc 0, and `--self-test` 9/9.
- **FULL tier, run once.** `lockf ... env LAYA_GATES_FULL=1 just laya-gates-selftest` returned rc 0 in 2066 s and printed the bare `LAYA GATES SELFTEST OK 27 rows`. It includes:
  - the armed laya-verify-suite: `LAYA VERIFY SUITE OK` (870 s), with the ladder rung armed;
  - the real resolver proof: `RESOLVER PROOF: crates/aprender-mcp-decide-lambda (control: ... -> crates/aprender-mcp-chronos-lambda)`;
  - a DRY_RUN upload of the deployed artifact;
  - a real repack of the deployed run, whose printed digest prefix `24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae` matches the deployed artifact. The row printer truncates at 150 characters, so 54 of the 64 hex digits were seen, and the row does not compare the digest.
- **Side effects.** None remained after any sweep. The real `crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml` came back byte-identical: its pin is still 24a44d7e. `resolver-proof.txt` keeps `proven_at=2026-09-27T04:05:49Z`, and `models/decide/swap-backup/` is empty.
- **AWS CALLS: 0.** The sweep's aws recorder stayed at 0 lines in every run. Every AWS-touching case used the PATH-shimmed fake aws.

## Decisions Made

- The gate enumeration is fail-closed. A laya recipe counts as a gate when its body has `exit 1|2|3`, `exit(N)`, `FAIL`, an `exec`, or a verdict token.
- An exec-ing recipe counts as verdict-printing, so it cannot be exempted as not-a-gate.
- Heavy must-pass cases form a named FULL tier, and the default OK line lists the cases it did not run.
- `_laya-grant-check` refuses every S3 grant other than the stack-declared read, so its success line claims only what it checked.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical] The drift check's gate definition covers exec-delegating and python-exit recipes**
- **Found during:** Task 3.
- **Issue:** The plan's rule counts a recipe as a gate when its body contains `exit 1|2|3` or `FAIL`. That rule misses the four recipes that `exec` pack_laya (laya-verify, laya-pack, laya-inspect, laya-pack-fixture). It also misses the python `sys.exit(1)` helpers (`_laya-grant-check`, `_laya-iam-check`, `_laya-edge-health-*`) and `laya-train-selftest`. Under the literal verdict ERE, the plan's own must-fail case would have been ACCEPTED: laya-verify's body is an `exec` with no verdict token, so a planted `not-a-gate` row for it passes.
- **Fix:** A recipe is also a gate when it runs `exec`, calls `exit(N)`, or prints a verdict token, and an `exec` counts as verdict-printing. This added 17 rows beyond the 12 the plan named, each with its own must-fail and must-pass cases.
- **Files modified:** justfile, scripts/laya_gates.tsv.
- **Verification:** The drift check reports 26 gates, all covered. Each planted exemption fails with `DRIFT: <recipe> prints a verdict`.
- **Committed in:** 4e6894882.

**2. [Rule 2 - Missing critical] FULL tier for heavy must-pass cases**
- **Found during:** Task 3.
- **Issue:** Four must-pass cases load multi-GB weights or build an SDK: the armed laya-verify-suite (903 s at 08-12, 870 s here), a real repack, a DRY_RUN upload, and the real resolver proof. Running them on every sweep makes the sweep take 35 minutes. Skipping them silently would be exactly the AL7 defect.
- **Fix:** These cases run only with `LAYA_GATES_FULL=1`, and without it the OK line names each case it did not run. The default sweep keeps the one real-weights leg the plan requires: the default-armed deploy selftest. The FULL tier was run once, with a bare OK.
- **Committed in:** 4e6894882.

**3. [Rule 3 - Blocking] `_laya-resolver-parse` factored out of laya-resolver-proof**
- **Found during:** Task 2.
- **Issue:** Testing the unanchored token read against a canned log needs the read as a separate unit. The recipe's original `sed | head -n 1` also risks SIGPIPE under `pipefail`.
- **Fix:** A private recipe using `awk match`. laya-resolver-proof calls it.
- **Committed in:** 47494e1c6.

**4. [Rule 2 - Missing critical] The grant check refuses ANY other S3 grant, not only wildcard ones**
- **Found during:** Task 2.
- **Issue:** The plan's success line says "no other S3 grant found". Refusing only wildcards would have left that claim untrue for an exact ARN on another bucket.
- **Fix:** Any S3 statement other than the stack-declared read is refused. The success line also names what is not inspected: the permission boundary, SCPs and the bucket policy.
- **Committed in:** 47494e1c6.

**5. [Rule 2 - Missing critical] A unit test for the NaN-visible max**
- **Found during:** Task 3 (the mutation proof).
- **Issue:** No gate row can inject a NaN into a real-weights rung, so the A2-6 fix could not be mutation-proven through the table.
- **Fix:** `nan_max_keeps_an_earlier_nan`. Its mutation M13 goes RED.
- **Committed in:** 4e6894882.

**6. [Rule 3 - Blocking] `LAYA_DEPLOY_SELFTEST_POSITIVE=0` disarm switch**
- **Found during:** Task 3.
- **Issue:** With the positive dry run armed by default, the honest-skip must-pass case needs a way to disarm it.
- **Fix:** Setting `LAYA_DEPLOY_SELFTEST_POSITIVE=0` disarms the positive run. The skip reason is printed in the final line.
- **Committed in:** 4e6894882.

**7. [Rule 3 - Blocking] The dup-bin-names must-fail case uses the guard's engine directly**
- **Found during:** Task 3.
- **Issue:** `check_duplicate_bin_names.sh` hardcodes its allowlist path, and the script is not in `files_modified`.
- **Fix:** The must-fail case runs `scripts/lib/bin_names.py` over a temp allowlist and fresh `cargo metadata` for both workspaces, the same invocation the guard makes. The must-pass case runs the guard itself.
- **Committed in:** 4e6894882.

---

**Total deviations:** 7 auto-fixed (4 missing-critical, 3 blocking).
**Impact on plan:** All seven serve the plan's own invariant. Each gate, the new rows included, can be shown to fail. There is no scope outside the gate surface.

## Issues Encountered

- `DRY_RUN=1 just laya-deploy-selftest`, Task 2's verify command, is now a real-weights leg: Task 3 arms its positive dry run by default. Run it under the host lock. This is intended (AL7), but anyone re-running Task 2's verify alone should know.
- Pre-existing and out of scope: `cargo clippy -p aprender-decide --test laya_parity -- -D warnings` fails on warnings in the aprender-compute dependency. Without `-D`, the changed test file has zero warnings.
- Pre-existing and out of scope, logged in deferred-items.md: aprender-compute, aprender-core and aprender-decide recompile on every `cargo run -p aprender-decide`, about 17 s each time.

## Known Stubs

None.

## Threat Flags

None. The only new surface is read-only IAM calls in laya-grant, which are listed in COVERAGE.md. Every selftest AWS case runs against a fake.

## User Setup Required

None.

## Next Phase Readiness

- 08-23 owns the parser-guard offender and 08-31 owns the cascade offender. Both are `external` rows naming their owner plan. 08-32's laya-gap-regression can call `just laya-gates-selftest`.
- A new Phase 8 gate recipe without a row now fails the sweep.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-28*

## Self-Check: PASSED

- FOUND scripts/laya_gates.tsv and this SUMMARY; FOUND commits 778417a86, 47494e1c6, 4e6894882, 4ae7219e4.
