---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 18
subsystem: infra
tags: [laya, aprender-mcp-decide-lambda, pmcp-run, lambda, live-record, accepted-region-cold, cold-start, deferred-items]
outcome: deployed-passed

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-17: deployed-passed on the option-1 resume. 4 cold samples, identity == H, the function left RUNNING"
provides:
  - "08-LIVE-DEPLOY-EVIDENCE.json FINAL (final: true, outcome deployed-passed). This is the record plan 08-12 binds accepted_region_cold from"
  - "an external cold observation in the record: HTTP 200 at 31.05 s client time, correlated to CloudWatch (REPORT 28008 ms, load_ms 26202)"
  - "the posture: RUNNING by the user's decision, verified read-only, with containment and teardown recorded and not run"
  - "deferred items closed or updated for 08-12's close-out, plus the new D-ITEM-08-18-A (open endpoint)"
  - "the Lambda crate README's Deployed section: endpoint, request shape, identity check, 3 GB limits, claim scope"
affects: [08-12, decide deploy recipes, pmcp.run edge behaviour]

actuals:
  tokens: 5289     # chars/4 over git diff 4f79d157c..b3ace9a3d (21155 chars), SUMMARY excluded
  tasks: 2         # Task 1 skipped by its own precondition (not applicable), Task 2 executed
  commits: 1       # MEASURED: git rev-list --count 4f79d157c..HEAD before this SUMMARY's docs commit
plan_head_before: 4f79d157c8f627a1390789fbcd082f43f5da6a18

tech-stack:
  added: []
  patterns:
    - "External observations are joined to CloudWatch by time window before they are recorded, so a client number always comes with its in-function number"

key-files:
  created:
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-18-SUMMARY.md
  modified:
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-LIVE-DEPLOY-EVIDENCE.json
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md
    - crates/aprender-mcp-decide-lambda/README.md

key-decisions:
  - "The label stays deployed-passed. The plan's rule is defined on the laya-deploy-verify samples (CloudWatch-proven cold, maximal, probe-id matched), and all 4 are < 30000 ms. The external 31.05 s call is risk evidence, not a relabel: it is non-maximal, has no probe id, and its in-function time (28008 ms) is itself under the cap"
  - "Task 1 (blocking-human, exceeded region) is not applicable. Its precondition (outcome deployed-exceeded with exceeded_resolution pending-human) is false, so no checkpoint was raised"
  - "The function stays RUNNING by the user's decision. The deployed-passed branch of the verify does not require reserved concurrency 0, so there is no plan/user conflict. Containment is recorded as what it WOULD be, and it was not run"
  - "D-ITEM-08-10-C is superseded, not resolved. The plan resolves it only if every live assumption was confirmed, and the GET-health one was refuted. Its replacement check passed live, so nothing is left to measure"
  - "The rule samples' elapsed_ms is client-side through the same edge (not in-AWS). The client-minus-REPORT gap was 753-785 ms for the samples and about 3042 ms for the external call. So the contract's 895 ms gateway+client term is not a bound on client overhead"

patterns-established:
  - "A final live record carries the posture it leaves behind: state, how it was verified, what containment would be, and the commands, even when nothing is contained"

requirements-completed: [D-10, D-18]

coverage:
  - id: D1
    description: "08-LIVE-DEPLOY-EVIDENCE.json finalised: final true, outcome deployed-passed, samples unaltered, no exceeded_resolution, the Task 1 decision recorded as not applicable"
    requirement: "D-18"
    verification:
      - kind: other
        ref: "08-18-PLAN Task 2 <automated> verify (extracted and run under bash) -> 'final outcome=deployed-passed', rc 0, before and after commit b3ace9a3d"
        status: pass
    human_judgment: false
  - id: D2
    description: "External cold observation (31.05 s, HTTP 200, identity == H) and warm prolife -> against, correlated to CloudWatch REPORT 28008 ms / load_ms 26202 / graviton2, in the record and in D-ITEM-08-17-E"
    verification:
      - kind: other
        ref: "aws logs filter-log-events /aws/lambda/aprender-mcp-decide 23:10:00-23:11:30Z (read-only): performed_load=true load_ms=26202, platform.report durationMs 28008.462"
        status: pass
    human_judgment: true
    rationale: "The client-side numbers (31.05 s, 2.16 s, probabilities) come from the orchestrator's own observation and are recorded as given. 08-18 correlated them by time window only (probe_id=none) and did not re-measure them"
  - id: D3
    description: "Posture verified read-only: reserved concurrency unset (RUNNING), config 3008 MB arm64 Timeout 30 with pin == H, edge /health 200 serverId aprender-mcp-decide. No AWS write"
    verification:
      - kind: other
        ref: "aws lambda get-function-concurrency (empty) and get-function-configuration --query (mem/arch/state/sha) at 23:12:15Z; curl GET /health 200 at 23:12:23Z"
        status: pass
    human_judgment: false
  - id: D4
    description: "Deferred items: 08-11-A resolved, 08-10-D resolved, 08-10-C superseded, 08-17-A resolved, 08-17-C/D/E updated, new 08-18-A"
    requirement: "D-10"
    verification:
      - kind: other
        ref: "grep -A2 '### D-ITEM-08-11-A' deferred-items.md | grep 'status: resolved' (part of the plan verify), plus the status lines quoted below"
        status: pass
    human_judgment: false
  - id: D5
    description: "README Deployed section: server name, endpoint and request shape, identity check, 3 GB limits (2 texts, 120 tokens, cold about 25-31 s), eval_set.claim quoted, no account id or bucket name"
    verification:
      - kind: other
        ref: "plan verify: grep 'Deployed' in the README; no 'aprender-decide-weights-' and no live account id (sts) in the README or the record"
        status: pass
    human_judgment: true
    rationale: "Whether the request shape and limits are enough for the admin UI's WASM client is the user's judgment"

duration: 6min
completed: 2026-09-27
status: complete
---

# Phase 8 Plan 18: Final Live Outcome Record Summary

**The live branch closes as `deployed-passed`, recorded final.**
- `aprender-mcp-decide` serves on pmcp.run at 3,008 MB arm64, pinned to H `24a44d7e…`, with auth off.
- All 4 `laya-deploy-verify` cold samples came in under 30 s: 24.2-29.4 s, 650 ms margin at worst.
- An external cold call from the user's laptop answered HTTP 200 at **31.05 s** client time. Its
  in-function time was 28008 ms (CloudWatch). This is recorded as margin-risk evidence, not a relabel.
- The function is left RUNNING by the user's decision. The README now documents how to call it.

## Performance

- **Duration:** about 6 min (2026-09-27T23:12:07Z to about 23:18Z).
- **Tasks:** 2. Task 1 was skipped by its precondition, and Task 2 was executed.
- **Files modified:** 3, plus this SUMMARY.
- **Start sha:** `4f79d157c8f627a1390789fbcd082f43f5da6a18` (`/tmp/p08-18-start.sha`).

## outcome: deployed-passed

Task 2's verify was extracted from the plan and run under bash. It printed `final outcome=deployed-passed`
with rc 0, both before and after commit `b3ace9a3d`. It checks all of the following:
- `final: true`;
- every sample is CloudWatch-cold and < 30000 ms;
- there is no `exceeded_resolution`;
- the README has a `Deployed` section;
- D-ITEM-08-11-A reads `status: resolved`;
- neither the record nor the README contains the bucket prefix or the live account id (read with sts);
- nothing under `contracts/`, the three decide crates' `src/` or `binding.yaml` changed since the start sha.

## Task 1: not applicable

- **Precondition:** `outcome: deployed-exceeded` and `exceeded_resolution: pending-human`.
- **The record says:** `deployed-passed`, with no resolution key.
- **Result:** the precondition is false, so the blocking-human decision does not fire. No checkpoint was
  invented. The record carries `final_record.exceeded_decision: "not applicable: ..."`.

## Task 2: what the final record adds

All additions are appended keys. No existing value or sample was changed.

| Key | Content |
|---|---|
| `final` | `true` |
| `final_record` | The label rule applied unchanged: 4 of 4 samples < 30000 ms, max 29350. Also: why the external call does not relabel, and the measurement note below |
| `external_observation` | The cold call: HTTP 200, time_total 31.05 s, identity == H. The warm call: 2.16 s, "Every life deserves protection from conception. #prolife" -> `against` (none 0.072, against 0.835, favor 0.094), which is correct for the TweetEval abortion target. The CloudWatch correlation, and the risk arithmetic |
| `posture` | RUNNING by user decision, verified at 23:12:15Z. What containment would be. 0 AWS writes by 08-18 |
| `recommendations_not_acted_on` | The S3 download, sha2 asm, the 10,240 MB tier, a warm floor or async MCP Task front door, and the pmcp.run issue (D-ITEM-08-17-D) |

**CloudWatch correlation.** Read-only, and matched by time window only, because the call sent no probe id:
- The cold invocation's init started at 23:10:11Z.
- `load_ms=26202`: download 15125, sha 3521, build 7468. That is **slower than every rule sample**.
- 8 S3 part attempts timed out at 8 s and were retried.
- It ran on graviton2 and classified 67 tokens in 1791 ms.
- REPORT: **28008 ms**, Max Memory Used 2482 MB, Init 64 ms, `success`.
- Two warm invocations followed, at 1.80 s each.

**Measurement note (a finding).** The rule samples were not measured in AWS:
- `laya-deploy-verify` runs the probe locally and POSTs through the same edge. Only the cold proof comes
  from CloudWatch.
- For the samples, client minus REPORT was 753-785 ms. For the external curl it was about 3042 ms.
- So client overhead is not a fixed ~0.8 s. The contract's 895 ms "gateway + client" term does not
  bound it (the cause is unmeasured).
- The function's own 30 s timeout still bounds the in-function time.

**Risk arithmetic (not a measurement).** A maximal 120-token cold call on that graviton2 environment
would run about 26202 + 120 x 26.8 = 29418 ms in-function. That leaves about 580 ms under the Lambda
timeout, and comes to about 32.5 s at the client.

## Posture: RUNNING by the user's decision

| Check (read-only) | Result |
|---|---|
| `get-function-concurrency` | no reservation, so the function is RUNNING |
| `get-function-configuration` | MemorySize 3008, Timeout 30, arm64, ephemeral 512, Active / Successful, LastModified 23:00:20Z, pin == H |
| edge `GET /health` | 200 `{"status":"healthy","serverId":"aprender-mcp-decide",...}` |

- **The plan's "containment verified (reserved concurrency 0)"** is required only on the
  `defer-and-contain` and `deploy-refused` branches. The deployed-passed branch does not require it. So
  keeping the function running is consistent with the plan, and there was no conflict to override.
- **Containment would be:** `just laya-teardown aprender-mcp-decide dev ze-kasher-dev`.
  - It sets reserved concurrency 0, and `get-function-concurrency` must then read 0.
  - The stack read stays attached, where it is inert.
  - Resume with `delete-function-concurrency`.
- **Not run.** Neither was the destroy or the S3 removal (both in `teardown_commands_not_run`).
- **Tracked as D-ITEM-08-18-A,** owned by the user.

## Deferred-item status lines (quoted)

- **D-ITEM-08-11-A:** `- status: resolved (08-18, 2026-09-27). The final live outcome is **deployed-passed**`
- **D-ITEM-08-10-C:** `- status: superseded (08-18, 2026-09-27). This is NOT `resolved`: the rule is resolved only if every live`
  - Tally: 4 assumptions confirmed. 1 was refuted (GET health) and replaced by the edge `/health` check, which passed live.
  - The identity probe, `laya-deploy-verify` and containment/resume have all now run live.
- **D-ITEM-08-10-D:** `- status: resolved (08-18, 2026-09-27). The deploy option that ran was `deploy-auth-off-accept-risk`.` Auth is off, with provider none.
- **D-ITEM-08-17-A:** `- status: resolved (08-18, 2026-09-27). Its one pending condition was the live run of the replacement`
- **D-ITEM-08-17-C:** open. Adds the live `sha_ms` of 2655-3521 over 5 cold loads.
- **D-ITEM-08-17-D:** open, recommended, no action taken. Adds that the edge forwarded a 31.05 s client-time call, so the edge's cutoff should be documented in the same platform issue.
- **D-ITEM-08-17-E:** open. Adds the external observation, the CloudWatch correlation, the risk arithmetic, why the label stays, and the four levers (recommendations only).
- **D-ITEM-08-18-A (new):** `- status: open (the user's call; nothing for an executor to do)`. This is the running, open endpoint.

## README Deployed section

Added to `crates/aprender-mcp-decide-lambda/README.md`:
- the server name, the endpoint (POST only), auth off, and the task and labels;
- the admin-UI request shape and the response fields;
- the identity check (`model.artifact_sha256 == 24a44d7e…`);
- the 3 GB limits: 2 texts or fewer, 120 built tokens or fewer, cold about 25-31 s, warm about 1.4-2.2 s;
- `eval_set.claim` quoted: in-distribution calibration, not shift robustness, beside the 0.1896 shift probe;
- no account id, bucket name or token.

## Task Commits

1. **Task 1: exceeded-region decision.** Not applicable; the precondition is unmet, so there is no commit.
2. **Task 2: final live record, deferred items, README Deployed section.** `b3ace9a3d` (feat)

**Plan metadata:** the docs commit that carries this SUMMARY.

## Deviations from Plan

**1. [Rule 2 - Missing critical] The external observation correlated to CloudWatch before recording**
- **Found during:** Task 2.
- **Issue:** a client time alone cannot say whether the cap was crossed inside the function or at the edge.
- **Fix:** one read-only `logs filter-log-events` over 23:10:00-23:11:30Z. It found REPORT 28008 ms and
  load_ms 26202, and it also showed that the rule samples' `elapsed_ms` is client-side.
- **Files:** the record and deferred-items.md. **Commit:** `b3ace9a3d`.

**2. [Scope, documentation] D-ITEM-08-17-A resolved and D-ITEM-08-18-A added**
- The plan lists only 08-11-A, 08-10-C and 08-10-D.
- 08-17-A's only open condition, the live run of its replacement check, was met in 08-17, so leaving it
  open would mislead 08-12's close-out.
- The running endpoint needed an owner line.
- Commit: `b3ace9a3d`.

**Total deviations:** 2 (1 missing-critical evidence step, 1 documentation scope). No contract, Rust
constant, binding or AWS resource changed.

## Issues Encountered

- **The orchestrator's framing says "in-AWS samples". The samples' `elapsed_ms` is client-side** (see
  the measurement note). This does not change the label: the rule is defined on those samples as
  measured. The difference is recorded instead of repeated.
- **The rtk hook condenses `aws` and `curl` output,** so every read used `rtk proxy`.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: open-endpoint | 08-LIVE-DEPLOY-EVIDENCE.json | Still RUNNING with auth off, by user decision (D-ITEM-08-18-A). Each cold call buys about 25-28 s of 3 GB compute. Contain with `just laya-teardown` |

## Next Phase Readiness

- **ROADMAP and STATE move to 17/18.** Next is **08-12** (Wave 16 close-out).
- Its preconditions from this record:
  - `final: true` and `outcome: deployed-passed`, which means `accepted_region_cold` is `implemented`;
  - `PHASE8_LIVE_EXEMPT` is EMPTY;
  - FALSIFY-DECIDE-TOOL-009's `LIVE-PENDING` is swapped for the live harness plus the evidence path.
- D-ITEM-08-17-E (the cold margin) is carried as open work with its levers.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-27*

## Self-Check: PASSED

- Files exist: 08-LIVE-DEPLOY-EVIDENCE.json (`final: true`), deferred-items.md, crates/aprender-mcp-decide-lambda/README.md (`## Deployed`).
- Commit `b3ace9a3d` exists (`git log --oneline -1` after the commit), with no file deletions.
- Plan verify: rc 0 (`final outcome=deployed-passed`), before and after the commit.
- No live account id or bucket prefix in the record, deferred-items.md or the README (checked against sts, read-only).
- AWS writes by this plan: 0.
