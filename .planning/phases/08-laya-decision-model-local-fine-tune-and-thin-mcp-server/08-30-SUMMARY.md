---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 30
subsystem: infra
tags: [lambda, pmcp-run, s3, cold-start, download-deadline, decide-tool-boundary, accepted-region, laya]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-22 deploy selftest, 08-24 hardened bootstrap (one sha pass), 08-25/08-26 ladder rungs, 08-28 isError refusals and tier-derived description"
provides:
  - "aprender-mcp-decide live on pmcp.run at 10,240 MB (8 texts / 800 built tokens), serving b615d8244"
  - "s3::DOWNLOAD_DEADLINE 10739 ms derived from live 10,240 MB cold samples, and download_budget(remaining, reserve) bounding the download by the Lambda invocation deadline (V4-b)"
  - "decide-tool-boundary-v1 9.0.0: priced vs measured 10,240 MB terms, accepted_region_cold implemented on a FINAL live record"
  - "08-LIVE-REDEPLOY-EVIDENCE.json: 8 proven-cold samples at 10,240 MB with every term, identity, probe verdict and the derived deadline"
affects: [08-31, 08-32, phase-8-verification, laya-decision-server, D-18]

actuals:
  tokens: 25463
  tasks: 3
  commits: 10
plan_head_before: e0be0262b6e31a3e55d6ada7a79f39d8d3a555be

tech-stack:
  added: []
  patterns:
    - "A network deadline derived from live per-term cold samples: R = max(post-download work) + contract margin; deadline = min(old, cap - R); the constant's doc names the samples"
    - "Invocation-deadline budget: min(DOWNLOAD_DEADLINE, remaining - reserve), saturating to zero so a late-started load fails fast and the next invocation retries"
    - "Prove the served binary is HEAD by a log line that exists only since the deployed commit (decide.load download_budget_ms=)"

key-files:
  created:
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-LIVE-REDEPLOY-EVIDENCE.json
  modified:
    - crates/aprender-mcp-decide-lambda/src/s3.rs
    - crates/aprender-mcp-decide-lambda/src/lib.rs
    - crates/aprender-mcp-decide-lambda/src/main.rs
    - crates/aprender-mcp-decide-lambda/src/tests.rs
    - crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template
    - crates/aprender-mcp-decide-lambda/README.md
    - contracts/decide-tool-boundary-v1.yaml
    - contracts/aprender/binding.yaml
    - CLAUDE.md
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-CONTEXT.md
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "Owner redeploy-and-measure (Task 2): redeploy the hardened bootstrap with the same artifact and size the download deadline from cold samples"
  - "Owner restore-10240-tier: the 3,008 MB tier was a forced downgrade; decide-tool-boundary-v1 7.0.0 re-derives every bound for 10,240 MB (8 texts / 800 tokens)"
  - "Owner keep-800-apply-10739: keep 800 tokens / 8 texts on the acceptance rule (every cold sample < 30 s) although the per-term-max rule gives 648; DOWNLOAD_DEADLINE 10739 ms; the stop-guard's reference set is the 10,240 MB tier's own downloads"
  - "Contract bumped per pv diff twice: 8.0.0 (classify_token_budget invariant, priced vs measured) before the re-measure, 9.0.0 (accepted_region_cold invariant, the live pass) after it"

patterns-established:
  - "Derived-deadline constants carry their sample ids and arithmetic in the doc and a test that recomputes them from the contract"
  - "A budget accepted on the whole-request rule while a per-term derivation disagrees is written down as such in the contract and tracked as an open deferred item (D-ITEM-08-30-B)"

requirements-completed: [D-18]

coverage:
  - id: D1
    description: "Offline positive dry run on HEAD accepts the deployed artifact 24a44d7e with zero AWS calls"
    requirement: "D-18"
    verification:
      - kind: integration
        ref: "lockf ... env DRY_RUN=1 just laya-deploy-selftest (MAIN checkout): DRY-RUN OK, bare DEPLOY SELFTEST OK, AWS CALLS: 0"
        status: pass
    human_judgment: false
  - id: D2
    description: "DOWNLOAD_DEADLINE 10739 ms derived from the 10,240 MB samples, and download_budget(remaining, reserve) wired to the Lambda invocation deadline"
    requirement: "D-18"
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#download_deadline_is_derived_from_the_contract_and_the_samples"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#download_budget_is_the_smaller_of_the_deadline_and_what_the_invocation_leaves"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#remaining_before_counts_down_to_zero_and_is_unbounded_without_a_deadline"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#zero_budget_download_is_abandoned_as_a_deadline"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/s3.rs#tests::deadline_abandons_stalled_parts"
        status: pass
      - kind: other
        ref: "cargo mutants -F 'download_budget|remaining_before' (2/2 caught) + 7-row hand mutation table (7/7 caught)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Redeployed b615d8244 at 10,240 MB; 4 more proven-cold samples (2 per shape) all under 30 s; served binary proven to be HEAD's"
    requirement: "D-18"
    verification:
      - kind: e2e
        ref: "lockf ... just laya-deploy-verify models/decide/laya-stance-64.apr aprender-mcp-decide ze-kasher-dev 2 -> DEPLOY VERIFY OK (4 cold samples, both shapes, < 30000 ms, CloudWatch cold evidence)"
        status: pass
      - kind: e2e
        ref: "CloudWatch: every cold environment logged `decide.load download_budget_ms=10726|10727` (exists only since b615d8244) and a platform.report with initDurationMs"
        status: pass
    human_judgment: false
  - id: D4
    description: "accepted_region_cold implemented at 10,240 MB; contract-audit-phase8 green without PHASE8_LIVE_EXEMPT; pv validate clean"
    requirement: "D-18"
    verification:
      - kind: other
        ref: "env -u PHASE8_LIVE_EXEMPT make contract-audit-phase8 -> 4 contract(s) audited, no binding finding"
        status: pass
      - kind: other
        ref: "pv validate contracts/decide-tool-boundary-v1.yaml -> 0 error(s), 0 warning(s)"
        status: pass
    human_judgment: false
  - id: D5
    description: "CLAUDE.md decision row and the lambda README state only measured 10,240 MB facts"
    requirement: "D-18"
    verification:
      - kind: other
        ref: "cargo test -p aprender-core --test readme_contract (15 passed)"
        status: pass
    human_judgment: true
    rationale: "readme_contract checks cited paths, not whether the prose matches the evidence; a reader should confirm the numbers against 08-LIVE-REDEPLOY-EVIDENCE.json `live`"

duration: 3h50m (across five owner checkpoints; this continuation 28 min)
completed: 2026-09-28
status: complete
---

# Phase 8 Plan 30: Live Redeploy and Measured Download Deadline Summary

**aprender-mcp-decide now runs live at 10,240 MB with 8 texts and 800 tokens per call. Its S3 download deadline is 10,739 ms, derived from measured cold terms, and the handler also respects the Lambda invocation deadline. Eight proven-cold maximal samples took 22.6-27.0 s against the 30 s cap.**

## Performance

- **Duration:** about 3h50m of wall time across the plan's five owner checkpoints. This continuation took 28 min, 2026-09-29T02:27Z to 02:55Z.
- **Started:** 2026-09-28T22:58Z (Task 1 dry run)
- **Completed:** 2026-09-29T02:55Z
- **Tasks:** 3 of 3
- **Files modified:** 17 across the plan, 11 of them in this continuation
- **AWS writes, whole plan:** 2 accepted deploys (attempts 3 and 4), 8 DECIDE_COLD_BUMP env updates and 0 containments. pmcp.run refused 2 deploy attempts before any AWS change. Every other AWS call was read-only.

## Decisions (owner)

| # | Decision id | Date | Effect |
|---|---|---|---|
| 1 | `redeploy-and-measure` | 2026-09-28 | Redeploy the hardened bootstrap (same artifact) and size the deadline from cold samples |
| 2 | `restore-10240-tier` | 2026-09-28 | Tier back to 10,240 MB. Contract 7.0.0 re-derived to 8 texts / 800 tokens. No fallback to 8000 or 3008 |
| 3 | `raise-pmcp-run-cap` / `retry-after-cap-confirmed` | 2026-09-28 | pmcp.run's MemorySize cap refused 2 attempts. The platform fix then deployed and attempt 3 was accepted |
| 4 | `keep-800-apply-10739` | 2026-09-28 | Keep 800/8 on the acceptance rule (the per-term rule gives 648). DOWNLOAD_DEADLINE 10739. The stop-guard reference set is corrected to this tier |

All four are recorded in 08-LIVE-REDEPLOY-EVIDENCE.json and under D-18 in 08-CONTEXT.md.

## Every cold sample at 10,240 MB (Graviton2, identity 24a44d7e, each proven cold)

| Attempt / # | Shape | Client ms | Lambda ms | load | download | sha | build (+probe replay) | classify | gateway | budget |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 / 1 | CONC 2x/800 | 27015 | 24264 | 15356 | 8954 | 3484 | 2824 | 8908 | 2751 | (25 s const) |
| 3 / 2 | DIST 8x/800 | 22922 | 22134 | 15546 | 9065 | 3484 | 2904 | 6588 | 788 | (25 s const) |
| 3 / 3 | CONC | 25076 | 24279 | 15368 | 8927 | 3498 | 2852 | 8911 | 797 | (25 s const) |
| 3 / 4 | DIST | 22647 | 22023 | 15402 | 8986 | 3484 | 2837 | 6621 | 624 | (25 s const) |
| 4 / 1 | CONC | 25092 | 24201 | 15433 | 9052 | 3483 | 2808 | 8768 | 891 | 10727 |
| 4 / 2 | DIST | 22855 | 22181 | 15543 | 9110 | 3483 | 2859 | 6638 | 674 | 10726 |
| 4 / 3 | CONC | 25305 | 24380 | 15533 | 9072 | 3484 | 2885 | 8847 | 925 | 10726 |
| 4 / 4 | DIST | 23116 | 22295 | 15684 | 9139 | 3522 | 2928 | 6611 | 821 | 10726 |

- **Cold wall:** 22,647 to 27,015 ms. The worst sample (attempt 3 #1) finished 2,985 ms under the cap.
- **Attempt 4 alone:** 22,855 to 25,305 ms.
- **Warm calls:** 866 to 980 ms over 10 calls. p50 was 914 ms in attempt 3 and 915 ms in attempt 4.
- **Memory:** Max Memory Used was 2483-2485 MB of 10240.
- **Init:** 61.6 to 66.5 ms.
- **Deadline check:** no download was cut. The largest download, 9139 ms, finished 1587 ms inside its budget. The largest post-download cost was 15,216 ms, within the 15,261 ms the reserve was derived from.

**Derivation (applied in s3.rs):**
- R = max(sha + build + classify) over the attempt-3 samples + margin_ms = 15261 (sample 3) + 4000 = 19261.
- DOWNLOAD_DEADLINE = min(25000, 30000 - 19261) = **10739 ms**.
- The handler's budget is min(10739, remaining - 19261). Live it read 10726-10727, which shows the invocation deadline was read.

**The served binary is HEAD's.** Three facts show it:
- CodeSha256 changed at the deploy (4J509c1g…).
- The compile log names only aprender-mcp-decide-lambda.
- Every cold environment logged `decide.load download_budget_ms=`, a line that exists only since b615d8244.

## Accomplishments

- **Task 1 (tracer):** the offline positive dry run on HEAD accepted the deployed artifact with zero AWS calls.
- **Task 3:**
  - The live function moved from the pre-round 3,008/8000 MB binary to the hardened bootstrap at 10,240 MB. It now serves the 08-28 surface: isError refusals and a tier-derived truncation promise.
  - V4-b is closed with evidence. The deadline is derived from measured terms, and the handler reads the invocation deadline.
  - decide-tool-boundary-v1 went 7.0.0 → 8.0.0 → 9.0.0.
  - accepted_region_cold is `implemented` again, backed by a FINAL 10,240 MB record.
  - `make contract-audit-phase8` passes without the exemption.
- CLAUDE.md and the lambda README now carry only these measured facts.

## Task Commits

1. **Task 1: offline dry run accepts the deployed artifact** - `defe91e09` (test)
2. **Task 2: owner decision redeploy-and-measure** - `6f7d9569c` (docs)
3. **Task 3a: contract re-priced for 10,240 MB (7.0.0)** - `26b1ebf44` (feat)
4. **Task 3b: restore-10240-tier recorded in D-18** - `98f4187fe` (docs)
5. **Task 3c: first refused deploy (pmcp.run cap 3008)** - `5fa0e7df9` (docs)
6. **Task 3d: second refused deploy** - `007edcb6f` (docs)
7. **Task 3e: accepted 10,240 MB deploy + 4 cold samples** - `6291fb116` (docs)
8. **Task 3f: keep-800-apply-10739 applied (deadline, budget, contract 8.0.0)** - `b615d8244` (feat)
9. **Task 3g: re-measured deploy, accepted_region_cold implemented (contract 9.0.0)** - `f76a8b2fc` (docs)
10. **Task 3h: CLAUDE.md row + README measured facts** - `8473ea04d` (docs)

**Plan metadata:** the final docs commit (SUMMARY, STATE, ROADMAP)

## Files Created/Modified

- `crates/aprender-mcp-decide-lambda/src/s3.rs`: adds DOWNLOAD_DEADLINE 10739 ms and POST_DOWNLOAD_RESERVE 19261 ms. The doc gives the sample-by-sample derivation and the tier reference-set note. The deadline test is updated.
- `crates/aprender-mcp-decide-lambda/src/lib.rs`: adds `download_budget`, `remaining_before` and `resolve_model_within`.
- `crates/aprender-mcp-decide-lambda/src/main.rs`: reads the Lambda context deadline, passes the budget into the load, and logs `download_budget_ms`.
- `crates/aprender-mcp-decide-lambda/src/tests.rs`: adds tests for the derivation-vs-contract check, the budget arithmetic (including remaining < reserve), remaining_before, and the zero-budget `s3_deadline`.
- `contracts/decide-tool-boundary-v1.yaml`: 9.0.0. Records priced vs measured (b), the 648-vs-800 statement, and the accepted_region_cold live pass.
- `contracts/aprender/binding.yaml`: accepted_region_cold `implemented`.
- `08-LIVE-REDEPLOY-EVIDENCE.json`: adds `deadline_decision`, `live_attempts[3]` and the aggregate `live` block (8 samples, final: true).
- `08-CONTEXT.md`: dated D-18 line for keep-800-apply-10739.
- `deferred-items.md`: D-ITEM-08-17-E resolved (tier superseded, measured); D-ITEM-08-30-B opened.
- `CLAUDE.md` and `crates/aprender-mcp-decide-lambda/README.md`: measured 10,240 MB facts.
- `.pmcp/deploy.toml.template`: the deadline comment.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] The plan's Task 3 verify reads `e["live"]["cold_samples"]`, which no earlier attempt wrote**
- **Found during:** Task 3, recording the re-measure.
- **Issue:** the samples lived under `live_attempts[n].cold_samples`, so the plan's own verify would have raised KeyError.
- **Fix:** added an aggregate `live` block holding all 8 samples with the fields the verify reads, plus the derived deadline and its sample ids.
- **Files modified:** 08-LIVE-REDEPLOY-EVIDENCE.json
- **Verification:** the plan's evidence script prints `EVIDENCE OK redeploy-and-measure 8`.
- **Committed in:** f76a8b2fc

**2. [Rule 2 - Missing critical] A second contract bump (9.0.0) for the live pass**
- **Found during:** Task 3, flipping accepted_region_cold.
- **Issue:** the accepted_region_cold invariant said "until that record passes, the binding row is partial". Leaving it would contradict the flipped binding.
- **Fix:** restated the invariant, FALSIFY-DECIDE-TOOL-009 and the qa_gate text on the live record. `pv diff` classified it MAJOR, hence 9.0.0 (same precedent as v3.0.0).
- **Committed in:** f76a8b2fc

**Other scope differences.**
- The plan as written targeted the 3,008 MB tier. The owner's decisions moved it to 10,240 MB (decisions table), so the plan's "3,008 MB amendment" is superseded by the dated D-18 lines.
- The owner's instruction also called for the CLAUDE.md/README edits and the D-ITEM bookkeeping.

**Total deviations:** 2 auto-fixed (1 blocking, 1 missing critical). **Impact:** both keep the records consistent with the measurement. No scope creep beyond the owner's instructions.

## Issues Encountered

- **pmcp.run refusals (attempts 1-2).** pmcp.run capped MemorySize at 3008 and refused twice, before any AWS change. The platform fix then let attempt 3 through.
- **Two priced cold terms exceed their prices at 10,240 MB.**
  - Build including the probe replay ran 404-524 ms over on every sample.
  - Gateway overhead ran 1,855 ms over once.
  - The budget is accepted on the whole-request rule only. D-ITEM-08-30-B tracks it.
- **CloudWatch uses the JSON log format.** The earlier REPORT-text parser found nothing, so a platform.report-aware reader (scratch, not committed) extracted durations.

## Known Stubs

None.

## Threat Flags

None. The handler now reads the Lambda context deadline, which is platform data and not request data. The new log line carries only a millisecond count.

## User Setup Required

None. AWS and pmcp.run credentials were already in place and verified read-only before the deploy.

## Next Phase Readiness

- **Live state.** The endpoint is live with auth off (D-ITEM-08-18-A unchanged) and reserved concurrency unset. It serves b615d8244 at 10,240 MB.
- **Next plans.** Ready for 08-31 (claim honesty, D-14 publication) and 08-32 (gap regression, CI).
- **Open.**
  - D-ITEM-08-30-A: the /tmp idea.
  - D-ITEM-08-30-B: the per-term vs whole-request acceptance.
  - D-ITEM-08-17-C: software sha256, 3.5 s per cold load, the largest remaining lever.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-28*

## Self-Check: PASSED

- 5/5 key files present; 10/10 plan commits resolve (defe91e09 .. 8473ea04d); commits measured from the ledger: 10 (e0be0262b..8473ea04d)
- Re-run gates: plan Task 3 evidence script EVIDENCE OK (8 samples); lambda lib tests 47 passed; clippy -D warnings rc 0; fmt --check rc 0; pv validate 0 errors; contract-audit-phase8 without PHASE8_LIVE_EXEMPT rc 0; readme_contract 15 passed
