---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 28
subsystem: mcp-decide-tool-boundary
tags: [pmcp, mcp, classify, tool-description, isError, refusals, admission, contracts, pv, e2e-stdio]
status: complete
gap_closure: true

requires:
  - phase: 08-23
    provides: "served_fields / untrusted_input_bounds tables and their sweeps; parse_args count-first"
  - phase: 08-24
    provides: "Lambda crate without a crate-wide disallowed_methods allow; lambda_request_rows_are_swept"
  - phase: 08-25
    provides: "probe.rs LABELS_SEGMENT / labels_in_order parsing `The labels, in this order: [`"
provides:
  - "classify_admission restated to the measured transport behaviour (pmcp 2.19.3 serial dispatch) and pinned over live stdio by pipelined_calls_are_serialized_not_refused (Task 1)"
  - "Owner decision `A-derive B-iserror` recorded as the dated D-09 amendment (gap round 08-28)"
  - "truncation_sentence(max_len, limits): the served description promises truncation only when a full-window row fits classify_max_total_tokens (A-derive)"
  - "Every bound refusal is pmcp::Error::tool_rejected -> on the wire a successful tools/call result with isError true (B-iserror); model/internal failures stay JSON-RPC -32603"
  - "refusal_names_bound satisfiable: quantifies over distinctive texts of at least refusal_echo_min_chars (12) (A4-7)"
  - "decide-tool-boundary-v1 6.0.0 (pv diff: major): new constant refusal_echo_min_chars, served_fields description.truncation row"
affects: [08-30-redeploy-decision, 08-verify-work, aprender-mcp-decide, aprender-mcp-decide-lambda, decide-tool-boundary-v1]

actuals:
  tokens: 11567      # chars/4 over the realized diff (git diff b4fdd08b1..HEAD | wc -c = 46269)
  tasks: 3
  commits: 3         # MEASURED: git rev-list --count b4fdd08b1..HEAD (before this SUMMARY's commit)
plan_head_before: b4fdd08b1285e3601a922022ee9daee070baa869

tech-stack:
  added: []
  patterns:
    - "Bound refusal = pmcp::Error::tool_rejected(message, None): caller-fixable errors are in-band tool results, not protocol errors"
    - "Served description sentences that depend on the tier are DERIVED from the artifact and the limits, and pinned by a test covering both branches and the exactly-at boundary"
    - "A no-echo formula quantifies over a stated minimum length of distinctive text, and the test's secret sits exactly on that minimum"

key-files:
  created: []
  modified:
    - crates/aprender-mcp-decide/src/lib.rs
    - crates/aprender-mcp-decide/src/tests.rs
    - crates/aprender-mcp-decide/tests/e2e_stdio.rs
    - crates/aprender-mcp-decide/README.md
    - crates/aprender-mcp-decide-lambda/src/tests.rs
    - contracts/decide-tool-boundary-v1.yaml
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-CONTEXT.md

key-decisions:
  - "Owner chose A-derive B-iserror (verbatim ids) at the Task 2 checkpoint; recorded as the D-09 amendment 'gap round 08-28'; takes effect at the next deploy (plan 08-30)"
  - "Bound refusals carry no `details` (no structuredContent): the message is the whole refusal, so no second copy can drift or carry caller text"
  - "check_served_task_fits' over-budget branch is a bound refusal (tool_rejected) per the owner's list, but it is BUILD-time and never reaches a tools/call; its prepare-failure branch stays internal"
  - "refusal_echo_min_chars = 12 is a machine-readable contract constant the unit test reads, not prose"

patterns-established:
  - "isError wire shape asserted on BOTH shipped transports: live stdio (e2e_stdio) and the Lambda crate's streamable-HTTP loopback (through the probe's own classify_payload branch)"

requirements-completed: [D-09, D-10, D-12]

coverage:
  - id: D1
    description: "classify_admission restated to pmcp 2.19.3's serial dispatch; pipelined stdio calls all served in order"
    requirement: "D-10"
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-decide/tests/e2e_stdio.rs#pipelined_calls_are_serialized_not_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#admission_refuses_over_pending"
        status: pass
    human_judgment: false
  - id: D2
    description: "Served description's truncation sentence derived from the artifact window vs the tier budget (A-derive)"
    requirement: "D-12"
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#description_truncation_sentence_matches_tier"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#every_served_field_has_a_bound_source"
        status: pass
    human_judgment: false
  - id: D3
    description: "Bound refusals arrive as isError tool results on stdio and HTTP; model failures stay -32603 (B-iserror)"
    requirement: "D-09"
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-decide/tests/e2e_stdio.rs#bound_refusals_are_iserror_results_over_live_stdio"
        status: pass
      - kind: integration
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#loopback_bound_refusal_is_an_iserror_result"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#error_taxonomy_budget_rejected_model_internal"
        status: pass
    human_judgment: false
  - id: D4
    description: "refusal_names_bound satisfiable with a 12-character minimum; distinctive-text test at that length (A4-7)"
    requirement: "D-09"
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#every_refusal_names_key_without_text"
        status: pass
    human_judgment: false
  - id: D5
    description: "The changed served surface reaches the live Lambda only through plan 08-30's redeploy decision"
    verification: []
    human_judgment: true
    rationale: "Deploy is an owner decision (plan 08-30); the live function (D-ITEM-08-18-A) still serves the old description and -32603 refusals"

duration: ~22min (Task 1 in the prior executor session; Tasks 2-3 22:27-22:49Z)
completed: 2026-09-28
---

# Phase 8 Plan 28: Classify Tool Claim Honesty Summary

**The classify tool now tells the truth about itself: a tier-derived truncation sentence (A-derive), bound refusals as MCP `isError` tool results instead of JSON-RPC -32603 (B-iserror, measured on stdio and HTTP), a satisfiable no-echo formula with a 12-character minimum, and an admission claim that matches pmcp's serial dispatch.**

## Owner decision (verbatim)

**`A-derive B-iserror`** — recorded in 08-CONTEXT.md under D-09 as
"— **Amended 2026-09-28 (user, gap round 08-28):** truncation sentence A-derive; refusal wire shape B-iserror; takes effect at the next deploy (plan 08-30)."

Neither change reaches the live Lambda (RUNNING, D-ITEM-08-18-A) until plan 08-30's redeploy decision. Nothing was deployed.

## Performance

- **Duration:** ~22 min for Tasks 2-3 (this continuation); Task 1 ran in the prior executor session (commit 890b769e9 at 22:21Z)
- **Started (continuation):** 2026-09-28T22:27:05Z
- **Completed:** 2026-09-28T22:49Z
- **Tasks:** 3 (1 tracer, 1 checkpoint:decision, 1 auto)
- **Files modified:** 7

## Accomplishments

- **Task 1 (tracer):** `pipelined_calls_are_serialized_not_refused` writes `classify_max_pending + 1` = 5 frames over live stdio before reading and gets 5 classified replies in request order; `classify_admission` restated (bound applies to concurrent callers of `ClassifyService::call`; `transport_dispatch` invariant: pmcp 2.19.3 serialises stdio via one worker and HTTP via `Mutex<Server>`); contract 4.0.0 -> 5.0.0.
- **Task 2 (decision):** owner chose `A-derive B-iserror`; dated D-09 amendment committed.
- **Task 3, A-derive:** `truncation_sentence(max_len, limits)` — the old sentence when `max_len <= classify_max_total_tokens`, otherwise "A text whose built row would exceed the {budget}-token request budget is refused (classify_max_total_tokens) rather than truncated, because the model's own window ({max_len} tokens) is larger than that budget: send a shorter excerpt of a long document instead." On the tiny fixture at the contracted tier the served `tools/list` description is **byte-identical** to before (cmp of live output); at 3 008 MB with Laya-en (512 > 120) the refusal sentence is served; at 10 240 MB (1024) truncation returns with no edit. The `The labels, in this order: [` segment is unchanged.
- **Task 3, B-iserror:** `refusal()`, `Busy::into_pmcp`, `ClassifyFailure::TokenBudget` and `check_served_task_fits`' over-budget branch are `pmcp::Error::tool_rejected(message, None)`; model/label/join failures stay `pmcp::Error::internal`.
- **Task 3, A4-7:** `refusal_names_bound` quantifies over distinctive caller texts of at least `refusal_echo_min_chars` (12, new contract constant); `every_refusal_names_key_without_text` uses a 12-char secret read against that constant.
- **Contract:** decide-tool-boundary-v1 5.0.0 -> **6.0.0** (`pv diff` suggested major: codomains of four equations, refusal_names_bound formula/domain/codomain/invariants, truncation_flag invariants); `pv validate`: 0 errors, 0 warnings.

## Measured wire shape (live stdio, raw JSON-RPC lines, tiny fixture)

Before (HEAD 890b769e9 binary):

```
count_over:   {"jsonrpc":"2.0","id":2,"error":{"code":-32603,"message":"Validation error: classify: 3 texts exceeds classify_max_texts 2 (contracts/decide-tool-boundary-v1.yaml); split the batch"}}
token_budget: {"jsonrpc":"2.0","id":6,"error":{"code":-32603,"message":"Validation error: classify: built rows total 128 tokens (per text [64, 64]), over classify_max_total_tokens 120 (contracts/decide-tool-boundary-v1.yaml); send fewer or shorter texts"}}
```

After (this plan's binary):

```
count_over:   {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"classify: 3 texts exceeds classify_max_texts 2 (contracts/decide-tool-boundary-v1.yaml); split the batch"}],"isError":true}}
count_zero:   {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"classify: 0 texts is fewer than classify_min_texts 1 (contracts/decide-tool-boundary-v1.yaml); send at least one text"}],"isError":true}}
shape_string: {"jsonrpc":"2.0","id":4,"result":{"content":[{"type":"text","text":"classify: the arguments must be exactly {\"texts\": [string, ...]} and nothing else (classify_count_bound precondition: deny_unknown_fields, contracts/decide-tool-boundary-v1.yaml); detail withheld so no caller text is echoed"}],"isError":true}}
unknown_key:  (same text as shape_string, "isError":true)
token_budget: {"jsonrpc":"2.0","id":6,"result":{"content":[{"type":"text","text":"classify: built rows total 128 tokens (per text [64, 64]), over classify_max_total_tokens 120 (contracts/decide-tool-boundary-v1.yaml); send fewer or shorter texts"}],"isError":true}}
bytes_over:   {"jsonrpc":"2.0","id":7,"result":{"content":[{"type":"text","text":"classify: texts[0] is 16385 UTF-8 bytes, over classify_max_text_bytes 16384 (contracts/decide-tool-boundary-v1.yaml); send a shorter document"}],"isError":true}}
ok:           {"jsonrpc":"2.0","id":8,"result":{"content":[{"type":"text","text":"{\"model\":...}"}]}}   (classify still served afterwards)
```

pmcp 2.19.3 behaved exactly as read in `src/server/mod.rs:2458` (no `error` member, no `structuredContent`, the `Validation error: ` prefix gone). The HTTP half was measured too: `loopback_bound_refusal_is_an_iserror_result` drives the Lambda crate's stateless streamable-HTTP loopback through the probe, whose `classify_payload` returns `tool error: ...` (the isError branch), where the pre-change server produced `JSON-RPC error: {"code":-32603,...}` (mutation M2-lambda below).

## Mutation table

Every row was applied, run, and restored (restore checked with `cmp`).

| # | Mutation | Test run | Result |
|---|----------|----------|--------|
| T1-a | `CONTRACTED.max_pending: 0` (Task 1, prior session) | e2e `pipelined_calls_are_serialized_not_refused` | RED: "pipelined call 0 of 1 was refused (names classify_max_pending: true)" |
| T1-b | `max_pending: 1` over stdio (Task 1) | same | 5/5 classified in order, 3/3 runs (serial dispatch measured) |
| T1-c | Control: `max_pending: 1`, 5 concurrent `ClassifyService::call` futures (Task 1) | library door | 1 ok, 4 refused, 3/3 runs |
| M4 | `max_pending: 0` again, AFTER B-iserror changed Busy to an isError result | e2e `pipelined_calls_are_serialized_not_refused` | RED: "pipelined call 0 of 1 was refused (names classify_max_pending: true): classify: 0 requests are already admitted..." — the test reads both refusal shapes |
| M1 | `truncation_sentence` condition forced `true` (old sentence always) | `description_truncation_sentence_matches_tier` | RED: "a full-window row cannot fit 63 tokens, so truncation is unreachable" |
| M1b | condition forced `false` | same | RED: "a full-window row fits: truncation promised" |
| M1c | `<=` -> `<` (off by one at window == budget) | same | first run GREEN (mutant survived) -> exactly-at case added -> RED: "a 64-token row fits a 64-token budget exactly" |
| M1d | description ignores `limits` (always CONTRACTED) | same | RED at the 63-token branch |
| M5 | `description.truncation` row deleted from served_fields | `every_served_field_has_a_bound_source` | RED: row count 3 != 4 |
| M2-e2e | `refusal()` back to `pmcp::Error::validation` | e2e `bound_refusals_are_iserror_results_over_live_stdio` | RED: "a bound refusal must not be a JSON-RPC error: {...\"code\":-32603...}" |
| M2-lambda | same | `loopback_bound_refusal_is_an_iserror_result` (HTTP) | RED: detail was "JSON-RPC error: {\"code\":-32603,...}" |
| M2-unit | same | `every_refusal_names_key_without_text` | RED: "expected a bound refusal (tool_rejected, no details), got Validation(...)" |
| M2b | only `TokenBudget` back to validation | e2e `bound_refusals_are_iserror_results_over_live_stdio` | RED at the token-budget case (id 7, -32603) |
| M2c | only `Busy::into_pmcp` back to validation | `error_taxonomy_budget_rejected_model_internal` | RED: got Validation("...classify_max_pending...") |
| M2d | refusal carries `details` (structuredContent) | e2e `bound_refusals_are_iserror_results_over_live_stdio` | RED: "no structured copy" |
| M3 | test secret -> `"a"` | `every_refusal_names_key_without_text` | RED at the length gate (1 != 12) |
| M3b | secret `"a"` with the length gate disabled | same | RED: "classify_min_texts echoed the text" — the old any-non-empty-text formula is unsatisfiable (A4-7) |
| M3c | contract `refusal_echo_min_chars: 11` | same | RED: length gate (12 != 11) — the test reads the contract, not a literal |

## Task Commits

1. **Task 1 (tracer): admission claim measured over live stdio and restated** - `890b769e9` (test)
2. **Task 2: owner decision recorded as the D-09 amendment** - `5e91d2b10` (docs)
3. **Task 3: A-derive + B-iserror + A4-7 formula + contract 6.0.0** - `d3de3e741` (feat)

## Files Created/Modified

- `crates/aprender-mcp-decide/src/lib.rs` - `truncation_sentence`, private `tool_description_for`; `refusal()` / `Busy` / `TokenBudget` / served-task fit -> `tool_rejected`; docs
- `crates/aprender-mcp-decide/src/tests.rs` - `rejection_message` (was `validation_message`), `error_taxonomy_budget_rejected_model_internal` (renamed), 12-char secret, `description.truncation` served-field case, new `description_truncation_sentence_matches_tier`
- `crates/aprender-mcp-decide/tests/e2e_stdio.rs` - strict isError `refusal()`, new `bound_refusals_are_iserror_results_over_live_stdio`, pipelined test reads both refusal shapes
- `crates/aprender-mcp-decide-lambda/src/tests.rs` - new `loopback_bound_refusal_is_an_iserror_result`
- `crates/aprender-mcp-decide/README.md` - truncation paragraph (both branches), isError refusal paragraph with the measured line
- `contracts/decide-tool-boundary-v1.yaml` - 6.0.0: v6 paragraph, consequences text, `refusal_echo_min_chars`, `description.truncation` row, codomains, refusal_names_bound rewrite, safety obligation, FALSIFY-001/005/006 and qa_gate
- `08-CONTEXT.md` - D-09 amendment

## Decisions Made

- Owner: `A-derive B-iserror` (see above).
- No `details` on bound refusals (keeps one copy of the message; pinned by M2d).
- `check_served_task_fits` follows the owner's list (tool_rejected) but is documented as build-time only.
- `probe.rs` NOT touched: `classify_payload` (probe.rs:297-301) already returns `tool error: <text>` for `isError: true`, and `Rpc::call` still handles a JSON-RPC error for internal failures — verified by the new Lambda loopback test.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing verification] HTTP wire shape measured in the Lambda crate**
- **Found during:** Task 3 (B-iserror)
- **Issue:** The plan's live assertion was stdio only; the owner required the isError claim be measured, and the Lambda (HTTP) transport is the one deployed.
- **Fix:** Added `tests::loopback_bound_refusal_is_an_iserror_result` to `crates/aprender-mcp-decide-lambda/src/tests.rs` (not in the plan's files_modified), named on FALSIFY-DECIDE-TOOL-006's test line.
- **Verification:** passes; RED under M2-lambda.
- **Committed in:** d3de3e741

**2. [Rule 1 - Bug] Surviving off-by-one mutant in the description test**
- **Found during:** Task 3 mutation proof
- **Issue:** `<=` -> `<` survived: window == budget was untested.
- **Fix:** exactly-at case (budget = tiny window 64 -> truncation promised).
- **Verification:** M1c RED after the fix.
- **Committed in:** d3de3e741

**3. [Rule 2 - Claim honesty] `served_fields` row and a contract constant**
- **Found during:** Task 3
- **Issue:** The derived truncation sentence is a new served field, and the plan's "stated minimum length" was prose only.
- **Fix:** `served_fields` gains `description.truncation` (the class-A sweep now dispatches it); `constants.refusal_echo_min_chars: 12`, which the unit test reads.
- **Verification:** M5 and M3c RED.
- **Committed in:** d3de3e741

**4. [Plan text] Tiny-fixture branches reversed in the plan**
- The plan said contracted limits give the refusal sentence on the tiny fixture; with max_len 64 <= 120 they give the TRUNCATION sentence (as the owner's note said). The refusal branch is reached at budget 63; Laya-en's 512 window is covered on the pure helper at 120 (refused) and 1024 (truncates).

**5. [Process] Task 2 has its own commit**
- The D-09 amendment was committed as Task 2's record (5e91d2b10) rather than inside Task 3's commit.

---

**Total deviations:** 3 auto-fixed (1 bug, 2 missing verification/claim honesty) + 2 process/plan-text notes.
**Impact on plan:** All within scope; no served surface beyond the owner's two decisions changed.

## Issues Encountered

None. pmcp 2.19.3 behaved on the wire exactly as its source reads, so no checkpoint was needed.

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The served surface changed (description at tiers where max_len > budget, and the refusal shape). The live `aprender-mcp-decide` Lambda still serves the OLD description and -32603 refusals until plan 08-30's redeploy decision.
- Ready for 08-30.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-28*

## Self-Check: PASSED

- Commits found: 890b769e9, 5e91d2b10, d3de3e741, dfc682dc6 (SUMMARY)
- Files found: lib.rs, tests.rs, e2e_stdio.rs, lambda tests.rs, decide-tool-boundary-v1.yaml, this SUMMARY
- Re-run at close: `cargo test -p aprender-mcp-decide` (lib 32 + e2e 4, all ok), `cargo test -p aprender-mcp-decide-lambda --lib` (43 ok), clippy `-D warnings` clean on both crates, no module-wide allow in probe.rs, labels segment present, D-09 amendment present, `pv validate` 0 errors 0 warnings, `cargo fmt --check` clean
