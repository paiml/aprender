---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 25
subsystem: deploy-evidence
tags: [aprender-mcp-decide-lambda, probe, deploy-identity, mutation-testing, pmcp-run]
status: complete

requires:
  - phase: 08-12
    provides: the identity probe, the maximal-request builder and the loopback test harness this plan hardens
provides:
  - ProbeReport.response_labels_match, with ok() the AND of all four identity checks
  - labels_in_order reads the exact `The labels, in this order: [...]` segment as a list
  - build_maximal_request decides fit with aprender_mcp_decide::check_token_budget (one budget function)
  - examples/probe.rs rejects non-UTF-8 argv with usage and exit 2 (previously a panic, exit 101)
  - deploy.toml.template comments now describe the identity tell and the shared crates root correctly
affects: [08-24, 08-28, laya-deploy, laya-deploy-verify]

actuals:
  tokens: 4458        # chars/4 over the realized diff (17,831 chars); the 3 touched files total 55,649 chars (13,912)
  tasks: 3
  commits: 3
plan_head_before: 99e72f1f958071f75e3576f4321503ce825b09b7

tech-stack:
  added: []
  patterns:
    - "A probe verdict ANDs every field it reports as deploy evidence; a unit test flips each field alone"
    - "Parse the one literal description segment the server promises; do not substring-walk prose"
    - "The probe sizes its requests with the server's own refusal function, and its error carries the server's refusal verbatim"

key-files:
  created: []
  modified:
    - crates/aprender-mcp-decide-lambda/src/probe.rs
    - crates/aprender-mcp-decide-lambda/examples/probe.rs
    - crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template

key-decisions:
  - "MaximalError::OverBudget now carries the server's ClassifyFailure (`refusal`) instead of its own total/budget pair, so the probe's refusal is the server's text word for word (R6)"
  - "Non-UTF-8 argv is reported by position and never quoted, because the offending argument may be the bearer token"
  - "The probe example's identity JSON line also prints response_labels_match. This adds a key; the recipes gate on the exit code, which ok() already drives"

patterns-established:
  - "ok_requires_every_check: build a report with every check true, flip each one alone, and assert that ok() is false"

requirements-completed: [D-11, D-18]

coverage:
  - id: D1
    description: "The identity probe fails when the endpoint answers with the right sha256 but the wrong label order (V12-a)"
    requirement: D-11
    verification:
      - kind: integration
        ref: "crates/aprender-mcp-decide-lambda/src/probe.rs#probe_fails_when_response_labels_differ"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/probe.rs#ok_requires_every_check"
        status: pass
    human_judgment: false
  - id: D2
    description: "The identity probe fails when the endpoint serves a sha256 other than the pin, even with the right labels"
    requirement: D-11
    verification:
      - kind: integration
        ref: "crates/aprender-mcp-decide-lambda/src/probe.rs#probe_fails_when_identity_differs"
        status: pass
    human_judgment: false
  - id: D3
    description: "The description check parses the exact labels segment as a list; question text and substrings (`none` inside `nonetheless`) cannot satisfy it (IN-03)"
    requirement: D-11
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/probe.rs#labels_segment_is_parsed_exactly"
        status: pass
    human_judgment: false
  - id: D4
    description: "build_maximal_request decides fit with the server's check_token_budget, and its refusal carries the server's text (R6)"
    requirement: D-18
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/probe.rs#maximal_request_uses_the_server_budget"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/probe.rs#maximal_distributed_refuses_a_count_the_budget_cannot_admit"
        status: pass
    human_judgment: false
  - id: D5
    description: "The probe example exits 2 with usage on a non-UTF-8 argument (the old build exits 101)"
    verification:
      - kind: other
        ref: "target/debug/examples/probe --apr $(printf 'x\\377y'); exit 2 (old build from 99e72f1f9: 101)"
        status: pass
    human_judgment: false
  - id: D6
    description: "deploy.toml.template comments name the POST identity probe as the tell and the shared crates root as the deploy root; only comment lines changed"
    verification:
      - kind: other
        ref: "grep -v '^#' old/new template, then cmp: identical (50 lines, 981 B); deploy_memory_is_the_contract_tier passes"
        status: pass
    human_judgment: true
    rationale: "Whether the prose is accurate and readable is an editorial judgment. The automated check only proves that no TOML value changed."

duration: 12min
completed: 2026-09-28
---

# Phase 8 Plan 25: Probe identity checks against the artifact Summary

**The deploy probe's ok() is now the AND of four checks against the local artifact: one `classify` tool, the exact labels segment of the description, the response's label list, and the sha256. A reordered label list now fails the probe; the old substring walk accepted it on the real tiny-fixture description. The probe also sizes requests with the server's own `check_token_budget`, and its example exits 2 on non-UTF-8 argv instead of panicking.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-09-28T19:21:06Z
- **Completed:** 2026-09-28T19:32:56Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- **V12-a closed.** `ProbeReport.response_labels_match` compares the `tools/call` response's `labels` with the labels derived from `--apr` (same length, same order). `ok()` now requires it along with the other three checks. Before this plan, an endpoint that returned the pinned sha256 with its labels in the wrong order passed.
- **IN-03 closed, and reproduced on real data first.** Against the served description of the tiny fixture, the old `labels_in_order` accepted `[billing, shipping, account]` for a model whose labels are `[shipping, billing, account]`. It found `billing` in the segment, then `shipping` and `account` in the criteria lines that follow it. The new parser reads only the literal `The labels, in this order: [` segment up to `]`, splits on `, ` and compares the result as a list. The doc comment says plan 08-28 keeps this segment.
- **R6 closed.** `build_maximal_request` decides fit with `aprender_mcp_decide::check_token_budget`. `MaximalError::OverBudget { shape, refusal }` carries the server's `ClassifyFailure`, so the builder's refusal contains the server's `classify_max_total_tokens` text word for word.
- **V12-c (probe half) closed.** `examples/probe.rs` iterates `std::env::args_os()`. A non-UTF-8 argument is a usage error: the usage text is printed and the process exits 2. The argument is reported by position and never quoted. The old build, rebuilt from 99e72f1f9, exits **101** on the same input.
- **IN-04 (template half) closed.** The template now says the tell is the identity probe over POST (`model.artifact_sha256` equal to the pin, with the labels in order), and that no GET reaches the bootstrap on pmcp.run (D-ITEM-08-17-A). It names the shared crates root, `--manifest-path crates` inside `_laya-crates-root-swap` (plan 08-10, D-ITEM-08-10-A), and step 3 of the flow names that root. The non-comment lines are byte-identical (`cmp`: 50 lines, 981 B both sides), and `deploy_memory_is_the_contract_tier` still passes.

## Mutation table (Task 3)

Each mutant was applied to a clean copy of `probe.rs`, `cargo test -p aprender-mcp-decide-lambda --lib probe::` was run, and the file was restored. The restore was checked with a byte comparison against the clean copy, and `git diff` was empty afterwards.

| Mutant | Test(s) turned RED | RED | Restored |
|--------|--------------------|-----|----------|
| M1 `ok()` without `response_labels_match` | `ok_requires_every_check` | yes: exit 101, 11 passed / 1 failed | yes (byte-identical) |
| M2 `ok()` without `description_has_labels_in_order` | `ok_requires_every_check` | yes: exit 101, 11 / 1 | yes |
| M3 `ok()` without `identity_matches` | `probe_fails_when_identity_differs`, `ok_requires_every_check` | yes: exit 101, 10 / 2 | yes |
| M4 `labels_in_order` back to the substring walk | `labels_segment_is_parsed_exactly`, `probe_fails_when_response_labels_differ` | yes: exit 101, 10 / 2 | yes |
| M5 `build_maximal_request` back to its own plain sum (the code at 99e72f1f9: own `total > budget` test, own error text) | `maximal_request_uses_the_server_budget` | yes: exit 101, 11 / 1 | yes |
| M5b (extra, equivalent) own plain sum wrapped in a hand-built `ClassifyFailure::TokenBudget` | none | **survives** (12 / 0) | yes |

Two limits of this table:

- **M1 and M2 are caught only by the unit test.** A real loopback cannot make the response labels disagree while the description agrees, or the other way round, because both come from the same artifact. `ok_requires_every_check` flips each field alone. The loopback test `probe_fails_when_response_labels_differ` proves the end-to-end path: the right sha with the wrong labels gives ok() false.
- **M5b is an equivalent mutant.** Below `usize::MAX` the probe's plain sum and the server's saturating fold agree on every input, so a mutant that recomputes the sum and wraps it in the server's own error type behaves identically. The shared function shows up only through where the refusal comes from, which is what M5 (the real pre-plan code) and `maximal_request_uses_the_server_budget` pin.

## Task Commits

1. **Task 1 (tracer): response labels in the verdict.** `4f03998ee` (feat). Tracer gate: interactive, `end-of-phase`, automated-only verify. It was re-run and passed, and no checkpoint was synthesized.
2. **Task 2: exact segment parse, shared budget, args_os, template comments.** `305ae8226` (feat)
3. **Task 3: identity test and mutation proof.** `6f82bdab8` (test)

## Files Created/Modified

- `crates/aprender-mcp-decide-lambda/src/probe.rs`: `response_labels_match`; `ok()` over four checks; `LABELS_SEGMENT` and the exact-segment `labels_in_order`; `build_maximal_request` using `check_token_budget`; `MaximalError::OverBudget { shape, refusal }`; five new tests. No `json!` was added, and every `json!` stays inside a function, so plan 08-24's item-level allows can scope them.
- `crates/aprender-mcp-decide-lambda/examples/probe.rs`: `args_os` parsing with a positional, unquoted usage refusal (exit 2); the identity line reports `response_labels_match`.
- `crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template`: comment lines only.

## Decisions Made

- `OverBudget` carries the server's `ClassifyFailure` instead of a probe-computed total/budget pair. With one budget function there is one refusal text. The existing refusal test now destructures `refusal: TokenBudget { total, limit, .. }`.
- The non-UTF-8 argument is named by position, never quoted: it can be the `--bearer` value, and T-08-07-04 says a token is never printed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Task 1's "description false too" assertion moved to Task 2**
- **Found during:** Task 1
- **Issue:** The plan's Task 1 test asserts `description_has_labels_in_order == false` for a reordered list. Against the real tiny-fixture description the old substring walk returned **true** for `[billing, shipping, account]`, because the criteria lines follow the segment. That is IN-03 itself, and the fix belongs to Task 2.
- **Fix:** Task 1 asserts `!response_labels_match` and `!ok()`. Task 2 added the description assertion back, where it was one of Task 2's three REDs before the parser rewrite.
- **Files modified:** crates/aprender-mcp-decide-lambda/src/probe.rs
- **Committed in:** 4f03998ee, 305ae8226

**2. [Rule 2 - Missing critical] The example's identity JSON reports `response_labels_match`**
- **Found during:** Task 2
- **Issue:** Without the new key, the evidence line from `laya-deploy` / `laya-deploy-verify` could fail on the label check (non-zero exit) while showing every check it prints as true (T-08-25-02).
- **Fix:** One additive key. No recipe parses the identity line's keys: `grep` found only `identity_matches`, and only in the maximal-sample reader.
- **Files modified:** crates/aprender-mcp-decide-lambda/examples/probe.rs
- **Committed in:** 305ae8226

**3. [Rule 2 - Missing critical] Added `ok_requires_every_check`**
- **Found during:** Task 1 (needed by Task 3)
- **Issue:** The loopback cannot isolate the response-labels check from the description check (see the M1/M2 limit above), so without a unit test the M1 and M2 mutants would survive.
- **Fix:** A unit test that flips each of the four fields alone.
- **Committed in:** 4f03998ee

---

**Total deviations:** 3 auto-fixed (1 blocking, 2 missing critical)
**Impact on plan:** All three are needed for the mutation proof or for honest evidence. No scope creep, and no file outside the plan's three was touched.

## Issues Encountered

- The rtk hook condenses `cargo test` output, so the plan's `grep '<test> ... ok'` verify lines see nothing through a plain `cargo test`. Every verify ran as `rtk proxy cargo test ...`. The hook also mangled `grep -v` / `wc -l` in the template check (it printed 0 lines). That check was re-run with `/usr/bin/grep`, and both sides hold 50 non-empty lines, so the `cmp` is not vacuous.
- zsh treats `"$B:c..."` as a history modifier, so the first `git show "$BASE:crates/..."` failed. It was re-run as `"${B}:..."`.

## Known Stubs

None.

## User Setup Required

None. No AWS call was made; the only network traffic was the in-process 127.0.0.1 loopback.

## Next Phase Readiness

- Plan 08-24 (wave 18) can now take `probe.rs`: every `json!` is inside a function (`Rpc::call`, `run_identity_probe`, `classify_params`, tests).
- Plan 08-28 must keep the `The labels, in this order: [...]` segment; `LABELS_SEGMENT` documents that.

## Self-Check: PASSED

- FOUND: crates/aprender-mcp-decide-lambda/src/probe.rs, crates/aprender-mcp-decide-lambda/examples/probe.rs, crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template
- FOUND commits: 4f03998ee, 305ae8226, 6f82bdab8 (`git rev-list --count 99e72f1f9..HEAD` = 3 before this SUMMARY)
- `cargo test -p aprender-mcp-decide-lambda --lib`: 35 passed, 0 failed. clippy `--all-targets --no-deps -D warnings` exit 0 (log shows `Checking aprender-mcp-decide-lambda`). `cargo fmt --check` exit 0. The example exits 2 on non-UTF-8 argv.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-28*
