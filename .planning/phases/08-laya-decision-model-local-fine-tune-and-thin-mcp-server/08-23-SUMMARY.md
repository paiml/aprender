---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 23
subsystem: api
tags: [mcp, pmcp, clap, clippy, contracts, untrusted-input, provenance, laya, gap-closure]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-12 bound decide-tool-boundary-v1 (v3.0.0); 08-19 decide-apr-v1 manifest.bindings (the load-side sources served_fields cites); 08-22 laya_gates.tsv external row parser-guard -> 08-23"
provides:
  - "Count-first parse_args: the classify `texts` count is checked on the parsed JSON array before any element becomes a String (V5-c), with one shared check_count refusal"
  - "aprender-mcp-decide argv parsed by a clap derive Cli; check_no_hand_rolled_parsers.sh no longer lists it (5 -> 4)"
  - "IN-05: the crate-wide clippy::disallowed_methods allow removed; the only allow is on `mod args` (ClassifyArgs + JsonSchema derive), re-exported at the old path"
  - "decide-tool-boundary-v1 4.0.0: served_fields (class A, served half) and untrusted_input_bounds (class B, request half) tables, each swept by a test"
  - "Private build_server_with_limits so the served-task fit check is exercised through the build path"
affects: [08-26, 08-28, 08-31, 08-32, phase-8-verification]

actuals:
  tokens: 12897
  tasks: 3
  commits: 3
plan_head_before: 444d7be00510884dc9999e8b2b1b785019a28505

tech-stack:
  added: [clap (workspace dep, already in Cargo.lock) for aprender-mcp-decide]
  patterns:
    - "Contract table + sweep test: every row dispatched by id; an unknown id or a row without a case fails; external rows must name a test that exists in the owner crate's source"
    - "Served-field provenance: the response's leaf paths (arrays as [*]) must EQUAL the table's rows, and each identity value is re-derived independently (sha2 over the served bytes, recipe.json)"
    - "Allow scoping: a lint allow lives on the smallest module that holds the macro expansion needing it, and is proven by planting the banned call elsewhere"

key-files:
  created: []
  modified:
    - crates/aprender-mcp-decide/src/lib.rs
    - crates/aprender-mcp-decide/src/main.rs
    - crates/aprender-mcp-decide/src/tests.rs
    - crates/aprender-mcp-decide/tests/e2e_stdio.rs
    - crates/aprender-mcp-decide/Cargo.toml
    - Cargo.lock
    - contracts/decide-tool-boundary-v1.yaml

key-decisions:
  - "decide-tool-boundary-v1 bumped 3.0.0 -> 4.0.0 because pv diff classifies the classify_count_bound invariant edit MAJOR (the v3.0.0 precedent); the two added tables are recorded in the same v4.0.0 paragraph"
  - "frame_stdio is an ACCEPTED row (pmcp 2.19.3 has no stdio framing cap; local transport started by the user's own client); the sweep asserts the count bound on a 100 000-element parsed array and pins pmcp 2.19.3 in Cargo.lock so a pmcp bump re-opens the row"
  - "build_server delegates to a private build_server_with_limits: the only way the served_task_min_row hostile case (and the skip-check mutation) can run through the build path with the tiny fixture, whose task fits the contracted tier"
  - "External rows (frame_http, probe_id_header) are verified by existence of the named test in the Lambda crate's source, plus a literal cross-check of frame_http's 4194304 against pmcp StreamableHttpServerConfig::stateless().max_request_bytes; the Lambda crate was not edited"

patterns-established:
  - "untrusted_input_bounds / served_fields: top-level contract tables pv ignores but a crate test sweeps row by row"

requirements-completed: [D-09, D-10, D-11, D-15]

coverage:
  - id: D1
    description: "The classify count is checked on the parsed JSON array before any text is materialised, over the real stdio transport"
    requirement: D-10
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#count_is_checked_before_element_shapes"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-decide/tests/e2e_stdio.rs#the_tiny_decide_server_classifies_over_live_stdio (3 numeric texts -> classify_max_texts)"
        status: pass
    human_judgment: false
  - id: D2
    description: "aprender-mcp-decide parses argv with clap derive and the parser guard no longer lists it"
    requirement: D-15
    verification:
      - kind: other
        ref: "bash scripts/check_no_hand_rolled_parsers.sh (36 scanned, 4 hand-rolled, aprender-mcp-decide absent)"
        status: pass
      - kind: manual_procedural
        ref: "target/debug/aprender-mcp-decide --bogus / --model / no args -> exit 2; --help -> exit 0; APRENDER_DECIDE_MODEL fallback read"
        status: pass
    human_judgment: false
  - id: D3
    description: "The crate-wide disallowed_methods allow is gone; a planted unwrap fails clippy"
    verification:
      - kind: other
        ref: "cargo clippy -p aprender-mcp-decide --all-targets --no-deps -- -D warnings (clean; planted unwrap -> error; control with the old allow -> clean)"
        status: pass
    human_judgment: false
  - id: D4
    description: "served_fields table: every served field names its sha-bound source and the served value equals it"
    requirement: D-11
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#every_served_field_has_a_bound_source"
        status: pass
    human_judgment: false
  - id: D5
    description: "untrusted_input_bounds table: every request-derived size has a bound and a hostile case; five code mutations and two table mutations each turn the sweep RED"
    requirement: D-10
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#request_bounds_table_is_swept (REQUEST BOUNDS swept=6 accepted=1 external=2)"
        status: pass
    human_judgment: false
  - id: D6
    description: "Public API used by the Lambda crate unchanged"
    requirement: D-15
    verification:
      - kind: unit
        ref: "cargo test -p aprender-mcp-decide-lambda --lib (31 passed)"
        status: pass
    human_judgment: false

duration: 17min
completed: 2026-09-28
status: complete
---

# Phase 8 Plan 23: Class A served fields, class B request bounds, V14-c and IN-05 on aprender-mcp-decide Summary

**The classify count is now checked on the parsed JSON array before any text is copied (proven over live stdio). `aprender-mcp-decide` parses argv with clap, and `.unwrap()` is banned again across the crate. decide-tool-boundary-v1 4.0.0 lists every served field with its sha-bound source and every request-derived size with its bound, and both tables are swept by tests that fail when a row loses its source or its bound.**

## Performance

- **Duration:** 17 min
- **Started:** 2026-09-28T18:59:53Z
- **Completed:** 2026-09-28T19:17:35Z
- **Tasks:** 3 (1 tracer + 2 auto)
- **Files modified:** 7

## Accomplishments

- **Count first (V5-c, T-08-23-01).** `parse_args` checks the `texts` array length against `ClassifyLimits::CONTRACTED` before serde builds any `String`. One `check_count` helper produces the refusal text for both `parse_args` and `precheck`, so the two messages cannot drift. `{"texts":[0,1,2]}` is now refused naming `classify_max_texts`, where before it got the shape message. This is tested at unit level and over the live stdio server.
- **V14-c.** `main.rs` uses a `#[derive(clap::Parser)] struct Cli { --model }`. The `APRENDER_DECIDE_MODEL` fallback and the "no model" message are kept. The guard now reports 36 scanned and 4 hand-rolled (5 before), and `aprender-mcp-decide` is not among them. The other 4 are other phases' crates, which plan 08-31 records as deferred.
- **IN-05 (T-08-23-04).** The crate-level `#![allow(clippy::disallowed_methods)]` is gone. `ClassifyArgs` and its `JsonSchema` derive live in `mod args`, which carries the only allow, and are re-exported at the old path.
- **Class A, served half (T-08-23-03).** `served_fields` has 9 `classify_response` rows and 3 `tools_list` rows. Each row names its source, and the load-side sources are the decide-apr-v1 `manifest.bindings` rungs.
- **Class B, request half.** `untrusted_input_bounds` has 9 rows. 7 are owned by this crate (1 of them accepted) and 2 by the Lambda crate. Each has a bound, `checked_by`, `owner_crate`, disposition and test.

## Class A: served fields (decide-tool-boundary-v1 `served_fields`)

| path | surface | source |
|------|---------|--------|
| model.artifact_sha256 | classify_response | sha256 of the whole served .apr, computed by the ladder and minted at rung 8. On the Lambda door it comes from HashedArtifact, the same digest the S3 pin is checked against |
| model.recipe_id | classify_response | manifest.recipe_id, bound at rung 4 to sha256(decide.recipe_json) and to the gate report's recipe_id |
| model.method | classify_response | manifest.method, bound at rung 3 to METHOD_LAYA |
| model.base | classify_response | manifest.base display string. manifest.base is bound at rung 4 to the recipe blob (plan 08-19) |
| labels[*] | classify_response | decide.task_json criteria names in document order, equal to manifest.labels at rung 4 |
| results[*].label | classify_response | labels[argmax(probabilities)], first on ties |
| results[*].probabilities[*] | classify_response | the Laya forward of the ladder-minted Decider. Weights are covered by the whole-file sha, the temperature is bound at rung 4, and the probes are replayed at rung 7 |
| results[*].tokens | classify_response | the built-row length from Model::prepare, which is exactly what the budget charged |
| results[*].truncated | classify_response | the truncation flag from Model::prepare (D-12) |
| description.question | tools_list | decide.task_json instructions |
| description.labels | tools_list | decide.task_json criteria in document order |
| description.bounds | tools_list | ClassifyLimits::CONTRACTED, asserted equal to the contract's constants |

`every_served_field_has_a_bound_source` makes these checks:
- The response's leaf paths must equal the 9 response rows.
- `artifact_sha256` must equal an independent `sha2` digest of the served tiny bytes.
- `recipe_id` must equal `sha256(recipe.json)`, and `base` must be re-derived from recipe.json.
- `method` must be `laya`, and all four `model.*` values must equal the Decider's identity.
- In each result, label must equal the argmax, and tokens and truncated must equal `prepare`'s output. Both truncation values are served.
- Every `tools_list` row needs a case.

Three mutations each turned the test RED: dropping the `results[*].tokens` row, serving `recipe_id` as `artifact_sha256`, and adding an unknown `description.examples` row.

## Class B: request bounds (decide-tool-boundary-v1 `untrusted_input_bounds`)

| id | bound | disposition | checked_by | owner | hostile case in the sweep |
|----|-------|-------------|------------|-------|---------------------------|
| frame_stdio | none | accepted | pmcp stdio (no cap in 2.19.3); parse_args bounds the parsed value | aprender-mcp-decide | 100 000 nulls refused naming classify_max_texts ("100000 texts"); Cargo.lock pins pmcp 2.19.3 |
| frame_http | 4194304 | enforced | pmcp `StreamableHttpServerConfig::stateless().max_request_bytes` | aprender-mcp-decide-lambda | external: `tests::server_config_is_stateless` must exist; the literal is cross-checked against pmcp's value |
| args_shape | deny_unknown_fields | enforced | parse_args (serde's message withheld) | aprender-mcp-decide | unknown key refused, no echo |
| texts_count_min | classify_min_texts | enforced | parse_args (array) and precheck, via check_count | aprender-mcp-decide | `[]` through parse_args and through the service |
| texts_count_max | classify_max_texts | enforced | parse_args (array, before any String) and precheck (before bytes) | aprender-mcp-decide | N+1 numbers through parse_args; N+1 texts with the first over the byte bound, through the service |
| text_bytes | classify_max_text_bytes | enforced | precheck, per text, before tokenization | aprender-mcp-decide | a text of classify_max_text_bytes + 1 bytes |
| built_tokens_total | classify_max_total_tokens | enforced | classify_blocking -> check_token_budget | aprender-mcp-decide | 2 long texts: built rows total past 120 under the contracted limits (asserted non-vacuous) |
| served_task_min_row | classify_max_total_tokens | enforced | build_server -> check_served_task_fits | aprender-mcp-decide | build_server_with_limits with a budget one short of min_row x max_texts |
| probe_id_header | 64 | enforced | aprender_mcp_decide_lambda::parse_probe_id | aprender-mcp-decide-lambda | external: `tests::probe_id_accepts_only_short_safe_ids` must exist |

Sweep output: `REQUEST BOUNDS swept=6 accepted=1 external=2`.

## Mutation table (each measured RED, then restored)

| # | Mutation | Test | Result |
|---|----------|------|--------|
| M1 | remove the array-count check from parse_args | request_bounds_table_is_swept | RED. The frame_stdio case got the shape message instead of classify_max_texts. The live e2e leg was also RED under the same mutation (Task 1) |
| M2 | precheck checks bytes, then count | request_bounds_table_is_swept | RED. texts_count_max: "texts[0] is 16385 UTF-8 bytes, over classify_max_text_bytes" |
| M3 | raise the byte bound by one in the service (`> max_text_bytes + 1`) | request_bounds_table_is_swept | RED. text_bytes: the 16385-byte text was classified (Ok response) |
| M4 | skip check_token_budget in classify_blocking | request_bounds_table_is_swept | RED. built_tokens_total: the over-budget request was classified |
| M5 | skip check_served_task_fits in build_server_with_limits | request_bounds_table_is_swept | RED. "build_server refuses a task the budget cannot serve at the count". Re-measured after the `match` rewrite |
| M6 (table) | add an owned row `texts_nesting_depth` with no case | request_bounds_table_is_swept | RED. "owned row texts_nesting_depth has no hostile case" |
| M7 (table) | external row names a nonexistent test | request_bounds_table_is_swept | RED. "must name a test that exists in it" |
| IN-05 | plant `"1".parse::<usize>().unwrap()` in tool_description | clippy -D warnings | RED: `use of a disallowed method core::result::Result::unwrap` at lib.rs:502. Control: re-adding the old crate-wide allow on top of the plant gave clippy rc=0, so the old allow hid it |

## Task Commits

1. **Task 1 (tracer): count checked on the JSON array, proven over live stdio** - `43d57f5f2` (feat)
2. **Task 2: clap in main.rs, allow narrowed, served_fields table + sweep** - `787014203` (feat)
3. **Task 3: untrusted_input_bounds table + sweep, mutation proof** - `da00143ff` (feat)

## Files Created/Modified

- `crates/aprender-mcp-decide/src/lib.rs`: `check_count` (the shared count refusal), count-first `parse_args`, `mod args` holding the only allow, and the private `build_server_with_limits`.
- `crates/aprender-mcp-decide/src/main.rs`: the clap derive `Cli`, replacing the hand-rolled `model_path_from`.
- `crates/aprender-mcp-decide/src/tests.rs`: three new tests (`count_is_checked_before_element_shapes`, `every_served_field_has_a_bound_source`, `request_bounds_table_is_swept`) and their helpers (`tiny_bytes`, `sha256_hex`, `contract_rows`, `leaf_paths`, `hostile_case`, `named_tests_exist`).
- `crates/aprender-mcp-decide/tests/e2e_stdio.rs`: the live leg that sends classify_max_texts + 1 numeric texts.
- `crates/aprender-mcp-decide/Cargo.toml`, `Cargo.lock`: `clap = { workspace = true }`.
- `contracts/decide-tool-boundary-v1.yaml`: 4.0.0, with the classify_count_bound invariants and precondition rewritten, the `served_fields` and `untrusted_input_bounds` tables added, and FALSIFY-DECIDE-TOOL-001 now citing the new count test.

## Decisions Made

See `key-decisions` in the frontmatter.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Private `build_server_with_limits` added so the served_task_min_row row can be swept through the build path**
- **Found during:** Task 3
- **Issue:** `build_server` always serves CONTRACTED limits, and the tiny fixture's task fits them. So the sweep had no way to make `build_server` refuse, and the plan's mutation "skip check_served_task_fits in build_server" could not turn anything RED.
- **Fix:** `build_server` now delegates to a private `build_server_with_limits(model, limits, name, version)`, which holds the check. The public signature is unchanged, and the Lambda tests pass.
- **Files modified:** crates/aprender-mcp-decide/src/lib.rs
- **Verification:** M5 RED, then restored.
- **Committed in:** da00143ff

**2. [Rule 1 - Bug, in my own draft] Corrected rung numbers in two served_fields sources**
- **Found during:** Task 2
- **Issue:** My first draft said blobs are sha-checked at rung 2 and that weights come from blobs. decide-apr-v1 checks blob sha256 at rung 4, and the weights are covered by the whole-file sha, rung 5 (finite) and rung 6 (rebuild).
- **Fix:** Rewrote the `results[*].probabilities[*]` and `results[*].tokens` sources from the ladder text before committing.
- **Committed in:** 787014203

**3. [Rule 2 - Missing critical] External rows are checked beyond "non-empty test"**
- **Found during:** Task 3
- **Issue:** The plan only asked for a non-empty `test:` on rows owned by another crate, with existence left to plan 08-31. A typo'd test name would therefore pass until 08-31.
- **Fix:** `named_tests_exist` now requires `fn <name>(` to exist in the owner crate's `src/` or `tests/`. frame_http's literal is also cross-checked against pmcp. M7 proves the existence check bites.
- **Committed in:** da00143ff

**Contract version:** the plan did not mention a version bump. pv diff suggested MAJOR, so I bumped 3.0.0 to 4.0.0, following the v3.0.0 precedent. Nothing in the repo pins the version string (checked with `git grep`).

---

**Total deviations:** 3 auto-fixed (1 blocking, 1 bug in my own draft, 1 missing critical)
**Impact on plan:** All three strengthen the gates the plan asked for. There is no scope creep, and the Lambda crate was not edited.

## Issues Encountered

- **The repo-wide `scripts/check_contract_test_binding.sh` is VACUOUS right now.** It reports "strict-test-binding gate was SKIPPED (contract validation failed)" because of `contracts/spectral-indices-v1.yaml` PV-PRV-001. That error predates this plan (fdf6b1802) and is already in deferred-items.md.
  - A scoped run (a temp tree holding only this contract, with `crates` symlinked) found **25 refs, 25 found, 0 missing**. The baseline before this plan was 24.
  - The first scoped attempt resolved nothing ("0 contracts", then "25 missing") until the tree layout matched the repo's. I noted that so the scoped result is not mistaken for a first-try pass.
- `make contract-audit-phase8`: passed, "No binding gaps found".
- The real-model e2e leg (`APR_MCP_E2E_DECIDE_MODEL`) was not set, so it took its SKIP path. The plan does not require it.
- The two known pre-existing reds (apr-format golden writer, aprender-compute clippy) were not touched. The `--no-deps` clippy of this crate is clean.

## Known Stubs

None.

## Threat Flags

None. `build_server_with_limits` is private, and no new endpoint, auth path or file access was added.

## User Setup Required

None.

## Next Phase Readiness

- Plan 08-26 adds the ARTIFACT half of class B to decide-apr-v1. The request half is here.
- Plan 08-28 owns the tool description sentence, the admission claim and the refusal wire code. None of them were changed here.
- Plan 08-31's claims check can read `untrusted_input_bounds[*].test` for the Lambda-owned rows. `frame_http`'s named test asserts only `stateless()`, and its 4 MiB value is cross-checked from this crate (see the row's `note:`).
- In `scripts/laya_gates.tsv`, the `parser-guard external:08-23` row's offender is now fixed. The guard still fails overall on 4 crates from other phases (08-31's deferred set).

## Self-Check: PASSED

- The files exist: lib.rs, main.rs, tests.rs, e2e_stdio.rs, Cargo.toml, Cargo.lock, decide-tool-boundary-v1.yaml.
- Commits 43d57f5f2, 787014203 and da00143ff are present in `git log`. `git rev-list --count 444d7be00..HEAD` = 3 before this SUMMARY commit.
- Final run:
  - `cargo test -p aprender-mcp-decide`: lib 31/31, e2e 2/2.
  - `cargo test -p aprender-mcp-decide-lambda --lib`: 31/31.
  - clippy `-D warnings` on both crates: clean.
  - `pv validate`: 0 errors.
  - `cargo fmt --check`: clean.
  - The parser guard self-test passes, and the guard no longer lists aprender-mcp-decide.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-28*
