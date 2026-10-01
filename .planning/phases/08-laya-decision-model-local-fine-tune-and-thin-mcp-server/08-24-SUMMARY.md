---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 24
subsystem: infra
tags: [lambda, mcp, pmcp, cors, health, clippy, mutation, untrusted-input-bounds]

requires:
  - phase: 08-23
    provides: decide-tool-boundary-v1 v4.0.0 untrusted_input_bounds (frame_http, probe_id_header owned by aprender-mcp-decide-lambda)
  - phase: 08-25
    provides: probe.rs as merged, every json! inside a function so item-level allows can scope them
provides:
  - watch_loopback(handle, exit) — a dead loopback ends the process (exit 1), wired in main.rs to std::process::exit
  - route(method) / Route — only POST loads; GET health, OPTIONS preflight, anything else 405 without loading
  - health(source, loaded) — 200 ok:true only on a parsed APRENDER_DECIDE_* config, else 503 ok:false with config_error
  - copy_upstream_headers / proxied_headers — exactly one access-control-allow-origin on a proxied response
  - map_bounded_read_error — rung 1's ArtifactTooLarge surfaces as ResolveError::TooLarge (kind too_large)
  - crate-wide disallowed_methods allows removed from lib.rs and main.rs; item-level allows with reasons only
  - tests::lambda_request_rows_are_swept — class-B request rows this crate owns, frame_http exactly 413
  - ten-row mutation record including the V4-a resume fix (commit ea940faec)
affects: [08-28, 08-30, 08-31, decide-tool-boundary-v1]

actuals:
  tokens: 10133
  tasks: 3
  commits: 4
plan_head_before: 4d00f50e17116e4a6d4ed25481b864d7b32da062

tech-stack:
  added: [serde_yaml (dev-dependency of aprender-mcp-decide-lambda; workspace crate, already locked)]
  patterns:
    - "Injected exit: a process-ending watcher takes `exit: impl FnOnce(i32)` so a test observes the call on a real loopback"
    - "The bootstrap's decisions (route, health, header merge) are pure lib functions; main.rs only switches on them"
    - "Lint allows are item-scoped with a one-line reason; a planted unwrap proves the ban still bites around them"

key-files:
  created: []
  modified:
    - crates/aprender-mcp-decide-lambda/src/lib.rs
    - crates/aprender-mcp-decide-lambda/src/main.rs
    - crates/aprender-mcp-decide-lambda/src/probe.rs
    - crates/aprender-mcp-decide-lambda/src/s3.rs
    - crates/aprender-mcp-decide-lambda/src/tests.rs
    - crates/aprender-mcp-decide-lambda/Cargo.toml
    - crates/aprender-mcp-decide-lambda/README.md
    - Cargo.lock

key-decisions:
  - "map_bounded_read_error passes rung 1's own `what` through (stream or declared_length) instead of hard-coding stream: the refusal names the check that fired"
  - "server_name/DEFAULT_SERVER_NAME/PACKAGE moved from main.rs into lib.rs so health(source, loaded) keeps the planned two-argument signature and is testable"
  - "main.rs's json! sites go through one allowed helper, error_body, so handler itself carries no allow and an unwrap in it fails clippy"
  - "frame_http is swept with an at-bound control (exactly 4194304 bytes is classified, 200) beside the hostile case (4194305 bytes is exactly 413, no classify), so the 413 is proven to be the size bound"

patterns-established:
  - "Contract row sweep per owning crate: the Lambda crate dispatches its own untrusted_input_bounds rows by id, mirroring aprender-mcp-decide's request_bounds_table_is_swept"

requirements-completed: [D-10, D-15, D-18]

coverage:
  - id: D1
    description: "A container whose loopback MCP server task ended exits(1) after logging, so Lambda replaces it (WR-06)"
    requirement: D-15
    verification:
      - kind: integration
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#loopback_end_exits_the_process"
        status: pass
      - kind: other
        ref: "mutant M1 (watch_loopback does not call exit) -> RED left None right Some(1)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Only POST triggers the cold load; GET/OPTIONS never load; any other method is a 405 without loading (V4-c)"
    requirement: D-15
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#non_post_methods_do_not_load"
        status: pass
      - kind: other
        ref: "mutant M2 (route(\"HEAD\") -> Load) -> RED"
        status: pass
    human_judgment: false
  - id: D3
    description: "GET health is ok:true only when the model-source config parses; otherwise 503 ok:false naming the config error (A4-6)"
    requirement: D-18
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#health_is_not_ok_on_invalid_config"
        status: pass
      - kind: other
        ref: "mutant M3 (health ignores the source) -> RED left 200 right 503"
        status: pass
    human_judgment: false
  - id: D4
    description: "A proxied response carries exactly one access-control-allow-origin (IN-06)"
    requirement: D-15
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#proxied_response_has_one_cors_origin"
        status: pass
      - kind: other
        ref: "mutant M4 (access-control-* copied) -> RED left [\"*\", \"https://evil.example\"]"
        status: pass
    human_judgment: false
  - id: D5
    description: "Over-cap local artifact is too_large on the declared path (real file), and rung 1's ArtifactTooLarge maps to too_large; no second copy of rung 1's arithmetic (V3-d, R2)"
    requirement: D-18
    verification:
      - kind: integration
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#local_over_cap_is_too_large"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#artifact_too_large_maps_to_too_large"
        status: pass
      - kind: other
        ref: "mutant M5 (ArtifactTooLarge -> Read) -> RED left read right too_large"
        status: pass
    human_judgment: false
  - id: D6
    description: "Load-once docs describe the tokio OnceCell; no 'load lock' in main.rs/s3.rs/lib.rs; lock-named test renamed (V3-c, S6)"
    requirement: D-15
    verification:
      - kind: other
        ref: "cat main.rs s3.rs lib.rs | grep -c 'load lock' -> 0; s3::tests::failed_load_leaves_the_cell_empty_and_the_next_call_loads passes"
        status: pass
    human_judgment: false
  - id: D7
    description: "Crate-wide disallowed_methods allows removed (lib.rs, main.rs); item-level allows with reasons in lib.rs, main.rs, probe.rs; clippy -D warnings clean; planted unwraps in lib.rs, probe.rs (and main.rs) fail clippy (IN-05, Lambda half)"
    requirement: D-10
    verification:
      - kind: other
        ref: "cargo clippy -p aprender-mcp-decide-lambda --all-targets --no-deps -- -D warnings (rc 0; planted unwrap rc 101 at lib.rs:774, probe.rs:320, main.rs:93)"
        status: pass
    human_judgment: false
  - id: D8
    description: "The two Lambda-owned untrusted_input_bounds rows are swept; frame_http requires exactly HTTP 413"
    requirement: D-10
    verification:
      - kind: integration
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#lambda_request_rows_are_swept (LAMBDA REQUEST BOUNDS swept=2)"
        status: pass
      - kind: other
        ref: "mutants M6, M8, M9, M10 -> RED"
        status: pass
    human_judgment: false
  - id: D9
    description: "V4-a resume-at-first-missing-byte fix (ea940faec) has a recorded red-side proof"
    requirement: D-18
    verification:
      - kind: other
        ref: "mutant M7 (fill_part discards cut-attempt progress) -> s3::tests::cut_attempt_resumes_at_the_first_missing_byte RED left 2 right 1"
        status: pass
    human_judgment: false

duration: 16min
completed: 2026-09-28
status: complete
---

# Phase 8 Plan 24: Lambda runtime shim hardening Summary

**The decide Lambda bootstrap now exits when its loopback MCP server dies, loads the model only for POST (405 for anything else), reports config errors as a 503 health body, forwards one CORS origin, labels over-cap reads too_large, keeps the unwrap ban crate-wide via item-level allows, and sweeps its two contract request rows (frame_http exactly 413).**

AWS CALLS: 0. No deploy, no live action; the running function (D-ITEM-08-18-A) is untouched. The
behaviour changes here ship only when plan 08-30 (or a later plan) redeploys.

## Performance

- **Duration:** 16 min
- **Started:** 2026-09-28T19:37:41Z
- **Completed:** 2026-09-28T19:53:09Z
- **Tasks:** 3
- **Files modified:** 8 (5 sources, Cargo.toml, README.md, Cargo.lock)

## Accomplishments

- **WR-06:** `watch_loopback(handle, exit)` awaits the loopback task, logs
  `decide.loopback ended: <outcome>; exiting so Lambda replaces this environment` at error level
  and calls `exit(1)`. `main.rs` spawns it with `std::process::exit`. Proven on a real tiny-model
  loopback that first answers a POST initialize, then has its handle aborted.
- **V4-c:** `route(method)` sends only `POST` to the load. `GET` gets the health body, `OPTIONS` the
  preflight, and every other method (HEAD, PUT, DELETE, PATCH, TRACE, CONNECT, lowercase `post`, the
  empty string) gets a 405 with `allow: POST, GET, OPTIONS` before `LOADED` is touched.
- **A4-6:** `health(source, loaded)` answers 200 `ok:true` only for a parsed S3 (pinned) or local
  source. Missing, ambiguous, unpinned-S3 and malformed-pin configs get 503 `ok:false` with
  `config_error` set to `SourceError`'s Display. The test proves the pin value is never echoed.
- **IN-06:** `copy_upstream_headers` drops upstream `access-control-*`, `transfer-encoding` and
  `content-length`. The test builds a real `lambda_http::Response` and counts exactly one origin.
- **V3-d / R2:** `map_bounded_read_error` maps rung 1's `ArtifactTooLarge` to
  `ResolveError::TooLarge` (kind `too_large`). The unreachable post-read length branch is deleted,
  so the Lambda crate carries no second copy of rung 1's cap arithmetic.
- **V3-c / S6:** the OnceCell is described in the `main.rs` header, the `s3.rs` module docs, the
  `LoadOnce` and `ResolveError` docs and the crate README. The s3 test is renamed
  `failed_load_leaves_the_cell_empty_and_the_next_call_loads`.
- **IN-05 (Lambda half):** both crate-wide `#![allow(clippy::disallowed_methods)]` are gone.
  Item-level allows, each with `// serde_json::json! expands to .unwrap()`, sit on `health`
  (lib.rs), `error_body` (main.rs), probe.rs `Rpc::call`, `classify_params` and
  `run_identity_probe`, and one test. Nothing precedes `pub mod probe;`, and probe.rs's diff is
  exactly those 3 added lines.
- **Class B, Lambda half:** `lambda_request_rows_are_swept` covers both owned rows
  (`LAMBDA REQUEST BOUNDS swept=2`).

## Task Commits

1. **Task 1 (tracer): dead loopback ends the process**: `c82158de0` (feat). `99a2499a7` (style) is the rustfmt fix for that test.
2. **Task 2: routing, health, CORS, too_large, docs, item-scoped allows**: `64813d0e5` (feat)
3. **Task 3: Lambda request-row sweep + mutation proof**: `4309a816d` (test)

**Plan metadata:** this SUMMARY commit (docs)

## RED-first evidence

| Test | RED against | Observed |
|------|-------------|----------|
| loopback_end_exits_the_process | log-only watcher (HEAD's behaviour) | `left: None right: Some(1)`; the initialize assertion had passed |
| non_post_methods_do_not_load | HEAD-equivalent route (only GET/OPTIONS special) | `"HEAD"` routed to Load |
| health_is_not_ok_on_invalid_config | HEAD's always-ok body | `Missing: {"ok":true,...}` `left: 200` |
| proxied_response_has_one_cors_origin | HEAD's copy loop | `left: ["*", "https://evil.example"]` |
| artifact_too_large_maps_to_too_large | HEAD's `map_err(ResolveError::Read)` | `left: "read"` |
| local_over_cap_is_too_large | passed at HEAD: the declared-length refusal already existed | characterization pin, on a real non-sparse file, plus the at-cap acceptance |

## Mutation record (Task 3; each applied alone, its named test run, file restored)

Each run was `cargo test -p aprender-mcp-decide-lambda --lib -- <test> --exact`. Every mutated
file was `cmp`-verified against its pre-mutant copy. Afterwards an md5 of all sources matched the
pre-loop hashes. s3.rs has no committed change after Task 2 (`git diff --quiet 64813d0e5..HEAD -- s3.rs`
exit 0) and no uncommitted change.

| # | Mutant | Test | RED how | Restored |
|---|--------|------|---------|----------|
| M1 | watch_loopback does not call exit | loopback_end_exits_the_process | left None, right Some(1) | yes |
| M2 | route("HEAD") returns Load | non_post_methods_do_not_load | `"HEAD"`: left Load, right MethodNotAllowed | yes |
| M3 | health ignores the source (always 200 / ok true) | health_is_not_ok_on_invalid_config | Missing: left 200, right 503 | yes |
| M4 | copy_upstream_headers copies access-control-* | proxied_response_has_one_cors_origin | left ["*", "https://evil.example"], right ["*"] | yes |
| M5 | map_bounded_read_error maps ArtifactTooLarge back to Read | artifact_too_large_maps_to_too_large | left "read", right "too_large" | yes |
| M6 | PROBE_ID_MAX_LEN 64 -> 65 | lambda_request_rows_are_swept; probe_id_accepts_only_short_safe_ids | sweep: probe_id_header literal left 64, right 65; unit: 65-char id accepted | yes |
| M7 | **V4-a (ea940faec):** fill_part discards what a cut attempt landed (done stays 0, retry re-fetches the part from its first byte) | s3::tests::cut_attempt_resumes_at_the_first_missing_byte | "the part's first byte is fetched once": left 2, right 1 | yes (s3.rs byte-identical) |
| M8 | parse_probe_id admits bound+1 (constant unchanged) | lambda_request_rows_are_swept; probe_id_accepts_only_short_safe_ids | "probe_id_header: one over the bound" accepted | yes |
| M9 | server_config doubles max_request_bytes | lambda_request_rows_are_swept | frame_http: left 4194304, right 8388608 | yes |
| M10 | M9 plus the sweep's literal check removed (does the live case bite on its own?) | lambda_request_rows_are_swept | "one byte over the bound": left 200, right 413 (the over-cap frame was classified) | yes |

M1–M7 are the seven the plan asked for. M8–M10 are additions. M8 shows the sweep bites on
`parse_probe_id`'s behaviour, not only on the constant. M9 and M10 show the frame_http case bites
twice: once through the literal cross-check, and once through the live 413 assertion alone.

## Planted-unwrap checks (Task 2, IN-05)

| Where | Function | clippy -D warnings |
|-------|----------|--------------------|
| lib.rs | parse_probe_id (no allow) | rc 101: `use of a disallowed method core::result::Result::unwrap` at lib.rs:774 |
| probe.rs | labels_in_order (no allow) | rc 101: same lint at probe.rs:320 |
| main.rs (extra) | http() (no allow) | rc 101: same lint at main.rs:93 |

All three were removed and clippy re-ran clean (rc 0).

## Verification (final tree)

- `cargo test -p aprender-mcp-decide-lambda`: lib 42 passed, 0 failed. The bin and doc targets
  have 0 tests. That covers the eight new or renamed tests and `tiny_identity_oracle_is_the_committed_golden`.
- `cargo test -p aprender-mcp-decide-lambda --lib -- tests::lambda_request_rows_are_swept --nocapture`
  prints `LAMBDA REQUEST BOUNDS swept=2`.
- `cargo clippy -p aprender-mcp-decide-lambda --all-targets --no-deps -- -D warnings`: rc 0.
- `cargo fmt -p aprender-mcp-decide-lambda -- --check`: rc 0.
- `cargo build -p aprender-mcp-decide-lambda --bin bootstrap`: rc 0.
- `grep -c '^#!\[allow(clippy::disallowed_methods)\]'` gives 0 for lib.rs, main.rs and probe.rs.
  No allow precedes `pub mod probe;`, and probe.rs carries 3 item-level allows.
- `cat main.rs s3.rs lib.rs | grep -c "load lock"` gives 0.
- Cross-crate check: `cargo test -p aprender-mcp-decide --lib -- tests::request_bounds_table_is_swept`
  still passes. The Lambda-owned rows' named tests still exist.

## Files Created/Modified

- `crates/aprender-mcp-decide-lambda/src/lib.rs`: adds watch_loopback, Route/route, health,
  copy_upstream_headers/proxied_headers, server_name/PACKAGE/ALLOWED_METHODS and
  map_bounded_read_error. Deletes the post-read length branch, updates the OnceCell and
  ResolveError docs, and removes the crate-wide allow.
- `crates/aprender-mcp-decide-lambda/src/main.rs`: the handler switches on route() and answers 405
  before LOADED. GET answers health(). Proxied headers go through proxied_headers(), and error
  bodies go through the error_body helper. The watcher is watch_loopback with
  std::process::exit. OnceCell header docs; crate-wide allow removed.
- `crates/aprender-mcp-decide-lambda/src/probe.rs`: 3 item-level allow lines, nothing else.
- `crates/aprender-mcp-decide-lambda/src/s3.rs`: module docs describe the OnceCell; the lock-named
  test is renamed.
- `crates/aprender-mcp-decide-lambda/src/tests.rs`: 7 new tests plus sweep helpers.
- `crates/aprender-mcp-decide-lambda/Cargo.toml`, `Cargo.lock`: serde_yaml dev-dependency (a lock
  edge only).
- `crates/aprender-mcp-decide-lambda/README.md`: lazy-load paragraph (OnceCell, POST-only, health
  503, loopback exit).

## Decisions Made

- `map_bounded_read_error` keeps rung 1's own `what` rather than hard-coding `"stream"`, so the
  error names the check that fired.
- `server_name` and its constants moved to lib.rs so `health(source, loaded)` keeps the planned
  signature and stays testable.
- One allowed helper, `error_body`, holds main.rs's `json!` calls, so `handler` itself carries no
  allow.
- The frame_http case includes an at-bound control (exactly 4 MiB is classified, 200). That makes
  the 413 attributable to the size bound and not to some other refusal.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] serde_yaml dev-dependency for the contract sweep**
- **Found during:** Task 3
- **Issue:** `lambda_request_rows_are_swept` must read decide-tool-boundary-v1, and the Lambda crate had no YAML parser. Its files list did not include Cargo.toml.
- **Fix:** Added `serde_yaml = { workspace = true }` under `[dev-dependencies]`. It is the same workspace crate that aprender-mcp-decide's sweep uses and was already in Cargo.lock. The lock diff is one dependency edge; nothing was downloaded.
- **Files modified:** crates/aprender-mcp-decide-lambda/Cargo.toml, Cargo.lock
- **Committed in:** 4309a816d

**2. [Rule 1 - Doc honesty] README described the removed lock**
- **Found during:** Task 2
- **Issue:** The crate README still said the model loads "behind a lock", outside the plan's files list.
- **Fix:** Rewrote the lazy-load paragraph to cover the OnceCell, POST-only loads, the 503 health body and the loopback exit.
- **Files modified:** crates/aprender-mcp-decide-lambda/README.md
- **Committed in:** 64813d0e5

**3. [Rule 1 - Format] Task 1 commit was not rustfmt-clean**
- **Found during:** Task 1 post-commit fmt check
- **Fix:** A separate style commit, `99a2499a7`.
- **Committed in:** 99a2499a7

---

**Total deviations:** 3 auto-fixed (1 blocking, 2 doc/format).
**Impact on plan:** None on scope. The mutation record has 3 more rows than planned (M8–M10), and
the planted-unwrap check also covers main.rs.

## Issues Encountered

None.

## Deferred (logged in deferred-items.md)

- `examples/probe.rs` still carries a crate-wide `#![allow(clippy::disallowed_methods)]`. It is its
  own target and was not in this plan's files list.
- The decide-tool-boundary-v1 rows `frame_http` and `probe_id_header` still name
  `server_config_is_stateless` and `probe_id_accepts_only_short_safe_ids` as their `test:`. The
  hostile cases now run in `lambda_request_rows_are_swept`. That is a contract edit for the next
  plan that versions the file.

## User Setup Required

None. No external service configuration was required.

## Next Phase Readiness

- Plan 08-28 may edit probe.rs `classify_payload` under the same item-level allow rule. The lib's
  crate-wide allow is gone, so any `json!` it adds needs its own allow.
- Plan 08-30 owns the S3 download deadline (V4-b) and any redeploy. These runtime fixes reach the
  live function only through that redeploy.

## Self-Check: PASSED

- Modified files exist: lib.rs, main.rs, probe.rs, s3.rs, tests.rs, Cargo.toml, README.md.
- Commits c82158de0, 99a2499a7, 64813d0e5 and 4309a816d are in `git log`.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-28*
