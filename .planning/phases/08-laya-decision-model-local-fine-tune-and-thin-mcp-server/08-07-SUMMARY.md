---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 07
subsystem: infra
tags: [aprender-mcp-decide-lambda, aws-lambda, pmcp, streamable-http, stateless, s3, aws-sdk-s3, cold-start, sha256-pin, probe, laya]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-06 aprender-mcp-decide: build_server, load_model_from_bytes, ClassifyLimits::CONTRACTED, precheck, check_token_budget, TOOL_NAME"
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-04/08-05 aprender-decide: ArtifactLimits::CONTRACTED, read_decide_apr_bytes_bounded, pack_run_dir, Decider::prepare / manifest().agent.max_len / identity()"
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-01 decide-tool-boundary-v1 accepted_region_cold (CONCENTRATED / DISTRIBUTED shapes)"
provides:
  - "crates/aprender-mcp-decide-lambda (publish = false): lib aprender_mcp_decide_lambda + `bootstrap` bin"
  - "server_config() (StreamableHttpServerConfig::stateless), start_loopback(server, addr), build_server delegating to aprender-mcp-decide"
  - "ModelSource::{from_env, from_lookup} (S3 URI + mandatory Sha256Pin, or local path + optional pin), SourceError, parse_s3_uri"
  - "resolve_model / resolve_local / resolve_from_fetcher -> (Model, LoadTimeline); ResolveError with a caller-safe kind()"
  - "s3::{RangeFetcher, S3Fetcher, DownloadPolicy::DEPLOYED, download_into_memory(_with), S3LoadError, FetchError, PART_BYTES, CONCURRENCY, RETRIES, ATTEMPT_TIMEOUT, DOWNLOAD_DEADLINE}"
  - "LoadOnce<T> (lazy once-per-container load, re-armed after failure), parse_probe_id, load_header_value, load_log_line, graviton_generation"
  - "probe::{run_identity_probe, run_cold_first, build_maximal_request, MaximalShape, ProbeReport, ColdSample, new_probe_id}"
  - "examples/probe.rs CLI; .pmcp/deploy.toml.template + .pmcp/.gitignore"
affects: [08-10, 08-11, 08-12]

actuals:
  tokens: 31410   # chars/4 over the realized diff (125642 chars, git diff c9a639816..HEAD, code commits only)
  tasks: 3
  commits: 5
plan_head_before: c9a63981603bc33d7703c16fe8a5416b938e9851

tech-stack:
  added:
    - "aprender-mcp-decide-lambda: aws-config 1.8.18 / aws-sdk-s3 1.137.0 (already locked, no version moved), futures 0.3, lambda_http 0.13, reqwest 0.12 (rustls), pmcp 2.19; dev: tempfile, tokio test-util"
  patterns:
    - "Loopback Lambda shim over a thin server crate: the bootstrap and the tests share server_config() through start_loopback, so what ships is what is tested"
    - "Cold-start weights in memory: head length -> cap check -> one pre-sized Vec split into disjoint slices, each filled in place by a ranged GET; bounded FuturesUnordered, per-attempt timeout, overall deadline"
    - "Pin before parse: sha256(buffer) == APRENDER_DECIDE_SHA256 before load_model_from_bytes, then model.identity() re-checked against the hash"
    - "Cold evidence independent of the client's timing: probe id header -> decide.load log line + x-decide-load response header"
    - "A lib-test Send guard for a future only the bin awaits (cargo test --lib never builds the bin)"

key-files:
  created:
    - crates/aprender-mcp-decide-lambda/Cargo.toml
    - crates/aprender-mcp-decide-lambda/README.md
    - crates/aprender-mcp-decide-lambda/src/lib.rs
    - crates/aprender-mcp-decide-lambda/src/main.rs
    - crates/aprender-mcp-decide-lambda/src/s3.rs
    - crates/aprender-mcp-decide-lambda/src/probe.rs
    - crates/aprender-mcp-decide-lambda/src/tests.rs
    - crates/aprender-mcp-decide-lambda/examples/probe.rs
    - crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template
    - crates/aprender-mcp-decide-lambda/.pmcp/.gitignore
  modified:
    - Cargo.toml
    - Cargo.lock
    - README.md
    - crates/aprender-core/tests/monorepo_invariants.rs

key-decisions:
  - "FALSIFY-MONO-011 DEPLOYMENT_UNIT_BASELINE raised 8 -> 9 for aprender-mcp-decide-lambda under Phase 8 CONTEXT D-15, which names the Lambda bootstrap as the pmcp.run deployment unit (same pattern as 08-06's 7 -> 8). This needs human confirmation"
  - "RangeFetcher::fetch_range writes into the caller's slice rather than returning a Vec. The download stays one buffer with no per-part copy, which avoids up to 16 x 64 MiB of transient memory"
  - "The concurrency is a hand-rolled bounded FuturesUnordered, not stream::iter(..).map(closure).buffer_unordered(n). The closure form makes rustc unable to prove the lambda_http handler future Send for every lifetime"
  - "CONCENTRATED is capped at max_texts, and its last text is sized to the remainder when the budget is not a whole number of rows. For Laya-en (512, 1024) that is exactly 2 full max-byte rows. The tiny fixture under CONTRACTED limits gets 8 full 64-token rows"
  - "A failed load returns HTTP 503 with only ResolveError::kind(). The full detail, which can name the bucket and key, goes only to the log"
  - "The deploy template documents cargo-pmcp's package-resolution trap instead of prescribing `--manifest-path crates/aprender-mcp-decide-lambda`, which would ship another *-lambda binary. The deploy root is 08-10's decision"

patterns-established:
  - "Thin Lambda shim = LoadOnce(resolve_model + start_loopback) on the first MCP POST, proxy with x-decide-load, 503 kind-only on a failed load"
  - "S3 loader tests: an in-memory fault-injecting RangeFetcher (failures per offset, short body, stall) on a paused tokio clock, with no AWS involved"

requirements-completed: [D-11, D-15, D-18]

coverage:
  - id: D1
    description: "The stateless loopback server over the tiny fixture answers over real HTTP on 127.0.0.1. run_identity_probe sees exactly one `classify` tool, labels [shipping, billing, account] in order in its description, and identity == sha256 of the packed bytes, and a wrong expectation reports identity_matches false. run_cold_first against a FRESH server that never saw initialize is served, with the right identity"
    requirement: "D-11"
    verification:
      - kind: integration
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#loopback_identity_probe_over_real_http"
        status: pass
      - kind: integration
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#loopback_cold_first_call_is_served_without_initialize"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#server_config_is_stateless"
        status: pass
    human_judgment: false
  - id: D2
    description: "The in-memory S3 loader refuses every failure mode with a typed error, tested with an injected fetcher and no AWS. The cases covered:\n- byte-exact reassembly of a 3.5-part object;\n- a part failing twice is retried (3 attempts recorded);\n- 5 failures give PartFailed{offset 4, attempts 5};\n- ShortBody{8,4,3};\n- MissingLength with zero range fetches;\n- over-cap refused before allocating (length u64::MAX under the contracted cap);\n- DOWNLOAD_DEADLINE abandons stalled parts at 25 s and ATTEMPT_TIMEOUT cuts a slow attempt at 8 s, then retries (paused clock);\n- a wrong pin is refused before the ladder (garbage under a wrong pin is HashMismatch, and under its own hash is Load);\n- s3:// parsing;\n- the load lock re-arms after a failed load;\n- a local file over a shrunk cap is refused from its metadata"
    requirement: "D-18"
    verification:
      - kind: unit
        ref: "cargo test -p aprender-mcp-decide-lambda --lib s3:: (12 passed)"
        status: pass
    human_judgment: false
  - id: D3
    description: "aws-sdk-s3 resolves to the already-locked v1.137.0. The Cargo.lock diff is additions only: the new package entry and its dependency edges, with no existing version moved. `cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.34 --bin bootstrap` builds a 24.8 MB aarch64 ELF, with no weights baked in"
    requirement: "D-18"
    verification:
      - kind: other
        ref: "cargo tree -p aprender-mcp-decide-lambda -i aws-sdk-s3 --depth 0 -> aws-sdk-s3 v1.137.0"
        status: pass
      - kind: other
        ref: "ulimit -n 65536; cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.34 -p aprender-mcp-decide-lambda --bin bootstrap; file -> ELF 64-bit LSB pie executable, ARM aarch64"
        status: pass
    human_judgment: false
  - id: D4
    description: "build_maximal_request builds both accepted_region_cold shapes. Under shrunk limits, CONCENTRATED gives 2 texts of max_text_bytes with every row truncated and a total within 8 of the budget, and an uneven budget sizes the last text. DISTRIBUTED gives max_texts texts within 8 of the budget. Both pass precheck and check_token_budget. Both CONTRACTED-limit requests are served as the first and only POST to a fresh loopback server. examples/probe.rs builds, and --help lists --maximal, --cold-first and --probe-id"
    requirement: "D-11"
    verification:
      - kind: unit
        ref: "cargo test -p aprender-mcp-decide-lambda --lib probe:: (7 passed, incl. loopback_maximal_requests_are_served_cold_first)"
        status: pass
      - kind: other
        ref: "cargo build -p aprender-mcp-decide-lambda --examples; target/debug/examples/probe --help"
        status: pass
    human_judgment: false
  - id: D5
    description: ".pmcp/deploy.toml.template is tracked and sets memory_mb 10240 and timeout_seconds 30. The UNSET placeholder sits on the [server] name, APRENDER_DECIDE_S3_URI and APRENDER_DECIDE_SHA256 lines. auth is a placeholder pending 08-11, and the generated deploy.toml path is gitignored"
    requirement: "D-18"
    verification:
      - kind: other
        ref: "git check-ignore -q --no-index (template, s3.rs, examples/probe.rs not ignored; .pmcp/deploy.toml ignored)"
        status: pass
    human_judgment: true
    rationale: "The template's values are deploy policy (auth left open pending the 08-11 checkpoint; the deploy root still undecided because of the cargo-pmcp resolver trap, D-ITEM-08-07-B). Nothing asserts they are the right policy until 08-10/08-11 deploy it"
  - id: D6
    description: "aprender-mcp-decide-lambda is registered in FALSIFY-MONO-011 deployment_unit_bins with publish = false (baseline 8 -> 9 under D-15); README workspace-crate count 90 == cargo metadata"
    requirement: "D-15"
    verification:
      - kind: integration
        ref: "cargo test -p aprender-core --test monorepo_invariants --test readme_contract (11 + 15 passed)"
        status: pass
    human_judgment: true
    rationale: "The ratchet says raising DEPLOYMENT_UNIT_BASELINE is a recorded-CONTEXT decision; D-15 names this unit but also said 'no baseline change is expected'. A human should confirm the 8 -> 9 raise, as for 08-06's 7 -> 8"
  - id: D7
    description: "At runtime, the bootstrap handler loads lazily on the first MCP POST behind LoadOnce, logs `decide.load performed_load=<bool> probe_id=<id> load_ms=...`, sets `x-decide-load: cold;load_ms=<n>` or `warm`, and answers 503 with a kind only on a failed load. Its pure pieces are unit-tested (probe-id filter, header and log strings, LoadOnce re-arm, Send property)"
    requirement: "D-15"
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#probe_id_accepts_only_short_safe_ids, #load_evidence_names_the_loading_request, #resolve_model_future_is_send; s3::tests#retry_after_failure_rearms_the_load_lock"
        status: pass
    human_judgment: true
    rationale: "main.rs::handler itself never ran under a Lambda runtime (D-ITEM-08-07-A); the live proof is 08-10/08-11"

duration: 24min
completed: 2026-09-26
status: complete
---

# Phase 8 Plan 07: Lambda deployment unit `aprender-mcp-decide-lambda` Summary

**`aprender-mcp-decide-lambda` is the pmcp.run deployment unit for the decide server. Its `bootstrap` loads the model lazily on the first MCP POST, fetching the pinned `.apr` from S3 by 16-way 64 MiB ranged GETs straight into one pre-sized in-memory buffer. Each attempt is capped at 8 s and the whole download at 25 s. The buffer's sha256 must equal `APRENDER_DECIDE_SHA256` before the decide-apr-v1 ladder parses a byte. The bootstrap then serves `classify` through a stateless loopback pmcp server. Every request is tagged cold or warm, both in a `decide.load` log line keyed by the probe id and in an `x-decide-load` header. A probe CLI verifies a live endpoint's identity and sends both maximal legal requests as the first POST.**

## Performance

- **Duration:** 24 min
- **Started:** 2026-09-26T03:45:15Z
- **Completed:** 2026-09-26T04:09:07Z
- **Tasks:** 3 (Task 1 tracer, Task 2 TDD, Task 3 auto)
- **Files modified:** 14 (10 created, 4 modified)

## Accomplishments

- **The tracer passed on its first run, cold-first included.** The test packs the tiny fixture with the production packer, loads it with `load_model_from_bytes`, and serves it through `start_loopback`. `start_loopback` uses the same `server_config()` the bootstrap uses, on `127.0.0.1:0`. `run_identity_probe` then sees:
  - one `classify` tool;
  - the labels in order;
  - identity equal to the sha256 of the packed bytes.

  Against a FRESH server, a `tools/call` with no `initialize` before it is served. pmcp 2.19.3's stateless mode has no handshake gate. Plan 08-10's cold verification depends on exactly this.
- **The S3 loader never touches disk and refuses every failure mode with a type.**
  - `head` reports no length: `MissingLength`, never read as 0.
  - The length is over the decide-apr-v1 cap: `TooLarge`, before `vec![0; len]` runs. A `u64::MAX` length under the contracted cap would abort the process if the check came after the allocation.
  - The fill: disjoint slices of one buffer, each filled in place by `S3Fetcher`'s streamed ranged `get_object`, 16 in flight.
  - Per part: 5 attempts, each cut at `ATTEMPT_TIMEOUT`, and an exact-length check.
  - The whole download: `DOWNLOAD_DEADLINE`, after which the load lock is released.

  Twelve behaviour tests cover it on an injected fetcher, with paused-clock tests for both timeouts.
- **The pin is checked before the ladder parses.** `resolve_from_fetcher` and `resolve_local` hash the buffer and compare it to the pin before `load_model_from_bytes`, then check `model.identity().artifact_sha256` against the hash again. The test proves the order: garbage under a wrong pin is refused as `HashMismatch`, and the same garbage under its own hash reaches the ladder and is refused as `Load`.
- **The local source is bounded before it is read.** The metadata length is checked against the cap before any read. The read then goes through decide-apr-v1's `read_decide_apr_bytes_bounded`, capped at `cap + 1`.
- **Proven to bite.** Four induced negatives each turn a named test red:
  - a short body accepted;
  - the pin not compared;
  - the overall deadline removed;
  - a missing length read as 0.

  The `Send` guard fails to compile against the closure-based stream it replaced.
- **Both maximal legal requests are built from the artifact's own `prepare`.** CONCENTRATED is 2 x 512 full rows for Laya-en. DISTRIBUTED is 8 texts totalling up to the budget. Under the CONTRACTED limits, both are served as the first and only POST over real loopback HTTP.

## Task Commits

1. **Task 1: tracer (bootstrap, lib, probe, register, README count)**: `df46a1c5c` (feat)
2. **Task 2: S3 cold-start loader**: RED `56c7f819c` (test) → GREEN `b89316621` (feat) → `b05d44b0c` (fix: Send across the download)
3. **Task 3: maximal-request probe CLI and deploy template**: `14f2622ca` (feat)

**Plan metadata:** recorded in the docs commit that carries this SUMMARY.

## Files Created/Modified

- `crates/aprender-mcp-decide-lambda/Cargo.toml`: `publish = false`, `[[bin]] bootstrap`, `aprender-decide` as a NORMAL dependency, and aws-config/aws-sdk-s3 `"1"` on the locked versions.
- `src/lib.rs`: the stateless config, the loopback, `ModelSource`, `Sha256Pin`, `resolve_*`, `LoadTimeline` (fetch/sha/build ms, RSS/peak RSS, Graviton generation), `LoadOnce`, and the cold-evidence helpers.
- `src/main.rs`: the `bootstrap` handler (GET health naming the package, OPTIONS, lazy load, proxy, `x-decide-load`, 503 kind-only).
- `src/s3.rs`: `RangeFetcher`, `S3Fetcher`, `DownloadPolicy`, `download_into_memory`, `S3LoadError`, and 12 tests.
- `src/probe.rs`: the identity and cold-first probes, `build_maximal_request`, and 7 tests.
- `src/tests.rs`: the loopback tests, the Send guard, and the source, probe-id and evidence tests.
- `examples/probe.rs`: the probe CLI.
- `.pmcp/deploy.toml.template`, `.pmcp/.gitignore`: the pmcp.run config template, with the generated file ignored.
- `crates/aprender-core/tests/monorepo_invariants.rs`: the register entry, and the baseline 8 → 9 citing D-15.
- `Cargo.toml`, `Cargo.lock`, `README.md`: the workspace member, and the crate count 89 → 90.

## Decisions Made

See `key-decisions` in the frontmatter. The FALSIFY-MONO-011 baseline raise needs human confirmation (coverage D6).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The bootstrap bin stopped compiling after GREEN, and the lib tests could not see it**
- **Found during:** Task 2 acceptance (the aarch64 zigbuild).
- **Issue:** `stream::iter(chunks_mut).map(closure).buffer_unordered(16)` inside the `resolve_model` future made rustc fail `run(service_fn(handler))` with "implementation of `Send`/`FnOnce` is not general enough". `cargo test --lib` never builds the bin, so all tests stayed green. The first zigbuild's background notification also reported exit 0 while its log held 8 errors, because the notification reports the shell wrapper's status, not cargo's. A stale `bootstrap` from another crate sat at the output path.
- **Fix:** A hand-rolled bounded `FuturesUnordered` with the same semantics. `resolve_model_future_is_send` pins the property in the lib tests; it was shown to fail to compile against the old form.
- **Files modified:** `src/s3.rs`, `src/tests.rs`, `Cargo.toml` (comment)
- **Verification:** the zigbuild log itself shows `Finished` and `zig_rc=0`, with no `error` lines. `file` reports an aarch64 ELF, and `strings` finds `aprender-mcp-decide-lambda` in it.
- **Committed in:** `b05d44b0c`

**2. [Rule 3 - Blocking] FALSIFY-MONO-011 baseline raised 8 → 9 alongside the register entry**
- **Found during:** Task 1.
- **Issue:** The ratchet caps the register's length, so the new entry fails the gate unless the baseline rises with it. The raise was announced in 08-06's comment.
- **Fix:** The baseline is now 9, with a comment citing 08-07 and D-15.
- **Committed in:** `df46a1c5c`

**3. [Interface detail] `RangeFetcher::fetch_range(start, dest: &mut [u8]) -> usize`**
- The plan says "fetch inclusive byte range -> bytes". The trait writes the inclusive range `start..=start+len-1` into the caller's slice instead, so the download is one buffer with no per-part copy. `ShortBody` is raised when the count differs from the slice length.

**4. [Plan text] The deploy template does not prescribe `--manifest-path crates/aprender-mcp-decide-lambda`**
- cargo-pmcp's resolver would ship another `*-lambda` binary from that root (D-ITEM-08-07-B). The template documents the trap and defers the root to 08-10, which already has a task for it.

**5. [TDD shape] Two of the twelve s3 tests were green at RED**
- `uri_parse_accepts_s3_and_refuses_the_rest` and `local_bounded_refuses_by_metadata_before_reading` exercise `parse_s3_uri` and `resolve_local`. The tracer had to implement both, because `from_env` and the Local source needed them. The other 10 failed on their target assertions. For example, `download_ok` failed with `left: [] right: [3, 20, 37, ...]`.
- `gsd check tdd-red-evidence` parses only Node/TAP output, so it cannot read cargo's. That evidence is recorded here instead.

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking), plus 3 documented plan-shape notes.
**Impact on plan:** The Send fix was necessary: without it the plan's own zigbuild criterion fails. No scope creep.

## Issues Encountered

- **The rtk hook compacts `git diff` output.** `git diff … | wc -c` returned 24,447 against 125,642 real bytes. The `actuals.tokens` figure above was measured with `env git -c core.pager=cat diff`.
- **`main.rs::handler` never ran under a Lambda runtime** (D-ITEM-08-07-A). Only its composed pieces are proven.
- **`.planning/WINDOWS.md` still refuses appends** (`Ledger entry 24 has invalid status: "resolved"`), so both new items are in deferred-items.md.
- **Lint scope.** `cargo clippy -p aprender-mcp-decide-lambda --all-targets --no-deps -- -D warnings` exits 0. A multi-crate clippy still dies on the pre-existing aprender-compute errors, which were not touched. `cargo fmt -p aprender-mcp-decide-lambda -p aprender-core -- --check` exits 0.
- **No AWS, no pmcp.run and no deploy happened.** Every S3 path was exercised through the injected fetcher.

## User Setup Required

None. No external service configuration required.

## Next Phase Readiness

- **08-10** can:
  - upload content-addressed to `s3://<bucket>/decide/<server>/<sha256>.apr`;
  - fill the three UNSET placeholders;
  - grant `s3:GetObject`;
  - probe with `examples/probe.rs`.

  It must settle the deploy root first (D-ITEM-08-07-B). After the deploy, a GET must name `aprender-mcp-decide-lambda`.
- **08-11** runs `probe --apr <real.apr> --maximal concentrated|distributed --cold-first` and correlates each `probe_id` with `decide.load performed_load=true`. The real `.apr` still waits on 08-08, which is HALTED at its human decision and was not touched here.

## Self-Check: PASSED

- FOUND: all 10 created files, and the 4 modified ones
- FOUND commits: df46a1c5c, 56c7f819c, b89316621, b05d44b0c, 14f2622ca (`git rev-list --count c9a639816..HEAD` = 5)
- Re-ran after the last code commit:
  - lib 27/27 (loopback 3, s3 12, probe 7, other 5);
  - monorepo_invariants 11/11, readme_contract 15/15;
  - clippy and fmt clean;
  - aws-sdk-s3 v1.137.0;
  - aarch64 zigbuild clean.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-26*
