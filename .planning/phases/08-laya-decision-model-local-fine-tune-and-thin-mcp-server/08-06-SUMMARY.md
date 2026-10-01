---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 06
subsystem: api
tags: [aprender-mcp-decide, mcp, pmcp, stdio, laya, decide-tool-boundary-v1, admission, tokio-semaphore, provable-contracts]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-01 decide-tool-boundary-v1 (constants, equations, FALSIFY-DECIDE-TOOL-001..009, staged bindings)"
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-02 laya_tiny run dir + oracle (max_len 64, an over-window task row)"
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-04/08-05 aprender-decide: Decider (ladder-only), prepare / classify_prepared, PreparedRow::tokens, identity(), Task::owned_labels, pack_run_dir, read_decide_apr_bytes_bounded"
provides:
  - "crates/aprender-mcp-decide (publish = false): lib aprender_mcp_decide + stdio bin aprender-mcp-decide"
  - "TOOL_NAME, CONTRACT, ClassifyArgs { texts } (deny_unknown_fields), ClassifyLimits::CONTRACTED"
  - "precheck(&ClassifyLimits, &ClassifyArgs) — count then bytes, no model; check_token_budget; classify_blocking(model, limits, texts)"
  - "Admission::new / try_admit -> Ticket::run_blocking (owned permits held until the blocking work returns); Busy; AdmissionError"
  - "ClassifyResponse { model: ModelView, labels, results: [ResultView] }, ClassifyFailure (budget -> validation, else internal)"
  - "ClassifyService::served (the one handler path, always CONTRACTED), tool_description(model), build_server(Arc<Model>, name, version)"
  - "load_model_from_path / load_model_from_bytes, ModelLoadError { Io, Artifact }, Model re-export (= aprender_decide::Decider)"
  - "FALSIFY-DECIDE-TOOL-001..008 bound to concrete tests; KANI-DECIDE-TOOL-001 names a runnable proptest"
affects: [08-07, 08-09, 08-11, 08-12]

actuals:
  tokens: 22872   # chars/4 over the realized diff (91489 chars, git diff 71e2306e5..HEAD)
  tasks: 2
  commits: 2
plan_head_before: 71e2306e5e70e1d5c792bfccb43fadc3a5b63116

tech-stack:
  added:
    - "aprender-mcp-decide crate: pmcp 2.19 (schema-generation, streamable-http), schemars 1.0, tokio; dev: sha2, tempfile, serde_yaml, proptest"
  patterns:
    - "Bounds split by cost: count and bytes on the async path (a signature with no model), tokenize + token budget + forward inside one admitted spawn_blocking"
    - "Process-wide admission from two tokio Semaphores; both OWNED permits move into the blocking closure, so a disconnect cannot free CPU still being spent"
    - "One handler path (ClassifyService::call) shared by the MCP tool and the unit tests; the only public constructor pins CONTRACTED"
    - "Tool description built from the artifact's task (question, ordered labels with criteria, bounds, one-document guidance)"
    - "Induced-negative table: each test group proven to bite by a mutation that turns a named test red"

key-files:
  created:
    - crates/aprender-mcp-decide/Cargo.toml
    - crates/aprender-mcp-decide/README.md
    - crates/aprender-mcp-decide/src/lib.rs
    - crates/aprender-mcp-decide/src/main.rs
    - crates/aprender-mcp-decide/src/tests.rs
    - crates/aprender-mcp-decide/tests/e2e_stdio.rs
  modified:
    - Cargo.toml
    - Cargo.lock
    - README.md
    - crates/aprender-core/tests/monorepo_invariants.rs
    - contracts/decide-tool-boundary-v1.yaml

key-decisions:
  - "FALSIFY-MONO-011 DEPLOYMENT_UNIT_BASELINE raised 7 -> 8 for aprender-mcp-decide under CONTEXT D-15. D-15's note 'no baseline change is expected' is factually wrong: the register's LENGTH is the ratcheted quantity, so any entry raises it. 08-07's lambda (also D-15) takes it to 9"
  - "ClassifyService::served is the only public constructor, and it pins ClassifyLimits::CONTRACTED. Shrunk limits reach the handler path only through a private with_limits that the in-crate tests use"
  - "A tokenizer refusal's detail is withheld from the response, because the tokenizers message may quote caller text (ASVS V7). Every other model failure is surfaced as an internal error"
  - "pmcp is declared as 2.19, the line Cargo.lock resolves (2.19.3) and the contract cites, rather than the plan's 2.9. sha2 is a dev-dependency only (an independent hash for the e2e identity check)"
  - "TOOL-004 binds three legs: aprender-decide's stance order (none, against, favor) standalone (preserve_order OFF) and with --features serde-preserve-order (ON), plus this crate's response-order test (ON via pmcp)"

patterns-established:
  - "Thin decide server = precheck (no model) -> try_admit -> Ticket::run_blocking(classify_blocking) -> ClassifyResponse; the Lambda crate reuses build_server + load_model_from_bytes"
  - "Admission tests drive Admission directly with parked blocking jobs (oneshot 'started' + std mpsc 'release'). Every wait is bounded, and a panicking test drops the release sender so no blocking thread outlives it"

requirements-completed: [D-09, D-10, D-11, D-12, D-15]

coverage:
  - id: D1
    description: "An MCP client spawning `aprender-mcp-decide --model <tiny.apr>` over live stdio sees exactly one strict `classify` tool whose description states the artifact's question and its labels in task order; a call on the 5 oracle task rows returns model.artifact_sha256 == sha256 of the served file, recipe_id == sha256(recipe.json), labels [shipping, billing, account], probabilities within probs_abs (max|d| 2.98e-8), argmax labels, tokens == oracle row length, truncated == oracle (the over-window row comes back true)"
    requirement: "D-09"
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-decide/tests/e2e_stdio.rs#the_tiny_decide_server_classifies_over_live_stdio"
        status: pass
    human_judgment: false
  - id: D2
    description: "The served process enforces the CONTRACTED bounds: live 9-text call refused naming classify_max_texts; caller-supplied `labels` refused end to end; ClassifyService::served limits == CONTRACTED; every Rust bound and the admission sizes equal decide-tool-boundary-v1 constants; the declared token budget fits the Lambda envelope"
    requirement: "D-10"
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-decide/tests/e2e_stdio.rs#the_tiny_decide_server_classifies_over_live_stdio"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#bounds_match_contract"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#served_service_uses_contracted_limits"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#token_budget_fits_the_envelope"
        status: pass
    human_judgment: false
  - id: D3
    description: "Each request bound refuses at N+1 and accepts at N: 0 and 9 texts refused, 8 classified; 16385 bytes (and a multi-byte text whose char count fits) refused naming index and key, 16384 bytes tokenized and truncated; built-row sum at the (shrunk) budget accepted, one over refused with the sum and per-text lengths; every refusal class is validation-class, names the contract and key, and omits the caller's text; first failing bound in order count -> bytes -> tokens is the one named (proptest)"
    requirement: "D-10"
    verification:
      - kind: unit
        ref: "cargo test -p aprender-mcp-decide --lib (25 passed): empty_texts_refused_naming_min_texts, nine_texts_refused_naming_max_texts, eight_texts_accepted_and_classified, oversized_text_refused_without_echo, multibyte_text_over_byte_limit_refused, text_at_byte_limit_is_tokenized_and_truncated, token_budget_accepts_exactly_at, token_budget_refuses_one_over, every_refusal_names_key_without_text, bound_order_is_count_bytes_tokens"
        status: pass
    human_judgment: false
  - id: D4
    description: "Admission is process-wide and counts real CPU work: pending+1 refused at once naming classify_max_pending; a caller dropped mid-computation keeps both slots until its blocking work returns; a caller dropped while waiting releases its pending slot and never runs; precheck's signature takes no model"
    requirement: "D-10"
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#admission_refuses_over_pending, #admission_slot_held_until_blocking_ends, #admission_waiting_cancel_frees_pending, #admission_limits_match_contract, #precheck_takes_no_model"
        status: pass
    human_judgment: false
  - id: D5
    description: "Response shape (D-11, D-12): top-level model identity, labels in task order (not sorted), per-result label == labels[argmax], K probabilities summing to 1 within 1e-6, arrays never maps; truncation flag false when a text exactly fills the room and true one filler word later (both at the 64-token window)"
    requirement: "D-11"
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#response_labels_follow_task_order"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#truncation_flag_flips_one_past_the_room"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide/src/tests.rs#long_text_truncated_short_text_not"
        status: pass
    human_judgment: false
  - id: D6
    description: "aprender-mcp-decide is registered in FALSIFY-MONO-011 deployment_unit_bins with publish = false (baseline 7 -> 8 under D-15); README workspace-crate count 89 == cargo metadata"
    requirement: "D-15"
    verification:
      - kind: integration
        ref: "cargo test -p aprender-core --test monorepo_invariants --test readme_contract (11 + 15 passed)"
        status: pass
    human_judgment: true
    rationale: "The baseline raise follows from D-15, but D-15 itself said 'no baseline change is expected'. A human should confirm that reading of the recorded decision before 08-07 raises the baseline again"
  - id: D7
    description: "FALSIFY-DECIDE-TOOL-001..008 carry test: lines bound to concrete test names (TOOL-009 stays LIVE-PENDING); on a lifted copy the guard resolves 644 refs with 0 dangling in decide-tool-boundary-v1, and two mutated names are flagged"
    requirement: "D-10"
    verification:
      - kind: other
        ref: "cargo run -p aprender-contracts-cli --bin pv -- validate contracts/decide-tool-boundary-v1.yaml -> 0 errors, 0 warnings"
        status: pass
      - kind: other
        ref: "bash scripts/check_contract_test_binding.sh -> VACUOUS (pre-existing D-ITEM-08-01-A); measured on a lifted copy instead (D-ITEM-08-06-A)"
        status: unknown
    human_judgment: true
    rationale: "The plan's strict-binding PASS line cannot be printed while spectral-indices-v1 (another phase) makes the guard skip; the lifted-copy measurement is evidence, not the gate itself"

duration: 21min
completed: 2026-09-26
status: complete
---

# Phase 8 Plan 06: Thin stdio decide MCP server (`aprender-mcp-decide`) Summary

**`aprender-mcp-decide` serves one verified decide `.apr` behind one `classify` tool. The tool takes only `texts`: its question and ordered labels come from the artifact's own task. Every response carries the served file's sha256 and recipe_id, plus per-text argmax label, calibrated probabilities as arrays in task order, `tokens` and `truncated`. The bounds are split by cost. Count and bytes are checked on the async path by a function that takes no model. Tokenization, the token budget and the forward all run inside one admitted `spawn_blocking` section. Process-wide admission holds its slots until the CPU work actually ends.**

## Performance

- **Duration:** 21 min
- **Started:** 2026-09-26T03:21:47Z
- **Completed:** 2026-09-26T03:42:06Z
- **Tasks:** 2 (Task 1 tracer, Task 2 TDD expansion)
- **Files modified:** 11 (6 created, 5 modified)

## Accomplishments

- **The tracer passed on its first run** over live stdio. The test packs the checked-in tiny fixture with the production packer, writes it to a temp file, spawns `env!("CARGO_BIN_EXE_aprender-mcp-decide")` and runs initialize, tools/list and tools/call:
  - exactly one tool, with `additionalProperties: false`;
  - the description contains the artifact's question and `shipping, billing, account` in order;
  - the identity is `37d65159…`, the 08-05 golden, and equals an independent sha256 of the file;
  - on the 5 oracle task rows the maximum probability difference is **2.98e-8** (probs_abs 1e-5), the argmax labels match, and `tokens` equals the oracle's row lengths;
  - the oracle's over-window row comes back `truncated: true`.

  The same session proves the served bounds are the contracted ones (9 texts is refused naming `classify_max_texts`) and that a caller-supplied `labels` key is refused.
- **Bounds split by cost.** `precheck(&ClassifyLimits, &ClassifyArgs)` checks the count, then per-text UTF-8 bytes. It has no model parameter, and `precheck_takes_no_model` pins that with a function-pointer type ascription. `classify_blocking` runs `prepare` once, then the sum of `PreparedRow::tokens` against the budget, then `classify_prepared`, all inside the admitted blocking section.
- **Admission counts real work.** Both owned permits move into the `spawn_blocking` closure. The tests show:
  - pending+1 is refused at once, naming `classify_max_pending`;
  - a caller aborted mid-computation keeps both slots, and a probe's work does not start until the parked job is released;
  - a caller dropped while waiting frees its pending slot and never runs its closure.
- **Refusals never echo text.** Every bound refusal and the busy refusal are `pmcp::Error::Validation`. Each names `contracts/decide-tool-boundary-v1.yaml`, the key and the observed value, and a test checks that a distinctive caller string is absent in every class. The tokenizer's own error detail is withheld, because it may quote input.
- **Contract mirror and bindings.** `bounds_match_contract` and `admission_limits_match_contract` read every constant with a local `constant_u64`. `token_budget_fits_the_envelope` checks the second proof obligation from the YAML. The `bound_order_is_count_bytes_tokens` proptest is now the runnable evidence the KANI-DECIDE-TOOL-001 entry names. FALSIFY-DECIDE-TOOL-001..008 carry `test:` lines bound to concrete test names, and TOOL-009 stays `LIVE-PENDING`.
- **Proven to bite, not just green.** Twelve induced negatives were each applied to a saved copy and reverted, and every one turns a named test red:
  - byte bound off-by-one and byte bound removed;
  - permits dropped before the blocking work, and a detached wait;
  - pending oversized, max_texts changed to 9, budget off-by-one;
  - tokenizer detail leaking, labels sorted, guidance dropped;
  - the YAML budget edited alone, and the refusal echoing text.

## Task Commits

1. **Task 1: tracer (crate, stdio bin, e2e, register, README count)**: `3b009ad68` (feat)
2. **Task 2: boundary tests, induced negatives, contract bindings**: `4eb91e840` (test)

**Plan metadata:** recorded in the docs commit that carries this SUMMARY.

## Files Created/Modified

- `crates/aprender-mcp-decide/Cargo.toml`: `publish = false` deployment unit, lib + bin, pmcp 2.19.
- `crates/aprender-mcp-decide/src/lib.rs`: the args, limits, precheck, token budget, blocking section, admission, response, description, service, load doors and `build_server`.
- `crates/aprender-mcp-decide/src/main.rs`: `--model` / `APRENDER_DECIDE_MODEL`, stderr-only, exit code 2 on usage errors, logs the identity and load time.
- `crates/aprender-mcp-decide/src/tests.rs`: 25 lib tests, including 4 admission tests and 1 proptest.
- `crates/aprender-mcp-decide/tests/e2e_stdio.rs`: a live stdio leg on the tiny fixture that always runs, plus the `APR_MCP_E2E_DECIDE_MODEL` leg, which prints SKIP when the variable is unset.
- `crates/aprender-mcp-decide/README.md`: run, tool table, bounds and tests.
- `crates/aprender-core/tests/monorepo_invariants.rs`: register entry, and baseline 7 → 8 citing D-15.
- `contracts/decide-tool-boundary-v1.yaml`: TOOL-001..008 `test:` lines, and the KANI harness now names the proptest.
- `Cargo.toml`, `Cargo.lock`, `README.md`: workspace member, and the crate count 88 → 89.

## Decisions Made

See `key-decisions` in the frontmatter. The one that needs a human eye is the FALSIFY-MONO-011 baseline (next section, deviation 1).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] FALSIFY-MONO-011 baseline raised 7 → 8 alongside the register entry**
- **Found during:** Task 1 (monorepo_invariants verify)
- **Issue:** The plan says to add `aprender-mcp-decide` to `deployment_unit_bins`, and CONTEXT D-15 authorizes it, adding "no baseline change is expected". But `DEPLOYMENT_UNIT_BASELINE` caps the register's length at 7, so any new entry fails the gate with "grew to 8".
- **Fix:** Raised the baseline to 8, with a comment citing D-15 and noting that 08-07's lambda (also D-15) will take it to 9. The chronos-lambda precedent (`fdf6b1802`) did the same thing in the same commit as its crate.
- **Files modified:** crates/aprender-core/tests/monorepo_invariants.rs
- **Verification:** monorepo_invariants passes 11/11. With the entry but without the raise, it failed exactly on the ratchet.
- **Committed in:** 3b009ad68

**2. [Rule 1 - Bug, in the test] `admission_waiting_cancel_frees_pending` did not prove B was waiting**
- **Found during:** Task 2 (induced-negative table, mutation M10 "detached wait" survived)
- **Issue:** The test aborted B's spawned task after a `yield_now`. An unpolled future drops its ticket trivially, so a regression that detaches the wait into its own task passed.
- **Fix:** The test now drives B's pinned future itself. A bounded poll must time out while A holds the slot, and only then is the future dropped. M10 now fails that test.
- **Files modified:** crates/aprender-mcp-decide/src/tests.rs
- **Committed in:** 4eb91e840

**3. [Plan shape] RED could not precede GREEN for Task 2**
- **Found during:** Task 2
- **Issue:** Task 1's tracer action specified, and shipped, the whole implementation (precheck, admission, response, description), so the Task 2 tests passed on their first run.
- **Substitute evidence:** The induced-negative table stands in for the missing RED. It has 12 mutations, each killed by a named test, listed above.

**4. [Minor] Dependency lines**
- pmcp is declared as `2.19`, not the plan's `2.9`. Both resolve to the locked 2.19.3, and 2.19 is the version the contract cites. `sha2` is a dev-dependency rather than a normal one, because only the e2e uses it, as an independent hash. `proptest` was added as a dev-dependency for KANI-DECIDE-TOOL-001's evidence, which the contract already attributed to this plan.

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 test bug) plus 2 documented plan-shape notes.
**Impact on plan:** The baseline raise is the one item a human should confirm (coverage D6). No scope creep.

## Issues Encountered

- `bash scripts/check_contract_test_binding.sh` is still VACUOUS: `spectral-indices-v1` has no `kani_harnesses` (D-ITEM-08-01-A). I measured on a lifted copy instead. It resolved 644 refs, which is 622 plus exactly this plan's 22, with 0 dangling in decide-tool-boundary-v1, and two mutated names were flagged (D-ITEM-08-06-A). `.pv/lint-previous.json` was restored.
- `cargo fmt --all -- --check` still fails only on the pre-existing `fdf6b1802` files (D-ITEM-08-04-B). `cargo fmt -p aprender-mcp-decide -- --check` exits 0.
- `cargo clippy -p aprender-mcp-decide --all-targets --no-deps -- -D warnings` exits 0. A multi-crate clippy still dies on pre-existing aprender-compute errors (the orchestrator's caveat), and I did not touch that.
- `.planning/WINDOWS.md` refuses every append (`Ledger entry 24 has invalid status: "resolved"`), so the unrun-verify item is recorded in deferred-items.md.
- README's layout tree still says "(82 crates total)", which no gate reads. It is logged as D-ITEM-08-06-B and not fixed.
- The e2e test is dark in CI until plan 08-12 adds it to the ci.yml `--test` line. The real-model leg SKIPs until 08-09 packs a real decide `.apr`; none exists yet (only the halted 08-08 run dir).

## User Setup Required

None. No external service configuration required.

## Next Phase Readiness

- 08-07 (Lambda) can wrap `build_server` and `load_model_from_bytes`, and reuse `ClassifyLimits::CONTRACTED` for its maximal-request probe. It must also add `aprender-mcp-decide-lambda` to the register and raise the baseline to 9.
- 08-09 can arm the real-model e2e leg with `APR_MCP_E2E_DECIDE_MODEL`.
- 08-08 remains HALTED at its human decision (stance ECE). This plan did not touch it.

## Self-Check: PASSED

- FOUND: crates/aprender-mcp-decide/{Cargo.toml, README.md, src/lib.rs, src/main.rs, src/tests.rs, tests/e2e_stdio.rs}
- FOUND commits: 3b009ad68, 4eb91e840 (`git rev-list --count 71e2306e5..HEAD` = 2)
- Re-ran: e2e 2/2, lib 25/25 (4 admission), monorepo_invariants 11/11, readme_contract 15/15, clippy and fmt on the crate clean, pv validate 0/0

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-26*
