# PMAT-3775 implementation receipt

Issue #3775: `apr code -p --output-format json` printed no JSON document on a driver error (rc 1, stderr only).
Branch `PMAT-3775-apr-code-json-error-envelope`. Author: aprender-f8.

## Schema
These are the #3720 response contract's fields: aprender-f8 proposed them on #3720 (comment 5767761121), and aprender-fd, #3720's owner, accepted them as written (comment 5767765474), with one precision: `empty_completion` uses the command's own inference-failure exit code (apr code: 1).

- `status`: `ok` | `refused` | `failed`, on every document.
- `error`: `{kind, message, exit_code}`, iff `status != ok`.
  - `kind` is the snake_case error variant.
  - `exit_code` is the process's real exit status.
- The existing Claude-Code-parity fields stay, with `is_error == (status != ok)`.

## done_when → evidence

1. **Every exit writes exactly one JSON document, with #3720's field names.**
   - `crates/aprender-orchestrate/src/agent/code_envelope.rs`: `CodeOutcome` and `envelope()`, replacing `build_json_result_envelope`.
   - `code.rs`, `run_single_prompt`: a driver or agent error writes a failed or refused document (`capability_denied` is a refusal). An empty completion is `failed`/`empty_completion` and exits 1; before, it printed a document and exited 0.
   - An error `cmd_code` RETURNS is written by its caller. apr-cli's `dispatch.rs` calls the new `pub emit_error_document`, passing the exit code that error becomes (`CliError::exit_code_value()`, 1 for `Aprender`). batuta's own `cli/code.rs` always passes `"text"` and owes no document.
   - The early refusals leave through `anyhow::bail!(CodeOutcome …)`, which keeps them downcastable (test `a_bailed_outcome_survives_as_a_downcastable_error`). Kinds: `invalid_input`, `max_turns_exhausted`, `hook_blocked`. Anything untyped is `agent_error`.
   - The no-model exit writes `no_model` with exit 5 before `process::exit` (`exit_no_model`).
   - `cmd_code` keeps its name, and every addition is a helper call (`json_document_mode`, `max_turns_refusal`, `exit_no_model`). `check_complexity_ratchet.sh`: "PASS (D2): none new, none grown". A first version had renamed the body `cmd_code_inner`, and the ratchet read that as a NEW over-threshold function.
   - `capacity` ("where #3720 / #3596's CapacityRefused applies"): nothing applies yet. At the base (`a9502d992`) no `CapacityRefused`/`capacity_refused` exists anywhere in `crates/` or `contracts/`. #3596 is a time-to-first-token issue, not a capacity type. `apr code` receives any serve refusal only as an HTTP error string. The field is part of the accepted #3720 schema (present iff `kind == "capacity_refused"`), but this diff has NO code path that emits it, because there is no capacity kind for it to accompany. Whoever introduces a `capacity_refused` kind adds the field with it.
2. **The three cases each give a parseable document, rc ≠ 0, and a test that is RED at 0.69.0.** `code_tests.rs` runs the EXISTING `run_single_prompt` in a child test process and parses its whole stdout:
   - `falsify_3775_driver_error_writes_one_json_document`: the serve child fails, or loads 0 layers and answers HTTP 500 (#3571). Expects `inference_failed`, exit 1.
   - `falsify_3775_tool_error_writes_one_json_document`: a tool that keeps failing. Measured, not assumed: in `-p` mode the loop guard BLOCKS the third identical call and the turn ends with no answer (4 iterations, 3 tool calls), so the kind is `empty_completion`, exit 1.
   - `falsify_3775_empty_completion_is_a_failure_document`: `empty_completion`, exit 1.
   - **Measured RED on the pre-fix code** (base `a9502d992` with only the test code appended): all 3 FAILED. driver_error: "found []" (empty stdout). tool_error and empty: "a failed run must not exit 0".
   - The "model refused (0 layers)" case reaches `apr code` as the driver error above. No separate capacity path exists in `apr code` (see `capacity` under done_when 1).
3. **Nothing but the JSON document on stdout; diagnostics go to stderr.** End to end with the built binary (`apr 0.69.0 (4ca7fdcc9)`, debug):
   - A fake `apr serve` child that answers every completion with HTTP 500 gives rc 1. Stdout is exactly 1 line: `{"status":"failed",…,"error":{"kind":"network_error",…,"exit_code":1}}`. The `Launched`/`ready`/`Error:` lines are on stderr.
   - No model gives rc 5 and 1 line: `status:"refused"`, `kind:"no_model"`, `exit_code:5`.
   - Contrast: the same HTTP-500 fake on a binary without this change (`7b161d743`) gives rc 1 and **0 bytes** on stdout.
   - Early refusals (`apr 0.69.0 (5f626637c)`):
     - `--max-turns 0` gives rc 1 and 1 line: `refused` / `max_turns_exhausted` / `exit_code:1`.
     - `--project /nonexistent` gives rc 1 and 1 line: `refused` / `invalid_input` / `exit_code:1`.

## Checks
- `cargo test -p aprender-orchestrate --lib agent::`: 893 passed.
- `cargo clippy -p aprender-orchestrate --lib --tests -- -D warnings`: clean. `cargo clippy -p apr-cli --lib -- -D warnings`: clean. `cargo fmt --check`: clean.
- `contracts/apr-code-parity-v1.yaml` 5.5.0: the non-interactive-mode row states the rule, and FALSIFY-CODE-PARITY-006 names the tests. `pv validate` and `pv lint` pass.
- `scripts/guard_tree.sh --no-cargo` at `4ca7fdcc9`: only `check_complexity_ratchet.sh` failed (the `cmd_code_inner` rename, above). At HEAD, `check_complexity_ratchet.sh` passes and `check_roadmap_fragment_required.sh` passes.

## Not in scope
- The HTTP-500 driver error is classified `network_error`, the `DriverError` variant `AprServeDriver` uses for any non-2xx. That is the variant name, per the schema; renaming the variant is not this issue.
