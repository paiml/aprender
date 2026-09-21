# PMAT-3719 implementation receipt

Issue #3719: `apr code` on Qwen3.5 Q4_K_M (4B, 9B), on CUDA, on lambda and gx10. Does it complete a scripted edit-and-verify task?
Branch `PMAT-3719-apr-code-qwen35-cuda`. Author: aprender-f8 (routed by the cop, aprender-04). Evidence: `evidence/apr-code-3719/`.

## done_when 1: the gap, measured and posted

- **Fixture:** `tests/fixtures/apr-code-edit-verify/`. `test_mean` fails, and the fix is one line in `stats.py`; `task.txt` asks for the fix and the test run.
- **Harness:** `scripts/apr_code_edit_verify.sh`. It runs `apr code -p --model M --project <copy> --output-format json --emit-trace <t>` under the fleet GPU lock (bounded flock, or `--gpu-q PRIO` with `GPUQ_WAIT`), choom 1000.
- **Judge:** `scripts/lib/apr_code_edit_verify.py` reads artifacts only:
  - the working-copy diff;
  - an independent test re-run;
  - python shims on the agent's PATH (`--emit-trace` never records a tool call);
  - the serve child's own output, captured through an `APR_BIN` wrapper.
- **Baseline gap** (`evidence/apr-code-3719/baseline-856009cc9/`), binary `apr 0.69.0 (856009cc9)` = main `52f43da71` + #3726 `c57c260bc` (the cop's rule: measure with #3726 in). All 4 cells FAIL "serve child did not load": `Model ready: 0 layers` · `gpu-layers: resolved=0 total=0`, then `CUDA optimized model ready`, then HTTP 500. That is #3571. Posted as https://github.com/paiml/aprender/issues/3719#issuecomment-5765960437.

## done_when 2: every PASS cites the CUDA forward

Final binary: `apr 0.69.0 (7b161d743)` = main `52f43da71` + #3726 + aprender-c7's #3571 step (2) `5a4a8e102` + this branch. Every PASS cell's `evidence` is the serve child's own line `Model ready: Qwen3.5 hybrid, 32 layers resident on the GPU, declared context 262144 tokens`, plus `gpu-layers: requested=all resolved=32 total=32 (backend=cuda)`, zero `[GPU->CPU FALLBACK]` lines, and a completion that came back. The judge reports `backend: cuda` only when all layers are resident on CUDA AND an answer returned. `CUDA optimized model ready` alone was measured to be printed for a 0-layer model, so it is not accepted.

| Cell | v6 `7b161d743` |
|---|---|
| lambda · Qwen3.5-4B-Q4_K_M | PASS |
| lambda · Qwen3.5-9B-Q4_K_M | PASS |
| gx10 · Qwen3.5-4B-Q4_K_M | PASS |
| gx10 · Qwen3.5-9B-Q4_K_M | PASS |

Honest limit: the route prints no per-request CUDA line, so the evidence is residency plus completions with no fallback marker. A per-request `prompt_tokens=N` line is promised by aprender-c7 after its receipt; until then rows are keyed `context: task`.

## done_when 3: every FAIL was fixed here (falsifier RED on the pre-fix code) or points to its issue

| Mechanism measured | Where | Fix |
|---|---|---|
| serve child did not load (0 layers → HTTP 500) | baseline, all 4 | #3571 (aprender-c7), under Refs |
| model never told it has tools: `-p` sent the system prompt "Answer the question. Be direct." to EVERY model (fake-serve capture) | gx10·4B on the #3571 route | `b2b89d69e`; `falsify_3719_single_prompt_run_tells_the_model_about_its_tools`, RED with the override restored |
| tool call not parsed: `<tool_call>` JSON missing only its outer `}` | gx10·4B v4 | `a35f7b8a0`; `falsify_3719_delimited_tool_call_missing_outer_brace_is_executed`, RED with the repair disabled |
| wrong edit: the prompt taught `file_edit` `old`/`new` (tool requires `old_string`/`new_string`) and `memory` `key`/`value` (requires `content`); the model repeated a rejected call until the loop guard ended the turn (logging-proxy capture) | lambda·4B v4/v5 | `ed714f055`; `falsify_3719_prompt_tool_examples_supply_every_required_field`, RED with 5 findings before |

## done_when 4: the cells become release-matrix rows

- The rows use the keys aprender-62 (#3712) and aprender-97 (#3715) agreed:
  - the v2 cell fields;
  - `thinking` read from the serve child's `chat template: … (thinking off|on)` line;
  - `context` of `4k` only on a measured per-request `prompt_tokens ≥ 4096`, else `task`;
  - `max_tokens` 1024.
- aprender-62's ruling: the ladder (#3712 row B2) calls this harness outside `apr_locked` with `--gpu-q 1`, once per derived code cell, and writes the ON row for a both-mode model as `verdict: error`, reason "apr code has no thinking toggle (#3723)". B2 waits on #3745.
- 62's condition: the harness must prove its `apr` call is locked, choom'd and bounded. `scripts/check_apr_code_edit_verify.sh` (dispatched by `guard_tree --no-cargo`) runs the real harness against a fake `apr` and a scratch lock, in all gate modes. Rows: free lock gives `held=yes oom=1000`; held lock gives a DECLINE within the bound. Mutants each turn it red: choom dropped, gpu-q bound dropped, `--caps` check dropped, a double lock.
- #3715's shape already derives verb `code` for every inventory model × host. A missing or non-pass row refuses the tag.

## Checks
- `cargo test -p aprender-orchestrate --lib agent::`: 888 passed. `cargo clippy -p aprender-orchestrate --lib --tests -- -D warnings`: clean. `cargo fmt --check`: clean.
- `scripts/check_apr_code_edit_verify.sh`: 30 rows ok, 0 fail, mutation-checked.
- bashrs: 0 errors. `check_apr_bin_pinned`, `check_no_competing_harnesses`, `check_no_pipe_into_grep_q`, `check_guards_are_wired` and `check_roadmap_fragment_required`: pass.
- `scripts/guard_tree.sh --no-cargo` at `b9d78124c`: 77 checks, 1 failed (`check_complexity_ratchet.sh`: the tool-call repair grew `parse_tool_calls_envelope` and added an over-threshold function). Split in `e060bc114`; the ratchet then reads "PASS (D2): none new, none grown".

## Out of scope, noted
- `apr code -p --output-format json` printed nothing on a driver error: #3775 (aprender-f8, separate branch).
- `--emit-trace` never records tool calls: filed by the cop for 0.70.0.
