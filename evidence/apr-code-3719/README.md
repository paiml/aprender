# #3719: `apr code` edit-and-verify on Qwen3.5 Q4_K_M, CUDA, lambda + gx10

Each cell is one run of `scripts/apr_code_edit_verify.sh --model <M> --host <H> --out <dir>`
against the fixture `tests/fixtures/apr-code-edit-verify` (one failing test, a one-line fix).
`cell.json` is the verdict of `scripts/lib/apr_code_edit_verify.py`, re-run over each directory's
artifacts with the judge at `005d53f79` (md5 `72fab3e4cd98b8f17ed1ee18c2a4b032` on both hosts).
The judge reads only artifacts, never the agent's claims:

- the working-copy diff;
- an independent test re-run;
- the python-shim log;
- the serve child's own stdout and stderr.

Case table: `scripts/check_apr_code_edit_verify.sh`.

## baseline-856009cc9: the gap (#3719 done_when 1)

Binary `apr 0.69.0 (856009cc9)`, built with `--features cuda`. It is a local merge, never pushed:
origin/main `52f43da71` + `PMAT-3726-canonical-bpe` `c57c260bc` (the cop: measure on a binary that
includes #3726) + the harness `e63e90849`. No 0.69.1 tag or version bump existed at measurement time.

| Model | lambda (RTX 4090) | gx10 (GB10) |
|---|---|---|
| Qwen3.5-4B-Q4_K_M | FAIL: serve child did not load | FAIL: serve child did not load |
| Qwen3.5-9B-Q4_K_M | FAIL: serve child did not load | FAIL: serve child did not load |

All four fail the same way. The serve child printed `Model ready: 0 layers` and
`gpu-layers: requested=all resolved=0 total=0 (backend=cuda)`. It then printed
`CUDA optimized model ready` anyway, and answered the first completion with HTTP 500:
`Operation 'generate_gpu_resident_streaming' not supported: Model architecture not supported for
GPU-resident path`. Fix: #3571 (step 1 refuses the 0-layer load; step 2 adds the Qwen3.5 route).

No CUDA forward ran in any cell. `CUDA optimized model ready` is a load-time line, and it is printed
for a model with no layers.

## route-cc3892acd: the next mechanism

Binary `apr 0.69.0 (cc3892acd)` = the baseline + aprender-c7's #3571 step (2), branch
`PMAT-3571-serve-qwen35-session-v2` @ `5a4a8e102` (pre-quorum-receipt), + the harness `005d53f79`.

gx10 / Qwen3.5-4B:

- **Serve child:** loaded, `Model ready: Qwen3.5 hybrid, 32 layers resident on the GPU` ·
  `chat template: Qwen3NoThink (thinking off)` · `gpu-layers: requested=all resolved=32 total=32
  (backend=cuda)`. The model then answered coherently.
- **Agent:** made no tool call and left `stats.py` unchanged. Its answer began "Without seeing the
  actual code, I'll assume…". The run was `num_turns 1` and `tokens_in 81`.
- **Cause:** `fake-serve-request-capture.jsonl` shows what `apr code -p` sent. The system message
  was `Answer the question. Be direct.` (31 chars): `run_single_prompt` replaced the coding prompt
  with `COMPACT_SYSTEM_PROMPT` for every model, so no model was told it had tools.
- **Fix:** in this PR (`b2b89d69e`), with falsifier
  `falsify_3719_single_prompt_run_tells_the_model_about_its_tools`. It is RED with the old
  override restored.

## v6-7b161d743: all four cells PASS (#3719 done_when 1–3)

Binary `apr 0.69.0 (7b161d743)`, `--features cuda`, the same SHA on both hosts. It is a local merge, never pushed: the baseline + aprender-c7's #3571 step (2) `5a4a8e102` + this branch through `ed714f055`. The harness ran through `gpu-q --prio 1` (`GPUQ_WAIT` bound).

| Model | lambda (RTX 4090) | gx10 (GB10) |
|---|---|---|
| Qwen3.5-4B-Q4_K_M | **PASS** (5 turns, 81 s) | **PASS** |
| Qwen3.5-9B-Q4_K_M | **PASS** | **PASS** (4 turns, 214 s) |

Each PASS cell carries:

- **the serve child's own lines:** `Model ready: Qwen3.5 hybrid, 32 layers resident on the GPU, declared context 262144 tokens` · `chat template: Qwen3NoThink (thinking off)` · `gpu-layers: requested=all resolved=32 total=32 (backend=cuda)`, and zero `[GPU->CPU FALLBACK]` lines;
- **the edit:** exactly `- return sum(values) / (len(values) - 1)` → `+ return sum(values) / len(values)`, with `test_stats.py` untouched and no files added;
- **the test run:** the agent's own `python3 -m unittest test_stats -v` (shim log), and the independent re-run `OK`;
- **the answer:** it names the off-by-one and reports the passing tests.

The fixes between route-cc3892acd and v6, each with a falsifier that was RED on the pre-fix code:

1. `b2b89d69e`: `-p` stopped replacing the coding prompt with "Answer the question. Be direct.".
2. `a35f7b8a0`: a delimited `<tool_call>` missing only its closing brackets is executed.
3. `ed714f055`: the prompt teaches `file_edit` `old_string`/`new_string` and `memory` `content`, the fields the tools require. Before, Qwen3.5-4B repeated a rejected `old`/`new` call until the loop guard ended the turn; a logging proxy in front of the serve child showed it.

Rows are keyed `thinking: off` (read from the child) and `context: task` (the route prints no per-request prompt size yet). The ON row that `apr code` cannot produce is written by the #3712 ladder as `verdict: error`.
