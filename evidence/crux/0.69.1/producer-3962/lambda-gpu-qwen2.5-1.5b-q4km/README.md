# #3962 producer receipt: CODE + SERVE verbs, lambda gpu lane

The first real run of the route-derived serve cell and the `apr code` cell
(`scripts/lib/crux_cells_serve_code.sh`). This is the PRODUCER's receipt: its rows,
graded by the prompt-set oracle. It is NOT a release verdict: no certification
receipt exists yet (#3962 Q1), and #3957's judge refuses without one.

| | |
|---|---|
| harness | `crux-3962-code-serve-producer` @ `6e294e1a0` (`scripts/crux_inference_dogfood.sh`, as launched) |
| apr | `apr 0.69.1 (6e294e1a0)`, `--features cuda`, pinned by `scripts/apr_bin.sh` |
| host / lane | lambda, NVIDIA GeForce RTX 4090, `--backend gpu`, every cell through `gpu-q --prio 1` (cop ruling) |
| model | `qwen2.5-1.5b-instruct-q4_k_m.gguf`, sha256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| comparators | llama.cpp build 10987 (`d1d3c3396`, the pin); ollama server 0.34.2. hf/vllm were not requested; their drivers are on #3952 branches |
| prompts | `prompts.json` = prompt-set v2 @ `682eacc28fbc` (aprender-dd), restricted to its `code` / `serve run` / `serve stream` verbs; `run` kept only on the control. sha256 `18a0b7886728…` |
| grading | `grade.txt`: every row through `crux_oracles.evaluate` (feat/3962-crux-prompt-certification @ `a8621dbf4`) |
| thinking | off (the only lane this slice drives). apr serve has no per-request toggle; the comparators were sent `chat_template_kwargs.enable_thinking=false` |

**Driver note.** The cells ran to completion and every row was appended to
`manifest.jsonl`. After that the driver died, before the judge step, with a bash syntax
error: I edited `crux_inference_dogfood.sh` in the worktree while it was running, and
bash reads a script incrementally. So no `collect` output exists. The rows are
complete; the judge step never ran.

## Coverage

- **Route universe.** apr's own `GET /` listed **40 routes**. **11** are generation
  routes and were driven in every mode their wire has. **0** were unclassified.
  The rest are recorded with their reasons in the `serve_routes` row of
  `manifest.jsonl`.
- **Rows.** 270 apr serve rows (18 prompts × 15 route-modes) and 4 apr code rows.
  Per prompt, llama.cpp and ollama each gave 2 serve rows (OpenAI chat
  route, non-stream + stream) and 1 code row.

## Result (from `grade.txt`)

| | apr | llama.cpp | ollama |
|---|---|---|---|
| `code` (4 prompts, EXECUTED against the asserts) | **4/4** | 4/4 | 4/4 |
| `serve run` OpenAI chat | 1/18 | 10/18 | 10/18 |
| `serve stream` OpenAI chat | 1/18 | 10/18 | 10/18 |

apr is right on 1/18 on EVERY answering serve route. It is wrong mostly as
`no_answer_tag`: it answers `4`, `The capital city of Japan is Tokyo.`, where
llama.cpp on the identical GGUF answers `<answer>4</answer>`, `<answer>Tokyo</answer>`.

## Findings (RED, each measured here)

1. **apr's Qwen2.5 chat template drops the template's default system prompt.**
   - apr's run-verb prompt is 26 tokens; llama.cpp's rendering of the same
     messages is 47 tokens.
   - The user span `[3838 … 14276]` is identical in both. What apr omits is
     `<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>`,
     which the GGUF's own template inserts when there is no system message.
   - Consequence: a same-quant greedy differential against llama.cpp, and the
     `<answer>` format lost on every chat route. This is the #3672 class (template
     fidelity).
2. **`POST /v1/batch/completions` → 503 `No GPU-capable model loaded`** on the gpu
   lane for a GGUF that `/v1/chat/completions` serves on the GPU. The route is
   mounted and advertised, and it cannot answer.
3. **`POST /stream/generate` and `POST /realize/generate` → 503
   `Model registry error: No model available`.** Both are mounted and
   advertised, and both are dead for a GGUF served by `apr serve run`.
4. **`POST /generate` (and `/batch/generate`, `/realize/batch`) returns the PROMPT
   echoed in `text`.** The completion is `…assistant\nThe final answer is 4.`
   Even with llama.cpp's own rendered prompt (sha256 in each row), the completion
   carries no `<answer>` tag. So finding 1 is not the only difference on the raw
   routes; the next step is a token-level check of apr's `/tokenize` on the
   rendered string.
5. `apr code` hardcodes `apr serve --gpu`, has no max-tokens or thinking control,
   and uses port `19384 + pid % 1000`. Filed and fixed as **#3978**; this
   producer feature-detects the fix.

No protocol fault was seen on any stream that answered (terminal events present).
The 54 faults (3 routes × 18 prompts) are the 503s above.
