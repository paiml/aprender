# #3990 — chat-template fidelity: receipt (aprender-f5)

Branch `fix/3990-chat-template-fidelity`, off `66c911359`, release head `4ce169a4d` merged
in (`0a537d224`, clean). Split per the cop (aprender-6a): **f5** owns the renderer, the oracle,
serve, `apr code`, the qa golden DEFAULT leg, `chat_session_02` and `safetensors.rs`;
**aprender-36** owns run (infer ×3) and chat `build_formatted_prompt` (`165578f17`, uses
`official_or_legacy`); **aprender-6c [8b6b78]** owns the golden thinking-ON legs.

## What changed

The model's OWN chat template is now what apr renders, instead of a hand-coded family
template picked by architecture/name:

| Source | Entry point |
|---|---|
| GGUF `tokenizer.chat_template` | `realizar::chat_template::render_official_for_model(gguf, msgs, thinking)` |
| SafeTensors `tokenizer_config.json` | `render_official_from_tokenizer_config(json, msgs, thinking)` (string or list-of-templates; string or AddedToken bos/eos) |
| raw | `render_official(tpl, bos, eos, msgs, add_generation_prompt, thinking)` |

minijinja with `trim_blocks`/`lstrip_blocks` (HF/llama.cpp semantics), `raise_exception`,
`tojson` (feature `json`), and the Python str methods templates call (`startswith`,
`endswith`, `strip`/`lstrip`/`rstrip`, `split`). No template → the old detector (said, not
silent); a template that FAILS to render → a `[#3990] WARNING` on stderr, then the detector.

**Thinking default = OFF** (`enable_thinking=false`) on every default path — production's
default since #3801 — now rendered the template's own way. Behaviour deltas a reader should
expect:
- Qwen3 / Qwen3.5 no-think prefill is now `<think>\n\n</think>\n\n` (official), was `<think>\n</think>\n`.
- Qwen2.5 with no system message now gets its template's default system turn
  ("You are Qwen, created by Alibaba Cloud…"), which the hand ChatML omitted.
- Measured: `qwen35_serve_tests` answer moved `"Lima."` → `"Lima"` on the official prompt.

## Oracle

llama.cpp `df03399` `llama-server` (CPU-only, `CUDA_VISIBLE_DEVICES=""`, `-ngl 0`, reaped):
`/apply-template` for the string, `/tokenize` for ids. 4 models × {system, none} ×
{thinking on, off} = **16 cells**, fixture
`crates/aprender-serve/src/fixtures/chat_template_3990/llama_cpp_df03399.json`:
`qwen2.5-1.5b-instruct-q4_k_m`, `Qwen3-1.7B-Q4_K_M`, `Qwen3.5-0.8B-IQ4_XS`,
`tinyllama-1.1b-chat-v1.0.Q4_K_M`. TinyLlama is in the matrix because its template uses
BARE `{% %}` tags and `eos_token`; with the Qwen templates alone, trim_blocks and bos/eos were
equivalent mutants.

## Results (post-merge)

| Test | Result |
|---|---|
| hermetic string equality, renderer (CI) | **16/16** byte-equal |
| rendered-prompt ids vs llama.cpp (real GGUFs) | **16/16 compared: 12 equal, 4 pinned RED to #3993** |
| TinyLlama HF `tokenizer_config.json` → oracle (CI) | 4/4 byte-equal |
| serve helper on real GGUFs (thinking=false cells) | 8/8 byte-equal |
| `apr code` on real GGUFs (thinking=false cells) | 8/8 byte-equal |
| qa golden default prompt on real GGUFs | 4/4 byte-equal |
| OpenAI handler wiring (tokenizes the official render) | pass |
| `qwen35_serve_tests` (serve == run) | 10/10 |
| `apr-cli commands::qa::` | 296 pass |
| fmt `--all --check` | clean |
| clippy `-D warnings`: aprender-serve `--features cuda`, apr-cli, aprender-orchestrate `--features inference` | clean |

**#3993 (pinned RED, not skipped):** the rendered TinyLlama STRING is byte-equal; apr's SPM
encode does not split the control token `</s>`, so `word.</s>` → `.</` `s` `>` where llama.cpp
has `.` `</s>`. The pin accepts ONLY that divergence and fails if the ids start matching
(remove the pin then). Owner: aprender-19 [337148].

## Mutants (all restored; `git status` clean after each table)

Renderer (`chat_template_official.rs`), 16-cell matrix: **7/8 killed**: trim_blocks off,
enable_thinking never passed, no str methods, no generation prompt (model path), eos never
inserted, model path eos not looked up, bos/eos swapped. **SURVIVED: lstrip_blocks off**,
equivalent on all four oracle templates (none has an indented block tag). Named gap.

Wiring: **8/9 killed**: OpenAI handler → legacy; qwen35 serve → legacy (killed by `a_chat_request_answers_what_apr_run_answers`); serve helper thinking None; serve helper
never official; tokenizer_config eos ignored; `apr code` never official; golden thinking None;
golden ignores tokenizer_config. **SURVIVED: qa gate call site back to the detector**; no
test drives `run_golden_output_gate` (it runs the model). Named gap.

## Gaps, named

1. `lstrip_blocks` untested: no oracle template has an indented tag.
2. `run_golden_output_gate`'s call to `golden_test_cases_for_model` is untested (mutant survived).
3. `apr run`'s GGUF prompt in `safetensors.rs::format_gguf_prompt` is wired but untested,
   and still gated on the FILE NAME containing "instruct" (pre-existing; unchanged here).
4. SafeTensors CUDA serve (`cuda_chat_backend.rs::try_safetensors_cuda_backend`) still
   renders the detector template: its state is built only in realizar's library CLI
   (`cli/mod_server_commands.rs`), which no `apr` verb calls (apr-cli has no `realizar::cli`
   reference). Not wired; not reachable from `apr`.
5. APR-format models carry no GGUF/tokenizer_config here → detector template (unchanged).
6. The golden gate was NOT run end-to-end against real generation in this row. The prompt
   changed (deltas above), so the qa ladder's next run is the measurement.
