# #3801b — `apr code`'s fourth template mechanism

Measured by aprender-c7 on lambda (RTX 4090), 2026-09-22. Binaries snapshotted outside every cargo
target dir; both named in every run below.

| role | binary | sha256 (24) |
|---|---|---|
| BEFORE | `apr 0.69.0 (ba22a9591)` = `release/0.69.1-batch-2` | `54b1142a35b545653c83d16c` |
| AFTER | `apr 0.69.0 (98e1744ab)` = this branch | `d5b8d24392277f4f0c86530f` |

## A correction to my own finding, before anything else

I reported that `apr code`'s **default** path uses `RealizarDriver` and therefore renders Qwen3 in
plain ChatML. **That is wrong, and the measurement is what says so.** `driver/mod.rs:165` carries a
doc comment reading *"Default implementation: `RealizarDriver` (sovereign, local)"* — which is what I
read — but `agent/code.rs:439` says *"Try AprServeDriver first … Falls back to embedded RealizarDriver
if `apr` binary not found"*, and `build_fallback_driver` is named for exactly that.

Measured, BEFORE binary, ordinary invocation:

```
$ apr code -p "What is 2+2? Answer in one line." --model ~/models/Qwen3-1.7B-Q4_K_M.gguf
Launched apr serve on port 19399 (pid 1503045)
apr serve ready (3.5s)
2 + 2 = 4.
```

The primary path goes over HTTP to `apr serve`, which takes realizar's detector and already gives
Qwen3 the no-think template. **No `<think>` in the output, correct answer, unaffected by this row.**

So the defect is confined to the **embedded fallback** — which is precisely the cell #3801's issue
lists: *"`apr code --thinking on` Qwen3.5-4B (**embedded fallback**, agent prompt)"*. The fourth
mechanism is real and it is on a reachable path; it is not on the path most users take.

## The defect, on the path that has it

`aprender-orchestrate/src/agent/driver/chat_template.rs` had its own three-variant
`ChatTemplate {ChatMl, Llama3, Generic}` selected by **filename** (`ChatTemplate::from_model_path`):
any name containing `qwen` → `ChatMl`. A filename cannot see `general.architecture`, and the enum has
no no-think variant — the same shape as the `apr chat` defect in #3801, one crate over, and the
fourth mechanism answering "which template does this model want".

## Measured on the fallback path, forced by removing `apr` from PATH

Both runs: `PATH=<empty dir> apr code -p "What is 2+2? Answer in one line." --model Qwen3-1.7B-Q4_K_M.gguf`,
which prints `⚠ apr serve unavailable … using embedded inference` and takes `RealizarDriver`.

| | rc | model text | reasoning markers ("Okay/Wait/Maybe/…") | answer |
|---|---|---|---|---|
| BEFORE `ba22a9591` | 0 | **13 lines** | **3** | `2+2=4` |
| AFTER `98e1744ab` | 0 | **5 lines** | **0** | `2+2=4` |

BEFORE reasons at length before answering — *"Since the gp tool isn't working, I need to think of
another way… Wait, the user's previous interaction used the gp tool…"* — the chain-of-thought this
row is about. AFTER emits one sentence and answers.

**What this measurement does NOT prove, stated rather than glossed:** the driver post-processes with
its own `strip_thinking_blocks`, so the raw prompt and raw generation are not visible in this output;
the reasoning-length collapse is consistent with the no-think prefill being applied but does not by
itself prove it. The prompt is pinned at unit level instead, where it can be asserted exactly:
`a_qwen3_model_gets_the_production_no_think_prompt` requires the rendering to end with
`<|im_start|>assistant\n<think>\n</think>\n`, and `the_old_filename_guess_produced_the_thinking_prompt`
keeps the old bare-ChatML rendering as the negative control.

Neither run looped or hit a budget, so #3801's *"unclosed within 4096"* cell did not reproduce here —
that cell is `--thinking on` on the 4B, and this is the default mode on the 1.7B. This row narrows
the mode; it is not the think-budget guard.

## The change

`format_prompt_for_model(request, model_path)` takes the selection from realizar's
`detect_format_from_name` — the same function `apr serve`, `apr run --chat`, `apr chat` and `apr qa`
ask — keyed on the architecture the file declares, read from a **bounded GGUF prefix** (never
`MAP_POPULATE`, which #3817 measured pre-faulting 18 GB on a 30B MoE). Rendering is delegated to
realizar's own template.

**One exception, named:** realizar maps every `llama` to `Llama2` (`[INST]`) and has **no Llama-3
template**, so a Llama-3 model keeps the local `format_llama3` — delegating there would replace
`<|start_header_id|>` with `[INST]`, a downgrade. `is_llama3_name` is the only filename test left.

Tool definitions still reach the system turn; tool-use and tool-result turns flatten to the assistant
and user turns they already rendered as.

## Tests

`cargo test -p aprender-orchestrate --lib --features inference` — **6550 passed, 0 failed**; without
`inference` — **6550 passed, 0 failed**.

Five new in `one_detector_tests`: the no-think prompt for qwen3 · the old rendering as negative
control · qwen2 unchanged (plain ChatML) · the Llama-3 exception asserting `<|start_header_id|>` and
**not** `[INST]` · tool definitions still in the system turn.

## Left alone, deliberately

`apr_serve.rs:463` holds a **third** copy of `strip_thinking_blocks` that truncates at an unclosed
block — the defect #3724 fixed in the golden gate. It is on the HTTP driver, not this one, and
changing what a driver returns to the agent loop is a behavioural change I am not making inside a
template row without a ruling.
