# PMAT-3715 — `/api/chat` and the Qwen3.5 hybrid

**Row:** Alfredo, via #3715 (issuecomment-5773344579) and the cop's measurement
(issuecomment-5773376796). Milestone 0.69.1.
**Worker:** aprender-d8. **Branch:** `PMAT-3715-ollama-qwen35` off
`origin/release/0.69.1-batch-2` @ `71421b6e5`.

## Verdict

**The reported defect does not exist. `/api/chat` reaches the Qwen3.5 hybrid
correctly, on both a default and a `--features cuda` build.** No routing fix was
needed, and none was written.

**The risk behind the report is real, and it is now guarded.** The ollama-compat
wire reaches the hybrid by *delegation*, an invariant nothing asserted — so the
next refactor giving `/api/chat` its own generation path would have been a silent
regression on the route Ollama-compatible harnesses actually hit. Three tests and
a contract now assert it.

## 1. The premise, measured

The report's evidence is a reference count:

| file | `generate_gpu_resident_streaming` | qwen35-aware |
|---|---|---|
| `api/cuda_chat_backend.rs` | 1 | yes |
| `api/ollama_handlers.rs` | 0 | **no** |
| `api/cuda_batch_scheduler.rs` | 2 | no |

I reproduced that count exactly on `71421b6e5`. **The count is accurate; the
conclusion does not follow from it.** `ollama_chat_handler` (and
`ollama_generate_handler`) call `openai_chat_completions_handler` directly and
re-shape its response — the delegation is stated in their own doc comments. That
handler is defined at `cuda_chat_backend.rs:800`, and its **first** arm, at line
835, is `try_qwen35_backend`, unconditional and not `cfg`-gated. So the Ollama
wire reaches the hybrid without naming it, and a grep is not a routing proof in
either direction.

### Measured live, both builds

`apr serve run /home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf`, server log
`Model ready: Qwen3.5 hybrid, 24 layers resident on the CPU`:

| route | default build | `--features cuda` |
|---|---|---|
| `POST /v1/chat/completions` | **200**, `"4"` | **200** |
| `POST /api/chat` | **200**, `"4"` | **200**, `"4"` |
| `POST /api/generate` | **200**, `"4"` | — |
| `POST /api/chat` `stream:true` | **200**, NDJSON | — |
| `GET /api/tags` | **200** | — |

Binaries pinned outside any cargo target dir: default
`sha256 41662d2619e4192c812ae134`, cuda `sha256 e89a9d80538553f5f83c5d1a`, both
`apr 0.69.0 (71421b6e5)`.

I tested the cuda build specifically because the report likely came from one and
a non-cuda result would not have answered it. It makes no difference:
`try_qwen35_backend` precedes every `cfg(cuda)` arm in the chain.

### The batch scheduler's two unguarded calls are unreachable for a hybrid

`generate_single_request_inner` takes `&mut OwnedQuantizedModelCuda` — a dense
model. Its callers are in `cuda_chat_backend.rs` (lines 140, 203), **downstream
of the line-835 early return**. And `AppState::with_qwen35_session` sets
`model: None` (`mod_app_state_qwen35.rs:46`), so a hybrid state holds no dense
model for anything downstream to decode through. Two independent reasons; either
alone suffices.

## 2. What was actually built

No routing change. The deliverable is the guard:

| artifact | what it does |
|---|---|
| `qwen35_chat_backend_tests.rs` +3 tests | asserts the Ollama wire reaches the hybrid |
| `contracts/qwen35-hybrid-serve-dispatch-v1.yaml` | new, mirrors `qwen3-moe-serve-dispatch-v1` |
| `contracts/apr-serve-model-backend-coverage-v1.yaml` | `resident_backends` widened to admit the hybrid |

The three tests were added to the **existing** `#3571` harness rather than a new
file, so they reuse its `state_or_skip` / `post` / `one_shot_answer` helpers and
the same real model. They are not `#[ignore]`d — they run in the normal `--lib`
suite whenever the model is present, and skip with a printed reason when it is
not.

* `the_ollama_chat_wire_answers_from_the_hybrid` — `/api/chat` is 200 **and its
  content equals what `apr run`'s one-shot path produces for the same prompt**.
* `the_ollama_generate_wire_answers_from_the_hybrid` — `/api/generate` likewise.
* `both_chat_wires_agree_on_the_same_request` — the two wires return **identical
  content**. This is the row that catches a divergence whose symptom is a *wrong
  answer* rather than an error, which a status assertion cannot see and which is
  the shape #3825's `tool_calls` gap had.

## 3. The mutant

Make `ollama_chat_handler` stop delegating and answer 500 directly — the
divergence the report describes, as its own reported symptom:

```
assertion `left == right` failed: #3715: /api/chat must reach the Qwen3.5 hybrid, not fail:
  {"model":"…","message":{"role":"assistant","content":"generation unavailable"},"done":true,…}
  left: 500
 right: 200
```

RED: `the_ollama_chat_wire_answers_from_the_hybrid`,
`both_chat_wires_agree_on_the_same_request`.
**Green under the same mutant: `a_chat_request_answers_what_apr_run_answers`** —
the pre-existing OpenAI-wire test. So the new rows cover the Ollama wire
*specifically*, not the chain in general, which is the whole point of the guard.

Note the mutant's body: the translation layer folded the 500 into the assistant
content as `"generation unavailable"`. A client reads that as the model's reply —
the #2609 class the file's own docs describe, still live in the error path.

## 4. Checks

| check | result |
|---|---|
| `cargo test -p aprender-serve --lib` | **15985 passed**, 0 failed, 59 ignored |
| `cargo test -p aprender-serve --lib --features cuda qwen35_serve_tests` | 9 passed, **1 failed — not mine, see below** |
| `cargo fmt --all --check` | rc=0 |
| `clippy -p aprender-serve --lib -D warnings` | rc=0 |
| `clippy -p aprender-serve --lib --features cuda -D warnings` | rc=0 |
| `pv validate` (both contracts) | 0 errors, 0 warnings |

The cuda failure is `gpu_a_chat_request_answers_from_the_gpu_session`, a
**pre-existing** test (0 lines of my diff mention it), failing on
`CUDA_ERROR_OUT_OF_MEMORY`. Cause, measured: another agent's
`apr-cuda-a9502d992 run Qwen3.5-27B-Q4_K_M.gguf` held **22638 MiB of the 24564
MiB card**, and `/tmp/apr-gpu-queue` was **empty** — so it took the VRAM without
queueing, while I *was* holding the lock. That is the #3794 class in the wild via
`apr run`: an operator path, not a code defect, but it is why the card cannot be
rationed by the lock alone.

## 5. Not claimed

* Not claimed that the report was careless. The reference count is real, the
  translation layer has genuinely diverged twice (#3715's report and #3825), and
  a delegated route is exactly the thing a grep cannot see. The guard exists
  because the concern was well-founded even though the defect was not present.
* Streaming on the Ollama wire is asserted only by a manual NDJSON probe recorded
  above, not by a cargo-level row. Listed as an open obligation in the contract.
* `GENERATION_ROUTES` is discharged by three named routes, not enumerated from
  the router. A route mounted later is not covered until added. Also recorded as
  an open obligation rather than implied.
