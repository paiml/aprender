# PMAT-4810 implementation receipt

Chat routes stop on every end-of-generation marker, not only the declared EOS
(sub-ticket of #4661). Parent `cf84e9def9`. Commits:

| Commit | What |
|--------|------|
| `611d5fb846` | the fix: chat paths use `completion_stop_tokens` |
| `450940a338` | the RED/GREEN unit test |
| `3bba10b0c7` | roadmap entry |
| `9dbcdf975a` | this receipt (quorum round 1) |
| `7491a6f3e4` | every chat path calls `chat_stop_tokens`; the helper `stop_tokens_unless_ignore_eos` is removed (quorum round 2) |

## AC 1: one stop set for every chat path

`chat_stop_tokens(request, tokenizer, eos: Option<u32>)`
(`crates/aprender-serve/src/api/openai_handlers.rs`) returns
`completion_stop_tokens(tokenizer, eos)`, or an empty set when `ignore_eos` is
set. Each chat path reaches it as follows. Line numbers are at `7491a6f3e4`:

| Path | Call |
|------|------|
| CUDA dense, `try_cuda_backend` | `chat_quantized_config` at `cuda_chat_backend.rs:137` |
| quantized CPU, `try_quantized_backend` | `chat_quantized_config` at `cuda_chat_backend.rs:423` |
| `chat_quantized_config` itself | `chat_stop_tokens` at `openai_handlers.rs:283` |
| CUDA MoE, `moe_gen_config` | `chat_stop_tokens` at `cuda_chat_backend.rs:1175` |
| Qwen3.5, `try_qwen35_backend` | `chat_stop_tokens` at `qwen35_chat_backend.rs:144` |
| GPU, `try_gpu_backend` | `chat_stop_tokens` at `openai_handlers.rs:1140` |
| cached, `try_cached_backend` | `chat_stop_tokens` at `openai_handlers.rs:1254` |

The first two rows are in code this branch does not change, which is why the
diff alone does not show them. No other non-test code builds a chat stop set:
`stop_tokens_unless_ignore_eos` is gone, and `chat_stop_tokens` is its only
replacement. So every chat path stops on the same set as `/v1/completions`.

## AC 2: RED on the parent, GREEN on the fix

`chat_stops_on_every_eog_marker_not_just_eos` builds a three-token vocabulary
(`<unk>`, `<|endoftext|>`, `<|im_end|>`) with the declared EOS set to
`<|im_end|>`. It reads the stop set through `chat_quantized_config`, whose
signature is the same on both commits, so the test compiles on the parent and
the RED result is real.

| Tree | Command | Result |
|------|---------|--------|
| parent `cf84e9def9` plus the test file only | `cargo test -p aprender-serve --lib perf039_ignore_eos_tests` | 5 passed, 1 failed, exit 101. The failure is the intended one: `<\|endoftext\|> must end a chat turn: [2]` |
| fix `450940a338` | `cargo test -p aprender-serve --lib` | see AC 4 |

## AC 3: measured behaviour change on a 30B MoE model

This is an end-to-end measurement and not a CI test: it needs a 30B GGUF and a
GPU. Both builds were served with `apr serve run <gguf> --gpu`. Each prompt was
sent to `/v1/chat/completions` with `max_tokens 256`, `temperature 0.8`,
`top_p 0.95` and seeds 1, 5 and 42. The tokenizer was the same on both sides.
Model: Qwen3-Coder-30B-A3B-Instruct Q4_K_M.

| Prompt | Parent `cf84e9def9` | Fix `611d5fb846` |
|--------|---------------------|------------------|
| "What is 2+2?" | `length` 256 on 3/3 seeds; the output repeats the prompt | `stop` 8 on 3/3 seeds: `2 + 2 = 4` |
| "Write a haiku about rain." | `stop` 99 on 3/3 seeds | `stop` 24 on 3/3 seeds; the same haiku |
| the other three prompts | identical on both builds | identical on both builds |

For Qwen2.5-0.5B-Instruct f16, every row is identical on both builds except one:
"Why is the sky blue?" with seed 42 goes from `length` 256 to `stop` 82. Its
garbled text, including a literal `|i|m_end|`, is the same on both builds. That
is a separate defect (#4804: the GGUF serve tokenizer splits special tokens into
pieces), and this ticket does not claim it.

Limit of this measurement: the server log was not captured, so the receipt does
not prove which device ran the forward pass. The device was requested with
`--gpu`. What the table proves is the behaviour change between the two builds
on the same model, prompts and seeds.

## AC 4: complexity gate and the lib suite

`try_cuda_backend` was at cognitive complexity 43, which failed the pmat
pre-commit gate (limit 25). Two blocks were moved into helpers without changing
behaviour: the non-streaming batch/fallback path became `collect_cuda_turn` and
the streaming path became `dispatch_cuda_stream`. The commit then passed the gate.

The lib suite was measured in three runs:

| Head | Run | Result |
|------|-----|--------|
| `450940a338` | `cargo test -p aprender-serve --lib` (16,281 tests), stopped to free a shared test host | 16,215 passed, 64 ignored, 0 failed. 2 had not reported yet |
| `450940a338` | those 2, by name with `--exact`, on the same built test binary: `second_chat_turn_resumes_from_the_first_turns_checkpoint_4274` and `falsify_4228_005_prefill_bit_identical_4b` | 2 passed, 0 failed, exit 0, 3440 s |
| `7491a6f3e4` | rebuilt; `cargo test -p aprender-serve --lib api::openai_handlers -- --skip qwen35_serve_tests` | 52 passed, 0 failed, exit 0. Includes `chat_stops_on_every_eog_marker_not_just_eos` and all of `perf039_ignore_eos_tests` |

Together, the first two rows are the complete lib suite at `450940a338`:
16,217 passed, 64 ignored, 0 failed.

`7491a6f3e4` changes only how the stop set is reached, not what it contains.
For that commit, the third row covers the module that holds every chat path.
`cargo check -p aprender-serve --lib --tests`, with and without
`--features cuda`, and `cargo clippy -p aprender-serve --lib --features cuda --
-D warnings` were both clean. The default build already compiles
`moe_gen_config` and `try_qwen35_backend`; `--features cuda` adds
`try_cuda_backend`. The full 16,281-test suite was not run again at
`7491a6f3e4`.
