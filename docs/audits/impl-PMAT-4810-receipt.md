# PMAT-4810 implementation receipt

Chat routes stop on every end-of-generation marker, not only the declared EOS
(sub-ticket of #4661). Fix commit `611d5fb846`, test commit `450940a338`,
parent `cf84e9def9`.

## AC 1: one stop set for every chat path

`chat_stop_tokens(request, tokenizer, eos)` returns
`completion_stop_tokens(tokenizer, Some(eos))`, or an empty set when
`ignore_eos` is set. The quantized CPU, CUDA dense, CUDA MoE (`moe_gen_config`),
Qwen3.5, cached and GPU chat paths all call it, so they stop on the same set as
`/v1/completions`.

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

`cargo test -p aprender-serve --lib` at `450940a338` ran 16,281 tests. It was
stopped to free a shared test host after 16,215 had passed, 64 were ignored and
0 had failed. The 2 tests that had not reported yet, both slow Qwen3.5 tests
(`second_chat_turn_resumes_from_the_first_turns_checkpoint_4274` and
`falsify_4228_005_prefill_bit_identical_4b`), are being re-run by name on the
same built test binary. This receipt will be updated with that result. Until
then, AC 4's "green" is partly measured.
