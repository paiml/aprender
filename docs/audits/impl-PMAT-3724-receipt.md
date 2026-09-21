# PMAT-3724 implementation receipt

Issue #3724 (epic #3710, 0.69.1). Every figure below was measured by the implementer and can be re-run.

## It compiles, and why the private helpers are visible across files
These files are not separate modules. They are textually `include!`d into one module each, so a private `fn` / `struct` in one is visible in the other:
- `crates/apr-cli/src/commands/qa.rs:560` `include!("output_verification.rs");` and `:561` `include!("golden_output.rs");`: `golden_answer`, `GoldenCase`, `golden_test_cases_for`
- `crates/apr-cli/src/commands/chat.rs:623` `include!("chat_session.rs");`, which at `:56` `include!("chat_load_tokenizers.rs");` and `:57` `include!("chat_session_02.rs");`: `select_chat_template`, `production_chat_template`, `RealizarTemplate`

This tree (`git diff 6577d369c <this commit>` is empty) built and ran:
- `cargo build --release --features cuda --bin apr`: rc 0, `apr 0.69.0 (6577d369c)`
- `cargo test -p apr-cli --lib -- golden_output_tests production_chat_template_tests`: 9 passed, 0 failed
- `cargo fmt --all -- --check`: rc 0. `scripts/guard_tree.sh --no-cargo`: 76 checks, 0 failed. `scripts/check_complexity_ratchet.sh`: PASS (none new, none grown)
- `cargo clippy -p apr-cli --lib --features cuda`: no warning in the four changed files

## Hardware verification (both modes, both hosts)
`apr qa <model> --json --offline` with the ladder's skip flags, under the host GPU lock. Golden is 6 cases for a thinking model (3 direct + 3 thinking at 4096), all rc 0:

| model | lambda (x86, RTX 4090) | gx10 (aarch64, GB10) |
|---|---|---|
| Qwen3-8B-Q4_K_M | 6/6 | 6/6 |
| Qwen3.5-0.8B-Q4_K_M | 6/6 | 6/6 |
| Qwen3.5-2B-Q4_K_M | 6/6 | 6/6 |
| Qwen3.5-4B-Q4_K_M | 6/6 | 6/6 |
| Qwen3.5-9B-Q4_K_M | 6/6 | 6/6 |
| Qwen3.5-27B-Q4_K_M | 6/6 | 6/6 |

Binary `apr 0.69.0 (b89d126fc)`. Later changes are comments, the fragment, and extracting `select_chat_template` (re-smoked at 6577d369c: `apr chat --no-gpu` prints "Detected ChatML chat template (Qwen3 no-think, as `apr serve`)" and answers "2 + 2 equals 4.").

## Modalities, qwen3-8b, lambda CUDA (done_when 4)
- `apr chat --gpu --temperature 0`: "2 + 2 equals 4." / "The capital of France is Paris." (0.69.0 printed only `<think>` text for 2+2)
- `apr serve run --gpu`, `/v1/chat/completions`: "2 + 2 = 4." / "The capital of France is Paris.", finish_reason stop
- `apr run --prompt Q --chat --gpu`: "2 + 2 equals 4." / "The capital of France is Paris.", rc 0, no CPU fallback

## Think length, x86 vs aarch64 (done_when 5)
Env-gated token dump of the gate's CPU leg, built from 225b2a9ab on each host, greedy, budget 2048, Qwen3-8B sha d98cdcbd…:

| prompt | lambda x86 | gx10 aarch64 |
|---|---|---|
| What is 2+2? | 545 | 396 |
| Hello there, how are you doing today my friend? | 114 | 114 |
| What is the capital of France? | 126 | 126 |

The 2+2 trajectories share a 141-token prefix and split at generated token #126 (gx10 id 807, lambda id 279): a near-tie argmax the two CPUs break differently.
