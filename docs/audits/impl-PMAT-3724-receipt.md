# #3724 — the golden gate asks the model the way production asks it

**Branch** `PMAT-3724-qa-golden-template` @ `3ebb1c61e` · **author** aprender-c7 · **2026-09-22**
**Binary under measurement** `apr 0.69.0 (3ebb1c61e)`, built `--release --features cuda` on each host from that sha.

## The defect, restated from the measurement

`apr qa ~/models/Qwen3-8B-Q4_K_M.gguf` returned rc 5 `golden_output: Empty output` on lambda and was GREEN on
gx10 — same binary, same file sha. Neither was a CUDA defect: the GPU leg passed and a non-cuda build failed the
same way. The gate sent plain ChatML, which leaves Qwen3 in **thinking mode**, a mode no production path used.
On x86 the greedy reasoning for "What is 2+2?" ran the entire 512-token budget (527 tokens, 15 of prompt) and was
still inside `<think>`; `strip_thinking_blocks` truncated at the unclosed tag and the answer became `""`, which
`verify_output` reported as "Empty output". aarch64 closes the same block under 512 tokens. **Same code, different
token counts — which is why a prompt defect presented as a host defect.**

## What changed

**1. The gate renders through production's detector, keyed on `general.architecture`** (`golden_prompt_for`,
`crates/apr-cli/src/commands/golden_output.rs`). Ruled option (A) by the cop, 2026-09-22: the detector for every
architecture, not just for the one that was broken.

**2. An unclosed `<think>` is reported with its budget, never truncated to an empty answer.**
`strip_thinking_blocks -> String` became `split_thinking_blocks -> ThinkingSplit {Answer, Unclosed}`
(`output_verification.rs`), and all three legs — CPU (`golden_output.rs`), CUDA (`validate_gpu_golden_output`) and
runtime/hybrid (`run_golden_output_gate_runtime`) — report
`"<leg>: think block unclosed within N tokens (the model was still reasoning at the budget; M chars generated,
no answer was reached)"`.

**3. done_when 3 — both modes.** A model whose production template *suppresses* thinking is also judged with the
suppression removed (`thinking_on_case`), at 2048 tokens: the block must CLOSE and the answer after it must be
correct; a block that never closes is the named failure above, and generating no block at all is reported as
"the thinking mode this leg exists to judge was never entered" rather than passing quietly.
The suppression is detected by *reading production's rendered prompt* (`without_thinking_prefill`), not by
matching a template name — so #3755's replacement template is followed rather than bypassed.

**No budget was raised on the production leg.** It is still `config.max_tokens.max(512)`, and
`the_thinking_on_budget_does_not_touch_the_production_leg` pins that. The fix is the prompt.

## Blast radius — exactly which architectures change

`ChatMLTemplate::format_conversation` of a single user message is
`<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n` — byte-identical to the three strings the gate
hardcoded. So:

| architecture | production template | golden prompt after #3724 | ladder rung? |
|---|---|---|---|
| `qwen2`, `qwen2.5` | ChatML | **unchanged, byte for byte** | yes (qwen2-1.5b-q4km) |
| `qwen3_moe` / `qwen3moe` | ChatML (PMAT-181: trained without `<think>`) | **unchanged, byte for byte** | no |
| `qwen3` | Qwen3NoThink | **+ `<think>\n</think>\n`** — the fix | yes (qwen3-1.7b, qwen3-8b) |
| `qwen35` | Qwen3NoThink | **+ `<think>\n</think>\n`** | yes (0.8b/2b/4b/9b/27b) |
| `llama`, `mistral`, `phi2`, `stablelm`, … | Llama2 / Mistral / Phi / … | **now production's template** | **no rung exists** |
| APR / SafeTensors with no declared architecture | — | unchanged ChatML | n/a |

Pinned by `chatml_architectures_render_the_legacy_prompt_byte_for_byte`, which classifies by asking the detector
rather than by a hand-written list, and asserts both partitions non-empty so it cannot pass vacuously.

## Which production paths put qwen3 in thinking mode — the cop's question, answered

**Not all of them agree, and one still thinks.** Measured on `apr 0.69.0 (3ebb1c61e)`:

| path | detector | `qwen3` gets |
|---|---|---|
| `apr serve` | `realizar::chat_template::detect_format_from_name` | `Qwen3NoThink` |
| `apr run --chat` | same, via `prepare_tokens_gguf` → `apr_arch_to_template_hint` → `format_messages` | `Qwen3NoThink` |
| `apr qa` golden gate | same, **after this branch** | `Qwen3NoThink` |
| **`apr chat`** | **`aprender::text::chat_template::detect_format_from_name`** (`chat.rs:34`) | **`ChatML` — it thinks** |

Measured, `apr chat ~/models/Qwen3-1.7B-Q4_K_M.gguf --temperature 0 --max-tokens 64`:
`Detected ChatML chat template` … `Assistant: <think>\nOkay, the user is asking "What is 2+2?" Let me think.…`

aprender-core's `TemplateFormat` (`crates/aprender-core/src/text/chat_template/mod.rs:195-201`) has seven variants
and **no `Qwen3NoThink`** — no-think is unrepresentable there, not merely unselected. Two mechanisms answer the
same question and only one of them knows about no-think. **So thinking-on remains a reachable production mode
under #3715's shape, and #3724 does not delete that cell.** Fixing `apr chat` is one import away but is a change
to `apr chat`, not to the golden gate; it is not in this row and was not touched.

## Tests

`cargo test -p apr-cli --lib` — **7312 passed, 0 failed, 12 ignored**. `cargo clippy -p apr-cli --lib -- -D warnings` rc=0. `cargo fmt --all -- --check` rc=0.

New, all under `commands::qa`:
- `chatml_architectures_render_the_legacy_prompt_byte_for_byte` — done_when 2, byte-identity, derived classification
- `qwen3_is_asked_with_the_production_no_think_prompt` — the fix, positive and negative
- `the_gate_renders_through_the_same_detector_production_uses` — fails if a template is ever re-hardcoded in the gate
- `an_unknown_architecture_keeps_the_legacy_chatml`
- `changing_the_architecture_changes_the_wrapper_and_never_the_question` — "no check is relaxed"
- `the_thinking_on_prompt_is_production_minus_the_suppression`
- `an_architecture_without_suppression_has_no_thinking_on_leg`
- `a_think_block_with_content_is_not_a_suppression_prefill`
- `the_thinking_on_leg_judges_closure_and_the_answer` — unclosed named with budget, never "Empty output"
- `the_thinking_on_budget_does_not_touch_the_production_leg`
- `strip_thinking_unclosed` / `an_unclosed_block_after_a_closed_one_is_still_unclosed` — the old assertion
  `strip_thinking_blocks(unclosed) == ""` **was the defect asserted as behaviour**, and is now `Unclosed`
- `the_unclosed_reason_names_the_budget_and_never_says_empty`

## Measured — both hosts, 8/8 rungs green

Every run below names its binary in the log it is quoted from, and the binary was snapshotted outside any cargo
target dir before use (cop's standing ruling, 2026-09-22). `/home/noah/src/aprender/target/release/apr` on lambda
read `apr 0.69.0 (85d8491f6)` — another session's build — at the time of these runs; **nothing here resolved
through it.**

| run | binary | sha256 (24) | where |
|---|---|---|---|
| ladder, lambda | `apr 0.69.0 (3ebb1c61e)` `--features cuda` | `939a9c9506650e9b8ef98ada` | `/mnt/nvme-raid0/targets/c7-3724/release/apr`, snapshot `apr-3ebb1c61e` |
| ladder, gx10 | `apr 0.69.0 (3ebb1c61e)` `--features cuda` | — | built in gx10's own worktree `/mnt/nvme-raid0/agent-wt/c7-3724` |
| control | `apr 0.69.0 (3ebb1c61e)` **no-cuda** | `2c63161b0fe7cf563e957da6` | `apr-nocuda-control` |
| mutant A | `apr 0.69.0 (3ebb1c61e)` no-cuda + mutation | `078df7427cc82269382596bb` | `apr-mutantA` |
| mutant AB | `apr 0.69.0 (3ebb1c61e)` no-cuda + mutation | `0479c1004da75242fc019f49` | `apr-mutantAB` |

### `scripts/model_ladder.sh`, ladder flags, both hosts — done_when 1

`--- model capability ladder on lambda (NVIDIA GeForce RTX 4090, cc 8.9) apr=/mnt/nvme-raid0/targets/c7-3724/release/apr sha=3ebb1c61e version=0.69.0 ---`
`--- model capability ladder on gx10 (NVIDIA GB10, cc 12.1) apr=/mnt/nvme-raid0/targets/c7-3724/release/apr sha=3ebb1c61e version=0.69.0 ---`

| rung | lambda (x86_64, RTX 4090 sm_89) | gx10 (aarch64, GB10 cc 12.1) |
|---|---|---|
| qwen2-1.5b-q4km | OK | OK |
| qwen3-1.7b-q4km | OK | OK |
| **qwen3-8b-q4km** | **OK** | **OK** |
| qwen35-0.8b-q4km | OK | OK |
| qwen35-2b-q4km | OK | OK |
| qwen35-4b-q4km | OK | OK |
| qwen35-9b-q4km | OK | OK |
| qwen35-27b-q4km | OK | OK |

Every row is `qa cap+golden pass, backends cpu,cuda honoured`. Receipts:
`evidence/qa/3724/ladder-lambda.json` (`executed: 8, red: 0`, 2026-09-22T06:20Z) and
`evidence/qa/3724/ladder-gx10.json` (`executed: 8, red: 0`, 2026-09-22T06:23Z).

**qwen3-8b-q4km — the 0.69.0 stopper — is green on lambda**, which is where it failed, **and stays green on gx10**,
which is where it always passed. The five Qwen3.5 rungs stay green on both, as done_when 1 requires.

### Falsification — two mutants, each RED, each with its production symptom

All three runs are `apr qa <qwen3-8b> --json --offline` with the ladder's skip flags, on the **no-cuda** build, so
the falsification half took **zero GPU time** and never contended for the lock.

| build | golden_output | message |
|---|---|---|
| **control** (both fixes) | **PASS** | `3 golden test cases passed` |
| **mutant A** — `golden_prompt_for` forced back to hardcoded ChatML, unclosed-reporting kept | **RED**, rc 5 | `golden_output: think block unclosed within 512 tokens (the model was still reasoning at the budget; 2086 chars generated, no answer was reached)` |
| **mutant AB** — prompt reverted **and** `split_thinking_blocks` reverted to truncating: pre-#3724 behaviour | **RED**, rc 5 | `golden_output: Empty output` |

Mutant AB reproduces **the issue's exact symptom** — rc 5, "Empty output" — from this branch's own source, so the
defect this row fixes is the defect that was reported. Mutant A isolates the two changes: with the prompt reverted
but the reporting kept, the gate still fails (the prompt is the root cause) and now says *why* — 2086 characters of
reasoning that never reached an answer, rather than a claim that the model produced nothing.

## Coupling with #3755 — asked and answered

I filed the hazard that three qwen35 sites must move to `format_chat_prompt` in the same batch as #3755
(https://github.com/paiml/aprender/issues/3755#issuecomment-5771254612). **#3724 does not subsume it and does not
conflict with it.** This row changes only `apr qa`'s golden gate; it touches none of those three sites. It does
*reduce* the coupling's blast radius in one specific way: `without_thinking_prefill` detects suppression by reading
production's rendered prompt rather than by matching a template name, so when #3755 replaces the template the gate
follows it instead of silently bypassing it. The three call sites still need to move together.

## Not done, deliberately

- **`apr chat` still thinks.** See the table above. Fixing it is a change to `apr chat`'s detector import, not to
  the golden gate; the cop has routed it as the #3801 remedy, after #3817.
- **No full `apr qa` run is cited.** The one I ran at 05:22Z was built before I committed, so its version string
  said `a9502d992` (the parent). The code was identical, but "the code was the same" is an inference and the version
  string is the measurement — so it is not citable and is not cited. The ladder rungs above are.
- **No budget was raised.** Deliberate, per the cop's ruling: the gate's job is to exercise production's mode, not
  to pass.
