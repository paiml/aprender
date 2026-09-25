# #3870 — the golden gate claimed "CPU passed" about a leg it had not judged

**Worker:** aprender worker two. **Branch:** `fix/3870-golden-cpu-claim` off
`origin/release/0.69.1-batch-2` @ `3142987fb`. **Commit:** `ac58d507c`.

## Verdict

Fixed. And the sweep the cop asked for (item c) found the **root cause of
tinyllama**, which is §2 below and is a separate row.

## 1. The instrument defect

Two halves, and each alone was survivable:

* `GpuGoldenLeg::failure()` hardcoded `"GPU output failed (CPU passed)"` into
  its `WrongAnswer` arm. Nothing measured "CPU passed".
* `validate_golden_test_case` returned on a GPU failure **before** the CPU
  pattern check ran (`strip_prefix` → `split_thinking_blocks` → `verify_output`
  all sat below the early return).

Together: the message asserted the outcome of a measurement that had not
happened. One measurement and one string, read as two measurements — and read
that way by a release process.

### The fix

```rust
failure(&self)  ->  failure_given_cpu(&self, cpu: &CpuGoldenVerdict)
```

Taking the verdict **by argument** is the whole point: the claim cannot be
composed without the other leg in hand, so a future edit reintroducing the early
return does not compile. `CpuGoldenVerdict` (`Passed` / `WrongAnswer` /
`Unclosed`) judges with the same split-then-verify shape `GpuGoldenLeg::judge`
uses, so the two legs cannot drift apart in **how** they decide — only in what
they decide, which is the thing the gate exists to compare.

`Unclosed` is the third state and it is not cosmetic: a CPU leg still inside a
`<think>` at the budget reached **no verdict**, and claiming either a pass or a
failure for it would be the same defect pointing the other way.

The CPU-side messages and the gate's behaviour are unchanged. Only the order.

### Red in both directions, with DISJOINT red sets

| mutant | red | green |
|---|---|---|
| **A** — restore the hardcoded `"(CPU passed)"` | the false-claim row, the no-verdict row, the distinctness row | **`a_gpu_failure_still_names_a_genuine_cpu_pass`** |
| **B** — drop the `"(CPU passed)"` wording (the over-correction) | `a_gpu_failure_still_names_a_genuine_cpu_pass`, plus the pre-existing foreign `gpu_wrong_answer_fails_the_gate` | the false-claim row |
| **C** — move the CPU judge back below the GPU early return | `error[E0425]: cannot find value cpu in this scope` | — |

The disjointness is the load-bearing part. Neither row can be satisfied by the
other's fix, so #3477's signature (GPU wrong, CPU right — a real GPU-specific
defect) survives with its **exact** wording, which is what losing it would have
cost. Mutant A leaving the genuine-pass row GREEN is what proves the new rows
are specific to the false claim rather than blanket assertions about the string.

Mutant C is the strongest of the three and is not a test at all: the ordering
cannot be reintroduced.

## 2. The sweep found tinyllama's root cause — and it is arithmetic

The cop's item (c) was to grep the file for other "propagate past a check you
then report on" shapes. There is only one other cross-leg claim in either file,
but the sweep surfaced something bigger in `gpu_golden_leg`:

**Both legs hardcode `SpecialTokens::qwen2()`.** `golden_output.rs:50`, `:332`,
`:648` — the CPU leg, the GPU leg, and the token-id helper:

```rust
let specials = aprender::demo::SpecialTokens::qwen2();
let prompt_tokens = gguf.encode(prompt).unwrap_or_else(|| vec![specials.bos_id, 9707]);
let gen_config = QuantizedGenerateConfig { stop_tokens: vec![specials.eos_id], .. };
```

`SpecialTokens::qwen2().eos_id` is **151645**. Measured from the GGUF metadata
of `/mnt/nvme-raid0/cache/apr-home/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`:

| | |
|---|---|
| `general.architecture` | `llama` |
| `tokenizer.ggml.eos_token_id` | **2** |
| vocab size (`tokenizer.ggml.tokens`) | **32000** |
| gate's stop token | **151645** |
| is 151645 a valid id in a 32000 vocab? | **no** |

The stop condition **cannot fire**. Not "fires late", not "fires on the wrong
token" — 151645 is not a token id that exists in that model. So the model runs
the full 512-token budget and drifts, which is precisely what `#1864`'s own
comment in that function says the line exists to prevent:

> without stop_tokens, the model runs for the full max_tokens budget and starts
> emitting in-distribution chat-template tokens … from accumulated drift

That is the observed `[S][INST]` loop. And because **both** legs hardcode the
same wrong constant, both produce it — which is why the GPU and CPU outputs were
byte-identical, the observation that started this whole thread.

### Blast radius, measured across the local model set

```
REACHABLE   eos=151645  vocab=151936  arch=qwen3moe   Qwen3-Coder-30B-A3B-Instruct-Q4_0.gguf
REACHABLE   eos=151645  vocab=151936  arch=qwen3moe   Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
REACHABLE   eos=151645  vocab=151936  arch=qwen3      Qwen3-1.7B-Q4_K_M.gguf
UNREACHABLE eos=2       vocab=32000   arch=llama      tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Perfect correlation, and the mechanism explains it rather than merely fitting
it. **`apr qa`'s golden gate is structurally Qwen-only.** Every Qwen model in
the set passes; the one non-Qwen model is the one that fails. tinyllama is not
special — it is the only model present that could expose this.

Against the cop's three hypotheses: #2 (special-token decode) is the cause, and
it is a *stop token*, not a decode. #1 (the `[INST]` template) is real and
secondary — tinyllama's `tokenizer.chat_template` is Zephyr-style (`<|user|>`),
so a ChatML golden prompt is wrong for it regardless; that is why the drift is
shaped like chat scaffolding. #3 (a 1.1B model against a 3-case gate) is not
needed to explain anything.

**Filed, not fixed.** Per-architecture specials in the golden gate is a real row
with its own blast radius across the ten other call sites of
`SpecialTokens::qwen2()` in `apr-cli`, and it is not a release-night edit.

## 3. Checks

| check | result |
|---|---|
| `cargo fmt --all --check` | rc=0 |
| `cargo clippy -p apr-cli --lib -- -D warnings` | rc=0 |
| `cargo test -p apr-cli --lib` | **7369 passed, 1 failed** |

The one failure is `a_deltanet_projection_with_no_gpu_kernel_is_refused` — *"no
GPU GEMV kernel for IQ4_XS: parity cannot build the GPU half"*. **Pre-existing
on the base**, proved rather than asserted: it fails identically with my two
files stashed (`git stash push -- <the two files>`, rc=101, same panic, same
line). Zero lines of the diff touch `parity_hybrid_tests.rs`.

Worth the cop's attention on its own: `release/0.69.1-batch-2` currently carries
a **red row** in `apr-cli --lib`, and it is the row the IQ4_XS GEMV kernel work
(#3869 family) closes.

## 4. Not claimed

* Not claimed that tinyllama passes after this change. It does not, and it
  should not — the model genuinely fails the golden gate. What changes is that
  the gate now **reports what it measured**: both legs failed, so it is not a
  GPU-specific defect and not a CUDA blocker.
* Not claimed the gate is correct for non-Qwen models. §2 says it is not. This
  branch makes that visible instead of mislabelling it.
* The blast-radius table covers the four GGUFs in the local model cache, not the
  full release ladder. It is sufficient to establish the mechanism; it is not a
  ladder audit.

---

# Addendum — the stop-token fix, and what it did NOT fix

Cop's ruling: fix the golden gate alone for 0.69.1. Done in `3307506e5`. The
acceptance run then refuted the easy conclusion, so this section reports what
was measured rather than what was intended.

## The fix is correct and proven

`golden_stop_tokens(gguf)` reads the model's own eos. Unit matrix over all four
local GGUFs, both directions red (mutant D restores the constant, mutant E drops
stop tokens entirely; E reds at the `checked > 0` vacuity guard). The Qwen
control holds: Qwen still stops on exactly 151645, so the fix did not become
"ignore stop tokens".

## It is NOT sufficient for tinyllama, and I nearly said it was

`apr qa` on tinyllama still FAILS `golden_output`. The displayed text is
truncated, which made it *look* shorter than before. It is not:

```
apr=<scratchpad>/apr-3870 sha=740c65faf0918c699e5916c3 version=apr 0.69.1 (3307506e5)
model sha=9fecc3b3cd76bba89d504f29   worktree clean

apr run --max-tokens  64  →  wall  3.62 s   300 chars
apr run --max-tokens 512  →  wall 22.11 s  1410 chars, cut off mid-word
```

512 tokens of budget, 512 tokens consumed. **tinyllama never emits its own eos
under the prompt the gate builds**, so a correct stop token has nothing to stop
on. Had I read the `apr qa` line alone I would have reported a fix that does not
fix the symptom.

## The residual cause, measured rather than hypothesized

`golden_prompt_for` renders the template keyed on `general.architecture` —
`"llama"` → Llama-2 `[INST]`. tinyllama-1.1b-chat-v1.0's GGUF declares a
**Zephyr** template (`tokenizer.chat_template` = `{% ... %}<|user|>\n...`). The
gate asks a Zephyr-tuned model in Llama-2 turn structure, so it never reaches a
turn boundary and never emits eos.

Asked with its **own declared template**, same binary, same model:

| asked as | wall | chars | answer |
|---|---|---|---|
| `[INST]` (what the gate does) | 22.11 s | 1410 | `[S][INST]` loop, no "Paris" |
| `<\|user\|>` (what the GGUF declares) | **2.45 s** | **170** | **"The capital of France is Paris."** |

That output **passes** the golden pattern `["Paris"]`.

**So tinyllama is not a broken model and was never a CUDA defect. It is a
correct model being asked the wrong question by the gate**, twice over, in the
same defect class both times: a constant keyed on architecture where the model
carries the answer in its own metadata.

## Why the second one is not a release-night change

Rendering the model's declared jinja needs new plumbing:
`chat_template/raw_template.rs` exposes `detect_format_from_name` /
`detect_format_from_tokens` / `auto_detect_template` — name-and-architecture
*detection*, not "render this model's declared template". And it would change
the prompt the gate sends to **every** model, not just tinyllama.

Filed for 0.70.0 alongside the ten other `SpecialTokens::qwen2()` call sites.

## What this settles

* tinyllama cannot be made green tonight by any small change. The cop's ruling
  moving it off the tag blockers is correct, and now for a measured reason.
* The two fixes are complementary, not alternatives. With the right template the
  model emits eos — and for the gate to *observe* that, the stop token has to be
  the model's own, which is what `3307506e5` fixed. Neither alone makes the row
  green.
* Not claimed: that `apr run`'s 2.45 s termination exercises `golden_stop_tokens`.
  It does not — `apr run` has its own path. What it proves is that the model
  terminates and answers correctly when asked properly; the gate's ability to
  see that is what the stop-token fix restores.
