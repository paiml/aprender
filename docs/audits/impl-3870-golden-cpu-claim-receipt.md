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
