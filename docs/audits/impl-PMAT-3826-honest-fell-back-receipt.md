# PMAT-3826 — `"fell_back"` required the user to have ASKED

**Row:** GH #3826, milestone 0.69.1. **Worker:** aprender worker two. **Cop:** aprender-3e.
**Branch:** `fix/honest-backend-reporting` off `origin/release/0.69.1-batch-2` @ `cdad23729`
(already carries `f4d5b7fe9`, #3827).

## Verdict

Fixed. And the fix is **not** the one the ticket implies, because the obvious
version of it silently retires #3602. That is the finding worth carrying forward.

`apr run --format json` emitted `"backend": {"fell_back": …}` computed as

```rust
accel_forced && result.used_gpu == Some(false)
```

A bare `apr run` on a cuda build **attempts CUDA without being asked**. When the
F2 parity gate refuses that result (cosine 0.4153 on qwen2.5-coder-0.5b,
#3804/#3602) the run falls back to CPU and reported `"fell_back": false` — while
its own stderr on the same run said `attempting fallback`. The machine-readable
surface contradicted the human-readable one, and a consumer counting fallbacks
saw none, ever.

## 1. Why `used_gpu` could not answer it alone

`used_gpu` records whether the GPU **produced the tokens**. It therefore
collapses two different runs into one `false`:

* nothing was asked for and nothing was tried — a plain CPU run;
* something was tried and its result was **refused** — a fallback.

So dropping `accel_forced` and testing `used_gpu` alone is the opposite error:
every CPU-only run would claim a fallback. The missing term is a third state —
whether a GPU backend was **entered**, recorded before its result is judged.
That is `gpu_attempted`.

## 2. The correction that mattered: asking is SUFFICIENT, not NECESSARY

My first predicate was `gpu_attempted && used_gpu != Some(true)` — `accel_forced`
removed outright. It passed compilation and its own new case table, and it broke
a shipped guard:

```
---- the_json_distinguishes_a_deliberate_cpu_run_from_a_rejected_gpu_run ----
assertion `left == right` failed: a requested GPU that did not run is the whole
finding of #3602
  left: Bool(false)
 right: true
```

That fixture is `used_gpu: Some(false)`, `gpu_attempted: None`, `accel_forced: true`
— the user asked for the GPU and nothing reported entering one. #3602 says that
is a fallback. It is, and my narrower predicate had quietly retired it.

**#3826 was never that `accel_forced` is wrong. It is that `accel_forced` was
treated as NECESSARY.** The two terms are a disjunction, not a replacement:

```rust
"fell_back": result.used_gpu == Some(false)
    && (accel_forced || result.gpu_attempted == Some(true)),
```

`used_gpu == Some(false)`, not `!= Some(true)`: absent is Unknown, never Fail, so
a backend that did not report has not reported a fallback. That is
`reconcile_accelerator`'s own rule, and the JSON beside it must not contradict
the check it sits next to.

**I wrote the case table that agreed with my own error.** It asserted
`attempted=Some(false), forced=true → fell_back=false`, contradicting a guard
already in the same file. The suite caught it, not the compiler and not review.
Recorded because a mutant table authored from the same wrong premise as the fix
confirms the premise instead of testing it.

## 3. Mutants — both edges, opposite directions

| mutant | predicate | result |
|---|---|---|
| **A** — the shipped defect | `accel_forced && used_gpu == Some(false)` | **RED 1 of 10** + the counting row |
| **B** — the over-correction | `gpu_attempted && used_gpu != Some(true)` | **RED 4 of 10** + #3602's guard RED |

Mutant A names the case and nothing else:

```
#3826 REGRESSION: 1 of 10 (state x accel_forced) combinations report the wrong
fallback. A harness counting fallbacks reads this field and nothing else:
  - attempted=Some(true) used_gpu=Some(false) accel_forced=false:
    expected fell_back=true, got false. THE #3826 CASE: an accelerator was
    entered and its result refused, so the CPU redid the work. A fallback
    WHETHER OR NOT it was asked for — this is the row where the old predicate
    reported none
```

and the consumer symptom, asserted as a symptom rather than as a field value:

```
#3826: a harness counting fallbacks across these three runs must see 2 (the bare
run whose CUDA result was refused, and the explicit --gpu one). Seeing 1 means
the unasked-for fallback is invisible again — which is the defect, not a
rounding difference.
  left: 1
 right: 2
```

Under mutant A the #3602 guard stays **GREEN** — so the new rows cover #3826
specifically, not the field in general. Under mutant B it goes RED, which is the
half that would otherwise have shipped.

Mutant B also re-opens two Unknown rows: `attempted=Some(true), used_gpu=None`
reported a fallback in both `accel_forced` states. Unknown is not a Fail.

## 4. What was changed

| file | change |
|---|---|
| `infer/inference_result.rs` | `pub gpu_attempted: bool` on `InferenceResult`; threaded through `run_gguf_inference`'s three arches |
| `infer/gguf_gpu_generate.rs` | `run_gguf_generate` → `Result<(Vec<u32>, bool, bool)>`; `gpu_attempted = true` set in both CUDA arms and the wgpu arm **before** the outcome is known |
| `apr-cli/commands/run.rs`, `inference_output.rs` | field threaded through `RunResult` / `InferenceOutput` |
| `apr-cli/commands/run_entry.rs` | the predicate above |
| `run_tests_accel_reconcile.rs` | `pmat3826_fell_back_names_a_real_fallback` — 5 states × both `accel_forced` values, plus the counting row |

The dense path is where the reported defect lives. `qwen3_moe` is CPU-only so it
cannot attempt; the qwen35 dispatch does not report its attempt separately and is
threaded as `false` rather than guessed.

### An inconsistency in my own mechanical edit

The scripted insert wrote `gpu_attempted: false` at every site keyed on
`used_gpu`, including **three production sites where `used_gpu: true`** — the APR
wgpu, APR CUDA and SafeTensors CUDA constructors. A backend cannot produce tokens
without having been entered, so those were simply false. 29 sites corrected (3
production, 26 fixtures). Not user-visible under the final predicate, which tests
`used_gpu == Some(false)` first — but the field would have been lying, and the
next predicate to read it would have inherited that.

The edit also missed a site the first time and again the second: both were the
field-init **shorthand** form (`used_gpu,`), which the literal-keyed script
cannot see. Both were caught by compiling, neither by re-reading or re-counting.

## 5. Checks — both feature settings, as asked

| check | result |
|---|---|
| `cargo test -p apr-cli --lib` | **7356 passed**, 0 failed, 12 ignored |
| `cargo test -p aprender-serve --lib` | **15985 passed**, 0 failed, 59 ignored |
| `cargo test -p aprender-serve --test infer_deep_coverage` | 71 passed, 0 failed |
| `cargo test -p aprender-serve --lib --features cuda --no-run` | rc=0 |
| `cargo test -p apr-cli --lib --features cuda --no-run` | rc=0 |
| `cargo test -p apr-cli --lib --features cuda -- pmat3826 the_json_distinguishes` | 3 passed |
| `cargo clippy -p aprender-serve --lib -- -D warnings` | rc=0 |
| `cargo clippy -p aprender-serve --lib --features cuda -- -D warnings` | rc=0 |
| `cargo clippy -p apr-cli --lib -- -D warnings` | rc=0 |
| `cargo clippy -p apr-cli --lib --features cuda -- -D warnings` | **rc=101, 60 errors — PRE-EXISTING, see below** |
| `cargo fmt --all -- --check` | rc=0 |

**`clippy -p apr-cli --lib --features cuda` is red on this branch's base.**
Measured, not assumed: `git stash` and re-run gave the same **60** errors, every
one of them in `aprender-train` (`autograd/cuda_forward/matmul.rs` ×28,
`normalization.rs` ×13, …), a crate my diff does not touch at all. 60 → 60.
Mostly `trivial numeric cast: u32 as u32` and dead-code findings. Flagged, not
fixed: it is a separate row, and a lint sweep of another crate is not a
release-night edit.

That first cuda clippy attempt **did not run at all** — I passed `$f` unquoted in
zsh, which does not word-split, so cargo saw `--features cuda` as one argument
and exited 1 on `unexpected argument`. An rc=1 read as a lint failure would have
been a false finding in this receipt.

## 6. Not claimed

* **Not measured on the card.** The claim is code-path-derived and labelled as
  such throughout. The live case (`apr run` on a cuda build whose F2 gate refuses
  the result) is cuda-only and was not reproduced on hardware for this row: the
  GPU was at 23317/24564 MiB and 99% util with an empty `/tmp/apr-gpu-queue` for
  the duration. The cuda evidence here is that both test binaries **build** under
  the feature and the backend rows pass under it.
* The `used_gpu ⟹ gpu_attempted` invariant is **documented on the field and
  honoured at every construction site, not enforced structurally.** The struct is
  built by literal at ~70 sites and nothing stops the next one from violating it.
  A constructor or a `debug_assert` is owed; it is not in this diff.
* `fell_back` still says nothing about *why* the accelerator was refused. A run
  refused by the F2 parity gate and one refused by an OOM are the same `true`.
  #3606-shaped, named in the code comment, not approximated with a guess here.
* No qwen35 model was involved in any measurement in this receipt.
