# PERF-060 — c=1 calibration of `intel` and `mini`, the two non-CUDA hosts

**Calibration only. No cell was measured. `scripts/perf-matrix.yaml` was not modified.
No full band set was run.**

Measured 2026-08-30 against commit-under-test **`34248e8fecf1e354f3426cd077d4c5b0536d6cc7`**
(`origin/main` at the time of the run). Refs paiml/aprender#2706, PERF-060.

---

## 0. Answers to the three questions that were asked

**1. The measured rates.** Per host, W1 shape (513-token prompt, 128 generated),
c=1, `apr` serving the pinned 7B GGUF. Every replicate is listed in §7; these are
the ranges, not medians.

| host | prefill tok/s | decode tok/s | `t_req` |
|---|---|---|---|
| `intel` (Xeon W-3245, CPU) | **8.14 – 8.97** | **6.57 – 7.66** | **73.9 – 81.9 s** |
| `mini` (M4, CPU — see §3) | **2.371 – 2.372** | **2.318** | **270.2 – 271.1 s** |

**2. The runbook's estimates do not hold, and the error is concentrated in prefill.**
`docs/PERF-054-campaign-scope.md` §5 (branch `feat/x3-campaign-plan`) assumed `intel` 40 / 6 and `mini`
60 / 8 tok/s. Decode was close on `intel` (estimate 6, measured 6.6–7.7 — the
estimate was *pessimistic* by ~1.2×). Prefill was wrong by **~4.8× on `intel`**
and by **~25× on `mini`**. Because W1's prompt is 4× its generation, prefill is
**76–80 % of `t_req`**, so the whole-cell figure moves with it: `intel`'s apr lane goes
from 8.9 h to **19.2 – 21.3 h**, `mini`'s from 6.4 h to a number that does not
matter, because of §2.

**3. The 7B produces coherent text on CPU on both hosts, and Metal does not exist
in `apr` to test.** §4. This is the finding that changes the shape of #2785.

**And one thing nobody asked, which outranks all three:** the §4.4 harness has a
**120 s hard per-request timeout** (spec §4.4.3, `perf_gate::protocol::REQUEST_TIMEOUT`,
no CLI knob). `mini`'s `t_req` is 270 s. **Every `mini` request — warmup and sampled —
is aborted before it completes, so `mini/W1` yields a receipt with zero samples no
matter how long it runs.** `intel` fits, but its first unwarmed W1 request took
118.55 s: a 1.2 % margin. §2.

---

## 1. Hosts, as found

`intel` and `mac-server` are the same machine, confirmed: both SSH aliases answer
`hostname` = `mac-server`. The matrix's `intel` row is that box.

| | `intel` / `mac-server` | `mini` |
|---|---|---|
| `uname -s -m` | `Linux x86_64` | `Darwin arm64` |
| CPU | `Intel(R) Xeon(R) W-3245 @ 3.20GHz`, 16 c / 32 t | `Mac16,10` (Mac mini M4), 10 cores |
| RAM | 283 GB | `hw.memsize` = 17179869184 (16 GiB) |
| disk | `/` 3.6 T, **994 G – 1004 G free**, 72 % used | `/` 460 Gi, **371 Gi free**, 4 % used |
| load during the run | **19.3 → 24.2** (1-min) | 1.7 → 8.5 |
| cargo | 1.97.1 | 1.96.0 |
| other tenants | **16 `actions.runner.paiml.intel-clean-room*` services running**, one `Runner.Worker` active with ~13 `rustc` and two `realizar-*` test binaries (4.9 G + 2.8 G RSS) | none |
| `apr serve` RSS, model loaded, c=1 | 12.99 GB | **11.84 GB of 16 GB (74 %)** |

Two corrections to `docs/PERF-054-campaign-scope.md` §3, both in the campaign's
favour: `intel`'s disk is **72 % used with ~1 TB free**, not "345 G free (91 % used)";
and its cargo target dir already held a warm 2.1 G build. Nothing was installed on
either host. Nothing was removed. Peak added disk was the incremental `target/`
growth of one `--bin apr` build.

`mini` has no `cmake` and no llama.cpp tree (`~/src/llama.cpp` does not exist).
`intel`'s `~/src/llama.cpp/build/bin/` holds `llama-server` and shared objects and
**nothing else** — no `llama-cli`, no `llama-bench` — and reports `version: 1 (35bee03)`,
not the `7746 (39173bcac)` pin. Both confirm the already-taken decision to drop
Arm B on these two hosts; neither was re-litigated and nothing was built for it.

---

## 2. The blocker: a 120 s hard per-request timeout that `mini` cannot meet

This was not on any prior list and it is the reason `mini/W1` is not schedulable.

Spec §4.4.3 defines `timeouts` as "120 s hard per request". The implementation:

```
crates/aprender-test-lib/src/perf_gate/protocol.rs:33
    pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

crates/aprender-test-lib/src/llm/band.rs:235
    let timed_out = tokio::time::timeout(w.timeout, issue(&w.client, prompt, w.stream)).await;

crates/aprender-test-lib/src/llm/client.rs:240
    .timeout(Duration::from_secs(120))          # the reqwest client, which also bounds warmup
```

`BandConfig::conformant` always sets it, `BandConfig::relaxed` is unreachable from
the CLI, and `apr test llm bench --band` exposes no flag that changes it — by
design (`test_llm_band.rs` header: "there is deliberately no flag that shrinks the
window").

Measured `t_req` against that ceiling:

| host | `t_req` warmed | margin to 120 s | first (unwarmed) W1 request |
|---|---|---|---|
| `intel` | 73.9 – 81.9 s | **+31.8 % at worst observed** | **118.55 s → +1.2 %** |
| `mini` | 270.9 – 271.1 s | **−126 %: exceeded on every request, 4/4** | 270.21 s |

Consequences, from the harness's own code paths:

- **`mini`.** Every sampled request is killed at 120 s and recorded as
  `Outcome::Timeout`; `completed` = 0, so `agg_tok_s` has an empty numerator and
  `bootstrap_agg_tok_s_ci` returns `None` (it needs n ≥ 2 completions). Warmup is
  bounded by the same 120 s through the reqwest client, so `warmup_completed` = 0
  too. `violations()` correctly emits *"§4.4.2 only 0 of 30 required sampled
  requests completed"* — the harness fails honestly, it does not fabricate. But a
  c=1 band still costs 2 × 120 s warmup + 5 s + 30 × 120 s ≈ **64 minutes to
  produce nothing**. Across the cell — `ceil(n/c) × 120 s` per band once `c`
  workers time out in parallel — that is ≈ **6.2 h of guaranteed-empty receipts**.
- **`intel`.** Fits when warm. The 118.55 s first-request figure is the warmup
  rule earning its keep: it is the *first* W1-size request after server start
  (first large KV-cache allocation on that path), and every request after it landed
  at 73.9 – 81.9 s. But `2 × c` warmup requests are themselves only bounded by the
  reqwest timeout, so at c=16 that is 32 requests each of which must come in under
  120 s on a box whose load moved between 19.3 and 24.2 during a 7-minute window.

**This is a spec-level fact, not an implementation detail**, so "raise the timeout"
is an amendment to §4.4.3, not a patch. It is also arguably the *correct* verdict:
a runtime that needs 270 s to answer one 512-token request is not a serving
configuration anyone should record a scaling number for.

---

## 3. `mini` cannot report `compute_class: metal`, and the reason is that no Metal path exists

`perf-matrix.yaml` declares `mini: compute_class: metal`. On the default build of
the commit-under-test:

```
$ apr serve run --list-devices
accelerators this BUILD can dispatch to:
  cpu     always available

This build has NO accelerator compiled in.
```

`apr-cli/Cargo.toml` defines `wgpu = ["inference"]` — a feature that enables no GPU
backend — and `config.gpu` is read only inside `#[cfg(feature = "cuda")]` blocks
(`commands/serve/mod.rs` header, handlers.rs:776 / :1084). `aprender-serve`'s real
`gpu = ["trueno/gpu"]` is on by default in that crate but nothing in the GGUF chat
path dispatches to it. `/health` on the running server reports `"compute_mode":"cpu"`.

So the question *"does the 7B produce coherent text on Metal"* has no answer,
because **there is no Metal path in `apr` to produce anything on**. Everything
below labelled `mini` is the CPU path on Apple silicon. Confirms
`PERF-054-campaign-scope` §4.3 by execution rather than by reading.

---

## 4. The 7B produces coherent text on CPU — on both hosts, from the same bytes

The model is present and byte-identical on both hosts and matches the runbook:

```
509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c
  intel  /home/noah/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
  mini   /Users/noahgift/models_perf012_7b_q4km.gguf
```

`llama.cpp`'s loader on `intel` reads it as 339 tensors: 141 `f32`, **169 `q4_K`,
29 `q6_K`**.

**Natural prompt, greedy** (`temperature: 0`), *"Write a Rust function that reverses
a string. Code only."* — captured in full and **byte-identical on both hosts**, and
correct:

```rust
fn reverse_string(s: &str) -> String {
    s.chars().rev().collect()
}
```

**The W1 corpus itself.** W1 prompts are synthetic word-salad by construction
(`_meta.provenance`: "SYNTHETIC … from a seeded PRNG over a fixed word pool"), so
"coherent" means the model responds sensibly *to nonsense*. It does, and again
identically across the two architectures — the W1 completions were compared over
their first 200 characters, which is what the probe retained: W1 record `id: 0`, greedy, both hosts:

> "It appears that the code you've provided is a mix of various programming concepts
> and syntax elements from different languages, possibly a combination of Rust,
> JavaScript, and possibly some other lang…"

W1 record `id: 1`, greedy, both hosts:

> "It appears that the provided code snippet is a mix of various programming languages
> and syntax elements, making it difficult to understand its intended functionality.
> However, I can attempt to break down…"

Nothing resembling #2785's `"#.asd0ius totalement…"`.

### What this does to #2785

#2785 says the 7B CUDA path "produces garbage text on baseline and fixed builds
alike", and concludes *"W1 remains unmeasurable on lambda … That is 4 of the 8
matrix cells."*

**Two of those four cells are not affected.** The same GGUF bytes, decoded by the
CPU kernels on two independent architectures — x86-64 AVX-512/VNNI and Apple
arm64 — produce coherent, mutually identical greedy output. So:

- the **GGUF file is exonerated**, as is its quantisation mix and the tokenizer/
  chat-template path;
- the defect is localised to the **CUDA** decode path, which is where #2785's own
  suggested first cut already points (per-weight qtype handling in the graph
  dispatcher). This calibration is independent corroboration of that prior from a
  direction the issue did not have: the issue's control was *a different model*
  (1.5B) on the same backend; this is *the same model* on a different backend.
- `intel/W1` and `mini/W1` are **not blocked on #2785**. They are blocked on §2
  (`mini`) and on wall-clock plus the clean-room concurrency group (`intel`).

---
## 5. Binary provenance — proved by content, not by `--version` (#2768, I-10)

Both hosts ran `apr 0.64.0` from `~/.cargo/bin`, which answers
`error: unrecognized subcommand 'llm'` — `apr test llm bench` was added after the
`v0.64.0` tag. So both were built from source at the commit-under-test:

```
git -C ~/src/aprender fetch origin && git checkout 34248e8fe
nice cargo build --release --bin apr -j 8      # -j 8 + nice, to stay out of the runners' way
```

| | `intel` | `mini` |
|---|---|---|
| build time (warm target dir) | 2 m 59 s (2.1 G) | 1 m 43 s (9.3 G) |
| binary | `~/src/aprender/target/release/apr` | same |
| `sha256sum` | `68efb93ffd949255526d272039e3f74e178d8c0f17627953edd46303367ed1bc` | `c381e2c9a7d26e5aa3617020103c7f55955270c2439fb9124dff048452f27a11` |
| mtime | 2026-08-30 07:57:51 UTC | 2026-08-30 09:56:35 CEST |

`--version` is reported but not relied on (both say `apr 0.64.0 (34248e8fe)`).
The pin is proved by content, with a negative control so the grep is shown able
to return 0:

| `strings -a "$BIN" \| grep -c …` | `intel` | `mini` | why this string |
|---|---|---|---|
| `34248e8fe` | 1 | 3 | the commit-under-test, baked by `apr-cli/build.rs` |
| `receipt.rN.json` | 1 | 1 | exists only in `commands/test_llm_band.rs`, added **after** `v0.64.0` |
| `conformant band protocol` | 2 | 2 | the `--band` clap help, same provenance |
| `PERF-060-NOT-A-REAL-STRING` | **0** | **0** | negative control |

`apr test llm bench --help` resolves on both binaries. Neither host had any `APR_*`
variable set; `APR_LAYER_TRACE` was off and **`grep -c '\[CB-006-OUT\]'` over each
server log returned 0** (#2764).

---

## 6. Protocol — what this run was, and what it was not

**This is NOT a §4.4-conformant measurement, and it is not offered as one.** Stated
plainly so no reader mistakes it for a receipt:

- one client, **no `2 × c` warmup**, **no 5 s quiesce**, no `max(30, 8c)` sample
  window, no 60 s wall-clock floor, **N = 1 per prompt**, no bootstrap CI, no
  receipt emitted, nothing signed;
- `ignore_eos` was **not** sent. It did not bind: every `apr` replicate returned
  exactly 128 content chunks, and the two non-streaming replicates reported
  `finish_reason: length`, so the §4.3.1 token budget was met anyway. (It did bind for the llama.cpp reference in §8, which
  stopped at 102–104 tokens — normalised there.);
- each replicate used a **different** W1 prompt (corpus records 0–4 on `intel`,
  0–3 on `mini`), so these are independent samples of the workload, not repeats of
  one input. Corpus prompts are distinct by design, which also rules out
  prefix-cache help;
- requests went to `POST /v1/chat/completions` with the raw corpus prompt as a
  single `user` message, `max_tokens: 128`, `temperature: 0`, `stream: true`
  (except the first replicate per host, non-streaming, to capture `usage`);
- `prefill_tok_s` is derived as `prompt_tokens / ttft_s`, `decode_tok_s` as
  `(n_chunks − 1) / (t_last − t_first)`, matching §4.4.3's definitions at c=1.

The probe is `evidence/perf-060/probe.py`; raw logs are
`evidence/perf-060/intel-calib.txt`, `evidence/perf-060/mini-calib.txt` and
`evidence/perf-060/intel-llamacpp-reference.txt` (§8). They carry a `.txt`
extension because `.gitignore:38` is `*.log`.

The committed `probe.py` is the run script split into functions, because the
pre-commit complexity gate rejected the single-`main` form it was run as. The split
is mechanical and the only behavioural change is that `text_head` is now sliced to
exactly 200 characters instead of "at least 200"; nothing timed changed. It was
re-run on `mini` afterwards against the same endpoint and corpus to show the
committed file reproduces the measurement — see §7.2's verification line.

### 6.1 `512 ± 8` — resolved, empirically, for the first time

`_meta.token_count_verified` is `false` and `PERF-054-campaign-scope` §4.4 records
that nothing asserts the budget. It can now be asserted for this model and template:

```
usage.prompt_tokens = 513        # identical on intel and mini, W1 record id=0
```

**Inside 512 ± 8.** The count is taken **after** the chat-template wrapper (raw
prompt in, server applies the template, `usage` counts what the model saw) — which
is the side of the boundary §4.3.1 leaves open, and is what a §4.4.6 `tokenization`
block would have to declare. The corpus does not need a `--body-words` retune.

---
## 7. Every replicate

`prefill tok/s` = `513 / ttft_s`. `decode tok/s` is the streamed inter-token rate.
No medians: all of them.

### 7.1 `intel` — five replicates, five distinct W1 prompts, load 19.3 → 24.2

| replicate | W1 record | `total_s` | `ttft_s` | prefill tok/s | `itl_p50` ms | decode tok/s |
|---|---|---|---|---|---|---|
| r0 (non-stream) | 0 | 80.587 | — | — | — | — |
| r1 | 1 | 78.804 | 62.210 | **8.25** | 127.9 | **7.657** |
| r2 | 2 | 73.869 | 57.186 | **8.97** | 129.6 | **7.614** |
| r3 | 3 | 81.325 | 61.994 | **8.27** | 151.9 | **6.574** |
| r4 | 4 | 81.891 | 63.006 | **8.14** | 148.4 | **6.728** |
| *(pre-calibration, first W1 request after server start — unwarmed)* | 0 | **118.552** | — | — | — | — |

- prefill: **8.14, 8.27, 8.25, 8.97** tok/s
- decode: **6.574, 6.728, 7.614, 7.657** tok/s
- `t_req`: **73.869, 78.804, 80.587, 81.325, 81.891** s (+ 118.552 s unwarmed)

Every replicate returned exactly 128 content chunks; r0 reported
`finish_reason: length`.

### 7.2 `mini` — four replicates, four distinct W1 prompts, load 1.7 → 8.5

| replicate | W1 record | `total_s` | `ttft_s` | prefill tok/s | `itl_p50` ms | decode tok/s |
|---|---|---|---|---|---|---|
| r0 (non-stream) | 0 | 270.911 | — | — | — | — |
| r1 | 1 | 271.016 | 216.239 | **2.372** | 431.1 | **2.318** |
| r2 | 2 | 271.110 | 216.318 | **2.371** | 430.7 | **2.318** |
| r3 | 3 | 271.049 | 216.278 | **2.372** | 430.8 | **2.318** |
| *(pre-calibration, first W1 request after server start)* | 0 | **270.208** | — | — | — | — |

- prefill: **2.371, 2.372, 2.372** tok/s
- decode: **2.318, 2.318, 2.318** tok/s
- `t_req`: **270.208, 270.911, 271.016, 271.049, 271.110** s

Dispersion on an idle box is 0.2 s across 271 s. There is no warmup effect visible
on `mini` — unlike `intel`, the first request is the same as the fourth.

**Verification of the committed probe** (see §6): after the calibration, the server
was restarted and the refactored `evidence/perf-060/probe.py` was run once against
W1 record 1 on a freshly loaded model. `total_s` 271.740, `ttft_s` 216.819,
`itl_p50` 432.0 ms, `decode_tok_s` 2.312, 128 content chunks, same completion text
— within 0.3 % of r1 on every field. The committed file reproduces the measurement.

### 7.3 Against the runbook

| | runbook `E` | measured | error |
|---|---|---|---|
| `intel` prefill | 40 tok/s | 8.14 – 8.97 | **4.8× optimistic** |
| `intel` decode | 6 tok/s | 6.57 – 7.66 | 1.2× *pessimistic* — the only estimate that held |
| `intel` `t_req` | 34.1 s | 73.9 – 81.9 s | **2.3× optimistic** |
| `mini` prefill | 60 tok/s | 2.371 – 2.372 | **25.3× optimistic** |
| `mini` decode | 8 tok/s | 2.318 | **3.5× optimistic** |
| `mini` `t_req` | 24.5 s | 270.9 – 271.1 s | **11.1× optimistic** |

The derivation in `PERF-054-campaign-scope` §5 was a memory-bandwidth ceiling.
Bandwidth was not the mistake — §8 shows llama.cpp reaching 54–58 tok/s prefill on
the same `intel` box, close to the 40 the ceiling predicted. The mistake was
assuming `apr` prefills like a batched runtime. §8.1.

### 7.4 Recomputed wall-clock, per §4.4.2, apr lane only, Arm B dropped

`T_band = 2c warmup + 5 s quiesce + max(60 s, max(30, 8c) × t_req)`, N = 3, bands
1/4/8/16, drain not modelled — so every figure is a lower bound. The generator is
`evidence/perf-060/wallclock.py`; feeding it the runbook's own `t_req` reproduces
its 8.88 h and 6.39 h exactly, so the formula is not in dispute, only its input.

| band | `intel` (t_req 73.9 – 81.9 s) | `mini` (t_req 270.9 – 271.1 s) |
|---|---|---|
| c=1, N=3 | 1.97 – 2.19 h | 7.23 h |
| c=4, N=3 | 2.47 – 2.73 h | 9.03 h |
| c=8, N=3 | 4.93 – 5.46 h | 18.1 h |
| c=16, N=3 | 9.85 – 10.92 h | 36.1 h |
| **cell total** | **19.2 – 21.3 h** | **70.5 h** |
| runbook said | 8.9 h apr lane / ~11 h cell | 6.4 h apr lane / ~8 h cell |
| **factor** | **2.2× the apr-lane estimate** | **11.0× — and see §2, it buys nothing** |

And the calibration itself. The runbook budgeted ≈18 min on `intel` and ≈13 min on
`mini` for the 30-request c=1 pass — figures that count only the sampled requests
(`30 × t_req` = 17.1 and 12.2 min). Measured, the same sampled phase is
**36.9 – 40.9 min on `intel`** and **2.26 h on `mini`**; with §4.4.2's warmup and
quiesce, the whole c=1 band is **39.5 – 43.8 min** and **2.41 h**. That is why this
ticket ran 4–5 single-request replicates instead of the 30. Reporting a 5-sample
calibration and calling it 30 would have been the cheaper lie.

`mini`'s 70.5 h is arithmetic, not a plan — no request ever runs to 271 s under the
harness. The real cost is `2 × 120 s + 5 s + ceil(n/c) × 120 s` per band, i.e.
**64 min for the c=1 replicate and ≈ 6.2 h for the whole cell, with zero completed
samples at every band**.

---
## 8. A host-capability reference on `intel` — NOT an Arm B measurement

Arm B is dropped on both hosts and nothing here reopens that. But "the runbook's
estimate was 4.8× optimistic" has two possible causes — the *box* is slower than
the bandwidth ceiling suggested, or `apr` is slower than the box — and only one of
them is actionable. So the llama.cpp binary **already present** on `intel` was run
once against the same model file and the same two W1 prompts, immediately after the
`apr` calibration, then stopped. Nothing was installed or built.

> **This is not, and may not be recorded as, an Arm B number.** That binary is
> `version: 1 (35bee03)`; `scripts/llama_pin.toml` pins `7746 (39173bcac)`. It is a
> reference for *what the hardware can do*, nothing more. `intel`'s Arm B sub-cell
> stays `UNMEASURED`.

Same box, same 4.7 GB GGUF, same prompts, within four minutes of each other, load
20.4 → 21.7 (llama.cpp: `n_threads = 16`, AVX512 + AVX512_VNNI + LLAMAFILE + REPACK):

| | prefill tok/s | decode tok/s | `t_req` (128 gen, normalised) |
|---|---|---|---|
| `apr` @ `34248e8fe` | 8.14 – 8.97 | 6.57 – 7.66 | 73.9 – 81.9 s |
| llama.cpp `1 (35bee03)` | **53.8 / 57.5** | **12.28 / 13.82** | **≈ 19.9 / 18.2 s** |
| ratio | **apr ≈ 6.6× slower** | apr ≈ 1.8× slower | apr ≈ 4.1× slower |

(llama.cpp stopped at 104 and 102 tokens on EOS rather than running to 128;
`t_req` above is renormalised to 128 generated tokens as `ttft + 128/decode`, which
is the like-for-like figure. Raw totals were 18.005 s and 16.301 s.)

**So the box was never the problem.** The scope document's bandwidth-derived
~40 tok/s prefill estimate was a reasonable model *of the hardware* — llama.cpp
gets 54–58 on it. What the estimate got wrong was assuming `apr` prefills with a
batched GEMM.

### 8.1 The mechanism: `apr` has no batched CPU prefill

The measured prefill-to-decode ratio is **1.07× on `intel` and 1.02× on `mini`** —
`apr` prefills a 513-token prompt at almost exactly the rate it decodes one token at
a time. That is the signature of a per-token loop, and it is one, at the serving
path:

```
crates/aprender-serve/src/gguf/inference/generate_quantized.rs:568   # generate_with_cache_streaming
    // Process prompt tokens (prefill)
    for (pos, &token_id) in prompt.iter().enumerate() {
        logits = self.forward_single_with_cache(token_id, &mut cache, pos)?;
    }
```

`prefill_batch` in `gguf/inference/forward/batch_size.rs:282` is documented as
"Prefill prompt tokens with batched forward pass (IMP-106)" and does the same thing,
with the honest comment `// (True batch prefill would compute all positions at once
with causal attention)`; it has no non-test caller. Every batched `prefill_*`
implementation in the tree is under `crates/aprender-serve/src/cuda/`.

For W1 this costs 76–80 % of `t_req`. On `intel`, closing this one gap to
llama.cpp's rate would take `t_req` from ~79 s to ~27 s and the cell from ~20 h to
~7 h — below the runbook's original 8.9 h estimate. On `mini` it is the difference
between a cell that cannot be measured and one that can: `mini` has no llama.cpp so
no ratio is available there, but the hard bound needs none — with prefill reduced to
zero `t_req` is still `128 / 2.32 = 55.2 s`, which is **inside** the 120 s ceiling,
against 270 s today. Prefill alone is what puts `mini/W1` out of reach.

That makes "batched CPU prefill" the single highest-leverage item for this half of
the matrix — worth its own ticket, ahead of scheduling either slot.

---
## 9. `apr` does serialise here — measured, not inherited

The §5 wall-clock arithmetic in `PERF-054-campaign-scope` assumes `max_in_flight = 1`,
so the sampling phase for `n` requests costs `n × t_req` regardless of the band.
That was inherited from PERF-000 rather than observed on these hosts, and the whole
per-cell estimate rests on it, so it was checked on `intel` (short 24-token prompt,
64 `max_tokens`, load 5.7, immediately after the calibration):

| concurrent requests | wall-clock | vs 1× serial |
|---|---|---|
| 1 | 4.54 s | — |
| 2 | 7.84 s | 1.73× (perfectly parallel would be 1.0×) |
| 4 | 14.55 s | 3.20× |

**Serialised.** Aggregate goes from 0.220 to 0.275 req/s between c=1 and c=4 — a
1.25× gain over four workers, i.e. `scaling_efficiency(4) ≈ 0.31`, which is the same
regime the v2.1 `lambda` figures recorded. The assumption holds well enough to
schedule on, with one correction in the campaign's favour: because there *is* a
~1.25× overlap benefit, the c=4/8/16 rows in §7 are **upper bounds**, and the real
cell is plausibly ~20 % below them.

(This is a mechanism check, not an Arm A measurement — 3 samples, no warmup, no
window, W1 not used. Arm A still needs the conformant harness.)

---
## 10. What a full cell actually needs on each host

One correction to `PERF-054-campaign-scope` §0 first: **PERF-054-A has landed.**
`crates/apr-cli/src/commands/test_llm_band.rs` (PERF-025, on main since `9d45b927d`)
builds `perf_gate::ReceiptInput` from `BandInput` and writes `receipt.rN.json` plus
gzipped samples. The scope document's headline — "no conformant receipt producer
exists, 0 of 8 cells measurable" — is no longer true. *Verified by source and by the
subcommand resolving on both binaries; a receipt was deliberately not produced here,
because doing so requires a full band and this ticket forbids one.*

### `intel/W1` — measurable, expensive, and it blocks releases while it runs

| prerequisite | status |
|---|---|
| `apr` at commit-under-test | ✅ built, 2 m 59 s warm, pinned by content (§5) |
| W1 model | ✅ present, sha matches |
| harness reachable | ✅ `apr test llm bench --band` resolves |
| receipt bridge | ✅ landed |
| 512 ± 8 budget | ✅ 513 measured |
| coherent output | ✅ §4 |
| under the 120 s ceiling | ⚠️ yes when warm (32 % margin); the first unwarmed request was 118.55 s |
| Arm B | ❌ `UNMEASURED` — `llama-cli`/`llama-bench` absent, `llama-server` is the wrong build |
| wall-clock | **19.2 – 21.3 h**, apr lane only, N=3, 4 bands (§7) |
| contention | 16 clean-room runners live; I-7 puts the perf run in their concurrency group |

So the cell costs roughly **a full day of exclusive box time during which nothing
can be released** — not the ~11 h (or ~8.9 h apr-lane) the scope document budgeted.
The §9 overlap correction may bring it to ~16 h; it does not bring it to 9.

Two ways to make it schedulable, in preference order:

1. **Land batched CPU prefill (§8.1) first.** It is a single mechanism, `intel`'s
   own llama.cpp shows a 6.6× headroom, and at llama.cpp's prefill rate the cell
   drops to ~7 h — *below* the original estimate. This is the cheap fix.
2. Failing that, schedule it as a declared multi-hour clean-room outage, and expect
   the c=16 replicate alone (3.3 – 3.6 h each, 9.9 – 10.9 h for N=3) to dominate.

### `mini/W1` — not measurable today, and not for a reason more compute fixes

| prerequisite | status |
|---|---|
| `apr` at commit-under-test | ✅ built, 1 m 43 s warm, pinned by content (§5) |
| W1 model | ✅ present, sha matches (renamed) |
| harness reachable | ✅ |
| coherent output | ✅ §4 |
| `compute_class: metal` | ❌ **unreachable** — no Metal path exists in `apr` (§3) |
| under the 120 s ceiling | ❌ **`t_req` = 270 s. Every request aborts.** |
| 16 GB ceiling above c=1 | ❌ `apr` RSS = **11.84 GB of 16 GB at c=1** (74 %) |
| Arm B | ❌ `UNMEASURED` — no llama.cpp, no cmake |
| wall-clock | irrelevant: 0 completed samples at any budget |

**`mini/W1` should not be scheduled.** Committing the ~13 min the runbook budgeted —
or the 135 min the measured `t_req` implies, or the 64 min per c=1 band the timeout
implies — buys a receipt whose every field is a violation. The honest matrix state
for `mini/W1` is `UNMEASURED` with a *named mechanism* rather than a date: blocked
on batched CPU prefill, plus a decision about `compute_class`.

The 16 GB observation stands independently and confirms
`PERF-054-campaign-scope` §4.1 by measurement: 11.84 GB resident at c=1 leaves
4 GB for the OS and every KV cache above the first, so c=8 and c=16 are out of reach
on memory grounds even after the timeout is fixed. `perf-matrix.yaml`'s
`bands: [1, 4, 8, 16]` still disagrees with spec §4.9 about this.

---

## 11. Recommendations

1. **Open a ticket for batched CPU prefill** (`generate_with_cache_streaming`'s
   per-token prompt loop). It is the single mechanism behind both hosts' verdicts,
   it has a measured 6.6× headroom on `intel` against llama.cpp on the same box and
   the same file, and it converts `mini/W1` from unmeasurable to measurable. Nothing
   else in this calibration is worth doing first.
2. **Do not schedule `mini/W1`.** Record `UNMEASURED` with the mechanism, not a date.
3. **Do not schedule `intel/W1` yet either** — it is ~20 h of clean-room outage for
   a number that item 1 would improve by ~3× and make cheaper to take. If it must be
   taken now, take it as a declared outage and expect ~a day.
4. **Retire `mini: compute_class: metal` from `perf-matrix.yaml`**, or fix the class
   producer. A receipt claiming `metal` from this build would be an I-2 violation and
   `bench_receipt.py` would pass it. (Matrix edit, not made here.)
5. **Amend §4.4.3's 120 s** or accept it as a de-facto floor on serving performance.
   It is currently an unstated eligibility criterion: a host slower than
   `512/prefill + 128/decode = 120 s` is excluded from the matrix by a constant, and
   nothing in the matrix says so.
6. **Update `PERF-054-campaign-scope` §0 and §5**: the receipt bridge has landed, and
   the `intel`/`mini` wall-clock rows are 2.2× and (unboundedly) optimistic.
7. Cross-reference this file from **#2785** — its "4 of the 8 matrix cells" is 2.

---

## 12. Host hygiene

Both boxes are shared. Nothing was installed, nothing was removed, no CI job was
touched, and every process this ticket started was stopped and its absence
confirmed with `ps -eo pid,rss,args` (`pgrep -f "apr serve run"` self-matches the
SSH command line and reports a false positive — it did, twice).

- `intel`: `apr serve run` and the llama.cpp reference process stopped; `ps` clean.
  RSS back to 7 GB of 283 GB. Disk at the end: 820 G free / 77 % used — the change
  from 1004 G is CI traffic on the box, not this run, whose only footprint was the
  incremental `target/` growth of one `--bin apr` build. Load rose from 19 to 46
  during the session, again CI, and the calibration replicates each record the load
  at which they were taken.
- `mini`: `apr serve run` stopped; nothing else was started. 371 Gi still free.
- Both hosts are left checked out at `34248e8fe` (`intel` was on `main`, `mini` on
  `de8fbc407`). Raw logs remain at `/tmp/perf060-*.log` on each host and are copied
  into this directory.
