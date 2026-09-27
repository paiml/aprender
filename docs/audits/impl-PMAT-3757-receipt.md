# PMAT-3757 — default `apr run` attempts wgpu, pays 1.7 GB, then rejects its own result

**Row:** GH #3757, milestone 0.69.1. **Worker:** aprender-d8. **Cop:** aprender-3e.
**Branch:** `PMAT-3757-default-backend` off `release/0.69.1-batch-2` @ `9f8836c71`.

## Verdict

**It reproduces on batch-2.** Batch-1's 148 commits did not fix it, and the issue
is not merely a description of the published 0.68.2 artifact.

Root cause is a **transitive default-feature leak**, and the fix makes the wgpu
fallback require an explicit accelerator request. The default path is now 2.4×
faster and allocates 1.7 GB less, for the identical answer from the identical
backend.

## Provenance

| | |
|---|---|
| Host | this dev box, NVIDIA RTX 4090 sm_89 |
| Model | `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (the issue's model) |
| Build | **default features** — what `cargo install aprender` produces. Not a `--features cuda` build, which would answer a different question |
| Target dir | `/mnt/nvme-raid0/cargo-target/PMAT-3757`, dedicated, so the shared `target/release/apr` is not clobbered |
| BEFORE | `apr=scratchpad/apr-3757 sha256=97da4f4571668e4102d74a76 version=apr 0.69.0 (9f8836c71)` |
| AFTER | `apr=scratchpad/apr-3757-fixed sha256=f5d9ce80a4cfd966065f7f53 version=apr 0.69.0 (9f8836c71)` |

Both binaries report the same `--version`, because apr-cli embeds the SHA of
HEAD and the fix was measured before commit. They are distinguished **by
sha256**, which is why the rule says to cite it.

## 1. It reproduces — verbatim, on batch-2

```
$ apr run qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    --prompt "What is 2+2? Answer with just the number." --max-tokens 16 --format json
Backend: wgpu (Vulkan)
Preparing GPU weights: dequantizing 28 layers to F32
GPU weights ready: 169 tensors, 1726.8 MB F32
warning: GPU (wgpu) path rejected, attempting fallback: cosine vs CPU = 0.955376 (< 0.99) at step 2/3
  "used_gpu": false,   "text": "4"
```

`1726.8 MB` and `cosine 0.955376` match the issue's intel row **to four
significant figures**, on completely different silicon. The divergence is the
wgpu path's own numerics, not a driver — consistent with the issue seeing the
same number on intel, gx10 and an Apple M4.

## 2. The detour, priced (the issue left this unmeasured)

| invocation | `inference_time_ms` | wall | wgpu attempted |
|---|---|---|---|
| **default (no flags)** | **7607.55** | 7.66 s | yes — 1726.8 MB F32, then rejected |
| `--no-gpu` | 3034.86 | 3.06 s | no |
| `--backend cpu` | 3161.61 | 3.19 s | no |

**2.5× wall and 1.7 GB, to reach the identical answer through the identical CPU
backend.**

## 3. The default was the only dishonest path

| invocation | behaviour on a default build |
|---|---|
| `--gpu` | **rc=9, refuses**: "this build has no GPU backend compiled in" |
| `--backend wgpu` | **rc=14, refuses**: attempts, fails parity, will not report a fallback as success (R-0b, #3002) |
| **default (no flags)** | **rc=0** — silently attempts, pays, fails, falls back, reports success |

So the binary already told the truth on both explicit paths, and contradicted
itself on the default: `--gpu` says there is no GPU backend compiled in, while
the bare default silently used one.

## 4. Root cause: a transitive default-feature leak

```toml
# crates/aprender-serve/Cargo.toml
default = ["server", "cli", "gpu"]
gpu = ["trueno/gpu"]          # wgpu

# crates/apr-cli/Cargo.toml
realizar = { workspace = true, optional = true }   # ← no default-features = false
inference = ["realizar", ...]                      # in `default`
wgpu = ["inference"]                               # pulls in NO wgpu dependency
```

apr-cli's own feature list has no wgpu — its `wgpu` feature is a no-op alias for
`inference` — but its `realizar` dependency carries realizar's `default`, which
includes `gpu`. So every `cargo install aprender` compiles the wgpu path in. The
built default binary contains **2798** wgpu/vulkan strings.

That is exactly the contradiction the issue's point 4 names: the
`check_multiplatform_dogfood.sh` comment claims "`cargo install aprender` builds
CPU-only on every host in this matrix (crates/apr-cli/Cargo.toml `default`
carries no cuda, no wgpu)" — true of apr-cli's own table, false of the artifact,
because the claim was checked against the wrong manifest.

The attempt itself was gated on `!config.no_gpu` alone, so with no flags it ran.

## 5. The fix

`accel_forced` — "the user EXPLICITLY asked for an accelerator (`--gpu`, or
`--backend cuda|wgpu|gpu`)" — **already existed**, added by #3602, and was
already threaded as far as `reconcile_accelerator`, the POST-HOC verdict that
produces the `--backend wgpu` rc=14. It was not consulted by the ATTEMPT.

This threads the same signal one step further — `RunOptions` → `InferenceConfig`
→ the attempt — and extracts the rule into a pure predicate so it has a case
table instead of living inline in two `if`s that could drift:

```rust
pub(crate) fn wgpu_fallback_allowed(no_gpu, accel_forced, has_legacy_quant) -> bool {
    !no_gpu && accel_forced && !has_legacy_quant
}
```

**Deliberately not done:** setting `default-features = false` on the realizar
dependency. That would also drop `server`/`cli` and change what is compiled into
the artifact — a bigger change than a release-night row should carry. It remains
the right follow-up if the goal is for the shipped default binary to stop
*containing* wgpu, rather than merely stop *using* it. Filed as a note, not done
here.

## 6. Measured after the fix

| invocation | result |
|---|---|
| **default** | **3162.61 ms**, `"text": "4"`, **zero** wgpu/dequant lines in stderr |
| `--backend wgpu` | rc=14, still prints `Backend: wgpu (Vulkan)` — capability preserved, opt-in |
| `--gpu` | rc=9, still refuses — unchanged |

7607.55 ms → 3162.61 ms on the default path, and the 1726.8 MB F32 copy is never
allocated.

## 7. The case table and its mutant

`pmat3757_wgpu_attempt_gate` covers five invocations, including the two failure
directions: the defect (bare run attempts wgpu) and the over-correction (the fix
removing the backend instead of making it opt-in). The whole table is evaluated
before asserting, so a regression names every invocation it broke.

**Mutant: delete the `accel_forced` conjunct** — the state batch-2 was in. RED:

```
#3757 REGRESSION: bare `apr run model.gguf` — the #3757 defect: on a default
(non-cuda) install this dequantized 1726.8 MB to F32, failed wgpu's cpu-parity
gate at cosine 0.9554 and fell back to CPU, costing 7607 ms against --no-gpu's
3035 ms for the identical answer

1 of 5 wgpu-attempt cases are wrong:
  - no_gpu=false accel_forced=false has_legacy_quant=false: expected allowed=false, got true.
```

This is the gap `FALSIFY-BACKEND-CUDA-HONESTY-001` left: it asserts a run never
prints `Backend: wgpu`, but only on the explicit `--backend cuda` path. Nothing
covered the bare invocation, which is the one that shipped broken.

## 8. Two defects found alongside, NOT fixed here

1. **The JSON lies about the fallback.** The before-run reported
   `"backend": {"requested": "default", "ran": "cpu", "fell_back": false}` while
   stderr said `GPU (wgpu) path rejected, attempting fallback`. A machine-readable
   field contradicting the human-readable log on the same run — any consumer
   counting `fell_back` sees none.
2. **A stale proof claim.** `try_wgpu_generate`'s doc says
   *"Proven: cosine=0.999863 on Blackwell sm_121"*. The issue measured **0.955046
   on gx10, which IS Blackwell (GB10)**. The proof does not describe the current
   path, on the very architecture it names.

## 9. Checks

`cargo fmt --all --check` rc=0 · `clippy -p aprender-serve --lib -D warnings`
rc=0 · `clippy -p apr-cli --lib -D warnings` rc=0 ·
`cargo test -p aprender-serve --lib infer::` **757 passed** ·
`cargo test -p apr-cli --lib` **7321 passed**.

## 10. Not claimed

* Not measured on intel, gx10 or mini — the fix is a routing change with no
  host-specific term, but the issue's three hosts are not re-measured here.
* Not claimed that wgpu is now correct. It still fails its own cpu-parity gate;
  this makes the user opt in to that rather than pay for it unasked. Whether the
  wgpu path should be repaired or removed is a separate row (see §8.2).
