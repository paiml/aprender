# PMAT-3794 — `apr chat --no-gpu` uploads to CUDA anyway; chat reports no backend

**Row:** GH #3794, milestone 0.69.1. **Worker:** aprender-d8. **Cop:** aprender-3e.
**Branch:** `PMAT-3794-chat-no-gpu`, rebased onto `origin/release/0.69.1-batch-2`.

## Verdict

Both `done_when` items met, each with a measurement rather than an assertion:

1. `--no-gpu` now initialises no accelerator for **any** of the three formats.
   Peak VRAM under `--no-gpu`: **4070 MiB → 0 MiB**.
2. `apr chat --json` now emits `backend: {requested, ran, fell_back}`, with `ran`
   recorded by the generate branch that actually answered.

## Provenance

| | |
|---|---|
| Host | this dev box, NVIDIA RTX 4090 sm_89 |
| Model | `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (the issue's model) |
| Build | `--features cuda` — the defect only exists on a CUDA build |
| Binary | `apr=scratchpad/apr-3794-v2 sha256=37ad2456fc22bfc54a0f2520 version=apr 0.69.0 (3fe9a6cff)`, snapshotted outside any cargo target dir |

## 1. The defect: `--no-gpu` was never consulted at the upload site

`ChatSession::new(path)` took **no `force_cpu` argument at all** — the signal did
not reach the function, so there was nothing to check. Generation honoured it
(`chat_generate_session_02.rs`), so the answer was correct and came from the CPU;
the session had nonetheless built an `OwnedQuantizedModelCuda`, uploaded the
weights, and held the VRAM for its whole lifetime.

This is the same shape as #3757: **a signal that exists and is simply not read on
the path that matters.** There, `accel_forced` reached the post-hoc verdict but
not the attempt. Here, `force_cpu` reached generation but not the upload.

### Why it is worse than wasted memory

VRAM taken outside `/tmp/apr-gpu.lock` is invisible to `gpu-q`, so a `--no-gpu`
run could starve the serialization rationing the card between agents — the lock
cannot account for memory it never saw claimed. A `--no-gpu` flag that takes the
GPU also fails its own claim on its face.

## 2. Measured, before and after

Peak VRAM attributable to the chat process, sampled from
`nvidia-smi --query-compute-apps=pid,used_memory` for the duration of a
two-turn session:

| invocation | peak VRAM |
|---|---|
| `apr chat --no-gpu` **before** | (issue) `[GGUF CUDA: RTX 4090 (24035 MB VRAM) — pre-cached]` |
| `apr chat --no-gpu` **after** | **0 MiB**, and no `[GGUF CUDA …]` line |
| `apr chat` (default) **after** | **4070 MiB** — capability preserved |

Answers unchanged under `--no-gpu`: `2 + 2 equals 4.` / `4 * 3 equals 12.`

## 3. The fix

`force_cpu` is threaded into `ChatSession::new`, and the three CUDA
initialisation sites now share **one** named predicate:

```rust
fn cuda_preload_allowed(force_cpu: bool, format: ModelFormat) -> bool { !force_cpu }
```

The issue names only GGUF. `try_init_apr_cuda` and `try_init_safetensors_cuda`
have the same shape and are gated too — `done_when` says "does not initialise
CUDA at all", and a fix that covered one of three would be the same bug again for
the other two. The predicate is shared precisely so a regression cannot drop the
check from one site and leave the others looking correct.

## 4. The backend report (`done_when` 2)

`apr chat --json` previously **accepted the flag and printed no JSON at all**, so
a harness could not hold chat to its lane the way the `run` cells hold `apr run`.

```
$ apr chat model.gguf --no-gpu --json …
{"backend":{"requested":"cpu","ran":"cpu","fell_back":false}}

$ apr chat model.gguf --json …
{"backend":{"requested":"default","ran":"gpu","fell_back":false}}
```

`ran` is **recorded by the generate branch that answered** (a
`generated_on_gpu` flag set inside the CUDA branches, mirroring the existing
`had_generate_error` pattern), not inferred from the flags. Inferring it would
make the report agree with the flags by construction and therefore incapable of
reporting a disagreement — which is the whole reason the field exists.

`fell_back` is the field a harness acts on and the easiest to get backwards:
**only an unasked-for demotion counts.** Asking for CPU and getting CPU is not a
fallback; never asking and getting GPU is not a fallback.

## 5. Case tables and mutant

* `pmat3794_chat_cuda_preload_gate` — all three formats, both failure directions
  (the defect, and the over-correction of disabling the accelerator outright).
* `pmat3794_chat_backend_report` — four `(force_cpu, generated_on_gpu)` states,
  including `force_cpu` + GPU, which is impossible after this fix and is reported
  honestly rather than smoothed over, so the contradiction stays visible if the
  gate regresses. `fell_back` is asserted separately so a regression in it cannot
  hide inside a whole-string diff.

**Mutant: remove the gate** (`cuda_preload_allowed` → `true`, the pre-fix state).
RED, naming every affected format and its call site:

```
#3794 REGRESSION: --no-gpu / force_cpu would still initialise CUDA for 3 of 3
formats, holding VRAM outside /tmp/apr-gpu.lock for a run that asked for none
(measured 4070 MiB peak before the fix):
  - Gguf (try_init_gguf_cuda — the format #3794 measured)
  - Apr (try_init_apr_cuda)
  - SafeTensors (try_init_safetensors_cuda)
```

## 6. Checks

`cargo fmt --all --check` rc=0 · `clippy -p apr-cli --lib -D warnings` rc=0 ·
`cargo test -p apr-cli --lib` **7326 passed, 0 failed**.

One clippy finding surfaced under `--features cuda` and is **not mine**: an
unused `GemmOp` import in `aprender-compute`, present in the clean batch-2
checkout and untouched by this branch (`git diff --name-only` lists no
`aprender-compute` file). The cop has since landed a fix for it on batch-2.

## 7. Not claimed

* Not measured on lambda or gx10. The fix is a flag-threading change with no
  host-specific term, and the VRAM evidence is from this box's RTX 4090.
* The APR and SafeTensors gates are covered by the case table but were **not**
  exercised end-to-end with a real model of those formats — only GGUF was
  measured. The predicate is shared, so the claim is structural for those two.
* `--json` emits a session-final backend line, not a per-turn transcript.
  `done_when` allows either ("a `--format json` transcript, **or** a final JSON
  line"); a per-turn transcript would change chat's stdout contract more than a
  release-night row should.
