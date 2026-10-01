---
spike: 019
idea: llm-decision-classifier
name: kev-lambda-inference
type: standard
validates: "Given the Rust Kev path from 017 + 020, when it runs as a Lambda would (6 threads, 10,240 MB), then binary and weight size, peak memory, cold start and per-request latency are measured against Lambda limits"
verdict: PARTIAL
related: [015, 017, 020]
tags: [kev, lambda, cold-start, memory, latency, deployment]
---

# Spike 019: Kev on Lambda (local proxy)

## What This Validates
Whether the Rust Kev inference path fits AWS Lambda: 10,240 MB memory (which buys 6 vCPU), 10 GB container image,
no GPU. Measured with a local proxy (decided 2026-09-23): a fresh process per run, `RAYON_NUM_THREADS=6`, peak RSS
from `/usr/bin/time -l`, on the M4 Pro. No AWS resources were created.

## How to Run
```bash
CARGO_TARGET_DIR=<repo>/target cargo build --release    # needs spike/016-upstream-sync @ 32103ba83
M=../017-kev-rust-forward-parity
RAYON_NUM_THREADS=6 /usr/bin/time -l <repo>/target/release/kev-lambda-inference \
  $M/models/kev-0.8b-merged-f32.gguf $M/models/kev-0.8b-head.safetensors $M/fixtures/kev-0.8b_fixture.json [--keep-mmap]
```

## Investigation Trail
1. **Binary is 1.4 MB** (`aprender-serve` with default features off): the deployment size is the weights.
2. **RSS timeline showed the load is wasteful**: mmap 52 MB → base model **2,963 MB** → owned layers 6,765 MB → drop
   mmap 3,885 MB. The base step owns the 1.0 GB f32 embedding AND a 1.0 GB owned `lm_head` (Qwen3.5 ties them; the
   loader materialises the head anyway) — Kev never calls `lm_head`. The layer step copies 2 GB of blocks out of the
   mmap while the mmap's touched pages stay resident.
3. **Dropping the mmap after the owned model is built returns 2.9 GB** (6,765 → 3,885 MB) with identical outputs, so
   nothing reads the map after construction. The transient peak (7.1 GB max RSS) is set during loading.
4. Steady-state latency is flat across rounds (p95 within 1–2 % of p50): no allocator or thermal drift over 50 requests.

## Results

**Size**

| artifact | bytes |
|---|---|
| binary | 1.4 MB |
| Kev-0.8B GGUF, f32 (merged LoRA) | 3.02 GB |
| same, bf16 / f16 | 1.52 GB (prefill GEMM is F32-only today, spike 020) |
| pointer head + temperature | 2 MB |

**Cold start** (fresh process, OS file cache warm)

| step | t since start | RSS |
|---|---|---|
| mmap GGUF | 21 ms | 52 MB |
| base (embeddings, norms, lm_head) | 184 ms | 2,963 MB |
| owned layers | 402 ms | 6,765 MB |
| drop mmap | 420 ms | 3,885 MB |
| **first decision complete** (87 tokens) | **819 ms** | 3,933 MB |

Peak RSS 7.1 GB (`time -l`), steady 3.9–4.1 GB; macOS "peak memory footprint" (dirty memory) 4.3 GB.

**Steady state, 6 threads (5 rounds per request)**

| request | questions | tokens | p50 | p95 |
|---|---|---|---|---|
| short text (tweets) | 1 | 36–42 | 185–207 ms | 188–208 ms |
| stance tweet | 1 | 81–87 | 352–375 ms | 354–383 ms |
| support ticket | 3 | 145 (state paid 3×) | 711 ms | 715 ms |
| long document | 1 | 915 | 3.73 s | 3.77 s |

Worst |Δp| vs Python fp32 across every request: 1.3e-6.

**What the proxy cannot measure, stated as estimates**
- **CPU gap to Lambda.** Lambda arm64 is Graviton2 (Neoverse N1, 2×128-bit FMA, ~2.5 GHz ≈ 40 GFLOP/s/core peak);
  an M4 P-core peaks ≈ 4× that. The GEMM-bound 87-token decision (~0.37 s here) projects to **~1–1.5 s on a 6-vCPU
  arm64 Lambda**. Unmeasured; a real deploy is the only way to settle it.
- **Cold read of 3 GB.** Here the file cache is warm. A Lambda container image pulls layers lazily from its cache; a
  first invocation must read 3 GB of weights, which is seconds-to-tens-of-seconds depending on image caching. bf16
  weights halve it; provisioned concurrency hides it.

**Kev-4B projection** (from `Qwen3.5-4B-Base/config.json`: hidden 2560, 32 layers, FFN 9216, 32 value heads)

| | Kev-0.8B | Kev-4B |
|---|---|---|
| block matmul params | 0.50 B | **3.57 B (7.2×)** |
| f32 weights | 3.0 GB | **16.8 GB — exceeds Lambda's 10 GB** |
| bf16 weights | 1.5 GB | 8.4 GB — fits only without the extra lm_head / mmap copies |
| 87-token decision, M4 6 thr | 0.37 s measured | ~2.1 s projected (GEMM-scaled) |
| same, Graviton2 6 vCPU | ~1–1.5 s est. | ~6–9 s est. |

**Verdict: PARTIAL ⚠.**
- **Kev-0.8B fits Lambda**: 1.4 MB binary + 3 GB weights, 3.9 GB steady (7.1 GB transient) under 10 GB, a short
  decision in ~0.2–0.4 s on 6 M4 threads (est. ~1–1.5 s on Graviton2), parity 1.3e-6.
- **Kev-4B — the size that beat SetFit in spike 015 — does not fit as built**: f32 weights exceed Lambda memory, and
  a BF16 prefill GEMM does not exist yet. Even with it, a decision is multi-second on Lambda CPU.
- For comparison, the deployed SetFit Lambda answers in ~32 ms warm with a 91 MB artifact. Kev on Lambda buys
  zero-shot steering and calibrated multi-question answers at 10–40× the latency.

**Cheap wins found for the real build**: skip the unused `lm_head` (−1 GB for 0.8B, −2.5 GB for 4B at f32); build
owned layers and release the mmap (already −2.9 GB), or keep the mmap and borrow F32 weights zero-copy instead of
owning them; reuse the state's recurrent state across a request's questions (the 3-question ticket pays the state
three times); BF16 GEMM for 4B.
