---
spike: 021
idea: mcp-model-hosting-aws
name: kev-mcp-default-lambda
type: comparison
validates: "Given Kev-0.8B wrapped as a stateless pmcp MCP server on arm64 Lambda (10,240 MB), when tools/call decide hits a cold environment, then the container-init -> weights-loaded -> first-decision timeline and real Graviton decision latency are measured for weights baked into the image vs streamed from S3"
verdict: PARTIAL
related: [019, 020, 022, 023]
tags: [lambda, cold-start, mcp, pmcp, graviton, deployment, kev]
---

# Spike 021: Kev MCP server on default Lambda — cold start, measured

## What This Validates
Spike 019 measured Kev on a *local proxy* of Lambda and left two numbers as estimates: decision latency on
Graviton, and the cold read of 3 GB of weights. This spike deploys the real thing — Kev-0.8B behind one
stateless pmcp `decide` tool — to arm64 Lambda at 10,240 MB, in two variants:

| Variant | Package | Weights |
|---|---|---|
| **021-a lambda-baked** | container image (3.2 GB), `provided.al2023-arm64` base | `COPY` into the image at `/opt/kev.gguf`, mmap'd |
| **021-b lambda-s3** | zip (13.6 MB `bootstrap`) | parallel ranged GETs from S3 into `/tmp` (4 GB ephemeral), then mmap'd |

## Research
- Lambda at 10,240 MB buys ~6 vCPU; `/tmp` is configurable to 10,240 MB; container images up to 10 GB are
  lazily loaded from a regional cache. The init phase of an on-demand function is capped (10 s), so the model
  loads inside the **first invocation** (the `aprender-mcp-chronos-lambda` pattern: loopback pmcp server started on
  first request) — `EAGER_LOAD=1` moves it into init for comparison.
- Account quotas (us-east-1, read 2026-09-23): no memory cap below 10,240 MB (the function was created at 10,240);
  1,000 concurrent executions; Fargate 140 on-demand vCPU. The SetFit trainer's 3,008 MB OOM was its own setting.
- Forcing a cold start: changing an environment variable publishes a new configuration, so the next invoke gets a
  fresh execution environment (confirmed per call by `first_call_in_process` and the REPORT line's `Init Duration`).

## How to Run
```bash
# build (aarch64, glibc <= 2.34 for provided.al2023)
CARGO_TARGET_DIR=<repo>/target cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.34
# local smoke test of the same MCP server
KEV_GGUF=../017-kev-rust-forward-parity/models/kev-0.8b-merged-f32.gguf PORT=8791 <repo>/target/release/server &
python3 tools/probe.py http http://127.0.0.1:8791/ --row 1 --warm 3
# cloud: images are built IN AWS by CodeBuild (deploy/buildspec.yml) from S3, never pushed from the laptop
uv run --with boto3 python tools/lambda_cold.py kev-spike-lambda-s3 --rounds 3 --out results/lambda-s3.jsonl
uv run --with boto3 python tools/lambda_cold.py kev-spike-lambda-baked --rounds 3 --out results/lambda-baked.jsonl
```

## Investigation Trail
1. **Local MCP smoke test (M4 Pro, 6 threads)**: the whole path — pmcp stateless streamable-HTTP →
   `decide` tool → batched prefill → pointer head — answers row 1 (81 tokens) with |Δp| 4.8e-7 vs Python fp32,
   437 ms first decision, 361 ms warm p50. Load from a **cold file cache** took 5.2 s (3 GB at ~580 MB/s from the
   SSD) — spike 019's 0.4 s was a warm page cache. Cold-start cost is bytes, as predicted.
2. **The laptop uplink is 3.7 MB/s** (measured with a 32 MB put): every 3 GB transfer costs ~14 min, and the
   default `aws s3 cp` (10 parallel 8 MB parts) dropped connections three times. Weights went up once with 2 × 16 MB
   parts; the baked images are built by CodeBuild inside AWS from S3 — the only sane way to ship multi-GB model
   images, and the pattern a real pipeline would use anyway.
3. `aws-sdk-s3` ≥ 1.138 requires rustc 1.94.1 against the repo's pinned 1.93: resolved with
   `CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo update` (aws-sdk-s3 1.137, aws-config 1.8.18).
4. **First cloud cold start (S3 variant)**: 42.9 s client wall; the timeline says **37.7 s of it is the S3 download
   at 80 MB/s**, then 3.1 s to build the model and 1.15 s to decide on a **Graviton2** (MIDR 0xd0c). 80 MB/s = 640
   Mbit/s — the account's Lambda quota list carries "Network bandwidth per execution environment: 625".
5. **Varied the input before believing a cap** (32 x 32 MB parts instead of 16 x 64 MB): 72 MB/s. Parallelism
   does not buy bandwidth — it is a per-environment ceiling. The first 32-part attempt **failed mid-body**
   (`streaming error` from the S3 body stream) and my downloader had no per-part retry; the load was not cached,
   so the next call paid the whole download again. Added a 5-attempt retry per part.
6. **Lambda arm64 hardware is not uniform**: the retried environment ran on **Graviton3** (MIDR 0xd40). The
   controlled rounds that followed all landed on Graviton2.
7. **Baked-image variant: 800 s** for the first cold call — **794 s inside `mmap`** (the GGUF open touches the
   whole file; RSS 2.9 GB after it). Reading 3 GB through Lambda's lazily loaded container-image store ran at
   ~3.8 MB/s. A second forced cold start (after the image had been read once) took **309 s** (304 s in the same
   step). Image chunk caching helps ~2.6x and is still ~7x slower than S3.
8. Image activation (`Pending` → `Active` after `CreateFunction` with a 3.2 GB image) took 50 s — a deploy-time
   cost, not a request-time one.

## Results

**Default Lambda, arm64, 10,240 MB (6 vCPU), Kev-0.8B f32 (3.02 GB), row 1 = 81 tokens, us-east-1**

| variant | cold start → first MCP answer | where it goes | warm decision p50 | peak memory |
|---|---|---|---|---|
| **S3 → /tmp** (3 controlled runs) | **42.8 / 43.2 / 43.2 s** | init 0.07 s · S3 37.7 s (80 MB/s) · build 3.1 s · decide 1.15 s | **1.13–1.16 s** (Graviton2) | 7.8–8.1 GB |
| **baked into image** | **800 s** first ever · **309 s** second | image-store read 794 / 304 s · build 4–5 s | – (not re-measured) | 6.8 GB |

Parity vs Python fp32 on every call: **|Δp| 4.8e-7**. Default-Lambda Graviton2 is **3.2x slower** per decision than
the laptop (1.15 s vs 0.36 s) — inside spike 019's 1–1.5 s estimate.

**Verdict: PARTIAL ⚠.** Kev-0.8B *runs* as an MCP server on default Lambda, scales to zero and is correct —
but a cold start is **~43 s, set by a platform network cap (80 MB/s per environment) that no client-side
parallelism moves**, and a warm decision is 1.1 s on Graviton2. Baking weights into the image is not a fallback: it
is 7–18x worse. Default Lambda is the right home for small models (SetFit, Chronos-Bolt), not for multi-GB ones
behind an interactive agent.

**Signal for the build**
- **Weights never go in a Lambda image** above a few hundred MB. S3 with parallel ranged GETs + per-part retry.
- A multi-GB model on default Lambda needs its cold start *hidden* (MCP Tasks / async), not optimised: the floor is
  bytes ÷ 80 MB/s. bf16 (1.5 GB) would halve it to ~19 s once the BF16 prefill GEMM exists.
- `LogType=Tail` gives `Init Duration`/`Max Memory Used` on default Lambda (not on LMI — see 023).
- Report which Graviton answered (`/proc/cpuinfo` CPU part): latency differs by generation within one function.
