---
spike: 026
idea: llm-decision-classifier
name: laya-mcp-default-lambda
type: standard
validates: "Given the spike-025 Rust Laya behind a stateless pmcp `decide` tool that takes a real /v1/systemone-shaped request, when it runs on default arm64 Lambda (10,240 MB and 4,096 MB) cold and warm, then cold start, Graviton latency, memory and cost per decision are measured against Kev-0.8B's spike-021 numbers (43 s cold / 1.15 s warm)"
verdict: VALIDATED
related: [025, 021, 024, 022, 023]
tags: [laya, lambda, cold-start, mcp, pmcp, graviton, deployment, cost]
---

# Spike 026: Laya MCP server on default Lambda

## What This Validates
The same measurement as spike 021 (Kev) for the model spike 024 found better: Laya-en, served as a thin stateless
pmcp MCP server on default Lambda. Unlike 021 the tool takes a **production-shaped request** — `state` + typed
`questions` (`choice` / `score` / `noul` with `instructions`, `criteria`, optional `labels`) — so option rendering,
tokenization, the forward and the calibrated probabilities are all on the served path.

## Research
- Reuses spike 021 wholesale: loopback `bootstrap` shim, `server` container bin, per-part-retrying parallel S3 GETs,
  response forensics (load timeline, Graviton generation, RSS), the API-GW v2 invoke probe, the IAM role and bucket.
- Weights stay the published **F16 `model.safetensors` (843 MB)** — spike 021 showed cold start = bytes / ~80–95 MB/s,
  so F16 on the wire and widening to f32 in memory is the right trade. Files: `model.safetensors`,
  `tokenizer/tokenizer.json`, `encoder/config.json`, `rl_agent_config.json` under `s3://…/laya-en-55cf4c4e/`.
- `render_options` ported from `laya/common.py` (choice object/list, score list, noul with default or custom labels);
  `serde_json` `preserve_order` keeps criteria in caller order, which IS the label index.

## How to Run
```bash
CARGO_TARGET_DIR=<repo>/target cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.34
# local: parity through the real text API against the spike-025 fixture
LAYA_DIR=$S PORT=8792 <repo>/target/release/server & python3 tools/probe.py http http://127.0.0.1:8792/ --records all --warm 2
# cloud (functions laya-spike-10g / laya-spike-4g, zip + LAYA_S3)
uv run --with boto3 python tools/lambda_cold.py laya-spike-10g --rounds 3 --out results/lambda-10g.jsonl
tools/teardown.sh
```

## Investigation Trail
1. **Local, through the real text API**: all 12 fixture requests (14 questions, incl. a 3-question ticket, an
   injection probe, unicode, a 512-token state) match Laya's torch fp32 probabilities to **≤ 3.8e-6, argmax 14/14**
   — so the Rust `render_options` + builder + forward is the Python request path, not just its forward.
2. **10 GB, 3 forced cold starts**: 11.3–12.0 s to the first MCP answer. The S3 phase is 846 MB in 8.9–9.0 s
   (**94–95 MB/s**, same 16 x 64 MB parts as spike 021, which measured 72–80 MB/s — the ~15 % difference is
   unexplained: different day, environments and object; still a per-environment ceiling, not a client knob), model built at ~10 s, first decision 0.5–0.75 s. Two rounds on Graviton2, one on Graviton3.
3. **4 GB, 3 forced cold starts**: cold 13.2–13.4 s — the download is not memory-scaled — and every warm call
   2.4x slower (1.8 s per tweet) on ~2.3 vCPU; peak 3.5–3.65 GB of 4.1 GB.
4. **Cost per decision is the same at both sizes** (Lambda bills GB-seconds): the smaller function buys nothing.

## Results

**Default Lambda, arm64, Laya-en F16 weights from S3, us-east-1** (parity vs Python fp32 on every call ≤ 4.5e-6)

| | **Laya 10,240 MB (6 vCPU)** | Laya 4,096 MB (~2.3 vCPU) | Kev-0.8B 10,240 MB (spike 021) |
|---|---|---|---|
| **cold start -> first MCP answer** | **11.3 / 12.0 / 12.0 s** | 13.2 / 13.4 / 13.4 s | 42.8–43.2 s |
| of which S3 download | 8.9–9.0 s (846 MB, 94 MB/s) | 8.9–9.1 s | 37.7 s (3.02 GB, 80 MB/s) |
| stance tweet (94 tokens), warm p50 | **0.75 s** Graviton2 · **0.49 s** Graviton3 | 1.80 s (G2) | 1.13–1.16 s (G2) |
| 3-question ticket (~150 tokens each) | 3.8 s G2 · 2.4 s G3 | 8.6 s | – (Kev pays the state 3x too) |
| 512-token state | 5.6 s G2 · 3.2 s G3 | 12.4 s | – |
| max memory used | 3.4–3.6 GB | 3.5–3.65 GB | 7.8–8.1 GB |
| GB-s per tweet decision (≈ $ at arm list price $0.0000133334/GB-s) | 7.5 (≈ $0.00010) | 7.2 (≈ $0.00010) | 11.5 (≈ $0.00015) |

Graviton2 vs the M4 laptop at 6 threads: 3.3x (0.75 s vs 0.226 s), the same ratio spike 021 measured for Kev.

**Verdict: VALIDATED ✓.** Laya makes the Qwen-class decision model **fit default Lambda comfortably**: a cold start
**3.6x shorter than Kev's (12 s vs 43 s)** because the wire weights are 3.6x smaller, warm decisions **1.5x faster
(0.75 s vs 1.15 s on Graviton2)**, half the memory, and a third less cost per decision — while still scaling to zero.
It does not make it *interactive-fast*: a multi-question request or a 512-token document is multi-second on
Graviton2, and a cold call is ~12 s.

**Signal for the build**
- **Default Lambda at 10,240 MB is the right host for Laya-class models** (the GB-s price is flat, so buy the
  vCPUs). The 12 s cold start still wants an async MCP Task front door or a warm floor for interactive agents.
- **Hardware lottery**: Graviton3 answered 1.5x faster than Graviton2 inside one function (0.49 vs 0.75 s). LMI
  (spike 023: c9g, 0.34 s for Kev) or Fargate on Graviton4 are the latency tiers if Lambda's is not enough.
- **Batch the questions of one request** into one GEMM pass (M = Σ row lengths): today a 3-question ticket is three
  sequential forwards (3.8 s); the rows are independent and the encoder is GEMM-bound.
- A fine-tuned Laya checkpoint (spike 024's back-office output) deploys by replacing the four S3 files; the server
  code does not change.
