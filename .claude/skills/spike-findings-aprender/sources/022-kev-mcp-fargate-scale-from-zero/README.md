---
spike: 022
idea: mcp-model-hosting-aws
name: kev-mcp-fargate-scale-from-zero
type: comparison
validates: "Given the spike-021 Kev MCP server as an arm64 Fargate task (8 vCPU / 16 GB), when a task starts from zero, then the RunTask -> image pulled -> container started -> weights loaded -> first MCP answer timeline is measured for weights streamed from S3 vs baked into the image"
verdict: VALIDATED
related: [021, 023, 019]
tags: [fargate, ecs, cold-start, scale-to-zero, mcp, pmcp, graviton, deployment, kev]
---

# Spike 022: Kev MCP server on Fargate, scale from zero

## What This Validates
The user's alternative to always-on capacity: ECS on Fargate scaled to zero, paying a cold start per wake-up.
I had estimated 20–60 s for it; this measures it. Same crate and binary as spike 021 (`server` bin: loads the
model, *then* binds :8080, so "port open" means "ready to decide"). Code, drivers and raw results live in
`../021-kev-mcp-default-lambda/` (`src/bin/server.rs`, `deploy/*.Dockerfile`, `deploy/taskdef-*.json`,
`tools/fargate_cold.py`, `results/fargate-*.jsonl`).

| Variant | Image | Weights |
|---|---|---|
| **022-a fargate-s3** | 50 MB compressed (al2023-minimal + 25 MB `server`) | 16 x 64 MB ranged GETs from S3 to the task's 20 GB ephemeral disk |
| **022-b fargate-baked** | 2.36 GB compressed | `COPY` into the image |

Task: arm64, 8 vCPU / 16 GB, `RAYON_NUM_THREADS=8`, public IP in default subnets 1a/1b/1c, SG open to one IP.

## Research
- Fargate task start = scheduling + ENI attach (10–30 s reported) + image pull (linear in size without SOCI) +
  container start. ECS exposes `pullStartedAt`, `pullStoppedAt`, `startedAt` per task, so each phase is measured on
  ECS's clock rather than inferred.
- SOCI lazy loading was **not** tried: 022-b shows the baked image is the wrong shape regardless (see Results),
  and SOCI would only defer the reads to the first forward pass, which touches every weight.

## How to Run
```bash
cd ../021-kev-mcp-default-lambda
uv run --with boto3 python tools/fargate_cold.py kev-spike-fargate-s3 --rounds 3 --out results/fargate-s3.jsonl
uv run --with boto3 python tools/fargate_cold.py kev-spike-fargate-baked --rounds 3 --out results/fargate-baked.jsonl
```
Each round: `RunTask` → poll `DescribeTasks` + ENI public IP → TCP connect to :8080 → MCP `tools/call decide`
(cold) → a second call with 5 warm rounds → `StopTask`.

## Investigation Trail
1. **022-b first** (the obvious design): 95–105 s. ECS's own timestamps split it: ~17–21 s before the pull starts,
   **53–65 s pulling 2.36 GB**, ~17–22 s from pull-stopped to container-started (layer unpack), then 1.1–1.3 s to
   build the model from local disk.
2. **022-a**: 22–30 s. The pull of a 50 MB image is 2 s; the **S3 download inside the task runs at 674–789 MB/s**
   (9x default Lambda's cap) — 3 GB in 3.8–4.5 s; model built 1.1–1.4 s later.
3. **Provisioning dominates what is left**: 13–21 s from `RunTask` to pull-start in every run, with no image-size
   dependence. That is the Fargate floor for this configuration; nothing in the binary or the weights moves it.
4. **Hardware is a lottery here too**: the same task definition landed on **Graviton3** (MIDR 0xd40, 1a) and
   **Graviton4** (0xd4f, 1b). Warm decision: 0.63–0.65 s on G3, **0.51–0.52 s on G4** (8 threads).
5. The task's `/proc/meminfo` reports the host (19.6 or 31.5 GB), not the task's 16 GB — size by the task
   definition, not by what the process sees.

## Results

**From `RunTask` (scale from zero) to the first MCP answer, Kev-0.8B f32, row 1 (81 tokens), us-east-1**

| variant | runs (first answer) | provisioning → pull start | image pull + unpack | weights | model build | warm decision p50 |
|---|---|---|---|---|---|---|
| **S3 weights** | **26.9 · 22.2 · 29.6 s** | 13.4–20.9 s | 2.0–2.6 s | S3 3.8–4.5 s (674–789 MB/s) | 1.1–1.4 s | 0.51–0.65 s |
| baked image | 94.6 · 103.3 · 105.1 s | 17.0–20.9 s | 74.8–81.3 s | from image | 1.1–1.3 s | 0.52–0.64 s |

Parity vs Python fp32 on every call: |Δp| 4.8e-7. Peak RSS 7.75 GB (the as-built loader's transient; steady 4.9 GB).

**Verdict: VALIDATED ✓.** Fargate scale-from-zero with weights in S3 answers the first MCP call **22–30 s after
`RunTask`**, then decides in 0.5–0.65 s — faster to wake than default Lambda (43 s) because its S3 bandwidth is
~9x higher, and 2x faster per decision (8 vCPU of Graviton3/4 vs 6 of Graviton2). My "20–60 s" estimate holds
at its low end, and the user's "a couple of seconds" does not: **~15–20 s of every wake-up is Fargate
provisioning**, independent of the model. Baking the weights into the image is 4x worse.

**Signal for the build**
- **Weights in S3, never in the image.** A 50 MB image + parallel ranged GETs beats a 2.4 GB image by ~75 s.
- The wake-up is ~25 s, so an interactive agent must not wait on it synchronously: a front door that returns an
  **MCP Task** while the task starts (spike 024) is the shape, or the first call must be allowed to take ~30 s.
- Record the Graviton generation per response; G3 vs G4 is 20 % latency inside one task definition.
- For Kev-4B (8.4 GB bf16) the S3 phase scales to ~11–13 s at the same bandwidth, so ~35 s per wake-up.
