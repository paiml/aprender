# LLM Classifier Deployment on AWS Lambda (default Lambda vs Lambda Managed Instances)

> **Superseded in part by measurement (spikes 021–023, 026, 2026-09-24/25) — read `aws-mcp-model-hosting.md` first.**
> Corrections to this page's doc-derived projections:
> - LMI does **not** force 3 hosts: after a successful first publish, `min = max = 1` runs **one** host. But an
>   8-vCPU function landed on a **`c9g.8xlarge` (32 vCPU / 64 GB)**, so even one host is large.
> - Default Lambda S3 bandwidth is capped at **~80–95 MB/s per environment**: Kev-0.8B's measured cold start is
>   **43 s**. Weights baked into a Lambda image are far worse (800 s / 309 s).
> - Measured Graviton2 Kev-0.8B decision: **1.13–1.16 s**, inside the 1–1.5 s estimate below. Lambda also served
>   Graviton3.
> - **Laya** (spike 026) fits default Lambda comfortably: 12 s cold, 0.75 s warm, 3.5 GB peak.

Where a Qwen-based decision model (Kev and the Qwen models expected after it) runs on AWS, next to
the small Rust MCP servers. Spike 019 measured the model against **default** Lambda with a local
proxy. Lambda Managed Instances (LMI) was added at wrap-up (2026-09-23) because it removes the limit
that made Kev-4B a PARTIAL. The LMI facts below come from AWS docs. **No LMI deployment has been
measured yet.**

## Requirements

From idea `llm-decision-classifier`:

- **Inference must be Rust on aprender**; the served path (speed, security, AWS Lambda) is Rust.
- **Lambda is the deployment target**: size, memory, cold start and latency are measured against
  Lambda limits, not assumed.
- **Lambda was measured by a local proxy** (6 threads, 10 GB cap) for spike 019, not a real AWS
  deploy (decided 2026-09-23). Every number marked *est.* below still needs a real deploy.
- **LMI is the intended host for Qwen-sized models**, running alongside the existing small Rust MCP
  servers (recorded 2026-09-23 at wrap-up).

## How to Build It

### 1. What spike 019 measured (Kev-0.8B, f32, local proxy: fresh process, `RAYON_NUM_THREADS=6`)

| Artifact | Bytes |
|---|---|
| Binary (`aprender-serve`, default features off) | **1.4 MB**. The deployment size is the weights |
| Kev-0.8B GGUF f32 (merged LoRA) | 3.02 GB (bf16/f16: 1.52 GB, but the prefill GEMM is F32-only) |
| Pointer head + temperature | 2 MB |

Cold start, fresh process, OS file cache warm:

| Step | t since start | RSS |
|---|---|---|
| mmap GGUF | 21 ms | 52 MB |
| base (embeddings, norms, `lm_head`) | 184 ms | 2,963 MB |
| owned layers | 402 ms | **6,765 MB** |
| drop mmap | 420 ms | 3,885 MB |
| **first decision complete** (87 tokens) | **819 ms** | 3,933 MB |

Steady state (p50, 6 threads): 1-question tweet 185–207 ms; stance tweet (81–87 tokens) 352–375 ms;
3-question ticket 711 ms; 915-token document 3.73 s. p95 is within 1–2 % of p50, with no drift over
50 requests. Worst |Δp| vs Python fp32: 1.3e-6.

### 2. Fix the loader before sizing anything (required for 4B)

The load is wasteful in two ways:

1. **Skip the unused `lm_head`.** Qwen3.5 ties it to the embedding and the loader materialises it
   anyway, but Kev never calls it. Saves 1.0 GB at 0.8B f32 and ~2.5 GB at 4B f32.
2. **Don't hold the mmap and owned copies at once.** Owned layers copy 2 GB out of the mmap while
   the touched pages stay resident. Dropping the mmap after construction returned 2.9 GB with
   identical outputs, but the **transient peak (7.1 GB, ~2.35× the file) is set during loading**.
   The better fix: keep the mmap and **borrow F32 weights zero-copy** instead of owning them.
   Resident memory then ≈ file size.

### 3. Choose the host by model size

| | Default Lambda | **Lambda Managed Instances** |
|---|---|---|
| Max memory / vCPU | 10,240 MB / ~6 vCPU | **32 GB / 16 vCPU** (2:1), 8 vCPU (4:1), 4 vCPU (8:1); min 2 GB / 1 vCPU |
| Hardware | Graviton2 (arm64) or x86 | EC2 catalog incl. **Graviton4 (C8g)**; `Architectures` default `x86_64` |
| Cold start | per scale-out, in the request path | **none**: scales asynchronously on CPU utilisation and concurrency saturation; 3 environments start before the version goes ACTIVE |
| Concurrency | 1 request per environment | multi-concurrent; **Rust: `run_concurrent` + `concurrency-tokio` feature, handler `Clone + Send`**, up to 64 per vCPU |
| Scale to zero | yes | **no**: default min 3 environments (AZ resilience). `min=max=0` deactivates, and reactivation is explicit |
| Price | per request + GB-s | EC2 instance price (On-Demand / RI / Savings Plans) **+ 15 % management fee** (not discounted) |
| Timeout | 15 min | 15 min sync and init; 90 min async / event-source |
| Isolation | Firecracker per environment | **capacity provider is the security boundary**; containers on an instance share it |

Model fit, **projected** (not measured) from spike 019's ratios:

| Model / weights | Resident (after §2 fixes) | Default Lambda | LMI |
|---|---|---|---|
| Kev-0.8B f32 | ~3 GB | fits (3.9 GB steady, 7.1 GB transient as built) | fits in a 8 GB 2:1 function (4 vCPU) |
| Kev-4B bf16 | ~8.4 GB | tight; needs BF16 GEMM + §2 fixes | fits in 16 GB (8 vCPU @ 2:1); needs BF16 GEMM |
| Kev-4B f32 | ~16.8 GB | **does not fit** | fits in 32 GB / 16 vCPU **only with the §2 fixes**. As built, the ~2.35× transient is ~39 GB and still does not fit |
| Kev-9B | ~18 GB bf16 | no | bf16 in 32 GB; f32 no |

**Recommendation.** Run the Qwen models on LMI with an **arm64** capacity provider. Keep the small
MCP servers (SetFit 91 MB at ~32 ms warm; Chronos-Bolt-tiny 24 MB with a 52 ms cold start; the
forecast servers) on **default Lambda**, where they scale to zero. Moving a small server onto LMI
makes it always-on (≥ 3 × 2 GB environments) to save a ~50 ms cold start it barely has. Put a small
server on the models' capacity provider only when its traffic is steady and the servers are mutually
trusted. Then it uses the provider's idle headroom. Chronos-2 (228 MB f16, 1.45 GB peak,
0.59 s/forward single-threaded, GEMM-bound) is the one forecaster that benefits from 16 vCPU.

### 4. LMI build pattern for a Rust model server

This is a sketch assembled from the AWS Rust guidance and spike 019's driver. It has not been
compiled or deployed.

```rust
// Load once in init (init is limited to 15 min; the weight read happens here, not in the request path).
// Qwen35Model<'a> borrows from the mapped file, so give the handler 'static data:
// build the owned model at init and Box::leak it (or Arc an owned variant), then clone the handle.
lambda_runtime::run_concurrent(service_fn(move |ev| {
    let m = model;                        // &'static, Copy
    async move { tokio::task::spawn_blocking(move || decide(m, ev)).await? }  // never block a tokio worker
})).await
```

- **Set `RAYON_NUM_THREADS` from the function's vCPU setting.** Don't let rayon read the host's
  core count on a shared instance. (Whether `available_parallelism` sees the function's vCPU limit on
  LMI is unverified.)
- **One decision saturates the pool.** Concurrent requests in one environment share the rayon pool,
  so they add latency and no throughput. Keep per-environment max concurrency low (1–2 per decision
  model) and scale by environments. AWS warns that very low concurrency can throttle during
  scale-up, so measure the trade-off. Weights are shared across concurrent requests in the one Rust
  process: 16.8 GB is loaded once per environment, not per request. This is where Rust beats
  Python's process-per-request model on LMI.
- **Pin arm64.** The NEON 8×6 microkernel (65–77 GFLOP/s) is aarch64-only, and the x86 GEMM path is
  unmeasured on these shapes. On Graviton one vCPU is one physical core (no SMT).
- **Headroom**: by default LMI keeps enough headroom for traffic to double within 5 minutes. Faster
  bursts throttle, so raise `MinExecutionEnvironments` or schedule it with EventBridge Scheduler
  (`PutFunctionScalingConfig`).
- **Capacity provider**: VPC subnets in ≥ 2 AZs (it launches 3 instances by default); an operator
  role; `InstanceRequirements Architectures=arm64`; optional `AllowedInstanceTypes` (e.g. `c8g.*`).
  Leaving the type open gives better availability. Limits: 100 function versions per provider
  (cannot be raised); `MaxVCpuCount` defaults to 400.

## What to Avoid

- **Don't size Kev-4B from the file size.** The as-built loader peaks at ~2.35× the file. Fix the
  loader first.
- **Don't treat "fits in memory" as "servable".** Kev-4B is ~7× 0.8B's matmul work. Short decisions
  are multi-second on server CPUs (est. ~6–9 s on Graviton2 6 vCPU; perhaps 2–4 s on Graviton4
  8–16 vCPU, unmeasured). 16 vCPU does not rescue short rows (spike 020: 6 → 14 threads bought
  0.36 → 0.34 s at 87 tokens). Build the state-prefix cache and BF16 GEMM before promising latency.
- **Don't put an untrusted tenant's function on the models' capacity provider.** Containers are not
  a security boundary on LMI.
- **Don't move scale-to-zero servers onto LMI to "consolidate".** You pay for 3 always-on
  environments plus 15 %.
- **Don't report the proxy's M4 numbers as Lambda numbers.** An M4 P-core is roughly 4× a Graviton2
  core at peak FMA. Graviton4 (Neoverse V2) has twice N1's 128-bit FMA pipes, so roughly half an M4
  P-core per core (est.).

## Constraints

- Default Lambda: 10,240 MB buys ~6 vCPU; container image ≤ 10 GB; no GPU.
- LMI docs (2026-09-23): ≤ 32 GB / 16 vCPU per function (since 2026-03-27); ratios 2:1 / 4:1 / 8:1;
  Rust through OS-only runtime `provided.al2023`. The minimum function is 2 GB / 1 vCPU.
- **Unverified on LMI; settle each with a real deploy before planning around it:**
  - GPU instance types (sources conflict; the Rust path is CPU anyway);
  - container-image vs zip packaging and the size limit (16.8 GB f32 exceeds a 10 GB image, so
    load from S3);
  - `/tmp` size (zero-copy mmap needs the weights on a file);
  - Function URL / HTTP ingress for streamable-HTTP MCP;
  - whether **pmcp.run** can target LMI (the Chronos forecaster is live there today);
  - real Graviton latency, and cold weight read time from S3.
- Kev on Lambda buys zero-shot steering and calibrated multi-question answers at **10–40× SetFit's
  latency**. Route by shots (see `kev-few-shot-evaluation.md`); don't replace SetFit.

## Origin

Synthesized from spike 019 (PARTIAL), with spike 020's thread-scaling table. The LMI facts come from
AWS documentation read on 2026-09-23:
[LMI overview](https://aws.amazon.com/lambda/lambda-managed-instances/),
[32 GB / 16 vCPU announcement](https://aws.amazon.com/about-aws/whats-new/2026/03/lambda-32-gb-memory-16-vcpus/),
[scaling](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-scaling.html),
[best practices](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-best-practices.html),
[capacity providers](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-capacity-providers.html),
[runtimes](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-runtimes.html).
Source files: `sources/019-kev-lambda-inference/` (driver with the RSS timeline, RUN-OUTPUT with and
without `--keep-mmap`).
