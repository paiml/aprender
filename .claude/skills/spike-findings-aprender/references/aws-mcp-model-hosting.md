# Hosting Rust Model MCP Servers on AWS (measured)

Which AWS host to use for a thin pmcp model server, decided on **measured** cold start, latency and idle cost.
Default Lambda, ECS on Fargate scaled to zero, and Lambda Managed Instances (LMI) were each deployed with the
same binary in us-east-1, arm64. This supersedes the doc-derived estimates in `llm-classifier-lambda-deployment.md`
wherever they differ.

## Requirements

From idea `mcp-model-hosting-aws`:
- **Every model is wrapped as an MCP server on the pmcp SDK** (`~/Development/mcp/sdk/rust-mcp-sdk`), so it
  integrates with AI agents; the thin one-model-per-server rule holds.
- **Idle cost matters.** The cost to beat is 24/7 capacity × 3 AZs plus the 15 % LMI fee. Scale-to-zero is
  preferred when its cold start is acceptable, and the cold start is measured, not assumed.
- **The candidate hosts are default Lambda, ECS on Fargate (scale to zero) and LMI.** The decision is per model and
  may differ between small and large models.
- Measure in us-east-1 (where pmcp.run deploys) on arm64, tag every resource, and tear down or leave idle-free.

From idea `llm-decision-classifier` (spike 026):
- **Inference is Rust on aprender; Lambda is the deployment target**, measured rather than assumed.

## How to Build It

### 1. One crate, every host
One library holds the pmcp tool and the model, with two bins (`sources/021-kev-mcp-default-lambda/src/`,
`sources/026-laya-mcp-default-lambda/src/`):
- **`bootstrap`** is the Lambda/LMI entry: the `aprender-mcp-chronos-lambda` loopback pattern. A stateless pmcp
  `StreamableHttpServer` runs on 127.0.0.1, and each invocation is proxied to it. The model loads on the first
  call; `EAGER_LOAD=1` moves loading into init.
- **`server`** is the container entry: it loads the model and only then binds `0.0.0.0:8080`, so an open port means
  the server can decide.
- Use `StreamableHttpServerConfig::stateless()` everywhere. Any instance can then answer any call, and
  scale-to-zero never breaks a session.

### 2. Weights live in S3, pulled at start, never baked into an image
- **Parallel ranged GETs** (16 × 64 MB) write straight into a pre-sized file with `write_all_at`. **Retry each part
  up to 5 times**: a 32-part run failed mid-body with `streaming error` and, uncached, forced the whole reload.
- Keep weights in their smallest faithful dtype on the wire (Laya: F16, 843 MB) and widen at load. Cold start is
  bytes divided by bandwidth.

### 3. Choose the host by weight size and traffic

| host (arm64) | cold start → first MCP answer | what the cold start is made of | warm decision | idle cost |
|---|---|---|---|---|
| **default Lambda 10,240 MB**, Kev-0.8B 3.0 GB | **42.8–43.2 s** | S3 **~80 MB/s** per environment (37.7 s) + load 3 s | 1.13–1.16 s (Graviton2) | 0 |
| **default Lambda 10,240 MB**, **Laya 0.84 GB** | **11.3–12.0 s** | S3 94 MB/s (9 s) + load 1.4 s | **0.75 s** G2 · 0.49 s G3 | 0 |
| Fargate 8 vCPU / 16 GB, Kev S3 weights | **22–30 s** from `RunTask` | provisioning 13–21 s + S3 **674–789 MB/s** (4 s) + load 1.3 s | 0.51 s G4 · 0.64 s G3 | 0 (stopped) |
| Fargate, Kev weights baked into image | 95–105 s | image pull + unpack 75–81 s | same | 0 |
| **LMI**, 16 GB / 8 vCPU function, Kev | **none on the request path** (lazy first call ~5 s) | S3 875–962 MB/s | **0.34 s** (c9g, Neoverse-V3) | ≥ 1 × `c9g.8xlarge` 24/7 + 15 % |
| default Lambda, Kev weights baked into image | **800 s** first · 309 s next | lazy image-store read ~4 MB/s | – | 0 |

Decision rule:
- **Small models (≤ ~1 GB: SetFit, Chronos-Bolt, Laya-class): default Lambda at 10,240 MB**, via pmcp.run.
  Scale-to-zero, ~$0.0001 per Laya decision.
- **Multi-GB models: Fargate with weights in S3**, behind a front door that answers with an **MCP Task** during the
  ~25 s wake-up. This reuses the pmcp-tasks + DynamoDB machinery proven for SetFit training.
- **LMI only when several models with steady traffic share one capacity provider's hosts.** For a single model it
  is the most expensive option measured.

### 4. LMI specifics (spike 023)
- Create the capacity provider **with its final `MaxVCpuCount`**. The first publish failed at 64 (my cap), and later
  raising it to 128 and then 400 **never unstuck that provider**. A fresh provider created at 400 worked:
  3 × `c9g.8xlarge` (one per AZ) in 33 s, and the version went Active in 87.7 s.
- **Instance granularity is the host, not the environment.** An 8-vCPU / 16 GB function got 32-vCPU / 64 GB hosts,
  so the default of 3 environments means 96 vCPU for one model.
- **One instance is possible**: set `min = max = 1` on a *successfully published* version; 3 hosts dropped to 1 at
  t+314 s. `$LATEST.PUBLISHED` does not exist before the first successful publish, and a failed version rejects
  `PutFunctionScalingConfig` (`ResourceConflictException`).
- **`$LATEST` is `ActiveNonInvocable`**; only published versions run. New versions reuse running hosts (48 s).
- Set `EphemeralStorage` (≤ 10,240 MB) for S3-staged weights: `/tmp` is 512 MB and `/dev/shm` is small.
- `LogType=Tail` is rejected, so take timings from the server's own response.
- Operator role: trust `lambda.amazonaws.com`, attach `AWSLambdaManagedEC2ResourceOperator`. us-east-1c is refused.

### 5. Measure like this
- Force a Lambda cold start by bumping an env var, then `lambda.invoke` with an API-GW v2 event (no public URL) and
  `LogType=Tail` for Init Duration and Max Memory. Put forensics in every response: load timeline, RSS per step,
  `/proc/cpuinfo` CPU part (the Graviton generation), `available_parallelism`, `first_call_in_process`.
- Fargate: `RunTask` → `DescribeTasks` timestamps (`pullStartedAt`, `pullStoppedAt`, `startedAt`) → TCP connect →
  first MCP call. Drivers: `sources/021-kev-mcp-default-lambda/tools/{lambda_cold,fargate_cold,lmi,probe}.py`.
- Build images **inside AWS** (CodeBuild ARM_CONTAINER, privileged, source zip in S3); the laptop uplink is ~3.7 MB/s.

## What to Avoid
- **Baking weights into a container image**: 7–18× worse on Lambda, 4× worse on Fargate.
- **Trusting "a couple of seconds" for Fargate**: 13–21 s of every wake-up is provisioning, independent of the model.
- **Sizing Lambda below 10,240 MB for a CPU-bound model.** Lambda bills GB-seconds and scales vCPU with memory:
  Laya at 4 GB cost the same per decision and was 2.4× slower.
- **Auditing LMI cost with a default `DescribeInstances`**: LMI hosts carry `Operator.HiddenByDefault = true` and are
  listed only with `IncludeManagedResources=True` (or by id). Two hosts billed ~5 min unseen.
- **Assuming one Graviton**: default Lambda gave Graviton2 and Graviton3, Fargate gave Graviton3 and Graviton4, in the
  same function or task definition. Record it per response.
- Importing a driver script that runs at import time, reading `$?` through a pipe, and zsh's no-word-split `set --`.

## Constraints
- Default Lambda: 10,240 MB → 6 vCPU, 900 s timeout, S3 at ~80–95 MB/s per environment (account quota "Network
  bandwidth per execution environment: 625"), 50 MB direct-upload zip, `provided.al2023` needs glibc ≤ 2.34
  (`cargo zigbuild --target aarch64-unknown-linux-gnu.2.34`).
- Fargate arm64 list price: $0.03238 per vCPU-hour + $0.00356 per GB-hour; 16 vCPU / 120 GB max; 20 GB ephemeral.
- Lambda arm list price $0.0000133334 per GB-second. Laya at 10 GB is 7.5 GB-s per decision (≈ $0.00010); Kev-0.8B is
  11.5 GB-s (≈ $0.00015).
- aws-cli 2.29 does not know the LMI API; use boto3 ≥ 1.43. `aws-sdk-s3` newer than 1.137 needs rustc 1.94.1:
  `CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo update`.
- **Not measured**: the MCP Task front door, SOCI, pmcp.run's own host, LMI with several models packed, x86.

## Origin
Synthesized from spikes 021 (PARTIAL), 022, 023 and 026 (VALIDATED).
Source files: `sources/021-kev-mcp-default-lambda/` (crate, deploy/ Dockerfiles, buildspec, task definitions,
tools/, results/), `sources/022-*/`, `sources/023-*/`, `sources/026-laya-mcp-default-lambda/`.
Teardown scripts: `tools/teardown.sh` in 021 and 026.
