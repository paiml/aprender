---
spike: 023
idea: mcp-model-hosting-aws
name: lmi-minimum-footprint
type: standard
validates: "Given an arm64 Lambda Managed Instances capacity provider and the spike-021 Kev MCP function (16 GB / 8 vCPU), when a version is published and then scaled to min = max = 1, then the EC2 instances actually run, time to Active, warm latency on the chosen Graviton, and whether one instance (not three) is possible are measured"
verdict: VALIDATED
related: [021, 022]
tags: [lambda-managed-instances, lmi, capacity-provider, cost, graviton, mcp, deployment, kev]
---

# Spike 023: Lambda Managed Instances — the real minimum footprint

## What This Validates
The user's cost worry: "LMI means 24/7 x 3 AZs of EC2 plus 15 %". The docs allow a function minimum of 1
environment but do not say what instances back it. This publishes the spike-021 `bootstrap` (zip, S3 weights) on a
capacity provider and counts the instances. Driver: `../021-kev-mcp-default-lambda/tools/lmi.py`; raw records:
`results/lmi.jsonl`, `results/lmi-calls.jsonl` there.

## Research
- API: boto3 1.43.48 carries `CreateCapacityProvider`, `PutFunctionScalingConfig` and
  `CreateFunction(CapacityProviderConfig=…)`; the installed aws-cli 2.29 does not know them.
- Operator role: trust `lambda.amazonaws.com`, managed policy `AWSLambdaManagedEC2ResourceOperator`.
- Function: 16,384 MB at `ExecutionEnvironmentMemoryGiBPerVCpu = 2.0` (8 vCPU, the Fargate size),
  `PerExecutionEnvironmentMaxConcurrency = 1` (one decision saturates the rayon pool).

## Investigation Trail
1. **`us-east-1c` is refused** for capacity-provider subnets ("aren't supported"): used 1a, 1b, 1d.
2. **The provider itself is free**: Active in 1.8 s with 0 instances.
3. **`$LATEST` is `ActiveNonInvocable`** on a capacity provider — only published versions run.
4. **First publish FAILED at `MaxVCpuCount = 64`** (my cap): the scaler launched **two `c9g.8xlarge`** (32 vCPU /
   64 GB each) in 24 s for the default **3 environments** of 8 vCPU, then could not place the third. The instances
   ran ~5 min and were terminated by the scaler — and **`DescribeInstances` did not list them**: LMI instances carry
   `Operator.HiddenByDefault = true` and appear only with `IncludeManagedResources=True` (or by id). A cost audit
   using the default listing misses them.
5. **Raising `MaxVCpuCount` on that provider (128, then 400 = AWS default) never unstuck it**: v2 and v3 failed
   in 38–49 s with the same message and launched nothing (EC2 on-demand quota is 256 vCPU — not the limit).
   `put_function_scaling_config` on a failed version returns `ResourceConflictException`, and
   `$LATEST.PUBLISHED` does not exist before a first successful publish, so min environments cannot be lowered
   *before* the first publish.
6. **A fresh provider created at 400 worked**: 3 x `c9g.8xlarge`, one per AZ, launched within 33 s; version
   Active in **87.7 s**; applied `MinExecutionEnvironments: 3`.
7. **First invoke: `No space left on device`** — `/tmp` is 512 MB by default on LMI too, and `/dev/shm` is also
   small. `EphemeralStorage = 10240` is accepted on an LMI function; the next version (Active in **48 s, on the
   same 3 instances** — new versions reuse running capacity) worked.
8. **`LogType=Tail` is rejected on LMI** ("Tail logs are not supported for functions configured with capacity
   provider"), so there is no REPORT line in the response: timings come from the server's own timeline.
9. **Scale to one**: `min = max = 1` on the working version → applied after ~33 s → two instances
   `shutting-down` at **t+283 s**, one left at **t+314 s**. The surviving environment kept its loaded model and
   answered warm (341 ms). **One instance is possible.**

## Results

| quantity | measured |
|---|---|
| instance type chosen for a 16 GB / 8 vCPU function | **`c9g.8xlarge`** (32 vCPU / 64 GB), CPU part 0xd84 (Neoverse-V3) |
| default footprint (min 3 environments) | **3 instances = 96 vCPU / 192 GB**, one per AZ — 4x the vCPU the environments use |
| publish → version Active | 87.7 s (instances launched), 48 s (reusing running instances) |
| scale 3 → 1 environment | applied ~33 s, instances down to 1 at **~5 min** |
| first call per environment (lazy load) | **~5 s**: S3 3.1–3.5 s at **875–962 MB/s**, build ~1 s, decide 0.35 s |
| warm decision, 81 tokens, 8 threads | **339–349 ms** p50 (2x Fargate G3, 3.3x default Lambda G2) |
| `available_parallelism` inside the environment | 8 (the host shows 32 cores; the limit is enforced) |
| parity vs Python fp32 | 4.8e-7 |

**Verdict: VALIDATED ✓.** LMI can run **one** instance per function (min = max = 1 after the first publish), and on
it Kev-0.8B is the fastest of the three hosts (0.34 s) with no request-path cold start. But the granularity is the
**instance, not the environment**: an 8-vCPU function got a 32-vCPU host, so even the one-instance floor is a
`c9g.8xlarge` running 24/7 (+15 % fee) — and the default is three of them. The user's cost concern is right in
direction and understated in size for large functions, unless several models share the provider's hosts.

**Signal for the build**
- LMI pays off only when **several models are packed onto one provider's hosts** (a 32-vCPU host fits four 8-vCPU
  environments). For one model it is the most expensive option measured.
- Create each capacity provider **with its final `MaxVCpuCount`**; a failed first publish can wedge it.
- Audit cost with `IncludeManagedResources=True`; the default EC2 listing hides LMI hosts.
- Set `EphemeralStorage` (≤ 10,240 MB) for S3-staged weights, and `EAGER_LOAD` in init so environments come up
  loaded (LMI init may run up to 15 min — no request ever pays the 5 s).
- Instance choice is Lambda's; pin `AllowedInstanceTypes` only if cost per host matters more than availability.
