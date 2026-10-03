# aprender 0.70.1 — release notes

## How this release was gated

0.70.1 ships under a recorded operator emergency scope, `crux-smoke`, as 0.69.1 did.
The operator's words: "0.70.1 ships on CRUX smoke on lambda and gx10 GPU. Everything bigger is nightly."
The record is the `crux-smoke` entry for release "0.70.1" under `emergency_scopes` in
`contracts/model-capability-ladder-v1.yaml`.

**The gate.** The CRUX smoke ran on the lambda GPU (RTX 4090) and the gx10 GPU (GB10), with nothing else on either GPU. It
covered every certified model (Qwen3.5-2B Q4_K_M, Qwen3.5-4B Q4_K_M and Qwen3.5-4B UD-Q4_K_XL) and every admitted
thinking mode, on the control prompts. apr was checked against llama.cpp (pinned d1d3c3396), Hugging Face transformers
and vLLM, on the verbs run, chat, serve and code.

| host | apr | cells | result |
|---|---|---|---|
| lambda | `apr 0.70.1 (cc4463f7a6)` | 134 | 134 GREEN, 0 RED |
| gx10 | `apr 0.70.1 (cc4463f7a)` | 134 | 134 GREEN, 0 RED |

These rows are the smoke of the binary built at cc4463f7a6, and the judge passed them under the scope (rc 0).
The publish path judges the release commit itself, so both GPUs were smoked again with the binary built at the
release commit, whose source is identical to cc4463f7a6. A commit cannot hold results produced from itself, so those
receipts are published with the GitHub release for v0.70.1.

On lambda, the two 4B models ran with `APR_QWEN35_PREFILL_GEMM=f32`. This skips apr's fp16 prefill prewarm (6.8 GiB,
#4313), so the Hugging Face and vLLM reference engines fit beside apr on the 24 GB card. On the default path, every
4B serve cell had no reference answer because hf and vLLM could not start beside apr. apr agreed with llama.cpp on
all 80 of those cells. The 2B ran on the default path.

**Printed as evidence, not the gate for this release.** The full model matrix (R7) and the release-readiness grade
(R8) still run at publish, and their verdicts are printed. They stay red where they were red; see Known failures.

**Unchanged.** Clean tree, tag on HEAD, release branch, dogfood GO for this commit, probes 14/14, plants 22/22,
Lean T5, and the clean-room green on the tagged commit.

**Next (P0 for 0.70.2).** No check may be enforced at publish until its inputs have been produced green on main,
nightly, three times. The full CRUX lanes and the readiness check move to nightly, and CRUX gets the GPU to itself.

## Fixed
- `apr code` now gets an input budget on small-context models such as tinyllama. It failed with "context overflow:
  required 12 tokens, available 0" (#4599 regression, fixed in cc4463f7a6).

## Build note
The release commit differs from cc4463f7a6 only in `scripts/check_publish_preflight.sh`, one contract entry and
`evidence/`. crates/, src/, Cargo.toml and Cargo.lock are identical. `crates/aprender-contracts/build.rs` reads
`contracts/*.yaml`, so the contract entry recompiles aprender-contracts and the crates that depend on it. build.rs
does not read `emergency_scopes`, so the code is unchanged. `evidence/release/surface/0.70.1.json` records the
binary built at cc4463f7a6. A file inside the release commit cannot name that commit's own hash.

## Known failures (shipped, tracked as P0 for 0.70.2)
Each fails the same way on v0.69.5-rc.1, so none is new in 0.70.1. Both results are saved under
`evidence/release/0.70.1/known-red/`.

| model | host(s) | check | failure in 0.70.1 | same on v0.69.5-rc.1 | ticket |
|---|---|---|---|---|---|
| qwen35-0.8b-q4km | lambda, gx10 | golden_output thinking_on | think block unclosed within 2048 tokens | yes | https://github.com/paiml/aprender/issues/4666 (continues #4030 (closed)) |
| Qwen2.5-0.5B-Instruct-f16.gguf | lambda | cuda serve /api/chat | gibberish output | yes | https://github.com/paiml/aprender/issues/4661 |
| Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf | lambda, gx10 | cuda serve /api/chat | gibberish output | yes | https://github.com/paiml/aprender/issues/4662 |
| Qwen3.5-0.8B-IQ4_XS.gguf | lambda | golden_output thinking_on | no measured thinking budget (named refusal; the GPU ran). The old binary failed earlier, with a CPU fallback. | fails on both, differently | https://github.com/paiml/aprender/issues/4663 |
| Qwen3.5-0.8B-UD-IQ2_XXS.gguf | lambda, gx10 | golden_output | GPU forward fell back to CPU (chat rc=14); wrong answer | yes, it fell back on v0.69.5-rc.1 too (the old binary has no rc=14) | https://github.com/paiml/aprender/issues/4664 |
| Qwen3.5-35B-A3B-UD-IQ4_XS.gguf | lambda | capability_match | no CUDA forward for qwen35moe (named refusal) | yes | https://github.com/paiml/aprender/issues/4665 |

None of these is marked known red or green for this release. Each stays red in the printed evidence and in its ticket.

## Refused (does not fit)
The memory-fit check refuses these rows on lambda, the same as on v0.69.5-rc.1. They are not counted as failures.
- qwen2.5-coder-0.5b-instruct.apr
- qwen2.5-coder-1.5b-instruct-q4k-v2.apr
- qwen2.5-coder-1.5b-instruct-q4k.apr
- qwen2.5-coder-1.5b-instruct-st.apr
- qwen2.5-coder-1.5b-q4k.apr

On gx10, the memory-fit check refuses these rows (llama-fit-params exit 1). They are not counted as failures.
v0.69.5-rc.1 was not measured on gx10 for these rows, so there is no old result to compare.
- qwen2.5-coder-1.5b-instruct-fp16.apr
- qwen2.5-coder-1.5b-instruct-q4_k_m.apr
- qwen2.5-coder-1.5b-instruct-q4k.apr

The fit check stops apr before it runs on these `.apr` rows, so they have no CRUX result (#4668).

## Not measured for 0.70.1
- Capability cells (`model_ladder.sh --cells`) were not measured on any host. At the measured ~45 min per model for
  26 models, they take 15-20 GPU-hours per host. This is a gap, not a pass: https://github.com/paiml/aprender/issues/4667
- The full CRUX sweep (beyond the smoke above) and CPU lanes: nightly, per the scope.
