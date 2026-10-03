# R1 — Backend parity matrix (0.73 "Runs Everywhere", epic #3999) — DRAFT

Status: draft by la-73, 2026-09-27. There is no ticket yet: the cop mints it. It becomes a spec-only PR
when the repo is under the 10-PR cap. Every number is [A] until a cell is measured.

## 1. What it answers
Is each 0.73 backend cell correct (E1), fast enough (E2) and MoE-capable (E3)? The answer is one
receipt per cell. E6 "admissible cell" means: E1 PASS + E2 PASS receipts on main, subject to
PRM-001 agreeing (RQ-3).

## 2. Cells
| Cell | Host | apr backend | Comparator (llama.cpp d1d3c3396, ruled RQ-2) |
|---|---|---|---|
| C0 | lambda | CUDA sm_89 + x86 CPU | control: the existing parity_host_receipt.sh path |
| C1 | intel | wgpu/Vulkan, AMD GPU 0 | ggml-webgpu (like-for-like, RQ-2) + ggml-vulkan |
| C2 | intel | wgpu/Vulkan, AMD GPU 1 | same as C1 |
| C3 | mini | wgpu/Metal | ggml-webgpu + ggml-metal (+ MLX advisory, never gating) |
| C4 | gx10 | aarch64 CPU | ggml-cpu |
| C5 | gx10 | CUDA sm_121 | ggml-cuda |
A cell's backend is named by what the process *printed about itself*: the wgpu adapter name
and the CUDA banner (verification rule 2). The flag it was launched with does not count. A
cell whose engine silently fell back to CPU is RED, not "slow".

## 3. E1 — correctness (composed-leg cosine)
- Leg A: apr-backend vs apr-CPU on the same host, same GGUF, final-position logits. This is `apr parity`,
  which exists (registry `contracts/apr-cli-commands-v1.yaml`). `apr parity --per-op` is the diagnostic
  for a failing cell (`apr-parity-per-op-v1`).
- Leg B: apr-CPU vs llama.cpp-CPU, once per (model, quant), from saved logits
  (`--kl-divergence-base`, common/arg.cpp:2502-2509 at d1d3c3396).
- Prompts: n ≥ 16, the 78-token prompt of `scripts/check_model_parity.sh` plus variants (rule 6).
  Report min / median / max. The gate uses the MIN.
- Gate: each leg's min ≥ 0.995. Proof: acos(0.995) = 5.73°, and two legs sum to 11.46°, which is under
  acos(0.98) = 11.48°, so the end-to-end result is ≥ 0.98. Legs of 0.99 each only give ≥ 0.960, so
  0.99 per leg is NOT enough. The direct apr-backend vs llama.cpp-same-backend cosine is reported,
  but it does not gate: it mixes two engines' backend errors.
- Models: Qwen3 dense first (the wgpu forward has no q_norm/k_norm, so this is expected RED until
  fixed — K1). Qwen3.5 (gated delta; absent on wgpu). Qwen3-Coder-30B-A3B for E3.

## 4. E2 — speed
- Decode and prefill tok/s, n ≥ 5 runs per engine, with the same client, prompts and clock as
  `scripts/parity_host_receipt.sh`. Paired bootstrap 95% CI on apr/llama.cpp.
- Gate: the CI lower bound is ≥ 0.5 [A]. The first three measured cells recalibrate the 0.5 by an
  amendment, never silently.
- Receipt shape: `crux-perf-receipt-v1` (#4464), plus the APR-OBS §2.1 identity block.

## 5. E3 — MoE
The E1+E2 method on Qwen3-Coder-30B-A3B Q4_0 and Q4_K_M. Order: C0, C5, C4, then C1–C3 once
the wgpu MoE forward exists. `forward_qwen3_moe_wgpu` is a stub today, and `apr run` MoE is
CPU-only, so a trace line must show which forward ran.

## 6. Receipt (one per cell × model × quant)
Fields: cell, host (forjar name), apr version+sha, llama.cpp sha (== d1d3c3396), adapter name
as reported, leg_a {min,median,max,n}, leg_b {…}, composed_bound, e2 {decode,prefill: ratio,
ci_lo, ci_hi, n}, forward_trace_line, op_placement {op: device|host} and qtype_path {qtype:
device-kernel|host-widen|refused-cpu}, both read from the trace (R5 census F-R5-2/F-R5-4), and the
APR-OBS identity (model_sha256, binary_sha256, …).
The receipt is refused unless it has every identity field (FALSIFY-OBS-ID-001, shared lint).

## 7. Falsifiers (for the `backend-parity-matrix-v1` contract, Next #2)
- F1: a perturbed-logits fixture (leg A min 0.990) must FAIL the cell even when the median is ≥ 0.995.
- F2: a receipt whose adapter name is a CPU/llvmpipe/"Microsoft Basic" adapter must FAIL as a
  fallback, not pass as slow.
- F3: legs of 0.99 + 0.99 must FAIL. This guards against someone "simplifying" the gate to 0.99.
- F4: a receipt with a llama.cpp sha ≠ the ruled pin must FAIL.
- F5: an E2 ratio whose point estimate is ≥ 0.5 but whose CI lower bound is < 0.5 must FAIL.

## 8. Reuse, not new tools
check_model_parity.sh (prompts), parity_host_receipt.sh (E2 method, the rule that a lane is not
labelled by intent), `apr parity` / `--per-op` (leg A + diagnosis), #4464 receipt shape, APR-OBS identity lint.
New work: the leg-B logits harness, the matrix runner (one cell per invocation; it runs only when
train-active is clear, with the GPU lock around the binary), and the contract.

## 9. Out of scope
Fixing any cell: those are R2–R5. R1 only measures and gates. A thresholds change needs an amendment.

## 10. Census fold-in (R5, 2026-09-27; drafts/R5-kernel-key-census.md)
- **F6 (F-R5-2, silent CPU fallback):** wgpu refuses Q4_0/Q8_0 at `wgpu_adapter.rs:339` [V], but those qtypes pass the GPU whitelist. So the run falls back to CPU on a real GPU adapter, and F2's adapter check cannot see it. A cell whose trace has no wgpu forward line for every layer is **refused (NotRun{reason: "backend forward not in trace"})**, not scored. Planted RED: a Q8_0 wgpu run on a real adapter must be refused.
- **F7 (F-R5-4, hybrids):** on wgpu today, attention, RoPE, the LM head and argmax run on the host. Every wgpu receipt carries `op_placement`, and E2 reports `device_ops / total_ops`. A cell with any host-placed op is labelled `hybrid: true`. Whether a hybrid counts toward E6 admissibility is **RQ-4 (cop ruling needed)**. L3 recommends: E1 may pass hybrid, but E2/E6 name it and never call it "GPU".
- **F8 (F-R5-3, known-bad control):** current wgpu decode measures cosine 0.955 on intel, gx10 and mini (`gguf_gpu_generate.rs:104-110`) [V]. The harness's first run on today's wgpu Qwen3 **must FAIL** E1. If it passes, the harness is broken; that is the control, not a finding.
- The E5 kernel key per cell comes from `qtype_path` × `op_placement`. It needs the KREG backend field (K4).

## 11. K19 amendment: one CPU reference for both legs (2026-09-27)
- **Problem.** The E1 bound, end-to-end ≥ cos(angle A + angle B), only holds when leg A and leg B share **one** apr CPU logits vector. apr has two CPU paths:
  - `fp32_act`: exact FP32 activations, inside `with_fp32_activations`;
  - `q8k_act`: the production Q8_K activations.
- **Evidence** (quoted in the #3714 comment, not re-measured): the CUDA MoE forward scored cosine 1.000000 against `fp32_act` and 0.985 against `q8k_act`.
- **Current tools disagree.** On main aca6f2d7f6, `apr parity-moe` uses `fp32_act` (parity_moe.rs:110) [V]. Dense `apr parity` has no such scope, so it uses `q8k_act` unless `DIRECT_FP32_GEMV=1` is set (parallel_k.rs:304). That second point is [V] only by grep absence.
- **Rule.** The receipt records `cpu_ref_path`, and both legs must name the same one. FALSIFY-BPM-008 refuses a mixed or missing pair.
- **Open question RQ-5 (needs a cop ruling): which path is pinned?**
  - Recommended: `fp32_act`. Leg A then measures only the backend's own error, and leg B absorbs apr's activation-quantization gap against llama.cpp. That gap is unmeasured; llama.cpp's CPU k-quant dot also quantizes activations, to Q8_K.
  - Alternative: `q8k_act`, i.e. what users actually run. But then every exact-FP32 GPU path pays an error of roughly 0.985 that belongs to the CPU, not the GPU.

## §11a — `apr parity*` CPU activation path (origin/main aca6f2d7f6, read-only, 2026-09-27)

Mechanism: only `fused_q4k_parallel_matvec_into` (aprender-serve quantize/parallel_k.rs:304) honours
`fp32_activations_scoped()` (set by `with_fp32_activations`, parallel_k.rs:236) or `DIRECT_FP32_GEMV=1`;
otherwise Q4K quantizes activations to Q8K (parallel_k.rs:326). Q5K/Q6K `_into` kernels never read the flag.
`fused_matmul_into` (gguf/inference/fused_matmul_into.rs:47/61) dispatches Q4K→flag-honouring fn, Q6K→Q8K-only.

| command | source | CPU fn | activation path |
|---|---|---|---|
| `apr parity` (dense) | apr-cli parity_03.rs:185 | forward_single_with_cache | q8k_act (no wrapper) |
| `apr parity --moe` | parity_moe.rs:110 | with_fp32_activations(forward) | fp32_act for Q4K; Q5K/Q6K still q8k |
| `apr parity` hybrid/qwen35 | parity_hybrid.rs:268 | forward_single_qwen35 → fused_matmul_into (forward_qwen35.rs:977) | q8k_act (no wrapper); MoE arm → parity_moe |
| `apr parity` per-op | parity_per_op.rs:85 | forward_single_with_cache | q8k_act (no wrapper) |
| `apr kernel parity` | kernel_parity.rs | attention tiled vs naive | N/A (not a forward pass) |
| attn-parity-lint | attn_parity_lint.rs | consumes kernel-parity JSON | N/A |
| quantize-flag parity | quantize_flag_parity.rs | argv parity | N/A |

RQ-5 implication: only `--moe` compares an fp32-activation CPU reference; the other three forward-path
commands compare GPU against a Q8K-activation CPU path unless DIRECT_FP32_GEMV=1 is exported — and even
the `--moe` wrapper leaves Q5K/Q6K tensors on Q8K activations.

## 12. L25 review: vacuous-pass holes (2026-10-03, origin/main 316dee2cd4, read-only)
The draft gated the right quantities, but seven inputs could make a cell PASS without the check measuring anything. Each now has a planted falsifier in `backend-parity-matrix-v1` (BPM-009..015):

| # | Hole | Why it passes vacuously | Fix | Falsifier |
|---|---|---|---|---|
| 1 | NaN in one prompt | `f32::min` returns the non-NaN operand, so a fold drops the NaN prompt. apr's cosine helpers (parity_per_op_table.rs:61, forward_error.rs:254) return 0.0 on a length mismatch or zero norm, but NaN on NaN logits [V] | A non-finite cosine REFUSES the receipt | BPM-009 |
| 2 | Legs over different prompt sets | The triangle bound holds per prompt | Both legs carry `prompt_set_sha256` and `n`, which must match | BPM-010 |
| 3 | Legs over different model files | Leg B then measures the file, not the engine | `model_sha256` is equal across all 3 runs | BPM-011 |
| 4 | CPU cell C4 is its own reference | Same kernels on both sides read 1.0. `fp32_act` does not separate Q5K/Q6K (§11a), so the R3 NEON Q6K dot would meet itself | The reference `kernel_path` must differ per quantized tensor | BPM-012 |
| 5 | Prefill-final logits only | Never touches the KV cache or the M=1 decode GEMV, which is where the known 0.955 sits | Gate on prefill-final plus ≥ 8 teacher-forced decode positions | BPM-013 |
| 6 | Adapter name denylist | An unlisted software adapter passes | `AdapterInfo.device_type = Cpu` ⇒ FALLBACK; names stay as a second check. Banner from `AdapterInfo.backend` (F3) | BPM-014 |
| 7 | E2 over unequal token counts | An early EOS decodes fewer, shorter-context tokens | One `host_id`, fixed `n_gen`, EOS ignored on both engines | BPM-015 |

Reuse note: origin/main has at least 8 cosine copies (`git grep 'fn cosine'`). The R1 checker uses `parity_per_op_table::cosine`, which already fails closed on length and zero norm, and adds no ninth copy.
pv 0.70.0: validate 0 errors / 0 warnings; `pv lint contracts-draft/` PASS. Obligations went from 6 to 10.
