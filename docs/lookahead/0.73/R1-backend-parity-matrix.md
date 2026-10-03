# R1 — Backend parity matrix (0.73 "Runs Everywhere", epic #3999) — DRAFT

Status: draft by la-73, 2026-09-27. There is no ticket yet: the cop mints it. It becomes a spec-only PR
when the repo is under the 10-PR cap. Every number is [A] until a cell is measured.

## 1. What it answers
Is each 0.73 backend cell correct (E1), fast enough (E2) and MoE-capable (E3)? The answer is one
receipt per cell. E6 "admissible cell" means: E1 PASS + E2 PASS receipts on main, subject to
PRM-001 agreeing. This is RQ-3's provisional S-4 default (C293.3, 2026-10-03), not a ruling; the request
is in the handoff's "Rulings needed".

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
**Superseded in detail by `P1-receipt-checker-spec.md` §3/§3a (2026-10-03).** That spec is authoritative for the field list:
- leg A and leg B are `parity-receipt-v2` records, cited by sha256;
- `forward_trace_line` becomes the `apr-kernel-path-v1` entries;
- identity is per run, from the unmerged OBS-00 contract.

`qtype_path` here and `device_qtype` in wgpu-forward-v1 (WGF-004) name one fact. Keep `qtype_path`, which has the three
states device-kernel, host-widen and refused-cpu. WGF-004 FAILs on any host-widen or refused-cpu for a required qtype.

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
- **F7 (F-R5-4, hybrids):** on wgpu today, attention, RoPE, the LM head and argmax run on the host. Every wgpu receipt carries `op_placement`, and E2 reports `device_ops / total_ops`. A cell with any host-placed op is labelled `hybrid: true`. Whether a hybrid counts toward E6 admissibility is **RQ-4** (ruling requested in the handoff). **Provisional S-4 default (C293.3), not a ruling:** E1 may pass hybrid, but E2/E6 name it and never call it "GPU".
- **F8 (F-R5-3, known-bad control):** current wgpu decode measures cosine 0.955 on intel, gx10 and mini (`gguf_gpu_generate.rs:104-110`) [V]. The harness's first run on today's wgpu Qwen3 **must FAIL** E1. If it passes, the harness is broken; that is the control, not a finding.
- The E5 kernel key per cell comes from `qtype_path` × `op_placement`. It needs the KREG backend field (K4).

## 11. K19 amendment: one CPU reference for both legs (2026-09-27)
- **Problem.** The E1 bound, end-to-end ≥ cos(angle A + angle B), only holds when leg A and leg B share **one** apr CPU logits vector. apr has two CPU paths:
  - `fp32_act`: exact FP32 activations, inside `with_fp32_activations`;
  - `q8k_act`: the production Q8_K activations.
- **Evidence** (quoted in the #3714 comment, not re-measured): the CUDA MoE forward scored cosine 1.000000 against `fp32_act` and 0.985 against `q8k_act`.
- **Current tools disagree** (re-read at main 316dee2cd4, 2026-10-03; see §11a). `apr parity-moe` runs the CPU inside `with_fp32_activations` (parity_moe.rs:110) [V]. Dense `apr parity` has no scope, so its reference is mixed: Q4_K on Q8_K activations (f32 on crushed blocks), Q5_K/Q6_K on f32, Q4_0/Q8_0 on Q8_0 [V]. Neither label is true of it.
- **Rule.** The receipt records `cpu_ref_path`, and both legs must name the same one. FALSIFY-BPM-008 refuses a mixed or missing pair.
- **RQ-5: which path is pinned?** Ruled (cop, 2026-09-27 20:12Z): **`fp32_act`**. `q8k_act` is reported as an info row only, and dense parity and parity-moe use the same reference. The drafts carried `fp32_act` as the provisional S-4 default until 2026-10-03, when the cop pointed to the ruling (cop-inbox/processed/inbox-20260927T2110Z.md:1); nothing flipped.
  - Why `fp32_act`: leg A then measures only the backend's own error, and leg B absorbs apr's activation-quantization gap against llama.cpp. That gap is unmeasured; llama.cpp's CPU k-quant dot also quantizes activations, to Q8_K. It is also the only label one forward can honour per tensor (§11a). Under `q8k_act`, Q5_K/Q6_K stay on f32 and crushed Q4_K blocks switch to f32.
  - Alternative: `q8k_act`, earlier called "what users actually run". Users run the mixed path in §11a, so a `q8k_act` pin would have to name that forward and list Q5_K/Q6_K as `ref_mixed`. Every exact-FP32 GPU path would also pay an error of roughly 0.985 that belongs to the CPU, not the GPU.
  - **How the default is applied** (P1/P2; no code change needed at main for Q4_K_M):
    - The reference is `forward_single_with_cache` token by token inside `with_fp32_activations` (dense), or `forward_single_qwen3_moe_with_cache` inside it (MoE, as parity_moe.rs:110 does), or `forward_single_qwen35` inside it (Qwen3.5).
    - It is never the scratch or traced forward, and never multirow without `DIRECT_FP32_GEMV=1`.
    - The value is data: `bpm.cpu_ref_path` in thresholds.yaml, with a basis.
    - The receipt derives `act_path_by_qtype` from the reference run's `kernel_path` precision.
    - It lists in `ref_mixed` the qtypes with no kernel for the pinned path (`bpm.ref_mixed_qtypes`; for `fp32_act` that is Q4_0 and Q8_0). The list is pinned, never chosen by the run.
  - **Flip cost:** one line in thresholds.yaml plus the f008c fixture's expected verdict. No M receipt exists before LIVE 0.70.1, so nothing is re-measured.

## §11a — `apr parity*` CPU activation path (origin/main 316dee2cd4, re-read 2026-10-03)

Correction: the first reading (aca6f2d7f6, 2026-09-27) said Q5K/Q6K stay on Q8K activations under the scope. That
was wrong when written. Q5_K/Q6_K never quantize activations, and the code was the same at aca6f2d7f6. Only one
commit touched these files since: 989cb012e5 (#4655).

Mechanism, per qtype (aprender-serve):
- **Q4_K, single row.** `fused_q4k_parallel_matvec_into` (quantize/parallel_k.rs:304) takes f32 activations when
  `fp32_activations_scoped()` (set by `with_fp32_activations`, :236) or `DIRECT_FP32_GEMV=1`. Otherwise it quantizes
  to Q8_K (:326).
- **Q4_K, crushed blocks.** The honest forward switches one Q4_K matvec to f32 when `has_crushed_block(x)`
  (`matvec_into_honest`, ffn_block.rs; `fused_q4k_parallel_matvec_f32_into`, L0-1b #2971). This is data-dependent.
- **Q4_K, multirow (m > 1).** `fused_q4k_multirow_matmul_f32_into` reads only `DIRECT_FP32_GEMV`
  (q4k_q8k_multirow.rs:178). The scope is ignored.
- **Q5_K / Q6_K.** Always f32. `fused_q5k_parallel_matvec_into` and `fused_q6k_parallel_matvec_into`
  (q5k_q6k_matvec.rs:7, :48, included at parallel_k.rs:498) run `generic_parallel_matvec_into` with the f32 dots
  (`fused_q6k_dot_simd`, fused_q5k_q6k.rs:118). There is no Q8_K path for them.
  - quantize/fused_q.rs is an older copy of q5k_q6k_matvec.rs that nothing compiles (no `mod`, no `include!`).
    Four falsify tests still cite it as the CPU path: falsify_q6k_fp_accumulator_order_001.rs:101,
    falsify_q6k_activation_amplification_002.rs:84, falsify_q6k_chain_length_003.rs:104 and
    falsify_q4k_bisect_dequant_007.rs:128.
- **Q4_0 / Q8_0.** Always Q8_0 activations (`fused_q4_0_q8_0_*`, `fused_q8_0_q8_0_parallel_matvec_into`). There is no
  f32 path.
- **Callers that pre-quantize whatever the scope says:**
  - `fused_gate_up_q4k_into` (fused_gate_up.rs:177) uses Q8_K, except on crushed blocks. Its Q5_K/Q6_K
    siblings (:240, :265) use the f32 dots.
  - `forward_single_with_scratch` sets `use_q8k_path = hidden_dim % 256 == 0` (results.rs:541). It has no
    non-test callers.
  - The traced forward does the same (traced.rs:88).
- **Which FFN branch the honest forward takes** (`single_cache_ffn_block`, ffn_block.rs:14):
  - RMSNorm models with an FFN norm, other than Gemma-1, use `ffn_up_gate_honest` (:44).
    - A Q4_K pair runs `matvec_honest` per tensor (ffn_block.rs:844, in `ffn_up_gate_honest` at :833).
    - Any other pair goes to `fused_rmsnorm_ffn_up_gate` (fused_matmul_into.rs:463). There a Q4_0 pair uses
      the fused RMSNorm + Q8_0 kernel (quantize/activation.rs:342), and everything else runs `fused_matmul`
      per tensor (matmul_fused.rs:134).
    - `fused_matmul` never takes the multirow path: for seq_len > 1 it loops the single-row matvec.
    - So a Q5_K/Q6_K up/gate stays f32, and the label holds.
  - Gemma-1 (`is_gemma1`, config.rs:391: arch exactly "gemma" or "gemmaforcausallm"), gated LayerNorm
    models and gated models without an FFN norm use the non-fused gated branch.
    - It calls `fused_gate_up_matmul_into` (ffn_block.rs:58 → fused_matmul_into.rs:163), so their Q4_K
      up/gate stay on Q8_K inside the scope.
    - Its only other callers are in the scratch forward (results.rs:51, :103), which is never the reference.
    - The fused gate/up kernels have no caller except `fused_gate_up_matmul_into` (fused_matmul_into.rs:187,
      :198, :209).
- **LM head** (`single_cache_final_output`, ffn_block.rs:139). The dense forward calls it at :587 and the MoE
  forward at forward_qwen3_moe.rs:504.
  - RMSNorm models use `fused_rmsnorm_lm_head` (fused_matmul_into.rs:441). A Q4_0 head runs the fused
    RMSNorm + Q8_0 kernel (`fused_rmsnorm_q4_0_matmul`, quantize/activation.rs:273). Any other head runs
    `fused_matmul`.
  - Unit-offset RMSNorm and LayerNorm models normalize first, then run `fused_matmul` (ffn_block.rs:185, :195).
  - `fused_matmul` sends Q4_K to `fused_q4k_parallel_matvec` (matmul_fused.rs:368), which wraps the `_into`
    above, so it honours the scope. Q5_K/Q6_K are f32, Q4_0/Q8_0 are Q8_0, and F16/BF16/F32 are float.
  - Qwen3.5 runs `fused_matmul_into` on its head (forward_qwen35.rs:1051). That reaches the same `_into` kernels.
  - So the head keeps the label on all three reference forwards. A Q6_K head is f32 under either label.

| command | source | CPU fn | activation path |
|---|---|---|---|
| `apr parity` (dense) | apr-cli parity_03.rs:185 | forward_single_with_cache (honest path) | mixed: Q4_K on Q8_K (f32 on crushed blocks), Q5_K/Q6_K f32, Q4_0/Q8_0 Q8_0 |
| `apr parity --moe` | parity_moe.rs:110 | with_fp32_activations(forward_single_qwen3_moe_with_cache) | fp32_act on Q4_K and Q6_K; Q4_0 tensors stay Q8_0 |
| `apr parity` hybrid/qwen35 | parity_hybrid.rs:268 | forward_single_qwen35 (forward_qwen35.rs:993) → fused_matmul_into per tensor (:1077–:1177), no fused gate/up | mixed, as dense (no scope); MoE arm → parity_moe |
| `apr parity` per-op | parity_per_op.rs:85 | forward_single_with_cache | mixed, as dense (no scope) |
| `apr kernel parity` | kernel_parity.rs | attention tiled vs naive | N/A (not a forward pass) |
| attn-parity-lint | attn_parity_lint.rs | consumes kernel-parity JSON | N/A |
| quantize-flag parity | quantize_flag_parity.rs | argv parity | N/A |

The MoE forward reaches experts through `matvec_for_qtype` (qwen3_moe_load.rs:442). That calls
`fused_q4k_parallel_matvec`, which wraps the `_into` above, so it honours the scope. Its QKV and output
projections use `fused_matmul`. `SUPPORTED_EXPERT_QTYPES = [Q4_K, Q6_K]` (qwen3_moe_load.rs:84), so an
expert tensor of any other qtype refuses on the CPU reference.

Production CPU decode runs the same honest forward without the scope (sync_owned_quantized_02.rs:82/171,
forward/batch_size.rs:63/98, apr_q4k_scheduler.rs:469). So "what users run" is the mixed path, not a pure `q8k_act`.

RQ-5 implication (ruled `fp32_act`, §11):
- `fp32_act` is honest with no code change for Q4_K_M (Q4_K + Q6_K): run the honest or MoE forward token by
  token inside the scope.
- The label is false for the scratch and traced forwards, for fused gate/up, and for multirow without the env var.
  The LM head is not an exception: on every reference forward it honours the scope, or its qtype fixes the path.
  The receipt must name the forward, and derive the per-qtype path from the trace instead of trusting the label
  (FALSIFY-BPM-008, fixture f008d).
- Q4_0/Q8_0 tensors keep a Q8_0 reference under either label, so they are recorded as `ref_mixed`.
- The 0.73 models (§3: Qwen3 dense, Qwen3.5, Qwen3-Coder-30B-A3B) never reach fused gate/up:
  - Qwen3 is RMSNorm with an FFN norm.
  - Qwen3.5 and the MoE have their own forwards, with no fused gate/up.
  - A Gemma-1 cell would reach it, so its Q4_K up/gate would contradict the label. BPM-008 refuses that
    receipt from the trace (f008d), provided the fused gate/up kernel emits its `kernel_path` entry (P2).

## 12. L25 review: vacuous-pass holes (2026-10-03, origin/main 316dee2cd4, read-only)
The draft gated the right quantities, but seven inputs could make a cell PASS without the check measuring anything. Each now has a planted falsifier in `backend-parity-matrix-v1` (BPM-009..015):

| # | Hole | Why it passes vacuously | Fix | Falsifier |
|---|---|---|---|---|
| 1 | NaN in one prompt | `f32::min` returns the non-NaN operand, so a fold drops the NaN prompt. apr's cosine helpers (parity_per_op_table.rs:61, forward_error.rs:254) return 0.0 on a length mismatch or zero norm, but NaN on NaN logits [V] | A non-finite cosine REFUSES the receipt | BPM-009 |
| 2 | Legs over different prompt sets | The triangle bound holds per prompt | Both legs carry `prompt_set_sha256` and `n`, which must match | BPM-010 |
| 3 | Legs over different model files | Leg B then measures the file, not the engine | `model_sha256` is equal across all 3 runs | BPM-011 |
| 4 | CPU cell C4 is its own reference | Same kernels on both sides read 1.0. Q5K/Q6K have one CPU activation path (f32) under either label (§11a), so the R3 NEON Q6K dot would meet itself | No quantized matmul tensor shares a `kernel_id` between the two runs, over every kernel the tensor reached: compared per tensor, not per path, not per OBS slot, not on `arch` (below) | BPM-012 |
| 5 | Prefill-final logits only | Never touches the KV cache or the M=1 decode GEMV, which is where the known 0.955 sits | Gate on prefill-final plus ≥ 8 teacher-forced decode positions | BPM-013 |
| 6 | Adapter name denylist | An unlisted software adapter passes | `AdapterInfo.device_type = Cpu` ⇒ FALLBACK; names stay as a second check. Banner from `AdapterInfo.backend` (F3) | BPM-014 |
| 7 | E2 over unequal token counts | An early EOS decodes fewer, shorter-context tokens | One `host_id`, fixed `n_gen`, EOS ignored on both engines | BPM-015 |

Reuse note: origin/main has at least 8 cosine copies (`git grep 'fn cosine'`). The R1 checker uses `parity_per_op_table::cosine`, which already fails closed on length and zero norm, and adds no ninth copy.
pv 0.70.0: validate 0 errors / 0 warnings; `pv lint contracts-draft/` PASS. Obligations went from 6 to 10.

BPM-012 per tensor (2026-10-03). The first form compared whole kernel paths, and a mixed run evades that. A same-host reference differs from a default-route backend on Q4_K (q4k-q8k against q4k-f32), while every Q6_K tensor meets itself. The OBS-15 `kernel_diff` (#4574, unmerged) would not help:
- it keys by `(op, shape_class)` and keeps one entry per slot (obs_kernel_path.rs:118, last wins). A Q4_K_M file mixes Q4_K and Q6_K in one slot: in Qwen3.5-0.8B-Q4_K_M, ffn_down (3584, 1024) is Q6_K in 12 layers and Q4_K in 12 [V];
- it compares whole entries, and `arch` differs on every C0-against-C4 entry, so one scalar kernel on both hosts reads as different.

BPM-012 now compares, per (tensor, op), the sets of `kernel_id`s each run reached over the quantized matmul tensors, and refuses on any overlap. It needs sets because the route can switch per call: on the default route a crushed activation block sends that one Q4_K call to f32 (ffn_block.rs:790-792) [V]. P1 plants f012 (one equal tensor among differing ones), f012b (the C4 control), f012c (one kernel_id on two arches), f012d (a slot collision) and f012e (a tensor with two routes). The C4 leg meets the rule through R3's dispatch-honesty precondition, not through the second machine (R3 §13).
pv 0.70.0 after this: validate 0/0 on both contracts; lint 0 errors and the same 5 lean_theorem warnings. Counts unchanged (BPM: 4 equations, 17 falsifiers, 12 obligations).
