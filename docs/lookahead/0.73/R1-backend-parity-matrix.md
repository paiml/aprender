# R1 — Backend parity matrix (0.73 "Runs Everywhere", epic #3999) — DRAFT

Status: draft by la-73, 2026-09-27. There is no ticket yet: the cop mints it. It becomes a spec-only PR
when the repo is under the 10-PR cap. Every number is [A] until a cell is measured.

Paths: `results.rs` = `crates/aprender-serve/src/gguf/inference/forward/results.rs`, `config.rs` = `crates/aprender-serve/src/gguf/config.rs`, `router.rs` = `crates/aprender-serve/src/api/router.rs`, `batch.rs` = `crates/aprender-serve/src/api/batch.rs`, `types.rs` = `crates/aprender-serve/src/api/types.rs`, `run.rs` = `crates/apr-cli/src/commands/run.rs`, `fused_q5k_q6k.rs` = `crates/aprender-serve/src/quantize/fused_q5k_q6k.rs`.

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

Each cell has two routes, `apr run` and `apr serve`, since E1 names both. A route passes E1 only through a leg A
that ran its decoder, and a cell meets E1 only when both routes pass (§13, FALSIFY-BPM-019).

## 3. E1 — correctness (composed-leg cosine)
- Leg A: apr-backend vs apr-CPU on the same host, same GGUF, final-position logits. This is `apr parity`,
  which exists (registry `contracts/apr-cli-commands-v1.yaml`). `apr parity --per-op` is the diagnostic
  for a failing cell (`apr-parity-per-op-v1`).
  At 316dee2cd4 `apr parity` runs only on a cuda build (`apr-cli/src/commands/parity_03.rs:214-218`), so it is
  leg A for C0 and C5 only: C1–C3 have no leg-A tool, and C4's leg A compares its own CPU run with the C0
  reference (§12, row 4). Its Qwen3.5 arm runs one token at a time, while `apr run` and `apr serve` prefill in
  one batched call; §13 says what that does to E1.
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
labelled by intent), `apr parity` / `--per-op` (leg A on C0 and C5 + diagnosis), #4464 receipt shape, APR-OBS
identity lint.
New work: the leg-B logits harness, the leg-A arms and route runs of §13, the matrix runner (one cell per
invocation; it runs only when train-active is clear, with the GPU lock around the binary), and the contract.
The harness, the leg-A arms, the route runs and the runner have no bundle yet: they are landing-map row 10
until RQ-9 is ruled (§13).

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
  (q4k_q8k_multirow.rs:178). The scope is ignored. Fix: ticket S2 (`ticket-bodies-side-fixes.md`).
- **Q5_K / Q6_K.** Always f32. `fused_q5k_parallel_matvec_into` and `fused_q6k_parallel_matvec_into`
  (q5k_q6k_matvec.rs:7, :48, included at parallel_k.rs:498) run `generic_parallel_matvec_into` with the f32 dots
  (`fused_q6k_dot_simd`, fused_q5k_q6k.rs:118). There is no Q8_K path for them.
  - quantize/fused_q.rs is an older copy of q5k_q6k_matvec.rs that nothing compiles (no `mod`, no `include!`).
    Four falsify tests still cite it as the CPU path: falsify_q6k_fp_accumulator_order_001.rs:101,
    falsify_q6k_activation_amplification_002.rs:84, falsify_q6k_chain_length_003.rs:104 and
    falsify_q4k_bisect_dequant_007.rs:128. Ticket S3 (`ticket-bodies-side-fixes.md`) deletes the file and re-points the four cites.
- **Q4_0 / Q8_0.** Always Q8_0 activations (`fused_q4_0_q8_0_*`, `fused_q8_0_q8_0_parallel_matvec_into`). There is no
  f32 path.
- **Callers that pre-quantize whatever the scope says:**
  - `fused_gate_up_q4k_into` (fused_gate_up.rs:177) uses Q8_K through `fused_q4k_q8k_ffn_up_gate_into` (:227),
    except when the input has a crushed block. Its fallback for that case (#2971, :189-213) calls
    `fused_q4k_parallel_matvec_into`, which takes f32 only inside the scope or with `DIRECT_FP32_GEMV`
    (parallel_k.rs:303-304). Outside the scope the fallback quantizes too, and it never calls
    `note_crushed_fallback` (item e, §11b). Its Q5_K/Q6_K siblings (:240, :265) use the f32 dots.
  - `forward_single_with_scratch` sets `use_q8k_path = hidden_dim % 256 == 0` (results.rs:541). It has no
    non-test callers.
  - The traced forward does the same (traced.rs:88).
- **Which FFN branch the honest forward takes** (`single_cache_ffn_block`, ffn_block.rs:14):
  - RMSNorm models with an FFN norm, other than Gemma-1, use `ffn_up_gate_honest` (ffn_block.rs:44).
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

## §11b — RQ-5 info rows (item e, 2026-10-03, origin/main 316dee2cd4, read-only)

RQ-5 made `fp32_act` the E1 reference and `q8k_act` an info row only. Item (e) gives info rows a shape and a rule, and writes the first row.

- **Shape.** `receipt_shape` gains `info_rows`: name, route, the runs (run sha256, model and prompt-set sha256, `kernel_path`) and per-pair cosines {min, median, max, n}. The checker derives each row's status, reason and labels, and reads none of them from the producer.
- **Rule, FALSIFY-BPM-018.** A row never changes the verdict. f018a: a failing leg next to a 0.999 info row stays Fail(E1 min). f018b: a 0.985 info row, or one holding a "NaN" cosine, leaves a Pass receipt Pass, and the NaN row reads not_measured(non_finite). An info row moved into a leg is BPM-008 (f008c).
- **The first row, `c4_default_route_info`** (neon-q4k-q6k-v1, FALSIFY-NEON-Q4K-009). r1 = cos(C4 default, C0 `fp32_act` reference) is the end-to-end figure. r2 = cos(C0 default, the same reference) is the x86 activation-quantization gap. r3 = cos(C4 default, C0 default) is the cross-host kernel difference on one route. The row is measured only when `kernel_path` proves the default route on both hosts, the three runs are one bound triple (f0 is the run leg_a cites), and the angles close a triangle within 1e-3 rad. The f32 cosine helpers lose up to 2.44e-4 rad per angle, so three lose at most 7.3e-4.
- **Why an info row, not a leg.** #3714 quotes CUDA MoE at cos 1.000000 against `fp32_act` and 0.985 against the production Q8_K path (Qwen3-Coder-30B-A3B, not re-measured). So r2 may sit under the 0.995 floor, and a gated default-route leg would measure activation quantization, not NEON. Dense Qwen3 is [U]. GPU paths use exact-FP32 activations (§11), so the default route has no GPU-cell analogue, and the earlier framing (a C4 leg against the reference "as a GPU cell runs") is withdrawn. Whether the gx10 default route should gate is RQ-6 (handoff), with r3 as the candidate. The provisional S-4 default is info only.
- **Correction [V].** Dense Qwen3 decode in `forward_single_with_cache` reaches C and never D, on either route. D (`fused_q4k_q8k_ffn_up_gate_into`) runs only from `fused_gate_up_q4k_into` (fused_gate_up.rs:227), through the non-fused gated branch (ffn_block.rs:58, then fused_matmul_into.rs:180-187; taken by LayerNorm models, models with no FFN norm, and Gemma-1, ffn_block.rs:42-58), and from `scratch_q8k_up_gate` (results.rs:33), which only the scratch and traced forwards reach (results.rs:608, traced.rs:179). The 20:59Z handoff Next said the default route reaches C and D.
- **Side defect [V].** The #2971 crushed fallback in `fused_gate_up_q4k_into` (fused_gate_up.rs:189-213) calls `fused_q4k_parallel_matvec_into`. Outside the scope that function quantizes to Q8_K (parallel_k.rs:303-304), so the fallback changes nothing, and it never calls `note_crushed_fallback`. The fix calls `fused_q4k_parallel_matvec_f32_into`, as `matvec_into_honest` does (ffn_block.rs:791-792). It becomes a PROPOSE-TICKET after LIVE 0.70.1 (ticket body S1, `ticket-bodies-side-fixes.md`); no 0.73 model reaches it.

pv 0.70.0 after item (e): validate 0/0 on both contracts; lint 0 errors and 6 lean_theorem warnings (the new one is `c4_default_route_info`). BPM: 4 equations, 18 falsifiers, 13 obligations.

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
- it keys by `(op, shape_class)` and keeps one entry per slot (obs_kernel_path.rs:118 on #4574 @b1244f6fb7, last wins). A Q4_K_M file mixes Q4_K and Q6_K in one slot: in Qwen3.5-0.8B-Q4_K_M, ffn_down (3584, 1024) is Q6_K in 12 layers and Q4_K in 12 [V];
- it compares whole entries, and `arch` differs on every C0-against-C4 entry, so one scalar kernel on both hosts reads as different.

BPM-012 now compares, per (tensor, op), the sets of `kernel_id`s each run reached over the quantized matmul tensors, and refuses on any overlap. It needs sets because the route can switch per call: on the default route a crushed activation block sends that one Q4_K call to f32 (ffn_block.rs:790-792) [V]. P1 plants f012 (one equal tensor among differing ones), f012b (the C4 control), f012c (one kernel_id on two arches), f012d (a slot collision) and f012e (a tensor with two routes). The C4 leg meets the rule through R3's dispatch-honesty precondition, not through the second machine (R3 §13).
pv 0.70.0 after this: validate 0/0 on both contracts; lint 0 errors and the same 5 lean_theorem warnings. Counts unchanged (BPM: 4 equations, 17 falsifiers, 12 obligations).

## 13. The `apr serve` half of E1, and leg A per route (item m, 2026-10-04, origin/main 316dee2cd4, read-only)
E1 names `apr run` and `apr serve`. §2–§8 measure one route per cell and judge it by leg A, but the oracle judges
only the decoder that leg A ran, and at 316dee2cd4 neither route is shown to run that decoder. So the serve half
of E1 is a falsifier, FALSIFY-BPM-019, and not the run oracle reused.

### 13a. What leg A runs today
- `apr parity` builds only with cuda: without it the command returns FeatureDisabled
  (`apr-cli/src/commands/parity_03.rs:214-218`). Its arms are in `parity_hybrid.rs:15-25`. The Qwen3.5 arm
  builds `Qwen35CudaModel::with_max_seq_len` (:246), and `run_hybrid` (:294) compares `forward_single_qwen35`
  with `Qwen35CudaModel::forward_single` one token at a time.
- The Qwen3.5 session prefills "in one batched call on the GPU"
  (`aprender-serve/src/gguf/inference/forward/qwen35_session.rs:472`). `APR_QWEN35_SESSION_PREFILL=per-token`
  restores the per-token prefill that 0.69.1 served, so that the two "can be compared token for token on ONE
  binary" (:35-39). So by default leg A runs the prefill of neither route.
- The two routes do not plan the same prefill. `apr run` loads through `Qwen35Session::load_for_run`
  (`aprender-serve/src/infer/inference_result.rs:376`; `qwen35_session.rs:252-262`), which passes the run's
  positions through `from_host` (:380) to `GpuBackend::build` (:407). That plans the CUDA prefill only when it
  has positions (:85-87), and only then sets the rows per chunk and the attention path (:103-105), from
  `plan_capacity` (:149): cuBLAS f32 attention while its plan fits, flash only when flash alone fits, and bigger
  chunks on a unified-memory host. `apr serve` calls
  `Qwen35Session::load` (`apr-cli/src/commands/serve/server.rs:172`), which reaches `from_host` with no positions
  (`qwen35_session.rs:240-241`, :325-327), so it keeps the model defaults
  (`aprender-serve/src/gguf/cuda/forward_qwen35_cuda_prefill.rs`): the first attention candidate (:199-207) and
  512 rows per chunk (:39, :290-294), where a run on C5's unified memory tries 2048 first (:45). Rows per chunk
  are capped by the prompt (:211-213), so a prompt under 512 positions is one chunk on serve. The run's chunk
  and attention path come from its plan, so the two routes prefill alike only when the plan picks the defaults.
- CPU-GPU-006 in `apr run` is not a leg A. `try_wgpu_generate`
  (`aprender-serve/src/infer/gguf_gpu_generate.rs:160`) probes three steps (:236-264) against the CPU
  `forward_single_with_cache` (:267), not the `fp32_act` reference (RQ-5). It accepts at `cos >= 0.99` (:304)
  and logs and falls back below that (:306-310). Its note (:104-120, #3827) withdraws the 0.999863 figure:
  #3757 measured 0.955 on intel, gx10, mini and the RTX 4090.

### 13b. The two routes on each backend
| Model, backend | `apr run` | `apr serve` | Same decoder as leg A? |
|---|---|---|---|
| Qwen3.5, CUDA (C0, C5) | the session, through `load_for_run`; batched prefill on the run's plan | the session, through `load` (`server.rs:172`); batched prefill on the model defaults | No, by default: leg A runs per token. The per-token variable makes both routes prefill per token. Whether each decode step is leg A's `forward_single` is [U]; BPM-019 checks it |
| Qwen3.5, aarch64 CPU (C4) | the session on the CPU: the default route (Q4_K on Q8_K activations) | the same, when serve is started without an accelerator (`server.rs:172` passes `!config.wants_accelerator()` as no_gpu) | No: leg A runs `fp32_act` (RQ-5), and the default route is an info row while RQ-6 holds (`c4_default_route_info`). Both routes are Refused |
| Qwen3.5, wgpu (C1–C3) | refused, exit 14 (`apr-cli/src/commands/run_entry.rs:331`) | refused at load: `--backend wgpu` reaches `try_start_wgpu_backend` first (`apr-cli/src/commands/serve/handlers.rs:947`, fn :905), and its `build_serve_model` (:922) refuses the layerless Qwen3.5 base (`handler_gpu_completion.rs:487`) before the Qwen3.5 route (:396) | Neither route runs (landing-map row 6) |
| dense, wgpu (C1–C3) | `try_wgpu_generate`: raw Q4K weights (`gguf_gpu_generate.rs:205`), eps from the config (:197) | `serve_wgpu_backend` (`handlers.rs:746`): F32 weights unless `WGPU_Q4K` (:510), a literal final-norm eps of 1e-6 (:133-148), per-layer eps from the config (:771), and a smoke test on the argmax only (:534-580) | No: two decoders, and no wgpu leg A |
| dense, CUDA (C0, C5) | [U] | [U] | [U]; BPM-019 checks it |

### 13c. What a route can be compared on
- Serve returns no full-vocab logits. `/v1/logprobs` builds only with cuda and returns log probabilities per
  generated token (`aprender-serve/src/api/gpu_completions_handler.rs:697-703`, mounted at `router.rs:222`),
  and the Qwen3.5 completions backend sets `logprobs: None` (`qwen35_completions_backend.rs:348`).
- Serve returns tokens. `POST /generate` (`router.rs:97`; `batch.rs:442`, `qwen35_raw_generate.rs:90`) returns
  `GenerateResponse.token_ids` (`types.rs:110`), and the Qwen3.5 server mounts that router
  (`apr-cli/src/commands/serve/server.rs:153-160`, :418-430).
- `apr run --stream` returns tokens: `print_stream_output` (`apr-cli/src/commands/run_entry.rs:560-561`, fn :700)
  writes a `token_id` for each of `RunResult.generated_tokens` (:723-729). `execute_with_realizar` fills that
  field with the tokens after the prompt (`apr-cli/src/commands/inference_output.rs:366`, :380, :396-404;
  `run.rs:361`), and for Qwen3.5 those come from the session in `run_gguf_inference`
  (`aprender-serve/src/infer/inference_result.rs:239`, :267, :376-383, :396).
- The legs are teacher-forced (BPM-013) and a route generates freely, so a route is compared with the other
  route, never with a leg.

### 13d. FALSIFY-BPM-019 (route binding)
The receipt carries `routes`: `run` and `serve`, each {`binary_sha256`, `model_sha256`, `host`,
`prompt_set_sha256`, `n_gen`, `forward_trace_line` (the prefill forward with its attention path and rows per
chunk, and the decode forward), `kernel_path`, `tokens`}. Both routes are greedy and non-batched, and `serve`
goes through `POST /generate`. The checker applies five rules:
1. Identity: a route's binary, model, host and prompt set are the cell's, else Refused(route identity).
2. Decoder: per (tensor, op), a route's `kernel_id` sets equal those of leg A's backend run, and its forward
   line names the same forwards, else Refused(route binding).
3. Tokens: the run and serve tokens are equal on every prompt, else Fail(route tokens).
4. Inheritance: a bound route's verdict is the cell's, so a route never lifts a failing cell.
5. Absence: a missing route is NotRun(route).

The planted fixtures f019a..f019d are in the P1 spec §4, and the seven checker mutations in its §5.

### 13e. At 316dee2cd4, under the defaults
- Qwen3.5 on C0 and C5: Refused(route binding). Both routes prefill in one batched call and leg A runs per
  token. With `APR_QWEN35_SESSION_PREFILL=per-token` the prefill can bind, but it is then the 0.69.1 prefill,
  not the default, and the session prints that the variable is set.
- C1–C3: there is no leg A, so each route is NotRun until landing-map row 10 lands (RQ-9). The Qwen3.5 routes
  do not run at all (row 6).
- C4: Refused while RQ-6 holds (13b).
- Dense wgpu: serve is a second decoder, so a leg A built on `try_wgpu_generate` would bind the run route only.
- So no Qwen3.5 cell can pass E1 at 316dee2cd4. Only dense CUDA on C0 and C5 can, if its routes bind, which
  is [U].

### 13f. Known limits
- `kernel_path` carries no scalar parameters, so serve's literal eps of 1e-6 would bind on `kernel_path` alone.
  That is why rule 2 also reads the forward line, which names the forward that ran.
- Only matmul sites emit `kernel_path`, so a different attention path shows only in the forward line. That is
  why the prefill entry names its attention path.
- The batched serve (`start_gguf_server_gpu_batched`, `server.rs:479`) is out of scope: both routes are
  non-batched.

Open:
- RQ-9 (handoff): who builds leg A for each route and §8's harness? (a) a bundle P6 ranked after P2; (b)
  landing-map row 10, the default; (c) the wgpu arm in P4 and the rest in row 10. Under (b) no falsifier moves,
  the gate stays at 42, and the M runs that need a wgpu or Qwen3.5 leg A wait on row 10.
- If RQ-6 flips, the C4 routes are compared without the crushed-block f32 switch entries on either side
  (`aprender-serve/src/gguf/inference/forward/ffn_block.rs:790-792`, f012e), because that switch depends on the
  activations. Under the default, RQ-6 holds and the C4 routes are Refused.

pv 0.70.0 after this: validate passes on all four draft contracts; lint 0 errors and the same 9 warnings as before the edit (6 lean_theorem, 3 postconditions). BPM: 4 equations, 19 falsifiers, 14 obligations. `falsifier_gate.py`: 0 of 11 checks differ from the K9 table.
