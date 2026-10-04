# 0.73 L3 ticket bodies, P1 to P5 (DRAFT, not filed; la-73, 2026-10-03)

These are bodies ready to file, one per bundle in `falsifier-landing-map.md`.
- **Filing:** they are filed only after the cop writes LIVE 0.70.1, as PROPOSE-TICKET lines to the cop. L3 never mints (C277).
- **Format:** each body follows the repo shape: scope, files, acceptance, blockers, hosts, out of scope.
- **Citations:** at origin/main 316dee2cd4 unless a line says otherwise.

---

## P1 — backend-parity cell receipt checker (planted receipts, x86 CI)
**Refs** #3999 (0.73 E1/E2/E3). Contract: `backend-parity-matrix-v1` (draft). Spec: `docs/lookahead/0.73/P1-receipt-checker-spec.md`.

**Scope**
- A new extractor `crates/aprender-contracts/src/ontology/extract/bpm_cell_receipt.rs`, next to `parity_receipt.rs`. It turns `evidence/bpm/**/*.json` (`apr-bpm-cell-receipt/v1`) into a verdict: `Pass | Fail | Refused | NotRun | Unknown{WrongCorpus}`.
- Legs A and B are `parity-receipt-v2` records cited by sha256, so P1 adds no new logit shape.
- Floors come from a `bpm:` key in `evidence/parity/thresholds.yaml` and carry a basis: 0.98 reuses the measured default, and 0.995 is derived from it.
- Reach is pinned in `evidence/bpm/EXPECTED_RECEIPTS`, using an independent predicate script.

**Acceptance**
- **Control:** the planted `base.json` passes. Without this, a checker that refuses everything would pass.
- **Planted receipts:** 47 files under `tests/fixtures/bpm/` (`base.json`, the 37 rows of spec §4 with the second fixtures of f010, f012e, f013, f018b, f019a and f019c and the three of f017, and f002b) give their spec §4 verdicts. They cover BPM-001..006 and 008..019, NEON-Q4K-009, plus WGF-005, WGF-009, R4-003 and the uncaptured-stderr case.
- **Mutations:** the 29 checker mutations in spec §5 each flip their named receipt. The run is recorded in the PR body.
- **Composed bound:** a const test checks 2·acos(0.995) ≤ acos(0.98).
- **Command:** `cargo test -p aprender-contracts --lib` runs all of it.

**Blocked by**
- The OBS stack (#4487 identity block, #4574 kernel-path shape) is not on main. Until it merges, the identity check returns `NotRun(identity lint absent)`, never a pass.
- RQ-5 is ruled (cop, 2026-09-27 20:12Z): `cpu_ref_path` = `fp32_act`, held as data in `bpm.cpu_ref_path`. A `q8k_act` run is an info row only and never an E1 receipt (f008c). Info rows have their own field and never enter the verdict (BPM-018).

**Hosts** x86 CI only. No GPU, no model.
**Out of scope** Running any model; P2 trace fields; arming the shape in `lint-baseline.json` (shared file, a separate labelled follow-up).

---

## P2 — trace fields the cell receipt reads
**Refs** #3999. Depends on P1 for the consumer. Shapes reuse `apr-kernel-path-v1` (OBS-15, #4574).

**Scope:** aprender-serve emits these fields in the forward trace:
- `kernel_path` entries `{op, kernel_id, qtype, layout, arch, shape_class, precision}` plus `tensor` (the GGUF name), one per kernel each (tensor, op) reached, on every backend, CPU included. This is stricter than OBS `trace_cut`, which exempts CPU rows; BPM-012 needs CPU entries. It compares per tensor because the OBS slot `(op, shape_class)` merges the Q4_K and Q6_K layers of one role in a Q4_K_M file, and over sets because a crushed activation block switches one Q4_K call to f32.
- per-op `op_placement` over the full DECODE_OPS set.
- `qtype_path` per qtype: device-kernel, host-widen or refused-cpu.
- `head_dim_source`: key_length or fallback.
- a wgpu forward banner line, the counterpart of the CUDA `Backend: GPU (…)` line.
- `precision` in each `kernel_path` entry names the activation path (f32, q8k or q8_0), including the per-call crushed-block f32 switch. The receipt derives `act_path_by_qtype` from it (BPM-008, f008d), and the forward trace line names the forward that ran. Every matmul site emits its entry, including fused gate/up (fused_gate_up.rs:177) and the fused RMSNorm + Q8_0 kernels.
- Both routes name the prefill forward (its attention path and rows per chunk) and the decode forward apart in the forward trace line, and `apr serve` emits the same `kernel_path` and forward line as `apr run` for each non-batched request (FALSIFY-BPM-019). Today only the .apr CPU handler returns trace data (`apr-cli/src/commands/serve/handler_apr_cpu_completion.rs:299`), and `POST /generate` returns tokens only.

**Acceptance**
- A schema test asserts every field is present on a CPU run of a tiny fixture model.
- `op_placement` keys equal DECODE_OPS, and `kernel_path` is non-null on CPU.
- **Mutation:** drop the `attention` key; WGF-009's planted receipt in P1 must then be refused.
- **Act-path control:** a tiny gated model forced down the non-fused gated branch (ffn_block.rs:58) inside `with_fp32_activations` must show its Q4_K up/gate at precision q8k. If the trace says f32, the precision field is not evidence.
- **Route fields:** a non-batched `POST /generate` on the tiny fixture model returns a `kernel_path` and a forward line in the same schema as `apr run`'s, with the prefill and decode forwards named apart. Whether they are equal is BPM-019's check on a real cell, not P2's.

**Hosts** x86 CI (CPU forward). The wgpu values are checked later, in P4.
**Out of scope** New kernels; changing the OBS-15 contract (flag the CPU-row conflict to its owner instead).

---

## P3 — NEON Q4_K / Q6_K dots for aarch64 (C4)
**Refs** #3999 (C4 E1/E2). Contract: `neon-q4k-q6k-v1` (draft). Skeletons: `docs/lookahead/0.73/R3-test-skeletons.md`.

**Scope**
- NEON arms in the three live dispatchers: `fused_k.rs:193`, `quantize/fused_q5k_q6k.rs:118`, and `q4k_dot_avx2.rs:338` (compiled through `include!` at `fused_k.rs:370`).
- On gx10 the default Q8_K route reaches the `q4k_dot_avx2.rs:338` arm through `fused_q4k_q8k_dot_with_bsums_simd` (bsum_precompute.rs:220), which falls back to that dispatcher off x86. No fourth arm is needed for correctness (R3 §13, route).
- `kernel_path(k)` maps to OBS `kernel_id`.
- A widen entry exposed only to tests.
- The orphans `quantize/fused_q4k.rs` and `quantize/fused_q.rs` are not touched here. Each has its own PROPOSE-TICKET (R3 §13 rows 1 and 9). Body: S3 in `ticket-bodies-side-fixes.md`.

**Acceptance**
- NEON-Q4K-001..004 and 007 are green on gx10, with 10 000 cases each, f16-normal d/dmin, rows of 1, 2, 3, 7, 8 and 36 blocks under the derived bound 2·γ(K)·S, and the masked sweep.
- The masked-sweep power guard is green on x86. NEON-Q4K-002 asserts on both dotprod branches.
- The golden file is checked against a pinned sha256.
- NEON-Q4K-000 and 006 (the cross-target `cargo check` and the compile-site probe with the orphan as control) run off the release path.
- Every mutation listed in the contract turns its test red.
- The NEON arms add no Err condition beyond the scalar oracle's, because every CPU matvec turns a dot Err into a 0.0 row (R3 §13 row 10). FALSIFY-NEON-Q4K-008 is green on gx10. It calls every row's dot directly at 8 and 36 super-blocks per row, in five decode matvec entries and two prefill (multirow) entries, and checks that the matvec wrote that value to that row (R3-test-skeletons.md §4c).

**Hosts** gx10 (aarch64). x86 for the power guard, 000 and 006. No PR job runs these tests on aarch64 yet: mac-check only compiles them. Where the CI step goes is open (`docs/lookahead/0.73/R3-neon-q4k-q6k.md` §16).
**Out of scope** The C4 E1 measurement (M run, NEON-Q4K-005) and E2 speed.

---

## P4 — wgpu forward correctness (C1–C3)
**Refs** #3999 (C1–C3 E1). Contract: `wgpu-forward-v1` (draft). Scope doc: `R2-wgpu-forward-scope.md`.

**Scope**
- rope_theta and the RoPE pairing from metadata at every RoPE site, all at 316dee2cd4: the host decode RoPE that `apr run` uses (`wgsl_forward.rs:970`, pairing at `:972-1002`), the batch shader (`:278`, reached only from training) and the shader nothing dispatches (`:242`, deleted or fixed). A shader-only fix leaves decode unchanged.
- head_dim from `key_length` at all three sites (`gguf_gpu_generate.rs:185`, `:672` and `batch_wgpu.rs:151`, at 316dee2cd4).
- Q6_K, Q8_0 and Q4_0 WGSL GEMVs, so the exit model's Q6_K tensors stop being widened on the host.
- A qtype check on the ffn_gate weight, which today reaches the Q4_K GEMV unchecked (R2 D-1, static, unmeasured).
- Qwen3 q/k norm. The 0.955 was measured on qwen2.5-coder-1.5b, where none of the theta, head_dim or q/k-norm defects applies, so its cause is unattributed. The layer-diff trace (R2 item 0, K14) runs before any of these fixes is credited with it. The 0.955 first shows at position 1 (every 1.5B failure of the parity probe is at step 2/3, and position 0 passes 0.99), so the trace starts at position 1, layer 0: q and k after the bias, after RoPE, the scores and the attention output. Before the trace, one probe run with `DIRECT_FP32_GEMV=1` checks the reference: the probe judges the f32-activation wgpu GEMVs against a CPU forward on Q8_K activations (`q8k_act`), the #3714 shape (R2 item 0). If the 1.5B then passes 0.99 at all 3 steps, P4 moves the probe's CPU forward inside `with_fp32_activations`, as `crates/aprender-serve/src/infer/qwen3_moe_dispatch.rs:209` does [V at 316dee2cd4], and changes nothing in the wgpu forward for it.

**Acceptance**
- WGF-001, 002, 004 and 008 are green on C1, using synthetic fixtures at position ≥ 1 and pos_in_head ≥ 1.
- WGF-003 is green only after the K14 trace has run, on a fixture at position ≥ 1. WGF-002 and WGF-003 judge against the CPU forward on FP32 activations (`fp32_act`): against Q8_K a correct forward can miss 0.9999 (#3714). At position 0 attention has one key, so its output is that key's V and a skipped q/k norm cannot show.
- The mutations listed in the contract turn their tests red.

**Hosts** C1 (wgpu adapter), train-inactive. CI has no GPU.
**Out of scope**
- Moving attention, RoPE and the LM head onto the device (R2 item 5).
- The gated-delta route (R2 item 6, which may slip to 0.74). It is rank 6 in the landing map, and E1's WGPU and Metal legs wait on it.
- The ledger sweep (M run: WGF-006, 007 and 010).
- The batch wgpu path, which runs before CUDA and has no parity probe (R2 D-2). It is a separate ticket.

---

## P5 — MoE dispatch: streaming parity, and the wgpu MoE forward
**Refs** #3999 (E3). Scope doc: `R4-moe-gpu-wiring.md`.

**Scope**
- **(a)** A streaming variant of `run_qwen3_moe_generate_dispatch`, so a `stream: true` request takes the same path as non-streaming. Today it goes through `cuda_chat_backend.rs:1176`, bypassing the dispatch.
- **(b)** The wgpu MoE forward, which is stubbed today at `wgpu_backend/mod.rs:197`. (b) waits for P4.

**Acceptance**
- **(a)** R4-001 holds on C0: `used_gpu = true`, the CUDA banner and a judged F2 guard on both runs. A planted `SKIP_PARITY_GATE=1` run is refused by P1 (BPM-017).
- **(b)** R4-002 holds on C1–C3. Its leg A is ≥ 0.995, with a positive wgpu banner and `device_type ≠ Cpu`.

**Hosts** C0 for (a); C1–C3 for (b).
**Out of scope** Batched MoE prefill (K18, a candidate R4 item 5).

---

## Not a ticket: measurement runs (M)
These runs produce receipts, not code. They are scheduled when a host is train-inactive, and P1 checks each receipt.
- BPM-007 on C1: today's 0.955 control.
- NEON-Q4K-005 on C4.
- The WGF-006/007/010 ledger sweep on C1.
- R4-001 on C0 and R4-002 on C1–C3.
- Each E1 receipt carries both routes (FALSIFY-BPM-019). A run that needs a wgpu or Qwen3.5 leg A waits for landing-map row 10 (RQ-9).

---

## Status (2026-10-03 22:15Z)
- origin/main is 316dee2cd4. All file:line cites in R1–R5 and these bodies were re-checked there at 13:18Z (ac564391ab).
- **Filing:** still under S-1 hold (operator C292): nothing is filed or opened until LIVE 0.70.1. After LIVE, 0.73 moves from the floor to normal cadence: P1..P5 go to the cop as PROPOSE-TICKET lines, and this branch gets its PR.
- **Open rulings, not blocking (C293.3):** RQ-3, RQ-4 and RQ-6 are requested in the handoff. Work proceeds on provisional S-4 defaults: RQ-3 = E1 PASS + E2 PASS receipts on main; RQ-4 = a hybrid may pass E1, and E2/E6 name it; RQ-6 (gate the gx10 default route?) = info only, applied at c49bbb23cb. RQ-5 is ruled: `fp32_act` (cop, 2026-09-27 20:12Z). Added 2026-10-04: RQ-7 (where P3's aarch64 PR-CI step goes; default: P3's tests run by hand on gx10), RQ-8 (whether qwen35 on wgpu, landing-map row 6, goes above P5; default: it stays at 6) and RQ-9 (who builds leg A for each E1 route, and R1 §8's harness; default: landing-map row 10, and the M runs that need it wait).
- **External blocker:** the OBS stack, #4487 and #4574, is unmerged, so P1's identity check stays NotRun until it merges.
- **Side fixes:** S1..S3 (`ticket-bodies-side-fixes.md`) go out with P1..P5 at LIVE 0.70.1. They are not 0.73 gates, and none of P1..P5 waits on them.
