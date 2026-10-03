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
- **Planted receipts:** 24 under `tests/fixtures/bpm/` give their spec §4 verdicts. They cover BPM-001..006 and 008..017, plus WGF-005, WGF-009, R4-003 and the uncaptured-stderr case.
- **Mutations:** the 10 checker mutations in spec §5 each flip their named receipt. The run is recorded in the PR body.
- **Composed bound:** a const test checks 2·acos(0.995) ≤ acos(0.98).
- **Command:** `cargo test -p aprender-contracts --lib` runs all of it.

**Blocked by**
- The OBS stack (#4487 identity block, #4574 kernel-path shape) is not on main. Until it merges, the identity check returns `NotRun(identity lint absent)`, never a pass.
- RQ-5 sets the `cpu_ref_path` value, but it does not block (C293.3). P1 lands on the provisional default `fp32_act`, held as data in `bpm.cpu_ref_path`; a different ruling flips one line.

**Hosts** x86 CI only. No GPU, no model.
**Out of scope** Running any model; P2 trace fields; arming the shape in `lint-baseline.json` (shared file, a separate labelled follow-up).

---

## P2 — trace fields the cell receipt reads
**Refs** #3999. Depends on P1 for the consumer. Shapes reuse `apr-kernel-path-v1` (OBS-15, #4574).

**Scope:** aprender-serve emits these fields in the forward trace:
- `kernel_path` entries `{op, kernel_id, qtype, layout, arch, shape_class, precision}` on every backend, CPU included. This is stricter than OBS `trace_cut`, which exempts CPU rows; BPM-012 needs CPU entries.
- per-op `op_placement` over the full DECODE_OPS set.
- `qtype_path` per qtype: device-kernel, host-widen or refused-cpu.
- `head_dim_source`: key_length or fallback.
- a wgpu forward banner line, the counterpart of the CUDA `Backend: GPU (…)` line.
- `precision` in each `kernel_path` entry names the activation path (f32, q8k or q8_0), including the per-call crushed-block f32 switch. The receipt derives `act_path_by_qtype` from it (BPM-008, f008d), and the forward trace line names the forward that ran.

**Acceptance**
- A schema test asserts every field is present on a CPU run of a tiny fixture model.
- `op_placement` keys equal DECODE_OPS, and `kernel_path` is non-null on CPU.
- **Mutation:** drop the `attention` key; WGF-009's planted receipt in P1 must then be refused.

**Hosts** x86 CI (CPU forward). The wgpu values are checked later, in P4.
**Out of scope** New kernels; changing the OBS-15 contract (flag the CPU-row conflict to its owner instead).

---

## P3 — NEON Q4_K / Q6_K dots for aarch64 (C4)
**Refs** #3999 (C4 E1/E2). Contract: `neon-q4k-q6k-v1` (draft). Skeletons: `docs/lookahead/0.73/R3-test-skeletons.md`.

**Scope**
- NEON arms in the three live dispatchers: `fused_k.rs:193`, `fused_q5k_q6k.rs:118`, and `q4k_dot_avx2.rs:338` (compiled through `include!` at `fused_k.rs:370`).
- `kernel_path(k)` maps to OBS `kernel_id`.
- A widen entry exposed only to tests.
- The orphan `quantize/fused_q4k.rs` is not touched here; it has its own PROPOSE-TICKET (07:49Z).

**Acceptance**
- NEON-Q4K-001..004 and 007 are green on gx10, with 10 000 cases each, f16-normal d/dmin, and rows of 1, 2, 3 and 7 blocks.
- The magnitude-floor test is green on x86. NEON-Q4K-002 asserts on both dotprod branches.
- The golden file is checked against a pinned sha256.
- NEON-Q4K-000 and 006 (the cross-target `cargo check` and the compile-site probe with the orphan as control) run off the release path.
- Every mutation listed in the contract turns its test red.

**Hosts** gx10 (aarch64). x86 for the magnitude floor, 000 and 006.
**Out of scope** The C4 E1 measurement (M run, NEON-Q4K-005) and E2 speed.

---

## P4 — wgpu forward correctness (C1–C3)
**Refs** #3999 (C1–C3 E1). Contract: `wgpu-forward-v1` (draft). Scope doc: `R2-wgpu-forward-scope.md`.

**Scope**
- rope_theta from metadata in both WGSL RoPE shaders (`wgsl_forward.rs:242` and `:278` at 316dee2cd4).
- head_dim from `key_length`.
- Q6_K, Q8_0 and Q4_0 WGSL GEMVs, so the exit model's Q6_K tensors stop being widened on the host.
- Qwen3 q/k norm, only after the K14 layer-diff trace has attributed the 0.955 gap.

**Acceptance**
- WGF-001, 002, 004 and 008 are green on C1, using synthetic fixtures at position ≥ 1 and pos_in_head ≥ 1.
- WGF-003 is green only after the K14 trace has run.
- The mutations listed in the contract turn their tests red.

**Hosts** C1 (wgpu adapter), train-inactive. CI has no GPU.
**Out of scope**
- Moving attention, RoPE and the LM head onto the device (R2 item 5).
- The gated-delta route (R2 item 6, which may slip to 0.74).
- The ledger sweep (M run: WGF-006, 007 and 010).

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

---

## Status (2026-10-03 16:46Z)
- origin/main is 316dee2cd4. All file:line cites in R1–R5 and these bodies were re-checked there at 13:18Z (ac564391ab).
- **Filing:** still under S-1 hold (operator C292): nothing is filed or opened until LIVE 0.70.1. After LIVE, 0.73 moves from the floor to normal cadence: P1..P5 go to the cop as PROPOSE-TICKET lines, and this branch gets its PR.
- **Open rulings, not blocking (C293.3):** RQ-3, RQ-4 and RQ-5 are requested in the handoff. Work proceeds on provisional S-4 defaults: RQ-3 = E1 PASS + E2 PASS receipts on main; RQ-4 = a hybrid may pass E1, and E2/E6 name it; RQ-5 = `fp32_act`.
- **External blocker:** the OBS stack, #4487 and #4574, is unmerged, so P1's identity check stays NotRun until it merges.
