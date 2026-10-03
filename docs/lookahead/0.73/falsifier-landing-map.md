# 0.73 L3 falsifier landing map (draft, la-73, 2026-10-03)

There are 41 falsifiers not yet written: 18 in backend-parity-matrix-v1 (BPM), 10 in neon-q4k-q6k-v1 (NEON),
10 in wgpu-forward-v1 (WGF) and 3 in R4-moe-gpu-wiring.md (R4). The contract ones carry `test: NOT YET WRITTEN`.
R4 has no contract, so its three exist only as lines in that note. This map assigns each one to one landing bundle.
`count_audit.py` derives every count stated here and in the P1..P5 drafts from its source, and checks each one.
The bundles are ticket PROPOSALS for the mint after LIVE 0.70.1. Nothing here is minted (C277).

**Main finding:** 21 of the 41 need no GPU, no model and no aarch64 host. They are checks on planted receipt JSON,
so they can land first and run on x86 CI. The contracts said "lands with the R1 harness" for several of them. That
blocked them on the harness and GPU access for no reason. Only the checker is needed.

## Bundles

| Bundle | What lands | Runs where | Blocked on |
|---|---|---|---|
| **P1 receipt checker** | a pure function `check_receipt(json) -> Verdict` plus planted receipts under `tests/fixtures/bpm/` | x86 CI, no model | nothing (first to land) |
| **P2 trace fields** | aprender-serve trace emits `kernel_path`, the per-op `backend`, a total `op_placement`, `device_qtype`, `head_dim_source`, and a wgpu MoE banner | x86 CI (schema test) | nothing for the fields; their values need P3/P4 |
| **P3 R3 NEON kernels** | Q4_K/Q6_K NEON dots, `kernel_path`, the widen test entry, the matvec row test | aarch64 CI or gx10; 000/006 cross-check on x86 | R3 mint |
| **P4 R2 wgpu fixes** | theta and head_dim from metadata, q/k norm, Q6_K/Q8_0/Q4_0 WGSL | C1 (wgpu adapter); CI has no GPU | R2 mint; WGF-003 also on the K14 trace |
| **P5 R4 MoE dispatch** | the streaming variant of the dispatch; the wgpu MoE forward (R4 item 2) | C0 for streaming; C1-C3 for wgpu | R4 mint; item 2 after P4 |
| **M measurement runs** | receipts only, no code | GPU hosts, train-inactive | P1 + the bundle that is measured |

## Assignment

| Falsifier | Bundle | Planted input (P1) or the run it needs |
|---|---|---|
| BPM-001 per-prompt min | P1 | leg_a median 0.999, min 0.990 |
| BPM-002 composed bound | P1 | leg_a = leg_b = 0.99 |
| BPM-003 fallback refused | P1 | adapter llvmpipe, leg_a 1.0 |
| BPM-004 comparator pin | P1 | llamacpp_sha 39173bcac |
| BPM-005 CI lower bound | P1 | r 0.55, ci_lo 0.45 |
| BPM-006 no wgpu forward line = NotRun | P1 + P2 | Q8_0 receipt on a real adapter without a wgpu forward line |
| BPM-007 today's gap is visible | M (C1) | the harness on current wgpu Qwen3; expects < 0.995 |
| BPM-008 shared, pinned cpu_ref_path | P1 | mixed legs (f008a/b); equal legs off the pin (f008c); a label the trace contradicts (f008d). Pin: fp32_act (RQ-5, ruled) |
| BPM-009 non-finite cosine | P1 | one NaN cosine among 16 prompts |
| BPM-010/011 same inputs | P1 | prompt_set_sha256, n or model_sha256 mismatch |
| BPM-012 CPU cell not self-compared | P1 + P2 | f012 (one equal tensor among differing ones), f012c (one kernel_id on two arches), f012d (OBS slot collision), f012e (a tensor with two routes); control f012b; value from P3 |
| BPM-013 decode positions | P1 | prefill 1.0, decode pos 4 at 0.90 |
| BPM-014 unlisted software adapter | P1 | "FooSoft Renderer", device_type Cpu |
| BPM-015 equal n_gen | P1 | 37 vs 128 decoded tokens |
| BPM-016 per-op HYBRID | P1 + P2 | Q8_0 trace with CPU GEMVs on a real adapter |
| BPM-017 judged F2 guard | P1 | stderr with SKIP_PARITY_GATE=1; with "nothing was judged"; without the "GPU matches" line |
| BPM-018 info row never gates | P1 | f018a: a failing leg next to a 0.999 info row; f018b: a 0.985 or "NaN" info row on a Pass receipt |
| WGF-005 hybrid is derived | P1 | host op with hybrid: false |
| WGF-009 total op_placement | P1 + P2 | map without attention |
| R4-003 `--no-gpu` refused | P1 | plus a positive control: a real C0 GPU receipt is ACCEPTED |
| NEON-000 cross-check | P3 | `cargo check --target aarch64-unknown-linux-gnu`, off the release path |
| NEON-001 parity proptest | P3 | gx10; the masked-sweep power guard (scalar and bound only) runs on x86 |
| NEON-002 path matches dotprod | P3 | gx10 |
| NEON-003 pinned golden | P3 | golden recorded on x86 scalar, checked on gx10 |
| NEON-004 Q6_K parity | P3 | gx10 |
| NEON-005 C4 E1 leg | M (C4) | P1 + P3 |
| NEON-006 compile-site probe | P3 | x86 with the aarch64 target, scratch tree |
| NEON-007 widen entry | P3 | gx10 |
| NEON-008 matvec rows, decode and prefill | P3 | gx10 |
| NEON-009 default-route info row | P1 | fN9a..d: unproven route, scalar label on aarch64, unbound f0, broken triangle |
| WGF-001 theta | P4 | synthetic fixture on C1 |
| WGF-002 head_dim | P4 | synthetic fixture on C1 |
| WGF-003 q/k norm | P4 | after the K14 layer-diff trace |
| WGF-004 no silent widen | P4 + P2 | device_qtype in the receipt |
| WGF-008 required qtypes | P4 | Q6_K must be served |
| WGF-006/007/010 ledger sweep | M (C1) | base and head in one job; one discriminating model per equation |
| R4-001 stream = non-stream | P5 + M (C0) | `used_gpu = true` and the banner on both runs |
| R4-002 wgpu MoE leg A | P5 + M (C1-C3) | after P4 and R4 item 2 |

Count (41): 21 in P1 (BPM-001..006, 008..018, NEON-009, WGF-005, WGF-009, R4-003). BPM-006, 012, 016 and WGF-009 also need P2
for the real field, but their planted receipts carry it already. Then 8 in P3, 5 in P4 (WGF-003 gated on K14),
5 in M alone (BPM-007, NEON-005, WGF-006/007/010) and 2 in P5 + M (R4-001, R4-002).

## Order
1. P1, then P2. They are CPU-only and unblock every measurement. P1 is the cheapest falsifier in 0.73, and it is
   where the L25 fixes take effect.
2. P3 and P4 in parallel: different crates and different hosts.
3. P5 after P4 for the wgpu MoE part; the streaming part can follow P1.
4. M runs only when the host is train-inactive, each with its receipt checked by P1.

## Side fixes (not 0.73 gates)
Three falsifiers sit outside the 41. They come from the draft amendment `contracts-draft/cpu-q4k-activation-quant-v1.yaml` (1.1.0) and land with their tickets (`ticket-bodies-side-fixes.md`), not with P1..P5:
- FALSIFY-AQ-005 → S2: the multirow entry follows the fp32 scope.
- FALSIFY-AQ-006 and FALSIFY-AQ-007 → S1: the fused gate+up entry follows the selector, and its crushed fallback is f32 and noted.

All three are x86 lib tests: no GPU, no model, no aarch64 host. S3 has no contract falsifier; its acceptance is one-shot checks recorded in its PR body.

## Open
- RQ-5 is ruled (cop, 2026-09-27 20:12Z): BPM-008 pins `cpu_ref_path` = `fp32_act`, held as data in `bpm.cpu_ref_path`. A `q8k_act` run is an info row only. Item (e): BPM-018 keeps info rows out of the verdict, and NEON-009 judges the gx10 default-route row; both are P1 rows. Whether that route should gate is RQ-6 (handoff).
- RQ-3 decides the E6 admissible cell; it affects no row here. Provisional default: E1 PASS + E2 PASS receipts on main, with PRM-001 agreeing.
- RQ-4 decides whether a hybrid cell counts for E6. Provisional default: E1 may pass hybrid, and E2/E6 name it. It affects no row here (f016 asserts only the label).
