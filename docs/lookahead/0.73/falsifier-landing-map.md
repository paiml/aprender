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
| **P3 R3 NEON kernels** | Q4_K/Q6_K NEON dots, `kernel_path`, the widen test entry, the matvec row test | gx10, by hand: no PR job runs aprender-serve tests on aarch64 (R3 §16); 000/006 cross-check on x86 | R3 mint |
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

## Ranking
This is the one ranking for 0.73. The cop mints from it, and the PROPOSE-TICKET lines go out in this order. The
research rows R1..R5 (2026-09-27) are the evidence behind it, not a second list. The From column maps them.

| Rank | Bundle | From | Gate falsifiers | Why this rank |
|---|---|---|---|---|
| 1 | P1 receipt checker | R1, its checker | 21 | The cheapest falsifier in 0.73: x86 CI, no model. Every other bundle's receipts, and every M run, are judged by it. |
| 2 | P2 trace fields | R1, the fields its receipts read | 0. BPM-006, BPM-012, BPM-016, WGF-009 and WGF-004 need its fields | CPU only. Until it lands, those falsifiers are checked on planted fields only. |
| 3 | P4 wgpu fixes | R2 | 5 | It goes after a measured failure: wgpu decode reaches cosine 0.955 on every GPU measured, below the 0.995 floor of an E1 leg (F-R5-3). P5's wgpu half waits on it. |
| 4 | P3 NEON kernels | R3 | 8 | On aarch64 the gap is speed, plus a trace that says NEON while scalar code runs (F-R5-1). The output is already right. It runs in parallel with P4. |
| 5 | P5 MoE dispatch | R4 | 2, each with its M run | The CUDA MoE forward exists; the gap is wiring. Its wgpu half follows P4. |

- R1 was one row: the harness, its cells and a speed receipt. Minted as one ticket, it would put P1's falsifiers
  behind the GPU cells again. So it is split: P1 and P2 land first on CPU, and its GPU cells are M runs, last.
- R5, the kernel-key census (`R5-kernel-key-census.md`), is done and lands no bundle. Its one ask, a backend field
  in the KREG-001 key (#4539, train 0.71), is in the KREG draft: `kreg:backend` at `contracts/kernel-registry-v1.yaml:86`
  on branch kreg/4539-parity-receipts at 02b0f7f7fc. It is not on main at 316dee2cd4.
- M is not ranked. It is runs, not code; the Count line above gives its falsifiers.
- S1..S3 (`ticket-bodies-side-fixes.md`) are not 0.73 gates. Their lines go out after P5.

## Ranking, rows 6 to 20
Rows 6 to 20 continue the one ranking above; they are not a second list (APR-LOOKAHEAD-001 §4, L2 item 2: the
top 20 ranked). They are not bundles. None has a spec, a contract or a gate falsifier yet, so the gate total above
is unchanged, and no PROPOSE-TICKET line goes out for them; L2 specifies them in this order. Each row names the
exit criteria it serves (APR-LOOKAHEAD-001 §2a). Code paths are relative to `crates/` and were read at 316dee2cd4
unless the row says otherwise. M and S1..S3 stay out of the ranking, as above, and so does F3, the #4575
backend-label fix already written (`F3-pr-draft.md`).

Ranks 1 to 5 serve: P1 E1, E2; P2 E1, E2; P3 E1, E2; P4 E1, E4; P5 E3, E4.

So no bundle serves E5 or E6, and none runs Qwen3.5-4B, E1's model, on wgpu (row 6).

| Rank | Row | Serves | Evidence | Why this rank |
|---|---|---|---|---|
| 6 | qwen35 on wgpu: a wgpu forward in `Qwen35Session`, with the gated-delta WGSL route (R2 item 6) | E1, E4 | `aprender-serve/src/infer/inference_result.rs:365-385` and `apr-cli/src/commands/serve/server.rs:172` load the session on CUDA or on the CPU, never on wgpu. A forced wgpu run is refused with exit 14 (`apr-cli/src/commands/run_entry.rs:331`), and so is a forced serve (`server.rs:180`). The gated-delta step has no wgpu arm (`aprender-serve/src/gguf/inference/forward/forward_qwen35.rs:1341`). | P4 repairs the dense wgpu path only, so P1..P5 leave E1's WGPU and Metal legs refused. It is the long pole and reuses P4's WGSL GEMV, so it starts when P4 lands. RQ-8 asks whether it goes above P5. |
| 7 | OBS-18 (#4575): `gpu_proof` for Metal and wgpu, a device-kernel trace line | E1, E6 | Defined in `OBS-18-gpu-proof-metal-wgpu.md`, read from origin/main on 2026-09-28. Its finding F3, a hardcoded backend label, has a fix written on la-73/4575-wgpu-backend-label @fd02954a8e. | Already minted. A WGPU or Metal leg counts only with this proof; without it a CPU fallback reads as a GPU run. Every wgpu and Metal M run waits on it. |
| 8 | Kernel-registry coverage: register every kernel key the E1 backends dispatch, and a falsifier that fails on a dispatched key the registry lacks | E5 | No Rust kernel registry has both a backend and a qtype dimension (F-R5-5). The field is `kreg:backend` at `contracts/kernel-registry-v1.yaml:86` on kreg/4539-parity-receipts @02b0f7f7fc (#4539, train 0.71); that file is not on main. R5's census is the key list. | No bundle serves E5. It starts when #4539 is on main. |
| 9 | E4 census: every refusal and CPU fallback a WGPU or Metal run can hit, each with a falsifier that it is gone | E4 | Today: Q8_0 and Q4_0 are refused for wgpu dequant and run on the CPU (`aprender-serve/src/gpu/adapters/wgpu_adapter.rs:339`, F-R5-2); the wgpu MoE forward is a stub (`aprender-serve/src/gguf/wgpu_backend/mod.rs:196`); qwen35 (row 6); sampled requests (row 17). | E4 says the refusals are removed, but nothing lists them, so E4 cannot be checked. Reading only, no GPU. P4 and P5 then strike rows off it. |
| 10 | The `apr serve` half of E1: a serve parity cell per backend, greedy, judged like the run cell | E1 | R1's cells are all `apr run`: a case-insensitive grep of R1 for serve finds only `aprender-serve`. Serve loads the same `Qwen35Session` as run (`server.rs:172`), so the run oracle carries over. | E1 names `apr run` and `apr serve`, and the measurement plan covers only run: a gap in this slot's own plan. Cheap once the run cell exists. |
| 11 | llama.cpp at the ruled pin d1d3c3396 on each E1 host, built for that host's backend, with its sha in every receipt | E1, E2 | R1 line 13 names the comparator (RQ-2), and R1 F4 fails a receipt whose llama.cpp sha is not the pin. Which hosts have that build today is [U]. | Host setup, not aprender code. Every E1 cosine and E2 ratio needs it, but only before the first M run, and M runs come last. |
| 12 | wgpu device residency: attention, RoPE, the LM head and argmax on the device (R2 item 5) | E2, E6 | `aprender-compute/src/backends/gpu/device/linalg/wgsl_forward.rs:751` (embedding and LM head on the CPU), `:796` (LM head CPU matmul), `:876-877` (Q, K and V read back, attention on the CPU). | Until it lands every wgpu cell is hybrid (R1 F7), which RQ-4 rules on for E6, and the per-layer readback bounds E2 speed [U until measured]. Large, and it follows P4. |
| 13 | intel's AMD GPUs under wgpu: a working Vulkan driver and a pinned adapter choice | E1, E6 | The #3757 measurement quoted in the #3827 comment lists intel as "Vulkan, no working driver" (`aprender-serve/src/infer/gguf_gpu_generate.rs:108`). The driver state today, and whether E1 means one adapter or both, are [U]. | Host setup. Without it neither E1's intel leg nor E6's C1 intel-wgpu cell (`contracts/rex-cell-admission-v1.yaml:6`) can be measured. |
| 14 | NEON for the qwen35 CPU forward's gated-delta step | E2 | `delta_rule_head` (`forward_qwen35.rs:280`) is plain loops with no explicit SIMD, and R5 lists gated-delta as scalar on x86 and aarch64. Whether the compiler vectorizes it, and its share of decode time, are [U]. | P3 covers the Q4_K and Q6_K GEMVs only, and E2 on aarch64 is measured on Qwen3.5-4B. The rank holds until a profile on gx10 says how much time the step takes. |
| 15 | A PR-CI step that runs P3's parity tests on aarch64 (R3 §16, option (a)) | E1 | No PR job runs aprender-serve tests on aarch64 (R3 §16). | Without it P3's falsifiers gate nothing once they land. It edits `.github/workflows/ci.yml`, which needs an operator check-in (RQ-7). |
| 16 | E6 admission: `rex admit` admits C1 intel-wgpu and C5b mini-metal once their E1 and E2 receipts pass | E6 | `contracts/rex-cell-admission-v1.yaml:6-7` names both cells, and `rex admit` writes one row per cell (`aprender-review-experiment/src/admission.rs`). | It runs once those hosts have passing E1 and E2 receipts, so it ranks below the rows that produce them. RQ-3 sets which receipts count (Open). |
| 17 | Sampling on the wgpu decoder (#3760) | E4 | The wgpu decoders are greedy-only: a sampled request runs on the CPU with a notice (`gguf_gpu_generate.rs:62`, `:68`). | E1's cells are greedy, so it blocks no E1 leg. It is a fallback the E4 census lists (row 9). |
| 18 | wgpu parity in PR CI on a software adapter | E1, E4 | P4's tests run on C1 only: "CI has no GPU" (`ticket-bodies-P1-P5.md:94`). #3757 measured cosine 0.955 on every GPU it tried, intel among them with no working Vulkan driver (`gguf_gpu_generate.rs:108`), so the defect may reproduce with no GPU [U]. | The same gap as row 15, for P4. A CI change, so an operator check-in. Whether a software Vulkan adapter runs on the CI runners is [U]. |
| 19 | A Q5_K WGSL GEMV | E2, E4 | wgpu dequantizes Q5_K to F32 on the host (`wgpu_adapter.rs:312`, R5). P4 item 4 adds Q6_K, Q8_0 and Q4_0 only. | Whether Qwen3.5-4B's GGUF holds Q5_K tensors is [U]. If it does, this row moves up. |
| 20 | Batched MoE prefill (K18) | E3 | Out of P5's scope (`ticket-bodies-P1-P5.md:114`), a candidate R4 item 5. | E3 asks for a run with parity, not speed, so it comes last. It matters for long MoE prompts. |

## Order
1. P1, then P2. They are CPU-only and unblock every measurement. P1 is the cheapest falsifier in 0.73, and it is
   where the L25 fixes take effect.
2. P4 and P3 in parallel: different crates and different hosts. If only one can be staffed, P4 goes first (Ranking).
3. P5 after P4 for the wgpu MoE part; the streaming part can follow P1.
4. M runs only when the host is train-inactive, each with its receipt checked by P1.
5. Rows 6 to 20 in rank order, once L2 specifies them. Row 6 starts when P4 lands.

## Side fixes (not 0.73 gates)
Three falsifiers sit outside the 41. They come from the draft amendment `contracts-draft/cpu-q4k-activation-quant-v1.yaml` (1.1.0) and land with their tickets (`ticket-bodies-side-fixes.md`), not with P1..P5:
- FALSIFY-AQ-005 → S2: the multirow entry follows the fp32 scope.
- FALSIFY-AQ-006 and FALSIFY-AQ-007 → S1: the fused gate+up entry follows the selector, and its crushed fallback is f32 and noted.

All three are x86 lib tests: no GPU, no model, no aarch64 host. S3 has no contract falsifier; its acceptance is one-shot checks recorded in its PR body.

## Open
- RQ-5 is ruled (cop, 2026-09-27 20:12Z): BPM-008 pins `cpu_ref_path` = `fp32_act`, held as data in `bpm.cpu_ref_path`. A `q8k_act` run is an info row only. Item (e): BPM-018 keeps info rows out of the verdict, and NEON-009 judges the gx10 default-route row; both are P1 rows. Whether that route should gate is RQ-6 (handoff).
- RQ-3 decides the E6 admissible cell. It affects no bundle row, only row 16, which admits the cells it names. Provisional default: E1 PASS + E2 PASS receipts on main, with PRM-001 agreeing.
- RQ-4 decides whether a hybrid cell counts for E6. Provisional default: E1 may pass hybrid, and E2/E6 name it. It affects no bundle row (f016 asserts only the label). For row 12 it decides whether E6 needs device residency.
- RQ-7 (handoff, item (k)) asks where P3's aarch64 PR-CI step goes (row 15). Without a ruling, P3's tests run by hand on gx10.
- RQ-8 (handoff, item (l)) asks whether row 6, qwen35 on wgpu, goes above P5. Without a ruling it stays at 6, and E1's WGPU and Metal legs stay open until it lands, which may be in 0.74.
