# impl receipt — PMAT-3477 (EPIC #3477, 0.68.2 interrupt train)

## Identity
- ticket: PMAT-3477 · kind: code · branch: `PMAT-3477-qwen3-cuda-qk-norm` · PR #3484
- discover.json sha256: see `$XDG_RUNTIME_DIR/paiml-implement/<session>/3635291372/discover.json` (repo_root `/mnt/nvme-raid0/agent-wt/rel-0682`, gate_cmd `make gate`, required_check `ci / gate,workspace-test`, quorum_tool agy)
- orchestrator model: fable-5-1 (model-gate admit, basis=file; `orch:fable` + `orch-basis:release` on the roadmap entry)
- status-line join: statusLine session_id = hook session_id — not measured (no statusline transcript check run) `[U]`; `k_measured` = 6558 distinct assistant message ids in the session transcript (jq over the session jsonl) — this session predates the ticket, so the ticket's own turn count is not separable from it; `k` for the ticket is unmeasured `[U]`.

## Plan and routing (per phase)
| phase | what | route (route.sh, verbatim) | executed as | trigger |
|---|---|---|---|---|
| 1 | #3413 A: record `per_head_rmsnorm_into` into the manual CUDA graph | `route=agy-goal w=1.00 basis=absent note=fable-binding effort=1[U]` | direct (8-line fix fully specified by the RED test; the one writes=true agy slot was kept free) | Q1 (|M|≥3), Q2 plan grill |
| 2 | #3413 B: batched per-head RMSNorm in `batched_qkv_rope_phase` | same | subagent:opus worker B (2 crates) | — |
| 3 | #3413 C: F2 guard judges the resolved prefill path | same | subagent:opus worker D | — |
| 4 | #3432: one GGML type table; `apr qa` admits qwen35 on CPU; qa certifies CPU-only arch | same | subagent:opus workers C, F | — |
| 5 | #3091: IQ2_XXS/IQ4_XS CPU dequant, real files | same | subagent:opus worker E | — |
| 6 | #3090 disposition | — | direct (issue comment) | — |
| 7 | pre-PR review quorum; two-host ladder receipts | `route=agy-quorum w=1.00 basis=absent effort=1[U]` | delegate (agy, width 3) | Phase-4 review |
| 8 | train T-0…T-4 publish | route=self | pending | — |

Plan grill (phase 1, delegate, agy `--mode grillme`, width 3, 2 countable lanes; lane 2 BLIND/void): PASS-WITH-CHANGES — CtaIdY sequence offset in the PTX, guard must run the real batched prefill, keep `k_quant_bytes` row padding, ptx_map/parity_refusal keep refusing qwen35, P1 before P2. All applied.

## Dispatch ledger
| dispatch | mode | agent id | maxTurns hit | resumed | outcome |
|---|---|---|---|---|---|
| ph1 delegate grill | agy quorum w=3 | a365eeab5c4e1bee3 | yes (30) | once | receipt; conversations 29175450…, 7e9fdb7c…, 734ca55d…; child_conversations=3 |
| ph2 worker B | opus | ae2119108bc652da9 | yes (40) | once | e296147d8 |
| ph4 worker C | opus | a33dd8ca7410a3623 | yes (40) | once | 6181a30db |
| ph3 worker D | opus | a79b0d5fb7305de93 | yes (40) | once | e4fc96bd0 |
| ph5 worker E | opus | a812f7fcab60266aa | yes (40) | once (cut again; had committed) | 530b6d4dc |
| ph4.qa worker F | opus | ab5985a5b1bff400b | yes (40) | once (cut again; had committed) | 56a347cf7 |
| ph7 delegate review | agy quorum w=3 | a8b47a1a337ff1a7e | yes (30) | once | 3/3 PASS; conversations 054b523d…, 0b73af5e…, 2519edfb…; child_conversations=3 |

Slots: peak 2 Claude subagents concurrently (workers B+C, D+E) + 0–1 delegate; never 4. Denials: 0. Stalls: 0. I-3 line: `transcript-gate: attempted=0 denied=0 stalled=0 running_peak=0 slots=3` — **vacuous**: the gate scanned the worktree's project dir (`-mnt-nvme-raid0-agent-wt-rel-0682`) while the subagent transcripts live under the launch dir's project (`-home-noah-src-aprender`), so it saw none of the 7 dispatches above (memory: transcript-gate from a linked worktree is vacuous, PMAT-3429). Counted by hand from the Agent results instead.

## Verification (claimed vs re-run by the orchestrator)
| acceptance | worker claim | orchestrator re-run |
|---|---|---|
| A1 `cargo test -p aprender-serve --lib --features cuda test_3413_per_head_rmsnorm_is_recorded` | RED on 4a538ddef (0 recorded) | RED then GREEN (1 passed) |
| A2 aprender-gpu per_head_rmsnorm (cuda) + serve `3413` filter | 14 passed; mutant RED on device | 14 passed; 3 passed |
| A3 `f2_` tests (cuda) | 4 passed | 8 passed (incl. FP16 re-measure tests) |
| A4 ggml_type_table / tensor_byte_size / qa_capability | 8/7/4 passed | 8/7/4 passed |
| A5 IQ unit tests + real files | (cut) | 40 passed; IQ4_XS → `**Paris**.`; UD-IQ2_XXS coherent; rc=0 both |
| `apr qa Qwen3.5-0.8B-Q4_K_M` | rc=0 claimed | rc=0, 13 PASS, Golden Output PASS on CPU, no NEITHER |
| `apr qa Qwen3-1.7B-Q4_K_M` (cuda) | — | Golden Output PASS, GPU Speedup 18.2×, PTX parity 6/6 |
| `apr run --gpu` Qwen3-8B | — | `The capital of France is Paris` (serial prefill, FP8 off, printed) |
| model ladder @927587272 | — | lambda: qwen2, qwen3-1.7b, qwen35 OK; qwen3-8b optional red (#3486). gx10 (peer session): 4/4 OK. `check_model_ladder.sh`: every required rung green on every required host |
| `make gate` | workers: red on 3 inherited guards | complexity/baseline ratchets FAIL (instrument) on this box only — recorded pmat 3.40.1 vs box 3.40.2; CI runs 3.40.1. roadmap-fragment guard: fixed (83f85a141). |

Gate stages measured by `pmat verify` (format, complexity, clippy): ok. not_measured: satd, tests (Makefile's own `--skip`).

## Measurements that changed the plan
- FP8 batched prefill fails CPU parity at the post-prompt decode step: Qwen3-8B cos −0.0972 in every FP8 config, "Paris" in every FP16/serial config; Qwen3-1.7B 0.87 on 2/4 prompts; qwen2.5 control 0.9187 (same argmax) on one run. → `GpuProfile::disable_fp8_for_qk_norm` (serial prefill + FP8 off for QK-norm models; FP16 batched OOMed 8B beside apr qa's CPU twin on 24 GB) and the F2 FP16 re-measure. Root cause → #3483.
- Qwen3-8B `apr qa` golden "Empty output": thinking budget consumed by `<think>` → #3486; rung stays optional.

## Jidoka
- ph1 grill lane 2: BLIND (no tree witness) + isolation violated by another session's ref move — void, not counted.
- ph7 review lanes 1–2: exit 3 from the orchestrator's own ladder receipt write during the review window — verdicts kept with that caveat (the only changed path was the receipt).
- CI guard-cargo RED: `matmul_fused.rs::fused_matmul` cognitive 36→41 / cyclomatic 17→20 (worker E's early-return) — fixed by moving the fallback into the existing default arm (see the commit after this receipt).
- CI guard-tree R-2: PR body cited issues without close/no-close reasons — body amended.
- CI guard-cargo "No claim literals": my 47-line insertion in `gpu_profile.rs` shifted two baselined doc-comment claims (356→403, 410→457) and a worker's doc comment stated tok/s measurements — fixed by moving the new fn into a trailing `impl GpuProfile` block (baseline untouched) and rewording the comment to the mechanism.
- CI guard-cargo "pmat/bashrs must match the pinned fleet versions": the fleet moved to pmat 3.41.1 under the repo pin 3.40.1 (PMAT-1363 convergence) — every PR red. Folded in: `tools.toml` pin → 3.41.1, `scripts/complexity_baseline.txt` re-recorded under 3.41.1 (676 → 668 rows, D2 PASS: none new, none grown), `scripts/cb200_baseline.txt` header restamped with CB-200 re-measured at 601 (= baseline). check_tool_versions 4/4, check_baseline_ratchets PASS.
- CI cuda-unit (yoga): `driver::cublas_tests::cta64_vs_cta32_vs_cublas_fp16` → `CUDA_ERROR_OUT_OF_MEMORY` allocating B; 2608/2609 pass there, 2609/2609 on lambda; the test file is untouched by this PR — environment (yoga's GPU shared with a concurrent job), not code. Not in `ci / gate`'s needs; rerun when the run completes.
- CI guard-tree §11.1: touching `scripts/cb200_baseline.txt` (a known-red list) makes this a sweep PR that must carry an `ont-delta:` line — added to the body (`none`: the instrument moved, no ontology entity changed).
- `ci / lint` (first runs): the folded #3485 contract carried `qwen35` with aliases that duplicated the existing `qwen3_5` entry, so the generated `match` had an unreachable arm under `-D warnings` — duplicate entry removed; `check_arch_constraints_contract_covers_fallback.sh` ok.

## Gaps (NotRun / open)
- `present` / `pr-review-quorum` (Arm 4 signed receipt at `evidence/pr-review/3484`): not produced — review backlog class, not a required check.
- IQ2_XXS grid table: `iq2_grid_bytes_are_only_the_three_magnitudes` cannot catch a transposed entry (review quorum finding); covered only by the one-block fixture and the coherent real-file output. Follow-up test owed.
- pv contract in the same PR: contracts co-evolution rides in via #3480's merged head (qk-norm-v1 FALSIFY-QKN-006, parity qa_gate, qwen3-e2e); `pv validate` not re-run by me (`pv_lane=NotRun`).
- #3090 GDN GPU: dated non-goal (issue comment 2026-09-18).

## Estimates
- K̂ = 8 (`estimate.sh aprender 8`, basis=first-run[U], ROWS=1 EXCLUDED=43 UNMEASURED=4), K = 16. Actual: unmeasurable per ticket (session predates it) `[U]`; ~8 h wall-clock, 7 dispatches.

## Verdict
PARTIAL(pending-merge): all code phases green and re-run; two-host ladder green; merge waits on CI (guard-cargo complexity fix + R-2 body re-read) and the T-4 publish is phase 8.
