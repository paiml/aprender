# impl receipt — PMAT-1095 — 0.67 CUDA Rust fleet readiness

## Identity
- ticket: PMAT-1095 (kind:code, orch:fable, orch-basis:release) · github: #3067, epic #3062 · PR #3068
- branch: PMAT-1095-cuda-rust-fleet · HEAD at receipt time: b225adc74 · base: origin/main
- discover.json sha256[:16]: 4535d17a7372eec3 · repo_root: /home/noah/src/aprender-worktrees/PMAT-1095
- spec: docs/specifications/nvidia-cuda-rust-library-integration.md (#3061)
- status-line join: session_id = hook session_id (measured true, pid-file rule); tasks[].id = agent_id and transcript_path: not measured; k_measured: not measured by statusline.sh this session (declared `?` in every block).

## Plan and routing
| phase | scope | route (route.sh, verbatim) | trigger | acceptance |
|---|---|---|---|---|
| P1 | driver/memory_fuzz_tests/adversarial.rs | route=agy-goal w=1.00 basis=quota.json@43h note=fable-binding effort=1[U] | Q1 (three modules) | cargo test -p aprender-gpu --features cuda --lib test_alloc_oversize_100gb |
| P2 | driver/cublas_tests.rs | route=agy-goal w=1.00 basis=quota.json@43h note=fable-binding effort=1[U] | Q1 | cargo test -p aprender-gpu --features cuda --lib test_cublas_gemm_f16_training_shape |
| P3 | scripts/cuda_rust_fleet_check.sh, experiments/cuda-rust-probes | route=agy-goal w=1.00 basis=quota.json@43h note=fable-binding effort=1[U] | Q1 | bash scripts/cuda_rust_fleet_check.sh --self-test |
| P4 | receipts, PR | route=self w=11.11 basis=quota.json@43h | - | make gate |

## Dispatch ledger (Claude subagents: all paiml-agy-delegate, opus, foreground; one at a time)
| phase | description prefix | lane | width | outcome | agy conversations | children |
|---|---|---|---|---|---|---|
| ph1 | PMAT-1095/ph1.delegate | goal, writes=true | 1 | PARTIAL: maxTurns(30) spent rebuilding the lane harness for a worktree .git file; no edit; left a stray registered worktree at .claude/worktrees/lane-a9243905d-2793474 (323 MB) that later failed make gate. Fallback: direct, per R-4. | none recorded | 0 |
| ph3 | PMAT-1095/ph3.delegate | quorum, review | 3 | PARTIAL: maxTurns before the receipt; lane files complete: 3/3 do-not-implement-as-written, 8 findings | in lane files | 3 |
| ph3 | PMAT-1095/ph3.delegate-r2 | quorum, review | 3 | complete: 3/3 do-not-implement; F1 F3 F4 F5 F6 F7 FIXED, F2 PARTIAL, F8 NOT-FIXED, N1 NEW (delegate-verified fail-open) | 60a76d81-fc81-4be9-9541-b79331d0518e, db247b3b-ec31-45cc-9173-c8127b4efab2, e0507f4f-1e06-4148-b945-433c6de41191 | 3 |
| ph3 | PMAT-1095/ph3.delegate-r3 | quorum, review | 3 | complete: 3/3 implement-as-written, lane-reduce agreed=true, dissent=[] | 5320758e-58cf-4104-b719-6e4181d26f9b, 4b295568-659d-488c-9504-cb11d20abac6, 6e8d5f59-2cf6-46b9-bacf-9f903270f8f9 | 3 |

slots used: peak 1 of 3 · denials: 0 · I-3 (transcript-gate.sh): attempted=0 denied=0 running_peak=0 slots=3 — VACUOUS: it scanned this worktree's project dir while the four dispatches ran under the pp-066-spec session dir; the ledger above is the measured record (attempted 4, denied 0, running_peak 1). No agy lane wrote to the repository.

## Verification (claimed vs re-run)
| check | re-run result | exit |
|---|---|---|
| A_1 lambda-vector (RTX 4090) | ok. 1 passed | 0 |
| A_1 gx10 (GB10, systemd-run MemoryMax=48G) | ok. 1 passed, 0.08 s, MemAvailable unchanged | 0 |
| A_2 lambda-vector | ok. 1 passed (165.3 TFLOP/s reported, not asserted) | 0 |
| A_2 gx10 | ok (inside the fleet probe: 4 passed) | 0 |
| --self-test | SELF-TEST PASS (verdict table, receipt writer, mem_floor_ok vectors) | 0 |
| fleet gx10 / yoga | PASS 6/6 / PASS 6/6 | 0 / 0 |
| fleet lambda-vector | INCOMPLETE; blocked_on = one root entry (driver 570.207 < R580; reboot activates the staged R580+ package) | 2 |
| bashrs lint scripts/cuda_rust_fleet_check.sh | 0 error(s) (lanes could not prove this; re-run here) | 1 (warnings) |
| cargo fmt --all -- --check | clean | 0 |
| clippy -p aprender-gpu --features cuda -- -D warnings | clean | 0 |
| check_roadmap_diff_additive.sh | PASS (+25) | 0 |
| kind-gate --base main | kind=code | 0 |
| gate_cmd make gate | 39 checks, 0 failed (first run: 1 failed on the delegate's stray worktree; removed) | 0 |
| pv contract same PR | NotRun: no contract changed in this ticket (register_budget landed in #3064) | - |
| mutation observed RED | mem_floor_ok vectors refuse '' x 0 5 11; the receipt-writer self-test caught the bash-array-in-env-prefix defect on its first run | - |

## Jidoka log
- gx10 global OOM (13:36) and reboot (13:43), 2026-09-09: the first fleet run's cold cargo build plus P1 v2's 2x-physical cuMemAlloc on a unified-memory part. Owned. Fix: P1 v3 (2^60 on UnifiedMemory, validation-time rejection) and the runner's resource guard (CI-job check twice, 12 GiB floor fail-closed, nice, capped jobs, MemoryMax=min(48G, 60% MemAvailable)). Verified inside a cgroup.
- gx10 after reboot had no default route (DHCP profile autoconnect=false); restored with `nmcli con up enP7s7`, no configuration change.
- lambda-vector: driver 580 install replaced userspace libs under the running 570 module and broke CUDA; rolled back to the captured 570 set; 580 needs a reboot the operator owns.
- delegate ph1 left a registered stray worktree under .claude/worktrees; make gate scanned its stale roadmap copy; removed.
- yoga needed a full clang (bindgen resource dir); gx10 exposes only clang-21 to a non-interactive PATH; the runner discovers clang-N.
- an earlier receipt-writing script used an unquoted heredoc whose prose contained a backticked apt command; the substitution hung on sudo and produced an empty file. Nothing was installed (apt history checked). This receipt is written by python.
- quorum r1 (8 findings) -> r2 (3 residual + N1) -> r3 3/3 implement-as-written.

## Estimates
K_hat=4 (estimate.sh, first-run[U]); K=8; actual about 100 orchestrator turns. basis=first-run[U].

## Gaps
- lambda-vector cuda-core / cutile probes stay INCOMPLETE until the operator reboots with the R580+ driver (Lambda's own repo carries 580.119.02). The receipt names it.
- pr-review receipt (evidence/pr-review/<n>/, the `present` check): NotRun for this and the other five PRs; not a required check (branch protection and rulesets: ci / gate, workspace-test).
- transcript-gate is vacuous for this worktree (I-3 line above).
- cuda-nightly.yml:242 still runs only the perf053 filter; #3067 owns that.

## Verdict
DONE, merge pending required checks; the lambda-vector row is PARTIAL(blocker: operator reboot) and named as such.
