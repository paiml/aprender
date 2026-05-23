# Hiatus Resume Checkpoint — 2026-05-22

**Purpose:** single source of truth for picking up MODEL-2 distillation work after the
3-month hiatus starting 2026-05-22. Written at the end of the PMAT-701 cascade
(8 PRs, 6 contracts, 3 spec amendments) so future-you doesn't have to reverse-
engineer state from commit history.

## TL;DR — what's stable, what's pending

**Stable (no operator action needed):**
- The full `apr distill --backend cuda` pipeline runs correctly on Grace Blackwell GB10.
  cuBLAS dispatches by default; teacher up to 7B Q4K is feasible. Per-step loss is visible
  via `APR_DISTILL_LOG_EVERY` (default 50). Smoke validation via `APR_DISTILL_MAX_STEPS=N`.
- `apr eval --task humaneval` no longer returns the structural-validation false-positive
  (pass@1=1.0 on broken models). Exit code 8 on inference failure.
- 6 provable contracts codify the invariants. All validate clean (`pv validate` 0/0).

**Pending operator action (compute):**
- Re-dispatch **Phase 4 Stage D 50K** with the corrected defaults. ~30-50 h on GB10.
  Wrapper: `scripts/dispatch-distill-stage-d.sh`. The prior 50K + 10K Stage D runs
  (2026-05-20/21) are discharged as no-KD per §86 — they trained against
  teacher==student, not real KD signal.
- After Stage D completes, run **Phase 5 HumanEval discharge**.
  Wrapper: `scripts/dispatch-distill-phase-5-humaneval.sh`. Target pass@1 ≥ 25%
  per AC-DISTILL-004.
- After Phase 5 passes, run **Phase 6 publish** to `paiml/albor-370m-v2` per
  SPEC-HF-PUBLISH-001.

## Resume in 4 commands

Assuming all PMAT-701 family PRs (see below) have merged:

```bash
# 1. Disk preflight (gx10 was at 98% — clean old runs)
./scripts/cleanup-gx10-runs.sh           # dry-run, lists candidates
./scripts/cleanup-gx10-runs.sh --apply   # actually delete

# 2. Smoke validation FIRST (60s — catches cascade-level bugs before 30h)
APR_DISTILL_MAX_STEPS=10 ./scripts/dispatch-distill-stage-d.sh

#    Verify the smoke log shows:
#      [PMAT-705] ProgressCallback attached (log every 50 steps)
#      [PMAT-706] smoke mode: APR_DISTILL_MAX_STEPS=10
#      [CUDA] cuBLAS initialized — forward TF32 tensor cores (41x vs SIMD)
#      Step 1/N: loss: ... Step 2/N: loss: ... ...
#      [SMOKE] 10 steps in T.Ts: initial_loss=X.XXXX, final_loss=Y.YYYY, throughput=Z.Z step/s
#      [SMOKE] projected full-run wall time (50000 steps): H.Hh / WW min / SSs
#    If projected wall time > 100h, the cascade has regressed — debug before Stage D.

# 3. Dispatch real Stage D (background, ~30-50 h)
./scripts/dispatch-distill-stage-d.sh

# 4. After Stage D completes, run Phase 5 HumanEval
CHECKPOINT=/home/noah/runs/distill-stage-d-<TIMESTAMP>/student-trained.apr/model.apr \
  ./scripts/dispatch-distill-phase-5-humaneval.sh
```

If pass@1 ≥ 25 %, proceed to Phase 6 publish per SPEC-DISTILL-001 Phase 6 +
SPEC-HF-PUBLISH-001. If pass@1 < 25 %, re-train with wider corpus per the
fallback in SPEC-DISTILL-001 §4.

## PMAT-701 cascade — what landed and why

The session (2026-05-22) closed a cascade of 6 defects in the Phase 4 dispatch pipeline.
Each fix shipped as its own PR with a provable contract.

| PR | Ticket | What it fixed | Contract |
|---|---|---|---|
| #1863 ✅ | PMAT-701 Bug A | trueno-gpu allocator defaulted to `cuMemAlloc` (~30 GB ceiling on GB10) instead of `cuMemAllocManaged` (full 128 GB unified pool). Autodetects Grace Blackwell via `CU_DEVICE_ATTRIBUTE_INTEGRATED`. | `contracts/trueno-gpu/cuda-unified-memory-allocator-v1.yaml` |
| #1869 ✅ | PMAT-701 Bug B | Routed Q4K teachers around `CudaTrainerTeacher` to a CPU-bound `RealizarQ4KTeacher`. **DEMOTED** by PMAT-704 (see #1879) — kept as memory-constrained-device fallback. | `contracts/cuda-q4k-frozen-teacher-v1.yaml` (status: demoted, not retracted) |
| #1871 ✅ | SPEC §86 | Phase 4 Stage D 50K + 10K runs discharged as no-KD due to teacher == student staging defect. Dispatch script's `TEACHER_REPO` default flipped to `paiml/qwen2.5-coder-7b-apache-q4k-v1`. | (spec amendment) |
| #1874 🟡 | PMAT-702 | `apr eval --task humaneval` no longer returns pass@1=1.0 false-positive on broken models. Exit code 8 + `mode: "inference_failed"` JSON. MBPP parity. | `contracts/apr-eval-humaneval-inference-failure-handling-v1.yaml` |
| #1877 🟡 | PMAT-703 | Teacher logits truncated to student vocab when `teacher_vocab > student_vocab` (7B has 152064, 0.5B has 151936). | `contracts/apr-distill-teacher-vocab-alignment-v1.yaml` (superseded by #1879's TruncatingTeacher; harmless duplicate) |
| #1879 🟡 | PMAT-704 | Bug B post-mortem: revert default to cuBLAS-backed `CudaTrainerTeacher`. `APR_DISTILL_TEACHER_BACKEND=realizar-q4k` opt-in fallback. `TruncatingTeacher` wrapper for vocab alignment. | `contracts/apr-distill-teacher-backend-selection-v1.yaml` |
| #1880 🟡 | SPEC §87 | Post-mortem amendment documenting the Bug B wrong turn. Methodology lesson: cheap-experiment-before-design. | (spec amendment) |
| #1881 🟡 | PMAT-705 | Wires `ProgressCallback` into the distill `Pipeline`. Per-step loss visible via `APR_DISTILL_LOG_EVERY` (default 10 from CLI dispatch). | `contracts/distill-pipeline-observability-v1.yaml` |
| #1883 🟡 | (chore) | Stage D dispatch wrapper with PMAT-701 lessons baked in. | (no contract; operator script) |
| #1885 🟡 | (chore) | gx10 disk-cleanup script. | (no contract; operator script) |
| #1886 🟡 | (chore) | Phase 5 HumanEval dispatch wrapper. | (no contract; operator script) |
| #1888 🟡 | PMAT-706 | `APR_DISTILL_MAX_STEPS=N` smoke-validation mode. Catches cascade defects in ~60 s. | `contracts/apr-distill-smoke-validation-v1.yaml` |

Legend: ✅ MERGED · 🟡 OPEN, auto-merge armed at session end.

## Known unknowns

These are NOT defects — they're things we *don't* know that would matter for a real
ship decision:

1. **Per-step latency for 7B teacher on GB10 with cuBLAS.** PMAT-704 verified GPU
   utilization at 96 % during training; we did NOT measure wall-clock per step.
   Run `APR_DISTILL_MAX_STEPS=10 ./scripts/dispatch-distill-stage-d.sh` to measure.
   The `[SMOKE]` summary line prints projected wall time. If > 100 h for 50K steps,
   we have a perf regression somewhere — likely Q4K dequant on every step rather
   than once at upload (worth digging into before committing to Stage D).
2. **KD signal quality of the cuBLAS path.** We verified the cascade doesn't hang
   and the GPU runs hot. We did NOT verify that loss decreases meaningfully (only
   that the loop runs). The smoke summary's `initial_loss` vs `final_loss`
   delta is the first signal — if `final_loss > initial_loss` or `delta < 5 %`
   after 10 steps with batch=32, KD is broken (likely Q4K dequant numerical
   noise drowning out the gradient). Worth ~30 min of bisection before Stage D.
3. **Stage D corpus selection.** The dispatch wrapper defaults to synthetic batches.
   For real Phase 4, set `DATASET_DIR=/path/to/encoded/shards` per the spec
   §4 default (qwen-v3 1.24 B token Python corpus). If the encoded shards aren't
   on disk anymore (gx10 disk pressure during hiatus may have triggered cleanup),
   re-encode via `apr tokenize encode-corpus ...` before Stage D.

## Memory entries saved this session (relevant on resume)

- `feedback_smoke_defaults_leak_into_production.md` — Phase-3 smoke fallback became
  Phase-4 production default; ~30 h of GPU compute on degenerate KD objective hidden
  for 2 days by eval false-positive.
- (existing) `feedback_a_priori_theoretical_falsification.md` — 30 min of math saves
  8 h of GPU; the runtime analog landed as PMAT-706 (smoke validation).
- (existing) `feedback_workspace_test_missing_binary_transient.md` /
  `feedback_workspace_test_trueno_sigsegv_cleanup.md` — gh pr update-branch vs
  gh run rerun --failed; both used heavily this session.

## Worktrees outstanding (cleanup after hiatus)

All `/tmp/*-worktree/` directories are session-scratch. They can be removed
post-merge via `git worktree remove`. The branches themselves are tracked by the
PRs above; once a PR merges, the branch deletes automatically (`--delete-branch`
flag on every `gh pr merge`).

## What I deliberately did NOT do

- **No Stage D dispatch.** ~30-50 h compute job that shouldn't run unattended for
  3 months. Operator (future-you) starts it when actively monitoring.
- **No Phase 5 eval dispatch.** Same reason — dependent on Stage D output.
- **No new investigative work.** The cascade closure is the natural pause point.
- **No HF Hub publishes.** No model worth publishing exists until Stage D + Phase 5 pass.
- **No fixes for the in-flight CI infra flakes.** The PRs are auto-merge armed;
  the runners' transient errors (registry unreachable, sibling repo missing) are
  outside session scope per `feedback_workspace_test_missing_binary_transient.md`.

## If something feels off when you come back

1. `pmat query "PMAT-701" -G` — recover the cascade narrative from commits.
2. `cat docs/specifications/aprender-train/distillation-epic-spec.md` — §86 (PMAT-701)
   + §87 (PMAT-704 post-mortem) are the authoritative spec entries.
3. `ls evidence/distill-7b-cublas-cudatrainer/findings.json` — full 5-whys of the
   PMAT-704 cascade and why the cuBLAS path is the right default.
4. `gh pr list --search "PMAT-70 in:title"` — see what landed and what's still open.

Cascade closure verified: 96 % GPU utilization on gx10 with 7B Q4K teacher (PR #1879
verification). Operator path to Phase 4 production is clear.

— end of hiatus checkpoint, 2026-05-22.
