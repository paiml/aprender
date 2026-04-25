# Specification: Ship Two Models — Sovereign AI Stack Proof

**Document ID:** SPEC-SHIP-TWO-001
**Version:** 2.56.0
**Atomic next action (v2.56.0):** pretokenize The Stack v2 filtered-Python for MODEL-2 convergence (unchanged from v2.51.0). **FALSIFY-SHIP-010 PARTIAL_ALGORITHM_LEVEL → DISCHARGED** on 2026-04-25 (task #150) via live `apr validate-manifest --live --json` execution against all 3 paiml/qwen2.5-coder-7b-apache-q4k-v1 publish manifests on noah-Lambda-Vector RTX 4090. Each manifest's `overall: PASS` recorded: APR `0a854098...c73666` over 8 035 635 524 B; GGUF `e6cac5d6...e7981` over 8 037 129 408 B; safetensors `c1058ce7...d8954` over 15 231 938 404 B. Per-manifest gates green: PM-001/002-live/003/004/005/006. Format-specific PM-007/008/009 DEFER without local --artifact (algorithm-level proofs ship separately). Contract `publish-manifest-v1.yaml` v1.4.0 → **v1.5.0** stays DRAFT. Drift-prevention test `falsify_ship_010_yaml_binding_pins_discharged_status` added. Spec v2.55.0 → **v2.56.0**. Coverage tally is now **35 PARTIAL + 10 DISCHARGED** (was 36 + 9; SHIP-010 promoted; **fifth MODEL-1 PARTIAL → DISCHARGED of the cycle**, after SHIP-009 #1054 + SHIP-001 #1056 + SHIP-004 #1057 already merged). Evidence files: `evidence/ship-010-full-discharge/validate-manifest-{apr,gguf,safetensors}.json`. Methodology: pure stack tooling, no `eprintln!`, no bash workarounds — per `feedback_apr_trace_not_eprintln.md`.

**Atomic next action (v2.52.0):** pretokenize The Stack v2 filtered-Python for MODEL-2 convergence (unchanged from v2.51.0) AND address **two session-discovered defects** that block full discharge of MODEL-1 gates. (1) **SHIP-007 parity-gate blocker:** `apr bench /mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr --iterations 5 --max-tokens 128` fails at CUDA init with `PARITY-GATE FAILED: Cosine similarity: -0.005190 (required: ≥0.98), CPU argmax: 334 | GPU argmax: 8127, Max absolute logit difference: 19.5053`. Error message attributes divergence to the model's 28-head / 4-kv-head / 3584-hidden layout (Qwen2.5-Coder-7B canonical). Counter-evidence: the 370M training path on the same host does NOT trigger this gate — it's specific to the 7B/GQA-7:1 serving path. `apr parity` also fails at the same init point, so no layer-specific divergence map is yet available. Per `feedback_fix_root_cause_never_route_around.md`: do NOT `SKIP_PARITY_GATE=1`; fix is upstream attention kernel (likely `aprender-compute::gqa_attention_kernel` on sm_89). SHIP-007 live evidence blocks pending that upstream fix; algorithm-level PARTIAL from PR #1014 (f32-threshold `verdict_from_decode_tps`) stays valid. Memory: `project_ship_007_parity_gate_blocker.md`. (2) **Teacher provenance gap:** `apr inspect /mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr` shows `Provenance: license: (missing), data_source: (missing), data_license: (missing)`. The teacher was built at commit `06a3eae38` (spec v2.11.0) — before the `GATE-APR-PROV-001/002/003` provenance-writing gates shipped at commit `8f0607d42` (post-v2.19 evidence branch, task #113). The shipped teacher therefore has never had provenance fields populated. This directly affects **SHIP-009 full discharge** (AC-SHIP1-009: License & provenance recorded in `model.apr` metadata) — the algorithm-level bind (`apr-provenance-v1` v1.1.0 + GATE-APR-PROV-004) PASSES on any (Some, Some, Some) triple, but the shipped artifact is (None, None, None). No CLI stamping path exists today — `apr convert` has no `--license` flag, there's no `apr stamp` subcommand. Follow-up `task #141` covers authoring either (a) a dedicated `apr stamp --license X --data-source Y --data-license Z` subcommand with streaming tensor copy for 7.48 GiB, or (b) extending `apr convert` with provenance flags. After the tool lands, the teacher must be re-stamped + re-uploaded to HF + publish-manifest sha256 updated; that's a full release cycle so deferring to follow-up. Spec coverage tally unchanged at **39 PARTIAL + 6 DISCHARGED** — v2.52.0 is narrative + task inventory, not rule promotion.

**Atomic next action (v2.51.0):** pretokenize The Stack v2 filtered-Python (≥~15B tokens) into MODEL-2-tokenizer (vocab=50257) `.bin` shards and dispatch the multi-hour MODEL-2 convergence run. The infrastructure corridor is now fully evidence-discharged: FALSIFY-GPUTRAIN-004 flipped PARTIAL_ALGORITHM_LEVEL → **DISCHARGED** on 2026-04-24 (task #140) via three seed=0 `apr pretrain --device cpu --synthetic --num-steps 4` dispatches on `noah-Lambda-Vector` (binary apr 0.31.2 built `--features cuda`). All three runs produced **byte-identical scripted-loss traces** (sha256 `aeea198418ec50ba9897f95fa641bcdf43f4e05f99f4e5f7b9e308af29e7ff78` after stripping `tokens_per_sec` and `gpu_util_pct`) AND `nvidia-smi --query-compute-apps` returned empty during the CPU dispatch (training pid 1029007 absent from the CUDA-compute-apps list). This proves the CPU path remained functional after the Task #132 device-dispatch refactor — no silent GPU promotion, no silent CPU fallback, peer-contract gates GATE-TRAIN-005 (decreasing scripted loss) / GATE-TRAIN-007 (no NaN) / GATE-TRAIN-008 (grad_norm finite) all PASS under aggregate `status: "OK"`. Evidence: `evidence/task-132/cpu-fallback-peer-gates.json`. Contract `gpu-training-backend-v1.yaml` bumped v1.2.0 → **v1.3.0** (same ACTIVE status) to record the second Phase 3 evidence cycle. Coverage tally is now **39 PARTIAL + 6 DISCHARGED** (was 40 + 5; GPUTRAIN-004 promoted). The only remaining GPUTRAIN PARTIAL is **FALSIFY-GPUTRAIN-006** (same-device seed reproducibility, runtime_falsification_observed at |Δ|=1.19e-3 vs 1e-5 tolerance — holds at PARTIAL pending a `--deterministic` Rust code path that pins cuBLAS math mode + workspace + disables atomic reductions). Per Toyota Way: not widening the 1e-5 tolerance; authoring the deterministic code path is tracked separately from MODEL-2 convergence.

**Atomic next action (v2.46.0):** run `apr pretrain --num-steps 10000+ --device cuda:0` on `noah-Lambda-Vector` (RTX 4090, 24 GB VRAM, local) for a real MODEL-2 convergence pass. Phase 3 dispatch is PROVEN — FALSIFY-GPUTRAIN-002 and FALSIFY-GPUTRAIN-003 just flipped PARTIAL_ALGORITHM_LEVEL → **DISCHARGED** on 2026-04-24 via two `apr pretrain --mode from-scratch --device cuda:0` runs (5-step smoke + 50-step full) with nvidia-smi residency proof (pids 3960483 @ 404 MiB / 3964206 @ 10,300 MiB, both far above the 1 MiB floor). Evidence persisted to `evidence/task-132/rtx4090-370m-residency.json`. The hostname confusion — "lambda-labs" in §14.5 naming a remote cloud host vs. noah-Lambda-Vector as a local Lambda-class RTX 4090 workstation — falsified itself the moment the code-check ran: this box IS a lambda-labs-equivalent and Phase 3 dispatch runs locally. With Phase 2 runtime wiring already shipped (verified 2026-04-24 in prior §14.5 correction) and Phase 3 residency discharged, MODEL-2 training is unblocked at the *infrastructure* layer. What's pending is a multi-hour convergence run that brings val CE down toward the SHIP-013 ≤ 2.2 floor — the 5/50-step smokes ran fast (5s / 63s wall) with val CE = 10.08 (untrained), so the machinery is ready; only GPU clock-time remains. Contract `gpu-training-backend-v1.yaml` also flipped v1.1.0 PROPOSED → **v1.2.0 ACTIVE** per §14 Phase 4. Coverage tally is now **40 PARTIAL + 5 DISCHARGED** (was 42 + 3; two GPUTRAIN-promoted).

**Atomic next action (v2.45.0):** Task #132 **Phase 3** — lambda-labs RTX 4090 live dispatch + residency-evidence collection. **Phase 2 runtime wiring is SHIPPED** (my v2.44/v2.45 narrative above initially said "pending, 2 days" — that was stale, code-checked 2026-04-24 and falsified): `crates/apr-cli/src/commands/pretrain.rs::drive_real` at lines 252–301 already takes `device: Device`, dispatches `if device.is_cuda() { drive_real_cuda(...) } else { drive_real_cpu(...) }`, and `drive_real_cuda` (line 336, `#[cfg(feature = "cuda")]`) builds a real `CudaTransformerTrainer` via `entrenar::train::pretrain_real_cuda::build_shared_cuda_trainer` + wires `CudaRealStepFn`/`CudaRealValFn`/`CudaAprCheckpointFn`. The `#[cfg(not(feature = "cuda"))]` companion at line 373 returns a clear GATE-GPUTRAIN-002 error instead of silent CPU fallback. nvidia-smi querying lives in `crates/aprender-train/src/config/train/loader/mod.rs:445` + `crates/aprender-train/src/gpu/ledger.rs:404`. What's actually pending is pure hardware work: one `cargo build --release --features cuda` on lambda-labs, then `apr pretrain --mode from-scratch --device cuda:0 --num-steps 50 --json` emitting `evidence/task-132/rtx4090-370m-step-budget.json` with median step-wall < 500 ms + nvidia-smi residency proof — no Rust left to write. After Phase 3 evidence lands, FALSIFY-GPUTRAIN-003..007 flip PARTIAL_ALGORITHM_LEVEL → DISCHARGED, `gpu-training-backend-v1.yaml` goes PROPOSED → ACTIVE (Phase 4), and the same lambda-labs session can run MODEL-1 PARTIAL → DISCHARGED harnesses (`apr convert --quantize q4_k_m`, `apr eval --benchmark humaneval`, `apr qa`, `apr bench`) in parallel against the published 7B teacher artifact. MODEL-2 training itself (SHIP-013 val CE ≤ 2.2 + SHIP-014 ≤ 21 days) is a separate multi-day compute run that starts immediately after Phase 3 residency proof — the dispatch glue is ready, only the GPU clock hasn't started.

**Spec-drift five-whys (recorded 2026-04-24 during v2.45.0 authoring):**
 1 Why did v2.45.0 initially name "Phase 2 runtime wiring" as the atomic next action? I trusted §14.5's "Phase 2 (live-wire, pending) — 2 days" row without code-checking.
 2 Why was §14.5 stale? It was authored 2026-04-21 when the bug was fresh; the wiring landed in a later PR that didn't amend the spec.
 3 Why didn't the wiring PR amend the spec? The spec is narrative; the code passes tests; nothing enforces that narrative-claims and code-reality agree. Same class as README drift and const drift pre-v2.44.
 4 Why didn't v2.44 drift-prevention catch this? v2.44 guards (a) README numbers (`readme-claims-v1.yaml`) and (b) `AC_*` threshold consts (`ship_two_001_const_pinning.rs`). It does NOT guard *prose claims about which Rust functions exist*. Runtime-wiring claims were unguarded.
 5 Why is there no mechanism? Same five-whys that produced SHIP-TWO-001 in the first place: spec narrative outruns code verification unless someone bolts on a falsification test. This is the **next** drift-prevention class to add — a test that greps for `fn drive_real_cuda` existence under `#[cfg(feature = "cuda")]` and fails if the claimed runtime path is absent. Filed as follow-up in the same PR as this correction.

**v2.46.0 amendment (2026-04-24, first live Phase 3 discharge):** My v2.45.0 narrative above claimed Phase 3 dispatch required a remote lambda-labs cloud box. Noah falsified that within minutes: `hostname` on this workstation returns `noah-Lambda-Vector` (a Lambda Labs Vector workstation class), and `nvidia-smi` reports an **NVIDIA GeForce RTX 4090 with 24 GB VRAM**, driver 570.207, CUDA 12.8. The local workstation IS the lambda-labs-class host the spec wanted. Built `apr` from HEAD with `cargo build --release --features "cuda training" -p apr-cli --bin apr`, ran two `apr pretrain --mode from-scratch --device cuda:0` dispatches (5-step smoke: pid 3960483 / 404 MiB / val CE 10.082615; 50-step full: pid 3964206 / 10,300 MiB / val CE 10.082788; both `exit 0`, both `"status": "OK"` in the JSON output). `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` confirmed GPU residency in both runs, `verdict_from_residency` returned Pass, `AC_GPUTRAIN_003_MIN_USED_MEMORY_MIB = 1` floor cleared by 404× and 10,300× respectively. Task #132 outcome: FALSIFY-GPUTRAIN-002 (no silent CPU fallback — the exact defect Task #132 was filed against on 2026-04-21) **DISCHARGED**; FALSIFY-GPUTRAIN-003 (GPU residency proof) **DISCHARGED**. `contracts/entrenar/gpu-training-backend-v1.yaml` v1.1.0 PROPOSED → **v1.2.0 ACTIVE** per §14 Phase 4. Evidence JSON: `evidence/task-132/rtx4090-370m-residency.json` (schema-validated, captures hostname / GPU / driver / CUDA / dispatch command / both runs' nvidia-smi output / both runs' `apr` exit JSONs). Coverage shifts 42 PARTIAL + 3 DISCHARGED → **40 PARTIAL + 5 DISCHARGED** (two GPUTRAIN promotions). Remaining blockers for full 7/7 FALSIFY-GPUTRAIN discharge: 005 (step-time < 500 ms — needs per-step `--json` telemetry emission in `apr pretrain`), 006 (same-seed reproducibility — two back-to-back runs + loss-trajectory diff), 007 (apr --version --json schema — quick wire-up). Remaining blocker for MODEL-2 weights: a multi-hour `--num-steps 10000+` convergence run using the now-proven dispatch.

**v2.47.0 amendment (2026-04-24, GPUTRAIN-005 DISCHARGED + GPUTRAIN-006 runtime-FALSIFIED):** Extended `apr pretrain`'s `PretrainReport` JSON schema to emit per-step `StepMetrics` (the 6-field `GATE-TRAIN-001` tuple: step, train_loss, grad_norm, lr, tokens_per_sec, gpu_util_pct) so downstream Phase-3 harnesses don't have to parse run-dir checkpoint metadata. With per-step visibility in hand, ran two more Phase 3 checks against the live RTX 4090. **FALSIFY-GPUTRAIN-005** (step time < 500 ms at batch=1 seq=2048 per spec canonical) — PASSED cleanly: derived step_ms from tokens_per_sec (batch * seq * 1000 / tps) across 25 warmed-up steps, observed **median 101.30 ms** (min 97.58, max 110.39), well inside the 500 ms budget at 20.3% utilization. PARTIAL_ALGORITHM_LEVEL → **DISCHARGED**. **FALSIFY-GPUTRAIN-006** (same-seed reproducibility, |Δloss[k]| ≤ 1e-5) — ran two back-to-back `--device cuda:0 --seed 0 --num-steps 50` dispatches and diffed per-step train_loss trajectories. Observed max **|Δ| = 8.27e-4 at step 27**, with even step 0 (pure forward pass, no training update applied) showing **Δ = 2.3e-5**. **83× over the 1e-5 tolerance**. The algorithm-level verdict fn is provably correct; CUDA is simply not bit-deterministic without explicit mode configuration (cublasSetMathMode / CUBLAS_WORKSPACE_CONFIG / cuDNN determinism / atomics-off reductions). Per Toyota Way: **NOT widening** `AC_GPUTRAIN_006_MAX_SEED_LOSS_DELTA` to match observed drift — that would hide the defect. Instead: GPUTRAIN-006 **held at PARTIAL_ALGORITHM_LEVEL** with a `runtime_falsification_observed` block in the contract, a root-cause hypothesis, and a follow-up ticket for a `--deterministic` opt-in code path (estimate: 1-2 days Rust, acceptance: re-run the harness under `--deterministic` and assert Pass). Evidence: `evidence/task-132/rtx4090-370m-step-budget-and-repro.json`. Coverage: 40 PARTIAL + 5 DISCHARGED → **39 PARTIAL + 6 DISCHARGED** (005 promoted; 006 held with documented runtime falsification).

**v2.48.0 amendment (2026-04-24, FALSIFY-GPUTRAIN-007 DISCHARGED + MODEL-2 convergence training launched):** Added `emit_version_json()` in `crates/apr-cli/src/lib.rs` + `--version --json` intercept in `crates/apr-cli/src/main.rs` before clap's default `--version` exit. Output on a `--features cuda` build on noah-Lambda-Vector: `{ "cuda_feature": true, "cuda_runtime_available": true, "git_sha": "...", "name": "apr", "version": "0.31.2", "visible_devices": ["0"] }`. Both `verdict_from_version_json_keys` (schema completeness) and `verdict_from_version_json_fields` (len ≤ 16, no FM-GPUTRAIN-STALE-BUILD inconsistency) return Pass. FALSIFY-GPUTRAIN-007 PARTIAL_ALGORITHM_LEVEL → **DISCHARGED**. Second deliverable: **MODEL-2 convergence training launched** in background (pid 805184, 22.2 GB GPU residency on the RTX 4090, 2000-step `--batch-size 1 --seq-length 2048 --seed 0 --warmup-steps 200 --mode from-scratch` run, run-dir `/mnt/nvme-raid0/runs/model-2-convergence-20260424-181022`). At ~100 ms/step the full 2000-step sweep should complete in ~3-4 minutes of GPU time; the val-loss trajectory will be the first live descent signal toward the SHIP-013 ≤ 2.2 val CE floor (starting point 10.08 from the 50-step smoke). Evidence for 007: `evidence/task-132/version-json-schema.json`. Coverage: 39 PARTIAL + 6 DISCHARGED → **38 PARTIAL + 7 DISCHARGED** (full GPUTRAIN surface: 002/003/005/007 DISCHARGED + 001/004 ACTIVE in device.rs + 006 held with runtime-falsification note).

**v2.49.0 amendment (2026-04-24, MODEL-2 first live training — 1300-step EARLY_STOP smoke):** The 2000-step background convergence run launched by v2.48.0 completed with `status: EARLY_STOP` at step 1300 (val-loss plateau detected by the pretrain loop's guard). First real MODEL-2 training data on the stack. **Infrastructure: fully proven**; **SHIP-013 convergence: not yet achieved**. Observed val-loss descent: `[10.60, 10.50, 10.36, 10.22, 10.19, 10.21, 10.21, 10.14, 10.15, 10.10, 10.14, 10.17, 10.16]` over 1300 steps. `final_val_loss = 10.103599` vs `AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS = 2.2` target — **gap of 7.90 nats**. train_loss descended 11.04 → min 9.27 → settled ~10.47. Step-time median 100.83 ms (consistent with v2.47's 101.30 ms measurement under the same config). GPU peak residency 22.2 GB / 24 GB on the RTX 4090 (batch=1, seq=2048). Per Toyota Way: **NOT** promoting FALSIFY-SHIP-013 beyond PARTIAL_ALGORITHM_LEVEL — the decision rule is correct, the infrastructure works, but observed val CE (10.10) does not meet the 2.2 floor. The gap is a training/tuning problem, not an algorithm or infrastructure problem: longer runs, LR-schedule sweep, larger corpus, or larger effective batch are the next levers. Coverage unchanged at **38 PARTIAL + 7 DISCHARGED** — this amendment documents live training evidence without promoting the rule. Evidence file: `evidence/task-132/rtx4090-370m-convergence-smoke.json` (dispatch command + run-dir + val-loss history + train-loss trajectory + step-time median + gap analysis + next-action list). Next convergence iterations should re-run at `--num-steps 50000+` with tuned `--lr`/`--warmup-steps`/`--batch-size` to exit the plateau region around val CE ≈ 10 and push toward 2.2.

**v2.50.0 amendment (2026-04-24, second independent run confirms corpus bottleneck):** Launched a 10000-step re-run with explicit `--target-val-loss 2.2 --steps-per-epoch 500 --num-steps 10000 --seed 0` (4× longer budget than the v2.49 smoke). Result: early-stop again, this time at step **2500** with `final_val_loss = 10.103826` — essentially identical to the v2.49 run's 10.103599. Two independent runs converge to the **same val_loss ceiling (~10.10)** despite 4× step budget. Train-loss 10-bucketed means show slow descent on training data (10.53 → 10.14 over 2500 steps) but val-loss is frozen on plateau. Bucketed bucket-to-bucket deltas: `-0.17, -0.12, +0.03, +0.01, -0.10, +0.11, +0.02, -0.10, -0.09` — drift noise, not descent. Signature is textbook corpus-undersizing: a 370M model on ~10M tokens (CSN-Python-shards from task #118 pretokenization) has memorized what it can and cannot extract further generalization signal. Typical 370M-class convergence requires 10-100 **billion** tokens — this corpus is **1000-10000× undersized**. The bottleneck is NOT step count, NOT learning rate, NOT infrastructure — it is the input dataset. Per §5.1 MODEL-2 current state, the *intended* training corpus is 60 GB deduplicated Python from The Stack v2 filtered subset (≥~15B tokens); CSN-Python-shards is a task #118 pretokenize smoke, not the ship corpus. Evidence file: `evidence/task-132/rtx4090-370m-convergence-10k-plateau.json` (dispatch + plateau metrics + bucketed train-loss + corpus-bottleneck analysis + next-action list). Per Toyota Way: SHIP-013 **remains NOT DISCHARGED**; instead of widening the 2.2 floor, this amendment names the root cause (corpus size) with live two-run evidence and points at the corrective action (pretokenize The Stack v2 Python full). Coverage unchanged at **38 PARTIAL + 7 DISCHARGED** — the amendment is narrative + evidence, not rule promotion.

**Status:** SHIP-TWO-001-MODEL-1-TEACHER **RELEASED**; MODEL-2 pretraining **loop driver landed** (task #105 CLOSED — commit `9a5af3ac2`); 370M Llama scaffold + pretrain loop + `apr pretrain` CLI all dogfood-ready; Zero-Tolerance design principle codified (§3 row #8); `pv validate` dogfooded across all 760 contracts (task #101); 8 legacy contracts backfilled with kani_harnesses + falsification parity (task #102 CLOSED); MODEL-2 `--min-frequency` threaded end-to-end through aprender-train BPE (task #103 CLOSED); gx10 third-party framework capacity gate PASS at 38.0 tok/s decode with 26.7% margin (task #104 CLOSED); loader hardened to ignore co-located ModelFamilyVariant contracts (task #108 CLOSED — 32→0 workspace-test failures); **task #132 BLOCKER surfaced 2026-04-21** on lambda-labs RTX 4090 — `apr pretrain --mode from-scratch` ran 14 min at 114% CPU + 0 MiB GPU memory because `TransformerTrainer::new` has no Device argument; `contracts/entrenar/gpu-training-backend-v1.yaml` PROPOSED (task #132 Phase 0) with INV-GPUTRAIN-001..007 + GATE-GPUTRAIN-001..006, ship-blocks task #126 real-compute dispatch
**Author:** PAIML Engineering
**Reviewer:** Noah Gift
**Date:** 2026-04-17 (v1.0.0) / 2026-04-17 (v2.0.0 audit + pivot) / 2026-04-18 (v2.5.0 pre-flight Poka-Yoke) / 2026-04-18 (v2.6.0 PM-008 GGUF tensor-type Poka-Yoke) / 2026-04-18 (v2.7.0 PM-009 APR magic-bytes Poka-Yoke) / 2026-04-18 (v2.8.0 HF Hub Xet large-file upload contract) / 2026-04-18 (v2.8.1 Xet impl landed) / 2026-04-18 (v2.9.0 EX-04 DISCHARGED via NDJSON lfsFile schema) / 2026-04-18 (v2.10.0 MODEL-1 v2 QLoRA divergence root cause — teacher-only ship) / 2026-04-18 (v2.11.0 EX-05/06/07 DISCHARGED — teacher tagged SHIP-TWO-001-MODEL-1-TEACHER) / 2026-04-18 (v2.12.0 post-ship artifacts — MODEL-2 contracts + MODEL-1 retry plan + SHARD-003 probe) / 2026-04-18 (v2.13.0 FALSIFY-SHARD-003 DISCHARGED live yoga vs gx10) / 2026-04-18 (v2.14.0 MODEL-2 dataset contract drafted + BPE NFC gap identified) / 2026-04-18 (v2.15.0 MODEL-2 scaffold LANDED — BPE NFC + tokenizer CLI + corpus ingest binary) / 2026-04-18 (v2.16.0 Zero-Tolerance design principle codified — no bugs, no perf regressions, no carve-outs) / 2026-04-18 (v2.17.0 contracts schema harmonization shipped — pv validate works across all 760 contracts, unblocks dogfooded gate) / 2026-04-18 (v2.18.0 parallel dispatch lanes #102/#103/#104 all closed — 8 contracts backfilled + MODEL-2 --min-frequency plumbed + gx10 38.0 tok/s PASS) / 2026-04-18 (v2.19.0 MODEL-2 pretrain loop driver landed via task #105 sub-agent — GATE-TRAIN-005 + INV-TRAIN-007 wired; `apr pretrain` CLI gated by `training` feature; loader hardened for ModelFamilyVariant contracts via task #108) / 2026-04-19 (v2.20.0 FALSIFY-SHIP-021 + FALSIFY-SHIP-022 DISCHARGED — MODEL-2 seed-reproducibility harness + apr inspect provenance block wired; tasks #112 #113 closed on chore/post-v2.19-evidence) / 2026-04-19 (v2.21.0 FALSIFY-SHIP-011 DISCHARGED + FALSIFY-SHIP-012/015 PARTIAL_ALGORITHM_LEVEL — C-LLAMA-370M-SOVEREIGN v1.0.0 PROPOSED → v1.2.0 ACTIVE with Rust-YAML byte-equality binding + param-count algorithm proof; C-TOK-BPE v1.1.0 wires 3 tokenizer tests; tasks #114 #115 #116 closed; 3/12 ACTIVE + 2/12 PARTIAL) / 2026-04-19 (v2.22.0 FALSIFY-SHIP-019 PARTIAL_ALGORITHM_LEVEL — C-LLAMA-370M-SOVEREIGN v1.2.0 → v1.3.0 stays ACTIVE; GATE-ARCH-370M-004 wired to 3 algorithm proofs reusing `layout_contract.rs` per Spec §9 Risk #2; task #117 closed on commit `846cc1dbb`; 3/12 ACTIVE + 3/12 PARTIAL = 6/12 touched) / 2026-04-21 (v2.23.0 **task #132 CUDA training backend gap** surfaced on lambda-labs RTX 4090 real-compute dispatch at commit `f7ad11408` — `apr pretrain --mode from-scratch` 14min on CPU (0 MiB GPU memory) because `TransformerTrainer` has no Device awareness; `contracts/entrenar/gpu-training-backend-v1.yaml` PROPOSED (Phase 0) with INV-GPUTRAIN-001..007 + GATE-GPUTRAIN-001..006 + FALSIFY-GPUTRAIN-001..007; production `CudaTransformerTrainer` already exists at `crates/aprender-train/src/train/transformer_trainer/cuda_trainer.rs` — gap is wiring, not kernels; task #126 real-compute dispatch BLOCKED until Phase 3 residency-proof evidence lands) / 2026-04-22 (v2.24.0 **FALSIFY-SHIP-008 PARTIAL_ALGORITHM_LEVEL** — task #155 wires MODEL-1 chat-template render gate: `contracts/chat-template-v1.yaml` v1.0.0 → v1.1.0 adds `GATE-CHAT-SHIP-008` binding `ChatMLTemplate::format_conversation` to the canonical Qwen2.5-Coder-7B golden via a pure `verdict_from_chat_template_render` const fn + 5-section mutation survey (empty / missing-gen-prompt / wrong-delim / swapped-roles / single-byte flip) + provenance pin; `cargo test -p aprender-core --lib falsify_ship_008_chat_template_render_bind` green; full discharge blocks on live `apr run paiml/qwen2.5-coder-7b-apache-q4k-v1` completion diff; **MODEL-1 coverage 1/10 → 2/10** touched — first MODEL-1 non-provenance PARTIAL, mirrors MODEL-2 pattern set by SHIP-016/017/018/020; 8 PARTIAL + 3 DISCHARGED across both models) / 2026-04-22 (v2.25.0 **FALSIFY-SHIP-006 PARTIAL_ALGORITHM_LEVEL** — task #156 wires MODEL-1 `apr qa` 8-gate aggregate gate: `contracts/apr-model-qa-v1.yaml` v1.1.0 → v1.2.0 adds `FALSIFY-QA-SHIP-006` binding the aggregate-AND verdict fn `verdict_from_qa_gates(&[bool]) -> Ship006Verdict` to the 8-gate ship criterion (golden / throughput / ollama parity / gpu speedup / tensor contracts / format parity / ptx parity / metadata per `docs/specifications/components/qa.md` §3) via pure const fn + 7-section mutation survey (all-Pass / all-Fail / single-gate-flip × 8 / exhaustive 2^8=256 bitmask proof / monotonicity / length drift 0-7-9-16 / provenance pin); `cargo test -p aprender-core --lib falsify_ship_006_apr_qa_eight_gates_aggregate` green; full discharge blocks on live `apr qa paiml/qwen2.5-coder-7b-apache-q4k-v1 --json` on RTX 4090 host; **MODEL-1 coverage 2/10 → 3/10** touched — mirrors MODEL-2 SHIP-016 aggregate-AND shape but authored self-contained because SHIP-016 branch not yet on main; 9 PARTIAL + 3 DISCHARGED across both models) / 2026-04-22 (v2.26.0 **FALSIFY-SHIP-002 PARTIAL_ALGORITHM_LEVEL** — task #159 wires MODEL-1 canonical `def fib(n):` Python-syntax gate: `contracts/qwen2-e2e-verification-v1.yaml` v1.0.0 → v1.1.0 adds `FALSIFY-QW2E-SHIP-002` binding zero-tolerance `const fn verdict_from_syntax_error_count(usize) -> Ship002Verdict` in `crates/aprender-core/src/qa/ship_002.rs` + 6-section mutation survey (zero-errors → Pass / exactly-one-error → Fail / many-errors Fail band {2, 10, 100} / monotonicity sweep 0..=256 / `usize::MAX` sanity Fail / provenance pin tolerance = 0); `cargo test -p aprender-core --lib falsify_ship_002_python_syntax_error_threshold_logic` green; full discharge blocks on live `apr run paiml/qwen2.5-coder-7b-apache-q4k-v1 --prompt "def fib(n):"` on RTX 4090 + `rustpython`/`ruff` AST parse; **MODEL-1 coverage 3/10 → 4/10** touched — tightest MODEL-1 rule (0 tolerance on single canonical prompt vs MODEL-2 SHIP-017 which tolerates ≤1 across 100 held-out prompts) because spec §4.2 "emits valid Python" carries no noise allowance; 10 PARTIAL + 3 DISCHARGED across both models) / 2026-04-22 (v2.27.0 **FALSIFY-SHIP-005 PARTIAL_ALGORITHM_LEVEL** — task #158 wires MODEL-1 `apr eval --benchmark humaneval` pass@1 ship floor: `contracts/qwen2-e2e-verification-v1.yaml` v1.1.0 → v1.2.0 adds `FALSIFY-QW2E-SHIP-005` binding `AC_SHIP1_005_NOMINAL_HUMANEVAL_PASS_AT_1_PCT = 86.00` + `AC_SHIP1_005_NOISE_ALLOWANCE_PP = 1.20` + `AC_SHIP1_005_EFFECTIVE_HUMANEVAL_PASS_AT_1_PCT ≈ 84.80` to pure two-number threshold verdict fn `verdict_from_pass_at_1(correct, total, threshold_pct) -> Ship005Verdict` in `crates/aprender-core/src/metrics/ship_005.rs` + 8-section mutation survey (safe-margin Pass above effective / above nominal Pass / noise-window Fail at nominal / below-effective Fail including HumanEval-canonical 139/164 = 84.756% / monotonicity sweep 0..=164 / div-safety + sanity guards / non-finite threshold → Fail conservatively / tolerance-bounded provenance pin on all three constants because f32 `86.0 − 1.2 ≈ 84.79999924` ≠ exact 84.80); `cargo test -p aprender-core --lib falsify_ship_005_humaneval_pass_at_1_threshold_logic` green; full discharge blocks on live `apr eval --benchmark humaneval paiml/qwen2.5-coder-7b-apache-q4k-v1 --json` median of 3 seed=0 runs ≥ 86.00 (or ≥ 84.80 under the 1.2 pp noise allowance) on RTX 4090 with `--features cuda`; **MODEL-1 coverage 4/10 → 5/10** touched — mirrors MODEL-2 SHIP-018 pass@1 threshold shape but uniquely carries a 1.2 pp noise allowance (MODEL-2 has no noise window) and is self-contained because SHIP-018 PR #1004 and SHIP-007 PR #1019 are not yet on main; 11 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.28.0 **FALSIFY-SHIP-010 PARTIAL_ALGORITHM_LEVEL** — task #161 wires MODEL-1 published-artifact URL + SHA-256 ship gate: `contracts/publish-manifest-v1.yaml` v1.3.0 → v1.4.0 adds `FALSIFY-SHIP-010` binding two constants — `AC_SHIP1_010_SHA256_HEX_LEN = 64` (sha256sum canonical output) + `AC_SHIP1_010_REQUIRED_URL_SCHEME = "https://"` (TLS floor per §4.2) — to two pure verdict fns `verdict_from_sha256_match(expected_hex, actual_hex) -> Ship010Verdict` + `verdict_from_manifest_url(url) -> Ship010Verdict` in `crates/aprender-core/src/format/ship_010.rs` + twin 7-section mutation surveys: SHA-256 side covers identical-Pass / single-hex-flip / wrong-length / uppercase-rejected / non-hex-rejected / all-zero guard / provenance pin; URL side covers HF canonical / S3 canonical / plaintext-http rejected / scheme-less rejected / empty-host rejected / whitespace-control rejected / provenance pin; `cargo test -p aprender-core --lib format::ship_010` green (3/3 tests); full discharge blocks on live `curl -sSI <artifact_url>` 200-OK + `sha256sum <local_file> == <manifest_hash>` against `paiml/qwen2.5-coder-7b-apache-q4k-v1` on a host with network egress; **MODEL-1 coverage 5/10 → 6/10** touched — first MODEL-1 network-dependent PARTIAL, uniquely carries a TLS-floor byte-literal constant (no MODEL-2 counterpart because AC-SHIP2-012 is provenance-metadata not artifact-resolution); 12 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.29.0 **FALSIFY-SHIP-007 PARTIAL_ALGORITHM_LEVEL** — task #160 wires MODEL-1 `apr bench` decode throughput gate: `contracts/qwen2-e2e-verification-v1.yaml` v1.2.0 → v1.3.0 adds `FALSIFY-QW2E-SHIP-007` binding `verdict_from_decode_tps(f32) -> Ship007Verdict` in `crates/aprender-core/src/bench/ship_007.rs` at `AC_SHIP1_007_MIN_DECODE_TPS_RTX4090_7B = 30.0` tok/s + 7-section mutation survey (boundary at 30.0 → Pass / one-ULP-below → Fail / clear Pass band {45, 100} / clear Fail band {0, 10, 29.999999} / monotonicity above+below floor / non-finite → Fail conservatively {NaN, ±∞} / provenance pin 30.0); `cargo test -p aprender-core --lib falsify_ship_007_decode_tps_threshold_logic` green; full discharge blocks on live `apr bench --iterations 5 --max-tokens 128 paiml/qwen2.5-coder-7b-apache-q4k-v1` on RTX 4090 with `--features cuda` + median ≥ 30.0; **MODEL-1 coverage 6/10 → 7/10** touched — MODEL-1 twin of MODEL-2 SHIP-020 (identical f32-threshold shape, floor 30 vs 100 tok/s — 7B Q4_K is bandwidth-bound at ~3.5× the size of the 370M target); 13 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.30.0 **FALSIFY-SHIP-003 PARTIAL_ALGORITHM_LEVEL** — task #162 wires MODEL-1 `apr convert --quantize q4_k_m` per-layer cosine round-trip gate: `contracts/qwen2-e2e-verification-v1.yaml` v1.3.0 → v1.4.0 adds `FALSIFY-QW2E-SHIP-003` binding `AC_SHIP1_003_MIN_COSINE_SIMILARITY = 0.999` to two pure verdict fns in `crates/aprender-core/src/format/ship_003.rs`: `verdict_from_cosine_similarity(sim, threshold) -> Ship003Verdict` (single-layer threshold + cosine-range guard `[-1.0, 1.0]` + non-finite guard) and `verdict_from_per_layer_cosines(sims, threshold) -> Ship003Verdict` (aggregate-AND combinator, conservative Fail on empty input) + twin 8-section and 7-section mutation surveys (exact 0.999 boundary Pass / ULP-below `0x3F7FBE76` Fail / safe-above {0.9999, 1.0} Pass / safe-below {0.998, 0.5, 0.0, −1.0} Fail / monotonicity sweep [0.990..=1.000] step 1e-4 / non-finite sim+threshold Fail / out-of-range sim+threshold Fail / provenance pin 0.999 on the single-layer side, plus all-Pass / 1-of-196 single-flip Fail / all-Fail / empty-Fail / single-element both directions / first-layer NaN+OOR short-circuit Fail / last-layer Fail not short-circuited on the aggregate side); `cargo test -p aprender-core --lib format::ship_003` green (2/2); full discharge blocks on live `apr convert --quantize q4_k_m paiml/qwen2.5-coder-7b-apache-q4k-v1.safetensors` on RTX 4090 + `apr diff` per-layer cosine harness walking 28 × 7 = 196 projection matrices; **MODEL-1 coverage 7/10 → 8/10** touched — eighth compute-free MODEL-1 PARTIAL lever, first to combine a single-number threshold (mirrors SHIP-007/SHIP-020 decode-tps shape) with an aggregate-AND combinator (mirrors SHIP-016 `verdict_from_qa_gates`) in one discharge; 14 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.31.0 **FALSIFY-SHIP-004 PARTIAL_ALGORITHM_LEVEL** — task #164 wires MODEL-1 `apr export --format gguf` → llama.cpp round-trip gate: `contracts/qwen2-e2e-verification-v1.yaml` v1.4.0 → v1.5.0 adds `FALSIFY-QW2E-SHIP-004` binding three independent format-boundary verdict fns in `crates/aprender-core/src/format/ship_004.rs`: `verdict_from_llama_cli_exit(i32) -> Ship004Verdict` (POSIX zero-tolerance exit-code boundary bound to `AC_SHIP1_004_LLAMA_CLI_SUCCESS_EXIT_CODE = 0`), `verdict_from_gguf_magic_bytes(&[u8]) -> Ship004Verdict` (canonical 4-byte `AC_SHIP1_004_GGUF_MAGIC_BYTES = b"GGUF"` with short-slice Fail and single-byte-flip rejection), and `verdict_from_gguf_version(u32) -> Ship004Verdict` (set-membership over `AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS = &[2, 3]` with Fail-closed above-band rejection) + triple mutation survey: exit-code 6 sections (POSIX success / adjacent-value Fail / classic failure bands {2, 127, 137, 139, 255} / i32 extrema / monotonicity sweep [-256, 256] / provenance pin); magic 6 sections (canonical Pass / magic+version header Pass / 6 single-byte flips / 4 short-slice lengths / wrong-format magics {b"APR\\0", b"APRN", zeros} / 4-byte provenance pin); version 5 sections (supported {2, 3} / predecessor {0, 1} / above-band {4, 5, 10, 100, 1M, u32::MAX} / exhaustive 0..=64 / set-length pin); `cargo test -p aprender-core --lib format::ship_004` green (3/3); full discharge blocks on live `apr export --format gguf paiml/qwen2.5-coder-7b-apache-q4k-v1.safetensors` on RTX 4090 + shell out to upstream `llama-cli -m qwen2.5-coder-7b.gguf --prompt "hello" --n-predict 4` and assert magic, version, AND exit code each Pass; **MODEL-1 coverage 8/10 → 9/10** touched — ninth compute-free MODEL-1 PARTIAL lever, first MODEL-1 discharge binding three independent verdict fns in one AC (each on a different format boundary — executable tool, magic bytes, version set); 15 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.32.0 **FALSIFY-SHIP-001 PARTIAL_ALGORITHM_LEVEL** — task #115 wires MODEL-1 `realizar::Model::load_safetensors` safetensors-load ship gate: `contracts/qwen2-e2e-verification-v1.yaml` v1.5.0 → v1.6.0 adds `FALSIFY-QW2E-SHIP-001` binding three independent pure verdict fns in `crates/aprender-core/src/format/ship_001.rs`: `verdict_from_load_result(bool) -> Ship001Verdict` (Result-boundary collapse to Pass-on-Ok with no further decoding required), `verdict_from_safetensors_header_size(u64, u64) -> Ship001Verdict` (safetensors header-size invariant `0 < N <= file_len − AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN` bound at `AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN = 8` per the canonical little-endian u64 prefix), and `verdict_from_safetensors_json_open_byte(u8) -> Ship001Verdict` (byte-literal check that the JSON header starts with `AC_SHIP1_001_SAFETENSORS_JSON_OPEN_BYTE = b'{' = 0x7B`) + triple mutation survey: Result-boundary 2 sections (Ok → Pass / Err → Fail / provenance pin on the two-state boundary); header-size multiple sections (canonical in-band Pass / zero-size Fail / overflow-past-file-len Fail / off-by-one at `file_len - 8` boundary / 8-byte prefix constant pin); JSON open-byte sections (canonical `0x7B` Pass / adjacent-byte Fail / exhaustive 0..=255 sweep with only `0x7B` allowed / 0x7B constant pin); `cargo test -p aprender-core --lib format::ship_001` green (3/3); full discharge blocks on live `realizar::Model::load_safetensors(paiml/qwen2.5-coder-7b-apache-q4k-v1.safetensors)` on RTX 4090 with `--features cuda` (or a realizar binary equivalent that loads the tensor index and asserts the Result is Ok); **MODEL-1 coverage 9/10 → 10/10** touched — tenth compute-free MODEL-1 PARTIAL lever, completes MODEL-1 to 10/10 touched; only AC-SHIP1-009 (license / provenance metadata) remains untouched in the table if you count SHIP-009 as pending; second triple-verdict decomposition after SHIP-004, reinforcing the pattern where a tool-accepted-the-artifact rule is split into independent format-boundary gates (Result-boundary × header-size × open-brace byte); 16 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.33.0 **FALSIFY-SHIP-009 PARTIAL_ALGORITHM_LEVEL** — task #116 wires the FINAL MODEL-1 AC row (AC-SHIP1-009 "MODEL-1 teacher license + data provenance recorded in `model.apr` metadata") via a SECOND binding on `contracts/apr-provenance-v1.yaml` v1.0.0 → v1.1.0 (stays ACTIVE); the same contract already discharges MODEL-2's AC-SHIP2-012, so one contract now cleanly carries BOTH model bindings — the AprV2Metadata + serde-JSON decision rule is model-agnostic; GATE-APR-PROV-004 is added as the 2nd gate on that SAME contract (first multi-model multi-bind across the entire SHIP-TWO-001 surface); two harness tests in `crates/aprender-core/src/format/tests/provenance_tests.rs`: `falsify_ship_009_apr_metadata_applies_to_model_1_teacher` drives an AprV2Metadata teacher-representative round-trip (license="apache-2.0" + data_source="qwen2.5-coder-7b-instruct" + data_license="apache-2.0") through the serde-JSON path and asserts field-level recovery, and `falsify_ship_009_gate_apr_prov_004_has_partial_discharge_marker` uses `include_str!` on the YAML contract and `serde_yaml::Value` to assert that the new GATE-APR-PROV-004 block carries the correct `binds_to` / `falsification_id` / `discharge_status: PARTIAL_ALGORITHM_LEVEL` / `ship_blocking: true` flags — a byte-equal YAML-to-Rust binding test in the same style as the SHIP-011 Rust-scaffold binding; `cargo test -p aprender-core --lib provenance` green (5/5 including the 3 pre-existing SHIP-022 tests + the 2 new SHIP-009 tests); full ACTIVE promotion blocks on teacher `model.apr` republish under PMAT-686 populating license / data_source / data_license as named fields (fixture-swap only, no code change); **MODEL-1 coverage 9/10 → 10/10 touched** — SHIP-009 is the last MODEL-1 AC row needing a PARTIAL annotation, so MODEL-1 is now fully covered at the PARTIAL_ALGORITHM_LEVEL ship-gate surface; first multi-model multi-bind on a single contract (proves `apr-provenance-v1`'s decision rule is not model-specific) and sixth falsification of the repeatedly stated "exhausted" verdict (SHIP-019 → SHIP-017 → SHIP-020 → SHIP-018 → SHIP-016 → SHIP-009), this one strictly more surprising than the prior five because it is cross-model rather than just another MODEL-2 lever; 17 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.34.0 **FALSIFY-SHIP-017 PARTIAL_ALGORITHM_LEVEL (restacked)** — task #149 wires MODEL-2 (albor 370M Sovereign) `apr run` held-out Python-syntax gate on top of the v2.33.0 MODEL-1 stack: `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.5.0 → v1.6.0 (stays ACTIVE) adds `GATE-ARCH-370M-005` binding AC-SHIP2-007 ↔ FALSIFY-SHIP-017 with `discharge_status: PARTIAL_ALGORITHM_LEVEL`; decision rule is a pure integer threshold — "≤ 1 SyntaxError tolerated out of 100 held-out prompts, ≥ 2 is a ship-blocker" — bound in `crates/aprender-train/src/models/llama_370m.rs` via `AC_SHIP2_007_HELDOUT_PROMPT_COUNT = 100` + `AC_SHIP2_007_MAX_TOLERATED_SYNTAX_ERRORS = 1` consts feeding `verdict_from_syntax_error_count(errors: usize) -> Ship017Verdict` + 2 falsification tests (`falsify_ship_017_syntax_error_count_threshold_logic` covers Pass boundary 0/1 / Fail boundary 2/50/100 / monotonicity ∈ [0, 100] / provenance pin; `falsify_ship_017_gate_arch_370m_005_has_partial_discharge_marker` byte-binds the sovereign contract YAML shape via `include_str!`); `cargo test -p aprender-train --lib llama_370m` → 12/12 pass; full discharge blocks on real trained 370M .apr + 100-prompt `apr run` harness with EX-06-style Python AST parse (AC-SHIP2-003/004 pretraining compute-dispatch); **MODEL-2 coverage 4/12 → 5/12** touched — this is the first MODEL-2 PARTIAL to land on top of a completed 10/10 MODEL-1 stack, carrying the same integer-threshold shape as MODEL-1 SHIP-002 but with a 1-error tolerance (MODEL-1 has 0 on a single canonical prompt) to accommodate the 100-prompt held-out harness noise floor; 18 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.35.0 **FALSIFY-SHIP-020 PARTIAL_ALGORITHM_LEVEL (restacked)** — task #150 wires MODEL-2 `apr bench` decode-throughput gate on top of the v2.34.0 SHIP-017 restack: `contracts/model-families/llama-370m-sovereign-v1.yaml` stays at v1.6.0 ACTIVE adding `GATE-ARCH-370M-006` binding AC-SHIP2-010 ↔ FALSIFY-SHIP-020 with `discharge_status: PARTIAL_ALGORITHM_LEVEL`; decision rule is a pure f32 threshold — "median decode throughput ≥ 100 tok/s on RTX 4090 (370M target)" — bound in `crates/aprender-train/src/models/llama_370m.rs` via `AC_SHIP2_010_MIN_DECODE_TPS_RTX4090 = 100.0` + `verdict_from_decode_tps(measured_tps: f32) -> Ship020Verdict` + 2 falsification tests (`falsify_ship_020_decode_tps_threshold_logic` covers exact 100.0 Pass / one-ULP-below Fail / generous-green {120, 500} / hard-red {0, 50} / monotonicity in both directions / non-finite conservative Fail for {NaN, ±∞} / provenance pin; `falsify_ship_020_gate_arch_370m_006_has_partial_discharge_marker` byte-binds the sovereign YAML shape via `include_str!`); full discharge blocks on real trained 370M .apr + 3 seed=0 `apr bench --tokens 128 --json` medians on RTX 4090; **MODEL-2 coverage 5/12 → 6/12** touched — MODEL-2 twin of MODEL-1 SHIP-007 (identical f32-threshold shape, floor 100 vs 30 tok/s since the 370M target is ~3.5× smaller than the 7B Q4_K teacher); 19 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.36.0 **FALSIFY-SHIP-018 PARTIAL_ALGORITHM_LEVEL (restacked)** — task #151 wires MODEL-2 `apr eval --benchmark humaneval` pass@1 ship floor on top of the v2.35.0 SHIP-020 restack: `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.7.0 → v1.8.0 (stays ACTIVE) adds `GATE-ARCH-370M-007` binding AC-SHIP2-008 ↔ FALSIFY-SHIP-018 with `discharge_status: PARTIAL_ALGORITHM_LEVEL`; decision rule is a pure (correct, total, threshold_pct) → Pass/Fail comparison — "HumanEval pass@1 ≥ 30.0% on 164 tasks" — bound in `crates/aprender-train/src/models/llama_370m.rs` via `AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT = 30.0` + `verdict_from_pass_at_1(correct, total, threshold_pct) -> Ship018Verdict` + 2 falsification tests (`falsify_ship_018_humaneval_pass_at_1_threshold_logic` covers inclusive floor (30/100, 60/200, 50/164 = 30.49% all Pass) + just-below Fail (49/164 ≈ 29.88%, 29/100 = 29.0%) + inclusive-floor proof at f32-exact 50/100 with ±ULP asymmetry + generous-green {82/164, 164/164} + hard-red {0/164, 1/164} + monotonicity sweep correct∈[0,164] + div-safety total=0 Fail + correct>total sanity Fail + non-finite threshold {NaN, ±∞} conservative Fail + provenance pin 30.0; `falsify_ship_018_gate_arch_370m_007_has_partial_discharge_marker` byte-binds the sovereign YAML shape via `include_str!`); full discharge blocks on real trained 370M .apr + 3 independent `apr eval --benchmark humaneval --json` seed=0 medians each feeding `verdict_from_pass_at_1`; **MODEL-2 coverage 6/12 → 7/12** touched — MODEL-2 twin of MODEL-1 SHIP-005 (pass@1 threshold; MODEL-2 has 30.0% floor with no noise allowance vs MODEL-1's 86.0% ± 1.2pp because the 370M target is a trained-from-scratch student, not a distilled teacher); 20 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.37.0 **FALSIFY-SHIP-016 PARTIAL_ALGORITHM_LEVEL (restacked)** — task #152 wires MODEL-2 `apr qa <model>.apr` 8-gate aggregate gate on top of the v2.36.0 SHIP-018 restack: `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.8.0 → v1.9.0 (stays ACTIVE) adds `GATE-ARCH-370M-008` binding AC-SHIP2-006 ↔ FALSIFY-SHIP-016 with `discharge_status: PARTIAL_ALGORITHM_LEVEL`; decision rule is a pure aggregate-AND over 8 Boolean gate-results (golden_output / throughput / ollama_parity / gpu_vs_cpu_speedup / tensor_contract / cross_format_parity / ptx_parity / probar) bound in `crates/aprender-train/src/models/llama_370m.rs` via `AC_SHIP2_006_REQUIRED_QA_GATE_COUNT = 8` + `verdict_from_qa_gates(&[bool]) -> Ship016Verdict` + 2 falsification tests (`falsify_ship_016_apr_qa_aggregate_and_logic` covers exhaustive 2^8 = 256-combination proof (exactly 1 input yields Pass, 255 yield Fail) + 8-way single-gate-flip falsifiability + monotonicity + 4 contract-drift guards (length {0, 7, 9, 16} → Fail even when all-true) + provenance pin; `falsify_ship_016_gate_arch_370m_008_has_partial_discharge_marker` byte-binds the sovereign YAML shape via `include_str!`); full discharge blocks on real trained 370M .apr + exit-0 `apr qa <model>.apr` with all 8 gates PASS; **MODEL-2 coverage 7/12 → 8/12** touched — first MODEL-2 aggregate-AND PARTIAL (mirrors MODEL-1 SHIP-006's shape exactly, since both models gate on the same 8-of-8 `apr qa` aggregate); this completes the compute-free MODEL-2 PARTIAL harvest — remaining 4 MODEL-2 gates are all genuinely compute-bound (003 val loss / 004 wall-clock / 005 already covered / 009 handled via AC-SHIP2-009 from SHIP-019); 21 PARTIAL + 3 DISCHARGED across both models) / 2026-04-23 (v2.38.0 **FALSIFY-SHIP-013 + FALSIFY-SHIP-014 BUNDLED PARTIAL_ALGORITHM_LEVEL** — task #118 discharges the last two untouched MODEL-2 AC rows (AC-SHIP2-003 val CE ≤ 2.2 + AC-SHIP2-004 training ≤ 21 days on RTX 4090) as a SINGLE bundled PR on top of the v2.37.0 SHIP-016 restack: `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.9.0 → v1.10.0 (stays ACTIVE) adds TWO new gates in one bump — GATE-ARCH-370M-013 binding AC-SHIP2-003 ↔ FALSIFY-SHIP-013 via pure f32 threshold fn `verdict_from_val_ce_loss(f32) -> Ship013Verdict` + const `AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS = 2.2` + 7-section mutation survey (exact 2.2 boundary Pass inclusive-floor / ULP-above Fail + ULP-below Pass asymmetry / clear Pass band {0.0, 0.5, 1.0, 2.0, 2.199} / clear Fail band {2.201, 3.0, 10.0, f32::MAX} / non-finite {NaN, ±∞} conservative Fail / negative-CE domain-violation Fail because H(p,q) ≥ 0 by definition / 2.2 provenance pin); and GATE-ARCH-370M-014 binding AC-SHIP2-004 ↔ FALSIFY-SHIP-014 via pure u32 threshold fn `verdict_from_training_duration_days(u32) -> Ship014Verdict` + const `AC_SHIP2_004_MAX_TRAINING_DURATION_DAYS = 21` + 6-section mutation survey (exact 21 boundary Pass inclusive-ceiling / adjacent 20→Pass + 22→Fail / clear Pass band {0, 1, 7, 14, 20, 21} / clear Fail band {22, 30, 100, u32::MAX} / monotonicity sweep 0..=42 flipping exactly once at 21→22 / 21 provenance pin); both verdict fns colocated in `crates/aprender-train/src/models/llama_370m.rs`; `cargo test -p aprender-train --lib ship_013` + `cargo test -p aprender-train --lib ship_014` each green (1/1); full discharge of SHIP-013 blocks on live `apr pretrain --mode from-scratch --validate` loop on RTX 4090 with `--features cuda` producing a real MODEL-2 val cross-entropy at the final validation step; full discharge of SHIP-014 blocks on real wall-clock measurement of a MODEL-2 pretraining run from first `apr pretrain` dispatch to final checkpoint write, converted to integer days; **MODEL-2 coverage 8/12 → 12/12 PARTIAL_ALGORITHM_LEVEL touched — completes MODEL-2** — this is the first bundled double-discharge on the SHIP-TWO-001 surface, lifting the last two genuinely compute-bound gates (SHIP-013 and SHIP-014 both block on actual pretraining compute) to PARTIAL at the decision-rule level ahead of the AC-SHIP2-003/004 compute-dispatch; across both models: 23 PARTIAL + 3 DISCHARGED) / 2026-04-23 (v2.39.0 spec hygiene — back-annotates §5.2 MODEL-2 AC table (6 rows: 3 DISCHARGED + 3 PARTIAL_ALGORITHM_LEVEL from v2.20/21/22 amendments) + §7.1/§7.2 Falsification Tests (12 new PARTIAL cross-references) so the tables are a true single source of truth for SHIP-TWO-001 algorithm-level coverage; task #119; no Rust/contract/test changes) / 2026-04-23 (v2.40.0 **FALSIFY-SHIP-023 + FALSIFY-SHIP-024 BUNDLED PARTIAL_ALGORITHM_LEVEL** — task #120 discharges the last two MODEL-1 §7.1 stability rows (FALSIFY-SHIP-023 cross-run score drift + FALSIFY-SHIP-024 adversarial-suite 0-tolerance) as a SINGLE bundled PR on top of the v2.39.0 back-annotation: `contracts/qwen2-e2e-verification-v1.yaml` v1.6.0 → v1.7.0 adds TWO new falsification_tests in one bump — FALSIFY-QW2E-SHIP-023 binding `verdict_from_score_drift(day1_pct, day2_pct, tolerance_pp) -> Ship023Verdict` in `crates/aprender-core/src/format/ship_023.rs` at `AC_SHIP1_023_MAX_HUMANEVAL_DRIFT_PP = 1.2` + 7-section mutation survey (exact 1.2 boundary Pass / drift = 1.2 + 1e-4 Fail / clear Pass band {0, 0.5, 1.0, 1.199} / clear Fail band {1.3, 2.0, 10.0, 86.0} / symmetric `.abs()` order invariance / non-finite {NaN, ±∞} conservative Fail / out-of-range day values {−0.1, 100.1} + negative-tolerance + zero-tolerance boundary + degenerate 0.0/100.0 cases + provenance pin); and FALSIFY-QW2E-SHIP-024 binding `const fn verdict_from_adversarial_suite(inputs_run: usize, panic_count: u32, nan_count: u32) -> Ship024Verdict` in `crates/aprender-core/src/format/ship_024.rs` at `AC_SHIP1_024_MIN_ADVERSARIAL_SUITE_SIZE = 50` + `AC_SHIP1_024_MAX_TOLERATED_PANIC_COUNT = 0` + `AC_SHIP1_024_MAX_TOLERATED_NAN_COUNT = 0` + 7-section mutation survey (zero-tolerance boundaries (50,0,0) Pass + (50,1,0) (50,0,1) Fail / insufficient suite {0, 49, 1, 10, 25, 40} Fail / over-size Pass band {51, 100, 200, 500, 1000, 10_000, usize::MAX} / single-failure-class counts {panic∈[1,1000], nan∈[1,1000]} / compound failures {(50,1,1), (50,5,5), (10,10,10)} / u32::MAX overflow guard / all-three-constants provenance pin); `cargo test -p aprender-core --lib format::ship_023` + `cargo test -p aprender-core --lib format::ship_024` each green (1/1); full discharge of SHIP-023 blocks on live 2-day `apr eval --benchmark humaneval paiml/qwen2.5-coder-7b-apache-q4k-v1 --json` re-run on RTX 4090 with `--features cuda` + `(day1 − day2).abs() ≤ 1.2`; full discharge of SHIP-024 blocks on real 50-prompt adversarial torture suite feeding `verdict_from_adversarial_suite` with panic and NaN-logit counters; **completes MODEL-1 §7.1 at 12/12 falsification tests algorithmically bound** — SHIP-023 + SHIP-024 are the FIRST non-ship-blocking PARTIAL levers on the SHIP-TWO-001 surface because they live in §7.1 stability tests, not §4.2 AC table; SHIP-023's 1.2 pp budget is intentionally shared with SHIP-005's `AC_SHIP1_005_NOISE_ALLOWANCE_PP` (same budget, different semantics: noise vs nominal vs drift between two runs); across both models: 25 PARTIAL + 3 DISCHARGED) / 2026-04-23 (v2.41.0 **FALSIFY-GPUTRAIN-003..007 BUNDLED PARTIAL_ALGORITHM_LEVEL** — task #121 wires §14 Phase 2 algorithm-level discharges for the last five unbound FALSIFY-GPUTRAIN tests (001 and 002 already bound in `crates/aprender-train/src/train/device.rs` at 17 tests): `contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 PROPOSED → v1.1.0 PROPOSED (stays PROPOSED until Phase 3 live evidence) adds FIVE new `evidence_discharged_by` + `full_discharge_blocks_on` + `counter_example_classes` blocks on one bump — FALSIFY-GPUTRAIN-003 binding `nvidia-smi --query-compute-apps` parser + residency verdict via pure `parse_nvidia_smi_compute_apps(&str) -> Result<Vec<NvidiaSmiComputeApp>, ()>` + `verdict_from_residency(u32, &[NvidiaSmiComputeApp]) -> Gputrain003Verdict` bound to `AC_GPUTRAIN_003_NVIDIA_SMI_POLL_WINDOW_SECONDS = 5` + `AC_GPUTRAIN_003_MIN_USED_MEMORY_MIB = 1` (7-section survey: happy path / zero-mem Fail / other-pid Fail / empty Fail / multi-process both orderings Pass / malformed lines parse-Err / u32::MAX-u64::MAX boundary / dual provenance pin); FALSIFY-GPUTRAIN-004 binding device-class dispatch invariant via pure `verdict_from_dispatch_label(&str, &str) -> Gputrain004Verdict` bound to disjoint `AC_GPUTRAIN_004_CPU_DISPATCH_VARIANTS = &["Cpu"]` + `AC_GPUTRAIN_004_CUDA_DISPATCH_VARIANTS = &["Cuda"]` (7-section survey: cpu→cpu Pass / cuda→cuda Pass / cpu→cuda silent-promotion Fail / cuda→cpu task-#126 silent-fallback Fail / unknown labels Fail / empty string Fail / case-sensitivity `CPU` vs `Cpu` Fail / disjointness proof); FALSIFY-GPUTRAIN-005 binding 500-ms step-time ceiling via pure `const fn verdict_from_step_time_ms(f32, f32) -> Gputrain005Verdict` bound to `AC_GPUTRAIN_005_MAX_STEP_TIME_MS_RTX4090_370M = 500.0` (7-section survey: exact 500.0 inclusive-ceiling Pass / one-ULP above Fail / clear Pass band / clear Fail band incl. f32::MAX / non-finite Fail / negative measured + non-positive threshold Fail / provenance pin); FALSIFY-GPUTRAIN-006 binding same-device seed reproducibility via pure `const fn verdict_from_loss_delta(f32, f32)` + aggregate `fn verdict_from_loss_trajectories(&[f32], &[f32], f32)` bound to `AC_GPUTRAIN_006_MAX_SEED_LOSS_DELTA = 1e-5` (7-section survey: exact boundary Pass / ULP-above Fail / trajectory single-step-fail at k=42 / length mismatch 50-vs-100 Fail / empty Fail / NaN+∞ Fail / negative delta + negative tolerance + infinite tolerance Fail / provenance pin); FALSIFY-GPUTRAIN-007 binding `apr --version --json` schema + field-shape invariants via pure `fn verdict_from_version_json_keys(&[&str])` + `fn verdict_from_version_json_fields(&VersionJsonCudaFields)` bound to `AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS = &["cuda_feature", "cuda_runtime_available", "visible_devices"]` (7-section survey: all-keys-present Pass + extras tolerated / missing-each-key Fail × 3 / 3 valid (feature, runtime) combos Pass / claims-feature-without-runtime Fail (FM-GPUTRAIN-STALE-BUILD) / boundary exactly 16 Pass, 17 Fail, 100 Fail / combined happy path / provenance pin); `cargo test -p aprender-train --lib gputrain_003` + `gputrain_004` + `gputrain_005` + `gputrain_006` + `gputrain_007` each green (1/1), `cargo test -p aprender-train --lib device` still green (17/17) regression-clean; full discharge of every GPUTRAIN gate still blocks on the live lambda-labs RTX 4090 harness per §14 Phase 3 (nvidia-smi residency proof + 50-step median timing + 100-step cuda:0 seed=0 replay + `apr --version --json` dogfood); **§14 Phase 0 algorithm-level complete — 7/7 FALSIFY-GPUTRAIN tests now bound at or above PARTIAL_ALGORITHM_LEVEL** (GPUTRAIN-001/002 already bound in device.rs); promotes `gpu-training-backend-v1` v1.0.0 PROPOSED → v1.1.0 PROPOSED (stays PROPOSED until §14 Phase 3 lambda-labs RTX 4090 residency evidence lands and flips the C-GPUTRAIN-BACKEND status to ACTIVE per §14 Phase 4); across both models: 30 PARTIAL + 3 DISCHARGED) / 2026-04-23 (v2.42.0 **§6 Compound Ship Gates bundle — 6 PARTIAL_ALGORITHM_LEVEL bindings** — task #122 algorithmically binds the 6 bindable §6 compound ship gates (GATE-SHIP-001..006; 007-012 are CI/lint meta-policy and remain out of scope) via a NEW `contracts/compound-ship-gates-v1.yaml` v1.0.0 PROPOSED (kind: pattern, cross-cutting CompoundShipGatesContract) and 6 new sibling modules in `crates/aprender-core/src/format/gate_ship_00X.rs`: GATE-SHIP-001 binds `verdict_from_model1_ac_aggregate(&[bool]) -> GateShip001Verdict` to `AC_GATE_SHIP_001_MODEL_1_AC_COUNT = 10` via aggregate-AND + 6-section survey incl. exhaustive 2^10 = 1024 bitmask proof; GATE-SHIP-002 binds `verdict_from_model2_ac_aggregate(&[bool]) -> GateShip002Verdict` to `AC_GATE_SHIP_002_MODEL_2_AC_COUNT = 12` via aggregate-AND + 6-section survey incl. exhaustive 2^12 = 4096 bitmask proof; GATE-SHIP-003 binds `verdict_from_golden_output_diff(&[u8], &[u8]) -> GateShip003Verdict` via byte-identity + non-empty conservative Fail + 6-section survey (identical / length mismatch / single-byte flip / both-empty Fail / one-empty Fail / 10_000-byte stress); GATE-SHIP-004 binds `verdict_from_identical_humaneval_scores(f32, f32) -> GateShip004Verdict` via `to_bits()` bitwise-identity (STRICTLY STRICTER than FALSIFY-SHIP-023 1.2 pp drift) + 7-section survey (identical Pass / single-ULP Fail / close-but-not-equal Fail / non-finite Fail / out-of-range Fail / boundary 0.0 + 100.0 Pass / SHIP-023 contrast pin); GATE-SHIP-005 binds `verdict_from_license_metadata(&str, &str) -> GateShip005Verdict` to `AC_GATE_SHIP_005_REQUIRED_LICENSE_FIELD = "license"` via case-sensitive byte-equal + ASCII-printable guard + 6-section survey (happy path / case drift / empty / non-ASCII incl. NUL + BOM + tab + newline / trailing whitespace / provenance pin); GATE-SHIP-006 binds `const fn verdict_from_first_token_probability_delta(f32, f32, f32) -> GateShip006Verdict` to `AC_GATE_SHIP_006_MAX_FIRST_TOKEN_DELTA = 1e-3` via symmetric `.abs()` threshold + 7-section survey (delta=0 Pass / exact-tolerance Pass / one-ULP-over Fail / Pass band small probs + symmetric / out-of-range Fail / negative tolerance Fail / non-finite Fail / provenance pin); `cargo test -p aprender-core --lib format::gate_ship` green (6/6) + `cargo test -p aprender-core --doc format::gate_ship` green (6/6 doctests); `pv validate contracts/compound-ship-gates-v1.yaml` → 0 errors, 0 warnings; full discharge of each gate blocks on live compound-gate harness (all 10 per-AC MODEL-1 checks for GATE-SHIP-001 / all 12 per-AC MODEL-2 checks for GATE-SHIP-002 / `apr qa --golden-output` on pre+post quantize checkpoints for GATE-SHIP-003 / two consecutive `apr eval --seed 0` runs in the same session for GATE-SHIP-004 / `apr inspect .metadata.license` vs upstream HF card YAML for GATE-SHIP-005 / `apr run --emit-logprobs` + `apr export --format gguf` + `llama-cli --logits-all` first-token softmax diff for GATE-SHIP-006); promotes NEW compound-gates contract to PROPOSED; across both models: 36 PARTIAL + 3 DISCHARGED) / 2026-04-23 (v2.43.0 **§6 Compound Ship Gates meta-policy bundle — 6 PARTIAL_ALGORITHM_LEVEL bindings (12/12 §6 total)** — task #123 completes §6 by algorithmically binding the 6 merge-gate meta-policy rows (GATE-SHIP-007..012 — unwrap / contract-density / CI-green / deny / tdg / coverage) via `contracts/compound-ship-gates-v1.yaml` v1.0.0 → v1.1.0 (stays PROPOSED) and 6 new sibling modules in `crates/aprender-core/src/format/gate_ship_0XX.rs`: GATE-SHIP-007 binds `const fn verdict_from_unwrap_count(u32) -> GateShip007Verdict` to `AC_GATE_SHIP_007_MAX_TOLERATED_UNWRAP_COUNT = 0` via zero-tolerance threshold + 5-section survey (count=0 Pass / count=1 Fail / Fail band {2, 10, 100, u32::MAX} / monotonicity sweep 0..=256 / provenance pin); GATE-SHIP-008 binds `verdict_from_contract_density(u32, u32, f32) -> GateShip008Verdict` to `AC_GATE_SHIP_008_MIN_CONTRACT_DENSITY_NEW_CODE = 1.0` via divide-by-zero-guarded ratio threshold + 7-section survey (10/10 @ 1.0 Pass / 9/10 @ 1.0 Fail / 10/10 @ 0.9 Pass / zero-total Fail / contracted > total sanity Fail / non-finite or OOR density Fail / provenance pin); GATE-SHIP-009 binds `const fn verdict_from_ci_aggregate(bool, bool, bool) -> GateShip009Verdict` to `AC_GATE_SHIP_009_REQUIRED_CHECK_COUNT = 3` via aggregate-AND + 8-section survey incl. exhaustive 2^3 = 8 bitmask proof + AND-symmetry across argument permutations + provenance pin on 3-count; GATE-SHIP-010 binds `const fn verdict_from_advisory_count(u32) -> GateShip010Verdict` to `AC_GATE_SHIP_010_MAX_TOLERATED_ADVISORY_COUNT = 0` via zero-tolerance threshold + 5-section survey mirroring GATE-SHIP-007 at the security-advisory semantic layer; GATE-SHIP-011 binds `const fn verdict_from_tdg_score(f32, f32) -> GateShip011Verdict` to `AC_GATE_SHIP_011_MIN_PMAT_TDG_SCORE = 90.0` via inclusive-floor threshold + 7-section survey (exact 90.0 Pass / ULP-below Fail / clear Pass band {95, 100, 92.5} / clear Fail band {89.9, 80, 0, 50} / non-finite Fail / negative measured + threshold range guards / provenance pin); GATE-SHIP-012 binds `const fn verdict_from_line_coverage_pct(f32, f32) -> GateShip012Verdict` to `AC_GATE_SHIP_012_MIN_LINE_COVERAGE_PCT = 95.0` via inclusive-floor threshold + 7-section survey mirroring GATE-SHIP-011 at the line-coverage 95% floor; `cargo test -p aprender-core --lib format::gate_ship_0XX` green (6/6) + `cargo test -p aprender-core --doc format::gate_ship_0XX` green (6/6 doctests); `pv validate contracts/compound-ship-gates-v1.yaml` → 0 errors, 0 warnings; full discharge of each gate blocks on live CI tooling invocation (`cargo clippy --all-targets --all-features -- -D warnings` for GATE-SHIP-007; `pmat density --new-code --json` for GATE-SHIP-008; branch-protection `ci / gate` + `workspace-test` for GATE-SHIP-009; `cargo deny check advisories` for GATE-SHIP-010; `pmat tdg . --format json` for GATE-SHIP-011; `cargo llvm-cov report --json` for GATE-SHIP-012); **§6 Compound Ship Gates now 12/12 algorithmically bound** (6 ship-blocking 001..006 already covered by v2.42.0; 6 merge-gate 007..012 added this release); across both models: **42 PARTIAL + 3 DISCHARGED**). / 2026-04-24 (v2.44.0 **drift-prevention enforcement layer codified** — tasks #128 + #129 close the specification loop by pinning every quantitative README claim AND every `AC_*` threshold constant to a falsifiable re-derivation, so silent edits to either the narrative or the verdict-fn thresholds fail CI instead of shipping. No new PARTIAL levers — this is the meta-layer that keeps the 42-lever surface honest over time. TWO deliverables landed in PRs #1044 + #1046 off main: (1) `contracts/readme-claims-v1.yaml` v1.0.0 `kind: pattern` `status: enforced` + FALSIFY-README-001..004 + `scripts/check_readme_claims.sh` — binds the README's workspace crate count (`ls crates/`), provable contract count (`find contracts/ -name '*.yaml'`), CLI command count (`cargo run -p apr-cli --bin apr -- --help` subcommand lines), and apr-cookbook link presence (`grep -F 'apr-cookbook'`) to live repository state; `--regen` mode prints the current values for quick README edit, `--claim <name>` targets a single check; bashrs-clean with documented SC1020/SC1140/SC1009/SC2102 false-positive suppressions for regex-bracket parsing; `bash scripts/check_readme_claims.sh` → `PASS FALSIFY-README-001..004` on HEAD (80 / 1096 / 79 / present); (2) `crates/aprender-train/tests/ship_two_001_const_pinning.rs` — a 345-line integration test that imports every `AC_*` const across the 27 SHIP-TWO-001 verdict modules (4 crates) and asserts each value matches the spec section-by-section: 20 MODEL-1 consts (§4.2 AC-SHIP1-001..010 + §7.1 SHIP-023/024), 7 MODEL-2 consts (§5.2 AC-SHIP2-003..010), 10 compound-gate consts (§6 GATE-SHIP-001..012), 8 GPUTRAIN consts (§14 FALSIFY-GPUTRAIN-003..007); 44 value assertions + 1 tripwire meta-test (`pinned_const_count_tripwire` checks count stays synced to `^pub const AC_` across `crates/`); catches the class where a contributor edits `AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS` from 2.2 → 2.5 without touching the spec — every verdict fn still compiles, every unit test passes, but the ship floor silently drifts; `cargo test -p aprender-train --test ship_two_001_const_pinning` → 44 passed. Pairs with the README contract to close the loop: README contract guards narrative drift, const-pinning test guards threshold drift, `pv validate` guards contract-schema drift — all three fail CI on violation, all three re-derive from live repo state instead of frozen snapshots. Across both models: **42 PARTIAL + 3 DISCHARGED** (numbers unchanged from v2.43.0 — this is enforcement, not coverage; the 42 levers are now *durable* against silent drift). / 2026-04-24 (v2.45.0 **atomic-next-action clarity: Task #132 Phase 2 runtime wiring** — task #131. No code changes; 2-line spec edit (Version + new `**Atomic next action**` callout at top of header, ahead of Status line). Surfaces the distinction §14.5 already encodes but that was only discoverable by reading the full §14: Task #132 Phase 2 splits into TWO surfaces — **algorithm-level** (FALSIFY-GPUTRAIN-003..007 bound at PARTIAL_ALGORITHM_LEVEL, shipped v2.41.0, status: DONE) and **runtime wiring** (`SharedTrainer::CudaVariant` + `drive_real` dispatch glue to the existing `CudaTransformerTrainer`, status: PENDING, ~2 days Rust). Without runtime wiring, `apr pretrain --device cuda:0` silently runs on CPU — the exact Task #132 bug. With it, one lambda-labs dispatch promotes SHIP-013/014/016/017/018/020 from PARTIAL → DISCHARGED on live MODEL-2 weights, and MODEL-1 live-compute harnesses (apr convert q4_k_m / humaneval / apr qa) can run in parallel on the same host. After v2.44.0 made the 42-lever surface *durable*, this amendment makes the *next action* discoverable. Across both models: **42 PARTIAL + 3 DISCHARGED** (unchanged — still enforcement-layer, not coverage-layer).

**v2.21.0 amendment (2026-04-19):** Three MODEL-2 architecture + tokenizer
gates landed in the same post-v2.19 evidence window, on branch
`chore/post-v2.19-evidence`:

1. **FALSIFY-SHIP-011 (AC-SHIP2-001) — DISCHARGED** at commit `338c6eb3c`
   (task #114). `contracts/model-families/llama-370m-sovereign-v1.yaml`
   promoted v1.0.0 PROPOSED → v1.1.0 ACTIVE. Rust scaffold
   `Llama370MConfig` (crates/aprender-train/src/models/llama_370m.rs) now
   binds **byte-equally** to the YAML contract via the harness test
   `falsify_ship_011_rust_scaffold_matches_yaml_contract`, which uses
   `include_str!` to embed the contract at compile time and
   `serde_yaml::Value` to parse-and-compare every architecture.* and
   constraints.* field against the corresponding `Llama370MConfig::*`
   const. Any edit to either side that diverges fails
   `cargo test -p aprender-train --lib llama_370m` before a single step
   of compute runs. INV-ARCH-370M-002..008 remain enforced at compile
   time via `const _: () = Llama370MConfig::validate();`, so the
   compile-time tier is intact even without the new YAML-binding test.
   The deliberate *sibling* approach over amending `llama.yaml` with a
   `370m` entry is recorded in the discharge memo: albor's
   `tied_embeddings=true` and `rope_theta=10000.0` conflict with
   Meta Llama-3's family-wide `tied_embeddings=false` /
   `rope_theta=500000.0`, and GATE-ARCH-370M-001's
   "llama.yaml (or this sibling contract)" language explicitly permits
   it.

2. **FALSIFY-SHIP-012 (AC-SHIP2-002) — PARTIAL_ALGORITHM_LEVEL** at
   commit `2e8b8b8e2` (task #115). `contracts/tokenizer-bpe-v1.yaml`
   bumped v1.0.0 → v1.1.0, **status intentionally stays PROPOSED**.
   GATE-BPE-003 gains `evidence_discharged_by` pointing at 3 harness
   tests in
   `crates/apr-cli/tests/falsify_ship_012_tokenizer_roundtrip.rs`:
   byte-exact round-trip on a 20-doc Python-like holdout (ASCII
   keywords + Unicode identifiers + docstrings + emoji + combining
   marks) under `aprender::text::tokenize::BpeTokenizer`, standalone
   NFC idempotence (INV-BPE-005), and train/holdout disjointness. The
   gate's `evidence_required` explicitly asks for **10K** docs; the
   current harness runs 20 on a synthetic fixture, so the gate lands
   with `discharge_status: PARTIAL_ALGORITHM_LEVEL` and
   `full_discharge_blocks_on: "task #91 (10K The Stack v2 Python
   holdout)"`. The harness module doc-comment locks in the zero-rewrite
   swap path: when task #91's 10K corpus materializes, replacing
   `HOLDOUT_CORPUS` + `TRAIN_CORPUS` with shard readers is a data-only
   change, then the contract can bump to 2.0.0 and promote to ACTIVE.
   This is the first spec-level use of a PARTIAL gate inside a
   PROPOSED contract — the pattern is: if the algorithm is provable
   today but the production-scale evidence is deferred, wire the
   algorithm proof and surface the data gap as first-class contract
   state rather than leaving the `evidence_discharged_by` slot blank.

3. **FALSIFY-SHIP-015 (AC-SHIP2-005) — PARTIAL_ALGORITHM_LEVEL** at
   commit `bfb883199` (task #116). Sovereign contract v1.1.0 → v1.2.0,
   stays ACTIVE. GATE-ARCH-370M-003 gains `evidence_discharged_by`
   pointing at the pre-existing `estimated_param_count_within_contract_band`
   unit test plus the `estimated_param_count` /
   `estimated_stored_param_count` const fns in
   `crates/aprender-train/src/models/llama_370m.rs`. The gate's
   `evidence_required` asks for `apr inspect --json model.apr |
   jq '.param_count'` to yield an integer in [366_000_000,
   374_000_000]; no on-disk `.apr` exists pre-compute, so the gate
   lands with `discharge_status: PARTIAL_ALGORITHM_LEVEL` and
   `full_discharge_blocks_on: "real 370M .apr checkpoint from
   pretraining compute-dispatch (AC-SHIP2-003/004)"`. The unit test
   hard-asserts p ∈ [366M, 374M], |p − 370M|/370M < 5%, and that
   embedding tying reduces stored params by exactly
   VOCAB_SIZE × HIDDEN_DIM; any edit to `Llama370MConfig` that moves
   the count out of the INV-ARCH-370M-001 band fails
   `cargo test -p aprender-train --lib llama_370m` before any compute
   runs. Contract remains ACTIVE because SHIP-011 (not SHIP-015) is
   what gates the sovereign contract's ACTIVE promotion — a gate-level
   PARTIAL nested inside an ACTIVE contract is a valid shape.

**Pattern codified by v2.21.0 (PARTIAL_ALGORITHM_LEVEL):** when a gate's
`evidence_required` text describes a production-scale check (10K docs,
on-disk artifact, benchmark run) that is not yet runnable, but the
underlying invariant is provable at algorithm / compile / unit-test
level today, emit the gate with `evidence_discharged_by` listing the
algorithm proofs + `discharge_status: PARTIAL_ALGORITHM_LEVEL` +
`partial_discharge_note:` + `full_discharge_blocks_on:` +
`ship_blocking: true`. The last field is load-bearing: PARTIAL gates
MUST still block `apr publish` until full discharge lands. Downstream
auditors must treat `evidence_discharged_by` alone (without checking
`discharge_status`) as **not** sufficient green — the two fields
together are the authoritative read.

**v2.24.0 amendment (2026-04-22):** First **MODEL-1** PARTIAL discharge
lands on branch `feat/falsify-ship-009-partial-discharge` — and it does
so by reusing the exact contract that already discharges MODEL-2's
AC-SHIP2-012, proving the decision-rule / compute-harness separation
pattern extends across model families:

1. **FALSIFY-SHIP-009 (AC-SHIP1-009) — PARTIAL_ALGORITHM_LEVEL** (task
   #153). `contracts/apr-provenance-v1.yaml` v1.0.0 → v1.1.0, status
   **stays ACTIVE** (no downgrade — v1.0.0's MODEL-2 discharge is
   unaffected). A new `GATE-APR-PROV-004` block is appended alongside
   GATE-APR-PROV-001/002/003, binding `AC-SHIP1-009` / `FALSIFY-SHIP-009`
   with `discharge_status: PARTIAL_ALGORITHM_LEVEL`. The gate's rule
   declares that the SAME AprV2Metadata + serde-JSON decision rule
   proved for MODEL-2 also applies to MODEL-1, and the two new harness
   tests in `crates/aprender-core/src/format/tests/provenance_tests.rs`
   confirm it:

   - `falsify_ship_009_apr_metadata_applies_to_model_1_teacher` —
     constructs a teacher-representative `AprV2Metadata { license:
     Some("apache-2.0"), data_source:
     Some("qwen2.5-coder-7b-instruct"), data_license:
     Some("apache-2.0"), .. }`, round-trips it through `to_json` /
     `from_json`, and asserts byte-identical recovery of all three
     provenance fields with NO leak into the `custom` HashMap.
   - `falsify_ship_009_gate_apr_prov_004_has_partial_discharge_marker`
     — `include_str!()`s the updated contract YAML, parses via
     `serde_yaml`, and asserts the new gate has
     `binds_to: AC-SHIP1-009`,
     `falsification_id: FALSIFY-SHIP-009`,
     `discharge_status: PARTIAL_ALGORITHM_LEVEL`,
     `ship_blocking: true`, non-empty `evidence_discharged_by`, and
     both `partial_discharge_note` and `full_discharge_blocks_on` set.

   The algorithm is model-agnostic — only the input string values
   differ between the MODEL-1 teacher test and the MODEL-2 SHIP-022
   test. `full_discharge_blocks_on: "teacher .apr republish (planned
   regen lane, PMAT-686) with AprV2Metadata.license / data_source /
   data_license populated as named fields; apr inspect --json must
   emit each as non-null. Fixture-swap only — no code change, no
   contract change."` The teacher is already shipped on HF
   (paiml/qwen2.5-coder-7b-apache-q4k-v1) under Apache-2.0, distilled
   from Qwen2.5-Coder-7B-Instruct — the provenance fact is known; only
   embedding it in a .apr binary remains.

**Pattern extensions from v2.24.0:**

- **First MODEL-1 PARTIAL.** All prior algorithm-level PARTIAL work
  (SHIP-015, -017, -018, -019, -020, -016) targeted MODEL-2 ship
  gates. SHIP-009 is the first MODEL-1 gate to be PARTIAL-discharged
  via the decision-rule/compute-harness split pattern, demonstrating
  the pattern is not MODEL-2-specific.
- **First multi-model multi-bind on ONE contract.** Prior PARTIAL
  discharges each had their own contract. SHIP-009 attaches a second
  gate to `apr-provenance-v1` — a second binding to AC-SHIP1-009 sits
  alongside the existing v1.0.0 bindings to AC-SHIP2-012. Because the
  decision rule is literally the same code, one YAML file cleanly
  carries both discharges with no schema extension.
- **"Exhausted" verdict now falsified SIX times.** Counter-example
  hunting after v2.22.0's "exhausted" verdict has now found genuine
  PARTIAL levers six times: SHIP-019 → SHIP-017 → SHIP-020 → SHIP-018
  → SHIP-016 → SHIP-009. The sixth falsification is cross-model (a
  MODEL-1 gate discharged by a MODEL-2 contract) — strictly more
  surprising than the prior five.

Combined ship-gate status after v2.24.0:

- **MODEL-2:** 3/12 fully ACTIVE (001, 011, 012) + 7/12 PARTIAL (002,
  005, 006, 007, 008, 009, 010) = **10/12 touched (83.3%)**. Remaining
  2 truly compute-blocked: AC-SHIP2-003 (val loss ≤ 2.2), AC-SHIP2-004
  (≤21-day wall-clock).
- **MODEL-1:** 1/10 PARTIAL (009), with the remaining 9 gates already
  DISCHARGED via the tagged teacher release
  `SHIP-TWO-001-MODEL-1-TEACHER`. MODEL-1 provenance will flip to
  fully ACTIVE the moment teacher.apr republish (PMAT-686) populates
  the three named fields.

**v2.22.0 amendment (2026-04-19):** One additional MODEL-2 ship gate
attained PARTIAL_ALGORITHM_LEVEL in the same post-v2.19 evidence window,
on branch `chore/post-v2.19-evidence`:

4. **FALSIFY-SHIP-019 (AC-SHIP2-009) — PARTIAL_ALGORITHM_LEVEL** at
   commit `846cc1dbb` (task #117). Sovereign contract v1.2.0 → v1.3.0,
   stays ACTIVE. GATE-ARCH-370M-004 gains `evidence_discharged_by`
   pointing at two new harness tests + an enumerator helper in
   `crates/aprender-train/src/models/llama_370m.rs` plus three
   cross-referenced assets (`LayoutContract`, `validate_apr_shape`,
   `contracts/tensor-layout-v1.yaml`). The gate's `evidence_required`
   asks for GGUF-exported 370M first-token cosine similarity ≤ 1e-3 vs
   APR on 100 canary prompts — that runner is blocked on
   AC-SHIP2-003/004 pretraining compute plus GATE-SHIP-006 harness
   invocation, so the gate lands with `discharge_status:
   PARTIAL_ALGORITHM_LEVEL` + `full_discharge_blocks_on: "real 370M .apr
   checkpoint from pretraining compute-dispatch (AC-SHIP2-003/004) +
   harness invocation of GATE-SHIP-006 cosine-parity runner"`. The
   algorithm-level proofs collectively establish the conditional: *if*
   GGUF export invokes `LayoutContract::validate_apr_shape` on every
   tensor, *then* row-major layout and GH-202 regression rejection are
   mathematically enforced. The enumerator counts
   **3 + 9 × NUM_LAYERS = 219** tensors and cross-checks each with
   `LayoutContract::get_apr_contract`; adding a tensor to
   `Llama370MConfig` without a matching entry in
   `layout_contract_specs.rs` now fails
   `cargo test -p aprender-train --lib llama_370m` before any compute
   runs. Spec §9 Risk #2's explicit instruction to "reuse
   `layout_contract.rs` validator" was the load-bearing hint that
   pointed at a non-compute, algorithm-level asset.

**Pattern lesson codified by v2.22.0 (counter-example hunting):** the
v2.21.0 cycle declared all non-compute PARTIAL levers for MODEL-2
"exhausted". Re-running the 7-gate FALSIFY-SHIP survey (013/014/016/017/
018/019/020) with explicit counter-example hunting found exactly one
genuine lever (SHIP-019); SHIP-017/018/020 truly need compute,
SHIP-013/014/016 collapse into SHIP-011's wiring. Prior verdict was ~86%
correct. **Rule: before declaring a search space exhausted, re-run the
survey with explicit counter-example hunting — the spec's own Risk
mitigations are the highest-leverage hint source.**

Combined MODEL-2 ship-gate status after v2.22.0: **3/12 AC-SHIP2 gates
fully ACTIVE** (001, 011, 012) + **3/12 PARTIAL_ALGORITHM_LEVEL** (002
via SHIP-012, 005 via SHIP-015, 009 via SHIP-019) = **6/12 touched**
(50%). The remaining 6 (003/004/006/007/008/010) all require either
real 370M training compute, a trained on-disk `.apr` with evaluation
harness, or a wall-clock benchmark on RTX 4090, and will remain
untouched until compute-dispatch lands — the pretrain loop driver + CLI
from v2.19.0 are ready for them. Genuine algorithm-level PARTIAL
harvesting is now exhausted for MODEL-2.

**v2.34.0 amendment (2026-04-23):** After the full MODEL-1 stack landed
(v2.24.0 → v2.33.0 covering SHIP-008 → SHIP-006 → SHIP-002 → SHIP-005 →
SHIP-010 → SHIP-007 → SHIP-003 → SHIP-004 → SHIP-001 → SHIP-009, 10/10
MODEL-1 AC rows touched), MODEL-2 PARTIAL harvesting resumes on top of
the completed MODEL-1 surface — **FALSIFY-SHIP-017 (AC-SHIP2-007) —
PARTIAL_ALGORITHM_LEVEL** restacked at task #149. The decision rule of
SHIP-017 ("`apr run` produces syntactically valid Python on 100
held-out prompts; ≥ 2 SyntaxError → FAIL, tolerate ≤ 1") is a pure
integer threshold function that can be proven correct today, even
though the full 100-prompt harness requires a trained 370M .apr.
`contracts/model-families/llama-370m-sovereign-v1.yaml` bumped v1.5.0
→ v1.6.0 (stays ACTIVE) with new GATE-ARCH-370M-005 binding
AC-SHIP2-007 ↔ FALSIFY-SHIP-017 and `discharge_status:
PARTIAL_ALGORITHM_LEVEL`. Algorithm proof = two unit tests in
`crates/aprender-train/src/models/llama_370m.rs`:
(1) `falsify_ship_017_syntax_error_count_threshold_logic` — covers
Pass boundary (0, 1 errors), Fail boundary (2 errors), pathological
cases (50, 100 errors all Fail), monotonicity over all errors ∈
[0, 100], and provenance pinning
(`AC_SHIP2_007_HELDOUT_PROMPT_COUNT == 100`,
`AC_SHIP2_007_MAX_TOLERATED_SYNTAX_ERRORS == 1`). (2)
`falsify_ship_017_gate_arch_370m_005_has_partial_discharge_marker` —
binds the sovereign contract YAML shape (falsification_id, binds_to,
discharge_status, evidence_discharged_by, full_discharge_blocks_on,
ship_blocking) to the Rust tests via `include_str!`. Full discharge
blocks on real trained 370M .apr + 100-prompt `apr run` harness with
EX-06-style Python AST parse pipeline — fixture swap only, no
harness rewrite. Unlike the original v2.25.0 discharge, SHIP-017 now
lands on a surface where MODEL-1 is already at 10/10 PARTIAL-touched,
so the integer-threshold shape is shared across models (MODEL-1
SHIP-002 = 0 tolerance on 1 canonical prompt; MODEL-2 SHIP-017 = 1
tolerance on 100 held-out prompts). New combined status: **3/12
ACTIVE + 5/12 MODEL-2 PARTIAL = 8/12 touched** on MODEL-2 side, with
MODEL-1 fully saturated at **10/10 PARTIAL**; 18 PARTIAL + 3
DISCHARGED across both models.

**v2.35.0 amendment (2026-04-23):** Second MODEL-2 PARTIAL in the
post-v2.33.0 MODEL-1-stack restacking window — **FALSIFY-SHIP-020
(AC-SHIP2-010) — PARTIAL_ALGORITHM_LEVEL** restacked at task #150 on
top of v2.34.0 SHIP-017. The decision rule of SHIP-020 ("`apr bench`
median decode throughput ≥ 100 tok/s on RTX 4090 for the 370M
target") is a pure f32 threshold function, separable from the
compute-heavy `apr bench --median` harness which requires a trained
370M .apr. `contracts/model-families/llama-370m-sovereign-v1.yaml`
stays at v1.6.0 ACTIVE with new GATE-ARCH-370M-006 binding
AC-SHIP2-010 ↔ FALSIFY-SHIP-020 and `discharge_status:
PARTIAL_ALGORITHM_LEVEL`. Algorithm proof = two unit tests in
`crates/aprender-train/src/models/llama_370m.rs`: (1)
`falsify_ship_020_decode_tps_threshold_logic` — covers exact 100.0
tok/s Pass boundary / one-f32-ULP below → Fail / generous-green
{120.0, 500.0} / hard-red {0.0, 50.0} / monotonicity in both
directions / non-finite inputs {NaN, ±∞} → conservatively Fail (a
real `apr bench` median is always a finite positive; +∞ as Pass
would let an instrumentation bug silently green the ship-gate) /
provenance pinning (`AC_SHIP2_010_MIN_DECODE_TPS_RTX4090 == 100.0`).
(2) `falsify_ship_020_gate_arch_370m_006_has_partial_discharge_marker`
— byte-binds the sovereign contract YAML shape (falsification_id,
binds_to, discharge_status, evidence_discharged_by,
full_discharge_blocks_on, ship_blocking) to the Rust tests via
`include_str!`. Full discharge blocks on a real trained 370M .apr +
three independent `apr bench --tokens 128 --json` medians on the
RTX 4090 host — fixture-swap only, no decision-rule rewrite.
SHIP-020 is the MODEL-2 twin of MODEL-1 SHIP-007 (same f32-threshold
shape; floor 100 tok/s for 370M vs 30 tok/s for 7B Q4_K because the
student is ~3.5× smaller than the teacher). New combined status:
MODEL-2 at **6/12 touched**; MODEL-1 still fully saturated at
**10/10 PARTIAL**; **19 PARTIAL + 3 DISCHARGED** across both models.

**v2.36.0 amendment (2026-04-23):** Third MODEL-2 PARTIAL in the
post-v2.33.0 MODEL-1-stack restacking window — **FALSIFY-SHIP-018
(AC-SHIP2-008) — PARTIAL_ALGORITHM_LEVEL** restacked at task #151 on
top of v2.35.0 SHIP-020. The decision rule of SHIP-018 ("`apr eval
--benchmark humaneval` pass@1 ≥ 30.0% on 164 tasks") is a pure
(correct, total, threshold_pct) → Pass/Fail comparison, separable
from the compute-heavy 164-task sampling harness which requires a
trained 370M .apr. `contracts/model-families/llama-370m-sovereign-v1.yaml`
v1.7.0 → v1.8.0 ACTIVE with new GATE-ARCH-370M-007 binding
AC-SHIP2-008 ↔ FALSIFY-SHIP-018 and `discharge_status:
PARTIAL_ALGORITHM_LEVEL`. Algorithm proof = two unit tests in
`crates/aprender-train/src/models/llama_370m.rs`: (1)
`falsify_ship_018_humaneval_pass_at_1_threshold_logic` covers
inclusive floor (30/100, 60/200, 50/164=30.49% all Pass) +
just-below Fail (49/164≈29.88%, 29/100=29.0%) + f32-exact 50/100
±ULP asymmetry proof of `>=` (inclusive, not `>`) + generous-green
{82/164, 164/164} + hard-red {0/164, 1/164} + monotonicity sweep
correct∈[0,164] with Fail→Pass allowed once, Pass→Fail forbidden +
div-safety (total=0 → Fail) + sanity (correct>total → Fail) +
non-finite threshold {NaN, +∞, −∞} → conservatively Fail + provenance
pin (`AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT == 30.0`). (2)
`falsify_ship_018_gate_arch_370m_007_has_partial_discharge_marker`
byte-binds the sovereign YAML shape via `include_str!`. Full
discharge blocks on a real trained 370M .apr + three independent
seed=0 `apr eval --benchmark humaneval --json` medians each feeding
`verdict_from_pass_at_1` → all three Pass — fixture-swap only, no
decision-rule rewrite. SHIP-018 is the MODEL-2 twin of MODEL-1
SHIP-005 (both pass@1 threshold gates; MODEL-2 30.0% floor with no
noise allowance vs MODEL-1 86.0% ± 1.2 pp noise window because the
370M student is trained from scratch whereas the 7B teacher is a
Qwen2.5-Coder-Instruct derivative). New combined status: MODEL-2 at
**7/12 touched**; MODEL-1 still fully saturated at **10/10 PARTIAL**;
**20 PARTIAL + 3 DISCHARGED** across both models.

**v2.37.0 amendment (2026-04-23):** Fourth (and final compute-free)
MODEL-2 PARTIAL in the post-v2.33.0 MODEL-1-stack restacking window
— **FALSIFY-SHIP-016 (AC-SHIP2-006) — PARTIAL_ALGORITHM_LEVEL**
restacked at task #152 on top of v2.36.0 SHIP-018. The decision rule
of SHIP-016 ("`apr qa <model>.apr` — all 8 gates PASS") is a pure
aggregate-AND over a Boolean slice, separable from the compute-heavy
gate runner which requires a trained 370M .apr and a real RTX 4090
host. `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.8.0
→ v1.9.0 ACTIVE with new GATE-ARCH-370M-008 binding AC-SHIP2-006 ↔
FALSIFY-SHIP-016 and `discharge_status: PARTIAL_ALGORITHM_LEVEL`.
Algorithm proof = two unit tests in
`crates/aprender-train/src/models/llama_370m.rs`: (1)
`falsify_ship_016_apr_qa_aggregate_and_logic` covers the exhaustive
2^8 = 256-combination sweep (exactly one input — all-true — yields
Pass; 255 yield Fail) + 8-way single-gate-flip falsifiability +
monotonicity (flipping false→true never regresses Pass→Fail) + 4
contract-drift guards (gate slice of length {0, 7, 9, 16} → Fail
conservatively even when supplied entries are all true) + provenance
pin (`AC_SHIP2_006_REQUIRED_QA_GATE_COUNT == 8`). (2)
`falsify_ship_016_gate_arch_370m_008_has_partial_discharge_marker`
byte-binds the sovereign YAML shape via `include_str!`. Full
discharge blocks on real trained 370M .apr + `apr qa <model>.apr`
exit 0 with all 8 gates green on RTX 4090 — fixture-swap only, no
harness rewrite. SHIP-016 is the MODEL-2 twin of MODEL-1 SHIP-006
(both 8-gate aggregate-AND; same 8 canonical gates). This completes
the compute-free MODEL-2 PARTIAL harvest within the restacking
window: remaining 4 MODEL-2 gates (003 val loss ≤ 2.2 / 004 ≤21-day
wall-clock) + already-touched {005, 009} / {007-covered, 008-covered,
010-covered via 017/018/020} are all either compute-bound or already
discharged. New combined status: MODEL-2 at **8/12 touched**;
MODEL-1 still fully saturated at **10/10 PARTIAL**; **21 PARTIAL + 3
DISCHARGED** across both models.

**v2.20.0 amendment (2026-04-19):** Two MODEL-2 ship gates **DISCHARGED**
in the post-v2.19 evidence window on branch `chore/post-v2.19-evidence`:

1. **FALSIFY-SHIP-021 (AC-SHIP2-011) — DISCHARGED** at commit `0b8ca8c84`
   (task #112). `falsify_ship_021_seed_0_100_step_reproducibility` proves
   two seed=0 × 100-step training runs produce |Δloss| ≤ 1e-6 at every
   step and bit-identical AdamW-state sha256; a counter-test
   `falsify_ship_021_different_seeds_do_diverge` proves seed=0 vs seed=1
   diverge > 1e-4 within 10 steps. Root cause of the original green-run
   flake (step-0 6.854 vs 6.928 under parallel cargo test) was a sibling
   test racing on the global `INIT_SEED` atomic; fix landed as
   `transformer::init::lock_init_seed(seed) -> MutexGuard` which any
   future caller doing concurrent weight init under a set-before-read
   global MUST hold across the full init work. Contract
   `training-loop-pretrain-v1.yaml` bumped 1.0.0 → 1.1.0,
   status PROPOSED → ACTIVE, INV-TRAIN-006 + GATE-TRAIN-006 got
   harness/evidence_discharged_by blocks.

2. **FALSIFY-SHIP-022 (AC-SHIP2-012) — DISCHARGED** at commit `8f0607d42`
   (task #113). `apr inspect` now surfaces the three provenance keys —
   `license`, `data_source`, `data_license` — from every .apr binary,
   rendering absent values as the literal `(missing)` in text mode and
   `null` in JSON mode. Key design: `AprV2Metadata` gained
   `data_source` + `data_license` as NAMED Option<String> fields (not
   buried in `custom: HashMap`); no `skip_serializing_if` is allowed on
   any provenance field on either `AprV2Metadata` or `MetadataInfo`,
   because silent-skip via serde is the exact failure mode
   (`FM-APR-PROV-SILENT-SKIP`) the contract guards against. Text
   rendering goes through a pure helper `format_provenance_block()` so
   tests assert on a returned `String` rather than capturing stdout
   (`gag::BufferRedirect` is NOT parallel-test-safe — recorded as a
   reusable pattern). New schema contract `apr-provenance-v1.yaml`
   (C-APR-PROVENANCE v1.0.0 ACTIVE, `kind: schema`) declares 3
   invariants (round-trip, always-emit, publish-gate-rejects), 3 gates,
   and 3 failure modes, all bound to AC-SHIP2-012. `pv validate` PASS
   (0 errors). Live smoke test on
   `qwen2.5-coder-1.5b-instruct-q4k.apr` (no provenance stored)
   correctly prints the Provenance block with `(missing)` on all three
   rows. Together with PM-003/PM-008/PM-009/PM-007 pre-flight gates,
   any operator can now answer "what data trained this, under what
   license?" from a `.apr` alone — the sidecar-manifest dependency is
   severed.

Combined MODEL-2 ship-gate status after v2.20.0: 2/12 AC-SHIP2 gates
DISCHARGED (011, 012). The remaining 10 (001–010) all block on the
actual 370M checkpoint, which is the compute-dispatch long-pole; the
pretrain loop driver from v2.19.0 is ready to exercise them once
compute-dispatch for real weights lands.

**v2.17.0 amendment (2026-04-18):** Task #101 contracts schema
harmonization **SHIPPED** on `feat/pm-007-preflight-poka-yoke` at commit
`4fc453d57`. Closes the last parser barrier preventing `pv validate`
from serving as the canonical dogfooded gate across all SHIP-TWO-001
contract work. `crates/aprender-contracts/src/schema/types.rs` now
(a) accepts legacy ProofObligation field spellings (`statement`/
`verification`) via `#[serde(alias)]`, (b) accepts both map
`{id: Equation}` and list `[{id, ...}]` equation forms via a custom
polymorphic `deserialize_equations`, (c) adds `Safety` + `Liveness` to
`ObligationType` (28 variants now, up from 26), (d) uplifts 6 legacy
contracts (decode-gpu-resident-sampling, decode-hot-path-*,
eval-harness-humaneval, eval-sharding, profile-graph-vs-per-op-
methodology, publish-manifest) to the metadata-block form. Target
tests `load_contracts_real` + `parse_missing_metadata_returns_error`
both green; 1368/1371 aprender-contracts lib tests pass. Remaining
3 failures (lint_passes_on_real_contracts, validate_gate_passes,
lint_findings_on_failure) are downstream content checks — empty
`formula:` bodies, missing `kani_harnesses`, falsifications <
proof_obligations on the same 6 legacy contracts — dispatched as
task #102 follow-up content-authoring lane. This amendment ties the
"pv not bash for contracts" MEMORY.md policy (2026-04-18) to
concrete unblocked state: no more adhoc bash/grep workarounds when
the dogfood tool covers the workflow.

**v2.18.0 amendment (2026-04-18):** Parallel dispatch lanes #102/#103/#104
**ALL CLOSED** in a single concurrent compute window against
non-overlapping surfaces, demonstrating the monorepo's sub-agent
workflow scales. Results:
(a) **#102 contract backfill CLOSED** — 8 legacy contracts
(`decode-gpu-resident-sampling`, `decode-hot-path-{first-tokens,
prefix-cache,zero-syscalls}`, `eval-harness-humaneval`, `eval-sharding`,
`profile-graph-vs-per-op-methodology`, `publish-manifest`) received
metadata references, formula bodies, kani_harnesses, and falsification
parity. 22 ERROR findings → 0, `lint_passes_on_real_contracts` green.
Verified live via `pv validate` dogfood: 8/8 contracts parse clean
(1 advisory SCHEMA-013 qa_gate-missing on eval-sharding kept as
forward work). No bash/grep workaround needed.
(b) **#103 MODEL-2 `--min-frequency` plumbing CLOSED** — `apr-cli`
tokenize call-site swapped from `aprender::text::tokenize::BpeTokenizer`
→ `entrenar::tokenizer::BPETokenizer::train` via `train_bpe_via_entrenar`
helper; `TokenizerConfig::bpe().with_min_frequency(..).with_normalization(..)`
now threads user-provided `--min-frequency` + `--normalization` into
merge pruning. Public read-only `vocab()`/`merges()` accessors added to
`aprender-train::tokenizer::BPETokenizer`. 17 `apr-cli` tokenize tests
pass including new `run_train_honors_min_frequency_pruning` which
asserts singleton byte-pairs ("xyz" single occurrence) are pruned
from `merges.txt`/`vocab.json` at threshold 2. Closes v2.15.0 §1
"Known gap". Redundant `build_normalizer()` call-site removed since
`aprender-train`'s BPE applies NFC internally (no double-normalization).
(c) **#104 gx10 capacity gate PASS** — llama.cpp (b1-b0f0dd3 CUDA)
on teacher GGUF (sha256 `e6cac5d6…7981`) measured **38.0 tok/s decode**
(prompt eval 509.0 tok/s, 7.7 GiB VRAM, 5 s wall) vs 30 tok/s gate
threshold = PASS 26.7% margin. 2.45× the forbidden 15.5 tok/s fused
NF4 steady-state fallback — Zero-Tolerance §3 row #8 "no perf
regression" clause preserved. Two follow-ups flagged: (1) decode drift
from memory's 46 tok/s → 38.0 tok/s on current build; (2) gx10 disk
95% full (44 GB free) needs cleanup before MODEL-2 7B training lands.
Evidence: `evidence/ship-two-001/gx10-capacity-baseline-20260418-213928.json`.

With #102+#103 closed, **task #105** (370M MODEL-2 pretraining loop
wiring per `training-loop-pretrain-v1` GATE-TRAIN-005) is now the sole
long-pole item. Expected surface: `aprender-train/src/train/pretrain.rs`
loop driver calling the `llama_370m` forward pass with AdamW optimizer
and gradient accumulation, gated by the dataset ingest binary shipped
in v2.15.0.

**v2.19.0 amendment (2026-04-18):** Task #105 **CLOSED** via background
sub-agent `ac479445bcd722bf7` — commit `9a5af3ac2` on
`feat/pm-007-preflight-poka-yoke`. Surface landed (6 files, +1379 LOC):
(a) `crates/aprender-train/src/train/pretrain.rs` (963 LOC) — PretrainConfig
with `model_2_defaults()` baking LR=5e-5 + rank=32 + seed=42 remedies
from MODEL-1 v2 QLoRA divergence post-mortem;
(b) `crates/apr-cli/src/commands/pretrain.rs` (332 LOC) — CLI entrypoint
gated behind `training` cargo feature;
(c) extended_commands.rs + dispatch_analysis.rs — wired `apr pretrain`
into the apr-cli dispatch table.
Contract compliance verified: `contracts/training-loop-pretrain-v1.yaml`
passes `pv validate` with 0 errors; GATE-TRAIN-005 (val_loss[N] ≤ 2.0×
val_loss[N-1]) wired in `check_non_divergence`; INV-TRAIN-007 NaN/Inf
guard wired in `check_numerical_stability` before metric logging;
GATE-TRAIN-008 throughput bounds wired via `PretrainAbort::ThroughputOutOfRange`.
`per_step_metrics.required` and `per_epoch_artifacts.required_fields`
enforced as struct invariants. Checkpoint path template
`{run_dir}/ckpt/epoch-{N:03d}.apr` frozen in `EpochArtifact::new`.
Synthetic drive via injected `StepFn`/`ValFn` traits allows exercising
the full gate surface today while the real 370M forward pass wiring
(llama_370m.rs) completes. Test verification: 15/15 pretrain unit tests
pass, 3/3 CLI tests pass, 947/947 aprender-train lib tests no
regressions. Abort errors map 1:1 to contract gate IDs so operators
see the tripped gate via shell `$?`.

Concurrent with #105, **task #108** closed the 32-way workspace-test
regression discovered by CI run 24614757928. Root cause: five directory
iterators in `aprender-core/src/format/` were treating
`contracts/model-families/llama-370m-sovereign-v1.yaml` (a
ModelFamilyVariant CONTRACT starting with `contract_id:`) as a
ModelFamily REGISTRY entry. Fix (commit `21d43bd7a`): all iterators
now skip files whose first top-level key is `contract_id:` (family
registry YAMLs all begin with `metadata:` — a clean discriminator,
verified by corpus scan). `cargo test -p aprender-core --lib format::`
re-green at 13031 passed / 0 failed.

The ci/lint workspace package-ambiguity blocker (`aprender@0.27.8` vs
`aprender@0.31.0` — caused by transitive deps on published
`realizar ^0.7/^0.8`, `renacer ^0.9/^0.10`, `trueno ^0.15/^0.16/^0.17`,
`entrenar ^0.7`, `bashrs ^6.35/^6.65`, `pacha ^0.2` that all re-export
old `aprender@0.27`) was split into task #109 for a separate
path-dependency migration pass. This is orthogonal to SHIP-TWO-001 and
pre-dated the branch; the monorepo's `[patch.crates-io]` block was
removed during RC4 cc-cleanup and was never restored for the
aprender/realizar/renacer/trueno/trueno-gpu chain documented in
CLAUDE.md.

**Parallel dispatch state (2026-04-18 post-v2.17.0 — preserved for audit
trail):** three lanes ran concurrently against non-overlapping surfaces —
(a) task #102 contract backfill (content-authoring, contracts/*.yaml),
(b) task #103 MODEL-2 CLI `--min-frequency` plumbing (swap apr-cli
tokenize call-site from aprender-core BPE to aprender-train BPE;
closes v2.15.0 §1 "Known gap" — 0.5 day), (c) task #104 gx10
third-party framework capacity gate (llama.cpp on teacher GGUF,
enforces Zero-Tolerance §3 row #8). Tasks #105 (370M pretraining
loop wiring per training-loop-pretrain-v1 GATE-TRAIN-005) remains
the long-pole item awaiting #102+#103 closure. Compute pool utilization
is deliberately heterogeneous: lambda-labs (x86_64 RTX 4090) does
contract+code surgery, gx10 (aarch64 GB10 Blackwell) does remote
bench, yoga (x86_64 RTX 4060 Laptop) stays idle pending apr 0.31.0
upgrade per Zero-Tolerance §3 row #8. Jetson remains blocked per
`project_ship_two_001_jetson_blocked.md`.

**v2.16.0 amendment (2026-04-18):** Codified **Zero-Tolerance** as §3 row
#8. The operationalization, verbatim: "We never accept bugs or poor
performance. Defects and perf regressions are both blockers, not trade-
offs. All work improves or holds the line; never degrades it. No 'pre-
existing' carve-outs. No `#[ignore]` as a release valve." Why now: the
SHIP-TWO-001 compute-pool reality (lambda-labs x86_64 RTX 4090 +
yoga x86_64 RTX 4090 Laptop + gx10 aarch64 GB10 Blackwell + jetson
aarch64) surfaces cases where it is tempting to accept a regression
("gx10 is Blackwell — 15.5 tok/s fused NF4 is fine") or a bug ("yoga's
apr 0.4.11 is stale but it works for small models"). The Zero-Tolerance
principle writes the refusal explicitly: when a host drops to a slower
path OR runs stale software, that is a blocker on the host, not a
baseline to ship against. Concrete application to in-flight work:
(a) yoga stays blocked from SHIP-TWO-001 eval dispatch until apr
binary is upgraded to 0.31.0 AND cuBLAS smoke passes (no "it works
with the old binary" ship), (b) gx10 must run a non-fused third-party
framework (llama.cpp / PyTorch nightly cu128 / vllm) at ≥ reference
tok/s before counting as GPU capacity for MODEL-2 parity training —
the 15.5 tok/s fused fallback is FORBIDDEN as a steady-state (see
`project_pmat_587_*` memos for prior perf discipline). Ties to the
existing Toyota Way feedback memory: "all defects are your defects;
never 'pre-existing'" now extends to performance.

**v2.15.0 amendment (2026-04-18):** MODEL-2 pretraining scaffold **LANDED**
on `feat/pm-007-preflight-poka-yoke`. Three commits close the three
P0 blockers identified in the v2.14.0 readiness audit:

1. **Task #89 — BPE NFC patch SHIPPED (commit `b0e0a280b`):**
   `crates/aprender-train/src/tokenizer/{config,bpe}.rs` now enforce
   C-TOK-BPE-001 INV-TOK-003 (NFC before optional lowercase).
   `TokenizerConfig::normalization` defaults to `None` (`#[serde(default)]`
   for backward compat); set `Normalization::NFC` via
   `.with_normalization()` builder. Two falsification tests locked:
   (a) `test_bpe_nfc_composed_decomposed_parity` — composed `café`
   U+00E9 and decomposed `cafe\u{0301}` encode to identical token IDs
   under NFC; (b) `test_bpe_without_nfc_composed_decomposed_diverge` —
   live falsification witness: without NFC the two forms MUST diverge.
   If the witness test starts passing under `Normalization::None`, the
   invariant is no longer load-bearing and the contract should be
   revisited. `preprocess()` doc-comment records **why NFC before
   lowercase**: `char::to_lowercase()` is not closed over non-NFC input
   for every grapheme — normalizing first keeps the pipeline
   deterministic for composed/decomposed variants.

2. **Task #90 — `apr tokenize train` subcommand SHIPPED (commit
   `512ea51a6`):** new `TokenizeCommands::Train { corpus, vocab_size,
   min_frequency, output, normalization }` variant. Walks `.jsonl`
   files (file or directory), extracts `content` field per line,
   applies NFC via `unicode-normalization::UnicodeNormalization::nfc`
   when `--normalization nfc` (default), calls the BPE trainer, emits
   `vocab.json` + `merges.txt`. `--json` mode round-trips all
   parameters. 3 unit tests pass (happy-path JSONL, directory walk,
   unknown-normalization rejection). **Known gap** (follow-up, NOT a
   ship blocker): `--min-frequency` is accepted for contract parity
   but NOT threaded through — the CLI currently calls
   `aprender::text::tokenize::BpeTokenizer::train(corpus, vocab_size)`
   (aprender-core) which has no public `min_frequency` parameter.
   Strategic fix: switch the CLI to
   `aprender-train::tokenizer::BPETokenizer` (which both honors
   `with_min_frequency()` AND has the NFC plumbing task #89 added).
   Documented in memory `project_ship_two_001_nfc_bpe_patch.md`.

3. **Task #91 — `apr-corpus-ingest` binary SHIPPED (commit
   `512ea51a6`):** new `crates/apr-cli/src/bin/apr-corpus-ingest.rs`
   (+517 LOC) with `plan` and `validate-contract` subcommands over
   `C-DATA-THESTACK-PYTHON` v1.0.0. `plan` reads the contract,
   asserts the 6 required top-level keys (source, license_whitelist,
   pii_scrub, deduplication, split, budget), validates 7
   `INV-DATA-*` + 5 `FALSIFY-DATA-*` + 5 `GATE-DATA-*` prefixes, and
   emits `./output/dry-run-manifest.yaml` with TODO placeholders + UTC
   timestamp. `validate-contract` is exit-code-only. **Hard constraints
   honored:** NO network, NO writes outside `./output/`, deps limited
   to workspace `serde`/`serde_yaml`/`anyhow`/`clap`. Does NOT touch
   `aprender-train/` or `aprender-core/`. 2 unit tests pass.

**MODEL-2 training readiness estimate (post-v2.15.0):** 4 contracts +
3 scaffolding commits shipped. Remaining work to first pretraining
loss curve:
- Thread `--min-frequency` through CLI (switch call to aprender-train
  BPE) — 0.5 day, follow-up ticket.
- Actual corpus download + validated ingest into train/val split
  honoring the 6 C-DATA-THESTACK-PYTHON gates (MinHash-LSH dedup,
  PII scrub, license whitelist, deterministic hash-by-sha256 split,
  corpus_sha256 merkle gate yoga vs gx10) — 2-3 days.
- 370M Llama architecture implementation + pretraining loop wiring
  honoring `training-loop-pretrain-v1.yaml` GATE-TRAIN-005 (val_loss
  divergence abort) — 5-7 days.
- First pretraining smoke run on gx10 — 1-2 days.

**Total: ~10-14 days to first loss curve** (revised up from
v2.14.0's 5-7d estimate now that the scaffold is concrete and the
370M arch implementation is clearly the gating path). Post-v2.15.0,
MODEL-2 moves from contract+scaffold into execution.

**v2.14.0 amendment (2026-04-18):** MODEL-2 pretraining readiness audit
closed two gaps in contract + impl surface:

1. **Dataset contract drafted:** `contracts/dataset-thestack-python-v1.yaml`
   (C-DATA-THESTACK-PYTHON v1.0.0 PROPOSED). 7 invariants + 5
   falsification tests + 5 compound gates covering (a) upstream
   revision pin + raw_tar_sha256 reproducibility, (b) permissive-
   license whitelist (Apache/MIT/BSD/ISC/Unlicense/CC0/0BSD) with
   unknown→reject policy, (c) PII scrub (AWS/PEM/GH PAT/Slack/Google),
   (d) MinHash-LSH near-duplicate removal (seed=42, Jaccard ≥0.85 →
   drop), (e) deterministic hash-by-file-sha256 split (train=0.98,
   val=0.02, assertion: same seed → byte-identical split across
   hosts), (f) corpus_sha256 merkle-style parity gate (FALSIFY-DATA-003
   yoga vs gx10), (g) UTF-8 + NFC round-trip encoding hygiene
   (INV-DATA-007). Closes the P0 blocker identified by the 2026-04-18
   MODEL-2 training-readiness audit: `training-loop-pretrain-v1.yaml`
   line 22 referenced this peer contract, but the file did not exist.

2. **BPE NFC gap identified (IMPLEMENTATION BLOCKER):** The BPE
   tokenizer at `crates/aprender-train/src/tokenizer/bpe.rs` does NOT
   implement NFC normalization, despite `contracts/tokenizer-bpe-v1.yaml`
   INV-TOK-003 / `dataset-thestack-python-v1.yaml` INV-DATA-007
   requiring it. No HF `tokenizers` dep to defer to.
   `TokenizerConfig`/`BpeConfig` have no normalizer field. Fix surface:
   (a) add `normalization: Option<Normalization>` to BpeConfig, (b)
   apply `unicode_normalization::nfc()` at `encode()` entry, (c) add a
   round-trip property test on `café` (composed vs decomposed) + emoji.
   Without this, MODEL-2 tokenizer will drift between train-time and
   inference-time on non-ASCII code and GATE-DATA-005 will ship-block.

**MODEL-2 training readiness estimate (post-v2.14.0):** contract surface
is complete (4 contracts: llama arch, BPE tokenizer, pretrain loop,
dataset). Remaining code work: BPE NFC patch (~1 day), tokenizer
trainer CLI wiring (~3 days), corpus ingest harness honoring the
dataset contract (~2 days). **5-7 days to first pretraining run**
modulo Blackwell JIT warm-up and corpus download time.

**v2.13.0 amendment (2026-04-18):** FALSIFY-SHARD-003 DISCHARGED. Live
probe run yoga (RTX 4090, x86_64) vs gx10 (GB10 aarch64) on the released
teacher GGUF (`paiml/qwen2.5-coder-7b-apache-q4k-v1`, sha
`e6cac5d6…7981`) returned **16/16 byte-identical completions** on
HumanEval/0..15 at temperature=0.0, top-k=1, max_tokens=512. Evidence:
`evidence/ship-two-001/shard-003-determinism/probe_20260418_143041.json`.
Contract `contracts/eval-sharding-v1.yaml` bumped 1.0.0 → 1.1.0 and
flipped **DRAFT → ACTIVE**; `discharged:` block recorded on
FALSIFY-SHARD-003 mirroring the SHARD-004 pattern. Combined with the
prior SHARD-004 discharge (Δ=0.0039 pp merged-score identity), both
correctness gates for AC-EX-007 are green. The parallel eval-shard lane
(yoga+gx10) is now a legitimate accelerator for any future SHIP-TWO-001
re-audit that respects the contract prerequisites (temp=0.0, top-k=1).
Task #79 closed.


**v2.12.0 amendment (2026-04-18):** Post-ship artifacts landed (commit
`cc52e7bfc`) while the teacher is live on HF. All of these are
**out-of-scope for the current ship** but advance the next-wave deliverables:

1. **MODEL-2 Phase 1-B contracts** (task #81) — three new YAMLs:
   - `contracts/model-families/llama-370m-sovereign-v1.yaml` (9 invariants,
     4 gates, sovereign 370M arch with frozen intermediate_dim=2816)
   - `contracts/tokenizer-bpe-v1.yaml` (7 inv, 7 gates; vocab bounds,
     special tokens, byte-exact round-trip, NFC normalization)
   - `contracts/training-loop-pretrain-v1.yaml` (8 inv, 8 gates;
     GATE-TRAIN-005 ship-blocking: `val_loss[N] > 2.0 × val_loss[N-1]`
     → ABORT — encodes the MODEL-1 v2 divergence lesson)
2. **MODEL-1 QLoRA retry plan** (task #86) —
   `docs/specifications/aprender-train/model-1-qlora-retry-plan.md`,
   6 falsification gates, hyperparameter deltas from v2 (LR 2e-4 →
   5e-5, rank 16 → 32, temperature 4.0 → 2.0).
3. **FALSIFY-SHARD-003 determinism probe** (task #88) —
   `scripts/ship-two-001/eval-shard-determinism-probe.sh` (239 lines).
   Closes the one blocking gap for AC-EX-007 found by the eval-shard
   audit (contract `eval-sharding-v1.yaml` line 151 referenced a script
   that did not exist). DRY_RUN=1 validates the JSONL builder without
   dispatch. Full `--hosts yoga,gx10 --model <gguf> --probe-tasks 0-15`
   run requires teacher GGUF pre-cached on both hosts.

**Compute-pool reality check (2026-04-18):** yoga RTX 4090 + gx10 GB10
aarch64 are today's effective parallel pool. Jetson remains blocked by
the 5 blockers documented in memory `project_ship_two_001_jetson_blocked.md`.
Lambda-labs is referenced in spec docs but **not provisioned** — no SSH
alias, no memory file, no credentials surfaced; treat as aspirational
until provisioning is in place.

**v2.11.0 amendment (2026-04-18):** SHIP-TWO-001-MODEL-1-TEACHER **RELEASED**.
EX-05, EX-06, EX-07 all DISCHARGED on the teacher artifact (`paiml/qwen2.5-coder-7b-apache-q4k-v1`):

1. **EX-05 verify-manifest (live, 3 formats)**:
   `apr validate-manifest <m> --live --json` PASS for `.apr` (8.0 GiB, sha
   `0a854098…c73666`), `.safetensors` (15.2 GiB, sha `c1058ce7…d8954`),
   `.gguf` (7.5 GiB, sha `e6cac5d6…7981`). All five gates fire green:
   PM-001 (schema), PM-003 (HEAD content-length), PM-002-live
   (streaming sha256), PM-004 (SPDX), PM-005 (recipe_sha256), PM-006
   (parent chain). Evidence:
   `evidence/ship-two-001/ex-05-manifest-verify-*.json` (3 per-format +
   1 summary).

2. **EX-06 apr pull + re-inference**: `apr pull
   paiml/qwen2.5-coder-7b-apache-q4k-v1` → cached GGUF at
   `~/.cache/pacha/models/7bcabb852fedb36b.gguf`; sha256 of pulled file
   exactly matches the declared GGUF manifest sha (harness v3 auto-
   detects pulled format from file extension, fixes v2 bug that hard-
   coded the APR manifest and produced a spurious format-mismatch FAIL);
   `apr run <pulled> --prompt 'def fib(n):' --max-tokens 64 --temp 0 --top-k 1`
   produces output whose longest parseable prefix contains ≥1 non-trivial
   Python statement (spec §12.3 AC-EX-006 literal: "emits syntactically
   valid Python"). Both **AC-EX-005 (sha256 roundtrip)** and **AC-EX-006
   (Python validity)** PASS. Evidence:
   `evidence/ship-two-001/ex-06-pull-rerun.json` → `overall: PASS`.

3. **EX-07 tag release**: Git tag `SHIP-TWO-001-MODEL-1-TEACHER` created
   at HEAD of the ship branch; announcement blurb embedded in this
   amendment. The teacher artifact is live on HF Hub at
   https://huggingface.co/paiml/qwen2.5-coder-7b-apache-q4k-v1 (3 formats),
   and downloadable via `apr pull paiml/qwen2.5-coder-7b-apache-q4k-v1`.

**Announcement (v2.11.0):** Aprender ships its first sovereign model:
Qwen2.5-Coder-7B-Instruct Q4_K (Apache-2.0), 85.98% HumanEval pass@1
(141/164, confirmed via `apr eval --benchmark humaneval` on 2026-03-28),
8.0 GiB APR / 7.5 GiB GGUF / 15.2 GiB SafeTensors. Runs end-to-end on
`apr run` / `apr serve`. MODEL-1 v2 (distilled student) is falsified
at the adapter (non-converged QLoRA, task #86 holds the retry plan);
MODEL-2 (albor sovereign) follows in a separate ship per spec §12.4.

**v2.10.0 amendment (2026-04-18):** MODEL-1 v2 root cause is **DEFINITIVE**:
non-converged QLoRA adapter. Deep-probe sub-agent (memory:
`project_ship_two_001_model1_qlora_divergence.md`) found the smoking gun in
`instruct-qlora-7b/best/metadata.json` — `train_loss=15.41`,
`val_loss=31.99`, `train_perplexity=1e6`, `val_perplexity=1e6`,
`epoch=0` (of planned 3). The `best/` and `epoch-0/` adapter safetensors
are byte-identical; training halted at epoch 0 with both losses
diverging and perplexity saturated at the 1M cap. Merging this
non-converged adapter into Qwen2.5-Coder-7B produced the mode-collapsed
`ylkoylkoylko…` output observed by AC-SHIP1-005. **Hypotheses all
FALSIFIED**: tokenizer (embedded BPE loads cleanly, `embed_tokens`
byte-identical to teacher), tensor layout (`apr qa` Tensor Contract PASS,
339 tensors pass PMAT-235), quantization (Q4K lm_head stats match
teacher f32 within quant noise). Probable failure mode: LR=2e-4 too
hot for rank-16 actual (recipe specified rank=32) × soft-label
temperature=4.0. **Ship decision**: TEACHER-ONLY
(`qwen2.5-coder-7b-instruct-q4k.apr`, 85.98% pass@1 confirmed via
`/home/noah/src/apr-leaderboard/results/humaneval_20260328_121327.json`
— 141/164 pass). AC-SHIP1-005 (distilled student ≥30% HumanEval)
blocked by MODEL-1 retry (task #86, out of scope for current ship).
EX-05/06/07 proceeds with teacher artifacts only. Reduced-gate ship per
§ Failure Protocol (Hansei).

**v2.9.0 amendment (2026-04-18):** EX-04 **DISCHARGED**. Two falsifications
of the v2.8.1 code motivated two fixes:
1. v1.1.2 — `upload_via_xet` was early-returning on the Xet branch,
   skipping the LFS-pointer commit entirely (bytes in CAS but invisible
   on repo tree). Evidence: `evidence/ship-two-001/ex-04-xet-clobber-falsification.json`.
2. v1.1.3 — after the v1.1.2 fix, `commit_lfs_pointer` was using
   `application/json` with an `{operations:[{op:addOrUpdate,...}]}`
   schema that HF Hub accepts with HTTP 200 + `success:true` but
   silently no-ops (produces empty commits identical to parent tree).
   Evidence: `evidence/ship-two-001/ex-04-xet-postfix-still-falsified.json`.
   Fix: NDJSON body (newline-delimited JSON) with `Content-Type: application/x-ndjson`
   and `{"key":"header",...}` + `{"key":"lfsFile","value":{...}}` line
   schema. New gate **FALSIFY-PUB-LFS-011** (NDJSON schema) with source-
   invariant test `commit_lfs_pointer_uses_ndjson_lfsFile_schema`.
   Live discharge: `evidence/ship-two-001/ex-04-xet-postfix-v1.1.3-discharged.json`
   — all three formats (8.0 GiB .apr, 8.0 GiB .gguf, 15.2 GiB .safetensors)
   now present on `/tree/main` with sha256 oids matching staging; GGUF
   idempotent re-upload completed in 16.9s (CAS cache-hit). Contract
   `contracts/apr-publish-hf-large-file-v1.yaml` bumped to v1.1.3,
   `status` → `DISCHARGED`. **FALSIFY-PUB-LFS-009/010/011** all DISCHARGED.
   Next: EX-05 (verify-manifest live), EX-06 (apr pull + re-inference),
   EX-07 (tag release SHIP-TWO-001-MODEL-1-TEACHER).

**v2.8.1 amendment (2026-04-18):** Phase 2 of F-PUB-LFS-001 shipped in
commit `18fd9536e` (PR #882). The `xet` sub-feature wires `hf-xet`
1.5.1 (HF's Apache-2.0 reference impl) into `apr publish`. The
`reject_oversized_file` hard-abort is deleted; files > 5 GiB now
dispatch through `crates/aprender-core/src/hf_hub/xet.rs::XetUploader`,
which uses the `hf-xet` blocking API (`XetSessionBuilder` → token-
refresh URL → `upload_from_path_blocking` → `commit_blocking`). The
client-side surface is 178 lines because phases 3–7 of the Xet
protocol (chunking, dedup, xorb/shard CAS upload, hash encoding) are
delegated wholesale to the reference impl. **FALSIFY-PUB-LFS-001**
(file-size dispatch) and **-002** (token-refresh URL shape) are
deterministically discharged by 4 unit tests; **-003..-009** are
inherited from `hf-xet`; **-010** (three-format dogfood) still
pending HF_TOKEN in the ship environment. Contract
`contracts/apr-publish-hf-large-file-v1.yaml` bumped to v1.1.0 with
status `IMPLEMENTED`.

**v2.8.0 amendment (2026-04-18):** EX-04 discovered that `apr publish`
aborts on every SHIP-TWO-001 teacher artifact because all three formats
exceed the 5 GiB HTTP preupload threshold (.apr 8.0 GiB / .gguf 8.0 GiB /
.safetensors 15.2 GiB). The fix is NOT sharding (workaround) and NOT a
self-hosted S3 mirror (not sovereign — AWS-dependent). The fix is to
implement HF Hub's actual current large-file protocol: **Xet**
(huggingface.co/docs/xet/index v1.0.0, reference Rust impl
github.com/huggingface/xet-core Apache-2.0 v1.4.3). New contract
`contracts/apr-publish-hf-large-file-v1.yaml` v1.0.0 codifies the
10-gate falsification set **FALSIFY-PUB-LFS-001..010** (file-size
dispatch, token acquisition, chunk/xorb invariants, shard ordering,
idempotency, retry policy, hash-string encoding, LFS pointer commit,
three-format dogfood). See §12.8 for the full protocol amendment.

**v2.7.0 amendment (2026-04-18):** the pre-flight gate set grows to nine.
**FALSIFY-PM-009** (APR magic-bytes Poka-Yoke, contract
`publish-manifest-v1.yaml` v1.3.0) closes the three-format ship symmetry
— every shipped format (`.safetensors`, `.gguf`, `.apr`) now has a
pre-flight gate that aborts BEFORE any network I/O when the staged file
disagrees with the manifest. v1.0 scope for PM-009 is magic-bytes only:
first 4 bytes must be one of `APR\0`, `APRN`, `APR1`, `APR2`. The exact
class it catches is "wrong file staged under format=apr manifest" (e.g.
a GGUF renamed `.apr`, or a stray `.safetensors`). Tensor-index quant
validation deferred to v1.1. 45 unit tests on every push; real-artifact
dogfood evidence in §12.7.

**v2.6.0 amendment (2026-04-18):** the pre-flight gate set grew from seven
to eight. **FALSIFY-PM-008** (GGUF tensor-type Poka-Yoke, contract
`publish-manifest-v1.yaml` v1.2.1) closes the same ship-blocker class as
PM-007 but for the `.gguf` format. Evidence surfaced during the discharge
run that `general.file_type` is advisory: our own 8 GiB teacher GGUF ships
with stale `file_type = 0` (ALL_F32) despite fully Q4_K tensors, so PM-008
treats the **predominant GGML tensor type** as authoritative and the
metadata field as a fallback. Real-artifact verification at
`evidence/ship-two-001/ex-04-preflight-gate-smoketest.json`.

**v2.5.0 amendment (2026-04-18):** all seven ship manifest gates (PM-001..007)
now run inside `scripts/ship-two-001/ex-04-upload-hf.sh` as a pre-flight
Poka-Yoke. Any manifest-vs-artifact divergence aborts with non-zero exit
BEFORE any network I/O (contract `apr-cli-publish-extra-v1.yaml` v1.2.0,
`publish-manifest-v1.yaml` v1.1.0). Local validation shows all three ship
artifacts (`.apr`, `.safetensors`-fp16, `.gguf`) PASS every gate — the ship
is unblocked on `HF_TOKEN` alone.

---

## 1. Abstract

This specification defines the contract-first, falsification-driven plan to ship **production models**
through the aprender monorepo, proving end-to-end sovereignty (training → format → inference → eval) of the
Sovereign AI Stack.

**v2.0.0 scope change:** the original distilled-student artifact failed the 2026-04-17 contract-first
audit (see §1.5). The spec now pivots to an **expedited teacher-first ship** (see §12) while defering
distillation and sovereign training to follow-on releases. Either artifact reaching SHIP status
falsifies the null hypothesis "the stack cannot produce shippable weights"; the teacher-first ship
alone satisfies that falsification.

Original (v1.0.0) scope, retained for reference:
1. **MODEL-1 (apr-leaderboard):** A distilled Qwen2.5-Coder-7B student targeting **87.20% HumanEval pass@1**,
   shippable in **~36 engineering hours** from the current trained checkpoint.
2. **MODEL-2 (albor):** A sovereign, from-scratch **370M Python code-completion model** targeting **≥30%
   HumanEval pass@1**, shippable in **3–4 weeks** of compute + engineering.

All shipped artifacts must load via `apr run` (realizar backend), pass `apr qa` Golden Output gates, and
carry a contract-conforming manifest (`contracts/publish-manifest-v1.yaml`).

---

## 1.5. Audit Findings (2026-04-17)

**Verdict:** v1.0.0 MODEL-1 SHIP PATH IS BLOCKED. AC-SHIP1-005 falsified; teacher-first pivot in progress.

### 1.5.1 What was audited

Under contract `F-EVAL-HUMANEVAL-AUDIT-001` (`contracts/eval-harness-humaneval-v1.yaml` v1.1.0):
- Primary student checkpoint: `qwen2.5-coder-7b-distilled-v2-q4k.apr` (5.8 GB, Apr-3)
- Audit tool: `apr qa` + partial `apr eval --benchmark humaneval` via apr-leaderboard harness
- Binary: `/mnt/nvme-raid0/targets/aprender/release/apr` 0.31.0 (commit 9217e9c8a), RTX 4090

### 1.5.2 What we found

| Gate                 | Result            | Measured                                                           |
|----------------------|-------------------|--------------------------------------------------------------------|
| Capability Match     | ✓ PASS            | —                                                                  |
| Tensor Contract      | ✓ PASS            | 339 tensors pass PMAT-235                                          |
| Metadata Plausibility| ✓ PASS            | arch=qwen2, rope_theta=1000000                                     |
| **Golden Output**    | **✗ FAIL**        | For "2+2=" expected "4"; got `xxx9,x,x,,,,,,,,,,,,,999`            |
| Throughput           | ✓ PASS            | 9.7 tok/s (threshold=1)                                            |
| HumanEval pass@1     | **~0 (inferred)** | Batch output was incoherent BPE ("uardsylkoylkoiaÅĤ...") on 2/164  |
| Teacher pass@1       | 85.98 (prior run) | `results/humaneval_20260328_121327.json` — pipeline is sound       |

**The distilled student cannot generate coherent text.** Its tensors are structurally valid (all pass
shape, dtype, and non-finite checks) but its weights do not represent a working model.

### 1.5.3 Five-Whys (recorded in contract `validation_result_v1_1`)

1. **Why did the audit fail?** Student emits garbage BPE sequences on every prompt.
2. **Why is output garbage if weights load?** Tensor Contract validates structure, not semantics —
   weights with legal dtype+shape+finite values can still be a broken model.
3. **Why might weights be broken?** Three candidates, in decreasing likelihood:
   (a) distillation diverged and the run was saved without a sanity gate;
   (b) `apr convert --quantize q4_k_m` introduced a LAYOUT-001-class transpose bug;
   (c) BPE tokenizer / chat-template drift so generation samples from wrong token space.
4. **Why can't we tell which?** Diagnostics (`apr diff`, merged-checkpoint run, tokenizer round-trip)
   were not required gates — they remain diagnostic follow-ups in the contract.
5. **Why did no earlier gate catch this?** `apr qa` Tensor Contract exits PASS before Golden Output
   runs; Golden Output failure does NOT block publish in the current gate matrix. This is the
   root contract gap, now promoted to the expedited plan's first action (§12.1).

### 1.5.4 Notable gap surfaced

The 87.20% figure traces back to recipe-h-32b-distill.yaml's comment labelling the *base 7B-Instruct
few-shot* HumanEval — not a distilled-student zero-shot run. No `apr eval` result file for a
distilled student exists in `apr-leaderboard/results/`; all 17 archived HumanEval runs measure the
teacher. The headline number in v1.0.0 §4.1 was therefore never a reproducible claim.

---

## 2. Motivation

### 2.1 Why These Two Models

| Criterion                    | MODEL-1 (apr-leaderboard) | MODEL-2 (albor)        |
|------------------------------|---------------------------|------------------------|
| Current state                | trained, needs packaging  | architecture designed, pretraining required |
| Engineering distance to SHIP | 36 h                      | 3–4 weeks              |
| Proves distillation path     | yes                       | no                     |
| Proves sovereign path        | partial (uses HF teacher) | **yes (end-to-end)**   |
| Proves eval harness          | yes (HumanEval)           | yes (HumanEval)        |
| Risk profile                 | LOW                       | MEDIUM-HIGH (training) |

Shipping both gives orthogonal proof: one demonstrates the stack can finish what PyTorch started;
the other demonstrates the stack can start AND finish without PyTorch in the loop.

### 2.2 Explicit Non-Goals (v1)

- Not shipping: `entrenar-rl` (POC), `entrenar-rlhf` (POC), `verificar-agent` (research).
- Not targeting: chat tuning, multimodal, tool use, >10B params.
- Not blocking on: full leaderboard automation (post-SHIP), wandb integration, distributed training.

---

## 3. Design Principles

| #  | Principle                   | Operationalization                                                    |
|----|-----------------------------|-----------------------------------------------------------------------|
| 1  | Contract-first              | Every weight file, config, and eval path has a YAML contract BEFORE code |
| 2  | Falsification-driven        | Every acceptance criterion has a named, executable FALSIFY-* test     |
| 3  | Sovereign                   | No PyTorch in the production path; GGUF/APR/SafeTensors only          |
| 4  | Lean on existing artifacts  | Reuse `contracts/model-families/qwen2.yaml` and `llama.yaml` — do not fork |
| 5  | Dogfood tooling             | `apr qa`, `apr bench`, `apr trace`, `apr eval` — never bespoke scripts |
| 6  | Binary gates                | Every GATE-SHIP-* is pass/fail; no partial credit                     |
| 7  | Five-Whys on failure        | Any FALSIFY-* failure triggers documented Hansei (§10) before retry   |
| 8  | Zero tolerance              | We never accept bugs or poor performance. Defects and perf regressions are both blockers, not trade-offs. All work improves or holds the line; never degrades it. No "pre-existing" carve-outs. No `#[ignore]` as a release valve. |

---

## 4. Model 1 — apr-leaderboard (Distilled Qwen2.5-Coder-7B)

> **⚠ 2026-04-17 audit (v2.0.0):** The student checkpoint that was the subject of this section
> produces garbage tokens (see §1.5). MODEL-1 v1.0.0 as specified cannot ship. This section is
> retained unchanged as historical scope; the path forward is in §12 (teacher-first expedited ship).

### 4.1 Current State

- Teacher: `Qwen/Qwen2.5-Coder-7B-Instruct` (matches `contracts/model-families/qwen2.yaml` 7B variant).
- Student: same architecture, distilled on 20K code-instruction pairs.
- ~~Measured: **87.20% HumanEval pass@1** (source: POC notebook, pre-audit).~~
  **[v2.0.0] Falsified 2026-04-17:** this figure was a pre-distillation *few-shot* teacher score
  mis-attributed to the distilled student. The distilled checkpoint's actual pass@1 under `apr eval`
  is ~0 (garbage output). See §1.5 and `contracts/eval-harness-humaneval-v1.yaml` v1.1.0.
- Format: SafeTensors (HF-native), not yet exported to GGUF or APR.
- Eval: ran on reference Python harness; `apr eval` run terminated after garbage output detected.

### 4.2 Acceptance Criteria

| ID            | Criterion                                                                 | Verification            |
|---------------|---------------------------------------------------------------------------|-------------------------|
| AC-SHIP1-001  | Student weights load via `realizar::Model::load_safetensors`              | FALSIFY-SHIP-001 **(PARTIAL_ALGORITHM_LEVEL v2.32.0)** |
| AC-SHIP1-002  | `apr run <model>.safetensors --prompt "def fib(n):"` emits valid Python   | FALSIFY-SHIP-002 **(PARTIAL_ALGORITHM_LEVEL v2.26.0)** |
| AC-SHIP1-003  | Convert to APR via `apr convert --quantize q4_k_m`; round-trip weights match (cos ≥ 0.999) | FALSIFY-SHIP-003 **(PARTIAL_ALGORITHM_LEVEL v2.30.0)** |
| AC-SHIP1-004  | Export to GGUF via `apr export --format gguf`; loads in llama.cpp         | FALSIFY-SHIP-004 **(PARTIAL_ALGORITHM_LEVEL v2.31.0)** |
| AC-SHIP1-005  | `apr eval --benchmark humaneval` reproduces ≥86.00% pass@1 (allow 1.2% noise) | FALSIFY-SHIP-005 **(PARTIAL_ALGORITHM_LEVEL v2.27.0)** |
| AC-SHIP1-006  | `apr qa <model>` — all 8 gates PASS (Golden Output, layout, tensor stats, etc.) | FALSIFY-SHIP-006 **(PARTIAL_ALGORITHM_LEVEL v2.25.0)** |
| AC-SHIP1-007  | `apr bench` decode throughput ≥30 tok/s on RTX 4090 (7B Q4_K target)      | FALSIFY-SHIP-007 **(PARTIAL_ALGORITHM_LEVEL v2.29.0)** |
| AC-SHIP1-008  | Chat template (`contracts/chat-template-v1.yaml`) applies cleanly        | FALSIFY-SHIP-008 **(PARTIAL_ALGORITHM_LEVEL v2.24.0)** |
| AC-SHIP1-009  | License & provenance recorded in `model.apr` metadata (Qwen2 Apache-2.0) | FALSIFY-SHIP-009 **(PARTIAL_ALGORITHM_LEVEL v2.33.0)** |
| AC-SHIP1-010  | Published artifact URL resolves; SHA-256 matches manifest                 | FALSIFY-SHIP-010 **(PARTIAL_ALGORITHM_LEVEL v2.28.0)** |

### 4.3 Critical Path (MODEL-1)

```
[checkpoint.safetensors] ──► AC-001 load ──► AC-002 run ──► AC-005 eval (baseline)
                                                 │                    │
                                                 ▼                    ▼
                                        AC-008 chat-template   AC-006 qa gates
                                                 │                    │
                                                 ▼                    ▼
                                         AC-003 convert ──► AC-007 bench
                                                 │
                                                 ▼
                                         AC-004 export gguf
                                                 │
                                                 ▼
                                         AC-009 metadata ──► AC-010 publish
```

### 4.4 Contract Registry (MODEL-1)

Leverages 28 existing contracts from the apr-leaderboard POC, promoted into the monorepo:

| Kind             | Contract                                              | Status      |
|------------------|-------------------------------------------------------|-------------|
| model-family     | `contracts/model-families/qwen2.yaml`                 | EXISTS      |
| tensor-layout    | `contracts/tensor-layout-v1.yaml`                     | EXISTS      |
| chat-template    | `contracts/chat-templates-v1.yaml` (qwen2 variant)    | EXISTS      |
| eval-harness     | `contracts/eval-harness-humaneval-v1.yaml`            | **NEW**     |
| distillation     | `contracts/distillation-pipeline-v1.yaml`             | **NEW**     |
| publish-manifest | `contracts/publish-manifest-v1.yaml`                  | **NEW**     |

---

## 5. Model 2 — albor (Sovereign 370M Python Code Completion)

### 5.1 Current State

- Architecture: LLaMA-family decoder, 370M params (hidden=1024, layers=24, heads=16, kv_heads=4).
  Slot: registered as a new variant under `contracts/model-families/llama.yaml` `370m`.
- Tokenizer: BPE over 50K vocab, Python-biased corpus.
- Training data: 60GB deduplicated Python (The Stack v2 filtered subset).
- Target: ≥30% HumanEval pass@1 (baseline reference: CodeParrot 1.1B ≈ 4%, StarCoderBase 1B ≈ 15.4%).
- Current blocker: pretraining run not yet executed end-to-end via `entrenar` CUDA path.

### 5.2 Acceptance Criteria

| ID            | Criterion                                                                 | Verification           |
|---------------|---------------------------------------------------------------------------|------------------------|
| AC-SHIP2-001  | Architecture registered in `contracts/model-families/llama.yaml` 370m     | FALSIFY-SHIP-011 **(DISCHARGED v2.21.0)** |
| AC-SHIP2-002  | Tokenizer trained; `apr tokenize` round-trip exact on 10K held-out docs   | FALSIFY-SHIP-012 **(PARTIAL_ALGORITHM_LEVEL v2.21.0)** |
| AC-SHIP2-003  | `entrenar` pretraining loop reaches target loss (CE ≤ 2.2 on val)         | FALSIFY-SHIP-013 **(PARTIAL_ALGORITHM_LEVEL v2.38.0)** |
| AC-SHIP2-004  | Training on RTX 4090 completes within 21 days (hardware budget)           | FALSIFY-SHIP-014 **(PARTIAL_ALGORITHM_LEVEL v2.38.0)** |
| AC-SHIP2-005  | Checkpoint weights saved as `.apr` (native format, no PyTorch)            | FALSIFY-SHIP-015 **(PARTIAL_ALGORITHM_LEVEL v2.21.0)** |
| AC-SHIP2-006  | `apr qa <model>.apr` — all 8 gates PASS                                   | FALSIFY-SHIP-016 **(PARTIAL_ALGORITHM_LEVEL v2.37.0)** |
| AC-SHIP2-007  | `apr run` produces syntactically valid Python on 100 held-out prompts     | FALSIFY-SHIP-017 **(PARTIAL_ALGORITHM_LEVEL v2.34.0)** |
| AC-SHIP2-008  | `apr eval --benchmark humaneval` ≥30.0% pass@1                            | FALSIFY-SHIP-018 **(PARTIAL_ALGORITHM_LEVEL v2.36.0)** |
| AC-SHIP2-009  | GGUF export loads in llama.cpp AND produces matching tokens (tol ≤ 1e-3)  | FALSIFY-SHIP-019 **(PARTIAL_ALGORITHM_LEVEL v2.22.0)** |
| AC-SHIP2-010  | `apr bench` decode ≥100 tok/s on RTX 4090 (370M target)                   | FALSIFY-SHIP-020 **(PARTIAL_ALGORITHM_LEVEL v2.35.0)** |
| AC-SHIP2-011  | Training reproducible: seed fixed, two runs produce identical first 100 steps | FALSIFY-SHIP-021 **(DISCHARGED v2.20.0)** |
| AC-SHIP2-012  | Weights + tokenizer + config published with CC-BY-4.0 data provenance     | FALSIFY-SHIP-022 **(DISCHARGED v2.20.0)** |

### 5.3 Critical Path (MODEL-2)

```
[llama.yaml 370m entry] ──► AC-001 ──► AC-002 tokenizer
                                               │
                                               ▼
                                      AC-011 reproducibility check (dry run, 100 steps)
                                               │
                                               ▼
                                      AC-003 pretraining loop
                                               │
                                               ▼
                                      AC-004 hardware budget ── (MONITOR) ──► AC-005 save .apr
                                                                                    │
                                                              ┌─────────────────────┼─────────────────────┐
                                                              ▼                     ▼                     ▼
                                                       AC-006 qa gates      AC-007 run valid     AC-008 humaneval
                                                                                                         │
                                                                             ┌───────────────────────────┤
                                                                             ▼                           ▼
                                                                      AC-009 gguf export         AC-010 bench
                                                                             │
                                                                             ▼
                                                                      AC-012 publish
```

### 5.4 Contract Registry (MODEL-2)

Leverages 54 contracts from the albor POC, promoted into the monorepo:

| Kind             | Contract                                                   | Status  |
|------------------|------------------------------------------------------------|---------|
| model-family     | `contracts/model-families/llama.yaml` (add 370m variant)   | AMEND   |
| tokenizer        | `contracts/tokenizer-bpe-v1.yaml`                          | **NEW** |
| dataset          | `contracts/dataset-thestack-python-v1.yaml`                | **NEW** |
| training-loop    | `contracts/training-loop-pretrain-v1.yaml`                 | **NEW** |
| checkpoint       | `contracts/checkpoint-apr-native-v1.yaml`                  | **NEW** |
| eval-harness     | `contracts/eval-harness-humaneval-v1.yaml` (shared)        | SHARED  |
| publish-manifest | `contracts/publish-manifest-v1.yaml` (shared)              | SHARED  |

---

## 6. Compound Ship Gates

All gates are binary; any failure blocks publish.

| Gate             | Description                                                    | Blocks          |
|------------------|----------------------------------------------------------------|-----------------|
| GATE-SHIP-001    | MODEL-1: all 10 AC-SHIP1-* PASS **(PARTIAL_ALGORITHM_LEVEL v2.42.0)**          | MODEL-1 publish |
| GATE-SHIP-002    | MODEL-2: all 12 AC-SHIP2-* PASS **(PARTIAL_ALGORITHM_LEVEL v2.42.0)**          | MODEL-2 publish |
| GATE-SHIP-003    | Both models: `apr qa` Golden Output never regresses post-quantize **(PARTIAL_ALGORITHM_LEVEL v2.42.0)** | publish     |
| GATE-SHIP-004    | HumanEval harness produces identical score on two consecutive runs (seed=0) **(PARTIAL_ALGORITHM_LEVEL v2.42.0)** | AC-005, AC-008 |
| GATE-SHIP-005    | License metadata is present AND matches upstream declaration **(PARTIAL_ALGORITHM_LEVEL v2.42.0)**   | publish         |
| GATE-SHIP-006    | GGUF round-trip: APR → GGUF → load in llama.cpp → first-token match (tol ≤ 1e-3) **(PARTIAL_ALGORITHM_LEVEL v2.42.0)** | AC-004, AC-009 |
| GATE-SHIP-007    | No unwrap() in new code (enforced by `.clippy.toml`) **(PARTIAL_ALGORITHM_LEVEL v2.43.0)**          | merge           |
| GATE-SHIP-008    | Contract density: every new public fn has `#[contract]` **(PARTIAL_ALGORITHM_LEVEL v2.43.0)**        | merge           |
| GATE-SHIP-009    | CI green: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test --workspace` **(PARTIAL_ALGORITHM_LEVEL v2.43.0)** | merge |
| GATE-SHIP-010    | `cargo deny check advisories` — zero vulnerabilities in weight/tokenizer dependencies **(PARTIAL_ALGORITHM_LEVEL v2.43.0)** | merge |
| GATE-SHIP-011    | PMAT quality score ≥ A- (project), TDG ≥ 90 **(PARTIAL_ALGORITHM_LEVEL v2.43.0)**                    | merge           |
| GATE-SHIP-012    | Coverage ≥ 95% line on new modules (per `.pmat-gates.toml`) **(PARTIAL_ALGORITHM_LEVEL v2.43.0)**    | merge           |

---

## 7. Falsification Tests

Each test is named, executable, and has a defined failure signal.

### 7.1 MODEL-1 Falsification (12 tests)

| ID                 | Test                                                              | Failure Signal                         |
|--------------------|-------------------------------------------------------------------|----------------------------------------|
| FALSIFY-SHIP-001   | `realizar::Model::load_safetensors(path)` returns Ok (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-001` in `contracts/qwen2-e2e-verification-v1.yaml` v1.6.0; `cargo test -p aprender-core --lib format::ship_001`) | Err(_) returned |
| FALSIFY-SHIP-002   | Run `apr run ... --prompt "def fib(n):"`; parse output as Python AST (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-002` in `contracts/qwen2-e2e-verification-v1.yaml` v1.1.0; `cargo test -p aprender-core --lib falsify_ship_002_python_syntax_error_threshold_logic`) | SyntaxError (> 0 errors) |
| FALSIFY-SHIP-003   | Convert then compare per-layer cosine similarity (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-003` in `contracts/qwen2-e2e-verification-v1.yaml` v1.4.0; `cargo test -p aprender-core --lib format::ship_003`) | any layer cos < 0.999 |
| FALSIFY-SHIP-004   | Shell out to `llama-cli` on exported GGUF; prompt → logits (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-004` in `contracts/qwen2-e2e-verification-v1.yaml` v1.5.0; `cargo test -p aprender-core --lib format::ship_004`) | llama.cpp exit ≠ 0 |
| FALSIFY-SHIP-005   | Run HumanEval 164× via `apr eval`; pass@1 computed (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-005` in `contracts/qwen2-e2e-verification-v1.yaml` v1.2.0; `cargo test -p aprender-core --lib falsify_ship_005_humaneval_pass_at_1_threshold_logic`) | pass@1 < 86.00% (or < 84.80% under 1.2 pp noise allowance) |
| FALSIFY-SHIP-006   | `apr qa <model>` exit code = 0 (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QA-SHIP-006` in `contracts/apr-model-qa-v1.yaml` v1.2.0; `cargo test -p aprender-core --lib falsify_ship_006_apr_qa_eight_gates_aggregate`) | any gate reports FAIL |
| FALSIFY-SHIP-007   | `apr bench --iterations 5 --max-tokens 128`; median tok/s (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-007` in `contracts/qwen2-e2e-verification-v1.yaml` v1.3.0; `cargo test -p aprender-core --lib falsify_ship_007_decode_tps_threshold_logic`) | median < 30 |
| FALSIFY-SHIP-008   | Render chat template on canonical system+user; diff vs golden (**PARTIAL_ALGORITHM_LEVEL** — `GATE-CHAT-SHIP-008` in `contracts/chat-template-v1.yaml` v1.1.0; `cargo test -p aprender-core --lib falsify_ship_008_chat_template_render_bind`) | diff ≠ 0 |
| FALSIFY-SHIP-009   | `apr inspect <model>.apr`; grep for `license: apache-2.0` (**PARTIAL_ALGORITHM_LEVEL** — `GATE-APR-PROV-004` in `contracts/apr-provenance-v1.yaml` v1.1.0; `cargo test -p aprender-core --lib provenance`) | missing or mismatched |
| FALSIFY-SHIP-010   | curl + sha256sum against manifest (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-SHIP-010` in `contracts/publish-manifest-v1.yaml` v1.4.0; `cargo test -p aprender-core --lib format::ship_010`) | hash mismatch or 404 |
| FALSIFY-SHIP-023   | Re-run AC-005 on second day; score drift (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-023` in `contracts/qwen2-e2e-verification-v1.yaml` v1.7.0; `cargo test -p aprender-core --lib format::ship_023`) | drift > 1.2 pp |
| FALSIFY-SHIP-024   | Prompt-injection torture suite (50 adversarial inputs) (**PARTIAL_ALGORITHM_LEVEL** — `FALSIFY-QW2E-SHIP-024` in `contracts/qwen2-e2e-verification-v1.yaml` v1.7.0; `cargo test -p aprender-core --lib format::ship_024`) | any panic or NaN in logits |

### 7.2 MODEL-2 Falsification (10 tests)

| ID                 | Test                                                              | Failure Signal                         |
|--------------------|-------------------------------------------------------------------|----------------------------------------|
| FALSIFY-SHIP-011   | `llama.yaml` `370m` entry validates against `_schema.yaml` (**DISCHARGED v2.21.0** — `GATE-ARCH-370M-001` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.1.0 ACTIVE; `cargo test -p aprender-train --lib llama_370m`) | schema error |
| FALSIFY-SHIP-012   | Tokenize 10K docs, detokenize, byte-compare (**PARTIAL_ALGORITHM_LEVEL** — `GATE-BPE-003` in `contracts/tokenizer-bpe-v1.yaml` v1.1.0; `cargo test -p apr-cli --test falsify_ship_012_tokenizer_roundtrip`) | any byte mismatch |
| FALSIFY-SHIP-013   | Training val CE at final step (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-013` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.10.0; `cargo test -p aprender-train --lib ship_013`) | CE > 2.2 |
| FALSIFY-SHIP-014   | Wall-clock from train start to final checkpoint (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-014` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.10.0; `cargo test -p aprender-train --lib ship_014`) | > 21 days |
| FALSIFY-SHIP-015   | Load checkpoint via `apr inspect`; count params (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-003` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.2.0; `cargo test -p aprender-train --lib llama_370m`) | params ≠ 370M ± 1% |
| FALSIFY-SHIP-016   | `apr qa <model>.apr` exit code (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-008` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.9.0; `cargo test -p aprender-train --lib llama_370m`) | any gate FAIL |
| FALSIFY-SHIP-017   | 100 prompts → Python AST parse (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-005` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.6.0; `cargo test -p aprender-train --lib llama_370m`) | ≥2 SyntaxError (tolerate ≤1) |
| FALSIFY-SHIP-018   | `apr eval --benchmark humaneval` pass@1 (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-007` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.8.0; `cargo test -p aprender-train --lib llama_370m`) | < 30.0% |
| FALSIFY-SHIP-019   | GGUF export; first-token probability vs APR (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-004` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.3.0; `cargo test -p aprender-train --lib llama_370m`) | |Δp| > 1e-3 on top-1 |
| FALSIFY-SHIP-020   | `apr bench` median tok/s on RTX 4090 (**PARTIAL_ALGORITHM_LEVEL** — `GATE-ARCH-370M-006` in `contracts/model-families/llama-370m-sovereign-v1.yaml` v1.6.0; `cargo test -p aprender-train --lib llama_370m`) | < 100 |
| FALSIFY-SHIP-021   | Run training 100 steps × 2 with seed=0; diff loss trajectories (**DISCHARGED v2.20.0**) | any step diff > 1e-6 |
| FALSIFY-SHIP-022   | `apr inspect`; check `license`, `data_source`, `data_license` (**DISCHARGED v2.20.0** — `GATE-APR-PROV-001..003` in `contracts/apr-provenance-v1.yaml` v1.0.0 ACTIVE) | any field missing |

---

## 8. Execution Plan

### 8.1 Phase DAG

```
                    ┌─────────────────────────────────────┐
                    │          Phase 0: Scaffold          │
                    │  (contracts, schema, test harness)  │
                    └──────────────────┬──────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      │                                 │
                      ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │   Phase 1-A (MODEL-1)   │       │   Phase 1-B (MODEL-2)   │
         │  36h — packaging path   │       │  Week 1: tokenizer + dry │
         │                         │       │  run (AC-001,002,011)    │
         │  AC-001..004 load/conv  │       └───────────┬─────────────┘
         └───────────┬─────────────┘                   │
                     ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │  Phase 2-A: eval + qa   │       │  Phase 2-B: pretraining │
         │  AC-005..008            │       │  Weeks 2-3: AC-003,004  │
         └───────────┬─────────────┘       └───────────┬─────────────┘
                     ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │  Phase 3-A: publish     │       │  Phase 3-B: eval + qa   │
         │  AC-009,010  — SHIP-1   │       │  Week 4: AC-005..012    │
         └─────────────────────────┘       └───────────┬─────────────┘
                                                       ▼
                                           ┌─────────────────────────┐
                                           │  Phase 4-B: publish     │
                                           │  SHIP-2                 │
                                           └─────────────────────────┘
```

Phase 1-A and Phase 1-B are independent and run in parallel.

### 8.2 Effort Budget

| Phase | Model    | Effort       | Calendar  | Owner        |
|-------|----------|--------------|-----------|--------------|
| 0     | shared   | 6 h          | day 0     | eng          |
| 1-A   | MODEL-1  | 10 h         | day 1     | eng          |
| 2-A   | MODEL-1  | 12 h         | day 2     | eng          |
| 3-A   | MODEL-1  | 4 h          | day 3     | eng          |
| 1-B   | MODEL-2  | 40 h         | week 1    | eng          |
| 2-B   | MODEL-2  | compute-bound | weeks 2-3 | GPU node    |
| 3-B   | MODEL-2  | 16 h         | week 4    | eng          |
| 4-B   | MODEL-2  | 4 h          | week 4    | eng          |
| **Σ** |          | 92 h + 2 wk  | ~4 weeks  |              |

### 8.3 Integration with `apr run`

After SHIP, both models must satisfy:

```bash
apr run ./model-1.apr --prompt "def quicksort(arr):"       # MODEL-1
apr run ./model-2.apr --prompt "def binary_search(xs, t):" # MODEL-2
```

Both paths resolve through `realizar::Model` (see `crates/aprender-serve/CLAUDE.md` Realizar-First Architecture).
No code path in `aprender-core` may invoke generation.

---

## 9. Risk Matrix

| # | Risk                                           | Probability | Impact | Mitigation                                                           |
|---|------------------------------------------------|-------------|--------|----------------------------------------------------------------------|
| 1 | HumanEval eval non-deterministic               | MED         | HIGH   | seed=0, greedy; GATE-SHIP-004 enforces two-run identity              |
| 2 | GGUF export has tensor-layout bug (LAYOUT-001) | HIGH        | HIGH   | FALSIFY-SHIP-019 parity check; reuse `layout_contract.rs` validator  |
| 3 | MODEL-2 training diverges                      | MED         | HIGH   | AC-SHIP2-003 loss gate; fallback = reduce LR, resume from ckpt       |
| 4 | RTX 4090 insufficient for 370M in 21 days      | LOW         | HIGH   | AC-SHIP2-004 budget; overflow → rent 2× H100 week 3                  |
| 5 | Teacher (Qwen2.5-Coder-7B) license ambiguity   | LOW         | MED    | AC-SHIP1-009; confirm Apache-2.0 in `config.json`                    |
| 6 | `apr convert` quantize drops accuracy > 1pp    | MED         | MED    | FALSIFY-SHIP-003 cos-sim gate; fallback = Q5_K_M                     |
| 7 | Tokenizer round-trip bytes mismatch            | LOW         | HIGH   | FALSIFY-SHIP-012 on 10K corpus                                       |
| 8 | `cargo install aprender` breaks during release | MED         | HIGH   | CI `cargo install --path .` smoke test (GATE-SHIP-009)               |
| 9 | HF artifact hosting outage                     | LOW         | LOW    | dual-publish to HF + self-hosted bucket                              |
| 10| CUDA JIT regression (trueno#200/203)           | MED         | MED    | Pin trueno version per memory note 2026-03-22                        |

---

## 10. Failure Protocol (Hansei)

Any FALSIFY-SHIP-* failure triggers the following sequence. No retry is permitted before completion.

### 10.1 Five Whys

1. What check failed? (name the FALSIFY-*)
2. What invariant did it violate?
3. What code path was responsible?
4. Why was the code path wrong? (bug class: layout, numeric, eval harness, toolchain)
5. Why did no earlier gate catch it? (contract gap → file follow-up)

### 10.2 Decision Gate

| Condition                                              | Action                         |
|--------------------------------------------------------|--------------------------------|
| Single test failed, root cause known, fix ≤ 2 h        | Fix + re-run full gate         |
| Multiple tests failed OR root cause unknown            | Escalate to design review      |
| AC breaks but SHIP is blocking deadline                | Reduced-gate ship (below)      |
| Hardware / compute budget breached                     | Full failure escalation        |

### 10.3 Reduced-Gate Ship (emergency only)

Acceptable only with written Noah Gift approval. A reduced ship may drop:
- AC-SHIP1-007 / AC-SHIP2-010 (bench speed) — ship with "beta performance" label
- AC-SHIP1-010 / AC-SHIP2-012 (artifact publication) — ship to internal bucket only

All 8 `apr qa` gates (Golden Output, layout, tensor stats, license, etc.) MUST still pass.

### 10.4 Full Failure Escalation

If neither model ships within the budget, this specification is void. Retrospective must
answer: (a) which assumption was wrong, (b) what contract should have caught it earlier,
(c) what gets deleted from scope before restart.

---

## 12. Expedited Ship Plan (v2.0.0 — teacher-first)

**Goal:** publish ONE artifact within **10 engineering hours** of 2026-04-17 to falsify the null
hypothesis "the stack cannot produce shippable weights."

**Strategy:** ship the **teacher** (`qwen2.5-coder-7b-instruct-q4k.apr`) under a new artifact ID
`paiml/qwen2.5-coder-7b-apache-q4k-v1`. Defer distillation proof to v1.1. Defer MODEL-2 to v2.0.

### 12.1 Pre-requisite: plug the Golden Output gate gap

Before any publish, `apr qa` must be configured so that **Golden Output failure blocks publish**.
Today it is reported but non-fatal — exactly the hole that let the v1.0.0 plan rely on a
garbage checkpoint for 14 days before audit. Track as contract amendment to `apr-qa-v1.yaml`
(or equivalent); must land before §12.2.

### 12.2 Teacher-first critical path (10h budget)

```
[qwen2.5-coder-7b-instruct-q4k.apr]      # already in apr-leaderboard/checkpoints/, 7.5 GB
         │
         ▼
  EX-01  apr qa --require-golden-output   # must PASS after §12.1 gate fix  (1 h)
         │
         ▼
  EX-02  apr eval --benchmark humaneval   # reproduces ≥84.5 pass@1 (noise-band of 85.98)  (2 h)
         │
         ▼
  EX-03  Write contracts/publish-manifest-v1.yaml entry    (1 h)
           - sha256, size_bytes, license=Apache-2.0
           - provenance.pipeline=finetune
           - provenance.parent=Qwen/Qwen2.5-Coder-7B-Instruct
           - provenance.recipe=contracts/model-families/qwen2.yaml
         │
         ▼
  EX-04  Upload artifact to HF Hub AND self-hosted bucket  (2 h)
         │
         ▼
  EX-05  Verify manifest: sha256 match, URL 200, SPDX valid  (1 h)
         │
         ▼
  EX-06  apr pull <published_id> → local file; re-run EX-02 from downloaded artifact  (2 h)
         │
         ▼
  EX-07  Tag release in spec + announce  (1 h)
```

### 12.3 Expedited Acceptance Criteria

| ID            | Criterion                                                                        | Verification        |
|---------------|----------------------------------------------------------------------------------|---------------------|
| AC-EX-001     | Golden Output gate is a HARD BLOCKER in `apr qa`                                 | FALSIFY-EX-001      |
| AC-EX-002     | Teacher passes all 8 `apr qa` gates including Golden Output                      | FALSIFY-EX-002      |
| AC-EX-003     | `apr eval --benchmark humaneval` on teacher ≥84.5% pass@1 (85.98 − 1.5 noise)   | FALSIFY-EX-003      |
| AC-EX-004     | `publish-manifest-v1.yaml` instance for artifact passes `apr validate-manifest`  | FALSIFY-EX-004      |
| AC-EX-005     | `apr pull paiml/qwen2.5-coder-7b-apache-q4k-v1` resolves + SHA-256 matches       | FALSIFY-EX-005      |
| AC-EX-006     | `apr run <published>.apr --prompt "def fib(n):"` emits syntactically valid Python | FALSIFY-EX-006     |
| AC-EX-007     | Parallel eval lane: N-shard run on ≥2 hosts matches single-host pass@1 (Δ ≤ 0.01 pp) and completes in ≤ `single_host_wall_time / N × 1.25` | FALSIFY-SHARD-001..004 |

### 12.4 Explicit Scope Cut (v2.0.0)

Moved out of v1 ship:
- **Distilled student artifact** → v1.1 (requires diagnosis per `validation_result_v1_1` ACT-01..03,
  then re-distillation with contract-gated Golden Output at each epoch).
- **MODEL-2 (albor sovereign)** → v2.0 (3+ weeks of compute; no reason to couple to MODEL-1 ship).
- **GGUF round-trip export** (AC-SHIP1-004) → v1.1 (teacher already has GGUF on HF).

### 12.5 What falsifies the expedited plan

| Condition                                         | Action                                              |
|---------------------------------------------------|-----------------------------------------------------|
| AC-EX-002 FAIL on teacher                         | Pipeline regressed — block ship, investigate realizar |
| AC-EX-003 pass@1 < 84.5                           | Harness drift since 2026-03-28; do not ship until resolved |
| AC-EX-004 manifest invalid                        | Fix manifest schema compliance, retry                |
| AC-EX-005 SHA-256 mismatch                        | Re-upload; investigate CDN/transit corruption        |
| Any EX-* step takes > 2× budget                   | Escalate; triggers §13.2 retrospective update        |
| Shard merged pass@1 differs from single-host by > 0.01 pp | Parity FAIL — block ship, investigate shard determinism (FALSIFY-SHARD-003) |
| Any shard reports missing / duplicate task_ids    | Completeness or disjointness FAIL — block ship (FALSIFY-SHARD-001/002) |

### 12.6 Parallel Eval Lane (post-hoc lesson, 2026-04-17)

**Problem surfaced during v2.0.0 ship.** EX-02 (single-host HumanEval on yoga) ran
serially for ~2 hours while `gx10` (Blackwell GB10, `apr-cli` inference unaffected
by PMAT-587 JIT issues) and any Lambda-Labs GPU instance sat idle. 5-Whys (recorded
in contract `eval-sharding-v1.yaml::five_whys`):

1. **Why only yoga?** The orchestration script accepted a single `MODEL_PATH` and
   invoked `apr run --batch-jsonl` once, consuming all 164 tasks serially.
2. **Why serial batch?** `eval-pass-at-k.sh` (inherited from apr-leaderboard) has
   no shard dimension; it assumes one GPU.
3. **Why wasn't sharding added?** EX-02 was treated as a monolithic §12.2 step;
   decomposing "generate N completions" into `generate N/k × k hosts` was not
   considered because the 10h budget was written assuming yoga alone.
4. **Why was the budget yoga-alone?** `gx10` was mentally categorized as "training,
   blocked on JIT bug" without separating the inference path, which works today.
5. **Root cause.** Spec optimized for *matching existing tooling* (one-host eval
   harness) instead of *minimizing critical path*. A 2-way shard cuts EX-02 from
   ~2h → ~1h; a 3-way shard (yoga+gx10+Lambda) to ~40 min.

#### 12.6.1 Scope

This lane is **post-hoc for v2.0.0** (sunk cost on the in-flight serial run) and
**pre-requisite for v1.1 / v2.0** future evals (distilled student, MODEL-2 sovereign,
multi-seed reproducibility runs per FALSIFY-PUBLISH-RECIPE-001).

#### 12.6.2 Architecture

```
benchmark.jsonl  ──(round-robin split, stride N)──►  shard_0.jsonl … shard_{N-1}.jsonl
                                                           │ │ … │
                                                           ▼ ▼   ▼
                                                       host_0 host_1 … host_{N-1}
                                                       (yoga) (gx10) … (lambda)
                                                           │ │ … │
                                                           ▼ ▼   ▼
                                                   humaneval_shard_i.json (each host)
                                                           │ │ … │
                                                           └─┴───┘
                                                              ▼
                                          eval-shard-merge.py: concat problems[],
                                          recompute Chen pass@1 → humaneval_merged.json
```

- **Shard algorithm.** Round-robin stride: task `i` goes to host `i mod N`.
  Evens out per-task cost variance (long prompts, long generations) without
  needing a pre-estimated cost model.
- **Model sync.** `rsync -c` (content-checksum) pushes the .apr + tokenizer to
  each host once; subsequent runs are no-ops.
- **Merge.** Per-shard result JSONs share the `eval-pass-at-k.sh` schema
  (`problems[]`, `results.passed`, `results.total`). Merge = concat `problems`,
  sum totals, recompute pass@1 using Chen et al. unbiased estimator on merged
  array.

#### 12.6.3 Acceptance (AC-EX-007 discharge)

Run the 4 FALSIFY-SHARD tests in `contracts/eval-sharding-v1.yaml`:

- **FALSIFY-SHARD-001 (completeness):** `sum(shard_i.total) == benchmark.total`
  and every benchmark task_id appears in exactly one shard result.
- **FALSIFY-SHARD-002 (disjointness):** no task_id appears in two shards.
- **FALSIFY-SHARD-003 (determinism parity):** at temperature=0.0, completions for
  task T on host A == completions for task T on host B for a 16-task probe set.
- **FALSIFY-SHARD-004 (merged-score identity):** reshard an existing single-host
  humaneval_*.json result by task_id; merged pass@1 matches within 0.01 pp of the
  original.

Evidence location: `evidence/ship-two-001/shard-eval/`.

#### 12.6.4 Non-goals for this lane

- **Dynamic load-balancing.** Static stride-N is sufficient for ≤5 hosts and
  benchmarks under a few thousand tasks.
- **Remote-managed model caches.** `rsync -c` on each invocation is <2 min on
  gigabit for a 7.5 GB .apr; optimizing further is premature.
- **Fault-tolerant shard retry.** If one host dies mid-run, operator re-runs the
  missing shard manually — no automatic reassignment. (Revisit for v1.1 if
  experienced in practice.)

### 12.7 Dogfood Gate + Three-Format Ship (2026-04-18 amendment)

**Problem surfaced during EX-04.** The first-cut `ex-04-upload-hf.sh` called
`uv run --with huggingface-hub python3` instead of our own product. That is the
same failure class as §13.2 cause 7 ("Tooling investment vs tooling usage"):
we have `apr publish`, and we should be shipping through it.

Two product gaps had to be closed before EX-04 could run through `apr publish`:

1. `apr publish` did not natively consume `publish-manifest-v1.yaml` or upload
   arbitrary sidecar files (tokenizer.json, per-format manifests).
2. No contract stated that ships must be published in multiple ecosystem
   formats, and no contract stated the required safetensors dtype.

Both gaps are now closed by **contract `contracts/apr-cli-publish-extra-v1.yaml`
(F-PUBLISH-EXTRA-001)**, a peer of `publish-manifest-v1.yaml` that adds:

- `manifest_upload_roundtrip` — `apr publish --manifest <yaml>` validates, hashes
  the declared artifact locally, and aborts before network I/O on mismatch.
- `extra_file_passthrough` — `apr publish --extra-file <path>` (repeatable) uploads
  sidecars verbatim in CLI-argument order.
- `no_readme_when_manifest` — when `--manifest` is passed, the auto-generated
  `README.md` is suppressed; the manifest is the provenance document.
- `dogfood_shell_script` — `scripts/ship-two-001/ex-04-upload-hf.sh` MUST invoke
  `apr publish`; `uv run`, `huggingface_hub`, `huggingface-cli`, and `pip install`
  are forbidden in the ship script.
- `three_format_preference` — every SHIP-TWO-* release publishes `.apr`,
  `.safetensors`, and `.gguf` side-by-side in the same HF repo.
- `safetensors_dtype_fp16` — ship-bound `.safetensors` MUST be exported via
  `apr export --format safetensors --quantize fp16`. Default-fp32 export doubles
  disk/network cost; the `transformers` / `candle` / HF ecosystem reads fp16
  natively. Expected 7B sizes: `.apr` ≈ 7.5 GB, `.safetensors-fp16` ≈ 14 GB,
  `.gguf` ≈ 7.5 GB (a fp32 safetensors at ≈ 29 GB is forbidden for ships).

Discharged by falsification tests **FALSIFY-PUB-EXTRA-001 through -010**
(contract `apr-cli-publish-extra-v1.yaml` v1.2.0):

- **-001..-004** covered by `apr publish` unit tests
- **-005** dogfood gate (no Python in ship scripts)
- **-006** post-upload sha256 round-trip (discharged by EX-05)
- **-007** three-format HF repo (discharged by EX-05 + list-repo-files)
- **-008** no Python in `ex-05-verify-manifest.sh` (discharged)
- **-009** corrupt-manifest pre-flight abort (shows exit code 5 blocking upload)
- **-010** `preflight_validate_manifest` function present + invoked before any `publish_format`

Additionally, **FALSIFY-PM-007** (safetensors header dtype Poka-Yoke, contract
`publish-manifest-v1.yaml` v1.1.0) fires automatically inside every pre-flight
invocation for `.safetensors` format. Eight unit tests cover both the happy path
and the exact §12.7.2 ship-blocker scenario (`pm007_f32_weight_when_fp16_declared_fails`).

**FALSIFY-PM-008** (GGUF tensor-type Poka-Yoke, contract `publish-manifest-v1.yaml`
v1.2.1, added 2026-04-18) closes the same class for `.gguf` ships. Design pivot
made mid-discharge: the teacher GGUF that had to pass this gate ships with
`general.file_type = 0` (ALL_F32) despite fully Q4_K tensors — a known llama.cpp
quantize-tool bug. PM-008 therefore treats the **predominant non-float GGML
tensor type** from the tensor_metadata section as authoritative and the
metadata_kv ftype as an advisory fallback (used only when tensor metadata is
absent, e.g. for synthetic fixtures). 15 unit tests, including the real-teacher
scenario (`pm008_q4_k_tensors_override_stale_ftype_zero`) and the "wrong file
pointed at" scenario (`pm008_tensor_type_mismatch_fails`).

**FALSIFY-PM-009** (APR magic-bytes Poka-Yoke, contract `publish-manifest-v1.yaml`
v1.3.0, added 2026-04-18) closes the three-format ship symmetry. With PM-007
covering `.safetensors` and PM-008 covering `.gguf`, PM-009 ensures `.apr` ships
can't pass pre-flight with a mis-staged artifact. v1.0 scope = first 4 bytes
match one of `APR\0` / `APRN` / `APR1` / `APR2` (the four APR magic variants
recognised by `crates/aprender-registry/src/format.rs::parse_apr_header`). The
exact ship-blocker this catches is "GGUF file renamed `.apr` and staged under
format=apr manifest" — covered explicitly by
`pm009_gguf_magic_staged_as_apr_fails`. Dogfooded against the real 8 GiB
teacher APR: verdict PASS (`apr magic = APR\0 (v2) (valid)`). Expansion to
tensor-index quant validation is deferred to v1.1 until a real-world FAIL
demonstrates need.

The unit test matrix (`cargo test -p apr-cli validate_manifest`) runs 45 tests on
every push; the end-to-end pre-flight gate runs against real 8–15 GiB artifacts
only at ship time. All three staged teacher artifacts (`.apr` 8.0 GiB,
`.safetensors` 15.2 GiB, `.gguf` 8.0 GiB) discharged every applicable gate on
2026-04-18, with overall verdict **PASS** per format. Evidence:
`evidence/ship-two-001/ex-04-preflight-gate-smoketest-v2.json` (9-gate
coverage; supersedes v1 which only captured PM-001..007).

#### 12.7.1 Revised EX-04 invocation

EX-04 is now **one command per format**, pointed at a per-format manifest in
`contracts/publish-manifests/`:

```
apr publish /mnt/nvme-raid0/models/ship-two-001/ \
    paiml/qwen2.5-coder-7b-apache-q4k-v1 \
    --manifest contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1-apr.yaml \
    --extra-file /mnt/nvme-raid0/models/ship-two-001/tokenizer.json
```

and repeats for `-safetensors.yaml` and `-gguf.yaml`. Each invocation runs the
pre-flight sha256 guard *before* opening any network socket, then uploads the
artifact + tokenizer + manifest.yaml.

#### 12.7.2 What falsifies the dogfood gate

| Condition                                                                                   | Action                                                  |
|---------------------------------------------------------------------------------------------|---------------------------------------------------------|
| `ex-04-upload-hf.sh` contains `uv run` / `huggingface_hub` / `huggingface-cli` / `pip install` | FALSIFY-PUB-EXTRA-005 FAIL — fix script, rerun           |
| `ex-05-verify-manifest.sh` contains `uv run` / `python3` / `pip` / `huggingface_hub`        | FALSIFY-PUB-EXTRA-008 FAIL — ex-05 must use `apr validate-manifest --live` |
| HF repo missing any of `.apr` / `.safetensors` / `.gguf` after EX-04                        | FALSIFY-PUB-EXTRA-007 FAIL — re-upload missing format    |
| Staged `.safetensors` header declares F32 for weight tensors when manifest says fp16        | **FALSIFY-PM-007 FAIL — pre-flight gate aborts with exit 2 BEFORE any network I/O; re-export with `--quantize fp16`** |
| Staged `.gguf`'s predominant GGML tensor type disagrees with manifest quantization (e.g. manifest says `q4_k` but tensors are predominantly `Q6_K`) | **FALSIFY-PM-008 FAIL — pre-flight gate aborts with exit 2 BEFORE any network I/O; correct the manifest or re-quantize.** (Note: stale `general.file_type=0` does NOT trigger FAIL — it is surfaced as a diagnostic note.) |
| Staged `.apr` file's first 4 magic bytes are not one of `APR\0` / `APRN` / `APR1` / `APR2` (e.g. a GGUF file renamed `.apr`, or a stray `.safetensors`) | **FALSIFY-PM-009 FAIL — pre-flight gate aborts with exit 2 BEFORE any network I/O; restage the correct `.apr` artifact.** |
| Staged artifact's local sha256 ≠ per-format manifest sha256 at ship time                    | **FALSIFY-PUB-EXTRA-009 FAIL — pre-flight gate aborts with exit 5 BEFORE any network I/O** |
| `preflight_validate_manifest` removed or reordered after `publish_format`                   | FALSIFY-PUB-EXTRA-010 FAIL — Poka-Yoke bypassed; re-sequence |
| Any uploaded artifact's CDN-served sha256 ≠ per-format manifest sha256                      | FALSIFY-PUB-EXTRA-006 FAIL (post-upload) — investigate transit corruption |

**Ship-time Poka-Yoke:** prior to contract v1.2.0 (2026-04-18), the dtype mismatch
row above required post-hoc detection and a deprecation cycle on HF Hub. With
PM-007 + the pre-flight gate, it is structurally unreachable: an ex-04 invocation
with divergent bytes exits non-zero before the first HTTP connection opens.

---

### 12.8 Large-File Upload via Xet (2026-04-18 amendment — v2.8.0)

**Trigger:** a real EX-04 upload run with live `HF_TOKEN` (commit
`ec60b5c9e`, `--features cuda`) surfaced that every SHIP-TWO-001 teacher
artifact exceeds HF Hub's 5 GiB HTTP preupload threshold:

| Format         | Size     |
|----------------|----------|
| `.apr`         | 8.0 GiB  |
| `.gguf`        | 8.0 GiB  |
| `.safetensors` | 15.2 GiB |

HF Hub's `preupload/main` endpoint returned `200 OK` with `uploadMode:
"lfs"` but **both** `upload_url` and `chunk_urls` empty. Our upload
path (`crates/aprender-core/src/hf_hub/upload.rs:283 —
reject_oversized_file`) hard-aborts in that state. Five Whys evidence
at `evidence/ship-two-001/ex-04-five-whys-lfs-5gb-blocker.md`.

#### 12.8.1 Rejected paths (for the record)

| Option                                    | Why rejected                                                   |
|-------------------------------------------|----------------------------------------------------------------|
| A) `apr export --max-shard-size` sharding | **Workaround**, not a fix. Only helps `.safetensors`; `.apr` and `.gguf` lack native sharding conventions; loses single-file UX. |
| B) LFS batch API only                     | Pulls git-lfs subprocess / reimplements legacy protocol. HF has moved to Xet; LFS batch is legacy/fallback, not the current path. |
| C) Self-hosted S3 bucket                  | **Not sovereign** — still AWS-dependent. Decouples us from HF Hub discovery and breaks AC-SHIP1-006 (`apr pull` from HF). |
| D) Respec to a smaller parent model       | Q4_K of 7 B is already near the practical floor for coder-quality; changing parent is out of scope for SHIP-TWO-001. |
| E) Ship fewer formats                     | Violates `three_format_preference` equation in `apr-cli-publish-extra-v1.yaml`. |

The real fix is the real protocol: **Xet**, HF Hub's current
content-addressable storage backend for large files.

#### 12.8.2 The Xet protocol (normative summary)

Source of truth: [huggingface.co/docs/xet/index v1.0.0](https://huggingface.co/docs/xet/index).
Reference Rust impl: [github.com/huggingface/xet-core](https://github.com/huggingface/xet-core)
(Apache-2.0, v1.4.3 as of 2026-03-31). Crates on crates.io: `hf-xet`,
`xet-client`, `xet-data`, `xet-core-structures`, `xet-runtime`.

**Upload lifecycle** (MUST be performed in order):

1. **Token acquisition** —
   `GET https://huggingface.co/api/models/{repo_id}/xet-write-token/{revision}`
   with `Authorization: Bearer ${HF_TOKEN}`. Response:
   `{ accessToken, exp (unix seconds), casUrl }`. Refresh at
   `exp - 30s`.
2. **Chunking** — content-defined (gearhash) with 8 KiB min /
   ~64 KiB avg / 128 KiB max. Exception: last chunk of a file MAY
   be smaller than min.
3. **Deduplication** (OPTIONAL) —
   `GET ${casUrl}/v1/chunks/default-merkledb/{chunk_hash_hex}`.
4. **Xorb formation** — group chunks into xorbs, each ≤ 64 MiB
   serialized, avg ~1024 chunks. Hash via xet-core
   `xorb_hashing` procedure.
5. **Xorb upload** —
   `POST ${casUrl}/v1/xorbs/default/{xorb_hash_hex}` with
   `Authorization: Bearer ${accessToken}`, body
   `application/octet-stream`. Response: `{ was_inserted: bool }`.
   `was_inserted:false` is SUCCESS (idempotent replay).
6. **Shard assembly** — one shard references one or more xorbs
   plus file reconstructions. Shard ≤ 64 MiB. All referenced xorbs
   MUST already be uploaded (strict happens-before).
7. **Shard upload** — `POST ${casUrl}/v1/shards`. Response
   `{ result: 0|1 }`; both values are SUCCESS.
8. **LFS pointer commit** — `POST https://huggingface.co/api/models/{repo_id}/commit/{revision}`
   with an LFS pointer file (oid sha256 = sha256(file), size =
   bytes). Without this step the bytes are safe in CAS but the
   repo file tree does not show them.

**Hash-string encoding rule (CRITICAL)** — URLs embed 32-byte hashes
as 64 hex chars, but NOT naive hex. For each 8-byte block, reverse
bytes within the block, then concatenate hex. Equivalent to reading
each 8-byte block as a little-endian u64 and printing as 16 hex
chars. Naive hex triggers 400 Bad Request. `MerkleHash::to_string()`
in xet-core does this correctly; direct `hex::encode` is FORBIDDEN.

**Retry taxonomy:**
- RETRYABLE (exp. backoff, Retry-After on 429): 429, 500, 503, 504,
  connection-level errors.
- NON-RETRYABLE (abort immediately): 400, 403, 404, 416.
- 401 = refresh token once, then abort.

#### 12.8.3 Contract and Falsification Set

Contract file: `contracts/apr-publish-hf-large-file-v1.yaml` v1.1.1
(status `IMPLEMENTED` as of 2026-04-18, commit `18fd9536e`; evidence
fields added in v1.1.1 at commit `671535b44`). Ten falsifiable gates:

| Gate                      | What it falsifies                                                              |
|---------------------------|--------------------------------------------------------------------------------|
| FALSIFY-PUB-LFS-001       | File-size dispatch: > 5 GiB routes to Xet, not `reject_oversized_file()`.     |
| FALSIFY-PUB-LFS-002       | Xet token acquisition URL template + header + JSON response parsing.          |
| FALSIFY-PUB-LFS-003       | Chunk size bounds (8 KiB ≤ len ≤ 128 KiB) except last chunk.                 |
| FALSIFY-PUB-LFS-004       | Xorb size ≤ 64 MiB serialized.                                                |
| FALSIFY-PUB-LFS-005       | Strict shard-after-xorbs ordering (all referenced xorbs 2xx before shard).    |
| FALSIFY-PUB-LFS-006       | Content-addressable idempotency (`was_inserted:false` and `result:0` = OK).   |
| FALSIFY-PUB-LFS-007       | Retry policy matches Xet error taxonomy.                                      |
| FALSIFY-PUB-LFS-008       | Hash-in-URL uses 8-byte-reversed hex, not naive hex.                          |
| FALSIFY-PUB-LFS-009       | LFS pointer git commit uses one-pass sha256 + size from the Xet upload.       |
| FALSIFY-PUB-LFS-010       | Three-format real dogfood (8-15 GiB each) round-trips via `apr publish` only. |

#### 12.8.4 Implementation (shipped 2026-04-18, commit `18fd9536e`)

Actual wiring diverged from the v1.0.0 plan in two ways: (i) `hf-xet`
1.5.1 exposes a *blocking* API (`build_blocking`,
`upload_from_path_blocking`, `commit_blocking`), which obviates the
planned tokio↔sync bridge (step 3 below, deleted); (ii) phases 3–7
of the Xet protocol are fully internal to `hf-xet`, so the four-file
`xet/` module tree anticipated in v1.0.0 collapses to a single
178-line `xet.rs`. See
`contracts/apr-publish-hf-large-file-v1.yaml` v1.1.0 changelog for
the v1.0.0→v1.1.0 delta.

1. **Dependency surface** — ADDED `hf-xet = "1.5.1"` (Apache-2.0) to
   `[workspace.dependencies]` plus
   `hf-xet = { workspace = true, optional = true }` in
   `crates/aprender-core/Cargo.toml`. NEW `xet` sub-feature:
   `xet = ["hf-hub-integration", "hf-xet"]`. `apr-cli` forwards it
   via `xet = ["hf-hub", "aprender/xet"]`. Default `cargo install
   aprender` footprint unchanged (xet off by default; adds ~4 MB
   when enabled).
2. **Dispatch site** — DELETED
   `crates/aprender-core/src/hf_hub/upload.rs::reject_oversized_file`.
   ADDED `upload_via_xet` (tempfile materialize + `XetUploader`
   invoke) and `reject_needs_xet_feature` (clear error when built
   without `--features xet`). Dispatch gate in `upload_via_lfs`
   routes files > 5 GiB through `super::super::xet::should_use_xet`.
   The < 5 GiB HTTP-LFS path is untouched.
3. **Sync call surface** — `hf-xet` provides `*_blocking` variants,
   so we call them directly from the sync CLI path. No tokio
   runtime spawned in `apr publish`.
4. **Error surface** — ADDED `HfHubError::XetUpload(String)` and
   `HfHubError::PartialUpload { cas_success: bool,
   commit_success: bool, detail: String }`. Partial-upload splits
   "CAS xorbs landed but LFS pointer commit failed" from "nothing
   happened" — consumed by retry UX.
5. **Dogfood** — live upload still pending `HF_TOKEN`. Gate evidence
   paths for the live upload remain:
   `evidence/ship-two-001/ex-04-xet-upload.log` +
   `evidence/ship-two-001/ex-04-xet-verify.json`. Pre-live evidence
   already captured in two files:
   (a) Static wiring proof at
   `evidence/ship-two-001/ex-04-xet-phase2-wiring.json` (commit
   `ee6382803`) — `strings(apr)` confirms the full `hf-xet` 1.5.1
   runtime is linked into the canonical binary.
   (b) Live-on-teacher dry-run at
   `evidence/ship-two-001/ex-04-xet-dryrun-teacher.{json,txt}`
   (commit `18f8b5604`) — all three real SHIP-TWO-001 teacher
   artifacts (.apr 8.0 GiB / .gguf 8.0 GiB / .safetensors 15.2 GiB)
   route to the Xet CAS path under the canonical
   `/mnt/nvme-raid0/targets/aprender/release/apr` (features
   `cuda,xet`). This discharges FALSIFY-PUB-LFS-001 against real
   teacher sizes, not synthetic fixtures.

Actual edit sites (see `contracts/apr-publish-hf-large-file-v1.yaml`
`implementation_plan.edit_sites` for the authoritative list):

```
Cargo.toml                                      (+ hf-xet = "1.5.1")
crates/aprender-core/
├── Cargo.toml                                  (+ optional hf-xet dep, + xet feature)
└── src/hf_hub/
    ├── mod.rs                                  (+ pub mod xet; + XetUpload / PartialUpload variants)
    ├── upload.rs                               (- reject_oversized_file
    │                                            + upload_via_xet
    │                                            + reject_needs_xet_feature
    │                                            ~ upload_via_lfs dispatch)
    └── xet.rs                                  (NEW, 178 lines)
crates/apr-cli/
└── Cargo.toml                                  (+ xet feature forwarder; + xet in `full`)
```

Known Phase 3 follow-up (non-blocking): `push_to_hub` still takes
`&[u8]`, so `upload_via_xet` materializes bytes to a tempfile
before invoking `upload_from_path_blocking`. Threading `&Path`
through the upload stack eliminates the round-trip; tracked for a
follow-up contract amendment.

#### 12.8.5 Sovereignty position

The Sovereign AI Stack ships models **through** HF Hub (discovery
convenience) without **depending on** HF Hub (bytes are also
mirrored via `artifact_url_mirror` in every manifest, per
`publish-manifest-v1.yaml` §4.3). Xet-based upload does not
compromise sovereignty: we publish to the Hub via the Hub's own
public protocol, and the manifest links to an independent mirror
whose bytes match by sha256. Loss of HF Hub availability degrades
discovery, not operation.

#### 12.8.6 What falsifies the v2.8 amendment (v2.8.0 + v2.8.1)

| Event                                                                                   | Falsification verdict                                                        |
|-----------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| EX-04 succeeds via any path **other than** `apr publish`'s Xet code (e.g., `hf upload`) | §12.8 failed: we took a workaround, not the contract-mandated path.          |
| Any one of the 3 real 8-15 GiB artifacts does not round-trip by sha256                  | FALSIFY-PUB-LFS-010 FAIL — ship blocked; investigate CAS corruption or LFS pointer drift. |
| `reject_oversized_file` remains reachable in production code                            | FALSIFY-PUB-LFS-001 FAIL — code delete incomplete. (Already verified deleted at `18fd9536e`.) |
| Default `cargo install aprender` binary size regresses > 20 %                           | Feature gating broken; re-architect to push xet into a separate crate. (xet is off by default — `cargo install aprender` does NOT pull `hf-xet`.) |
| `cargo test -p aprender-core --features xet --lib hf_hub` fails on any of the 4 PUB-LFS-001/002 unit tests | Regression in dispatch-gate or token-URL builder. Phase 2 static proof void. |

Failure here is recoverable and distinct from §12.5/§12.7 failures:
a bug in the Xet path can be fixed by shipping an aprender patch
release without redoing training or re-evaluating the teacher.

---

## 13. Why Did This Take So Long? (Retrospective)

### 13.1 Timeline

- **2026-01-01 → 2026-04-17:** 3.5 months of work on the Sovereign AI Stack.
- 2141 commits to aprender (of which 181 = 8.5% are perf-path-to-1.5× commits).
- apr-leaderboard ran **12 distillation recipes** (a → l); multiple had broken checkpoint output.
- Commit `0fc5436 fix: LoRA merge was element-wise, not matrix multiply — root cause of distilled
  model garbage` on apr-leaderboard — **this exact failure class has been fixed before**.
- Commit `a20f234 docs: document Q4K roundtrip corruption blocker` — **also previously known**.
- Current distilled-v2 checkpoints are dated 2026-04-03; they sat **14 days** before any
  `apr qa`-driven audit.
- Spec SPEC-SHIP-TWO-001 v1.0.0 was written 2026-04-17, *after* the broken checkpoint had been
  sitting for 2 weeks and cited as "trained, needs packaging."

### 13.2 Root causes

1. **Contract came after POC, not before.** The 87.20% number lived in a recipe comment since
   Q1 and was promoted to a spec headline without a falsification test run. Design Principle 1
   (Contract-first) was violated by its own spec.
2. **`apr qa` gate matrix lets Golden Output fail silently.** A model that cannot generate "4"
   for "2+2=" passed `apr qa` overall because Golden Output was reported but non-blocking. Tensor
   Contract PASS became a false confidence signal. This is the structural defect that allowed
   broken weights to persist for 14 days.
3. **Perf work crowded out ship work.** 181 perf commits toward 1.5× Ollama parity between
   January and April; **zero** commits on publishing a *model* artifact (as opposed to a *crate*).
   Perf gains are visible in benchmarks; ship state was not gated anywhere.
4. **Monorepo reorg (APR-MONO) consumed weeks.** Phases 1–11 moved 70 crates, introduced shim
   layers, debugged publishing cycles — necessary ceremony but directly competed with model-ship
   bandwidth. Commits like "Phase 10d+10e done" / "Phase 11 (CI fix + publish babysit)" show
   sustained multi-week focus.
5. **Distillation recipe churn without shipping discipline.** Recipes a/b/c/d/e/f/g/h/i/j/k/l —
   twelve experiments, each generating a checkpoint — but no contract defining what makes a
   recipe's output "ship-quality." Each recipe was treated as a new chance; no recipe was ever
   contractually retired. `contracts/distillation-pipeline-v1.yaml` (per v1.0.0 §4.4) is listed as
   "NEW" but has never existed — the proof of this is that broken checkpoints sat unaudited for weeks.
6. **Two-model scope from the start.** v1.0.0 bundled distilled (quick) with sovereign
   (multi-week). This guaranteed the multi-week item would block the quick item from any
   expedited ship path. The fix is to ship MODEL-1 alone as a v1 and move MODEL-2 to v2.
7. **Tooling investment vs tooling usage.** `apr qa`, `apr trace`, `apr profile`, `apr diff` all
   exist and are well-built. They were not being *dogfooded on the shipping artifact* until
   2026-04-17. The audit that exposed this was a 10-minute `apr qa` invocation.
8. **"87.20%" was never in a results JSON.** 17 HumanEval result files exist in
   `apr-leaderboard/results/`; all 17 are teacher runs. The student has no recorded result. A
   spec claim not traceable to a results file is an unfalsified claim.

### 13.3 Lessons codified as contracts

| Contract (new)                                  | Prevents future occurrence of                                |
|-------------------------------------------------|--------------------------------------------------------------|
| `contracts/eval-harness-humaneval-v1.yaml` v1.1 | Headline pass@1 numbers without a results-JSON trail         |
| `contracts/publish-manifest-v1.yaml` v1.0       | Artifacts shipping without sha256 / license / provenance     |
| `contracts/publish-manifest-v1.yaml` v1.1 (PM-007) | Uploading `.safetensors` whose header dtype contradicts the manifest |
| `contracts/publish-manifest-v1.yaml` v1.2 (PM-008) | Trusting stale `general.file_type` over per-tensor `ggml_type` histogram for GGUF |
| `contracts/publish-manifest-v1.yaml` v1.3 (PM-009) | Uploading a renamed `.gguf`/`.safetensors` under a `.apr` manifest |
| **APR-QA GATE AMENDMENT (§12.1)**               | Tensor Contract masking a Golden Output failure              |
| `contracts/distillation-pipeline-v1.yaml` (TBD) | Recipes run without per-epoch Golden Output gating           |

### 13.4 Publish-fastest rules going forward

1. **No claim without a JSON.** Every pass@1 number in any spec must be a `jq`-extractable field
   in a file under version control. If it isn't, it doesn't exist.
2. **Golden Output is a ship-blocking gate.** `apr qa` must exit non-zero if Golden Output fails,
   even when all structural gates pass.
3. **Ship-first, proof-second.** Ship the teacher in 10 hours; use the published artifact as the
   reference against which distillation is measured. Do not wait to ship because distillation
   isn't done.
4. **One artifact per release.** v1 = teacher. v1.1 = distilled (if/when it works). v2 =
   sovereign. Coupling them couples their risks.
5. **Dogfood before declaring "trained."** A checkpoint is not "trained" until `apr qa` and
   `apr eval` agree it is. Until then it is "saved."

---

## 11. References

### 11.1 Existing Contracts

- `contracts/model-families/qwen2.yaml` — Qwen2 architecture descriptor
- `contracts/model-families/llama.yaml` — LLaMA architecture descriptor
- `contracts/model-families/_schema.yaml` — family schema validator
- `contracts/tensor-layout-v1.yaml` — row-major APR invariant (LAYOUT-001/002)
- `contracts/chat-templates-v1.yaml` — chat-template engine spec
- `contracts/apr-cli-commands-v1.yaml` — 57-command CLI contract
- `contracts/publish-manifest-v1.yaml` v1.3.0 — artifact-shipping schema (sha256, provenance, license) + **FALSIFY-PM-007** safetensors header dtype Poka-Yoke + **FALSIFY-PM-008** GGUF tensor-type Poka-Yoke (tensor-authoritative; `general.file_type` is advisory fallback) + **FALSIFY-PM-009** APR magic-bytes Poka-Yoke (three-format ship symmetry)
- `contracts/apr-cli-publish-extra-v1.yaml` v1.2.0 — **F-PUBLISH-EXTRA-001** (§12.7): manifest consumption, `--extra-file` passthrough, three-format ship, safetensors fp16 dtype, **preflight_validate_manifest** (FALSIFY-PUB-EXTRA-009/-010)
- `contracts/eval-harness-humaneval-v1.yaml` — pass@1 harness / AC-EX-003 floor
- `contracts/apr-model-qa-v1.yaml` — `apr qa` gate matrix / AC-EX-001/-002 (Golden Output hard-block)
- `contracts/training-loop-pretrain-v1.yaml` v1.4.0 — MODEL-2 training loop (GATE-TRAIN-001..010), peer of the new GPU backend contract below
- `contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 PROPOSED — **§14 (v2.23.0)** task #132 GPU training backend dispatch (INV-GPUTRAIN-001..007, GATE-GPUTRAIN-001..006, FALSIFY-GPUTRAIN-001..007)

### 11.2 Related Specifications

- `docs/specifications/aprender-train/hugging-face-distill-learn-pipeline-spec.md`
- `docs/specifications/aprender-train/comprehensive-qa-falsification.md`
- `docs/specifications/aprender-train/model-eval-framework-spec.md`
- `docs/specifications/aprender-monorepo-consolidation.md`

### 11.3 External

- HumanEval: Chen et al. 2021, *Evaluating Large Language Models Trained on Code*
- Qwen2.5-Coder: Hui et al. 2024, *Qwen2.5-Coder Technical Report*
- The Stack v2: BigCode, CC-BY-4.0

---

## 14. Task #132 — CUDA training backend gap (v2.23.0 amendment, 2026-04-21)

### 14.1 Surface (what broke)

First MODEL-2 from-scratch real-compute dispatch on lambda-labs RTX 4090
at commit `f7ad11408` (post-task-#131 vocab alignment):

- `apr pretrain --mode from-scratch --dataset … --tokenizer …`
- 14 minutes observed runtime
- 114% CPU (single-thread), 0 MiB GPU memory per `nvidia-smi`
- Empty run dir; no step logging; no checkpoints
- Killed after observing no GPU activity

The dispatch accepted flags, printed startup banner, and silently ran on
CPU. No error surfaced because there was no contract binding "operator
asked for GPU" to "training ran on GPU."

### 14.2 Root cause

`crates/aprender-train/src/train/transformer_trainer/trainer.rs:42`:

```rust
impl TransformerTrainer {
    pub fn new(config: TransformerTrainConfig) -> Self {
        let seed_guard = crate::transformer::init::lock_init_seed(config.seed);
        let model = Transformer::new(&config.model_config);
        drop(seed_guard);
        Self::build(model, config)
    }
}
```

`TransformerTrainer::new` takes no `Device`. Everything downstream —
`Transformer`, `AdamW`, autograd tape, `GradScaler` — uses CPU-backed
`aprender::Tensor` (trueno SIMD). The `--features cuda` flag gates
`realizar` inference kernels, **not** `aprender-train` training.

Why this was not caught before task #126:

1. `apr pretrain --synthetic` passes — the synthetic drive path never
   instantiates the real model, so GPU residency was never exercised.
2. Unit tests of the training path explicitly avoid the 370M scale
   (allocating ~5 GB of parameters is too expensive per test). CPU is
   tractable at toy scale, which masks the CPU-only dispatch.
3. Task #119's "real-compute smoke test PASS" on lambda-labs used the
   synthetic drive (or a toy config), not a 370M cold start.

Scale math: 370M × CPU forward+backward ≈ 30–60 s/step → 10 k steps ≈
100 + hours. Impractical. This is what task #126 actually dispatched,
which is why the run sat at 114% CPU with no log output.

### 14.3 Plan agent finding — existing GPU infrastructure

Phase 0 input (Plan agent survey, 2026-04-21):

| Artifact                                                             | Status        | LOC   |
|----------------------------------------------------------------------|---------------|-------|
| `crates/aprender-train/src/train/transformer_trainer/cuda_trainer.rs` | EXISTS        | 3,432 |
| `CudaTransformerTrainer` AdamW + fused CE + gradient clip + pre-warmed kernels | EXISTS | — |
| YAML training-config loader `loader/mod.rs:227`                      | EXISTS — HAS `if use_cuda { CudaTransformerTrainer::… → train_loop_cuda } else { CPU fallback }` | — |
| `apr pretrain` CLI `drive_real` path (`pretrain.rs:230`)              | MISSING — unconditionally calls `TransformerTrainer::new` (CPU) | — |

**The gap is wiring, not kernels.** The YAML-config path dispatches
correctly; the CLI-flag path does not. Task #132 converges them.

### 14.4 Contract (Phase 0 deliverable)

`contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 PROPOSED,
kind: `training-loop`, peer of `training-loop-pretrain-v1.yaml`.

**Invariants:**

| ID                | Rule                                                                       |
|-------------------|----------------------------------------------------------------------------|
| INV-GPUTRAIN-001  | `--device` grammar: `^(cpu\|cuda(:[0-9]\|:1[0-5])?\|auto)$`, reject others |
| INV-GPUTRAIN-002  | No silent CPU fallback when CUDA was explicitly requested                   |
| INV-GPUTRAIN-003  | GPU residency proof: `nvidia-smi` shows `pid == training_pid AND used_memory > 0` within 5 s of step 0 |
| INV-GPUTRAIN-004  | CPU fallback path remains fully functional (peer GATE-TRAIN-001..010 still PASS) |
| INV-GPUTRAIN-005  | 370M step time < 500 ms on RTX 4090 (seq_len=2048, batch=1, sm_89 pre-compiled) |
| INV-GPUTRAIN-006  | Same-device seed reproducibility holds (two `cuda:0` runs at seed=0, `\|Δloss[k]\| ≤ 1e-5`) |
| INV-GPUTRAIN-007  | `apr --version --json` reports `{cuda_feature, cuda_runtime_available, visible_devices[]}` |

**Ship-blocking gates:** GATE-GPUTRAIN-002 (no-silent-fallback) and
GATE-GPUTRAIN-003 (residency proof). Both must land before task #126
re-dispatches.

### 14.5 Implementation plan (5 phases)

| Phase | Deliverables                                                                                                   | Estimate |
|-------|----------------------------------------------------------------------------------------------------------------|----------|
| 0     | `contracts/entrenar/gpu-training-backend-v1.yaml` + this §14 amendment (PROPOSED status)                       | THIS PR  |
| 1     | `Device` enum + `resolve_device()` in `crates/aprender-train/src/train/device.rs` + `--device` CLI flag + SharedTrainer enum extended with `CudaVariant` (NotImplemented stub) + FALSIFY-GPUTRAIN-001/002 | 1 day    |
| 2 (algorithm-level, task #121 v2.41.0) | FALSIFY-GPUTRAIN-003..007 all bound at `PARTIAL_ALGORITHM_LEVEL` in `crates/aprender-train/src/train/gputrain_0{03..07}.rs`; 5 new verdict fns + 2 parsers/field types; 5 × 6–8 section mutation surveys all green; contract v1.0.0 → v1.1.0 records the algorithm-level discharges (status stays PROPOSED) | DONE     |
| 2 (live-wire, **DONE** 2026-04-24) | `crates/apr-cli/src/commands/pretrain.rs::drive_real` takes `device: Device` and dispatches `drive_real_cuda` (`#[cfg(feature = "cuda")]`, line 336) which builds a `CudaTransformerTrainer` via `entrenar::train::pretrain_real_cuda::build_shared_cuda_trainer` + `CudaRealStepFn`/`CudaRealValFn`/`CudaAprCheckpointFn`. `#[cfg(not(feature = "cuda"))]` companion (line 373) returns GATE-GPUTRAIN-002 error. nvidia-smi querying: `config/train/loader/mod.rs:445` + `gpu/ledger.rs:404`. Code-check discovered this already-landed state 2026-04-24 during v2.45.0 authoring — see "Spec-drift five-whys" at top of this document. | shipped |
| 3     | Lambda-labs re-dispatch: `apr pretrain --mode from-scratch --device cuda:0 --num-steps 50 --json` produces `evidence/task-132/rtx4090-370m-step-budget.json` with median step-wall < 500 ms; GATE-GPUTRAIN-001..006 all `verdict: pass` | 2 days   |
| 4     | Promote `gpu-training-backend-v1.yaml` PROPOSED → ACTIVE; spec v2.23.0 → v2.23.1 records promotion; MEMORY.md pointer for task #132 flipped to CLOSED | 0.5 day  |

Total estimate: **~6 days** (Plan agent), down from initial multi-week
scope because `CudaTransformerTrainer` already exists.

### 14.6 Critical path DAG

Task #131 (vocab bump) CLOSED at `f7ad11408`. Previous DAG claimed
task #126 was ready; the lambda-labs dispatch falsified that claim.
Updated DAG:

```
#118 BPE train 50_257  ──► #131 vocab align  ──► ( #126 blocked by #132 )
                                                        │
                                                        ▼
                                            #132 Phase 0 (this PR — contract + spec)
                                                        │
                                                        ▼
                                            #132 Phase 1 (device enum + CLI flag)
                                                        │
                                                        ▼
                                            #132 Phase 2 (wire existing CudaTransformerTrainer)
                                                        │
                                                        ▼
                                            #132 Phase 3 (RTX 4090 evidence)
                                                        │
                                                        ▼
                                                    #126 re-dispatches
                                                        │
                                                        ▼
                                                AC-SHIP2-003 (target_val_loss ≤ 3.0)
```

### 14.7 Risks + mitigations

| Risk                                                                 | Mitigation                                                                                   |
|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| CudaTransformerTrainer API drift since last exercise                 | Phase 1 adds FALSIFY-GPUTRAIN-006 same-device seed-reproducibility test — exercises full forward/backward/AdamW cycle before Phase 2 wires drive_real |
| `--features cuda` footgun (memory/feedback_cuda_feature_footgun.md)  | INV-GPUTRAIN-007 + GATE-GPUTRAIN-006 — `apr --version --json` must distinguish build-time feature from runtime availability |
| Seed plumbing broken across device-dispatch layer                    | INV-GPUTRAIN-006 explicit counter-test; `lock_init_seed` mutex stays in place                |
| Test cost for 370M × CUDA in unit tests                              | Keep INV-GPUTRAIN-005 as an evidence-file gate (JSONL from lambda-labs), not a unit test     |
| CPU path regression during refactor                                  | INV-GPUTRAIN-004 + GATE-GPUTRAIN-005 — peer-contract GATE-TRAIN-001..010 must still PASS on `--device cpu` |

### 14.8 Toyota Way — Five Whys

1. **Why** did task #126 burn 14 minutes of compute? — The run was CPU-only.
2. **Why** was the run CPU-only when the operator wanted GPU? — The CLI
   path never selected CUDA.
3. **Why** didn't the CLI select CUDA? — `TransformerTrainer::new` takes
   no `Device` and `drive_real` unconditionally constructs it.
4. **Why** was a CPU-only constructor accepted for a training CLI that
   advertises `--features cuda`? — No contract bound "requested device"
   to "actual device" at ship time.
5. **Why** was there no such contract? — The YAML-config loader has
   correct dispatch; no one noticed the CLI-flag path diverged. This
   contract (§14.4) closes that loop so the two paths converge on the
   same invariants.

**Lesson codified:** `contracts/entrenar/gpu-training-backend-v1.yaml`
GATE-GPUTRAIN-002 (ship-blocking: no silent CPU fallback when CUDA
requested) — prevents future occurrence.

---

**END OF SPECIFICATION**
