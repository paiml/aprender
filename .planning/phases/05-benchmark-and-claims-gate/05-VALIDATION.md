---
phase: 5
slug: benchmark-and-claims-gate
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-16
updated: 2026-09-13
---

# Phase 5 — Validation Strategy

> **RECONCILED 2026-09-08 against `05-CONTEXT.md` D-19 — the 9B LoRA arm is DESCOPED.**
> Every claim below about the 40 LoRA cells, the lambda-vector GPU host, or a
> SetFit-versus-LoRA comparison is **superseded**: the host is unreachable and aprender
> cannot run the Qwen3.5-9B hybrid architecture. EVAL-02/EVAL-04 and the Phase 5 goal were
> amended accordingly; the arm is deferred as `D-ITEM-05-15`. Findings about the SetFit half,
> the numerics substrate, the row schema and the claims gate are UNAFFECTED and still hold.
> This file was reconciled in place rather than regenerated — the descope narrows scope, it
> does not invalidate the surviving research.


> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (libtest) + pv contract validation + bashrs + spawned-binary tests |
| **Config file** | Cargo.toml (workspace), Makefile tier targets |
| **Quick run command** | `CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit setfit::` |
| **Full suite command** | `make setfit-all-tests && make setfit-bench-tests && make contract-audit-phase5 && target/release/pv validate contracts/setfit-benchmark-claims-v1.yaml` |
| **Estimated runtime** | ~120 s (quick), ~5 min (full; excludes the deliberate compute runs in 05-01/05-11/05-12) |

---

## Sampling Rate

- **After every task commit:** Run the task's `<automated>` verify (each < 60 s except the flagged calibration/ladder/cell runs)
- **After every plan wave:** Run the full suite command
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 300 seconds (the calibration matrix, the 05-07.T1 spawned production ladder — its automated verify includes a real s8 training rung — and the 40-cell runs are contracted exceptions with their own compute gates)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01.T1 | 05-01 | 1 | EVAL-02 | T-05-01-04 | probe before matrix; frozen E/B recorded | ignored harness (probe mode) | `APRENDER_CALIBRATION_PROBE=1 cargo test -p aprender-train --lib --features setfit production_calibration -- --ignored --nocapture` | ❌ Wave 0 | ⚪ compute-gated (not re-run) |
| 05-01.T2 | 05-01 | 1 | EVAL-02 | T-05-01-02 | ctrl_max < real_min per class per cell | ignored harness (full matrix) | same, probe var unset | ❌ Wave 0 | ⚪ compute-gated (not re-run) |
| 05-01.T3 | 05-01 | 1 | EVAL-02 | T-05-01-01 | entry byte-copied from measured id | doc assertion (grep) | `grep -c 'seeds=13,17,...' 05-01-calibration-measurements.md` | ❌ Wave 0 | ✅ green |
| 05-02.T1 | 05-02 | 1 | EVAL-02 | T-05-02-03 | one source for regimes + tables | unit | `cargo test -p aprender-train --lib --features setfit thresholds` | ◐ edits existing | ✅ green (30 passed) |
| 05-02.T2 | 05-02 | 1 | EVAL-02 | T-05-02-01 | single table_for lookup; induced-mutation control | unit + mutation control | `cargo test -p aprender-train --lib --features setfit setfit::` | ◐ edits existing | ✅ green (434 passed) |
| 05-03.T1 | 05-03 | 2 | EVAL-02 | T-05-03-01/02/03 | additive edit; envelope covered; NO commit | unit + pv | `cargo test ... thresholds` + `pv validate` + `pv diff` | ◐ edits existing | ✅ green (pv validate 0/0) |
| 05-03.T2 | 05-03 | 2 | EVAL-02 | T-05-03-04 | human approval before commit (D-04) | checkpoint (human) | — (human-check) | — | ✅ done (human checkpoint) |
| 05-03.T3 | 05-03 | 2 | EVAL-02 | T-05-03-01 | one commit; both gate directions | unit + audit | `cargo test ... setfit::` + `make contract-audit-phase4` | ◐ edits existing | ✅ green (phase4 audit rc=0) |
| 05-04.T1 | 05-04 | 1 | EVAL-01/04 | T-05-04-01 | manifested fixtures; A1/A3/A4 resolved | generator self-verify | `uv run python gen_claims_fixtures.py` + `shasum -c` | ❌ Wave 0 | ⚪ generator not re-run |
| 05-04.T2 | 05-04 | 1 | EVAL-01 | T-05-04-01 | fixture-parity ECE/Brier, contract-bound | unit (tdd) | `cargo test -p aprender-core --lib calibration` | ◐ edits existing | ✅ green (287 passed) |
| 05-04.T3 | 05-04 | 1 | EVAL-04 | T-05-04-02/03 | frozen t verified vs scipy; no RNG | unit (tdd) | `cargo test -p aprender-core --lib stats::` | ◐ edits existing | ✅ green (287 passed) |
| 05-05.T1 | 05-05 | 2 | EVAL-03/04 | T-05-05-02 | contract pv-valid + gate can fail | pv + induced red | `pv validate contracts/setfit-benchmark-claims-v1.yaml` | ❌ Wave 0 | ✅ green (pv validate 0/0) |
| 05-05.T2 | 05-05 | 2 | EVAL-03 | T-05-05-01/03/04/05 | 5 distinct refusals; 80-cell parity | unit (tdd) | `cargo test -p aprender-train --lib --features setfit bench_row` | ❌ Wave 0 | ✅ green (46 passed) |
| 05-06.T1 | 05-06 | 2 | EVAL-02 | T-05-06-01 | manifest->Selection door; hardcodes controllable | check + unit | `cargo check -p apr-cli --features setfit` | ◐ edits existing | ✅ green (builds) |
| 05-06.T2 | 05-06 | 2 | EVAL-02 | T-05-06-02 | A6 probed; epochs_completed exposed; refusals | unit (tdd) | `cargo test -p aprender-train --lib classify_trainer` + `cargo test -p apr-cli --lib --features setfit finetune` | ◐ edits existing | ✅ green (330 passed) |
| 05-07.T1 | 05-07 | 3 | EVAL-02 | T-05-07-01/03 | spawned production ladder; one sha across surfaces | spawned integration | `cargo test -p apr-cli --test setfit_cli_lifecycle --features setfit` | ◐ extends existing | ⚠️ 1 passed / 5 `#[ignore]`d — see Standing Reds |
| 05-07.T2 | 05-07 | 3 | EVAL-02 | T-05-07-02/04 | prose honest; zero assertion changes; floors re-run | suites + grep | both scoped lib suites | ◐ edits existing | ✅ green (434 passed) |
| 05-08.T1 | 05-08 | 3 | EVAL-01 | T-05-08-01 | credential-gated door; scalar consistency | unit (tdd) | `cargo test -p aprender-train --lib --features setfit apr_evaluate` | ❌ Wave 0 | ✅ green (22 passed) |
| 05-08.T2 | 05-08 | 3 | EVAL-01 | T-05-08-02/03/04 | ordered labels; validation-only calibration | unit (tdd) | `cargo test -p aprender-train --lib --features setfit bench_metrics` | ❌ Wave 0 | ✅ green (46 passed) |
| 05-09.T1 | 05-09 | 4 | EVAL-05 | T-05-09-04 | contracted resource protocol helpers | unit | `cargo test -p apr-cli --lib --features setfit setfit_bench` | ❌ Wave 0 | ✅ green (67 passed) |
| 05-09.T2 | 05-09 | 4 | EVAL-03 | T-05-09-02/03/05 | library doors only; --record verified | unit | same filter | ❌ Wave 0 | ✅ green (67 passed) |
| 05-09.T3 | 05-09 | 4 | EVAL-03/05 | T-05-09-01 | executed identity; lint-clean driver | bashrs + check | `bashrs lint scripts/run_bench_cells.sh` + `cargo check -p apr-cli --features setfit` | ❌ Wave 0 | ✅ green (bashrs 0 errors) |
| 05-10.T1 | 05-10 | 5 | EVAL-04 | T-05-10-01/02/03/04 | 5 negatives refused in every cargo test; deterministic aggregate | unit (tdd) | `cargo test -p aprender-train --lib --features setfit bench_gate` | ❌ Wave 0 | ✅ green (55 passed) |
| 05-10.T2 | 05-10 | 5 | EVAL-04 | T-05-10-05 | estimation-first rendering; refusal = no tables | unit | `cargo test -p apr-cli --lib --features setfit setfit_bench` | ◐ extends 05-09 | ✅ green (67 passed) |
| 05-10.T3 | 05-10 | 5 | EVAL-04 | T-05-10-01 | floors measured; two induced reds reverted | make gates | `make setfit-bench-tests` + `make contract-audit-phase5` | ❌ Wave 0 | ✅ green (**was red — GAP-1**) |
| 05-11.T1 | 05-11 | 6 | EVAL-02/04 | T-05-11-01/04 | contract narrowing approved at a blocking checkpoint; nothing committed before | checkpoint (human) | — (human-check; pv validate + pv diff automated first) | — | ✅ done (human checkpoint) |
| 05-11.T2 | 05-11 | 6 | EVAL-02/04 | T-05-11-02/07 | 40-cell active scope closed-form; six negatives preserved, four re-mutated at the new scope; out-of-scope row refused | unit (tdd) + pv | `cargo test -p aprender-train --lib --features setfit bench_gate` + `cargo test -p aprender-core --lib stats::` | ◐ edits existing | ✅ green (55 + 287) |
| 05-11.T3 | 05-11 | 6 | EVAL-04/05 | T-05-11-03/05/06 | no comparison section or literal; peak mechanisms separate; floors raised and provably able to fail | unit + make gates | `cargo test -p apr-cli --lib --features setfit setfit_bench` + `make setfit-bench-tests` | ◐ extends 05-09/05-10 | ✅ green (67 + rc=0) |
| 05-12.T1 | 05-12 | 7 | EVAL-02/03/05 | T-05-12-01 | compute NOT pre-authorized (05-03 deferred to wave 7); projection recorded as a number first | checkpoint (human) | — (human-check; projection automated) | — | ✅ done (human checkpoint) |
| 05-12.T2 | 05-12 | 7 | EVAL-03 | T-05-12-05/07 | pilot proved BOTH ways: per-row door exits 0, whole-set report refuses at a missing cell | spawned door + report | pilot row exists; door status 0; report refusal status non-zero | ❌ Wave 0 | ✅ green (pilot in evidence) |
| 05-12.T3 | 05-12 | 7 | EVAL-02/03/05 | T-05-12-02/03/04/06 | 40 sequential cells; 40/40 closure; 40-row mechanism tally; text-free | driver run + json count | `ls rows/setfit-*.json \| wc -l` == 40; manifest complete-count == 40 | ❌ Wave 0 | ✅ green (40 rows / 40 sel) |
| 05-13.T1 | 05-13 | 8 | EVAL-01/04/05 | T-05-13-01/02/04/05 | gate green on the real 40-cell set; bit-identical re-run; omission and no-comparison controls on copies | spawned report + digest compare | report.json/md exist; no comparative literal in either | ❌ Wave 0 | ✅ green (report.md/json present) |
| 05-13.T2 | 05-13 | 8 | (D-11) | — | refusal names remedy; exit code unchanged | unit | `cargo test -p apr-cli --lib qa` | ◐ edits existing | ✅ green (266 passed) |
| 05-13.T3 | 05-13 | 8 | EVAL-01..05 | T-05-13-03/06 | closing audits; no checkbox flips; amended ids marked met-narrowly | make + pv | `make setfit-bench-tests` + `make contract-audit-phase5` + `make contract-audit-phase4` | — | ✅ green (**was red — GAP-1**) |
| 05-14.T1 | 05-14 | 9 | EVAL-02 (D-18) | T-05-14-01/02/04 | fail-closed epsilon verdict falsified BOTH ways in the DEFAULT suite — no `#[ignore]`, no env gate; gated set read from `table_for`, never a local allowlist; coverage is a typed `BasisCoverage`, not banner prose | unit (tdd) + count floor | `cargo nextest run -p aprender-train --lib --features setfit -E 'test(/epsilon_basis/)'` (floor 8) | ✅ `evidence.rs` | ✅ green (9 passed) |
| 05-14.T2 | 05-14 | 9 | EVAL-02 (D-18) | T-05-14-03/05 | durable half re-runnable forever; the combine's expected status DERIVED from the derivation's own `REGIME TABLE FOR THE DERIVED REGIME:` line, never a contract-text proxy; 12 persisted digests re-verified, pair count asserted FIRST | unit + ignored combine | `cargo nextest run … -E 'test(/epsilon_basis/)'` + `APRENDER_CALIBRATION_COMBINE=… cargo test --release … production_calibration -- --ignored` | ✅ `evidence.rs`, `calibration-store/` | ✅ durable half green (9 passed); ⚪ `--ignored` combine compute-gated |
| 05-15.T1 | 05-15 | 10 | EVAL-04 | T-05-15-01..07 | the gate's input surface enumerated before the door is hardened; no debt markers standing in for analysis | doc assertion | `test -s …/05-15-gate-input-surface.md` + debt-marker count == 0 | ✅ `05-15-gate-input-surface.md` | ✅ green (debt markers 0) |
| 05-15.T2 | 05-15 | 10 | EVAL-04 | T-05-15-01/02/03/05 | row-supplied `lock_record_path` / `candidate_ledger_path` refused before any syscall, then canonicalized and compared component-wise; both arms through ONE `read_evidence` helper; 16 MiB two-stage cap; symlink escapes swept | unit + door probe + bashrs | `cargo nextest run … -E 'test(/bench_gate/)'` + `bash scripts/setfit_bench_gate_door_probe.sh` + `bashrs lint …` | ✅ `bench_gate.rs:857-925` | ✅ green (55 passed; probe rc=0, spot-check E rc=5) |
| 05-15.T3 | 05-15 | 10 | EVAL-04 | T-05-15-04 | foreign `contract_id` refused on BOTH row and manifest with `RowSchemaRefused` naming both ids; floors re-measured | unit + make gates | `cargo nextest run … -E 'test(/bench_gate/)'` + `make setfit-bench-tests` + `… -E 'test(/bench_metrics\|bench_row/)'` | ✅ `bench_gate.rs:1265-1272,1316-1327` | ✅ green (55 / 46; make rc=0) |
| 05-16.T1 | 05-16 | 11 | EVAL-02 | T-05-16-01/02/03 | `selection_manifest_hash` RECOMPUTED from the manifest at a **cell-key-derived** path (never a row field); exact 64-hex byte equality; `shots_per_class`/`root_seed` vs the cell key as an independent second statement; absent `selections/` is `EvidenceFileMissing` | unit (tdd) + shipped-door regression | `cargo nextest run … -E 'test(/bench_gate/)'` + `. scripts/apr_bin.sh && "$APR" setfit bench report --bench-dir benchmarks/tweeteval-stance` | ✅ `bench_gate.rs:1683-1733` | ✅ green (55 passed; door rc=0) |
| 05-16.T2 | 05-16 | 11 | EVAL-02 (OPS-03) | T-05-16-04 | the manifest's seal is checked by `SelectionManifest::from_bytes` and **not reimplemented** — held by a forbidden-symbol guard, not by a comment | unit (tdd) + clippy | `cargo nextest run … -E 'test(/bench_gate/)'` + `cargo clippy -p aprender-train --lib --features setfit -- -D warnings` | ✅ `bench_gate.rs:3175-3183` | ⚠️ tests green (55 / 46); clippy rc=101 — see Standing Reds |
| 05-16.T3 | 05-16 | 11 | EVAL-02 | T-05-16-05/06/07 | contract `pv`-valid; report's `verified:` block names all three recomputations; the structurally-absent path-escape is also *guarded* (signature asserted verbatim) | pv + door probe + make gates | `bash scripts/setfit_bench_gate_door_probe.sh` + `pv validate contracts/setfit-benchmark-claims-v1.yaml` + `make setfit-bench-tests` | ✅ contract + `report.md` | ✅ green (probe rc=0, spot-checks G+F rc=5; pv 0/0) |
| 05-17.T1 | 05-17 | 12 | EVAL-01 | T-05-17-07 | the closed-form recomputation cannot be DoS'd by an enormous matrix — a real cap shipped (`MAX_CROSS_CHECK_ROWS`, saturating + short-circuiting), NOT the plan's stated mitigation (deviation 1) | unit | `cargo nextest run … -E 'test(/bench_gate/)'` + `… -E 'test(/bench_metrics\|bench_row/)'` + `cargo nextest run -p apr-cli --lib --features setfit -E 'test(/setfit_bench/)'` | ✅ `bench_metrics.rs:383,458-468` | ✅ green (55 / 46 / 67) |
| 05-17.T2 | 05-17 | 12 | EVAL-01 (OPS-03) | T-05-17-01/02/03 | every published quality figure recomputed from the row's OWN confusion matrix through the SAME shipped entry points and compared by exact `to_bits()`; all five `_bits` siblings held; `n_test_rows` checked FIRST against the matrix total | unit (tdd) + shipped-door control | `cargo nextest run … -E 'test(/bench_gate/)'` + clippy + `"$APR" setfit bench report …` (expect rc=0, `quality_cross_check_mismatch` 0) | ✅ `bench_gate.rs:1770-1843` | ⚠️ tests green (55/46); door rc=0, mismatch 0; clippy rc=101 — see Standing Reds |
| 05-17.T3 | 05-17 | 12 | EVAL-01 (D-19) | T-05-17-04/05/06/08 | the `residual:` line neither over- nor under-states: three statements (module doc, `RESIDUAL_DISCLOSURE`, contract `amended_4_0_0`) present and MATCHING; matrix + calibration diagnostics disclosed as accepted residuals in four places | pv + door probe + make gates | `bash scripts/setfit_bench_gate_door_probe.sh` + `pv validate …` + `cargo nextest run -p apr-cli --lib --features setfit -E 'test(/setfit_bench/)'` | ✅ contract 4.0.0 | ✅ green (probe rc=0, spot-check D rc=5 refused by the cross-check **not** the row digest) |
| 05-BIND.T1 | 05-16/05-17 | gate wiring | EVAL-01/02 | — | every ACTIVE equation of the claims contract has a binding row naming a symbol that EXISTS; the gate refuses ANY BIND- line | make audit | `make contract-audit-phase5` | ✅ `binding.yaml:1609-1633` | ✅ green (rc=0, 12/12; red reproduced at rc=2 under induced mutation) |
| 05-DOOR.T1 | 05-14…05-17 | gate wiring | EVAL-01/02/04 | T-05-15-01/-03, T-05-16-01/-03, T-05-17-01 (secondary leg) | the door-level proof of spot-checks D/E/F/G now RUNS in a tier instead of in no gate; its `apr` prerequisite is BUILT by the target, with no skip path | make gate (build + spawned door) | `make setfit-bench-door-probe-build` (wired into `tier4`, `Makefile:945`) | ✅ `Makefile:2826-2874` | ✅ green (rc=0; control rc=0, then four attacks rc=5) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky · ⚪ compute-gated, not re-run this audit*

---

## Wave 0 Requirements

Mapped from RESEARCH.md "Wave 0 Gaps" to owning plans (all covered):

- [x] Production calibration harness (`production_calibration_matrix`, env-gated, `#[ignore]`d) — **05-01**
- [x] Per-regime `Thresholds` restructuring + updated `thresholds_match_the_contract` — **05-02** (structure) + **05-03** (the len 1→2 edit)
- [x] `contracts/setfit-benchmark-claims-v1.yaml` + `$(CONTRACTS)` append + scoped audit target — **05-05** (+ binding rows in **05-10**)
- [x] Multiclass ECE/Brier in `aprender-core::calibration` + uv-env fixture generator + SHA-256 manifest — **05-04**
- [x] f64 paired-stats helpers + frozen `t_{0.975,9}` + scipy fixtures — **05-04**
- [x] Row type + run manifest + doctored-row negatives — **05-05** (types) + **05-10** (five negatives)
- [x] `SetfitCommands::Bench` + `commands/setfit_bench.rs` + Make/tier wiring (non-vacuous) — **05-09** (run) + **05-10** (report + gates)
- [x] `--selection-manifest` on finetune + explicit `TrainingConfig` control + refusal tests — **05-06**
- [x] Per-row-predictions evaluator door beside `evaluate_validation_from_artifact` — **05-08**

---

## Standing Reds and Coverage Caveats

Measured at HEAD on 2026-09-13. None is a Phase 5 regression; each is named so a future
reader does not misread a red as new breakage — the T-05-13-06 discipline, applied to this file.

| Red / caveat | Measurement | Why it is not a Phase 5 gap |
|---|---|---|
| `make setfit-all-tests` — `setfit::artifact::determinism::the_fixture_artifact_hash_matches_the_committed_golden` | rc=2; **85 passed / 1 failed**. Hash pair left `cc17764d…aab675`, right `13e5c296…7000bf4` | **Re-verified independently this audit, not quoted from 05-13:** `git diff --stat 35cd7c210..HEAD -- crates/aprender-core/src/setfit/ crates/aprender-core/tests/fixtures/setfit/` is **0 bytes**, and `artifact.rs` was last touched by Phase 4 commit `fb7904bad`. Phase 5 changed neither the code under test nor its fixtures |
| `cargo clippy -p aprender-train --lib --features setfit -- -D warnings` — cited by 05-16.T2 and 05-17.T2 | rc=101, **20 errors** | **Zero findings in `aprender-train`.** All 20 are in dependency crates `aprender-compute` (17) and `aprender-present-terminal` (1) — unused imports, unreachable expressions, arch-gated dead code on this arm64 host. Of the 13 failing files only 3 were touched anywhere in the range, and `git log` attributes those to the NEON GEMM perf commit `9d67b7247` and dogfood batches `e759af233` / `a8c6807d8` — **no Phase 5 commit**. This is the CLAUDE.md non-monotonic-clippy / ceiling-gate class |
| 05-07.T1's spawned production ladder | `cargo nextest run -p apr-cli --test setfit_cli_lifecycle --features setfit` → rc=0, but **1 passed / 5 `#[ignore]`d** | 5 of the target's 6 tests are `#[ignore]`d; the only one running by default is `setfit_cli_the_binary_under_test_is_pinned_and_spawned_from_exactly_one_site` — the binary-pin source guard (T-05-07-01), **not** the end-to-end ladder. Already a contracted compute exception in Sampling Rate above; recorded here so the row's rc=0 is not read as ladder coverage |
| T-05-03-05's frozen-epsilon success leg | `PRODUCTION_LOWER_BOUND` (`evidence.rs:1749`) has exactly **one** caller, `:4221`, inside the `#[ignore]`d production combine | Default `cargo test` never exercises it. 05-14.T1's `epsilon_basis` tests cover the *rule*; the *frozen production constants* are proven only under an explicit `--ignored` run. Surfaced by the security audit (2026-09-12) and confirmed here |


---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Contract-edit approval (D-04) | EVAL-02..05 unblock | Human checkpoint by explicit ruling | 05-03.T2: executor presents pv diff + measured thresholds + MEASURED/COVERED margins; approve before commit |
| ~~lambda-vector GPU access + 9B base weights (A1)~~ **RESOLVED-AS-BLOCKED 2026-09-08 (D-19): arm descoped, row retired** | EVAL-02/03/05 | Remote host credentials + weight provenance | 05-11.T1: automation-first ssh probes presented; human supplies access, confirms weight hash, decides Q6 SetFit host |
| >1hr compute authorizations | CLAUDE.md rule | Compute-budget decisions reserved for the human | 05-01.T1 (matrix projection), 05-03.T2 (05-12 pre-auth), 05-12.T1 (fallback gate) |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (checkpoints carry `<human-check>`)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 300s (compute runs excepted via contracted gates)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated 2026-09-13

---

## Validation Audit 2026-09-13

| Metric | Count |
|--------|-------|
| Task rows audited | 46 (34 pre-existing + 12 added) |
| Gaps found | 3 |
| Resolved | 3 |
| Escalated | 0 |
| Standing reds recorded (not Phase 5's) | 4 |

**What the audit changed.** The file was a pre-execution artifact — `status: ready`,
`wave_0_complete: false`, every row `⬜ pending`, and a Per-Task Map that stopped at 05-13
while plans 05-14…05-17 had already landed. Statuses are now measured rather than planned.

| Gap | Finding | Resolution |
|-----|---------|-----------|
| GAP-1 | `make contract-audit-phase5` **rc=2** — `BIND-001` on `selection_binding_rule` (added by 05-16) and `quality_closed_form_crosscheck` (added by 05-17). 05-10 had deliberately tightened this gate to refuse ANY BIND- line, so a wired Phase 5 gate was red at HEAD and nothing recorded it | Two rows appended to `contracts/aprender/binding.yaml:1609-1633`, naming `verify_selection_binding` and `verify_quality_closed_form` (both confirmed present at `bench_gate.rs:1683` / `:1770` — the block's own comment records a prior audit finding rows that named functions which never existed, and `pv audit` cannot resolve a symbol). `make contract-audit-phase5` → **rc=0, 12/12 bound**. Falsified by mutating a new row's equation name: **rc=2** with exactly the expected BIND-001 line, then reverted |
| GAP-2 | `setfit-bench-door-probe` (`Makefile:2790`) was a prerequisite of nothing and absent from `.PHONY` — the sole door-level proof of spot-checks D/E/F/G, cited as the secondary leg of five security threats, ran in **no gate** | New `setfit-bench-door-probe-build` target builds the release `apr` with `--features setfit`, then runs the probe; wired into **tier4** (`Makefile:945`), unprefixed so a failure fails the tier. Chosen over a skip path because `scripts/apr_bin.sh` refuses a stale binary — a skip branch would have been the vacuous gate T-05-07-02 exists to prevent. Proven: rc=0 with the control passing **before** the four attacks (E, G, F, D each rc=5), so "refused" is distinguishable from "the scratch tree was broken" |
| GAP-3 | Plans 05-14…05-17 had no rows at all — 34 `<automated>` blocks unmapped | 12 rows added (05-14.T1…05-17.T3 plus `05-BIND.T1` and `05-DOOR.T1` for the two gate-wiring fixes), threat refs taken from `05-SECURITY.md` |

**Files changed by this audit:** `contracts/aprender/binding.yaml`, `Makefile`. No file under
`crates/**/src/` was modified.
