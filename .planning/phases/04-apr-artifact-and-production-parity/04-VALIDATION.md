---
phase: 4
slug: apr-artifact-and-production-parity
status: ready
nyquist_compliant: true
wave_0_complete: false
created: 2026-08-14
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Populated from 04-RESEARCH.md § Validation Architecture + the 04-01..04-16 plan set
> (16 plans, 9 waves, 38 tasks).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (libtest) + trybuild + proptest; cargo-nextest 0.9.102 in CI; cargo-mutants 25.3.1 (scoped, 04-11 only) |
| **Config file** | Makefile tiers (tier1–tier4) + `.github/workflows/ci.yml` (setfit steps at ci.yml:274–309) |
| **Quick run command** | `cargo test -p <touched-crate> --features setfit --lib setfit::` (aprender-core / aprender-train; apr-cli and aprender-serve use the per-task scoped filters below) |
| **Full suite command** | `make tier3` (setfit targets + contracts via `$(PV_BIN)` + feature matrix; parity harness runs in default `cargo test` after 04-09) |
| **Estimated runtime** | scoped filter ~30–120s incl. incremental compile; `make tier3` 1–5 min |

---

## Sampling Rate

- **After every task commit:** Run the task's scoped `<automated>` filter for the touched crate(s) — every filtered run carries a nonzero-ran guard (CR-02: zero-match exiting 0 is vacuous)
- **After every plan wave:** Run `make tier2` (arm64 scoped-clippy caveat per RESEARCH Pitfall 6) + all scoped setfit suites landed so far
- **Before `/gsd:verify-work`:** `make tier3` must be green (setfit targets, `$(PV_BIN) validate contracts/setfit-apr-v1.yaml`, feature matrix, parity harness + in-band negative)
- **Max feedback latency:** 120 seconds (scoped filter including incremental compile)

---

## Per-Task Verification Map

Rows are in WAVE order. All 38 tasks across 04-01..04-16 appear exactly once.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 1 | SAFE-01, APR-01 | T-04-01/02/37/38/58 | contract carries falsification gates; tolerances cite frozen provenance; nullable-path allowlist derived from the four sub-documents' `Option` fields | contract | `cargo run --release -p aprender-contracts-cli --bin pv -- validate contracts/setfit-apr-v1.yaml` | ❌ W0 | ⬜ pending |
| 4-01-02 | 01 | 1 | SAFE-01 | T-04-01 | contract wired into `$(CONTRACTS)` + blocking phase audit; count compared explicitly, not read off `grep -c`'s exit status | gate | `n=$(grep -v '^#' Makefile \| grep -c "setfit-apr-v1.yaml"); test "$n" -ge 2 && make contract-audit-phase4` | ❌ W0 | ⬜ pending |
| 4-01-03 | 01 | 1 | SAFE-01 | — | N/A (docs — D-09 exception row) | docs | `grep -c "SetFit" CLAUDE.md` | ✅ | ⬜ pending |
| 4-02-01 | 02 | 2 | APR-01 | T-04-03/04/05/38/58 | deterministic metadata; probes contain no dataset text; non-finite rejected before write; the production full-pin shape (`vocab_remap: None`) is ACCEPTED — counted separately under `setfit::artifact::nullable` | unit | `cargo test -p aprender-core --features setfit --lib setfit::artifact::` | ❌ W0 | ⬜ pending |
| 4-02-02 | 02 | 2 | APR-01 | T-04-03 | cross-process byte equality + hash stability on the full-pin fixture shape | unit | `cargo test -p aprender-core --features setfit --lib setfit::artifact::determinism` | ❌ W0 | ⬜ pending |
| 4-13-01 | 13 | 2 | APR-01, APR-05 | T-04-39/40/58 | provenance read off the run's Selection, never caller strings; schema-version bump refuses old payloads typed; allowlist completeness gate fails by naming a new `Option` path | unit | `cargo test -p aprender-train --features setfit --lib setfit::bundle` | ❌ W0 | ⬜ pending |
| 4-13-02 | 13 | 2 | APR-01 | T-04-39 | the single bundle-assembly site threads the run's selection; no policy step moves | unit | `cargo test -p aprender-train --features setfit --lib setfit::verify` | ✅ | ⬜ pending |
| 4-14-01 | 14 | 2 | OPS-02 | T-04-43 | overrides route through the single validating constructor; no in-place config mutator is added | unit | `cargo test -p aprender-train --features setfit --lib setfit::config` | ✅ | ⬜ pending |
| 4-14-02 | 14 | 2 | TRN-07 | T-04-41/42 | lock payload bounded BEFORE serde; lock_hash recomputed from the reconstructed record; no new float-taking door | unit | `cargo test -p aprender-train --features setfit --lib setfit::lock` | ✅ | ⬜ pending |
| 4-03-01 | 03 | 3 | APR-02 | T-04-06/07/08/09 | every corruption class dies typed before a model exists; length cap before parse | unit (induced corruption) | `cargo test -p aprender-core --features setfit --lib setfit::artifact::ladder` | ❌ W0 | ⬜ pending |
| 4-03-02 | 03 | 3 | APR-02, APR-05 | T-04-07/10 | probe replay is the last rung; typestate mint-on-success; embed accessor typed-fallible | unit | `cargo test -p aprender-core --features setfit --lib setfit::artifact::probe` | ❌ W0 | ⬜ pending |
| 4-03-03 | 03 | 3 | APR-04 | T-04-10 | out-of-crate code cannot mint the witness type | trybuild (compile-fail) | `cargo test -p aprender-train --features setfit --test ui` | ❌ W0 | ⬜ pending |
| 4-04-01 | 04 | 4 | OPS-04 | T-04-13 | non-finite values unrepresentable in serialized output, construction and deserialization alike | unit + golden | `cargo test -p aprender-core --features setfit --lib setfit::classify::envelope` | ❌ W0 | ⬜ pending |
| 4-04-02 | 04 | 4 | OPS-06 | T-04-12/59 | backend identity returned BY the encode invocation; no capability-detection symbol anywhere in the setfit surface; the filter matches a module this task creates | unit | `cargo test -p aprender-core --features setfit --lib setfit::classify::backend` | ❌ W0 | ⬜ pending |
| 4-04-03 | 04 | 4 | OPS-04, OPS-06 | T-04-11/12 | batch cap typed (256); token_count asserted against the exported MAX_SEQUENCE_LENGTH; latency finite and >= 0 only | unit | `cargo test -p aprender-core --features setfit --lib setfit::classify::` | ❌ W0 | ⬜ pending |
| 4-05-01 | 05 | 4 | APR-03 | T-04-14 | foreign format refused twice (trusted + codec self-check); closure holds twice | unit | `cargo test -p aprender-train --features setfit --lib setfit::apr_codec::` | ❌ W0 | ⬜ pending |
| 4-05-02 | 05 | 4 | APR-03 | T-04-15/16 | trusted policy closure + EXACT + production-loader cross-check; mutated byte fails typed | integration (in-lib) | `cargo test -p aprender-train --features setfit --lib setfit::apr_codec::round_trip` | ❌ W0 | ⬜ pending |
| 4-06-01 | 06 | 5 | OPS-02 | T-04-17 | feature-gated namespace compiles with AND without setfit; the check leg is the SAFE-02 gating evidence (dev-deps cannot reach it) | build + unit | `cargo check -p apr-cli --features setfit && cargo check -p apr-cli && cargo test -p apr-cli --features setfit --lib setfit_io` | ✅ (checks existing crates) | ⬜ pending |
| 4-06-02 | 06 | 5 | OPS-02, OPS-06 | T-04-17/18/19 | deny_unknown_fields config; typed CudaNotAvailable before data load; atomic temp+rename write | unit | `cargo test -p apr-cli --features setfit --lib setfit_train` | ❌ W0 | ⬜ pending |
| 4-06-03 | 06 | 5 | OPS-02 | T-04-19 | end-to-end tiny-fixture train leg (tier3-weight) | integration (ignored) | `cargo test -p apr-cli --features setfit --lib setfit_train -- --ignored` | ❌ W0 | ⬜ pending |
| 4-12-01 | 12 | 5 | OPS-01 | T-04-36/51 | zero apr-cli deps; train→save→load→embed→classify→inspect via public API only, every value read through an accessor | integration | `cargo test -p aprender-train --features setfit --test setfit_apr_lifecycle` | ❌ W0 | ⬜ pending |
| 4-16-01 | 16 | 5 | APR-04, TRN-07 | T-04-46/47/48/49/60 | fresh-process reload re-enters the ONE trusted policy; all three provenance identifiers (semantic hash, LEDGER hash, dataset fingerprint) refused by name; byte-identity gate before minting | unit | `cargo test -p aprender-train --features setfit --lib setfit::apr_reload::` | ❌ W0 | ⬜ pending |
| 4-07-01 | 07 | 6 | OPS-03 | T-04-20/22 | explicit-tag-only auto-detect; untagged APR stays plain (negative test) | unit | `cargo test -p apr-cli --features setfit --lib predict` | ❌ W0 | ⬜ pending |
| 4-07-02 | 07 | 6 | APR-05 | T-04-22 | inspect recovers every identity field via the full ladder | unit | `cargo test -p apr-cli --features setfit --lib inspect` | ❌ W0 | ⬜ pending |
| 4-07-03 | 07 | 6 | OPS-02, TRN-07 | T-04-21 | test split reachable only via lock→token→grant; typed refusal without lock | unit | `cargo test -p apr-cli --features setfit --lib eval::setfit` | ❌ W0 | ⬜ pending |
| 4-08-01 | 08 | 6 | OPS-05 | T-04-23/24/25 | slot holds only VerifiedSetFitModel; batch + 1 MiB body bounds | build | `cargo check -p aprender-serve --features setfit && cargo check -p aprender-serve` | ✅ (checks existing crates) | ⬜ pending |
| 4-08-02 | 08 | 6 | OPS-03, OPS-05 | T-04-24 | startup branches inside the APR arm after the typed tag; typed fail on ladder errors | unit | `cargo check -p apr-cli --features setfit && cargo test -p apr-cli --features setfit --lib serve` | ❌ W0 | ⬜ pending |
| 4-08-03 | 08 | 6 | OPS-05 | T-04-23/25 | oneshot suite in every `cargo test`; readiness hash pinned to the loaded model | in-process HTTP | `cargo test -p aprender-serve --features setfit --lib setfit` | ❌ W0 | ⬜ pending |
| 4-09-01 | 09 | 7 | SAFE-01, OPS-04, OPS-01 | T-04-29/53/61 | three-surface value parity on ONE ClassifyRequestDocument; CLI leg pinned via CARGO_BIN_EXE_apr, never PATH; the dev-dep feature-unification cost to SAFE-02's run leg is recorded where it happens | integration | `cargo test -p apr-cli --features setfit,inference --test setfit_parity` | ❌ W0 | ⬜ pending |
| 4-09-02 | 09 | 7 | SAFE-01 | T-04-27/28 | frozen goldens + SHA-256 manifest; in-band skewed negative FAILS the gate every run | golden + negative | `cargo test -p apr-cli --features setfit,inference --test setfit_parity golden` | ❌ W0 | ⬜ pending |
| 4-09-03 | 09 | 7 | SAFE-01 | T-04-29 | ONE spawned-serve smoke; reserved-port handoff; Drop-guarded child; status never read through a pipe | spawned smoke (ignored, tier3) | `cargo test -p apr-cli --features setfit,inference --test setfit_parity -- --ignored spawned_serve_smoke` | ❌ W0 | ⬜ pending |
| 4-15-01 | 15 | 7 | OPS-02 | T-04-54/29 | five real binary invocations consuming each other's outputs; the selection lock crosses a process boundary as a FILE; missing-lock negative exits nonzero | spawned lifecycle (ignored, tier3) | `cargo test -p apr-cli --features setfit --test setfit_cli_lifecycle -- --ignored lifecycle` | ❌ W0 | ⬜ pending |
| 4-15-02 | 15 | 7 | OPS-03 | T-04-55 | generic `apr tensors`/`inspect`/`qa` run against a real artifact; the U8 `tokenizer.blob` and both `setfit.head.*` entries are named; a nonzero qa verdict is surfaced as a finding, not suppressed | spawned tooling (ignored, tier3) | `cargo test -p apr-cli --features setfit --test setfit_cli_lifecycle -- --ignored tooling` | ❌ W0 | ⬜ pending |
| 4-10-01 | 10 | 8 | SAFE-01 | T-04-30/31/56/62 | ran-something guards on every filtered target incl. setfit-config-tests and setfit-evaluate-tests; ONE positional filter per invocation; `rc=$?` never through a pipe | gate (Make) | `make setfit-apr-tests && make setfit-classify-tests && make setfit-config-tests && make setfit-evaluate-tests && make setfit-codec-tests && make setfit-reload-tests && make setfit-cli-predict-tests && make setfit-serve-tests && make setfit-parity && make setfit-api-boundary` | ❌ W0 | ⬜ pending |
| 4-10-02 | 10 | 8 | SAFE-02 | T-04-32/57/61 | four-crate x three-profile matrix, build AND run legs; boundary greps carry a must-match case table; the dev-dep unification is stated in-recipe | build matrix | `make setfit-feature-matrix && make contract-audit-phase4` | ❌ W0 | ⬜ pending |
| 4-11-01 | 11 | 9 | SAFE-02 | T-04-33/63 | patch-file-only proposal; ci.yml untouched in the working tree; every 04-10 target included or excluded with a written rationale | patch evidence | `git apply --check .planning/phases/04-apr-artifact-and-production-parity/04-11-ci-setfit.patch && rtk proxy git status --porcelain -- .github/workflows/ci.yml` | ❌ W0 | ⬜ pending |
| 4-11-02 | 11 | 9 | SAFE-02 | T-04-33 | blocking human checkpoint gates the CI edit; workflow still clean at the moment of approval | checkpoint | `rtk proxy git status --porcelain -- .github/workflows/ci.yml` | ✅ | ⬜ pending |
| 4-11-03 | 11 | 9 | SAFE-02, TRN-07 | T-04-34/35 | evidence-cited requirement audit; per-crate mutation reports with baselines; OPS-01 cited with BOTH its Make target and its CI leg | gate | `grep -c "apr-cli --features setfit" .github/workflows/ci.yml && grep -A 2 "TRN-07" .planning/REQUIREMENTS.md \| grep -c "\[x\]"` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Note (revision 2): the OPS-01 lifecycle task moved from 04-05 Task 3 (wave 4) to 04-12 Task 1
(wave 5, depends_on 04-04 + 04-05) — its classify step consumes 04-04's
`VerifiedSetFitModel::classify`, which lands in the PARALLEL wave-4 plan and is not visible in a
wave-4 worktree.*

*Note (revision 3): this map was regenerated for the full 16-plan / 38-task set. Revision 2's map
covered only 04-01..04-12 and listed 30 rows — it was missing 04-04 Task 2 (the backend-identity
suite) and every task of 04-13, 04-14, 04-15 and 04-16, so its sign-off attested to a plan set that
no longer existed. Four commands also changed in revision 3: 4-01-02 now captures the grep count and
compares it explicitly (a bare `grep -c` exit status cannot distinguish 1 from 2); 4-10-01 gained
`setfit-config-tests` and `setfit-evaluate-tests`; 4-11-01 now checks the patch FILE rather than a
working-tree diff, per 04-11's M12 shape; 4-06-01 now includes the `setfit_io` unit leg its task
actually verifies.*

---

## Wave 0 Requirements

No standalone Wave 0 scaffold plan: every code-producing task is `tdd="true"` and creates its own
test surface RED-first inside the task, so each ❌ W0 marker above is satisfied by its own task's
RED step before implementation. The new test surfaces (mapping RESEARCH § Wave 0 Gaps → plans):

- [ ] `contracts/setfit-apr-v1.yaml` + `$(CONTRACTS)` append + `PHASE4_CONTRACTS` + `contract-audit-phase4` — 04-01
- [ ] `crates/aprender-core/src/setfit/artifact.rs` tests (ladder / determinism / probe) — 04-02, 04-03
- [ ] `crates/aprender-core/src/setfit/artifact.rs` `mod nullable` — the allowlist ACCEPT/REJECT suite, incl. the full-pin production shape — 04-02
- [ ] `crates/aprender-core/src/setfit/classify.rs` tests: `mod envelope` + goldens, `mod backend`, `mod classify_path` — 04-04
- [ ] `crates/aprender-train/src/train/setfit/apr_codec.rs` closure/determinism tests — 04-05
- [ ] `crates/aprender-train/src/train/setfit/apr_reload.rs` tests (fresh-process reload, three provenance refusals, byte-identity gate) — 04-16
- [ ] `crates/aprender-train/src/train/setfit/bundle_tests.rs`: provenance tests + the nullable-path allowlist COMPLETENESS gate — 04-13
- [ ] `crates/aprender-train/src/train/setfit/lock_tests.rs`: `SelectionLock::from_canonical_bytes` round-trip + bounded-parse + edited-candidate tests — 04-14
- [ ] `crates/aprender-train/src/train/setfit/config.rs`: `to_request` identity/override tests — 04-14
- [ ] `crates/aprender-train/tests/ui/setfit_verified_model_constructed.{rs,stderr}` trybuild case — 04-03
- [ ] `crates/aprender-train/tests/setfit_apr_lifecycle.rs` (OPS-01) — 04-12
- [ ] apr-cli setfit_train / predict / inspect / eval / serve / setfit_io test modules — 04-06, 04-07, 04-08
- [ ] aprender-serve oneshot suite — 04-08
- [ ] `crates/apr-cli/tests/setfit_parity.rs` + goldens + SHA-256 manifest + in-band negative — 04-09
- [ ] `crates/apr-cli/tests/setfit_cli_lifecycle.rs` (spawned OPS-02 chain + generic APR tooling / A3) — 04-15
- [ ] Make targets with ran-something guards (incl. setfit-config-tests, setfit-evaluate-tests); feature matrix growth — 04-10
- [ ] `.planning/phases/04-apr-artifact-and-production-parity/04-11-ci-setfit.patch` — the CI proposal artifact its own `git apply --check` verifies — 04-11

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ci.yml setfit-step extension approval | SAFE-02 | CLAUDE.md forbids autonomous CI workflow edits, and per review M12 preparing a working-tree edit is already acting; 04-11 Task 2 is a blocking human checkpoint | Read `.planning/phases/04-apr-artifact-and-production-parity/04-11-ci-setfit.patch` (a patch FILE — `.github/workflows/ci.yml` must be clean, confirmed by `rtk proxy git status --porcelain -- .github/workflows/ci.yml`). Confirm the added commands appear verbatim in the Makefile, and that the target-coverage accounting shows exactly two exclusions (both spawned-binary tier3 targets). Approve or reject; only then does Task 3 `git apply` it |

All other phase behaviors have automated verification.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (verified against the 04-01..04-16 task blocks — 38/38 tasks carry `<automated>`; per-plan counts 01:3, 02:2, 03:3, 04:3, 05:2, 06:3, 07:3, 08:3, 09:3, 10:2, 11:3, 12:1, 13:2, 14:2, 15:2, 16:1)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (every task has one; 04-11 Task 2's checkpoint is bracketed by automated evidence commands)
- [x] Wave 0 covers all MISSING references (tdd tasks create their own test files RED-first; no orphan MISSING markers exist in any plan)
- [x] Every gate can fail: 4-01-02 compares a captured count rather than a `grep -c` exit status; 4-02-01 carries counted ACCEPT cases on the production shape; 4-13-01's completeness gate and 4-10-01's guards are each falsified once and recorded
- [x] No watch-mode flags (audited: no `--watch`/watch-mode invocation in any `<automated>` command)
- [x] Feedback latency < 120s (scoped per-crate filters; tier3 reserved for wave/phase gates)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-08-14 (revision 3: regenerated for the 16-plan / 38-task set — added 4-04-02 and all of 04-13/04-14/04-15/04-16; corrected 4-01-02, 4-06-01, 4-10-01 and 4-11-01 commands; corrected the manual-verification instructions to the patch-file gate)
