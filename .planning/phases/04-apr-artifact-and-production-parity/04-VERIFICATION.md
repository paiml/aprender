---
phase: 04-apr-artifact-and-production-parity
verified: 2026-08-16T04:36:49Z
verified_at_commit: 0fb47958f
status: gaps_acknowledged
original_status: gaps_found
score: 94/100 must-haves verified
roadmap_success_criteria: 0/5 fully met (2 FAILED, 3 PARTIAL)
overrides_applied: 0
acknowledged: 2026-08-16T23:29:10Z
acknowledged_at_commit: b3f816c25
acknowledged_via: .planning/phases/04-apr-artifact-and-production-parity/04-UAT.md
acknowledged_note: >-
  The verdict below is UNAMENDED and its measurements stand as taken at 0fb47958f. Five
  gap-closure plans (04-18..04-22) landed afterwards; four of the gaps listed here are
  closed and re-measured, and the remainder were ruled on by a human. See the
  "## Acknowledged Gaps" section at the end of this file. Phase 4's success criteria are
  NOT retroactively marked met — SC1 and SC3 remain unmet pending F-10 closure in Phase 5.
gaps:
  - truth: "SC1 — A user can SAVE one checksummed F32 setfit-apr-v1 containing the complete encoder, tokenizer bytes/hash, policy, head, labels, config, evidence and provenance"
    status: failed
    reason: >-
      The WRITER half ships and is proven (write_setfit_apr, 86 green tests, cross-process
      byte-identical SHA-256). The PRODUCER half does not exist at any user-reachable tier.
      Independently re-measured at verification time: CALIBRATED_REGIMES
      (thresholds.rs:61-62) holds exactly ONE entry whose architecture component is compared
      for exact equality, so the phase-3 MiniLM slice is the only trainable encoder, and its
      97-row vocabulary closure cannot compute probe_unicode (canonical id 5915). F-10.
      The inspect and load halves of SC1 are VERIFIED.
    artifacts:
      - path: "crates/aprender-train/src/train/setfit/thresholds.rs"
        issue: "CALIBRATED_REGIMES has one entry; exact-equality architecture match gates tune_encoder"
      - path: "crates/aprender-train/src/train/setfit/apr_codec.rs"
        issue: "the only APR-capable run (fixture::apr_capable_run, line 774) is #[cfg(test)] and substitutes the encoder and head"
    missing:
      - "A calibration run on the production all-MiniLM-L6-v2 encoder"
      - "A deliberate edit adding its fingerprint to contracts/setfit-train-lifecycle-v1.yaml (D-10(c))"
  - truth: "SC2 — Training closes its in-memory model, reloads the APR through the production core loader, and proves exact tokenizer/configuration/tensor state plus tolerance-bounded embeddings, logits, probabilities and exact labels"
    status: partial
    reason: >-
      The MECHANISM is real and green: verify_artifact(AprCodec) drives the whole trusted
      verify policy (close -> serialize -> hash -> reload through read_setfit_apr_parts ->
      byte-canonical closure -> Tolerance::EXACT) and 17 setfit-codec tests pass. The
      no-bypass half is compile-proven from OUTSIDE the crate (tests/ui/
      setfit_verified_model_constructed.rs, green in a real trybuild run). What has never
      run is the policy over a model the SHIPPED TRAIN PATH produced — dataset, selection,
      config and evidence are real; the encoder and head are substituted (F-10).
    artifacts:
      - path: "crates/aprender-train/src/train/setfit/apr_codec.rs"
        issue: "fixture::apr_capable_run substitutes encoder + head; the module header states this and keeps the finding executable"
    missing:
      - "F-10 closure, then one execution of verify_artifact(AprCodec) over a genuinely trained encoder"
  - truth: "SC3 — A Rust caller and an `apr` user can complete the CPU train -> APR -> inspect -> eval -> predict lifecycle"
    status: failed
    reason: >-
      OPS-01 and OPS-02 are NOT met. Re-measured at verification time, not read off a
      SUMMARY: `make setfit-lifecycle-tests` passes 5/5 and those tests ASSERT the refusal —
      the save-as-setfit-apr-v1 rung returns a typed ProbeComputation{probe:"probe_unicode"}
      and load/embed/classify/inspect have no admissible input. `make setfit-cli-lifecycle`
      passes 2+1 spawned tests and rung 4 (`apr setfit train`) exits 6 with model.apr proven
      absent afterwards. The auto-detect and shared-core-path halves of SC3 ARE verified.
    artifacts:
      - path: "crates/aprender-train/tests/setfit_apr_lifecycle.rs"
        issue: "OPS-01 rung table: save-as-APR / load / embed / classify / inspect all marked NO — F-10"
      - path: "crates/apr-cli/tests/setfit_cli_lifecycle.rs"
        issue: "OPS-02 spawned ladder stops at rung 4, exit 6 (ModelLoadFailed)"
    missing:
      - "F-10 closure — same calibration + contract edit as SC1"
  - truth: "SC4 — Rust, CLI and native HTTP callers receive matching labels, probabilities, logits, margins, token/truncation facts, latency, BACKEND IDENTITY, and the same artifact hash in predictions and readiness"
    status: partial
    reason: >-
      Three-surface parity is real and falsifiable: `make setfit-parity` ran 20/20 green,
      the three legs are the library call, a spawned CARGO_BIN_EXE_apr and an in-process
      realizar router oneshot over ONE ClassifyRequestDocument, with frozen goldens + a
      SHA-256 manifest and an in-band skewed negative built through the public validating
      constructor. Readiness/response artifact agreement is proven. TWO clauses are not
      closed: (a) the parity fixture is SYNTHETIC, because no user-producible artifact
      exists (F-10); (b) the contract equation `backend_identity` is still `status: pending`
      in contracts/aprender/binding.yaml because its row names
      `aprender::setfit::classify::backend_identity`, which does not exist — verified at
      verification time by `make contract-audit-phase4` reporting BIND-004 for that one
      equation. The identity is produced by ExecutionBackend::identity in encoder.rs.
    artifacts:
      - path: "contracts/aprender/binding.yaml"
        issue: "backend_identity binds to a symbol that does not exist; equation left pending"
    missing:
      - "Correct the backend_identity binding row's `function` column to the shipped symbol (a semantic registry edit, deliberately out of 04-10's scope)"
      - "F-10 closure, then re-run parity over a produced artifact"
  - truth: "SC5 — A developer can run offline executable contracts and the supported CPU build/test feature matrix; an explicitly requested unavailable device fails instead of silently falling back or MISREPORTING ITS BACKEND"
    status: partial
    reason: >-
      Contracts and the local matrix are verified by execution, not by claim: `pv validate
      contracts/setfit-apr-v1.yaml` rc=0 (0 errors, 0 warnings); `make contract-audit-phase4`
      rc=0 with 15/15 equations bound; `make setfit-feature-matrix` rc=0 over 4 crates x 3
      CPU profiles with BUILD and RUN legs and two-sided graph negatives; `make
      setfit-api-boundary` rc=0 with an executed must-match control. The device gate fails
      closed (unit test green; two structured CLI refusals observed directly at the binary
      tier). NOT closed: (a) the "misreporting the backend" clause depends on the same
      unresolved backend_identity binding as SC4; (b) SAFE-02's "in CI" clause — the ci.yml
      setfit step IS applied (commit 57f7823ab, after the human ruling) but has never RUN,
      so its greenness is unproven.
    artifacts:
      - path: ".github/workflows/ci.yml"
        issue: "16 setfit legs applied at lines 378-397 but never executed on this branch"
    missing:
      - "One green CI run of the applied setfit step"
      - "backend_identity binding resolution"
  - truth: "04-11 must-have 4 — Scoped cargo-mutants runs PER CRATE with its own baseline, and the aggregate adjusted score is computed explicitly from the per-crate numbers"
    status: failed
    reason: >-
      Only ONE of the four crates was attempted (aprender-serve), the run was interrupted at
      ~68 min without finishing its 101 mutants, and NO score — per-crate or aggregate — was
      produced. The stop was measured and reported honestly (D-04-11-B projects >= 10 h wall
      clock for all 890 mutants) rather than silently rescoped, but the must-have as written
      is not met.
    artifacts:
      - path: ".planning/phases/04-apr-artifact-and-production-parity/deferred-items.md"
        issue: "D-04-11-B records the measured stop; no per-crate baselines, no adjusted score"
    missing:
      - "A scheduled per-crate cargo-mutants run for aprender-core, aprender-train, aprender-serve and apr-cli with per-crate baselines"
      - "An explicitly computed aggregate adjusted score"
      - "Closure of the two REAL production survivors recorded in D-04-11-A (has_setfit_model -> false; `>` -> `==` at setfit_handlers.rs:158)"
deferred:
  - truth: "F-10 — CALIBRATED_REGIMES admits only the phase-3 MiniLM slice, whose 97-row vocabulary closure cannot compute probe_unicode, so no user-reachable path produces a setfit-apr-v1"
    addressed_in: "Phase 5"
    evidence: >-
      ROADMAP.md Phase 5 carries an explicit blocking note: "Unblocking it requires
      calibrating on the production encoder and adding its fingerprint to
      contracts/setfit-train-lifecycle-v1.yaml — a deliberate, `pv diff`-flagged contract
      edit per Phase 3 D-10(c), never an inline relaxation by a Phase 5 executor."
      NOTE: this defers the MECHANISM only. OPS-01 and OPS-02 remain Phase 4 requirements —
      Phase 5's success criteria are EVAL-01..05 and do not claim them — so they are
      reported above as gaps, not as deferred items.
human_verification:
  - test: "Open a PR from gsd/phase-2-contract-gate (or push a branch that triggers ci.yml) and read the `setfit` step's result"
    expected: "All 16 applied setfit legs at .github/workflows/ci.yml:378-397 exit 0 on the Linux CI image"
    why_human: >-
      The legs are applied but have never executed. Every leg was verified GREEN locally on
      aarch64 Darwin at verification time, but SAFE-02's clause is literally 'in CI', and a
      Linux dependency closure can differ from the macOS dev host via cfg(target_os). This
      cannot be settled without a CI run.
  - test: "Run the per-crate cargo-mutants gate as a scheduled job, per D-04-11-B's four recommendations"
    expected: "Four per-crate baselines and an explicitly computed aggregate adjusted score"
    why_human: "D-04-11-B measured >= 10 h wall clock for 890 mutants — beyond any single session"
  - test: "Decide the disposition of the two REAL production mutation survivors in crates/aprender-serve/src/api/setfit_handlers.rs (D-04-11-A)"
    expected: >-
      Survivor A (AppState::has_setfit_model -> false) killed by an assertion; survivor B
      (`>` -> `==` at line 158) diagnosed — a test NAMED
      setfit_classify_refuses_a_batch_one_over_the_contract_bound should already kill it
    why_human: "Requires deciding whether the named test is mis-scoped or unreached; D-04-11-A explicitly declined to diagnose from one input (CLAUDE.md rule 6)"
  - test: "Rule on the 6 Warning and 5 Info findings in 04-REVIEW.md that were never closed or tracked"
    expected: "Each either fixed or recorded in deferred-items.md with an owner"
    why_human: >-
      Only the 2 Critical findings were closed (commit 0fb47958f). WR-02 in particular is
      security-relevant and STILL PRESENT — verified at eval/setfit.rs:620-636: write_lock's
      temp file is a predictable `.{name}.tmp`, opened with .create(true).truncate(true)
      rather than create_new, so it follows symlinks and is not exclusive.
  - test: "Run bashrs over Makefile and scripts/ on a host where it is installed"
    expected: "Clean, or findings triaged"
    why_human: "bashrs is genuinely absent from this host (F-07); CLAUDE.md mandates it over shellcheck. No bashrs check anywhere in Phase 4 may be reported as passing."
---

# Phase 4: APR Artifact and Production Parity — Verification Report

**Phase Goal:** Users deploy the exact trained SetFit model as one verified offline APR whose
shared core implementation produces equivalent results through library, CLI, evaluation, and HTTP
serving surfaces on the mandatory CPU profile.

**Verified:** 2026-08-16T04:36:49Z at `0fb47958f`
**Status:** gaps_found
**Re-verification:** No — initial verification
**Mode:** standard goal-backward (phase has no `mode: mvp`)

## Verdict in one paragraph

**The work that shipped is real, and the phase's own accounting of what did not ship is accurate.**
Every claim I could execute, I executed: 19 Make gates, 3 whole-crate regression runs, 9 trybuild
compile-fail cases, `pv validate`, `pv audit`, and two direct probes of the installed `apr` binary.
All were green. The 24 `aprender-train` failures are the known-red baseline **by NAME**, not by
count. Zero debt markers were introduced. **The phase goal is NOT achieved**, and the reason is a
single measured blocker (F-10) that the phase identified, refused three times to route around, and
escalated to Phase 5 — which is the correct outcome, not a failure of execution. One plan-level
must-have (04-11's per-crate mutation gate) is genuinely unmet. Four documentation-hygiene defects
are recorded as warnings below; one of them (a stale SAFE-02 paragraph in `REQUIREMENTS.md`, which
that file itself says "outlives the phase") is worth fixing before the milestone closes.

---

## Goal Achievement

### Roadmap Success Criteria (the contract — 0/5 fully met)

| # | Success Criterion | Status | Evidence |
|---|-------------------|--------|----------|
| 1 | Save, inspect, load one checksummed F32 `setfit-apr-v1`; malformed/oversized/incomplete/inconsistent/non-finite fail before prediction | ✗ FAILED | LOAD + REFUSAL verified: 8-rung ladder (`artifact.rs:2018-2045`) pinned to the contract's own `rungs:` block by `the_rung_numbering_matches_the_contracts_eight_rung_ladder` (reads it via `include_str!`, not a hand copy); `make setfit-apr-tests` = 86 passed. INSPECT verified: `make setfit-cli-inspect-tests` = 120 passed, renderer pinned to core's `SETFIT_ARTIFACT_DOC_FIELDS` (16 fields). **SAVE by a user: unreachable (F-10).** |
| 2 | Training closes, reloads through the production loader, proves exact state + tolerance-bounded outputs; every consumer rejects anything short of `ArtifactReloadedAndVerified` | ⚠ PARTIAL | `make setfit-codec-tests` = 17 passed over the real `verify_artifact`/`run_verify_policy`/`Tolerance::EXACT` path. No-bypass proven out-of-crate at COMPILE time — `tests/ui/setfit_verified_model_constructed.rs` ran and matched its `.stderr`. **Encoder + head are substituted; never run over a trained model.** |
| 3 | Rust caller and `apr` user complete CPU `train -> APR -> inspect -> eval -> predict`; generic APR commands auto-detect SetFit and reuse the shared core path | ✗ FAILED | Auto-detect + shared-path VERIFIED (tag-only routing; CLI defines no response struct; `SetFitBundle::from_run_parts` is `pub(crate)`). **Lifecycle cannot complete: OPS-01 and OPS-02 both blocked by F-10** — measured, not asserted (below). |
| 4 | Rust, CLI and HTTP receive matching labels, full probabilities, optional logits, margins, token/truncation facts, latency, backend identity, same artifact hash in predictions and readiness | ⚠ PARTIAL | `make setfit-parity` = 20 passed / 1 ignored across library + spawned binary + in-process router on ONE shared request document, with frozen goldens, a SHA-256 manifest and an in-band skewed negative. Readiness/response artifact agreement verified. **Fixture is synthetic; `backend_identity` equation is `pending` because it names a nonexistent symbol.** |
| 5 | Offline executable contracts + the supported CPU build/test feature matrix; an unavailable requested device fails instead of silently falling back or misreporting its backend | ⚠ PARTIAL | `pv validate` rc=0; `make contract-audit-phase4` rc=0 (15/15 bound, 14 implemented, 1 `BIND-004` warning); `make setfit-feature-matrix` rc=0 (4x3, BUILD **and** RUN); `make setfit-api-boundary` rc=0 with an executed MUST-MATCH control. Device gate fails closed. **"in CI" unproven (never ran); "misreporting the backend" unbound.** |

### Plan-level must-have truths (94/95 verified)

| Plan | Truths | Status | Evidence |
|------|--------|--------|----------|
| 04-01 contract | 9/9 | ✓ VERIFIED | `pv validate contracts/setfit-apr-v1.yaml` → `0 error(s), 0 warning(s)`, rc=0. Contract is 1302 lines: storage map with `setfit.head.weight`/`setfit.head.bias` as TENSORS, `doc_bundle_bijection` both directions + the byte-EXACT closure equation, `nullable_path_allowlist` = 4 paths over 5 walked sub-documents, `max_artifact_bytes: 268435456`, 6 probe ids, tolerances CITED from `setfit-encoder-conformance-v1`, backend-identity grammar. Listed in `$(CONTRACTS)` (Makefile:1428) and `PHASE4_CONTRACTS` (Makefile:1468). CLAUDE.md carries the SetFit row documenting the D-09 exception. |
| 04-02 writer | 5/5 | ✓ VERIFIED | `cross_process_writes_produce_the_same_artifact_sha256` spawns the test binary as a CHILD and compares SHA-256 — status read off `Output.status`, never through a pipe. Exactly one custom metadata key asserted against a deliberately-two-key control. Null walk covers all five sub-documents. |
| 04-03 loader | 8/8 | ✓ VERIFIED | One production door; `load_setfit_apr` = `read_setfit_apr_parts_within` + rungs 7-8, so there is literally ONE ladder. `read_setfit_apr_bytes_bounded` refuses an over-cap declared length before touching the reader, `take(cap+1)`, and clamps the reservation to the cap. `VerifiedSetFitModel` non-constructible out-of-crate (trybuild green). Numbering amended (parse door is rungs 2-6, not 1-5) to match the contract — substance identical. |
| 04-04 envelope | 4/4 | ✓ VERIFIED | `classify.rs:645` takes the backend identity FROM the encode invocation that ran (`encode_batch_traced` returns `(pooled, backend)`); there is no other expression in the function that could produce it. Private fields + one validating constructor + `Deserialize` routed through a private wire struct. `make setfit-classify-tests` = 50 passed. |
| 04-05 codec | 4/4 | ✓ VERIFIED | Sealed `AprCodec`; 20-field bijection proven field-by-field; `CodecError::Artifact { source: SetFitArtifactError }` typed, not stringified; codec output loads through the production loader incl. probe replay. 17 tests green. |
| 04-06 `apr setfit train` | 6/6 | ✓ VERIFIED | Probed the INSTALLED binary directly: an unknown config field produced `error: Validation failed: --config cfg.json: unknown field 'shots_per_class', expected one of ...` at rc=5 with no output file created. `atomic_write` + `refuse_existing_output` (temp + `create_new` + rename + cleanup on every error path). Device gate `setfit_train_device_cuda_fails_closed_with_a_nonzero_exit_code` green. |
| 04-07 consumers | 7/7 | ✓ VERIFIED | `apr predict --help` and `apr eval --help` on the built binary show tag-only routing and the `--lock-out` / `--selection-lock` pair. `eval/setfit.rs` calls the whole chain: `reload_verified_run_from_apr` → `create_selection_lock` → `mint_test_token` → `CanonicalTestAccess::grant`, adding no gate of its own. Every artifact byte goes through `setfit_io::read_setfit_apr_file_bounded` (no `fs::read` in predict/inspect/eval). 120 + 34 + 16 + 5 tests green. |
| 04-08 serve | 5/5 | ✓ VERIFIED | `/v1/classify` installed under `#[cfg(feature = "setfit")]` UNCONDITIONALLY on the slot (router.rs:132-141), handler returns 503 for an empty slot. Body IS core's `ClassifyRequestDocument`, response IS core's `ClassifyResponse`; the handler's only compute call is `model.classify(&request)` — **no tokenizer, pooling or head is reimplemented, so CLAUDE.md's documented D-09 exception is respected**. `classifier_artifact_sha256` + `classifier_verified` on readiness. 10 + 358 tests green. |
| 04-09 parity | 6/6 | ✓ VERIFIED | 20 tests green incl. `golden_negative_the_comparator_rejects_a_skewed_probability` (catch_unwind over the real comparator) and `golden_a_single_flipped_byte_fails_the_manifest`. MH6 satisfied by AMENDMENT: the dev-dep threat was measured not to arise; the Makefile records the general rule AND the different, measured apr-cli weakness (13 setfit-named tests run feature-off). |
| 04-10 Make gates | 7/7 | ✓ VERIFIED | 21 named targets, each with `assert_tests_ran` parsing libtest's own passed-count with `awk`, plus the inverse `assert_tests_absent` used only in PAIRS. Every recipe is `cmd > log 2>&1; rc=$?` — never through a pipe. One positional filter per invocation. tier3 wires `setfit-all-tests`, `setfit-parity`, `setfit-serve-smoke`, `setfit-cli-lifecycle`, `setfit-api-boundary`, `setfit-feature-matrix`, `contract-audit-phase4`. |
| 04-11 CI + audit | 4/5 | ✗ 1 FAILED | Patch-file-first verified by commit ORDER (`eb0f8e1c5` patch, then `57f7823ab` apply after the human ruling). CI step uses `set -e`, no pipes, and carries inline exclusion rationale for all three omitted targets plus tripwire comments naming the two standing reds verbatim. Closing audit is honest. **Mutation gate NOT achieved — see gaps.** |
| 04-12 OPS-01 proof | 3/3 | ✓ VERIFIED | 5/5 green. `make setfit-api-boundary` re-run at verification time: 4 must-not-match legs at 0 apr-cli nodes AND a MUST-MATCH control at 1, so the absence check cannot pass on a dead pattern. |
| 04-13 provenance | 5/5 | ✓ VERIFIED | 6-field `ProvenanceRecord::of(&Selection)` read off the run — no `Option`, no float, private constructor. `BUNDLE_SCHEMA_VERSION = 2` with a typed `UnsupportedSchemaVersion`. 33 tests green. |
| 04-14 public doors | 4/4 | ✓ VERIFIED | `SetFitTrainConfig::to_request` (config.rs:648) and `SelectionLock::from_canonical_bytes` with `MAX_SELECTION_LOCK_BYTES = 1_048_576` checked BEFORE parsing. 40 + 14 tests green. |
| 04-15 spawned OPS-02 | 4/4 | ✓ VERIFIED | 2 + 1 spawned tests green. Every verdict read off a reaped `ExitStatus`; binary pinned by `env!("CARGO_BIN_EXE_apr")`, never PATH. Rung 0 (`--version`) proves the mechanism before any later exit code is interpreted. |
| 04-16 fresh-process door | 4/4 | ✓ VERIFIED (2 superseded, both argued in-source) | `reload_verified_run_from_apr` runs `load_setfit_apr` (full 8 rungs incl. 6-probe replay) FIRST, then three coarse-to-fine provenance equalities each naming both values. The two superseded truths (re-enter `verify_artifact`; re-serialization gate) are each replaced by a written argument in the module header — the re-serialization gate would have compared a value with itself. 17 tests green. |
| 04-17 bytes door + credential | 9/9 | ✓ VERIFIED | See the four targeted checks below. |

---

## The four targeted checks the brief called out

### Check 1 — `into_artifact_bytes` and the re-hash proof: **VERIFIED**

`SetFitRun::<ArtifactReloadedAndVerified>::into_artifact_bytes(self) -> Vec<u8>` exists at
`mod.rs:1011` and moves `self.evidence.artifact_bytes.0` — it does not rebuild.
`verify_into_artifact_bytes_are_the_hashed_bytes` (`verify_tests.rs:648`) is not a non-emptiness
check: it takes `artifact_hash()` and `artifact_bytes()` BEFORE the consuming call, then asserts
`hex::encode(Sha256::digest(&bytes)) == recorded_hash`, and additionally re-observes the round-trip
closure from outside the policy. A guard (`verify_tests.rs:605-631`) requires the bytes-door `impl`
block to hold EXACTLY ONE `pub fn` with EXACTLY that signature and requires no THIRD block to exist.
The measured peak-RSS delta (+1,818,624 median against a 1,824,298-byte buffer) is recorded in
`verify.rs:675-693`, replacing the comment that used to justify dropping the buffer.

### Check 2 — `SetFitCredential` seal: **VERIFIED, by rustc**

`pub trait SetFitCredential: sealed::Sealed` with `mod sealed` private to `credential.rs`.
`tests/ui/setfit_external_credential_impl.rs` is a complete out-of-crate program implementing the
trait; its committed `.stderr` pins `E0277` naming the private supertrait. I ran the suite:
`make setfit-ui-tests` = 1 passed / 38.81 s, and the log names **all nine** cases including this one.
`credential_seal_is_a_private_supertrait` counts exactly two implementors.

### Check 3 — the three doors and the ladder order: **VERIFIED**

`create_selection_lock<C: SetFitCredential>` (lock.rs:693, the ONE implementation — the inherent
method at :666 is a one-line forward), `SelectionLock::mint_test_token<C: SetFitCredential>`
(lock.rs:594) and `CanonicalTestAccess::grant<'a, C: SetFitCredential>` (lock.rs:796) all take the
credential and read the artifact hash OFF it — no caller-supplied hash parameter survives.
`reload_verified_run_from_apr` (apr_reload.rs:315-382) calls `load_setfit_apr` on line 325, before
the provenance record is even deserialized; the three identity gates follow, coarse to fine, each
naming both values. `make setfit-reload-tests` = 17 passed, `make setfit-lock-tests` = 36 passed.

### Check 4 — no fabricated evidence: **VERIFIED (with one guard-scope caveat)**

Global construction-site scan of `crates/aprender-train/src/`:

- `PassedEvidence { ` — exactly ONE site, `tune.rs:1203`, inside `validate_evidence`.
- `HeadFittedEvidence { ` — TWO sites: `mod.rs:757` (the legitimate `fit_head` transition) and
  `apr_codec.rs:784`, which is a functional update `{ head: head(&labels), ..evidence }` inside
  `#[cfg(test)] pub(in crate::train::setfit) mod fixture` (gate at `apr_codec.rs:359`).

**No production code fabricates, defaults or reconstructs either type.**
*Caveat (WARNING W-05 below):* the in-repo guard
`credential_validate_evidence_is_still_the_only_passed_evidence_producer` says in its docstring
"Scanned over the whole `setfit` module", but its `SOURCES` array lists **6** of the module's ~25
files and does not include `apr_codec.rs` or `apr_reload.rs`. The property holds today — I verified
it globally — but the guard would not catch a violation in an unscanned file.

---

## The refusal held — and I tried to break the finding, not just read it

The brief asked me to confirm that no test manufactures an artifact and then claims the training
chain works. **It does not.** Three independent pieces of evidence:

1. **`crates/aprender-train/tests/setfit_apr_lifecycle.rs:43-56`** states the refusal explicitly:
   *"It does not hand-build a `SetFitArtifactView` and call core's public `write_setfit_apr` with a
   synthetic APR-capable encoder to manufacture an artifact. That would compile, and it would let
   this file call `.embed(` and `.classify(`."* It then records that the 04-05 in-crate remedy is
   `E0451` at this tier — so the shortcut is compiler-closed as well as declined.
2. **The one APR-capable fixture is `#[cfg(test)]`** and its module header (`apr_codec.rs:361-399`)
   names exactly what is substituted (encoder + head) and what is real (dataset, selection, resolved
   config, stage-one evidence), with the measured refusal transcript quoted verbatim.
3. **The blocker is executable, not documentation.** `round_trip::round_trip_the_phase_three_slice_
   fixture_cannot_carry_an_apr_artifact` keeps it red-on-close. And `setfit_cli_lifecycle`'s rung 4
   asserts `!trained.status.success()` with an `F10_CLOSED` message — the test will turn RED the day
   the blocker is fixed, which is the correct direction for a canary.

I re-derived the root cause independently rather than accepting it: `thresholds.rs:61-62` holds
`CALIBRATED_REGIMES = &["minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7|cells=s16e2b8,s8e1b4"]`
— one entry, exact-equality architecture match.

---

## Requirements Coverage

Every ID declared in a Phase 4 plan's `requirements` frontmatter, cross-referenced against
`.planning/REQUIREMENTS.md`. **No orphans:** the traceability table maps exactly APR-01..05,
OPS-01..06, SAFE-01, SAFE-02 to Phase 4 (13 IDs), and TRN-07 to "Phase 3-4"; all 14 are claimed by
at least one plan.

| Requirement | Claimed by | Status | Verifier's evidence |
|-------------|-----------|--------|---------------------|
| APR-01 | 04-01, 04-02, 04-13 | ✗ BLOCKED | Writer + 20-field view + 6-field provenance all ship and are green. Nothing user-reachable produces a `setfit-apr-v1`. |
| APR-02 | 04-03 | ⚠ PARTIAL | 8-rung refusal ladder verified and contract-pinned. Accept path exercised only over an in-crate fixture. |
| APR-03 | 04-05 | ⚠ PARTIAL | `verify_artifact(AprCodec)` green at `--lib` over a SUBSTITUTED encoder/head. |
| APR-04 | 04-03, 04-16, 04-17 | ⚠ PARTIAL | Negative half compile-proven OUT-OF-CRATE (trybuild, ran green). Positive half fixture-only. |
| APR-05 | 04-03, 04-07, 04-13 | ⚠ PARTIAL | Renderer complete and offline, pinned to `SETFIT_ARTIFACT_DOC_FIELDS`; 120 tests. No user-produced artifact to inspect. |
| OPS-01 | 04-09, 04-12, 04-17 | ✗ NOT MET | Graph half CLOSED and re-verified by me (`make setfit-api-boundary` rc=0, incl. must-match control). Lifecycle half blocked by F-10. |
| OPS-02 | 04-06, 04-07, 04-14, 04-15, 04-17 | ✗ NOT MET | Spawned ladder green through rung 3; rung 4 exits 6; `model.apr` proven absent twice. |
| OPS-03 | 04-07, 04-08, 04-15 | ⚠ PARTIAL | Routing proven EXECUTED (exit 6, not 4 — the distinction is the proof). Call-through unproven past the loader. |
| OPS-04 | 04-04, 04-09 | ⚠ PARTIAL | Envelope parity across 3 readers verified. `backend_identity` binds to a nonexistent symbol. |
| OPS-05 | 04-08 | ⚠ PARTIAL | Readiness/response artifact agreement verified over a fixture; spawned-serve smoke green. |
| OPS-06 | 04-04, 04-06 | ⚠ PARTIAL | CPU-only path delivered, device gate fails closed. "Misreporting the backend" clause unbound. |
| SAFE-01 | 04-01, 04-09, 04-10 | ⚠ PARTIAL | `make setfit-parity` is the detector and is proven able to fail. 18 scoped suites under their own floors. One binding row unresolved. |
| SAFE-02 | 04-10, 04-11 | ⚠ PARTIAL | Local 4x3 matrix rc=0. **CI legs are now APPLIED (not, as REQUIREMENTS.md still says, an unapplied patch) but have never RUN.** |
| TRN-07 | 04-07, 04-11, 04-14, 04-16, 04-17 | ⚠ PARTIAL | Negative half CROSS-PROCESS (04-15, green). Positive half IN-CRATE only. Both labels earned; neither relabelled. |

**Checked-box audit — orchestrator claim CONFIRMED.** `grep -c '^- \[x\] \*\*' .planning/REQUIREMENTS.md`
= **13**, and the 13 are DATA-01..06, TRN-01..06, SAFE-03. **Zero Phase 4 boxes are checked, and that
is correct.** Two plans (04-13, 04-14) still assert `requirements-completed` in their SUMMARY
frontmatter; the 04-11 closing audit explicitly refuses to honour both, by name, in the file that
outlives the phase.

---

## Behavioral Spot-Checks (all executed by the verifier at `0fb47958f`)

| # | Behavior | Command | Result | Status |
|---|----------|---------|--------|--------|
| 1 | Contract is schema-valid | `./target/release/pv validate contracts/setfit-apr-v1.yaml` | rc=0, `0 error(s), 0 warning(s)` | ✓ PASS |
| 2 | Every Phase 4 equation is bound | `make contract-audit-phase4` | rc=0; 15 bound / 14 implemented / 1 `BIND-004` (`backend_identity`) | ✓ PASS |
| 3 | OPS-01 dependency boundary | `make setfit-api-boundary` | rc=0; 4 must-not-match at 0 nodes + MUST-MATCH control at 1 | ✓ PASS |
| 4 | Core artifact suite | `make setfit-apr-tests` | 86 passed (floor 80) | ✓ PASS |
| 5 | Core classify suite | `make setfit-classify-tests` | 50 passed (floor 45) | ✓ PASS |
| 6 | Train bundle / config / evaluate / codec / reload / lock / verify | 7 targets | 33 / 40 / 14 / 17 / 17 / 36 / 18 passed | ✓ PASS |
| 7 | OPS-01 out-of-crate lifecycle | `make setfit-lifecycle-tests` | 5 passed | ✓ PASS |
| 8 | trybuild compile-fail proofs | `make setfit-ui-tests` | 1 passed; log names all 9 `tests/ui/setfit_*.rs` cases | ✓ PASS |
| 9 | CLI train / predict / inspect / eval / io / serve | 6 targets | 15+1 / 34 / 120 / 16 / 5 / 358 passed | ✓ PASS |
| 10 | Serve setfit surface | `make setfit-serve-tests` | 10 passed | ✓ PASS |
| 11 | Three-surface parity | `make setfit-parity` | 20 passed / 1 ignored | ✓ PASS |
| 12 | Spawned serve smoke | `make setfit-serve-smoke` | 1 passed | ✓ PASS |
| 13 | Spawned CLI lifecycle + tooling | `make setfit-cli-lifecycle` | 2 passed, then 1 passed | ✓ PASS |
| 14 | SAFE-02 feature matrix | `make setfit-feature-matrix` | rc=0; `setfit-feature-matrix: PASSED`; core 0→240, train 0→311, cli 13→69, serve 10 | ✓ PASS |
| 15 | `apr setfit` subcommand installed | `./target/debug/apr setfit --help` | rc=0; `train` subcommand present | ✓ PASS |
| 16 | Generic `apr predict` installed with tag routing | `./target/debug/apr predict --help` | rc=0; documents `--input` as the shared request document | ✓ PASS |
| 17 | `apr eval` lock flags installed | `./target/debug/apr eval --help` | rc=0; `--lock-out` and `--selection-lock` both present with split constraints | ✓ PASS |
| 18 | Unknown config field fails before training | `apr setfit train --config <bad>.json … --device cuda` | rc=5, structured `unknown field 'shots_per_class', expected one of …`; no output file created | ✓ PASS |
| 19 | Missing required config field fails before training | same with a partial config | rc=5, `missing field 'encoder_lr'`; no output file | ✓ PASS |

### Regression checks (whole-crate, name-diffed)

| Crate | Command | Result | Status |
|-------|---------|--------|--------|
| aprender-core | `cargo test -p aprender-core --lib --features setfit` | **14423 passed, 0 failed** | ✓ PASS |
| apr-cli | `cargo test -p apr-cli --lib --features setfit` | **6752 passed, 0 failed** | ✓ PASS |
| aprender-train | `cargo test -p aprender-train --lib --features setfit` | 7920 passed, **24 failed** | ✓ PASS — the 24 failing NAMES `diff` **empty** against `03-…/known-red-baseline.md` (21 `gpu::` + 3 `prune::snapshot_tests`). Zero regressions, verified by name and not by count. |

`aprender-serve --lib` was deliberately not run whole-crate: D-04-08-A records 51 pre-existing
failures from ONE overflow at `contract_gate.rs:428:21`, untouched by this phase. The scoped
`--features setfit --lib setfit` leg is green.

---

## Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `Makefile` | `contracts/setfit-apr-v1.yaml` | `$(CONTRACTS)` + `PHASE4_CONTRACTS` | ✓ WIRED (both lists) |
| `artifact.rs` | `AprV2Writer` / `AprV2Reader` | `add_f32_tensor` / `add_tensor(U8)` | ✓ WIRED |
| `artifact.rs` | `contracts/setfit-apr-v1.yaml` | `include_str!` of the contract in the rung-numbering guard | ✓ WIRED (parses the contract's own `rungs:` block) |
| `apr_codec.rs` | `write_setfit_apr` / `read_setfit_apr_parts` | serialize / deserialize delegate format semantics to core | ✓ WIRED |
| `apr_reload.rs` | `load_setfit_apr` | full 8-rung ladder runs FIRST, line 325, before provenance | ✓ WIRED |
| `lock.rs` × 3 doors | `SetFitCredential` | generic parameter; hash read OFF the object | ✓ WIRED |
| `credential.rs` | `sealed::Sealed` | private module, two impls, trybuild-proven | ✓ WIRED |
| `aprender-serve/setfit_handlers.rs` | `VerifiedSetFitModel::classify` | the single compute call in the handler | ✓ WIRED (D-09 respected) |
| `apr-cli/serve/handlers.rs` | `setfit_io::read_setfit_apr_file_bounded` → `load_setfit_apr` | bounded startup read then full ladder | ✓ WIRED |
| `eval/setfit.rs` | `reload_verified_run_from_apr` → `create_selection_lock` → `mint_test_token` → `grant` | the whole TRN-07 chain | ✓ WIRED |
| `predict.rs` / `inspect_setfit.rs` / `eval/setfit.rs` | `setfit_io::read_setfit_apr_file_bounded` | the ONE filesystem door; no `fs::read` in any of the three | ✓ WIRED |
| `.github/workflows/ci.yml` | `Makefile setfit-*` targets | 16 command-identical legs, `set -e`, no pipes | ✓ WIRED (applied, never executed) |
| `contracts/aprender/binding.yaml` | `aprender::setfit::classify::backend_identity` | binding row | ✗ **NOT WIRED — the symbol does not exist** |

---

## Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data? | Status |
|----------|------|--------|-----------|--------|
| `ClassifyResponse.backend` | backend identity | returned by `encode_batch_traced` at `classify.rs:645` | ✓ execution-derived, not constructible by classify | ✓ FLOWING |
| `ClassifyResponse.artifact_sha256` | artifact hash | `self.artifact_sha256()` on the verified model, itself the loader's hash of the input slice | ✓ | ✓ FLOWING |
| `HealthResponse.classifier_artifact_sha256` | readiness hash | `AppState.setfit_model` slot | ✓ tied to the fixture FILE's own hash with `classifier_verified: true` | ✓ FLOWING |
| `ProvenanceRecord` (bundle field 20) | 6 provenance values | `Selection` accessors via `ProvenanceRecord::of` | ✓ read off the run, never restated from config; private constructor | ✓ FLOWING |
| `SelectionLock.chosen_artifact_hash` | artifact identity | `SetFitCredential::artifact_hash()` | ✓ read OFF the credential; `from_candidates` is `pub(super)` | ✓ FLOWING |
| `VerifiedSetFitModel` (any surface) | the model itself | `load_setfit_apr` — its **only** constructor | ⚠ in production the constructor has no admissible input (F-10) | ⚠ HOLLOW at the user tier |

---

## Anti-Patterns

**Debt-marker gate: PASS.** `TODO|FIXME|XXX|TBD|HACK` over every `.rs`/`.yaml`/`.toml`/`.yml` file
changed since the phase base `d66678e7a` (72 files) plus the `Makefile`: **zero matches**, both on
added lines and on whole-file content.

| # | Finding | Severity | Impact |
|---|---------|----------|--------|
| W-01 | `REQUIREMENTS.md` SAFE-02 (lines 301-305) still says *"The CI half exists only as an unapplied proposal … Until it is applied, no CI job runs the apr-cli, aprender-serve, lifecycle or parity legs"*, and the traceability row repeats it. The patch WAS applied at `57f7823ab`, 72 minutes after the audit commit `608aa954e`. | ⚠ WARNING | The file that the phase itself says *"outlives the phase"* now understates what shipped. Two `.github/workflows/ci.yml` edits since the audit are unreflected. |
| W-02 | `ROADMAP.md` Phase 4 still reads **"Plans: 16 plans in 9 waves"** and its plan list contains no `04-17` entry, while the Progress table says **17/17**. | ⚠ WARNING | 04-17 delivered the two doors (`into_artifact_bytes`, `SetFitCredential`) that unblocked 04-12 and 04-16 — it is the least invisible plan in the phase, and it is invisible in the roadmap. |
| W-03 | 04-REVIEW.md's **6 Warning + 5 Info findings are unaddressed and untracked** — absent from `deferred-items.md`, `04-ORCHESTRATOR-NOTES.md`, `REQUIREMENTS.md` and `STATE.md`. Only the 2 Criticals were closed (`0fb47958f`). | ⚠ WARNING | WR-02 verified STILL PRESENT at `eval/setfit.rs:620-636`: `.{name}.tmp` is predictable, `.create(true).truncate(true)` follows symlinks and is not exclusive-create — while the in-repo precedent (`setfit_train::temp_path` + `create_new`) does the opposite. |
| W-04 | 04-13 and 04-14 SUMMARY frontmatter still assert `requirements-completed: [APR-01, APR-05]` and `[OPS-02, TRN-07]`. | ⚠ WARNING | Repudiated by name in the 04-11 audit, so no box was flipped — but a tool that reads frontmatter rather than prose would flip four. |
| W-05 | `credential_validate_evidence_is_still_the_only_passed_evidence_producer` claims in its docstring to scan *"the whole `setfit` module"*; its `SOURCES` array holds 6 of ~25 files and omits `apr_codec.rs` and `apr_reload.rs`. | ⚠ WARNING | Guard scope narrower than its stated claim — the CLAUDE.md rule-7 class. The property currently holds (verified globally by the verifier), so this is a future blind spot, not a live defect. |
| I-01 | `read_setfit_apr_bytes_bounded` pre-reserves `min(declared_len, 256 MiB)`, so a source that lies upward commits a quarter gigabyte before the first read (F-14 item 5, OPEN). | ℹ INFO | The must-have's literal claim ("a hostile **multi-gigabyte** file cannot be allocated") holds; the residual is recorded in `REQUIREMENTS.md` APR-02. |
| I-02 | `setfit` is NOT an `apr-cli` default feature, so `cargo install aprender` yields no SetFit surface at all. | ℹ INFO | Deliberate and argued in `Cargo.toml:112-115` (CR-01 discipline). Reinforces — does not cause — the OPS-02 gap. |
| I-03 | 04-10's `must_haves.artifacts` names `contracts/setfit-apr-v1.yaml` as providing the binding-status flips; the 14 flips actually landed in `contracts/aprender/binding.yaml`. | ℹ INFO | The plan's artifact description is wrong; the OUTCOME is right, and it is the outcome check 7 wanted — `contracts/setfit-apr-v1.yaml` has exactly ONE commit in its whole history (`488e307d5`, 04-01) and has not been touched since. |

### Check 7 in full — contract immutability and binding hygiene

- `git log --oneline -- contracts/setfit-apr-v1.yaml` → **one commit**, `488e307d5` (04-01).
  **The contract has not changed since it was authored, exactly as the brief expected.**
- `contracts/aprender/binding.yaml` was touched twice after 04-01's wiring commit:
  - `af466836e` (04-10) — **status-only**: 14 × `status: pending` → `implemented`, plus explanatory
    comments. Diff inspected line by line; no `equation`, `module_path` or `function` value moved.
  - `fb7904bad` (a post-04-10 fix) — **one semantic correction**: `function: replay_probes` →
    `rung8_replay_probes`, because the old value *"never existed at any point on this branch"*. This
    is a correction toward the shipped symbol, recorded in the row's own `notes`. I re-ran
    `pv audit` after it: rc=0, 15/15 bound.
- `backend_identity` was deliberately left `pending` rather than flipped, *"because flipping it
  would record a claim nothing can check."* Verified: `pmat`-style symbol search finds no
  `aprender::setfit::classify::backend_identity`; the identity comes from
  `ExecutionBackend::identity` in `encoder.rs`.

---

## Gaps Summary

The phase built essentially everything it set out to build, and it built it well: the artifact
schema is normative and unmodified since authoring, the loader is a single eight-rung ladder pinned
to that schema by a test that parses the contract itself, the classify envelope is one versioned
type shared verbatim by three surfaces, the credential is sealed with the compiler as the witness,
and twenty-one Make gates run it all with floors that make a zero-match filter impossible to
mistake for success. Nineteen of those gates, three whole-crate regression runs, nine trybuild
cases and two live binary probes were re-executed by this verification and every one was green.

**What is missing is one thing, and it is upstream of the phase.** `CALIBRATED_REGIMES` admits a
single encoder, that encoder cannot compute two of the six contract-resident probes, and therefore
no user-reachable path produces a `setfit-apr-v1`. Everything that routes through a produced
artifact — OPS-01, OPS-02, the "a user can" tier of APR-01/02/03/04/05, and the positive half of
TRN-07 — is blocked behind it. Three separate executors each found a way to compile past this by
synthesising an APR-capable encoder through the public `SetFitArtifactView` fields, and **each
refused**; I checked all three refusals and they held. The only APR-capable fixture in the tree is
`#[cfg(test)]`, names its substitutions in its own header, and is guarded by a test that goes red
if the slice ever gains the vocabulary coverage it lacks.

**One plan-level must-have is genuinely unmet**: 04-11's per-crate mutation gate produced no score
for any crate. That was reported and stopped rather than silently rescoped, with a measured
projection (≥10 h) — the right call — but the must-have stands open, and it left two REAL production
survivors in `aprender-serve/src/api/setfit_handlers.rs` undiagnosed.

**Four documentation defects should be closed before the milestone audit.** The most consequential
is W-01: `REQUIREMENTS.md` describes the CI half as an unapplied patch when it was applied 72
minutes after that paragraph was written. That file is the milestone's traceability record; a
verifier or auditor reading it next month would understate SAFE-02, not overstate it — the safe
direction, but still wrong. W-03 is the one with residual risk: six code-review Warnings, one of
them a symlink-following non-exclusive temp file in the selection-lock write path, are recorded in
`04-REVIEW.md` and in no tracking artifact at all.

**`gaps_found` is the honest verdict, and it is the right one.** Nothing here should be read as a
failure of execution.

---

_Verified: 2026-08-16T04:36:49Z at `0fb47958f`_
_Verifier: Claude (gsd-verifier) — 19 Make gates, 3 whole-crate regression runs, 9 trybuild cases,
2 contract tools and 5 live binary probes executed; no result in this report is taken from a SUMMARY_

---

## Acknowledged Gaps

_Appended 2026-08-16T23:29:10Z after `/gsd:verify-work 04` at `b3f816c25`, 25 commits after
this report was written. **The verdict above is NOT amended.** `status: gaps_found` stands,
and its measurements stand as taken at `0fb47958f`. This section records what changed
afterwards and which gaps a human explicitly ruled on — see `04-UAT.md` for the session._

**Why this section exists.** Five gap-closure plans (04-18…04-22) landed AFTER this report
and were written to close findings it raised. `roadmap.update-plan-progress` then flipped
Phase 4 to Complete on summary-count parity alone — the same false-completion defect
recorded at 03-10 — so the ROADMAP claimed complete while the verification still read
`gaps_found`. Neither artifact had been reconciled against the other until this UAT.

### Closed since this report

| Gap as reported | Closed by | Evidence re-measured at b3f816c25 |
|---|---|---|
| SC4 / SC5 — `backend_identity` binds to `aprender::setfit::classify::backend_identity`, a symbol that does not exist; equation left `status: pending` (BIND-004) | 04-19 | `make contract-audit-phase4`: 15 equations, 15 bound, 15 implemented, 0 partial, 0 not-implemented, 18 obligations / 270 covered, "No binding gaps found", **zero `BIND-` lines**. Ghost path absent from `contracts/` entirely; row now names `aprender::setfit::encoder` / `ExecutionBackend::identity`. Check was built RED before the flip (six tests in `setfit::classify::backend`). |
| WR-08 / WR-09 — over-cap metadata block handling | 04-18 | `make setfit-apr-tests` 86 passed / 0 failed. Typed refusal naming the limit; four consumers proven to agree over ONE over-cap file via an agreement table. |
| WR-10 — `apr eval --lock-out` ran a full multi-candidate sweep before refusing to overwrite the lock file it was asked to write | 04-20 | `make setfit-cli-eval-tests` 20 passed / 0 failed (floor 13). Ordering proven by two in-process tests failing at DIFFERENT later stages; the spawned three-leg witness was **falsified**, not reported green. |
| D-04-11-A survivors A and B (the two REAL production mutants) | 04-21 | `make setfit-serve-tests` 11 passed / 0 failed. Survivor set re-measured at HEAD (99 mutants: 78 `tests::`/`fixture::`, 21 production). Survivor A corrected — `has_setfit_model` was DELETED at `b47acc4fe`, so that mutant no longer exists; successor surface named. |
| F-07 — `bashrs-lint-makefile` ended in `\|\| echo`, exiting 0 whatever bashrs reported | 04-22 | `make bashrs-lint-makefile` now reads its own status; proven RED on an induced real defect. At HEAD: bashrs rc=2, SC2168 still present at `Makefile:2695`, gate reports "0 counted error-severity finding(s) (baseline 0), 1 discriminated as control-refuted false positive(s)". 35 warnings printed with stated deferral reasons, not suppressed. |

Also re-measured green this session, unchanged in substance from this report: `make
setfit-codec-tests` 17, `make setfit-reload-tests` 17, `make setfit-lifecycle-tests` 5,
`make setfit-parity` 20/20, `pv validate contracts/setfit-apr-v1.yaml` 0 errors / 0
warnings, `make setfit-api-boundary` PASSED with an executed MUST-MATCH control, `make
setfit-feature-matrix` PASSED with RUN legs (core 246, train 311, apr-cli delta 18→79,
serve 11) and two-sided negatives.

One measurement in this report has since moved: the spawned CLI ladder was `2+1` at
`0fb47958f`; it is `3+1` at `b3f816c25` (04-20 added a lifecycle-named case, selected by
the existing filter because `assert_tests_ran` is a floor).

### Ruled on by a human, still open

| Gap | Ruling | Where it goes |
|---|---|---|
| **F-10** — `CALIBRATED_REGIMES` admits only the phase-3 MiniLM slice, whose 97-row vocabulary cannot compute `probe_unicode`; no user-reachable path produces a `setfit-apr-v1`. Root cause of SC1 failed, SC2 partial, SC3 failed, SC4 synthetic fixture. | Accepted, deferred | **Phase 5**, per the existing blocking ROADMAP note. Unblocking requires calibrating on production `all-MiniLM-L6-v2` and adding its fingerprint to `contracts/setfit-train-lifecycle-v1.yaml` — a deliberate `pv diff`-flagged contract edit per Phase 3 D-10(c), never an inline relaxation by a Phase 5 executor. |
| **04-11 must-have 4** — per-crate cargo-mutants baselines and an explicitly computed aggregate adjusted score. 1 of 4 crates attempted, interrupted at ~68 min, no score produced. | Accepted, deferred | **A standalone compute ticket**, not a phase plan. ≥10 h wall clock for 890 mutants is a compute-budget decision CLAUDE.md reserves for the human. Owner unassigned; successor to D-04-11-B. **No mutation score exists for Phase 4 — a later reader must not infer one.** |
| **WR-01** — the APR write path is not race-free. `fs::rename` in `atomic_write` replaces its destination unconditionally, so a file created between check and rename is destroyed without `--force`. 04-20 narrowed the window, did not close it. | Accepted as open | Needs `O_CREAT\|O_EXCL`. Stated in `refuse_existing_output`'s doc and `write_lock`'s comment, so the source does not mislead. |
| **SAFE-02 "in CI"** — the 16 setfit legs at `.github/workflows/ci.yml:378-397` (applied `57f7823ab`) have never executed. No PR opened for `gsd/phase-2-contract-gate`. | Accepted as open | All Phase 4 evidence, including everything in this section, is **macOS/arm64 local**. The arch asymmetry is two-way and only one direction has been measured: `make tier2` is known RED on arm64 with 24 pre-existing clippy errors in arch-gated SIMD that X64-Linux CI never lints. |

### Not re-measured this session

Distinguishing "verified today" from "verified once, at `0fb47958f`":

- The unavailable-device structured refusal (SC5) — carried from this report's own evidence
  (green unit test, two structured CLI refusals at the binary tier).
- The trybuild no-bypass compile proof (SC2) — `make setfit-ui-tests` was not run; the
  command typed was `setfit-ui-test`, which is not a target.

### Surfaced this session, unrouted

`pv` is not on PATH. CLAUDE.md's contract-validation section instructs the reader to type
bare `pv validate …`, which returns command-not-found; the repo's own gates go through
`PV_BIN := cargo run --release -p aprender-contracts-cli --bin pv --` (`Makefile:1737`).
There is no `pv` analogue of `scripts/apr_bin.sh`, so unlike `apr` there is no staleness
proof for the contract CLI. Repo-wide, not a Phase 4 defect; same class as CLAUDE.md
rules 3 and 8.

_Acknowledged: 2026-08-16T23:29:10Z at `b3f816c25` (macOS/arm64) via `/gsd:verify-work 04`.
12 UAT tests, 0 issues. Every gate cited above was executed by the human in-session; no
result in this section is taken from a SUMMARY._
