---
status: complete
phase: 04-apr-artifact-and-production-parity
tested_at_commit: b3f816c25
host: macOS/arm64 (Darwin 25.6.0)
source: [04-01-SUMMARY.md, 04-02-SUMMARY.md, 04-03-SUMMARY.md, 04-04-SUMMARY.md, 04-05-SUMMARY.md, 04-06-SUMMARY.md, 04-07-SUMMARY.md, 04-08-SUMMARY.md, 04-09-SUMMARY.md, 04-10-SUMMARY.md, 04-11-SUMMARY.md, 04-12-SUMMARY.md, 04-13-SUMMARY.md, 04-14-SUMMARY.md, 04-15-SUMMARY.md, 04-16-SUMMARY.md, 04-17-SUMMARY.md, 04-18-SUMMARY.md, 04-19-SUMMARY.md, 04-20-SUMMARY.md, 04-21-SUMMARY.md, 04-22-SUMMARY.md]
started: 2026-08-16T21:47:26Z
updated: 2026-08-16T23:29:10Z
note: >-
  This session exists to reconcile 04-VERIFICATION.md (status gaps_found, written at
  0fb47958f, 0/5 success criteria fully met) with the five gap-closure plans that landed
  AFTER it (04-18..04-22, closing WR-08, WR-09, WR-10, SC4-backend_identity,
  SC5-misreporting-backend, BIND-004, D-04-11-A survivors A/B, F-07). The verification
  has not been re-run since a523db6b3, 25 commits back. Tests 2-6 restate the five
  ROADMAP success criteria; tests 7-12 test the gap closures directly.
---

## Current Test

[testing complete]

## Tests

### 1. Cold Start — pin the binary
expected: `. scripts/apr_bin.sh` exports $APR and proves the binary was built from HEAD; `"$APR" --version` prints a SHA matching the current commit. A stale or missing binary returns non-zero instead of silently resolving to an older `apr` on PATH.
result: pass
evidence: "resolved /Users/guy/Development/machine-learning/aprender/target/debug/apr; `apr 0.63.0 (b3f816c25)` — SHA equals HEAD"

### 2. SC1 — save, inspect, and load one checksummed setfit-apr-v1
expected: Split verdict, and the split is the phase's headline gap. The INSPECT and LOAD halves work — a well-formed `setfit-apr-v1` is readable offline with no Python, network, or sidecars, and malformed / oversized / incomplete / inconsistent / non-finite artifacts are refused before any prediction. The SAVE half is NOT user-reachable: `CALIBRATED_REGIMES` (thresholds.rs:61-62) holds exactly one entry matched on exact architecture equality, so only the phase-3 MiniLM fixture slice is trainable, and its 97-row vocabulary cannot compute `probe_unicode` (canonical id 5915). Finding F-10, explicitly deferred to Phase 5 by ROADMAP note.
result: pass
note: "Deferral of the SAVE half to Phase 5 accepted by the human. The inspect/load halves stand as verified."

### 3. SC2 — close, reload through the production loader, prove state
expected: `make setfit-codec-tests` and `make setfit-reload-tests` are green. `verify_artifact(AprCodec)` drives the whole trusted-verify policy end to end — close, serialize, hash, reload through `read_setfit_apr_parts`, byte-canonical closure, `Tolerance::EXACT` — and the no-bypass rule is compile-proven from OUTSIDE the crate by a real trybuild run (`make setfit-ui-tests`). Caveat you are being asked to accept: the policy has never run over a model the shipped train path produced; the encoder and head are substituted by `fixture::apr_capable_run`, which is `#[cfg(test)]`. Same F-10 root cause.
result: pass
evidence: "`make setfit-codec-tests` 17 passed / 0 failed (4.01s); `make setfit-reload-tests` 17 passed / 0 failed (4.86s), both at b3f816c25 on macOS/arm64"
note: "The trybuild compile-proof leg was NOT run — the user typed `setfit-ui-test`; the target is `setfit-ui-tests` (Makefile:2444). SC2's mechanism is carried by the two green suites; the no-bypass compile proof remains as last measured by the verifier."

### 4. SC3 — the CPU train -> APR -> inspect -> eval -> predict lifecycle
expected: `make setfit-lifecycle-tests` passes 5/5 and `make setfit-cli-lifecycle` passes its spawned ladder — but those tests ASSERT THE REFUSAL. The save-as-setfit-apr-v1 rung returns a typed `ProbeComputation{probe:"probe_unicode"}`, rung 4 (`apr setfit train`) exits 6 (ModelLoadFailed) and `model.apr` is proven absent afterwards. What IS verified: the auto-detect half (generic `apr` commands recognise SetFit) and the shared-core-path half (tokenizer, pooling and model path are reused, not reimplemented). OPS-01 and OPS-02 are not met end to end.
result: pass
evidence: "`make setfit-lifecycle-tests` 5 passed / 0 failed (5.37s); `make setfit-cli-lifecycle` spawned lifecycle ladder 3 passed / 0 failed (0.57s) + spawned tooling ladder 1 passed / 0 failed (0.14s), at b3f816c25"
note: "The spawned ladder is one leg richer than 04-VERIFICATION.md recorded (it measured 2+1 at 0fb47958f; now 3+1). Human accepted the refusal-shaped lifecycle — OPS-01/OPS-02 remain not-met pending F-10 closure in Phase 5."

### 5. SC4 — Rust, CLI and HTTP callers agree
expected: `make setfit-parity` runs 20/20 green across three real legs — a library call, a spawned `CARGO_BIN_EXE_apr`, and an in-process realizar router oneshot — over one `ClassifyRequestDocument`, with frozen goldens, a SHA-256 manifest, and an in-band skewed negative built through the public validating constructor. Labels, probabilities, logits, margins, token/truncation facts, latency, backend identity and artifact hash agree, and readiness reports the same hash the responses do. Caveat: the fixture is SYNTHETIC because F-10 means no user-producible artifact exists to run it over.
result: pass
evidence: "`make setfit-parity` 20 passed / 0 failed / 1 ignored (2.79s) at b3f816c25 — matches the 20/20 the verifier measured at 0fb47958f. The 1 ignored leg is the spawned-server smoke test, which lives in the separate `setfit-serve-smoke` target."
note: "Synthetic-fixture caveat accepted by the human; re-running parity over a produced artifact stays blocked on F-10."

### 6. SC5 — offline contracts, CPU feature matrix, honest device refusal
expected: All four gates green by execution: `pv validate contracts/setfit-apr-v1.yaml` rc=0 (0 errors, 0 warnings); `make contract-audit-phase4` rc=0 with 15/15 equations bound; `make setfit-feature-matrix` rc=0 over 4 crates x 3 CPU profiles with both BUILD and RUN legs plus two-sided graph negatives; `make setfit-api-boundary` rc=0 with an executed must-match control. An explicitly requested unavailable device fails closed with a structured refusal rather than silently falling back. Open clause: the 16 setfit legs applied to `.github/workflows/ci.yml:378-397` have never RUN on this branch, so SAFE-02's "in CI" wording is unproven.
result: pass
evidence: |
  All four gates re-measured by the human at b3f816c25 on macOS/arm64:
    - `pv validate contracts/setfit-apr-v1.yaml` -> "0 error(s), 0 warning(s) / Contract is valid."
    - `make contract-audit-phase4` -> 15 equations, 15 bound, 15 implemented, 0 partial,
      0 not-implemented, 18 obligations / 270 covered, "No binding gaps found", zero BIND- lines
    - `make setfit-api-boundary` -> PASSED; 4 must-not-match legs (aprender-core 0/121,
      core+setfit 0/222, aprender-train 0/633, train+setfit 0/707) plus an EXECUTED
      MUST-MATCH control (apr-cli's own tree -> 1 apr-cli node)
    - `make setfit-feature-matrix` -> PASSED with RUN legs, not just checks: aprender-core
      246 passed, aprender-train 311 passed, apr-cli delta 18 off -> 79 on (+61),
      aprender-serve 11 passed; two-sided negatives on both axes (feature OFF selects ZERO /
      ON selects many; default aprender-serve build has NO tokenizers node, with setfit it does)
note: |
  Two clauses NOT re-measured in this session, carried over from 04-VERIFICATION.md rather
  than re-proven: (a) the unavailable-device structured refusal (verifier observed a green
  unit test plus two structured CLI refusals at the binary tier); (b) SAFE-02's "in CI"
  clause — see test 12.

  Separate observation, repo-wide and NOT a Phase 4 defect: `pv` is not on PATH. CLAUDE.md's
  contract-validation section instructs the reader to type bare `pv validate ...`, which
  returns command-not-found; the repo's own gates go through
  `PV_BIN := cargo run --release -p aprender-contracts-cli --bin pv --` (Makefile:1737).
  There is no `pv` analogue of `scripts/apr_bin.sh`. Same class as CLAUDE.md rules 3 and 8.

### 7. WR-08 / WR-09 — an over-cap metadata block is a typed refusal, not a fail-open (04-18)
expected: Feeding a metadata block larger than the cap produces a typed refusal that names the limit, rather than being silently truncated, accepted, or read past. The four consumers of that path agree with each other over ONE over-cap file — proven by an agreement table rather than four independent assertions that could each be right about a different file.
result: pass
evidence: "`make setfit-apr-tests` 86 passed / 0 failed (0.85s) at b3f816c25 — the same writer-half count 04-VERIFICATION.md recorded, now including 04-18's over-cap refusal ladder and four-consumer agreement table."

### 8. WR-10 — `apr eval --lock-out` refuses BEFORE doing the work (04-20)
expected: Pointing `apr eval --lock-out` at a path whose lock file already exists refuses immediately. It no longer first reads the corpus, walks the eight-rung load ladder, and classifies the entire validation split for every candidate before declining to overwrite the file it was asked to write. The ordering is proven by two in-process tests that fail at DIFFERENT later stages (so a single shared failure cannot fake it), and the spawned three-leg witness was falsified rather than reported green.
result: pass
evidence: "`make setfit-cli-eval-tests` 20 passed / 0 failed (0.01s) at b3f816c25 — the same count 04-20 recorded against its floor of 13."
note: "WR-01 accepted as OPEN by the human: `fs::rename` in `atomic_write` still replaces its destination unconditionally, so a file created between check and rename is destroyed without `--force`. Closing it needs `O_CREAT|O_EXCL`; the limitation is stated in `refuse_existing_output`'s doc and `write_lock`'s comment."

### 9. backend_identity binding resolves to a real symbol (04-19)
expected: `make contract-audit-phase4` reports 15/15 bound and implemented with ZERO `BIND-` lines. The ghost path `aprender::setfit::classify::backend_identity` — which never existed — is gone from `contracts/` entirely, replaced by `aprender::setfit::encoder` / `ExecutionBackend::identity`. This closes the one clause SC4 and SC5 shared. Note recorded on the registry itself: `pv verify-bindings` is blind in this layout, with the arithmetic that proves it.
result: pass
evidence: "Measured at test 6 by the human's own `make contract-audit-phase4` run at b3f816c25: 15 equations / 15 bound / 15 implemented / 0 partial / 0 not-implemented, 'No binding gaps found', zero BIND- lines. 04-VERIFICATION.md's SC4 and SC5 both cited this BIND-004 as open; it is closed."

### 10. F-07 — the bashrs Makefile gate can actually fail (04-22)
expected: `make bashrs-lint-makefile` reads its own exit status. Before this plan it ended in `|| echo`, so it exited 0 whatever bashrs reported and had never been able to surface the 34 findings it really produces — a target that printed failures and reported success. It was proven RED on an induced real defect. It now exits 0 at HEAD while bashrs itself exits 2 and SC2168 is still present, because the two known findings are discriminated as false positives by controls rather than suppressed.
result: pass
evidence: |
  `make bashrs-lint-makefile` at b3f816c25:
    "✗ 2695:22-28 [error] SC2168: 'local' is only valid in functions"
    "Summary: 1 error(s), 35 warning(s), 0 info(s)"
    "bashrs-lint-makefile: 0 counted error-severity finding(s) (baseline 0),
     1 discriminated as control-refuted false positive(s), bashrs rc=2"
  The gate reports the real tool status (rc=2) and its own verdict separately, and prints
  the warning breakdown (20 MAKE012, 11 MAKE010, 2 MAKE003, 1 MAKE018, 1 MAKE001) with
  stated deferral reasons from D-04-22-A rather than swallowing them. Full report at
  target/bashrs-makefile.log.
note: "Line 2695 matches the dev-setup line the wave-12 tracking commit independently confirmed as byte-identical after moving from 2327. The repo-wide 59-script bashrs backlog remains explicitly NOT RUN, triaged by fix-shape with a named owner in D-04-22-A."

### 11. Mutation survivors: the two REAL production survivors (04-21)
expected: The `setfit_handlers.rs` survivor set is re-measured at HEAD rather than inherited — 99 mutants, 78 `tests::`/`fixture::`, 21 production. D-04-11-A's Survivor A is corrected: `has_setfit_model` was DELETED at b47acc4fe, so that mutant no longer exists, and its successor surface is named. Both measured batch-bound survivors are killed by an at-the-bound acceptance test. Still open and NOT claimed: 04-11 must-have 4 — the per-crate cargo-mutants runs across all four crates and the aggregate adjusted score were never produced (>= 10 h projected; the single attempted crate was interrupted at ~68 min).
result: pass
evidence: "`make setfit-serve-tests` 11 passed / 0 failed (0.46s) at b3f816c25 — the scoped aprender-serve setfit suite carrying 04-21's at-the-bound acceptance test that kills both measured batch-bound survivors."
note: |
  HUMAN RULING on 04-11 must-have 4: DEFER TO A STANDALONE COMPUTE TICKET.

  The >= 10 h cargo-mutants run across all four crates is a compute-budget decision
  CLAUDE.md reserves for the human, not phase work. It leaves the phase sequence and
  becomes its own scheduled job (nightly/background) producing per-crate baselines for
  aprender-core, aprender-train, aprender-serve and apr-cli plus an explicitly computed
  aggregate adjusted score.

  Phase 4 therefore closes with must-have 4 recorded as DEFERRED — not as met. Any later
  reader must not infer a mutation score exists for Phase 4; none was ever produced.
  Owner: unassigned. Successor to D-04-11-B.

### 12. CI has never executed the setfit legs
expected: `.github/workflows/ci.yml:378-397` contains 16 setfit legs (12 `cargo test`, 6 `cargo check`) applied at commit 57f7823ab. No CI run on `gsd/phase-2-contract-gate` has executed them, and no PR has been opened for this branch. Every setfit gate you have accepted above was measured locally on macOS/arm64 only.
result: pass
note: |
  Acknowledged by the human as an OPEN CLAUSE, not closed. Phase 4 closes with SAFE-02's
  "in CI" wording explicitly unproven.

  Every gate passed in this UAT session is macOS/arm64 LOCAL evidence at b3f816c25. The
  16 setfit legs (12 cargo test, 6 cargo check, applied at 57f7823ab) have never executed;
  no PR has been opened for gsd/phase-2-contract-gate.

  The arch asymmetry runs both ways and only one direction has ever been measured:
  `make tier2` is known RED on arm64 with 24 pre-existing clippy errors in arch-gated
  SIMD that CI, being X64-Linux-only, never lints.

## Summary

total: 12
passed: 12
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none — no test produced a failing observation]

## Accepted Open Items

These are NOT gaps requiring fix plans. Each was presented to the human during this
session and explicitly ruled on. They are recorded here so no later reader mistakes
Phase 4's closure for their closure.

- item: "F-10 — no user-reachable path produces a setfit-apr-v1"
  ruling: accepted, deferred to Phase 5
  scope: >-
    CALIBRATED_REGIMES (thresholds.rs:61-62) holds one entry matched on exact architecture
    equality, so only the phase-3 MiniLM fixture slice is trainable and its 97-row
    vocabulary cannot compute probe_unicode (canonical id 5915). Consequence: SC1's SAVE
    half is unreachable, SC2's policy has only ever run over a substituted encoder/head,
    SC3's lifecycle stops at rung 4 (exit 6), and SC4's parity fixture is synthetic.
  unblocking: >-
    Calibrate on the production all-MiniLM-L6-v2 and add its fingerprint to
    contracts/setfit-train-lifecycle-v1.yaml — a deliberate, `pv diff`-flagged contract
    edit per Phase 3 D-10(c), NEVER an inline relaxation by a Phase 5 executor.
  tests: [2, 3, 4, 5]

- item: "04-11 must-have 4 — per-crate mutation baselines and aggregate adjusted score"
  ruling: accepted, deferred to a STANDALONE COMPUTE TICKET (human decision this session)
  scope: >-
    1 of 4 crates attempted (aprender-serve), interrupted at ~68 min without finishing its
    101 mutants. No score produced, per-crate or aggregate. D-04-11-B projects >= 10 h wall
    clock for all 890 mutants.
  unblocking: >-
    A scheduled standalone job (nightly/background) producing per-crate baselines for
    aprender-core, aprender-train, aprender-serve and apr-cli plus an explicitly computed
    aggregate adjusted score. Compute budget is a human decision per CLAUDE.md, which is
    why this leaves the phase sequence rather than becoming a 04-23 plan.
  owner: unassigned
  tests: [11]

- item: "WR-01 — the APR write path is not race-free"
  ruling: accepted as open
  scope: >-
    `fs::rename` inside `atomic_write` replaces its destination unconditionally, so a file
    created between the check and the rename is still destroyed without `--force`. 04-20's
    early refusal narrows the window; it does not close it.
  unblocking: "Take the destination with O_CREAT|O_EXCL."
  recorded_in_source: "refuse_existing_output's doc and write_lock's comment both say so."
  tests: [8]

- item: "SAFE-02 'in CI' — the 16 setfit legs have never run"
  ruling: accepted as open
  scope: >-
    .github/workflows/ci.yml:378-397, applied at 57f7823ab, never executed. No PR opened
    for gsd/phase-2-contract-gate. All Phase 4 evidence is macOS/arm64 local. The arch
    asymmetry is two-way and only one direction has been measured — `make tier2` is known
    RED on arm64 with 24 pre-existing clippy errors in arch-gated SIMD that X64-Linux CI
    never lints.
  tests: [12]

- item: "Repo-wide bashrs backlog — 59 scripts, NOT RUN"
  ruling: noted, out of Phase 4 scope
  scope: "Triaged by fix-shape with a named owner in D-04-22-A."
  tests: [10]

- item: "`pv` is not on PATH (repo-wide docs/tooling inconsistency, NOT a Phase 4 defect)"
  ruling: surfaced this session, unrouted
  scope: >-
    CLAUDE.md's contract-validation section instructs the reader to type bare
    `pv validate ...`, which returns command-not-found on this box. The repo's own gates go
    through `PV_BIN := cargo run --release -p aprender-contracts-cli --bin pv --`
    (Makefile:1737). There is no `pv` analogue of scripts/apr_bin.sh, so there is also no
    staleness proof for the contract CLI the way there is for `apr`. Same class as
    CLAUDE.md rules 3 and 8.
  tests: [6]

## Not Re-Measured This Session

Carried over from 04-VERIFICATION.md rather than re-proven — listed so the distinction
between "verified today" and "verified once, at 0fb47958f" stays legible.

- The unavailable-device structured refusal (SC5). Verifier observed a green unit test plus
  two structured CLI refusals at the binary tier.
- The trybuild no-bypass compile proof (SC2). `make setfit-ui-tests` was not run — the
  command typed was `setfit-ui-test`, which is not a target.
