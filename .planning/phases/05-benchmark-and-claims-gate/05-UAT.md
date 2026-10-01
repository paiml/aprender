---
status: complete
phase: 05-benchmark-and-claims-gate
source: [05-VERIFICATION.md]
started: "2026-09-12T02:12:00Z"
updated: "2026-09-13T00:00:00Z"
---

## Current Test

[testing complete]

## Tests

### 1. Disposition of D-ITEM-05-17-A — the build-graph-dependent row seal
expected: An explicit decision — (a) pin and document the order-preserving seal, or (b) schedule the
  re-seal plan — plus correcting `bench_row.rs:39-44`, which claims key-sorted independence that is
  false in the binary that sealed the evidence.
evidence: `cargo tree -p aprender --features inference -e features -i serde_json` shows
  `preserve_order` via `pmcp v2.19.3`; the same query for `-p aprender-train --features setfit`
  shows none. Recomputed outside the codebase in Python from the JSON alone: key-sorted →
  `fafda648…`, file order → `1c54f3b4…` (what the envelope claims).
why_human: Architectural. The fix re-seals 40 rows, 40 selection manifests and the run manifest, or
  else commits the project to a feature pin. Auditability is NOT broken — the verifier reproduced
  the committed digest from committed bytes with no aprender code — so this is advisory, not a gap.
result: pass
reported: "pass"
note: Accepted by the user 2026-09-12. Recorded state at acceptance — no code or doc change was
  made in this run: `bench_row.rs:39-44` still carries the key-sorted-independence claim that
  measurement contradicts, and D-ITEM-05-17-A remains an open deferred item. Accepting is a valid
  call (auditability was proven intact outside the codebase), but the divergence is not yet closed.

### 2. Rule on the eleven judgment-tier prohibitions across 05-15/16/17
expected: Each prohibition explicitly accepted or turned into a fix. Two need a ruling:
  (i) WR-03 — the `verified:` line's trailing "rather than read off the rows" is literally false of
  the third source it lists (the confusion matrix IS a row field);
  (ii) WR-04 — `setfit-benchmark-claims-v1.yaml:628` asserts three residual statements agree "WORD
  FOR WORD"; they do not (four items vs three, closure clause dropped, paraphrases).
  Read `report.md:1-17`, `bench_gate.rs:64-98` and `setfit-benchmark-claims-v1.yaml:628` side by side.
why_human: Judgment-tier prohibitions in interactive mode; neither is decidable by grep. Neither
  claims MORE than the gate enforces — which is why these are rulings, not gaps — but both are
  claim-language defects in a phase whose entire subject is claim language.
result: pass
reported: "pass. I'm not sure what are the tests and if is OK to review and clean up the rule,
  especially, after the deployment of the servers were successful."
note: Accepted 2026-09-12, with the user asking whether cleanup is safe post-deployment. Verified
  and answered: the deployed servers (aprender-mcp-chronos / -lambda / -forecast) depend only on
  `aprender-forecast`; grep confirms NOTHING in the deploy path references `bench_gate` or
  `setfit_bench`. WR-03/WR-04 are wording-only defects in the phase-5 report and contract, both in
  the UNDER-claiming direction, so cleanup is non-urgent and carries no deployment risk. Deferred,
  not fixed, in this run.

### 3. Fix or consciously accept CR-01 (doctor-script containment guard)
expected: The guard refuses `<repo>/benchmarks/tweeteval-stance` from EVERY cwd, proven by a
  must-match/must-not-match case table run from at least two working directories (CLAUDE.md
  Verification Discipline rules 4 and 7). Suggested fix: anchor on
  `os.path.dirname(os.path.abspath(__file__))`, compare with `os.path.commonpath` rather than
  `str.startswith`, and correct the comment claiming the guard holds "whatever a caller passes".
evidence: CONFIRMED empirically — from `crates/`, `os.path.realpath("benchmarks")` resolves to
  `<repo>/crates/benchmarks`, so the guard cannot fire for the real tree and the script would
  `shutil.move` the committed lock record out of it.
why_human: Destructive-hazard triage on a fixture tool. The shipped probe path was proven safe
  (`cd $REPO_ROOT` + `mktemp`; tracked trees clean after two runs), so this is the unsafe
  hand-invocation path only. Fix-now vs accept-the-footgun is a maintainer call.
result: pass
reported: "Please fix the containment guard if they are irrelevant anymore after the work that we
  did."
resolution: FIXED 2026-09-12 in `480a06fe3`. First confirmed the script is NOT irrelevant — the
  door probe invokes it three times (escape / selection-hash-zeros / f-avg-to-0-99), so the guard
  had to be repaired rather than removed. Guard now anchors on
  `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` and compares with
  `os.path.commonpath`, replacing the cwd-relative `str.startswith` form. Ships a `--self-test`
  case table (CLAUDE.md rule 7) exercised from three working directories (rule 4).
  PROVEN RED-TURNING, not assumed: 8 of 24 cases fail against the old predicate — 6 under-refusals
  (committed tree not refused from `crates/` and `/tmp`) and 2 over-refusals (prefix siblings
  refused from the repo root). New guard 24/24 from 3 cwds. Regression: door probe rc=0 against
  `apr 0.63.0 (572fb2fec)` with control first and all four refusals; `benchmarks/` untouched.

### 4. Run `/gsd-secure-phase 05` — no 05-SECURITY.md exists
expected: A threat-mitigation record covering at minimum: the path-traversal class 05-15 closed
  (`resolve_committed_evidence_path`); the evidence-substitution class 05-16 closed (transplanted /
  deleted selection manifest); the TOCTOU residual the resolver discloses at `bench_gate.rs:844-849`
  (containment checked, then the file opened); and the `MAX_CROSS_CHECK_ROWS` expansion cap bounding
  the confusion-matrix recomputation.
why_human: `.planning/config.json` sets `security_enforcement: true`, `security_asvs_level: 1`,
  `security_block_on: high`, and all three gap plans declare `asvs_level: 1, block_on: high`. A
  security gate that never ran cannot be discharged by reading code — and this phase's subject
  matter IS path traversal and evidence tampering.
result: pass
reported: "pass"
resolution: DISCHARGED 2026-09-13 across two runs. `/gsd-secure-phase 05` (State B) built the
  record from all 17 plan `<threat_model>` blocks and verified it against code, not prose:
  115 threats, 115 closed, threats_open 0, committed `71948eccc`. A State A re-audit
  (`419cc2223`) carried the verdicts forward on a PROVEN-empty diff (0 bytes across the 15
  mitigation-bearing files and the committed evidence set) rather than re-asserting them.
  All four named expectations are covered: path traversal via `resolve_committed_evidence_path`
  (T-05-15-01, 13 path cases x 2 kinds); evidence substitution (T-05-16-02/03, transplanted and
  deleted manifests); the `MAX_CROSS_CHECK_ROWS` cap (T-05-17-07).
  THE FOURTH WAS INITIALLY MISSING AND THIS TEST IS WHAT CAUGHT IT: the TOCTOU residual at
  `bench_gate.rs:843-848` was absent from the first record, because it is disclosed in the
  source module doc but appears in NONE of the 17 plan threat models — a register derived from
  plan-time models structurally cannot contain it. Added as AR-11 (`46bb8b102`) with that
  limitation stated in the file. Test 4 did real work rather than rubber-stamping a run.

### 5. Re-run the shipped end-to-end door proof against a HEAD-built binary
expected: rc=0, positive control passing FIRST, then all four refusals (E, G, F, D).
result: PASSED — discharged by the orchestrator 2026-09-12, not deferred to the user.
  `cargo build --release --bin apr --features setfit` rc=0; `apr --version` reports
  `apr 0.63.0 (55cf37600)`, byte-matching HEAD `55cf37600` (binary pin proven, CLAUDE.md rule 3);
  `make setfit-bench-door-probe` rc=0 with CONTROL first then E/G/F/D all rc=5, F and D each
  explicitly "NOT at the row digest". 05-15's `verification: backstop` truth is now fully
  discharged. RESIDUAL (not blocking): `setfit-bench-door-probe` was still absent from `.PHONY`
  (WR-07) — a one-line Makefile fix left unmade because the phase is pending and it was not in scope.
  CLOSED 2026-09-13 in `b5bd11a65`: `/gsd-validate-phase 05` found the same orphan independently
  and fixed it properly — both targets added to `.PHONY`, and a new `setfit-bench-door-probe-build`
  target that BUILDS the release `apr` it needs, wired into tier4 unprefixed. The probe was a
  prerequisite of nothing, so the door-level proof of five threats ran in no gate at all; it now
  runs in one, re-measured rc=0 at HEAD with the control passing before all four refusals.

## Summary

total: 5
passed: 5
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

None. Verification scored 5/5 must-haves; all three previously graded gaps (EVAL-04 FAILED,
EVAL-02 PARTIAL, EVAL-01 advisory) are closed under adversarial fixtures built independently of the
executors' own probe. The four pending items are decisions and one un-run gate, not unmet must-haves.
