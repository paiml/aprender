# PMAT-4787 (sub-ticket of PMAT-4741) / IDLE-1130 receipt: coverage-report-scope callers switched, .py deleted

Code commit 9ee4de6d68 on batch/0702/py-port-coverage-report-scope (base eb60aef156).

Measured on the x86 build host (CARGO_TARGET_DIR private):
- cargo test -p aprender-ci-tools: 11 passed, 0 failed
- cargo fmt --all -- --check: rc 0
- scripts/check_coverage_report_scoped.sh --self-test: 10/10; scan of Makefile scripts/*.sh workflows: rc 0
- mutant: SCOPED_RE restored to the old `coverage_report_scope\.py|...` form -> self-test RED (stale-py, derived-bin)
- scripts/tests/ci_tools_py_parity_test.sh (old .py from blob 106561a2 vs new bin): 54/54 identical,
  incl. bad inputs: unknown exclude, unknown flag, bare --exclude, positional, cargo failure
- live scope with --exclude aprender-gpu: 82 `-p` entries, aprender-gpu absent
- bashrs lint warnings head == base (10/10 each file)
- PF: merged origin/main (d5c0b1234f, not pushed): all of the above green

Known red NOT introduced here: FALSIFY-MONO-011 (monorepo_invariants) on base eb60aef156, held by aprender-a7.

## Planted contrary question (reviewers MUST answer, with file:line)
Claim: "After this diff the guard still treats `$(python3 scripts/coverage_report_scope.py)` on an
`llvm-cov report` line as SCOPED, so a stale caller of the deleted script would pass the scan."
Is this claim TRUE or FALSE? Cite the regex line and the self-test case that decides it.

Pre-push (x86 build host, 9ee4de6d68): cargo deny check advisories ok. cargo test -p aprender-contracts --lib: 2288 pass,
1 fail (lint::tests::lint_passes_on_real_contracts, gates reverse-coverage + shapes); the SAME test fails the
same two gates on base eb60aef156, so it is not introduced here (this diff touches no contracts/).

Quorum (PMAT-4787, judged head 9ee4de6d68): AGREED 3/3 PASS. claude-sonnet-5-5 (x2), claude-haiku-4-5,
measured == declared; degraded: same-family (agy policy: not a tier-1 diff). Planted claim answered FALSE
by all three lanes; cited scripts/check_coverage_report_scoped.sh:19, :65 and
scripts/tests/ci_tools_py_parity_test.sh:47. Artifact sha256 00fbe8a8f2ae7809… kept out of tree (local paths).
