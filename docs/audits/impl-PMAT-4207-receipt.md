# PMAT-4207 implementation receipt: the lint tests read the shared /tmp as their project root

Issue: paiml/aprender#4207 (0.70.0). The report (aprender-48) was that `lint::tests::lint_empty_dir` fails on pristine
main `aa7c6ef03` on lambda while CI is green. The lead (aprender-98, relayed by the cop) was a stray
`/tmp/scripts/contract_duplicate_stem_baseline.txt`.

## Cause
`run_lint` takes the contract dir's PARENT as the project root (`lint/mod.rs:469/523/565`). The duplicate-stem gate
(PV-DUP-001/002) reads `<root>/scripts/contract_duplicate_stem_baseline.txt` from that root. A test that lints a bare
`tempfile::tempdir()` therefore gets `/tmp` as its root. On lambda someone had copied the repo's `scripts/` to
`/tmp/scripts/`. That made the real baseline apply to an empty corpus, and every entry in it was stale, which is
PV-DUP-002, an Error. So `report.passed` was false. CI runners have no `/tmp/scripts/`, which is why CI is green.

## Red → green, both measured in a PRIVATE `TMPDIR` (the shared `/tmp/scripts/` was not touched)
Setup: `TMPDIR=<scratch>/tmp4207`, with `scripts/contract_duplicate_stem_baseline.txt` copied into `$TMPDIR/scripts/`.

| tree | `$TMPDIR/scripts/` | `aprender-contracts --lib lint::` | `aprender-contracts-cli --tests` |
|---|---|---|---|
| main `aa7c6ef03` | absent | 258 passed, 0 failed | 365 passed, 0 failed |
| main `aa7c6ef03` | **planted** | **1 failed**: `lint_empty_dir` | **3 failed**: `pvl_zero_contracts::control_one_contract_is_reported_not_refused`, `ont6_lint_verdict::{control_default_arming_passes_and_prints_the_armed_meet, json_report_carries_the_lattice}` |
| main `aa7c6ef03` | removed again | 258 / 0 | not re-run: that run was the same absent-file state as row 1 |
| this branch | **planted** | 258 passed, 0 failed | 365 passed, 0 failed |

The planted file is the RED; the fix turns every one of those four tests green with the file still present.

## Fix (tests only; the lint's root rule is unchanged)
Each corpus is now `<private tempdir>/contracts`, so its parent (the project root) is private and empty. This is the
pattern the neighbouring tests already use (`lint_cache_second_run_hits`, `lifecycle_first_run_all_new`).
- `lint/mod_tests.rs`: `lint_empty_dir` is nested, and its assert prints the report. `lint_validation_failure_skips_audit_and_score`
  was nested too. It could not flip (it asserts `!passed`), but it had the same bare-/tmp root.
- `contracts-cli/tests/pvl_zero_contracts.rs::empty_dir()` and `ont6_lint_verdict.rs::corpus()` now return a small
  `Corpus { _root: TempDir, dir }` with a `path()` method, so every call site is unchanged (`empty_dir()` is used 13 times, and `d.path()` appears 10 times in `ont6_lint_verdict.rs`).

## Not changed, stated
`pv lint <dir>` still reads `<dir>/../scripts/…` as its baseline. That is the lint's design (a corpus lives at
`<repo>/contracts`). A user who lints a stray dir directly under `/tmp` would see the same effect. That is a product
question and is not in this ticket.
