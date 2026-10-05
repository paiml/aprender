# PMAT-4829 implementation receipt (head aefc7083f2)

## Defect
`module_of` in scripts/check_tree_reader_tests.sh accepted a `mod NAME;` only from the child's own directory.
crates/aprender-contracts/src/ontology/extract/json/github.rs is declared by json.rs (2018 layout), so it fell back
to the whole crate. The registry's whole-lib `aprender-contracts --lib` row then wins over every aprender-contracts
module atom (row 86 rule), and BSE-17 rows 6, 90 and 91 fail. That step runs only in guards-nightly.

## Change
- resolver: a `mod` record also matches when its declaring file is `<owner>.rs` (3 lines + comment).
- fixture: reader_mods/src/sib.rs (`pub mod kid;`) + src/sib/kid.rs (a test reading the tree); lib.rs `pub mod sib;`;
  the three goldens gain `sib::kid` (flat: `kid`); one new case row.
- registry regenerated with `--update`: one line changes, whole-lib -> `ontology::extract::json::github`.
  The only other WARN left (apr-cli src/bin/apr-corpus-ingest.rs) is the existing fail-open case row 92 expects.

## Evidence (measured)
- tree-reader self-test, head: 38 checks, 0 failed. main's resolver against the new fixture: rows 23/33/35 FAIL (golden diffs).
- fixture RED before the fix: `WARN unresolved-include .../src/sib/kid.rs` + a `reader_mods --lib` whole-crate row.
- BSE-17 `ci_test_tier.sh --self-test`, same run, scope TasksMax=8192 (pids.max read back 8192, peak 24),
  SELF_TEST/FILTERSET/FS_TARGETS unset:
  - main 11f844a772: 93 checks, 1 failed (rows 6, 90, 91 FAIL), rc 1.
  - head: 93 checks, 0 failed, rc 0.

## Effect on what blocks
Quick tier for aprender-contracts: whole lib (since #4655) -> 45 module atoms (as before #4655). Full tier unchanged.
Per-tier test counts: NOT_MEASURED yet (needs a `cargo nextest list` run).

## Review
- Planted round on aefc7083f2: 3/3 PASS (claude-sonnet-5 x2, claude-haiku-4-5).
  It is degraded same-family by policy: the non-Claude lane was not run on a non-tier-1 diff.
  That is not a measured quota exhaustion, so this round does NOT satisfy the arming rule.
  A two-family round replaces it before any arming.
- Plant (resolver reverted, sibling case row flipped to expect the whole-crate fallback): 3/3 FAIL.
  Every lane named scripts/check_tree_reader_tests.sh:142 and/or :473, so the plant was caught.
