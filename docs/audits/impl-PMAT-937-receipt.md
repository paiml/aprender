# Implementation receipt — PMAT-937: CB-200 as a head-vs-base release gate

Agent: apr-0d-b1. Cop ruling 2026-09-29 21:33Z: base = the previous RELEASE tag (last published, not rc, not merge-base).

## Defect

The release path judges CB-200 (TDG Grade Gate) against a stored number, `.pmat-gates.toml [tdg] baseline = 599`:
`scripts/dogfood.sh` (`pmat-comply` row) ← `scripts/release/autopilot.sh:118` (T-1, `die` on rc≠0) and
`scripts/release/t2_preflight.sh:127`. On main (00052c0128) under pmat 3.42.0 CB-200 counts 622, so FINAL is NO-GO with no
code change. A stored limit plus a moving scanner is not evidence (never-worse rule: head vs base, same scanner, same run).

## Change

1. `scripts/check_cb200_head_vs_base.sh` (new): measures HEAD and BASE with one pinned `$PMAT_BIN` in one run, `[tdg] baseline`
   set to 0 in both scratch trees (the stored value cannot decide anything). Exit 0 head ≤ base, 1 head > base, 2 usage,
   3 NOT MEASURED (Skip, no count, unparsable, unresolvable ref, `PMAT_BIN` unset). Never a pass on 3. `--selftest`: 9 rows.
2. `scripts/dogfood.sh`: on CB-200 `Fail` (stored baseline exceeded) compares head to `$DOGFOOD_CB200_BASE`, default the newest
   final `vX.Y.Z` tag (v0.69.3 today). Pass/Warn unchanged; Skip/absent still FAIL; anything unmeasurable still FAIL.
   It only turns a stored-limit FAIL into a head-vs-base verdict, so it cannot loosen anything that was RED because of new debt.
3. The 4 definitions main added over v0.69.3 (618 → 622), refactored into behaviour-identical helpers:
   `check_valid_under` (Rust), `scripts/coverage_serve_shards.py::main`, `scripts/lib/git_patch_id.py::get_one_patchid`,
   `scripts/nightly_manifest.py::main`.

## Proof

| Claim | Evidence |
|-------|----------|
| Stored 599 is read on the release path | `dogfood.sh` comply arm; main 622 vs 599 = Fail |
| Base v0.69.3 = 618, main = 622 (+4, not +2) | real pmat 3.42.0, baseline zeroed, DB diff by (file, fn): the 4 above |
| GREEN | `check_cb200_head_vs_base.sh --base v0.69.3 --head HEAD` → `head 618 <= base 618`, rc 0 (pmat 3.42.0) |
| RED | same script `--head origin/main` (00052c0128) → `head 622 > base 618`, rc 1 (see PR comment for the run) |
| Selftest polarities | 9/9: equal/lower green; higher red with stored 999; Skip/no-count/garbage/unresolvable = rc 3; no --base = rc 2 |
| valid_under refactor | `cargo test -p aprender-contracts --lib valid_under` 17 passed |
| shards refactor | golden diff old vs new, ALONE/PACK/DEEP shrunk so every branch runs: shard dirs byte-identical, solo and error paths identical |
| git_patch_id refactor | old vs new on 60 `git log -p` commits + 5 `--binary` commits, `--stable/--verbatim/--unstable`: output byte-identical |
| nightly_manifest refactor | `--self-test` PASS 0 failing rows; usage/error rc and text identical |
| Guards | check_dogfood_shim, check_dogfood_coverage, check_verifier_pinning (after removing a `PMAT_BIN=` bypass), check_apr_bin_pinned: all rc 0 |

## Scope and honesty

- No workflow files. The gate is kept or tightened: a Fail is now RED unless head ≤ the last release, and a PR that adds a
  below-grade definition goes RED with no baseline to raise.
- Known limit: head ≤ base tolerates debt that already existed at the last release. `[tdg] baseline` and
  `scripts/cb200_baseline.txt` (the CI ratchet, `check_complexity_ratchet.sh`) are untouched and still shrink-only.
- Base tag must be fetched (`git fetch --tags`); a missing tag is rc 3 = FAIL, not a pass.
