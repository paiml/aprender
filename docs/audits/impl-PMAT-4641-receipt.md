# Implementation receipt — PMAT-4641: CB-200 as a head-vs-base release gate

Agent: apr-0d-b1. Ticket PMAT-4641 (issue #4641, epic #3997). Cop ruling 2026-09-29 21:33Z: base = the previous RELEASE tag (last published, not rc, not merge-base).

## Defect

The release path judges CB-200 (TDG Grade Gate) against a stored number, `.pmat-gates.toml [tdg] baseline = 599`:
`scripts/dogfood.sh` (`pmat-comply` row) ← `scripts/release/autopilot.sh:118` (T-1, `die` on rc≠0) and
`scripts/release/t2_preflight.sh:127`. On main (00052c0128) under pmat 3.42.0 CB-200 counts 622, so FINAL is NO-GO with no
code change. A stored limit plus a moving scanner is not evidence (never-worse rule: head vs base, same scanner, same run).

## Change

1. `scripts/check_cb200_head_vs_base.sh` (new): measures HEAD and BASE with one pinned `$PMAT_BIN` in one run, `[tdg] baseline`
   set to 0 in both scratch trees (the stored value cannot decide anything). Exit 0 head ≤ base, 1 head > base, 2 usage,
   3 NOT MEASURED (Skip, no count, unparsable, unresolvable ref, `PMAT_BIN` unset). Never a pass on 3. `--selftest`: 13 rows.
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
| Selftest polarities | 13/13: clean Pass = 0; equal/lower green; higher red with stored 999; Skip/no-count/garbage/unresolvable = rc 3; no --base = rc 2 |
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

## Q0 round 1 (opus-5-5 FAIL, haiku-4-5 PASS, gpt-oss-120b PASS) — what changed

- Wrong ticket (PMAT-937 is the 609-baseline ticket): filed PMAT-4641 / #4641, receipt renamed.
- Stale comment in the comply case block said Fail stays NO-GO: reworded to say Fail is re-judged head-vs-release.
- Tag at HEAD made the comparison vacuous: base now comes from `--default-base` = newest final tag that does NOT contain HEAD
  (`git tag --no-contains`), and a base that resolves to the same commit as head is rc 3 (NOT MEASURED). Selftest rows added (12 total).
- "Fail can now be PASS": that is the cop ruling (no stored limit; base = previous release tag). It is bounded by head <= the last release
  and by rc 3 = FAIL; the CI ratchet (`check_complexity_ratchet.sh`, `scripts/cb200_baseline.txt`) is untouched and stays shrink-only.
- No dogfood-level fixture: the `Fail)` arm only maps the script's rc (0 PASS, else FAIL); the decision logic and the base choice live in the
  script and are what the selftest drives.
- Refactors in scope: the cop ruled the 4 added findings are fixed in this PR (main is +4 over v0.69.3, not +2).

## Round-4 dispositions
- Pass/Warn arms are NOT re-judged head-vs-base: they already pass on the scanner's own verdict, so the change only ever turns a stored-baseline Fail into a measured comparison (never weakens a pass; cost of two extra full scans on every green run not warranted by the ruling).
- Helper is called via `$SKILL_DIR` (cwd-independent); guards check_verifier_pinning/check_dogfood_shim/check_dogfood_coverage rc 0.
- Clean Pass with baseline neutralised (no count in message) = 0, selftest row added (13/13).
- Dirty tree in the release arm = FAIL NOT MEASURED (helper measures committed HEAD).
