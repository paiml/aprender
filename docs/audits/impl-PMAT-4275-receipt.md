# impl receipt — #4275 binary-release smoke: the asset belongs to its tag

## Done-when as written (issue body, 2026-09-24 14:17Z) and the rewrite this PR proposes

Written: compare against the tag **with `-rc.N` stripped**, assert the binary's sha equals the tag's commit, and a
case table: `v0.69.3`↔`0.69.3` ok; `v0.69.3-rc.1`↔`0.69.3` ok; `v0.69.3-rc.1`↔`0.69.2` refused; sha mismatch refused.

The strip clause was overtaken the same day. Operator, 2026-09-24: "we need actual version numbers", "version number
needs release canidate info in it". So the rc tag tree is stamped `X.Y.Z-rc.N` (`scripts/release/stamp_rc_version.sh`)
rather than the tag being stripped to fit a bare binary. Row 2 of the written table is therefore inverted.

**Proposed done-when (for the quorum to confirm or refuse):**
1. smoke-cpu and smoke-cuda judge each asset with `scripts/release/asset_version_check.sh TAG COMMIT "$(apr --version | head -1)"`,
   a whole-string compare. The old `*${TAG#v}*` substring glob is gone.
2. The printed version equals the tag exactly, `-rc.N` included. A promoted final (`vX.Y.Z` shipping rc bytes) accepts only
   `X.Y.Z-rc.N` of the same X.Y.Z.
3. The printed sha is the tag's commit. `+no-git` or any other sha is refused.
4. The case table covers: `v0.69.3`↔`0.69.3` ok; `v0.69.3-rc.1`↔`0.69.3-rc.1` ok; **`v0.69.3-rc.1`↔bare `0.69.3` refused**;
   `v0.69.3-rc.1`↔`0.69.2` refused; sha mismatch refused; `0.69.30` for `v0.69.3` refused.
5. The table **runs on every PR** and can be shown to turn RED. That is this PR's change; before it, nothing ran the table.

## Where each clause is met (base batch/ont-10 @ 8d021f61e1 = #4502 head)

| Clause | Where |
|---|---|
| 1 | `.github/workflows/binary-release.yml` smoke-cuda :820, smoke-cpu :898 call `asset_version_check.sh`; `git grep 'TAG#v}\*'` on the workflow: 0 hits |
| 1–2 | `stamp_rc_version.sh` runs on the tag tree before each build (binary-release.yml :168 :324 :495 :664) |
| 3 | `APR_GIT_SHA_OVERRIDE` binds the sha (:366 :536); the table refuses `+no-git` and a foreign sha |
| 4 | `bash scripts/release/asset_version_check.sh --self-test`: PASS, rc 0 (lambda, 0.02 s) |
| 5 | **this PR**: `scripts/check_release_asset_version.sh` |

Clauses 1–4 landed in the sealed car (be48bedba6 + 3d677a8c21) and ride #4502. aprender-91 verified them at 919a1dd950
and 8d021f61e1 (`handoff/091-close-4275-3569.md`). This PR adds only clause 5.

## Clause 5: the gap and the fix

`asset_version_check.sh` lives in `scripts/release/`. guard_tree.sh's universe is `git ls-files 'scripts/check_*.sh'`,
and check_guards_are_wired.sh scans `scripts/*.sh` at depth 1, so no CI step, Makefile target or dogfood gate ran
the table (91's receipt: "wired into no Makefile/CI target"). The new `scripts/check_release_asset_version.sh` is a
build-tool-free wrapper, so `guard_tree.sh --no-cargo` (ci/sections.yml :1314) dispatches it. No workflow or
sections.yml edit.

Measured in worktree 5d/4275-avc-wired (lambda, bash-only, no cargo):
- clean: rc 0, `check_release_asset_version: PASS`
- `guard_tree.sh --list --no-cargo` includes it (1); `--cargo-only` does not (0)
- mutant 1, the written done-when's strip (`want=${want%%-rc.*}` in a scratch copy): rc 1, 3 rows broke, including
  "an rc asset printing bare X.Y.Z is refused"
- mutant 2, `asset_version_check.sh` deleted in a scratch copy: rc 1, "FAIL: … is missing"
- `check_guards_are_wired.sh`: rc 0, 212 scanned, 3 unwired (the baseline, no growth). A first draft whose text used
  the build tool landed in `--cargo-only`, which CI does not dispatch, and this guard went RED "grew 3 -> 4". That
  draft was discarded.
- `bashrs lint`: no issues. Worktree `git status` after the mutants: only the new file.

## Not in this PR (finding FND-20260928-release-selftests-dark)
`stamp_rc_version.sh --self-test` uses cargo as its oracle, so it can only be wired as a named `ci/sections.yml` step,
which needs cop approval. Measured at 8d021f61e1 (grep of .github, ci, Makefile and scripts/check_*.sh for `<name>.sh … --self-test`):
13 `scripts/release/*.sh` ship a table. rc_cut and rc_publish_gate run in rc-cut.yml; carry_milestone_items,
t2_preflight and tag_coverage_gate run through a check_ wrapper. After this PR, 7 are still run by nothing:
stamp_rc_version, fleet_cells_gate, models_t1, ont_complete_gate, prepare_bump, promote_rc, rc_fleet_stage.
The root cause is that check_guards_are_wired.sh only scans depth 1.
[U] No live evidence yet: the first binary-release run under this code is v0.70.0-rc.1's smoke-cpu/cuda. Cite that run id
at close.
