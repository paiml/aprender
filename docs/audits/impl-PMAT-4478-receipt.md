# PMAT-4478 implementation receipt

**Defect.** `.github/workflows/ci.yml` (added by #4441) declared
`FAT_SECRET_PR_REVIEW_SIGNING_KEY_B: ${{ secrets.PR_REVIEW_SIGNING_KEY_B }}`. The
`pr-review-sign` section in `ci/sections.yml` reads `secrets.PR_REVIEW_SIGNING_KEY_B64`, and
`scripts/ci/fat_driver.py` serves `secrets.NAME` only from env `FAT_SECRET_NAME`. The section
therefore saw an empty key, refused every unsigned receipt, and no PR opened since #4441 can
obtain a signed receipt, so Arm 4 / `present` is RED everywhere. Measured: #4431, run
36246483181, `pr-review-sign — failure`: "carries an UNSIGNED receipt and
PR_REVIEW_SIGNING_KEY_B64 is empty".

**Change.**
1. `ci.yml`: one line, `_B` → `_B64` on both key and secret reference. No new secret, no
   runner change: `PR_REVIEW_SIGNING_KEY_B64` is the existing repository secret.
2. `scripts/check_fat_secrets_passed.sh` (new): S1 every `secrets.NAME` on a non-comment line
   of `ci/sections.yml` except `GITHUB_TOKEN` has `FAT_SECRET_NAME: ${{ secrets.NAME }}` in
   `ci.yml`; S2 every `FAT_SECRET_KEY` line reads `secrets.KEY`.
3. `ci/sections.yml`: wired next to `check_pr_review_wiring.sh` (case table, then tree).

**Evidence (this tree).**
- `--self-test`: 6/6 rows; `the_4441_truncation` (verbatim main line) → rc 1.
- tree: `OK` rc 0. Against `git show origin/main:.github/workflows/ci.yml`: `FAIL S1` rc 1.
- The comment filter is load-bearing: row `token_and_comment_need_none` plants
  `secrets.ONLY_IN_A_COMMENT` in a comment and expects rc 0; the first draft (no filter) went
  RED on its own sections.yml comment, which is how the rule was found.
- `check_guards_are_wired.sh`: rc 0 (RED before wiring: `NEW: check_fat_secrets_passed.sh`).
- Every `scripts/check_*.sh` that reads `sections.yml`: all rc 0.
- Both YAML files parse.

**Not measured here.** That the signer actually signs: that needs the repository secret, which
only a `pull_request` run of this head has. This PR's own CI is that measurement.
