# PMAT-4112 receipt: fold pushes to release/** run guard-tree + guard-cargo

**Ticket:** #4112. **Approval:** the operator, verbatim, relayed by the cop aprender-cf on 2026-09-24: "1. recommended approved". The recommended option was B: `push: branches: [..., 'release/**']`, scoped to the guard jobs with an `if: github.ref` condition. **Base:** origin/main aa7c6ef03.

## What changed (`.github/workflows/ci.yml`)
- `on.push.branches: [main, master, 'release/**']`.
- **Skipped on a push to `release/**`**, via `if: ${{ !(github.event_name == 'push' && startsWith(github.ref, 'refs/heads/release/')) }}`: `ci` (the sovereign-ci reusable), `workspace-test-shard`, `mac-check`, `vendored-schemas`, `determinism`, `determinism-compare`.
- `workspace-test` and `gate` keep `always()`, ANDed with the same exclusion.
- `mutants`, `pr-review-*`, `gpu-*` and `cuda-unit` were already pull_request-only and are unchanged.
- **Result:** a fold push runs exactly `guard-tree` and `guard-cargo`. The gate is skipped there, because it reads skipped jobs as failures. The run's conclusion is red iff a guard job is red.
- **Both guard jobs' comparand step unshallows on a release push.**
  - **Why:** a fold push is not on origin/main, and a depth-1 checkout cannot name `merge-base(origin/main, HEAD)`. `scripts/lib/resolve_base.sh` refuses a single-parent fold outright ("merge-base … unresolvable (shallow checkout)"). A merge-commit fold would fall back to the origin/main tip.
  - **Measured on 2026-09-24** (a depth-1 clone of chore/0.69.1-merge-back, then `git fetch --unshallow` of it plus main): 10.8 s, 159 MB `.git`. Afterwards `git merge-base origin/main HEAD` resolves (49fe19c28).
- **Unchanged on pull_request, push→main and merge_group:** each job's `if` evaluates exactly as before on those events (the guard below evaluates all three).
- **Scope of effect:** a push runs the `ci.yml` in the PUSHED tree, so this covers release branches cut from main after this lands, not the ones already cut.
- **Concurrency:** a push is in the `ci-<ref>` group with cancel-in-progress false. Rapid folds keep one running and one pending run per release branch, and GitHub replaces an older pending run with a newer one. That is acceptable: the guards judge the whole tree, so the latest fold's run covers the earlier folds.

## Guard: `scripts/check_ci_release_fold_scope.sh` (new; auto-wired by guard_tree.sh with both `[self-test]` and `[run]` rows)
- R1: `release/**` is in `on.push.branches`.
- R2: every job's `if:` is evaluated for pull_request→main, push→main and push→release. The evaluator knows only the shapes ci.yml uses, and any other shape is REFUSED, never guessed. Exactly `guard-tree` and `guard-cargo` run on a release push, and neither `needs` a job a release push skips.
- R3: both guard jobs unshallow on a release push.
- **Must-RED:** main's ci.yml fails with 11 violations (R1, R2 ×8, R3 ×2).
- **Self-test 8/8:** six planted mutants, each RED for its OWN reason (checked by reading each one's first violation), plus a missing file refusing (rc 2) and the real ci.yml passing:
  1. release/** dropped;
  2. the exclusion dropped from workspace-test-shard;
  3. guard-tree excluded;
  4. the unshallow dropped;
  5. an unknown `if` shape;
  6. guard-cargo `needs: [ci]`.

## Checks
- `guard_tree.sh --no-cargo`: 94 checks, 0 failed, including the new guard's two rows.
- `bashrs lint`: 0 errors.
- actionlint: the same 2 pre-existing findings as main's ci.yml (compared with the repo's `.github/actionlint.yaml` on both sides), 0 new.
- **Not measurable before merge:** a real fold-push run. The first release branch cut after this lands, and its first fold push, is the proof, and the cop should watch that run. A dispatch cannot stand in for it: `workflow_dispatch` is not a push, so the exclusion does not apply to it.
