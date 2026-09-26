# Retired guards

Guards whose subject no longer exists in the tree. They are kept, not deleted, so
that re-porting them is a `git mv` plus an edit, not an archaeology job. Nothing
runs them: `scripts/guard_tree.sh` dispatches `scripts/check_*.sh` only.

| Guard | Retired by | Why | Re-port |
|---|---|---|---|
| `check_workspace_test_waits_on_guard_tree.sh` (#3177/#4102) | merge of main into #4431 (fold/b3-onto-main), after #4441 | #4441 moved the jobs into `ci/sections.yml` run by `scripts/ci/fat_driver.py`; the guard reads `.github/workflows/ci.yml` for a `workspace-test-shard` job that is no longer there | finding `57-4431-ci-4102-4112-report` in `docs/findings/aprender-57.jsonl` |
| `check_ci_release_fold_scope.sh` (#4112) | same | main's fat `ci.yml` has no `release/**` push trigger, so "only guard-tree/guard-cargo run on a release push" has no subject | same finding |
