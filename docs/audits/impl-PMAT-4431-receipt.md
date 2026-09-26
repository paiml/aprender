# PMAT-4431 receipt — merge of origin/main e7a52949d into fold/b3-onto-main d99168e39

Merge commit e5929e6e5 (no rebase, no force), then three fix commits up to this receipt.
The judged diff (base e5929e6e5) is the three fix commits; the resolution itself is stated here.

## Resolution decisions (merge commit e5929e6e5)
- `.github/workflows/ci.yml`: main's version taken whole. Measured: HEAD vs e7a52949d is **identical**.
  #4441 moved the job bodies into `ci/sections.yml` run by `scripts/ci/fat_driver.py`; the fold's own
  ci.yml delta (#4112 release/** push scope, #4102 target register/release steps) does not apply there
  (`git apply --check` fails at ci/sections.yml:12). It is NOT carried; it is filed as finding
  `57-4431-ci-4102-4112-report` in `docs/findings/aprender-57.jsonl` for a re-port.
- The guards whose subject was that ci.yml delta are retired with `git mv` to `scripts/retired/`
  (`check_workspace_test_waits_on_guard_tree.sh`, `check_ci_release_fold_scope.sh`; both exit 1 against
  main's ci.yml) — table in `scripts/retired/README.md`. guard_tree.sh dispatches `scripts/check_*.sh` only.
- `docs/roadmaps/roadmap.yaml`: union of both sides by id. Measured: HEAD 1111 ids; ids in fold∪main
  missing from HEAD: 0. Then regenerated from fragments (PMAT-4438 adopted as a fragment).

## Fix commits (the judged diff)
- `.gitattributes`: two sub-directory ledgers named merge=union (`docs/audits/*.jsonl` does not cross `/`).
- `scripts/ci_run_target_release.sh` retired: its only caller was the dropped #4102 ci.yml step
  (check_guards_are_wired: unwired 3 -> 4). README row + finding ask updated.
- `model_ladder_cells_produce.py`: truncated reason now says `... and N more chars`.
  `check_ladder_cells_producer.sh` owed-set mutant rewritten without a slice; case table rc=0, mutant still killed.
- `admission.rs::row_errors`, `receipt.rs::problems`: split into helpers, same checks/messages/order.
  `cargo test -p aprender-review-experiment --lib` + clippy -D warnings: rc 0. check_complexity_ratchet PASS.
- PMAT-4072 fragment cites its quorum receipt (`proof:docs/audits/quorum-PMAT-4072-fe8c9dff7.json`).

## Guards
guard_tree.sh --no-cargo on the merged tree: remaining reds are check_baseline_ratchets (also RED on main
e7a52949d: host bashrs 7.4.2 vs baseline 7.4.1), and pipe-grep-q / silent-truncation hits that are only in the
gitignored, untracked `crates/aprender-contracts-staging/lean/.lake/`; both are green on a `git archive HEAD` export.
