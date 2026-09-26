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

## Fix commits (the judged diff) — each change is here because a named guard was RED without it

| Change | Guard RED without it (its own words, on the merged tree) |
|---|---|
| `.gitattributes`: `docs/audits/review-corpus/corpus-v1.jsonl`, `docs/audits/rex-001/rex-04-admission.jsonl` merge=union | check_append_only_ledgers: "FAIL  <path> is an append-only ledger and is NOT union-merged." — `docs/audits/*.jsonl` does not cross `/` |
| `scripts/ci_run_target_release.sh` → `scripts/retired/`, README row, finding ask | check_guards_are_wired: "unwired guards grew 3 -> 4. NEW: ci_run_target_release.sh" — its only caller was the dropped #4102 ci.yml step |
| `model_ladder_cells_produce.py:210` reason says `... and N more chars` | check_no_silent_truncation: "FAIL no_silent_truncation: a NEW truncation of a value a human reads later -- say how much was dropped" naming this line. Both files are fold-origin (absent on main); the guard landed on main after the fold branched |
| `check_ladder_cells_producer.sh:116` owed-set mutant without a slice | same guard, same FAIL, naming this line. A mutation fixture fits none of the baseline's classes (display/loud/fatal/id-prefix/numeric), so it is rewritten rather than baselined; same semantics (first owed rung only). `bash scripts/check_ladder_cells_producer.sh` rc=0: "mutant owed-set killed by: good: ..." |
| `admission.rs::row_errors`, `receipt.rs::problems` split into helpers, same checks/messages/order | check_complexity_ratchet: "RED NEW ... row_errors cyclomatic 19 cognitive 27", "... problems cyclomatic 19 cognitive 29" (cognitive limit 25). After: "PASS (D2): e7a52949d vs 9a9d64041 — none new, none grown". `cargo test -p aprender-review-experiment --lib` + clippy -D warnings rc 0 |
| PMAT-4072 fragment: ONLY `notes` changes (null → cites `docs/audits/quorum-PMAT-4072-fe8c9dff7.json`, which commit 5c1dbe01f landed beside its `status: completed`); status/assignee untouched | check_roadmap_completion_is_cited: "FAIL docs/roadmaps/roadmap.yaml PMAT-4072 claims completed and cites nothing, and is NOT in the frozen baseline — so it is a NEW unprovable claim" |
| PMAT-4438 fragment adopted, PMAT-4431 fragment, roadmap.yaml regenerated | check_roadmap_fragment_required / check_roadmap_sorted (read committed HEAD); rc 0 after |

## Guards
guard_tree.sh --no-cargo on the merged tree: remaining reds are check_baseline_ratchets (also RED on main
e7a52949d: host bashrs 7.4.2 vs baseline 7.4.1), and pipe-grep-q / silent-truncation hits that are only in the
gitignored, untracked `crates/aprender-contracts-staging/lean/.lake/`; both are green on a `git archive HEAD` export.
