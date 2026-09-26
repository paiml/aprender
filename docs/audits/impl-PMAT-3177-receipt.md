# impl receipt — PMAT-3177 (#3177): workspace-test shards wait on guard-tree

Author: aprender-36 (claude-opus-5-5). Assigned by: cop aprender-cf, 2026-09-24. PR #4196.

## Ticket intent

A merge group whose guard-tree has already failed must stop using workspace-test runner time. The cop's directive:
- Prove it with a case table: guard red → shards skipped or cancelled; guard green → they run.
- Check that the required check `workspace-test` still REPORTS in the skip case. Otherwise the merge queue blocks forever.
- No runs-on, secret or permission changes.

## What changed

| file | change |
|---|---|
| `.github/workflows/ci.yml` `workspace-test-shard` | `needs: [guard-tree]`, no `if:` (GitHub's default `success()`), plus a comment carrying the measurements below |
| `.github/workflows/ci.yml` `workspace-test` (fan-in) | `needs: [guard-tree, workspace-test-shard]`. Keeps `if: always()`. A new `skipped)` arm fails with `shards skipped because guard-tree was '<result>'`. Guard-tree is in `needs` only so the message can name it |
| `scripts/check_workspace_test_waits_on_guard_tree.sh` | the case table, with `--self-test` mutants. Dispatched by `guard_tree.sh --no-cargo`, as `--list` shows at line 89 |
| `docs/roadmaps/entries/PMAT-3177.yaml` + `roadmap.yaml` | the roadmap row (aggregate regenerated) |

Option 2 (`gh run cancel` when guard-tree fails) was rejected: it needs `actions: write`, a permission change, and the ticket forbids that.

## The main risk: does the required check still report?

Yes.
- `workspace-test` is the fan-in with `if: always()`, so it runs whatever its needs concluded. That structure already existed (PACK-001).
- When the shards are skipped, `needs.workspace-test-shard.result == 'skipped'`, and the fan-in exits 1 with an `::error::` naming guard-tree.
- The check-run named `workspace-test` therefore concludes **failure**. It is never missing and never skipped.
- The case-table row `guard-red` asserts exactly this by executing the fan-in's shipped `run:` script. The `fan-in-no-always` mutant proves the row would catch the fan-in being skipped.

## Case table (measured at 58fb5a080)

```
$ bash scripts/check_workspace_test_waits_on_guard_tree.sh ; rc=0
  ok   guard-red: guard-tree failure -> shards skipped -> `workspace-test` red
  ok   guard-cancelled: guard-tree cancelled -> shards skipped -> `workspace-test` red
  ok   guard-green: guard-tree success -> shards success -> `workspace-test` green
  ok   shard-red: guard-tree success -> shards failure -> `workspace-test` red
  ok   required-name: the fan-in reports as `workspace-test`
PASS
$ bash scripts/check_workspace_test_waits_on_guard_tree.sh --self-test ; rc=0
  ok    mutant shards-no-needs    killed by guard-red, guard-cancelled
  ok    mutant shards-always      killed by guard-red, guard-cancelled
  ok    mutant fan-in-no-always   killed by guard-red, guard-cancelled, shard-red
  ok    mutant skipped-arm-green  killed by guard-red, guard-cancelled
  ok    mutant fan-in-renamed     killed by required-name
SELF-TEST OK
$ bash scripts/check_workspace_test_waits_on_guard_tree.sh --ci <origin/main aa7c6ef03 ci.yml> ; rc=1
  FAIL guard-red: guard-tree failure -> shards RUN, want skipped
  FAIL guard-cancelled: guard-tree cancelled -> shards RUN, want skipped
```

The table is RED on the pre-fix ci.yml at exactly the rows the ticket is about, and GREEN with the fix.

**What the table models, and what it does not.**
- It evaluates GitHub's documented job-status rules: a job with `needs:` and no `if:` runs only when every need succeeded, and `always()` runs regardless.
- It refuses, rather than guesses, any other `if:` expression on these two jobs.
- It does not run GitHub itself. The live demonstration below covers that.

## Green-path cost (measured)

Taken from the last 10 successful ci.yml runs per event, using `gh api .../jobs`. The duration is each job's `started_at`→`completed_at`, so queue time is excluded. The added wall clock is `(guard + shard_max) − max(guard, shard_max)`.

| event | median | mean | worst | n |
|---|---|---|---|---|
| merge_group | +73 s | +196 s | +1048 s | 10 |
| push (main) | +432 s | +502 s | +1014 s | 10 |

- Runs: merge_group 35960661956 … 35564092115; push 35963300528 … 35563537942.
- The merge-queue cost is small because the merge_group shards mostly finish in 33–134 s.
- A push to main pays guard-tree's duration serially, but nothing waits on a main push.

## Live demonstration (deliberately RED guard)

Draft PR #4195 (branch `demo/3177-red-guard`, never to merge) runs the fixed `needs:`/`if:` structure. Its ci.yml is cut to guard-tree, workspace-test-shard and workspace-test, and guard-tree gets a deliberate `exit 1`. The run is 35974014776 (`pull_request` event):

| job | conclusion | started | completed |
|---|---|---|---|
| guard-tree | failure | 08:34:52Z | 08:35:00Z |
| workspace-test-shard | **skipped** | 08:35:01Z | 08:35:01Z |
| workspace-test (fan-in) | **failure** | 09:14:26Z | 09:16:07Z |

The fan-in's own log (job 107556798473):
```
workspace-test-shard matrix result: skipped (guard-tree: failure)
##[error]workspace-test: shards skipped because guard-tree was 'failure' (#3177: the shards wait on guard-tree; fix the guard first)
##[error]Process completed with exit code 1.
```

- The shards were skipped 1 s after guard-tree failed and took no runner.
- The required check `workspace-test` REPORTED, and it reported failure, naming guard-tree.
- The fan-in's 39 min queue (08:35→09:14Z) is runner starvation on the fleet that morning. It is not caused by this change.

## Guards run at the committed head

All returned rc 0:
- `check_guards_are_wired`
- `check_baseline_ratchets`
- `check_roadmap_fragment_required`
- `check_roadmap_completion_is_cited`
- `check_roadmap_diff_additive`
- `check_roadmap_ids_unique`
- `check_roadmap_sorted`
- `check_workflow_env_defined`
- `check_receipt_gate_base_owned`

## Not done here

The issue's full "Done" wants a deliberately red **merge group** with before/after wall clock in the build ledger. A red PR cannot enter the merge queue, so the demonstration runs on the `pull_request` event. `needs:` evaluation is the same for every event. The first real red merge group after this lands is the merge-group measurement. The before side is aprender-dd's 2.8 h over 20 red groups.
