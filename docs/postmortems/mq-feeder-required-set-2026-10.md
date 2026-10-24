# The merge-queue feeder required a check no pull request runs (PMAT-6175)

**What happened.** On 2026-10-22 heavy X64 jobs (`x86-main`, `determinism`) moved to `merge_group`/`push` only
(`ci.yml`, commit `ef6095b978`). The feeder source was updated in the same change to judge `pr-light`. The feeder
that actually ran was a pinned worktree (`915765011a`) that still required `x86-main`. From then until 2026-10-23
07:14Z every CLEAN PR was refused with "x86-main is skipped on <sha>", and PRs merged only by hand (#6083, 06:11Z).

**Class.** Same class as the 0.68 A1 finding: a required check that is vacuously unsatisfiable on the path it
guards, so the gate never goes green. The change moved the producer of a check; nothing compared it with the
consumer that requires it. A second hit: the fix landed in source while a pin kept the old copy running.

**Rules.**
1. A change that moves a job between event paths names every consumer of that job (feeder, arm script, rulesets).
2. A check a consumer requires must be a job the consumer's own event runs. The feeder asks `--check-required` on every
   tick against main's `ci.yml` read from GitHub (not a file beside the script, not a pin's copy), refuses to act with an
   inbox line when it fails or cannot be read, and `mq_feeder_install.sh` refuses to install such a feeder. CI runs
   `scripts/check_mq_feeder_required_set.sh` (`ci/explicit-test-commands.d/843-*`): planted incident, absent job, missing
   file, good file, and an end-to-end refusal.
3. After merging a change to a script a unit runs, read the unit's ExecStart and confirm it runs the merged copy.
