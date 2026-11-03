# PMAT-6677 receipt (C12 r7, aprender#6684)

Measured at code commit f040aa8e328f04df240c3320edfbf98c6546fa8d (the receipt commit that follows adds only this file; scripts/ and docs/roadmaps/ are byte-identical to it).

## Commands and verbatim summary lines

    bash scripts/release/falsify_promote_scheduler.sh
    falsify_promote_scheduler: 393 passed, 0 failed

    bash scripts/release/falsify_promote_scheduler.sh --mutants
    falsify_promote_scheduler --mutants: 57 mutants, 0 not caught

    make roadmap-aggregate && make roadmap-aggregate-check
    ok  roadmap.yaml == aggregate(459 fragment(s)), idempotent

## r6 rulings applied

- (a) PMAT-6677.yaml acceptance criteria name: killed run after a skip writes one RED; stale breadcrumb reported once; already-promoted fallback written once per (promoted sha, newest red sha).
- (b) dry-run path has the same per-pair marker; rows dry-run-pair-first (1 RED) and dry-run-pair-identical-second (0 RED).
- (c) "RED clean-room run ... ended/cannot be read" and "RED newest nightly crux ... falling back" restored verbatim; asserted by grep -F against `git show origin/main:scripts/release/promote_scheduler.sh`. The one fallback line is in addition.
- (d) falsify changes to pre-existing lines versus origin/main, each kept for one reason:
  - mutant dryrun-skipped-on-execute: regex re-anchored, because the dry-run line it matched is now a block (new dedupe).
  - stub `workflow run` branch: adds FX_DISPATCH_FAIL (default off) so the new dispatch-failure rows can run; behaviour unchanged when unset.
  - stub FX_TRAIN_RED_SHA: now a space list (single value behaves as before) so rows with several red candidates can run.
  - sched(): FX_KEEP_STATE / FX_PRESEED_SHA / signal reset added so the dedupe and breadcrumb rows can reuse state across ticks; default path identical.
  - assertion cleanroom-ended-red: REVERTED to origin/main text (inbox_has ... RED).
  - mutant cleanroom-red-second-line: removed (the second clean-room RED line is now pre-existing behaviour).

## r7 rulings applied

- (1) execute success path now goes through fallback_once() with the already-promoted key (fallback-reported-<newest12>); rows execute-then-already-promoted-first/second give 1 fallback RED line in total; mutant execute-marker-dropped is caught.
- (2) row renamed cleanroom-red-restored-line-plus-one-fallback: asserts the restored clean-room RED line separately and the fallback count with red_n == 1.
- Measured at final code commit f040aa8e328f04df240c3320edfbf98c6546fa8d; the make roadmap-aggregate line is unchanged since no roadmap fragment changed after r6.
