# PMAT-6677 receipt (C12 r6, aprender#6684)

Measured at code commit 0b2883a1a6c268dc760c1a5b223fa0491093fefc (the receipt commit that follows adds only this file; scripts/ and docs/roadmaps/ are byte-identical to it).

## Commands and verbatim summary lines

    bash scripts/release/falsify_promote_scheduler.sh
    falsify_promote_scheduler: 384 passed, 0 failed

    bash scripts/release/falsify_promote_scheduler.sh --mutants
    falsify_promote_scheduler --mutants: 56 mutants, 0 not caught

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
