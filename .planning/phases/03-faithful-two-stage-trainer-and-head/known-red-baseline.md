# Phase 3 known-red test baseline

Established after Wave 1 merge, by two-sided control (same tests run at the
pre-wave base commit `e2dee4be9` in a temporary detached worktree, single-threaded).

**These 24 `aprender-train --lib` failures are PRE-EXISTING and unrelated to Phase 3.**
Later waves MUST diff against this list rather than treating a red `aprender-train`
suite as a new regression. A failure NOT on this list is a real regression.

## gpu:: (21) — shared GPU reservation ledger; fail deterministically, not a parallelism flake

gpu::guard::tests::test_guard_available_mb_after_reserve
gpu::guard::tests::test_guard_direct_acquire
gpu::guard::tests::test_guard_multiple_reservations_sequential
gpu::guard::tests::test_guard_multiple_sequential_reserve_release
gpu::guard::tests::test_guard_small_gpu
gpu::guard::tests::test_guard_status_returns_string
gpu::guard::tests::test_guard_update_actual
gpu::guard::tests::test_guard_update_actual_reduces_reserved
gpu::ledger::tests::test_capacity_invariant_prevents_overallocation
gpu::ledger::tests::test_drop_releases_reservation
gpu::ledger::tests::test_gpu_status_display
gpu::ledger::tests::test_gpu_status_display_with_actual_mb
gpu::ledger::tests::test_ledger_data_prune_dead
gpu::ledger::tests::test_multiple_reservations_same_gpu
gpu::ledger::tests::test_read_reservations_returns_our_gpu_only
gpu::ledger::tests::test_reservation_is_alive_current_process
gpu::ledger::tests::test_reserve_and_release
gpu::ledger::tests::test_reserve_exact_capacity
gpu::ledger::tests::test_update_actual
gpu::ledger::tests::test_with_lease_hours_custom
gpu::wait::tests::test_timeout_when_full

## prune::snapshot_tests (3) — insta snapshots

prune::snapshot_tests::tests::snapshot_all_prune_methods
prune::snapshot_tests::tests::snapshot_pipeline_stages
prune::snapshot_tests::tests::snapshot_schedule_validation_errors

## Control evidence

| Run | Command | Result |
|-----|---------|--------|
| Merged tree | `cargo test -p aprender-train --lib --features setfit -- --test-threads=1 gpu::` | 196 passed, 21 failed |
| Base `e2dee4be9` | same, no `--features setfit` | 196 passed, 21 failed (identical names) |
| Merged tree | `... prune::snapshot_tests` | 14 passed, 3 failed |
| Base `e2dee4be9` | same | 14 passed, 3 failed (identical names) |

Single-threaded on both sides, so this is not a parallel-execution artifact.
