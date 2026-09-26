# impl-PMAT-4059 receipt: the KEPT binaries' CLI tests, dark in CI, now wired (capped), with a guard

Ticket #4059, part of epic #4057 (0.70.0). Author: aprender-36 (claude-opus-5-5). Base: origin/main @ aa7c6ef03.

## The finding, derived and not listed

`scripts/check_bin_cli_tests_wired.sh --print` derives every test target that SPAWNS a workspace binary (`CARGO_BIN_EXE_<bin>`, `cargo_bin(`, `cargo_bin_cmd!`) and that no `--test` line runs. A line counts as wiring only when it names `--test <name>` together with that target's own `-p <package>`, in a workflow or an explicit-test fragment.

On main @ aa7c6ef03, **93** such targets are dark:

| package | dark | KEEP? |
|---|---|---|
| apr-cli | 45 | yes (apr) |
| aprender-profile | 34 | yes (renacer) |
| aprender (facade) | 3 | yes (apr) |
| aprender-orchestrate | 1 | yes (batuta) |
| 8 other crates | 10 | no; stays ledgered for epic #4057's MERGE/DEPRECATE work |

Two corrections to the issue body:

- **pv is already wired.** All 17 aprender-contracts-cli CLI test files were already on fragment lines.
- **aprender-profile's count was wrong.** The issue says 13. The real number is 34, because 21 of its targets spawn through the `cargo_bin_cmd!` macro, which a `CARGO_BIN_EXE_` grep misses. apr-cli's `cli_integration` was also hidden: pv has a target with the same name, and only a package-aware match separates the two.

## Measured on lambda before wiring (the cop's order)

The first pass ran each of the 82 KEPT targets on its own (`cargo test -p C --test T`, built with `--no-run` first, load 33–70):

- **All 82 passed.** They took 1415 s summed.
- **Most of that was cargo overhead.** Only 187 s was spent inside the tests.
- **One target is huge:** `qwen3_moe_apr_run_live_falsifier`, 156.9 s, of which 135.4 s was a live 30B MoE `apr run`.
- **The next largest** was `command_coverage` at 58.5 s (108 tests).
- **Every other target** took 12–34 s wall, nearly all of it cargo.

The second pass timed the grouped fragments exactly as CI runs them, warm:

| fragment | targets | rc | wall | tests |
|---|---|---|---|---|
| 460-apr-cli-command-coverage | 1 | 0 | 57.2 s | 108 passed, 0 failed |
| 470-apr-cli-kept-cli-tests | 43 | 0 | 22.9 s | 599 passed, 0 failed |
| 480-aprender-facade-e2e | 3 | 0 | 21.8 s | 4 passed, 0 failed |
| 490-aprender-profile-cli-tests | 34 | 0 | 12.3 s | 291 passed, 0 failed |
| 500-aprender-orchestrate-integration | 1 | 0 | 47.3 s | 58 passed, 0 failed |

In total: 82 targets, 1060 tests, 0 failed, **161.5 s** wall. Fragment 470 includes apr-cli `cli_integration`.

**Capped** means one cargo invocation per crate: 5 fragments instead of 82. `command_coverage` has its own fragment so the explicit-command shards stay balanced. **Huge goes nightly:** the MoE live falsifier stays in the ledger and moves to **#4179**. It needs a nightly lane on a host that holds the model, with fail-closed behaviour; on a PR runner without the model it SKIPs and returns ok, which would be a vacuous green.

## The guard

`scripts/check_bin_cli_tests_wired.sh` checks the derived set against `scripts/bin_cli_unwired_baseline.txt` (11 rows) for an exact match. Drift in either direction is RED:

- a new dark target fails until it is wired or ledgered;
- a wired target left in the ledger fails as stale.

The ledger is registered as a `set` baseline in `scripts/check_baseline_ratchets.sh`, so it may only shrink.

- **No workflow edit.** The guard has no bare `cargo ` token and its `--help` advertises a self-test, so `guard_tree.sh --no-cargo` (guard-tree job) dispatches both modes. `check_guards_are_wired.sh` reports PASS.
- **`--self-test`** covers 14 fixture rows plus 4 ledger rows. It kills the planted mutants no-macro, name-only, no-dir-targets, no-boundary and no-continuation: `SELF-TEST OK`.

## Checks run on this branch

- `bash scripts/check_bin_cli_tests_wired.sh` passes, with 11 targets ledgered.
- `--self-test` prints SELF-TEST OK.
- `check_baseline_ratchets.sh` passes.
- `check_guards_are_wired.sh` passes (ratcheted).
- `check_explicit_test_commands.sh` passes.
- `ci_run_explicit_test_commands.sh --list` is ok.
- All five `check_roadmap_*.sh` return rc 0, measured at the committed head. `check_roadmap_fragment_required.sh` reads HEAD, so a run before the commit said nothing about it.

## Quorum round 1 (head 739d80fc4)

- **sonnet-5: FAIL, correct.** This receipt claimed all five `check_roadmap_*.sh` returned rc 0. Measured at 739d80fc4, `check_roadmap_fragment_required.sh` was rc 1: the new PMAT-4059 fragment was committed without `make roadmap-aggregate`. My rc 0 came from a run before the fragment was committed, and that guard reads HEAD. The next commit regenerates `docs/roadmaps/roadmap.yaml` and re-measures at the new head.
- **haiku-4-5: PASS.** It said the roadmap checks pass, and at 739d80fc4 that was wrong, so its PASS is recorded here, not counted as evidence. Round 2 re-runs every lane on the new head.
