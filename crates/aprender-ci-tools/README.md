# aprender-ci-tools

CI helpers ported from `scripts/**/*.py` to Rust (C301: no Python in the build).
`publish = false`, never installed. One binary, one subcommand per ported script:

| Subcommand | Ported from |
|------------|-------------|
| `publishable-crates` (reads `cargo metadata` JSON on stdin) | `scripts/lib/publishable_crates.py` |
| `package-include-diff <LISTING> <INCLUDES>` | `scripts/lib/package_include_diff.py` |
| `coverage-report-scope [--exclude NAME]...` | `scripts/coverage_report_scope.py` |
| `dag-status --root DIR` (reads `[id, row]` JSON pairs on stdin) | `scripts/lib/dag_status.py` (deleted; its callers call the binary) |

Each port must print the same stdout as its original and agree with it on success
or failure. `scripts/tests/ci_tools_py_parity_test.sh` checks this. The Python
files stay only as that test's external validator. `dag-status` is checked by
`scripts/tests/ci_tools_dag_status_parity_test.sh`, which takes its original from git.

## Where the argv surface differs from the originals (by design, not parity-checked)

Arguments are parsed by clap derive (`scripts/check_no_hand_rolled_parsers.sh`), so a
usage line differs from the originals' hand-rolled loops in four ways. Parity covers
what each subcommand prints and whether it fails on the inputs its callers pass, and
none of the callers uses these forms:

| Argv | Original | Port |
|------|----------|------|
| `--help` / `--version` | usage refusal, exit 1 | help or version, exit 0 (dogfood surface probe) |
| `coverage-report-scope --exclude=NAME` | usage refusal, exit 1 | same as `--exclude NAME` |
| `package-include-diff A B EXTRA` | `EXTRA` ignored | usage error, exit 1 |
| any usage error | the original's message | clap's message (stderr only; exit 1 on both) |

Part of the [aprender monorepo](https://github.com/paiml/aprender).
