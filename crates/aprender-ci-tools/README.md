# aprender-ci-tools

CI helpers ported from `scripts/**/*.py` to Rust (C301: no Python in the build).
`publish = false`, never installed. One binary, one subcommand per ported script:

| Subcommand | Ported from |
|------------|-------------|
| `publishable-crates` (reads `cargo metadata` JSON on stdin) | `scripts/lib/publishable_crates.py` |
| `package-include-diff <LISTING> <INCLUDES>` | `scripts/lib/package_include_diff.py` |
| `coverage-report-scope [--exclude NAME]...` | `scripts/coverage_report_scope.py` |

Each port must print the same stdout as its original and agree with it on success
or failure. `scripts/tests/ci_tools_py_parity_test.sh` checks this. The Python
files stay only as that test's external validator.

Part of the [aprender monorepo](https://github.com/paiml/aprender).
