# aprender-ci-tools

CI helpers ported from `scripts/**/*.py` to Rust (C301: no Python in the build).
`publish = false`, never installed. One binary, one subcommand per ported script:

| Subcommand | Ported from |
|------------|-------------|
| `publishable-crates` (reads `cargo metadata` JSON on stdin) | `scripts/lib/publishable_crates.py` |
| `package-include-diff <LISTING> <INCLUDES>` | `scripts/lib/package_include_diff.py` (deleted; caller `scripts/check_package_includes.sh` switched) |
| `coverage-report-scope [--exclude NAME]...` | `scripts/coverage_report_scope.py` (deleted; parity harness reads it from git blob 106561a2) |
| `tarball-shrink-report <PACKAGE_LOG> <WS_DIR>` | `scripts/lib/tarball_shrink_report.py` (kept: its caller runs on a gate path, so it switches once a released `aprender-ci-tools` carries the port) |
| `tarball-workspace DIR` · `--name DIR` · `--target-dir` (reads `cargo metadata` JSON on stdin) | `scripts/lib/tarball_workspace.py` (deleted; the parity test reads it from git) |
| `llama-fit-verdict TOOL_FOUND PIN RC FREE_MIB VERSION_FILE STDOUT_FILE MODEL` | `scripts/lib/llama_fit_verdict.py` (kept: `scripts/model_ladder.sh` calls it on the certification path, and `scripts/check_model_ladder.sh` mutates copies of it) |

Each port must print the same stdout as its original and agree with it on success
or failure. `scripts/tests/ci_tools_py_parity_test.sh` checks this. The Python
files stay only as that test's external validator; a port whose callers are all
switched deletes its `.py`, and the test reads that original from git.

## Where the argv surface differs from the originals (by design, not parity-checked)

Arguments are parsed by clap derive (`scripts/check_no_hand_rolled_parsers.sh`), so a
usage line differs from the originals' hand-rolled loops in the ways below. Parity covers
what each subcommand prints and whether it fails on the inputs its callers pass, and
none of the callers uses these forms:

| Argv | Original | Port |
|------|----------|------|
| `--help` / `--version` | usage refusal, exit 1 | help or version, exit 0 (dogfood surface probe), except `llama-fit-verdict`, where both are arguments as they were in the original (`aprender-ci-tools help llama-fit-verdict` prints its help) |
| `llama-fit-verdict` with a non-UTF-8 argument | read with `surrogateescape`; a non-UTF-8 pin prints as `\udcXX` | usage error, exit 1 (the caller passes a hex pin and plain paths) |
| `coverage-report-scope --exclude=NAME` | usage refusal, exit 1 | same as `--exclude NAME` |
| `package-include-diff A B EXTRA` | `EXTRA` ignored | usage error, exit 1 |
| any usage error | the original's message | clap's message (stderr only; exit 1 on both) |
| `tarball-shrink-report` usage error | exit 2 | exit 1 (a missing input file still exits 2 on both) |
| `tarball-shrink-report`, a Unicode `Other_Alphabetic` character that is not a letter or digit (a combining mark, a circled letter such as `Ⓐ`) before `_or_skip(` or `fn` | not a word character | a word character |
| `tarball-workspace --name` alone | `--name` taken as DIR | usage error, exit 1 |

## `llama-fit-verdict` follows one interpreter

Two of the original's answers depend on which `python3` runs it: which characters
`int()`, `isdigit()` and the `\d` in its `-c N -ngl N` match accept as digits (the
interpreter's Unicode version), and how many nested GGUF arrays its recursive skip walks
before `RecursionError` turns the trained context into `null`. The port follows CPython
3.12 and 3.13, which agree on both (Unicode 15; 996 nested arrays parse, 997 do not) and
are what the ladder's hosts run. Under 3.10 or 3.14 the original itself answers those
edges differently, so the parity test refuses any interpreter but 3.12/3.13 for this
section (`LFV_PYTHON=`).

## Where `tarball-workspace` output differs (by design; the parity test maps or skips each)

| Case | Original | Port |
|------|----------|------|
| first line of the written `DIR/Cargo.toml` | names `scripts/lib/tarball_workspace.py` | names `aprender-ci-tools tarball-workspace` (the parity test maps this one line, nothing else) |
| a refusal | exit 2 | exit 1 (parity compares success vs failure; every caller tests only that) |
| a non-string `name`/`version`/`target_directory` | printed via `str()` | refused: cargo rejects such a manifest before it is packaged |

Part of the [aprender monorepo](https://github.com/paiml/aprender).
