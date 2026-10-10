# aprender-ci-tools

CI helpers ported from `scripts/**/*.py` to Rust (C301: no Python in the build).
`publish = false`, never installed. One binary, one subcommand per ported script:

| Subcommand | Ported from |
|------------|-------------|
| `publishable-crates` (reads `cargo metadata` JSON on stdin) | `scripts/lib/publishable_crates.py` |
| `package-include-diff <LISTING> <INCLUDES>` | `scripts/lib/package_include_diff.py` (kept: its caller `scripts/check_package_includes.sh` is a gate, so it switches once a released `aprender-ci-tools` carries the port, N-1) |
| `coverage-report-scope [--exclude NAME]...` | `scripts/coverage_report_scope.py` (kept: its callers, the Makefile coverage targets, `ci.sh` and `prepare-release.sh`, are gate paths, N-1; the parity harness reads it from git blob 106561a2) |
| `tarball-shrink-report <PACKAGE_LOG> <WS_DIR>` | `scripts/lib/tarball_shrink_report.py` (kept: its caller runs on a gate path, so it switches once a released `aprender-ci-tools` carries the port) |
| `tarball-workspace DIR` · `--name DIR` · `--target-dir` (reads `cargo metadata` JSON on stdin) | `scripts/lib/tarball_workspace.py` (kept: its caller `scripts/package_tarball_build.sh` is on the publish gate path, N-1; the parity test reads it from a pinned git blob) |
| `tarball-build-errors LOG` | `scripts/lib/tarball_build_errors.py` (kept: its caller `scripts/package_tarball_build.sh` is on the publish gate path, N-1) |
| `dag-status --root DIR` (reads `[id, row]` JSON pairs on stdin) | `scripts/lib/dag_status.py` (kept for now, see below) |
| `git-patch-id [--stable\|--unstable\|--verbatim]` (reads a diff on stdin) | `scripts/lib/git_patch_id.py` (kept: callers `scripts/lib/pr_review_patch_id.sh` and `scripts/check_pr_review_arm4.sh` not yet switched) |
| `llama-fit-verdict TOOL_FOUND PIN RC FREE_MIB VERSION_FILE STDOUT_FILE MODEL` | `scripts/lib/llama_fit_verdict.py` (kept: `scripts/model_ladder.sh` calls it on the certification path, and `scripts/check_model_ladder.sh` mutates copies of it) |
| `nextest-fail-fast [--reader library\|fallback] TOML` | the inline Python judge in `scripts/check_nextest_ci_profile_no_fail_fast.sh` (kept: the guard runs it inline and switches once a released `aprender-ci-tools` carries the port, N-1) |

Each port must print the same stdout as its original and agree with it on success
or failure. `scripts/tests/ci_tools_py_parity_test.sh` checks this. The Python
files stay as that test's external validator, and a gate-path caller keeps its `.py` until
a released `aprender-ci-tools` carries the port (N-1). A port whose callers are all
switched deletes its `.py`, and the test reads that original from git.

`dag-status` is checked by `scripts/tests/ci_tools_dag_status_parity_test.sh` against
the working-tree `scripts/lib/dag_status.py`. Every caller still imports the `.py`
(`session_docs_commit.sh` and `pp066_state.sh` in inline Python, `render_dag.py`,
`lib/dag_invariants.py`). The shell scripts keep main's inline Python unchanged, and the
last two run in cargo-free CI steps, which only use released tool versions. They switch,
and the `.py` is deleted, in the first change after a released `aprender-ci-tools`
carries `dag-status`.

`git-patch-id` is checked by `scripts/tests/ci_tools_git_patch_id_parity_test.sh`, which
compares the bin with the `.py` and counts those cases on its own. Native `git patch-id`
is a separate verdict there. On a git older than 2.40 only the text-only cases are
compared, and the rest print NOT_MEASURED.

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
| `tarball-build-errors --help` / `-h` | read as the LOG path: cannot read, exit 2 | help, exit 0 (every other argv, `--` and `-x` included, matches: parity cases) |

## `llama-fit-verdict` follows one interpreter

Two of the original's answers depend on which `python3` runs it: which characters
`int()`, `isdigit()` and the `\d` in its `-c N -ngl N` match accept as digits (the
interpreter's Unicode version), and how many nested GGUF arrays its recursive skip walks
before `RecursionError` turns the trained context into `null`. The port follows CPython
3.12 and 3.13, which agree on both (Unicode 15; 996 nested arrays parse, 997 do not) and
are what the ladder's hosts run. Under 3.10 or 3.14 the original itself answers those
edges differently, so the parity test refuses any interpreter but 3.12/3.13 for this
section (`LFV_PYTHON=`).

## Where `nextest-fail-fast` differs (by design; the parity test maps the first two)

`scripts/tests/ci_tools_nextest_fail_fast_parity_test.sh` cuts the judge out of the
guard and runs it under tomllib and under its purpose-built reader
(`NEXTEST_GUARD_FORCE_FALLBACK=1`), against `--reader library` and `--reader fallback`.
It needs Python 3.11+ (tomllib) and counts its cases on its own. The original took the
purpose-built reader only where `tomllib`/`tomli` were missing; `--reader` picks it here.
Every verdict, exit status and ENV/FAIL reason is the original's, except:

| Case | Original | Port |
|------|----------|------|
| the library reader's name in `reader=` | `tomllib` or `tomli` | `toml crate` |
| the purpose-built reader's name | `purpose-built reader (no tomllib/tomli on this interpreter)` | `purpose-built reader` |
| the words of a TOML parse error | tomllib's | the `toml` crate's, on one line, with tomllib's `(at line L, column C)` |
| a value tomllib would read as a datetime, or an integer outside `i64`, as `fail-fast` | its Python `repr()` | its TOML text, or a parse error |
| an inline table with two or more keys as `fail-fast`, in a FAIL line | keys in file order | keys in the `toml` crate's map order (sorted unless its `preserve_order` feature is on) |
| `repr()` of a non-printable, non-ASCII character in a FAIL line | `\x..`/`\u....` escapes | the character itself |
| an invalid UTF-8 byte past the first 8 KiB, or a truncated sequence at the end, in the purpose-built reader | the position within the chunk text mode was decoding | the position in the file |
| which characters `\d` and `int()` take as digits | the interpreter's Unicode version | the `regex` crate's |

## Where `tarball-workspace` output differs (by design; the parity test maps or skips each)

| Case | Original | Port |
|------|----------|------|
| first line of the written `DIR/Cargo.toml` | names `scripts/lib/tarball_workspace.py` | names `aprender-ci-tools tarball-workspace` (the parity test maps this one line, nothing else) |
| a refusal | exit 2 | exit 1 (parity compares success vs failure; every caller tests only that) |
| a non-string `name`/`version`/`target_directory` | printed via `str()` | refused: cargo rejects such a manifest before it is packaged |

Part of the [aprender monorepo](https://github.com/paiml/aprender).
