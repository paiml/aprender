| `complexity-rows -- JSON` | `--` is a path: cannot open, exit 1 | clap's end-of-options marker, dropped (`check_complexity_ratchet.sh` passes pmat outputs only; `-x` and `--help` are paths on both, parity cases) |
| `complexity-rows`, a JSON integer beyond 64 bits or a float metric whose integer part is beyond `i128`, the `NaN` / `Infinity` literals or a lone surrogate escape | read by `json.load` | a refusal, exit 1 (pmat writes none of them) |
| `complexity-rows`, non-ASCII decimal digits in a threshold or a string metric | `int()` accepts them | refused (exit 2 for a threshold, 1 for a metric) |
| `complexity-rows [JSON]...` (thresholds from `CX_MAX_CYCLOMATIC` / `CX_MAX_COGNITIVE`) | `scripts/lib/complexity_rows.py` (kept: its caller `scripts/check_complexity_ratchet.sh` is a gate, N-1) |
# aprender-ci-tools

CI helpers ported from `scripts/**/*.py` to Rust (C301: no Python in the build).
`publish = false`, never installed. One binary, one subcommand per ported script:

| Subcommand | Ported from |
|------------|-------------|
| `publishable-crates` (reads `cargo metadata` JSON on stdin) | `scripts/lib/publishable_crates.py` |
| `package-include-diff <LISTING> <INCLUDES>` | `scripts/lib/package_include_diff.py` (kept: its caller `scripts/check_package_includes.sh` is a gate, so it switches once a released `aprender-ci-tools` carries the port, N-1) |
| `coverage-report-scope [--exclude NAME]...` | `scripts/coverage_report_scope.py` (kept: its callers, the Makefile coverage targets, `ci.sh` and `prepare-release.sh`, are gate paths, N-1; the parity harness reads it from git blob 106561a2) |
| `cascade-universe [--names] [REPO_ROOT]` (runs `cargo metadata` from PATH, once per workspace) | `scripts/lib/cascade_universe.py` (kept: its callers, the Makefile, `ci.yml` and the `cascade-*`/`check_cascade_*`/`check_publish_safety.sh` scripts, are on the publish gate path, N-1) |
| `tarball-shrink-report <PACKAGE_LOG> <WS_DIR>` | `scripts/lib/tarball_shrink_report.py` (kept: its caller runs on a gate path, so it switches once a released `aprender-ci-tools` carries the port) |
| `tarball-workspace DIR` · `--name DIR` · `--target-dir` (reads `cargo metadata` JSON on stdin) | `scripts/lib/tarball_workspace.py` (kept: its caller `scripts/package_tarball_build.sh` is on the publish gate path, N-1; the parity test reads it from a pinned git blob) |
| `tarball-build-errors LOG` | `scripts/lib/tarball_build_errors.py` (kept: its caller `scripts/package_tarball_build.sh` is on the publish gate path, N-1) |
| `dag-status --root DIR` (reads `[id, row]` JSON pairs on stdin) | `scripts/lib/dag_status.py` (kept for now, see below) |
| `git-patch-id [--stable\|--unstable\|--verbatim]` (reads a diff on stdin) | `scripts/lib/git_patch_id.py` (kept: callers `scripts/lib/pr_review_patch_id.sh` and `scripts/check_pr_review_arm4.sh` not yet switched) |
| `perf041-report [OUT_DIR]` (default `/tmp/perf041`; a report, decides nothing) | `scripts/perf041_report.py` (kept as the parity test's validator; it has no caller in the tree, so no gate path) |
| `annotate-book-examples [ROOT]` (default `.`; rewrites `ROOT/book/src/{cli,lib}/*.md` in place) | `scripts/annotate-book-examples.py` (kept as the parity test's validator; it has no caller in the tree, so no gate path) |
| `extract-book-examples [ROOT]` (default `.`; JSON lines for the bash/rust blocks in `ROOT/book/src/{cli,lib}/*.md`) | `scripts/extract_book_examples.py` (kept: its wrapper `scripts/extract-book-examples.sh` feeds `check_book_examples_executable.sh`, which `dogfood-book.sh` runs, and `_build_rust_compile_test.py`, so it switches once a released `aprender-ci-tools` carries the port, N-1) |
| `llama-fit-verdict TOOL_FOUND PIN RC FREE_MIB VERSION_FILE STDOUT_FILE MODEL` | `scripts/lib/llama_fit_verdict.py` (kept: `scripts/model_ladder.sh` calls it on the certification path, and `scripts/check_model_ladder.sh` mutates copies of it) |

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
| `cascade-universe` with JSON that `json.loads` takes and serde refuses (`NaN`, `Infinity`, a lone surrogate, UTF-16/32, a BOM) | parsed | exit 1 (cargo never prints these) |
| `cascade-universe`, a non-string package `name` | hashed: `1`, `1.0` and `true` are one key | exit 1 (cargo names are strings) |
| `cascade-universe`, invalid UTF-8 on cargo's stderr in the exit-2 message | Python `replace` decoding | Rust lossy decoding (both print U+FFFD; the count per bad sequence may differ) |

## Where `perf041-report` differs (by design, not parity-checked)

The port reads records as Python's `json` does (`NaN`, `Infinity`, `-Infinity`, `1e400` as
`inf`, integers beyond 64 bits exactly) and compares and divides them as Python does; parity
rows cover each. Where the original computes something the port cannot carry, the port
STOPS with exit 1 and a `not supported: ...` line on stderr, after the lines already printed.
It never prints a different answer.

| Input | Original | Port |
|-------|----------|------|
| an integer of magnitude 2^127 or more anywhere in a record (a `c` string too) | read exactly | a stop, exit 1 |
| a lone surrogate escape (`"\ud800"`) in a record | parsed (it fails only if printed) | a stop, exit 1 |
| JSON nested more than 512 levels | parsed (to Python's recursion limit) | a stop, exit 1 |
| a median of 3 or more replicates with a `NaN` among them | depends on where the NaN sits in the sort | a stop, exit 1 |
| two integers beyond 2^53 divided (`int / int`), or `c * agg_tok_s` of 2^127 or more | exact, rounded once | a stop, exit 1 |
| a record file name that is not UTF-8, printed in a skip line | depends on the locale's stdout error handler | a stop, exit 1 (read and sorted as Python does otherwise: parity row) |
| glob metacharacters (`*?[`) in `OUT_DIR` | expanded by `glob` | taken literally (unit test `out_dir_is_a_directory_not_a_pattern`) |
| non-ASCII Unicode digits in a string `c` (`"١"`) | read by `int()` | a stop, exit 1 |
| `-h` / `--help` | taken as `OUT_DIR` | help, exit 0 |
| the reason for a stop (stderr) | a Python traceback | one line naming the exception |

## Where `annotate-book-examples` differs (by design, not parity-checked)

| Input | Original | Port |
|-------|----------|------|
| which book it rewrites | the one in its own repository (the script's grandparent directory) | `ROOT`, default the current directory |
| any argument (`--help`, a path, anything) | ignored: the book is rewritten | `-h`/`--help` print help, one path is `ROOT`, more are a usage error (exit 1) |
| a chapter file name (one ending `.md`) that is not UTF-8 | processed (sorted by its surrogate-escaped name); if it gained an annotation, printing its name crashes under a UTF-8 locale after the file is rewritten, and prints the raw bytes under the C locale (UTF-8 mode) | a stop before any file is touched, exit 1 |
| the reason for a stop (stderr) | a Python traceback | one line naming the file and the error |

## Where `extract-book-examples` differs (by design, not parity-checked)

| Input | Original | Port |
|-------|----------|------|
| which book | the one under the script's own repo | the one under `ROOT` |
| arguments | ignored (even `--help`) | `ROOT`, or help / a usage error |
| the reason for a stop (stderr) | a Python traceback | one line naming the chapter and the reason |

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
