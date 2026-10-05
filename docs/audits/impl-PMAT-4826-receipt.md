# PMAT-4826 (sub-ticket of PMAT-4741) receipt: extract_book_examples.py ported to aprender-ci-tools extract-book-examples

Branch batch/0702/py-port-extract-book-examples (base 999a7ab805, py-port-2/ci-tools).

Scope: port only. `scripts/extract_book_examples.py` is run by `scripts/extract-book-examples.sh`,
which feeds `check_book_examples_executable.sh` (run by `dogfood-book.sh`) and
`_build_rust_compile_test.py`. Those callers are not switched (N-1: a gate runs a tool only from an
already released crate); the wrapper keeps its .py until a released `aprender-ci-tools` carries the
port. The .py is also the parity test's external validator.

New: `crates/aprender-ci-tools/src/extract_book_examples.rs`, subcommand
`extract-book-examples [ROOT]` (default `.`). Like the original, it prints one `json.dumps` line per
```` ```bash ```` / ```` ```rust ```` block in `ROOT/book/src/{cli,lib}/*.md` (`sorted(glob)` order:
code points of the `surrogateescape` name, dot-files, directories and dangling links named `.md`
included), with the cost class and model read from the last non-blank of the two lines above the
fence (default `trivial`), text-mode newlines, `str.splitlines` breaks, and an unclosed fence ending
the chapter. Exit 0, or exit 1 at the first chapter that cannot be read, is not UTF-8 or holds an
empty cost line, after the records already printed (that chapter's own records are not printed, as
the original builds a chapter's list before printing it).

Measured on the x86 build host (CARGO_TARGET_DIR private):
- cargo test -p aprender-ci-tools --lib: 45 passed, 0 failed (10 new extract_book_examples tests).
- cargo clippy -p aprender-ci-tools --all-targets -- -D warnings: clean; cargo fmt: clean.
- pmat complexity: max cyclomatic 4, max cognitive 15 per function.
- scripts/tests/ci_tools_py_parity_test.sh (PYTHON=python3.13): 136/136 identical (115 before + 21
  new, section 8). Each case builds one book tree, copies the .py into its `scripts/` and points the
  port at the same tree; `check_exact` compares stdout, the exact exit status, and stderr empty on
  both sides or on neither (its text is not compared: the original prints a traceback). Liars, one
  per comparison (stdout, success, exact status, stderr present, stderr absent), each paired with a
  twin that must pass. Fixtures: no book, cli as a file, empty dirs, the live book, every cost class,
  model forms, cost-line forms and the two-line look-back, fence forms, CRLF/CR/Unicode line breaks,
  JSON escaping (C0, DEL, non-BMP, `\r`), file order and names (dot-files, `c.MD`, newline and `\r`
  in a name, non-ASCII, invalid UTF-8 names), lib only, BOM, invalid UTF-8 content, an encoded
  surrogate, an empty cost line (read, and one no fence looks back to), a directory and a dangling
  link named `.md`, a link to a chapter, an unreadable chapter.
- Planted mutations in the bin: default cost `trivial` -> `trivia` (harness 120/136) and a stop's
  exit status 1 -> 2 (130/136, caught only by the exact-status comparison). Source restored, hash
  re-checked.
- cargo-mutants -f extract_book_examples.rs (lib tests): 79 tested, 74 caught, 3 timeouts, 2 missed.
  `json_str`'s `\r` arm was a real gap: a unit case now escapes `\r`, and a re-run on `json_str`
  caught 20 of 20. The other is equivalent: `i = k + 1` -> `k * 1` after a block resumes the scan on
  the closing fence line, which `fence_lang` never takes, so the next line is the same.

Known divergences, all in crates/aprender-ci-tools/README.md ("Where `extract-book-examples`
differs"): which book (the script's own repo vs ROOT); arguments (ignored by the original; ROOT,
help or a usage error in the port); stop reasons on stderr (one line, not a traceback). No parity
case covers them, by design.

## Planted contrary question (reviewers MUST answer, with file:line)
Claim: "A block whose cost line is two lines above the fence, with a blank line between, is
reported as `trivial`, because the port only looks at the line directly above the fence."
Is this claim TRUE or FALSE? Cite the line and the test that decides it.
