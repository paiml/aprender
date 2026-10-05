# PMAT-4822 (sub-ticket of PMAT-4741) receipt: annotate-book-examples.py ported to aprender-ci-tools annotate-book-examples

Branch batch/0702/py-port-annotate-book-examples (base 999a7ab805, py-port-2/ci-tools).

Scope: port only. `scripts/annotate-book-examples.py` has no caller anywhere in the tree (no script,
Makefile, workflow or doc runs it), so no gate path calls it (N-1 does not apply). It stays as the
parity test's external validator and is deleted in a later change, after which the test reads it
from git, as with the other ports.

New: `crates/aprender-ci-tools/src/annotate_book_examples.rs`, subcommand
`annotate-book-examples [ROOT]` (default `.`). Like the original, it inserts
`<!-- example-cost: ... -->` above every bash/rust fence in `ROOT/book/src/{cli,lib}/*.md` that has
none within the two lines above, and rewrites every listed chapter in text mode (so `\r\n`, a lone
`\r` and every other `str.splitlines` break become `\n`, and a trailing blank line is lost, as
before). Exit 0, or exit 1 at the first chapter that cannot be read, is not UTF-8 or cannot be
written, after the lines already printed (earlier chapters stay rewritten, later ones untouched).

Two parts of the original are not ported, on purpose: `needs_annotation()` (never called) and the
`apr help <command>` branch (a first line it calls trivial starts `apr<space>help`, so the subcommand
read next starts `help`, which no list names, and is trivial too; the unit and parity tables both
hold `apr help run`, `apr helpx` and `apr help` + a combining mark).

Measured on the x86 build host (CARGO_TARGET_DIR private):
- cargo test -p aprender-ci-tools --lib: 46 passed, 0 failed (11 new annotate_book_examples tests).
- cargo clippy -p aprender-ci-tools --all-targets -- -D warnings: clean; cargo fmt: clean.
- pmat complexity (pre-commit hook ON): max cyclomatic 7, max cognitive 10 per function. The first
  commit attempt was refused by the hook (annotate_text cognitive 31 > 30); the look-back, the
  annotation text, the subcommand parse and the model lookup are now their own functions, and every
  result below was re-measured on that code.
- scripts/tests/ci_tools_py_parity_test.sh (PYTHON=python3.13): 138/138 identical (115 before + 23
  new, section 7; section 6 is perf041-report on its own branch). The .py rewrites its own repo, so
  each case builds the same tree twice, runs the .py from one copy's scripts/ and the port on the
  other, compares stdout and success/failure, then `diff -r` of the two rewritten trees. Fixtures:
  no book, cli as a file, empty dirs, the live book (188 chapters) as is and with every annotation
  stripped, every cost class, model-name forms, prompts and Unicode/C0 whitespace, --help/--version
  and `apr help`, CRLF/CR/Unicode line breaks, trailing newlines, the two-line look-back, cost-line
  forms, fence forms (incl. a closing fence that names a language), file order and names (dot-files,
  `.md`, `c.MD`, a newline, non-ASCII), lib only, BOM, invalid UTF-8 mid-run, an encoded surrogate,
  a directory and a dangling link named `.md`, a link to a chapter, a read-only chapter.
- Planted mutations in the bin (cost line accepts `>`; trailing newline never restored): harness red
  on both (137/138 and 119/138), the second only by the tree comparison, so that check can
  fail. Source restored, hash re-checked (bd9a61fb…).
- cargo-mutants -f annotate_book_examples.rs (lib tests): 71 tested, 71 caught, 0 missed, 0 timeout,
  0 unviable.

Known divergences, all in crates/aprender-ci-tools/README.md ("Where `annotate-book-examples`
differs"): which book (the script's own repo vs ROOT); arguments (ignored by the original, so even
`--help` rewrites the book; help or a usage error in the port); a non-UTF-8 chapter name (a stop
before any file is touched); stop reasons on stderr (one line, not a traceback). No parity case
covers them, by design.

## Planted contrary question (reviewers MUST answer, with file:line)
Claim: "The port drops the original's `apr help <command>` check, so a fence whose first line is
`apr help run` is now annotated `model-required` with the default model instead of `trivial`."
Is this claim TRUE or FALSE? Cite the line and the test that decides it.
