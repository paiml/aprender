# PMAT-4818 (sub-ticket of PMAT-4741) receipt: perf041_report.py ported to aprender-ci-tools perf041-report

Branch batch/0702/py-port-perf041-report (base 999a7ab805, py-port-2/ci-tools).

Scope: port only. `scripts/perf041_report.py` has no caller anywhere in the tree, so no gate path
calls it (N-1 does not apply). It stays as the parity test's external validator and is deleted in a
later change, after which the test reads it from git, as with the other ports.

New: `crates/aprender-ci-tools/src/perf041_report.rs`, subcommand `perf041-report [OUT_DIR]`
(default `/tmp/perf041`). It prints the band table, the path penalty and the decomposed index, and
decides nothing. Exit 0 for a report. Exit 1 for no band, no fast c=1 band, or wherever the original
raised (ZeroDivisionError, KeyError, TypeError, ValueError, a directory or non-UTF-8 record file),
after the lines the original had already printed.

Measured on the x86 build host (CARGO_TARGET_DIR private):
- cargo test -p aprender-ci-tools --lib: 54 passed, 0 failed (19 new perf041_report tests; their golden
  strings are the original's output on the same fixtures).
- cargo clippy -p aprender-ci-tools --all-targets -- -D warnings: clean; cargo fmt: clean.
- scripts/tests/ci_tools_py_parity_test.sh: 149/149 identical (115 before + 34 new, section 6). Each new
  case compares the exact exit code, then stdout. Fixtures: full sweep, no forced band, no fast c=1
  band, empty dir, skipped records (bad JSON, `error` of every value type, BOM, hidden file, other
  extensions), int/bool medians, mode labels (empty, absent, null, false, 0, leading dash, code-point
  order), `c` in every form `int()` takes, first-extreme int/float for tokens_min/max, formatting
  (rounding ties, -0.0, overflow to inf, inf/inf = nan), four zero-divisor cases, eight inputs the
  original raised on, missing dir, a file as the dir, trailing slash, extra args, `-x`, `""` (cwd).
- Planted mutations in the bin (unknown-mode label `?` -> `!`, no-fast-band exit 1 -> 0): harness
  147/149, both flagged (modes stdout, nofast1 exit code); source restored, hash re-checked.
- cargo-mutants -f perf041_report.rs: 123 tested, 119 caught, 4 unviable, 0 missed. The first run
  missed one (deleting med's int/int arm); the test now has an int pair, 2^53+1 and 2^53+2, where
  summing exactly and summing their f64 roundings differ (...994 vs ...992; Python gives ...994).

Found while writing the divergence table: a `c` of magnitude 2^127 or more was read with a
saturating `as`, so it landed in a wrong band. It is now a stop (exit 1). Test:
`c_beyond_i128_stops_instead_of_saturating`. Python reads it exactly; the README lists this.

Known divergences, all in crates/aprender-ci-tools/README.md ("Where `perf041-report` differs"):
NaN/Infinity/1e400 literals, lone surrogates and JSON nested past 128 levels (file skipped by the
port); integers beyond 64 bits (nearest f64); `c` at or beyond 2^127 (a stop); non-UTF-8 file names
(not listed); glob metacharacters in OUT_DIR (literal); non-ASCII digits in a string `c` (a stop);
`-h`/`--help` (help); stop reasons on stderr (one line, not a traceback). No parity case covers them,
by design: each would fail parity.

## Planted contrary question (reviewers MUST answer, with file:line)
Claim: "`band_c` converts a float `c` with `f.trunc() as i128` and no bound, so a record with
`\"c\": 1e39` silently lands in the i128::MAX band instead of stopping."
Is this claim TRUE or FALSE? Cite the line and the test that decides it.

Quorum (PMAT-4818, judged head f9690c8e47, base 999a7ab805): AGREED 3/3 PASS. claude-sonnet-5-5 (x2),
claude-haiku-4-5, measured == declared; author claude-opus-5-5; degraded: same-family (agy policy: not a
tier-1 diff). All three lanes answered the planted claim FALSE. The haiku lane cited the guard at the
wrong lines (269-279); it is perf041_report.rs:164 (`Py::Float(f) if f.abs() < i128::MAX as f64`), decided
by the test `c_beyond_i128_stops_instead_of_saturating`. Artifact sha256 99fd4f41bf13d755… kept out of
tree (local paths).

Pre-push (x86 build host, f9690c8e47): cargo fmt --all -- --check rc 0; cargo test -p aprender-contracts --lib
2289 passed, 0 failed; cargo deny check advisories ok.
