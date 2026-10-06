# PMAT-4797 (sub-ticket of PMAT-4741) / IDLE-1130 receipt: tarball_build_errors.py ported to aprender-ci-tools

Code commit 211e3adbd7 on batch/0702/py-port-tarball-build-errors (base 22b4cf2ed9, py-port-2/ci-tools).

Scope: port only. The .py and both callers stay: scripts/package_tarball_build.sh:213 is on the
publish / clean-room gate path, and gate tools ship one train earlier (N-1). The other caller,
check_package_includes.sh:373-374, is a self-test of the same .py.

Measured on the x86 build host (CARGO_TARGET_DIR private):
- cargo test -p aprender-ci-tools: 19 passed, 0 failed
- cargo clippy -p aprender-ci-tools --all-targets -- -D warnings: clean; cargo fmt --all -- --check: rc 0
- scripts/tests/ci_tools_py_parity_test.sh: 78/78 identical (54 before + 24 new). Each new case compares
  the EXACT exit code (0/1/2/3/4) before stdout: clean, newline-only, empty, missing, directory, no args,
  two args, `-x`, `--`, host-beats-crate, 7 host patterns (cap 5), longest prefix with an rc version,
  no owner, cap 20, failed only, unowned only (rc 3), cap 10, \r \x0b \x0c \x1c breaks, U+2028 / U+0085
  breaks, invalid UTF-8, \x1f in a path, error[E-1] / error[] codes, nested pkgs/, non-ASCII crate.
- Planted mutations (cap 20->19, host exit 4->1) in the bin: harness 75/78, all three flagged.
- cargo-mutants -f tarball_build_errors.rs: 25 tested, 21 caught, 4 unviable (Default for Outcome), 0 missed.
  An earlier draft left `>`/`>=` in `owner` as an equivalent mutant; owner now uses max_by_key (the
  doc comment says why ties cannot occur), and two unit tests were added for the && and || survivors.
- PMAT pre-commit gate: passed after judge() was split (it was cyclomatic 30).

Known divergence, documented in crates/aprender-ci-tools/README.md: `tarball-build-errors --help` / `-h`
prints help (exit 0); the original reads it as the LOG path (exit 2). No caller passes it.

## Planted contrary question (reviewers MUST answer, with file:line)
Claim: "The port uses Rust's `\S` in DIAG, so a diagnostic whose file path holds `\x1f` is attributed to
its crate by the port but not by the Python original, and no parity case covers it."
Is this claim TRUE or FALSE? Cite the regex line and the parity case that decides it.

Quorum (PMAT-4797, judged head 211e3adbd7, base 22b4cf2ed9): AGREED 3/3 PASS. claude-sonnet-5-5 (x2),
claude-haiku-4-5, measured == declared; author claude-opus-5-5; degraded: same-family (agy policy: not a
tier-1 diff). All three lanes answered the planted claim FALSE, citing
crates/aprender-ci-tools/src/tarball_build_errors.rs:16 (`[^\s\x1c-\x1f]+?`) and the "unit separator in
path" parity case. Artifact sha256 95c29df12d347f2e… kept out of tree (local paths).

Pre-push (x86 build host, 211e3adbd7): cargo fmt --all -- --check rc 0; cargo test -p aprender-contracts --lib
2289 passed, 0 failed; cargo deny check advisories ok.
