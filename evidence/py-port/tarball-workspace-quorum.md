# Quorum receipt: PMAT-4789 tarball-workspace port

- Diff reviewed: `git diff eb60aef156...018759da8d` (three-dot). Original `scripts/lib/tarball_workspace.py` given alongside.
- Author model: claude-opus-5-5. Lanes are the same family, so this quorum is `degraded: same-family`. No lane ran the author's model.
- Runtime model ids below were read from each lane's transcript (`"model"` field of every assistant turn), not from the requested alias.

| Lane | Runtime model id | Verdict | Cited line (one per lane) | Planted contrary question | Lane's answer |
|------|------------------|---------|---------------------------|---------------------------|---------------|
| L1 | claude-sonnet-5-5 | FIX-FIRST at 018759da8d → APPROVE at 46057fc4f0 | `scripts/tests/ci_tools_py_parity_test.sh:206-216`: the validator was read from a commit that is not on `main` | "Python `sorted()` on `pathlib.Path` compares case-insensitively on POSIX, so the Rust sort misorders `Z-2`; confirm" | REJECTED. PurePosixPath's normcase is the identity, so the order is code-point order, which equals the byte order of `sort_unstable` (`crates/aprender-ci-tools/src/tarball_workspace.rs:130-132`) |
| L2 | claude-haiku-4-5-20251001 | APPROVE | `scripts/package_tarball_build.sh:109,205` (verified; the lane also cited 119 and 125, which are not calls; the remaining calls are at 187 and 192): every call is `\|\|` or `\|\| continue`|\|` or `\|\| continue` | "A refusal exits 1 instead of 2, and the caller distinguishes 2 from 1; confirm a caller-visible break" | REJECTED. The caller takes any non-zero status on `\|\|` and sets its own exit 2 |
| L3 | claude-sonnet-5-5 | APPROVE | `crates/aprender-ci-tools/src/tarball_workspace.rs:77-84`: the `//` root rule | "`str(pathlib.Path('//a'))` is `'/a'`, so keeping `//` is wrong; confirm" | REJECTED. CPython 3.13.1 prints `//a` for `PurePosixPath('//a')` and `/a` for `'///a'`. The port matches both |

## Finding and fix

L1's blocker: once a squash merge drops the base commit, no clone can read the deleted original from it, so the parity section would report "not measured" everywhere. The fix in 46057fc4f0 reads `tarball_workspace.py` (`86d653eb…`) and `toml_compat.py` (`9eca16fb…`) by blob id. Both blobs are identical on `main`. L1 re-reviewed the fix and returned APPROVE, citing `scripts/tests/ci_tools_py_parity_test.sh:208-210` (the pins are at 211-212 at head).

Result: 3/3 APPROVE, `degraded: same-family`.

## Pre-flight (branch at 46057fc4f0 merged with main, not pushed)

- `cargo fmt --check`, `clippy -D warnings` and `cargo test -p aprender-ci-tools`: all rc=0.
- Parity: `ci_tools_py_parity: 82/82 identical (declared 82)`.
- Planted mutant M1 (pkgs reversed): 4 MISMATCH.
- Planted mutant M2 (`path= ` in the patch line): 4 MISMATCH on the written Cargo.toml, rc=1.
- `scripts/check_package_includes.sh --self-test`: SELF-TEST PASSED, 0 FAIL rows. This included row 18, the `--name` substitution through the default `cargo run` path.
