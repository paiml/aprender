---
status: complete
ticket: GH-4175
github_issue: 4175
part: the shared in-tree helper + the dev-dep fix (one criterion of the umbrella; the rest are 3a's rows)
kind: code
model: claude-opus-5-5 (author)
---
# implementation receipt: GH-4175, the shared "in tree" helper

## Scope: which part of the umbrella this PR is

#4175 ("clean-room B2: packaged-tarball test gate, all crates") is aprender-3a's umbrella. It covers several rows.
This PR delivers ONE of its acceptance criteria and nothing else:

> Both-directions proof for the shared helper: out of tree it skips; in tree with the file removed it FAILs.

The other criteria are separate rows, owned by 3a: the tarball run step in `package_tarball_build.sh`, the
nightly `mode_b_tarball` in infra, RED measured on v0.69.1, and GREEN on the fix stack. They are not in this
diff, and they are not claimed.

The cop (aprender-cf) ruled on this on 2026-09-24. This is the cop's ruling, not an operator quotation:
- Deciding "in tree" by `contracts/.is_dir()` SKIPS when a real checkout lacks `contracts/`, and that breaks the
  both-directions proof. It must be: `../../Cargo.toml` exists AND contains `[workspace]`.
- ONE shared helper, reused by 3a's #4149 sites.
- A case table: in-tree with contracts/ → run; in-tree without contracts/ → FAIL; tarball → skip.
- FIX THE DEV-DEP, no local copies. For aprender-train, -core and -serve, aprender-contracts becomes a
  `{ workspace = true }` versioned dev-dep, after checking publish order and cycles. On a cycle: stop and report.
- Landing the helper first is fine. The sites follow.

3a added one required condition, which I accepted: the tarball gate unpacks crates into `<ws>/pkgs/<name>-<ver>/`
under a generated `[workspace]` manifest. So `[workspace]` alone would read IN TREE there. The helper therefore
also requires that `root/crates/<this dir name>` canonicalizes to the manifest dir.

## What the diff does

| file | change |
|---|---|
| `crates/aprender-contracts/src/tree.rs` (new) | `workspace_root_of`, `workspace_path_or_skip_at` and `workspace_file_or_skip_at`, plus two `#[macro_export]` macros (`workspace_path_or_skip!` and `workspace_file_or_skip!`) that pass the CALLER's `CARGO_MANIFEST_DIR`. Out of tree, it prints `SKIP <test>: out of tree …` on stderr and returns `None`. In tree, a missing or unreadable file panics. |
| `crates/aprender-contracts/src/lib.rs` | `pub mod tree;` |
| `crates/aprender-{core,train,serve}/Cargo.toml` | The `provable-contracts` dev-dep changes from path-only to `{ workspace = true }` (versioned alias). |
| `docs/roadmaps/…` | GH-4175 fragment, `kind:code`. |

No call site is migrated in this PR. The existing `*_or_skip` sites are on the unmerged #4129/#4140 branches,
and they move onto this helper after it lands, per the cop's ordering.

## Measured

Tests ran on gx10 from a clean worktree at the pushed SHA:
```
cargo test -p aprender-contracts --lib tree::   -> 3 passed (case_table, the decoy row, the macro test)
```
- The case table has 7 rows. There is also a separate decoy row: a same-named `crates/foo` that is a different directory.
- The macro test printed no SKIP, so it ran IN TREE.

Mutants (each one planted with an exact-string replace and restored with `git checkout`):
| mutant | result |
|---|---|
| M1: drop the `listed != me` check | RED, the decoy row. It SURVIVED before the decoy row existed, and that row was added for it. |
| M2: `[workspace]` check always true | RED, "parent manifest without [workspace]" and "[workspace] only in a comment" |
| M3: restore the old `contracts/.is_dir()` rule | RED, "in tree, contracts/ absent: want FAIL, got skip" |

The dev-dep, checked BEFORE the change:
- aprender-contracts' normal-dep closure is `{aprender-contracts-macros}`. None of core/train/serve/present-terminal
  are in it, so there is no cycle. (PMAT-955's cycle was test-lib → core. That does not happen here.)
- `scripts/release/publish-order.txt`: aprender-contracts is at line 22, before present-terminal (27), core (44),
  serve (56) and train (60).
- `cargo package -p {aprender-train,aprender-core,aprender-serve} --list` → rc 0 for all three.
- `cargo package -p aprender-train --no-verify` → the packaged manifest keeps
  `[dev-dependencies.provable-contracts] version = "0.69.0"`, `package = "aprender-contracts"`.

Lint: `cargo clippy -p aprender-contracts --lib --tests -- -D warnings` rc 0, and `rustfmt --check tree.rs` rc 0.

## Follow-up commit 718d72c58: trailing comma (stacked on #4183 at cfc55e447)

aprender-3a measured this on the #4149 branch after running `cargo fmt`. When a call is too long for one line,
rustfmt breaks it across lines and appends a trailing comma. Both macro arms rejected that with `error: no rules
expected ','` (5 sites; `registry_failure_catalogue` and `beat_apr_sibling_cli_reach` did not compile). Both arms
now take `($test:expr, $rel:expr $(,)?)`. The new test `the_macros_accept_the_trailing_comma_rustfmt_adds` calls
each macro in that multi-line, trailing-comma shape.

Measured on gx10 at 718d72c58 (a clean checkout of the pushed SHA, private target, deleted afterwards):
```
cargo test -p aprender-contracts --lib tree::   -> 4 passed
mutant: both arms back to `($test:expr, $rel:expr)` -> rc 101, "error: no rules expected `,`"
```
`rustfmt --check tree.rs` rc 0. #4183 itself is unchanged; its armed head stays cfc55e447.
