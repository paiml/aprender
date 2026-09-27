---
status: complete
ticket: GH-4175
github_issue: 4175
part: "phase 2 on car/0.70.0 (branch fix/4175-on-car-f6fa): every call site on the shared helper + the packaged-tarball gate GREEN. Phase 1 (the helper itself) is already on the car; its record is kept below."
kind: code
model: claude-opus-5-5 (author)
---
# implementation receipt: GH-4175, the shared "in tree" helper

## READ FIRST: what THIS diff is (phase 2, base `origin/car/0.70.0` @f6fa4cf0c)

Phase 1 (below, "Scope" onward) landed the helper: `crates/aprender-contracts/src/tree.rs`, the macros, and the
versioned dev-deps for core/train/serve. It is already on the car, so it is NOT in this diff, and its line
"No call site is migrated in this PR" describes phase 1 only.

This diff is phase 2, the call sites and the tarball gate. The cop (aprender-77) ruled, 2026-09-27, "run the quorum
now on fix/4175-on-car-f6fa". What it does:

| commit | change |
|---|---|
| 53248ad41b (#4193) | aprender-serve's `gguf-header-slices` fixtures move from `tests/fixtures/` (the package excludes `/tests/`) into `src/fixtures/`, so the `include_bytes!` in `tokenizer_tests_unk_3609.rs` compiles from the .crate. The `package_tarball_build.sh` header lines 7-10 and `NEG_NEEDLES` still name the OLD path on purpose: they describe the published 0.69.1 .crate, the negative control the gate must see RED. |
| 57607a048f (#4129) | four crates read out-of-crate test fixtures at run time instead of `include_str!` |
| dfe3dad0ca (#4130) | cbtop/core/orchestrate/test-showcase `exclude` the integration-test targets that cannot compile from a .crate (see "Why excludes" below) |
| fcb2f2d74a, 67ff6081fc (#4149) | apr-cli tree readers skip by name out of tree and FAIL in tree; the tree-reader oracle sees `workspace_*_or_skip!` callers |
| 71888b094b (#4130) | aprender-core unit tests read repo-root contracts at run time (`test_support.rs`) |
| 9f390a350f | aprender-serve `fusion_call_site_guard_3985.rs` reads `kernel-fusion-v1.yaml` through `workspace_path_or_skip!` |
| b331a306b7 | `scripts/tree_reader_tests.txt` / `_unwired_baseline.txt` regenerated with `check_tree_reader_tests.sh --update` |
| cec92504fa | **quorum round 1 fix**: the eight sites that still decided "in tree" by a local `contracts/.is_dir()` (the rule the phase-1 ruling forbids, mutant M3) now call the shared macros: aprender-contracts `schema::workspace_contract_or_skip`, aprender-core `test_support`, aprender-train `llama_370m`, aprender-present-terminal, `pv_surface_gate.rs`, apr-cli `golden_output.rs` + `thinking_budgets_mirror.rs`, aprender-orchestrate `chat_template.rs`. present-terminal and orchestrate gain the versioned `provable-contracts = { workspace = true }` dev-dep (aprender-contracts' normal closure is `{-macros}`: no cycle). |

No local copy is left: `grep -rn --include='*.rs' 'contracts").is_dir()' crates/` → no match (after merging the moved car at 04db69c29, which brought one more copy in
`aprender-present-terminal/src/ptop/mod.rs`; that site is migrated too, in the commit after the merge).

### Why excludes (#4130), not skips

A target listed in `exclude` either has a PATH-ONLY sibling dev-dependency, which `cargo package` strips and which
PMAT-955 / preflight R6 forbid versioning (the 0.65.0 publish cycle), or it `include_str!`s a file outside its crate
(repo-root `contracts/`, `CLAUDE.md`, `LICENSE`, apr-cli sources). Such a target cannot COMPILE from the .crate, so
no run-time skip can reach it. It still runs in the workspace, and CI runs the same targets as before. The tarball
gate tests what is published, and what is published now compiles. A target that could be made to compile is
migrated instead, which is the whole of #4129/#4149.

### Measured at cec92504fa (clean committed tree, private target `/mnt/nvme-raid0/cargo-targets/6b-4175car`)

```
cargo fmt --all -- --check                                    rc 0
scripts/check_tree_reader_tests.sh                            rc 0  (163 registered, 40 unwired baseline)
cargo test -p aprender-contracts --lib schema::               rc 0  (140 passed)
cargo test -p aprender-core --lib test_support                rc 0  (incl. in_tree_missing_file_panics)
cargo test -p aprender-core --lib setfit::                    rc 0
cargo test -p aprender-train --lib llama_370m                 rc 0
cargo test -p aprender-present-terminal --lib design_principles rc 0
cargo test -p aprender-contracts-cli --test pv_surface_gate every_rule  rc 0
cargo test -p apr-cli --test thinking_budgets_mirror          rc 0
cargo test -p aprender-orchestrate --lib a_real_gguf_renders  rc 0
cargo test -p apr-cli --lib golden_output                     33/35 ok, then SIGKILL. The 2 unfinished tests
    (the_golden_stop_token_is_reachable_in_every_model_we_have, a_qwen_model_still_stops_on_151645_…) load every
    local GGUF and do not call the helper; all 4 helper callers are among the 33 that passed.
cargo clippy -p {core,contracts,present-terminal,train,orchestrate,apr-cli} --lib -- -D warnings   rc 0
    (--lib --tests: no finding in any touched file; pre-existing findings elsewhere, e.g. spec_checklist_t_realizar.rs)
git status --porcelain Cargo.lock after the builds           empty
scripts/package_tarball_build.sh                              rc 0  "PASS  all 73 published tarball(s) compile their tests"
    (SHRINK: 391 integration targets not shipped across 21 crates; 82 run-time skip sites across 8 crates)
```


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
