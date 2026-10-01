---
type: defect
severity: high
created: 2026-08-17
found_in: phase-05-wave-3-post-merge-gate
resolves_phase:
---

# `make test` cannot run a single test — `aprender-profile` example fails to compile

## What

`make test` (i.e. `cargo test --no-run --workspace`) exits 101 during compilation.
Because the failure happens at `--no-run`, **zero tests in the entire workspace
execute**. The project's standard test gate currently reports nothing.

```
error[E0432]: unresolved import `renacer::validate`
  --> crates/aprender-profile/examples/validate_golden_trace.rs:10:14
error[E0282]: type annotations needed
  --> crates/aprender-profile/examples/validate_golden_trace.rs:38:36
error: could not compile `aprender-profile` (example "validate_golden_trace") due to 2 previous errors
```

## Evidence it is pre-existing

Reproduced at commit `f02a8aadfec04bdc5d175c977c0c2be01cb3c051` (the base Phase 5
Wave 3 forked from) in a detached control worktree:

```
BASE_COMMIT_EXAMPLE_BUILD_EXIT=101   # same two errors, same lines
```

`crates/aprender-profile/examples/validate_golden_trace.rs` was last touched by
`81c918b58` (2026-04-06, "subtree: merge renacer as crates/aprender-profile").
The example still imports `renacer::validate::*`, but after the subtree merge the
crate is `aprender-profile`, and the `validate` module it expects is not exposed
at that path.

## Why it matters

This is the failure mode CLAUDE.md's Verification Discipline §5 names directly: a
gate that does not evaluate is theater. Any green `make test` claim made while
this is broken would be meaningless — not because tests passed, but because the
runner never reached them. It also masks every genuine regression in all 78
workspace crates.

## Suggested fix

Either repair the import to the post-subtree path, or delete/`#[cfg]`-gate the
example if `renacer::validate` no longer exists in-tree. Confirm with:

```bash
cargo build -p aprender-profile --example validate_golden_trace   # must exit 0
make test                                                         # must reach test execution
```

A guard that fails when `cargo test --no-run --workspace` exits non-zero for a
*compile* reason would stop this recurring.
