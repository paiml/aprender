# aprender-test (probar) — upstream documentation archive

This directory holds the documentation that used to live in `crates/aprender-test/`.

## What `crates/aprender-test/` actually was

It was **not a crate**. Its `Cargo.toml` had no `[package]` section at all — it was
the vendored *workspace root* of the upstream `paiml/probar` repository, checked in
whole during the APR-MONO consolidation. It declared four members:

```toml
[workspace]
members = ["crates/probar", "crates/probar-derive", "crates/probar-cli", "crates/probar-js-gen"]

probar = { version = "=1.0.3", path = "crates/probar" }
```

Those paths resolved to `crates/aprender-test/crates/probar/…`, which never existed
in this repo — the member crates had already been relocated one level up. Because a
directory with no `[package]` cannot be a workspace member, the directory was placed
in the root `Cargo.toml` `exclude` list, where it sat unbuildable, referenced by
nothing, for the life of the monorepo. It was resolved in #2470.

## Where the code went

Every upstream member exists today as a first-class workspace crate:

| upstream member | monorepo crate | notes |
|---|---|---|
| `probar` | `crates/aprender-test-lib` | package `aprender-test-lib`, **`[lib] name = "jugar_probar"`** |
| `probar-derive` | `crates/aprender-test-derive` | |
| `probar-cli` | `crates/aprender-test-cli` | ships the `aprender-test-cli` binary |
| `probar-js-gen` | `crates/aprender-test-js-gen` | |
| `showcase-calculator` | `crates/aprender-test-showcase` | |

Nothing in the deleted directory was the only copy of any source code. Its single
`.rs` file, `src/generated_contracts.rs`, was an orphaned `pv codegen` artifact: it
was included by no crate root (there was no `lib.rs`), and it was a stale 25,663-line
snapshot of a file that is regenerated per-crate — the current generation lives in
`crates/aprender-test-lib/src/generated_contracts.rs` (26,823 lines).

## What was preserved here, and what was dropped

**Preserved** (had no counterpart anywhere else in the tree):

- `book/` — the upstream mdbook user guide for probar (60 chapters). This is the
  only long-form documentation of the ~211k-line engine now in `aprender-test-lib`.
  It is **not** wired into `book.yml`; the repo book only carries `apr probar` and a
  TUI case study.
- `qa/` — falsification reports and QA checklists from upstream development.
- `pmat-tickets/`, `roadmaps/`, `assets/`, `CHANGELOG.md`, `UPSTREAM-README.md`.

**Dropped** (duplicated or non-content):

- `docs/specifications/*.md` — all 17 files were already present, **byte-identical**,
  at `docs/specifications/aprender-test/`. Verified with `cmp` before deletion.
- The vendored upstream workspace scaffolding: `Cargo.toml`, `Cargo.lock`, `deny.toml`,
  `LICENSE`, `CLAUDE.md`, `.github/`, `scripts/`, `docker/`, `.config/`, `.pmat-*`.
- Tracked build artifacts that should never have been committed: `lcov.info` (1.6 MB),
  `test_output.txt` and `parallel_test_output.txt` (162 KB each), `qa_report.txt`,
  `ignored_test_output.txt`, `examples_test_output.txt`.

Total removed: ~5.3 MB. Everything remains recoverable from git history at
`79b06ae78:crates/aprender-test/`.

## Caveat: this documentation is stale

The book documents a `probar` CLI binary that no longer exists under that name. The
engine is reachable today as the `aprender-test-cli` binary, and the `apr probar`
subcommand currently exposes only `tensor`. Treat the book as a description of the
engine's capabilities, not as accurate invocation instructions.
