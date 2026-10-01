# Coding Conventions

**Analysis Date:** 2026-08-07

## Naming Patterns

**Files:**
- Use `snake_case.rs` for Rust modules and keep the module name aligned with the file or directory name, as in `crates/aprender-core/src/linear_model/mod.rs`, `crates/aprender-core/src/linear_model/elastic_net.rs`, and `crates/apr-cli/src/commands/inspect.rs`.
- Use `mod.rs` for a directory-backed module that owns private submodules and public re-exports, as in `crates/aprender-core/src/cluster/mod.rs` and `crates/aprender-core/src/linear_model/mod.rs`.
- Keep small unit tests in a trailing `#[cfg(test)] mod tests`; split large suites into sibling `tests.rs`, `*_tests.rs`, or `tests/` modules declared with `#[cfg(test)]`, as in `crates/aprender-core/src/error.rs`, `crates/aprender-core/src/linear_model/tests.rs`, and `crates/aprender-core/src/cluster/tests/mod.rs`.
- Name public integration tests after the behavior or contract they enforce: `property_tests.rs`, `monorepo_invariants.rs`, `cli_commands.rs`, `falsification_*.rs`, and `beat_*.rs` are established patterns in `crates/aprender-core/tests/` and `crates/apr-cli/tests/`.
- Reserve `include!()` composition for established large CLI/test surfaces and generated contract code; examples are `crates/apr-cli/src/lib.rs`, `crates/aprender-core/tests/property_tests.rs`, and `crates/aprender-core/src/generated_contracts.rs`. Prefer ordinary modules for new hand-written domain code.

**Functions:**
- Use `snake_case` for functions and methods: `resolve_model_path`, `registered_commands`, `matrix_strategy`, and `bench_linear_regression_fit` in `crates/apr-cli/src/error.rs`, `crates/apr-cli/tests/cli_commands.rs`, `crates/aprender-core/tests/property_tests.rs`, and `crates/aprender-core/benches/linear_regression.rs`.
- Name constructors `new`, fallible loaders `load`/`from_*`, predicates `is_*`/`has_*`, and consuming builder setters `with_*`, following `LinearRegression::new`, `ElasticNet::load`, `ElasticNet::is_fitted`, and `ElasticNet::with_tol` in `crates/aprender-core/src/linear_model/mod.rs` and `crates/aprender-core/src/linear_model/elastic_net.rs`.
- Prefix ordinary unit tests with `test_`; use `prop_` or an invariant name inside `proptest!`; use `falsification_`/`falsify_` and `beat_` when a test is tied to a named falsifier or comparative contract. Examples live in `crates/aprender-core/src/linear_model/tests.rs`, `crates/aprender-core/tests/property_tests.rs`, and `crates/aprender-core/tests/beat_sklearn_iris.rs`.

**Variables:**
- Use descriptive `snake_case` for application state and dimensions (`n_samples`, `n_features`, `fit_intercept`, `metadata_offset`) in `crates/aprender-core/src/linear_model/input.rs` and `crates/apr-cli/src/commands/inspect.rs`.
- Short mathematical names such as `x`, `y`, `i`, `j`, `k`, `n`, and `m` are explicitly allowed by `.clippy.toml`; do not expand standard mathematical notation merely to satisfy identifier-length preferences.
- Use `SCREAMING_SNAKE_CASE` for constants and contract-bound identifiers, such as `MAGIC_APR2` in `crates/apr-cli/src/output.rs` and `PROBAR_SNAPSHOT_DIR` in `crates/aprender-train/tests/tui_snapshot_test.rs`.

**Types:**
- Use `UpperCamelCase` for structs, enums, traits, and error types: `LinearRegression`, `AprenderError`, `CliError`, `Estimator`, and `StageStatus` in `crates/aprender-core/src/linear_model/mod.rs`, `crates/aprender-core/src/error.rs`, `crates/apr-cli/src/error.rs`, `crates/aprender-core/src/traits.rs`, and `crates/apr-cli/src/output.rs`.
- Give crate-specific errors a semantic `*Error` name and expose a local `Result<T>` alias where the module has a stable error boundary, as in `crates/aprender-core/src/error.rs`, `crates/apr-cli/src/error.rs`, `crates/aprender-compute/src/error.rs`, and `crates/aprender-train/src/error.rs`.

## Code Style

**Formatting:**
- Run `cargo fmt --check`; the pinned toolchain includes rustfmt in `rust-toolchain.toml`.
- Follow root `rustfmt.toml`: Rust 2021 formatting, 100-column width, four-space indentation, Unix newlines, field-init shorthand, and `?` shorthand.
- Respect crate-local format configuration where present. For example, `crates/aprender-compute/rustfmt.toml` retains the 100-column width but uses `use_small_heuristics = "Max"`.
- Let rustfmt choose wrapping and brace layout. Do not manually align code in ways rustfmt will undo; representative formatted code is in `crates/aprender-core/src/linear_model/mod.rs` and `crates/apr-cli/src/error.rs`.

**Linting:**
- New workspace crates should inherit `[workspace.lints]` with `[lints] workspace = true`, following `crates/aprender-core/Cargo.toml`, `crates/aprender-train/Cargo.toml`, and most current leaf crates.
- Treat Clippy warnings as build failures by running `cargo clippy --all-targets -- -D warnings`, matching `.github/workflows/ci.yml` and `scripts/ci.sh`. The workspace baseline enables `clippy::all` and `clippy::pedantic` in `Cargo.toml`.
- Preserve the ML-specific lint allowances in `Cargo.toml`: numeric casts, float comparisons, mathematical single-character names, explicit index loops, large test arrays, and long algorithmic functions are accepted when they make numerical code clearer.
- Production code must not use `Option::unwrap` or `Result::unwrap`; `.clippy.toml` disallows both. Use `?`, `ok_or`/`ok_or_else`, or a descriptive `expect` when the invariant is genuinely internal. Test builds explicitly relax this rule in `crates/aprender-core/src/lib.rs` and `crates/aprender-train/src/lib.rs`.
- Do not copy `crates/apr-cli/src/lib.rs`'s broad crate-level Clippy allowances into new crates; it is a localized monorepo-transition exception. Prefer the narrower manifest allowances in `crates/apr-cli/Cargo.toml` or workspace rules in `Cargo.toml`.
- Keep `unsafe` exceptional. The workspace sets `unsafe_code = "deny"` in `Cargo.toml`; crates that permit targeted unsafe operations must carry a `// SAFETY:` invariant immediately around every unsafe block and enforce `clippy::undocumented_unsafe_blocks`, as documented in `Cargo.toml` and implemented in `crates/aprender-core/src/lib.rs` and `crates/aprender-train/src/lib.rs`.

## Import Organization

**Order:**
1. Import the current crate through `crate::...` and `super::...`, as in `crates/aprender-core/src/linear_model/mod.rs` and `crates/apr-cli/src/commands/inspect.rs`.
2. Import external/workspace crates, grouping related names with braces, as in `serde::{Deserialize, Serialize}` in `crates/aprender-core/src/linear_model/mod.rs` and `criterion::{...}` in `crates/aprender-core/benches/linear_regression.rs`.
3. Import `std::...` types last, as in `crates/aprender-core/src/linear_model/mod.rs` and `crates/apr-cli/src/output.rs`.
- Keep function-local imports when they are only needed for one code path, such as `BTreeMap` and SafeTensors helpers in `crates/aprender-core/src/linear_model/mod.rs`.
- There is no import-sorting tool configured beyond rustfmt in `rustfmt.toml`; mirror the surrounding module rather than introducing a new grouping scheme.

**Path Aliases:**
- Rust path aliases are Cargo package/lib aliases rather than source-root aliases. Important examples are `aprender-core` exporting the `aprender` library in `crates/aprender-core/Cargo.toml`, and workspace compatibility aliases such as `realizar`, `entrenar`, and `trueno` in `Cargo.toml`.
- Prefer workspace dependencies (`workspace = true`) for in-monorepo crates and keep both `path` and publishable `version` where the surrounding manifest does so, as in `Cargo.toml` and `crates/apr-cli/Cargo.toml`.

## Error Handling

**Patterns:**
- Return a crate-local `Result<T>` from fallible library boundaries and propagate with `?`. The canonical aliases are `aprender::error::Result` in `crates/aprender-core/src/error.rs` and `crate::error::Result` in `crates/apr-cli/src/error.rs`.
- Use typed error enums at crate boundaries. `AprenderError` in `crates/aprender-core/src/error.rs` carries dimensions, convergence data, format versions, checksums, and I/O sources; `CliError` in `crates/apr-cli/src/error.rs` uses `thiserror` and stable semantic variants.
- Preserve error context when crossing crate seams through `From` implementations or `map_err`, as shown by the `AprFormatError` conversion in `crates/aprender-core/src/error.rs` and serialization context in `crates/aprender-core/src/linear_model/mod.rs`.
- Validate inputs early and return a specific error rather than silently falling back. `resolve_model_path` in `crates/apr-cli/src/error.rs` distinguishes missing files, non-files, and directories without model artifacts.
- CLI failures must print to stderr and return a meaningful non-zero code. `cli_main` in `crates/apr-cli/src/lib.rs` prints `error: {e}`, while `CliError::exit_code` in `crates/apr-cli/src/error.rs` maps failure classes to codes 1 through 11.
- When a public method intentionally panics on an invalid object state, use a descriptive `expect` and document `# Panics`, as `coefficients()` does in `crates/aprender-core/src/linear_model/mod.rs`. Prefer a `Result` for external input, I/O, parsing, or validation failures.
- Shell verification must capture the command status before piping. Use `cmd >log 2>&1; rc=$?` or Bash `PIPESTATUS[0]`; the required methodology is documented in `CLAUDE.md` and `.claude/skills/apr-dogfood/SKILL.md`.

## Logging

**Framework:** `tracing` for long-running services/orchestration; structured CLI output helpers and stdout/stderr for `apr`.

**Patterns:**
- Use structured `tracing::{debug, info, warn, error}` fields for daemon, scheduler, orchestration, and distributed execution code, as in `crates/aprender-distribute/src/executor/remote.rs`, `crates/aprender-orchestrate/src/agent/runtime.rs`, and `crates/aprender-zram/bins/trueno-ublk/src/ublk/multi_queue.rs`.
- Use `crates/apr-cli/src/output.rs` for reusable CLI sections, status badges, metrics, tables, sizes, and paths. Keep machine-readable JSON paths separate from decorated text output, as demonstrated by `crates/apr-cli/src/commands/inspect.rs`.
- Send requested command results to stdout and warnings/errors/progress that must not corrupt machine-readable output to stderr. Top-level CLI errors are handled centrally in `crates/apr-cli/src/lib.rs`.
- Tests should assert emitted structure or behavior rather than depend on incidental logging. CLI output contracts in `crates/apr-cli/tests/cli_commands.rs` normalize color with `NO_COLOR=1`.

## Comments

**When to Comment:**
- Explain invariants, numerical formulas, non-obvious performance constraints, and the reason for a workaround. The coordinate-descent normalization note in `crates/aprender-core/src/linear_model/input.rs` and the provenance serialization invariant in `crates/apr-cli/src/commands/inspect.rs` are representative.
- Attach contract, issue, or falsifier identifiers when code implements an externally tracked guarantee (`PMAT-*`, `GH-*`, `FALSIFY-*`, `INV-*`), following `crates/apr-cli/src/error.rs` and `crates/apr-cli/src/lib.rs`.
- Every unsafe block requires a `// SAFETY:` explanation under the policy in `Cargo.toml`; do not use a generic comment that fails to state bounds, alignment, ownership, device, or target-feature assumptions.
- Avoid comments that merely restate syntax. Test comments should explain the property, oracle, or regression being guarded, as in `crates/aprender-core/tests/property_tests.rs` and `crates/aprender-train/src/prune/snapshot_tests.rs`.

**Rustdoc:**
- Start public modules with `//!` documentation and public items with `///`, following `crates/aprender-core/src/lib.rs`, `crates/aprender-core/src/linear_model/mod.rs`, and `crates/apr-cli/src/error.rs`.
- Document public fallible APIs with `# Errors`, intentional panics with `# Panics`, and nontrivial APIs with runnable examples. Use `cargo test --doc` to verify examples, as required by `scripts/ci.sh`.
- Apply `#[must_use]` to constructors, builders, accessors, predicates, and pure calculations whose result should not be discarded, as in `crates/aprender-core/src/linear_model/mod.rs` and `crates/aprender-core/src/error.rs`.

## Function Design

**Size:** Keep public entry points focused on validation and orchestration, then move parsing, rendering, and computation into helpers. `run` in `crates/apr-cli/src/commands/inspect.rs` dispatches formats while helper functions own output and parsing. Long numerical kernels are accepted by workspace lint policy in `Cargo.toml` when splitting would obscure the algorithm.

**Parameters:**
- Borrow inputs by reference (`&Path`, `&Matrix<T>`, `&Vector<T>`, `&str`) unless ownership is required, following `crates/apr-cli/src/commands/inspect.rs` and `crates/aprender-core/src/linear_model/input.rs`.
- Use generics such as `P: AsRef<Path>` for public filesystem APIs, as in the save/load methods in `crates/aprender-core/src/linear_model/mod.rs`.
- Use configuration structs/builders when options form a reusable concept; builder methods consume and return `Self`, as in `ElasticNet` in `crates/aprender-core/src/linear_model/elastic_net.rs`.

**Return Values:**
- Return domain values directly for infallible calculations and `Result<T, E>` for validation, parsing, I/O, network, or state transitions. Examples are `predict` versus `fit` in `crates/aprender-core/src/linear_model/input.rs`.
- Use `Option<T>` only for genuine absence and preserve absence in serialized schemas intentionally. `MetadataInfo` in `crates/apr-cli/src/commands/inspect.rs` documents which fields must serialize as `null` rather than disappear.

## Module Design

**Exports:**
- Keep implementation modules private and re-export the intended API from `mod.rs` or `lib.rs`, as `crates/aprender-core/src/cluster/mod.rs` and `crates/aprender-core/src/lib.rs` do.
- Use `pub(crate)` for CLI command handlers and helpers that are internal to a crate, as in `crates/apr-cli/src/commands/inspect.rs` and `crates/apr-cli/src/output.rs`.
- Expose narrow test-support seams only when integration tests cannot reach the real behavior otherwise; document the reason, as in `serve_test_support` in `crates/apr-cli/src/lib.rs`.

**Barrel Files:**
- Rust `mod.rs`/`lib.rs` files serve as explicit barrels. Keep re-exports curated rather than wildcarding an entire implementation tree; examples are `crates/aprender-core/src/cluster/mod.rs` and `crates/aprender-core/src/lib.rs`.
- A crate prelude is appropriate for common user-facing traits and primitives; use `crates/aprender-core/src/prelude.rs` rather than creating competing preludes in feature modules.

## Feature-Gated Code

- Declare heavy/platform dependencies with `optional = true`, map them explicitly in `[features]`, and use `dep:` when the dependency name should not become a public implicit feature. See `crates/apr-cli/Cargo.toml` (`training`, `dhat-heap`) and `crates/aprender-train/Cargo.toml` (`tui`, `parquet`, `hub`).
- Gate modules and imports with the same feature expression that activates their dependencies, as in `crates/aprender-core/src/lib.rs` (`audio`, `hf-hub-integration`) and `crates/apr-cli/src/lib.rs` (`inference`, `training`).
- Forward facade features through intermediate crates instead of making users know internal package names; root feature passthroughs live in `Cargo.toml` (`cuda`, `training`, `full`).
- Add `required-features` to examples/benchmarks that cannot compile meaningfully without their backend, as in `crates/apr-cli/Cargo.toml`, `crates/aprender-core/Cargo.toml`, and `crates/aprender-train/Cargo.toml`.
- Keep platform-only dependencies in target-specific tables and pair them with matching `cfg` guards, as in `crates/apr-cli/Cargo.toml` for `libc` and `crates/aprender-core/Cargo.toml` for `memmap2`/WASM dependencies.

## Contract Co-Evolution

- Behavior with formal correctness, CLI, format, or performance obligations should link implementation and tests to YAML under `contracts/`; examples are the contract macros in `crates/apr-cli/src/commands/inspect.rs` and generated assertions loaded by `crates/aprender-core/src/lib.rs`.
- When improving coverage for a function, add or strengthen the corresponding contract at the same time and verify with `make contract-check`; this repository rule is documented in `CLAUDE.md` and implemented by contract targets in `Makefile`.

---

*Convention analysis: 2026-08-07*
