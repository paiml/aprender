# Testing Patterns

**Analysis Date:** 2026-08-07

## Test Framework

**Runner:**
- Rust's built-in libtest runner via Cargo is the baseline; commands and tiers are defined in `Makefile` and `scripts/ci.sh`.
- `cargo-nextest` is the preferred parallel workspace runner for fast/local and CI execution. Configuration: `.config/nextest.toml`.
- The CI profile in `.config/nextest.toml` is fail-fast, retries failed tests twice, warns after 60 seconds, writes JUnit XML to `target/nextest/ci/junit.xml`, and stores failure output.
- Property tests use `proptest` (workspace dependency in `Cargo.toml`) with defaults in `.proptest.toml`: 32 cases, 100 shrink iterations, recursion depth 8, and a 5-second case timeout. Make/CI targets override `PROPTEST_CASES` by tier.

**Assertion Library:**
- Use built-in `assert!`, `assert_eq!`, `assert_ne!`, and `matches!` for unit/integration tests, as in `crates/aprender-core/src/linear_model/tests.rs` and `crates/apr-cli/tests/cli_commands.rs`.
- Use `prop_assert!`/`prop_assert_eq!` inside `proptest!`, as in `crates/aprender-cuda-edge/src/supervisor/heartbeat.rs` and `crates/aprender-core/tests/property_tests.rs`.
- CLI crates may use `assert_cmd` and `predicates`; `apr-cli` also commonly uses `std::process::Command` with `env!("CARGO_BIN_EXE_apr")`, as in `crates/apr-cli/tests/cli_commands.rs` and dependencies in `crates/apr-cli/Cargo.toml`.
- Use `insta` for stable serialization/TUI snapshots in `crates/aprender-train/src/prune/snapshot_tests.rs` and `crates/aprender-train/tests/tui_snapshot_test.rs`; use `jugar-probar` for TUI/golden behavior where the project-specific harness is required.

**Run Commands:**
```bash
make test                       # Standard workspace tests; nextest when installed
make test-fast                  # Workspace lib tests, reduced property cases, -j2
make test-full                  # Workspace, all features, expanded property cases
make test-heavy                 # Explicitly run #[ignore] tests
make coverage                   # Authoritative single-phase LLVM line coverage + ratchet
```
- There is no repository-defined watch-mode target in `Makefile`; do not assume `cargo watch` is installed.

## Test File Organization

**Location:**
- Co-locate focused unit tests in the source file under `#[cfg(test)] mod tests`, as in `crates/aprender-core/src/error.rs` and `crates/apr-cli/src/error.rs`.
- For a larger module, declare `#[cfg(test)] mod tests;` and place tests in a sibling `tests.rs` or `tests/` tree, as in `crates/aprender-core/src/linear_model/mod.rs`, `crates/aprender-core/src/linear_model/tests.rs`, `crates/aprender-core/src/cluster/mod.rs`, and `crates/aprender-core/src/cluster/tests/mod.rs`.
- Put black-box crate integration tests in `<crate>/tests/*.rs`; major suites live in `crates/aprender-core/tests/`, `crates/apr-cli/tests/`, `crates/aprender-serve/tests/`, and equivalent directories across the workspace.
- Put Criterion benchmarks in `<crate>/benches/*.rs` and declare `[[bench]] harness = false` in that crate's `Cargo.toml`, as in `crates/aprender-core/benches/linear_regression.rs` and `crates/aprender-core/Cargo.toml`.
- Put libFuzzer targets in the non-workspace fuzz packages under `fuzz/fuzz_targets/` or a crate-local `fuzz/fuzz_targets/`; manifests include `fuzz/Cargo.toml` and `crates/aprender-gpu/fuzz/Cargo.toml`.

**Naming:**
- Ordinary tests: `test_<behavior>` in source modules, as in `crates/aprender-core/src/linear_model/tests.rs`.
- Property tests: `prop_<invariant>` or a concise invariant name inside `proptest!`, as in `crates/aprender-core/tests/property_tests.rs` and `crates/aprender-cuda-edge/src/supervisor/heartbeat.rs`.
- Contract falsifiers: `falsification_*`, `falsify_*`, or identifiers in comments/attributes, as throughout `crates/aprender-core/tests/` and `crates/apr-cli/tests/`.
- Comparative accuracy/performance gates: `beat_<incumbent>_<behavior>.rs`, as in `crates/aprender-core/tests/beat_sklearn_iris.rs` and `crates/aprender-core/tests/beat_pytorch_autograd_grad.rs`.
- Regression locks should describe the issue or invariant, such as `crates/aprender-core/tests/regression_never_again.rs` and `crates/aprender-core/tests/monorepo_invariants.rs`.

**Structure:**
```text
crates/<package>/
├── src/<module>.rs              # implementation + small #[cfg(test)] module
├── src/<module>/tests.rs        # larger white-box tests
├── tests/<behavior>.rs          # black-box integration/contract/CLI tests
├── benches/<operation>.rs       # Criterion benchmark, harness = false
└── fuzz/fuzz_targets/<input>.rs # optional libFuzzer target
```

## Test Structure

**Suite Organization:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_input_returns_error() {
        let result = operation(/* minimal deterministic input */);
        assert!(result.is_err());
    }
}
```
- This is the dominant white-box pattern in `crates/aprender-core/src/error.rs`, `crates/apr-cli/src/error.rs`, and `crates/aprender-core/src/linear_model/tests.rs`.

**Patterns:**
- Follow arrange/act/assert without mandatory helper frameworks: construct small deterministic inputs, invoke one behavior, then assert outputs and invariants. `crates/aprender-core/src/linear_model/tests.rs` is the canonical numerical example.
- Use tolerance assertions for floating-point algorithms (`(actual - expected).abs() < epsilon`) rather than exact equality unless exactness is itself the contract, as in `crates/aprender-core/src/linear_model/tests.rs`.
- Test both successful output and failure semantics. Error displays/conversions/exit codes are covered in `crates/aprender-core/src/error.rs` and `crates/apr-cli/src/error.rs`.
- Keep contract traceability in module docs and test comments; `crates/apr-cli/tests/cli_commands.rs` maps every assertion to `contracts/apr-cli-commands-v1.yaml` and `FALSIFY-CLI-*` identifiers.
- Use `#[ignore = "reason"]` for tests needing a model, GPU, network, timing isolation, environment mutation, or an unavailable implementation. Run them explicitly through `make test-heavy`, dedicated workflows, or the exact crate/test command. Examples are in `crates/aprender-core/tests/contracts/ssm_contract.rs`, `crates/aprender-train/src/hf_pipeline/tests/core.rs`, and `crates/aprender-compute/src/brick/tests/phases/phase11_profiling.rs`.
- Tests that shell out to Cargo/rustc must use a process-unique temporary directory and must be added to the `serial-build` nextest group in `.config/nextest.toml`.
- Throughput/timing tests named in `.config/nextest.toml` run in the isolated `serial-timing` group so global nextest parallelism cannot distort their wall-clock rates.

## Mocking

**Framework:** No general-purpose mocking framework is detected in workspace manifests; prefer real components, deterministic seams, and temporary resources. Evidence: `Cargo.toml`, `crates/apr-cli/Cargo.toml`, and representative tests under `crates/apr-cli/tests/` and `crates/aprender-core/tests/`.

**Patterns:**
```rust
fn test_server() -> (TestServer, NamedTempFile) {
    let file = NamedTempFile::with_suffix(".apr").expect("create APR fixture");
    let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default())
        .expect("valid server state");
    let server = TestServer::new(create_router(Arc::new(state)))
        .expect("router should start");
    (server, file)
}
```
- The in-process HTTP pattern comes from `crates/apr-cli/src/commands/serve/tests_e2e_http.rs` using `axum-test` rather than a mocked transport.
- For callbacks or external engines, create small recording/scripted implementations at the trait seam, as in `RecordingCallback` in `crates/aprender-train-distill/src/pipeline.rs` and scripted token support in `crates/apr-cli/src/commands/serve/handlers.rs`.
- For rendering, use a recording implementation such as `RecordingCanvas` from `crates/aprender-present-core/src/canvas.rs` and assert emitted draw commands/state.

**What to Mock:**
- Replace nondeterministic or expensive boundaries only: inference token streams, callbacks, GPU telemetry, clocks/data snapshots, or HTTP routing state. Existing seams are shown in `crates/apr-cli/src/commands/serve/handlers.rs` and `crates/aprender-train/tests/tui_snapshot_test.rs`.

**What NOT to Mock:**
- Do not mock core ML algorithms, serializers, routers, CLI argument parsing, or exit-code mapping. Exercise the real public API or Cargo-built binary as in `crates/aprender-core/src/linear_model/tests.rs`, `crates/apr-cli/tests/cli_commands.rs`, and `crates/apr-cli/src/commands/serve/tests_e2e_http.rs`.
- Do not fake a hardware claim; verification must prove the mechanism engaged through a trace/version/behavior signal as required by `CLAUDE.md` and `.claude/skills/apr-dogfood/SKILL.md`.

## Fixtures and Factories

**Test Data:**
```rust
fn matrix_strategy(rows: usize, cols: usize) -> impl Strategy<Value = Matrix<f32>> {
    proptest::collection::vec(-100.0f32..100.0, rows * cols).prop_map(move |data| {
        Matrix::from_vec(rows, cols, data).expect("test dimensions are valid")
    })
}
```
- Property strategy factories are defined near their suite in `crates/aprender-core/tests/property_tests.rs` and composed through `include!()` files under `crates/aprender-core/tests/includes/`.
- Prefer `tempfile::TempDir`/`NamedTempFile` for filesystem fixtures so cleanup is automatic. Examples include `crates/aprender-registry/src/resolver.rs` and `crates/apr-cli/src/commands/serve/tests_e2e_http.rs`.
- Seed random tests explicitly when deterministic replay matters, typically with seed `42`, as in `crates/aprender-tsp/src/solver/ga_tests.rs` and `crates/aprender-serve/src/infer/qwen3_moe_generate.rs`.
- Keep UI snapshots in committed snapshot directories adjacent to the suite: `crates/aprender-train/src/prune/snapshots/`, `crates/aprender-train/tests/snapshots/`, and `crates/apr-cli/playbooks/snapshots/`.
- Preserve proptest failure seeds under crate-local `proptest-regressions/`, including `crates/aprender-compute/proptest-regressions/`, `crates/aprender-serve/proptest-regressions/`, and `crates/aprender-train/proptest-regressions/`.

**Location:**
- Small builders/fixtures stay in the test module (`mock_snapshot` in `crates/aprender-train/tests/tui_snapshot_test.rs`).
- Shared integration fragments live in `tests/includes/`, as in `crates/aprender-core/tests/includes/`; stable contract-generated cases live in `crates/aprender-core/tests/contracts/`.
- Tests requiring external model files belong behind a feature or `#[ignore]`, not in the default suite; `model-tests` is defined in `crates/aprender-core/Cargo.toml` and invoked by `make test-model` in `Makefile`.

## Coverage

**Requirements:**
- Aspirational line coverage is 95%; the enforced non-regression floor is 88%, defined by `COV_THRESHOLD` and `COV_FLOOR` in `Makefile`.
- `make coverage` is the authoritative measurement. It instruments and writes LCOV in one `cargo llvm-cov test` invocation, avoiding the root-facade 0/0 reporting trap documented in `Makefile`.
- Coverage output is written to `target/coverage/lcov.info` with a summary in `target/coverage/summary.txt`; the nightly ratchet runs in `.github/workflows/coverage-nightly.yml`.
- Coverage work must co-evolve with contracts: add/strengthen the relevant YAML and run contract compliance when adding tests for uncovered code, per `CLAUDE.md` and the targets in `Makefile`.
- `coverage-html` and `coverage-full` in `Makefile` document an unscoped-report limitation; use `make coverage` for the gate until those report-only paths are scoped consistently.

**View Coverage:**
```bash
make coverage
pmat query --coverage-gaps --exclude-tests
```
- Use PMAT's coverage queries rather than manually parsing LLVM JSON, as prescribed in `CLAUDE.md` and automated in `.github/workflows/coverage-nightly.yml`.

## Test Types

**Unit Tests:**
- White-box, deterministic, and generally co-located under `#[cfg(test)]`; examples are `crates/aprender-core/src/error.rs`, `crates/aprender-core/src/linear_model/tests.rs`, and `crates/apr-cli/src/error.rs`.
- Default fast tiers reduce generated-case counts with `PROPTEST_CASES`/`QUICKCHECK_TESTS` in `Makefile`; no QuickCheck usages are currently detected, but the environment variable remains in shared recipes.

**Property Tests:**
- Use `proptest!` for algebraic, dimensional, monotonicity, round-trip, and numerical invariants. Examples are `crates/aprender-core/tests/property_tests.rs`, contract files under `crates/aprender-core/tests/contracts/`, and `crates/aprender-cuda-edge/src/supervisor/heartbeat.rs`.
- Keep generated sizes bounded and avoid pathological recursion; repository defaults live in `.proptest.toml`.
- Run extended properties with `make property-test`, `make property-test-fast`, or `make property-test-extensive`, defined in `Makefile`.

**Integration Tests:**
- Use crate `tests/` suites for public-API, file-format, cross-crate, contract, and regression behavior. The main integration surfaces are `crates/aprender-core/tests/`, `crates/apr-cli/tests/`, and `crates/aprender-serve/tests/`.
- CI explicitly runs monorepo/readme/CLI contracts and high-value parity/fail-closed tests after workspace lib tests; the exact list is in `.github/workflows/ci.yml`.

**CLI Tests:**
- Run the Cargo-built test binary via `env!("CARGO_BIN_EXE_apr")`, set `NO_COLOR=1`, assert status/stdout/stderr, and exercise real Clap registration. The canonical helper is `apr_binary()` in `crates/apr-cli/tests/cli_commands.rs`.
- `assert_cmd::Command::cargo_bin` is acceptable in crates already using it, such as `crates/aprender-shell/tests/cli_integration.rs` and `crates/aprender-serve/tests/integration_cli.rs`.
- For installed-binary dogfood, never invoke bare `apr`; source `scripts/apr_bin.sh`, use `$APR`, and enable `APR_BIN_STRICT=1` as required by `.claude/skills/apr-dogfood/SKILL.md` and `CLAUDE.md`.
- When asserting a CLI exit code in shell, capture it before any pipe. The accepted patterns are documented in `CLAUDE.md` and `.claude/skills/apr-dogfood/SKILL.md`.

**E2E Tests:**
- In-process HTTP E2E tests use real Axum routers with `axum-test`, as in `crates/apr-cli/src/commands/serve/tests_e2e_http.rs`.
- Model/GPU/network E2E checks are feature-gated or ignored and run through `make test-model`, `.github/workflows/cuda-nightly.yml`, `.github/workflows/qwen-story-daily.yml`, or the `apr-dogfood` local skill in `.claude/skills/apr-dogfood/SKILL.md`.
- Snapshot/TUI E2E uses `insta` and `jugar-probar` in `crates/aprender-train/tests/tui_snapshot_test.rs`.

**Contract Tests:**
- YAML contracts live under `contracts/`; generated/property bindings live under `crates/aprender-core/tests/contracts/` and are loaded through generated contract modules such as `crates/aprender-core/src/generated_contracts.rs`.
- Use `make contracts` for the release-facing lint plus contract-engine tests, or `make contract-check` for validation, property tests, and binding audit; both are defined in `Makefile`.

**Fuzz and Formal Tests:**
- Run the default 60-second libFuzzer target with `make fuzz`; additional targets are declared in `fuzz/Cargo.toml` and crate-local fuzz manifests such as `crates/aprender-gpu/fuzz/Cargo.toml`.
- Kani proofs are gated with `#[cfg(kani)]`/`#[kani::proof]` in files such as `crates/aprender-core/src/format/kani_proofs.rs`, `crates/aprender-registry/src/model/version.rs`, and `crates/aprender-zram/src/verification_specs.rs`.

## Common Patterns

**Async Testing:**
```rust
#[tokio::test]
async fn e2e_health_returns_200_when_ready() {
    let (server, _fixture) = test_server();
    let response = server.get("/health").await;
    response.assert_status_ok();
}
```
- Use `#[tokio::test]` with an in-process server or deterministic async component, as in `crates/apr-cli/src/commands/serve/tests_e2e_http.rs`. Tokio test features are dev dependencies in `crates/apr-cli/Cargo.toml`.

**Error Testing:**
```rust
#[test]
fn test_dimension_mismatch_error() {
    let result = model.fit(&x, &wrong_length_y);
    assert!(result.is_err());
}
```
- For library errors, assert the semantic variant/message/context where stable, as in `crates/aprender-core/src/error.rs` and `crates/aprender-core/src/linear_model/tests.rs`.
- For CLI errors, assert both displayed message and exact `ExitCode`, as in `crates/apr-cli/src/error.rs`; black-box tests should assert process status plus stderr.

## Benchmarks and Performance Gates

- Criterion is the standard microbenchmark framework. Use `criterion_group!`, `criterion_main!`, `black_box`, `BenchmarkId`, and `Throughput` as appropriate; examples are `crates/aprender-core/benches/linear_regression.rs` and `crates/aprender-bench-tokenizer/benches/encode.rs`.
- Declare every Criterion target with `harness = false` and add `required-features` for backend-specific targets, as in `crates/aprender-core/Cargo.toml` and `crates/aprender-train/Cargo.toml`.
- Run all benchmarks with `make bench`/`cargo bench`; `scripts/bench.sh` supports saving and comparing text baselines, while `.github/workflows/nightly-bench.yml` delegates scheduled Criterion tracking to sovereign CI.
- Keep absolute timing gates out of normal PR tests. Comparative `beat_*_speed` tests are ignored by default and run on a fixed self-hosted host in `.github/workflows/beat-speed-nightly.yml`, which checks every expected measurement is present.
- If a timing assertion must remain in the regular suite, add it to the isolated `serial-timing` group in `.config/nextest.toml` rather than weakening its threshold.

## Mutation Testing

- Local targets are `make mutants`, `make mutants-fast`, and `make mutants-file FILE=...` in `Makefile`; configuration is in `.cargo-mutants.toml` and `.cargo/mutants.toml`.
- Pull requests run blocking, diff-scoped mutation testing in `.github/workflows/ci.yml`. The default tolerated count is zero surviving or timed-out mutants on touched lines.
- Extend a guard only with a discriminating test that proves the new scope can turn red; the repository's re-mutation rule is documented in `CLAUDE.md`.

## CI Verification

- Fast local progression is `make tier1`, `make tier2`, `make tier3`, then `make tier4`; definitions and property-case budgets live in `Makefile`.
- Core CI delegates format, Clippy, test, coverage, security, and provenance to the reusable workflow called by `.github/workflows/ci.yml`.
- Workspace CI runs `cargo nextest run --profile ci --workspace --lib` while excluding `aprender-gpu`, `aprender-cuda-edge`, and `aprender-compute`; the latter has a dedicated harness-status check because its process can fault during cleanup. See `.github/workflows/ci.yml`.
- High-value integration tests run as explicit `cargo test -p ... --test ...` commands in `.github/workflows/ci.yml`, keeping fail-closed, parity, contract, and CLI surfaces visible.
- Pre-commit and pre-push hooks add audit, cargo-deny, docs, PMAT, and format/Clippy gates in `.githooks/pre-commit` and `.githooks/pre-push`.

## Recommended Verification by Change

```bash
cargo fmt --check
cargo clippy -p <changed-package> --all-targets -- -D warnings
cargo test -p <changed-package> --lib
cargo test -p <changed-package> --test <affected-integration-suite>
make contract-check              # when a contract-bound behavior changes
make coverage                    # before claiming coverage improvement
```
- For `apr-cli`, add `cargo test -p apr-cli --test cli_commands`, matching `.claude/skills/pre-release/SKILL.md` and `.claude/skills/apr-dogfood/SKILL.md`.
- For release readiness, also run the readme and monorepo invariant suites in `crates/aprender-core/tests/readme_contract.rs` and `crates/aprender-core/tests/monorepo_invariants.rs`, plus package-safety scripts listed in `.claude/skills/pre-release/SKILL.md`.

---

*Testing analysis: 2026-08-07*
