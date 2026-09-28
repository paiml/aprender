# `make gate` vs CI: the checks a local gate misses

**Date:** 2026-09-28, 06:40Z. **Author:** la-73 (Opus 5.5). **Measured on:** origin/main `c115c5ed02`.
**Asked for by:** the operator ruling relayed by the cop (aprender-77). The first thing to land after 0.70 is one
command that runs, in the CI container image, every gate CI runs, in 10 minutes or less on fw16 with the shared cache.
This is step 1: the gap list. There is no PR until 0.70 is final.

## How the table was built

Nothing here is quoted from memory. Every row was extracted mechanically from main:

- `ci/sections.yml` has 13 jobs. `ci/vendor/sovereign-ci.yml` has 7 jobs; they were parsed after undoing its
  `@SCCACHE_HOST_DIR@` token, the same way `scripts/ci/fat_driver.py resolve_vendored` does it. Together these are
  every section `.github/workflows/ci.yml` runs through `fat_driver.py`.
- Every `run:` step was extracted with `yq`/`jq`, which gives 239 steps.
- Each step was classified against the three lines of `make gate` (Makefile:881):
  `pmat verify --skip satd --skip tests`, `scripts/guard_tree.sh --no-cargo` and `scripts/gate_touched_crates.sh`.
- A `scripts/check_*.sh` step counts as covered only when it runs with **no arguments** and the script is
  cargo-free by guard_tree.sh's own textual rule, `grep -E '(^|[^a-z_-])cargo '`. That is the exact set the
  `--no-cargo` step runs. The same guard with `--self-test`, `--check` or `--full-if-capable` is a different check.
- "Required" means that the `gate` job in `ci.yml` reads the result: guard-tree, guard-cargo, sov.gate (test, lint,
  security, provenance), workspace-test and determinism-compare. mutants, pr-review-*, vendored-schemas and gpu-* are
  advisory: the required gate never reads them. sov.coverage runs on tags only (`coverage_on: 'tag'`), and sov.bench
  is off (`run_benchmarks` defaults to false).

## Result

Of 239 CI steps, 68 are runner plumbing (ownership restore, target-watch probes, image pull, sccache stats and
artifact uploads). Of the remaining 171, `make gate` covers **8**, all through `guard_tree.sh --no-cargo`, which is
the same single command CI's guard-tree section runs. **145 required steps are missed.** The step counts per group below are from the classifier; the appendix is the authoritative per-step list.

| # | Gap (required unless noted) | CI section: steps | What a local gate must add |
|---|---|---|---|
| G1 | **Cargo guards**: 30 `check_*.sh` scripts that call cargo, including lockfile current, cargo-deny, duplicate bin names, README claims, strict-test-binding, pv_bin resolution, apr pinning and tool versions | guard-cargo: 30 | Run the CI's named list, **not** `guard_tree.sh --cargo-only`. That flag also runs 12 cargo guards CI never runs on a PR (see G9), which would blow the 10-minute budget |
| G2 | **Guard case tables and check modes**: the same guards run as `--self-test`, `--check` or `--full-if-capable`, plus mutation tables (`mutate_pr_review_wiring_guard.sh`, 15/15 kill) | guard-tree: 23, guard-cargo: 32 | Run each step verbatim; these are cargo-free apart from guard-cargo's |
| G3 | **Generated-file freshness**: `bump-version.sh --check`, `render_dag.py --check`, `derive_model_manifest.sh --check`, `dogfood_baseline.py --check`, roadmap-fragment-parity, and the explicit integration-test list `ci/explicit-test-commands.d` | guard-tree: 5, sov.security: 1, workspace-test-shard: 2 | Run all the `--check` forms; each takes seconds |
| G4 | **fmt and clippy** | sov.lint: 2 | `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings -A unused-variables`. **Note:** that clippy call has no `-p` or `--workspace`, so it lints only the ROOT FACADE package. The local gate should match CI and file the weakness as its own finding, not quietly be stricter |
| G5 | **Supply chain**: `cargo deny check advisories licenses sources` and `cargo audit` | sov.lint: 1, sov.security: 1, guard-cargo: 2 | Both. The advisory DB fetch needs the network or the shared cache |
| G6 | **pv codegen**: `pv` generates contract assertions before the tests and clippy build | sov.test: 1, sov.lint: 1 | The container's pv, which is the pinned one |
| G7 | **Affected tests (BSE-17 tier)**: CI decides quick, full, reuse or none, then runs the selected crates' lib and integration tests **plus every tree-reader test target** in the registry, and checks Σ-executed (ran == owed, #4433). The full tier adds the GPU crates on their default features, compute tests, and a build of every example | workspace-test-shard: 9, sov.test: 1 | `gate_touched_crates.sh` selects only the touched crates and their direct reverse dependents, so **a contracts/, docs/ or scripts/-only diff runs 0 tests locally and the tree readers in CI.** The local gate should call the same tier decider (`--event pull_request`) instead |
| G8 | **Rule checks outside guard_tree**: roadmap-valid, the build.rs crate-root escape, the apr-format leaf sovereignty guard, and the PR-body rules (closes every issue it cites, sweep PR carries an ontology delta) | guard-tree: 4, sov.security: 1, workspace-test-shard: 4, workspace-test: 1, sov.gate: 1 | The PR-body rules need the body text; a local gate reads it from `gh pr view` or a file |
| G9 | **Neither CI nor `make gate` runs these (not a gap, a boundary)**: 12 cargo guards no required CI step names — `check_book_examples_compile`, `check_book_examples_executable`, `check_book_linkcheck`, `check_clippy_current_stable`, `check_clippy_feature_matrix`, `check_mcp_never_path_resolves_apr`, `check_model_ladder`, `check_msrv`, `check_multiplatform_dogfood`, `check_publish_safety`, `check_tokenizer_nonascii_parity`, `check_wasm32_core_builds`. No proof-count guard exists on main. `make contracts` (pv lint + census + graph freshness + README count) is not a CI step: CI runs only its exit-propagation table (`scripts/tests/make_contracts_propagates.sh`). The feature matrix (`check_clippy_feature_matrix.sh`) is not wired either | — | Out of scope for "every gate CI runs". Listed so nobody assumes CI covers them. The ontology ratchets that ARE gated run through cargo-free guards already inside `guard_tree --no-cargo`: `check_ont_ratchet`, `check_fleet_pv_shapes_gate`, `check_baseline_ratchets` |
| G10 | **determinism-compare**: the X64 and ARM64 raster receipts compared | determinism: 7, determinism-compare: 4 | Cannot run on one host. A local gate runs the X64 half and compares it against the last main ARM64 receipt, or marks this row as a declared remote-only check |

`make gate` also has one thing CI does not: `pmat verify`. It is extra, not a gap.

## Next (step 2, not started)

- Time each group on intel/fw16 in the sovereign-ci image with the shared sccache. G7 full tier and G1 are the budget
  risks.
- Then write one entry point, for example `make gate-ci`, driven by `fat_driver.py` over the SAME section
  definitions (`--sections guard-tree,guard-cargo,sov.lint,sov.test,sov.security,workspace-test-shard`) rather than
  a second hand-written list. That way it cannot drift from CI.

## Appendix: every missed step (generated)

### required (145 steps)

| section | class | CI step | first command |
|---|---|---|---|
| `workspace-test-shard` | MISSED:generated-file freshness | Decide the test tier (BSE-17, PMAT-1077) | `args=(--event "${GITHUB_EVENT_NAME}")` |
| `workspace-test-shard` | MISSED:tests | Workspace lib tests (25,300+) | `docker run --rm \\` |
| `workspace-test-shard` | MISSED:tests | GPU crates on their default features (per-package resolve, no device needed) | `docker run --rm \\` |
| `workspace-test-shard` | MISSED:tests | Compute tests (tolerate SIGSEGV at exit — all tests pass but harness crashes on cleanup) | `docker run --rm \\` |
| `workspace-test-shard` | MISSED:generated-file freshness | Integration tests | `bash scripts/ci_run_explicit_test_commands.sh --list ci/explicit-test-commands.d` |
| `workspace-test-shard` | MISSED:tests | Σ-executed: the tests this job ran are exactly the tests it owes (#4433) | `sig="$RUNNER_TEMP/sigma"` |
| `workspace-test-shard` | MISSED:build/feature matrix | Build every example (workspace, keep-going) | `docker run --rm \\` |
| `workspace-test-shard` | MISSED:tests | Quick tier: lib + integration tests of the selected crates (BSE-17) | `read -ra crates <<< "$CRATES"` |
| `workspace-test-shard` | MISSED:tests | Quick tier: every test target that reads the tree (BSE-17) | `read -ra targets <<< "$TARGETS"` |
| `workspace-test-shard` | MISSED:tests | Σ-executed (quick tier): the tests this job ran are exactly the tests it owes (#4433) | `sig="$RUNNER_TEMP/sigma"` |
| `workspace-test-shard` | MISSED:build/feature matrix | Quick tier: cargo check --workspace (over-the-cap integration, BSE-17 rule (i)) | `printf 'over the cap: %s\\n' "$REASON"` |
| `workspace-test-shard` | MISSED:other rule check | Reuse: this tree already passed workspace-test on the PR head (BSE-17) | `printf 'workspace-test REUSED: %s\\n' "${{ steps.tier.outputs.reason }}"` |
| `workspace-test-shard` | MISSED:other rule check | None: nothing for workspace-test to measure (#3658) | `printf 'workspace-test NOT RUN: %s\\n' "$REASON"` |
| `workspace-test-shard` | MISSED:other rule check | Build.rs crate-root escape check (v0.31.1 yank guard) | `docker run --rm --init \\` |
| `workspace-test-shard` | MISSED:other rule check | apr-format leaf sovereignty guard (#2231) | `docker run --rm --init \\` |
| `workspace-test` | MISSED:other rule check | Every shard passed | `printf 'workspace-test-shard matrix result: %s\\n' "$SHARD_RESULT"` |
| `guard-tree` | COVERED:guard_tree --no-cargo | Every cargo-free guard runs, and every failure is reported | `bash scripts/guard_tree.sh --no-cargo` |
| `guard-tree` | MISSED:case table (guard self-test) | guard_tree.sh's own case table (BSE-01, wired BSE-02) | `bash scripts/tests/guard_tree_test.sh` |
| `guard-tree` | MISSED:case table (guard self-test) | guard_tree.sh's parallel dispatcher case table (PMAT-1098) | `bash scripts/tests/guard_tree_parallel_test.sh` |
| `guard-tree` | MISSED:pv/ontology | make contracts propagates pv's exit (PVL-001 EV-4, aprender#4168) | `bash scripts/tests/make_contracts_propagates.sh` |
| `guard-tree` | MISSED:case table (guard self-test) | Fleet hygiene case tables (steward, history, utilization, tier ledger, resolve-dirty, nightly manifest) | `bash scripts/ci_queue_steward.sh --selftest` |
| `guard-tree` | MISSED:generated-file freshness | Dogfood baselines must re-derive from the ledger | `python3 scripts/dogfood_baseline.py --check` |
| `guard-tree` | MISSED:case table (guard self-test) | Version-bump case table (reaches the excluded workspace) | `bash scripts/bump-version.sh --self-test` |
| `guard-tree` | MISSED:generated-file freshness | Every workspace is at a consistent version | `bash scripts/bump-version.sh --check` |
| `guard-tree` | MISSED:guard run with args (self-test/check mode) | Every DAG row marked complete has a receipt whose marker says so (C0-7) | `bash scripts/check_receipt_complete.sh --dag docs/specifications/pp-066-dag.yaml` |
| `guard-tree` | MISSED:case table (guard self-test) | Receipt signer refuses what it must (PERF-007, case table) | `bash scripts/perf_receipt_sign.sh --selftest` |
| `guard-tree` | MISSED:other rule check | Every ARMED PP row names cases that exist (PP-29) | `bash scripts/spec_conformance.sh` |
| `guard-tree` | MISSED:case table (guard self-test) | Spec-conformance guard can still go RED (case table) | `bash scripts/spec_conformance.sh --selftest` |
| `guard-tree` | MISSED:other rule check | PP-066 spec v1.6 defect table (22 rows) and its v1.5 RED proof | `bash tests/spec/pp066_v16_defects.sh && bash tests/spec/pp066_v16_defects.sh --v15-red` |
| `guard-tree` | MISSED:generated-file freshness | The spec's rendered DAG block matches the yaml (G-4 --check) | `python3 scripts/render_dag.py --check --dag docs/specifications/pp-066-dag.yaml --spec docs/specific` |
| `guard-tree` | MISSED:other rule check | The PP-066 obligation DAG holds its invariants (G-4, C10) | `bash scripts/check_dag_invariants.sh docs/specifications/pp-066-dag.yaml --min-slack-days 6` |
| `guard-tree` | MISSED:guard run with args (self-test/check mode) | A row PR writes no shared file (G-11, PMAT-1062) | `bash scripts/check_row_pr_write_set.sh --event "${GITHUB_EVENT_NAME:-pull_request}" --branch "${GITH` |
| `guard-tree` | MISSED:case table (guard self-test) | Release criteria can still go RED (release_criteria.sh --self-test) | `bash scripts/release_criteria.sh --self-test` |
| `guard-tree` | MISSED:generated-file freshness | Supported-model manifest case table (L0-1a, | `bash scripts/derive_model_manifest.sh --self-test` |
| `guard-tree` | MISSED:generated-file freshness | The manifest equals its derivation (a named model is never absent, an entry is never typed) | `bash scripts/derive_model_manifest.sh --check` |
| `guard-tree` | MISSED:guard run with args (self-test/check mode) | C14 model-parity case table (L0-1a, | `bash scripts/check_model_parity.sh --self-test` |
| `guard-tree` | MISSED:case table (guard self-test) | Parity-receipt denominator case table (#3577) | `bash scripts/parity_receipt_denominator.sh --self-test` |
| `guard-tree` | MISSED:case table (guard self-test) | Sharded-fixture splitter case table (F-1, #3024) | `python3 scripts/make_sharded_safetensors.py --self-test` |
| `guard-tree` | MISSED:guard run with args (self-test/check mode) | format x command honesty case table (F-1, #3022) | `bash scripts/check_format_command_matrix.sh --self-test` |
| `guard-tree` | MISSED:case table (guard self-test) | Bandwidth probe can still refuse (case table, no GPU) | `bash scripts/measure_bandwidth.sh --selftest` |
| `guard-tree` | MISSED:case table (guard self-test) | Receipt converter round-trips into a gate verdict | `python3 scripts/lib/perf_receipt.py --selftest` |
| `guard-tree` | COVERED:guard_tree --no-cargo | Tests must not gate on paths outside the workspace | `bash scripts/check_test_fixture_paths.sh` |
| `guard-tree` | MISSED:guard run with args (self-test/check mode) | Fixture-path guard case table | `bash scripts/check_test_fixture_paths.sh --self-test` |
| `guard-tree` | MISSED:guard run with args (self-test/check mode) | sovereign-ci.yml pin guard case table | `bash scripts/check_ci_reusable_workflow_pinned.sh --self-test` |
| `guard-tree` | COVERED:guard_tree --no-cargo | ci.yml must pin sovereign-ci.yml by commit sha, not @main | `bash scripts/check_ci_reusable_workflow_pinned.sh` |
| `guard-tree` | COVERED:guard_tree --no-cargo | No contract may name a machine-specific path | `bash scripts/check_hardcoded_paths.sh` |
| `guard-tree` | MISSED:guard run with args (self-test/check mode) | Machine-specific-path guard case table | `bash scripts/check_hardcoded_paths.sh --self-test` |
| `guard-tree` | MISSED:guard run with args (self-test/check mode) | Whole-tree machine-specific-path ratchet (pinned analyser via scripts/pmat_bin.sh, PMAT-1059) | `bash scripts/check_hardcoded_paths.sh --full-if-capable` |
| `guard-tree` | MISSED:guard run with args (self-test/check mode) | Receipt-wiring guard case table (both polarities of the if:) | `bash scripts/check_pr_review_wiring.sh --self-test` |
| `guard-tree` | COVERED:guard_tree --no-cargo | The PR-review receipt guard is wired job-level, not path-filtered | `bash scripts/check_pr_review_wiring.sh` |
| `guard-tree` | MISSED:guard run with args (self-test/check mode) | Receipt-gate base-owned case table (17 rows, both polarities) | `bash scripts/check_receipt_gate_base_owned.sh --self-test` |
| `guard-tree` | COVERED:guard_tree --no-cargo | The PR's own receipt is judged from the base, not the head (PRQ-013) | `bash scripts/check_receipt_gate_base_owned.sh` |
| `guard-tree` | MISSED:case table (guard self-test) | Receipt-wiring guard: every rule it states, mutated (15/15 kill) | `bash scripts/mutate_pr_review_wiring_guard.sh` |
| `guard-tree` | MISSED:guard run with args (self-test/check mode) | A PR body must close every issue it cites, or say why not (§6 R-2) | `bash scripts/check_pr_closes_issue.sh --self-test` |
| `guard-tree` | MISSED:other rule check | A sweep PR closes with an ontology delta, not a paragraph (§11.1) | `printf '%s' "$PR_BODY" > "$RUNNER_TEMP/ont-body.txt"` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | Every beat must be executed by some workflow | `bash scripts/check_beats_gated.sh` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | Every execution-surface `apr` reference must be pinned | `bash scripts/check_apr_bin_pinned.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | `apr`-pinning guard must still turn RED (case table + surfaces) | `bash scripts/check_apr_bin_pinned.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | Cargo.lock must match the manifests | `bash scripts/check_lockfile_current.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | lockfile guard must still turn RED (case table) | `bash scripts/check_lockfile_current.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | The release runner must not resolve a verifier through PATH | `bash scripts/check_verifier_pinning.sh` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | One dogfood runner, and the user-scope copy is a gated shim | `bash scripts/check_dogfood_shim.sh` |
| `guard-cargo` | MISSED:supply chain | cargo-deny (licences, bans, sources, advisories) | `B64=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" \| base64 \| tr -d '\\n')` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | apr_bin.sh must resolve the HEAD-built binary, whatever its profile | `bash scripts/check_apr_bin_resolution.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Tree-reader registry guard case table (BSE-17) | `bash scripts/check_tree_reader_tests.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | Tree-reader registry and unwired ledger agree with the sources (BSE-17) | `bash scripts/check_tree_reader_tests.sh` |
| `guard-cargo` | COVERED:guard_tree --no-cargo | bashrs must see every script in scripts/ (shrink-only) | `bash scripts/check_shell_lint_ratchet.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Coverage is a RUN, not a runner: 6 fixtures + the probe-deletion mutation (R-5) | `bash scripts/check_silicon_coverage.sh --self-test` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | A cuda-named silicon axis must invoke CUDA (R-6) | `bash scripts/check_silicon_cuda.sh --self-test` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Every cargo -p in a workflow names a real crate (R-8) | `bash scripts/check_workflow_cargo_packages.sh --self-test` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | bashrs SEC/DET/IDEM: the release gate, on the PR (#3196) | `bash scripts/check_bashrs_gate.sh --self-test` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Per-function complexity may only fall (shrink-only) | `bash scripts/check_complexity_ratchet.sh --selftest` |
| `guard-cargo` | MISSED:case table (guard self-test) | Ratchet verdicts depend on (comparand, merge) only -- readme + complexity case tables (BSE-03) | `bash scripts/tests/ratchet_semantics_test.sh --class readme` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | No binary may hand-roll argv parsing | `bash scripts/check_no_hand_rolled_parsers.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | hand-rolled-parser guard must still turn RED (case table) | `bash scripts/check_no_hand_rolled_parsers.sh --self-test` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | inherited-stdin guard must still turn RED (case table) | `bash scripts/check_hermetic_stdin_tests.sh --self-test` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Enforcement guard's own case table must pass before it judges | `bash scripts/check_contract_enforcement.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | No contract may name an enforcement command that cannot run | `bash scripts/check_contract_enforcement.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Gate-5 stage guard's own case table must pass before it judges | `bash scripts/check_gate5_stage.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | A pre-release gate must name the stage it is valid at (#2543) | `bash scripts/check_gate5_stage.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | README claims must match measurement | `bash scripts/check_readme_claims.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | pv_bin.sh must resolve the HEAD-built pv and refuse a stale one | `bash scripts/check_pv_bin_resolution.sh` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | Every contract cites a test that exists (strict-test-binding) | `bash scripts/check_contract_test_binding.sh` |
| `guard-cargo` | MISSED:tests | Contract corpus integrity (three dark test targets, now green) | `cargo test -p aprender-contracts --test validate_contracts` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Renamed-crate facades case table | `bash scripts/check_facade_compat.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | provable-contracts 0.3.1 code still compiles against the facades | `bash scripts/check_facade_compat.sh` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | The cascade's facade order gate arms on every spelling cargo accepts | `bash scripts/check_facade_order_gate.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Duplicate bin-name case table | `bash scripts/check_duplicate_bin_names.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | No two crates may claim one bin name | `bash scripts/check_duplicate_bin_names.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Cascade-coverage case table | `bash scripts/check_cascade_covers_all_crates.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | The release cascade must cover and order every publishable crate | `bash scripts/check_cascade_covers_all_crates.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Publish preflight gate case table (F-9) | `bash scripts/check_publish_preflight.sh --selftest` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | No fabricated comparator baselines | `bash scripts/check_no_fabricated_baselines.sh` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | No claim literals on user-facing surfaces | `bash scripts/check_no_claim_literals.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | No claim-literal guard regression (case table) | `bash scripts/check_no_claim_literals.sh --selftest` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | Performance claims cite their receipts (PERF-010) | `bash scripts/check_perf_claims_cite_receipts.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Claims-cite-receipts guard can still go RED (case table) | `bash scripts/check_perf_claims_cite_receipts.sh --selftest` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | perf-matrix.yaml means what it says (PP-1, PP-16, PP-33) | `bash scripts/check_perf_matrix_schema.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Matrix-schema guard can still go RED (case table) | `bash scripts/check_perf_matrix_schema.sh --selftest` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | Perf workflows cannot run concurrently with themselves (PP-19) | `bash scripts/check_perf_concurrency_groups.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Concurrency-group guard can still go RED (case table) | `bash scripts/check_perf_concurrency_groups.sh --selftest` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | dogfood live LLM probe is armed | `bash scripts/check_dogfood_llm_probe_armed.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | dogfood LLM probe guard case table | `bash scripts/check_dogfood_llm_probe_armed.sh --self-test` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Workflow env-var case table | `bash scripts/check_workflow_env_defined.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | A `run:` block may only interpolate names its job defines | `bash scripts/check_workflow_env_defined.sh` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | Every model-tests target is named by a workflow | `bash scripts/check_model_tests_wired.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | model-tests wiring case table | `bash scripts/check_model_tests_wired.sh --self-test` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Explicit test command list: runner + guard case table (PMAT-3313) | `bash scripts/check_explicit_test_commands.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | Explicit test commands stay fragments: wired, one command per file, no && mega-line (PMAT-3313) | `bash scripts/check_explicit_test_commands.sh` |
| `guard-cargo` | MISSED:tests | model-tests falsification suites (was gated by nothing, aprender#2522) -- SATD ceiling measured on the comparand (BSE-03) | `dr() { # dr <dir mounted as /workspace> <cmd...>` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | include!() targets must survive cargo package (CB-510) | `bash scripts/check_package_includes.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Package-includes guard case table | `bash scripts/check_package_includes.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | No in-tree crate name may resolve from crates.io | `bash scripts/check_lockfile_no_registry_siblings.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Lockfile sibling guard case table | `bash scripts/check_lockfile_no_registry_siblings.sh --self-test` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Sibling-crate pathing guard case table | `bash scripts/check_workspace_siblings_pathed.sh --self-test` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | In-tree siblings must be pathed, never pulled from crates.io | `bash scripts/check_workspace_siblings_pathed.sh` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | Publishable crates may not depend on unpublishable ones | `bash scripts/check_publishable_deps_publishable.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | Publishable-deps guard case table | `bash scripts/check_publishable_deps_publishable.sh --self-test` |
| `guard-cargo` | MISSED:supply chain | Advisories must pass, with deny.toml exemptions honoured | `GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GITHUB_TOKEN" \| base64 \| t` |
| `guard-cargo` | MISSED:other rule check | Every deny.toml exemption must still be live | `GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GITHUB_TOKEN" \| base64 \| t` |
| `guard-cargo` | MISSED:cargo guard (guard_tree --cargo-only) | No GHSA-only vulnerable crate in Cargo.lock | `bash scripts/check_no_ghsa_banned_crates.sh` |
| `guard-cargo` | MISSED:guard run with args (self-test/check mode) | GHSA guard must still turn RED (case table) | `bash scripts/check_no_ghsa_banned_crates.sh --self-test` |
| `guard-cargo` | MISSED:other rule check | nextest must ignore no key in .config/nextest.toml | `docker run --rm --init \\` |
| `guard-cargo` | MISSED:case table (guard self-test) | nextest-key guard must still turn RED (case table) | `docker run --rm --init \\` |
| `determinism` | MISSED:other rule check | A target dir of this job's own | `` |
| `determinism` | MISSED:generated-file freshness | No receipt from an earlier run may survive into this one | `rm -rf target/aprender-viz` |
| `determinism` | MISSED:other rule check | The libm ban must be able to fail | `bash scripts/ci/libm-ban-live.sh` |
| `determinism` | MISSED:tests | Render the fixture and write this host's receipt | `cargo test -p aprender-viz --features text-path,raster --test render_determinism` |
| `determinism` | MISSED:tests | Rasterise the SVG fixture and write this host's raster receipt | `cargo test -p aprender-viz --features raster --test raster_determinism` |
| `determinism` | MISSED:tests | No font may be resolved on the raster path | `cargo test -p aprender-viz --features raster --lib raster::` |
| `determinism` | MISSED:other rule check | No font stack may be compiled into the raster path | `bash scripts/ci/raster-fontdb-ban.sh` |
| `determinism-compare` | MISSED:case table (guard self-test) | The compare must be able to fail | `bash scripts/ci/determinism-compare.sh --self-test` |
| `determinism-compare` | MISSED:generated-file freshness | Compare the hosts and write the receipt | `mkdir -p target/aprender-viz` |
| `determinism-compare` | MISSED:case table (guard self-test) | The raster compare must be able to fail | `bash scripts/ci/raster-compare.sh --self-test` |
| `determinism-compare` | MISSED:generated-file freshness | Compare the hosts' rasters and write the raster receipt | `mkdir -p target/aprender-viz` |
| `sov.test` | MISSED:pv/ontology | Generate contract assertions (pv codegen) | `PV=""` |
| `sov.test` | MISSED:tests | Run tests | `git config --global --add safe.directory "$GITHUB_WORKSPACE"` |
| `sov.lint` | MISSED:pv/ontology | Generate contract assertions (pv codegen) | `PV=""` |
| `sov.lint` | MISSED:fmt/clippy | Format check | `cargo fmt --all -- --check` |
| `sov.lint` | MISSED:fmt/clippy | Clippy | `cargo clippy $CLIPPY_ARGS -- -D warnings -A unused-variables 2>&1 \|\| \\` |
| `sov.lint` | MISSED:supply chain | Supply chain audit (cargo deny) | `GIT_CONFIG_COUNT=1` |
| `sov.security` | MISSED:other rule check | roadmap-valid | `if [ ! -f docs/roadmaps/roadmap.yaml ]; then` |
| `sov.security` | MISSED:generated-file freshness | roadmap-fragment-parity | `entries=docs/roadmaps/entries` |
| `sov.security` | MISSED:supply chain | Audit | `GIT_CONFIG_COUNT=1` |
| `sov.gate` | MISSED:other rule check | Check results | `rc=0` |

### tag-only (3 steps)

| section | class | CI step | first command |
|---|---|---|---|
| `sov.coverage` | MISSED:pv/ontology | Generate contract assertions (pv codegen) | `PV=""` |
| `sov.coverage` | MISSED:other rule check | Run coverage | `git config --global --add safe.directory "$GITHUB_WORKSPACE"` |
| `sov.coverage` | MISSED:other rule check | Enforce coverage floor (OPT-IN ratchet — PMAT build-system audit gap | `if [ ! -f lcov.info ]; then` |

### off by default (2 steps)

| section | class | CI step | first command |
|---|---|---|---|
| `sov.bench` | MISSED:pv/ontology | Generate contract assertions (pv codegen) | `PV=""` |
| `sov.bench` | MISSED:other rule check | Run criterion benchmarks | `if ls benches/*.rs 2>/dev/null \| head -1 \| grep -q '.'; then` |

### advisory (not in `gate`) (19 steps)

| section | class | CI step | first command |
|---|---|---|---|
| `vendored-schemas` | COVERED:guard_tree --no-cargo | schemas/ are the bytes we vendored, and validate offline | `bash scripts/check_vendored_schemas.sh` |
| `vendored-schemas` | MISSED:case table (guard self-test) | the guard can still turn RED | `bash scripts/mutate_vendored_schemas_guard.sh` |
| `pr-review-shadow` | MISSED:case table (guard self-test) | Publisher case table: 6 rows, upsert-by-marker and every silent-skip path | `bash scripts/pr_review_shadow_publish.sh --self-test` |
| `pr-review-shadow` | MISSED:other rule check | Put the verdict on the pull request itself | `{` |
| `pr-review-sign` | MISSED:case table (guard self-test) | Signer case table: 14 rows on a throwaway keypair | `bash scripts/pr_review_sign_receipt.sh --self-test` |
| `pr-review-sign` | MISSED:other rule check | Sign this PR's receipt, if it has an unsigned one | `root="evidence/pr-review/$PR_NUMBER"` |
| `pr-review-sign` | MISSED:generated-file freshness | Commit the signature back to the PR branch | `git config user.name  "aprender-pr-review-signer"` |
| `gpu-touched` | MISSED:case table (guard self-test) | The GPU selector can still go RED (case table, no GPU, no cargo) | `bash scripts/ci_gpu_touched.sh --self-test` |
| `gpu-touched` | MISSED:other rule check | Does this PR touch the GPU set? (rows 67-C1 / 67-D1) | `git fetch --no-tags --depth=1 origin \\` |
| `gpu-quick` | MISSED:other rule check | Preflight — the self-hosted toolset this job needs (67-C2) | `bash scripts/ci_self_hosted_preflight.sh --cuda` |
| `gpu-quick` | MISSED:other rule check | Decide whether to run (yield-to-training) | `used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \| head -1 \| tr` |
| `gpu-quick` | MISSED:tests | aprender-gpu cuda unit tests (--features cuda --lib --release) | `nice -n 19 cargo test -p aprender-gpu --features cuda --lib --release` |
| `gpu-quick` | MISSED:tests | PERF-053 determinism filter (the one cuda-nightly runs) | `nice -n 19 cargo test -p aprender-gpu --features cuda --lib --release perf053 \\` |
| `cuda-unit` | MISSED:other rule check | Preflight — the self-hosted toolset this job needs (67-C2) | `bash scripts/ci_self_hosted_preflight.sh --cuda` |
| `cuda-unit` | MISSED:other rule check | Host GPU lock — the host's file, and the runner takes it | `l=/run/lock/fleet-gpu/gpu.lock` |
| `cuda-unit` | MISSED:tests | aprender-gpu cuda unit tests (--features cuda --lib --release) | `nice -n 19 cargo test -p aprender-gpu --features cuda --lib --release` |
| `cuda-unit` | MISSED:tests | aprender-serve cuda unit tests (the derived cuda-only set, by name) | `t="$RUNNER_TEMP"` |
| `mutants` | MISSED:other rule check | Compute PR diff for mutation scoping | `BASE_REF="${{ github.event.pull_request.base.ref }}"` |
| `mutants` | MISSED:tests | Run diff-scoped mutation testing (BLOCKING) | `if [ ! -s pr.diff ]; then` |
