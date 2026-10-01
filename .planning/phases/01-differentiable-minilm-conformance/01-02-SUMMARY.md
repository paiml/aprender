---
phase: 01-differentiable-minilm-conformance
plan: 02
subsystem: core-nn
tags: [rust, autograd, module-trait, named-parameters, tdd, aprender-core]

# Dependency graph
requires: []
provides:
  - "Module trait extension: named_parameters / named_parameters_mut with a POSITIONAL-FALLBACK default, so named==positional arity and order hold for every implementor in the crate, including non-overriding ones"
  - "Module::set_training(bool) as the recursive mode-propagation channel (D-17), distinct from the leaf-local train()/eval()"
  - "Semantic named overrides on the full BERT encoder path: Linear, LayerNorm, Dropout, Sequential, MultiHeadAttention"
  - "crate::nn::tests_named_module::snapshot_named — shared pub(crate) f32::to_bits byte-identity helper for reuse by 01-06's encoder conformance tests"
  - "Sequential::get(index) -> Option<&dyn Module> child accessor (mirrors ModuleList::get)"
affects: [01-06-encoder, 01-07-freeze-groups, 01-08-gradient-gates]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Positional-fallback trait default: a new traversal method delegates to the existing accessor rather than returning empty, making the cross-method invariant hold by construction instead of by convention"
    - "Index-dot prefixing that does not renumber around parameterless children"
    - "Mode propagation via a dedicated set_training channel so children overriding only set_training are not skipped"
    - "pub(crate) test helper in a #[cfg(test)] #[path] module (not inside `mod tests`) for cross-module reuse"

key-files:
  created:
    - crates/aprender-core/src/nn/tests_named_module.rs
  modified:
    - crates/aprender-core/src/nn/module.rs
    - crates/aprender-core/src/nn/linear.rs
    - crates/aprender-core/src/nn/normalization/mod.rs
    - crates/aprender-core/src/nn/dropout/mod.rs
    - crates/aprender-core/src/nn/container.rs
    - crates/aprender-core/src/nn/transformer/mod.rs
    - crates/aprender-core/src/nn/mod.rs

key-decisions:
  - "named_parameters default enumerates parameters() with numeric names rather than returning Vec::new(), so the arity/order invariant that freeze grouping and optimizer partitioning depend on cannot be silently violated by a non-overriding implementor"
  - "Sequential child indices are container positions, not positions among parameter-bearing children — [Linear, Dropout, Linear] yields 0.* and 2.*, so inserting or removing a dropout never re-addresses existing freeze groups"
  - "Sequential and MultiHeadAttention propagate mode via set_training on children rather than train()/eval(), pinning the propagation channel"
  - "Dropout states its empty named list explicitly rather than inheriting it, documenting that p/RNG/seed/training-flag are module state and never parameters"
  - "Sequential::get added as test observability (mirrors ModuleList::get) because child mode state is otherwise unobservable behind Vec<Box<dyn Module>>"

patterns-established:
  - "Invariant-by-construction defaults: when adding a trait method that must agree with an existing one, delegate to the existing one in the default body"
  - "Order-sensitive assertions: Vec<String> equality and TensorId identity, never HashSet membership or shape-only comparison"
  - "Byte-identity proofs via f32::to_bits, which distinguishes -0.0 from 0.0 and makes NaN comparable"

requirements-completed: [ENC-04, ENC-05]

# Metrics
duration: ~35min
completed: 2026-08-08
---

# Phase 1 Plan 02: Named Module Traversal Summary

**Extended the existing `Module` trait with named parameter traversal that provably cannot disagree with positional traversal, plus a recursive `set_training` channel, and gave every BERT-encoder-path module semantic names.**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-08-08T05:37:03Z (final task commit)
- **Tasks:** 2 (Task 2 was TDD: RED + GREEN)
- **Files modified:** 8 (1 created, 7 modified)

## Accomplishments

- **The empty-default defect is closed by construction.** `named_parameters()` / `named_parameters_mut()` delegate to `parameters()` / `parameters_mut()` with enumerated numeric names. Any parameter-bearing implementor anywhere in the crate — present or future, overriding or not — reports N names for N parameters. Plans 01-07 (freeze partition) and 01-08 (optimizer grouping) can trust that invariant universally rather than per-type.
- **Every BERT-path module returns semantic names**, asserted against exact ordered sequences: `Linear` → `weight`/`bias` (mirroring the `Some`/`None` bias branch), `LayerNorm` → `weight`/`bias` (empty when non-affine), `Dropout` → empty, `Sequential` → `0.weight`-style index-dot prefixes, `MultiHeadAttention` → the exact 8-name `q_proj`/`k_proj`/`v_proj`/`out_proj` sequence.
- **ENC-05 byte-identity harness landed and is reusable.** `snapshot_named` is `pub(crate)` in a `#[cfg(test)] #[path]` module — reachable from `crate::setfit::…`, proven by a cross-module smoke test in `nn::module` that would fail to compile if visibility regressed.
- **26 named-traversal tests, TDD-verified.** 14 failed before implementation and all 26 pass after. Full `aprender-core` suite: 14,008 tests pass, zero regressions.

## Task Commits

1. **Task 1: Extend Module trait with positional-fallback defaults** — `1cdc26f18` (feat)
2. **Task 2 (RED): failing named-traversal and mode-propagation tests** — `3528218e6` (test)
3. **Task 2 (GREEN): semantic overrides across the BERT encoder path** — `a2803ba2f` (feat)

No REFACTOR commit — the GREEN implementation was already clippy-clean and formatted.

## Files Created/Modified

- `crates/aprender-core/src/nn/tests_named_module.rs` (new, 26 tests) — named/positional agreement, uniqueness, ordered sequences, Sequential recursion, mode-flip byte identity; hosts `snapshot_named`
- `crates/aprender-core/src/nn/module.rs` — three trait defaults + the four documented naming rules; 12 default-behavior tests incl. the cross-module reachability proof
- `crates/aprender-core/src/nn/linear.rs` — `weight`/`bias` overrides mirroring the bias branch
- `crates/aprender-core/src/nn/normalization/mod.rs` — `LayerNorm` `weight`/`bias` overrides
- `crates/aprender-core/src/nn/dropout/mod.rs` — explicit empty override documenting why RNG/seed/mode state is excluded
- `crates/aprender-core/src/nn/container.rs` — `Sequential` prefixing, `set_training` recursion, `get(index)` accessor
- `crates/aprender-core/src/nn/transformer/mod.rs` — `MultiHeadAttention` overrides, confined to `impl Module` (73 insertions, 0 deletions)
- `crates/aprender-core/src/nn/mod.rs` — `pub(crate)` test-module wiring

## Decisions Made

See `key-decisions` in frontmatter. The load-bearing one: **child indices in `Sequential` are container positions, not parameter-bearing positions.** `[Linear, Dropout, Linear]` yields `0.*` and `2.*`, never `0.*` and `1.*`. Renumbering would silently re-address every downstream freeze group the moment a dropout layer is added or removed — a failure with no error signal.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added `Sequential::get(index) -> Option<&dyn Module>`**
- **Found during:** Task 2 (RED phase)
- **Issue:** The plan requires proving `set_training(false)` flips every `Dropout` child inside a `Sequential`. `Sequential` stores children as a private `Vec<Box<dyn Module>>` with no accessor, so child mode state was unobservable and the required assertion could not be written.
- **Fix:** Added a 3-line `get(index)` accessor mirroring the existing `ModuleList::get` in the same file. No behavior change to any existing path.
- **Files modified:** `crates/aprender-core/src/nn/container.rs`
- **Verification:** `named_module_sequential_set_training_recurses_into_dropout_children` now asserts the child's `training()` directly in both directions.
- **Committed in:** `3528218e6` (RED commit, as test scaffolding)

**2. [Rule 2 - Missing Critical] Added a propagation-CHANNEL test beyond the plan's outcome test**
- **Found during:** Task 2 (RED phase, fail-fast investigation)
- **Issue:** The plan's Sequential recursion assertion passed *before* implementation — `Sequential::eval()` already recursed via `child.eval()`, so the default `set_training` inherited working behavior. An outcome-only test would have been a tautological green and would not have detected a `set_training` override that failed to recurse.
- **Fix:** Added `SetTrainingOnlyProbe`, a child overriding only `set_training` (leaving `train`/`eval` as trait no-ops), and asserted a containing `Sequential` flips it. This fails under `train()/eval()`-only propagation and passes only when the parent recurses through `set_training`. Both `Sequential` and `MultiHeadAttention` were then implemented to use that channel.
- **Files modified:** `crates/aprender-core/src/nn/tests_named_module.rs`, `container.rs`, `transformer/mod.rs`
- **Verification:** `named_module_sequential_propagates_through_set_training_channel` — RED before, green after.
- **Committed in:** `3528218e6` (test) / `a2803ba2f` (implementation)

**3. [Plan-permitted choice] `Dropout` states its empty named list explicitly**
- The plan allowed either omitting the override or returning empty explicitly. Chose explicit, with a doc comment recording that `p`, the RNG, the seed, and the training flag are module state and must never be named — naming RNG state would also break the mode-flip byte-identity proof, since RNG state legitimately changes across a forward pass.

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 missing-critical test strength) + 1 plan-permitted choice
**Impact on plan:** No scope creep. The `Sequential::get` accessor is 3 lines with an exact in-file precedent. The channel test strictly strengthens a criterion the plan already required.

## TDD Gate Compliance

Task 2 carried `tdd="true"`. Gate sequence verified in git log:

- **RED** — `3528218e6` `test(01-02)`: 14 of 26 tests failed before implementation. Failures were assertion failures with real diffs (e.g. `["0","1","2","3"]` vs `["0.weight","0.bias","2.weight","2.bias"]`), not compile errors.
- **GREEN** — `a2803ba2f` `feat(01-02)`: 26/26 pass.
- **REFACTOR** — not needed; no cleanup commit.

**Fail-fast check performed.** 12 tests passed during RED. Each was investigated and confirmed legitimate rather than evidence of pre-existing functionality: 6 cover parameter-less modules (`Dropout`, non-affine `LayerNorm`) where the empty result is correct by construction; 4 are mode-flip byte-identity checks that hold because mode flips never touched parameters; 2 are arity-parity checks that pass *because Task 1 established that invariant universally* — they are the intended regression guards for it. The one genuine finding — that `Sequential` mode recursion already worked through `train()/eval()` — is documented as deviation #2 above and produced an additional, genuinely-RED test.

## Issues Encountered

Three verification commands from the plan could not pass as literally written on this platform, for reasons independent of this plan's changes. Each was verified by an equivalent that actually exercises the intended surface (details and evidence in `deferred-items.md`):

- **`cargo check --workspace`** fails because `crates/aprender-profile` hard-stops with `compile_error!("renacer requires Linux (ptrace syscall tracing)")` on non-Linux. Verified instead with `--exclude aprender-profile`: **exit 0**, all other 77 crates clean.
- **`cargo clippy -p aprender-core -- -D warnings`** exits 101 from 20 warnings-as-errors inside the **`aprender-compute` dependency** (cfg-gated SIMD paths inactive on darwin). Zero diagnostics point at `aprender-core`, and `git diff --name-only` confirms no file under `crates/aprender-compute/` was touched. Verified instead with `cargo clippy -p aprender-core --lib --tests`: **exit 0**, and **zero warnings in all seven touched files**.
- **`scripts/check_include_files.sh`** exits 0 but is vacuous on macOS — it uses `grep -oP` (GNU PCRE), so BSD grep errors out and it reports `All 0 include!() files are tracked` on a repo documented as having 562. This plan added no `include!()` file (it uses `#[path]`), and the new source file was verified tracked directly via `git ls-files`, not gitignored (`git check-ignore` exit 1), and matched by no `exclude` pattern in `aprender-core/Cargo.toml`.

## Threat Model Coverage

| Threat ID | Disposition | How mitigated |
|-----------|-------------|---------------|
| T-1-03 (Repudiation — ordering) | mitigated | Positional-fallback default makes named==positional arity/order universal; order pinned by `Vec<String>` equality and `TensorId` identity comparison, not set membership |
| T-1-04 (Tampering — mode switch mutating params) | mitigated | `f32::to_bits` snapshot comparison across train→eval→train on leaves and on a composite containing `Dropout` |
| T-1-17 (Spoofing — duplicate/fallback names) | mitigated | Uniqueness assertions on every touched implementor, a dedicated test that two structurally identical `Sequential` children stay distinct, and an explicit test that no BERT-path module emits a purely numeric (fallback) name |
| T-1-SC (third-party installs) | accepted | No packages installed |

No new trust boundaries, network endpoints, auth paths, file access patterns, or schema changes. No threat flags raised.

## Known Stubs

None. `Dropout::named_parameters` returning empty is semantically correct (it has no learnable parameters), documented, and tested — not a placeholder.

## Next Phase Readiness

Ready for the plans that build on this:

- **01-06 (encoder)** — `snapshot_named` is reachable as `crate::nn::tests_named_module::snapshot_named` from outside `nn`, proven by a compile-time-enforced cross-module test. HF-dotted full names compose from the leaf names and prefix mechanics landed here.
- **01-07 (freeze groups)** — prefix matching against `MultiHeadAttention`'s semantic names will address real tensors; a `LayerAttention(n)` prefix can no longer match zero tensors via a positional fallback.
- **01-08 (gradient gates)** — optimizer partitioning can rely on named/positional arity and order agreement for every implementor.

**Note for 01-06:** `MultiHeadAttention`'s seeded attention-probs dropout extension was deliberately left untouched here, per this plan's instruction to confine changes to the `impl Module` block. That remains 01-06's work.

## Self-Check: PASSED

- All 8 files verified present and tracked (`git ls-files`)
- All 3 commits verified present (`git log`)
- `cargo test -p aprender-core --lib named_module`: 26 passed
- `cargo test -p aprender-core --lib nn::module`: 24 passed
- `cargo test -p aprender-core --lib`: 14,008 passed, 2 ignored, 0 failed
- `cargo fmt -p aprender-core --check`: exit 0

---
*Phase: 01-differentiable-minilm-conformance*
*Completed: 2026-08-08*
