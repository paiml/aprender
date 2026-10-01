---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 03
subsystem: ml-models
tags: [modernbert, laya, encoder, parity, gemm_blis, apr, f16, rope, sliding-window, rayon, provable-contracts]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-01 laya-parity-v1 (embeddings_abs, per_layer_rel_rms, final_norm_abs, window_mutation bars; staged FALSIFY-LAYA-PARITY-002/004)"
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-02 modernbert_tiny fixture (plain-HF F16 safetensors, fp32 transformers ladder oracle, window_mutation record)"
provides:
  - "aprender::models::modernbert: a reusable ModernBERT encoder (feature `parallel`), with no dependency on models/bert or autograd"
  - "ModernBertConfig::from_json_bytes: typed HF parse. It validates the supported domain before any shape is derived and honours norm_eps"
  - "ModernBertEncoder::from_apr(&AprV2ReaderRef, prefix, &config): prefix-aware loader that refuses by tensor name (MissingTensor / ShapeMismatch / Undecodable / NonFinite)"
  - "ModernBertEncoder::forward(ids, tap) -> Result<[L, d]>, with ladder taps emb / layer{i} / final"
  - "Reusable primitives for aprender-decide: Linear (gemm_blis, spike-020 layout), layer_norm(x, d, w, b, eps), gelu_exact, rope_rotate_half, attention(q, k, v, l, heads, hd, window). All return typed errors"
  - "expected_modernbert_tensor_names(config, prefix): the import/load contract"
  - "17 CI-listed lib tests, including tiny_parity (FALSIFY-LAYA-PARITY-002) and window_mutation (FALSIFY-LAYA-PARITY-004), both bound in laya-parity-v1"
affects: [08-04, 08-05, 08-09, 08-10, 08-12]

actuals:
  tokens: 20662   # chars/4 over the realized code diff (82648 chars, rtk proxy git diff db2f820cc..HEAD)
  tasks: 2
  commits: 2
plan_head_before: db2f820cc92e92aee2d649cd398549369bf98daa

tech-stack:
  added:
    - "safetensors 0.4 as an aprender-core dev-dependency (already locked at 0.4.5; tests read the fixture's raw F16 bytes)"
  patterns:
    - "Validated config type: private fields and accessors. Only from_json_bytes constructs one, and weights-bearing structs are built only by the loader (pub(crate) from_parts)"
    - "Reusable numeric primitives return Result with a typed InputShape error instead of panicking on inconsistent buffer lengths"
    - "Parity tests read every bar from the contract YAML at test time, compare NaN-visibly and print measured max|delta| plus ARCH"
    - "In-memory .apr round trip in tests: fixture safetensors -> AprV2Writer::add_tensor(F16) -> AprV2ReaderRef::from_bytes"

key-files:
  created:
    - crates/aprender-core/src/models/modernbert/mod.rs
    - crates/aprender-core/src/models/modernbert/config.rs
    - crates/aprender-core/src/models/modernbert/gemm.rs
    - crates/aprender-core/src/models/modernbert/embeddings.rs
    - crates/aprender-core/src/models/modernbert/layer.rs
    - crates/aprender-core/src/models/modernbert/encoder.rs
    - crates/aprender-core/src/models/modernbert/load.rs
  modified:
    - crates/aprender-core/src/models/mod.rs
    - crates/aprender-core/Cargo.toml
    - contracts/laya-parity-v1.yaml

key-decisions:
  - "The reusable primitives (Linear::forward, layer_norm, attention, rope_rotate_half) return Result<_, ModernBertError> rather than a bare Vec, so aprender-decide's head and scorer cannot panic on a bad buffer. Consumers use `?`."
  - "layer_norm rounds the configured eps to f32 before widening, because torch's CPU LayerNorm casts eps to the f32 accumulation type. That keeps the spike's f64::from(1e-5f32) bit-identical while still honouring norm_eps."
  - "ModernBertConfig is a validated type (private fields, accessors). Encoder, layer and embeddings are constructed only by the loader, so the domain checks cannot be bypassed."
  - "The #[contract] equations are from_apr -> embeddings_abs (the load/widening rung) and forward -> per_layer_rel_rms."
  - "An empty row is refused (EmptyInput): a zero-row transpose would otherwise chunk by 0. The loader also refuses an undecodable dtype or truncated data (Undecodable), which goes beyond the three planned refusals."

patterns-established:
  - "Prefix-aware loading: one expected-names function drives both the load contract and the load order"
  - "A test-only #[cfg(test)] override (window) makes a mutation rung possible without a runtime knob"

requirements-completed: [D-13, D-17]

coverage:
  - id: D1
    description: "A plain HF ModernBERT (tiny fixture) loads from an in-memory .apr through the prefix-aware loader and matches the transformers fp32 oracle on embeddings, every layer and the final norm, within laya-parity-v1 bars read from the YAML"
    requirement: "D-17"
    verification:
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/mod.rs#tests::tiny_parity"
        status: pass
    human_judgment: false
  - id: D2
    description: "The local window is |i-j| <= local_attention/2 inclusive: w-1 and w+1 each fail the first local layer while the first global layer passes. The attended range is also checked exhaustively over 512x512"
    requirement: "D-17"
    verification:
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/mod.rs#tests::window_mutation"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/layer.rs#tests::window_predicate_exhaustive_512"
        status: pass
    human_judgment: false
  - id: D3
    description: "Reusable encoder (D-13): the same tensors under the `encoder.` prefix load bit-identically, and the loader refuses a missing tensor, a wrong shape, and a NaN or Inf F16 value by name"
    requirement: "D-13"
    verification:
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/mod.rs#tests::prefix_reuse"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/load.rs#tests::refuses_missing_tensor_by_name"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/load.rs#tests::refuses_shape_mismatch"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/load.rs#tests::refuses_non_finite"
        status: pass
    human_judgment: false
  - id: D4
    description: "The config's supported domain is enforced before any allocation (nine structural refusals, eps/theta, bias flags, missing theta, layer_types agreement), and norm_eps is honoured"
    requirement: "D-13"
    verification:
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/config.rs#tests::config_domain"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/config.rs#tests::layer_types_agree"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/config.rs#tests::unsupported_config"
        status: pass
      - kind: unit
        ref: "crates/aprender-core/src/models/modernbert/mod.rs#tests::norm_eps_honoured"
        status: pass
    human_judgment: false
  - id: D5
    description: "The modernbert tests are compiled into CI's workspace lib run, and the new module is clippy-clean against the base-commit baseline"
    verification:
      - kind: other
        ref: "cargo nextest list --workspace --lib --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute (17 modernbert:: tests)"
        status: pass
      - kind: other
        ref: "cargo clippy -p aprender-core --lib --no-deps -- -D warnings (0 under models/modernbert; 1 outside = base 1)"
        status: pass
    human_judgment: false

duration: 26min
completed: 2026-09-26
status: complete
---

# Phase 8 Plan 03: ModernBERT Encoder in aprender-core Summary

**The spike-025 ModernBERT encoder now lives in aprender-core on `gemm_blis` and the prefix-aware `.apr` F16 loader. Its config parse validates the domain, every primitive returns a typed error, and on the first run it matches transformers fp32 on the tiny fixture: layer rel-rms ≤ 2.2e-7, final norm ≤ 1.1e-6. The window mutation, prefix reuse and loader refusals each have a test, and all 17 tests are in CI's lib run.**

## Performance

- **Duration:** 26 min
- **Started:** 2026-09-26T00:56:32Z
- **Completed:** 2026-09-26T01:23:22Z
- **Tasks:** 2 (Task 1 tracer, Task 2 tdd expansion)
- **Files modified:** 10 (7 created, 3 modified)

## Accomplishments

- **`aprender::models::modernbert`** (behind the default `parallel` feature) is a lift of the spike-025 encoder. Every numeric operation keeps the spike's order; the erf is now the house `batuta_common::math::erfc_precise`. The module does not use `models::bert` or `autograd` (D-13).
- **tiny_parity passed on the first run** (aarch64, Apple M4). The path is fixture F16 safetensors, then an in-memory `.apr`, then a load with prefix `""`, then forward. Bars come from `laya-parity-v1`:

  | row | emb max\|Δ\| (bar 1e-5) | layer0 / 1 / 2 rel_rms (bar 1e-3) | final max\|Δ\| (bar 1e-4) |
  |---|---|---|---|
  | 0 (24 tok) | 4.77e-7 | 1.25e-7 / 1.72e-7 / 2.13e-7 | 9.54e-7 |
  | 1 (21 tok) | 4.77e-7 | 1.32e-7 / 1.79e-7 / 2.12e-7 | 1.07e-6 |

- **window_mutation discriminates:** half-window 3 moves layer1 to 8.19e-2 and 1.020e-1, and half-window 5 moves it to 6.53e-2 and 7.87e-2. Layer0 stays at 1.25e-7 and 1.32e-7. These values reproduce the record in `oracle.json` `window_mutation` exactly (0.0819 / 0.101981 / 0.065296 / 0.0787).
- **Domain validation happens before allocation.** There are nine structural refusals, plus eps/theta, bias flags, missing theta, and `layer_types` length, value and agreement. Every dimension product is checked with `checked_mul`. Setting `norm_eps` to 1e-1 moves the final norm by 1.77 (bar 1e-4), which shows the value is used.
- **Loader refusals name the tensor:** a missing tensor (with or without the prefix in the name), a shape mismatch, and F16 NaN `0x7E00` / Inf `0x7C00`. The `encoder.` prefix loads bit-identically to `""`.
- **CI reach is proven:** `cargo nextest list` with CI's exact exclude set lists 17 `modernbert::` tests, and `cargo check -p aprender-core --no-default-features --lib` exits 0.
- The contract binds `tiny_parity` to FALSIFY-LAYA-PARITY-002 and `window_mutation` to FALSIFY-LAYA-PARITY-004. KANI-LAYA-PARITY-002's prose now names the exhaustive 512×512 window-predicate test. `pv validate` reports 0 errors.

## Task Commits

1. **Task 1: Tracer — tiny HF ModernBERT → in-memory .apr → prefix load → forward → oracle**: `06b5a34d0` (feat)
2. **Task 2: Expansion — window mutation, prefix reuse, loader refusals, config domain, CI reach**: `3437afdc1` (test). No feat commit, because no implementation was needed; see TDD Gate Compliance.

**Plan metadata:** recorded in the docs commit that carries this SUMMARY.

## Files Created/Modified

- `crates/aprender-core/src/models/modernbert/mod.rs`: architecture and supported-domain docs, re-exports, `ModernBertError`, the `test_support` fixture plumbing (YAML bars, NaN-visible `within`, rel_rms), and the parity, window, prefix, eps and oov tests
- `.../modernbert/config.rs`: `ModernBertConfig` (validated, accessors), `ModernBertConfigError`, and 5 config tests
- `.../modernbert/gemm.rs`: `Linear` on `trueno::blis::gemm_blis`, using the spike-020 layout and the band rule, with typed `Gemm` and shape errors
- `.../modernbert/embeddings.rs`: `ModernBertEmbeddings` (gather + bias-free LayerNorm, with a typed out-of-vocab error)
- `.../modernbert/layer.rs`: `layer_norm`, `gelu_exact`, `rope_rotate_half`, `attention` (inclusive window), `ModernBertLayer`, and the gelu, window-predicate and shape tests
- `.../modernbert/encoder.rs`: `ModernBertEncoder::forward` with its ladder tap and the `#[cfg(test)]` window override
- `.../modernbert/load.rs`: `expected_modernbert_tensor_names`, `ModernBertEncoder::from_apr`, `ModernBertLoadError`, and 4 loader tests
- `crates/aprender-core/src/models/mod.rs`: `#[cfg(feature = "parallel")] pub mod modernbert;` plus a `pub use`
- `crates/aprender-core/Cargo.toml`: `safetensors = "0.4"` dev-dependency
- `contracts/laya-parity-v1.yaml`: `test:` lines on FALSIFY-002 and FALSIFY-004, and the KANI-002 harness prose

## Decisions Made

See `key-decisions` in the frontmatter. What plan 08-04 needs to know:
- `layer_norm(x, d, w, b, eps) -> Result<Vec<f32>, ModernBertError>`. Pass the encoder config's `norm_eps()` for encoder-shaped norms and 1e-5 for Laya's head `nn.LayerNorm`s.
- `attention(q, k, v, l, heads, hd, window) -> Result<Vec<f32>, _>` and `Linear { w, b, out, inp }::forward(x, m) -> Result<Vec<f32>, _>` validate every length.
- `gelu_exact(f32) -> f32` is infallible.
- `ModernBertEncoder::from_apr(&reader, "encoder.", &config)` is the Laya entry point, and `forward(ids, tap)` returns the final-norm `[L, d]`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical] Added tests beyond the plan's named set**
- **Found during:** Tasks 1 and 2
- **Issue:** Some mitigations had no test. T-08-03-01 (out-of-vocab ids and empty rows) was untested, the eps/theta refusals were outside `config_domain`, KANI-LAYA-PARITY-002 declared "an exhaustive 512×512 enumeration test (plan 08-03)" that did not exist, and the reusable primitives' shape refusals were untested.
- **Fix:** Added `forward_refuses_out_of_vocab_and_empty`, `config_domain_eps_and_theta`, `parses_the_tiny_fixture`, `expected_names_match_the_hf_fixture`, `window_predicate_exhaustive_512` and `primitives_refuse_bad_shapes`. KANI-002's prose now names the test.
- **Files modified:** modernbert/{mod,config,load,layer}.rs, contracts/laya-parity-v1.yaml
- **Verification:** 17/17 pass. The window predicate test went RED under an exclusive-bound mutation (`i=0 j=64`).
- **Committed in:** 06b5a34d0, 3437afdc1

**2. [Rule 3 - Blocking, not fixable in scope] Neither strict-binding acceptance criterion can print its PASS line**
- **Found during:** Task 1 and Task 2 verify (`bash scripts/check_contract_test_binding.sh`)
- **Issue:** The guard exits rc=1 with `VACUOUS ... SKIPPED`, because `contracts/spectral-indices-v1.yaml` has no `kani_harnesses:`. This is the pre-existing D-ITEM-08-01-A. As a result, Task 2's first `<automated>` verify also exits 1, even though its cargo-test half passes (17 ok).
- **Fix:** None in scope. I measured around it instead: I ran `pv lint --strict-test-binding` on a lifted copy (a temporary sibling dir, deleted afterwards). It resolved 587 refs (08-01 had 585, and this plan adds 2), with **0 dangling in laya-parity-v1**. The other 15 contracts and 44 dangling references are unchanged from 08-01. A mutated binding name was flagged dangling, so the check is not vacuous.
- **Files modified:** deferred-items.md (D-ITEM-08-03-A)
- **Committed in:** the docs commit carrying this SUMMARY

**3. [Process] The window_mutation `test:` line was amended into the test commit**
- The test commit was first made without the contract edit. I amended it (the commit is unpushed) so that the binding lands in the same commit as the test, as the plan requires.

---

**Total deviations:** 1 auto-added test set (Rule 2), 1 pre-existing blocker documented (Rule 3, out of scope), 1 process note.
**Impact on plan:** The code changes match the plan exactly. The only unmet acceptance clause is the guard's PASS line, which is blocked by another phase's contract.

## TDD Gate Compliance

- **RED (Task 2):** `3437afdc1 test(08-03)`. Every Task 2 test passed on its first run, so RED was an **unexpected GREEN** (fail-fast rule 1). The cause is structural. Task 1's own `<action>` required the domain validation, loader refusals, eps threading and `#[cfg(test)]` window override, and all of them had to exist for the tracer to be production-quality. Task 2 therefore had nothing left to go RED against.
- **Substitute RED evidence:** I gave each targeted test an induced mutation in the committed code, ran it, and reverted with `git checkout -- <file>`. **11 RED records** were persisted and `gsd check tdd-red-evidence` returned `RED_EVIDENCE_OK` for each. The mutations and the tests that caught them:
  - odd-head-dim check removed: `config_domain`
  - disagreement check removed: `layer_types_agree`
  - missing theta defaulted: `unsupported_config`
  - eps ignored: `norm_eps_honoured` (Δ = 0)
  - tanh GELU: `gelu_exact_reference` (4.1e-4), and `tiny_parity` also went RED at the final norm (3.6e-4)
  - finiteness check skipped: `refuses_non_finite`
  - shape check skipped: `refuses_shape_mismatch`
  - prefix dropped from the error: `refuses_missing_tensor_by_name`
  - prefix dropped from a name: `prefix_reuse`
  - exclusive window: `window_mutation`, `window_predicate_exhaustive_512` and `tiny_parity` (layer1 5.0e-2)
- **GREEN:** there is no `feat` commit for Task 2, because no implementation change was needed. Task 1's `feat` commit `06b5a34d0` precedes the tests.
- **REFACTOR:** none.

## Issues Encountered

- **The rtk hook filtered the base clippy capture.** The first baseline log held only rtk's summary line, with no `Checking aprender-core` line. I recovered it from rtk's raw tee log, which is the same run, and ran every later cargo command through `rtk proxy` or the plan's `rtk run`. The base count is **1**: `demo/reliable/performance.rs:126` `unreachable_code`, the known arm64 finding.
- **Is the clippy gate vacuous?** I checked. A probe `needless_return` in modernbert is reported alongside the pre-existing base error, so clippy's late lint pass does run. My first probe (`&Vec` / `ptr_arg`) was invalid, because clippy's `avoid-breaking-exported-api` suppresses it on pub fns. Pedantic lints are active: `unreadable_literal` and `bool_to_int_with_if` stayed silent only because the workspace allows them. The test target (`--lib --tests`) also has 0 modernbert diagnostics.
- **`pv lint` rewrote the tracked `.pv/lint-previous.json`.** I restored it with `git checkout -- .pv/lint-previous.json`, so it is not part of this plan.
- **The windows ledger refuses every append** (`Ledger entry 24 has invalid status: "resolved"`), a pre-existing state. Deviation 2 is therefore recorded in deferred-items.md (D-ITEM-08-03-A) instead.

## Known Stubs

None. A scan for TODO/FIXME/placeholder in `models/modernbert/*.rs` found nothing (rc 1).

## Threat Flags

None. The only new trust boundary is .apr bytes into the loader, and the plan's threat model already covers it (T-08-03-01, -02, -05 and -06 are each mitigated and tested).

## User Setup Required

None. No external service configuration is required.

## Next Phase Readiness

- Plan 08-04 can build Laya's head and scorer on `ModernBertEncoder::from_apr(reader, "encoder.", &config)` and reuse `Linear`, `layer_norm`, `attention` and `gelu_exact`. All primitives return `Result`, so consumers propagate with `?`.
- The x86_64 value for the laya-parity-v1 provisional bar is still unrecorded: CI's first `workspace-test` run prints it through `tiny_parity`.
- D-ITEM-08-01-A / 08-03-A still keep the strict-binding guard from printing PASS.

## Self-Check: PASSED

- All 10 key files were found on disk (the 7 module files plus models/mod.rs, Cargo.toml and laya-parity-v1.yaml).
- Commits `06b5a34d0` and `3437afdc1` exist (`git log --oneline --all`). The ledger count is `git rev-list --count db2f820cc..HEAD` = 2.
- Plan verification was re-run:
  - tiny_parity verify rc 0
  - clippy verify rc 0 (now=1, base=1, 0 modernbert)
  - 17 modernbert tests pass (≥ 10)
  - nextest list shows 17 (≥ 8)
  - `--no-default-features` check rc 0
  - The strict-binding guard is the one exception: rc 1, pre-existing (Deviation 2).

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-26*
