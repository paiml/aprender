# Phase 1: Differentiable MiniLM Conformance - Research

**Researched:** 2026-08-07
**Domain:** Differentiable BERT sentence encoder (pure Rust, aprender-core autograd), HF tokenizer parity, contract-gated numerical conformance
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Encoder Home and Structure
- **D-01:** Build a new `BertSentenceEncoder` in `aprender-core` that reuses `crates/aprender-core/src/models/bert/config.rs` and `bert/load.rs` but implements a fully graph-connected forward path. **Do not refactor `crates/aprender-core/src/models/bert/` in place during this phase.** `apr embed`, `apr rerank`, and `CrossEncoder` stay untouched.
- **D-02:** Convergence onto a single encoder is deferred to **Phase 4**, where OPS-03 already requires generic APR commands to call the shared core model. That is where the inference regression sweep belongs, with APR parity tests to prove it. This ordering is deliberate and follows STATE's constraint: *validate the real-weight mixed-batch graph before committing the full BERT refactor.*
- **D-03:** The six missing differentiable primitives — batched embedding gather, additive attention masking, masked mean pooling with a checked non-zero denominator, row-wise L2 normalization with explicit epsilon, cosine similarity, and MSE reduction — are implemented as **model-agnostic ops in `crates/aprender-core/src/autograd/ops/`**, alongside the existing matmul/elementwise backward implementations. They are **not** gated behind the `setfit` feature. This directly retires the CONCERNS.md "embedding and pooling operations detach the graph" debt rather than routing around it.
- **D-04:** Each new op ships with its own central-finite-difference test.

#### Feature Gating
- **D-05:** `BertSentenceEncoder` and the new `tokenizers` 0.23.1 dependency live behind a **`setfit` feature** in `aprender-core` from this phase forward. The `autograd/ops` primitives stay ungated.
- **D-06:** Establish the CPU feature matrix SAFE-02 will need now, while it is cheap: `--no-default-features`, `--features setfit`, and all-features must each build and test. A minimal inference consumer must never transitively pull `tokenizers`.

#### Tokenizer Boundary
- **D-07:** Tokenizer and encoder are **separate types**. A tokenizer type produces a typed `SentenceBatch` (input ids, token type ids, attention mask, truncation facts, input provenance); the encoder consumes only `SentenceBatch`.
- **D-08:** Both are owned by **one bound model type that is the sole public entry point**, so a mismatched tokenizer/encoder pair is not constructible. Tokenizer parity and encoder-forward parity remain independently falsifiable gates.

#### Reference Fixtures and Offline CI
- **D-09:** Commit a **small deterministic real-weight slice** of the pinned MiniLM in APR form (reduced layer/vocab subset, target a few hundred KB). Every CI job runs the graph, masking, pooling, normalization, and gradient gates against **real weight values**, not shapes. Synthetic-shape-only testing is explicitly rejected — PF-011 identifies it as how overclaimed compatibility ships.
- **D-10:** The **full ~90MB real-weight parity suite** runs behind an ignored/feature-gated test target fed by a locally fetched artifact. It is not in the default gate and the full weights are **not** vendored in git.
- **D-11:** Fixtures are committed as **human-readable JSON** under the crate's test tree.
- **D-12:** A **hash-locked `uv` environment** (`setfit==1.1.3`, `sentence-transformers==5.7.0`, `torch==2.13.0`, `scikit-learn==1.9.0`) plus a `scripts/` generator regenerates them as a **deliberate developer workflow — never in CI**. Commit the resolved lockfile with hashes.
- **D-13:** Commit a **manifest of fixture SHA-256s** so any regeneration appears as a reviewable diff. This is the structural guard against quietly re-baselining a fixture to match Rust.
- **D-14:** **Tolerances live in the phase contract**, derived from Python-side f32/f64 round-trip noise, and are committed **in their own commit before any Rust comparison executes**. Loosening one then requires a contract edit that `pv diff` flags with a semver bump suggestion. This satisfies the STATE blocker: *freeze numerical tolerances from pinned reference fixtures before examining Rust discrepancies.*
- **D-15:** Freeze the **full ENC-01..06 fixture corpus in one pass**: tokenizer ids / type ids / masks / truncation facts, **per-layer transformer token outputs**, masked mean, normalized sentence embedding, pair cosine-MSE forward, selected parameter gradients, one controlled optimizer step, plus mixed-length padded batches and batch-1 vs batch-N. Per-layer intermediates are required so a mismatch localizes to a layer instead of the whole encoder.
- **D-16:** Dropout is **disabled** for all cross-framework forward and gradient fixtures. Rust seeded dropout placement, reproducibility, and statistics are tested separately; Python and Rust RNG streams are not expected to match bit-for-bit.

#### Named Parameters, Modes, and Freezing
- **D-17:** **Extend the existing `Module` trait** in `crates/aprender-core/src/nn/module.rs` with named recursive traversal and train/eval propagation. **Do not introduce a separate `NamedModule` / `ParameterStore`** — core keeps exactly one parameter abstraction. Existing `Module` implementors (`nn/linear.rs`, `nn/normalization/`, `nn/dropout/`, `nn/container.rs`) gain named traversal as a side effect.
- **D-18:** Parameter names are **HF dotted, matching the source checkpoint verbatim** (e.g. `encoder.layer.0.attention.self.query.weight`), so gradient fixtures align with torch's `named_parameters()` with zero translation at the phase's highest-risk gate.
- **D-19:** The canonical mapping to `contracts/tensor-names-v1.yaml` is declared and validated at the **Phase 4 APR write boundary**, using `crates/aprender-core/src/format/converter/`. Phase 1 does not carry that mapping.
- **D-20:** **Default freeze policy is all-trainable**, matching SetFit's full-body fine-tuning. Any other default would be an unlabeled deviation from SetFit, which PF-008 treats as a claims defect.
- **D-21:** Freeze groups are opt-in configuration, and the test suite **pins at least one deliberately frozen group** so success criterion 4's "frozen components remain byte-identical" has something real to assert.
- **D-22:** Groups are addressed **per-module, per-layer** — `embeddings`, `encoder.layer.3.attention`, `encoder.layer.3.ffn`, `encoder.layer.3.norm` — matching exactly the components ENC-04 names. Not coarse-only (top-N freezing must be expressible) and not a per-tensor glob DSL (a freeze policy in the Phase 4 APR must be validated structure, not a string).

#### Gate Expression
- **D-23:** Author **one new `contracts/setfit-encoder-conformance-v1.yaml`** owning all six ENC criteria plus the frozen tolerance table. It **references** the existing contracts rather than editing them; `encoder-forward-v1`, `nn-training-gradient-path-v1`, and `pool-flatten-embedding-backward-gradflow-v1` stay stable for their current consumers. `pv status` then reports one coherent Phase 1 gate and `pv diff` versions the whole conformance surface — including the tolerances — together.
- **D-24:** Prove detachment failure **in-band**: a test-only detached encoder variant that the gradient gate must reject, running in every `cargo test`. This is the evidence the gate is not theater.
- **D-25:** Back it with **`cargo-mutants` scoped to the new `autograd/ops` and encoder forward** to catch detachment nobody thought to write a test for.
- **D-26:** Wire the fixture-parity, gradient, and detach-negative tests into **`make tier2`** (pre-commit), with `pv validate contracts/setfit-encoder-conformance-v1.yaml` in **tier3/tier4** alongside the other contract gates. The slow gated real-weight suite stays out of the fast loop. A dedicated target outside the tiers was rejected — a target outside the tiers is a target that stops being run.
- **D-27:** Annotate each new autograd op and the encoder forward with **`#[contract]`**, binding equations through `BindingRegistry` into `crates/aprender-core/src/generated_contracts.rs`, backed by YAML falsification tests. Satisfies Rule 7 (coverage + contracts co-evolution) as the code lands rather than retroactively.

### Claude's Discretion

The user did not delegate any decision explicitly. The following were surfaced as remaining gray areas and consciously left to research/planning as implementation detail:

- Exact dropout placement within the BERT block and the seeded-RNG policy proving ENC-05's "registered parameters do not change merely because the mode changes"
- Exact signatures and shape conventions of the six new autograd ops
- The reproducible derivation procedure for the committed small-slice APR from the pinned revision
- Whether `bert/load.rs` is reused directly or wrapped for pinned-revision import validation

### Deferred Ideas (OUT OF SCOPE)

- **Converging `apr embed` / `rerank` / `CrossEncoder` onto the shared encoder** — Phase 4, where OPS-03 already requires it and APR parity tests can prove it (D-02).
- **Canonical `tensor-names-v1.yaml` mapping for encoder tensors** — Phase 4 APR write boundary (D-19).
- **Alternative encoder objectives** (InfoNCE, SupCon, CoSENT, triplet) — v2, per REQUIREMENTS.md EXT-02. Not a Phase 1 option; they change SetFit fidelity and numerical references.
- **Additional sentence-transformer families** — v2 (EXT-01), each requiring its own named adapter and parity corpus.
- **Accelerator paths** — deferred until the full CPU lifecycle is proven; ACC-01 in v2.
- **Persistent embedding/token caches** — explicitly out of scope for v1 (CACHE-01); PF-010 documents the staleness risk.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENC-01 | Import pinned `all-MiniLM-L6-v2` revision into a typed SetFit encoder contract; unsupported architecture/tokenizer/pooling/config variants fail with typed errors | Pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` verified live on Hub API (Apache-2.0, `BertModel`). `BertConfig::minilm_l6()` preset matches the contract table exactly. `BertLoadError` typed-error precedent in `bert/load.rs`. Import validation wraps (not modifies) `bert/config.rs`/`bert/load.rs` — see Architecture Patterns |
| ENC-02 | Batch-tokenize once; ordered token IDs, type IDs, attention masks, truncation facts, stable input provenance | `tokenizers = 0.23.1` verified current on crates.io; `default-features = false, features = ["fancy-regex"]` confirmed to exclude onig (C), esaxx (C++), progressbar, http. `SentenceBatch` type per D-07; tokenizer-parity JSON fixtures per D-15 |
| ENC-03 | Single/mixed-length padded batches through one shared `Transformer -> masked mean pooling -> L2 normalization` path with fixture-verified outputs | `MultiHeadAttention::forward_self(x, attn_mask: Option<&Tensor>)` already accepts `[B,S,E]` + additive mask; attention backward gradflow exists (PMAT-914). New ops: masked mean pool (checked denominator), row L2 normalize (explicit eps). Per-layer fixture intermediates localize mismatches |
| ENC-04 | Named parameter enumeration, frozen/trainable groups, finite non-zero gradients and parameter changes after a controlled AdamW step | `Module` trait exists (positional only) — D-17 extends it with named traversal. `AdamW::step_with_params(&mut [&mut Tensor])` at `nn/optim/mod.rs:391`. Freezing = exclusion from optimizer param set + `requires_grad(false)`; `lora-adapter-trains-base-frozen-v1.yaml` is the frozen-vs-trainable proof precedent |
| ENC-05 | Recursive train/eval switching with dropout, without changing registered parameters | Dropout placement verified from HF transformers v4.57.1 source (see Architecture Patterns). Existing `Dropout::with_seed(p, seed)` + train/eval flags; PMAT-922 already fixed the dropout graph-severing bug via constant-mask `Tensor::mul`. Mode-flip byte-identity test pattern specified |
| ENC-06 | Finite tensor-valued cosine-similarity MSE pair loss, graph-connected, matching frozen forward/gradient fixtures; deliberate detachment fails the gate | Cosine + MSE as new autograd ops with `GradFn` structs following the exact `ops/mod.rs` pattern; existing f32-returning loss utilities (`loss/loss.rs`, `nn/self_supervised.rs`) must NOT be reused. In-band detach-negative test (D-24) + scoped cargo-mutants 25.3.1 (installed, verified) |
</phase_requirements>

## Summary

This phase builds the first fully graph-connected sentence-encoder path in `aprender-core`. The codebase research shows the risk is narrower than it first appears: **most of the hard autograd machinery already exists**. `EmbeddingBackward` (scatter-add gather backward) is already in `grad_fn.rs:1100` and is batch-agnostic (it operates on a flat index list — batched gather is the same grad_fn with `B*S` flattened indices). Attention backward gradflow was fixed in PMAT-914, dropout graph-severing was fixed in PMAT-922, LayerNorm backward has dedicated gradflow tests, and `MultiHeadAttention::forward_self` already takes `[batch, seq, embed]` input with an optional additive mask, a dropout probability, and a training flag. What is genuinely missing is exactly the six primitives D-03 names, the named-parameter/mode extension to `Module`, the typed import contract, and the fixture/contract harness.

The single highest technical risk carried from milestone research is unproven **gradient parity at batch > 1** through the existing attention/LayerNorm broadcast paths — repository inspection establishes intent but not end-to-end proof. The first plan should therefore be a spike that runs a two-sentence mixed-length batch through embeddings-gather → 2 layers → masked pool → normalize → cosine-MSE → backward and asserts finite non-zero grads on every weight, before the full encoder, fixtures, and contract land. A second structural risk found during this research: the committed small-slice APR (D-09) cannot itself pass the ENC-01 full-architecture pin (6 layers/384 hidden), so the plan must give the slice a test-only load path that bypasses only the architecture pin while keeping every structural check — otherwise the fixture gates and the typed-rejection gates contradict each other.

All external pins verified today: MiniLM revision `1110a243…` is the live main sha on the Hub (Apache-2.0, BertModel); `tokenizers` 0.23.1 is current on crates.io with the pure-Rust `fancy-regex` feature confirmed against the docs.rs source manifest; all four Python reference packages are the current PyPI releases and pass slopcheck. HF dropout placement was verified directly from transformers v4.57.1 source, resolving the main discretion area.

**Primary recommendation:** Plan the phase as (1) batched-graph spike on the six ops, (2) Module named-traversal extension, (3) tokenizer boundary + typed import, (4) Python fixture corpus + tolerance-first contract commit, (5) full conformance gate wiring (tier2/tier3, detach-negative, cargo-mutants) — in that order, with the tolerance table committed before any Rust-vs-fixture comparison runs (D-14).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Six differentiable primitives (gather, additive mask, masked mean, L2 norm, cosine, MSE) | `aprender-core/src/autograd/ops/` + `grad_fn.rs` (ungated) | — | D-03: model-agnostic, retires the detachment debt for all consumers, not just SetFit |
| Named parameter traversal + train/eval propagation | `aprender-core/src/nn/module.rs` (`Module` trait, ungated) | Existing implementors (`linear`, `normalization`, `dropout`, `container`, `transformer`) | D-17: exactly one parameter abstraction in core |
| Tokenization → `SentenceBatch` | New `setfit` module in `aprender-core` (feature `setfit`) | `tokenizers` 0.23.1 crate | D-05/D-07: separate type, feature-gated, produces first-class token facts |
| Graph-connected encoder forward (`BertSentenceEncoder`) | New `setfit` module (feature `setfit`) | `nn/` building blocks (`Linear`, `LayerNorm`, `MultiHeadAttention`, `Dropout`), `autograd/ops` | D-01: new type; existing `models/bert/` untouched |
| Pinned-revision import validation | New `setfit` import module wrapping `bert/config.rs` + `bert/load.rs` | `format/v2` `AprV2Reader` | D-01 reuse; typed errors per ENC-01; wrap, don't modify (see Open Questions Q2 for slice bypass) |
| Pair cosine-MSE loss (tensor-valued) | New `setfit` module (thin composition over ungated ops) | `autograd/ops` | ENC-06; must not reuse f32-returning loss utilities |
| Controlled optimizer step | Existing `nn/optim` `AdamW::step_with_params` | — | ENC-04; no new optimizer |
| Conformance gate + tolerances | `contracts/setfit-encoder-conformance-v1.yaml` via `pv` | `generated_contracts.rs` (`pv codegen`), `make tier2/tier3` | D-23–D-27; `pv` is the only sanctioned contract tool |
| Fixture corpus + regeneration | `crates/aprender-core/tests/` JSON + committed slice APR; `scripts/` uv generator (dev-only) | Hash-locked Python env | D-09–D-16; never in CI |

## Project Constraints (from CLAUDE.md)

Directives the planner MUST honor (verified against `./CLAUDE.md`):

1. **Branch protection:** `main` is protected; all work on feature branches, PRs via `gh pr create`, CI (`ci / gate` + `workspace-test`) must pass.
2. **Code search:** `pmat query` (semantic), never grep/glob for code search.
3. **Contract tooling:** DOGFOOD `pv`, never bash/yq/python workarounds. If `pv validate` rejects the new contract kind, the sanctioned fixes are: restructure to the existing `KernelContract` shape, or extend `aprender-contracts/src/schema/` as its own engineering task.
4. **Tensor layout:** LAYOUT-001/002 — row-major everywhere; `contracts/tensor-layout-v1.yaml` is source of truth; `_colmajor` imports forbidden. Every new op is row-major.
5. **Lints:** `unsafe_code = "forbid"`; `unwrap()` banned via `.clippy.toml` disallowed-methods — use `expect()` or `ok_or_else(|| ...)?`; clippy pedantic `-D warnings`.
6. **Quality tiers:** `make tier1` (<1s) / `tier2` (<5s pre-commit) / `tier3` (pre-push) / `tier4` (CI). Coverage floor 88% (target ≥95%); SATD 0; complexity ≤10/fn; mutation ≥85%.
7. **Rule 7 (coverage + contracts co-evolution):** tests for new code must land with `#[contract]` annotations and YAML falsification strengthening — "Coverage without contracts is REJECTED."
8. **Publishing safety (CB-510):** root-anchored gitignore patterns; after adding committed fixture files run `git check-ignore -v <path>` (must exit 1) and re-run `scripts/check_include_files.sh` / `check_package_includes.sh` if `include!()` files are added.
9. **Shell scripts:** `bashrs lint` conventions (`set -euo pipefail`, quoted vars); sourced libraries must be option-neutral. NOTE: `bashrs` is not currently installed on this machine (see Environment Availability).
10. **Verification discipline:** never read `$?` through a pipe; prove mechanisms engaged (a "gradient test passed" claim must cite the actual gradient values/graph evidence, not intent); one failing input is an anecdote — vary before naming a cause.
11. **Realizar-first does not apply here:** this phase is training-side library work in `aprender-core` (aprender owns model training per the responsibility table); no inference/serving surface is added.
12. **justfile preference (user-global):** the user prefers justfile for project scripts, but this repository's established convention is `Makefile` tiered gates (`make tier2` is locked by D-26). Follow the repo convention; do not introduce a parallel justfile for this phase.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `aprender-core::autograd::Tensor` | workspace (in-tree) | Only tensor/graph for the trainable path | Locked by STATE decision; existing `GradFn`/graph registry, `no_grad()` scope, backward tests [VERIFIED: codebase] |
| `nn::{Linear, LayerNorm, Dropout, MultiHeadAttention}` | workspace (in-tree) | Encoder building blocks | All are `Module` implementors with backward-gradflow test coverage (`tests_attention_backward_gradflow.rs`, `tests_norm_backward_gradflow.rs`); MHA takes `[B,S,E]` + `Option<&Tensor>` additive mask + dropout_p + training flag [VERIFIED: codebase] |
| `nn::optim::AdamW` | workspace (in-tree) | Controlled optimizer step (ENC-04) | `step_with_params(&mut self, params: &mut [&mut Tensor])` at `nn/optim/mod.rs:391` [VERIFIED: codebase] |
| `models/bert/{config,load}.rs` | workspace (in-tree) | Config preset + APR weight loading, reused per D-01 | `BertConfig::minilm_l6()` matches the encoder contract exactly (384/6/12/1536/30522/512/eps 1e-12/pad 0); `BertLoadError` typed error precedent [VERIFIED: codebase] |
| `tokenizers` | 0.23.1 | Exact HF WordPiece pipeline, batch encoding | Locked by D-05. Current on crates.io [VERIFIED: crates.io registry via `cargo search`, cross-checked docs.rs 0.23.1 source manifest]. `default-features = false, features = ["fancy-regex"]` removes default `onig` (C regex), `esaxx_fast` (C++), `progressbar`; `http` stays off [CITED: docs.rs/crate/tokenizers/0.23.1/source/Cargo.toml] |
| `sentence-transformers/all-MiniLM-L6-v2` | rev `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Pinned encoder | Revision is the live main sha; Apache-2.0; `architectures: ["BertModel"]` [VERIFIED: HF API, fetched 2026-08-07] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `serde` / `serde_json` | workspace | JSON fixture load, config parsing | Fixture corpus (D-11), config validation |
| `sha2` | workspace | Fixture SHA-256 manifest (D-13) | Manifest generation/verification test |
| `proptest` | workspace | Property tests (bounded via `PROPTEST_CASES`) | Padding invariance, mask properties |
| `cargo-mutants` | 25.3.1 (installed) | Detachment mutation gate (D-25) | Scoped: `cargo mutants --file 'crates/aprender-core/src/autograd/ops/*' ...` |
| `pv` (aprender-contracts-cli) | in-tree | Contract authoring/validation/codegen (D-23, D-27) | `pv validate`, `pv codegen contracts/ -o src/generated_contracts.rs`; not on PATH — use Makefile's `PV_BIN := cargo run --release -p aprender-contracts-cli --bin pv` [VERIFIED: Makefile:801] |

### Reference environment (dev-only, never a Cargo dependency, never in CI)

| Package | Version | PyPI status | slopcheck |
|---------|---------|-------------|-----------|
| `setfit` | 1.1.3 | Latest release [VERIFIED: pip index] | [OK] |
| `sentence-transformers` | 5.7.0 | Latest release [VERIFIED: pip index] | [OK] |
| `torch` | 2.13.0 | Latest release [VERIFIED: pip index] | [OK] |
| `scikit-learn` | 1.9.0 | Latest release [VERIFIED: pip index] | [OK] |

**Installation (Rust):**
```toml
# root Cargo.toml [workspace.dependencies] — the ONLY new production third-party dependency
tokenizers = { version = "0.23.1", default-features = false, features = ["fancy-regex"] }

# crates/aprender-core/Cargo.toml
[dependencies]
tokenizers = { workspace = true, optional = true }

[features]
setfit = ["tokenizers"]
```

**Reference env (dev workflow, D-12):**
```bash
# uv 0.9.5 available on this machine; commit the resolved lockfile with hashes
uv init scripts/setfit_fixtures && cd scripts/setfit_fixtures
uv add setfit==1.1.3 sentence-transformers==5.7.0 torch==2.13.0 scikit-learn==1.9.0
```

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Extending `EmbeddingBackward` for batched gather | New `BatchedEmbeddingBackward` grad_fn | Existing struct is already index-list based (batch-agnostic); reuse with flattened `B*S` indices unless per-batch error context is needed. Prefer reuse — fewer mutation surfaces |
| `tokenizers` with `fancy-regex` | `onig` default feature | onig compiles C (Oniguruma); breaks the pure-Rust/publishability posture and adds build fragility. fancy-regex is the documented pure-Rust alternative [CITED: docs.rs tokenizers 0.23.1 Cargo.toml] |
| Committed sliced-APR real-weight gate | Synthetic random weights in CI | Explicitly rejected by D-09/PF-011 — not an option |
| `Module` trait default methods for named traversal | Proc-macro derive for names | Trait-methods-only keeps one abstraction (D-17) and zero new macro crates; derive can come later if boilerplate hurts |

## Package Legitimacy Audit

slopcheck 0.x is installed at `/Users/guy/.local/bin/slopcheck` and was run (`slopcheck scan` on a pinned requirements file; the tool has no `--json` flag in this version).

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| tokenizers 0.23.1 | crates.io | mature (HF, years) | high | github.com/huggingface/tokenizers | n/a (scan supports pypi/npm; verified via `cargo search` + docs.rs + GitHub release tag) | Approved |
| setfit 1.1.3 | PyPI | mature | high | github.com/huggingface/setfit | [OK] | Approved (dev-only) |
| sentence-transformers 5.7.0 | PyPI | mature | high | github.com/UKPLab/sentence-transformers | [OK] | Approved (dev-only) |
| torch 2.13.0 | PyPI | mature | very high | github.com/pytorch/pytorch | [OK] | Approved (dev-only) |
| scikit-learn 1.9.0 | PyPI | mature | very high | github.com/scikit-learn/scikit-learn | [OK] | Approved (dev-only) |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

Cross-ecosystem note: `tokenizers` was verified on **crates.io** specifically (`cargo search` returned `tokenizers = "0.23.1"` as the top hit), matching the docs.rs 0.23.1 source manifest and the GitHub v0.23.1 release cited in milestone research — not merely assumed from the PyPI package of the same name.

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────── setfit feature ───────────────────────────┐
 texts: &[&str]          │                                                                      │
      │                  │  SetFitMiniLm (bound model type — sole public entry point, D-08)     │
      ▼                  │  ┌──────────────────┐          ┌───────────────────────────────┐     │
 ┌───────────────┐       │  │ MiniLmTokenizer  │          │ BertSentenceEncoder (D-01)    │     │
 │ import path   │──────▶│  │ (tokenizers      │─────────▶│                               │     │
 │ (typed reject │       │  │  0.23.1 wrapper) │ Sentence │  embedding_gather ─┐          │     │
 │  ENC-01):     │       │  └──────────────────┘ Batch    │  (+pos +type, add) │          │     │
 │ config.json ✓ │       │   ids/type_ids/mask/           │  LayerNorm→Dropout │          │     │
 │ tokenizer ✓   │       │   truncation facts/            │        │           │          │     │
 │ modules ✓     │       │   provenance (D-07)            │        ▼           │          │     │
 │ revision ✓    │       │                                │  6× BertLayer:     │          │     │
 │ tensors ✓     │       │                                │   MHA(+additive    │          │     │
 └───────────────┘       │                                │       mask, drop)  │          │     │
        │                │                                │   →drop→res→LN     │          │     │
        │ AprV2Reader +  │                                │   FFN(GELU)        │          │     │
        │ bert/load.rs   │                                │   →drop→res→LN     │          │     │
        │ (reused, D-01) │                                │        │           │          │     │
        └───────────────▶│                                │        ▼           │          │     │
                         │                                │  masked_mean_pool (checked    │     │
                         │                                │  denom) → l2_normalize (eps)  │     │
                         │                                └──────────────┬────────────────┘     │
                         │                                               │ [B, 384] Tensor      │
                         │       pair loss (ENC-06):                     ▼  (graph-connected)   │
                         │       cosine_similarity_rows(z_a, z_b) ──▶ mse(cos, labels) ──▶ [1]  │
                         └──────────────────────────────────────────────┬───────────────────────┘
                                                                        │ backward()
      ungated: crates/aprender-core/src/autograd/ops/* + grad_fn.rs     ▼
      (six primitives, D-03)                     named grads (HF dotted, D-18)
                                                                        │
                                                  AdamW::step_with_params (trainable set only;
                                                  frozen groups byte-identical, D-20/D-21/D-22)
```

Verification plane (crosscutting): JSON fixtures (D-11/D-15) + committed slice APR (D-09) → tolerance table in `contracts/setfit-encoder-conformance-v1.yaml` (D-14/D-23) → `pv codegen` → `generated_contracts.rs` macros in op/forward bodies (D-27) → tier2/tier3 (D-26) + detach-negative (D-24) + cargo-mutants (D-25).

### Recommended Project Structure

```
crates/aprender-core/src/
├── autograd/
│   ├── grad_fn.rs                 # + MaskedMeanPoolBackward, L2NormalizeRowsBackward,
│   │                              #   CosineSimilarityBackward, MseBackward
│   │                              #   (EmbeddingBackward already exists at :1100 — reuse)
│   └── ops/
│       ├── mod.rs                 # wire new files via mod or include! (follow existing style)
│       ├── embedding.rs           # batched gather (ungated, typed OOV error)
│       ├── masking.rs             # additive attention mask builder + validation
│       ├── pooling.rs             # masked_mean_pool (checked denominator)
│       ├── normalize.rs           # l2_normalize_rows (explicit eps)
│       ├── similarity.rs          # cosine_similarity_rows, mse_loss
│       └── tests_*_backward.rs    # central finite-difference per op (D-04)
├── nn/
│   └── module.rs                  # + named traversal + set_training propagation (D-17)
├── setfit/                        # entire module #[cfg(feature = "setfit")]
│   ├── mod.rs                     # SetFitMiniLm bound type (D-08), public API
│   ├── error.rs                   # typed error enum (import/tokenize/forward)
│   ├── tokenizer.rs               # MiniLmTokenizer + SentenceBatch (D-07)
│   ├── import.rs                  # pinned-revision contract validation (wraps bert/load)
│   ├── encoder.rs                 # BertSentenceEncoder forward (train/eval)
│   └── loss.rs                    # pair cosine-MSE (tensor-valued)
crates/aprender-core/tests/
├── setfit_conformance/            # fixture-driven gates (feature = "setfit")
└── fixtures/setfit/               # committed JSON fixtures + slice APR + SHA-256 manifest
contracts/setfit-encoder-conformance-v1.yaml
scripts/setfit_fixtures/           # uv project (lockfile committed), generator, slice tool
```

### Pattern 1: New autograd op = forward + GradFn + graph record

Every existing op follows this exact shape; the six new ops must too.

```rust
// Source: crates/aprender-core/src/autograd/ops/mod.rs (add/mul/mean pattern)
pub fn masked_mean_pool(hidden: &Tensor, mask: &[u8], /* B, S from shape */)
    -> Result<Tensor, PoolError>
{
    // 1. validate: shape [B, S, H]; per-row mask sum > 0 → else typed error (PF-004)
    // 2. compute forward into Vec<f32>, build result [B, H]
    let mut result = Tensor::from_vec(data, &[batch, hidden_dim]);
    // 3. record backward
    if is_grad_enabled() && hidden.requires_grad_enabled() {
        result.requires_grad_(true);
        let grad_fn = Arc::new(MaskedMeanPoolBackward { mask: mask.to_vec(), /* shapes */ });
        result.set_grad_fn(grad_fn.clone());
        with_graph(|graph| {
            graph.register_tensor(hidden.clone());
            graph.record(result.id(), grad_fn, vec![hidden.id()]);
        });
    }
    Ok(result)
}
```

### Pattern 2: Batched embedding gather reuses `EmbeddingBackward`

The existing grad_fn is index-list based and therefore batch-agnostic — `[B, S]` ids flatten to `B*S` gather rows, and scatter-add accumulation already handles repeated token ids correctly.

```rust
// Source: crates/aprender-core/src/models/qwen2/mod.rs:146-165 (record_embedding_backward)
// and crates/aprender-core/src/autograd/grad_fn.rs:1100 (EmbeddingBackward)
let grad_fn = Arc::new(EmbeddingBackward {
    indices: flat_ids,          // B*S flattened
    vocab_size, hidden_size,
});
```

Divergence required from the qwen2/aprender-train precedents: **fail closed on out-of-vocabulary ids in the forward** (typed error), instead of qwen2's OOB-escape or aprender-train's silent zero-fill (known trap in CONTEXT.md).

### Pattern 3: `Module` named traversal as trait default-extension (D-17)

```rust
// Extension to crates/aprender-core/src/nn/module.rs — sketch, planner refines
pub trait Module: Send + Sync {
    // ... existing methods unchanged ...

    /// Named parameters, prefix-composed by containers.
    /// Names are HF-dotted at the encoder level (D-18), e.g.
    /// "encoder.layer.0.attention.self.query.weight".
    fn named_parameters(&self) -> Vec<(String, &Tensor)> { Vec::new() }
    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Tensor)> { Vec::new() }

    /// Recursive mode propagation (existing train()/eval() are flat).
    fn set_training(&mut self, training: bool) {
        if training { self.train() } else { self.eval() }
    }
}
```

Leaf impls (Linear: `weight`/`bias`; LayerNorm: `weight`/`bias`) return their local names; composites (MHA, the new encoder) prefix child names. ENC-05's proof: snapshot `named_parameters()` (names + bytes), flip `set_training` both ways, assert identical set and identical bytes.

### Pattern 4: Dropout placement (verified from HF transformers v4.57.1 source)

Four site classes, all `p = 0.1` for MiniLM:

| Site | Placement (exact, verified) |
|------|------------------------------|
| Embeddings | `embeddings = LayerNorm(word+pos+type)` **then** `dropout(embeddings)` |
| Attention probs | `attention_probs = softmax(scores)` **then** `dropout(attention_probs)` (before `@ V`) |
| Attention output | `dense(attn_out)` → `dropout` → `LayerNorm(x + residual)` |
| FFN output | `dense(ffn_out)` → `dropout` → `LayerNorm(x + residual)` |

[VERIFIED: raw.githubusercontent.com/huggingface/transformers/v4.57.1/.../modeling_bert.py]

**Seeded-RNG policy (recommendation):** one root seed `u64` on the encoder config; per-site seed derived deterministically from `(root_seed, dotted site name)` via a stable hash — stable under refactoring, no positional drift, and independent per site. Use existing `Dropout::with_seed`. Fixtures never exercise dropout (D-16); Rust-side tests cover placement (count/site of active dropouts in train mode), determinism (same root seed ⇒ same masks), and statistics (drop rate ≈ p).

### Pattern 5: Contract binding (D-27)

```rust
// Source: crates/aprender-core/src/models/qwen2/mod.rs:120-124 (live example)
#[provable_contracts_macros::contract("setfit-encoder-conformance-v1", equation = "masked_mean_pool")]
pub fn masked_mean_pool(...) -> ... {
    contract_pre_masked_mean_pool!(mask);
    // ...
    contract_post_masked_mean_pool!(result.data());
}
```

Regeneration: `pv codegen contracts/ -o src/generated_contracts.rs` (header of the generated file). Contract YAML follows the existing `KernelContract` shape — `metadata` / `equations.{name}.formula/domain/codomain/invariants/preconditions` (see `contracts/encoder-forward-v1.yaml`) plus falsification tests. The tolerance table lives in this YAML (D-14).

### Recommended op signatures (discretion area — prescriptive proposal)

Shapes are row-major throughout (LAYOUT-001). `B`=batch, `S`=padded seq len, `H`=384.

| Op | Signature | Backward |
|----|-----------|----------|
| Batched embedding gather | `embedding_gather(weight: &Tensor[V,H], ids: &[u32], b: usize, s: usize) -> Result<Tensor[B,S,H]>` | existing `EmbeddingBackward` (flattened ids) |
| Additive attention mask | `additive_attention_mask(mask: &[u8], b: usize, s: usize) -> Result<Tensor[B,1,1,S]>` (constant: 0.0 keep, NEG_MASK pad) | none — constant tensor; flows through existing broadcast-add inside SDPA |
| Masked mean pool | `masked_mean_pool(hidden: &Tensor[B,S,H], mask: &[u8]) -> Result<Tensor[B,H]>` (typed error on zero denominator) | new `MaskedMeanPoolBackward` |
| Row L2 normalize | `l2_normalize_rows(x: &Tensor[B,H], eps: f32) -> Tensor[B,H]` (`x / max(‖x‖₂, eps)`) | new `L2NormalizeRowsBackward` |
| Cosine similarity | `cosine_similarity_rows(a: &Tensor[B,H], b: &Tensor[B,H], eps: f32) -> Result<Tensor[B]>` | new `CosineSimilarityBackward` |
| MSE reduction | `mse_loss(pred: &Tensor[B], target: &[f32]) -> Result<Tensor[1]>` | new `MseBackward` |

`NEG_MASK`: use a large finite negative constant (e.g. `-1e9`), not `f32::MIN`. Rationale: all-padding rows are rejected before forward (typed error), so every softmax row has ≥1 valid key; `exp(-1e9 - max)` underflows to exactly 0.0 in f32, giving parity with torch on valid positions within the declared tolerances while avoiding `-inf`/NaN arithmetic hazards. Padded-query outputs are excluded by masked pooling and by fixture comparison scope. Fixtures are the arbiter (see Assumptions A3).

### Anti-Patterns to Avoid

- **`Tensor::new`/`from_vec` on computed values inside the forward:** this is THE detachment bug class (PMAT-913/914/922 all fixed instances of it). Any intermediate built from `.data()` reads severs the graph. Only op entry points may materialize tensors, and only with a recorded grad_fn.
- **Reusing `loss/loss.rs` or `nn/self_supervised.rs` names/functions:** they return `f32`, not tensors — reuse would falsely imply encoder fine-tuning occurs (CONTEXT.md known trap).
- **Routing through `models/bert/embeddings.rs::forward`:** it `assert!`s on length and uses unchecked slice ranges; the new path returns typed errors and never panics.
- **Zero-filling OOV ids** (aprender-train precedent): fail closed with a typed error.
- **Tolerance edits inside test files:** tolerances live only in the contract YAML (D-14); a test that carries its own epsilon is a re-baselining side channel.
- **A bespoke test target outside make tiers:** rejected by D-26.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| WordPiece tokenization + BERT normalization | Custom tokenizer | `tokenizers` 0.23.1 (`fancy-regex`) | Normalization/pre-tokenization edge cases (CJK, accents, casing) cause silent parity failure; official pipeline is the parity target itself |
| Gather backward | New scatter-add grad_fn | Existing `EmbeddingBackward` (grad_fn.rs:1100) | Already handles repeated-id accumulation correctly; proven by OBLIG-EMBEDDING-BACKWARD-GRAD-FLOW |
| Attention core | New attention math | `MultiHeadAttention::forward_self` + additive mask | PMAT-914 backward chain already proven; rewriting reintroduces the severed-graph class |
| Optimizer step | Custom SGD/Adam | `nn::optim::AdamW::step_with_params` | Contracted (`tests_adamw_contract.rs`); ENC-04 requires a *controlled* step, not a new optimizer |
| Contract validation/diff/codegen | bash/yq/python scripts | `pv` (via Makefile `PV_BIN`) | CLAUDE.md hard rule; `pv diff` gives the semver-flagged tolerance versioning D-14 depends on |
| Hashing | Custom digest | `sha2` workspace dep | Already used at APR boundaries |
| Grad-check harness | Ad-hoc numeric diff | Follow `tests_matmul_backward.rs` / `tests_norm_backward_gradflow.rs` patterns; central differences per D-04 | Existing conventions define tolerance/reporting style reviewers expect |
| No-grad evaluation scope | Manual flag plumbing | `autograd::no_grad(|| ...)` | Exists, tested (`autograd/mod.rs:76`) |

**Key insight:** in this codebase the expensive failures have never been "missing math" — they were *silently detached graphs* around existing math. The leverage is in reusing the proven grad-flow components and spending the new effort on gates that make detachment loud (D-24/D-25).

## Common Pitfalls

### Pitfall 1: Silent no-op encoder tuning (PF-001 — the phase's reason to exist)
**What goes wrong:** loss decreases, shapes correct, encoder weights byte-identical.
**Why it happens:** intermediates built via `Tensor::new`/`.data()` copies sever the graph; history shows three shipped instances (PMAT-913 embeddings, PMAT-914 attention, PMAT-922 dropout).
**How to avoid:** every new op records a grad_fn (Pattern 1); in-band detached-encoder negative test (D-24) runs in every `cargo test`; cargo-mutants scoped to ops + forward (D-25).
**Warning signs:** any new `Tensor::from_vec`/`Tensor::new` in forward code without an adjacent `set_grad_fn`; gradient tests asserting only "finite" instead of "finite AND non-zero AND parameter delta after step".

### Pitfall 2: Batch>1 broadcast paths lose or corrupt gradients
**What goes wrong:** batch-1 parity passes, mixed-length batch gradients are wrong/absent (PF-014 + STACK.md research gap).
**Why it happens:** existing paths were exercised mostly at batch 1 (qwen2 gather hardcodes `batch_size = 1`; `BertEncoder` docs say `[seq_len, hidden]`); broadcast backward through `[B,1,1,S]` masks and ND `Linear` flattening is unproven end-to-end.
**How to avoid:** make the first plan a spike: 2-sentence mixed-length batch, full chain, assert per-parameter finite non-zero grads AND batch-1-vs-batch-N equivalence (per padding invariance) before building the rest.
**Warning signs:** grad shapes that don't match parameter shapes; gradients identical across batch rows that should differ.

### Pitfall 3: The committed slice APR conflicts with the ENC-01 architecture pin
**What goes wrong:** the D-09 slice (reduced layers/vocab/hidden to hit "few hundred KB") cannot pass the same typed import validation ENC-01 requires (6 layers / 384 hidden / 30522 vocab), so either the import gate is weakened or the slice can't load.
**How to avoid:** plan a test-only constructor (e.g., `#[cfg(any(test, feature = "conformance-fixtures"))]`) that bypasses ONLY the architecture-pin equality checks while keeping all structural/shape/finite checks. The public entry point keeps the full pin. Document this split in the contract.
**Warning signs:** import validation parameterized by caller-supplied dims on the public path (that is PF-011's exact failure mode).

### Pitfall 4: Fixture re-baselining and tolerance drift
**What goes wrong:** a Rust discrepancy is "fixed" by regenerating fixtures or widening an epsilon in a test file.
**How to avoid:** tolerances only in the contract YAML, committed before any comparison runs (D-14, its own commit); SHA-256 manifest makes regeneration a reviewable diff (D-13); fixture-generation is a dev workflow never run in CI (D-12).
**Warning signs:** a PR that touches both a fixture JSON and a Rust op; any epsilon literal in a comparison test.

### Pitfall 5: Feature-matrix and dependency leakage
**What goes wrong:** `tokenizers` (or its transitive C/C++ deps) leaks into the minimal inference build; or default features silently pull `onig`/`esaxx` native code.
**How to avoid:** `default-features = false, features = ["fancy-regex"]` exactly (verified against the 0.23.1 manifest: defaults are `progressbar`, `onig`, `esaxx_fast`); D-06 matrix in CI: `--no-default-features`, `--features setfit`, all-features; `cargo tree -e features` check that a no-setfit consumer has no `tokenizers` node.
**Warning signs:** `cargo build --no-default-features -p aprender-core` compiling ring/onig/cc.

### Pitfall 6: Committed fixture shadowed by gitignore (CB-510 class)
**What goes wrong:** the slice APR or JSON fixtures match an ignore pattern and silently vanish from the package/repo.
**How to avoid:** `.gitignore` currently root-anchors `/*.apr` and `/models/` ("test fixtures in crates/ are OK" — verified), but after adding files run `git check-ignore -v <each fixture path>` (must exit 1) and re-run `scripts/check_include_files.sh` + `scripts/check_package_includes.sh`.
**Warning signs:** `git status` not listing a newly written fixture.

### Pitfall 7: Mode-switch mutates state (ENC-05 failure)
**What goes wrong:** `train()`/`eval()` changes registered parameters (e.g., a mode-dependent cached transform is registered as a parameter, or dropout RNG state is treated as a parameter).
**How to avoid:** RNG state lives outside `named_parameters()`; the mode-flip byte-identity test (snapshot → train→eval→train → snapshot compare) is a required gate; `no_grad` is orthogonal to eval mode — don't conflate.
**Warning signs:** `named_parameters()` count differing between modes.

### Pitfall 8: Panics instead of typed errors on hostile input
**What goes wrong:** OOV token id, oversize sequence, all-padding row, or mismatched id/type lengths panic (existing `bert/embeddings.rs` asserts; `LayerNorm::forward` also asserts on shape).
**How to avoid:** validate the entire `SentenceBatch` once at the encoder boundary with typed errors; ops then operate on validated shapes but still return `Result` for their own failure modes (zero denominator). `unwrap()` is lint-banned anyway.
**Warning signs:** `assert!`/`panic!` in any new non-test code path reachable from public API.

## Code Examples

### Central finite-difference gradient check (D-04 pattern)

```rust
// Convention synthesized from tests_matmul_backward.rs / tests_norm_backward_gradflow.rs
// f'(x) ≈ (f(x+h) - f(x-h)) / 2h, h ≈ 1e-3 for f32; compare within contract tolerance.
fn central_diff_grad(f: impl Fn(&[f32]) -> f32, x: &[f32], i: usize, h: f32) -> f32 {
    let mut xp = x.to_vec(); xp[i] += h;
    let mut xm = x.to_vec(); xm[i] -= h;
    (f(&xp) - f(&xm)) / (2.0 * h)
}
// Each new op's test: analytic backward vs central diff over every input element
// of a small non-degenerate case, plus the frozen Python fixture comparison.
```

### Controlled optimizer step + frozen-group byte identity (ENC-04, D-21)

```rust
// AdamW API verified: crates/aprender-core/src/nn/optim/mod.rs:391 (struct); impl in rm_sprop.rs (decoupled decay :80, step_with_params :87) — do NOT use coupled-decay Adam
let mut opt = AdamW::new(trainable_params, 2e-5);       // trainable set ONLY (freeze = exclusion)
let frozen_before: Vec<Vec<f32>> = frozen.iter().map(|(_, t)| t.data().to_vec()).collect();
loss.backward();
opt.step_with_params(&mut trainable_refs);
// every trainable named tensor: finite, non-zero grad AND >=1 changed element
// every frozen named tensor: bitwise-identical to frozen_before (f32::to_bits comparison)
```

### Eval-mode deterministic encode (ENC-05)

```rust
// no_grad verified: crates/aprender-core/src/autograd/mod.rs:76
model.set_training(false);
let z1 = no_grad(|| model.encode(&batch))?;
let z2 = no_grad(|| model.encode(&batch))?;
assert_eq!(z1.data(), z2.data()); // deterministic in eval mode, bitwise
```

### Sentence-transformers reference semantics (fixture generator side)

```python
# Source: sbert.net CosineSimilarityLoss docs + all-MiniLM-L6-v2 model card (see Sources)
# loss = MSE(cos_sim(u, v), label), label in {0.0, 1.0}
# pooling = sum(token_emb * mask) / clamp(mask.sum(), min=1e-9)   [ASSUMED exact clamp constant — A1]
# normalize = x / max(||x||_2, 1e-12)
# Generator MUST: torch.manual_seed(...), model.eval() for forward fixtures (D-16),
# dtype float32, and dump per-layer hidden states (D-15) + named grads via
# named_parameters() so names match D-18 verbatim.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| aprender BERT: inference-only, batch-1, panicking embeddings | This phase: graph-connected batched encoder with typed errors | Phase 1 (now) | New `setfit` module; `models/bert/` untouched until Phase 4 |
| Positional `parameters()` only | Named recursive traversal on the same `Module` trait | Phase 1 (D-17) | Enables freezing, optimizer grouping, fixture alignment |
| `tokenizers` 0.22 (bench crate, default-features off + progressbar) | 0.23.1 workspace pin, `fancy-regex` only | This phase | Workspace-level pin per STACK.md; bench crate promotion optional, not required for the gate |
| SetFit ecosystem: `num_iterations` sampling | oversampling/undersampling/unique strategies | setfit ≥1.0 (deprecated param) | Phase 2 concern; fixtures here only need pair cosine-MSE semantics |
| ST recommends newer losses in some contexts | Cosine-MSE remains SetFit's default and the parity target | current (setfit 1.1.3) | Alternatives explicitly deferred to v2 (EXT-02) |

**Deprecated/outdated:**
- `aprender-train/src/transformer/` + `aprender-train/src/autograd/` — legacy stack, explicitly NOT the target (CONTEXT.md); zero-fill OOV behavior there is a documented trap.
- `Qwen2Model::generate()/forward()` already deleted — precedent that inference paths in `aprender` get removed, reinforcing D-01's "don't grow the old BERT path".

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | ST mean pooling uses `clamp(mask.sum(), min=1e-9)`; Normalize uses eps `1e-12` (`torch.nn.functional.normalize` default) | Code Examples / op signatures | Low — fixtures are the arbiter; Rust uses checked denominator + explicit eps, and the contract tolerance absorbs sub-eps differences. Verify exact constants when writing the Python generator (read the pinned ST 5.7.0 source in the locked env) |
| A2 | `torch==2.13.0` wheels resolve for Python 3.13.7 on this machine (uv lock will prove) | Standard Stack (reference env) | Medium — if not, pin the uv project's Python (e.g. 3.12) in `.python-version`; D-12 only requires a hash-locked env, not a specific interpreter |
| A3 | Additive-mask constant `-1e9` (vs torch's `finfo.min`) yields identical softmax weights on valid positions within declared tolerances | Op signatures | Low — `exp(-1e9 - max)` underflows to 0.0 in f32; fixture comparison over valid positions verifies; if a discrepancy surfaces, match torch's constant exactly |
| A4 | `tokenizers` 0.23.1 with only `fancy-regex` reproduces identical WordPiece output for the MiniLM `tokenizer.json` (no onig-only behavior) | Standard Stack | Low-Medium — the ENC-02 tokenizer parity fixtures gate this directly; if parity fails, feature set is the first suspect |
| A5 | The internal SDPA dropout (`scaled_dot_product_attention(..., dropout_p, training)`) can be seeded/controlled for the per-site RNG policy | Pattern 4 | Medium — if MHA's internal dropout is unseedable, either extend MHA with a seeded dropout hook or apply attention-probs dropout outside SDPA in the new layer; inspect `attention_helpers.rs` during the spike |
| A6 | The `KernelContract` schema (or a modest extension) can host the frozen tolerance table so `pv diff` versions it | Pattern 5 | Medium — CLAUDE.md pre-authorizes schema extension as its own task if `pv validate` rejects; budget for it in planning (Open Question Q4) |
| A7 | Slice-model fixture strategy (reduced hidden/heads/layers/vocab, fixtures generated from the SAME sliced checkpoint) satisfies D-09's "real weight values" intent at a few hundred KB | Pitfall 3 / Open Questions | Low — D-09 targets real value distributions vs synthetic shapes; full-dimension parity is separately covered by the D-10 gated ~90MB suite |

## Open Questions (RESOLVED)

All five questions have a committed resolution or a plan-owned resolution path (annotated per
question below); none block execution.

1. **Does gradient flow survive batch>1 through existing MHA/LayerNorm broadcast paths?**
   - **RESOLVED (empirical) — owner: plan 01-03 Task 3.** The ungated `batched_graph_spike.rs` integration test proves batch>1 grad flow end-to-end (per-parameter finite/non-zero grads at batch 2 with lengths 5/9, padding invariance); if broadcast backward is broken at B>1, fixing the responsible op/grad_fn is explicitly in-scope within that task (stated in its action).
   - What we know: shapes support `[B,S,E]` (`forward_qkv` reads `batch_size = query.shape()[0]`); backward tests exist but were not audited for B>1 coverage.
   - What's unclear: end-to-end grad correctness with a broadcast `[B,1,1,S]` additive mask and ND Linear flattening.
   - Recommendation: first plan = spike (Pitfall 2). If broadcast backward is broken, fixing it is in-scope for this phase (it's the "additive attention masking" primitive's proof).

2. **How does the committed slice APR pass loading without weakening the ENC-01 pin?**
   - **RESOLVED (design adopted) — owners: plan 01-05 Task 3 + plan 01-04 Tasks 1-2.** Test-only slice constructor (`from_slice_fixture`, 01-05 Task 3) bypasses only the architecture-pin equality; slice APR and all slice fixtures are generated from the SAME sliced torch model (01-04), so the real import path's ENC-01 pin is untouched.
   - What we know: slice dims can't equal the pinned architecture at a few-hundred-KB budget (one full 384-hidden layer alone is ~7MB F32).
   - Recommendation: test-only constructor bypassing only the architecture-pin equality (Pitfall 3); slice shape suggestion: 2 layers, hidden 64 (**2 heads × 32** — CORRECTED 2026-08-08; the original “4 heads × 16” suggestion is SUPERSEDED by 01-04, because 16-dim heads cut MiniLM's real 32-dim heads in half and manufacture a synthetic attention structure out of real weights, weakening the exact property D-09 exists to protect — take original heads 0 and 1 and record them in `slice_config.json.source_head_indices`), intermediate 256, vocab = fixture-token closure (~256 ids, remapped, remap table in fixture JSON), positions 64 → ≈0.5MB F32. Python generator slices the pinned checkpoint AND generates all slice fixtures from the same sliced torch model, so Rust-vs-Python parity is exact on the slice.

3. **Seeding of attention-probs dropout inside SDPA (A5).**
   - **RESOLVED BY INSPECTION (updated 2026-08-08) — owner: plan 01-06 Task 2.** No spike is needed and none is scheduled to answer this: `scaled_dot_product_attention` (`nn/transformer/mod.rs:66-70`) calls `apply_dropout` (`nn/transformer/positional_encoding.rs:401`), which calls `nn::functional::dropout(x, p, training)` (`nn/functional.rs:333`) — a signature with NO seed parameter. The internal attention-probs dropout is therefore **not seedable today**, as a fact of the source rather than a prediction. 01-06 Task 2 accordingly treats the seeded-dropout extension as the PRIMARY path, not a contingency: an additive `apply_dropout_seeded(x, p, Option<u64>)` plus an optional `MultiHeadAttention::with_attention_dropout_seed(seed)` builder defaulting to `None`, so existing callers are byte-for-byte unchanged. 01-03 Task 3's spike still records what it observes, but 01-06 does not wait on it.
   - Recommendation: extend MHA construction with an optional seeded dropout; never re-implement attention. Plan for the extension, not for a discovery.

4. **Will `pv validate` accept `setfit-encoder-conformance-v1.yaml` with a tolerance table and references to six existing contracts?**
   - **RESOLVED (procedural) — owner: plan 01-01 Task 1.** The contract is authored and `pv validate`d FIRST, before any Rust comparison is written; if pv rejects, the sanctioned path is restructure-to-KernelContract, and if the schema genuinely cannot host it the task STOPs and surfaces the schema-extension work (`aprender-contracts/src/schema/`) for user visibility — never a bash workaround.
   - What we know: existing contracts use `metadata.depends_on` for references (seen in `encoder-forward-v1.yaml`); schema kinds are enforced by `pv`.
   - Recommendation: draft the YAML early in planning and run `PV_BIN validate` before building tests against it; if rejected, the sanctioned path is a schema extension task in `aprender-contracts/src/schema/` (never a bash workaround).

5. **`bert/load.rs` reuse shape (discretion area).**
   - **RESOLVED (decision adopted: wrap) — owner: plan 01-05 Task 3.** `setfit/import.rs` reuses `bert/load.rs` per-tensor helpers via a wrapper; its typed error enum `#[from]`-wraps `BertLoadError`; `bert/` is not modified (D-01). Encoded in 01-05's key_links ("Q5: wrap").
   - Recommendation: wrap. Reuse its per-tensor fetch/shape helpers and `BertLoadError` from the new `setfit/import.rs`; the pinned-revision contract (config field equality, tokenizer bytes hash, module-graph policy, revision recording) lives in the new module with its own typed error enum that `#[from]`-wraps `BertLoadError`. Do not modify `bert/` (D-01).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| rustc / cargo | everything | ✓ | 1.93.0 (MSRV 1.91) | — |
| cargo-mutants | D-25 mutation gate | ✓ | 25.3.1 | — |
| `pv` binary | D-23/D-26/D-27 contract ops | ✗ (not on PATH) | in-tree | `cargo run --release -p aprender-contracts-cli --bin pv` (Makefile `PV_BIN`, verified at Makefile:801) |
| uv | D-12 hash-locked env | ✓ | 0.9.5 | — |
| Python 3 | fixture generator | ✓ | 3.13.7 | pin uv project Python if torch wheel unavailable (A2) |
| slopcheck | package audit | ✓ | installed | — |
| bashrs | shell-script lint (CLAUDE.md) | ✗ | — | keep `scripts/setfit_fixtures/` Python-first; if a shell wrapper is unavoidable, follow bashrs conventions manually and note lint debt |
| pmat | code search / coverage queries | not probed | — | assumed present per CLAUDE.md workflows; verify with `pmat --version` at plan start |
| Network + HF Hub | fixture generation, full-weight suite artifact | dev machine only | — | intentionally absent from CI (D-09/D-10/SAFE-02) — not a blocker, a design constraint |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** `pv` (cargo run), `bashrs` (Python-first scripts).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (rustc 1.93.0) + proptest (bounded) + pv contracts + cargo-mutants 25.3.1 |
| Config file | `Cargo.toml` workspace lints, `.proptest.toml`, `.clippy.toml`, `Makefile` tiers |
| Quick run command | `cargo test -p aprender-core --lib --features setfit` |
| Full suite command | `make tier2` (= `PROPTEST_CASES=5 cargo test --lib` + `cargo clippy -- -D warnings`) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENC-01 | Typed rejection of mutated config/tokenizer/pooling/revision | unit (mutation matrix) | `cargo test -p aprender-core --features setfit setfit::import` | ❌ Wave 0 |
| ENC-02 | Tokenizer ids/type-ids/masks/truncation facts match fixtures | fixture parity | `cargo test -p aprender-core --features setfit --test setfit_conformance tokenizer_` | ❌ Wave 0 |
| ENC-03 | Slice forward per-layer + pooled + normalized parity; batch-1 vs batch-N padding invariance | fixture parity + property | `cargo test -p aprender-core --features setfit --test setfit_conformance forward_` | ❌ Wave 0 |
| ENC-04 | Named grads finite/non-zero; trainable deltas; frozen byte-identity; loss-reducing embedding movement | integration | `cargo test -p aprender-core --features setfit --test setfit_conformance gradient_` | ❌ Wave 0 |
| ENC-05 | Recursive mode switch; registered params byte-identical across mode flips; dropout placement/determinism | unit | `cargo test -p aprender-core --features setfit setfit::encoder::mode` | ❌ Wave 0 |
| ENC-06 | Tensor-valued cosine-MSE parity; detach-negative gate fails on deliberate detachment | fixture parity + negative | `cargo test -p aprender-core --features setfit --test setfit_conformance loss_ detach_` | ❌ Wave 0 |
| D-04 (per-op) | Six ops match central finite differences | unit | `cargo test -p aprender-core --lib autograd::ops` (ungated — runs in every matrix leg) | ❌ Wave 0 |
| D-10 | Full ~90MB real-weight parity | ignored/feature-gated | `cargo test -p aprender-core --features setfit,model-tests -- --ignored` (reuse existing `model-tests` pattern + `make test-model`) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p aprender-core --lib --features setfit` (targeted module filter while iterating)
- **Per wave merge:** `make tier2` + feature-matrix build check (`cargo check -p aprender-core --no-default-features && cargo check -p aprender-core --features setfit`)
- **Phase gate:** `make tier3` including `PV_BIN validate contracts/setfit-encoder-conformance-v1.yaml`, scoped `cargo mutants` on `autograd/ops` + `setfit/encoder.rs`, full workspace lib tests green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `scripts/setfit_fixtures/` uv project + committed `uv.lock` (hashes) — D-12
- [ ] Fixture generator (tokenizer, per-layer, pooling, normalize, loss, grads, optimizer step, mixed batches) — D-15
- [ ] Slice derivation tool + committed slice APR + SHA-256 manifest — D-09/D-13 (+ `git check-ignore` verification)
- [ ] `contracts/setfit-encoder-conformance-v1.yaml` with tolerance table, committed BEFORE any Rust comparison (D-14 — its own commit, sequenced first)
- [ ] `crates/aprender-core/tests/setfit_conformance/` harness + `tests/fixtures/setfit/`
- [ ] Six `tests_*_backward.rs` finite-difference files under `autograd/ops/`
- [ ] tier2 wiring for the new tests (they enter via `cargo test --lib` automatically once feature-enabled in CI matrix; D-06 matrix legs need CI additions)
- [ ] Framework install: none (all Rust-side tooling present; `pv` via cargo run)

## Security Domain

`security_enforcement` enabled (ASVS L1). Library-only phase — no auth/session/network surface is added.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no auth surface) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Typed error enums at the `SentenceBatch`/import boundary: OOV ids, oversize sequences (>512 positions / >256 sentence max), all-padding rows, mismatched id/type/mask lengths, malformed config/tokenizer JSON (serde strict), non-finite tensor values. Fail closed — this IS ENC-01/ENC-02 |
| V6 Cryptography | limited | SHA-256 via `sha2` workspace crate for fixture manifest + tokenizer-bytes hash — never hand-rolled; no secrets handled |
| V10/V14 Supply chain & config | yes | Model pinned by immutable revision sha (never branch name); hash-locked uv env; `tokenizers` pinned + registry-verified; cargo-deny/audit already in CI; no network in test/CI paths |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious/corrupted model artifact (oversized dims, NaN weights, shape lies) | Tampering | Validate config/shapes/dtypes/finiteness BEFORE allocation (PF-011 prevention); existing `AprV2Reader` checksums; typed `BertLoadError`-style failures |
| Mutable-ref model swap (branch `main` re-pointed upstream) | Tampering/Repudiation | Pin full revision sha `1110a243…`; record revision + file hashes in the import contract |
| Fixture re-baselining to hide regressions | Repudiation | SHA-256 manifest (D-13) + tolerances in versioned contract (D-14) — regeneration is a reviewable diff |
| Slopsquatted/hallucinated dependency | Spoofing | slopcheck [OK] on all Python refs; `tokenizers` verified on crates.io specifically (cross-ecosystem confusion check done) |
| Panic-as-DoS on hostile input (assert paths) | DoS | New paths return `Result`; no `unwrap()` (lint-enforced); no `unsafe` (forbidden workspace-wide) |
| Native-code build injection via default features | Elevation | `default-features = false` drops onig (C) / esaxx (C++) build scripts from the dependency graph |

## Sources

### Primary (HIGH confidence)
- Codebase (read directly this session): `autograd/ops/mod.rs`, `autograd/grad_fn.rs:1090-1140` (EmbeddingBackward + PMAT-914 note), `autograd/mod.rs` (no_grad), `nn/module.rs`, `nn/dropout/mod.rs` (PMAT-922 fix, with_seed), `nn/transformer/attention_gqa.rs:320-375` (forward_qkv shapes/mask), `nn/optim/mod.rs` (AdamW, step_with_params), `models/bert/{config,embeddings,encoder,layer}.rs`, `models/bert/load.rs` (BertLoadError), `models/qwen2/mod.rs:120-165` (contract macro + gather recording), `generated_contracts.rs` header (pv codegen command), `contracts/encoder-forward-v1.yaml` (schema shape), `Makefile` (tier2, PV_BIN), `.gitignore`, root + core `Cargo.toml` (features, MSRV)
- HF Hub API `sentence-transformers/all-MiniLM-L6-v2` — revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` is current main sha; apache-2.0; BertModel (fetched 2026-08-07)
- docs.rs `tokenizers` 0.23.1 source Cargo.toml — feature semantics (defaults: progressbar, onig, esaxx_fast; fancy-regex pure-Rust alternative)
- HF transformers v4.57.1 `modeling_bert.py` (raw GitHub) — dropout placement (embeddings post-LN; attention probs post-softmax; dense→dropout→residual→LN)
- crates.io registry via `cargo search` — tokenizers 0.23.1 current
- PyPI via `pip index versions` + slopcheck scan — all four reference pins are latest, [OK]

### Secondary (MEDIUM-HIGH confidence)
- `.planning/research/STACK.md` (2026-08-07) — encoder contract table, feature-gate table, verification strategy, SetFit defaults (body LR 2e-5, batch 16, 1 epoch, 10% warmup), authoritative URL list (sbert.net CosineSimilarityLoss, SetFit docs, model card)
- `.planning/research/PITFALLS.md` — PF-001/002/003/004/011/014 with falsification tests; phase-specific exit gates
- `.planning/codebase/TESTING.md` — fixture location conventions, `model-tests` feature pattern, contract test wiring, coverage rules

### Tertiary (LOW confidence, flagged)
- ST pooling clamp constant `1e-9` / Normalize eps default — training knowledge (A1); verify in the locked ST 5.7.0 source when writing the generator

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all pins registry-verified this session; only new production dep is one crate with verified feature semantics
- Architecture: HIGH for op/grad_fn/Module patterns (read from working precedents); MEDIUM for batch>1 gradient parity (open spike, Q1) and the slice/import split (Q2)
- Pitfalls: HIGH — sourced from repo history (PMAT-913/914/922, CB-510) and milestone falsification research, not speculation

**Research date:** 2026-08-07
**Valid until:** ~2026-09-07 for stack/pins (re-verify PyPI pins when the uv lock is created); codebase facts valid until the referenced files change
