# Phase 1: Differentiable MiniLM Conformance - Context

**Gathered:** 2026-08-07
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers **one contracted, graph-connected MiniLM sentence encoder** in
`aprender-core`: pinned-revision import with typed rejection of unsupported variants, batched
tokenization producing first-class token facts, a single batched
`Transformer -> masked mean pooling -> L2 normalization` forward path, named parameter
enumeration with train/eval modes, proven finite non-zero gradients through a controlled
optimizer step, and a tensor-valued cosine-similarity MSE pair loss — all verified against
frozen Python-generated reference fixtures.

**In scope:** ENC-01 through ENC-06 only.

**Out of scope for this phase** (each belongs to a later phase, do not pull forward):
- Pair sampling, few-shot selection, dataset provenance, split isolation — Phase 2
- The `SetFitTrainer`, the multinomial head, lifecycle states — Phase 3
- APR persistence, CLI/serve surfaces, production parity — Phase 4
- Benchmarks, metrics, claims — Phase 5

Phase 1 is library-only. It exposes no new `apr` subcommand.
</domain>

<decisions>
## Implementation Decisions

### Encoder Home and Structure

- **D-01:** Build a new `BertSentenceEncoder` in `aprender-core` that reuses
  `crates/aprender-core/src/models/bert/config.rs` and `bert/load.rs` but implements a fully
  graph-connected forward path. **Do not refactor `crates/aprender-core/src/models/bert/` in
  place during this phase.** `apr embed`, `apr rerank`, and `CrossEncoder` stay untouched.
- **D-02:** Convergence onto a single encoder is deferred to **Phase 4**, where OPS-03 already
  requires generic APR commands to call the shared core model. That is where the inference
  regression sweep belongs, with APR parity tests to prove it. This ordering is deliberate and
  follows STATE's constraint: *validate the real-weight mixed-batch graph before committing the
  full BERT refactor.*
- **D-03:** The six missing differentiable primitives — batched embedding gather, additive
  attention masking, masked mean pooling with a checked non-zero denominator, row-wise L2
  normalization with explicit epsilon, cosine similarity, and MSE reduction — are implemented as
  **model-agnostic ops in `crates/aprender-core/src/autograd/ops/`**, alongside the existing
  matmul/elementwise backward implementations. They are **not** gated behind the `setfit`
  feature. This directly retires the CONCERNS.md "embedding and pooling operations detach the
  graph" debt rather than routing around it.
- **D-04:** Each new op ships with its own central-finite-difference test.

### Feature Gating

- **D-05:** `BertSentenceEncoder` and the new `tokenizers` 0.23.1 dependency live behind a
  **`setfit` feature** in `aprender-core` from this phase forward. The `autograd/ops` primitives
  stay ungated.
- **D-06:** Establish the CPU feature matrix SAFE-02 will need now, while it is cheap:
  `--no-default-features`, `--features setfit`, and all-features must each build and test.
  A minimal inference consumer must never transitively pull `tokenizers`.

### Tokenizer Boundary

- **D-07:** Tokenizer and encoder are **separate types**. A tokenizer type produces a typed
  `SentenceBatch` (input ids, token type ids, attention mask, truncation facts, input
  provenance); the encoder consumes only `SentenceBatch`.
- **D-08:** Both are owned by **one bound model type that is the sole public entry point**, so a
  mismatched tokenizer/encoder pair is not constructible. Tokenizer parity and encoder-forward
  parity remain independently falsifiable gates.

### Reference Fixtures and Offline CI

- **D-09:** Commit a **small deterministic real-weight slice** of the pinned MiniLM in APR form
  (reduced layer/vocab subset, target a few hundred KB). Every CI job runs the graph, masking,
  pooling, normalization, and gradient gates against **real weight values**, not shapes.
  Synthetic-shape-only testing is explicitly rejected — PF-011 identifies it as how overclaimed
  compatibility ships.
- **D-10:** The **full ~90MB real-weight parity suite** runs behind an ignored/feature-gated test
  target fed by a locally fetched artifact. It is not in the default gate and the full weights are
  **not** vendored in git.
- **D-11:** Fixtures are committed as **human-readable JSON** under the crate's test tree.
- **D-12:** A **hash-locked `uv` environment** (`setfit==1.1.3`, `sentence-transformers==5.7.0`,
  `torch==2.13.0`, `scikit-learn==1.9.0`) plus a `scripts/` generator regenerates them as a
  **deliberate developer workflow — never in CI**. Commit the resolved lockfile with hashes.
- **D-13:** Commit a **manifest of fixture SHA-256s** so any regeneration appears as a reviewable
  diff. This is the structural guard against quietly re-baselining a fixture to match Rust.
- **D-14:** **Tolerances live in the phase contract**, derived from Python-side f32/f64
  round-trip noise, and are committed **in their own commit before any Rust comparison
  executes**. Loosening one then requires a contract edit that `pv diff` flags with a semver bump
  suggestion. This satisfies the STATE blocker: *freeze numerical tolerances from pinned
  reference fixtures before examining Rust discrepancies.*
- **D-15:** Freeze the **full ENC-01..06 fixture corpus in one pass**: tokenizer ids / type ids /
  masks / truncation facts, **per-layer transformer token outputs**, masked mean, normalized
  sentence embedding, pair cosine-MSE forward, selected parameter gradients, one controlled
  optimizer step, plus mixed-length padded batches and batch-1 vs batch-N. Per-layer
  intermediates are required so a mismatch localizes to a layer instead of the whole encoder.
- **D-16:** Dropout is **disabled** for all cross-framework forward and gradient fixtures. Rust
  seeded dropout placement, reproducibility, and statistics are tested separately; Python and
  Rust RNG streams are not expected to match bit-for-bit.

### Named Parameters, Modes, and Freezing

- **D-17:** **Extend the existing `Module` trait** in `crates/aprender-core/src/nn/module.rs`
  with named recursive traversal and train/eval propagation. **Do not introduce a separate
  `NamedModule` / `ParameterStore`** — core keeps exactly one parameter abstraction. Existing
  `Module` implementors (`nn/linear.rs`, `nn/normalization/`, `nn/dropout/`, `nn/container.rs`)
  gain named traversal as a side effect.
- **D-18:** Parameter names are **HF dotted, matching the source checkpoint verbatim**
  (e.g. `encoder.layer.0.attention.self.query.weight`), so gradient fixtures align with torch's
  `named_parameters()` with zero translation at the phase's highest-risk gate.
- **D-19:** The canonical mapping to `contracts/tensor-names-v1.yaml` is declared and validated
  at the **Phase 4 APR write boundary**, using `crates/aprender-core/src/format/converter/`.
  Phase 1 does not carry that mapping.
- **D-20:** **Default freeze policy is all-trainable**, matching SetFit's full-body fine-tuning.
  Any other default would be an unlabeled deviation from SetFit, which PF-008 treats as a claims
  defect.
- **D-21:** Freeze groups are opt-in configuration, and the test suite **pins at least one
  deliberately frozen group** so success criterion 4's "frozen components remain byte-identical"
  has something real to assert.
- **D-22:** Groups are addressed **per-module, per-layer** — `embeddings`,
  `encoder.layer.3.attention`, `encoder.layer.3.ffn`, `encoder.layer.3.norm` — matching exactly
  the components ENC-04 names. Not coarse-only (top-N freezing must be expressible) and not a
  per-tensor glob DSL (a freeze policy in the Phase 4 APR must be validated structure, not a
  string).

### Gate Expression

- **D-23:** Author **one new `contracts/setfit-encoder-conformance-v1.yaml`** owning all six ENC
  criteria plus the frozen tolerance table. It **references** the existing contracts rather than
  editing them; `encoder-forward-v1`, `nn-training-gradient-path-v1`, and
  `pool-flatten-embedding-backward-gradflow-v1` stay stable for their current consumers.
  `pv status` then reports one coherent Phase 1 gate and `pv diff` versions the whole
  conformance surface — including the tolerances — together.
- **D-24:** Prove detachment failure **in-band**: a test-only detached encoder variant that the
  gradient gate must reject, running in every `cargo test`. This is the evidence the gate is not
  theater.
- **D-25:** Back it with **`cargo-mutants` scoped to the new `autograd/ops` and encoder forward**
  to catch detachment nobody thought to write a test for.
- **D-26:** Wire the fixture-parity, gradient, and detach-negative tests into **`make tier2`**
  (pre-commit), with `pv validate contracts/setfit-encoder-conformance-v1.yaml` in
  **tier3/tier4** alongside the other contract gates. The slow gated real-weight suite stays out
  of the fast loop. A dedicated target outside the tiers was rejected — a target outside the
  tiers is a target that stops being run.
- **D-27:** Annotate each new autograd op and the encoder forward with **`#[contract]`**, binding
  equations through `BindingRegistry` into `crates/aprender-core/src/generated_contracts.rs`,
  backed by YAML falsification tests. Satisfies Rule 7 (coverage + contracts co-evolution) as the
  code lands rather than retroactively.

### Claude's Discretion

The user did not delegate any decision explicitly. The following were surfaced as remaining
gray areas and consciously left to research/planning as implementation detail:

- Exact dropout placement within the BERT block and the seeded-RNG policy proving ENC-05's
  "registered parameters do not change merely because the mode changes"
- Exact signatures and shape conventions of the six new autograd ops
- The reproducible derivation procedure for the committed small-slice APR from the pinned
  revision
- Whether `bert/load.rs` is reused directly or wrapped for pinned-revision import validation

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase requirements and scope
- `.planning/ROADMAP.md` — Phase 1 goal and the five success criteria that define "done"
- `.planning/REQUIREMENTS.md` — ENC-01 through ENC-06 verbatim; traceability table
- `.planning/PROJECT.md` — milestone core value, constraints, out-of-scope list

### SetFit domain research (authoritative for this milestone)
- `.planning/research/STACK.md` — **Supported Encoder Contract** table (pinned revision, 6
  layers, 384 hidden, 12 heads, 1536 intermediate, 30522 vocab, 512 positions, 256 sentence max,
  LayerNorm eps `1e-12`, dropout `0.1`); the six missing primitives; `SentenceBatch` /
  `SentenceEncoder` sketch; crate ownership table; feature gate table; numerical verification
  strategy
- `.planning/research/PITFALLS.md` — **PF-001** (silent no-op encoder tuning) is the phase's
  reason to exist; **PF-004** (numerical instability), **PF-011** (overclaimed model
  compatibility), **PF-014** (batching changes semantics) all gate in Phase 1
- `.planning/research/ARCHITECTURE.md`, `.planning/research/FEATURES.md`,
  `.planning/research/SUMMARY.md` — supporting milestone research

### Codebase state
- `.planning/codebase/CONCERNS.md` — "Parallel encoder and autograd stacks", "BERT
  implementation is explicitly inference-only", "Embedding and pooling operations detach the
  graph", "Tokenizer logic is duplicated at command boundaries", plus the batch-size-one and
  gradient-flow test-coverage gaps this phase closes
- `.planning/codebase/ARCHITECTURE.md` — layer boundaries and crate ownership rules
- `.planning/codebase/STACK.md` — workspace versions, feature flags, toolchain (Rust 1.93.0,
  MSRV 1.91), publishability constraints
- `.planning/codebase/TESTING.md` — contract, property, mutation, numerical-tolerance patterns
- `.planning/codebase/CONVENTIONS.md` — code conventions to match

### Existing contracts to reference (do not edit)
- `contracts/encoder-forward-v1.yaml` — existing encoder forward obligations
- `contracts/nn-training-gradient-path-v1.yaml` — existing gradient-path obligations
- `contracts/pool-flatten-embedding-backward-gradflow-v1.yaml` — closest precedent for
  pooling/embedding backward gradient flow
- `contracts/transformer-end-to-end-trainable-v1.yaml` — end-to-end trainability precedent
- `contracts/embedding-lookup-v1.yaml`, `contracts/learned-position-embedding-v1.yaml` —
  embedding gather semantics
- `contracts/lora-adapter-trains-base-frozen-v1.yaml` — **the frozen-vs-trainable proof
  pattern**; closest precedent for D-21
- `contracts/codebert-tokenizer-validation-v1.yaml` — tokenizer validation precedent
- `contracts/apr-pytorch-autograd-equivalence-beat-v1.yaml` — torch-equivalence precedent
- `contracts/tensor-layout-v1.yaml` — **row-major is mandatory**; the LAYOUT-001/002 rules in
  `CLAUDE.md` apply to every new op
- `contracts/tensor-names-v1.yaml` — canonical naming scheme; relevant to D-19 at the Phase 4
  boundary, not to Phase 1 traversal names

### Repository rules
- `CLAUDE.md` — Verification Discipline (all 8 rules), LAYOUT-001/002 tensor layout safety,
  "Contract Validation: DOGFOOD `pv`, NEVER bash", code search policy (`pmat query`, not
  grep/glob), tiered quality gates
- `.claude/skills/pre-release/SKILL.md` — publishability, MSRV, and feature-combination
  constraints that D-05/D-06 must not break
- `.claude/skills/apr-dogfood/SKILL.md` — dogfooding conventions

### Upstream sources (pinned)
- `sentence-transformers/all-MiniLM-L6-v2` @ revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` — never a mutable branch name
- SetFit `CosineSimilarityLoss` semantics — see `.planning/research/STACK.md` Sources section for
  the full authoritative URL list

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `crates/aprender-core/src/autograd/` — the canonical stack. `grad_fn.rs` (47K), `graph.rs`,
  `tensor.rs`, `ops/`, plus `tests_elementwise_backward.rs`, `tests_matmul_backward.rs`, and
  `tests_tensor_contract.rs`. New ops land in `ops/` and follow these existing backward-test
  patterns.
- `crates/aprender-core/src/nn/module.rs` (6.1K) — the `Module` trait already exists. D-17
  extends it; BERT simply never implemented it.
- `crates/aprender-core/src/nn/` — `linear.rs`, `normalization/`, `activation.rs`, `dropout/`,
  `transformer/`, `container.rs` are existing `Module` implementors and candidate building
  blocks for the new encoder.
- `crates/aprender-core/src/models/bert/config.rs` (3.0K) and `bert/load.rs` (30K) — reused by
  D-01 for architecture validation and weight loading.
- `crates/aprender-core/src/nn/optim/` (AdamW) and `crates/aprender-core/src/optim/lbfgs.rs` —
  AdamW is the optimizer for the controlled step in ENC-04; L-BFGS is Phase 3's head.
- `crates/aprender-core/src/models/qwen2/` — has differentiable gather/backward logic worth
  reading before writing the batched embedding gather op.
- `crates/aprender-core/src/format/converter/` — the existing name/shape mapping layer that D-19
  will use at the Phase 4 boundary.

### Established Patterns

- **Contract-first.** YAML in `contracts/` binds to Rust via `#[contract]` and
  `generated_contracts.rs`. `pv` is the only sanctioned contract tool — never a bash/yq/python
  workaround.
- **Row-major everywhere.** `contracts/tensor-layout-v1.yaml` is the source of truth. Every new
  op must be row-major; the forbidden `_colmajor` imports listed in `CLAUDE.md` apply.
- **Tiered gates.** `make tier1` (<1s) / `tier2` (<5s) / `tier3` (1–5min) / `tier4` (CI). D-26
  places the phase gate in tier2 and tier3.
- **`unwrap()` is banned** via `.clippy.toml` disallowed-methods; `unsafe_code = "forbid"`
  workspace-wide. All new fallible paths return typed errors.
- **Crate-name split.** `aprender-train`'s library name is `entrenar`; `aprender-compute`'s is
  `trueno`; `aprender-serve`'s is `realizar`. Use library names in Rust imports, directory names
  in path discussion.

### Integration Points

- New ops attach to `crates/aprender-core/src/autograd/ops/` and register through
  `crates/aprender-core/src/generated_contracts.rs`.
- The new encoder attaches under `crates/aprender-core/src/` behind the `setfit` feature; the
  feature must be declared in `crates/aprender-core/Cargo.toml` and forwarded consistently per
  `.planning/research/STACK.md`'s feature-gate table.
- `tokenizers = { version = "0.23.1", default-features = false, features = ["fancy-regex"] }`
  is added as a **workspace** dependency and enabled only from the `setfit` feature.
- **Nothing in `crates/apr-cli/` changes this phase** — Phase 1 is library-only.
- **Nothing in `crates/aprender-train/`** — `SetFitTrainer` is Phase 3. Do not touch
  `aprender-train/src/transformer/` or `aprender-train/src/autograd/`; that legacy stack is
  explicitly not the target.

### Known Traps in This Territory

- `crates/aprender-core/src/models/bert/embeddings.rs` **asserts** on excessive length and uses
  unchecked slice ranges (CONCERNS.md known bug). The new path must return typed errors.
- `crates/aprender-train/src/transformer/embedding.rs` silently zero-fills out-of-vocabulary IDs
  and repeats the final position embedding past the maximum. The new path must fail closed.
- Existing contrastive utilities (`crates/aprender-core/src/loss/loss.rs`,
  `nn/self_supervised.rs`) return `f32`, not tensors. Reusing their **names** in SetFit code
  would falsely imply encoder fine-tuning is occurring — do not.

</code_context>

<specifics>
## Specific Ideas

- The phase's single highest-value artifact is the **in-band detached-encoder negative test**
  (D-24). It is the difference between a gradient gate and a gradient gate that is theater, and
  it is what makes success criterion 5 checkable.
- **Per-layer fixture intermediates** (D-15) were chosen specifically so that the first
  mismatch is a localized failure rather than a six-layer bisect with no ground truth.
- The **fixture SHA-256 manifest** (D-13) and **contract-resident tolerances** (D-14) exist as a
  matched pair: together they make "widen the tolerance until it passes" a visible, versioned,
  argued act rather than a one-character test edit.
- Ordering note carried from STATE: prove the real-weight mixed-batch graph **first**; the full
  BERT convergence is Phase 4's problem, not this phase's.

</specifics>

<deferred>
## Deferred Ideas

- **Converging `apr embed` / `rerank` / `CrossEncoder` onto the shared encoder** — Phase 4,
  where OPS-03 already requires it and APR parity tests can prove it (D-02).
- **Canonical `tensor-names-v1.yaml` mapping for encoder tensors** — Phase 4 APR write boundary
  (D-19).
- **Alternative encoder objectives** (InfoNCE, SupCon, CoSENT, triplet) — v2, per
  REQUIREMENTS.md EXT-02. Not a Phase 1 option; they change SetFit fidelity and numerical
  references.
- **Additional sentence-transformer families** — v2 (EXT-01), each requiring its own named
  adapter and parity corpus.
- **Accelerator paths** — deferred until the full CPU lifecycle is proven; ACC-01 in v2.
- **Persistent embedding/token caches** — explicitly out of scope for v1 (CACHE-01); PF-010
  documents the staleness risk.

</deferred>

---

*Phase: 1-Differentiable MiniLM Conformance*
*Context gathered: 2026-08-07*
