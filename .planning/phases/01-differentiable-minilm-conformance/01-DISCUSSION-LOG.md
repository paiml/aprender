# Phase 1: Differentiable MiniLM Conformance - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-07
**Phase:** 1-Differentiable MiniLM Conformance
**Areas discussed:** Encoder refactor strategy, Fixtures & offline CI, Param names & freeze, Gate expression (contracts)

---

## Encoder refactor strategy

### Q1 — Where does the differentiable encoder live?

| Option | Description | Selected |
|--------|-------------|----------|
| New encoder, converge later | Add `BertSentenceEncoder` in aprender-core reusing `bert/config.rs` + `bert/load.rs` with a graph-connected forward; leave `apr embed` / rerank / CrossEncoder untouched; convergence lands in Phase 4 under OPS-03 | ✓ |
| Refactor `bert/` in place now | Implement Module recursively on BertEmbeddings/BertLayer/BertEncoder and replace every `.data()` read; exactly one BERT from day one, but changes `load.rs` (30K) and the CrossEncoder/rerank paths before the graph is proven on real weights | |
| New encoder + delegate in Phase 1 | Build the trainable encoder then rewire inference BERT to delegate before the phase closes; one implementation by end of phase, but couples the encoder proof to an inference-regression sweep in the same phase | |

**User's choice:** New encoder, converge later
**Notes:** Honors STATE's constraint — validate the real-weight mixed-batch graph before committing the full BERT refactor. Smallest blast radius for the highest-risk gate.

### Q2 — How broadly are the six missing differentiable primitives built?

| Option | Description | Selected |
|--------|-------------|----------|
| General ops in `autograd/ops/` | Model-agnostic ops alongside existing matmul/elementwise backward, each with its own finite-difference test; retires the CONCERNS.md graph-detachment debt directly | ✓ |
| SetFit-scoped, behind the feature | Narrower blast radius and faster to land, but gather/pool logic then exists in two places — the duplication CONCERNS.md flags as tech debt | |
| Generalize Qwen2's existing gather first | Reuses proven backward code, but couples the phase to Qwen2's baked-in shape assumptions | |

**User's choice:** General ops in `autograd/ops/`
**Notes:** Qwen2's differentiable gather remains worth reading as prior art even though its logic is not being promoted wholesale.

### Q3 — Feature gating for the encoder and the new `tokenizers` dependency

| Option | Description | Selected |
|--------|-------------|----------|
| `setfit` feature from Phase 1 | Encoder + tokenizers gated; `autograd/ops` primitives ungated; establishes the SAFE-02 CI matrix while it is cheap | ✓ |
| Ungated in core, add gate in Phase 4 | Simpler builds now, but `tokenizers` becomes an unconditional aprender-core dependency inherited by every downstream consumer | |
| Gate everything including primitives | Default build stays byte-identical, but fragments the autograd op surface by feature flag | |

**User's choice:** `setfit` feature from Phase 1
**Notes:** Retrofitting a gate after Phase 3 depends on the encoder would be substantially more painful.

### Q4 — Where does the tokenizer sit relative to the encoder?

| Option | Description | Selected |
|--------|-------------|----------|
| Separate types, bound public entry | Tokenizer produces typed `SentenceBatch`; encoder consumes only `SentenceBatch`; both owned by one bound model type that is the sole public entry point | ✓ |
| Encoder owns the tokenizer | Cannot drift by construction, but every encoder-forward fixture test runs through tokenization so a tokenizer bug and a graph bug present identically | |
| Fully separate, bind in Phase 3 | Maximum testability and smallest Phase 1 surface, but leaves a window where callers can pair the wrong tokenizer with the encoder | |

**User's choice:** Separate types, bound public entry
**Notes:** Keeps tokenizer parity and encoder-forward parity independently falsifiable while making a mismatched pair unconstructible.

---

## Fixtures & offline CI

### Q1 — How do pinned MiniLM weights reach tests under SAFE-02's no-Python/no-network rule?

| Option | Description | Selected |
|--------|-------------|----------|
| Committed small-slice APR + gated full run | Small deterministic real-weight slice runs the gates in every CI job; full ~90MB parity suite behind an ignored/feature-gated target fed by a locally fetched artifact | ✓ |
| Synthetic MiniLM-shaped weights in CI | Nothing large in git and trivially regenerable, but PF-011 is explicit that shape-only testing is how overclaimed compatibility ships | |
| Commit the full pinned weights | Maximum fidelity and total offline determinism, at real cost to clone size, crates.io packaging, and the publish path CLAUDE.md guards | |

**User's choice:** Committed small-slice APR + gated full run
**Notes:** Keeps the falsifiers in the default gate rather than in a job most people never run.

### Q2 — Fixture generation, storage, and drift control

| Option | Description | Selected |
|--------|-------------|----------|
| Committed JSON + hash-locked uv generator | Human-readable JSON in the crate test tree; hash-locked uv env and `scripts/` generator run as a deliberate developer workflow, never in CI; committed SHA-256 manifest makes regeneration a reviewable diff | ✓ |
| Committed npz/binary + generator | Smaller and faster to load, but a regenerated fixture is an opaque diff — a reviewer cannot see that an expected gradient moved | |
| Nightly CI regenerates and diffs | Continuous proof the oracle still agrees, but adds a Python-dependent CI job and lets upstream releases redden the build for unrelated reasons | |

**User's choice:** Committed JSON + hash-locked uv generator
**Notes:** Reviewability of the diff was the deciding property.

### Q3 — What structurally prevents post-hoc tolerance tuning?

| Option | Description | Selected |
|--------|-------------|----------|
| Contract YAML, committed before Rust runs | Tolerances derived from Python-side f32/f64 round-trip noise, landing in their own commit before any Rust comparison; loosening one requires a contract edit that `pv diff` flags with a semver bump | ✓ |
| Central Rust consts module | Simple and close to the code, but relaxing a value is an ordinary source edit reading like any other diff | |
| Per-test inline tolerances | Maximum locality, but no single place shows the tolerance budget and nothing distinguishes a considered value from one widened until green | |

**User's choice:** Contract YAML, committed before Rust runs
**Notes:** Directly discharges the STATE blocker on freezing tolerances before examining Rust discrepancies.

### Q4 — Fixture corpus scope for Phase 1

| Option | Description | Selected |
|--------|-------------|----------|
| Full ENC-01..06 surface | Tokenizer facts, per-layer transformer outputs, masked mean, normalized embedding, pair cosine-MSE, selected gradients, one optimizer step, mixed-length and batch-1 vs batch-N — one generator run, one review | ✓ |
| Endpoints only, add depth on failure | Less Python work and smaller fixtures, but the first mismatch becomes a six-layer bisect with no intermediate ground truth | |
| Forward now, gradients in a follow-up | Faster to first green, but PF-001 is precisely the gradient half — deferring it defers the phase's reason to exist | |

**User's choice:** Full ENC-01..06 surface
**Notes:** Per-layer intermediates chosen specifically so mismatches localize.

---

## Param names & freeze

### Q1 — Which named-parameter abstraction?

| Option | Description | Selected |
|--------|-------------|----------|
| Extend the existing `Module` trait | Add named recursive traversal and train/eval propagation to `nn/module.rs`; existing implementors gain it at once; core keeps exactly one parameter abstraction | ✓ |
| Separate `NamedModule` / `ParameterStore` | Zero risk to current implementors and a clean sheet, but core then has two parameter abstractions — the shape of the dual-stack problem this phase exists to fix | |
| Static manifest, no trait change | Names become diffable data, but freeze/unfreeze and train/eval still need a runtime mechanism, so it mostly moves the problem | |

**User's choice:** Extend the existing `Module` trait

### Q2 — Naming convention

| Option | Description | Selected |
|--------|-------------|----------|
| HF dotted internally, mapped at APR write | Traversal names match the source checkpoint verbatim so gradient fixtures align with torch `named_parameters()` without translation; canonical mapping validated at the Phase 4 APR boundary | ✓ |
| APR canonical names throughout | `apr tensors`/`diff`/`qa` read naturally and Phase 4 needs no mapping, but every Phase 1 fixture comparison goes through a hand-maintained translation table at the riskiest boundary | |
| HF dotted everywhere, no mapping | Simplest end to end, but the SetFit artifact names tensors differently from every other model in the repo | |

**User's choice:** HF dotted internally, mapped at APR write
**Notes:** Removes a class of name-translation bugs from the phase's highest-risk gate; `format/converter/` already exists for the Phase 4 mapping.

### Q3 — Default freeze policy

| Option | Description | Selected |
|--------|-------------|----------|
| All trainable; freeze exercised in tests | Matches SetFit full-body fine-tuning; freeze groups are opt-in and at least one deliberately frozen group is pinned in tests so the byte-identical assertion is real | ✓ |
| Freeze embeddings by default | Fewer parameters to move on 8-shot data and faster CPU steps, but a deviation from SetFit's default that must be declared wherever the method is named | |
| Top-N layers only by default | Cheapest CPU path, but makes the headline configuration something other than SetFit | |

**User's choice:** All trainable; freeze exercised in tests
**Notes:** PF-008 treats an unlabeled SetFit deviation as a claims defect, not a tuning choice.

### Q4 — Freeze group granularity

| Option | Description | Selected |
|--------|-------------|----------|
| Per-module, per-layer | `embeddings`, `encoder.layer.3.attention`, `encoder.layer.3.ffn`, `encoder.layer.3.norm` — matching the components ENC-04 names | ✓ |
| Coarse groups only | Trivial to configure and validate, but top-N layer freezing becomes unexpressible without a later schema change | |
| Per-tensor glob patterns | Maximum flexibility, but typos silently freeze nothing and the Phase 4 APR records a string rather than validated structure | |

**User's choice:** Per-module, per-layer

---

## Gate expression (contracts)

### Q1 — New contract or extensions to existing ones?

| Option | Description | Selected |
|--------|-------------|----------|
| One new `setfit-encoder-conformance-v1` | Owns all six ENC criteria plus the frozen tolerances; references existing contracts rather than editing them; `pv status` reports one coherent gate and `pv diff` versions the whole surface together | ✓ |
| Extend the existing contracts | No new file and existing gates get stronger, but the Phase 1 exit criterion spreads across three documents with three version histories, and each edit risks current consumers | |
| New contract per concern | Each small and independently versionable, but three files must be read together to know whether the phase passed | |

**User's choice:** One new `setfit-encoder-conformance-v1`

### Q2 — How is "deliberate detachment makes the gate fail" proven?

| Option | Description | Selected |
|--------|-------------|----------|
| Negative test in default gate + scoped mutants | Test-only detached encoder variant the gradient gate must reject, running in every `cargo test`, backed by `cargo-mutants` scoped to the new ops and encoder forward | ✓ |
| cargo-mutants only | Finds cases a human wouldn't enumerate, but runs are slow and out-of-band so a green PR carries no in-band evidence the gate can fail | |
| Dedicated negative test only | Fast and always runs, but only proves the gate catches the one detachment someone remembered to write | |

**User's choice:** Negative test in default gate + scoped mutants
**Notes:** CLAUDE.md's verification discipline is explicit that a guard is only proven by re-running the mutation that turns it RED.

### Q3 — Where does the gate run?

| Option | Description | Selected |
|--------|-------------|----------|
| Wire into existing tiers | Fixture-parity, gradient, and detach-negative tests in `make tier2`; `pv validate` in tier3/tier4; slow gated real-weight suite stays out of the fast loop | ✓ |
| Dedicated `make setfit-gate` target | Clean to reason about and easy to run in isolation, but blocks nothing by default | |
| New CI job only | Guaranteed to run before merge, at the cost of a slow local feedback gap during active development | |

**User's choice:** Wire into existing tiers
**Notes:** A target outside the tiers is a target that stops being run.

### Q4 — Contract binding depth

| Option | Description | Selected |
|--------|-------------|----------|
| `#[contract]` on new ops + YAML falsifiers | Annotate each new autograd op and the encoder forward, binding equations through `BindingRegistry` into `generated_contracts.rs`; satisfies Rule 7 as the code lands | ✓ |
| YAML falsification tests only | Less ceremony and `pv validate` still gates the phase, but equations stay documentation with no compile-time binding to the code | |
| `#[contract]` on the six new ops only | Covers the reusable primitives while keeping annotation off the composed path | |

**User's choice:** `#[contract]` on new ops + YAML falsifiers

---

## Claude's Discretion

The user did not explicitly delegate any decision. The following were surfaced as remaining gray
areas and consciously left to research/planning as implementation detail:

- Dropout placement within the BERT block and the seeded-RNG policy proving ENC-05
- Exact signatures and shape conventions of the six new autograd ops
- Reproducible derivation of the committed small-slice APR from the pinned revision
- Whether `bert/load.rs` is reused directly or wrapped for pinned-revision import validation

## Deferred Ideas

- Converging `apr embed` / `rerank` / `CrossEncoder` onto the shared encoder — Phase 4 (OPS-03)
- Canonical `tensor-names-v1.yaml` mapping for encoder tensors — Phase 4 APR write boundary
- Alternative encoder objectives (InfoNCE, SupCon, CoSENT, triplet) — v2, EXT-02
- Additional sentence-transformer families — v2, EXT-01
- Accelerator paths — v2, ACC-01
- Persistent embedding/token caches — out of scope for v1, CACHE-01
