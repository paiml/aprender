# Phase 3: Faithful Two-Stage Trainer and Head - Context

**Gathered:** 2026-08-09
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers **the training lifecycle itself**: consume Phase 2's typed `Selection` and
replayed pair stream, tune the Phase 1 `SetFitMiniLm` encoder with the contrastive objective, then
fit **one** multiclass head on each unique tuned embedding **exactly once** — with stage ordering,
SetFit-identity evidence, canonical-validation-only selection, and CPU reproducibility all enforced
**by construction** rather than by convention.

Requirements: TRN-01..TRN-07, SAFE-03.

**In scope:** the typed lifecycle and its transitions; encoder tuning with the existing
`pair_cosine_mse` objective; a general multiclass logistic head; the SetFit-identity evidence gate;
the selection-lock record; the two-clean-runs reproducibility contract; configuration validation
before training begins.

**Not in scope (and why):**
- The real `setfit-apr-v1` write path and the production loader — Phase 4 (APR-01/APR-02). Phase 3
  owns the *verification boundary* only (see D-07).
- Any CLI/HTTP/eval parity surface — Phase 4 (OPS-01..06).
- Benchmark cells, F_avg reporting, LoRA comparison — Phase 5 (EVAL-01..05).
- Re-litigating encoder internals, pair semantics, or data protocol — settled in Phases 1 and 2.

</domain>

<decisions>
## Implementation Decisions

### Classifier Head (TRN-04, TRN-05)

- **D-01:** The multiclass head is a **new general `MultinomialLogisticRegression` in
  `crates/aprender-core/src/classification/`** — a dataset-agnostic capability with SetFit as its
  **first consumer, not its owner**. This is deliberately the Phase 2 D-01 pattern applied again.
  The existing binary `LogisticRegression` (`classification/mod.rs:114`) and its
  `apr-stochastic-lr-v1` contract are **left untouched**: it is binary-only (`"Labels must be 0 or
  1"`), plain-GD, and returns `Result<(), String>` where TRN-04 requires typed failure. Widening it
  in place would mean a Phase 3 gate defending code this milestone did not write.
  PROJECT.md's "fragmented binary/multiclass classifier heads" risk narrows without a public-API
  break; a later deliberate retirement of the binary type is out of scope here.

- **D-02:** The solver is the **existing `crates/aprender-core/src/optim/lbfgs.rs`**, reused as-is
  in shape: `minimize(objective, gradient, x0) -> OptimizationResult` with Wolfe line search, a
  gradient-norm `tol`, and a `ConvergenceStatus`. The head supplies only the softmax-NLL + L2
  objective and its analytic gradient. TRN-04's "explicit convergence or typed failure" maps onto
  `ConvergenceStatus` rather than being invented. L-BFGS is also the same solver family sklearn's
  multinomial default uses, so reference agreement is a *tolerance* question, not an *algorithm*
  question. No new numerical algorithm is introduced.

- **D-03:** **Fit in f64, store f32.** The L-BFGS path is widened to f64 (generic over the float, or
  a distinct f64 entry point); fitted coefficients are downcast to f32 at the APR boundary, which
  keeps APR-01's F32 artifact requirement intact. Rationale: near the optimum the softmax-NLL
  gradient norm can approach f32 epsilon (~1.2e-7), so an f32 Wolfe line search can stall and report
  progress it did not make — and the 64-shot low-regularization cell is exactly where that bites.
  Fitting in the reference's precision also means "reference-matching regularization behavior" is a
  real comparison rather than one confounded by f32 noise.
  **Consequence the planner must handle:** this touches a *contracted* optimizer. Expect `pv diff`
  to suggest a semver bump on the L-BFGS contract, and note the blast radius extends beyond this
  phase because `optim::LBFGS` is shared code.

- **D-04:** Regularization is parameterized **natively** — public API is mean-NLL + `λ‖W‖²` with an
  **unpenalized intercept** — and the sklearn relation is **one contracted equation with its own
  falsification test**: `λ = 1/(2·C·n)`, sum-vs-mean, intercept exclusion.
  **Amended (phase-3 review):** the relation was written `1/(C·n)`; the correct constant is
  `1/(2·C·n)`, because sklearn's penalty carries a half inside `r(W) = ½‖W‖²_F` while aprender's
  native form is `λ‖W‖²_F`. Both reviewers independently flagged the factor of two and plan 03-04
  already implements the corrected constant; this amendment makes CONTEXT agree with the contract
  instead of silently contradicting it. Rationale: a general
  `classification::` type should not inherit sklearn's idiosyncratic `C` forever, and this repo has
  the exact precedent for validating a numerical relation against a Python reference —
  `crates/aprender-core/src/glm/glm_tests.rs:280` catches a swapped IRLS link derivative against
  statsmodels/scipy. The three silent-mismatch sources (inverse scale, sum vs mean, intercept
  penalization) are named explicitly so the conversion cannot drift unnoticed.

### Trainer Home and Lifecycle (TRN-01, TRN-02)

- **D-05:** The two-stage trainer lives in **`crates/aprender-train/src/train/setfit/`** — the
  architecture map's declared home for training policy. `optim/adamw.rs`, `optim/clip.rs`,
  `optim/scheduler/`, and `train/device.rs::resolve_device` (which already **fails closed** on an
  explicitly requested unavailable CUDA device, exactly what TRN-02 needs) are reused **in place**.
  Adds exactly one dependency edge: `aprender-train -> aprender-contrastive-data`.
  **Rejected alternatives and why, so the planner does not relitigate:**
  - `aprender-core/src/setfit/trainer.rs` would require **reimplementing AdamW inside
    `aprender-core::optim`**, because `aprender-train -> aprender-core` means reaching train's
    AdamW would be a cycle. `aprender-core::optim` has L-BFGS/SGD/CG/ADMM/FISTA but **no AdamW**
    (verified). Duplicating a 24 KB optimizer recreates precisely the fragmentation D-01 avoids;
    falling back to core's SGD is a silent deviation from SetFit's reference training.
  - A new `aprender-setfit` crate is clean but adds a **third link to the crates.io publish
    cascade**, and Phase 2's one still-open UAT item is exactly that cascade (`apr-cli` cannot
    package until `aprender-contrastive-data` is published by hand).
  **Consequence the planner must handle:** the `setfit` feature now has to propagate through a
  37-module crate that also carries GPU/LoRA/distill/server. Keeping the CPU-only feature-matrix job
  green there is real work, not a flag rename. (`aprender-contrastive-data` has **no** edge to
  `aprender-core` — verified — so this direction stays acyclic either way.)

- **D-06:** The lifecycle is a **phantom typestate in Phase 2's house style** —
  `SetFitRun<Prepared>` / `<EncoderTuned>` / `<HeadFitted>`, each transition consuming `self` and
  returning the next state. Illegal orderings become **non-constructible**, not merely rejected,
  and this inherits Phase 2's **trybuild non-constructibility test** pattern directly (established
  for `Split<Train>` and `PreparedDataset<Canonical|Compatibility>`), so "you cannot express it" is
  proven rather than asserted. A runtime state machine was rejected as a regression against the bar
  the data side already meets.

- **D-07:** **`ArtifactReloadedAndVerified` — Phase 3 owns the boundary, Phase 4 owns the format.**
  Phase 3 defines the marker state *and* the verification trait: close the in-memory model, reload
  from bytes, re-encode, re-predict, compare within contracted tolerances. Phase 3 ships a
  **serde-based implementation** of that trait so the round-trip invariant is written and gated in
  the phase whose criterion demands it; **Phase 4 supplies the APR implementation of the same
  seam**, swapping the format rather than inventing the check.
  **Amended (phase-3 review):** the seam is split in two. The implementable half is a pure
  **codec** (`SetFitCodec`: bytes <-> bundle, plus a format id) carrying **no comparison policy,
  no tolerances and no hashing**. The verification policy — artifact hashing, close, reload,
  re-encode, re-predict, compare, and minting `ArtifactReloadedAndVerified` — stays in trusted
  crate-internal lifecycle code no external implementor can override. The codec trait is
  **sealed** for Phase 3 (un-sealing later is non-breaking; sealing later is not), and Phase 4's
  APR codec lands as a thin adapter module inside `aprender-train/src/train/setfit/` that calls
  `aprender-core`'s APR format code — the FORMAT stays in core, only the adapter moves. Reason: a
  public unsealed verifier whose `reload()` the implementor controls can return the pre-close
  bundle unchanged and mint the final state without a real persistence boundary ever existing. Shipping the real
  `setfit-apr-v1` here was rejected: it would freeze the artifact schema before the consumers that
  must read it (APR-03/APR-04) have been designed.

- **D-08:** **TRN-05 is enforced structurally.** The `HeadFitted` transition accepts **only** Phase
  2's `Selection` (the typed unique-ID set) and has **no access to the pair stream at all** — pair
  multiplicity is *inexpressible*, not rejected at runtime. Each unique selected row is encoded
  exactly once in evaluation/no-gradient mode with dropout disabled. Backed by a **test-only
  pair-weighted fitter that must FAIL its gate in every `cargo test`**, per the Ph1 D-24 / Ph2 D-25
  in-band-negative discipline. The encoding **batch composition is pinned**, so head embeddings are
  reproducible rather than padding-dependent (Phase 1 proved padding invariance *within tolerance*,
  which is not bitwise).

### SetFit-Identity Gate (TRN-03, SAFE-03)

- **D-09:** Evidence covers **every trainable parameter minus the declared freeze policy**. The
  exemption set is exactly the `FreezeGroup` list — already a validated structured enum from Ph1
  D-22 where a group matching zero names is a typed error. Records **aggregate stats per named
  parameter** (gradient norm, delta norm), not raw tensors. This makes SAFE-03 **automatic**: an
  all-frozen run has an empty trainable set and therefore cannot pass, with no separate rule needed
  to catch it. Component-level aggregation was rejected because a single dead parameter inside a
  live component would be invisible.

- **D-10:** The threshold is **relative to the initial norm with a contracted ε**:
  `‖Δθ‖ / max(‖θ_init‖, s_class) > ε_class` per parameter, alongside finite non-zero gradient
  norms and a strict `‖Δθ‖ > 0`.
  **Amended (phase-3 review), three changes:**
  (a) **Denominator floor.** A bare `‖θ_init‖` denominator is undefined for an exactly
  zero-initialized parameter (transformer biases are the standard case): it yields NaN/Inf, and
  `NaN > ε` is false, so a legitimate run would be rejected. The denominator becomes
  `max(‖θ_init‖, s_class)` with `s_class` a contracted positive per-class scale floor, and
  `‖Δθ‖ > 0` is required as a separate strict predicate.
  (b) **ε is per parameter class, not one global number.** This decision's own argument —
  "LayerNorm gains and embedding tables differ by orders of magnitude" — applies to a single
  relative ε exactly as it applies to a single absolute floor. Classes derive from the HF dotted
  name prefix: embedding tables / LayerNorm / dense-and-attention.
  (c) **ε carries its calibration regime.** A number measured on one architecture at one
  shot/epoch setting does not generalize: a sparse embedding table's relative delta shrinks as the
  vocabulary grows, so a fixture-derived ε can make the *legal* lifecycle unusable at production
  scale. The contract records the calibrated regime (architecture fingerprints, seeds, shot/epoch
  boundaries); the gate **fails closed** with a typed `UncalibratedRegime` error for a run outside
  it. Extending to a new architecture is a deliberate contract edit that `pv diff` flags. ε is frozen in the
  contract per the Ph1 D-14 tolerance discipline (committed before any comparison runs; loosening it
  requires a contract edit `pv diff` flags). Scale-free is required, not preferred: LayerNorm gains
  and embedding tables differ by orders of magnitude, so one absolute floor is necessarily wrong
  somewhere. A bare non-zero-change test was rejected — a learning rate of 1e-30 passes it, which is
  the numerically meaningless update SAFE-03 exists to keep from being labeled SetFit.

- **D-11:** The gate fires **inside the transition**: `tune_encoder()` returns
  `Result<SetFitRun<EncoderTuned>, _>` and **fails** when evidence does not pass. `EncoderTuned`
  therefore cannot exist without evidence; since `HeadFitted` requires `EncoderTuned` and export
  requires `HeadFitted`, the whole chain is gated by construction rather than by every consumer
  remembering to check.
  **Direct consequence:** frozen linear probes, centroid classifiers, and other non-updating
  baselines run through a **distinct, differently-named type that never claims SetFit**. That is how
  SAFE-03's "remain explicitly labeled as such" becomes structural, and it gives the existing
  `contracts/linear-probe-classifier-v1.yaml` a real binding. Phase 5 still needs these baselines
  runnable — this is the path.

- **D-12:** The evidence record is a **compact summary plus a hash-bound full table**. The artifact
  and the Phase 5 benchmark row carry: verdict, trainable/frozen counts, min/median/worst relative
  delta, the worst-offending parameter name, the ε and contract version used, and a **hash of the
  complete per-parameter table**. The full table is emitted separately as machine-readable JSON.
  This keeps the APR small (APR-01 specifies an oversized-artifact rejection) and the 40 rows
  readable, while the detail stays auditable and the hash **binds** the summary to it.
  **Amended (phase-3 review):** "non-forgeable" overclaims. A SHA-256 stored beside the mutable
  content it hashes is **tamper-evidence and linkage**, not authenticity — whoever edits the table
  can recompute the hash. It catches accidental divergence and un-recomputed edits; genuine
  non-forgeability needs an anchor outside the artifact, which arrives with Phase 4's APR
  checksum. Contract and plan wording say "binding", never "non-forgeable". Directly mirrors Phase 2 D-09: replay the detail, persist the hash.

### Determinism and Selection Lock (TRN-06, TRN-07)

- **D-13:** **Bitwise at any thread count, via fixed-order reductions.** Every trainer-side
  reduction is routed through fixed chunking with an order-fixed combine, so the loss trace is
  bitwise identical regardless of thread count. This extends D-20's stated principle from sampling
  to arithmetic: Phase 2 chose Philox specifically so thread-count independence is **structural
  rather than asserted**, and a `par_iter().sum()` is not bitwise reproducible even at a fixed
  thread count because work-stealing changes the reduction tree. "Declared deterministic portions of
  the loss trace" therefore resolves to: **all of it**.
  **Open for the planner:** nothing in the repo currently guarantees fixed-order reductions, and the
  existing parallel reductions live in `aprender-compute`. Where this capability lands — and whether
  `aprender-compute` should own it — is unresolved and needs research.

- **D-14:** The **selection lock is hash-committing, with a typestate token**. The record commits to the **full candidate
  history** — every candidate's configuration hash, artifact hash and canonical-validation
  evaluation — plus the deterministic **selection rule**, the chosen candidate, the dataset and
  validation-split fingerprints, and the access-ledger hash.
  **Amended (phase-3 review), three changes:**
  (a) A single chosen metric cannot prove validation-only selection, so the lock commits the
  candidate list and applies the selection rule itself — a hand-picked "winner" is unexpressible
  because there is no `chosen` parameter.
  (b) A validation metric is **computed by a trusted evaluator** from the verified artifact and
  the `Split<Validation>`, never supplied as a caller-asserted `f64`. Possessing a validation
  split does not prove a number was computed from it.
  (c) Token minting takes the **verified run object**, not caller-supplied hash bytes, and reads
  that object's own artifact hash. A `[u8; 32]` parameter lets a caller hand over the locked hash
  and then evaluate a completely different artifact. Canonical test access requires a token minted **only** from a lock whose
  artifact hash **matches the model about to be evaluated** — so lock, keep tuning, then test
  **invalidates** rather than passes. An existence-only record was rejected precisely because that
  sequence would sail through it. This gives Phase 5's "any post-test-selected cell invalidates the
  report" something mechanical to check. Phase 2's access ledger remains the record of *which splits
  were touched*; it cannot alone distinguish honest selection from selection redone after a test
  peek, which is why it is a complement and not the mechanism.

- **D-15:** **Dropout stays ON during encoder tuning**, matching SetFit's reference recipe, with
  masks drawn from `aprender-rand` keyed by `(root_seed, "dropout", layer, step, block)` so mask
  element *i* is a pure function of its index.
  **Amended (phase-3 review) — clarification, not a change of direction:** `block` is the
  **forward-call ordinal** `2*step + branch`, where `branch` is 0 for the pair's A sentence and 1
  for its B sentence. The coordinate is load-bearing and the first plan draft dropped it: the two
  siamese branches are two separate encoder forwards at the same `step`, so a key without it hands
  corresponding elements the identical mask in both branches — artificial correlation and a silent
  deviation from the reference recipe, while still looking perfectly deterministic. That construction is what makes dropout compatible
  with D-13's bitwise-at-any-thread-count guarantee — replay-exact and thread-count independent by
  the same mechanism D-20 chose for sampling. Disabling dropout was rejected as an undeclared
  deviation from the reference recipe (PF-008 treats that as a claims defect). Note this closes the
  question Ph1 D-16 explicitly deferred: fixtures disabled dropout for *cross-framework* comparison
  only, leaving Rust seeded-dropout reproducibility "tested separately" — this is that test.
  Dropout is disabled for head fitting per D-08 and ROADMAP criterion 3.

- **D-16:** The reproducibility gate runs **in-process in tier2, cross-process in tier3**. The fast
  in-process comparison gives every `cargo test` a signal; the **authoritative** claim is two
  separate processes compared by hash, wired into tier3 with the other heavy gates. Follows Ph1
  D-26's split exactly. In-process alone was rejected as structurally blind: both runs share the
  thread pool, allocator state, and lazily-initialized statics, so it would pass for the wrong
  reason — which is the entire class of nondeterminism a "clean run" exists to expose.

### Carried Forward (not re-litigated)

- `aprender-core::autograd::Tensor` is the only SetFit graph (Ph1 D-01/D-03).
- `SetFitMiniLm` is the sole public encoder entry point, behind the `setfit` feature; tokenizer and
  encoder are not separately constructible (Ph1 D-05/D-07/D-08).
- `pair_cosine_mse` already exists (`crates/aprender-core/src/setfit/loss.rs`) — Phase 3 consumes
  it, it is not rebuilt.
- Default freeze policy is **all-trainable**; `FreezeGroup` addresses per-module/per-layer (Ph1
  D-20/D-22).
- `aprender-rand` (Philox, library name `trueno_rand`) keyed `(root_seed, domain)` with
  `counter = ordinal`; draw *i* is a pure function of its index (Ph2 D-20/D-21).
- `aprender-contrastive-data` is bytes-in/typed-out — no `std::fs`, no network, no path-shaped APIs
  (Ph2 D-04). Any filesystem adapter belongs to `apr-cli`.
- Pairs are **replayed, not stored**; only the pair-manifest hash persists (Ph2 D-09).
- Typed split roles + runtime validation + access ledger, all three (Ph2 D-16); compatibility
  profile emits no `Split<Validation>` so a compatibility selection run is non-constructible
  (Ph2 D-19).
- **In-band negative variants must fail their gates in every `cargo test`** (Ph1 D-24, Ph2 D-25). A
  self-reported invariant is worth exactly as much as a self-reported decreasing loss.
- `cargo-mutants` scoped to the new code (Ph1 D-25, Ph2 D-26).
- Contracts via `pv` only — never a bash/yq/python workaround. `#[contract]` annotations land with
  the code, binding through `BindingRegistry` (Ph1 D-27).
- Tier wiring: fast tests in `make tier2`, `pv validate` in tier3/tier4. A gate outside the tiers is
  a gate that stops being run (Ph1 D-26).
- `unwrap()` banned via `.clippy.toml`; `unsafe_code = "forbid"`; all fallible paths return typed
  errors.
- CPU feature matrix stays green: `--no-default-features`, `--features setfit`, all-features
  (Ph1 D-06).

### Claude's Discretion

The user selected the recommended option in all sixteen questions and delegated nothing explicitly.
The following were surfaced and consciously left to research and planning as implementation detail:

- **TRN-02's config validation surface** — one validated-at-construction config type vs a fallible
  builder, and which of the twelve knobs (encoder LR, epochs, batch size, warmup, gradient clipping,
  max length, pair policy/budget, freeze policy, head regularization, root seed, device) validate
  where. `resolve_device` already exists and fails closed; the rest is open.
- **Epoch and batch ordering across the pair stream** — Phase 2 D-14 fixed the per-epoch pair count
  and left reshuffling across epochs explicitly to this phase ("Phase 3 consumes this; Phase 2 must
  not preclude it").
- **Where fixed-order reductions live** (see D-13) — new capability in `aprender-compute`, local to
  the trainer, or a shared primitive. Needs research; this is the single largest unresolved
  implementation question in the phase.
- **Whether the f64 L-BFGS widening should land as its own contracted change** ahead of the trainer
  work, given it touches shared contracted code (D-03).
- **What "pair-loss behavior passed" means concretely** for TRN-03 — monotone decrease, endpoint
  comparison, or a contracted trend test — and which sentences the embedding-delta is measured on.
- **The `OptimizationResult` → typed-error mapping** for the head, and ordered-label-map semantics.
- **Any CLI surface for training.** Phase 2 shipped `apr data select` / `apr data pairs`; whether
  Phase 3 adds a training command or leaves that to Phase 4's OPS-03 lifecycle is unresolved.
- **Class weighting** — TweetEval abortion-stance is imbalanced; whether the head exposes weighting
  at all is undecided. Note SetFit's reference head does not use it by default.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase requirements and prior decisions
- `.planning/ROADMAP.md` § "Phase 3: Faithful Two-Stage Trainer and Head" — goal and the five
  success criteria this phase is judged against.
- `.planning/REQUIREMENTS.md` lines 55-82 (TRN-01..TRN-07) and line 157 (SAFE-03) — verbatim
  requirement text.
- `.planning/PROJECT.md` — Core Value, Constraints (algorithm fidelity, correctness, benchmark
  integrity, reproducibility), and the Context section naming "fragmented binary/multiclass
  classifier heads" as a live risk.
- `.planning/phases/01-differentiable-minilm-conformance/01-CONTEXT.md` — Phase 1 D-01..D-27. D-14
  (tolerance discipline), D-16 (dropout in fixtures), D-20/D-22 (freeze policy), D-24 (in-band
  negatives), D-26 (tier wiring), D-27 (`#[contract]`) are all load-bearing here.
- `.planning/phases/02-deterministic-pair-and-data-protocol/02-CONTEXT.md` — Phase 2 D-01..D-27.
  D-04 (bytes boundary), D-09 (replay not store), D-14 (pair count, and the explicit hand-off of
  cross-epoch reshuffling to Phase 3), D-16 (split typestate + ledger), D-19
  (`Split<CompatibilityTest>`), D-20 (Philox derivation) are load-bearing here.
- `.planning/phases/02-deterministic-pair-and-data-protocol/02-HUMAN-UAT.md` — records "Phase 3 is
  not blocked by any item in this file". The one open item is the **manual** crates.io publish
  cascade (`aprender-contrastive-data` then `apr-cli`); it constrains D-05's crate-count reasoning.
- `.planning/phases/02-deterministic-pair-and-data-protocol/deferred-items.md` — four pre-existing
  repo defects. D-ITEM-02 (25 arm64 clippy errors making `make tier2` red locally) and D-ITEM-03
  (tier2's headline `cargo test --lib` runs ZERO tests) directly affect how D-16's tier wiring can
  be verified.

### Contracts (use `pv`, never bash/yq/python)
- `contracts/setfit-encoder-conformance-v1.yaml` — Phase 1's gate incl. the frozen tolerance table;
  Phase 3's evidence thresholds must not contradict it.
- `contracts/contrastive-pair-protocol-v1.yaml` — pair semantics, budget, singleton policy, RNG
  derivation, bytes boundary. Phase 3 is its first training consumer.
- `contracts/linear-probe-classifier-v1.yaml` — existing; D-11's separate baseline type binds here.
- `contracts/tweet-eval-stance-benchmark-v1.yaml` — dataset-specific half of the data protocol.
- `contracts/classifier-pipeline-v1.yaml`, `contracts/classification-finetune-v1.yaml` — check for
  overlap before authoring a new Phase 3 contract.
- L-BFGS contract bindings: `crates/aprender-core/src/optim/tests_lbfgs_contract.rs` — D-03's f64
  widening must account for this.

### Code the phase builds on
- `crates/aprender-core/src/setfit/mod.rs` — `SetFitMiniLm`, `FreezeGroup`, `apply_freeze`,
  `trainable_parameters_mut`, `frozen_parameters`, `set_training`. The exact surface D-09/D-11 read.
- `crates/aprender-core/src/setfit/loss.rs` — `pair_cosine_mse`, the stage-one objective.
- `crates/aprender-core/src/setfit/encoder.rs` — graph-connected forward path.
- `crates/aprender-core/src/optim/lbfgs.rs` — `LBFGS::new(max_iter, tol, m)`, `minimize(f, grad, x0)`,
  `ConvergenceStatus`, `WolfeLineSearch`. D-02/D-03 target this file.
- `crates/aprender-core/src/classification/mod.rs:114` — the existing binary `LogisticRegression`
  that D-01 deliberately does not touch; read it to site the new type alongside.
- `crates/aprender-core/src/glm/mod.rs` and `glm_tests.rs:280` — the IRLS + scipy-referenced
  falsification test that D-04 cites as precedent for validating a numerical relation.
- `crates/aprender-train/src/optim/adamw.rs`, `optim/clip.rs`, `optim/scheduler/` — reused in place
  by D-05.
- `crates/aprender-train/src/train/device.rs` — `resolve_device`, already fails closed on an
  explicit unavailable CUDA request (TRN-02).
- `crates/aprender-train/src/train/trainer/core.rs` and `train_loop/basic.rs` — the existing generic
  trainer; read before deciding whether the SetFit trainer composes with it or sits beside it.
- `crates/aprender-contrastive-data/src/` — `select.rs` (`Selection`), `pairs.rs`, `split.rs`
  (typed roles), `ledger.rs` (access ledger), `manifest.rs`. Phase 3's inputs.

### Project rules
- `CLAUDE.md` — Verification Discipline (all eight rules; rules 1, 4, 5 and 7 have already produced
  findings in this milestone), Contract Validation (dogfood `pv`), Code Search Policy (`pmat query`),
  tiered quality gates.
- `.planning/codebase/ARCHITECTURE.md` — crate-boundary rules; the sentence "training policy in
  `crates/aprender-train/`" is what D-05 follows.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`optim::LBFGS`** (`aprender-core/src/optim/lbfgs.rs`): already exposes exactly the shape the
  head needs — `minimize(objective, gradient, x0) -> OptimizationResult` with Wolfe line search and
  a `ConvergenceStatus`. TRN-04's "explicit convergence or typed failure" largely falls out of it.
- **`aprender-train::optim`**: AdamW (24 KB), gradient clipping, LR schedulers — all present and
  matching SetFit's reference training recipe. This is the decisive reason D-05 chose that crate.
- **`resolve_device`** (`aprender-train/src/train/device.rs`): already fails closed on an explicitly
  requested unavailable CUDA device — TRN-02's device-validation requirement is partly satisfied.
- **`SetFitMiniLm`** surface: `trainable_parameters_mut()` / `frozen_parameters()` return
  `(String, &mut Tensor)` pairs keyed by **HF dotted names** (Ph1 D-18), which is exactly the
  granularity D-09's per-parameter evidence table needs — no name translation required.
- **Phase 2's `Selection`, typed `Split<Role>`, and access ledger**: D-08's structural exactly-once
  rule and D-14's selection lock both consume these directly.
- **`glm/` IRLS + scipy-referenced falsification test**: the working precedent for D-04's contracted
  sklearn relation.

### Established Patterns
- **Typestate over runtime checks** — Phase 2 shipped `Split<Train>` and
  `PreparedDataset<Canonical|Compatibility>` with trybuild non-constructibility tests. D-06, D-08,
  D-11 and D-14 all extend this rather than inventing a new enforcement style.
- **In-band negatives** — Phase 1's detached encoder and Phase 2's leaky/materializing samplers must
  fail their gates in every `cargo test`. D-08's pair-weighted fitter is this phase's instance.
- **Frozen tolerances committed before comparison** (Ph1 D-14) — D-10's ε follows this.
- **Summary-plus-hash instead of bulk persistence** (Ph2 D-09) — D-12 follows this.
- **Structural over asserted determinism** (Ph2 D-20) — D-13 and D-15 follow this.
- **Library crates are bytes-in/typed-out; `apr-cli` owns every filesystem adapter** (Ph2 D-04).

### Integration Points
- `aprender-train -> aprender-contrastive-data` — the single new dependency edge D-05 introduces.
  Verified acyclic: `aprender-contrastive-data` has no edge to `aprender-core`, and
  `aprender-train -> aprender-core` already exists.
- The `setfit` feature must propagate from `aprender-core` into `aprender-train`, which carries
  GPU/LoRA/distill/server modules. The CPU-only feature-matrix job is the risk surface.
- D-07's verification trait is the seam Phase 4 implements for APR; define it so the APR impl is a
  drop-in, not a rewrite.
- D-12's evidence summary is a field in Phase 4's artifact (APR-01 "training evidence") and in
  Phase 5's per-run row (EVAL-03 "encoder-update evidence"). Its schema has two downstream readers.

</code_context>

<specifics>
## Specific Ideas

- **"SetFit is the first consumer, not the owner"** — invoked explicitly for the head (D-01),
  repeating the reasoning Phase 2 used for `aprender-contrastive-data`. A capability that is
  generally useful goes in a general place with SetFit as its first caller.
- **sklearn is the reference, and the three named mismatch sources are `C` vs λ, sum vs mean, and
  intercept penalization** (D-04). These are called out by name because each is a silent failure.
- **`par_iter().sum()` is not bitwise reproducible even at a fixed thread count** — work-stealing
  changes the reduction tree. This fact is what made D-13 choose fixed-order reductions over
  thread-count pinning, and any plan that tries to satisfy TRN-06 by pinning threads is wrong.
- **A frozen probe must still be runnable** — Phase 5 needs the baseline. D-11's answer is a
  differently-named type, not a disabled code path.
- **Phase 2's open publish cascade is a live constraint on crate count**, not a hypothetical: it is
  why D-05 rejected a new crate.

</specifics>

<deferred>
## Deferred Ideas

- **Retiring the binary `LogisticRegression`** in favour of the new multiclass type (D-01 leaves it
  in place). A deliberate deprecation with a migration path — its own future ticket, not this phase.
- **Fixing the repo-wide defects in `deferred-items.md`** — vacuous CB-510 packaging guards on
  macOS (D-ITEM-01), 25 arm64 clippy errors plus the missing arm64 CI lane (D-ITEM-02), tier2's
  zero-test headline step (D-ITEM-03), and `make contract-audit` reporting 132 unbound equations
  while exiting 0 (D-ITEM-04). Each needs its own PMAT ticket. **D-ITEM-02 and D-ITEM-03 will
  affect how D-16's tier wiring can be verified locally** — the planner should expect them, not be
  surprised by them.
- **Running the crates.io publish cascade** (`aprender-contrastive-data` then `apr-cli`) — Phase 2
  UAT item 2, manual, human-run, still pending. Not Phase 3 work but it gates pre-release Gate 5.
- **Multilabel / hierarchical / token-level classification** — PROJECT.md Out of Scope for v1.
- **Accelerator paths for training** — CPU is the mandatory baseline; optional GPU is feature-gated
  and belongs with Phase 4's support matrix (SAFE-02), not here.

</deferred>

---

*Phase: 3-Faithful Two-Stage Trainer and Head*
*Context gathered: 2026-08-09*
