# Phase 3: Faithful Two-Stage Trainer and Head - Research

**Researched:** 2026-08-09
**Domain:** Two-stage SetFit training lifecycle in pure Rust — contrastive encoder tuning (AdamW), multinomial logistic head (L-BFGS), typestate lifecycle enforcement, bitwise CPU reproducibility
**Confidence:** HIGH (codebase claims verified by direct reading this session; external reference facts cited from official sources; residual assumptions logged)

## Summary

Phase 3 is unusually well-constrained: sixteen locked decisions already name the solver (existing
`optim::LBFGS`, widened to f64), the trainer home (`crates/aprender-train/src/train/setfit/`), the
head's home (`crates/aprender-core/src/classification/`), and the enforcement style (phantom
typestate with evidence gates inside transitions). This research verified every claimed code asset
by reading it, resolved the eight discretion questions with prescriptive recommendations, and
quantified the phase's single largest unresolved question — where fixed-order reductions live —
down to a concrete, testable answer.

The three headline findings the planner must not miss: **(1)** the D-13 nondeterminism hazard in
trueno's parallel GEMM is real but *narrow and quantified* — partitioning depends on
`rayon::current_num_threads()` only when the M dimension is ≤ 128 AND the matmul is above the
serial threshold; the recommendation is trainer-local sequential reductions plus an empirical
thread-count falsification gate, not a new aprender-compute capability. **(2)** D-15's keyed
dropout is NOT satisfiable by the existing `nn::Dropout` — it holds a stateful `Mutex<StdRng>`
(draw *i* depends on every prior draw, and `StdRng` is not stable across `rand` versions); the
per-site seed *plumbing* from Phase 1 exists, but the mask *source* must be replaced with a
counter-based Philox construction, requiring a new (acyclic, leaf-safe) `aprender-core ->
aprender-rand` dependency edge. **(3)** D-04's stated sklearn relation `λ = 1/(C·n)` hides a
factor-of-2 trap: sklearn's L2 penalty term is `r(W)/(S·C)` with `r(W) = ½‖W‖²_F`, so the relation
is `λ = 1/(2Cn)` if aprender's penalty is written `λ‖W‖²` — exactly the class of silent mismatch
D-04's own falsification test exists to catch; the contract equation must pin the ½ convention
explicitly.

**Primary recommendation:** Plan the f64 L-BFGS widening as its own first-wave contracted plan;
build the trainer beside (not composed with) the existing `Trainer`; put fixed-order reductions in
a small `train/setfit/reduce.rs`; gate GEMM thread-count independence empirically in tier3; add a
`WarmupLinearDecayLR` scheduler (the reference recipe decays, the repo's `LinearWarmupLR` does
not); and generate the sklearn reference fixture from the pinned Python workflow with the exact
objective convention written into the contract equation.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Classifier Head (TRN-04, TRN-05)

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
  falsification test**: `λ = 1/(C·n)`, sum-vs-mean, intercept exclusion. Rationale: a general
  `classification::` type should not inherit sklearn's idiosyncratic `C` forever, and this repo has
  the exact precedent for validating a numerical relation against a Python reference —
  `crates/aprender-core/src/glm/glm_tests.rs:280` catches a swapped IRLS link derivative against
  statsmodels/scipy. The three silent-mismatch sources (inverse scale, sum vs mean, intercept
  penalization) are named explicitly so the conversion cannot drift unnoticed.

#### Trainer Home and Lifecycle (TRN-01, TRN-02)

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
  trait**, swapping the format rather than inventing the check. Shipping the real
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

#### SetFit-Identity Gate (TRN-03, SAFE-03)

- **D-09:** Evidence covers **every trainable parameter minus the declared freeze policy**. The
  exemption set is exactly the `FreezeGroup` list — already a validated structured enum from Ph1
  D-22 where a group matching zero names is a typed error. Records **aggregate stats per named
  parameter** (gradient norm, delta norm), not raw tensors. This makes SAFE-03 **automatic**: an
  all-frozen run has an empty trainable set and therefore cannot pass, with no separate rule needed
  to catch it. Component-level aggregation was rejected because a single dead parameter inside a
  live component would be invisible.

- **D-10:** The threshold is **relative to the initial norm with a contracted ε**:
  `‖Δθ‖ / ‖θ_init‖ > ε` per parameter, alongside finite non-zero gradient norms. ε is frozen in the
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
  readable, while the detail stays auditable and the hash makes the summary non-forgeable against
  it. Directly mirrors Phase 2 D-09: replay the detail, persist the hash.

#### Determinism and Selection Lock (TRN-06, TRN-07)

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

- **D-14:** The **selection lock is hash-committing, with a typestate token**. The record commits to
  the chosen configuration, the **artifact hash**, the validation metric that selected it, and the
  selection-run hashes. Canonical test access requires a token minted **only** from a lock whose
  artifact hash **matches the model about to be evaluated** — so lock, keep tuning, then test
  **invalidates** rather than passes. An existence-only record was rejected precisely because that
  sequence would sail through it. This gives Phase 5's "any post-test-selected cell invalidates the
  report" something mechanical to check. Phase 2's access ledger remains the record of *which splits
  were touched*; it cannot alone distinguish honest selection from selection redone after a test
  peek, which is why it is a complement and not the mechanism.

- **D-15:** **Dropout stays ON during encoder tuning**, matching SetFit's reference recipe, with
  masks drawn from `aprender-rand` keyed by `(root_seed, "dropout", layer, step, block)` so mask
  element *i* is a pure function of its index. That construction is what makes dropout compatible
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

#### Carried Forward (not re-litigated)

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

### Deferred Ideas (OUT OF SCOPE)

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
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRN-01 | Typed SetFit lifecycle `Prepared -> EncoderTuned -> HeadFitted -> ArtifactReloadedAndVerified` | Typestate pattern + trybuild precedent verified in `aprender-contrastive-data/tests/ui.rs` + `ui/`; `trybuild = "1"` already a workspace dep. Lifecycle diagram and module layout below. |
| TRN-02 | Validate 12 config knobs before training begins | `resolve_device` verified fail-closed (`device.rs:110`, `CudaNotAvailable` on explicit unavailable CUDA, grammar `cpu\|auto\|cuda\|cuda:0-15`). Reference defaults for all 12 knobs pinned from SetFit v1.1.3 + HF Trainer (table below). Recommendation: one validated-at-construction `SetFitTrainConfig`. |
| TRN-03 | Evidence proof (gradients, deltas, embedding deltas, pair-loss behavior) before SetFit identity | `trainable_parameters_mut() -> Vec<(String, &mut Tensor)>` with HF dotted names verified — exactly the per-parameter granularity D-09 needs. Pair-loss criterion recommendation (deterministic endpoint comparison) below. |
| TRN-04 | Deterministic L2 multinomial softmax head, K≥2, finite outputs, explicit convergence or typed failure | `LBFGS::minimize -> OptimizationResult{status: ConvergenceStatus}` verified (6 variants). f32-hardwiring confirmed — D-03 widening is real work, mapping table `ConvergenceStatus -> typed error` below. sklearn objective formula cited from official docs. |
| TRN-05 | Head fit exactly once per unique selected row, eval/no-grad mode; pair multiplicity inexpressible | `Selection` API verified (`ordered_ids`, `examples`, `ids_in_class`, `semantic_hash`); `set_training(false)` + per-site training flags verified in encoder. Structural enforcement pattern below. |
| TRN-06 | Two clean CPU runs bitwise-reproduce IDs, ordering, step count, loss trace, hashes, predictions | GEMM thread-count-dependence hazard quantified (m ≤ 128 AND above serial threshold); no rayon anywhere in core's autograd/setfit/nn (verified). Fixed-order reduction recommendation + falsification gate design below. `OptimizationResult.elapsed_time` identified as a hash poison. |
| TRN-07 | Canonical-validation-only selection; selection-lock record before canonical test access | `Split<Validation>`/`Split<CompatibilityTest>` typestate + `AccessLedger` verified in contrastive-data; `hash::exact_hash` reusable for lock records. Hash-committing lock design is D-14 (locked). |
| SAFE-03 | Frozen probe / centroid / non-updating baselines cannot be labeled SetFit | Automatic under D-09 (empty trainable set cannot pass); `contracts/linear-probe-classifier-v1.yaml` verified present (frozen-encoder invariants already written) — D-11's baseline type binds there. |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

Directives that bind this phase's plans (treat as locked):

- **Branch protection:** `main` protected; feature branch → PR → CI (`ci / gate` + `workspace-test`).
- **Code search:** `pmat query` for intent search, never grep/glob for code search.
- **Contracts:** dogfood `pv` (in-tree: `PV_BIN := cargo run --release -p aprender-contracts-cli --bin pv --`, Makefile:1035). Never bash/yq/python workarounds. `pv diff` takes TWO FILESYSTEM PATHS (materialize old revision with `git show` first).
- **Verification Discipline (all 8 rules):** never read `$?` through a pipe; never label a run by intent — prove the mechanism engaged; pin binaries; re-mutate when extending guard scope; guards must scan the decision surface; one failing input is an anecdote; guard regexes ship a case table; check for shadowed artifacts.
- **Lints:** `unsafe_code = "forbid"`, `unwrap()` banned via `.clippy.toml` disallowed-methods (use `expect()` or `ok_or_else(...)?`), clippy pedantic.
- **Shell:** bashrs conventions (`set -euo pipefail` for scripts, option-neutral sourced libs). Note bashrs is NOT installed on this host (CI runs it).
- **Tiered gates:** `make tier1/tier2/tier3/tier4`; coverage floor 88%; complexity ≤10/fn; SATD 0; mutation ≥85%.
- **Coverage + Contracts co-evolution:** coverage without contract-density improvement is REJECTED (monorepo spec Rule 7).
- **Debug protocol:** `. scripts/apr_bin.sh` before any `apr` invocation; never a bare `apr`.
- **`rtk` hook** rewrites common commands; porcelain-emptiness assertions must run through `rtk proxy` (Phase 2 lesson).
- **Justfile preference** (user-global): prefer justfile for project scripts — but this repo's established quality-gate entry points are the Makefile tiers; follow the repo convention (Makefile) for gate wiring, per the "gate outside the tiers stops being run" rule.

## Architectural Responsibility Map

Tiers here are workspace crates (the monorepo's architectural boundaries):

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Lifecycle typestate + two-stage trainer (TRN-01/02/03/06/07) | `aprender-train` (`train/setfit/`) | — | D-05 locked; ARCHITECTURE.md: "training policy in `crates/aprender-train/`" |
| `MultinomialLogisticRegression` head (TRN-04) | `aprender-core` (`classification/`) | — | D-01 locked; dataset-agnostic capability, SetFit is first consumer |
| f64 L-BFGS widening (D-03) | `aprender-core` (`optim/`) | — | Shared contracted solver; own first-wave plan (see Open Questions) |
| Encoder forward/eval, freeze policy, dropout sites | `aprender-core` (`setfit/`, `nn/`) | `aprender-compute` (GEMM/gemv execution) | Phase 1 surface; Phase 3 modifies ONLY the dropout mask source (D-15) |
| Keyed dropout mask source (D-15) | `aprender-core` (`nn/` or `setfit/`) | `aprender-rand` (new dep edge) | Masks are drawn inside encoder forward; `aprender-rand` is a leaf crate (thiserror only — verified), edge is acyclic |
| Selection, pair replay, splits, ledger, hashing | `aprender-contrastive-data` | — | Phase 2 deliverable, consumed not modified; single new edge `aprender-train -> aprender-contrastive-data` (D-05) |
| Fixed-order reductions (D-13) | `aprender-train` (`train/setfit/reduce.rs`) | `aprender-compute` (empirical determinism gate only; one-line partitioner fix ONLY if gate is red) | Recommendation — see Open Questions #1; trainer-side scalar reductions are small (≤ thousands of elements), no performance case for parallel |
| Evidence record + selection lock + reproducibility hashes | `aprender-train` (`train/setfit/`) | — | Training-policy artifacts with two downstream readers (APR-01 field, EVAL-03 row) |
| Reload-verify trait + serde impl (D-07) | `aprender-train` (trait + serde impl) | Phase 4 supplies APR impl | Seam design: Phase 4 swaps format, not check |
| Frozen-probe baseline type (D-11/SAFE-03) | `aprender-train` (`train/setfit/` sibling module) | binds `linear-probe-classifier-v1.yaml` | Never claims SetFit; Phase 5 needs it runnable |
| CLI surface | none in Phase 3 (recommend defer to Phase 4 OPS-02/03) | `apr-cli` | See Open Questions #7 |

## Standard Stack

Everything is in-tree. **Zero new external crates are required.** Versions are workspace-pinned.

### Core (reused in place — all verified by reading this session)

| Asset | Location | Purpose | Verified Shape |
|-------|----------|---------|----------------|
| `LBFGS` | `aprender-core/src/optim/lbfgs.rs` (341 lines) | Head solver | `new(max_iter, tol, m)`, `minimize(f, grad, x0) -> OptimizationResult`; `WolfeLineSearch::new(1e-4, 0.9, 50)`; **f32-hardwired via `Vector<f32>`** [VERIFIED: read] |
| `OptimizationResult` / `ConvergenceStatus` | `aprender-core/src/optim/mod.rs:122,171` | Convergence reporting | Status variants: `Converged, MaxIterations, Stalled, NumericalError, Running, UserTerminated`; result carries `elapsed_time: Duration` (nondeterministic — exclude from hashes) [VERIFIED: read] |
| `AdamW` | `aprender-train/src/optim/adamw.rs` (671 lines) | Encoder optimizer | `new(lr, beta1, beta2, epsilon, weight_decay)`, `step_refs(&mut [&mut Tensor])`; moments as `ndarray::Array1<f32>`; already carries `provable_contracts_macros::requires`; `crate::Tensor` = re-exported `aprender-core::autograd::Tensor` (lib.rs:139) — compatible with `trainable_parameters_mut()` [VERIFIED: read] |
| `clip_grad_norm_refs` | `aprender-train/src/optim/clip.rs:65` | Global-norm clipping | `(&mut [&mut Tensor], max_norm) -> f32` (returns pre-clip norm — feed the evidence record) [VERIFIED: read] |
| `resolve_device` | `aprender-train/src/train/device.rs:110` | TRN-02 device knob | Fails closed: explicit `cuda`/`cuda:N` with no CUDA → `DeviceError::CudaNotAvailable`; `auto` falls back to CPU; grammar rejects `cuda:01`, index ≤ 15 [VERIFIED: read] |
| `SetFitMiniLm` | `aprender-core/src/setfit/mod.rs` | Encoder | `trainable_parameters_mut() -> Vec<(String, &mut Tensor)>` (HF dotted names), `frozen_parameters()`, `apply_freeze(&[FreezeGroup])` (zero-match = typed error), `set_training(bool)`, `encode_texts(&[&str])` [VERIFIED: read] |
| `pair_cosine_mse` | `aprender-core/src/setfit/loss.rs:71` | Stage-one objective | `(za, zb, labels: &[f32]) -> Result<Tensor, SetFitError>` — graph-connected [VERIFIED: read] |
| `Selection` / pair machinery | `aprender-contrastive-data/src/{select,pairs}.rs` | Phase 3 inputs | `Selection::{ordered_ids, examples, ids_in_class, semantic_hash, ledger_hash, replay}`; `PairConfig::new(root_seed)`, `resolve_budget`, capacity fns, `SamplerStateReport` [VERIFIED: read] |
| `Split<Role>` / `AccessLedger` / `hash` | `aprender-contrastive-data/src/{split,ledger,hash}.rs` | TRN-07 enforcement | `Train/Validation/Test/CompatibilityTest` roles; `AccessLedger::record`, `ledger_hash`; `hash::exact_hash(&str) -> [u8;32]`, `hex` [VERIFIED: read] |
| Philox derivation pattern | `aprender-contrastive-data/src/rng.rs` | RNG keying template | `DomainKey` via SHA-256 over `(domain_tag, root_seed, domain)`, counter = ordinal, multiply-shift bounded draws, **stateless by construction** — copy this pattern for dropout and epoch-shuffle domains [VERIFIED: read] |
| trybuild UI pattern | `aprender-contrastive-data/tests/ui.rs` + `tests/ui/` | D-06 non-constructibility | `trybuild = "1"` workspace dep already present [VERIFIED: read] |

### New code this phase writes (no new external deps)

| Component | Home | Notes |
|-----------|------|-------|
| `SetFitRun<Prepared|EncoderTuned|HeadFitted|ArtifactReloadedAndVerified>` | `aprender-train/src/train/setfit/` | Phantom typestate, transitions consume `self` |
| `SetFitTrainConfig` | `aprender-train/src/train/setfit/config.rs` | Validated at construction (recommendation below) |
| `MultinomialLogisticRegression` | `aprender-core/src/classification/multinomial.rs` (or submodule) | Softmax-NLL + L2, f64 fit, f32 storage, typed error enum |
| f64 L-BFGS path | `aprender-core/src/optim/` | See Open Questions #4 for shape recommendation |
| `WarmupLinearDecayLR` | `aprender-train/src/optim/scheduler/` | **Gap found:** repo has `LinearWarmupLR` (constant after warmup), `WarmupCosineDecayLR`, but NOT warmup + linear decay — which is the reference schedule (below) |
| Counter-based dropout mask source | `aprender-core` | Replaces `Mutex<StdRng>` mask draws on the SetFit path (Pitfall 2); needs new dep `aprender-core -> aprender-rand` (leaf, acyclic — verified) |
| Fixed-order reductions | `aprender-train/src/train/setfit/reduce.rs` | Sequential index-order loops (see Open Questions #1) |
| Evidence + lock + verify modules | `aprender-train/src/train/setfit/` | serde/serde_json already in aprender-train; **sha2 is NOT** — add workspace `sha2` dep or route hashing through `aprender-contrastive-data::hash::exact_hash` (takes `&str`; canonical-JSON-then-hash works) [VERIFIED: Cargo.toml read] |
| Frozen-probe baseline type | `aprender-train/src/train/setfit/` sibling | Distinct name, never claims SetFit; binds `linear-probe-classifier-v1.yaml` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| L-BFGS head fit (D-02, locked) | IRLS/Newton | Exact but K·d × K·d Hessian machinery; D-02 locked L-BFGS — do not relitigate |
| Composing with `train/trainer/core.rs::Trainer` | Standalone SetFit trainer beside it | `Trainer` owns `Vec<Tensor>` params + `Box<dyn Optimizer>`; SetFit needs *borrowed named* params from `SetFitMiniLm` and evidence capture between step and zero_grad. **Recommend: sit beside it**, reusing `AdamW`/`clip`/`scheduler` directly (D-05 anticipated exactly this by naming the pieces, not the `Trainer`) |
| New `sha2` dep in aprender-train | `aprender-contrastive-data::hash::exact_hash` | exact_hash takes `&str` only; hashing canonical JSON strings through it avoids a new dep line AND keeps one hashing convention; either is acceptable — pick one and state it |
| `nn::Dropout` rework in place | New counter-based mask path used only by SetFit encoder sites | Reworking `Dropout` globally touches every consumer; a SetFit-scoped mask source (site-keyed Philox) with the same `training()` flag semantics is the smaller blast radius. Phase 1's `mha_seeded_dropout_*` and `encoder_mode_dropout_*` tests pin site NAMES and mode flags — those survive; the mask VALUES change (numeric fixtures are unaffected: Ph1 D-16 generated them with dropout disabled) |

**Installation:** none. No new external packages.

## Package Legitimacy Audit

**This phase installs no new external packages.** All dependencies are workspace-internal crates
(`aprender-core`, `aprender-train`, `aprender-contrastive-data`, `aprender-rand`,
`aprender-compute`, `aprender-contracts*`) or already-present workspace deps (`trybuild = "1"`,
`serde`, `serde_json`, `proptest`, `sha2`, `thiserror`, `rayon`). The only Cargo.toml changes are
**new edges between existing in-repo crates**:

| Edge | Direction OK? | Evidence |
|------|--------------|----------|
| `aprender-train -> aprender-contrastive-data` | Acyclic [VERIFIED] | contrastive-data deps: no aprender-core edge (Ph2-verified, re-confirmed via Cargo.toml read: `aprender-rand`, sha2-family only) |
| `aprender-core -> aprender-rand` (for D-15 dropout) | Acyclic [VERIFIED] | `aprender-rand` `[dependencies]` = `thiserror = "2"` only — leaf crate |
| `sha2` into `aprender-train` (optional — see Alternatives) | Workspace dep already used by core | — |

**Packages removed due to slopcheck [SLOP] verdict:** none (no external installs).
**Packages flagged as suspicious [SUS]:** none.

slopcheck was not run — there is nothing for it to check. No `checkpoint:human-verify` gates needed
for installs.

## Architecture Patterns

### System Architecture Diagram

```
        Phase 2 (consumed, not modified)                Phase 3 (this phase)
┌─────────────────────────────────────────┐   ┌────────────────────────────────────────────────┐
│ attested bytes ──► PreparedDataset      │   │  SetFitTrainConfig (12 knobs, fail-closed      │
│        │                <Canonical>     │   │  validation at construction; resolve_device)   │
│        ▼                                │   │                    │                           │
│ Selection (typed unique IDs,            │───┼──────────────►  SetFitRun<Prepared>            │
│   semantic_hash, per-class index)       │   │   (holds SetFitMiniLm + Selection + config)    │
│        │                                │   │                    │ tune_encoder()            │
│ pair stream (REPLAYED per epoch,        │───┼───────────────►    │  per epoch e:             │
│   Philox keyed, budgeted, O(S+B))       │   │   ├ epoch order: Philox(root_seed,             │
│        │                                │   │   │   "epoch-shuffle", e) — new domain         │
│ AccessLedger + Split<Validation>        │   │   ├ fwd: SetFitMiniLm (train mode,             │
│   (canonical-only selection; ledger     │   │   │   keyed dropout ON) ── Tensor::matmul      │
│    complements, lock is the mechanism)  │   │   │   ──► trueno gemv/gemm (aprender-compute)  │
└─────────────────────────────────────────┘   │   ├ loss: pair_cosine_mse ─► backward          │
                                              │   ├ clip_grad_norm_refs(max 1.0)               │
                                              │   ├ AdamW.step_refs + WarmupLinearDecayLR      │
                                              │   ├ evidence capture: per-named-param          │
                                              │   │   grad-norm / Δ-norm (fixed-order reduce)  │
                                              │   └ loss trace: f32 bits, step-ordered hash    │
                                              │                    │                           │
                                              │   EVIDENCE GATE (inside transition):           │
                                              │   ‖Δθ‖/‖θ_init‖ > ε per trainable param,       │
                                              │   finite non-zero grads, pair-loss trend,      │
                                              │   embedding delta ── fail ─► typed error       │
                                              │   (run can NEVER claim SetFit)                 │
                                              │                    ▼                           │
                                              │        SetFitRun<EncoderTuned>                 │
                                              │                    │ fit_head(&Selection)      │
                                              │   (NO pair-stream access in signature —        │
                                              │    multiplicity inexpressible)                 │
                                              │   ├ set_training(false); pinned batches;       │
                                              │   │  each unique row encoded EXACTLY once      │
                                              │   ├ MultinomialLogisticRegression:             │
                                              │   │  f64 softmax-NLL + λ‖W‖² (intercept        │
                                              │   │  unpenalized), f64 L-BFGS,                 │
                                              │   │  ConvergenceStatus → typed error map       │
                                              │   └ store f32 W, ordered labels                │
                                              │                    ▼                           │
                                              │        SetFitRun<HeadFitted>                   │
                                              │                    │ close ► reload ►          │
                                              │                    │ re-encode ► re-predict    │
                                              │   ReloadVerify trait (serde impl now,          │
                                              │   Phase 4 swaps in APR impl)                   │
                                              │                    ▼                           │
                                              │  SetFitRun<ArtifactReloadedAndVerified>        │
                                              │        │                        │              │
                                              │  SelectionLock (hash-commits    │              │
                                              │  config + artifact hash +       │              │
                                              │  validation metric) ─► token ─► canonical      │
                                              │  test access (mismatched artifact hash         │
                                              │  INVALIDATES the token)                        │
                                              │                                                │
                                              │  Evidence summary + hash-bound JSON table      │
                                              │  ──► Phase 4 APR field / Phase 5 EVAL-03 row   │
                                              └────────────────────────────────────────────────┘
   Distinct sibling type (never claims SetFit): FrozenProbeRun / centroid baseline
   ──► binds contracts/linear-probe-classifier-v1.yaml (SAFE-03 structural)
```

### Recommended Project Structure

```
crates/aprender-train/src/train/setfit/
├── mod.rs          # SetFitRun<S> phantom typestate; transition fns; state markers
├── config.rs       # SetFitTrainConfig — 12 knobs, validated at construction (TRN-02)
├── evidence.rs     # per-parameter evidence table, summary + hash binding (D-09..D-12)
├── reduce.rs       # fixed-order reductions: sum/mean/norm in index order (D-13)
├── epoch.rs        # cross-epoch pair-order derivation (Philox "epoch-shuffle" domain)
├── head_input.rs   # Selection -> pinned-batch, encode-once embedding matrix (D-08)
├── lock.rs         # SelectionLock record + access token typestate (D-14, TRN-07)
├── verify.rs       # ReloadVerify trait + serde round-trip impl (D-07)
├── baseline.rs     # FrozenProbeRun — distinct non-SetFit type (D-11, SAFE-03)
└── negative.rs     # test-only pair-weighted fitter that must FAIL its gate (D-08)

crates/aprender-core/src/classification/
└── multinomial.rs  # MultinomialLogisticRegression + typed error enum (D-01, TRN-04)

crates/aprender-core/src/optim/
└── (f64 L-BFGS widening — see Open Questions #4 for the two shapes)

crates/aprender-train/src/optim/scheduler/
└── warmup_linear_decay.rs  # reference schedule: linear warmup → linear decay to 0
```

### Pattern 1: Evidence gate inside the typestate transition (D-11)

**What:** `tune_encoder()` computes evidence during the loop and validates it before minting the
next state. The state type is the proof-of-evidence.
**When to use:** Every transition in this lifecycle.

```rust
// Pattern — matches Phase 2's house style (Split<Role>, PreparedDataset<Profile>)
pub struct SetFitRun<S: LifecycleState> {
    encoder: SetFitMiniLm,
    selection: Selection,
    config: SetFitTrainConfig,
    evidence: S::Evidence,          // associated type: Prepared has (), EncoderTuned
    _state: PhantomData<S>,         // has UpdateEvidence — Ph2 02-03 precedent: absent
}                                    // field beats Option+expect, trybuild-provable

impl SetFitRun<Prepared> {
    pub fn tune_encoder(self) -> Result<SetFitRun<EncoderTuned>, SetFitTrainError> {
        // ... training loop, evidence capture ...
        let evidence = evidence.validate(&self.config.contracted_epsilon)?; // gate INSIDE
        Ok(SetFitRun { evidence, /* .. */ _state: PhantomData })
    }
}
```

Phase 2's `DatasetProfile`-with-associated-`Splits` decision (STATE.md 02-03) is the direct
precedent: make the evidence a field that *does not exist* in earlier states rather than an
`Option` — that is what makes the trybuild non-constructibility gate provable.

### Pattern 2: Domain-separated Philox for every new random decision (D-15, epoch order)

**What:** Copy `aprender-contrastive-data/src/rng.rs`'s construction — SHA-256 domain derivation,
counter = ordinal, multiply-shift bounded draws, stateless functions — with new domain strings.
**When to use:** dropout masks (`"dropout"` domain keyed by site/step/element), cross-epoch pair
order (`"epoch-shuffle"` keyed by epoch), any future trainer randomness.

```rust
// Source: crates/aprender-contrastive-data/src/rng.rs (verified pattern, Ph2-contracted)
// The frozen construction: key = LE-u64-truncated SHA-256(domain_tag ‖ root_seed_le ‖ domain),
// counter = [ordinal_lo, ordinal_hi, stream_id, 0]; draw i is a pure fn of (key, stream, i).
// Phase 3 needs its OWN domain tag (e.g. b"apr-setfit-train-v1\0") — do NOT reuse
// b"apr-contrastive-v1\0"; cross-phase domain collision would be a silent seed-reuse bug.
```

### Pattern 3: Fixed-order reduction (D-13)

**What:** Every trainer-side scalar reduction (batch-loss mean, gradient norms, delta norms,
evidence aggregates) is a sequential index-order loop, or fixed-size chunks combined in index
order. Never `par_iter().sum()`.

```rust
/// Order-fixed sum: identical bit pattern at any thread count, because there is
/// exactly one reduction tree. Sizes on this path are ≤ a few thousand elements
/// (batch ≤ 64 pairs; params ≤ ~200 named tensors) — parallelism buys nothing.
pub fn sum_in_index_order(xs: &[f32]) -> f64 {
    xs.iter().fold(0.0_f64, |acc, &x| acc + f64::from(x)) // accumulate in f64, one order
}
```

(Accumulating evidence norms in f64 also sidesteps f32 cancellation in ‖Δθ‖ for near-frozen
parameters — the exact regime D-10's ε discriminates.)

### Pattern 4: In-band negative (D-08)

The test-only pair-weighted fitter lives in the crate (cfg(test) or #[doc(hidden)] test-support),
runs in every `cargo test`, and its gate must observe it FAIL. Phase 2's
`tests/negative_leaky.rs` / `negative_materializing.rs` are the templates (verified present).

### Anti-Patterns to Avoid

- **Satisfying TRN-06 by pinning `RAYON_NUM_THREADS=1`:** explicitly rejected by D-13. Thread-count
  independence must be structural; the falsification gate VARIES thread count to prove it.
- **`Option<Evidence>` in a shared state struct:** Ph2 02-03 showed the absent-field form is what
  makes non-constructibility a type error instead of whole-program reasoning.
- **Hashing structs that contain wall-clock or Duration fields:** `OptimizationResult.elapsed_time`
  must never enter a semantic hash (TRN-06 poison).
- **A runtime `enum State` machine:** rejected by D-06.
- **Reusing `b"apr-contrastive-v1\0"` as the trainer's RNG domain tag:** silent seed-reuse across
  phases; mint a new tag.
- **Adding the head as a mode of the binary `LogisticRegression`:** rejected by D-01.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Quasi-Newton optimization | A new solver | `optim::LBFGS` (widened per D-03) | Contracted (`lbfgs-kernel-v1.yaml` + inline FALSIFY tests); Wolfe line search edge cases are subtle |
| Encoder optimizer | AdamW re-implementation | `aprender-train::optim::AdamW` | 671 lines, contracted (`adamw-kernel-v1` + `requires` macros), `step_refs` matches the borrowed-params surface |
| Gradient clipping | Custom norm-scaling | `clip_grad_norm_refs` | Returns pre-clip global norm — feed it straight into the evidence record |
| Device validation | String parsing | `resolve_device` | Already fail-closed with a grammar case table (`cuda:01` rejected, index ≤ 15) |
| RNG derivation | New keying scheme | `aprender-rand` Philox + rng.rs derivation pattern | The byte encoding is contracted and golden-pinned; a second scheme = a second audit surface |
| Content hashing | New digest conventions | SHA-256 via workspace `sha2`; canonical-bytes-then-hash per `contrastive-data::hash` | One convention across selection/pair/evidence/lock hashes keeps Phase 5 recomputation sane |
| Pair generation, budgets, capacities | Any pair logic | `aprender-contrastive-data::pairs` replay | Phase 2's contract; 80 KB of contract YAML already defends it |
| Non-constructibility proof | Doc claims | trybuild UI tests | Pattern shipped in `contrastive-data/tests/ui.rs` |
| Numerically stable softmax-NLL | Naive `exp/sum` | Hand-write WITH the log-sum-exp shift (this one you DO write, in f64, but follow the standard form) | TRN-04 demands finite logits/probabilities; naive softmax overflows at logit ≈ 88 in f32, ≈ 709 in f64 |

**Key insight:** the phase's risk is not missing capability — every numerical building block
exists and is contracted. The risk is *conventions drifting between components* (hash encodings,
RNG domains, regularization scaling, label order). Reuse pins conventions; rebuilding forks them.

## Common Pitfalls

### Pitfall 1: The factor of 2 in the sklearn regularization relation (D-04)
**What goes wrong:** sklearn's documented objective is `min_W  mean-weighted-NLL + r(W)/(S·C)` with
`r(W) = ½‖W‖²_F` for L2 and `S = Σ sample_weights` (= n unweighted) [CITED:
scikit-learn.org/stable/modules/linear_model.html]. So against aprender's `mean-NLL + λ‖W‖²`, the
relation is **`λ = 1/(2·C·n)`** — NOT the `λ = 1/(C·n)` written in D-04 — unless aprender's penalty
is defined as `(λ/2)‖W‖²`. D-04 names three silent-mismatch sources; the ½ inside `r(W)` is a
fourth, and it is invisible to any test that only checks "some regularization happened."
**Why it happens:** Every library picks its own ½ convention; the docs bury it inside `r(w)`.
**How to avoid:** The contracted equation must write BOTH sides fully expanded — e.g.
`mean_NLL + (1/(2·C·n))·‖W‖²_F ≡ sklearn(C)` — and the falsification test must use a λ where the
factor-2 error changes coefficients beyond tolerance (small n, moderate λ: at n=24 (8-shot K=3),
C=1 → λ = 1/48 vs 1/24 — easily distinguishable). Numeric magnitudes for this project: n ∈ {24, 48,
96, 192} for shots {8,16,32,64} × K=3.
**Warning signs:** Coefficients match at large n but drift at 8-shot; ‖W‖ systematically off by ~√2-ish factors.

### Pitfall 2: `nn::Dropout` is stateful and version-fragile — D-15 cannot reuse it as-is
**What goes wrong:** `Dropout` holds `Mutex<StdRng>` seeded via `seed_from_u64`
(`nn/dropout/mod.rs:40`, verified). Draw *i* depends on every prior draw (violates D-15's
pure-function-of-index requirement), and rand's `StdRng` is explicitly NOT portable across `rand`
versions — a dependency bump would silently change every mask [VERIFIED: read;
StdRng-instability is rand's documented policy — ASSUMED for the exact wording].
**Why it happens:** Phase 1 only needed per-site seeding for mode tests; fixtures disabled dropout
(Ph1 D-16), so the mask *values* were never load-bearing until now.
**How to avoid:** New counter-based mask source: element *i* of the mask at
`(site, step)` = Philox draw at ordinal `f(step, i)` under key `(root_seed, "dropout", site)`.
Requires: (a) new dep `aprender-core -> aprender-rand` (leaf, acyclic — verified); (b) a step
counter threaded into forward (e.g. `SetFitMiniLm::set_train_step(u64)` bumping each site's
counter base — the encoder already owns per-site state); (c) keep the existing `training()` flag
semantics so Ph1's `encoder_mode_dropout_*` tests survive. The existing FNV-1a/SplitMix64
`site_seed` (`encoder.rs:113`) is NOT the contracted SHA-256 derivation — decide deliberately
whether to migrate site keying to the rng.rs construction or contract the existing one; do not
leave two half-documented schemes.
**Warning signs:** Two clean runs match only when batch/forward call order is byte-identical;
`cargo update -p rand` changes the loss trace.

### Pitfall 3: The GEMM thread-count window (D-13's real, quantified hazard)
**What goes wrong:** `Tensor::matmul` (verified: `autograd/ops/activation.rs:274`) routes m==1 to
serial gemv, small shapes to serial `matmul_naive`, and everything ≥64-dim to
`gemm_blis_parallel`. There, serial execution is forced when `flops < 8M || (flops < 64M && n <
192)`; otherwise partition size is `ps = MC(=128)` when `m > 128` — thread-count-INDEPENDENT — but
**`ps = MR(=8).max(m / rayon::current_num_threads())` when `m ≤ 128`** (`blis/parallel.rs:137`,
`HeijunkaScheduler::default().num_threads = rayon::current_num_threads()` — verified). Partition
boundaries move with the rayon pool size in that window. Each partition runs serial `gemm_blis` on
disjoint C rows (no cross-thread FP reduction), so results differ ONLY IF edge-tile handling
(m_local % 8) changes an element's accumulation order — plausible-but-unproven either way.
**Quantified for MiniLM (hidden 384, FFN 1536):** dense 384×384: serial for m ≤ 54, hazard window
m ∈ [55, 128]; FFN 384↔1536: hazard window m ∈ [14, 128] (n=1536 ≥ 192 so the second serial clause
does not apply). m here = rows of the LHS = tokens in the flattened batch — a tail batch of 4 rows
× 20 tokens = 80 tokens lands squarely in the window.
**How to avoid:** Three-pronged (this is the D-13 recommendation, expanded in Open Questions #1):
trainer-side reductions sequential; batch shapes pinned; an empirical tier-level falsification gate
that runs identical forwards at `RAYON_NUM_THREADS=1` vs max and byte-compares. If the gate is red,
the minimal fix is one line in `gemm_blis_parallel` (`ps` for m ≤ MC becomes a constant), which the
file's own measurements support ("small problems barely benefit from parallelism") — but that is an
aprender-compute change with its own contract diligence; only take it if falsified into necessity.
**Warning signs:** tier3 cross-process hashes match on the dev box (same pool size twice) but
diverge in CI or under `--test-threads` variation.

### Pitfall 4: `OptimizationResult.elapsed_time` poisons semantic hashes
**What goes wrong:** The head's convergence record naturally wants to serialize the whole
`OptimizationResult` — which carries `elapsed_time: std::time::Duration` (verified,
`optim/mod.rs:122`). Any hash over it breaks TRN-06's two-run reproduction.
**How to avoid:** Define the head's own `HeadFitReport` (status, iterations, final grad norm,
objective) and hash THAT; never the raw result. Same rule for any chrono timestamps in aprender-train.

### Pitfall 5: Intercept gauge freedom in the multinomial comparison
**What goes wrong:** Softmax is shift-invariant per-sample; sklearn fits K (not K−1) weight rows
[CITED: sklearn user guide] and the L2 penalty pins W's gauge — but **intercepts are unpenalized**,
so the intercept vector's mean is not determined by the objective. Comparing raw intercepts to
sklearn is comparing two arbitrary gauge choices.
**Why it happens/why tests still work:** With zero-init, the NLL intercept-gradient sums to zero
across classes, so gradient-based trajectories keep `Σ_k b_k ≈ 0` in both implementations — but
that is a trajectory accident, not an objective property.
**How to avoid:** The D-04 falsification test compares (a) W within tolerance, (b) predicted
probabilities within tolerance, (c) intercepts only after centering (`b − mean(b)`); it feeds BOTH
implementations the same fixed design matrix X (do not confound with encoder differences — see
also Assumption A4 on `normalize_embeddings`).

### Pitfall 6: The repo has no warmup-then-linear-decay scheduler, and that IS the reference schedule
**What goes wrong:** SetFit v1.1.3 delegates the embedding phase to `SentenceTransformerTrainer`
(HF `transformers.Trainer` family) [VERIFIED: fetched v1.1.3 trainer.py — `self.st_trainer.train()`,
`warmup_ratio = warmup_proportion`]; HF's default `lr_scheduler_type` is `"linear"` = linear warmup
then linear decay to zero, with `max_grad_norm = 1.0`, AdamW(0.9, 0.999, 1e-8), `weight_decay = 0.0`
[CITED: huggingface.co/docs/transformers TrainingArguments]. The in-repo `LinearWarmupLR` holds the
LR CONSTANT after warmup (verified — `linear_warmup.rs`), and the other schedulers are cosine/step.
Using it silently deviates from the reference recipe — the exact PF-008 claims-defect class D-15
cited when it kept dropout ON.
**How to avoid:** Add `WarmupLinearDecayLR` (≈40 lines, mirrors `WarmupCosineDecayLR`'s structure)
and pin the default schedule to it. Declare in the contract which reference (setfit 1.1.3 →
st-trainer → HF defaults) each of the 12 knobs' defaults traces to.

### Pitfall 7: sklearn WARNS on non-convergence; TRN-04 must FAIL — a declared deviation
**What goes wrong:** sklearn's lbfgs path emits `ConvergenceWarning` at `max_iter` and still
returns coefficients; TRN-04 maps `MaxIterations` to a typed error. A reference-comparison fixture
generated from a non-converged sklearn run would compare garbage to a typed failure.
**How to avoid:** The fixture-generation workflow must assert `n_iter_ < max_iter` on the Python
side; the contract notes the deviation (aprender is strictly fail-closed where sklearn warns).

### Pitfall 8: Local tier verification is compromised by pre-existing defects (D-ITEM-02/03)
**What goes wrong:** On this arm64 host `make tier2`'s clippy step is RED with 24 pre-existing
errors in 5 untouched crates, and tier2's headline `cargo test --lib` runs ZERO tests
(deferred-items.md, verified present). D-16 wires gates into tier2/tier3 — naive "run tier2, see
green" verification is impossible locally.
**How to avoid:** Wire and verify Phase 3 gates as SCOPED commands (`cargo test -p aprender-train
--lib --features setfit`, the Phase 2 `contract-audit-phase2` precedent), and record the
D-ITEM-02/03 caveat in plan must_haves, exactly as Phase 2's plans did for Gate 5.

### Pitfall 9: `cargo package -p aprender-train` goes KNOWN-RED the moment the dep edge lands
**What goes wrong:** aprender-train is a publishable workspace crate (no `publish = false`,
version 0.63.0 — verified). Adding the path+version dep on unpublished
`aprender-contrastive-data` reproduces Phase 2's measured failure mode: manifest RESOLUTION (not
the build) rewrites the path dep to a registry dep and fails — `--no-verify` does NOT dodge it
(STATE.md, control-verified in 02-02).
**How to avoid:** Extend the known-red inventory: publish cascade order becomes
`aprender-contrastive-data` → {`apr-cli`, `aprender-train`}. Plans must record this in
`must_haves.caveats`; `/gsd:verify-work` must read the red Gate 5 as the expected state.

### Pitfall 10: A Phase 3 contract not wired into the Makefile is decoration
**What goes wrong:** `$(CONTRACTS)` is an explicit hardcoded list (Makefile:1037, per plan 02-01
D-24), and repo-wide `contract-audit` exits 0 while printing 132 binding errors (D-ITEM-04).
**How to avoid:** Follow the `PHASE2_CONTRACTS` + blocking `contract-audit-phase2` precedent
(Makefile:1086): a `PHASE3_CONTRACTS` list and a scoped blocking audit target wired into tier3.
Binding-registry traps from Phase 2 apply: `contract:` must be the BARE filename; `status:` accepts
only `implemented|partial|not_implemented|pending`.

### Pitfall 11: Feature closure across a 37-module crate
**What goes wrong:** `aprender-train` has no `setfit` feature today (verified — features list read);
it must gain `setfit = ["aprender/setfit", "dep:aprender-contrastive-data"]` and the new modules
must be `#[cfg(feature = "setfit")]`. The crate also carries GPU/LoRA/server/tui modules; Phase 1's
D-06 discipline (feature is dependency-CLOSED: enabling `setfit` alone must build) now applies to a
much bigger crate, and the CPU matrix (`--no-default-features`, `--features setfit`, all-features)
must stay green. Note `default = ["tui"]` — the `--no-default-features` leg exercises a different
module set than developers usually build.
**Warning signs:** `cargo check -p aprender-train --no-default-features --features setfit` failing
on an import from a tui/gpu module.

### Pitfall 12: Host workarounds that plans must inherit
- `cargo check/test --workspace` cannot exit 0 on Darwin — `aprender-profile` has an intentional
  `compile_error!` off-Linux; use `--exclude aprender-profile` (STATE.md, control-measured).
- `target/debug/incremental` regrew to ~25 GB twice in Phase 2 (two ENOSPC stops); adopt
  `export CARGO_INCREMENTAL=0` for Phase 3 execution (STATE.md mitigation, measured effective).
- GSD state handlers corrupted STATE.md three times in Phase 2 — run handlers, then READ THE FILE
  and repair (02-07/02-09 lessons).

## Code Examples

### The L-BFGS surface the head programs against (verified)

```rust
// crates/aprender-core/src/optim/lbfgs.rs — CURRENT (f32) shape; D-03 widens this
let mut optimizer = LBFGS::new(100, 1e-5, 10);          // max_iter, tol (grad-norm), history m
let result: OptimizationResult = optimizer.minimize(f, grad, x0);
// result.status: Converged | MaxIterations | Stalled | NumericalError | Running | UserTerminated
// result.solution: Vector<f32>; result.gradient_norm: f32; result.elapsed_time: Duration (⚠ hash poison)
// WolfeLineSearch::new(1e-4, 0.9, 50) — c1, c2, max line-search iters (private field, set in new())
```

### ConvergenceStatus → typed-error mapping (recommendation for TRN-04)

```rust
pub enum HeadFitError {
    /// L-BFGS hit max_iter with grad_norm above tol. sklearn would WARN here; we fail (declared deviation).
    NotConverged { iterations: usize, gradient_norm: f64, tol: f64 },
    /// Wolfe line search could not make progress (typically λ≈0 + separable data ⇒ ‖W‖→∞ direction).
    Stalled { iterations: usize, gradient_norm: f64 },
    /// NaN/Inf in objective or gradient — always a bug or non-finite input, never “retry”.
    NumericalError { iterations: usize },
    /// Pre-fit validation: K < 2, label out of range, non-finite feature, λ < 0, n == 0, dim mismatch.
    InvalidInput(HeadInputError),
}
// Converged  -> Ok(FittedHead)
// Running / UserTerminated are unreachable from minimize()’s return path for this call pattern —
// map to NumericalError-class internal error with a message, do not silently accept.
```

### The evidence-capture loop shape (verified interop)

```rust
// SetFitMiniLm::trainable_parameters_mut() -> Vec<(String, &mut Tensor)>   [HF dotted names]
// AdamW::step_refs(&mut [&mut Tensor]); clip_grad_norm_refs(&mut [&mut Tensor], 1.0) -> f32
let mut named: Vec<(String, &mut Tensor)> = model.trainable_parameters_mut();
let init_norms: BTreeMap<String, f64> = /* ‖θ_init‖ per name, BEFORE any step (BTreeMap: fixed iteration order for hashing — Ph2 02-03 precedent) */;
// per step: backward → capture per-name grad norms (fixed-order reduce, f64 accumulate)
//         → clip (record returned pre-clip global norm) → AdamW.step_refs → scheduler.step
// at end:  ‖Δθ‖/‖θ_init‖ per name vs contracted ε; empty trainable set ⇒ gate unpassable (SAFE-03)
```

### sklearn reference relation — the contracted equation, both conventions spelled out

```text
sklearn (docs):    min_W  (1/S)·Σ_i NLL_i  +  (1/(S·C))·(1/2)·‖W‖²_F        S = n (unweighted)
aprender (D-04):   min_W  (1/n)·Σ_i NLL_i  +  λ·‖W‖²_F
relation:          λ = 1 / (2·C·n)                      ← the ½ comes from sklearn’s r(W)
                   (equivalently: if aprender’s API used (λ/2)‖W‖², then λ = 1/(C·n))
intercepts:        excluded from the penalty on BOTH sides; compare only after mean-centering
fixture must pin:  sklearn version, C, n, K, X (fixed design matrix), max_iter, tol, and
                   assert n_iter_ < max_iter (Pitfall 7)
```

### Reference-recipe defaults for the 12 TRN-02 knobs

| Knob | Reference default | Source / note |
|------|------------------|---------------|
| encoder LR | 2e-5 | setfit 1.1.3 `body_learning_rate[0]` [CITED: HF setfit docs] |
| epochs | 1 | `num_epochs[0]` [CITED] |
| batch size | 16 | `batch_size[0]` [CITED] |
| warmup | 0.1 of total steps | `warmup_proportion` [CITED] |
| grad clipping | max_norm 1.0 | HF Trainer `max_grad_norm` [CITED; A1 assumption on non-override] |
| LR schedule | linear warmup → linear decay to 0 | HF `lr_scheduler_type="linear"` [CITED; A1] |
| AdamW | β=(0.9, 0.999), ε=1e-8, weight_decay=0.0 | HF defaults [CITED; A1] |
| max length | Phase 1's pinned tokenizer truncation | not a Phase 3 decision — reuse ENC-02 facts |
| pair policy/budget | Phase 2 `PairConfig` + `resolve_budget` | contracted, replayed |
| freeze policy | all-trainable default; `FreezeGroup` list | Ph1 D-20/D-22 (carried forward) |
| head regularization | λ native; sklearn head default C=1.0 ⇒ λ=1/(2n) | head defaults: sklearn `LogisticRegression()` — `head_params` empty dict [VERIFIED: fetched v1.1.3 modeling.py]; C/tol/max_iter numeric defaults [ASSUMED — confirm via `get_params()` in pinned env, A2] |
| root seed / device | user-supplied; `resolve_device` fail-closed | verified in-repo |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| setfit v0.x: embedding phase via `SentenceTransformer.fit()` (ST's own WarmupLinear loop) | v1.x: `Trainer` delegates to `SentenceTransformerTrainer` (HF `transformers.Trainer` family) | setfit 1.0 (2023) | The reference recipe's optimizer/scheduler/clipping facts are HF Trainer defaults; the repo pin is **setfit 1.1.3** (verified: `reference_fixtures.rs:140` asserts `setfit_version == "1.1.3"`) |
| sklearn `multi_class="multinomial"` parameter | `multi_class` deprecated; lbfgs is always multinomial for K>2 | sklearn 1.5+ (deprecation), removal later | Fixture scripts must NOT pass `multi_class`; local env has sklearn 1.9.0 (verified) — the pinned fixture env must record its exact version |
| rand `StdRng` treated as stable | rand documents StdRng as non-portable across versions | longstanding rand policy | Reinforces Pitfall 2: the SetFit path must not keep StdRng-derived masks |

**Deprecated/outdated:** nothing else on this path; all in-repo assets are current HEAD.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `SentenceTransformerTrainingArguments` does not override HF Trainer defaults for `weight_decay` (0.0), `max_grad_norm` (1.0), `lr_scheduler_type` ("linear"), AdamW betas/eps on the setfit 1.1.3 path | Pitfall 6, knob table | Wrong defaults for 4 of the 12 knobs → recipe deviation the contract would mislabel as faithful. **Cheap confirmation:** in the pinned Python fixture env, instantiate the setfit trainer and print `trainer.st_trainer.args`; record in the fixture manifest |
| A2 | sklearn `LogisticRegression()` defaults are C=1.0, tol=1e-4, max_iter=100, fit_intercept=True, class_weight=None | knob table, D-04 test design | λ conversion and convergence budget wrong. **Cheap confirmation:** `LogisticRegression().get_params()` in the pinned env, recorded in the fixture |
| A3 | GEMM edge-tile handling makes partition boundaries numerically visible (or not) — direction unknown | Pitfall 3 | If assumed-safe and wrong: TRN-06 fails in CI with different pool sizes. Resolved by the empirical falsification gate, not by more reading |
| A4 | setfit 1.1.3's `SetFitModel.normalize_embeddings` default is False (head sees unnormalized ST embeddings in the reference), while aprender's ENC-03 path always L2-normalizes | Pitfall 5 note | No Phase 3 impact (D-04's test fixes X for both sides); affects Phase 5 comparability narrative only. Confirm when generating fixtures |
| A5 | rand's StdRng non-portability across versions (exact policy wording) | Pitfall 2 | Weakens one of two reasons to replace the mask source; the stateful-draw argument stands alone regardless |
| A6 | `aprender-train` participates in the crates.io publish cascade (no `publish = false` found; workspace version) | Pitfall 9 | If it is actually never published, the Gate-5 extension is moot — the caveat costs one sentence either way |

## Open Questions (RESOLVED)

All eight discretion items, resolved to recommendations (1–3 are the load-bearing ones):

1. **Where do fixed-order reductions live? (D-13, "single largest unresolved question")**
   - What we know: no rayon anywhere in core's autograd/setfit/nn (verified); `Tensor::matmul` is
     the only parallel entry on the SetFit path; the hazard window is exactly `m ∈ [~14..128]`
     above the serial FLOP threshold (Pitfall 3); trainer-side scalar reductions don't exist yet.
   - Recommendation (prescriptive): **(a)** trainer-local `train/setfit/reduce.rs` — sequential
     index-order, f64 accumulate — for every reduction Phase 3 writes; **(b)** an empirical
     falsification gate (tier2 in-process across `RAYON_NUM_THREADS` values via subprocess env;
     tier3 cross-process) that byte-compares forward/loss outputs for shapes INSIDE the hazard
     window — this converts A3 from an assumption into a measurement; **(c)** touch
     `aprender-compute` ONLY if the gate is red, and then with the minimal one-line partitioner
     change (`ps` constant for m ≤ MC), as its own contracted change. Do NOT build a general
     "deterministic reduction" capability in aprender-compute this phase — no caller needs it yet
     and the contract surface is pure cost.
2. **TRN-02 config surface** — one `SetFitTrainConfig` validated at construction (typed error per
   knob; `resolve_device` called inside), not a fallible builder chain: a builder validates
   per-setter and misses cross-field rules (warmup fraction vs total steps; pair budget vs
   selection capacity — `unique_capacity_check` exists for exactly this). Precedent:
   `PairConfig::new` + `resolve_budget` two-stage validation.
3. **Epoch/batch ordering across epochs** — derive epoch e's pair order with the Phase 2 Philox
   construction under a NEW trainer domain (`"epoch-shuffle"`, key includes e), Fisher–Yates with
   `bounded_draw`. Batches = consecutive fixed-size windows of that order (last batch short, never
   resampled). Ph2 D-14 fixed the per-epoch count; this keys the permutation without touching the
   pair contract.
4. **f64 L-BFGS as its own contracted change first? YES — first wave.** It touches shared
   contracted code (`lbfgs-kernel-v1.yaml` + `tests_lbfgs_contract.rs`, both f32-typed today) and
   unblocks the head plan. Shape choice: `Vector<T>` is already generic, but the arithmetic impls
   are f32-only (verified: `impl Vector<f32>` at primitives/vector.rs:81) — so EITHER genericize
   `LbfgsImpl<T>` + the handful of Vector ops it needs (keeping `LBFGS` = f32 alias, zero API
   break), OR add a parallel `lbfgs_f64` module. Recommend the generic-with-alias form: no public
   surface change, one algorithm, `pv diff` sees an additive minor bump.
5. **"Pair-loss behavior passed" (TRN-03)** — on the DECLARED-deterministic loss trace (which is
   all of it, per D-13): (a) every step finite; (b) endpoint comparison: mean of last k steps <
   mean of first k steps by a contracted relative margin (k and margin frozen in the contract
   before any run, per Ph1 D-14 discipline). Reject monotone-decrease (mini-batch noise makes it
   false even for correct training) and statistical trend tests (needless on a deterministic
   trace). **Embedding delta measured on the selected training rows themselves** — already
   ledger-accessible, no new split touches; encode before tuning and after, eval mode, pinned
   batches, per-row relative delta aggregated.
6. **`OptimizationResult` → typed error and label semantics** — mapping table in Code Examples.
   Ordered labels: the head stores `Vec<String>` (index = class = W row); predictions argmax with
   lowest-index tie-break, contracted. `Estimator` (f32 `Matrix`/`Vector` surface) can be
   implemented additionally for ecosystem fit, but the typed API is primary (Estimator's
   `Result<()>` cannot carry `HeadFitError` variants losslessly).
7. **CLI surface — recommend NONE in Phase 3.** OPS-02/03 (Phase 4) own the train lifecycle CLI;
   Phase 3's outputs are library types + JSON records, and the D-04-bytes-boundary rule means a
   trainer CLI would drag filesystem adapters into scope early. Phase 2's `apr data …` commands
   already demonstrate the adapter pattern Phase 4 will follow.
8. **Class weighting — NO for v1.** The reference head is `LogisticRegression()` with
   `class_weight=None` (verified via modeling.py: `head_params` defaults empty); adding weighting
   would be an undeclared deviation AND an untested surface. TweetEval imbalance is an evaluation
   concern (F_avg, Phase 5), not a head-fitting concern. Leave the API without the knob; a v2
   addition is compatible.

**Genuinely still open after research (for the planner to schedule, not decide):**
- The outcome of the GEMM falsification gate (A3) — plan the gate early so the contingent
  aprender-compute fix has runway.
- The numeric ε for D-10 — must be frozen in the contract from a controlled instrumentation run
  BEFORE any pass/fail comparisons (the Ph1 D-14 process is the requirement; the number needs one
  measured run over the 40-cell grid extremes: 8-shot/1-epoch is the weakest-update cell).
- Whether the migrated dropout site-keying replaces or contracts the existing FNV/SplitMix
  `site_seed` (Pitfall 2, last sentence) — either is defensible; pick one in the plan.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| cargo / rustc | everything | ✓ | 1.93.0 | — |
| pmat | code search policy | ✓ | 3.15.0 | — |
| pv | contract work | ✓ (in-tree) | via `cargo run -p aprender-contracts-cli --bin pv` (Makefile `PV_BIN`) | not on PATH — always use the Makefile targets or PV_BIN form |
| cargo-mutants | Ph1 D-25 mutation gates | ✓ | installed | use `--timeout 20` per 02-08's measured hang data |
| just | user-global preference | ✓ | 1.46.0 | repo convention is Makefile tiers — follow repo |
| python3 + sklearn | D-04 reference fixture generation (pinned separate workflow per SAFE-02) | ✓ | 3.13.7 / sklearn 1.9.0 | uv 0.9.5 available for a pinned venv — fixture env MUST be pinned and recorded, not the ambient one |
| bashrs | script linting | ✗ | — | CI runs it (Linux); local scripts still must follow conventions |
| trybuild / proptest / serde / sha2 | tests, evidence records | ✓ | workspace deps | — |
| CUDA | none (CPU-only phase) | ✗ (Darwin) | — | `resolve_device("cpu")`; explicit-cuda tests assert the ERROR path, which is exactly right on this host |

**Missing dependencies with no fallback:** none that block execution.
**Host caveats (from STATE.md, all measured):** `--exclude aprender-profile` for workspace-wide
cargo on Darwin; `CARGO_INCREMENTAL=0`; tier2 RED on arm64 (D-ITEM-02); tier2 zero-test headline
(D-ITEM-03); `rtk` hook rewrites (`rtk proxy` for porcelain checks).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (libtest) + proptest 1 + trybuild 1 + cargo-mutants (scoped) |
| Config file | per-crate `Cargo.toml`; `.clippy.toml` (unwrap ban); `.pmat-gates.toml` |
| Quick run command | `cargo test -p aprender-train --lib --features setfit` and `cargo test -p aprender-core --lib --features setfit` |
| Full suite command | `cargo test --workspace --lib --exclude aprender-profile` (Darwin form) |

**Phase 2 lesson (02-06) that applies verbatim:** every contract test command must carry `--lib`
(or a concrete `--test` target) — a bare filter form emits `test result: ok` from a suite that ran
ZERO matching tests and satisfies an `expected_output` grep vacuously.

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command (quick form) | File Exists? |
|--------|----------|-----------|-------------------------------|-------------|
| TRN-01 | Illegal transitions non-constructible | trybuild + unit | `cargo test -p aprender-train --test ui --features setfit` | ❌ Wave 0 |
| TRN-02 | Each invalid knob fails before training | unit (typed-error per knob, case table) | `cargo test -p aprender-train --lib --features setfit config_` | ❌ Wave 0 |
| TRN-03 | Evidence gate passes real run, fails frozen/1e-30-LR run | unit + in-band negative | `cargo test -p aprender-train --lib --features setfit evidence_` | ❌ Wave 0 |
| TRN-04 | Head convergence/typed failure, finite outputs, sklearn relation | unit + fixture falsification (GLM precedent) | `cargo test -p aprender-core --lib multinomial_` | ❌ Wave 0 |
| TRN-05 | Pair-weighted fitter FAILS its gate; unique rows encoded once | in-band negative + unit | `cargo test -p aprender-train --lib --features setfit head_input_` | ❌ Wave 0 |
| TRN-06 | In-process two-run hash equality (tier2); cross-process (tier3) | unit + Make target spawning two processes | `cargo test -p aprender-train --lib --features setfit repro_` + tier3 target | ❌ Wave 0 |
| TRN-07 | Test access blocked without matching lock; stale-artifact-hash token invalidates | unit + trybuild (token non-constructibility) | `cargo test -p aprender-train --lib --features setfit lock_` | ❌ Wave 0 |
| SAFE-03 | All-frozen run cannot pass gate; probe type never claims SetFit | unit + contract binding | `cargo test -p aprender-train --lib --features setfit baseline_` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** scoped quick commands above (seconds each)
- **Per wave merge:** both quick commands + `cargo check -p aprender-train --no-default-features --features setfit` (feature-closure leg)
- **Phase gate:** Darwin full-suite form + tier3 (scoped contract-audit-phase3, cross-process repro gate, mutation run scoped to new code at `--timeout 20`)

### Wave 0 Gaps
- [ ] `crates/aprender-train/src/train/setfit/` module tree + `setfit` feature in aprender-train's Cargo.toml (dependency-closed)
- [ ] `crates/aprender-train/tests/ui.rs` + `tests/ui/*.rs` trybuild cases (copy contrastive-data pattern)
- [ ] `contracts/` Phase 3 contract(s) + `PHASE3_CONTRACTS` Makefile list + scoped blocking audit target
- [ ] sklearn reference fixture (pinned Python workflow; record versions, `get_params()`, `n_iter_ < max_iter`)
- [ ] f64 L-BFGS widening plan (first wave; `pv diff` on materialized old lbfgs-kernel-v1)
- [ ] GEMM thread-count falsification harness (early — its outcome steers a contingent plan)

## Security Domain

`security_enforcement` enabled (ASVS L1). This phase is a pure-Rust library phase — no network,
auth, session, or user-input surface. Applicable categories:

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | analogous only | Canonical-test access control is the DOMAIN's integrity mechanism (typestate token + hash-committing lock, D-14) — enforced by construction, not runtime ACLs |
| V5 Input Validation | yes | Fail-closed typed validation of all 12 config knobs, labels (K≥2, in-range), finite features; `deny_unknown_fields` on serde records (Ph2 precedent) |
| V6 Cryptography | yes (integrity, not secrecy) | SHA-256 (workspace `sha2`) for evidence/lock/trace hashes — never hand-rolled; Philox is contractually documented as a STATISTICAL generator, never a CSPRNG (rng.rs doc, verified) — keep that warning on any new trainer RNG module |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Forged/tampered evidence summary | Tampering | Hash-bound full table (D-12); summary is non-forgeable against the hash |
| Lock-then-tune-then-test | Repudiation/Tampering | Artifact-hash-matching token (D-14) — stale hash invalidates |
| Panic-based DoS in library code | DoS | `unwrap()` banned, `unsafe_code = "forbid"`, typed errors on all fallible paths (repo-wide, enforced) |
| Nondeterminism masking result manipulation | Repudiation | TRN-06 bitwise reproduction + cross-process tier3 gate |

## Sources

### Primary (HIGH confidence — read directly this session)
- `crates/aprender-core/src/optim/{lbfgs.rs, mod.rs, tests_lbfgs_contract.rs}` — solver shape, f32 hardwiring, contract tests
- `crates/aprender-core/src/{classification/mod.rs, traits.rs, glm/glm_tests.rs}` — binary LR, Estimator, IRLS precedent
- `crates/aprender-core/src/setfit/{mod.rs, loss.rs, encoder.rs, encoder_tests.rs}` — encoder surface, dropout sites/seeding
- `crates/aprender-core/src/nn/dropout/mod.rs` — stateful `Mutex<StdRng>` finding
- `crates/aprender-core/src/autograd/ops/activation.rs` — `Tensor::matmul` routing (+`matmul-kernel-v1` binding)
- `crates/aprender-compute/src/{matrix/ops/arithmetic.rs, blis/parallel.rs, blis/mod.rs}` — GEMM dispatch, partitioner, MR/MC, serial thresholds
- `crates/aprender-train/src/{lib.rs, optim/*, train/device.rs, train/config.rs, train/trainer/core.rs}` + Cargo.toml — Tensor re-export, AdamW/clip/schedulers, resolve_device, features
- `crates/aprender-contrastive-data/src/{select.rs, pairs.rs, split.rs, ledger.rs, rng.rs, hash.rs}` + `tests/` — Phase 3 inputs, trybuild pattern, setfit 1.1.3 pin
- `crates/aprender-rand/Cargo.toml` — leaf-crate verification
- `contracts/{lbfgs-kernel,linear-probe-classifier,classification-finetune,classifier-pipeline}-v1.yaml` — overlap check
- `Makefile` (CONTRACTS/PHASE2_CONTRACTS/PV_BIN), `.planning/STATE.md`, `deferred-items.md`
- https://raw.githubusercontent.com/huggingface/setfit/v1.1.3/src/setfit/trainer.py — st_trainer delegation, warmup wiring [VERIFIED: fetched]
- https://raw.githubusercontent.com/huggingface/setfit/v1.1.3/src/setfit/modeling.py — `LogisticRegression(**head_params)`, encode path [VERIFIED: fetched]

### Secondary (MEDIUM-HIGH — official docs)
- https://scikit-learn.org/stable/modules/linear_model.html — LogisticRegression objective (mean form, r(W)/(S·C), ½ factor, intercept exclusion, K rows) [CITED]
- https://huggingface.co/docs/setfit/reference/trainer — TrainingArguments defaults [CITED]
- https://huggingface.co/docs/transformers/.../trainer — HF TrainingArguments defaults (adamw_torch, max_grad_norm 1.0, linear scheduler) [CITED]

### Tertiary (LOW — flagged in Assumptions Log)
- A1 (ST trainer non-override), A2 (sklearn numeric defaults), A4 (normalize_embeddings default), A5 (StdRng policy wording) — each with a named cheap confirmation step

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every reused asset read this session; zero external installs
- Architecture: HIGH — pattern is Phase 2's house style, precedents verified in-tree
- Reference recipe: MEDIUM-HIGH — v1.1.3 source fetched for head + delegation; four HF-default facts carry assumption A1 with a one-command confirmation path
- Pitfalls: HIGH for in-repo findings (all measured/read); the GEMM edge-tile question (A3) is deliberately left to a falsification gate rather than asserted either way

**Research date:** 2026-08-09
**Valid until:** ~2026-09-08 (stable in-repo facts; re-verify only if `optim/`, `blis/`, or `nn/dropout` change under this phase's feet — `git log` those paths at plan time)
