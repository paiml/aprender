# Phase 3: Faithful Two-Stage Trainer and Head - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-09
**Phase:** 3-faithful-two-stage-trainer-and-head
**Areas discussed:** Head design & solver, Trainer home & lifecycle, SetFit-identity gate, Determinism contract

**Area selection:** all four offered areas were selected. Two further candidates were surfaced in
the preamble and not selected: selection-lock mechanics (later folded into Determinism as D-14) and
the Phase 3/4 seam (later folded into Trainer lifecycle as D-07).

---

## Head design & solver

### Q1 — Where does the multiclass head live, and what happens to the existing binary LogisticRegression?

| Option | Description | Selected |
|--------|-------------|----------|
| New general type in classification/ | `MultinomialLogisticRegression` in `aprender-core/src/classification/`; SetFit is first consumer not owner; existing binary type and its `apr-stochastic-lr-v1` contract untouched | ✓ |
| New head inside the setfit boundary | `MulticlassLogisticHead` in `setfit/head.rs`; zero blast radius but adds a third entry to the fragmentation PROJECT.md flags | |
| Extend the existing type in place | Widen `LogisticRegression` to K>=2; retires fragmentation but breaks a contracted public API and returns `Result<(), String>` where TRN-04 wants typed failure | |

**User's choice:** New general type in classification/ (recommended option)
**Notes:** Explicitly the Phase 2 D-01 pattern applied a second time.

### Q2 — Which solver fits the multinomial head?

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse `optim::LBFGS` | Existing `minimize(f, grad, x0) -> OptimizationResult` with Wolfe line search and `ConvergenceStatus`; head supplies softmax-NLL + L2 objective and analytic gradient; same solver family as sklearn's multinomial default | ✓ |
| Autograd full-batch (AdamW / GD) | Reuses Phase 1 Tensor ops and train's AdamW, but "converged" degrades to "ran max_iter" | |
| Newton / IRLS, GLM-style | Second-order with tight sklearn agreement; (K·D)² Hessian and new numerical code | |

**User's choice:** Reuse `optim::LBFGS` (recommended option)
**Notes:** Discovered during scouting — `aprender-core/src/optim/lbfgs.rs` already had exactly the
needed signature, which reframed this from "write a solver" to "write an objective".

### Q3 — What precision does the head fit in, given the artifact must be F32?

| Option | Description | Selected |
|--------|-------------|----------|
| Fit f64, store f32 | Widen the L-BFGS path to f64, downcast at the APR boundary; removes the conditioning question and puts the fit in the reference's precision. Costs a semver bump on a contracted optimizer | ✓ |
| Fit f32 as-is | No contracted-code changes; risk that near the optimum the gradient norm approaches f32 epsilon (~1.2e-7) and the Wolfe line search stalls while reporting progress | |
| Measure first, then decide | Run both against a frozen sklearn fixture and pick by measured agreement; honest, but defers a decision the planner would rather have locked | |

**User's choice:** Fit f64, store f32 (recommended option)
**Notes:** Fit precision and artifact precision were treated as separable; APR-01's F32 requirement
is satisfied by the downcast.

### Q4 — How should the head parameterize regularization relative to the sklearn reference?

| Option | Description | Selected |
|--------|-------------|----------|
| Native λ + contracted conversion | Public API is mean-NLL + λ‖W‖² with unpenalized intercept; the sklearn relation (λ = 1/(C·n), sum-vs-mean, intercept exclusion) becomes one contracted equation with a falsification test | ✓ |
| Mirror sklearn's objective exactly | `C·Σ NLL + ½‖W‖²` with C as the public knob; zero conversion arithmetic, but a general type inherits sklearn's idiosyncratic C forever | |
| Native λ, no sklearn coupling | Validate regularization only in the abstract; ROADMAP criterion 4's "reference-matching" loses its reference | |

**User's choice:** Native λ + contracted conversion (recommended option)
**Notes:** Precedent cited: `glm_tests.rs:280` already catches a swapped IRLS link derivative against
a scipy reference, so validating a numerical relation against Python is an established pattern here.

**Continue check:** "Next area" — remaining head details (ordered-label semantics, the
`OptimizationResult` → typed-error mapping, initialization determinism, class weighting) routed to
Claude's discretion.

---

## Trainer home & lifecycle

### Q1 — Where does the two-stage trainer live?

| Option | Description | Selected |
|--------|-------------|----------|
| `aprender-train/src/train/setfit/` | Architecture map's declared home for training policy; AdamW, clip, scheduler, `resolve_device` reusable in place; one new dep edge; `setfit` feature must propagate through a 37-module crate | ✓ |
| `aprender-core/src/setfit/trainer.rs` | One crate, one feature flag, verified acyclic — but AdamW would have to be reimplemented in core, since reaching train's would be a cycle | |
| New `aprender-setfit` crate | Cleanest boundary and AdamW for free, but a third link in the crates.io publish cascade Phase 2's open UAT item already describes | |

**User's choice:** `aprender-train/src/train/setfit/` (recommended option)
**Notes:** Decided by a measured fact — `aprender-core::optim` has L-BFGS/SGD/CG/ADMM/FISTA but no
AdamW, and `aprender-train -> aprender-core` makes borrowing train's AdamW a cycle.

### Q2 — How is Prepared -> EncoderTuned -> HeadFitted -> ArtifactReloadedAndVerified encoded?

| Option | Description | Selected |
|--------|-------------|----------|
| Phantom typestate, Phase 2 house style | `SetFitRun<Prepared>` etc., transitions consume self; inherits Phase 2's trybuild non-constructibility tests | ✓ |
| Distinct concrete types per stage | Same compile-time guarantee, better error messages, more boilerplate, diverges from the committed idiom | |
| Runtime state machine | Serializes naturally and supports resumability, but an illegal transition becomes a runtime error — weaker than the data side already delivers | |

**User's choice:** Phantom typestate, Phase 2 house style (recommended option)

### Q3 — What does Phase 3 do about ArtifactReloadedAndVerified, whose implementation is Phase 4?

| Option | Description | Selected |
|--------|-------------|----------|
| Define the boundary, test-only implementation | Phase 3 owns the marker state and the verification trait with a serde impl; Phase 4 supplies the APR impl of the same trait | ✓ |
| Marker only, unreachable in Phase 3 | Cleanest seam and zero throwaway code, but criterion 1 provable for only three of four states | |
| Ship the real APR write in Phase 3 | Criteria 1 and 2 fully provable here, but freezes the artifact schema before its consumers are designed | |

**User's choice:** Define the boundary, test-only implementation (recommended option)

### Q4 — How is "exactly once per unique selected row" enforced?

| Option | Description | Selected |
|--------|-------------|----------|
| Structural + in-band negative fitter | HeadFitted transition accepts only the `Selection` and has no access to pairs — multiplicity inexpressible; backed by a pair-weighted fitter that must fail its gate in every `cargo test`; encoding batch composition pinned | ✓ |
| Structural only | Real compiler-enforced guarantee, but the first structural claim in the milestone shipped without an induced failure proving the gate can fail | |
| Runtime dedup with assertion | Most flexible, but "cannot reweight" becomes a runtime check on a path where the wrong thing is expressible | |

**User's choice:** Structural + in-band negative fitter (recommended option)

**Continue check:** "Next area" — TRN-02's config validation surface, epoch/batch ordering,
checkpointing, and how the trainer reaches replayed pairs without an fs dependency all routed to
Claude's discretion.

---

## SetFit-identity gate

### Q1 — Which parameters must show update evidence in a production run?

| Option | Description | Selected |
|--------|-------------|----------|
| Every trainable parameter minus the freeze policy | Exemption set is exactly the `FreezeGroup` list; aggregate stats per named parameter; makes SAFE-03 automatic since an all-frozen run has an empty trainable set | ✓ |
| The four ENC-04 component classes, aggregated | Matches Phase 1's granularity and a far smaller record, but a single dead parameter inside a live component is invisible | |
| A contracted subset | Smallest record and reviewable in YAML, but the phase would be choosing which parameters it is willing to be wrong about | |

**User's choice:** Every trainable parameter minus the freeze policy (recommended option)

### Q2 — What threshold makes the update evidence pass?

| Option | Description | Selected |
|--------|-------------|----------|
| Relative to initial norm, contracted ε | `‖Δθ‖/‖θ_init‖ > ε` plus finite non-zero gradient norms; scale-free across LayerNorm gains vs embedding tables | ✓ |
| Absolute epsilon on the delta norm | Simplest to state and falsify, but one absolute floor is necessarily wrong somewhere across orders of magnitude | |
| Strictly non-zero change | Trivially checkable, but a learning rate of 1e-30 passes it | |

**User's choice:** Relative to initial norm, contracted ε (recommended option)

### Q3 — Where does the identity gate fire, and how do legitimate non-SetFit baselines still run?

| Option | Description | Selected |
|--------|-------------|----------|
| In the transition, with a separate baseline path | `tune_encoder()` fails when evidence fails, so the whole chain is gated by construction; probes/centroids run through a differently-named type that never claims SetFit | ✓ |
| Record at tuning, check at export/label | One code path for all methods and a failed run stays inspectable, but an unverified `EncoderTuned` exists and can be passed around | |
| Both | Defense in depth across the three SAFE-03 surfaces, but a redundant check that can never fire is hard to prove is not theater | |

**User's choice:** In the transition, with a separate baseline path (recommended option)
**Notes:** Gives `contracts/linear-probe-classifier-v1.yaml` a real binding, and keeps Phase 5's
baseline runnable.

### Q4 — What shape does the evidence record take?

| Option | Description | Selected |
|--------|-------------|----------|
| Compact summary + hash-bound full table | Summary in the artifact and benchmark row; full per-parameter table emitted separately as JSON and bound by hash. Mirrors Phase 2 D-09 | ✓ |
| Full per-parameter table everywhere | Maximum auditability, but inflates every APR and multiplies across 40 cells toward APR-01's oversized rejection | |
| Summary only | Smallest schema, but the central honesty claim becomes unauditable exactly when disputed | |

**User's choice:** Compact summary + hash-bound full table (recommended option)

**Continue check:** "Next area" — what "pair-loss behavior passed" means concretely, and the
embedding-delta probe set, routed to Claude's discretion.

---

## Determinism contract

### Q1 — What is bitwise reproducible across two clean CPU runs?

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed-order reductions, bitwise at any thread count | Every trainer-side reduction routed through fixed chunking with an order-fixed combine; extends D-20's structural-not-asserted principle from sampling to arithmetic | ✓ |
| Discrete facts bitwise, loss tolerance-bounded | Threading unconstrained, but predictions come from argmax over tolerance-bounded logits and need a deterministic tie-break | |
| Single-threaded reference mode | Airtight in its own scope, but the claim would not cover the configuration anyone trains in | |

**User's choice:** Fixed-order reductions, bitwise at any thread count (recommended option)
**Notes:** Framed by the fact that `par_iter().sum()` is not bitwise reproducible even at a fixed
thread count, because work-stealing changes the reduction tree.

### Q2 — What must the selection-lock record commit to, and how does it block canonical test access?

| Option | Description | Selected |
|--------|-------------|----------|
| Hash-committing lock + typestate token | Commits to config, artifact hash, validation metric and selection-run hashes; test access needs a token minted only from a matching lock, so lock-then-keep-tuning-then-test invalidates | ✓ |
| Existence-only lock record | Satisfies the literal wording, but the lock-then-keep-tuning sequence sails through it | |
| Reuse Phase 2's access ledger alone | Zero new machinery, but the ledger proves which splits were read, never which model the reading selected | |

**User's choice:** Hash-committing lock + typestate token (recommended option)

### Q3 — How does dropout behave during encoder tuning?

| Option | Description | Selected |
|--------|-------------|----------|
| On, masks Philox-derived by index | Matches SetFit's reference recipe; masks keyed `(root_seed, "dropout", layer, step, block)` so element i is a pure function of its index — replay-exact and thread-count independent | ✓ |
| Off during tuning | Determinism free and one less op, but an undeclared deviation from the reference recipe (PF-008 claims defect) unless recorded like Phase 2 D-15 | |
| Configurable, default on | Cheap ablations, but a second axis crossing 40 cells and a knob whose off-position changes what "SetFit" means | |

**User's choice:** On, masks Philox-derived by index (recommended option)
**Notes:** Closes the question Phase 1 D-16 explicitly deferred — fixtures disabled dropout for
cross-framework comparison only, leaving Rust seeded-dropout reproducibility "tested separately".

### Q4 — How is the two-clean-runs reproducibility gate actually executed?

| Option | Description | Selected |
|--------|-------------|----------|
| In-process in tier2, cross-process in tier3 | Fast signal every `cargo test`; authoritative two-process hash comparison in tier3, per Ph1 D-26's split | ✓ |
| Cross-process only | One authoritative gate, but too slow for tier2 so the pre-commit loop carries no reproducibility signal | |
| In-process only | Fast and trivial, but structurally blind to shared thread-pool, allocator and statics state — would pass for the wrong reason | |

**User's choice:** In-process in tier2, cross-process in tier3 (recommended option)

---

## Claude's Discretion

The user selected the recommended option in all sixteen questions and delegated nothing explicitly.
The following were surfaced during discussion and consciously routed to research and planning:

- TRN-02's config validation surface, and which of the twelve knobs validate where
- Epoch and batch ordering across the pair stream (Phase 2 D-14 explicitly handed this to Phase 3)
- **Where fixed-order reductions live** — the largest unresolved implementation question in the phase
- Whether the f64 L-BFGS widening should land as its own contracted change ahead of the trainer work
- What "pair-loss behavior passed" means concretely, and the embedding-delta probe set
- The `OptimizationResult` → typed-error mapping and ordered-label-map semantics
- Whether Phase 3 adds a training CLI command or leaves that to Phase 4's OPS-03
- Whether the head exposes class weighting at all, given TweetEval's imbalance

## Deferred Ideas

- Retiring the binary `LogisticRegression` in favour of the new multiclass type — its own future ticket
- The four repo-wide defects in `deferred-items.md` (vacuous CB-510 guards on macOS; 25 arm64 clippy
  errors and the missing arm64 CI lane; tier2's zero-test headline step; `make contract-audit`
  reporting 132 unbound equations while exiting 0) — each needs its own PMAT ticket, and two of them
  affect how D-16's tier wiring can be verified locally
- The manual crates.io publish cascade (`aprender-contrastive-data` then `apr-cli`) — Phase 2 UAT
  item 2, still pending, gates pre-release Gate 5
- Multilabel / hierarchical / token-level classification — PROJECT.md Out of Scope for v1
- Accelerator paths for training — CPU is the mandatory baseline; optional GPU belongs with Phase 4's
  SAFE-02 support matrix

## Notes on scope

No scope creep occurred. Two candidate areas offered in the preamble but not separately selected
(selection-lock mechanics, the Phase 3/4 seam) were folded into the discussion as D-14 and D-07
respectively, so both are captured rather than lost.
