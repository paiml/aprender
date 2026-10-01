# Roadmap: Aprender Native SetFit Classification

## Overview

This milestone delivers a native Rust SetFit lifecycle through five hard capability gates. It first
proves that the pinned MiniLM encoder is numerically conformant and genuinely differentiable, then
locks deterministic leakage-safe data and pair semantics, builds the faithful two-stage trainer,
turns its exact result into a self-contained production APR used by every Rust/CLI/HTTP surface, and
only then permits complete TweetEval comparison claims against the existing 9B LoRA baseline.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Differentiable MiniLM Conformance** - Prove the pinned encoder's shared batched Rust path matches fixtures and updates named parameters through the graph. (completed 2026-08-08)
- [ ] **Phase 2: Deterministic Pair and Data Protocol** - Make few-shot selection and bounded pair generation reproducible, provenance-complete, leakage-safe, and non-quadratic. (code-complete 2026-08-09; verification returned `human_needed` — 4 items pending in `02-HUMAN-UAT.md`)
- [x] **Phase 3: Faithful Two-Stage Trainer and Head** - Deliver an auditable encoder-tuning then unique-row classifier-fitting lifecycle that alone may identify as SetFit. (all 10 plans code-complete 2026-08-11; verification returned `human_needed` — 5/5 roadmap criteria verified, 47/50 must-have truths. **All 5 items adjudicated 2026-08-14**: `03-HUMAN-UAT.md` is `status: complete`, `03-VERIFICATION.md` reads `passed`. The 4 code-review blockers were VERIFICATION-layer weaknesses, not implementation defects — see `03-REVIEW.md`.) (completed 2026-08-14)
- [ ] **Phase 4: APR Artifact and Production Parity** - Persist, reload, inspect, predict, evaluate, and serve the exact verified model through shared CPU-first APIs. (all 22 plans code-complete 2026-08-16; verification returned `gaps_found` — 0/5 roadmap criteria fully met (2 FAILED, 3 PARTIAL), 94/95 must-have truths, and 04-11's per-crate mutation gate unmet. Five gap-closure plans 04-18..04-22 landed 2026-08-16 for the user-scoped subset that does NOT depend on F-10. **UAT ran 2026-08-16 at `b3f816c25`: 12 tests, 12 passed, 0 issues — `04-UAT.md`.** Verification reconciled to `gaps_acknowledged` with its verdict UNAMENDED. Closed since the report: BIND-004/backend_identity, WR-08, WR-09, WR-10, both D-04-11-A production survivors, F-07. **Still NOT closed:** F-10 (→ Phase 5, keeps SC1/SC3 UNMET and OPS-01/OPS-02 NOT MET), the per-crate mutation baselines + aggregate score (→ standalone compute ticket by human ruling; no mutation score exists for Phase 4), WR-01 (`fs::rename` not race-free), and SAFE-02's "in CI" clause (the 16 ci.yml setfit legs have never executed). **Blocked on `/gsd:secure-phase 04`** — security enforcement is ON and no `04-SECURITY.md` exists; `04-REVIEW.md` holds six unreferenced Warnings incl. a symlink-following non-exclusive temp file in the selection-lock write path. **This box has now been wrongly `[x]`-ed TWICE by `roadmap.update-plan-progress`, which marks a phase complete as soon as summary_count reaches plan_count — before any verifier runs and regardless of unmet must-haves. Do not re-check it until secure-phase lands and F-10 closes.**)
- [ ] **Phase 5: Benchmark and Claims Gate** - Produce the complete reproducible 40-cell SetFit evidence set and reject incomplete or unequal claims. (**AMENDED 2026-09-07, `05-CONTEXT.md` D-19**: the 9B LoRA comparison arm is descoped — aprender cannot run the Qwen3.5-9B hybrid architecture and no GPU host is reachable. The fail-closed gate is unchanged; only the declared matrix shrinks. Deferred as `D-ITEM-05-15`.)
- [ ] **Phase 6: Native Time-Series Forecasting Stack** - Ship pure-Rust Prophet, NeuralProphet and Chronos-Bolt forecasters behind stateless `forecast` MCP tools, each proven to parity with its Python original. (added 2026-09-05 from ten VALIDATED spikes; independent of Phases 1–5)

## Phase Details

### Phase 1: Differentiable MiniLM Conformance

**Goal**: Developers have one contracted, graph-connected MiniLM sentence encoder whose tokenizer,
batched forward path, train/eval behavior, gradients, and controlled updates are proven before any
SetFit trainer is exposed.
**Depends on**: Nothing (first phase)
**Requirements**: ENC-01, ENC-02, ENC-03, ENC-04, ENC-05, ENC-06
**Success Criteria** (what must be TRUE):

  1. A developer can import the pinned `all-MiniLM-L6-v2` revision and receives typed errors for any unsupported architecture, tokenizer, pooling, normalization, or configuration mutation.
  2. A developer can batch-tokenize once and run single or mixed-length padded inputs through the same `Transformer -> masked mean pooling -> L2 normalization` path, with ordered token facts and fixture-matching embeddings that are invariant to valid padding within declared tolerances.
  3. A developer can enumerate stable named parameter groups, switch the full encoder between train and deterministic evaluation behavior, and prove that registered parameters do not change merely because the mode changes.
  4. On a controlled non-degenerate pair batch, every contracted trainable embedding, attention, FFN, and normalization component receives a finite non-zero gradient and changes after one optimizer step, while frozen components remain byte-identical and sentence embeddings move in the loss-reducing direction.
  5. The cosine-similarity MSE objective remains a finite graph-connected tensor and its forward values and gradients, together with all new gather/mask/pool/normalize primitives, match frozen reference and finite-difference fixtures; deliberate detachment makes the gate fail.

**Plans**: 9 plans in 6 waves

Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Conformance contract skeleton (ten equations) + gather/mask/pool autograd ops (wave 1)
- [x] 01-02-PLAN.md — Module trait named traversal (positional-fallback default) + train/eval propagation (wave 1)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-09-PLAN.md — Attention-mask broadcast repair + exact erf GELU op (wave 2)
- [x] 01-04-PLAN.md — Fixture corpus, slice APR, SHA-256 manifest, tolerance-first contract commit (wave 2)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-03-PLAN.md — normalize/cosine/MSE ops + batched mixed-length graph spike (wave 3)
- [x] 01-05-PLAN.md — setfit feature, tokenizer boundary (SentenceBatch), typed pinned import (wave 3)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 01-06-PLAN.md — BertSentenceEncoder graph-connected forward + mode/dropout contract (wave 4)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 01-07-PLAN.md — Pair cosine-MSE loss, SetFitMiniLm bound type, freeze groups (wave 5)

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 01-08-PLAN.md — Conformance gates (all-trainable + frozen), detach-negative, mutation + tier wiring (wave 6)

### Phase 2: Deterministic Pair and Data Protocol

**Goal**: Users can prepare and replay exact few-shot training inputs and bounded contrastive pairs
without split leakage, provenance ambiguity, silent row loss, or Cartesian-product growth.
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06
**Success Criteria** (what must be TRUE):

  1. A user can acquire the pinned TweetEval abortion-stance source and produce canonical 587/66/280 train/validation/test JSONL plus exact labels, hashes, and provenance without committing tweet text; malformed, duplicate-ID, conflicting, or unknown data fails with typed errors, while cross-split duplicate *content* is excluded from the training pool and recorded in the manifest (D-27).
  2. A user can select exactly 8, 16, 32, or 64 unique canonical-training examples per class for every contracted seed and replay the same ordered selected-ID manifest and semantic hashes.
  3. A user can replay positive and negative pair manifests whose endpoints are distinct selected training IDs, whose targets agree with class identity, whose unordered identities cannot conflict, and whose singleton-class behavior is explicit and versioned.
  4. A user can impose a per-epoch pair budget, and increasing the example count under a fixed budget retains `O(examples + pair budget)` state and storage instead of materializing a Cartesian product.
  5. Validation/test endpoints, cross-split duplicate content, and the merged SetFit compatibility test used for model selection are rejected fail-closed, while the manifest proves canonical train/validation/test isolation. (Per D-27, "rejected fail-closed" means excluded from the training pool — no selection or pair may span split roles, and pool exhaustion below `shots_per_class` is a typed error; prepare-time duplicate *content* is excluded and recorded, not fatal.)

**Plans**: 9 plans in 6 waves

Plans:
**Wave 1**

- [x] 02-01-PLAN.md — Hash-attested D-06 baseline in its own PR + pv-valid tweet-eval contract + $(CONTRACTS) wiring + CLAUDE.md `pv diff` fix (wave 1)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02-02-PLAN.md — aprender-contrastive-data scaffold (3 missing workspace deps + full module skeleton + widened error enum) + contrastive-pair-protocol-v1 contract + positive-allowlist D-04 gates (wave 2)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 02-03-PLAN.md — Bytes→typed layer: schema, dual hashes, typestate splits, coalesced dedup, persistable ledger, PreparedDataset<Canonical|Compatibility> profile typestate (wave 3)
- [x] 02-04-PLAN.md — Measured + contracted SetFit pair-count fixture families (incl. K=N adversarial layout) + shared test models + integrity verifier (wave 3)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 02-05-PLAN.md — Philox few-shot selection (frozen LE encoding, labeled SelectedExample) + non-circular manifest payload with persisted ledger + strict Selection::replay + goldens (wave 4)
- [x] 02-06-PLAN.md — data_tweeteval thin-adapter relocation on the D-05 seam + dataset-attestation ingest boundary + real-duplicate golden + contract growth (wave 4)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 02-07-PLAN.md — Pair protocol: fallible capacity, binding hard cap, total degenerate policy, O(K) streaming sampler, public state diagnostics, untrusted-pair DTO, tuple-committing replay hash (wave 5)

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 02-08-PLAN.md — Honesty gates: in-band leaky/materializing negatives (incl. K=N), trybuild non-constructibility, binding-coverage audit gate, bounded scoped mutation (wave 6)
- [x] 02-09-PLAN.md — `apr data select` / `apr data pairs` CLI surface: attested ingest, required contracted seed, atomic writes, strict replay + documented workflow (wave 6)

### Phase 3: Faithful Two-Stage Trainer and Head

**Goal**: Users can reproducibly tune the encoder and then fit one stable multiclass head on each
unique tuned embedding exactly once, with SetFit identity and test access enforced by lifecycle
evidence.
**Depends on**: Phase 2
**Requirements**: TRN-01, TRN-02, TRN-03, TRN-04, TRN-05, TRN-06, TRN-07, SAFE-03
**Success Criteria** (what must be TRUE):

  1. A developer can run only the legal `Prepared -> EncoderTuned -> HeadFitted -> ArtifactReloadedAndVerified` transitions, and invalid learning, batching, warmup, clipping, length, pairing, freeze, regularization, seed, or device configuration fails before training begins.
  2. A run cannot identify itself as SetFit or export a model until named encoder gradients, parameter deltas, embedding deltas, and pair-loss behavior pass; frozen probes, centroids, and other non-updating baselines remain explicitly labeled as such.
  3. After encoder tuning, dropout is disabled and each unique selected row is encoded exactly once in evaluation/no-gradient mode before one deterministic L2-regularized multinomial softmax head is fit for any ordered `K >= 2` labels; pair multiplicity cannot reweight the head data.
  4. The shared binary/multiclass head reports explicit convergence or typed failure, finite logits and probabilities summing to one, stable ordered-label semantics, and reference-matching regularization behavior.
  5. Two clean CPU runs reproduce selected IDs, ordered pairs and batches, step count, declared loss trace, semantic hashes, and predictions, and canonical test access remains blocked until canonical-validation selection emits a selection-lock record.

**Plans**: 10 plans in 7 waves

Plans:
**Wave 1** *(**orchestrator owns the branch**: create `gsd/phase-3-two-stage-trainer` from the
current HEAD of `gsd/phase-2-contract-gate` BEFORE dispatching this wave, and record the base SHA in
the phase SUMMARY set — no plan creates it, all three CHECK and stop. Sequential execution is the
recommended default; per-executor `git worktree` if true parallelism is wanted. See each wave-1
plan's `<wave_1_concurrency>` block.)*

- [x] 03-01-PLAN.md — f64 L-BFGS widening with a frozen f32 golden trajectory and the four-channel non-finite matrix (wave 1)
- [x] 03-02-PLAN.md — Keyed Philox dropout with the forward-ordinal (branch) coordinate + fixed-pool GEMM thread-count falsification gate (wave 1)
- [x] 03-03-PLAN.md — aprender-train setfit feature + typestate skeleton + 12-knob config with validated deserialization + scheduler/reduce/epoch primitives (wave 1)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 03-04-PLAN.md — MultinomialLogisticRegression head + central-difference gradient suite + sklearn factor-of-2 falsification + multinomial-head-v1 contract (wave 2)
- [x] 03-05-PLAN.md — tune_encoder loop with a pinned step order and in-band execution digests + evidence capture + the epsilon calibration matrix (wave 2)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 03-06-PLAN.md — setfit-train-lifecycle-v1 contract (per-class epsilon + calibration regime) + armed evidence gate + in-band negatives + FrozenProbeRun (SAFE-03) (wave 3)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 03-07-PLAN.md — Encode-once head input with an encode ledger + fit_head transition (pair multiplicity inexpressible) + pair-weighted in-band negative (wave 4)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 03-08-PLAN.md — Sealed SetFitCodec seam + complete bundle + bytes-reconstruction path + ArtifactReloadedAndVerified + recorded-digest accessors (wave 5)

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 03-09-PLAN.md — Trusted validation evaluator + candidate-committing SelectionLock + object-bound CanonicalTestToken (TRN-07) (wave 6)

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 03-10-PLAN.md — trybuild non-constructibility proofs (7 cases) + cross-process two-clean-runs gate + adjusted-score mutation + full-suite closing audit (wave 7)

### Phase 4: APR Artifact and Production Parity

**Goal**: Users deploy the exact trained SetFit model as one verified offline APR whose shared core
implementation produces equivalent results through library, CLI, evaluation, and HTTP serving
surfaces on the mandatory CPU profile.
**Depends on**: Phase 3
**Requirements**: APR-01, APR-02, APR-03, APR-04, APR-05, OPS-01, OPS-02, OPS-03, OPS-04, OPS-05, OPS-06, SAFE-01, SAFE-02
**Success Criteria** (what must be TRUE):

  1. A user can save, inspect, and load one checksummed F32 `setfit-apr-v1` containing the complete encoder, exact tokenizer bytes/hash, preprocessing policy, head, ordered labels, resolved configuration, training evidence, and provenance without Python, network access, or sidecars; malformed, oversized, incomplete, inconsistent, or non-finite artifacts fail before prediction.
  2. Training closes its in-memory model, reloads the APR through the production core loader, and proves exact tokenizer/configuration/tensor state plus tolerance-bounded embeddings, logits, and probabilities and exact labels; every production consumer rejects anything short of `ArtifactReloadedAndVerified`.
  3. A Rust caller and an `apr` user can complete the CPU `train -> APR -> inspect -> eval -> predict` lifecycle through stable fallible APIs and machine-readable output, while generic APR commands auto-detect SetFit and reuse the shared core tokenizer, pooling, and model path.
  4. Rust, CLI, and native HTTP callers can classify ordered single or mixed-length batches and receive matching labels, full probabilities, optional logits, margins, token/truncation facts, latency, backend identity, and the same artifact hash in predictions and readiness.
  5. A developer can run offline executable contracts and the supported CPU build/test feature matrix to detect detached gradients, invalid data/math, leakage, label drift, artifact mismatches, and core/CLI/HTTP parity failures; an explicitly requested unavailable device fails instead of silently falling back or misreporting its backend.

**Plans**: 22 plans in 12 waves — 17 delivered, then 5 gap-closure plans added 2026-08-16 after `04-VERIFICATION.md` returned `gaps_found` (revised 2026-08-15 after cross-AI review — 04-REVIEWS.md; then 04-17 added mid-phase at a user checkpoint to land the two public-API doors — `into_artifact_bytes` and the sealed `SetFitCredential` — that waves 5-6 proved missing, which shifted the dependent plans one wave later)

**Branch base**: `gsd/phase-2-contract-gate` @ d66678e7a (Phase 3 complete + UAT + CR-01..04 fixes).
The orchestrator creates `gsd/phase-4-apr-parity` from that HEAD before dispatching wave 1; wave
merges land back on `gsd/phase-2-contract-gate` (Phase 3 precedent). No PR to `main` — human's call.

Plans:
**Wave 1**

- [x] 04-01-PLAN.md — setfit-apr-v1 contract: normative storage map (incl. head tensors), SetFitArtifactDoc field list, doc<->bundle bijection table, cap/probes/tolerances, backend-identity grammar, lock lifecycle + $(CONTRACTS)/audit wiring + CLAUDE.md realizar-first SetFit row (wave 1)

**Wave 2** *(blocked on Wave 1; three parallel plans, zero file overlap)*

- [x] 04-02-PLAN.md — Core artifact writer: canonical tensors + setfit.head.weight/bias + U8 tokenizer blob, one-key deterministic metadata, embedded synthetic probes + cross-process determinism proofs (wave 2)
- [x] 04-13-PLAN.md — SetFitBundle provenance (field 20) read off the run + schema-version bump — makes byte-canonical closure achievable (wave 2)
- [x] 04-14-PLAN.md — aprender-train public API for the CLI: SetFitTrainConfig::to_request (validated override merge) + SelectionLock::from_canonical_bytes (durable lock reconstruction) (wave 2)

**Wave 3** *(blocked on Wave 2)*

- [x] 04-03-PLAN.md — Production loader: bounded reader + fail-closed ladder + probe replay + VerifiedSetFitModel typestate + induced-corruption suite + trybuild non-constructibility (wave 3)

**Wave 4** *(blocked on Wave 3; two parallel plans, zero file overlap)*

- [x] 04-04-PLAN.md — ClassifyRequestDocument + ClassifyResponse with enforced validation (D-08) + classify + execution-derived backend identity (D-12) (wave 4)
- [x] 04-05-PLAN.md — AprCodec sealed adapter with a proven 20-field bijection + typed CodecError::Artifact + APR-03 round trip at Tolerance::EXACT (wave 4)

**Wave 5** *(blocked on Wave 4; three parallel plans, zero file overlap)*

- [x] 04-06-PLAN.md — `apr setfit train`: setfit feature, namespace, config-file-first with validated override merge, bounded artifact reader, atomic write, fail-closed device gate (wave 5)
- [x] 04-12-PLAN.md — OPS-01 public-API lifecycle proof (train -> save -> load -> embed -> classify -> inspect) + cargo-tree boundary evidence (wave 5)
- [x] 04-16-PLAN.md — reload_verified_run_from_apr: the fresh-process door to a verified run, minting only by re-entering the existing trusted policy (wave 5)

**Wave 6** *(blocked on Wave 5; two parallel plans, zero file overlap)*

- [x] 04-07-PLAN.md — Generic `apr predict` (JSON request document) + inspect APR-05 recovery + eval validation-lock artifact and gated canonical test access (TRN-07 positive, D-16) (wave 6)
- [x] 04-08-PLAN.md — Serve surface: setfit feature, AppState slot, always-installed /v1/classify with 503 handler, readiness hash, bounded startup read, oneshot tests (wave 6)

**Wave 7** *(blocked on Wave 6; two parallel plans, zero file overlap)*

- [x] 04-09-PLAN.md — Three-surface parity harness on one shared request document + frozen goldens + in-band skewed negative + ONE tier3 spawned-serve smoke with a specified port protocol (wave 7)
- [x] 04-15-PLAN.md — Spawned-binary OPS-02 lifecycle chain (train -> inspect -> validation lock -> test eval -> predict) + generic APR tooling compatibility (D-01 / A3) (wave 7)

**Wave 8** *(blocked on Wave 7)*

- [x] 04-10-PLAN.md — Make gates with one filter per invocation and ran-something guards + four-crate x three-profile SAFE-02 matrix + tier wiring + OPS-01 boundary gate (wave 8)

**Wave 9** *(blocked on Wave 8; NOT autonomous — human checkpoint)*

- [x] 04-11-PLAN.md — ci.yml extension proposed as a patch file then human-approved, per-crate mutation gate, closing requirements audit incl. TRN-07 (wave 9)

**Wave 10** *(added mid-phase at a user checkpoint; the two public-API doors waves 5-6 proved missing)*

- [x] 04-17-PLAN.md — `SetFitRun::into_artifact_bytes` (the consuming bytes door, borrowck-enforced read-then-take) + the sealed `SetFitCredential` trait, which unblocked 04-12 and 04-16 (wave 10)

**Gap closure** *(added 2026-08-16 after `04-VERIFICATION.md` returned `gaps_found`; user-scoped to six items. Does NOT close F-10 — OPS-01 and OPS-02 stay NOT MET and are Phase 5 work per the blocking note below)*

**Wave 11** *(four parallel plans, zero `files_modified` overlap)*

- [x] 04-18-PLAN.md — WR-09 + WR-08: bound `apr inspect`'s attacker-controlled metadata allocation by the shared 16 MiB cap, and make the over-cap case one typed refusal so predict/eval/inspect/serve stop contradicting each other about one file (wave 11)
- [x] 04-19-PLAN.md — `backend_identity` binding: point the registry row at the shipped `ExecutionBackend::identity`, earn the `implemented` flip with a compile-witnessed resolution guard, clear the last BIND-004 (wave 11)
- [x] 04-20-PLAN.md — WR-10: run `apr eval --lock-out`'s no-clobber gate before the dataset ingest instead of after the full multi-candidate sweep, restoring the ordering discipline `setfit_train.rs:12-20` states; end-to-end evidence via a spawned decoy case, since the dispatch tag gate makes an untagged live probe unreachable (wave 11)
- [x] 04-21-PLAN.md — the two named mutation survivors in `api/setfit_handlers.rs`, re-measured at HEAD and diagnosed from varied inputs; each kill confirmed by a `-F`-scoped cargo-mutants re-run (wave 11)

**Wave 12** *(blocked on Wave 11 — 04-21 also edits the Makefile, and the Makefile is this plan's subject)*

- [x] 04-22-PLAN.md — F-07: run bashrs for real over the Makefile and `scripts/`, make `bashrs-lint-makefile` capable of failing, wire one scoped baseline-non-increase gate founded on shell semantics (two error findings are measured bashrs false positives), and triage the repo-wide backlog with an owner — no skipped check reported as passing (wave 12)

### Phase 5: Benchmark and Claims Gate

**Goal**: Users can audit and recompute a complete, selection-safe TweetEval evidence set for the
verified SetFit APR across every contracted shot and seed, with each cell bound to a recorded
selection manifest so a second method can later be paired against it without re-running this half.
**AMENDED 2026-09-07 (`05-CONTEXT.md` D-19)** — the original goal read "a complete, selection-safe
TweetEval comparison between the verified SetFit APR and the existing 9B LoRA path". The comparison
is deferred, not cancelled (`D-ITEM-05-15`): the Qwen3.5-9B checkpoint is hybrid-attention (24
linear + 8 full layers), gated, and multimodal, and `TransformerConfig` models none of that, so the
arm was never buildable in this phase; separately, no GPU host is reachable. No report produced
under this phase may state or imply a SetFit-versus-LoRA result.
**Depends on**: Phase 4
**Blocked by a Phase 3 gate until a contract edit lands**: Phase 3's SetFit-identity gate
freezes its per-parameter update thresholds against a CALIBRATED REGIME, and by user decision
that regime contains only the fixture encoder's architecture fingerprint. A benchmark run against
the production `all-MiniLM-L6-v2` therefore returns `UncalibratedRegime` and fails closed rather
than returning a verdict. Unblocking it requires calibrating on the production encoder and adding
its fingerprint to `contracts/setfit-train-lifecycle-v1.yaml` — a deliberate, `pv diff`-flagged
contract edit per Phase 3 D-10(c), never an inline relaxation by a Phase 5 executor.
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05
**UI hint**: no
**Success Criteria** (what must be TRUE):

  1. A user can evaluate ordered predictions with the official `F_avg = (F1_against + F1_favor) / 2`, per-class metrics, three-class macro-F1, MCC, confusion matrix, and validation-only calibration diagnostics bound to explicit ordered labels.
  2. A user can run all 40 shot/seed cells for SetFit—shots `{8,16,32,64}` crossed with the ten contracted seeds—each cell recording the selection-manifest hash that would bind a second method to an identical sampled-ID set. *(AMENDED 2026-09-07, D-19: originally "for both SetFit and 9B LoRA ... for both methods in every cell". The pairing mechanism is delivered; the second arm is deferred.)*
  3. A user receives one machine-readable row per method/shot/seed run containing dataset/model revisions, selection lock, artifact hash, encoder-update evidence, backend/hardware identity, quality metrics, and consistently bounded resource measurements.
  4. A user can exactly recompute headline means, dispersion, and uncertainty from all stored rows, while any missing, selectively omitted, unmatched, or post-test-selected cell invalidates the report. *(AMENDED 2026-09-07, D-19: the "paired SetFit-versus-LoRA deltas" clause is deferred. The invalidate-on-omission behaviour is NOT weakened — `verify_run` is unchanged. Corrected 2026-09-08: two of the six doctored negatives target a second method's row and are re-sited to a deferred scope; the other four are re-mutated at the 40-cell scope rather than inherited. See `05-CONTEXT.md` D-19.)*
  5. A user can compare training time, cold/warm latency, throughput with batch/warmup boundaries, peak memory, artifact size, calibration, and classification quality measured from the same reloaded production artifacts.

**Branch base**: Phase 5 continues on `gsd/phase-2-contract-gate` per the 02-01 policy (phases
2-4 all ride this branch; no PR has been opened — opening one is the human's call).

**Plans**: 17/17 plans executed — 14/14 executed in 8 waves, plus 3 GAP-CLOSURE plans (05-15..05-17, waves 9-11) added 2026-09-11 after `05-VERIFICATION.md` graded 3/5 must-haves with EVAL-04 FAILED and EVAL-02 PARTIAL; the gap round is scoped to those two plus advisory 2's closed-form quality cross-check and the stale test-module header, and is planned against the defect CLASS the verifier named (every row-supplied value is recomputed at a location the row cannot choose) rather than against the two measured probes. Phase history below. **Replanned 2026-09-08 for D-19**: 05-11 repurposed from the LoRA dispatch to the 80→40 matrix retarget (the selection manifests it claimed to own were already shipped by 05-09's `run_bench_cells.sh::generate_selections`), 05-12 became the only cell-generation plan, 05-13 retargeted to 40 cells. Waves and `depends_on` edges unchanged. Verified by gsd-plan-checker over two revisions: 3 blockers → 0. Previously replanned 2026-08-17 after 05-01's measurements refuted 05-03's premise. At production step count (s64 = 1536 steps) five of six parameter classes have NO legal ε under the contracted 10×/10× rule (only `layer_norm_weight` survives); recorded as `05-CONTEXT.md` D-16..D-18. 05-03 was reshaped from "prepare → approve → commit" to "derive candidates → SELECT → commit" (D-04 ceremony intact) and now decides which of two already-contracted lower bounds binds. 05-14 was ADDED (wave 1, before 05-03) to make the evidence gate fail-closed on window collapse — it currently exits `rc=0` while printing `EMPTY` five times, because separation is asserted while `supports_margin` is only reported. No other plan's wave or depends_on changed; 05-14 is numbered 14 rather than inserted so the dependency graph is not renumbered. **Replanned again 2026-09-08 after `05-CONTEXT.md` D-19 descoped the 9B LoRA arm** — the declared matrix shrinks from 40 comparison cells (80 rows) to 40 SetFit rows. Only 05-11, 05-12 and 05-13 were rewritten; 05-01..05-10 and 05-14 are executed and untouched. 05-11 was REPURPOSED (LoRA dispatch retired, matrix retarget in its place) rather than vacated, so no wave and no `depends_on` edge changed and no plan is left depending on an artifact nothing produces. The fail-closed gate is not weakened: 05-10's `verify_run`, its refusal order and its six doctored negatives all survive, with four of the six re-mutated at the new 40-cell scope because the old proof does not transfer to a different expectation set.

Plans:
**Wave 1** *(the F-10 unblock work — D-01 — plus the independent numerics substrate 05-04, whose t_critical.json fixture is the source of 05-05's frozen contract literal; nothing F-10-downstream runs until the edit lands)*

- [x] 05-01-PLAN.md — Production calibration measurement: freeze E/B from pinned setfit 1.1.3, timed s8 probe, boundary matrix, ε windows + proposed regime entry (NOT autonomous: conditional >1hr compute check-in)
- [x] 05-02-PLAN.md — Per-regime Thresholds restructuring (table_for lookup, fixture semantics byte-identical, len==1 preserved)
- [x] 05-04-PLAN.md — Numerics substrate: multiclass top-label ECE + Brier (calibration-v1-bound), f64 paired-t + frozen t_{0.975,9}, scipy/sklearn fixtures in the pinned uv env
- [x] 05-14-PLAN.md — **(added 2026-08-17, D-18)** Fail-closed evidence gate: a class with no legal ε must FAIL the run, not merely print `EMPTY`. Rule-agnostic (enforces "the emitted window is non-empty for every class this regime gates", not the arithmetic), so 05-03's rule choice leaves it holding. Doubles as 05-03's own verification — RED today, GREEN once a production table lands, on identical input; the verify derives its expected status from the contract's seed list rather than hardcoding an assertion that would invert. Protects the 12 persisted evidence files by digest (12/12) and asserts no `to_canonical_bytes` surface was removed

**Wave 2** *(blocked on Wave 1 for 05-03 and for 05-05 — which copies its frozen t literal from 05-04's t_critical.json; 05-06 is an independent foundation)*

- [x] 05-03-PLAN.md — **(revised 2026-08-17; now also depends on 05-14)** Derive candidate ε bases → blocking human SELECT (six enumerated options incl. HALT) → the three-place synchronized contract/code/test edit at the D-04 checkpoint (NOT autonomous), one commit, pv diff evidence. Chooses which of two already-contracted lower bounds binds — the DERIVATION invariant's 1e-8 near-null bound (now unsatisfiable at 1536 steps) or the noise-floor clearance invariant (satisfiable for all six classes; already the operative bound for `layer_norm_weight`, whose derivation row reads `worst 1e-8 = 0.000e0`). Any multiplier on the noise floor is recorded as CHOSEN, never cited — the bound is contracted, the factor is not. "Relax the safety factors" is refuted by arithmetic (largest admissible factor ~0.31, largest product ~3.13 against a contracted 100 — an inverted margin where a near-null run would PASS). Includes the 11-site `sole()` migration 05-02 deliberately armed, with `sole()` deleted rather than re-armed. Recommends measuring `s64:31`/`s64:53` first (~5.5 h): under the near-null bound those passes provably could not change the verdict, but under the noise-floor bound they SET the worst floor and can
- [x] 05-05-PLAN.md — setfit-benchmark-claims-v1.yaml + BenchRow/RunManifest (method-tagged, deny_unknown_fields, digest-verify-before-return) + $(CONTRACTS)/audit wiring
- [x] 05-06-PLAN.md — `apr finetune --task classify --selection-manifest` + explicit seed/val_split/early-stop control + A6 probe + run_classify_core shared entry + the LoRA reload preflight (tracer: adapter save → fresh-process reload → ordered probability vector; hard gate before any 9B compute)

**Wave 3** *(blocked on Wave 2)*

- [x] 05-07-PLAN.md — Production chain proof: spawned train→inspect→eval(lock)→eval(test)→predict ladder green (04-15's rung 4 flips 6→0) + F-10 blast-radius prose re-audit
- [x] 05-08-PLAN.md — evaluate_rows_from_artifact per-row evaluator door + QualityBlock assembly (F_avg/MCC/confusion/validation-only ECE+Brier)

**Wave 4** *(blocked on Wave 3)*

- [x] 05-09-PLAN.md — `apr setfit bench run`: one cell per method, contracted resource protocol (cold/warm/throughput/peak-RSS), --record transport ingest, 40-cell driver script

**Wave 5** *(blocked on Wave 4 — the claims gate must EXIST before any expensive cell is generated; cross-AI review consensus item 4)*

- [x] 05-10-PLAN.md — bench_gate fail-closed verify (incl. recomputed lock/ledger provenance) + closed-form aggregation + `bench report` (estimation-first, mechanism-labelled) + six in-band doctored negatives + non-vacuous Make/binding gates

**Wave 6** *(blocked on Wave 5 — the declared matrix must shrink to 40 BEFORE any cell is generated, because the first `bench run` writes a run manifest derived from `RunManifest::expectation()`)*

- [x] 05-11-PLAN.md — **(REPURPOSED 2026-09-08, D-19)** Retarget the declared matrix from 80 comparison cells to 40 SetFit cells: blocking human approval of the claims-contract narrowing (`pv diff`, RETAINED-vs-NARROWED inventory), the 40-cell active expectation set with the two-method design retained as an explicitly deferred scope, all six doctored negatives preserved with four RE-MUTATED at the new scope, an out-of-scope-row refusal, single-method uncertainty on the frozen t constant, the comparison-free renderer, and a single-cell verification door. The LoRA GPU-dispatch arm this number used to carry is RETIRED in-band with its evidence (`D-ITEM-05-15`); `scripts/dispatch-bench-lora.sh` is not written. The number is repurposed rather than vacated so 05-12 and 05-13 keep pointing at a plan that produces what they consume — the dependency graph is unchanged

**Wave 7** *(blocked on Wave 6 — a cell generated before the retarget would declare 80 expected cells, 40 of which can never exist)*

- [x] 05-12-PLAN.md — **(now the phase's ONLY cell-generation plan)** The 40 SetFit CPU cells from reloaded production artifacts (NOT autonomous: the compute is explicitly NOT pre-authorized — 05-03's checkpoint deferred the question to wave 7 — so the projection is written down as a number and approved first; sequential by design, parallel cells would invalidate every EVAL-05 resource number). Selection manifests come from `scripts/run_bench_cells.sh::generate_selections`, shipped by 05-09, so the retired plan's ownership claim was already stale. The pilot-cell gate proof is REPLACED by a two-half proof — the single-cell verification door against the real row, plus `bench report` on a copy refusing at a missing cell — because `verify_run` refuses at completeness before it reads any row, so the copy-and-refuse half alone never touches the pilot row's bytes. 40/40 manifest closure + a 40-row peak-RSS mechanism tally (sampled train peak vs child-measured inference peak, declared not averaged)

**Wave 8** *(blocked on Wave 7)*

- [x] 05-13-PLAN.md — The 40-row single-method report (exact-recompute demonstrated bit-for-bit) with the paired-delta section removed and 05-04's paired-t machinery retained-but-unexercised, three layered controls against implying a second-method result, + D-11 qa refusal message + a closing evidence map that marks EVAL-02/04/05 as met by a NARROWER deliverable under the 2026-09-07 amendment

**Gap closure** *(added 2026-09-11 after `05-VERIFICATION.md` returned `gaps_found` at 3/5 must-haves — EVAL-04 FAILED (gap 1: row-controlled provenance paths escape `bench_dir`) and EVAL-02 PARTIAL (gap 2: the selection-manifest hash is recorded but never recomputed). User-scoped to four items: the two graded gaps, the closed-form quality cross-check from advisory 2, and the stale `bench_gate_tests.rs` doc header. CR-01, advisory 3, the other 15 WR / 4 IN review findings and `D-ITEM-05-15` are explicitly OUT and are recorded as visible deferrals in each plan. Sequential waves — all three touch `bench_gate.rs`, so zero same-wave file overlap is only achievable by ordering. No plan re-runs any of the 40 cells; every plan is autonomous.)*

**Wave 9**

- [x] 05-15-PLAN.md — Gap 1 (EVAL-04): enumerate the gate's WHOLE row-supplied input surface by class (which found a third unvalidated field — `contract_id` is read, carried and never compared), then resolve every row-supplied path through one validating door (syntactic refusal of absolute/`..`/rooted/empty, then component-wise canonical containment), proven RED-before/GREEN-after at the 40-cell scope and end-to-end through `apr setfit bench report` on the tree that produced spot-check E. Corrects WR-06's error typing as an unavoidable consequence (option (a), decided in-plan), sweeps path shape x attacked field in one case table across `EvidenceKind::Lock` and `::Ledger` so the deferred arm inherits the fix, and replaces the stale eighty-cell test-module header

**Wave 10**

- [x] 05-16-PLAN.md — Gap 2 (EVAL-02): bind the selection manifest. Resolve `selections/s{shots}-seed{seed}/selection-manifest.json` at a path derived from the CELL KEY (never a row field), recompute its `semantic_hash` through the existing `SelectionManifest::from_bytes` door, and refuse on hash disagreement AND on a manifest whose own `shots_per_class`/`root_seed` disagree with the cell — so a transplanted manifest fails even with the row hash doctored to match. Three RED-turning negatives as one swept table plus the shot-axis boundary transplants, the spot-check F and G replays through the shipped door, and an ACTIVE `selection_binding_rule` contract equation with `OBLIG-CLAIMS-SELECTION-BOUND` / `FALSIFY-CLAIMS-011`. Single-method and intra-cell: `D-ITEM-05-15`'s cross-method shapes are neither constructed nor exercised

**Wave 11**

- [x] 05-17-PLAN.md — Advisory 2 hardening (EVAL-01): recompute `f_avg`, `macro_f1`, `mcc`, the per-class vectors, `n_test_rows` and every `*_bits` sibling from the row's OWN `confusion_matrix` + `ordered_labels`, routed through the same shipped surfaces `assemble_quality_block` uses (OPS-03 — no second implementation), refusing spot-check D. The acceptance band is measured over all 40 committed rows BEFORE it is chosen rather than fitted to make the tree pass. Makes the synthetic fixtures internally consistent with their own matrices first, then corrects what the report, the gate header and the contract each concede — the residual that remains is a doctored confusion matrix and the two non-recomputable calibration diagnostics, and nothing more

### Phase 6: Native Time-Series Forecasting Stack

**Goal**: Users can forecast a time series in one stateless MCP call — `ds[]`, `y[]`, horizon in;
forecast with bands and components out — from pure-Rust Prophet and NeuralProphet ports and an
embedded zero-shot Chronos-Bolt, each proven to parity with its Python original and served the
way SetFit is served (thin pmcp servers, stdio + streamable-HTTP, Lambda-shaped).
**Depends on**: none of Phases 1–5 functionally — an independent track that reuses the Phase 4
thin-MCP pattern (`crates/aprender-mcp-setfit`, `crates/aprender-mcp-setfit-lambda`) and may run
alongside Phase 5's remaining GPU waves.
**Requirements**: TBD
**Requirements note**: forecasting has no REQ-IDs in `.planning/REQUIREMENTS.md` (that document is
the SetFit milestone's). The binding requirements are the five `prophet-forecast-mcp` decisions in
`.planning/spikes/MANIFEST.md`, transcribed as D-01..D-05 in `06-CONTEXT.md`, plus the Success
Criteria below.
**UI hint**: no (the demo page is a static, spike-proven MCP client copied into each server crate;
not a product UI)
**Spike evidence**: ten VALIDATED spikes (001–010), packaged as `Skill("spike-findings-aprender")`
in `.claude/skills/spike-findings-aprender/`; raw experiments, oracle fixtures and run outputs in
`.planning/spikes/`.
**Success Criteria** (what must be TRUE):

  1. A user can call one stateless `forecast` tool on `aprender-mcp-forecast` (stdio and streamable-HTTP) with `ds`, `y`, `horizon`, `freq` and `model: prophet | neuralprophet`, and receive `yhat`, `yhat_lower`, `yhat_upper`, `trend`, named components, timing and a diagnostics object in under 2 s for a 3 000-point daily series; every malformed input (unknown field, fewer than 10 or more than 20 000 points, unsorted, duplicate or impossible dates, constant `y`, horizon 0 or above 3 650, unknown `freq`, logistic growth without a valid `cap`) is a validation error, never a silent default.
  2. The Prophet port in `aprender-forecast` reproduces Python Prophet 1.4.0 on the committed Peyton Manning, air passengers, retail sales and `wp_log_R` fixtures as tests that run in CI: data preparation 0.0 diff, objective at Python's MAP within 1e-9, Python's parameters through the Rust predict path within 1e-10, fitted objective no worse than Python's + 0.5, future forecast inside Prophet's own Newton-vs-L-BFGS band, components reconstructing `yhat`, and 80 % band widths within 2 % of Python's.
  3. The NeuralProphet port reaches 365-day-ahead holdout MAE ≤ 0.47 on Peyton with the lag-free model (Python NeuralProphet 0.9.0: 0.461) and beats the naive one-step baseline with `n_lags = 30`, using a graph-connected Huber loss and data preparation that matches the committed NeuralProphet oracle fixture.
  4. A user can call the same `forecast` shape on `aprender-mcp-chronos` with Chronos-Bolt-tiny f16 weights embedded in the binary: the nine native quantiles match the Python `chronos-forecasting` 2.3.1 oracle within 2 % of the series std through the server (1e-6 absolute with f32 weights), a 2 048-point context forecasts in under 100 ms, `horizon > 64` is refused unless `allow_long_horizon: true` and then carries a warning, the release binary is under 30 MB, and cold start to first forecast over stdio is under 150 ms.
  5. Eight concurrent requests to the Prophet/NeuralProphet streamable-HTTP server return responses bit-identical to their sequential results in less than half the sequential wall time (router pool), and every gate is green: both servers' e2e tests, the workspace lib tests, `cargo clippy -- -D warnings` on the new crates, `cargo fmt --all -- --check`, and `pv validate` on every new contract.

**Branch base**: continues on `gsd/phase-2-contract-gate` per the 02-01 policy. The NEON GEMM
kernel (spike 008) is already in this tree and on `perf/neon-gemm-8x6-microkernel` in the
`aprender-neon-upstream` worktree; opening that upstream PR is a human checkpoint, not a Phase 6
task.

**Plans**: 17/17 plans executed — 9/9 in waves 1-7, plus 4 GAP-CLOSURE plans executed in waves 8-11 (06-10..06-13, planned 2026-09-06 against `06-VERIFICATION.md`'s three measured SC1 gaps), plus 4 ROUND-3 GAP-CLOSURE plans planned 2026-09-06 in waves 12-15 (06-14..06-17, against `06-REVIEW.md`; see the two "Gap closure" blocks in the plan list below). (Planned 2026-09-05, revised twice after plan-check, then revised again 2026-09-05 after cross-AI review — 06-06 declares its `crate::chronos` dependency on 06-05 and moves to wave 3, and the root-manifest profile decision moves into 06-01 so no wave-2 plan edits the `Cargo.toml` its siblings read; tracer-first — 06-01 proves one Prophet path end-to-end before any expansion; three `autonomous: false` plans carry the phase's human decisions: the FALSIFY-MONO-011 `[[bin]]` allowlist treatment (06-02, wave 1), the D-13 memory-clause amendment that D-14's `dot8` single-row routing forces (06-05, wave 3 — D-13 and D-14 cannot both hold as written, and a locked CONTEXT decision is not the planner's to rewrite alone; recorded as `D-ITEM-06-08` and grep-gated in 06-09) and the `.github/workflows/ci.yml` embedded-weights leg (06-08, wave 6). Four contracts, not three: `forecast-tool-boundary`, `prophet-parity`, `neuralprophet-parity`, `chronos-bolt-parity` — NeuralProphet's SC3 bars get their own file rather than riding in the Prophet contract. Fixtures are copied in-crate; Lambda wrappers deferred; `SmoothL1Loss` filed as a core ticket.

**Cross-AI review pass (codex + gemini, 06-REVIEWS.md at plans commit `88da44dcd`)**: six findings incorporated — the NeuralProphet port moved out of the tracer into 06-04 (the one structural change; 06-06 gains it as a dependency); the Chronos "only transposed weights" truth restated to match the code being ported, with a test that proves single rows actually reach `dot8`; an architecture-keyed provisional f32 bar plus a `measure-x86-first` checkpoint option, because every CI job is x86_64 and the 1e-6 bar was measured on aarch64 with 4.6 % margin; `just fetch-chronos-tiny` made verify-always with a tamper control, and `chronos-gate` now calls it unconditionally; the pool speed-up assertion moved out of the unit test into the host-gated `forecast-pool-ratio` benchmark; Wave 2 sequential execution made an enforced precondition. Two of the reviewers' most emphatic findings — `[profile.dev.package.X]` not reaching test builds, and `cargo test -- a b` dropping the second filter — were REFUTED by experiment and are recorded as rejected in the affected plans so they cannot be re-raised as new.)

Plans:
**Wave 1** *(no file overlap; run one at a time on this shared branch — export `CARGO_INCREMENTAL=0`, the disk is at 94 %)*

- [x] 06-01-PLAN.md — TRACER (T1): `aprender-forecast` (dates with a strict `parse_date`, types, fit, forecast door, Prophet port) + `aprender-mcp-forecast` (pmcp server, stdio + streamable-HTTP, one e2e happy path) + root members; T2: 17 fixtures copied byte-verified, Peyton rung-2 parity test, both crate READMEs, the MEASURED `[profile.dev.package.aprender-forecast]` decision (taken in wave 1 so wave 2 reads a stable root manifest). `model: neuralprophet` is a declared refusing stub filled by 06-04 — the np.rs port left the tracer in the `--reviews` replan so the tracer is one honest end-to-end slice (wave 1)
- [x] 06-02-PLAN.md — HUMAN DECISION (blocking): FALSIFY-MONO-011 `[[bin]]` allowlist treatment for the six thin MCP deployment units, applied with a two-sided control; the three README drift fixes (two missing READMEs, the setfit monorepo link) (wave 1, NOT autonomous)

**Wave 2** *(blocked on 06-01; two plans with zero `files_modified` overlap that MUST be RUN ONE AT A TIME, not concurrently — they share one Cargo target directory, one `target/.package-cache` build lock and one disk at 94 % after three ENOSPC halts; each carries a Task-1 precondition that halts if another cargo build is running. Sequential execution is the instruction, not a recommendation — both cross-AI reviewers, 2026-09-05. 06-05 left this wave entirely: it is Wave 3 on a declared 06-04 dependency, because both plans append to the same `pub mod` block in `crates/aprender-forecast/src/lib.rs`)*

- [x] 06-03-PLAN.md — `contracts/prophet-parity-v1.yaml` + the full seven-fixture Prophet ladder as `--lib` tests reading the contract + the warm debug ladder wall recorded under the profile 06-01 decided — no manifest edit (wave 2)
- [x] 06-04-PLAN.md — the np.rs port + the real `model: neuralprophet` arm and its refusals (moved here from 06-01) + `contracts/neuralprophet-parity-v1.yaml` + NP parity (oracle data prep, lag-free MAE ≤ 0.47, AR-Net beats naive, Huber connectivity, mini-batch/tape invariants) (wave 2)

**Wave 3** *(blocked on 06-01 and 06-04 — 06-05 appends `bolt`/`safetensors`/`chronos` to the same `pub mod` block in `crates/aprender-forecast/src/lib.rs` that 06-04 adds `np` to, and a shared file is a real dependency, not a scheduling preference; NOT autonomous — human checkpoint on the D-13 memory clause)*

- [x] 06-05-PLAN.md — `just fetch-chronos-tiny` (pinned revision + sha256 verified on EVERY run, with a tamper control; f16 derived and hashed) + `build.rs` cfg(chronos_weights) + bolt/safetensors/chronos-door ports (both weight layouts, with the single-row `dot8` routing proven) + `contracts/chronos-bolt-parity-v1.yaml` (arch-keyed f32 bar) + gated Bolt ladder with the two-sided counted-skip proof (wave 3, NOT autonomous: a blocking human decision ratifies the D-13 memory-clause amendment before `bolt.rs` is ported)

**Wave 4** *(blocked on 06-01, 06-04 and 06-05 — `types::tests::chronos_bounds_match_contract` reads `crate::chronos`, and the NP happy-path e2e needs the `neuralprophet` arm 06-04 now lands)*

- [x] 06-06-PLAN.md — `contracts/forecast-tool-boundary-v1.yaml` + the complete D-11 refusal e2e set + strict schema + router pool (`pooled_app`, `--pool 8`) + equality-under-load + spawned-binary stdio e2e (wave 4)

**Wave 5** *(blocked on 06-05 and 06-06)*

- [x] 06-07-PLAN.md — `aprender-mcp-chronos`: embedded-weights build.rs, resolve_model, server, page, `--coldstart`/`--bench`, gated e2e through the server (1e-6 f32 / 2 % f16, `allow_long_horizon` + warning, forwards 46), shared-shape invariant, embedded-build proof (wave 5)

**Wave 6** *(blocked on 06-06 and 06-07; NOT autonomous — human checkpoint on ci.yml)*

- [x] 06-08-PLAN.md — host-gated `just` recipes (chronos-gate, embed-build, bench, coldstart, forecast-bench, pool-ratio, mase-rolling-origin) + the spike-006 MASE example + `06-EVIDENCE.md` measured on aarch64 release + the ci.yml proposal as a patch → blocking human decision (wave 6, NOT autonomous)

**Wave 7** *(blocked on everything)*

- [x] 06-09-PLAN.md — Makefile `$(CONTRACTS)`/`PHASE6_CONTRACTS`/`contract-audit-phase6` (tier3) + binding rows (zero BIND-) + README counts re-derived + CLAUDE.md Realizar-first exception row (D-07) + both drift gates green + the CI decision applied + `deferred-items.md` + SmoothL1Loss ticket + closing clippy/fmt/check/nextest sweep (wave 7)

**Gap closure** *(planned 2026-09-06 from `06-VERIFICATION.md` `status: gaps_found`, 4/5 must-haves.
SC2/SC3/SC4/SC5 hold on evidence the verifier generated; SC1 fails on three MEASURED grounds. Waves
8-11 continue the executed numbering so they cannot collide with waves 1-7; every plan carries
`gap_closure: true` and is selected by `/gsd-execute-phase 06 --gaps-only`. 06-01..06-09 are
untouched. Three items the verifier routed to `human_verification` — the browser MCP handshake +
chart rendering on both demo pages, measuring `quantiles_abs_f32_nonaarch64` on an x86_64 host
(D-ITEM-06-04), and the decision on SC4's Chronos ladder staying dark in CI (D-ITEM-06-03) — are NOT
planned here and remain open.)*

- [x] 06-10-PLAN.md — GAP 1, TRACER: refuse `cap` on every non-logistic growth arm at THE door (`cap is logistic-only; set growth to "logistic"`), the two unit cases the verifier named plus a logistic positive control, two e2e refusals through a live server, the `cap_is_logistic_only` contract equation + FALSIFY-BOUNDARY-017 + binding row, and the phase's `COVERAGE.md` no-external-API declaration (wave 8)
- [x] 06-11-PLAN.md — GAP 2a: ATTRIBUTE the measured 16.113 s wall (design/fit/predict split at five configurations, release), replace the per-row holiday `.any()` scan with prebuilt `HashSet` membership, then bound the product at the door — `MAX_HOLIDAY_DESIGN_COST` on `(points + horizon) x holiday_columns` and `MAX_HOLIDAY_DATES_TOTAL` — both contract-owned and re-mutated in `cost_bounds_match_contract`'s new scope (wave 9)
- [x] 06-12-PLAN.md — GAP 2b: the over-cost refusal and the near-miss acceptance through the server with the RED side observed under a temporarily raised constant, the `holiday_design_cost_bounded` equation + obligation + FALSIFY entry + binding row, and `just forecast-holiday-bench` turned from a printout into an SC1 2 s gate (wave 10)
- [x] 06-13-PLAN.md — GAP 3: extract the changepoint-count sampler as `poisson`, observe it saturating near 745 at lambda 900/2839, add the valid-domain branch above the threshold, and pin the bar as `poisson_sampler_domain` with its `float_tolerance` read by the test — parity ladder unmoved at 32 (wave 11)

**Gap closure, round 3 — CLASS closure** *(planned 2026-09-06 from `06-REVIEW.md`, an incremental code
review over `ce3e5a8ea..HEAD` finding 1 Critical + 4 Warnings + 4 Info. `06-VERIFICATION.md` is STALE:
it predates 06-10..06-13, which closed all three of its gaps. Waves 12-15 continue the executed
numbering; every plan carries `gap_closure: true` and `requirements: [SC1]`, and all four are
sequential because they overlap on `forecast.rs`, `types.rs`, `prophet.rs` and the tool-boundary
contract. 06-01..06-13 are untouched.*

*Why this round is shaped differently. Every gap this phase has produced is one instance of a single
class — the door accepts a request whose cost or whose semantics it never checks — and rounds 1 and 2
each fixed exactly the measured probe, so the next adversarial pass found the next instance. CR-01 is
round 2's own fix walking into round 3: 06-13 correctly removed Knuth's accidental saturation near 745,
and the simulated changepoint count now tracks lambda, which nothing at the door bounds; a 1 132-byte
accepted request went 0.157 s -> 2.334 s (14.8x), over the same 2 s SC1 bar 06-11/06-12 were created to
defend. So round 3 enumerates the door's WHOLE surface — every caller-settable knob and every cost axis
spent outside `FIT_BUDGET_SECS` — checks that enumeration into `contracts/forecast-tool-boundary-v1.yaml`
as `door_surface:`, and makes two tests go red on a new knob, a phantom knob, or an axis claiming a
bound that does not exist. The enumeration found two unbounded axes no review finding pointed at:
`holidays[].name` (no enforcement at all; ~2 000:1 amplification through `prophet::columns`) and the
NeuralProphet training path (no budget of any kind). All nine review findings are dispositioned in a
ledger repeated in every plan; 06-17 Task 3 verifies each row against the tree.*

*Three `human_verification` items remain OPEN and are NOT planned here: the browser MCP handshake +
chart rendering on both demo pages, the x86_64 `quantiles_abs_f32_nonaarch64` measurement
(D-ITEM-06-04), and the SC4-dark-in-CI decision (D-ITEM-06-03). Wiring `just forecast-sc1-sweep` into
CI is likewise a human decision this round does not take.*

- [x] 06-14-PLAN.md — TRACER: enumerate the door's whole caller-settable and cost surface by inspection (`pmat query` then source), land it as a machine-CHECKED `door_surface:` block with `every_request_knob_is_enumerated` / `every_cost_axis_names_a_real_bound`, then close CR-01 through it — `MAX_LOGISTIC_CHANGEPOINT_LAMBDA` refused before `make_design` with `prophet::changepoint_count` as the one changepoint-count implementation, a red-side e2e pair using the review's exact 1 132-byte payload, plus WR-01's ownership half (`poisson_normal_branch_lambda` contract-mirrored via a new `constant_f64`) and IN-03 (wave 12)
- [x] 06-15-PLAN.md — the two sibling axes the enumeration found and no review pointed at: `MAX_HOLIDAY_NAME_LEN` bounding the ~2 000:1 holiday-name amplification, and the unbudgeted NeuralProphet training path MEASURED at its structural maximum on release and then bounded or recorded `measured_at_structural_maximum` by that number; plus WR-03, the aggregate-dates refusal moved INSIDE the holiday loop with the exact-total refusal kept and the position proven by a test that would answer differently if it moved back. No `door_surface.cost_axes` sentinel survives this plan (wave 13)
- [x] 06-16-PLAN.md — WR-04 + IN-01: one parameterized SC1 gate (`crates/aprender-forecast/src/sc1_wall.rs`, `just forecast-sc1-sweep`) sweeping freq {D,W,MS} x growth {linear,logistic,flat} x holiday shape at the tightest legal history span, on release, with no ignore flag — and OBSERVED failing on the CR-01 configuration with the lambda bound raised; the three geometry-hard-coded benches folded onto one composition builder; and a shared numeric-shape validator with a must-match/must-not-match case table replacing the `awk … (v + 0 < 2.0)` coercion at ALL FIVE bar sites, each re-mutated in its own scope (wave 14)
- [x] 06-17-PLAN.md — WR-02 + WR-01's second half + IN-02/IN-04 + the round's closing evidence sweep: the sampler sweep gains contract-owned VARIANCE and ZERO-MASS bars (the zero-variance stub and a threshold lowered to 3.2 each OBSERVED red, both at >= 4 sigma from their own sampling noise); `feature_row`'s lookup made total without reverting 06-11's measured fix; a `debug_assert_eq!` making `bolt::transpose`'s safety claim checkable; the two 0.63.0 public-API notes; and every gate green in one recorded sweep with all nine findings resolved to the tree (wave 15)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5; Phase 6 is an independent track that may run alongside Phase 5's remaining GPU waves

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Differentiable MiniLM Conformance | 9/9 | Complete   | 2026-08-08 |
| 2. Deterministic Pair and Data Protocol | 9/9 | Complete   | 2026-08-09 |
| 3. Faithful Two-Stage Trainer and Head | 10/10 | Complete   | 2026-08-14 |
| 4. APR Artifact and Production Parity | 22/22 | UAT passed, awaiting secure-phase |  |
| 5. Benchmark and Claims Gate | 17/17 | In Progress|  |
| 6. Native Time-Series Forecasting Stack | 17/17 | In Progress|  |

### Phase 06.1: Forecast exogenous inputs: Prophet regressors, NeuralProphet events, and tier-safe cost bounds (INSERTED)

**Goal**: An operator can supply the information the forecast models cannot infer — known future
events and numeric drivers (price, promotions) — through the same one stateless `forecast` door,
with Prophet regressors proven to Python Prophet 1.4.0 parity, NeuralProphet events trained on the
autograd, every unsupported combination refused rather than silently ignored, and the existing
deployed results reproducing **byte-identically** when no new argument is passed.
**Depends on**: Phase 6.
**Requirements**: TBD — the binding inputs are the seven `forecast-exogenous-inputs` decisions in
`.planning/spikes/MANIFEST.md` (to be transcribed as D-xx in `06.1-CONTEXT.md`) plus the Success
Criteria below. There are no REQ-IDs in `.planning/REQUIREMENTS.md` for this work — that document
is the SetFit milestone's.
**UI hint**: no (no product surface; the demo page is unchanged)
**Spike evidence**: four VALIDATED spikes (011–014), packaged as `Skill("spike-findings-aprender")`
— `references/prophet-external-regressors.md`, `references/neuralprophet-exogenous-inputs.md`,
`references/no-argument-invariance-gate.md`. Raw experiments, oracle fixtures and run outputs in
`.planning/spikes/011-*` … `014-*`.

**Why this is a phase and not a feature add.** The architectural risk is already retired — spike 011
proved Prophet regressors are an **additive** change (a 219-line splice on the shipped `Design`,
nothing under `crates/` touched, full parity at 24 and 30 columns), and spike 012 proved the
consumer's byte-identical tag-bump gate is **free** because regressor columns append. What makes
this a phase is two measured defects that the feature cannot ship over:

- **The door's budget promise does not survive events.** `train_cost(n_samples, epochs, n_lags)`
  (`np.rs:464`) has no event-column term. Measured work is linear in E (`µs/step ≈ 12.9 + 0.085·E`)
  while the priced cost stays flat, so at `MAX_HOLIDAY_COLUMNS` (1 000 — the ceiling events inherit
  by reusing `HolidayArg`) a request buys **7.6× the work it was priced at**. This is cost axis
  **C-08**, the one the door refuses on under Lambda's timeout.
- **The proposed `RegressorArg` contract does not cover NeuralProphet with lags.** NP trains on an
  imputed daily grid denser than `ds`. Lag-free never reads an imputed-day regressor value (all
  fill rules including a garbage probe are bit-identical); with `n_lags > 0` it reads all of them,
  and two *defensible* fill rules differ by **10.48 on a series of scale 35.32 (~30 %)** while a
  wrong value on 12 % of grid days **flips the sign of both coefficients**. The CR's
  `values.len() == ds.len() + horizon` leaves those values undefined. This phase owns the decision
  (refuse / require grid-complete / impute-and-disclose) — it is not the planner's to invent.

**Relationship to Phase 7 (deliberate, not overlooked).** Both bounds above are hard-coded
constants — `MAX_NP_TRAIN_COST` (`types.rs:258`) and `MAX_HOLIDAY_COLUMNS` (`types.rs:43`) — that
Phase 7 converts into a resolved `DoorLimits` profile. Sequencing this phase first was a decided
trade (2026-09-20): Forecast Coach is a waiting consumer and Phase 7 ships no consumer-visible
feature. The **accepted cost** is that the event-column term lands on a hard constant here and is
re-expressed as tier policy in Phase 7. Phase 7's scope therefore grows by one item: the C-08 term
this phase adds must become tier-resolved along with the rest. Record the term's shape
(linear in E) and its calibration constant separately, so Phase 7 re-prices the constant without
re-deriving the shape.

**Success Criteria** (what must be TRUE):

  1. A caller can pass external regressors to `model: prophet` through the one stateless `forecast`
     door, and the port reproduces Python Prophet 1.4.0 on committed regressor fixtures as tests
     that run in CI: column order identical to Python at both 24 columns (regressors only) and 30
     (regressors plus two holidays with non-zero windows), standardisation constants within 1e-14
     using pandas `Series.std()` (**ddof = 1**), `prior_scales` / `s_a` / `s_m` exact `0.0`,
     Python's parameters through the Rust predict path within **1e-15 of `y_scale`**, every named
     component including `extra_regressors_additive` and `extra_regressors_multiplicative`, and a
     fitted objective slack no worse than the contract's `+0.5` bar. The components rung binds
     **relative to `y_scale`**, not the absolute 1e-10 — on `retail_sales` (y_scale 518 253) the
     absolute bar passes with only 3× headroom and would fail for arithmetic reasons.
  2. Every committed pre-change fixture reproduces its `ForecastResponse` **byte-identically** when
     no new argument is passed, across both models, linear / logistic / multiplicative growth,
     holidays with windows and AR lags. The signature covers `ds`, `yhat`, `yhat_lower`,
     `yhat_upper`, `trend`, `components` and `diagnostics`, hashes f64s by `to_bits()`, excludes
     `fit_seconds` / `predict_seconds`, and is **proven able to fail** by a 1-ULP mutation test that
     runs in CI — a green gate with no falsification probe beside it does not satisfy this
     criterion. Uncertainty bands are inside the comparison, not outside it.
  3. A caller can pass events to `model: neuralprophet` through the **same argument shape** the
     prophet arm already takes, trained as an additive block on the f32 autograd: a planted
     synthetic effect is recovered to within 10 % on every indicator column, fixed-seed runs are
     bit-identical with a differing-seed control, and the tape does not grow with the event-column
     count. `holidays` on `neuralprophet` either works or keeps refusing with a message naming the
     limitation — it is never silently ignored.
  4. The NeuralProphet training-cost bound carries an event-column term: `request_train_cost` takes
     the event-column count, the per-column constant is **measured on the deployment target** (not
     inherited from the spike's dev box) and rounded up, and
     `a_neuralprophet_request_over_the_train_cost_bound_is_refused` is re-derived against it. A
     request at `MAX_HOLIDAY_COLUMNS` is priced at no less than its measured work — the 7.6×
     under-pricing is observed closed, not asserted closed.
  5. The NeuralProphet-with-lags gappy-series regressor case has one decided, documented and
     **enforced** behaviour, with a test at `n_lags = 0` proving the imputed-day value is unread and
     a test at `n_lags > 0` proving the chosen rule (or the refusal) actually fires. If the decision
     is impute-and-disclose, the invented values appear in the response diagnostics; if it is
     refuse, the message names the gap. Binary drivers are never linearly interpolated.
  6. Every gate is green and the consumer can bump one line: workspace lib tests, `cargo clippy
     -- -D warnings` on every touched crate, `cargo fmt --all -- --check`, `pv validate` on every
     touched contract, the forecast server e2e tests, and a tag cut that builds `--locked`.

**Branch base**: continues on `gsd/phase-2-contract-gate` per the 02-01 policy.
**Plans:** 8/8 plans executed

Plans:
**Wave 1**

- [x] 06.1-01-PLAN.md — TRACER: external regressors end-to-end through the prophet door at 24-column Python Prophet 1.4.0 parity, with the pre-change no-argument invariance signature and baseline committed first (wave 1)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 06.1-02-PLAN.md — the regressor parity ladder at 24 and 30 columns, plus the regressor rungs in `contracts/prophet-parity-v1.yaml` (wave 2)
- [x] 06.1-03-PLAN.md — regressor cost ceilings and cost axis C-17, four door refusals, and the VIF / condition-number identifiability diagnostic that warns and never refuses (wave 2)
- [x] 06.1-04-PLAN.md — the three-part invariance gate: determinism, the 1-ULP mutation proof, and the part-C mechanism test with the uncertainty bands inside it (wave 2)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 06.1-05-PLAN.md — the NeuralProphet event block beside `NpModel`, and the calibrated C-08 event-column cost term with its enumeration decision (wave 3, has a checkpoint)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 06.1-06-PLAN.md — hoist all four holiday bounds above the model dispatch with an operand-aware design cost, then unblock `holidays` on the neuralprophet arm with per-event components (wave 4, has a checkpoint)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 06.1-07-PLAN.md — regressors on the neuralprophet arm: the by-construction lag-free guarantee, the gappy-series refusal at lags, and the multiplicative refusal (wave 5)

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 06.1-08-PLAN.md — the READMEs and the advertised tool description, the coverage declaration, and the SC6 all-gates-green sweep ending in a pristine-worktree `--locked` build (wave 6)

### Phase 7: Tier-Resolved Door Limits

**Goal**: Every forecast door bound keeps its enforcement but loses its hard-coded value:
the 13 `pub const MAX_*` in `crates/aprender-forecast/src/types.rs` become a resolved
`DoorLimits` profile whose `Default` is today's numbers, so one server binary can serve
four deployment envelopes that differ by orders of magnitude without mis-refusing legal
requests on any of them.
**Depends on:** Phase 6
**Requirements**: TBD — the binding inputs are `06-UAT.md` item 4 (DECIDED 2026-09-07) and
`06-REVIEW.md` CR-01 / WR-02.

**Why this is a phase and not a constant edit.** CR-01: `MAX_NP_TRAIN_COST = 15_000_000`
refuses `20 000 points x n_lags=7` — the canonical NeuralProphet setting, inside every
advertised bound — at `2 x 50 x 19 993 x 8 = 15 994 400`, 6.63% over, while its
10 000-point neighbour completes in ~1.3 s. The bound was fitted from BELOW by one parity
fixture (~95-97% of it) and validated from ABOVE by the 47.9 s structural worst case;
the region between was never priced. That generalizes: **a guard whose ACCEPTED region is
unmeasured is the same class of unfalsified claim as a bar that cannot fail** — the defect
class gap-closure round 3 existed to end, recurring one level up.

**Flexible is NOT unbounded.** Round 3's enumeration, completeness tests and contract
ownership all stand. The invariant strengthens:

- before — "cost axis C-08 is bounded at 15 000 000"
- after — "C-08 is ALWAYS bounded; its value comes from the resolved profile; no profile
  can disable a bound or set it above its tier's structural maximum"

**Success Criteria (draft — to be firmed at plan time):**

1. `DoorLimits` struct with `Default` byte-equal to today's 13 constants; every door check
   reads the resolved profile, no call site reads a bare `const`.
2. `contracts/forecast-tool-boundary-v1.yaml` `constants:` becomes the DEFAULT profile and
   `types::tests::cost_bounds_match_contract` keeps pinning it; a new test proves no profile
   can disable a bound or exceed its tier ceiling.
3. Named tier profiles whose ceilings derive from each envelope's REAL structural maximum
   (AWS Lambda, Docker on GCP/Azure, CloudFlare WASM, customer-hosted pmcp.run).
4. Every bound gains an ACCEPTED-region test at every profile — the gap CR-01 exposed.
   CR-01's own case (20 000 x n_lags=7) is accepted under the tier that can afford it and
   still refused under the tier that cannot, both observed.
5. WR-02 corrected wherever it is repeated (`types.rs:188-194`, three places in
   `forecast-tool-boundary-v1.yaml`, `binding.yaml`): pmcp 2.19.3
   `StreamableHttpServerConfig::stateless()` sets `max_request_bytes` = 4 MiB
   (`limits.rs:46`), enforced with a 413 at `streamable_http_server.rs:4571`, so C-07's
   `no_structural_maximum: true` is false for HTTP and the structural maximum is itself
   tier-dependent. The stdio transport genuinely has no framing cap — say so precisely.

**UI hint**: no
**Plans:** 0 plans

Plans:

- [ ] TBD (run /gsd-plan-phase 7 to break down)

### Phase 8: Laya Decision Model: Local Fine-Tune and Thin MCP Server

**Goal:** Productise spikes 024–026: a user can fine-tune Laya (ModernBERT-large decision model) on their own
8–64 labelled shots locally, calibrate it, convert it to .apr, and serve it through a thin, task-bound `classify` MCP server — in-tree,
linted, contracted and CI-tested — live on pmcp.run (default Lambda) the way the SetFit and Chronos servers are.
ModernBERT lands as a reusable aprender-core model; the decision layer is a method-neutral `aprender-decide`
crate with Laya as its first method (Kev/Jev later). Decisions: `08-CONTEXT.md`.
**Requirements**: TBD
**Depends on:** none of Phases 1–7 functionally — an independent track, like Phase 6. Reuses the thin-server
template (`crates/aprender-mcp-setfit/`) and the spike-020 `gemm_blis` layout.
**Source evidence:** `.claude/skills/spike-findings-aprender/references/laya-decision-model.md`,
`laya-rust-inference.md`, `aws-mcp-model-hosting.md` (spikes 024, 025, 026 — all VALIDATED).
**Plans:** 33/34 plans executed (08-33 executed locally 2026-09-29, unpushed; 08-34 planned, not yet executed), **08-32's CI must-have unmet; Phase 8 NOT complete** (08-33 contract-hygiene repair done, then 08-34 CI audit + push + maintainer-approved CI evidence; then verification); gap round 08-19..08-32 planned in waves 17-22 (2026-09-28; revised after plan checking). Executed history: 18 plans in 16 waves (08-08 complete under option 3: the D-19 demo's recorded outcome is GATE FAIL, both runs kept as fail-closed vectors; 08-09 complete under option A: both vectors refused in Rust by pack and verify; 08-10 complete under shared-crates-root: resolver executed on this workspace, deploy refusals proven offline; 08-11 complete: D-18 go/no-go recorded as a human-decided HOLD (hold-no-aws). Gap closure 08-13..08-18 added 2026-09-27 (user decision, option 1 from spikes 027/028): declare A1 noise-referenced pack bar + A2 in-distribution eval set + A3 median-of-three seed policy, wire them, ONE declared s64 gate run, then a human-approved live deploy; 08-12 close-out moves to Wave 16)

Plans:
**Wave 1**

- [x] 08-01-PLAN.md — four contracts declared before any run (gate thresholds, classify bounds, parity bars, decide-apr-v1 schema) + blocking contract-audit-phase8

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 08-02-PLAN.md — package-legitimacy checkpoint, pinned uv back office, tiny ModernBERT + tiny Laya fixtures from Laya's own code

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 08-03-PLAN.md — ModernBERT encoder in aprender-core (prefix-aware .apr loader, window mutation, CI-listed tests)
- [x] 08-08-PLAN.md — `just laya-train`: fine-tune, calibration, fail-closed gate, seed policy; TweetEval stance demo run

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 08-04-PLAN.md — `aprender-decide` crate: DecisionMethod seam, order-preserving task parser, Laya head/scorer/builder/temperature

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 08-05-PLAN.md — decide-apr-v1 artifact: packer, bounded load ladder, probes, identity, private constructor, determinism

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 08-06-PLAN.md — `aprender-mcp-decide` stdio server: one task-bound `classify` tool, contract-owned bounds, identity in every response

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 08-07-PLAN.md — `aprender-mcp-decide-lambda`: bootstrap loopback, in-memory S3 loader with sha256 pin, probe, deploy template
- [x] 08-09-PLAN.md — pack/verify CLI: gate recomputed in Rust from verified probabilities; both failing stance-demo run dirs REFUSED by pack and verify, nothing written (option A: early_stopping GateFailed[ece_post] exit 3, fixed_epochs RescoreDrift fine_tuned exit 2); full-model parity vs spike 025 (ids 14/14, |dp| 3.841e-6)

**Wave 8** *(blocked on Wave 7 completion)*

- [x] 08-10-PLAN.md — cargo-pmcp wrong-package decision (checkpoint, resolver executed on this workspace) + fail-closed, identity-checked deploy recipes that refuse any non-eligible artifact — proven offline against the tiny fixture (AWS CALLS: 0) (shared-crates-root: root crates -> crates/aprender-mcp-decide-lambda, cargo-pmcp 0.24.3 @ SDK e0561f8c9; setfit-train crates root restored byte-identical)

**Wave 9** *(blocked on Wave 8 completion)*

- [x] 08-11-PLAN.md — D-18 live stance deploy DEFERRED (option 3, 2026-09-26): HOLD go/no-go checkpoint + optional read-only AWS readiness, machine-checked hold record in 08-DEPLOY-EVIDENCE.json

**Wave 10** *(blocked on Wave 9 completion)*

- [x] 08-13-PLAN.md — gap closure: declare A1 (laya-parity-v1: pack bar max(1e-5, 4 x torch-vs-float64 noise), fixture-only final_norm/logits, x86_64 risk), A2 + A3 (laya-finetune-gate-v1 1.4.0: in-distribution eval set by rule, shift probe reported, median-of-three seeds; new demo_s64 cell) and the CONTEXT amendments BEFORE any run

**Wave 11** *(blocked on Wave 10 completion)*

- [x] 08-14-PLAN.md — gap closure: Python back office — per-seed training + median selection, float64 noise record, s64 data by rule + shift probe, self-tests (tiny fixture, CPU)

**Wave 12** *(blocked on Wave 11 completion)*

- [x] 08-15-PLAN.md — gap closure: Rust verifier — noise-referenced bound recomputed from the record, median seed re-derived, shift probe recomputed, legacy ordering; forged-bound/selection negative controls; 1.x vectors re-derived on real weights; FALSIFY-LAYA-GATE-014 test

**Wave 13** *(blocked on Wave 12 completion)*

- [x] 08-16-PLAN.md — gap closure: ONE declared s64 gate run (--seeds 3, early stopping, max 12 epochs) -> pack -> laya-verify -> stdio real-model leg; stop-rule checkpoint on anything but an eligible artifact; outcome of record + 08-GATE-RUN-EVIDENCE.json

**Wave 14** *(blocked on Wave 13 completion)*

- [x] 08-17-PLAN.md — gap closure: D-18 go/no-go (blocking-human: AWS approval, auth posture + provider, memory) -> read-only readiness -> live deploy with identity == H -> cold accepted region of both shapes -> 08-LIVE-DEPLOY-EVIDENCE.json **COMPLETE 2026-09-27: deployed-passed** (auth off, risk accepted, 3,008 MB; decide-tool-boundary-v1 2.0.0). Halted twice at deploy-refused, then option 1 declared the S3 read in the stack: identity == H through the pmcp.run edge, 4 cold samples 24.2-29.4 s under the 30 s cap, function left running

**Wave 15** *(blocked on Wave 14 completion)*

- [x] 08-18-PLAN.md — gap closure: exceeded-region response (blocking-human, only if exceeded) + the final live outcome record (containment verified, deferred items, Lambda README Deployed section)

**Wave 16** *(blocked on Wave 15 completion)*

- [x] 08-12-PLAN.md — CI-edit checkpoint (six targets + strict audit), CLAUDE.md D-16 exception row (claims licensed by the gate-run and live-deploy records), bindings implemented (live row per the final live outcome), tightened audit, focused/full/excluded CI-equivalent run

**Gap round 08-19..08-32** (added 2026-09-28 from 08-VERIFICATION gaps_found, 08-REVIEW and 08-CODE-REVIEW-FINDINGS; user decision: ONE class-wide round — each class A provenance / B input bounds / C gate honesty / D claim honesty / E Rust-Python agreement gets an invariant, a checked-in enumeration, a sweep and mutation proofs)

**Wave 17** *(blocked on Wave 16 completion)*

- [x] 08-19-PLAN.md — class A load side: every manifest leaf bound to its sha-bound blob at rung 4 on every load door (CR-01 load side), leaf sweep vs a decide-apr-v1 table, deployed artifact re-verified
- [x] 08-20-PLAN.md — class B shared readers: apr-format refuses duplicate tensor names and bounds its index reservation (test that fails when reverted), ModernBertLayer empty/shape guards
- [x] 08-21-PLAN.md — class A verify side: one typed contract-to-policy mapping, base identity + base-file pins, recipe block, slice fraction, run-field table and sweep
- [x] 08-22-PLAN.md — class C: scripts/laya_gates.tsv + `just laya-gates-selftest`; SKIP/ladder verdict, IAM listing and grant honesty, one hash helper, resolver ERE case table, bootstrap intent
- [x] 08-23-PLAN.md — stdio server: count before materialising texts, clap argv, scoped lint allow, served-fields and request-bounds tables with sweeps
- [x] 08-25-PLAN.md — deploy probe: response labels bound to the artifact, exact labels segment, shared budget, args_os, template comments

**Wave 18** *(blocked on Wave 17 completion)*

- [x] 08-24-PLAN.md — Lambda runtime: dead loopback exits, POST-only loads, config-checked health (A4-6), one CORS origin, too_large mapping, honest OnceCell docs, item-scoped lint allows incl. probe.rs, Lambda request rows swept, V4-a resume mutant
- [x] 08-26-PLAN.md — class B artifact side: duplicate names at rung 4, criteria bound, tokenizer truncation off, header version, rung-7 replay, bounded inspect, contract table + sweep
- [x] 08-27-PLAN.md — class E numeric: f64 + exactly-rounded sums in aprender-core and metrics.py, frozen boundary cases replayed bit-for-bit by both, f_avg recomputed, shared f32 metrics pinned by a to_bits snapshot

**Wave 19** *(blocked on Wave 18 completion; 08-28 moved here 2026-09-28 because it edits probe.rs after 08-24)*

- [x] 08-28-PLAN.md — classify tool claims: admission measured over stdio, owner decision on the truncation sentence (WR-03) and refusal wire code (V5-a), refusal formula fixed
- [x] 08-29-PLAN.md — class E Python: typed contract reader, '\n'-only rows, REFUSED not tracebacks, early-stopping contract made one rule, python_refusals sweep

**Wave 20** *(blocked on Wave 19 completion)*

- [x] 08-30-PLAN.md — offline positive dry run of the hardened deploy path, owner decision on redeploy + cold measurement (V4-b), D-18 3,008 MB amendment

**Wave 21** *(blocked on Wave 20 completion)*

- [x] 08-31-PLAN.md — class D: scripts/laya_claims.tsv + `just laya-claims-check`, pack-free load path, argmax/ui.rs, owner decision on aprender-decide publication (D-14), deferred-items final statuses

**Wave 22** *(blocked on Wave 21 completion)*

- [x] 08-32-PLAN.md — `just laya-gap-regression` (all five class sweeps + phase regression + real artifact) and the first CI run of Phase 8 code (workspace-test green) — **EXECUTED, CI MUST-HAVE UNMET (2026-09-29):** Task 1 passed; the branch was scrubbed, merged with upstream main and opened as draft PR paiml/aprender#4634, but every workflow run is `action_required` (fork PR awaiting a paiml maintainer) so `workspace-test` has NOT run, and upstream's contract-hygiene gates would stop it before the decide fragments 510/520. Decided; planned as 08-33 (repair all). Evidence: `08-CI-RUN-EVIDENCE.json`
- [x] 08-33-PLAN.md — contract-hygiene repair (gap closure, wave 23): 105 non-vocabulary `formal:` entries, 12 missing `valid_under` worlds, neon-blis/spectral-indices validate errors, chronos `#[test]` reorder, regenerated graph; upstream's shrink-only gates green locally (committed, NOT pushed). Owner decision "Repair all (new gap plan)". **EXECUTED (2026-09-29):** formal_prose 1464 and without-world 386 (both at their baselines, untouched), 105 formals in 11 contracts by ledger, 12 worlds, `pv lint contracts` all armed gates Pass, committed locally and NOT pushed (08-34 publishes). Evidence: `08-33-HYGIENE-EVIDENCE.json`, `08-33-FORMAL-REWRITES.json`.
- [ ] 08-34-PLAN.md — CI-order audit of all 65 fragments, HYGIENE stage in `laya-gap-regression`, secret-scanned fast-forward push, maintainer-approval checkpoint, workspace-test evidence (terminal state GREEN or BLOCKED-ATTRIBUTED; only GREEN closes 08-32's unmet must-have) (gap closure, wave 24, autonomous: false).
