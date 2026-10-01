# Project Research Summary

**Project:** Aprender Native SetFit Classification  
**Domain:** Native pure-Rust SetFit-style few-shot sentence classification  
**Researched:** 2026-08-07  
**Confidence:** HIGH overall; MEDIUM-HIGH for the exact core-BERT refactor surface and the new APR semantic adapter until their first executable conformance contracts pass

## Executive Summary

This milestone is a production model lifecycle, not a classifier utility. A valid SetFit implementation must first contrastively fine-tune a sentence encoder on deterministic same-class/different-class pairs and then fit a learned classifier on embeddings from that tuned encoder. The recommended v1 is deliberately narrow: one pinned `sentence-transformers/all-MiniLM-L6-v2` encoder, one graph-connected `aprender-core::autograd::Tensor` path, SetFit-compatible cosine-similarity MSE, one regularized multinomial softmax head, CPU-first execution, and one self-contained F32 APR artifact. A frozen encoder plus a probe or centroid is a useful baseline, but it must never be labeled SetFit.

Experts make this reliable by treating differentiability, data isolation, and artifact identity as executable contracts. Before a trainer or CLI is exposed, real MiniLM weights must reproduce tokenization/embedding fixtures and named encoder parameters must receive finite non-zero gradients and change after an optimizer step. Training must use bounded deterministic pair streams over canonical-train IDs, fit the head once per unique selected row in eval/no-grad mode, and select configuration only on canonical validation. The resulting model must be saved, closed, reloaded through the common core loader, and proven equivalent across core, CLI, and serving; only that reloaded APR may be evaluated, benchmarked, registered, or served.

The largest risks are silent graph detachment across Aprender's two tensor stacks, quadratic or leaky pair construction, train/serve reconstruction drift, and misleading few-shot claims. The roadmap should therefore be gated in this order: graph-connected MiniLM conformance; deterministic data and pairs; faithful two-stage training and head fitting; one-file APR reload parity and user surfaces; then the complete TweetEval claims gate. Do not trade these gates for wider model support, alternative losses, quantization, or GPU optimization in v1.

## Key Findings

### Recommended Stack

The canonical trainable path is the existing core BERT graph, upgraded where necessary; no new tensor framework or `aprender-setfit` crate should be introduced. `aprender-train` orchestrates optimization over core parameters but must never copy SetFit activations or weights into `aprender-train::autograd::Tensor`. SafeTensors and optional `hf-hub` are import conveniences only. Python is restricted to generating immutable numerical fixtures and is neither a training nor serving fallback.

**Core technologies:**

- `aprender-core::autograd::Tensor` (current workspace): the sole tensor/autograd graph for embedding gather, BERT, masked pooling, normalization, cosine-MSE, and AdamW updates; the legacy train autograd stack is not a SetFit backend.
- Repository-owned `BertSentenceEncoder` plus `NamedModule`/`ParameterStore`: batched `[B,S] -> [B,384]` graph-connected encoding, stable dotted tensor names, recursive train/eval propagation, selective freezing, optimizer grouping, and APR mapping.
- `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`: the only v1 encoder; reject unsupported families and mutated module/tokenizer/config contracts.
- `tokenizers` `0.23.1` with `default-features = false, features = ["fancy-regex"]`: exact Hugging Face batch tokenization without HTTP, native Oniguruma, or C++ ESAXX in the production feature.
- Core AdamW: encoder optimization using named graph-connected parameters; begin from explicit SetFit defaults of body LR `2e-5`, batch size `16`, one encoder epoch, and 10% warmup, with all resolved values persisted.
- Repository-owned K-class softmax regression plus core L-BFGS: one fallible, deterministic, L2-regularized head for every `K >= 2`; binary is the two-logit case, bias is not regularized, and the existing manual-SGD `LinearProbe` is not the production head.
- APR v2 plus semantic schema `setfit-apr-v1`: one checksummed F32 artifact containing the complete inference contract and provenance. `apr-format` remains a semantic-free byte container.
- Existing `rand_chacha` `0.9`, `serde`, `serde_json`, and `sha2`: domain-separated deterministic sampling/shuffling streams, typed manifests, and content hashes; do not add a second RNG family.

**Initial encoder contract:**

| Property | Required v1 value |
|----------|-------------------|
| Model | Pinned `all-MiniLM-L6-v2` revision above |
| Architecture | Post-LayerNorm BERT; 6 layers, hidden size 384, 12 heads, intermediate size 1536 |
| Tokenizer | Exact embedded Hugging Face `tokenizer.json`; WordPiece vocabulary 30,522 |
| Positions/input | Learned absolute positions up to 512; sentence maximum 256 wordpieces |
| Pooling | Attention-mask-weighted mean with checked non-zero denominator |
| Final transform | Row-wise L2 normalization with explicit epsilon |
| Dropout | Hidden/attention dropout 0.1 in train; disabled in eval |
| Module graph | `Transformer -> mean Pooling -> Normalize` only |

**Resolved objective and head choices:**

- Encoder objective: `mean((cosine(normalize(pool(encoder(a))), normalize(pool(encoder(b)))) - y)^2)` with `y in {0.0, 1.0}`. It must remain a scalar tensor connected to the encoder graph.
- Pair policy: deterministic, class-aware, approximately 1:1 positive/negative, streamed under persisted `max_pairs_per_epoch`; unordered IDs are canonicalized, self-pairs are rejected, and singleton behavior must be explicit.
- Classifier: one stable multinomial logistic-regression objective, `mean(logsumexp(logits_i) - logits_i[y_i]) + lambda/2 * ||W||^2`, solved full-batch with L-BFGS and convergence reported as a fallible result.
- Artifact tensor names: use existing `bert.*` names and canonical `setfit.classifier.weight` / `setfit.classifier.bias`. The conceptual component may be called the head, but `setfit.head.*` is not a second persisted namespace.

See [STACK.md](./STACK.md) for dependency versions, encoder details, alternatives, and the reference-fixture strategy.

### Expected Features

**Must have (table stakes):**

- Contracted tokenizer/encoder/pooling/normalization conformance with typed rejection of unsupported models.
- A genuinely graph-connected batched encoder: named embedding, attention, FFN, and normalization tensors receive finite non-zero gradients and update; padded batch and batch-one behavior agree within predeclared fixture-derived tolerances.
- A tensor-valued cosine-MSE pair objective that matches reference values/gradients and decreases in a controlled step.
- Fallible typed JSONL/split handling; balanced deterministic 8/16/32/64-per-class selection from canonical train only; stable IDs, hashes, source roles, and leakage rejection.
- A deterministic bounded pair sampler with correct labels, uniqueness/orientation rules, explicit singleton policy, stable replay, and `O(examples + pair budget)` state/storage.
- One fallible regularized K-class softmax head fit exactly once per original selected example; probabilities are finite and sum to one; ordered labels remain bound to columns.
- An explicit `Prepared -> EncoderTuned -> HeadFitted -> ArtifactReloadedAndVerified` lifecycle; export is forbidden without named encoder-update evidence.
- Fully resolved configuration, domain-separated seeds, reproducibility dossier, validation-only selection, and canonical-test access ledger.
- One offline self-contained checksummed F32 APR containing encoder, exact tokenizer bytes/hash, pooling/normalization/truncation policy, classifier, ordered labels, resolved configuration, update evidence, and source/data/sampling provenance.
- Mandatory train-save-load-predict parity for exact tokenizer/config/tensor bytes where applicable and tolerance-bounded embeddings/logits/probabilities, with exact labels across core, real CLI, and in-process serving.
- A cohesive CPU Rust/CLI lifecycle (`train -> APR -> inspect -> eval -> predict`) and native batch HTTP classification using the common loaded model.
- Machine-readable quality, calibration, latency/throughput, training time, peak memory, artifact size, backend/hardware, artifact hash, and paired SetFit-versus-9B-LoRA evidence.
- Executable failure contracts for invalid data, masks, dimensions, labels, model families, non-finite tensors, feature combinations, leakage, and malformed artifacts.

**Should have (competitive):**

- Pure-Rust, network-free, one-file training-to-serving lifecycle.
- Artifact/report evidence that mechanically proves the SetFit encoder changed, preventing a frozen probe from being mislabeled.
- A content-addressed reproducibility dossier joining data/model revisions, selected IDs, pair hash, resolved config, seeds, backend, access ledger, artifact, and metrics.
- Cross-surface semantic parity by construction: one core model for library, CLI, evaluation, and HTTP.
- Claims-gated TweetEval comparison against the existing 9B LoRA path on identical few-shot rows and measurement boundaries.
- Honest prediction evidence: full logits/probabilities, winner margin, truncation/token count, artifact/config identity, and backend, explicitly not causal token attribution.
- An explicit CPU reference profile and later accelerator support matrix that fails an unavailable requested device rather than silently relabeling CPU work.

**Defer (v2+):**

- Alternative encoders/families, generic Hugging Face or Sentence Transformers compatibility, and remote-code models.
- InfoNCE, SupCon, CoSENT, triplet, or multiple selectable body objectives.
- MLP, k-NN, centroid, one-vs-rest, end-to-end differentiable, or other production heads; frozen probes remain clearly named baselines only.
- Multilabel, hierarchical, token/span, generative, online/continual, distributed, and automated hyperparameter-search workflows.
- Quantization until F32 algorithm and artifact parity pass; every later quantized derivative requires fresh quality and calibration evidence.
- GPU-required training and broad backend optimization until every CPU correctness, lifecycle, and benchmark gate passes.
- Persistent embedding/token caches until keyed by the complete semantic model and dataset fingerprint.
- Token saliency, SHAP/LIME, counterfactuals, bundled nearest-example explanations, registry automation, and vendored TweetEval text.

See [FEATURES.md](./FEATURES.md) for all table stakes, differentiators, anti-features, and observable acceptance behavior.

### Architecture Approach

SetFit is one semantic model in `aprender-core` with lifecycle adapters around it. The core model owns tokenizer interpretation, batched trainable BERT, masked mean pooling, normalization, the multiclass classifier, ordered labels, and semantic APR load/save. `aprender-train` owns the explicit two-stage orchestration; data owns typed rows and deterministic index streams; CLI and serving are thin adapters over the reloaded core model. This preserves the Cargo DAG (`aprender-train -> aprender-core`), prevents a serving dependency on training, and makes train/serve parity structural rather than aspirational.

**Crate ownership:**

| Owner | Concrete path | Responsibility and boundary |
|-------|---------------|-----------------------------|
| Contracts | `contracts/`, `crates/aprender-contracts/src/` | Versioned data, gradient, sampler, artifact, feature, and reload-parity obligations; no implementations |
| Data | `crates/aprender-data/src/` | Fallible JSONL, stable examples/splits, balanced selection, class buckets, bounded `PairRef` stream, collation/fingerprints; no model math or serialization policy |
| Core | `crates/aprender-core/src/autograd/`, `src/models/bert/`, preferably `src/models/setfit/`, and `src/classification/` | Canonical graph primitives, named trainable BERT, tokenizer/batch boundary, pooling/normalization, K-class head, `SetFitModel`, typed manifest, semantic APR load/save |
| Training | `crates/aprender-train/src/finetune/setfit/`, `src/eval/classification/` | `SetFitTrainer`, AdamW/L-BFGS orchestration, schedule/clipping, stage transitions, detached head-embedding pass, validation selection, metrics/checkpoints; no second encoder/tensor engine |
| APR bytes | `crates/apr-format/` | Existing byte encoding, checksums, metadata limits, integrity; no SetFit logic or tokenizer interpretation |
| CLI | `crates/apr-cli/src/commands/`, `extended_commands.rs`, `dispatch_analysis.rs` | Arguments, feature-gated dispatch, structured output/errors; no tokenization, pooling, sampling, or model forward logic |
| Serving | `crates/aprender-serve/src/classification/`, `src/api/` | Immutable `SetFitRuntime`, batching, `/v1/classify`, probabilities/labels/readiness; no Hub, Python, sidecars, or alternate reconstruction |

**Major components:**

1. Contract suite and frozen fixtures — defines schema versions, names, split roles, numerical/failure expectations, and lifecycle gates before public surfaces.
2. Canonical tokenizer and graph-connected sentence encoder — exact batch tokenization through differentiable BERT, masked pooling, and L2 normalization on core autograd.
3. Typed data protocol and pair stream — selects immutable canonical-train IDs and emits deterministic bounded pair references without copying text or materializing Cartesian products.
4. Two-stage trainer and multinomial classifier — tunes the body, records named update evidence, deliberately switches to eval/no-grad, embeds each unique row once, and fits the head.
5. Semantic APR adapter — stores and validates the complete `setfit-apr-v1` inference model and provenance over existing APR bytes/checksums.
6. Common production model API — reloads the APR once and supplies identical embeddings/predictions to evaluation, CLI, benchmarking, and HTTP serving.

**Non-negotiable artifact lifecycle:**

```text
pinned SafeTensors/tokenizer import
  -> validate exact MiniLM contract and hashes
  -> graph-connected encoder tuning
  -> eval/no-grad unique-row head fit
  -> construct SetFitModel
  -> atomic self-contained F32 APR write
  -> close in-memory model
  -> offline SetFitModel::load_apr
  -> exact/tolerance parity probe
  -> only then evaluate, inspect, benchmark, register, predict, or serve
```

The final APR includes no optimizer moments, pair cursor, or scheduler state; those belong only in clearly marked `__training__.*` checkpoint tensors. Embed exact tokenizer bytes as a typed U8 payload with SHA-256 rather than a sibling file or base64-heavy manifest. Validate manifest version, complete tensor names/shapes/dtypes, tokenizer vocabulary/hash, label count/order, dimensions, and finite values before allocating the runtime. Keep the existing APR metadata 16 MiB limit fail-closed.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for API sketches, tensor names, manifest fields, data flow, adapters, and repository evidence.

### Critical Pitfalls

1. **Silent no-op encoder tuning** — raw `.data()`/vector copies can produce a new `requires_grad` tensor without any path to source weights. Use core autograd end to end; gate progress on named finite non-zero gradients, named parameter deltas, embedding deltas, finite differences, and a deliberate-detach mutation test.
2. **Pair/split leakage or quadratic sampling** — generate pairs only after selecting immutable canonical-train IDs; preserve roles and content hashes; stream canonicalized unordered pairs under a fixed budget; property-test labels, orientation, balance, singleton behavior, replay, and fixed-budget scaling.
3. **Numerically invalid encoder or head state** — reject empty/all-masked inputs, check denominators and normalization epsilon, use stable cosine/log-sum-exp/softmax, clip encoder gradient norm, regularize only head weights, treat optimizer convergence as fallible, and reject every non-finite tensor at save/load.
4. **Training and production are different models** — remove duplicate tokenization/pooling/reconstruction paths. Persist exact tokenizer/config/labels/head and require train-save-load parity through core, CLI, and HTTP with no network or sidecars.
5. **Seeded but irreproducible runs** — derive sorted, domain-separated ChaCha streams for selection, pairs, shuffle, dropout, and initialization; persist seeds, manifests, backend, threads, and hashes; make CPU the semantic reproducibility reference.
6. **Misleading few-shot claims** — require every one of 40 shot/seed cells, shared sampled-ID hashes across SetFit and LoRA, validation lock before test, official explicit-label `F_avg`, per-seed rows/uncertainty, and equal time/memory/throughput boundaries.

See [PITFALLS.md](./PITFALLS.md) for all falsification tests and phase-specific warning signs.

## Implications for Roadmap

Based on the combined research, use five capability phases. Each phase is a hard dependency boundary, not merely a feature grouping.

### Phase 1: Differentiable MiniLM Conformance

**Rationale:** Every downstream capability is invalid if the body graph detaches or the Rust model does not reproduce the contracted encoder. This is the highest-risk code boundary and must be proved before trainer or CLI work.

**Delivers:** Executable SetFit gradient/artifact contracts and frozen Python fixtures; missing core graph primitives; recursive named BERT parameters; shared tokenizer/batch types; the pinned real-weight MiniLM `Transformer -> masked mean -> Normalize` path; train/eval propagation; one controlled AdamW step.

**Concrete ownership:** `contracts/`, `crates/aprender-contracts/src/`, `crates/aprender-core/src/autograd/`, `crates/aprender-core/src/models/bert/`, a new core text/tokenizer boundary, and early `crates/aprender-core/src/models/setfit/` model contract.

**Exit gates:**

- Token IDs, type IDs, masks, token outputs, pooled embeddings, normalized embeddings, pair cosine-MSE, selected gradients, and one optimizer step match pinned reference fixtures under tolerances declared before observing Rust discrepancies.
- For a non-degenerate mixed pair batch, intended word embedding plus representative Q/K/V/O, FFN, and LayerNorm parameters have present finite non-zero gradients and change; frozen tensors remain byte-identical.
- Loss is a finite scalar graph tensor; fixed embeddings change after the step and pair cosine moves in the loss-reducing direction.
- New gather, row selection, masked reduction, normalization, cosine, and MSE primitives pass central finite differences.
- Batch size 1 and mixed-length padded batches preserve each example's embedding/order within tolerance; empty/all-masked and unsupported encoder mutations fail with typed errors.

**Addresses:** TS-01, TS-02, TS-03, DF-02, and their TS-17 contracts.  
**Avoids:** PF-001, PF-004, PF-011, and the encoder portion of PF-014.

### Phase 2: Deterministic Pair and Data Protocol

**Rationale:** Pair semantics and split isolation must be frozen before encoder epochs can produce meaningful or publishable results. This phase can begin after the Phase 1 contracts; head math may be developed in parallel only after core numerical primitives are green, but it integrates in Phase 3.

**Delivers:** Fallible typed JSONL/splits, stable IDs/content hashes, balanced few-shot selection, immutable sampled-ID manifests, typed split roles and access records, sorted class buckets, and a restartable bounded `PairRef` iterator keyed by `(dataset_hash, seed, epoch, policy)`.

**Concrete ownership:** `crates/aprender-data/src/`, relevant `contracts/`, and thin dataset command integration in `crates/apr-cli/src/commands/`. Preserve the existing TweetEval worktree implementation as input to these contracts rather than duplicating it.

**Exit gates:**

- Exactly 8/16/32/64 unique canonical-train IDs per class are selected for seeds 13, 17, 23, 29, 31, 37, 41, 43, 47, and 53; validation/test IDs and content hashes are disjoint.
- Every positive shares a label, every negative differs, endpoints differ, unordered identities cannot conflict, counts satisfy the resolved strategy/budget, and identical seeds reproduce the same ordered manifest.
- Increasing example count 10x under fixed `max_pairs_per_epoch` retains `O(examples + pair budget)` state/storage, not Cartesian-product growth.
- Duplicate IDs, unknown labels, malformed rows, cross-split content, test endpoints, and use of the merged SetFit compatibility test as validation fail closed.
- Singleton-class behavior and the bounded policy's explicit deviation from exhaustive SetFit oversampling are versioned and fixture-tested.

**Addresses:** TS-04, TS-05, sampling portions of TS-09, and DF-04.  
**Avoids:** PF-002, PF-003, and the data-order portion of PF-006.

### Phase 3: Faithful Two-Stage Trainer and Head

**Rationale:** Only after graph and data invariants pass can training prove SetFit identity. Encoder improvement and classifier fitting must be separate observable stages so head success cannot hide a frozen encoder.

**Delivers:** One `SetFitTrainer`, resolved configuration and scheduler/clipping policy, domain-separated RNG streams, named gradient/update audit, explicit `EncoderTuned -> HeadFitted` detach boundary, deterministic regularized K-class softmax head, ordered type-tagged labels, validation-only selection, and reproducibility reports.

**Concrete ownership:** `crates/aprender-train/src/finetune/setfit/`, core `src/classification/` and `src/models/setfit/`, and `crates/aprender-train/src/eval/classification/`. The trainer orchestrates core AdamW and L-BFGS; it does not own a second tensor or BERT implementation.

**Exit gates:**

- The full trainer repeats Phase 1 named gradient/parameter/embedding update assertions before it can report `algorithm=setfit` or export.
- State transitions are enforced; after tuning, dropout is disabled and each original selected row is encoded exactly once in eval/no-grad mode. Pair multiplicity cannot change the head dataset.
- Binary and multiclass reference fixtures pass through the same `K >= 2` implementation; head loss decreases, convergence/failure is explicit, logits/probabilities are finite, rows sum to one, and label permutation preserves semantics.
- Two clean CPU runs reproduce selected IDs, ordered pairs, batch order, step count, loss trace, semantic tensor/config/tokenizer hashes, and predictions apart from declared volatile metadata.
- Model/config selection consumes canonical validation only and emits a selection-lock record before any test access.

**Addresses:** TS-06 through TS-10, DF-02, and DF-03.  
**Avoids:** PF-001, PF-004, PF-006, PF-009, PF-010, and PF-012.

### Phase 4: APR Artifact and Production Parity

**Rationale:** Product value depends on deploying the exact trained model. Artifact semantics and mandatory reload must precede CLI/HTTP claims so no user surface can bypass the production bytes.

**Delivers:** Typed `setfit-apr-v1` manifest, stable encoder/classifier names, raw tokenizer U8 payload/hash, atomic self-contained F32 APR, fail-closed loader, mandatory close/reload probe, public Rust model API, generic CLI dispatch, inspect/eval/predict lifecycle, immutable HTTP runtime, and audit-friendly prediction evidence.

**Concrete ownership:** `crates/aprender-core/src/models/setfit/` and `src/serialization/apr/` own semantics; `crates/apr-format/` owns bytes/checksums only; `crates/apr-cli/src/commands/`, `extended_commands.rs`, and `dispatch_analysis.rs` remain thin; `crates/aprender-serve/src/classification/` and `src/api/` call the loaded core model.

**Exit gates:**

- F32 encoder/classifier tensor bytes, typed configuration, tokenizer bytes/hash, and ordered labels round-trip exactly; embeddings, logits, and probabilities meet predeclared tolerances; semantic labels are exact.
- Core, real CLI, and in-process HTTP agree on single and mixed-length batched Unicode fixtures, preserve input order, and report the same artifact hash.
- With network disabled and sidecars removed, the APR can inspect, evaluate, predict, and serve.
- Missing/mutated tensor names/shapes/dtypes, tokenizer hash, label map, pooling/max length, oversized metadata, checksums, and NaN/Inf values fail before runtime allocation or prediction.
- Only `ArtifactReloadedAndVerified` may emit the production path; evaluation, registration, benchmarking, and serving reject an in-memory-only or checkpoint model.

**Addresses:** TS-11 through TS-15, DF-01, DF-05, and DF-07.  
**Avoids:** PF-005, PF-011, PF-012, PF-014, and artifact portions of PF-004/PF-010.

### Phase 5: Benchmark and Claims Gate

**Rationale:** Comparative claims are meaningful only after the production artifact and all evidence from Phases 1–4 exist. The benchmark is a release gate, not an exploratory shortcut around validation discipline.

**Delivers:** One shared SetFit/9B-LoRA harness, canonical validation lock/test access ledger, complete machine-readable per-run results, recomputable aggregates and paired deltas, classification/calibration diagnostics, and resource/performance evidence from the reloaded APR.

**Concrete ownership:** benchmark contracts in `contracts/tweet-eval-stance-benchmark-v1.yaml`, metric/evaluation code in `crates/aprender-train/src/eval/classification/`, and thin command/reporting integration in `crates/apr-cli/src/commands/`.

**Exit gates:**

- All 40 cells exist: shots `{8,16,32,64}` per class crossed with the ten contracted seeds; each method shares the same sampled-ID hash in a cell and every row references dataset/model revisions, selection lock, artifact hash, backend, and encoder-update evidence.
- The primary score is recomputed as `F_avg = (F1_against + F1_favor) / 2` from the stored confusion matrix and ordered labels. Reports also include per-class metrics, three-class macro-F1, MCC, calibration diagnostics, and explicit between-seed uncertainty.
- Headline mean/dispersion/paired deltas are exactly derivable from per-run rows; dropping any cell fails completeness; no best-seed or post-test checkpoint selection is permitted.
- SetFit and 9B LoRA use identical IDs and declared measurement boundaries. Each report includes training time, latency/throughput with batch and warmup details, peak memory, model size, calibration, hardware/backend, and artifact hash.
- Quality, latency, memory, and size are measured from the same reloaded F32 APR. Any later quantized derivative is a separate model with new quality/calibration evidence.

**Addresses:** TS-16, the complete TS-17 matrix, DF-06, and CPU evidence for DF-08.  
**Avoids:** PF-002, PF-007, PF-008, PF-013, and performance portions of PF-014.

### Phase Ordering Rationale

The five phases compress the architecture's stricter build graph without changing its order:

| Build order | Deliverable | Phase dependency |
|------------:|-------------|------------------|
| 1 | Versioned data/gradient/artifact/lifecycle contracts and fixtures | Phase 1 foundation |
| 2 | Core differentiable primitives and recursive named BERT parameters | After order 1 |
| 3 | Shared tokenizer, batched BERT, masking, pooling, normalization, real-weight parity | After order 2 |
| 4A | Fallible dataset, deterministic selector, bounded sampler | After order 1; Phase 2 |
| 4B | Fallible multinomial head math and serialization tensors | After order 2; may develop alongside Phase 2, integrates in Phase 3 |
| 5 | Two-stage trainer over core encoder and head | After 3, 4A, 4B |
| 6 | Self-contained APR writer/loader plus mandatory reload parity | After 5 |
| 7 | Rust API and generic CLI lifecycle | After 6 |
| 8 | Serving runtime and HTTP classification through the same model | After 6 and shared API |
| 9 | Complete TweetEval/9B-LoRA benchmark and claims report | After 7 and 8 |
| 10 | Optional accelerator enablement/optimization | Only after CPU orders 2–9 pass |

- Do not build the trainer before the graph-connected encoder-update proof.
- Do not expose a production CLI before the APR reload round trip.
- Do not implement a serving-specific forward path.
- Do not benchmark the in-memory trainer.
- Do not begin accelerator optimization before CPU numerical, lifecycle, and benchmark contracts pass.

### Research Flags

Phases likely needing deeper research or a formal spike during planning:

- **Phase 1:** run the planned single-batch real-weight MiniLM spike before committing the full refactor. Repository evidence selects core autograd confidently, but broadcast/masking behavior for `batch > 1`, recursive parameter coverage, exact dropout placement, and fixture-derived tolerances require executable validation.
- **Phase 2:** targeted research/decision work is required for singleton classes and the precise bounded-oversampling compatibility rule. The implementation must state where it intentionally differs from exhaustive SetFit enumeration.
- **Phase 5:** choose the calibration statistic, binning/support policy, and uncertainty presentation before collecting results. Split isolation and final-APR identity are already fixed; the estimator is not.
- **Optional post-v1 accelerator phase:** research backend-specific tolerances, deterministic-kernel availability, and thread/device reproducibility before making support or speed claims.

Phases with established patterns that can skip a broad research phase:

- **Phase 3 head/trainer structure:** SetFit's two stages, cosine-MSE default, multinomial logistic objective, stable softmax, L2 regularization, and L-BFGS behavior are well documented. Planning should focus on integration and executable gates, not reopen algorithm selection.
- **Phase 4 APR/CLI/serve lifecycle:** Aprender already has APR bytes/checksums, atomic persistence, CLI dispatch, and serving boundaries. Use targeted schema/contract work, not a new architecture survey.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official SetFit/Sentence Transformers sources and concrete repository inspection agree on cosine-MSE, two-stage fitting, pinned MiniLM, core autograd, native tokenization, CPU-first execution, and APR. Exact BERT refactor scope remains MEDIUM-HIGH until the Phase 1 spike. |
| Features | HIGH | SetFit identity, production lifecycle, data isolation, artifact parity, and benchmark requirements have explicit observable acceptance behavior. Optional explanation and accelerator extensions are MEDIUM-HIGH and deferred. |
| Architecture | HIGH for crate/DAG direction; MEDIUM-HIGH for schema details | Existing Cargo boundaries require model semantics in core and orchestration in train. The `setfit-apr-v1` manifest and tokenizer U8 high-level API need ratification through contracts, but no new container is needed. |
| Pitfalls | HIGH | Graph detachment, dual stacks, pair leakage/explosion, duplicated inference paths, head weighting, label drift, and benchmark failure modes are grounded in repository evidence and authoritative algorithm/metric definitions. Cross-backend determinism is MEDIUM pending measurement. |

**Overall confidence:** HIGH in the roadmap direction and non-negotiable gates; MEDIUM-HIGH in implementation estimates until Phase 1 conformance and the APR schema contract pass.

### Gaps to Address

- **Exact numerical tolerances:** derive and freeze them from pinned Python fixtures before comparing Rust outputs; never tune tolerances after seeing discrepancies.
- **Core BERT batch/autograd surface:** prove every required attention/broadcast/mask path preserves gradients for `batch > 1` and that the named traversal covers the intended trainable/frozen set.
- **Singleton and bounded sampler policy:** version the exact behavior and explicitly label its deviation from exhaustive SetFit oversampling.
- **Reference environment:** solve and commit a hash-locked dev environment for `setfit==1.1.3`, `sentence-transformers==5.7.0`, `torch==2.13.0`, and `scikit-learn==1.9.0`; individual version availability does not prove joint CI compatibility.
- **CPU/thread reproducibility:** measure deterministic behavior across supported thread counts. CPU is the reference, but the acceptable concurrency profile must be documented.
- **Calibration:** select validation-only ECE/Brier-style diagnostics, bin/support reporting, and any serialized calibrator policy before benchmark execution.
- **APR tokenizer payload:** confirm the high-level typed U8 reader/writer and fail-closed 16 MiB handling through an executable `setfit-apr-v1` contract without moving semantics into `apr-format`.
- **Accelerators:** establish backend identity, feature combinations, tolerances, and deterministic expectations before adding GPU support; do not promise cross-device bit equality.

## Sources

### Primary (HIGH confidence)

- [Hugging Face SetFit conceptual guide](https://huggingface.co/docs/setfit/en/conceptual_guides/setfit) — two-stage body and classifier lifecycle.
- [Hugging Face SetFit trainer reference](https://huggingface.co/docs/setfit/reference/trainer) — current objective, sampling, learning-rate, epoch, warmup, maximum-length, seed, and body-freezing behavior.
- [Hugging Face SetFit sampling strategies](https://huggingface.co/docs/setfit/en/conceptual_guides/sampling_strategies) — pair identities, oversampling/undersampling/unique behavior, and pair-count implications.
- [Original SetFit paper](https://arxiv.org/abs/2209.11055) — contrastive Siamese body tuning followed by classifier training.
- [Sentence Transformers loss reference](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) — cosine-similarity targets with MSE.
- [`all-MiniLM-L6-v2` model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — model identity, masked mean pooling, L2 normalization, and 256-wordpiece sentence limit.
- [Sentence Transformers custom model/save documentation](https://sbert.net/docs/sentence_transformer/usage/custom_models.html) — tokenizer, transformer, pooling, normalization, and module reconstruction state.
- [Hugging Face Tokenizers API](https://huggingface.co/docs/tokenizers/main/api/tokenizer) and [`tokenizers` 0.23.1 docs](https://docs.rs/tokenizers/0.23.1/tokenizers/) — exact batch encoding, padding/truncation/masks, and Rust API.
- [TweetEval paper](https://aclanthology.org/2020.findings-emnlp.148/) — canonical stance evaluation and the `favor`/`against` F1 average.
- `.planning/PROJECT.md` — milestone scope, non-negotiable encoder tuning, benchmark protocol, and artifact parity.
- `contracts/tweet-eval-stance-benchmark-v1.yaml` and `docs/examples/tweet-eval-stance.md` — project-authoritative splits, labels, shots, seeds, metric, provenance, and reporting requirements.
- `crates/aprender-core/src/models/bert/`, `crates/aprender-core/src/autograd/`, `crates/aprender-core/src/nn/optim/`, and `crates/aprender-core/src/optim/lbfgs.rs` — existing production BERT graph and optimizer capabilities.
- `crates/aprender-train/src/autograd/`, `crates/aprender-train/src/transformer/`, and `crates/aprender-train/src/finetune/linear_probe.rs` — legacy parallel graph, detachment, and head limitations that the new path must not inherit.
- `crates/apr-format/src/v2/` and `crates/aprender-core/src/serialization/apr/` — existing byte container, integrity/metadata limits, semantic adapter, and atomic persistence boundaries.
- `crates/aprender-serve/src/api/` and `crates/apr-cli/src/commands/` — production surface boundaries and duplicated paths to consolidate.

### Secondary (MEDIUM confidence)

- No roadmap-critical recommendation relies solely on secondary sources. Python `setfit`, `sentence-transformers`, `torch`, and `scikit-learn` are numerical reference tools only; their exact joint lock must still be validated in CI.

### Tertiary (LOW confidence)

- None. Open questions are explicitly carried as gaps rather than converted into low-confidence requirements.

---
*Research completed: 2026-08-07*  
*Ready for roadmap: yes*
