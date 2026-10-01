# Feature Landscape

**Domain:** Native SetFit-style few-shot sentence classification
**Project:** Aprender Native SetFit Classification
**Researched:** 2026-08-07
**Overall confidence:** HIGH for SetFit identity, two-stage behavior, artifact lifecycle, and benchmark requirements; MEDIUM-HIGH for optional explainability and accelerator extensions

## Product Boundary

A v1 feature may be called **SetFit** only when it contrastively fine-tunes a sentence encoder and then fits a learned classifier head on embeddings from that tuned encoder. A frozen-encoder linear probe, nearest-centroid classifier, or similar embedding baseline is useful for comparison, but is not SetFit.

The faithful algorithm is narrower than the production product. SetFit itself requires the trainable sentence-transformer body, same/different-class pair objective, and second-stage classifier head. Aprender's target additionally requires deterministic data handling, APR persistence, CLI and serving integration, artifact parity, benchmark integrity, inspectability, and operational measurements.

## Table Stakes

Missing any **SetFit identity** item makes the algorithm unfaithful. Missing any **production lifecycle** item makes the milestone incomplete even if the algorithm trains in memory.

| ID | Feature | Requirement class | Why expected | Complexity | Dependencies | Observable acceptance behavior |
|----|---------|-------------------|--------------|------------|--------------|--------------------------------|
| TS-01 | Contracted encoder, tokenizer, pooling, and normalization | Encoder conformance | Training and inference must construct the same sentence embedding. V1 should support the pinned MiniLM/BERT family explicitly instead of implying generic Hub compatibility. | High | Existing APR/BERT importer; typed model manifest; exact tokenizer payload | The importer accepts the pinned `all-MiniLM-L6-v2` contract, reproduces token IDs, masks, masked-mean pooling, and normalized embeddings on fixed fixtures, and rejects unsupported architecture/module/tokenizer mutations with typed errors. |
| TS-02 | Graph-connected batched sentence encoder | **SetFit identity** | Encoder adaptation is the defining capability. Batched masks and pooling are also necessary for practical training and serving. | Very High | TS-01; canonical core autograd; named parameter traversal; differentiable gather, mask, pool, normalize, and cosine primitives | On a mixed positive/negative batch, named embedding, attention, FFN, and normalization parameters receive finite non-zero gradients and change after an optimizer step; frozen parameters do not. Batch-one and padded-batch embeddings agree within tolerance. |
| TS-03 | Tensor-valued cosine-similarity pair objective | **SetFit identity** | V1 needs a precise compatibility target for supervised contrastive sentence-encoder tuning. Scalar metric helpers cannot train the encoder. | High | TS-02; stable cosine and MSE reductions | Loss is a finite scalar attached to the encoder graph, matches reference fixtures/finite differences, decreases in a controlled step, and moves positive/negative pair cosines in the expected direction. |
| TS-04 | Fallible dataset, split, and balanced few-shot selection | Benchmark and data integrity | Small-data claims depend on exact labeled rows and strict train/validation/test roles. | Medium-High | Typed JSONL examples; stable IDs; dataset hashes; TweetEval contract | Selection draws exactly 8/16/32/64 unique canonical-train IDs per class for the contracted seeds, preserves labels and source split, and rejects duplicates, unknown labels, invalid counts, cross-split content, or compatibility-test use as validation. |
| TS-05 | Deterministic, class-correct, bounded pair sampler | **SetFit identity** and operational safety | Positive pairs must share a label, negative pairs must differ, and pair construction must not materialize a quadratic Cartesian product. | High | TS-04; stable class buckets; seeded RNG; explicit singleton policy | The same seed produces the same ordered pair manifest; positives/negatives, self-pair, orientation, duplicate, balance, and budget invariants are property-tested; memory remains `O(examples + pair budget)` at fixed budget. |
| TS-06 | One fallible regularized multiclass linear head | **SetFit identity** | Standard SetFit fits a classifier after body tuning. TweetEval needs three classes, while binary should remain the same `K=2` model. | High | TS-02; stable softmax/log-sum-exp; deterministic solver; ordered label vocabulary | The head fits each unique selected example exactly once, returns finite logits and probabilities summing to one, handles `K >= 2`, reports convergence/failure, preserves label-column meaning, and round-trips weights/bias. |
| TS-07 | Explicit faithful two-stage trainer | **SetFit identity** | Encoder contrastive tuning and classifier fitting are separate phases with a deliberate detach boundary. Head improvement must not mask a frozen body. | High | TS-02, TS-03, TS-05, TS-06 | The state flow is `Prepared -> EncoderTuned -> HeadFitted`; head embeddings are generated once per original training row with the tuned encoder in eval/no-grad mode; export is blocked unless named encoder-update evidence exists. |
| TS-08 | Validated, fully resolved training configuration | Production lifecycle | Hidden defaults and caller-supplied inference dimensions make runs irreproducible and artifacts ambiguous. | Medium | TS-01, TS-05, TS-07 | Public API and CLI expose validated body/head learning settings, epochs, batch size, warmup, regularization, max length, pair policy/budget, freeze policy, and device. The resolved configuration—not only user overrides—is persisted and inspectable. Invalid/contradictory values fail before training. |
| TS-09 | End-to-end reproducibility and provenance | Production lifecycle and benchmark integrity | A single seed does not automatically control subset selection, pair sampling, shuffle, dropout, or head initialization. | High | TS-04, TS-05, TS-07, TS-08; domain-separated RNG streams; stable iteration order | Two clean CPU runs with the same inputs reproduce selected IDs, pair manifest, batch order, step count, loss trace, semantic tensor/config/tokenizer hashes, and predictions. Changing the root seed changes a stochastic stage when alternatives exist. Dataset/model revisions, hashes, backend, and derived seed identities are recorded. |
| TS-10 | Leakage-safe evaluation and model selection | Benchmark integrity | Canonical validation must select configuration; canonical test is read only after selection is fixed. Accuracy alone is misleading for imbalanced stance data. | High | TS-04, TS-06, TS-09; classification metrics | The evaluator reports official `F_avg = (F1_against + F1_favor)/2`, per-class metrics, three-class macro-F1, MCC, confusion matrix, calibration diagnostics, and uncertainty from explicit ordered labels. An access ledger proves no test read occurred before a selection-lock record. |
| TS-11 | One self-contained, checksummed SetFit APR | Artifact lifecycle | Deployment must receive exactly one model containing all inference-required state, with no Python, Hub, or sidecar dependency. | High | TS-01, TS-06, TS-07, TS-08, TS-09; APR semantic adapter | The F32 APR contains encoder tensors, exact tokenizer state/hash, pooling/normalization and max-length policy, linear head, ordered labels, resolved configuration, source/dataset provenance, and sampling seeds/hashes. It loads offline and fails closed on missing, mismatched, non-finite, or malformed state. |
| TS-12 | Mandatory train-save-load-predict parity | Artifact lifecycle | Evaluating an in-memory object while serving a reconstructed model invalidates quality claims. | High | TS-11; common core loader/model API | After training, the model is closed and reloaded; tokenizer outputs are exact and embeddings, logits, probabilities, and labels match the pre-save object within documented tolerances. Core, CLI, and serve agree on single and batched fixtures. Only the reloaded artifact may be benchmarked, registered, or served. |
| TS-13 | Cohesive Rust API and CLI lifecycle | Product usability | Users need one supported route through training, inspection, evaluation, prediction, and benchmarking rather than internal modules. | Medium-High | TS-08, TS-10, TS-12; existing CLI dispatch and feature gates | A CPU workflow performs `train -> APR -> inspect -> eval -> predict` with structured errors and machine-readable output. Generic run/eval/inspect commands auto-detect the SetFit architecture; CLI code remains a thin adapter over library APIs. |
| TS-14 | Native batch inference and serving | Production lifecycle | Business value depends on using the classifier cheaply and consistently outside the trainer. | High | TS-01, TS-06, TS-12; immutable serving runtime; shared tokenizer/model | Library, CLI, and HTTP accept one or many texts and return ordered label, probability vector, optional logits, model hash, and latency. Tokenizer state is reused, mixed lengths are masked correctly, input order is preserved, explicit unavailable devices fail rather than silently falling back, and readiness reports the classifier artifact loaded. |
| TS-15 | Honest inspectability and prediction explanation | Product trust; not SetFit identity | Users need to understand what model/config produced a prediction without receiving unsupported causal claims. | Medium | TS-08, TS-11, TS-14 | `inspect` exposes encoder/tokenizer revision and hashes, pooling/truncation policy, label order, head type/regularization, training data fingerprint, seeds, and update evidence. Prediction explanation exposes all class probabilities/logits, winner margin, truncation/token-count facts, artifact hash, and backend. It is explicitly described as decision evidence, not token-level causality. |
| TS-16 | Operational performance and fair benchmark reporting | Production readiness | "Fast" and "small" require measured training and serving behavior, not only quality. | High | TS-09, TS-10, TS-12, TS-14; shared SetFit/LoRA harness | Machine-readable results cover all 40 shot/seed cells and the same sampled IDs for SetFit and 9B LoRA. Reports include per-run and aggregate quality, paired deltas, training time, latency/throughput with batch/warmup boundaries, peak memory, artifact size, calibration, hardware/backend, and artifact hash. Pair generation and tokenization remain bounded and batched. |
| TS-17 | Executable failure and compatibility contracts | Production safety | Shape-correct or happy-path tests can miss detached gradients, label drift, panics, feature-gate failures, and hostile resource inputs. | High | All preceding table stakes; Aprender contracts and CI feature matrix | Contracts reject empty/all-masked input, invalid IDs/labels/classes, overlength-policy violations, unsupported model families, oversized metadata/model dimensions, non-finite tensors, pair/split leakage, missing feature combinations, and artifact mismatches without panic or silent truncation. |

## Differentiators

These are not required by the SetFit algorithm, but they make Aprender's implementation materially stronger as a production system.

| ID | Feature | Value proposition | Complexity | Dependencies | Observable acceptance behavior |
|----|---------|-------------------|------------|--------------|--------------------------------|
| DF-01 | Pure-Rust, offline, one-file lifecycle | Train and deploy a useful few-shot classifier without Python or a mutable model directory. | High | TS-11 through TS-14 | CPU training and serving work with network disabled and no sidecars; Python is used only to generate frozen verification fixtures. |
| DF-02 | Executable proof of SetFit identity | Prevents a frozen probe from being marketed as SetFit and makes gradient fidelity auditable. | Medium-High | TS-02, TS-03, TS-07, TS-17 | Artifact/report metadata references passing named-gradient, parameter-delta, and embedding-delta evidence; benchmark tooling refuses `algorithm=setfit` without it. |
| DF-03 | Content-addressed reproducibility dossier | Lets teams reproduce or audit exactly which data, pairs, configuration, code path, and artifact produced a decision. | High | TS-09, TS-11, TS-12 | A machine-readable dossier joins dataset/model revisions, selected IDs, pair hash, resolved config, seeds, backend, artifact hash, access ledger, and metric report. |
| DF-04 | Bounded sampler with faithful semantics | Preserves SetFit's class-aware supervision while making memory behavior predictable beyond toy few-shot sets. | High | TS-04, TS-05 | A declared compatibility mode matches tiny reference pair fixtures, while the default bounded mode states its explicit pair cap/deviation and passes fixed-budget scaling tests. |
| DF-05 | Cross-surface semantic parity | One core semantic model eliminates common train/CLI/serve drift structurally. | High | TS-12 through TS-14 | The same APR and probe corpus produce equivalent tokenization, embeddings, logits, probabilities, labels, and ordering through the Rust API, real CLI, and in-process HTTP runtime. |
| DF-06 | Claims-gated TweetEval versus 9B LoRA comparison | Gives buyers an honest accuracy/resource tradeoff against Aprender's existing large-model option. | High | TS-10, TS-16 | Headline numbers are derivable from complete per-run rows; missing seeds, unequal sample IDs, test leakage, metric drift, or unequal measurement boundaries invalidate the report. |
| DF-07 | Audit-friendly explanation envelope | Provides useful operational context without pretending that linear-head scores are causal explanations. | Medium | TS-15 | Every prediction can include the full score vector, margin, preprocessing/truncation facts, artifact/config identity, and optional request audit ID in both CLI JSON and HTTP output. |
| DF-08 | Explicit CPU reference and accelerator support matrix | Keeps specialized hardware optional and makes performance claims attributable to an actual backend. | High | TS-14, TS-16, TS-17 | CPU is always supported. Each enabled accelerator passes parity and feature-combination tests, emits backend identity, and rejects an unavailable explicit request instead of relabeling CPU work as accelerated. |

## Anti-Features

These are tempting scope expansions or shortcuts that should be explicitly excluded from v1.

| ID | Anti-feature | Why avoid now | Complexity if built | Dependencies / conflict | What to do instead |
|----|--------------|---------------|--------------------|-------------------------|-------------------|
| AF-01 | Frozen linear probe or centroid classifier presented as SetFit | It omits the defining contrastive encoder update and can pass superficial accuracy/loss tests. | Low | Conflicts with TS-02, TS-03, TS-07 | Keep frozen probe and centroid results as clearly named baselines; require update evidence for the SetFit label. |
| AF-02 | InfoNCE, SupCon, CoSENT, triplet, or multiple selectable body losses in v1 | Additional objectives multiply numerical, sampling, and configuration contracts before the faithful default is stable. | High | Would fork TS-03 and TS-05 reference semantics | Ship cosine-similarity MSE first; add alternative objectives later as separately named experiments with their own fixtures. |
| AF-03 | MLP, k-NN, centroid, one-vs-rest, or end-to-end differentiable classifier as the production head | Multiple heads fragment binary/multiclass behavior, persistence, probabilities, and benchmarking. | Medium-High | Would fork TS-06, TS-07, TS-11 | Use one regularized `K`-class softmax-linear head; retain other heads only as non-SetFit baselines after the lifecycle is stable. |
| AF-04 | Multilabel, hierarchical, token-level, span, or generative classification | These tasks require different targets, losses, metrics, APIs, and artifact semantics. | Very High | Conflicts with v1 single-label TS-06/TS-10/TS-14 contracts | Support single-label binary and multiclass sentences only. |
| AF-05 | Generic compatibility with arbitrary Hugging Face/Sentence Transformers models | Shape-compatible models can differ in tokenizer, architecture, pooling, position scheme, activation, or remote-code requirements. | Very High | Dilutes TS-01 and TS-17 fail-closed contract | Support the pinned MiniLM/BERT family; add each new family through a named adapter and parity corpus. |
| AF-06 | Python, PyTorch, ONNX Runtime, or live SetFit as a production fallback | It violates the pure-Rust/offline value and creates a second behavior to serialize and serve. | High | Conflicts with DF-01 and TS-12 | Use pinned Python tooling only to generate immutable numerical fixtures. |
| AF-07 | Exhaustive Cartesian pair materialization | It can consume quadratic memory before a cap is applied and makes class imbalance accidental weighting. | Medium | Conflicts with TS-05 and TS-16 | Stream deterministic index pairs under an explicit budget; allow tiny compatibility enumeration only inside contracted limits. |
| AF-08 | Network- or sidecar-dependent serving artifact | Mutable tokenizer/config files make the served model differ from the evaluated model and break offline use. | Medium | Conflicts with TS-11 and TS-12 | Embed exact tokenizer and all model policy in one checksummed APR. |
| AF-09 | Hyperparameter search or checkpoint selection on canonical test or merged SetFit compatibility test | It leaks evaluation data and invalidates benchmark claims. | Low technically, high methodological cost | Conflicts with TS-04 and TS-10 | Select only on canonical validation, lock configuration, then evaluate canonical test once; reserve merged profile for reproduction-only labeling. |
| AF-10 | Quantization before F32 algorithm/artifact parity | Quantization can hide whether discrepancies come from SetFit, serialization, or reduced precision and can alter calibration. | High | Depends on completed TS-12 and TS-16 | Ship and benchmark F32 first; treat each quantized derivative as a distinct artifact with fresh quality/calibration evidence. |
| AF-11 | GPU-required training or broad multi-backend optimization in the first correctness slice | Backend work can obscure graph and numerical defects and excludes the small-data CPU use case. | Very High | Depends on all CPU gates TS-01 through TS-17 | Make CPU the reference and add optional accelerators only after complete parity and feature-matrix tests. |
| AF-12 | Persistent embedding cache in v1 | Stale embeddings can silently feed the head after encoder/tokenizer/pooling changes. | Medium | Requires a complete semantic fingerprint from TS-11 | Re-encode the tiny selected dataset after tuning; add caching later only when keyed by the full model and dataset fingerprint. |
| AF-13 | Token-level saliency, SHAP/LIME, counterfactual generation, or bundled nearest-example explanations | These can be expensive, misleading, privacy-sensitive, or require storing training text in the artifact. They are not SetFit requirements. | High | Requires post-v1 privacy, fidelity, and storage contracts beyond TS-15 | Provide score/margin, preprocessing, provenance, and artifact evidence in v1; research causal or exemplar explanations separately. |
| AF-14 | Automated hyperparameter search, online/continual learning, distributed training, or model registry automation | These broaden orchestration and state management before the core vertical slice is trustworthy. | Very High | Depends on stable TS-07 through TS-17 | Provide deterministic explicit configurations and machine-readable outputs suitable for later orchestration. |
| AF-15 | Vendoring TweetEval tweet text | It creates licensing and provenance risk and is unnecessary for reproducible acquisition. | Low | Conflicts with dataset provenance policy in TS-04/TS-09 | Download or mirror the pinned upstream revision on demand and record source/generated hashes. |

## Feature Dependencies

```text
TS-01 Encoder/tokenizer contract
  -> TS-02 Differentiable batched encoder
       -> TS-03 Contrastive objective
       -> TS-06 Multiclass linear head

TS-04 Typed splits + few-shot selection
  -> TS-05 Bounded pair sampler

TS-02 + TS-03 + TS-05 + TS-06
  -> TS-07 Faithful two-stage trainer
       -> TS-08 Resolved configuration
       -> TS-09 Reproducibility/provenance
       -> TS-10 Evaluation/selection
       -> TS-11 Self-contained APR
            -> TS-12 Mandatory reload parity
                 -> TS-13 Rust/CLI lifecycle
                 -> TS-14 Inference/serving
                      -> TS-15 Inspectability/explanation
                      -> TS-16 Operational benchmark

TS-17 Executable contracts gate every transition.
```

## MVP Recommendation

Build the production vertical slice in five ordered capability groups:

1. **Differentiable MiniLM conformance** — TS-01, TS-02, TS-03, and the relevant TS-17 gates. Stop unless real weights, batched masking, named gradients, and a controlled update agree with fixtures.
2. **Deterministic data and pair protocol** — TS-04, TS-05, and the sampling portions of TS-09. Freeze singleton behavior, pair-budget semantics, selected-ID manifests, and leakage rejection.
3. **Faithful trainer and head** — TS-06 through TS-10. Prove the tuned-body-to-frozen-head transition, deterministic replay, validation-only selection, stable probabilities, and explicit labels.
4. **APR production lifecycle** — TS-11 through TS-15. Reload before evaluating; make API, CLI, and serving consume the same offline artifact and expose honest inspection/explanation evidence.
5. **Benchmark and claims gate** — TS-16 plus the full TS-17 matrix. Complete all 40 shot/seed cells and compare the reloaded SetFit APR with the 9B LoRA baseline on identical IDs and declared resource boundaries.

Prioritize one pinned CPU encoder, one loss, one linear head, one artifact schema, and one benchmark. Defer alternative encoders/losses/heads, quantization, accelerator optimization, advanced explainability, multilabel tasks, caching, and orchestration until the complete CPU lifecycle passes.

## Sources

- `.planning/PROJECT.md` — milestone value, active requirements, scope exclusions, and benchmark decisions; HIGH confidence.
- `.planning/codebase/CONCERNS.md` — concrete dual-autograd, gradient-detachment, batching, tokenizer, head, metric, artifact, and reproducibility risks; HIGH confidence.
- `.planning/research/STACK.md` — recommended MiniLM/core-autograd stack, faithful loss/head behavior, APR contents, and verification strategy; HIGH confidence.
- `.planning/research/ARCHITECTURE.md` — component ownership, two-stage data flow, artifact contract, serving boundary, and build-order dependencies; HIGH confidence.
- `.planning/research/PITFALLS.md` — falsification tests, benchmark safeguards, phase gates, and deferred-risk rationale; HIGH confidence.

## Open Decisions for Phase Planning

- Freeze the singleton-class pair policy and precisely name the bounded sampler's deviation from exhaustive SetFit oversampling before implementing TS-05.
- Derive numerical tolerances from pinned fixtures during encoder conformance; do not choose them after observing Rust outputs.
- Choose the calibration statistic and uncertainty presentation during benchmark planning. Test isolation and artifact identity are required regardless of the method.
- Measure accelerator tolerances and determinism before promoting DF-08 beyond CPU; do not promise cross-device bit identity.
