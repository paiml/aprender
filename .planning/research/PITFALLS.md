# Domain Pitfalls

**Domain:** Native pure-Rust SetFit-style few-shot text classification  
**Project:** Aprender Native SetFit Classification  
**Researched:** 2026-08-07  
**Overall confidence:** HIGH for SetFit algorithm fidelity, data isolation, artifact parity, and evaluation safeguards; MEDIUM-HIGH for cross-backend determinism until Aprender's batched encoder is implemented and measured

## Roadmap Phase Vocabulary

The phase names below are recommendations for roadmap construction. A pitfall is assigned to the earliest phase that must make it impossible; later phases should retain the same invariant as a regression gate.

| Phase | Required outcome |
|-------|------------------|
| **Phase 1 — Differentiable MiniLM Conformance** | One pinned MiniLM batch matches reference tokenization, forward values, pooling, normalization, gradients, and a controlled optimizer step through `aprender-core` autograd. |
| **Phase 2 — Deterministic Pair and Data Protocol** | Few-shot selection and bounded pair sampling are deterministic, class-correct, leakage-proof, provenance-recorded, and non-quadratic. |
| **Phase 3 — Faithful Two-Stage Trainer and Head** | Encoder contrastive tuning and regularized multinomial head fitting are separate, fallible, numerically stable stages over the correct data. |
| **Phase 4 — APR Artifact and Production Parity** | One self-contained APR reproduces the trained model through core, CLI, and serving without sidecars or reconstruction. |
| **Phase 5 — Benchmark and Claims Gate** | The contracted shot/seed protocol, canonical split isolation, paired comparisons, uncertainty, resource metrics, and machine-readable evidence are enforced. |

## Critical Pitfalls

Mistakes in this section can make an implementation look successful while it is not actually SetFit, invalidate the benchmark, or force a model/runtime rewrite.

### PF-001: Encoder Tuning Is a Silent No-Op

**What goes wrong:** The contrastive stage computes embeddings by copying values through `.data()`, slice-to-vector conversions, detached pooling, or a scalar-only loss. `backward()` may run and total pipeline quality may improve because the later head learns, while the sentence encoder is byte-for-byte unchanged.

**Why it happens:** Aprender currently has two tensor/autograd stacks, inference-oriented BERT components without recursive parameter traversal, graph-detaching embedding/pooling paths, and contrastive utilities that return `f32` rather than a graph-connected tensor. A decreasing loss is therefore not evidence that encoder tuning occurred.

**Consequences:** The shipped feature is a frozen-embedding linear probe mislabeled as SetFit. Expected few-shot gains may disappear, optimizer and pair-sampler work becomes irrelevant, and aggregate tests can still pass.

**Warning signs:**

- Contrastive loss decreases but pre/post encoder tensor hashes are identical.
- Intended encoder parameters have absent, all-zero, or non-finite gradients after a non-degenerate pair batch.
- Pair similarities change only after the classifier is fit, or tuned embeddings equal frozen-baseline embeddings exactly.
- Tests assert only finite loss, output shape, parameter count, or end-to-end accuracy.

**Prevention:** Use only `aprender-core::autograd::Tensor` in the trainable forward path. Implement graph-connected batched embedding gather, masked pooling, row normalization, cosine similarity, reduction, and MSE. Enumerate parameters by stable name, derive the optimizer set from an explicit trainable allowlist, and prohibit raw-value extraction before the documented detach boundary between encoder tuning and head fitting.

**Falsification tests:**

1. On a pinned two-class MiniLM fixture with both positive and negative pairs, assert every intended trainable named tensor has a present, finite, non-zero gradient and at least one changed element after one optimizer step; assert frozen tensors do not change.
2. Re-encode fixed sentences before and after the step and require a finite, non-zero embedding delta and movement of pair cosine in the loss-reducing direction.
3. Match selected analytic gradients and the controlled optimizer step to frozen Python fixtures, then finite-difference every new differentiable primitive.
4. Add a mutation/regression test whose deliberate detach causes the gradient-reachability gate to fail even if a head can still reduce classification loss.

**Roadmap phase:** **Phase 1 is a hard stop.** Repeat the named-gradient and parameter-delta assertion inside the full trainer in Phase 3.

**Confidence:** HIGH — directly evidenced by current Aprender graph boundaries and by SetFit's required body-finetuning stage.

### PF-002: Pair or Split Leakage Inflates Few-Shot Results

**What goes wrong:** Pairs are generated before the few-shot subset is fixed, validation/test examples become endpoints, duplicate IDs or identical text cross splits, model selection reads the canonical test set, or the SetFit compatibility profile's merged validation+test split is used for tuning.

**Why it happens:** Pair expansion hides the original example boundary, and the compatibility profile intentionally exposes 346 validation-plus-test rows as one `test` split. A trainer that accepts generic paths cannot infer which accesses are legitimate.

**Consequences:** The contrastive encoder sees evaluation text or labels, reported test performance is optimistically biased, and results are not comparable to the canonical TweetEval protocol.

**Warning signs:**

- A pair endpoint is not in the selected canonical-train ID set.
- Pair generation logs more unique source IDs than `3 * shots_per_class`.
- Validation or test content hashes appear in training/pair manifests.
- Hyperparameter trials, early stopping, or checkpoint selection access the canonical test hash.
- A report names profile `setfit` but describes its score as canonical held-out test performance.

**Prevention:** Resolve and hash canonical splits first; select the balanced few-shot IDs only from canonical train; generate pairs only from those immutable IDs. Carry `id`, `source_split`, content hash, and dataset revision through sampling. Make the training orchestrator accept typed split roles rather than interchangeable file paths. Emit an access ledger, lock the selected configuration after canonical validation, and permit canonical test evaluation only from that locked run. Treat the merged compatibility profile as reproduction-only with a machine-readable leakage warning.

**Falsification tests:**

1. Assert every pair endpoint belongs to the exact selected train-ID set and that selected IDs are unique, class-balanced, and disjoint from validation/test IDs and content hashes.
2. Inject a validation/test endpoint, duplicate cross-split content, mislabeled `source_split`, and the compatibility profile into training; each must fail closed with a typed error.
3. Assert the benchmark access ledger contains no canonical-test read before a signed/hashed model-selection lock record.
4. Preserve the existing exact 587/66/280 canonical and 587/346 compatibility count falsifiers, but add an orchestrator test proving compatibility test cannot be supplied as validation.

**Roadmap phase:** **Phase 2** prevents contaminated pair construction; **Phase 5** enforces the selection/test access sequence.

**Confidence:** HIGH — the project benchmark contract explicitly defines the split boundary and warns that the compatibility profile merges validation and test.

### PF-003: Pair Generation Is Semantically Wrong or Quadratic

**What goes wrong:** The sampler emits `(x, x)`, both `(x, y)` and `(y, x)`, cross-class positives, same-class negatives, uncontrolled duplicate pairs, or an unintended positive/negative ratio. A faithful-looking oversampler may materialize all within-class and cross-class combinations before applying a cap, causing quadratic memory and startup time.

**Why it happens:** SetFit's default oversampling balances positive and negative pairs and covers possible pairs, but direct Cartesian construction is unsafe as data grows. Singleton classes and bounded sampling also require explicit semantics that a generic combination helper does not provide.

**Consequences:** The encoder receives contradictory or biased supervision; pair counts become an accidental class-weighting scheme; the implementation can exhaust memory before the advertised pair budget takes effect.

**Warning signs:**

- Pair storage or generation time scales approximately with `N^2` when the configured budget is fixed.
- The same unordered pair occurs under both labels or appears in both orientations.
- Positive/negative counts differ from the resolved policy without an explicit `unique` strategy.
- A singleton class is silently self-paired, dropped, or duplicated with itself.
- Pair count changes when class buckets are iterated through a hash map with the same seed.

**Prevention:** Build stable, sorted class buckets and sample on demand with a persisted pair budget. Canonicalize unordered ID pairs; reject self-pairs; derive labels from source classes, never caller-supplied pair labels. Define and serialize the singleton-class policy. Preserve the intended 1:1 oversampling behavior within the bounded Aprender policy, and label that bound as an explicit deviation from exhaustive SetFit oversampling rather than claiming exact pair-count equivalence.

**Falsification tests:**

1. Property-test that positives share a class, negatives do not, endpoints differ, unordered pairs cannot carry conflicting labels, and emitted counts obey the resolved strategy/budget.
2. On tiny hand-enumerated class layouts, compare all pair identities and counts with frozen SetFit reference fixtures, including imbalanced and singleton-class cases.
3. With fixed `max_pairs_per_epoch`, increase `N` by 10x and assert retained sampler state and emitted pair storage remain `O(N + budget)`, never proportional to the Cartesian product.
4. Run the same seed twice and require identical ordered pairs; run a different seed and require a changed order or selection when multiple valid samples exist.

**Roadmap phase:** **Phase 2**, before any full encoder epoch or performance promise.

**Confidence:** HIGH for pair semantics and the quadratic risk; MEDIUM-HIGH for the exact bounded compatibility policy until singleton behavior is decided and frozen.

### PF-004: Numerical Instability Produces NaNs, Collapsed Embeddings, or False Convergence

**What goes wrong:** All-padding masked means divide by zero, L2 normalization amplifies zero/tiny norms, cosine gradients become non-finite, attention masks leak padding, softmax/logistic loss overflows, or a nearly separable few-shot head drives unregularized weights toward extreme values. An optimizer may report completion despite non-finite state or failure to converge.

**Why it happens:** SetFit composes several numerically sensitive reductions in a short path, and small balanced training subsets can still yield separable embeddings. Existing Aprender probes already have empty-input and weighting failure modes.

**Consequences:** Training may collapse to one class, probabilities become exactly 0/1 or NaN, calibration is meaningless, and corrupted parameters can be serialized into an otherwise checksummed APR.

**Warning signs:**

- Any non-finite loss, gradient, parameter, embedding, logit, probability, or optimizer diagnostic.
- Normalized row norms materially differ from 1 for valid inputs, or depend on padding length.
- Probability rows do not sum to 1; weights grow while objective improvement stalls.
- L-BFGS reaches its iteration limit or line search fails but the CLI reports success.
- Class predictions collapse while contrastive pair cosines saturate near one value.

**Prevention:** Reject empty/all-masked sequences and out-of-range IDs before forward. Use checked masked denominators, explicit normalization epsilon, stable cosine/MSE reductions, log-sum-exp/shifted softmax, finite checks after every stage, F32 artifacts first, gradient-norm clipping for encoder tuning, and L2 regularization on head weights but not bias. Treat optimizer convergence as a fallible result and never serialize non-finite tensors.

**Falsification tests:**

1. Test empty text after tokenization, all-padding masks, mixed-length padding, exact/over maximum length, zero/tiny vectors, very large logits, empty datasets, and invalid class counts; require finite correct output or a typed error, never panic.
2. Check new primitive gradients by central finite differences and compare masked pooling, normalization, cosine-MSE, stable softmax, and regularized head objective with reference fixtures.
3. Assert normalized valid embeddings have norm `1 +/- tolerance`, padding does not change an example's embedding, probabilities are finite and sum to one, and regularized loss decreases on a controlled head fixture.
4. Inject NaN/Inf into each artifact tensor class and require save/load validation to reject it.

**Roadmap phase:** Encoder math is blocked in **Phase 1**; head stability and fallible convergence are blocked in **Phase 3**; APR finite-value validation is retained in **Phase 4**.

**Confidence:** HIGH — the operations and current failure paths are known; exact tolerances require the Phase 1 fixtures.

### PF-005: Training and Serving Implement Different Models

**What goes wrong:** Training, CLI evaluation, and serving disagree on tokenizer normalization/special tokens, truncation length, attention masks, pooling, L2 normalization, encoder configuration, label order, head parameters, or dropout/eval mode. Evaluation may use the live in-memory model while serving reconstructs a similar but non-identical model from sidecars or manual flags.

**Why it happens:** Aprender currently duplicates tokenizer and pooling logic, supplies BERT dimensions manually at CLI boundaries, and separates training and inference encoder stacks.

**Consequences:** The benchmark evaluates a model users never receive. Labels can swap even when logits match, batch and single prediction can disagree, and deployments depend on missing Hub files or mutable configuration.

**Warning signs:**

- In-memory and reloaded APR embeddings/logits differ beyond documented tolerance.
- Batch-one and mixed-length batched predictions differ for the same text.
- Serving needs tokenizer/config flags or network access not contained in the APR.
- Tensor names or label indices are remapped during load.
- CLI evaluation passes on a training checkpoint but not on the final APR.

**Prevention:** Make one typed `setfit-apr-v1` model own encoder tensors, exact tokenizer bytes/hash, validated architecture, max length, truncation/padding, masked pooling, normalization epsilon, multinomial head, and ordered typed labels. Core, CLI, and serve must call the same loader/model implementation. Evaluate the serialized-and-reloaded APR, not only the pre-save object.

**Falsification tests:**

1. Train-save-load-predict and require exact F32 tensor bytes/config/tokenizer bytes plus embedding, logit, probability, and label parity within documented numerical tolerances.
2. Load the same APR through core, real CLI, and in-process serving; compare single and batched results on empty/short/padded/truncated Unicode-heavy fixtures.
3. Remove network access and all sidecars; the APR must still inspect, evaluate, predict, and serve.
4. Permute label metadata, pooling policy, tokenizer hash, tensor name/shape, or max length and require load to fail before model allocation or prediction.

**Roadmap phase:** **Phase 4**. Serving integration must not start from an alternate model reconstruction.

**Confidence:** HIGH — current duplication is documented and the single-APR requirement is explicit.

### PF-006: “Seeded” Runs Are Still Non-Deterministic

**What goes wrong:** Only few-shot selection is seeded while pair sampling, pair order, batch shuffling, dropout, head initialization, or backend kernels use ambient randomness. Hash-map iteration or parallel scheduling changes order. Conversely, an ignored seed can make different-seed runs suspiciously identical.

**Why it happens:** SetFit has multiple stochastic stages and a single public seed does not automatically control them. Optional accelerators may use different reduction orders even with deterministic input.

**Consequences:** Runs cannot be reproduced, seed comparisons confound data and optimizer changes, artifacts cannot be audited, and benchmark variance is unreliable.

**Warning signs:**

- Same configuration/seed/data yields different selected IDs, ordered pairs, batch order, semantic tensor hash, or CPU predictions.
- Different seeds yield identical sampled IDs and pair order despite available alternatives.
- Output changes with hash-map insertion order or worker count.
- APR metadata records one seed but cannot identify derived RNG streams or backend.

**Prevention:** Derive domain-separated ChaCha streams from the run seed for subset selection, pair sampling, shuffle, dropout, and initialization. Sort IDs/classes before sampling; never depend on map iteration. Persist root and derived seed identifiers, sample/pair hashes, resolved configuration, thread/backend information, and deterministic-algorithm policy. Define CPU as the byte/semantic reproducibility reference; require accelerator parity within tolerance rather than promising cross-device bit identity.

**Falsification tests:**

1. Run two clean CPU trainings with identical inputs and compare selected IDs, ordered pair manifest, step count, loss trace, tensor/config/tokenizer semantic hashes, and predictions. Exclude only explicitly declared volatile metadata such as creation time.
2. Repeat with 1 and multiple data-loader workers/threads and require identical logical order and result under the deterministic CPU profile.
3. Change only the root seed and require the selected subset or pair order to change when combinatorially possible; record the exact causal difference.
4. For each optional accelerator, compare outputs/metrics against CPU within a documented tolerance and emit backend identity; never silently fall back when acceleration was explicitly requested.

**Roadmap phase:** RNG and stable-order contracts begin in **Phase 2**; full replay is mandatory in **Phase 3** and recorded in Phase 4 artifacts; **Phase 5** verifies benchmark reruns.

**Confidence:** HIGH for CPU policy and stochastic-stage coverage; MEDIUM for accelerator tolerances until kernels are measured.

### PF-007: Few-Shot Statistics Hide Seed Sensitivity

**What goes wrong:** A report presents the best or a single seed, treats pair duplicates as additional labeled examples, compares models on different sampled IDs, pools predictions across runs, or reports a confidence interval over test rows while ignoring training/subset variability. The sampling seed may cover optimizer order but not the labeled subset.

**Why it happens:** With 8–64 examples per class, subset composition is part of the experiment and often dominates small score differences. Pair expansion can create the illusion of a larger independent sample.

**Consequences:** Rankings can reverse across seeds, claimed gains are not attributable to the method, and a point estimate overstates evidence.

**Warning signs:**

- Fewer than ten contracted seeds or missing per-seed rows.
- A run has other than exactly `shots_per_class` unique train IDs for each of `none`, `against`, and `favor`.
- SetFit and LoRA reports cannot prove identical sample IDs for the same shot/seed cell.
- Only mean/best score is shown; dispersion and paired deltas are absent.
- Pair count is described as the number of labeled examples.

**Prevention:** Execute the complete 8/16/32/64-per-class by ten-seed matrix from the benchmark contract. Use the same immutable sampled-ID manifest for every comparator in a cell. Report every run, mean, standard deviation, median/range, and paired SetFit-minus-baseline deltas; keep test-example uncertainty distinct from between-seed/subset uncertainty. State labeled examples as unique source rows and pair examples separately.

**Falsification tests:**

1. Validate 40 run cells, exact contracted seeds, exact per-class unique counts, no replacement, and one shared sampled-ID hash across all methods in a cell.
2. Recompute aggregates from machine-readable per-run rows and fail if headline values cannot be derived exactly or if best-seed selection is used.
3. Shuffle run-row order and require unchanged aggregates; drop any seed and require the completeness gate to fail.
4. Report paired deltas only after joining on shot, seed, dataset revision, and sampled-ID hash; reject unpaired comparisons.

**Roadmap phase:** The sampling manifest is created in **Phase 2**; the full statistical contract is a release blocker in **Phase 5**.

**Confidence:** HIGH — the project contract already fixes shots, balanced sampling, ten seeds, and shared IDs.

### PF-008: Performance Claims Use the Wrong Metric or an Unequal Comparison

**What goes wrong:** A frozen probe is called SetFit; accuracy is headlined on the imbalanced stance test; `none` is incorrectly included in or substituted for official `F_avg`; label order is implicit; the best test checkpoint is selected post hoc; SetFit and 9B LoRA use different IDs, hardware boundaries, or timing definitions; only quality is reported while memory/model size/latency are omitted.

**Why it happens:** “SetFit” and “few-shot” are easy labels to apply to superficially similar pipelines, and one scalar metric conceals class, calibration, seed, and resource tradeoffs.

**Consequences:** Results are technically incomparable or misleading even if every recorded number is computed correctly.

**Warning signs:**

- No proof of non-zero encoder updates before the `SetFit` name is used.
- Headline metric is accuracy, generic macro-F1, or an unlabeled `F1` rather than explicit `F_avg = (F1_against + F1_favor)/2`.
- The report omits per-class confusion/F1, calibration, seeds, IDs, provenance, or validation-selection record.
- Training time includes download/import for one method but not another, or throughput uses different batch/hardware settings.
- Quality comes from the in-memory checkpoint while size/latency come from a derivative or quantized artifact.

**Prevention:** Reserve `SetFit` for a graph-proven contrastively tuned encoder followed by a learned head; name frozen-encoder results `linear probe`. Make official `F_avg` the primary stance metric with explicit label mapping, while also publishing three-class macro-F1, per-class metrics, MCC, calibration, confidence intervals, and confusion diagnostics. Benchmark the same reloaded APR used for predictions, define measurement boundaries, record warmup/batch/hardware/backend, and publish quality, train time, latency/throughput, peak memory, artifact size, and calibration together.

**Falsification tests:**

1. Require the machine-readable report to reference passing encoder-gradient/update evidence before accepting algorithm=`setfit`.
2. Recompute `F_avg` from the stored confusion matrix and explicit labels `against` and `favor`; permute class indices while preserving the label map and require the same score.
3. Reject reports missing any shot/seed cell, sampled-ID hash, dataset/model revision, selection lock, artifact hash, resource field, or per-class diagnostics.
4. Run all comparators through one benchmark harness and assert identical sample manifests and declared timing/memory boundaries; mark quantized derivatives as separate models with separate quality/calibration.

**Roadmap phase:** **Phase 5**, backed by evidence emitted by Phases 1–4.

**Confidence:** HIGH — SetFit's two-stage definition and TweetEval's stance metric are both authoritative and the project comparison requirements are explicit.

## Moderate Pitfalls

### PF-009: The Head Is Fit on the Wrong Rows or with Incorrect Weighting

**What goes wrong:** The multinomial head is trained on pair-expanded endpoints rather than each original labeled example once, so frequently sampled sentences receive accidental weight. Alternatively, the existing weighted `LinearProbe` applies each logit's class weight instead of scaling the entire sample gradient.

**Warning signs:** Head-fit row count equals pair endpoints, repeated text receives different multiplicity, class weighting changes non-target logit gradients inconsistently, or binary and multiclass paths produce different artifact semantics.

**Prevention:** After encoder tuning, switch to eval/no-grad, encode each unique selected training example exactly once, and fit one regularized K-class softmax head for all `K >= 2`. Do not reuse the current manual-SGD `LinearProbe`. If class/sample weights are exposed, multiply the complete sample loss and every logit gradient by the target sample's weight.

**Falsification tests:** Assert head row IDs equal the selected unique training IDs exactly; duplicate pair frequency must not alter the head dataset. Compare weighted loss/gradients and probabilities with a fixed reference, test label-permutation equivariance, and require binary to be the K=2 softmax case.

**Roadmap phase:** **Phase 3**.

**Confidence:** HIGH — SetFit's classifier stage consumes embeddings of original examples, and Aprender's current weighting defect is documented.

### PF-010: Stale or Stochastic Embeddings Feed the Head

**What goes wrong:** A cache created before contrastive tuning is reused after encoder weights change, or head embeddings are generated with dropout/training mode active. The classifier then sees frozen or non-repeatable representations despite a correctly tuned body.

**Warning signs:** Cache key omits encoder tensor/config/tokenizer hashes; repeated head-embedding passes differ; deleting the cache changes quality; tuned and frozen head inputs are identical while encoder tensors differ.

**Prevention:** Invalidate embeddings on any encoder/tokenizer/pooling/normalization change. Prefer no persistent embedding cache in v1; if used, key it by complete semantic model fingerprint and dataset ID hash. Explicitly transition the body to eval/no-grad after tuning and record that boundary.

**Falsification tests:** Modify one encoder tensor, tokenizer byte, max length, or pooling value and require a cache miss. Encode the head dataset twice in eval mode and require parity; force train mode and require the trainer to reject head fitting.

**Roadmap phase:** **Phase 3**; artifact cache identity is retained in **Phase 4**.

**Confidence:** HIGH.

### PF-011: Model or Tokenizer “Compatibility” Is Overclaimed

**What goes wrong:** An arbitrary Sentence Transformers/Hugging Face model is accepted because tensor shapes mostly load, despite unsupported architecture, pooling module, activation, position scheme, tokenizer behavior, or remote code.

**Warning signs:** Import succeeds with missing/ignored tensors, caller-supplied dimensions override source configuration, numerical parity is only shape-tested, or a model outside the contracted MiniLM family is advertised as supported.

**Prevention:** Contract v1 to the pinned `all-MiniLM-L6-v2` revision and its exact Transformer -> masked-mean Pooling -> Normalize graph. Validate the complete config, tokenizer, module graph, tensor names/shapes/dtypes, revision, and file hashes before allocation. New families require a named adapter and their own parity corpus.

**Falsification tests:** Mutate architecture, layer/head counts, activation, position scheme, pooling, normalization, tokenizer special IDs, tensor set, or revision and require a typed import error. Run real-weight token/embedding/cosine parity fixtures, not shape-only tests.

**Roadmap phase:** **Phase 1** for the first encoder; **Phase 4** revalidates the persisted contract on load.

**Confidence:** HIGH.

### PF-012: Label Vocabulary Drift Corrupts Probabilities and Metrics

**What goes wrong:** Training, APR, CLI, and serving infer label order independently; `none`, `against`, and `favor` logits are swapped, or a binary sigmoid shortcut leaks into the three-class path. Scores can look plausible while representing the wrong classes.

**Warning signs:** Labels are stored as an unordered map, class indices are accepted without typed label metadata, loaded probabilities have changed column meaning, or `F_avg` changes under a pure internal index permutation.

**Prevention:** Persist one ordered, type-tagged label vocabulary with the head. Bind every probability column, confusion-matrix row/column, and metric selector to that vocabulary. Reject missing, duplicate, unknown, or count-mismatched labels.

**Falsification tests:** Round-trip ordered labels; permute internal indices and require semantic prediction/metrics invariance through explicit remapping; use hand-authored confusion matrices to prove `F_avg` selects only `against` and `favor`; reject a three-class artifact routed through binary scoring.

**Roadmap phase:** **Phase 3** defines it, **Phase 4** persists it, and **Phase 5** recomputes metrics from it.

**Confidence:** HIGH.

## Minor Pitfalls

### PF-013: Calibration Is Claimed from Too Little or Reused Test Data

**What goes wrong:** Confidence quality is inferred from accuracy/F1, calibration parameters are fitted on the canonical test split, or a tiny validation set is summarized by one unstable scalar without bin/support details.

**Warning signs:** No reliability-bin counts, calibration method/source split is absent, test labels are consumed before final scoring, or post-hoc calibration changes the artifact after its reported predictions were generated.

**Prevention:** Treat calibration as a separately measured property. If a calibrator is added, fit/select it only on canonical validation and serialize it as part of the evaluated APR. Report ECE/Brier-style diagnostics with bin counts and acknowledge uncertainty; do not promise well-calibrated probabilities merely because the head is logistic regression.

**Falsification tests:** Assert calibrator fit IDs are validation-only, save/load includes calibrator state, probabilities used for calibration metrics come from the final APR, and missing/undersupported bins remain visible rather than silently omitted.

**Roadmap phase:** **Phase 5**; Phase 4 is involved only if calibration transforms become artifact state.

**Confidence:** MEDIUM-HIGH — the data-isolation rule is firm; the exact calibration method remains a roadmap choice.

### PF-014: Throughput Improvements Change Semantics

**What goes wrong:** Batched padding changes embeddings, length bucketing reorders labels, tokenization caches cross tokenizer/model revisions, or a requested GPU path silently falls back to CPU while retaining an accelerator performance label.

**Warning signs:** Batch-one and batched outputs differ, result order follows buckets rather than input IDs, warm/cold cache results differ, or runtime metadata does not prove the backend engaged.

**Prevention:** Make batching an equivalence-preserving transformation with stable IDs/order and mask-aware pooling. Key caches by full semantic fingerprint. Require explicit backend identity and fail an explicit accelerator request if unavailable. Benchmark semantics and mechanism before speed.

**Falsification tests:** Compare batch sizes 1 and greater than 1 across mixed lengths; shuffle bucket boundaries and require the same ID-keyed results; mutate cache identity components; verify backend trace plus numerical parity before reporting accelerator throughput.

**Roadmap phase:** Batched semantics begin in **Phase 1**, trainer ordering in **Phase 3**, and production/backend claims in **Phases 4–5**.

**Confidence:** HIGH for batching/caching risks; MEDIUM for backend-specific behavior until implemented.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Required exit gate |
|-------------|----------------|--------------------|
| Phase 1 — Differentiable MiniLM Conformance | A shape-correct encoder silently detaches or only matches batch size one | Real-weight tokenizer/forward/pooling/normalization fixtures; named finite non-zero gradients; controlled parameter and embedding update; batch/padding equivalence; finite-difference primitives |
| Phase 2 — Deterministic Pair and Data Protocol | Pair labels, balance, uniqueness, singleton behavior, memory bound, or split provenance are wrong | Property/reference sampler suite; exact selected-ID manifest; train-only endpoints; stable replay; fixed-budget scaling test; compatibility-profile rejection for tuning |
| Phase 3 — Faithful Two-Stage Trainer and Head | Head improvement masks encoder failure; stale/dropout embeddings or pair multiplicity contaminate head fit | Full trainer named-update assertion; explicit train->eval/no-grad boundary; one embedding per unique train row; stable regularized K-class reference parity; deterministic replay |
| Phase 4 — APR Artifact and Production Parity | The served model is reconstructed differently from the evaluated model | Exact tensor/config/tokenizer/label round trip; in-memory vs reloaded parity; core/CLI/serve contract; offline load; malformed-artifact fail-closed suite |
| Phase 5 — Benchmark and Claims Gate | Test leakage, seed cherry-picking, wrong `F_avg`, unequal baselines, or incomplete resource evidence | 40-cell shot/seed completeness; paired sampled-ID hashes; validation lock before test; recomputed metrics/aggregates; per-seed uncertainty; artifact/resource provenance schema |

## Sources

### Authoritative Algorithm and Model Sources

- [SetFit conceptual guide](https://huggingface.co/docs/setfit/en/conceptual_guides/setfit) — defines a SetFit model as a sentence-transformer body plus classifier head trained in separate embedding-finetuning and classifier phases; HIGH confidence.
- [SetFit sampling strategies](https://huggingface.co/docs/setfit/en/conceptual_guides/sampling_strategies) — defines self-pair/orientation exclusions and oversampling, undersampling, unique, and deprecated `num_iterations` behavior; HIGH confidence.
- [SetFit trainer reference](https://huggingface.co/docs/setfit/reference/trainer) — current defaults and semantics for sampling, cosine-similarity loss, learning rates, epochs, warmup, max length, seeds, and model reinitialization; HIGH confidence.
- [Original SetFit paper](https://arxiv.org/abs/2209.11055) — establishes contrastive Siamese body tuning followed by a classification head and the few-shot comparison context; HIGH confidence.
- [Sentence Transformers `CosineSimilarityLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) — defines cosine comparison to pair labels with MSE by default; HIGH confidence.
- [`all-MiniLM-L6-v2` model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — documents attention-mask-aware mean pooling, L2 normalization, and default 256-wordpiece truncation; HIGH confidence.

### Benchmark Sources

- [TweetEval paper](https://aclanthology.org/2020.findings-emnlp.148/) — defines TweetEval's standardized task evaluation; its stance exception averages F1 for `favor` and `against`; HIGH confidence.
- `contracts/tweet-eval-stance-benchmark-v1.yaml` — project-authoritative split sizes, label map, official score invariant, balanced shot counts, ten seeds, and compatibility-profile warning; HIGH confidence.
- `docs/examples/tweet-eval-stance.md` — project workflow and reporting requirements; HIGH confidence.

### Repository Evidence

- `.planning/PROJECT.md` — milestone algorithm, reproducibility, artifact, and benchmark requirements; HIGH confidence.
- `.planning/codebase/CONCERNS.md` — concrete graph detachment, dual-stack, pair explosion, tokenizer/config drift, head-weighting, metric, batching, and round-trip risks; HIGH confidence.
- `.planning/codebase/TESTING.md` — available contract, property, mutation, CLI, E2E, numerical tolerance, benchmark, and CI patterns; HIGH confidence.
- `.planning/research/STACK.md` — canonical core-autograd path, pinned MiniLM contract, bounded sampler, regularized multinomial head, APR schema, and reference-fixture strategy; HIGH confidence.

## Research Gaps to Carry into Planning

- Freeze the exact singleton-class sampler behavior and bounded-oversampling compatibility rule in Phase 2; exhaustive SetFit oversampling and a strict memory cap cannot both be claimed without defining the deviation.
- Measure CPU reproducibility across thread counts and establish backend-specific tolerances before promising GPU reproducibility.
- Choose the calibration estimator and uncertainty presentation during Phase 5 planning; test isolation and artifact identity are required regardless of method.
- Define numerical tolerances from the pinned Python fixtures rather than selecting them after observing Rust discrepancies.

