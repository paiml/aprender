# Codebase Concerns

**Analysis Date:** 2026-08-07

## Tech Debt

**Parallel encoder and autograd stacks:**
- Issue: Sentence-encoder functionality is split across two substantially different implementations. The CLI-facing BERT inference path uses `aprender-core` types in `crates/aprender-core/src/models/bert/`, while trainable encoder and classifier code uses the legacy `entrenar` library surface in `crates/aprender-train/src/transformer/` and `crates/aprender-train/src/finetune/`.
- Files: `crates/aprender-core/src/models/bert/mod.rs`, `crates/aprender-core/src/models/bert/encoder.rs`, `crates/aprender-train/src/transformer/encoder.rs`, `crates/aprender-train/src/autograd/`, `crates/aprender-core/src/autograd/`
- Impact: A SetFit pipeline cannot reuse `apr embed` for training without crossing incompatible tensor, loader, parameter, and optimizer APIs. Fixes made to inference pooling or BERT loading can drift from the training encoder without compiler enforcement.
- Fix approach: Select one canonical BERT/sentence-encoder implementation, expose it through a shared trainable module abstraction, and make both CLI inference and SetFit training call that implementation. Treat adapters between tensor stacks as temporary and cover them with numerical parity tests.

**BERT implementation is explicitly inference-only:**
- Issue: The BERT module declares training and batched scoring out of scope. `BertEmbeddings`, `BertLayer`, `BertEncoder`, and `CrossEncoder` do not implement the `Module` parameter enumeration used by optimizers.
- Files: `crates/aprender-core/src/models/bert/mod.rs`, `crates/aprender-core/src/models/bert/embeddings.rs`, `crates/aprender-core/src/models/bert/layer.rs`, `crates/aprender-core/src/models/bert/encoder.rs`, `crates/aprender-core/src/models/bert/cross_encoder.rs`, `crates/aprender-core/src/nn/module.rs`
- Impact: There is no supported way to enumerate, freeze, unfreeze, zero, update, or serialize all BERT parameters for the contrastive encoder phase of SetFit.
- Fix approach: Implement `Module` recursively for every BERT component, add named parameter groups, and provide explicit freeze policies for embeddings, encoder layers, pooler, and classifier. Require a test that every intended trainable tensor receives a finite non-zero gradient and changes after an optimizer step.

**Embedding and pooling operations detach the graph:**
- Issue: BERT embedding lookup reads parameter data into a new tensor, and sentence pooling copies hidden-state slices into new vectors/tensors. Setting a new tensor's gradient flag does not connect it to the source operation.
- Files: `crates/aprender-core/src/models/bert/embeddings.rs`, `crates/aprender-core/src/models/bert/cross_encoder.rs`, `crates/apr-cli/src/commands/embed.rs`, `crates/aprender-train/src/transformer/embedding.rs`, `crates/aprender-train/src/transformer/encoder.rs`, `crates/aprender-train/src/finetune/classification.rs`
- Impact: A superficially successful loss/backward call can leave token embeddings and upstream encoder parameters frozen. This is a silent failure mode for SetFit because the classifier head can still learn and make aggregate loss decrease.
- Fix approach: Add differentiable gather/index-select and masked pooling primitives to the canonical autograd engine. Prohibit `.data()` extraction inside a trainable forward path except at an explicit detach boundary.

**Contrastive losses are scalar utilities, not trainable losses:**
- Issue: `InfoNCE`, triplet loss, `ContrastiveTask`, and `SimCSE` consume `Vector<f32>` or nested `Vec<f32>` values and return `f32`. They do not return an autograd tensor or expose gradients.
- Files: `crates/aprender-core/src/loss/loss.rs`, `crates/aprender-core/src/loss/mod.rs`, `crates/aprender-core/src/nn/self_supervised.rs`, `crates/aprender-core/src/nn/self_supervised_byol_simcse.rs`, `crates/aprender-train/src/aprender_compat.rs`
- Impact: Existing contrastive APIs are useful for evaluation but cannot train a sentence encoder. Reusing their names in SetFit code could falsely imply that encoder fine-tuning is occurring.
- Fix approach: Implement a tensor-valued cosine-similarity/contrastive loss on the canonical autograd graph, including stable normalization, temperature validation, masked in-batch negatives, and a finite-difference gradient check.

**Classifier implementations are fragmented:**
- Issue: `aprender-core` logistic regression is binary-only while multiclass heads live separately as `ClassificationHead` and `LinearProbe` in `aprender-train`. Softmax regression is documented as planned in the core classifier module.
- Files: `crates/aprender-core/src/classification/mod.rs`, `crates/aprender-train/src/finetune/classification.rs`, `crates/aprender-train/src/finetune/linear_probe.rs`
- Impact: SetFit's binary and multiclass cases would follow different APIs and persistence formats. TweetEval stance is three-class, so the core logistic regression cannot serve as its final head.
- Fix approach: Define one fallible multiclass linear-classifier API with probability prediction, class weighting, regularization, calibration metadata, and deterministic serialization. Reuse it for binary classification as the two-class case.

**Tokenizer logic is duplicated at command boundaries:**
- Issue: `apr embed` intentionally inlines vocab and tokenizer JSON loading rather than sharing the reranker implementation, while `aprender-train` has its own `HfTokenizer` and tokenizer traits.
- Files: `crates/apr-cli/src/commands/embed.rs`, `crates/apr-cli/src/commands/rerank.rs`, `crates/aprender-train/src/tokenizer/traits.rs`, `crates/aprender-train/src/tokenizer/hf.rs`, `crates/aprender-train/src/finetune/classify_pipeline/mod.rs`
- Impact: Training and inference can disagree about normalization, special tokens, truncation, added tokens, or tokenizer JSON features. Sentence-transformer parity is highly sensitive to those details.
- Fix approach: Use one tokenizer implementation and one encoded-example structure containing IDs, token types, attention mask, and provenance. Persist tokenizer identity and a content hash with every trained classifier.

**Model configuration is supplied manually instead of derived from the artifact:**
- Issue: `apr embed` accepts hidden size, layer count, head count, intermediate size, vocabulary size, and position limits as independent CLI values, then constructs `BertConfig` before loading tensors.
- Files: `crates/apr-cli/src/extended_commands.rs`, `crates/apr-cli/src/commands/embed.rs`, `crates/aprender-core/src/models/bert/config.rs`, `crates/aprender-core/src/models/bert/load.rs`
- Impact: A valid model can be rejected or interpreted with the wrong architecture. SetFit checkpoints may become non-reproducible if the training command and inference command use different overrides.
- Fix approach: Persist and load an authoritative encoder configuration from APR metadata. Allow overrides only behind an explicit unsafe/debug option and verify every override against tensor shapes.

**Data CLI contains local copies of upstream data-library logic:**
- Issue: The CLI documents several implementations as local stubs for APIs not yet published by `alimentar`, including text statistics and resampling.
- Files: `crates/apr-cli/src/commands/data.rs`, `crates/aprender-data/src/quality/`, `crates/aprender-data/src/imbalance.rs`
- Impact: Dataset behavior can diverge between the CLI and library. A SetFit workflow that audits or balances data through the CLI may not match programmatic training behavior.
- Fix approach: Move all data transformations to `aprender-data`, publish a compatible version, and keep `apr data` as argument parsing plus presentation only.

**Legacy package identity remains visible:**
- Issue: The package is named `aprender-train` but its library name is `entrenar`, its repository metadata points to the former repository, and CLI feature wiring refers to `entrenar`.
- Files: `crates/aprender-train/Cargo.toml`, `crates/apr-cli/Cargo.toml`, `crates/aprender-train/src/lib.rs`
- Impact: New SetFit modules can be placed in the wrong crate or imported under inconsistent names. Documentation, error messages, and crates.io dependency resolution remain harder to reason about.
- Fix approach: Establish the supported public package/import name, provide a bounded compatibility alias, and update examples and generated metadata together.

## Known Bugs

**Weighted linear-probe gradient applies the wrong class weight:**
- Symptoms: The loss is multiplied by the target class weight, but each logit gradient is multiplied by its own class weight. For weighted cross-entropy, every logit gradient for a sample must be scaled by the target class weight.
- Files: `crates/aprender-train/src/finetune/linear_probe.rs`
- Trigger: Train `LinearProbe` or `MlpProbe` with non-uniform `class_weights`; this is especially relevant to imbalanced TweetEval stance data.
- Workaround: Do not enable probe class weights until the gradient is corrected. Prefer balanced sampling and verify against a reference implementation.

**Empty probe training returns non-finite loss:**
- Symptoms: `LinearProbe::train` and `MlpProbe::train` divide epoch loss by `n` without rejecting `n == 0`; with one or more epochs the result is `NaN`.
- Files: `crates/aprender-train/src/finetune/linear_probe.rs`
- Trigger: Pass empty embeddings and labels with `epochs > 0`.
- Workaround: Validate non-empty training and validation splits before constructing the probe.

**Bootstrap confidence intervals panic on empty inputs or zero iterations:**
- Symptoms: Empty predictions lead to modulo by zero during resampling; `n_bootstrap == 0` underflows the upper index and indexes an empty vector.
- Files: `crates/aprender-train/src/finetune/linear_probe.rs`
- Trigger: Call `bootstrap_mcc_ci` with no samples or zero bootstrap iterations.
- Workaround: Enforce both counts as positive at the caller and return a structured error for invalid evaluation requests.

**Long or out-of-range BERT input panics instead of returning an error:**
- Symptoms: `BertEmbeddings::forward` asserts on excessive length and uses unchecked slice ranges for word and token-type IDs. `apr embed` does not truncate or validate the encoded sequence before calling it.
- Files: `crates/aprender-core/src/models/bert/embeddings.rs`, `crates/apr-cli/src/commands/embed.rs`
- Trigger: Encode text longer than `max_position_embeddings`, supply a tokenizer whose IDs exceed `vocab_size`, or supply an invalid token-type ID.
- Workaround: Pre-tokenize and reject/truncate inputs to the model limit; verify tokenizer vocabulary size and special-token IDs before training.

**Training encoder silently accepts invalid token and position ranges:**
- Symptoms: Out-of-vocabulary token IDs become zero vectors after only a warning, and positions beyond the configured maximum repeat the final position embedding.
- Files: `crates/aprender-train/src/transformer/embedding.rs`, `crates/aprender-train/src/transformer/encoder.rs`
- Trigger: Use a mismatched tokenizer or overlength sample in `EncoderModel::forward`.
- Workaround: Validate examples before the forward pass and fail closed. Do not use zero-vector or repeated-position fallback in benchmark training.

**Micro-averaged classification metrics are reported as macro averages:**
- Symptoms: The `Average::Micro` branch explicitly delegates to macro averaging; tests encode that fallback rather than the standard micro formula.
- Files: `crates/aprender-train/src/eval/classification/metrics.rs`, `crates/aprender-train/src/eval/classification/basic_tests.rs`
- Trigger: Request micro precision, recall, or F1 on an imbalanced multiclass dataset.
- Workaround: Use per-class metrics, explicit class-index averaging, or compute micro totals directly until the implementation is corrected.

**DataLoader can silently truncate an epoch:**
- Symptoms: Missing rows are removed with `filter_map`, batch concatenation errors become `None`, and iterator consumers cannot distinguish end-of-data from a failed row or batch.
- Files: `crates/aprender-data/src/dataloader.rs`, `crates/aprender-data/src/dataset.rs`
- Trigger: A custom `Dataset::get` returns `None` for an in-range row or Arrow concatenation fails.
- Workaround: Validate dataset integrity before iteration and count consumed rows. A SetFit loader should yield `Result<Batch>` and fail on missing examples.

**Two-label cross-encoder scoring uses only the first logit:**
- Symptoms: `CrossEncoder::new` supports `num_labels > 1`, but `score` reads element zero and applies sigmoid rather than softmax over the logits.
- Files: `crates/aprender-core/src/models/bert/cross_encoder.rs`
- Trigger: Load a two-logit binary classification/reranking head and call `score`.
- Workaround: Call `forward` and interpret logits according to model metadata; reserve `score` for verified single-logit heads.

## Security Considerations

**Unbounded model and configuration allocations:**
- Risk: User-controlled model files and dimension flags can cause very large allocations, integer-product overflow, or process termination before shape validation.
- Files: `crates/apr-cli/src/commands/embed.rs`, `crates/aprender-core/src/models/bert/config.rs`, `crates/aprender-core/src/models/bert/embeddings.rs`, `crates/aprender-core/src/models/bert/encoder.rs`
- Current mitigation: APR parsing and tensor element-count checks reject many malformed artifacts after construction begins.
- Recommendations: Validate dimension products with `checked_mul`, impose configurable byte/parameter/sequence limits, inspect artifact metadata before allocating the model, and convert all panics to structured errors at CLI boundaries.

**Whole-file parsing permits local denial of service:**
- Risk: Models, tokenizer JSON, text files, and several JSONL corpora are read entirely into memory. A large or malicious local input can exhaust memory.
- Files: `crates/apr-cli/src/commands/embed.rs`, `crates/aprender-train/src/finetune/classification.rs`, `crates/aprender-data/src/dataset.rs`
- Current mitigation: File I/O errors are generally converted to structured errors.
- Recommendations: Use memory mapping or streaming APR reads, parse JSONL incrementally, cap tokenizer/model metadata sizes, and report limits before training begins.

**Predictable shared temporary resampling file:**
- Risk: Resampling writes to the fixed path `apr-resample-tmp.jsonl` under the system temporary directory. Concurrent processes can clobber one another, and an attacker with access to the same temporary directory may pre-create or redirect the path.
- Files: `crates/apr-cli/src/commands/data.rs`
- Current mitigation: The file is removed after reloading, but creation is not unique or atomic.
- Recommendations: Use `tempfile::NamedTempFile` or a private temporary directory and retain the handle through reload.

**Downloaded benchmark content is recorded but not checked against an authored digest:**
- Risk: TweetEval downloads are pinned to a full revision and hashed after download, but the code verifies expected row/class counts rather than an allowlist of known file digests.
- Files: `crates/apr-cli/src/commands/data_tweeteval.rs`, `contracts/tweet-eval-stance-benchmark-v1.yaml`
- Current mitigation: HTTPS, a full commit SHA, exact file paths, exact sample counts, exact class counts, and generated SHA-256 provenance substantially limit drift.
- Recommendations: Add expected source digests to the benchmark contract for the canonical revision and verify them before emitting training data.

## Performance Bottlenecks

**Vocabulary is parsed once per input text:**
- Problem: `tokenize_single` calls `load_vocab`, rebuilds a `HashMap`, and constructs a new `WordPieceTokenizer` for every text.
- Files: `crates/apr-cli/src/commands/embed.rs`
- Cause: The tokenizer is scoped to the single-text helper instead of the command run.
- Improvement path: Load and validate one tokenizer before the loop, then encode all texts through it. Cache tokenized examples across SetFit epochs and pair-generation passes.

**Sentence encoding is strictly batch size one:**
- Problem: `BertEmbeddings::forward` documents an implicit batch of one and `apr embed` runs a complete encoder forward serially for each string.
- Files: `crates/aprender-core/src/models/bert/embeddings.rs`, `crates/aprender-core/src/models/bert/encoder.rs`, `crates/apr-cli/src/commands/embed.rs`
- Cause: There is no padded batch representation or attention-mask-aware pooling path.
- Improvement path: Introduce batched IDs, token types, and masks; bucket by sequence length; implement masked mean pooling; and benchmark throughput at realistic SetFit batch sizes.

**APR sentence model load has high peak memory:**
- Problem: `apr embed` reads the complete model into a `Vec<u8>`, constructs zero/initialized BERT tensors, then replaces them with copied f32 tensors from the reader.
- Files: `crates/apr-cli/src/commands/embed.rs`, `crates/aprender-core/src/models/bert/embeddings.rs`, `crates/aprender-core/src/models/bert/layer.rs`, `crates/aprender-core/src/models/bert/load.rs`
- Cause: The loader is copy-based and model construction allocates full-sized placeholders before validation/load.
- Improvement path: Validate tensor metadata first, construct modules directly from loaded tensors, and use mapped/borrowed storage where the tensor backend permits it.

**CLI retains every text and embedding until completion:**
- Problem: All input lines are cloned into memory and all result vectors are collected before output.
- Files: `crates/apr-cli/src/commands/embed.rs`
- Cause: JSON and text formatting operate on a complete `Vec<(String, Vec<f32>)>`.
- Improvement path: Stream JSONL or incremental records, or write embeddings directly into an Arrow/Parquet cache used by classifier training.

**Arrow DataLoader reconstructs batches row by row:**
- Problem: Each iteration calls `Dataset::get` for individual rows and then concatenates them, creating avoidable slicing and concatenation overhead.
- Files: `crates/aprender-data/src/dataloader.rs`, `crates/aprender-data/src/dataset.rs`
- Cause: The loader does not batch contiguous indices at the RecordBatch level.
- Improvement path: Add vectorized `take`/slice paths and a text-classification collator that tokenizes and pads a batch once.

**Naive SetFit pair construction can become quadratic:**
- Problem: The repository has generic contrastive loss helpers but no bounded, class-aware sentence-pair sampler.
- Files: `crates/aprender-core/src/loss/loss.rs`, `crates/aprender-core/src/nn/self_supervised.rs`, `crates/aprender-core/src/nn/self_supervised_byol_simcse.rs`
- Cause: SetFit-style positive/negative pair generation is not represented as a streaming dataset abstraction.
- Improvement path: Sample a deterministic bounded number of positive and negative pairs per class/epoch, record the seed and sampling policy, and avoid materializing the Cartesian product.

## Fragile Areas

**Sentence pooling semantics:**
- Files: `crates/apr-cli/src/commands/embed.rs`, `crates/aprender-train/src/finetune/classification.rs`, `crates/aprender-train/src/transformer/encoder.rs`
- Why fragile: Mean, CLS, and last-token pooling exist in separate modules. CLI mean pooling assumes no padding, while training paths may pad for device-specific shapes. Several poolers copy raw values and detach gradients.
- Safe modification: Define one masked pooling operation with explicit shape, mask, dtype, normalization, and gradient contracts. Persist the selected pooling mode in the model artifact.
- Test coverage: Unit tests verify small forward values and shapes, but there is no end-to-end gradient test or sentence-transformers numerical parity test for masked batches.

**BERT tensor-name and configuration loading:**
- Files: `crates/aprender-core/src/models/bert/load.rs`, `crates/aprender-core/src/models/bert/config.rs`, `crates/aprender-core/src/format/converter/`, `crates/apr-cli/src/commands/embed.rs`
- Why fragile: Loading depends on exact Hugging Face tensor names, a limited set of classifier prefixes, and caller-supplied dimensions. Encoder-only and cross-encoder artifacts take different paths.
- Safe modification: Use a typed model manifest, validate the complete expected tensor set before mutation, and retain exact source architecture/tokenizer metadata.
- Test coverage: Synthetic APR tests cover names and shapes; the module states that Hugging Face numerical parity is out of scope.

**Classification training pipeline:**
- Files: `crates/aprender-train/src/finetune/classification.rs`, `crates/aprender-train/src/finetune/classify_pipeline/training.rs`, `crates/aprender-train/src/finetune/linear_probe.rs`
- Why fragile: Forward and backward logic manually extracts tensor data, manually adds biases, manually seeds logits gradients, and updates some parameters outside a uniform module abstraction.
- Safe modification: Add one differentiable classifier module and loss, then prove parameter updates by name. Keep the frozen-probe path explicitly detached and the contrastive/full-finetune path explicitly connected.
- Test coverage: Existing tests emphasize finite loss, shapes, parameter counts, and aggregate loss decrease; these can pass when only the head learns.

**Benchmark split selection:**
- Files: `crates/apr-cli/src/commands/data_tweeteval.rs`, `crates/apr-cli/src/data_commands.rs`, `contracts/tweet-eval-stance-benchmark-v1.yaml`
- Why fragile: The compatibility profile intentionally merges validation and canonical test data, while the canonical profile preserves model-selection isolation.
- Safe modification: Default all development and hyperparameter selection to the canonical profile. Permit the SetFit compatibility profile only for final reproduction and surface a machine-readable leakage warning.
- Test coverage: Exact sizes and class counts are covered, but no training orchestrator prevents tuning against the merged compatibility test split.

**Feature-gated training integration:**
- Files: `crates/apr-cli/Cargo.toml`, `Cargo.toml`, `crates/aprender-train/Cargo.toml`
- Why fragile: Training, CUDA, inference, and root facade features are wired independently, and the training crate uses a legacy library name.
- Safe modification: Define a feature matrix for SetFit CPU inference, CPU training, and GPU training; exercise every supported combination in CI before exposing the command.
- Test coverage: The workspace has extensive crate tests, but feature-combination behavior is distributed across manifests and specialized QA skills.

## Scaling Limits

**BERT sentence encoder throughput:**
- Current capacity: The public core BERT embedding API processes one variable-length sequence per forward call.
- Limit: Contrastive training requires at least two sentence forwards per pair and commonly many in-batch negatives, so serial batch-one execution scales poorly even for a few hundred examples.
- Scaling path: Add true batching, padding masks, length bucketing, reusable tokenizer state, and GPU/parallel backend support before promising benchmark training times.

**In-memory dataset representations:**
- Current capacity: `ArrowDataset` stores all RecordBatches in memory, `load_safety_corpus` reads complete JSONL content, and `apr embed` accumulates all outputs.
- Limit: Larger text corpora and cached embeddings multiply memory usage across raw text, token IDs, pair samples, and f32 embeddings.
- Scaling path: Stream source rows, cache tokenized examples/embeddings in Parquet or Arrow IPC, and expose bounded prefetching rather than whole-corpus vectors.

**Full encoder fine-tuning:**
- Current capacity: The trainable encoder exposes immutable per-layer parameter lists and embedding operations that are not connected to their source weights.
- Limit: Full or top-layer SetFit fine-tuning cannot be implemented reliably through the current public APIs.
- Scaling path: Add named mutable parameter traversal, optimizer groups, gradient checkpointing, mixed precision policy, and explicit freeze/unfreeze controls.

## Dependencies at Risk

**SafeTensors version split:**
- Risk: `aprender-core` depends on SafeTensors 0.4 while `aprender-train` depends on 0.7, reinforcing separate loader/type paths.
- Impact: Shared encoder artifacts and classifier checkpoints can require conversion or duplicate code; direct type sharing is unavailable.
- Migration plan: Centralize serialization behind repository-owned tensor metadata types and converge dependency versions after compatibility tests.

**Published `alimentar` API lag:**
- Risk: `apr data` carries copied implementations because required local APIs are not available in the published dependency used by the CLI.
- Impact: SetFit data preparation can behave differently across workspace builds and crates.io installs.
- Migration plan: Publish the required `aprender-data` APIs, remove CLI stubs, and add a standalone packaged-command test.

**Mixed MSRV and edition declarations:**
- Risk: Core and CLI require Rust 1.91, `aprender-train` declares 1.87, and other workspace crates declare older MSRVs or edition 2024.
- Impact: A SetFit feature spanning core, train, CLI, and data can compile in the workspace but fail for a crate consumer expecting the training crate's lower declared MSRV.
- Migration plan: Test declared MSRVs per publishable crate and raise/lower declarations based on actual transitive requirements.

## Missing Critical Features

**Differentiable sentence encoder:**
- Problem: No public BERT sentence-encoder path combines tokenizer, batched attention masks, differentiable masked pooling, L2 normalization, parameter traversal, and optimizer integration.
- Blocks: The contrastive fine-tuning phase that distinguishes SetFit from a frozen-embedding linear probe.

**SetFit pair sampler and training objective:**
- Problem: There is no deterministic class-aware pair dataset, epoch sampler, or tensor-valued SetFit loss.
- Blocks: Reproducible positive/negative sampling, bounded memory use, seed sweeps, and in-batch-negative training.

**Production multiclass head artifact:**
- Problem: `LinearProbe` supports multiclass training but has no fallible training contract, robust regularization API, or save/load artifact that binds encoder, tokenizer, label map, pooling, and normalization.
- Blocks: TweetEval's three-class task and reliable deployment of a trained SetFit model through `apr`.

**Training-to-inference round trip:**
- Problem: Core BERT inference loads APR, while trainable encoder/probe code loads and represents weights differently. There is no SetFit checkpoint round-trip through the exact `apr` inference path.
- Blocks: Proof that the model evaluated during training is the model users receive.

**Few-shot experiment orchestration:**
- Problem: The TweetEval contract records shots per class and ten seeds, but the framework lacks an end-to-end SetFit runner that samples, trains, selects on canonical validation, evaluates once on canonical test, and aggregates seed statistics.
- Blocks: Comparable benchmark claims and protection against favorable-seed reporting.

## Test Coverage Gaps

**Real sentence-transformer numerical parity:**
- What's not tested: Tokenization, encoder activations, masked mean pooling, normalization, and final cosine similarities against a pinned Hugging Face sentence-transformer on real weights.
- Files: `crates/aprender-core/src/models/bert/mod.rs`, `crates/aprender-core/src/models/bert/load.rs`, `crates/apr-cli/src/commands/embed.rs`
- Risk: Shape-correct inference can still produce semantically wrong embeddings.
- Priority: High

**End-to-end encoder gradient flow:**
- What's not tested: Named BERT embedding, attention, FFN, normalization, and selected layer parameters all receive correct gradients and change after a contrastive optimizer step.
- Files: `crates/aprender-core/src/models/bert/embeddings.rs`, `crates/aprender-core/src/models/bert/layer.rs`, `crates/aprender-core/src/models/bert/encoder.rs`, `crates/aprender-train/src/transformer/encoder.rs`
- Risk: SetFit reports training while only the classifier or no encoder parameters update.
- Priority: High

**Weighted multiclass reference parity:**
- What's not tested: Weighted `LinearProbe` and `MlpProbe` gradients and convergence against a reference implementation on imbalanced data.
- Files: `crates/aprender-train/src/finetune/linear_probe.rs`
- Risk: Minority-class performance and TweetEval F_avg can be distorted even while loss stays finite.
- Priority: High

**Batching, masking, and truncation:**
- What's not tested: Mixed-length padded batches, attention-mask propagation, masked mean pooling, empty text, exact maximum length, overlength policy, and tokenizer/model vocabulary mismatch.
- Files: `crates/aprender-core/src/models/bert/embeddings.rs`, `crates/apr-cli/src/commands/embed.rs`, `crates/aprender-train/src/transformer/embedding.rs`
- Risk: Crashes, padding-dependent embeddings, or silent zero/repeated embeddings.
- Priority: High

**SetFit artifact round trip:**
- What's not tested: Save a trained encoder plus head, reload it through the supported CLI/runtime, and reproduce logits, probabilities, labels, and embeddings within tolerance.
- Files: `crates/aprender-core/src/models/bert/load.rs`, `crates/aprender-train/src/finetune/linear_probe.rs`, `crates/apr-cli/src/commands/embed.rs`
- Risk: Training succeeds but deployment cannot load or faithfully reproduce the model.
- Priority: High

**Canonical TweetEval multi-seed benchmark:**
- What's not tested: The complete 8/16/32/64-shot, ten-seed protocol with selection restricted to validation and official F_avg computed only for labels 1 and 2.
- Files: `contracts/tweet-eval-stance-benchmark-v1.yaml`, `crates/apr-cli/src/commands/data_tweeteval.rs`, `crates/aprender-train/src/eval/classification/metrics.rs`
- Risk: Benchmark results are irreproducible, leak test data, or use the wrong aggregate metric.
- Priority: High

**Failure-path and resource-limit tests:**
- What's not tested: Oversized model dimensions, oversized tokenizer JSON, malformed IDs, zero classes, zero heads, empty probes, zero bootstrap iterations, and DataLoader row failures all return structured errors without panic or silent truncation.
- Files: `crates/aprender-core/src/models/bert/config.rs`, `crates/aprender-core/src/models/bert/embeddings.rs`, `crates/aprender-train/src/finetune/linear_probe.rs`, `crates/aprender-data/src/dataloader.rs`
- Risk: Local denial of service, misleading success, and brittle automation.
- Priority: Medium

---

*Concerns audit: 2026-08-07*
