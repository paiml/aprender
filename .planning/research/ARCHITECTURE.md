# Architecture Patterns

**Domain:** Native SetFit-style few-shot text classification in a contract-first Rust ML monorepo  
**Project:** Aprender Native SetFit Classification  
**Researched:** 2026-08-07  
**Confidence:** HIGH for repository boundaries and dependency order; MEDIUM-HIGH for the proposed APR semantic schema until it is ratified as an executable contract

## Recommended Architecture

SetFit should be a new **model family and lifecycle**, not a training-only utility and not an extension of the existing LoRA `--task classify` implementation. The production model is one semantic object composed of a tokenizer, one trainable sentence encoder, an explicit pooling/normalization policy, and one multiclass linear head. Training owns optimization policy; `aprender-core` owns the model math and APR semantics; serving calls the same core forward path that training evaluates.

The central decision is to make the existing `aprender-core` BERT/autograd stack canonical for SetFit:

```text
contracts/ + aprender-contracts
        │ define data, gradient, artifact, and round-trip obligations
        ▼
aprender-data                    apr-format
typed JSONL + deterministic       byte container only
sampling                          (no SetFit semantics)
        │                               │
        └──────────────┐                │
                       ▼                ▼
                 aprender-core
      tokenizer → batched trainable BERT → masked pool → normalize
                                      └── multiclass linear head
                    one SetFitModel + one APR semantic loader/writer
                       ▲                         ▲
                       │                         │
                aprender-train             aprender-serve
          pair objective, two-stage       immutable SetFitRuntime,
          optimization, selection          HTTP classification
                       ▲                         ▲
                       └──────── apr-cli ───────┘
                         thin commands/dispatch
```

This direction respects the current Cargo DAG: `crates/aprender-train/Cargo.toml` already depends on `aprender-core`, while `crates/aprender-core/Cargo.toml` intentionally does not have a runtime dependency on `aprender-train`. Moving the inference model into `aprender-train`, or making core load a train-owned encoder, would create the wrong dependency direction. `crates/aprender-serve/Cargo.toml` can consume core model semantics without acquiring training dependencies.

The official SetFit architecture is also two-stage: contrastively fine-tune a sentence-transformer body, then train a classifier on embeddings from that tuned body. The official implementation defaults to a simple logistic-regression head and freezes the body during the head stage unless end-to-end differentiable-head training is explicitly selected. Aprender v1 should follow the simpler, auditable two-stage form and use one native multiclass softmax-linear head for both binary (`K=2`) and multiclass (`K>2`) tasks.

## Component Boundaries

| Component | Owning crate/path | Responsibility | Public boundary | Must not own |
|-----------|-------------------|----------------|-----------------|--------------|
| SetFit lifecycle contracts | `contracts/`, bindings in `crates/aprender-contracts/src/` and generated crate bindings | Specify JSONL schema, split isolation, pair bounds, differentiability, named parameter updates, artifact schema, and train-save-load-predict parity | Versioned `setfit-*-v1.yaml` contracts and generated assertions | Model implementation or CLI presentation |
| Text classification dataset | `crates/aprender-data/src/` | Fallible JSONL parsing, stable sample IDs, label-map validation, split/provenance checks, deterministic few-shot selection | `TextClassificationDataset`, `TextExample`, `FewShotSelection` | Tokenizer, tensors, losses, model parameters |
| Contrastive pair sampler | `crates/aprender-data/src/` | Deterministically stream bounded same-label/different-label index pairs per epoch without a Cartesian materialization | `ContrastivePairSampler: Iterator<Item = Result<PairRef>>`, with seed and sampling manifest | Sentence encoding or loss math |
| Canonical tokenizer and encoded batch | `crates/aprender-core/src/text/` or `crates/aprender-core/src/models/bert/` | Load one tokenizer state, enforce special tokens/vocabulary/max length, batch encode, pad, and emit IDs, type IDs, and masks | `SentenceTokenizer`, `EncodedTextBatch { input_ids, token_type_ids, attention_mask, batch, seq_len }` | Dataset split selection or CLI file reading |
| Trainable sentence encoder | Extend `crates/aprender-core/src/models/bert/`; add a sentence-encoder wrapper beside `embeddings.rs`, `encoder.rs`, and `load.rs` | One batched BERT forward for training and inference; differentiable gather, attention masking, masked pooling, optional L2 normalization; named parameters and freeze policy | `SentenceEncoder::forward_batch(&EncodedTextBatch) -> Result<Tensor>` and an explicit `encode_no_grad` inference helper | Optimizer loop, pair generation, HTTP |
| Multiclass linear head | `crates/aprender-core/src/classification/` or a core SetFit model module | Fallible `H -> K` logits, stable softmax, prediction, label lookup, regularized fitting support, deterministic tensor names | `MulticlassLinearHead`, `logits`, `predict_proba`, `predict`; binary is `K=2` | A second encoder or task-specific serialization |
| SetFit model and APR semantics | New core module, preferably `crates/aprender-core/src/models/setfit/` | Compose tokenizer identity, sentence encoder, pooling/normalization, head, label map and manifest; validate and read/write one APR | `SetFitModel::load_apr`, `save_apr`, `embed_batch`, `predict_batch`, `manifest` | Training-state policy or network server |
| SetFit trainer | New `crates/aprender-train/src/finetune/setfit/` | Two explicit stages, tensor-valued contrastive loss, optimizer/scheduler, parameter-update audit, validation selection, checkpoint callbacks | `SetFitTrainer`, `SetFitTrainingConfig`, stage reports, `fit_encoder`, `fit_head`, `export_verified` | A train-only encoder, tokenizer implementation, or final inference math |
| Classification evaluation | `crates/aprender-train/src/eval/classification/` | Per-class metrics, official explicit-class averages, calibration and confusion matrix | Existing metrics plus a model-neutral `ClassificationEvaluator` | Loading different weights from those served |
| SetFit runtime | New `crates/aprender-serve/src/classification/` and state/router integration in `crates/aprender-serve/src/api/` | Load immutable verified `SetFitModel`, batch text requests, return logits/probabilities/labels, record latency/audit | `SetFitRuntime::from_apr`, `classify_batch`; native `/v1/classify` request/response | BERT math, tokenizer reconstruction rules, training |
| CLI lifecycle adapter | `crates/apr-cli/src/extended_commands.rs`, `dispatch_analysis.rs`, and a focused command module under `crates/apr-cli/src/commands/` | Parse commands/config, invoke data/train/core/serve APIs, render JSON/human output | `apr setfit train|benchmark`; generic `apr run`, `apr eval`, `apr inspect`, and `apr serve` auto-detect `architecture=setfit` | Tokenization, pooling, pair sampling, model forward |

### Why no new crate

Do not create an `aprender-setfit` crate in the first milestone. SetFit has no independent low-level dependency boundary: its model semantics belong with other core models, its optimization belongs in training, and its runtime belongs in serving. A new crate would either duplicate these owners or force a shared-tensor extraction before correctness is established. Revisit extraction only if a later milestone needs multiple encoder families that can no longer fit the core model boundary.

### Canonical public model APIs

The model-facing API should make detached and differentiable paths visibly different:

```rust
pub struct EncodedTextBatch {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub batch_size: usize,
    pub sequence_length: usize,
}

pub struct SentenceEncoderConfig {
    pub bert: BertConfig,
    pub pooling: PoolingPolicy,
    pub normalize: bool,
    pub max_length: usize,
}

impl SentenceEncoder {
    // Graph-connected [B, S] tokens -> [B, H] sentence embeddings.
    pub fn forward_batch(&self, batch: &EncodedTextBatch) -> Result<Tensor>;

    // Explicit inference/cache boundary; never used inside contrastive backward.
    pub fn encode_no_grad(&self, batch: &EncodedTextBatch) -> Result<Vec<Vec<f32>>>;

    pub fn named_parameters(&self) -> Vec<(ParameterName, &Tensor)>;
    pub fn named_parameters_mut(&mut self) -> Vec<(ParameterName, &mut Tensor)>;
}

impl SetFitModel {
    pub fn load_apr(path: impl AsRef<Path>) -> Result<Self>;
    pub fn save_apr(&self, path: impl AsRef<Path>) -> Result<()>;
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    pub fn predict_batch(&self, texts: &[&str]) -> Result<Vec<ClassificationOutput>>;
}
```

`SetFitTrainer` should use typestate or an equivalent private state machine so an unverified in-memory model cannot be confused with a deployable artifact:

```text
Prepared -> EncoderTuned -> HeadFitted -> ArtifactReloadedAndVerified
```

Only the final state may emit the production output path. Intermediate checkpoints may be written separately and clearly marked as checkpoints.

## Encoder and Autograd Consolidation Strategy

### Final direction

Use `aprender-core::autograd::Tensor` throughout the SetFit forward graph and upgrade the existing core BERT modules into recursively trainable `Module`s. This is the only practical direction that simultaneously:

- reuses the current production BERT/APR tensor naming and loading in `crates/aprender-core/src/models/bert/load.rs`;
- allows `aprender-train` to optimize the model without reversing the Cargo DAG;
- lets `aprender-serve` execute exactly the same encoder and head without a training dependency;
- removes the current `apr embed` versus training-encoder drift.

Implement recursive parameter traversal for `BertEmbeddings`, `BertLayer`, `BertEncoder`, and the sentence wrapper. Add stable names matching the APR tensor names, plus parameter groups for embeddings, selected encoder layers, and head. Use the core AdamW implementation under `crates/aprender-core/src/nn/optim/` directly from the SetFit trainer, or a train-owned orchestration wrapper around it; do not copy parameters into `aprender-train::autograd::Tensor` for optimizer steps.

The first prerequisite is graph-connected primitives in core autograd:

- batched embedding gather/index-select whose backward scatters into the source tables;
- differentiable slicing/gather for selected rows;
- masked reduction over sequence positions;
- stable L2 norm/normalization with epsilon;
- cosine similarity and a tensor-valued contrastive objective;
- any reshape/broadcast/mask operations needed by `[B,S,H]` attention and pooling.

The current copies are correctness blockers, not optimization opportunities: `crates/aprender-core/src/models/bert/embeddings.rs` calls `.data()` and constructs a new tensor; `crates/aprender-core/src/models/bert/cross_encoder.rs` copies the CLS slice; `crates/aprender-train/src/transformer/embedding.rs` and `encoder.rs` do the same in the legacy stack. Setting `requires_grad` on the new tensor does not reconnect it to the source weights.

### What remains legacy

`crates/aprender-train/src/autograd/` and `crates/aprender-train/src/transformer/encoder.rs` may remain for existing decoder/LoRA workflows during this milestone. They are not a supported SetFit backend. Consolidation is directional: all new sentence-encoder work lands on the core graph; later migrations can retire the train-owned encoder after parity. Trying to migrate every LLM trainer before SetFit would unnecessarily broaden the milestone.

### Proof gate before any trainer or CLI

A tiny real BERT-shaped test must record every intended parameter before and after one contrastive optimizer step and assert:

1. the loss is a scalar graph tensor and finite;
2. named gradients exist, are finite, and are non-zero for word embeddings plus representative Q/K/V/O, FFN, and normalization parameters;
3. those named parameters change after the step;
4. frozen parameters do not change;
5. a finite-difference check agrees for the new gather, pooling, normalization, and contrastive primitives.

No `apr setfit train` command should be exposed until this gate passes. Aggregate loss decrease is insufficient because the head can learn while the encoder is detached.

## Data Flow: JSONL to APR to Serving

### 1. Dataset acquisition and validation

`crates/apr-cli/src/commands/data_tweeteval.rs` already emits rows shaped as `{ id, input, label, label_text, source_split }` and a benchmark manifest with source revision, hashes, split identities, label map, shot counts, and seeds. The SetFit data API should accept this schema directly.

`aprender-data` parses JSONL with an explicit schema, rejects empty input/unknown labels/duplicate IDs, and preserves IDs and source splits. For benchmark runs it validates hashes and forbids canonical test rows from entering training or model selection. Do not route correctness-sensitive training through the current `DataLoader` iterator in `crates/aprender-data/src/dataloader.rs` until its `filter_map`/`Option` failure path is replaced with `Result<Batch>`; otherwise a malformed row can look like end-of-epoch.

### 2. Few-shot selection and pair generation

Few-shot selection returns stable sample indices and IDs, balanced without replacement by label and seed. The pair sampler then yields bounded `PairRef { left, right, target }` values. It must be restartable per epoch, deterministic from `(dataset_hash, seed, epoch, policy)`, exclude self-pairs, and define whether `(i,j)` and `(j,i)` are equivalent.

Match SetFit semantics—same-label positive pairs and different-label negative pairs—but make the Aprender default a bounded balanced sampler. The official SetFit `unique`/over/undersampling modes enumerate possible pairs and can become quadratic; expose those only when the contracted maximum pair count is not exceeded. Persist the selected IDs and sampling policy in run provenance.

### 3. Tokenization and collation

Load one tokenizer instance before training. Batch encode the distinct sentence IDs needed by a pair batch, add special tokens, truncate or reject according to the persisted policy, pad to the batch maximum (optionally a device multiple), and emit `input_ids`, `token_type_ids`, and `attention_mask` together. Fail closed on out-of-range IDs, vocabulary/config mismatch, or sequence overflow.

The tokenizer and preprocessing behavior must move out of `crates/apr-cli/src/commands/embed.rs`; that command currently reconstructs a WordPiece vocabulary/tokenizer for each text and omits padded attention masks. Training, `apr embed`, `apr run`, and serving must all call the same core tokenizer/encoder boundary.

### 4. Encoder fine-tuning

Collate the unique sentences in a pair batch once, run a graph-connected `[B,S] -> [B,H]` forward, gather the left/right sentence rows without detaching, and compute a scalar tensor loss. Backpropagate through normalization, masked pooling, encoder, and embedding lookup. The trainer records pair counts, seed, learning rate, gradient audit, and updated parameter hashes.

For v1, use one contracted objective—cosine-similarity pair loss is the closest match to the default official SetFit path. Additional triplet or supervised-contrastive objectives are later model-selection options, not parallel initial implementations.

### 5. Head fitting and validation

Freeze the tuned encoder and intentionally enter `no_grad` to encode the selected training examples. Fit one regularized `MulticlassLinearHead` with stable softmax cross-entropy and optional target-class weighting. Correct the weighted-gradient issue in `crates/aprender-train/src/finetune/linear_probe.rs`, but do not make that fragmented probe the production head. Fit binary tasks with two logits, not a separate sigmoid-only model.

Validation calls `SetFitModel::predict_batch`, the same composite inference API used later by the loaded artifact. Model selection is based only on canonical validation. The benchmark contract in `contracts/tweet-eval-stance-benchmark-v1.yaml` keeps canonical test separate and defines TweetEval `F_avg` over explicit class indices 1 and 2.

### 6. APR assembly

Build one `SetFitModel` in memory and write it atomically through the existing core APR writer path in `crates/aprender-core/src/serialization/apr/mod.rs`. `crates/apr-format/` remains a semantic-free byte container; SetFit interpretation and validation live in core.

The final artifact contains all inference state. Optimizer moments, pair cursor, and scheduler state belong only in optional tensors prefixed `__training__.` in checkpoint artifacts and are excluded from the final deployable APR.

### 7. Mandatory reload boundary

Close the in-memory model, load the written APR through `SetFitModel::load_apr`, and compare on a fixed probe batch:

- token IDs, type IDs, and attention masks exactly;
- sentence embeddings within the documented float tolerance;
- logits and probabilities within tolerance;
- label IDs and label strings exactly.

Only the reloaded artifact is evaluated for final reports, registered, benchmarked, or served. This is the architectural guard against train/serve drift.

### 8. CLI and serving

`apr run`, `apr eval`, `apr inspect`, and `apr serve` inspect APR metadata and dispatch `architecture=setfit` to the core model loader. `aprender-serve` holds an `Arc<SetFitRuntime>` in `AppState`, registers a typed `/v1/classify` endpoint accepting one or many strings, and returns ordered labels, probabilities, optional logits, model hash, and latency. Keep the existing numeric-feature `PredictRequest` in `crates/aprender-serve/src/api/mod_create_demo.rs` backward compatible rather than overloading its `features: Vec<f32>` field.

Do not implement SetFit in the demo-style linear `apr_predict_handler` in `crates/aprender-serve/src/api/apr_handlers.rs`, and do not use the stub `/predict` path in `crates/apr-cli/src/commands/serve/routes.rs`. Both currently bypass text tokenization and the real encoder.

## APR Artifact Contract

Define a versioned SetFit semantic manifest in APR metadata. The low-level `AprV2Metadata` in `crates/apr-format/src/v2/header_impl.rs` already provides typed architecture/dimension/provenance fields plus flattened custom JSON; the schema should use typed fields where available and reserve `setfit.*` keys for model-family policy.

### Required typed metadata

| Field | Required value/purpose |
|-------|------------------------|
| `model_type` | `text_classifier` |
| `architecture` | `setfit` |
| `hidden_size`, `num_layers`, `num_heads`, `vocab_size`, `intermediate_size`, `max_position_embeddings` | Authoritative encoder configuration; no normal CLI overrides |
| `source`, `original_format` | Base encoder identity and import format |
| `data_source`, `data_license` | Training data provenance when known |
| `created_at`, `version`, `param_count` | Artifact identity and audit information |

### Required SetFit metadata

Use a single canonical JSON object under `setfit.manifest` (or equivalently namespaced flattened keys) with at least:

```json
{
  "schema_version": "1.0.0",
  "encoder_family": "bert",
  "type_vocab_size": 2,
  "layer_norm_eps": 1e-12,
  "pad_token_id": 0,
  "max_length": 256,
  "pooling": { "mode": "mean", "mask": "attention_mask" },
  "normalization": { "enabled": true, "epsilon": 1e-12 },
  "head": { "type": "linear_softmax", "in_features": 384, "out_features": 3, "bias": true },
  "labels": ["none", "against", "favor"],
  "tokenizer": { "format": "huggingface_tokenizer_json", "sha256": "..." },
  "training": {
    "objective": "cosine_similarity_pairs",
    "pair_policy": "bounded_balanced",
    "seed": 13,
    "selected_id_hash": "...",
    "dataset_hash": "..."
  }
}
```

Embed the exact tokenizer JSON/state used by training, not only a vocabulary. Sentence-transformer reconstruction depends on tokenizer preprocessing as well as the encoder and pooling/normalize modules. Store a SHA-256 and fail if the embedded state does not match it. The APR metadata section has an explicit 16 MiB cap in `crates/apr-format/src/v2/mod.rs`; reject oversized tokenizer state with a structured error rather than silently falling back to a sibling file. The requirement is one self-contained serving artifact.

### Canonical tensor names

Retain the existing BERT names consumed by `crates/aprender-core/src/models/bert/load.rs` so import, training, and inference share the same mapping:

```text
bert.embeddings.word_embeddings.weight
bert.embeddings.position_embeddings.weight
bert.embeddings.token_type_embeddings.weight
bert.embeddings.LayerNorm.{weight,bias}
bert.encoder.layer.{N}....
setfit.classifier.weight        # [num_classes, hidden_size]
setfit.classifier.bias          # [num_classes]
```

The core SetFit loader validates the complete tensor set, shapes, dtype support, label count, head dimensions, tokenizer vocabulary, and manifest version before allocating the runtime. It must not infer architecture from user-supplied CLI dimensions as `crates/apr-cli/src/commands/embed.rs` currently does.

## Explicit Build-Order Dependencies

The roadmap should use this dependency graph. Items on the same row may be developed in parallel only after their prerequisites are green.

| Order | Deliverable | Depends on | Exit gate before downstream work |
|------:|-------------|------------|----------------------------------|
| 1 | SetFit data/gradient/artifact/lifecycle contracts and small authored fixtures | Existing contract infrastructure; TweetEval benchmark contract | Schema versions, tensor names, manifest fields, tolerances, split rules and failure behavior are executable |
| 2 | Core autograd primitives and recursive named BERT parameters | 1 | Finite-difference primitive checks and a named end-to-end encoder update test pass |
| 3 | Canonical tokenizer, batched BERT, attention masks, masked pooling, normalization, and HF/Sentence-Transformer parity | 2 | Mixed-length batch parity and pinned real-weight embedding/cosine parity pass on CPU |
| 4A | Fallible JSONL dataset, deterministic few-shot selector, bounded pair sampler | 1 | Stable IDs/hashes, class balance, bounded memory, repeatability, and error propagation pass |
| 4B | One fallible multiclass linear head | 2 | Binary/multiclass probability simplex, regularization, weighted-gradient reference parity, and serialization tensor parity pass |
| 5 | Two-stage `SetFitTrainer` using the core sentence encoder and head | 3, 4A, 4B | Encoder gradients/updates are audited by name; head stage uses the tuned frozen body; validation never touches test |
| 6 | Self-contained APR writer/loader and train-save-load-predict round trip | 5 | Tokenization, embeddings, logits, probabilities, labels, metadata, and checksums round-trip within contract tolerances |
| 7 | Rust public API plus generic CLI lifecycle dispatch | 6 | `train -> APR -> inspect -> eval -> predict` runs from the reloaded artifact on CPU; no in-memory shortcut |
| 8 | `aprender-serve` runtime and HTTP classification | 6, 7 model API | CLI and HTTP outputs match the same artifact byte-for-byte/tolerance-for-tolerance; readiness reports classifier loaded |
| 9 | Canonical multi-seed benchmark and 9B LoRA comparison | 7, 8 | Identical sampled IDs, validation-only selection, one final test evaluation, machine-readable quality/resource/calibration report |
| 10 | Optional GPU/device acceleration | CPU gates 2-9 | Feature matrix passes and CPU/GPU embeddings/logits stay within explicit tolerances; CPU remains mandatory |

The key ordering rules are non-negotiable:

- Do not build the trainer before the graph-connected encoder-update proof.
- Do not expose a production CLI command before the APR reload round trip.
- Do not build a serving-specific forward path; serving follows the core loaded-model API.
- Do not publish benchmark claims from the in-memory trainer; benchmark the reloaded production artifact.
- Do not start GPU optimization before the CPU numerical and lifecycle contracts pass.

## Temporary Adapters Versus Final Architecture

| Existing/possible adapter | Allowed temporary use | Final disposition |
|---------------------------|-----------------------|-------------------|
| `crates/apr-cli/src/commands/embed.rs` local tokenizer/pooler | Numerical migration oracle while core batch tokenization/pooling is introduced | Refactor command to call `SentenceEncoder`; remove local tokenization, manual dimensions, and pooling |
| `crates/aprender-train/src/transformer/encoder.rs` | Test-only weight/activation comparison during migration | Not a SetFit backend; deprecate for encoder classification after core parity |
| Train-loop adapter around core AdamW/parameters | Progress, callback, checkpoint, and scheduler integration only | May remain as orchestration; it must never convert activations or parameters between tensor engines |
| Tensor-copy bridge between `entrenar::Tensor` and core Tensor | Offline one-time weight conversion test only | Forbidden in the trainable forward or optimizer path |
| `crates/aprender-train/src/finetune/linear_probe.rs` | Baseline/reference after correcting known bugs | Replace with/rebase onto the production core multiclass head; do not serialize it separately |
| Existing generic APR reader/writer | Container I/O and atomic write | Retain; add SetFit semantics in core, not `apr-format` |
| Existing serve `AprModel`/mmap wrapper | Container validation or future zero-copy storage | Do not implement SetFit math there; runtime delegates to core `SetFitModel` |
| Sibling `tokenizer.json` lookup | Import-time source discovery only | Final SetFit APR is self-contained; serving must not require a sibling file |

## Patterns to Follow

### Pattern 1: One Semantic Model, Multiple Lifecycle Adapters

**What:** Training, CLI prediction, evaluation, and serving all call `SetFitModel` from core.  
**When:** Every post-training operation.  
**Why:** Compiler-enforced reuse is stronger than parity tests between duplicated implementations.

### Pattern 2: Explicit Detach Boundary

**What:** The contrastive stage stays entirely graph-connected. Detachment is legal only when the tuned encoder is frozen before head fitting or when producing inference outputs.  
**When:** Transition from `EncoderTuned` to `HeadFitted`, and request-time inference.  
**Why:** It turns the current silent `.data()` failure mode into an API-visible decision.

### Pattern 3: Contracted Artifact Before User Surfaces

**What:** Implement and test artifact schema and reload first; make CLI/HTTP thin clients afterward.  
**When:** Any new model family in this contract-first repository.  
**Why:** The artifact is the stable integration boundary between training and serving.

### Pattern 4: Index-Based Pair Streams

**What:** Pair samplers yield references/indices into immutable examples; collation deduplicates sentence IDs within a batch.  
**When:** Contrastive training.  
**Why:** Avoids copying text and quadratic pair materialization while preserving reproducibility.

### Pattern 5: Fail-Closed Configuration

**What:** Encoder dimensions, pooling, normalization, tokenizer, label order and head shape come from APR and are cross-validated against tensor shapes.  
**When:** Load, inspect, run, or serve.  
**Why:** The current manual `apr embed` dimensions permit a valid model to be interpreted incorrectly.

## Anti-Patterns to Avoid

### Anti-Pattern 1: A SetFit Encoder Inside `aprender-train`

**What:** Add another sentence encoder next to `crates/aprender-train/src/transformer/encoder.rs` and export its weights later.  
**Why bad:** Serving cannot share it without the wrong dependency direction, and train/serve drift remains structural.  
**Instead:** Train the upgraded core BERT model directly.

### Anti-Pattern 2: “Gradient-Capable” Copies

**What:** Read values with `.data()`, construct a new `requires_grad` tensor, then call backward.  
**Why bad:** The new tensor has no edge to the source parameters.  
**Instead:** Implement tracked gather, slice, mask, pool, and normalize operations.

### Anti-Pattern 3: Classifier-Only Success as SetFit Success

**What:** Accept decreasing loss or improved accuracy without named encoder updates.  
**Why bad:** A frozen embedding probe can pass those tests and is explicitly out of scope.  
**Instead:** Make encoder gradient/update evidence a release gate.

### Anti-Pattern 4: Metadata as Documentation Only

**What:** Store loosely named custom keys but let CLI flags override them.  
**Why bad:** Artifact meaning becomes caller-dependent.  
**Instead:** Version and validate a typed semantic manifest; overrides are debug-only and cannot produce benchmark/served artifacts.

### Anti-Pattern 5: Separate Benchmark Model

**What:** Evaluate trainer memory, then save a best-effort APR for users.  
**Why bad:** Quality claims do not describe the deployed bytes.  
**Instead:** Reload first, then evaluate and benchmark that artifact.

## Scalability Considerations

| Concern | Few-shot / hundreds of rows | 10K examples | 1M examples |
|---------|-----------------------------|--------------|-------------|
| JSONL | Arrow/stream reader with explicit schema; in-memory IDs acceptable | Stream batches and cache tokenized rows in Arrow IPC/Parquet | Sharded dataset and bounded prefetch; never retain all text/tokens |
| Pair sampling | Bounded balanced stream; optional contracted unique enumeration | Fixed pairs per class/example/epoch | Reservoir/class-index sampling; no pair materialization |
| Tokenization | Cache each selected example once; bucket by length | Persistent token cache keyed by tokenizer hash + text hash | Sharded cache, worker parallelism, bounded queue |
| Encoder execution | CPU batches with padding masks | Length buckets and optional accelerator | Distributed/device work is a later milestone; artifact semantics remain identical |
| Head fitting | Full-batch or minibatch native linear softmax | Minibatch AdamW/L-BFGS-style solver with validation | Streaming/online solver if needed; same `H x K` artifact |
| APR load | Owned core tensors acceptable for initial CPU milestone | Mmap/copy-on-load optimization after parity | Sharding only if encoder family requires it; tokenizer/manifest remains coherent |
| Serving | One immutable runtime, request batching optional | Dynamic batch window and tokenization cache | Multiple runtime replicas; model artifact remains read-only and content-addressed |

## Sources

### Primary external sources

- [Hugging Face SetFit conceptual guide](https://huggingface.co/docs/setfit/conceptual_guides/setfit) — model is a sentence-transformer body plus classifier trained in two phases; same/different labels define positive/negative pairs. **HIGH confidence.**
- [Hugging Face SetFit trainer reference](https://huggingface.co/docs/setfit/reference/trainer) — current sampling modes, two-stage arguments, cosine-similarity default, seed, maximum length, and body-freezing behavior. **HIGH confidence.**
- [Hugging Face SetFit sampling strategies](https://huggingface.co/docs/setfit/conceptual_guides/sampling_strategies) — unique, over-, under-, and iteration-based pair semantics and their pair-count behavior. **HIGH confidence.**
- [SetFit paper](https://arxiv.org/abs/2209.11055) — contrastive Siamese body tuning followed by classifier training on encoded examples. **HIGH confidence.**
- [Sentence Transformers custom model/save documentation](https://sbert.net/docs/sentence_transformer/usage/custom_models.html) — saved reconstruction includes transformer, pooling, optional normalize, tokenizer, and module configuration. **HIGH confidence.**
- [Hugging Face Tokenizers API](https://huggingface.co/docs/tokenizers/main/api/tokenizer) — batch encoding, explicit padding, truncation, and attention-mask behavior. **HIGH confidence.**

### Concrete repository evidence

- `crates/aprender-core/src/models/bert/{embeddings.rs,encoder.rs,layer.rs,load.rs}` — production BERT/APR inference path and current graph detaches.
- `crates/aprender-core/src/autograd/`, `crates/aprender-core/src/nn/module.rs`, `crates/aprender-core/src/nn/optim/` — the core tensor graph, recursive module contract, and compatible optimizers.
- `crates/aprender-train/src/autograd/`, `crates/aprender-train/src/transformer/{embedding.rs,encoder.rs}` — legacy parallel train tensor/encoder stack and copy-based embedding path.
- `crates/aprender-train/src/finetune/{linear_probe.rs,classification.rs}` — fragmented classifier implementations and current probe behavior.
- `crates/apr-cli/src/commands/embed.rs` — batch-one CLI tokenizer/pooling path with manual config and repeated tokenizer construction.
- `crates/aprender-data/src/{dataset.rs,dataloader.rs}` — JSONL/Arrow loading and the current non-fallible iterator boundary.
- `crates/apr-format/src/v2/{mod.rs,header_impl.rs}` — semantic-free APR v2 container, custom metadata and 16 MiB metadata limit.
- `crates/aprender-core/src/serialization/apr/mod.rs` — high-level model metadata and atomic APR persistence.
- `crates/aprender-serve/src/api/{mod.rs,router.rs,apr_handlers.rs,mod_create_demo.rs}` and `crates/apr-cli/src/commands/serve/routes.rs` — current serving state/routes and the paths that must not be mistaken for production text classification.
- `contracts/tweet-eval-stance-benchmark-v1.yaml`, `crates/apr-cli/src/commands/data_tweeteval.rs`, and `crates/aprender-train/src/eval/classification/metrics.rs` — benchmark split, provenance, label and metric foundation already present in the worktree.

---
*Architecture research: 2026-08-07*
