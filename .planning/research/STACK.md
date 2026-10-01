# Technology Stack

**Project:** Aprender Native SetFit Classification  
**Researched:** 2026-08-07  
**Overall confidence:** HIGH for the algorithm, encoder contract, and dependency choices; MEDIUM-HIGH for the exact internal refactor surface until the first end-to-end gradient-parity spike

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| `aprender-core::autograd::Tensor` | Current workspace | Canonical differentiable tensor and autograd stack for the encoder, pooling, contrastive loss, and optimizer parameters | Aprender's existing BERT inference path already uses this tensor, and the stack already has differentiable matmul, softmax, LayerNorm, GELU, dropout, and AdamW. Keeping training and inference on one tensor graph avoids detached copies and conversion bugs. | HIGH |
| Repository-owned `BertSentenceEncoder` | New internal abstraction | Batched MiniLM/BERT forward pass, attention-mask handling, masked mean pooling, and L2 normalization | SetFit tunes the sentence-transformer body, not merely a head over frozen embeddings. Aprender therefore needs a trainable sentence-encoder abstraction above its existing BERT blocks. | HIGH |
| Repository-owned `NamedModule` / `ParameterStore` | New internal abstraction | Stable dotted parameter names, recursive traversal, trainable/frozen groups, and train/eval mode propagation | Positional `parameters()` lists are insufficient for selective freezing, optimizer grouping, APR tensor naming, and parity diagnostics. Stable names should be the common contract for training and serialization. | HIGH |
| `sentence-transformers/all-MiniLM-L6-v2` | Revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | First supported pretrained sentence encoder | It is a small Apache-2.0 BERT-family model whose 6-layer, 384-hidden architecture fits Aprender's existing BERT implementation and CPU-first deployment goal. Pin the full Hub revision, never a mutable branch name. | HIGH |
| Repository-owned multinomial logistic regression | New internal abstraction | Production multiclass SetFit head with `predict_proba` | SetFit's standard head is logistic regression. A single K-class softmax implementation gives correct binary and multiclass behavior and can reuse Aprender's deterministic full-batch L-BFGS optimizer. | HIGH |
| APR v2 semantic model adapter | Current format; new `setfit-apr-v1` model schema | One-file encoder, tokenizer, head, label vocabulary, and training provenance | APR is already Aprender's native deployment boundary. The low-level format already supports custom metadata and U8 tensors, so SetFit needs a typed high-level adapter rather than another model container. | HIGH |

### Data and Artifact Storage

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| APR v2 | Current workspace | Native production artifact | Store all inference-required state in one checksummed artifact; deployment must not depend on a Hub checkout, Python, SafeTensors, or sidecar tokenizer files. | HIGH |
| SafeTensors importer | Current workspace | One-time import of pinned upstream encoder weights | SafeTensors is an interchange input, not the serving format. Keep it behind a repository-owned importer and do not make version convergence between existing crates a SetFit milestone blocker. | HIGH |
| Typed manifest plus U8 tokenizer payload | New high-level APR capability | Persist model contract and raw `tokenizer.json` bytes | JSON metadata is appropriate for the small typed manifest; raw bytes avoid escaping/base64 bloat and preserve tokenizer identity exactly. Add a byte-payload method to the high-level writer while leaving `apr-format` byte mechanics unchanged. | HIGH |

### Runtime Infrastructure

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| CPU through existing Trueno-backed kernels | Current workspace | Required training and inference baseline | The milestone is pure Rust and native. CPU correctness is the reference path and must work without CUDA or external runtimes. | HIGH |
| Existing `training-gpu` / `cuda` features | Current workspace; deferred enablement | Optional acceleration after parity | Reuse existing backend gates only after the same named-parameter, loss, gradient, and APR round-trip suite passes. An explicit accelerator request should fail clearly when unavailable rather than silently changing device. | HIGH |
| Existing optional `hf-hub` feature | Current workspace | Development-time download/import | Network retrieval is convenience only. Loading and serving a completed APR must remain offline and must not enable `hf-hub`. | HIGH |

### Supporting Libraries

#### Production Rust

| Library | Version | Purpose | When to Use | Confidence |
|---------|---------|---------|-------------|------------|
| `tokenizers` | **0.23.1** | Exact Hugging Face tokenizer pipeline and batch encoding | Add at the workspace level and enable from Aprender's focused `setfit` feature with `default-features = false, features = ["fancy-regex"]`. This keeps regex support pure Rust while excluding HTTP, progress bars, Oniguruma/native code, and the C++ ESAXX path. Promote the benchmark's older direct dependency to this workspace pin. | HIGH |
| `rand_chacha` | Current workspace (`0.9`) | Reproducible pair sampling, batch shuffling, and training seeds | Reuse; persist the seed and sampling policy in APR metadata. Do not add a second RNG crate. | HIGH |
| `serde`, `serde_json`, `sha2` | Current workspace | Typed manifest, tokenizer/config validation, source and payload hashes | Reuse existing dependencies at the semantic APR boundary. | HIGH |
| Aprender core `AdamW` | Current workspace | Encoder fine-tuning | Use only with parameters returned by the named core-autograd graph. | HIGH |
| Aprender core `LBFGS` | Current workspace | Full-batch convex head optimization | Use after encoder tuning on detached sentence embeddings. Tiny few-shot datasets make deterministic full-batch optimization a better fit than a separate SGD loop. | HIGH |

No additional production tensor framework, BLAS facade, Python bridge, or serving runtime is recommended.

#### Reference and Verification Only

| Tool | Version | Purpose | Boundary | Confidence |
|------|---------|---------|----------|------------|
| `setfit` | **1.1.3** | Behavioral reference for pair generation and two-stage training | Python dev environment only; exact pin plus transitive lock hashes. Never import, embed, or spawn it in production. | HIGH |
| `sentence-transformers` | **5.7.0** | Reference tokenizer/model/pooling/loss outputs | Python fixture generator only. | HIGH |
| `torch` | **2.13.0** | Reference gradients and optimizer-step fixtures | Python fixture generator only; CPU fixtures are sufficient. | HIGH |
| `scikit-learn` | **1.9.0** | Logistic-regression comparison and probability sanity checks | Reference tests only. Aprender's persisted head remains repository-owned. | HIGH |
| `proptest`, `approx`, `criterion`, `tempfile` | Current workspace | Properties, tolerances, performance regression, and artifact round trips | Reuse existing Rust dev dependencies; no new verification crate is required initially. | HIGH |

The Python pins are current releases as of the research date, but the reference environment must still be solved and committed as a complete lockfile. They are not Cargo dependencies and do not weaken the pure-Rust product boundary.

## Supported Encoder Contract

Support one explicit family first: post-LayerNorm BERT encoders with learned absolute position embeddings, WordPiece tokenization, token-type embeddings, GELU feed-forward blocks, and the Sentence Transformers module sequence `Transformer -> mean Pooling -> Normalize`.

The initial conformance target is exactly `all-MiniLM-L6-v2` at the pinned revision:

| Property | Required value |
|----------|----------------|
| Base architecture | `BertModel` |
| Encoder layers | 6 |
| Hidden size | 384 |
| Attention heads | 12 |
| Intermediate size | 1536 |
| Vocabulary | 30,522 WordPieces |
| Position capacity | 512 |
| Sentence maximum | 256 tokens, matching `sentence_bert_config.json` |
| LayerNorm epsilon | `1e-12` |
| Hidden/attention dropout | `0.1` in training; disabled in evaluation |
| Pooling | Attention-mask-weighted mean over token embeddings |
| Final transform | Row-wise L2 normalization with explicit epsilon |

The importer must validate the Hub config, tokenizer, and Sentence Transformers module graph against this contract. Reject unsupported architectures, remote code, position schemes, pooling modules, activations, tensor shapes, or missing fields with a typed error. Do not advertise generic Hugging Face or generic Sentence Transformers compatibility until each additional family has an adapter and parity corpus.

## Concrete Technical Approach

### 1. Make the Existing BERT Path Truly Differentiable

Define a numeric boundary such as:

```rust
pub struct SentenceBatch {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub batch_size: usize,
    pub sequence_length: usize,
}

pub trait SentenceEncoder {
    fn encode_train(&mut self, batch: &SentenceBatch) -> Result<Tensor>;
    fn encode_eval(&self, batch: &SentenceBatch) -> Result<Tensor>;
}
```

`encode_train` must return a `[batch, 384]` core-autograd tensor whose graph reaches every unfrozen encoder weight. The current embedding path's manual reads from `.data()` are not acceptable in trainable forward code. Promote or generalize the differentiable gather/backward logic already used by the Qwen2 embedding implementation, then add the missing batched primitives in `aprender-core`:

- batched word, position, and token-type embedding gather;
- broadcast addition and residual paths without detachment;
- additive attention masking with correct padded-query/key behavior;
- attention-mask-weighted mean pooling with a checked nonzero denominator;
- row-wise L2 normalization with an explicit epsilon;
- cosine similarity, mean reduction, and MSE on the same graph.

Add BERT's dropout configuration and exact placement, and propagate recursive `train()` / `eval()` state. No new SetFit forward path should convert to `aprender-train::autograd::Tensor`; adapters may exist only at legacy outer boundaries and should be removable.

### 2. Tune the Encoder with SetFit's Pair Objective

Use deterministic class buckets and stream balanced pairs on demand:

- positive pairs draw examples from the same class; negative pairs draw from different classes;
- canonicalize unordered example IDs so duplicates can be detected without retaining a Cartesian product;
- target a 1:1 positive/negative ratio and SetFit-style oversampling semantics, but impose and persist `max_pairs_per_epoch`;
- expose an explicit one-shot-class policy rather than silently manufacturing unexpected pairs;
- seed pair sampling and shuffling with the existing ChaCha RNG and persist all relevant settings.

The default v1 objective should match current SetFit behavior:

```text
z_a = normalize(masked_mean(encoder(a)))
z_b = normalize(masked_mean(encoder(b)))
loss = mean((cosine(z_a, z_b) - pair_label)^2)
pair_label in {0.0, 1.0}
```

Use core AdamW. Treat the documented SetFit defaults—body learning rate `2e-5`, batch size `16`, one encoder epoch, and 10% warmup—as explicit starting configuration, not hidden constants. Persist resolved values, weight decay, gradient clipping, seed, and pair budget. A linear warmup/decay schedule belongs in `aprender-train`; a default gradient-norm cap of `1.0` is a conservative Aprender policy and must be separately labeled rather than attributed to SetFit.

Do not substitute InfoNCE, supervised contrastive loss, triplet loss, or CoSENT in the first slice. They are useful later experiments, but they alter SetFit fidelity, batch semantics, and numerical references. Sentence Transformers now recommends stronger alternatives in some contexts; that is not a reason to skip parity with SetFit's current default cosine-MSE path.

### 3. Fit One Multiclass Logistic Head

After encoder tuning, switch the encoder to evaluation/no-grad mode and encode each original labeled example once. Fit one K-class softmax regression model for all `K >= 2`; binary classification is the same two-logit path, not a separate sigmoid artifact.

Use the stable objective:

```text
logits = X W^T + b
loss = mean(logsumexp(logits_i) - logits_i[y_i])
     + (lambda / 2) * ||W||^2
```

Do not regularize the bias. Initialize deterministically, optimize full-batch with core L-BFGS, return convergence diagnostics as a fallible result, and calculate probabilities with stable softmax. Persist `[K, H]` weights, `[K]` bias, ordered type-tagged label vocabulary, L2 coefficient, solver settings, and convergence summary. If class or sample weights are later added, each weight must scale the entire sample contribution and gradient.

Do not reuse the existing training-only `LinearProbe`: its manual SGD path is not the desired convex production head, it does not provide the required artifact contract, and its known weighting/empty-input concerns should not be inherited. Do not make centroids, k-NN, an MLP, or a frozen-encoder shortcut the default called “SetFit.”

### 4. Package One Self-Contained APR

Define a semantic model type such as `setfit_sentence_classifier` with manifest schema `setfit-apr-v1`. The artifact should contain:

- encoder tensors under stable names derived from `NamedModule`;
- `setfit.head.weight` and `setfit.head.bias`;
- exact raw `tokenizer.json` bytes in a U8 payload;
- validated encoder architecture, maximum length, truncation/padding rules, pooling, normalization epsilon, and label order;
- head objective and solver configuration;
- source model ID, full revision, imported-file hashes, dataset fingerprint, trainer configuration, pair-sampler policy/budget, and seeds.

Write F32 tensors for the first production artifact. This permits byte-exact APR tensor round trips and ensures the artifact evaluated before serialization is the artifact deployed afterward. Quantized derivative artifacts can follow with separately reported accuracy and calibration; quantization should not obscure initial SetFit parity.

Validate manifest values, tensor names, dtypes, and shapes before allocating model storage. Use the existing checksum and atomic-writer lifecycle. Extend the high-level APR semantic writer/reader to expose typed U8 payloads; do not redesign `apr-format` if its existing low-level representation is sufficient.

## Crate Ownership

| Crate | Owns | Must Not Own |
|-------|------|--------------|
| `aprender-core` | Canonical tensor/autograd primitives, named module traversal, trainable BERT sentence encoder, pooling/normalization, multinomial head, typed SetFit model/manifest, APR semantic load/save | Dataset iteration, CLI policy, network fetching |
| `aprender-train` | `SetFitTrainer`, schedules, batching loop, AdamW orchestration, gradient clipping, detached embedding pass, L-BFGS head-fit orchestration, metrics/checkpoints | A second BERT/tensor implementation or serving-only model type |
| `aprender-data` | Deterministic bounded pair sampler, class buckets, tokenized batch collation, dataset fingerprints, optional safe cache | Optimizers or model serialization semantics |
| `apr-format` | Existing byte-level APR encoding/decoding and integrity checks | SetFit-specific logic, tokenizer interpretation, or training metadata policy |
| `apr-cli` | Train/import/evaluate command arguments, feature-gated dispatch, actionable errors | Mathematical training implementations |
| `aprender-serve` | Load the same APR through core, tokenize requests once per loaded model, batch inference, expose label probabilities | Python, Hub access, alternate model reconstruction |
| `aprender-contracts` | Cross-crate compatibility and artifact contract tests | Production implementations |

## Feature Gates

Add one model-capability feature instead of backend-specific SetFit variants:

| Feature | Recommendation |
|---------|----------------|
| `setfit` | Enables the typed model, MiniLM/BERT adapter, and `tokenizers`; forward it consistently through core/train/CLI/serve where relevant. |
| `training` | Required for trainer and optimizers, not for a minimal inference consumer. |
| `inference` | Required for APR loading/prediction; a SetFit APR must work with `setfit,inference` and no network feature. |
| `hf-hub` | Optional import/download only; never transitively required by `setfit`. |
| `training-gpu` / `cuda` | Existing optional acceleration gates; do not fork model semantics or artifact formats. |

CI should cover the smallest CPU inference feature set, CPU training, and all-features builds. Avoid one feature per encoder model or per loss; express those differences as validated configuration only after implementations exist.

## Numerical Reference and Verification Strategy

Use Python as an oracle that emits small, immutable fixtures, not as a runtime fallback.

1. Pin the exact MiniLM revision and the reference Python environment in a hash-locked file.
2. Export fixtures for tokenizer IDs/type IDs/masks, transformer token outputs, masked mean, normalized sentence embedding, pair cosine-MSE, selected parameter gradients, one controlled optimizer step, head logits, probabilities, and label order.
3. Disable dropout for cross-framework forward/gradient fixtures. Test Rust seeded dropout placement, reproducibility, and statistics separately; Python and Rust RNG streams should not be expected to match bit-for-bit.
4. Add finite-difference checks for every new differentiable primitive and an integration assertion that representative encoder weights receive finite nonzero gradients.
5. Test masking with padding, all-padding rejection, batch sizes 1 and greater than 1, sequence truncation at 256, singleton/imbalanced classes, unseen/invalid labels, and empty data.
6. Property-test sampler balance, bounded memory, uniqueness rules, seed determinism, and no cross-class positives or same-class negatives.
7. Verify L-BFGS head loss decreases, probabilities are finite and sum to one, and class/sample weighting scales full sample gradients.
8. Round-trip the complete APR and require identical tensor bytes/configuration plus prediction parity. Load the same fixture through core, CLI, and serve contract tests.
9. Benchmark peak memory and examples/pairs per second. The pair sampler must remain O(number of examples + pair budget metadata), never O(all possible pairs).

The first implementation spike should stop at a single MiniLM batch and prove tokenizer parity, full encoder gradient reachability, and one decreasing cosine-MSE step. That is the highest-risk boundary; head fitting and APR assembly should follow only after it passes.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Canonical autograd | `aprender-core::autograd::Tensor` | `aprender-train::autograd::Tensor` | The public BERT model already uses core tensors. Training tensors would force conversions, duplicate transformer work, and risk silent graph detachment. |
| Tensor framework | Extend repository-owned core | Burn, Candle, `tch`, ONNX Runtime | Adds a second model/runtime abstraction, complicates APR and feature gates, or violates the pure-Rust/no-external-runtime deployment intent. Candle may remain useful as non-production comparison tooling, not as SetFit's implementation. |
| First encoder | Pinned `all-MiniLM-L6-v2` | MPNet, BGE, arbitrary Hub models | MiniLM fits Aprender's existing BERT path and is small enough for CPU training. Wider compatibility before strict config/module validation would create misleading support claims. |
| Tokenization | `tokenizers` 0.23.1, pure-Rust features | Handwritten WordPiece or native Oniguruma defaults | The official tokenizer pipeline prevents subtle normalization/pre-tokenization mismatch. Disabling native/default features preserves portability. |
| Pair construction | Bounded deterministic streaming oversampling | Materialized within/cross-class Cartesian products | Full materialization grows quadratically and is unnecessary for balanced stochastic training. |
| Encoder loss | Cosine similarity MSE | InfoNCE, SupCon, CoSENT, triplet | Cosine-MSE is the current SetFit default and establishes a precise compatibility target. Alternatives change semantics and should be separately named experiments. |
| Classification head | K-class softmax regression plus L-BFGS | Existing `LinearProbe`, binary one-vs-rest, MLP, centroids | One convex multinomial head is deterministic, probabilistic, compact, and naturally supports binary and multiclass output. |
| Deployment format | APR v2 | SafeTensors directory, ONNX, pickle/joblib | APR already supplies Aprender's validation/checksum/deployment lifecycle and can contain tokenizer plus typed model metadata. Python formats are not native or safe deployment boundaries. |
| Artifact precision | F32 first | Immediate quantization | F32 isolates algorithm and serialization correctness. Quantization requires its own post-training quality evidence. |
| Reference strategy | Frozen Python-generated fixtures | Python bridge or live-network comparison tests | Fixtures are deterministic and keep production pure Rust; a bridge/network makes tests and deployment environment-dependent. |

## Planned Manifest Changes

```toml
# Workspace dependency: the only new production third-party dependency recommended.
tokenizers = { version = "0.23.1", default-features = false, features = ["fancy-regex"] }
```

```bash
# Reference environment only; commit the resolved lockfile and hashes.
uv add --dev setfit==1.1.3 sentence-transformers==5.7.0 torch==2.13.0 scikit-learn==1.9.0
```

Do not upgrade SafeTensors merely for this milestone. The existing import boundary is sufficient, and unrelated dependency convergence should be handled separately unless the pinned MiniLM file exposes a concrete incompatibility.

## Sources

### Official SetFit and Sentence Transformers Sources

- [SetFit conceptual guide: two-stage body and logistic-regression head](https://huggingface.co/docs/setfit/conceptual_guides/setfit) — HIGH confidence
- [SetFit sampling strategies](https://huggingface.co/docs/setfit/conceptual_guides/sampling_strategies) — HIGH confidence
- [SetFit Trainer reference and defaults](https://huggingface.co/docs/setfit/reference/trainer) — HIGH confidence
- [Sentence Transformers loss reference: `CosineSimilarityLoss`](https://sbert.net/docs/package_reference/sentence_transformer/losses.html) — HIGH confidence

### Official Encoder Sources

- [Hugging Face model API: `sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2) — revision, license, architecture, and parameter metadata; HIGH confidence
- [MiniLM BERT configuration](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/config.json) — HIGH confidence
- [Sentence Transformers module graph](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/modules.json) — HIGH confidence
- [Pooling configuration](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/1_Pooling/config.json) — HIGH confidence
- [Sentence maximum-length configuration](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/sentence_bert_config.json) — HIGH confidence
- [MiniLM model card: masked mean pooling, normalization, and 256-wordpiece truncation](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — HIGH confidence

### Official Dependency Sources

- [`tokenizers` 0.23.1 Rust documentation](https://docs.rs/tokenizers/0.23.1/tokenizers/) — HIGH confidence
- [`tokenizers` 0.23.1 Cargo features](https://docs.rs/crate/tokenizers/0.23.1/source/Cargo.toml) — HIGH confidence
- [`tokenizers` v0.23.1 release](https://github.com/huggingface/tokenizers/releases/tag/v0.23.1) — HIGH confidence
- [SetFit on PyPI](https://pypi.org/project/setfit/) — HIGH confidence for current reference version
- [Sentence Transformers on PyPI](https://pypi.org/project/sentence-transformers/) — HIGH confidence for current reference version
- [PyTorch on PyPI](https://pypi.org/project/torch/) — HIGH confidence for current reference version
- [scikit-learn on PyPI](https://pypi.org/project/scikit-learn/) — HIGH confidence for current reference version

### Repository Evidence

- `.planning/codebase/STACK.md`, `.planning/codebase/ARCHITECTURE.md`, and `.planning/codebase/CONCERNS.md` — existing dependency, crate-boundary, tensor-stack, APR, and known-risk evidence; HIGH confidence
- `crates/aprender-core/src/models/bert/`, `crates/aprender-core/src/models/qwen2/`, `crates/aprender-core/src/autograd/`, `crates/aprender-core/src/nn/optim/`, and `crates/aprender-core/src/optim/lbfgs.rs` — existing BERT/autograd/embedding-backward/optimizer capabilities; HIGH confidence

## Research Gaps for Phase Planning

- Confirm by spike that all existing BERT attention and LayerNorm broadcast paths preserve gradients for `batch > 1`; repository inspection establishes the intended stack but not end-to-end parity.
- Verify the hash-locked Python reference set resolves together on the CI Python version; current individual release versions were verified, but environment compatibility must be captured by the lock solver.
- Decide the typed behavior for singleton classes before freezing the sampler schema. It must be explicit, deterministic, and reflected in reference fixtures.
- Add other Sentence Transformers families only after a model-specific config/module/tensor-name compatibility matrix exists.
