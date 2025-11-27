# Future Learning Approaches Specification

Candidate architectures and capabilities for aprender expansion.

## Current Inventory

**Have:**
- Supervised: LinearReg, LogisticReg, SVM, KNN, NaiveBayes, DecisionTree, RandomForest, GBM
- Unsupervised: KMeans, DBSCAN, Hierarchical, GMM, PCA, ICA, t-SNE, Spectral
- Deep: Sequential NN, Conv2D, Transformer, Attention, LSTM (partial)
- Ensemble: MoE, Bagging, Boosting
- Optimization: SGD, Adam, AdaGrad, RMSprop, LBFGS, Differential Evolution
- Text: TF-IDF, Tokenizers, Stemming, NER, Sentiment, Topic
- Graph: Centrality, Community, Pathfinding, PageRank
- Time Series: ARIMA
- Recommender: Content-based
- AutoML: TPE, Grid/Random Search

**Missing:** See below

## 1. Generative Adversarial Networks (GAN)

**Architecture:** Generator G(z) vs Discriminator D(x), minimax game

**Use Cases:**
- Synthetic data augmentation
- Domain adaptation (source → target distribution)
- Anomaly detection (discriminator as novelty scorer)

**Variants:**
| Variant | Description | Complexity |
|---------|-------------|------------|
| Vanilla GAN | Original minimax | Low |
| WGAN | Wasserstein distance, stable training | Medium |
| WGAN-GP | Gradient penalty regularization | Medium |
| Conditional GAN | Class-conditioned generation | Medium |

**Priority:** Low (synthetic module covers data augmentation)

---

## 2. Transformer Encoder

**Status:** ✅ Implemented (`src/nn/transformer.rs`)

**Enhancements:**
- [ ] Pre-trained weights loading (BERT, RoBERTa)
- [ ] Efficient attention (Linformer, Performer)
- [ ] Rotary position embeddings (RoPE)
- [ ] Flash Attention (memory-efficient)

**Use Cases:**
- Code understanding (depyler oracle)
- Text classification
- Sequence embedding

**Priority:** High (foundation exists, needs pretrained support)

---

## 3. Contrastive Learning

**Core Idea:** Learn embeddings where similar items cluster, dissimilar repel

**Loss Functions:**
```
InfoNCE: -log(exp(sim(z_i, z_j)/τ) / Σ exp(sim(z_i, z_k)/τ))
Triplet:  max(0, d(a,p) - d(a,n) + margin)
NT-Xent:  Normalized temperature-scaled cross-entropy
```

**Variants:**
| Method | Description | Use Case |
|--------|-------------|----------|
| SimCLR | Augmentation + projection head | Vision, general |
| MoCo | Momentum encoder, queue | Large batch alternative |
| CLIP | Image-text pairs | Multimodal |
| SimCSE | Dropout as augmentation | Text embeddings |

**Use Cases:**
- Error similarity (depyler oracle)
- Code clone detection
- Semantic search

**Priority:** High (complements existing embedding infrastructure)

---

## 4. Variational Autoencoder (VAE)

**Architecture:** Encoder → latent z ~ N(μ, σ) → Decoder

**Loss:** Reconstruction + KL divergence

**Use Cases:**
- Latent space interpolation
- Anomaly detection (reconstruction error)
- Controlled generation

**Priority:** Medium

---

## 5. Diffusion Models

**Core Idea:** Learn to denoise, reverse diffusion process

**Variants:**
| Model | Description |
|-------|-------------|
| DDPM | Denoising diffusion probabilistic |
| DDIM | Deterministic sampling (faster) |
| Latent Diffusion | Diffuse in latent space (efficient) |

**Use Cases:**
- High-quality generation
- Inpainting/completion
- Code synthesis (emerging)

**Priority:** Low (compute-intensive, specialized)

---

## 6. Graph Neural Networks (GNN)

**Status:** Partial (`src/graph/` has algorithms, no learned GNN)

**Architectures:**
| Model | Aggregation | Use Case |
|-------|-------------|----------|
| GCN | Mean neighbors | Node classification |
| GAT | Attention-weighted | Heterogeneous graphs |
| GraphSAGE | Sampled neighbors | Large graphs |
| GIN | Sum (injective) | Graph classification |

**Use Cases:**
- AST analysis (depyler)
- Dependency graphs
- Molecular property prediction

**Priority:** Medium (graph infra exists)

---

## 7. Mixture of Experts (MoE)

**Status:** ✅ Implemented (`src/ensemble/`)

**Enhancements:**
- [ ] Expert parallelism
- [ ] Auxiliary load balancing loss
- [ ] Dynamic expert allocation

---

## 8. Reinforcement Learning (RL)

**Components Needed:**
- Environment abstraction
- Policy networks (Actor-Critic)
- Replay buffer
- TD/MC value estimation

**Use Cases:**
- Hyperparameter tuning (RL-based AutoML)
- Code optimization decisions
- Interactive systems

**Priority:** Low (specialized domain)

---

## 9. Self-Supervised Learning

**Core Idea:** Learn representations from unlabeled data via pretext tasks

**Methods:**
| Method | Pretext Task | Domain |
|--------|--------------|--------|
| Masked LM | Predict masked tokens | Text |
| Next Sentence | Predict if sentences consecutive | Text |
| Rotation | Predict image rotation | Vision |
| Jigsaw | Solve puzzle patches | Vision |
| BYOL | Bootstrap own representations | General |

**Priority:** High (enables pretraining on unlabeled code)

---

## 10. Meta-Learning (Learning to Learn)

**Approaches:**
| Method | Description |
|--------|-------------|
| MAML | Model-Agnostic Meta-Learning, gradient-based |
| Prototypical Networks | Class prototypes in embedding space |
| Matching Networks | Attention over support set |
| Reptile | First-order MAML approximation |

**Use Cases:**
- Few-shot classification
- Quick adaptation to new error types
- Transfer learning acceleration

**Priority:** Medium

---

## 11. Neural Architecture Search (NAS)

**Search Strategies:**
| Strategy | Description |
|----------|-------------|
| Random Search | Baseline |
| Evolutionary | Mutate/crossover architectures |
| DARTS | Differentiable architecture search |
| ENAS | Weight sharing, controller RNN |

**Use Cases:**
- AutoML architecture optimization
- Task-specific network design

**Priority:** Low (expensive, specialized)

---

## 12. Attention Variants

**Beyond Vanilla:**
| Variant | Complexity | Description |
|---------|------------|-------------|
| Multi-Head | O(n²d) | Parallel attention heads |
| Linear Attention | O(nd²) | Kernel approximation |
| Sparse Attention | O(n√n) | Fixed/learned patterns |
| Flash Attention | O(n²d) | Memory-efficient, fused |
| Cross Attention | O(nmd) | Attend to different sequence |
| Grouped Query | O(n²d/g) | Shared KV heads |

**Priority:** High (transformer enhancement)

---

## 13. Sequence-to-Sequence

**Have:** Transformer encoder/decoder

**Missing:**
- [ ] Beam search decoding
- [ ] Nucleus (top-p) sampling
- [ ] Encoder-decoder attention mask
- [ ] Teacher forcing scheduler
- [ ] Label smoothing

**Priority:** High (generation tasks)

---

## 14. Object Detection / Segmentation

**Architectures:**
| Model | Type | Description |
|-------|------|-------------|
| YOLO | One-stage | Fast, single pass |
| Faster R-CNN | Two-stage | Region proposals |
| U-Net | Segmentation | Encoder-decoder skip |
| Mask R-CNN | Instance seg | Box + mask heads |

**Priority:** Low (vision-specific)

---

## 15. Recurrent Architectures

**Have:** LSTM (partial)

**Missing:**
| Architecture | Description |
|--------------|-------------|
| GRU | Simplified LSTM, fewer gates |
| Bidirectional | Forward + backward pass |
| Stacked RNN | Multiple layers |
| Attention RNN | Bahdanau/Luong attention |

**Priority:** Medium (transformers often preferred)

---

## 16. Normalization Techniques

**Have:** LayerNorm, BatchNorm

**Missing:**
| Technique | Description |
|-----------|-------------|
| GroupNorm | Normalize over channel groups |
| InstanceNorm | Per-sample, per-channel |
| RMSNorm | Root mean square (faster) |
| PowerNorm | Running stats without momentum |

**Priority:** Medium

---

## 17. Regularization Techniques

**Have:** L1, L2, Dropout

**Missing:**
| Technique | Description |
|-----------|-------------|
| DropConnect | Drop weights not activations |
| DropBlock | Drop contiguous regions |
| Mixup | Interpolate samples |
| CutMix | Cut and paste patches |
| Label Smoothing | Soft targets |
| Stochastic Depth | Drop entire layers |
| R-Drop | Consistency between dropout runs |

**Priority:** Medium

---

## 18. Loss Functions

**Have:** MSE, MAE, CrossEntropy, Huber

**Missing:**
| Loss | Use Case |
|------|----------|
| Focal Loss | Class imbalance |
| Dice Loss | Segmentation |
| CTC Loss | Sequence alignment |
| Contrastive Loss | Embedding learning |
| Triplet Loss | Metric learning |
| Hinge Loss | SVM-style margin |
| KL Divergence | Distribution matching |
| Wasserstein | GAN training |

**Priority:** High

---

## 19. Interpretability / Explainability

**Methods:**
| Method | Type | Description |
|--------|------|-------------|
| SHAP | Feature attribution | Shapley values |
| LIME | Local surrogate | Interpretable local model |
| Integrated Gradients | Gradient-based | Attribution along path |
| Attention Viz | Attention weights | What model attends to |
| Saliency Maps | Gradient-based | Input sensitivity |
| Counterfactuals | Example-based | "What if" explanations |

**Priority:** High (trust, debugging, compliance)

---

## 20. Calibration

**Problem:** Model confidence ≠ actual accuracy

**Methods:**
| Method | Description |
|--------|-------------|
| Temperature Scaling | Single parameter post-hoc |
| Platt Scaling | Logistic regression on logits |
| Isotonic Regression | Non-parametric |
| Expected Calibration Error | Metric |

**Priority:** Medium (production systems)

---

## 21. Active Learning

**Strategies:**
| Strategy | Description |
|----------|-------------|
| Uncertainty Sampling | Query most uncertain |
| Query-by-Committee | Disagreement among models |
| Expected Model Change | Maximize gradient magnitude |
| Core-Set | Diverse coverage |

**Use Cases:**
- Label-efficient training
- Human-in-the-loop

**Priority:** Medium

---

## 22. Federated Learning

**Components:**
- Local training on devices
- Secure aggregation (FedAvg)
- Differential privacy
- Communication compression

**Priority:** Low (infrastructure-heavy)

---

## 23. Quantization-Aware Training

**Have:** Post-training quantization (GGUF)

**Missing:**
- [ ] QAT (fake quantization during training)
- [ ] Mixed precision training (FP16/BF16)
- [ ] Dynamic quantization
- [ ] Quantization-aware fine-tuning

**Priority:** Medium (deployment optimization)

---

## 24. Knowledge Distillation Variants

**Have:** Basic distillation

**Missing:**
| Variant | Description |
|---------|-------------|
| Feature Distillation | Match intermediate layers |
| Attention Transfer | Match attention maps |
| Self-Distillation | Deeper layers teach shallower |
| Online Distillation | Co-train teacher and student |

**Priority:** Medium

---

## 25. Causal Inference

**Methods:**
| Method | Description |
|--------|-------------|
| Propensity Scoring | Treatment effect estimation |
| Instrumental Variables | Handle confounders |
| DoWhy | Causal graph framework |
| Double ML | Debiased ML estimators |

**Priority:** Low (specialized domain)

---

## 26. Survival Analysis

**Models:**
| Model | Description |
|-------|-------------|
| Kaplan-Meier | Non-parametric survival |
| Cox Proportional Hazards | Semi-parametric |
| DeepSurv | Neural Cox model |
| Random Survival Forest | Ensemble for censored data |

**Priority:** Low (specialized domain)

---

## 27. Multi-Task Learning

**Architectures:**
| Type | Description |
|------|-------------|
| Hard Sharing | Shared backbone, task heads |
| Soft Sharing | Cross-stitch networks |
| Task-Specific Adapters | Frozen backbone + adapters |
| Gradient Surgery | Resolve conflicting gradients |

**Priority:** Medium

---

## 28. Continual/Lifelong Learning

**Challenge:** Catastrophic forgetting

**Methods:**
| Method | Description |
|--------|-------------|
| EWC | Elastic weight consolidation |
| PackNet | Iterative pruning |
| Progressive Nets | Add capacity per task |
| Replay Buffer | Store old examples |

**Priority:** Low

---

## 29. Data Labeling & Weak Supervision

**Problem:** Labels are expensive, noisy, or missing

**Have:** `synthetic/weak_supervision.rs` (basic)

**Approaches:**
| Method | Description |
|--------|-------------|
| Labeling Functions | Programmatic heuristics (Snorkel) |
| Label Model | Combine noisy LFs probabilistically |
| Data Programming | Write functions, not labels |
| Crowd Aggregation | Dawid-Skene, GLAD |
| Self-Training | Pseudo-labels from confident predictions |
| Co-Training | Multi-view bootstrapping |

**Noise Handling:**
| Technique | Description |
|-----------|-------------|
| Confident Learning | Find label errors via C_ij matrix |
| MentorNet | Curriculum from clean subset |
| Co-teaching | Two networks filter noise |
| Label Smoothing | Soft targets reduce overconfidence |

**Missing:**
- [ ] Snorkel-style label model (generative)
- [ ] Confident learning (cleanlab)
- [ ] Programmatic labeling DSL
- [ ] Annotation conflict resolution
- [ ] Active labeling prioritization

**Priority:** High (data-centric AI)

---

## 30. Semi-Structured Data ML

**Problem:** Data between structured (SQL) and unstructured (text)

**Data Types:**
| Type | Examples |
|------|----------|
| JSON/JSONL | API responses, logs |
| XML/HTML | Web pages, configs |
| Nested Tables | Hierarchical spreadsheets |
| Key-Value | Forms, receipts |
| Markdown/RST | Documentation |
| YAML/TOML | Config files |
| AST/IR | Code representations |

**Tasks:**
| Task | Description |
|------|-------------|
| Schema Inference | Discover structure automatically |
| Entity Extraction | Pull typed values from docs |
| Table Detection | Find tables in documents |
| Form Parsing | Key-value from forms |
| Nested Flattening | Hierarchical → tabular |
| JSON Path Prediction | Learn extraction paths |

**Architectures:**
| Model | Use Case |
|-------|----------|
| LayoutLM | Document understanding (spatial + text) |
| TAPAS | Table QA |
| TaBERT | Joint table-text encoding |
| Donut | Document OCR + understanding |
| TreeLSTM | AST/tree structures |

**Have:**
- `data/` DataFrame (flat tabular)
- `text/` tokenizers
- `graph/` for tree structures

**Missing:**
- [ ] JSON schema inference
- [ ] Nested DataFrame (hierarchical)
- [ ] Document layout features
- [ ] Table extraction from text
- [ ] Form field detection
- [ ] Semi-structured serialization

**Priority:** High (real-world data is messy)

---

## 31. Code-Specific ML

**Relevance:** depyler, transpilation, oracle

**Representations:**
| Representation | Description |
|----------------|-------------|
| Token Sequence | Raw code as text |
| AST | Abstract syntax tree |
| CFG | Control flow graph |
| DFG | Data flow graph |
| PDG | Program dependence graph |
| IR | Intermediate representation |

**Tasks:**
| Task | Description |
|------|-------------|
| Code Classification | Bug detection, vulnerability |
| Code Search | Semantic retrieval |
| Code Completion | Next token/line prediction |
| Code Summarization | Docstring generation |
| Code Translation | Language-to-language |
| Type Inference | Predict missing types |
| Error Prediction | Compiler error classification |

**Models:**
| Model | Description |
|-------|-------------|
| CodeBERT | Pretrained on code+NL |
| GraphCodeBERT | + data flow edges |
| CodeT5 | Encoder-decoder for code |
| TreeSitter | Fast incremental parsing |
| Code2Vec | Path-based embeddings |

**Have:**
- `synthetic/code_eda.rs` - code analysis
- `synthetic/code_features.rs` - feature extraction
- `graph/` - could represent CFG/AST

**Missing:**
- [ ] AST-aware tokenizer
- [ ] Tree-structured encoders
- [ ] Code-specific pretraining
- [ ] Error message embeddings
- [ ] Cross-language alignment

**Priority:** 🔴 Critical (depyler oracle core need)

---

## 32. Embedding & Retrieval

**Problem:** Semantic search, similarity, RAG

**Components:**
| Component | Description |
|-----------|-------------|
| Encoder | Text/code → dense vector |
| Index | ANN search (HNSW, IVF) |
| Retriever | Query → top-k docs |
| Reranker | Refine initial results |

**Have:**
- `index/hnsw.rs` - approximate NN search
- `text/vectorize.rs` - TF-IDF (sparse)

**Missing:**
| Component | Description |
|-----------|-------------|
| Dense Retriever | Learned bi-encoder |
| ColBERT | Late interaction |
| Hybrid Search | Sparse + dense fusion |
| Cross-Encoder | Reranking |
| Embedding Fine-tune | Contrastive on domain |

**Priority:** High (RAG, semantic search)

---

## 33. Transfer Learning for Transpilers

**Context:** Cross-project knowledge sharing across transpiler ecosystem

**Projects:**
| Project | Direction | Domain |
|---------|-----------|--------|
| depyler | Python → Rust | General purpose |
| ruchy | Ruby → Rust | Ruby idioms |
| ruchyruchy | Ruby → Ruby (refactor) | Same-language |
| decy | ? → Rust | TBD |
| bashrs | Bash → Rust | Shell scripts |

**Transfer Opportunities:**

### Shared Representations
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Python    │     │    Ruby     │     │    Bash     │
│   Source    │     │   Source    │     │   Source    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────┐
│          Shared Error Embedding Space               │
│     (E0308, E0277, E0425, E0599, ... )             │
└─────────────────────────────────────────────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  depyler     │   │   ruchy      │   │   bashrs     │
│  Oracle      │   │   Oracle     │   │   Oracle     │
└──────────────┘   └──────────────┘   └──────────────┘
```

### Transfer Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Feature Transfer** | Shared encoder, task-specific heads | Related tasks |
| **Fine-tuning** | Pretrain on all, fine-tune per project | Large base, small target |
| **Multi-task** | Joint training with shared backbone | Concurrent development |
| **Domain Adaptation** | Align source→target distributions | Different syntax styles |
| **Zero-shot** | Apply directly without target data | New transpiler bootstrap |

### Transferable Components

| Component | Transferable? | Notes |
|-----------|---------------|-------|
| Error embeddings | ✅ High | Rust errors shared across all |
| Type inference | ✅ High | Rust type system universal |
| Lifetime analysis | ✅ High | Same borrow checker |
| AST patterns | 🟡 Medium | Language-specific syntax |
| Idiom mapping | 🟡 Medium | Source-language dependent |
| API mapping | ❌ Low | stdlib differs per source |

### Pretraining Strategy

**Phase 1: Unified Error Corpus**
```
Collect errors from all 5 transpilers
         ↓
Train shared error encoder (CodeBERT/ContrastiveError)
         ↓
Freeze encoder, share across projects
```

**Phase 2: Multi-Source Pretraining**
```
┌─────────────────────────────────────┐
│  Combined AST Dataset               │
│  - Python ASTs (depyler)            │
│  - Ruby ASTs (ruchy, ruchyruchy)    │
│  - Bash ASTs (bashrs)               │
└─────────────────────────────────────┘
         ↓
Train language-agnostic code encoder
         ↓
Fine-tune per transpiler
```

**Phase 3: Cross-Transpiler MoE**
```
┌─────────────────────────────────────┐
│        Gating Network               │
│   (detect source language)          │
└───────────────┬─────────────────────┘
        ┌───────┼───────┐
        ▼       ▼       ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │depyler │ │ ruchy  │ │ bashrs │
   │ Expert │ │ Expert │ │ Expert │
   └────────┘ └────────┘ └────────┘
```

### Implementation Checklist

- [ ] Unified error schema across projects
- [ ] Shared Rust target embedding space
- [ ] Cross-project error dataset
- [ ] Language-agnostic AST encoder
- [ ] Transfer learning primitives in aprender
- [ ] MoE router for multi-transpiler

### Concrete APIs Needed

```rust
// Transfer learning module
pub trait TransferEncoder: Module {
    fn freeze_base(&mut self);
    fn unfreeze_base(&mut self);
    fn get_features(&self, x: &Tensor) -> Tensor;
}

// Domain adaptation
pub struct DomainAdapter {
    source_encoder: Box<dyn TransferEncoder>,
    target_encoder: Box<dyn TransferEncoder>,
    discriminator: Linear,  // adversarial alignment
}

// Multi-task head
pub struct MultiTaskHead {
    shared_encoder: Box<dyn TransferEncoder>,
    task_heads: HashMap<String, Linear>,
}
```

**Priority:** 🔴 Critical (5x leverage across transpiler ecosystem)

---

## 34. Distillation Ingestion from entrenar

**Context:** entrenar trains/distills large models → aprender consumes for inference

**Ecosystem Flow:**
```
┌─────────────────────────────────────────────────────────────┐
│                       TRAINING (entrenar)                   │
├─────────────────────────────────────────────────────────────┤
│  Teacher (7B+)  ──distill──►  Student (1B)  ──quantize──►  │
│                                                    Q4_0     │
└──────────────────────────────┬──────────────────────────────┘
                               │ .apr file
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      INFERENCE (aprender)                   │
├─────────────────────────────────────────────────────────────┤
│  Load .apr  ──►  Verify provenance  ──►  SIMD inference    │
│                  (teacher hash,                             │
│                   distill params)                           │
└─────────────────────────────────────────────────────────────┘
```

**Have (in .apr format):**
- DistillationInfo metadata (teacher hash, method, params)
- Quantization support (Q4_0, Q8_0)
- Model provenance chain

**Missing:**
| Feature | Description |
|---------|-------------|
| Teacher verification | Validate teacher hash matches known models |
| Distillation metrics | Ingest final_loss, layer alignment scores |
| Progressive loading | Stream large distilled models |
| Adapter ingestion | Load LoRA/adapter weights from entrenar |
| Checkpoint resume | Continue distillation from aprender |

### Adapter/LoRA Support

```rust
// entrenar trains adapters, aprender loads them
pub struct AdapterConfig {
    pub rank: usize,           // LoRA rank (4, 8, 16)
    pub alpha: f32,            // Scaling factor
    pub target_modules: Vec<String>,  // ["q_proj", "v_proj"]
}

pub struct AdapterWeights {
    pub lora_a: HashMap<String, Tensor>,  // down-projection
    pub lora_b: HashMap<String, Tensor>,  // up-projection
}

// Load base + adapter
let base = load::<TransformerEncoder>("base.apr")?;
let adapter = load_adapter("adapter.apr")?;
let merged = base.merge_adapter(&adapter);  // W' = W + BA
```

### Distillation Chain Verification

```rust
// Verify provenance before inference
let info = inspect("student.apr")?;
let distill = info.distillation.expect("distilled model");

// Check teacher is trusted
assert!(TRUSTED_TEACHERS.contains(&distill.teacher.hash));

// Check distillation quality
assert!(distill.params.final_loss < 0.5);
assert!(distill.method == DistillMethod::Progressive);
```

### Integration Points

| entrenar produces | aprender consumes |
|-------------------|-------------------|
| `student.apr` | `load()` for inference |
| `adapter.apr` | `load_adapter()` + merge |
| `checkpoint.apr` | Resume training (future) |
| `teacher.apr` | Reference for verification |

### Shared Format Extensions

```rust
// New fields for entrenar→aprender handoff
pub struct TrainingMetadata {
    pub trained_by: String,       // "entrenar-v0.5.0"
    pub training_steps: u64,
    pub final_metrics: HashMap<String, f32>,
    pub hardware: String,         // "A100-80GB"
    pub training_time_secs: u64,
}

pub struct AdapterInfo {
    pub base_model_hash: [u8; 32],
    pub adapter_type: AdapterType,  // LoRA, Prefix, Adapter
    pub config: AdapterConfig,
}
```

**Priority:** 🔴 High (completes training→inference pipeline)

---

## Implementation Roadmap (Revised)

### 🔴 Critical (transpiler ecosystem)
| Phase | Items | Effort |
|-------|-------|--------|
| 1 | Code-specific ML: AST tokenizer, error embeddings | 2 weeks |
| 2 | Contrastive/Triplet Loss | 1 week |
| 3 | GNN layers (GCN, GAT) for AST/CFG | 2 weeks |
| 4 | Transfer learning: TransferEncoder trait, freeze/unfreeze | 1 week |
| 5 | Unified error corpus (depyler+ruchy+bashrs) | 1 week |
| 6 | Cross-transpiler MoE router | 1 week |

### 🔴 High Priority
| Phase | Items | Effort |
|-------|-------|--------|
| 7 | entrenar integration: LoRA/adapter loading | 1 week |
| 8 | entrenar integration: Provenance verification | 1 week |
| 9 | Weak supervision: Snorkel-style label model | 2 weeks |
| 10 | Semi-structured: JSON schema inference, nested DF | 2 weeks |
| 11 | Embedding/Retrieval: Dense retriever, hybrid search | 2 weeks |
| 12 | SHAP/LIME interpretability | 2 weeks |
| 13 | Attention variants (Flash, Linear) | 2 weeks |
| 14 | Beam search + nucleus sampling | 1 week |
| 15 | Self-supervised pretext tasks | 2 weeks |

### 🟡 Medium Priority
| Phase | Items | Effort |
|-------|-------|--------|
| 11 | Additional loss functions (Focal, CTC, KL) | 1 week |
| 12 | Normalization (RMSNorm, GroupNorm) | 1 week |
| 13 | Regularization (DropBlock, Mixup, CutMix) | 1 week |
| 14 | RNN variants (GRU, Bidirectional) | 1 week |
| 15 | VAE module | 1 week |
| 16 | Calibration (temperature scaling) | 1 week |
| 17 | Active learning strategies | 1 week |
| 18 | Multi-task learning | 2 weeks |
| 19 | Meta-learning (MAML, Prototypical) | 2 weeks |

### 🟢 Low Priority
| Phase | Items | Effort |
|-------|-------|--------|
| 20 | GAN (WGAN-GP) | 2 weeks |
| 21 | Diffusion (DDPM) | 3 weeks |
| 22 | RL (PPO, A2C) | 3 weeks |
| 23 | NAS (DARTS) | 3 weeks |
| 24 | Object Detection (YOLO) | 3 weeks |
| 25 | Federated Learning | 3 weeks |
| 26 | Causal Inference | 2 weeks |
| 27 | Survival Analysis | 2 weeks |
| 28 | Continual Learning (EWC) | 2 weeks |

---

## 35. Toyota Way Review & Annotations

**Reviewer:** Gemini CLI (2025-11-27)

### Principles Applied
1.  **Muda (Waste):** The inclusion of specialized domains like **Survival Analysis (Sec 26)** and **Causal Inference (Sec 25)** may represent *overproduction* relative to the core transpiler ecosystem goals. Recommendation: Isolate these as optional extensions to preserve a lean core.
2.  **Genchi Genbutsu (Go and See):** The roadmap prioritizes the `depyler` integration (Sec 31, 33), grounding development in actual downstream user needs.
3.  **Visual Control:** The consistent tabular structure for "Variants" and "Missing" features provides excellent visual management of the backlog.

---

## References

**Architectures:**
- [Attention Is All You Need (Vaswani 2017)](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su 2024)](https://doi.org/10.1016/j.neucom.2023.127063) *[Added]*
- [Switch Transformers: Scaling to Trillion Parameter Models (Fedus 2022)](https://arxiv.org/abs/2101.03961) *[Added]*
- [GCN (Kipf 2017)](https://arxiv.org/abs/1609.02907)
- [Inductive Representation Learning on Large Graphs (GraphSAGE) (Hamilton 2017)](https://arxiv.org/abs/1706.02216) *[Added]*
- [Flash Attention (Dao 2022)](https://arxiv.org/abs/2205.14135)

**Generative:**
- [WGAN-GP (Gulrajani 2017)](https://arxiv.org/abs/1704.00028)
- [DDPM (Ho 2020)](https://arxiv.org/abs/2006.11239)
- [VAE (Kingma 2014)](https://arxiv.org/abs/1312.6114)

**Contrastive/Self-Supervised:**
- [SimCLR (Chen 2020)](https://arxiv.org/abs/2002.05709)
- [SimCSE: Simple Contrastive Learning of Sentence Embeddings (Gao 2021)](https://arxiv.org/abs/2104.08821) *[Added]*
- [MoCo (He 2020)](https://arxiv.org/abs/1911.05722)
- [BYOL (Grill 2020)](https://arxiv.org/abs/2006.07733)
- [Masked Autoencoders Are Scalable Vision Learners (He 2022)](https://arxiv.org/abs/2111.06377) *[Added]*

**Weak Supervision:**
- [Snorkel (Ratner 2017)](https://arxiv.org/abs/1711.10160)
- [Confident Learning (Northcutt 2021)](https://arxiv.org/abs/1911.00068)

**Code ML:**
- [CodeBERT (Feng 2020)](https://arxiv.org/abs/2002.08155)
- [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models (Wang 2021)](https://arxiv.org/abs/2109.00859) *[Added]*
- [GraphCodeBERT (Guo 2021)](https://arxiv.org/abs/2009.08366)
- [Code2Vec (Alon 2019)](https://arxiv.org/abs/1803.09473)

**Semi-Structured:**
- [LayoutLM (Xu 2020)](https://arxiv.org/abs/1912.13318)
- [TAPAS (Herzig 2020)](https://arxiv.org/abs/2004.02349)

**Interpretability:**
- [SHAP (Lundberg 2017)](https://arxiv.org/abs/1705.07874)
- [LIME (Ribeiro 2016)](https://arxiv.org/abs/1602.04938)
- [Axiomatic Attribution for Deep Networks (Integrated Gradients) (Sundararajan 2017)](https://arxiv.org/abs/1703.01365) *[Added]*

**Distillation & Efficiency:**
- [Distilling the Knowledge in a Neural Network (Hinton 2015)](https://arxiv.org/abs/1503.02531) *[Added]*
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu 2021)](https://arxiv.org/abs/2106.09685) *[Added]*

**Meta-Learning:**
- [MAML (Finn 2017)](https://arxiv.org/abs/1703.03400)
- [Prototypical Networks (Snell 2017)](https://arxiv.org/abs/1703.05175)

**Specialized Domains:**
- [DeepSurv: Personalized Treatment Recommender System (Katzman 2018)](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1) *[Added]*

**Continual Learning:**
- [EWC (Kirkpatrick 2017)](https://arxiv.org/abs/1612.00796)
