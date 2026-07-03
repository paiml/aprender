# Aprender Roadmap

Next Generation Machine Learning in Pure Rust

## Current status

> **v0.57.0** (2026-07-03) — published to crates.io: `cargo install aprender` → `apr`.
> **Actively shipping** the four-pillar **BEAT campaign** (below) plus a **QLoRA
> training-correctness wave** (v0.55–v0.57). The earlier "3-month hiatus" was closed
> out; development resumed at v0.42 and is ongoing.

Aprender is now an **86-crate monorepo** spanning model **training**, **inference**
(realizar), **SIMD/GPU compute** (trueno), and a **104-subcommand `apr` CLI** — well
beyond the single-algorithm releases this file originally tracked. 25,300+ tests,
1,400+ provable contracts. *(Live counts drift; the enforced source of truth is
[`contracts/readme-claims-v1.yaml`](contracts/readme-claims-v1.yaml), which gates the
README's counts against `find contracts/ -name '*.yaml'` at HEAD.)*

**This file is a high-level version overview, not the working backlog.** The
canonical, issue-linked roadmap is
[`docs/roadmaps/roadmap.yaml`](docs/roadmaps/roadmap.yaml); per-release detail is in
[`CHANGELOG.md`](CHANGELOG.md) and active design specs in
[`docs/specifications/`](docs/specifications/).

| Version | Status | Headline |
|---------|--------|----------|
| v0.1.0 | ✅ Released | Foundation — Linear Regression + K-Means |
| v0.2.0–v0.4.1 | ✅ Released | TOP 10 ML algorithms, trees/ensembles, cross-validation, graph algorithms, stats |
| v0.7.x | ✅ Released | ARIMA, text processing, Bayesian inference, GLMs, ICA |
| v0.11–v0.30 | ✅ Released | Monorepo consolidation, `apr` CLI, APR/GGUF/SafeTensors formats, quantization, inference server |
| v0.31–v0.34 | ✅ Released | MoE serving, distillation pipeline, MODEL-1 code model, provable-contract expansion |
| v0.35.x | ✅ Released | Hiatus close-out (eval/distill fixes, book completeness, DX) |
| v0.36–v0.41 | ✅ Released | sklearn-parity breadth — model-selection stack, 13 metrics, preprocessing, `datasets` |
| v0.42–v0.49 | ✅ Released | **Four-pillar BEAT campaign** — CI-gated falsifiable wins vs sklearn / PyTorch / Unsloth / Ollama (below) |
| v0.50–v0.54 | ✅ Released | Correctness wave — sampling params, Blackwell-GPU coherence, Gemma CPU, APR-format safety, RoPE/LoRA/MCP, `apr-format` leaf extraction (each contract-backed) |
| **v0.55–v0.57** | ✅ **Current** | **GPU QLoRA training actually works** — a 6-defect cascade fixed (deadlock → loss-window → cross-stream NaN → wrong-model → stale-adapter eval → CPU-only eval), each with a mutation-verified falsifier + contract; runnable `--merge`; rank-aware auto-lr (default no longer diverges) |
| _in flight_ | 🔧 Hardening | CUDA CI lane so the GPU correctness falsifiers are actually enforced (see doctrine gap below); apr-code tool-call flip envelope measurement |
| _next_ | 📋 Planned | tracked in [`docs/roadmaps/roadmap.yaml`](docs/roadmaps/roadmap.yaml) |

### 🎯 The mission (north star)

**One pure-Rust binary that REPLACES and BEATS the four incumbents at what each does
best** — where "beat" is not parity but a *falsifiable, CI-gated benchmark* that fails
the build if `apr` regresses below the incumbent's pinned baseline. The cross-cutting
wedge none of the four have: **provable, contract-gated correctness.**

| Pillar | Incumbent | Signature strength | ✅ WON beat (CI-gated falsifier) |
|--------|-----------|--------------------|----------------------------------|
| **P1** | scikit-learn | Classical-ML breadth & ergonomics | LinearRegression **2.0× faster** (LAPACK-free O(nd)), iris-RF accuracy. *(LAPACK-bound Ridge/Lasso/KMeans/PCA honestly conceded.)* |
| **P2** | PyTorch | Tensors + autograd + training | Autograd gradients **≡ PyTorch, max\|Δ\|=5e-7**. *(Training speed conceded — ~11× MKL gap; the win is provable gradient correctness.)* |
| **P3** | Unsloth | Fast low-VRAM PEFT (LoRA/QLoRA) | NF4 quant **≡ bitsandbytes (4.9e-7)** + LoRA-merge forward-**equivalence (1.5e-8)**; **GPU QLoRA training runs correct end-to-end** (v0.55–v0.57: 6-defect cascade fixed, trains+validates at GPU speed on RTX 4090 & GB10 Blackwell). *(GPU-Triton tok/s conceded; training-correctness falsifiers are developer-run pending a CUDA CI lane.)* |
| **P4** | Ollama / llama.cpp | Fast local quantized inference | **Fail-closed correctness** — `apr` rejects 10/10 semantically-broken models that Ollama/llama.cpp silently run; decode **1.2–1.37× on RTX 4090**. |

All four beats are **adversarially mutation-verified** — a deliberately injected
regression must make the gate FAIL, or the gate is theater. Cross-cutting through-line:
the model is trained → fine-tuned → distilled → served as one CODE model
(qwen2.5-coder), measured by the apr-code-parity matrix. As of v0.55–v0.57 the
**fine-tune stage of that through-line trains and validates correctly** (GPU QLoRA), so
the lifecycle is closer to demonstrable end-to-end.

**Campaign mode (2026-06):** ≥10 days fully-autonomous, BEATS-as-CI-artifacts across all
four pillars plus the `cuda-oxide` pure-Rust→PTX spike on Blackwell. Every correctness
fix ships a named `proof_obligation` + a falsifier verified RED-on-bug / GREEN-on-fix +
a `pv`-validated contract bump (a bug shipped green ⇔ its falsifier was missing or too
weak).

### ⚠️ Honest doctrine gap: GPU falsifiers are not yet CI-enforced

The v0.55–v0.57 QLoRA training-correctness falsifiers (deadlock, cross-stream NaN,
wrong-model rope/bias, stale-adapter eval, GPU-vs-CPU eval-forward) require a CUDA
device and are `#[ignore]`-gated — the CI fleet has **no CUDA-labeled runner**, so
today they only run developer-side on the pre-authorized RTX 4090. By the "gates or
theater" rule this is a real hole: a correctness gate that never runs unattended is
weaker than one that does. **The highest-EV hardening item is a nightly CUDA lane on
the 4090** that executes these falsifiers as a scheduled (eventually blocking) check.
Until it lands, these wins are stated as *developer-verified*, not *CI-enforced* — the
same discipline §6 applies to conceded speed.

**Case-file lesson (CF-5, 2026-07-03):** the GPU eval-forward falsifier's tolerance band
*fired against its own CPU reference* (Δ=12.29 ≫ 0.5). Rather than loosen the band, the
right move was to bisect — which exposed an **unrelated** defect: `forward_with_lora`
had been silently dropping the Q/K/V projection biases, so every CPU LoRA train/eval ran
a bias-less model. New doctrine rule: *a falsifier whose band fails against its reference
may have caught a broken reference — bisect before loosening.* The escape became two
permanent falsifiers (`FALSIFY-CPU-LORA-QKV-BIAS-001/002`, CI-enforced, no GPU needed).

---

**Early-version history (detail).** The per-version sections below are the
original v0.1–v0.8 development notes, retained for historical context. They
**predate the monorepo consolidation** and do not reflect the current
architecture — see the canonical roadmap linked above.

## v0.2.0: Tree Models & Cross-Validation

**Target**: Decision Trees, Random Forests, Model Selection, Model Persistence

### Completed

- [x] **Tree Module**: Complete decision tree implementation
  - DecisionTreeClassifier with GINI impurity
  - Configurable max_depth
  - Recursive tree building
  - Integration tests with Iris dataset
- [x] **Random Forest**: Bootstrap aggregating ensemble
  - RandomForestClassifier with configurable n_estimators
  - Bootstrap sampling with replacement
  - Majority voting for predictions
  - Reproducible with random_state
- [x] **Cross-Validation**: Model evaluation utilities
  - train_test_split() with reproducible random seeds
  - KFold cross-validator with optional shuffling
  - cross_validate() with statistics (mean, std, min, max)
  - CrossValidationResult struct
- [x] **Model Serialization**: Save/load models to disk
  - Serde + bincode integration
  - Works with all models (LinearRegression, KMeans, DecisionTree, RandomForest)
  - Example demonstrating persistence
- [x] **Examples**: New comprehensive examples
  - decision_tree_iris.rs - Decision tree classification
  - random_forest_iris.rs - Ensemble classification
  - cross_validation.rs - Model evaluation workflow
  - model_persistence.rs - Save/load demonstration
- [x] **Documentation**: EXTREME TDD Book
  - 90+ chapter structure on GitHub Pages
  - Complete case study: Cross-Validation
  - RED-GREEN-REFACTOR methodology
  - Live at https://paiml.github.io/aprender/
- [x] **Bug Fixes**
  - Clear error messages for underdetermined systems

### Quality Metrics Achieved

- TDG Score: 93.3/100 (A grade)
- Total Tests: 184 passing (+64 from v0.1.0)
- Property Tests: 22
- Doc Tests: 16
- Test Coverage: ~97%
- Max Cyclomatic Complexity: ≤10
- Zero clippy warnings
- Zero SATD comments

### Released ✅

- [x] Published to crates.io (2024-11-18)
- [x] GitHub Release created with release notes
- [x] EXTREME TDD Book deployed
- [x] CI/CD pipeline passing

---

## v0.1.0: Foundation

**Target**: Linear Regression + K-Means (2 algorithms, viable from day one)

### Completed

- [x] Project scaffolding (Cargo workspace, quality gates)
- [x] Core primitives: Vector, Matrix (with Cholesky solver)
- [x] DataFrame (named column container with `to_matrix()`)
- [x] Linear Regression (OLS via normal equations)
- [x] K-Means clustering (k-means++ initialization + Lloyd's algorithm)
- [x] Metrics: R², MSE, RMSE, MAE, inertia, silhouette_score
- [x] Estimator/UnsupervisedEstimator/Transformer traits
- [x] Property-based tests (19 proptest properties)
- [x] Unit tests (120 tests)
- [x] Doc tests (13 tests)
- [x] Examples: boston_housing, iris_clustering, dataframe_basics
- [x] Benchmarks: linear_regression, kmeans

### Quality Metrics Achieved

- TDG Score: 95.6/100 (A+ grade)
- Repository Score: 95.0/100 (A+)
- Test Coverage: 97.72%
- Mutation Score: 85.3%
- Max Cyclomatic Complexity: 5
- Zero clippy warnings
- Zero SATD comments
- Total Tests: 149 (127 unit + 22 property)

### Released ✅

- [x] Published to crates.io (2024-11-18)
- [x] GitHub Release created with artifacts
- [x] Complete rustdoc coverage
- [x] CI/CD pipeline operational

---

## v0.3.0: Regularization & Optimization

**Target**: Regularized Linear Models, Optimizers, Advanced Model Selection

### Completed

- [x] **Regularized Linear Models**
  - Ridge regression (L2 regularization)
  - Lasso regression (L1 via coordinate descent)
  - Elastic Net (L1 + L2 combination)
  - Full builder pattern API with max_iter, tolerance
- [x] **Optimizers**
  - SGD with optional momentum
  - Adam with adaptive learning rates
  - Trait-based design for extensibility
- [x] **Loss Functions**
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - Huber loss (smooth combination)
  - Both functional and OOP APIs
- [x] **Advanced Model Selection**
  - Stratified K-Fold cross-validation
  - Grid search for hyperparameter tuning
  - Works with all regularized models
- [x] **Preprocessing**
  - StandardScaler (z-score normalization)
  - MinMaxScaler (range scaling)
  - Transformer trait integration
- [x] **Examples**: New comprehensive demonstrations
  - regularized_regression.rs - Ridge, Lasso, ElasticNet with grid search
  - optimizer_demo.rs - SGD and Adam optimization
- [x] **Chaos Engineering**
  - ChaosConfig from renacer integration
  - Property-based testing with proptest
  - Fuzz testing infrastructure with cargo-fuzz
  - 41 new tests (6 property + 14 integration + 1 fuzz target)
- [x] **Refactoring**
  - Reduced tree module complexity: 10 → 7 cyclomatic, 22 → 13 cognitive
  - Reduced grid search complexity: 9 → 4 cyclomatic, 23 → 6 cognitive
  - Extracted 6 helper functions for better maintainability

### Quality Metrics Achieved

- TDG Score: 95.6/100 (A+ grade)
- Total Tests: 498 passing (+314 from v0.2.0)
- Property Tests: 32 (+10)
- Doc Tests: 49 (+33)
- Integration Tests: 6
- Unit Tests: 387 (+203)
- Max Cyclomatic Complexity: 7 (down from 14)
- Max Cognitive Complexity: 13 (down from 23)
- Zero clippy warnings
- Zero SATD comments
- All quality gates passing

### Released ✅

- [x] All 10 features implemented and tested
- [x] Chaos engineering infrastructure integrated
- [x] Code complexity significantly reduced
- [x] 9 comprehensive examples running
- [x] All quality gates passing

---

## v0.4.0: TOP 10 ML Algorithms Complete

**Target**: Industry's most popular machine learning algorithms with comprehensive testing

### Completed

- [x] **Logistic Regression** (Issue #12)
  - Binary and multi-class classification
  - Gradient descent optimization with learning rate decay
  - L2 regularization support
  - predict_proba() for probability estimates
- [x] **K-Nearest Neighbors** (Issue #23)
  - Distance-based classification
  - Configurable k parameter
  - Brute-force and optimized distance computation
  - Works with any distance metric
- [x] **Support Vector Machine** (Issue #24)
  - Linear SVM with hinge loss
  - Subgradient descent optimizer
  - C regularization parameter
  - decision_function() for margin-based predictions
- [x] **Naive Bayes** (Issue #25)
  - GaussianNB with probabilistic classification
  - Variance smoothing parameter
  - predict_proba() returns class probabilities
  - Fast training O(n·d)
- [x] **Gradient Boosting Machine** (Issue #26)
  - Adaptive boosting with residual learning
  - Configurable n_estimators and learning_rate
  - Tree-based weak learners
  - Feature importance scores
- [x] **Decision Trees & Random Forests** (v0.2.0)
  - GINI impurity-based splitting
  - Bootstrap aggregating for Random Forest
  - Configurable max_depth and n_estimators
- [x] **Linear Regression** (v0.1.0)
  - OLS via normal equations
  - Ridge/Lasso/ElasticNet (v0.3.0)
- [x] **K-Means Clustering** (v0.1.0)
  - k-means++ initialization
  - Lloyd's algorithm
  - Configurable max_iter
- [x] **Principal Component Analysis** (Issue #13)
  - Eigendecomposition-based dimensionality reduction
  - Configurable n_components
  - explained_variance_ratio for feature analysis
  - transform() for new data projection
- [x] **Classification Metrics**
  - accuracy_score, precision_score, recall_score
  - f1_score, confusion_matrix
  - Multi-class support (macro/micro averaging)

### Quality Metrics Achieved

- Total Tests: 528 passing
- Zero clippy warnings
- Zero SATD violations
- All quality gates passing
- Comprehensive documentation with examples
- 10/10 TOP algorithms implemented ✅

### Released ✅

- [x] Published to crates.io as v0.4.0 (2024-11-19)
- [x] All TOP 10 algorithms tested and documented
- [x] Examples for each algorithm
- [x] Book chapters for theory + case studies

---

## v0.4.1: Graph Algorithms, Advanced Clustering & Statistics

**Target**: Expand beyond TOP 10 with graph theory, advanced clustering, anomaly detection, and statistical analysis

### Completed

- [x] **Graph Algorithms** (Issue #9)
  - Betweenness Centrality (shortest path counting)
  - PageRank (iterative power method)
  - Graph data structure with adjacency list
  - Weighted and unweighted edge support
- [x] **Community Detection** (Issue #22)
  - Louvain algorithm for modularity optimization
  - Modularity computation Q = (1/2m) Σ[A_ij - k_i*k_j/2m] δ(c_i, c_j)
  - Detects densely connected groups in networks
  - O(m·log n) complexity
- [x] **Advanced Clustering**
  - DBSCAN (Issue #14) - Density-based clustering
  - Hierarchical Clustering (Issue #15) - Agglomerative with linkage methods
  - Gaussian Mixture Models (Issue #16) - EM algorithm for soft clustering
  - Spectral Clustering (Issue #19) - Graph Laplacian eigendecomposition
- [x] **Anomaly Detection**
  - Isolation Forest (Issue #17) - Ensemble of isolation trees
  - Local Outlier Factor (Issue #20) - Density-based outlier detection
  - score_samples() and predict() methods
  - Contamination parameter for threshold setting
- [x] **Dimensionality Reduction**
  - t-SNE (Issue #18) - Non-linear visualization
  - Perplexity-based similarity computation
  - KL divergence minimization via gradient descent
  - 2D/3D embedding support
- [x] **Association Rule Mining**
  - Apriori Algorithm (Issue #21) - Frequent itemset mining
  - Support, confidence, and lift metrics
  - Market basket analysis support
  - Efficient pruning with apriori property
- [x] **Descriptive Statistics** (Issue #9)
  - Mean, median, mode, variance, std deviation
  - Quartiles (Q1, Q2, Q3), IQR
  - Histograms with multiple binning strategies
  - Five-number summary (min, Q1, median, Q3, max)

### Quality Metrics Achieved

- Total Tests: 683 passing (+155 from v0.4.0)
- Zero clippy warnings
- Zero critical SATD violations (1 low-priority Bayesian Blocks TODO)
- All quality gates passing
- Comprehensive EXTREME TDD book with case studies
- mdbook tests: 0 failures across 119 chapters

### Released ✅

- [x] Published to crates.io as v0.4.1 (2024-11-21)
- [x] All 6 advanced clustering algorithms implemented
- [x] Graph algorithms with social network examples
- [x] Complete anomaly detection suite
- [x] Association rule mining for market basket analysis
- [x] Comprehensive book chapters for all algorithms

---

## v0.5.0: Regression Trees & Advanced Ensemble Methods

- [ ] Decision tree regression (CART algorithm)
- [ ] Random Forest regression
- [ ] Out-of-bag error estimation for Random Forests
- [ ] Feature importance visualization
- [ ] XGBoost-style optimizations (histogram binning, approximate split finding)

---

## v0.6.0: Neural Networks

- [ ] Autodiff integration (feature-gated)
- [ ] Dense layers (fully connected)
- [ ] Activation functions (ReLU, Leaky ReLU, ELU, SELU)
- [ ] Optimizers: SGD, Adam, AdaGrad, RMSprop
- [ ] Batch normalization
- [ ] Dropout regularization
- [ ] Sequential model API

---

## v0.7.0: Advanced Statistics & Inference

- [ ] Generalized Linear Models (GLM)
- [ ] Statistical tests: t-test, chi-square, ANOVA, F-test
- [ ] Covariance/correlation matrices
- [ ] Independent Component Analysis (ICA)
- [ ] Factor Analysis
- [ ] Hypothesis testing framework

---

## v0.8.0: Showcase & QA Protocol (Completed)

**Target**: Unified Inference Architecture (GGUF/SafeTensors/APR) & Severe Testing Protocol

### Completed

- [x] **Unified Inference Architecture**
  - Multi-format support: GGUF, SafeTensors, APR
  - Hybrid backend: CPU + GPU (CUDA)
  - Rosetta ML Diagnostics for format conversion
  - `apr run`, `apr chat`, `apr serve` commands
- [x] **Severe Testing Protocol (PMAT-QA-PROTOCOL-001)**
  - Hang Detection: 60s timeout wrapper for all tests
  - Garbage Detection: Strict Level 5 verification (regex + patterns)
  - Zombie Mitigation: SIGINT resiliency for `apr serve`
  - RAII Model Fixtures: Automated setup/teardown
- [x] **QA Matrix**
  - 21-cell test matrix (Modality × Format × Trace)
  - Full traceability with `--trace` flag
  - Performance regression baselines
- [x] **Documentation**
  - Updated showcase spec (v1.7.0) with Dr. Popper's audit
  - Falsification suite (qa_falsify.rs)
  - Comprehensive QA Report

### Quality Metrics Achieved

- QA Pass Rate: 100% (21/21 matrix cells)
- Falsification Coverage: 5 attack vectors verified
- Zombie Processes: 0 (verified by SIGINT tests)
- Documentation: Epistemologically audited (Corroborated)

### Released ✅

- [x] QA Protocol fully operational
- [x] Showcase demo verified
- [x] Falsification suite integrated

---

## v0.8.1: Time Series (Planned)

- [ ] ARIMA models
- [ ] Exponential smoothing (Holt-Winters)
- [ ] Seasonal decomposition
- [ ] Forecasting metrics: MAPE, SMAPE

---

## v1.0.0: Production Hardening

- [ ] GPU benchmarks and optimization
- [ ] WASM examples (in-browser training)
- [ ] Model serialization versioning
- [ ] Complete EXTREME TDD Book content
- [ ] Performance whitepaper
- [ ] Production deployment examples

---

## Quality Targets

All releases must meet:

- **TDG Score**: A+ (95.0+/100)
- **Test Coverage**: 95%+ line coverage
- **Mutation Score**: 85%+ (cargo-mutants)
- **Complexity**: ≤10 cyclomatic per function
- **Documentation**: 100% rustdoc coverage

---

## Contributing

Contributions welcome! Please ensure:
- All tests pass: `cargo test --all`
- No clippy warnings: `cargo clippy --all-targets`
- Code is formatted: `cargo fmt`

Priorities:
1. Bug fixes and test coverage improvements
2. Documentation and examples
3. Performance optimizations
4. New algorithms (must include tests and benchmarks)
