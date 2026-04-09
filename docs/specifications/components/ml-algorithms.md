# ML Algorithms

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version**: 1.0.0
**Status**: Active
**Parent**: [aprender-spec.md](../aprender-spec.md) §4

---

## 1. Overview

Aprender implements the TOP 10 ML algorithms plus advanced modules covering
time series, NLP, Bayesian inference, GLMs, decomposition, graph algorithms,
and neural network building blocks. All algorithms are backend-agnostic via
the Trueno compute layer.

---

## 2. Architecture: Three-Tier API

### 2.1 High Level — Estimator Traits

```rust
pub trait Estimator<T: Float> {
    fn fit(&mut self, x: &Matrix<T>, y: &Vector<T>) -> Result<()>;
    fn predict(&self, x: &Matrix<T>) -> Vector<T>;
}

pub trait UnsupervisedEstimator<T: Float> {
    fn fit(&mut self, x: &Matrix<T>) -> Result<()>;
    fn predict(&self, x: &Matrix<T>) -> Vector<T>;
}

pub trait Transformer<T: Float> {
    fn fit(&mut self, x: &Matrix<T>) -> Result<()>;
    fn transform(&self, x: &Matrix<T>) -> Matrix<T>;
}
```

Julia-inspired multiple dispatch: algorithm selection is trait-based,
not inheritance-based. Any type implementing `Estimator` works with
model selection, cross-validation, and pipeline combinators.

### 2.2 Mid Level — Optimizers, Loss, Regularization

| Component | Implementations |
|-----------|----------------|
| Optimizer | SGD, Adam, AdaGrad, RMSProp, L-BFGS |
| Loss | MSE, CrossEntropy, Hinge, Huber, Focal |
| Regularizer | L1 (Lasso), L2 (Ridge), ElasticNet |
| Scheduler | StepLR, CosineAnnealing, ReduceOnPlateau |

### 2.3 Low Level — Trueno Primitives

```rust
use trueno::{Vector, Matrix, Backend};

// Backend-agnostic: same code on CPU SIMD, GPU, WASM
let result = Matrix::matmul(&weights, &input);
let activated = Vector::relu(&result);
```

---

## 3. TOP 10 Algorithms (v0.4.0)

### 3.1 Supervised — Regression

| Algorithm | Module | Key Features |
|-----------|--------|-------------|
| Linear Regression | `linear_model` | OLS, Ridge (L2), Lasso (L1), ElasticNet |

### 3.2 Supervised — Classification

| Algorithm | Module | Key Features |
|-----------|--------|-------------|
| Logistic Regression | `linear_model` | Binary + multiclass (softmax), regularized |
| Decision Tree | `tree::classifier` | CART, Gini/entropy, pruning |
| Random Forest | `tree` | Bagging, feature subsampling, OOB error |
| Gradient Boosted Trees | `tree::gradient_boosting` | Histogram-based, shrinkage, subsampling |
| Naive Bayes | `naive_bayes` | Gaussian, Multinomial, Bernoulli |
| KNN | `neighbors` | k-d tree, ball tree, brute force |
| SVM | `svm` | Linear, RBF, polynomial kernels; SMO solver |

### 3.3 Unsupervised

| Algorithm | Module | Key Features |
|-----------|--------|-------------|
| K-Means | `cluster` | k-means++, mini-batch, elbow method |
| PCA | `decomposition` | SVD-based, explained variance ratio |

---

## 4. Advanced Modules (v0.7.x+)

### 4.1 Time Series

| Component | Description |
|-----------|-------------|
| ARIMA | AutoRegressive Integrated Moving Average |
| Exponential Smoothing | Simple, double, triple (Holt-Winters) |
| Stationarity Tests | ADF, KPSS |
| Differencing | First and seasonal differencing |

### 4.2 NLP / Text Processing

| Component | Description |
|-----------|-------------|
| BPE Tokenizer | Byte-pair encoding with vocabulary training |
| Chat Templates | Jinja2-based (ChatML, LLaMA, Mistral, Gemma, etc.) |
| Stop Words | Multi-language stop word lists |
| Stemming | Porter, Snowball stemmers |
| Text Vectorization | TF-IDF, count vectorizer |

Chat templates use minijinja for sandboxed Jinja2 rendering. Supports
6+ format families. Template auto-detection from model metadata.

### 4.3 Bayesian Inference

| Component | Description |
|-----------|-------------|
| Conjugate Priors | Beta-Binomial, Normal-Normal, Gamma-Poisson |
| Bayesian Linear Regression | Full posterior with credible intervals |
| Prior/Posterior | Analytical updates for exponential family |

### 4.4 Generalized Linear Models (GLM)

| Family | Link Function | Use Case |
|--------|--------------|----------|
| Poisson | Log | Count data |
| Gamma | Inverse | Positive continuous |
| Binomial | Logit | Binary outcomes |

IRLS (Iteratively Reweighted Least Squares) solver.

### 4.5 Decomposition

| Algorithm | Description |
|-----------|-------------|
| PCA | Principal Component Analysis (SVD-based) |
| ICA | Independent Component Analysis (FastICA) |

### 4.6 Graph Algorithms

| Category | Algorithms |
|----------|-----------|
| Shortest Path | Dijkstra, A*, Bellman-Ford |
| Centrality | PageRank, betweenness, closeness, degree |
| Community | Louvain, label propagation |
| Traversal | BFS, DFS, topological sort |
| MST | Kruskal, Prim |

### 4.7 Neural Network Building Blocks

| Layer | Module | Description |
|-------|--------|-------------|
| Linear | `nn::linear` | Dense layer with optional bias |
| RMSNorm | `nn::normalization` | Root mean square normalization |
| GroupNorm | `nn::normalization` | Group normalization |
| LayerNorm | `nn::normalization` | Layer normalization |
| Attention (GQA) | `nn::transformer` | Grouped Query Attention |
| RoPE | `nn::transformer` | Rotary Position Embeddings |
| SwiGLU | `nn::functional` | Gated linear unit with SiLU |
| Softmax | `nn::functional` | Numerically stable softmax |

These are training-side building blocks. Inference uses realizar's
fused kernels (see [compute-backends.md](compute-backends.md)).

---

## 5. Model Selection and Metrics

### 5.1 Cross-Validation

- k-fold, stratified k-fold, leave-one-out
- Train/test split with stratification
- Time series split (expanding window)

### 5.2 Classification Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Correct predictions / total |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | Harmonic mean of precision and recall |
| ROC AUC | Area under ROC curve |
| Confusion Matrix | Full TP/FP/TN/FN breakdown |
| Log Loss | Cross-entropy loss |

### 5.3 Regression Metrics

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R-squared | Coefficient of determination |
| Adjusted R-squared | R-squared with penalty for features |

### 5.4 Ranking Metrics

| Metric | Description |
|--------|-------------|
| NDCG | Normalized Discounted Cumulative Gain |
| MRR | Mean Reciprocal Rank |
| MAP | Mean Average Precision |

---

## 6. Preprocessing

| Component | Description |
|-----------|-------------|
| StandardScaler | Zero mean, unit variance |
| MinMaxScaler | Scale to [0, 1] range |
| LabelEncoder | Categorical → integer encoding |
| OneHotEncoder | Categorical → binary vectors |
| PolynomialFeatures | Generate polynomial and interaction features |
| Imputer | Missing value imputation (mean, median, mode) |

---

## 7. Calibration

| Method | Description |
|--------|-------------|
| Platt Scaling | Sigmoid fit on logits |
| Isotonic Regression | Non-parametric calibration |
| Temperature Scaling | Single-parameter softmax scaling |

---

## 8. Loss Functions

| Loss | Module | Use Case |
|------|--------|----------|
| MSE | `loss` | Regression |
| Cross-Entropy | `loss` | Classification |
| Binary Cross-Entropy | `loss` | Binary classification |
| Hinge | `loss` | SVM |
| Huber | `loss` | Robust regression |
| Focal | `loss` | Imbalanced classification |
| KL Divergence | `loss` | Distribution matching / distillation |
| Contrastive | `loss` | Similarity learning |
