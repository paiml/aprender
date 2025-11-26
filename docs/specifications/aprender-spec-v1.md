# Aprender v1.0 Specification

**Project**: Aprender (Spanish: "to learn")  
**Version**: 1.0.0  
**Status**: Specification Phase  
**Repository**: `github.com/paiml/aprender`  
**TDG Target**: A+ (95.0+/100)  
**Mutation Score Target**: 85%+  
**Test Coverage Target**: 95%+  

---

## Executive Summary

Aprender is a next-generation machine learning library in pure Rust that synthesizes best practices from Julia (multiple dispatch), JAX (automatic differentiation), scikit-learn (ergonomic API), and R (comprehensive statistics) while building on Trueno's unified SIMD/GPU/WASM compute primitives. It enforces Toyota Way quality standards via PMAT, achieves asymptotic test effectiveness via Certeza methodology, and provides production tracing via Renacer integration.

**Core Differentiation**: Unlike legacy ML frameworks that bolt on GPU support, Aprender's algorithms are backend-agnostic from inception, leveraging Trueno's runtime dispatch to transparently scale from WASM in-browser to multi-GPU clusters with identical user code.

---

## 1. Foundation: Trueno Integration

### 1.1 Compute Primitive Layer

Aprender builds exclusively on Trueno's abstractions:

```rust
// Trueno provides unified backend dispatch
use trueno::{Vector, Matrix, Backend};

// Aprender algorithms are backend-agnostic
pub trait Model<T: Float> {
    fn fit(&mut self, x: &Matrix<T>, y: &Vector<T>) -> Result<()>;
    fn predict(&self, x: &Matrix<T>) -> Vector<T>;
}

// Backend selection is automatic
let model = LinearRegression::new(); // Uses Backend::Auto
model.fit(&x_train, &y_train)?; // Dispatches to AVX2/GPU/WASM
```

**Key Trueno Capabilities Leveraged**:
- Vector operations: `add`, `mul`, `dot`, `sum`, `max` (SIMD-optimized)
- Matrix operations: `matmul`, `transpose`, `convolve2d` (GPU for >500×500)
- Activation functions: `relu`, `sigmoid`, `tanh`, `gelu`, `softmax`
- Reductions: `mean`, `variance`, `stddev`, `covariance`
- Backend dispatch: Automatic selection (AVX2 > NEON > GPU > Scalar)

### 1.2 Performance Characteristics

From Trueno benchmarking:
- **SIMD Operations**: 182-348% faster than scalar (dot product, reductions)
- **GPU Threshold**: >500×500 matrices (2-10x speedup)
- **Memory-Bound Ops**: 3-15% SIMD gains (element-wise add/mul)
- **GPU Overhead**: 14-55ms (PCIe transfer) - only beneficial for O(n³) ops

**Aprender Design Principle**: Algorithms must amortize GPU transfer cost. Only operations with O(n²) or higher complexity use GPU by default.

---

## 2. Quality Infrastructure

### 2.1 PMAT Integration (Technical Debt Grading)

All Aprender code must pass PMAT quality gates:

```bash
# Pre-commit hook enforcement
pmat quality-gate --strict --fail-on-violation

# Continuous quality monitoring
pmat tdg dashboard --port 8081 --open

# TDG scoring (6 orthogonal metrics)
pmat tdg . --include-components
```

**Quality Requirements**:
- **Complexity**: ≤10 cyclomatic per function (Toyota Way standard)
- **SATD**: Zero tolerance (no TODO/FIXME/HACK comments)
- **Documentation**: 90%+ rustdoc coverage
- **Dead Code**: Zero unused code (enforced by `pmat analyze dead-code`)
- **Duplication**: <5% Type-3 clones (semantic similarity detection)
- **TDG Score**: Maintain A+ grade (95.0+/100)

**Tooling Philosophy: Assistants, Not Gatekeepers**

PMAT/Certeza/Renacer are learning tools, not obstacles:

- **Automate ceremony**: PMAT auto-generates rustdoc templates, fixes simple clippy lints
- **Focus on insights**: Renacer profile showing bottleneck = optimization puzzle, not failure
- **Psychological safety**: Strict gates apply to `main` branch only. Feature branches allow experimentation.
- **Human judgment**: Surviving mutants require categorization, not blind score-chasing

**Rationale**: High cognitive load (Rust + SIMD + ML + property testing) requires tool support, not additional burden. Sustainable excellence requires respecting developer flow. (Forsgren et al., 2018: psychological safety predicts high performance)

### 2.2 Certeza Testing Methodology (4-Tier Verification)

Aprender adopts Certeza's asymptotic test effectiveness with Toyota Way flow optimization:

```makefile
# Tier 1: On-save (instant, non-blocking)
tier1:
    @cargo check
    @cargo fmt --check
    @cargo clippy -- -W clippy::all

# Tier 2: Pre-commit (<5 sec, changed files only)
tier2:
    @cargo test --lib  # Fast unit tests only
    @cargo clippy -- -D warnings

# Tier 3: Pre-push (1-5 min, full validation)
tier3:
    @cargo test --all
    @cargo llvm-cov --all-features --workspace
    @./scripts/check_coverage.sh  # Generates report, does not fail

# Tier 4: CI/CD (asynchronous, heavyweight)
tier4:
    @cargo mutants --no-times
    @pmat tdg . --include-components
    @./scripts/mutation_review.sh  # Flags for human review
```

**Test Distribution** (from Certeza pyramid):
- 60% Unit tests: Basic functionality, edge cases
- 30% Property tests: Algorithmic correctness (proptest with 10K cases)
- 10% Integration tests: End-to-end workflows

**Jidoka Principle: Metrics as Signals**

Quality targets trigger human review, not automatic failure:

**Coverage <95%?**
- CI generates uncovered-lines report
- Developer documents *why* in PR: "panic path theoretically impossible" or "Windows-only branch, tested manually"
- Reviewer validates reasoning

**Mutation score <85%?**
- CI generates surviving-mutants report
- Developer categorizes each:
  - **Equivalent mutant**: Mark + justify
  - **Valuable mutant**: Write test to kill
  - **Trivial mutant**: Accept risk with comment
- Reviewer validates categorization

**Rationale**: Humans judge quality. Metrics detect anomalies. Strict gates encourage metric-chasing over bug-finding. (Inozemtseva & Holmes, 2014: "Coverage not strongly correlated with test suite effectiveness")

**Critical Paths** (require formal verification):
- Gradient computation (Kani proofs for autodiff tape correctness)
- Matrix inversion (condition number checks, singularity detection)
- Unsafe memory operations (if any - minimize to <1% codebase)

### 2.3 Renacer Profiling Integration

Performance-critical paths must be profiled:

```bash
# Profile training loop
renacer --function-time --source -- ./examples/linear_regression > profile.txt

# Generate flamegraph
cat profile.txt | flamegraph.pl > aprender_train.svg

# Detect I/O bottlenecks
renacer -c --stats-extended -- cargo bench matrix_multiply

# Real-time anomaly detection during benchmarks
renacer --anomaly-realtime --hpu-analysis -- ./benches/kmeans
```

**Profiling Requirements**:
- All `pub` API functions must be profiled under realistic workloads
- GPU operations must show >2x speedup vs SIMD to justify dispatch
- No function should consume >10% of total training time (hotspot threshold)

### 2.4 Bashrs Script Quality

All shell scripts (benchmarks, CI/CD, installation) must be generated via bashrs:

```rust
// benches/run_benchmarks.rs (transpiled to POSIX shell)
#[rash::main]
fn main() {
    let cargo_path = env_var_or("CARGO", "cargo");
    
    echo("Running Aprender benchmark suite...");
    
    // Ensure Trueno is built with GPU support
    if !exec("cargo build --release --features gpu") {
        eprint("Failed to build with GPU support");
        exit(1);
    }
    
    // Run criterion benchmarks
    exec("{cargo_path} bench --all-features");
    
    // Lint Makefile and shell scripts
    exec("bashrs check scripts/*.rs");
}
```

**Shell Quality Gates**:
- All scripts pass `shellcheck --severity=warning`
- POSIX compliance (tested on dash, bash 3.2+, ash)
- Injection-safe: All variables properly quoted
- Deterministic: Same input always produces identical output

---

## 3. Architecture

### 3.1 Core Data Structures

**Minimal DataFrame** (~300 LOC, zero dependencies):
```rust
// aprender::data::DataFrame - Named column container
pub struct DataFrame {
    columns: Vec<(String, Vector<f32>)>,
    n_rows: usize,
}

impl DataFrame {
    pub fn new(columns: Vec<(String, Vector<f32>)>) -> Result<Self, &'static str>;
    pub fn select(&self, names: &[&str]) -> Result<Self, &'static str>;
    pub fn column(&self, name: &str) -> Result<&Vector<f32>, &'static str>;
    pub fn to_matrix(&self) -> Matrix<f32>; // Horizontal stack, column-major
    pub fn row(&self, idx: usize) -> Result<Vector<f32>, &'static str>;
    pub fn shape(&self) -> (usize, usize); // (rows, cols)
}
```

**Scope boundaries** (intentionally NOT implemented):
- Joins, groupby, pivots → delegate to ruchy/polars
- Mixed types (strings, dates) → f32/f64 numeric only
- Missing value semantics → caller handles NaN
- I/O (CSV, Parquet) → aprender-io integration crate
- In-place mutation → immutable by design

**Rationale**: DataFrame is column-named `Vec<Vector<f32>>` with `to_matrix()` convenience. Prevents `(0..n_cols).map(|i| matrix.column(i))` boilerplate. Full data wrangling stays in ruchy/polars.

### 3.2 Type System (Julia-Inspired Multiple Dispatch)

Rust traits emulate Julia's multiple dispatch:

```rust
// Generic algorithm trait
pub trait Fit<X, Y, Backend> {
    fn fit(&mut self, x: &X, y: &Y, backend: Backend) -> Result<()>;
}

// Specialized implementations
impl Fit<Matrix<f32>, Vector<f32>, CpuBackend> for LinearRegression {
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>, backend: CpuBackend) -> Result<()> {
        // SIMD-optimized normal equations
        self.coefficients = solve_normal_equations_simd(x, y)?;
        Ok(())
    }
}

impl Fit<Matrix<f32>, Vector<f32>, GpuBackend> for LinearRegression {
    fn fit(&mut self, x: &Matrix<f32>, y: &Y, backend: GpuBackend) -> Result<()> {
        // GPU-accelerated via Cholesky decomposition
        self.coefficients = solve_cholesky_gpu(x, y)?;
        Ok(())
    }
}
```

**Type Hierarchy**:
```
Float: f32, f64
Backend: Auto, Cpu(Simd), Gpu, Wasm
Data: Vector<T>, Matrix<T>, DataFrame, Tensor<T> (future)
Model: Supervised, Unsupervised, Reinforcement (future)
```

### 3.3 Automatic Differentiation (JAX-Inspired, Phased Implementation)

**Phase 1 (v0.5.0): Leverage Existing Library**

Initial neural network release uses battle-tested autodiff:

```toml
[dependencies]
dfdx = { version = "0.13", optional = true }  # Proven tape-based autodiff
# OR
tch-rs = { version = "0.16", optional = true }  # PyTorch bindings
```

**Rationale**: Production autodiff handles higher-order derivatives, control flow (loops/conditionals), efficient tape pruning, and cycle detection. Building from scratch is multi-year research (Bradbury et al., 2018; Paszke et al., 2019). Focus MVP on ergonomic API.

**Phase 2 (v0.7.0+): Custom Engine (Research Track)**

Parallel development of Trueno-native autodiff:

```rust
// Target architecture (illustrative)
pub struct Variable<T: Float> {
    data: Vector<T>,
    grad: Option<Vector<T>>,
    tape: Arc<RwLock<Tape>>,
}

pub struct Tape {
    operations: Vec<BackpropOp>,
    checkpoint_nodes: Vec<NodeId>,  // For memory efficiency
}

pub enum BackpropOp {
    Add { lhs_id: VarId, rhs_id: VarId, out_id: VarId },
    MatMul { lhs_id: VarId, rhs_id: VarId, out_id: VarId },
    Sigmoid { in_id: VarId, out_id: VarId },
    ControlFlow { branches: Vec<Tape> },  // Nontrivial: requires dominance analysis
}
```

**Integration criterion**: Custom engine surpasses `dfdx` on:
- Memory efficiency (tape pruning via liveness analysis)
- GPU kernel fusion (leverages Trueno's wgpu backend)
- Higher-order derivatives (Hessian-vector products)

**Differentiation Rules** (when implemented):
- Element-wise: `d/dx (f(x) + g(x)) = f'(x) + g'(x)`
- Matrix multiply: `d/dA (AB) = B^T`, `d/dB (AB) = A^T`
- Activations: `d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))`

### 3.3 API Layers (sklearn-Inspired Ergonomics)

Three-tier API design:

```rust
// High-level: sklearn-like estimator API
pub trait Estimator<X, Y> {
    fn fit(&mut self, x: &X, y: &Y) -> Result<&mut Self>;
    fn predict(&self, x: &X) -> Y;
    fn score(&self, x: &X, y: &Y) -> f64; // R² for regression, accuracy for classification
}

// Mid-level: Optimizer/loss abstractions
pub trait Optimizer {
    fn step(&mut self, params: &mut [Variable]) -> Result<()>;
}

pub trait Loss<Y> {
    fn compute(&self, y_pred: &Y, y_true: &Y) -> f32;
    fn gradient(&self, y_pred: &Y, y_true: &Y) -> Y;
}

// Low-level: Direct Trueno primitives (expert users)
use trueno::{Vector, Matrix};
```

---

## 4. Phase 1: Minimal Viable Product (v0.1.0)

**Target**: Linear Regression + K-Means (2 algorithms, viable from day one)

### 4.1 Linear Regression

**Algorithm**: Ordinary Least Squares via Normal Equations
- Closed-form solution: `β = (X^T X)^-1 X^T y`
- Complexity: O(n²p + p³) where n=samples, p=features
- GPU threshold: p > 500 (matrix inversion benefits from GPU)

```rust
pub struct LinearRegression {
    coefficients: Vector<f32>,
    intercept: f32,
    backend: Backend,
}

impl Estimator<Matrix<f32>, Vector<f32>> for LinearRegression {
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<&mut Self> {
        // Add intercept column (1s)
        let x_design = add_intercept_column(x);
        
        // Compute X^T X (leverages Trueno's matmul)
        let xtx = x_design.transpose().matmul(&x_design)?;
        
        // Compute X^T y
        let xty = x_design.transpose().matvec(y)?;
        
        // Solve normal equations (Cholesky decomposition)
        let beta = cholesky_solve(&xtx, &xty)?;
        
        self.intercept = beta[0];
        self.coefficients = beta.slice(1..);
        Ok(self)
    }
    
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
        x.matvec(&self.coefficients).add_scalar(self.intercept)
    }
    
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f64 {
        let y_pred = self.predict(x);
        r_squared(&y_pred, y)
    }
}
```

**Extensions** (future phases):
- Ridge regression (L2 regularization): `β = (X^T X + λI)^-1 X^T y`
- Lasso regression (L1 regularization): Coordinate descent solver
- Elastic Net (L1 + L2): Proximal gradient method
- Gradient descent solver (for large-scale, online learning)

### 4.2 K-Means Clustering

**Algorithm**: Lloyd's algorithm with k-means++ initialization
- Complexity: O(nkdi) where n=samples, k=clusters, d=features, i=iterations
- GPU threshold: n > 10,000 && d > 100 (distance matrix computation)

```rust
pub struct KMeans {
    n_clusters: usize,
    max_iter: usize,
    centroids: Matrix<f32>,
    labels: Vector<usize>,
    inertia: f32, // Sum of squared distances
}

impl KMeans {
    pub fn fit(&mut self, x: &Matrix<f32>) -> Result<&mut Self> {
        // 1. Initialize centroids (k-means++ for faster convergence)
        self.centroids = kmeans_plusplus_init(x, self.n_clusters)?;
        
        for iter in 0..self.max_iter {
            // 2. Assign samples to nearest centroid (uses Trueno's parallel ops)
            let distances = compute_distance_matrix(x, &self.centroids)?;
            self.labels = argmin_along_axis(&distances, Axis::Columns);
            
            // 3. Update centroids (mean of assigned samples)
            let new_centroids = compute_cluster_means(x, &self.labels, self.n_clusters)?;
            
            // 4. Check convergence (centroids unchanged)
            if centroids_converged(&self.centroids, &new_centroids, 1e-4) {
                break;
            }
            
            self.centroids = new_centroids;
        }
        
        self.inertia = compute_inertia(x, &self.centroids, &self.labels);
        Ok(self)
    }
    
    pub fn predict(&self, x: &Matrix<f32>) -> Vector<usize> {
        let distances = compute_distance_matrix(x, &self.centroids)?;
        argmin_along_axis(&distances, Axis::Columns)
    }
}
```

**Extensions** (future phases):
- Mini-batch K-Means (online learning)
- K-Means++ initialization (already in v0.1)
- Silhouette score (cluster quality metric)
- Elbow method (optimal k selection)

### 4.3 v0.1.0 Deliverables

**Crate Structure**:
```
aprender/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── dataframe.rs (~300 LOC)
│   │   └── tests.rs (100+ tests)
│   ├── linear_model/
│   │   ├── mod.rs
│   │   ├── linear_regression.rs
│   │   └── tests.rs (250+ tests)
│   ├── cluster/
│   │   ├── mod.rs
│   │   ├── kmeans.rs
│   │   └── tests.rs (200+ tests)
│   ├── metrics/
│   │   ├── mod.rs
│   │   ├── regression.rs (r_squared, mse, mae)
│   │   └── clustering.rs (inertia, silhouette)
│   ├── traits.rs (Estimator, Transformer, etc.)
│   └── prelude.rs
├── benches/
│   ├── dataframe.rs
│   ├── linear_regression.rs
│   └── kmeans.rs
├── examples/
│   ├── dataframe_basics.rs
│   ├── boston_housing.rs (Linear Regression)
│   └── iris_clustering.rs (K-Means)
└── docs/
    ├── spec/
    │   └── aprender-spec-v1.md (this file)
    └── tutorials/
        ├── getting_started.md
        └── ruchy_integration.md
```

**Test Coverage**:
- Unit tests: 400+ tests (DataFrame: 100, LinReg: 250, K-Means: 200, edge cases)
- Property tests: 60+ properties (10K cases each via proptest)
  - DataFrame: `select().column(name) == original.column(name)`
  - DataFrame: `to_matrix().n_cols() == n_selected_columns`
  - Linear regression: `predict(fit(X, y), X) ≈ y` (within tolerance)
  - K-Means: `inertia decreases monotonically across iterations`
  - Invariants: `coefficients.len() == n_features`, `labels.max() < n_clusters`
- Integration tests: 12+ end-to-end workflows
- Benchmarks: 6+ criterion benchmarks (DataFrame ops, regression detection)

**Documentation**:
- 100% rustdoc coverage for public API
- Tutorial: "Linear Regression in 5 minutes"
- Tutorial: "DataFrame basics and Matrix conversion"
- Example: Boston Housing dataset (regression)
- Example: Iris dataset (clustering)
- Comparison table: Aprender vs scikit-learn (API parity checklist)

---

## 5. Ruchy Integration

### 5.1 DataFrame Conversion Bridge

**Zero-copy polars → aprender**:
```rust
// In ruchy-aprender bridge crate
impl From<&polars::DataFrame> for aprender::DataFrame {
    fn from(df: &polars::DataFrame) -> Self {
        let columns: Vec<(String, Vector<f32>)> = df
            .get_columns()
            .iter()
            .filter_map(|series| {
                if let Ok(ca) = series.f32() {
                    let vec_data: Vec<f32> = ca.to_vec();
                    let vec = Vector::from_vec(vec_data); // Trueno Vector
                    Some((series.name().to_string(), vec))
                } else {
                    None // Skip non-numeric columns
                }
            })
            .collect();
        
        DataFrame::new(columns).expect("Invalid column lengths")
    }
}

// Reverse conversion for output
impl From<&aprender::DataFrame> for polars::DataFrame {
    fn from(df: &aprender::DataFrame) -> Self {
        let series: Vec<polars::Series> = df
            .columns()
            .map(|(name, vec)| {
                Series::new(name, vec.as_slice())
            })
            .collect();
        polars::DataFrame::new(series).unwrap()
    }
}
```

### 5.2 Seamless Syntax Transpilation

Ruchy provides Python-like ergonomics that transpile to Aprender:

```python
# Ruchy syntax (ruchy/examples/ml_workflow.ruchy)
from aprender import LinearRegression, train_test_split

# Load data (uses ruchy's polars backend)
df = read_csv("boston_housing.csv")

# Select numeric columns, auto-converts to aprender::DataFrame
X = df.select(["sqft", "bedrooms", "bathrooms"])
y = df["price"]

# Train model (X is aprender::DataFrame internally)
model = LinearRegression()
model.fit(X, y)

# Evaluate
predictions = model.predict(X)
r2 = model.score(X, y)
println(f"R² score: {r2:.3f}")
```

**Transpiles to**:
```rust
// Generated Rust code
use aprender::prelude::*;
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Polars loads data
    let df = CsvReader::from_path("boston_housing.csv")?.finish()?;
    
    // Convert to aprender::DataFrame (bridge layer)
    let x_polars = df.select(&["sqft", "bedrooms", "bathrooms"])?;
    let x: aprender::DataFrame = (&x_polars).into(); // Zero-copy
    
    let y_series = df.column("price")?;
    let y: Vector<f32> = y_series.f32()?.to_vec().into();
    
    // Aprender core (pure, no polars dependency)
    let mut model = LinearRegression::new();
    model.fit(&x.to_matrix(), &y)?;
    
    let predictions = model.predict(&x.to_matrix());
    let r2 = model.score(&x.to_matrix(), &y);
    
    println!("R² score: {:.3}", r2);
    Ok(())
}
```

### 5.3 Core Library Independence

**Key architectural principle**: `aprender` core never imports polars.

```
┌─────────────────────────────────────────┐
│ Ruchy (has polars built-in)            │
│   - Data loading, cleaning, transforms  │
│   - polars::DataFrame                   │
└──────────────┬──────────────────────────┘
               │ Conversion bridge
               ↓ (ruchy-aprender crate)
┌─────────────────────────────────────────┐
│ aprender::DataFrame                     │
│   - Named columns: Vec<(String, Vec)>  │
│   - .to_matrix() for ML algorithms      │
└──────────────┬──────────────────────────┘
               │ Core ML
               ↓
┌─────────────────────────────────────────┐
│ aprender::LinearRegression              │
│   - Operates on Matrix<f32>             │
│   - Zero polars knowledge               │
└──────────────┬──────────────────────────┘
               │ Compute
               ↓
┌─────────────────────────────────────────┐
│ trueno::{Vector, Matrix}                │
│   - SIMD/GPU/WASM dispatch              │
└─────────────────────────────────────────┘
```

**Rationale**: DataFrame is thin column naming layer. Heavy lifting (joins, aggregations) stays in ruchy/polars. Conversion happens at API boundary, not in algorithm implementations.

---

## 6. Roadmap (Incremental Releases)

### v0.1.0: Foundation (Sprint 1-3, ~6 weeks)
- [x] Project scaffolding (Cargo workspace, PMAT integration)
- [ ] Linear Regression (OLS via normal equations)
- [ ] K-Means clustering (Lloyd's algorithm + k-means++)
- [ ] Metrics: R², MSE, MAE, inertia
- [ ] 450+ tests (Certeza Tier 2 compliance)
- [ ] 2 examples (Boston Housing, Iris)
- [ ] Documentation: API reference + getting started guide

### v0.2.0: Regularization & Optimization (Sprint 4-6, ~6 weeks)
- [ ] Ridge regression (L2 regularization)
- [ ] Lasso regression (L1 via coordinate descent)
- [ ] SGD optimizer (mini-batch gradient descent)
- [ ] Adam optimizer (adaptive learning rates)
- [ ] Loss functions: MSE, MAE, Huber
- [ ] Cross-validation: k-fold, stratified
- [ ] Grid search (hyperparameter tuning)

### v0.3.0: Classification (Sprint 7-9, ~6 weeks)
- [ ] Logistic regression (binary classification)
- [ ] Softmax regression (multi-class)
- [ ] Support Vector Machines (SVM via libsvm bindings or pure Rust SMO)
- [ ] Naive Bayes (Gaussian, Multinomial, Bernoulli)
- [ ] Metrics: accuracy, precision, recall, F1, ROC-AUC
- [ ] Confusion matrix utilities

### v0.4.0: Tree Ensembles (Sprint 10-12, ~8 weeks)
- [ ] Decision tree (CART algorithm)
- [ ] Random Forest (bagging + feature sampling)
- [ ] Gradient Boosting (XGBoost-style)
- [ ] Feature importance scores
- [ ] Out-of-bag error estimation

### v0.5.0: Neural Networks (Sprint 13-16, ~10 weeks)
- [ ] Autodiff integration (dfdx or tch-rs, feature-gated)
- [ ] Dense layers (fully connected)
- [ ] Activation functions (ReLU, Leaky ReLU, ELU, SELU)
- [ ] Optimizers: SGD, Adam, AdaGrad, RMSprop
- [ ] Batch normalization
- [ ] Dropout regularization
- [ ] Sequential model API
- [ ] Python FFI wrapper (aprender-py, separate crate)
- [ ] Custom autodiff engine (parallel research track, optional)

### v0.6.0: Advanced Statistics (Sprint 17-19, ~6 weeks)
- [ ] Generalized Linear Models (GLM): Poisson, Gamma, Binomial
- [ ] Statistical tests: t-test, chi-square, ANOVA
- [ ] Covariance/correlation matrices
- [ ] Principal Component Analysis (PCA)
- [ ] Independent Component Analysis (ICA)

### v0.7.0: Time Series (Sprint 20-22, ~6 weeks)
- [ ] ARIMA models (auto-regressive integrated moving average)
- [ ] Exponential smoothing (Holt-Winters)
- [ ] Seasonal decomposition
- [ ] Forecasting metrics: MAPE, SMAPE, MAE

### v1.0.0: Production Hardening (Sprint 23-26, ~8 weeks)
- [ ] GPU benchmarks (validation of Trueno dispatch heuristics)
- [ ] WASM examples (in-browser training demos)
- [ ] Distributed training (multi-node via MPI or gRPC)
- [ ] Model serialization (serde with versioning)
- [ ] MLOps toolkit (experiment tracking, model registry)
- [ ] Comprehensive documentation (book, API reference, tutorials)
- [ ] Performance whitepaper (Aprender vs PyTorch/TensorFlow)

---

## 7. Integration Strategy & Boundaries

### 7.1 Python FFI: Pragmatic Adoption Path

**Tier 1 (Adoption)**: Official thin Python wrapper
```python
# aprender-py (PyO3 bindings, separate crate)
import aprender as ap
import numpy as np

model = ap.LinearRegression()
model.fit(X_train, y_train)  # Accepts np.ndarray, zero-copy via PyO3
predictions = model.predict(X_test)
```

**Tier 2 (Idiomatic)**: Ruchy recommended for performance
```python
# Ruchy transpiles to pure Rust, no FFI overhead
from aprender import LinearRegression
model = LinearRegression()
model.fit(X, y)  # Compiles to native code
```

**Tier 3 (Core)**: Pure Rust internal API
```rust
// aprender core, zero Python knowledge
let mut model = LinearRegression::new();
model.fit(&x, &y)?;
```

**Rationale**: ML ecosystem is Python-native. Forcing Ruchy adoption creates adoption barrier. PyO3 wrapper enables immediate use while guiding toward optimal path. (Pedregosa et al., 2011: scikit-learn success via NumPy integration)

### 7.2 Explicit Anti-Patterns

### 7.2.1 No Direct CUDA/cuDNN Calls
**Rationale**: Trueno provides portable GPU via wgpu/WebGPU. Direct CUDA bindings lock into NVIDIA ecosystem and complicate cross-platform support (Metal, Vulkan, WebGPU).

### 7.3 No Legacy Algorithm Variants
**Rationale**: No "NumPy compatibility mode" or "sklearn-exact behavior" flags. Aprender implements algorithms correctly per primary literature, not bug-for-bug compatibility.

### 7.4 No String-Based Configuration
**Rationale**: All configuration via typed builders, not stringly-typed JSON/YAML. Catch errors at compile time, not runtime.

```rust
// ✅ Good: Type-safe builder
let model = LinearRegression::builder()
    .regularization(Regularization::Ridge(0.01))
    .solver(Solver::CholeskyGpu)
    .build();

// ❌ Bad: Stringly-typed config (not supported)
let model = LinearRegression::from_config(r#"
    {
        "regularization": "ridge",
        "alpha": 0.01,
        "solver": "cholesky_gpu"
    }
"#)?;
```

---

## 8. Benchmarking Strategy

### 8.1 Criterion-Based Performance Tracking

All algorithms have criterion benchmarks:

```rust
// benches/linear_regression.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use aprender::prelude::*;

fn bench_linear_regression_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_regression_fit");
    
    for size in [100, 500, 1000, 5000, 10000].iter() {
        let x = Matrix::random(*size, 10); // n x 10 features
        let y = Vector::random(*size);
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut model = LinearRegression::new();
                model.fit(black_box(&x), black_box(&y))
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_linear_regression_fit);
criterion_main!(benches);
```

### 8.2 GPU vs SIMD vs Scalar Comparison

Explicit backend benchmarks to validate dispatch heuristics:

```bash
# Compare backends
cargo bench --bench linear_regression -- --features gpu
cargo bench --bench linear_regression -- --features cpu-only

# Profile with Renacer
renacer --function-time -- cargo bench matrix_multiply

# Generate flamegraph
renacer --source -- cargo bench kmeans > profile.txt
flamegraph.pl profile.txt > kmeans_flamegraph.svg
```

### 8.3 Regression Detection

Criterion auto-detects performance regressions:

```toml
# .criterion.toml
[regression]
# Fail CI if >5% slower than baseline
max_percentage = 5.0
```

---

## 9. Example: End-to-End Workflow

### 9.1 Ruchy Script (High-Level API)

```python
# examples/complete_workflow.ruchy
from aprender import (
    LinearRegression, Ridge, Lasso,
    train_test_split, StandardScaler,
    cross_val_score, GridSearchCV
)
from ruchy.dataframe import read_csv

# Load and preprocess data
df = read_csv("housing.csv")
X = df.select(["sqft", "bedrooms", "bathrooms", "age"])
y = df["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
}
grid_search = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)
grid_search.fit(X_train_scaled, y_train)

# Evaluate best model
best_model = grid_search.best_estimator()
r2 = best_model.score(X_test_scaled, y_test)
predictions = best_model.predict(X_test_scaled)

println(f"Best α: {grid_search.best_params()['alpha']}")
println(f"Test R²: {r2:.3f}")
```

### 9.2 Transpiled Rust (Low-Level)

```rust
// Auto-generated from Ruchy (simplified)
use aprender::prelude::*;
use polars::prelude::*;

fn main() -> Result<()> {
    // Load data
    let df = CsvReader::from_path("housing.csv")?.finish()?;
    let x: Matrix<f32> = df.select(&["sqft", "bedrooms", "bathrooms", "age"])?.into();
    let y: Vector<f32> = df["price"].into();
    
    // Split
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42))?;
    
    // Scale features
    let mut scaler = StandardScaler::new();
    let x_train_scaled = scaler.fit_transform(&x_train)?;
    let x_test_scaled = scaler.transform(&x_test)?;
    
    // Grid search
    let param_grid = ParamGrid::new()
        .add("alpha", vec![0.001, 0.01, 0.1, 1.0, 10.0]);
    
    let mut grid_search = GridSearchCV::new(
        Ridge::builder().build(),
        param_grid,
        CrossValidation::KFold(5),
        Metric::NegMSE
    );
    
    grid_search.fit(&x_train_scaled, &y_train)?;
    
    // Evaluate
    let best_model = grid_search.best_estimator();
    let r2 = best_model.score(&x_test_scaled, &y_test);
    let predictions = best_model.predict(&x_test_scaled);
    
    println!("Best α: {}", grid_search.best_params()["alpha"]);
    println!("Test R²: {:.3}", r2);
    
    Ok(())
}
```

---

## 10. Dependency Policy

### 10.1 Core Runtime Dependencies (PAIML Projects Only)

```toml
[dependencies]
trueno = { git = "https://github.com/paiml/trueno", version = "0.3" }
# Zero other runtime dependencies
```

**Rationale**: Core library is pure Rust with single compute backend dependency. All algorithms operate on `trueno::{Vector, Matrix}` primitives. No serialization, no I/O, no parallelism abstractions—Trueno handles SIMD/GPU dispatch internally.

### 10.2 Development Toolchain (PAIML Projects)

```toml
[dev-dependencies]
# Property-based testing (from certeza methodology)
proptest = "1.6"  # External, but required by certeza framework

[build-dependencies]
# None - pure library, no build scripts
```

**Quality Gates** (CI/CD tools, not compiled into library):
- `pmat` (github.com/paiml/paiml-mcp-agent-toolkit): TDG analysis, complexity metrics
- `renacer` (github.com/paiml/renacer): Profiling, flamegraph generation
- `bashrs` (github.com/paiml/bashrs): Shell script linting/generation

### 10.3 Integration Layer (Separate Crate)

```toml
# aprender-io crate (optional, separate from core)
[dependencies]
aprender = { path = "../aprender" }
trueno = { git = "https://github.com/paiml/trueno" }
# No polars - use ruchy's DataFrame directly
```

**Data flow**:
```
User Data → Ruchy (has polars) → ruchy::DataFrame.to_ndarray() 
         → Vec<f32> → trueno::Matrix → aprender::Model
```

Zero-copy achieved via raw pointer aliasing in unsafe boundary (tested via certeza).

### 10.4 Banned Dependencies

**External crates prohibited in core**:
- `serde`, `bincode`: Serialization handled by caller (see aprender-serde integration crate)
- `rayon`, `tokio`: Parallelism is Trueno's responsibility via backend dispatch
- `thiserror`, `anyhow`: Use `Result<T, &'static str>` or custom error enum
- `ndarray`, `nalgebra`: Competing abstractions—use Trueno exclusively
- `polars`, `arrow`: Data loading orthogonal to ML algorithms
- **Any crate with C/C++ FFI** (except via Trueno's wgpu/SIMD intrinsics)

**Justification**: Each external dependency increases attack surface, complicates WASM compilation, and fragments the ecosystem. PAIML projects share quality standards (PMAT A+, Certeza testing) and maintenance guarantees.

---

## 11. Documentation Requirements

### 11.1 API Documentation (rustdoc)

Every public item must have:

```rust
/// Ordinary Least Squares (OLS) linear regression.
///
/// Fits a linear model by minimizing the residual sum of squares between
/// observed targets and predicted targets. The model equation is:
///
/// ```text
/// y = X β + ε
/// ```
///
/// where `β` is the coefficient vector and `ε` is random error.
///
/// # Solver
///
/// Uses normal equations: `β = (X^T X)^-1 X^T y` via Cholesky decomposition.
/// For large datasets (n > 10,000), consider using gradient descent instead.
///
/// # Examples
///
/// ```rust
/// use aprender::prelude::*;
///
/// let x = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// let y = Vector::from_slice(&[7.0, 8.0, 9.0]);
///
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y)?;
///
/// let predictions = model.predict(&x);
/// assert_eq!(predictions.len(), 3);
/// ```
///
/// # Performance
///
/// - Time complexity: O(n²p + p³) where n = samples, p = features
/// - Space complexity: O(np)
/// - GPU threshold: p > 500 (Cholesky decomposition benefits from GPU)
///
/// # References
///
/// - Hastie, Tibshirani, Friedman (2009). *The Elements of Statistical Learning*. Section 3.2.
pub struct LinearRegression {
    // ...
}
```

### 11.2 Book Structure

Mdbook documentation (`docs/book/`):

1. **Introduction**
   - Why Aprender? (vs PyTorch, TensorFlow, scikit-learn)
   - Installation (Cargo, Homebrew, Docker)
   - First example (5-minute tutorial)

2. **User Guide**
   - Data loading (polars, CSV, JSON)
   - Feature engineering (scaling, encoding, PCA)
   - Model selection (cross-validation, grid search)
   - Evaluation metrics (R², accuracy, F1)

3. **Algorithm Reference**
   - Linear models (OLS, Ridge, Lasso, Logistic)
   - Clustering (K-Means, DBSCAN, Hierarchical)
   - Neural networks (MLP, CNN, RNN)

4. **Advanced Topics**
   - Custom models (implementing `Estimator` trait)
   - GPU optimization (when to use GPU, profiling)
   - Distributed training (multi-node setup)

5. **Integration Guides**
   - Ruchy integration (Python-like syntax)
   - WASM deployment (in-browser inference)
   - MLOps (experiment tracking, model versioning)

---

## 12. Acceptance Criteria (v0.1.0 Release)

### 12.1 Functional Requirements
- [x] Linear Regression implements `Estimator` trait
- [ ] K-Means implements `fit` and `predict` methods
- [ ] R², MSE, MAE metrics implemented
- [ ] Train/test split utility
- [ ] 2 working examples (Boston Housing, Iris)

### 12.2 Quality Requirements
- [ ] PMAT TDG score: A+ (95.0+/100)
- [ ] Test coverage: 95%+ line coverage
- [ ] Mutation score: 85%+ (cargo-mutants)
- [ ] Zero SATD comments (no TODO/FIXME/HACK)
- [ ] Zero clippy warnings (strict mode)
- [ ] All public API documented (rustdoc)

### 12.3 Performance Requirements
- [ ] Linear regression: <1ms for 100×10 matrix (SIMD)
- [ ] K-Means: <100ms for 1000 samples × 10 features × 10 iterations
- [ ] GPU dispatch: 2x speedup for 1000×1000 matrix multiply
- [ ] WASM: <2x overhead vs native (compiled with `wasm-opt -O3`)

### 12.4 Integration Requirements
- [ ] Ruchy transpilation: `aprender` module available in Ruchy
- [ ] Polars integration: Zero-copy DataFrame → Matrix conversion
- [ ] CI/CD: GitHub Actions with quality gates
- [ ] Crates.io: Published as `aprender = "0.1.0"`

---

## 13. Acknowledgments

This specification incorporates feedback from Gemini Code Review Board's Toyota Way-inspired analysis (November 17, 2025). Key Kaizen improvements:

1. **4-tier feedback loop**: Eliminated developer flow interruption (Tier 2: <5sec, changed files only)
2. **Jidoka metrics**: Quality targets trigger human review vs hard gates (respects developer judgment)
3. **Phased autodiff**: De-risked v0.5.0 via proven libraries, custom engine as parallel research
4. **Pragmatic Python FFI**: 3-tier adoption strategy (PyO3 wrapper → Ruchy → pure Rust)
5. **Tooling as assistants**: PMAT/Certeza/Renacer positioned as learning aids, not gatekeepers

These refinements maintain technical excellence while honoring *Genchi Genbutsu* (face reality), *Jidoka* (automation with human touch), and *respect for people* (sustainable cognitive load).

---

## 14. References

### 14.1 Academic Literature
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
- Bradbury, J., et al. (2018). *JAX: Composable transformations of Python+NumPy programs.* arXiv:1807.02017.
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS.
- Inozemtseva, L., & Holmes, R. (2014). *Coverage is not strongly correlated with test suite effectiveness.* ICSE.
- Just, R., et al. (2014). *Are mutants a valid substitute for real faults in software testing?* FSE.

### 14.2 Algorithm Implementation References (v0.9.1)
- **K-Means**: Lloyd, S. (1982). "Least squares quantization in PCM". *IEEE Transactions on Information Theory*, 28(2), 129-137.
- **DBSCAN**: Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise". *KDD-96*.
- **Isolation Forest**: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest". *ICDM '08*.
- **LOF**: Breunig, M. M., et al. (2000). "LOF: identifying density-based local outliers". *SIGMOD 2000*.
- **GMM (EM)**: Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm". *Journal of the Royal Statistical Society*.
- **Random Forest**: Breiman, L. (2001). "Random Forests". *Machine Learning*, 45(1), 5-32.
- **CART**: Breiman, L., et al. (1984). *Classification and Regression Trees*. Wadsworth & Brooks/Cole.
- **Linear SVM**: Cortes, C., & Vapnik, V. (1995). "Support-vector networks". *Machine Learning*, 20(3), 273-297.
- **Logistic Regression**: Cox, D. R. (1958). "The regression analysis of binary sequences". *Journal of the Royal Statistical Society*.
- **Agglomerative Clustering**: Sokal, R. R., & Michener, C. D. (1958). "A statistical method for evaluating systematic relationships". *University of Kansas Science Bulletin*.

### 14.3 Software Engineering & Quality
- Dijkstra, E. W. (1972). "The Humble Programmer". ACM Turing Award Lecture.
- Beck, K. (2002). *Test-Driven Development: By Example*. Addison-Wesley.
- Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press.
- Forsgren, N., Humble, J., & Kim, G. (2018). *Accelerate: The Science of Lean Software and DevOps*. IT Revolution.
- Meyer, A. N., et al. (2014). *Software developers' perceptions of productivity.* FSE.
- Parnin, C., & DeLine, R. (2010). *Evaluating cues for resuming interrupted programming tasks.* CHI.
- Gousios, G., et al. (2015). *Work practices and challenges in pull-based development.* ICSE.

### 14.3 Ecosystem References
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR, 12, 2825-2830.
- Bezanson, J., et al. (2017). *Julia: A fresh approach to numerical computing.* SIAM Review, 59(1), 65-98.

### 14.4 Rust Ecosystem
- Trueno: `github.com/paiml/trueno` (Compute primitives)
- PMAT: `github.com/paiml/paiml-mcp-agent-toolkit` (Quality gates)
- Certeza: `github.com/paiml/certeza` (Testing methodology)
- Renacer: `github.com/paiml/renacer` (Profiling/tracing)
- Bashrs: `github.com/paiml/bashrs` (Shell script generation)
- Ruchy: `github.com/paiml/ruchy` (High-level syntax)

### 14.5 Comparative Analysis
- JAX: Automatic differentiation via function transformations
- Julia: Multiple dispatch via type system
- Scikit-learn: Estimator/Transformer API design
- PyTorch: Dynamic computation graphs
- R: Comprehensive statistical modeling

---

## 15. Appendix: Quality Gate Commands

```bash
# Pre-commit (Tier 1: <1 sec)
make tier1
    cargo fmt --check
    cargo clippy -- -D warnings
    cargo test --lib

# Pre-push (Tier 2: 1-5 min)
make tier2
    cargo test --all
    cargo llvm-cov --all-features --workspace
    pmat analyze complexity --fail-on-violation
    pmat analyze satd --fail-on-violation

# CI/CD (Tier 3: 5-60 min)
make tier3
    cargo mutants --no-times
    pmat tdg . --fail-on-grade B
    cargo audit
    cargo deny check

# Profiling (on-demand)
make profile
    renacer --function-time --source -- cargo bench
    renacer -c --stats-extended -- ./target/release/examples/linear_regression

# Documentation
make docs
    cargo doc --all-features --no-deps
    mdbook build docs/book
    pmat validate-docs

# Release checklist
make release-check
    make tier3
    make docs
    cargo package --allow-dirty
    git tag v0.1.0
    cargo publish
```

---

**END OF SPECIFICATION**

*This document is a living specification. All changes must be tracked via Git and reviewed via pull request. Version bumps follow semantic versioning (SemVer 2.0.0).*

*Contact: noah@paiml.com | Pragmatic AI Labs | github.com/paiml/aprender*
