# Case Study: Batuta - Automated Migration to Aprender

Using Batuta to automatically convert Python ML projects to Aprender/Rust.

## Overview

**Batuta** (Spanish for "conductor's baton") is an orchestration framework that converts Python ML projects to high-performance Rust using Aprender. It automates the migration of scikit-learn codebases to Aprender equivalents with:

- Automatic API mapping (sklearn → Aprender)
- NumPy → Trueno tensor conversion
- Mixture-of-Experts (MoE) backend routing
- Semantic-preserving transformation

```
┌─────────────────────────────────────────────────────────────────┐
│                     BATUTA MIGRATION FLOW                       │
│                                                                 │
│   Python Project                    Rust Project                │
│   ──────────────                    ────────────                │
│   sklearn.linear_model    ═══►     aprender::linear_model      │
│   sklearn.cluster         ═══►     aprender::cluster           │
│   sklearn.ensemble        ═══►     aprender::ensemble          │
│   sklearn.preprocessing   ═══►     aprender::preprocessing     │
│   numpy operations        ═══►     trueno primitives           │
│                                                                 │
│   Result: 2-10× performance improvement with memory safety      │
└─────────────────────────────────────────────────────────────────┘
```

## The 5-Phase Workflow

Batuta follows a Toyota Way-inspired Kanban workflow:

```
┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────┐
│ Analysis │──►│ Transpilation│──►│ Optimization │──►│ Validation │──►│ Deployment │
└──────────┘   └──────────────┘   └──────────────┘   └────────────┘   └────────────┘
     │                │                   │                │               │
     ▼                ▼                   ▼                ▼               ▼
   PMAT          Depyler           MoE Backend        Renacer          Reports
  TDG Score     Type Inference      Routing          Tracing         Migration
```

### Phase 1: Analysis

```bash
$ batuta analyze ./my-sklearn-project

Primary language: Python
Total files: 127
Total lines: 8,432

Dependencies:
  • pip (42 packages) in requirements.txt
  • ML frameworks detected:
    - scikit-learn 1.3.0 → Aprender mapping available
    - numpy 1.24.0 → Trueno mapping available
    - pandas 2.0.0 → DataFrame support

Quality:
  • TDG Score: 73.2/100 (B)
  • Test coverage: 68%

Recommended transpiler: Depyler (Python → Rust)
Estimated migration complexity: Medium
```

### Phase 2: Transpilation

```bash
$ batuta transpile --output ./rust-project
```

### Phase 3: Optimization

```bash
$ batuta optimize --enable-simd --enable-gpu
```

### Phase 4: Validation

```bash
$ batuta validate --trace-syscalls --benchmark
```

### Phase 5: Deployment

```bash
$ batuta build --release
$ batuta report --format markdown --output MIGRATION.md
```

## scikit-learn to Aprender Mapping

Batuta provides complete mappings for sklearn algorithms:

### Linear Models

| scikit-learn | Aprender | Complexity |
|--------------|----------|------------|
| `LinearRegression` | `aprender::linear_model::LinearRegression` | Medium |
| `LogisticRegression` | `aprender::linear_model::LogisticRegression` | Medium |
| `Ridge` | `aprender::linear_model::Ridge` | Medium |
| `Lasso` | `aprender::linear_model::Lasso` | Medium |

### Tree-Based Models

| scikit-learn | Aprender | Complexity |
|--------------|----------|------------|
| `DecisionTreeClassifier` | `aprender::tree::DecisionTreeClassifier` | High |
| `RandomForestClassifier` | `aprender::ensemble::RandomForestClassifier` | High |
| `GradientBoostingClassifier` | `aprender::ensemble::GradientBoosting` | High |

### Clustering

| scikit-learn | Aprender | Complexity |
|--------------|----------|------------|
| `KMeans` | `aprender::cluster::KMeans` | Medium |
| `DBSCAN` | `aprender::cluster::DBSCAN` | High |

### Preprocessing

| scikit-learn | Aprender | Complexity |
|--------------|----------|------------|
| `StandardScaler` | `aprender::preprocessing::StandardScaler` | Low |
| `MinMaxScaler` | `aprender::preprocessing::MinMaxScaler` | Low |
| `LabelEncoder` | `aprender::preprocessing::LabelEncoder` | Low |

### Model Selection

| scikit-learn | Aprender | Notes |
|--------------|----------|-------|
| `train_test_split` | `aprender::model_selection::train_test_split` | Same API |
| `cross_val_score` | `aprender::model_selection::cross_validate` | Same API |
| `GridSearchCV` | `aprender::model_selection::GridSearchCV` | Parallel by default |

## Conversion Examples

### Example 1: Basic ML Pipeline

**Python (Original):**

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X, y = data.data, data.target

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
```

**Rust (Batuta Output):**

```rust
use aprender::datasets::load_iris;
use aprender::preprocessing::StandardScaler;
use aprender::model_selection::train_test_split;
use aprender::ensemble::RandomForestClassifier;
use aprender::metrics::accuracy_score;
use aprender::{Estimator, Transformer};

fn main() -> anyhow::Result<()> {
    // Load data
    let data = load_iris()?;
    let (X, y) = (&data.features, &data.targets);

    // Preprocess
    let mut scaler = StandardScaler::new();
    let X_scaled = scaler.fit_transform(X)?;

    // Split (80/20, seed=42)
    let (X_train, X_test, y_train, y_test) = train_test_split(
        &X_scaled, y, 0.2, Some(42)
    )?;

    // Train
    let mut model = RandomForestClassifier::new()
        .with_n_estimators(100)
        .with_seed(42);
    model.fit(&X_train, &y_train)?;

    // Evaluate
    let predictions = model.predict(&X_test)?;
    let accuracy = accuracy_score(&y_test, &predictions)?;
    println!("Accuracy: {:.4}", accuracy);

    Ok(())
}
```

### Example 2: Linear Regression with Cross-Validation

**Python (Original):**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([1.5, 3.5, 5.5, 7.5, 9.5])

model = LinearRegression()
scores = cross_val_score(model, X, y, cv=3, scoring='r2')
print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f}")
```

**Rust (Batuta Output):**

```rust
use aprender::linear_model::LinearRegression;
use aprender::model_selection::cross_validate;
use aprender::Estimator;
use trueno::Matrix;

fn main() -> anyhow::Result<()> {
    let X = Matrix::from_slice(&[
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
    ]);
    let y = vec![1.5, 3.5, 5.5, 7.5, 9.5];

    let model = LinearRegression::new();
    let scores = cross_validate(&model, &X, &y, 3)?;

    println!("R² scores: {:?}", scores.test_scores);
    println!("Mean R²: {:.4}", scores.mean_test_score());

    Ok(())
}
```

### Example 3: Clustering with KMeans

**Python (Original):**

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

X = np.random.randn(1000, 5)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

score = silhouette_score(X, labels)
print(f"Silhouette score: {score:.4f}")
print(f"Inertia: {kmeans.inertia_:.2f}")
```

**Rust (Batuta Output):**

```rust
use aprender::cluster::KMeans;
use aprender::metrics::silhouette_score;
use aprender::UnsupervisedEstimator;
use trueno::Matrix;

fn main() -> anyhow::Result<()> {
    // Generate random data (using trueno's random)
    let X = Matrix::random(1000, 5);

    let mut kmeans = KMeans::new(3)
        .with_seed(42)
        .with_n_init(10);
    let labels = kmeans.fit_predict(&X)?;

    let score = silhouette_score(&X, &labels)?;
    println!("Silhouette score: {:.4}", score);
    println!("Inertia: {:.2}", kmeans.inertia());

    Ok(())
}
```

## NumPy to Trueno Mapping

Batuta converts NumPy operations to Trueno equivalents:

| NumPy | Trueno | Notes |
|-------|--------|-------|
| `np.array([...])` | `Vector::from_slice(&[...])` | Direct mapping |
| `np.zeros((m, n))` | `Matrix::zeros(m, n)` | Same semantics |
| `np.ones((m, n))` | `Matrix::ones(m, n)` | Same semantics |
| `np.dot(a, b)` | `a.dot(&b)` | SIMD-accelerated |
| `a @ b` | `a.matmul(&b)` | MoE backend selection |
| `np.sum(a)` | `a.sum()` | Reduction operation |
| `np.mean(a)` | `a.mean()` | Statistical operation |
| `np.max(a)` | `a.max()` | Reduction operation |
| `np.min(a)` | `a.min()` | Reduction operation |
| `a.T` | `a.transpose()` | View-based (zero-copy) |
| `a.reshape(m, n)` | `a.reshape(m, n)` | Same API |

### Example: Matrix Operations

**Python:**

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiply
C = A @ B

# Element-wise operations
D = A + B
E = A * B

# Reductions
total = np.sum(A)
mean = np.mean(A)
```

**Rust (via Batuta):**

```rust
use trueno::{Matrix, Vector};

fn main() {
    let A = Matrix::from_slice(&[
        [1.0, 2.0],
        [3.0, 4.0],
    ]);
    let B = Matrix::from_slice(&[
        [5.0, 6.0],
        [7.0, 8.0],
    ]);

    // Matrix multiply (MoE selects SIMD for small matrices)
    let C = A.matmul(&B);

    // Element-wise operations (SIMD-accelerated)
    let D = &A + &B;
    let E = &A * &B;

    // Reductions
    let total = A.sum();
    let mean = A.mean();
}
```

## Mixture-of-Experts Backend Routing

Batuta automatically selects optimal backends based on operation complexity and data size:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MoE BACKEND SELECTION                        │
│                                                                 │
│   Operation Type          Data Size      Backend Selected       │
│   ──────────────          ─────────      ────────────────       │
│   Element-wise (Low)      < 1M           Scalar/SIMD            │
│   Element-wise (Low)      ≥ 1M           SIMD                   │
│                                                                 │
│   Reductions (Medium)     < 10K          Scalar                 │
│   Reductions (Medium)     10K - 100K     SIMD                   │
│   Reductions (Medium)     ≥ 100K         GPU                    │
│                                                                 │
│   MatMul (High)           < 1K           Scalar                 │
│   MatMul (High)           1K - 10K       SIMD                   │
│   MatMul (High)           ≥ 10K          GPU                    │
└─────────────────────────────────────────────────────────────────┘
```

Based on the **5× PCIe dispatch rule** (Gregg & Hazelwood 2011): GPU dispatch is only beneficial when compute time exceeds 5× the PCIe transfer time.

### Using the Backend Selector

```rust
use batuta::backend::{BackendSelector, OpComplexity};

fn main() {
    let selector = BackendSelector::new();

    // Element-wise on 1M elements → SIMD
    let backend = selector.select_with_moe(OpComplexity::Low, 1_000_000);
    println!("1M element-wise: {}", backend);  // "SIMD"

    // Matrix multiply on 50K elements → GPU
    let backend = selector.select_with_moe(OpComplexity::High, 50_000);
    println!("50K matmul: {}", backend);  // "GPU"

    // Reduction on 5K elements → Scalar
    let backend = selector.select_with_moe(OpComplexity::Medium, 5_000);
    println!("5K reduction: {}", backend);  // "Scalar"
}
```

## Performance Comparison

Real-world benchmarks from migrated projects:

```
┌─────────────────────────────────────────────────────────────────┐
│              BATUTA MIGRATION PERFORMANCE GAINS                 │
│                                                                 │
│   Operation               Python      Rust        Improvement   │
│   ────────────────────    ──────      ────        ───────────   │
│   Linear regression fit   45ms        4ms         11.2× faster  │
│   Random forest predict   890ms       89ms        10.0× faster  │
│   KMeans clustering       2.3s        0.21s       10.9× faster  │
│   StandardScaler          12ms        0.8ms       15.0× faster  │
│   Matrix multiply (1K)    5.2ms       0.3ms       17.3× faster  │
│                                                                 │
│   Memory Usage:                                                 │
│   Peak RSS               127MB        24MB        5.3× smaller  │
│   Heap allocations       45K          3K          15.0× fewer   │
│                                                                 │
│   Binary Size:           N/A          2.1MB       Static linked │
│   Startup Time:          ~500ms       23ms        21.7× faster  │
└─────────────────────────────────────────────────────────────────┘
```

## Oracle Mode

Batuta includes an intelligent query interface for component selection:

```bash
# Find the right approach
$ batuta oracle "How do I train random forest on 1M samples?"

Recommendation: Use aprender::ensemble::RandomForestClassifier
  • Data size: 1M samples → High complexity
  • Recommended backend: GPU (via Trueno)
  • Memory estimate: ~800MB for training
  • Parallel trees: Enable with --n-jobs=-1

Code template:
```rust
use aprender::ensemble::RandomForestClassifier;

let mut model = RandomForestClassifier::new()
    .with_n_estimators(100)
    .with_max_depth(Some(10))
    .with_seed(42);
model.fit(&X_train, &y_train)?;
```

# List all stack components
$ batuta oracle --list

# Show component details
$ batuta oracle --show aprender
```

## Plugin Architecture

Extend Batuta with custom transpilers:

```rust
use batuta::plugin::{TranspilerPlugin, PluginMetadata, PluginRegistry};
use batuta::types::Language;

struct MyCustomConverter;

impl TranspilerPlugin for MyCustomConverter {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "custom-ml-converter".to_string(),
            version: "0.1.0".to_string(),
            description: "Custom ML framework converter".to_string(),
            author: "Your Name".to_string(),
            supported_languages: vec![Language::Python],
        }
    }

    fn transpile(&self, source: &str, _lang: Language) -> anyhow::Result<String> {
        // Custom conversion logic
        Ok(convert_custom_framework(source))
    }
}

fn main() -> anyhow::Result<()> {
    let mut registry = PluginRegistry::new();
    registry.register(Box::new(MyCustomConverter))?;

    // Use plugin for conversion
    let plugins = registry.get_for_language(Language::Python);
    if let Some(plugin) = plugins.first() {
        let output = plugin.transpile(source_code, Language::Python)?;
    }
    Ok(())
}
```

## Integration with CITL

Batuta integrates with the Compiler-in-the-Loop (CITL) system for iterative refinement:

```
┌─────────────────────────────────────────────────────────────────┐
│                  BATUTA + CITL INTEGRATION                      │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│   │  Batuta  │───►│  Depyler │───►│   rustc  │                 │
│   │ Analyzer │    │Transpiler│    │ Compiler │                 │
│   └──────────┘    └──────────┘    └────┬─────┘                 │
│                                        │                        │
│                         ┌──────────────┘                        │
│                         ▼                                       │
│   ┌──────────────────────────────────────────────────────┐     │
│   │                    CITL Oracle                        │     │
│   │                                                       │     │
│   │   Error E0308 → TypeMapping fix                       │     │
│   │   Error E0382 → BorrowStrategy fix                    │     │
│   │   Error E0597 → LifetimeInfer fix                     │     │
│   └──────────────────────────────────────────────────────┘     │
│                         │                                       │
│                         ▼                                       │
│   ┌──────────────┐    ┌───────────┐    ┌────────────┐          │
│   │ Apply Fix    │───►│  Recompile │───►│  Success!  │          │
│   └──────────────┘    └───────────┘    └────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

When transpiled code fails to compile, Batuta queries the CITL oracle for fixes:

```rust
use batuta::citl::CITLIntegration;

let citl = CITLIntegration::new()
    .with_max_iterations(5)
    .with_confidence_threshold(0.8);

// Transpile with automatic fix attempts
let result = citl.transpile_with_repair(python_source)?;

match result {
    TranspileResult::Success { rust_code, fixes_applied } => {
        println!("Successfully transpiled with {} fixes", fixes_applied.len());
    }
    TranspileResult::Partial { rust_code, remaining_errors } => {
        println!("Partial success, {} errors remain", remaining_errors.len());
    }
}
```

## Best Practices

### 1. Start with Analysis

Always analyze your project before migration:

```bash
batuta analyze ./my-project --tdg --languages --dependencies
```

### 2. Migrate Incrementally

Use Ruchy for gradual migration:

```bash
batuta transpile --incremental --modules core,utils
```

### 3. Validate Thoroughly

Run semantic validation with syscall tracing:

```bash
batuta validate --trace-syscalls --diff-output --benchmark
```

### 4. Optimize Last

Enable optimizations only after validation:

```bash
batuta optimize --enable-simd --enable-gpu --profile aggressive
```

### 5. Document the Migration

Generate a migration report:

```bash
batuta report --format markdown --output MIGRATION.md
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Type mismatch errors | Python dynamic typing | Add type hints in Python first |
| Missing algorithm | Unsupported sklearn feature | Check Aprender docs for equivalent |
| Performance regression | Wrong backend selected | Use `--force-backend` flag |
| Memory explosion | Large intermediate tensors | Enable streaming mode |

### Debugging Tips

```bash
# Verbose transpilation
batuta transpile --verbose --debug

# Show backend selection reasoning
batuta optimize --explain-backend

# Profile memory usage
batuta validate --profile-memory
```

## See Also

- [Compiler-in-the-Loop Learning](../ml-fundamentals/compiler-in-the-loop.md)
- [CITL Automated Program Repair](./citl-automated-repair.md)
- [Case Study: aprender-tsp Sub-Crate](./tsp-solver-crate.md)
