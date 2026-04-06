# sklearn to Aprender Migration

Batuta's `SklearnConverter` maps scikit-learn algorithms to their `aprender`
equivalents. The Rust API preserves sklearn's familiar `fit`/`predict` pattern
while providing compile-time type safety and SIMD acceleration.

## Linear Regression

**Python (sklearn)**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Rust (Aprender)**

```rust
use aprender::linear_model::LinearRegression;
use aprender::model_selection::train_test_split;
use aprender::Estimator;

let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.25)?;
let mut model = LinearRegression::new();
model.fit(&x_train, &y_train)?;
let predictions = model.predict(&x_test)?;
```

The `Estimator` trait provides `fit` and `predict`. Error handling uses Rust's
`Result` type instead of Python exceptions.

## KMeans Clustering

**Python (sklearn)**

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(X)
labels = model.predict(X)
```

**Rust (Aprender)**

```rust
use aprender::cluster::KMeans;
use aprender::UnsupervisedEstimator;

let mut model = KMeans::new(3);
model.fit(&x)?;
let labels = model.predict(&x)?;
```

Unsupervised algorithms implement `UnsupervisedEstimator`, which takes only
feature data (no labels) in `fit`.

## Preprocessing

**Python (sklearn)**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Rust (Aprender)**

```rust
use aprender::preprocessing::StandardScaler;
use aprender::Transformer;

let mut scaler = StandardScaler::new();
scaler.fit(&x_train)?;
let x_train_scaled = scaler.transform(&x_train)?;
let x_test_scaled = scaler.transform(&x_test)?;
```

Preprocessors implement the `Transformer` trait. The `fit` and `transform` steps
are explicit, avoiding the hidden state mutation that `fit_transform` can mask.

## Decision Trees and Ensembles

**Python (sklearn)**

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Rust (Aprender)**

```rust
use aprender::tree::DecisionTreeClassifier;
use aprender::Estimator;

let mut model = DecisionTreeClassifier::new();
model.fit(&x_train, &y_train)?;
let predictions = model.predict(&x_test)?;
```

Tree-based models and ensemble methods are classified as high-complexity
operations. On large datasets, Batuta routes them to GPU via the MoE backend
selector.

## Metrics

**Python (sklearn)**

```python
from sklearn.metrics import accuracy_score, mean_squared_error

acc = accuracy_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
```

**Rust (Aprender)**

```rust
use aprender::metrics::{accuracy_score, mean_squared_error};

let acc = accuracy_score(&y_true, &y_pred)?;
let mse = mean_squared_error(&y_true, &y_pred)?;
```

## Conversion Coverage

| sklearn Module            | Aprender Equivalent          | Status         |
|--------------------------|------------------------------|----------------|
| `sklearn.linear_model`   | `aprender::linear_model`     | Full           |
| `sklearn.cluster`        | `aprender::cluster`          | Full           |
| `sklearn.tree`           | `aprender::tree`             | Full           |
| `sklearn.ensemble`       | `aprender::ensemble`         | Full           |
| `sklearn.preprocessing`  | `aprender::preprocessing`    | Full           |
| `sklearn.model_selection`| `aprender::model_selection`  | Full           |
| `sklearn.metrics`        | `aprender::metrics`          | Full           |

## Key Takeaways

- The `fit`/`predict` pattern is preserved across all algorithm families.
- Three traits map sklearn's implicit duck typing: `Estimator` (supervised),
  `UnsupervisedEstimator` (clustering), and `Transformer` (preprocessing).
- All operations return `Result` for explicit error handling.
- Backend selection is automatic: small datasets use scalar, medium use SIMD,
  large use GPU.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
