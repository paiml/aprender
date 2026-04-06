# Aprender ML Integration

Trueno-viz integrates seamlessly with the aprender machine learning library
for end-to-end ML visualization pipelines.

## Enabling Integration

```toml
[dependencies]
trueno-viz = { version = "0.1", features = ["ml"] }
aprender = "0.1"
```

## DataFrame Conversion

```rust
use aprender::DataFrame as AprenderDF;
use trueno_viz::interop::aprender as viz;

let aprender_df = AprenderDF::from_csv("data.csv").unwrap();

// Direct visualization
let scatter = viz::scatter(&aprender_df, "x_column", "y_column")
    .color_by("category")
    .build();
```

**Test Reference**: `src/interop/aprender.rs::test_scatter_from_aprender`

## Model Evaluation Plots

### ROC Curve from Classifier

```rust
use aprender::classification::LogisticRegression;
use trueno_viz::interop::aprender as viz;

let model = LogisticRegression::new().fit(&x_train, &y_train);
let y_pred_proba = model.predict_proba(&x_test);

// Direct ROC curve
let roc = viz::roc_curve(&y_test, &y_pred_proba)
    .title("Logistic Regression ROC")
    .build();
```

### Confusion Matrix from Predictions

```rust
use trueno_viz::interop::aprender as viz;

let y_pred = model.predict(&x_test);

let cm = viz::confusion_matrix(&y_test, &y_pred)
    .class_names(&["Negative", "Positive"])
    .normalize(true)
    .build();
```

### Learning Curves

```rust
use trueno_viz::interop::aprender as viz;

let history = model.training_history();

let loss_curve = viz::loss_curve(&history)
    .title("Training Progress")
    .build();
```

## Feature Analysis

### Correlation Heatmap

```rust
use trueno_viz::interop::aprender as viz;

let correlation_matrix = df.correlation();

let heatmap = viz::correlation_heatmap(&df)
    .title("Feature Correlations")
    .build();
```

### Feature Importance

```rust
use aprender::ensemble::RandomForest;
use trueno_viz::interop::aprender as viz;

let rf = RandomForest::new().fit(&x, &y);

let importance = viz::feature_importance(&rf, &feature_names)
    .top_n(10)
    .build();
```

## Distribution Analysis

### Histogram from DataFrame Column

```rust
use trueno_viz::interop::aprender as viz;

let hist = viz::histogram(&df, "age")
    .bins(20)
    .title("Age Distribution")
    .build();
```

**Test Reference**: `src/interop/aprender.rs::test_histogram_from_aprender`

### Box Plot by Category

```rust
use trueno_viz::interop::aprender as viz;

let boxplot = viz::boxplot(&df, "value", "category")
    .title("Value Distribution by Category")
    .build();
```

## Clustering Visualization

### K-Means Results

```rust
use aprender::clustering::KMeans;
use trueno_viz::interop::aprender as viz;

let kmeans = KMeans::new(3).fit(&data);
let labels = kmeans.labels();

let scatter = viz::scatter_clusters(&data, &labels)
    .show_centroids(true)
    .title("K-Means Clustering")
    .build();
```

### PCA Projection

```rust
use aprender::decomposition::PCA;
use trueno_viz::interop::aprender as viz;

let pca = PCA::new(2).fit_transform(&data);

let scatter = viz::scatter_2d(&pca)
    .color_by(&labels)
    .title("PCA Projection")
    .build();
```

## Complete ML Pipeline Example

```rust
use aprender::{DataFrame, classification::LogisticRegression, metrics};
use trueno_viz::prelude::*;
use trueno_viz::interop::aprender as viz;

fn main() -> Result<()> {
    // Load data
    let df = DataFrame::from_csv("iris.csv")?;

    // 1. Exploratory visualization
    let scatter = viz::scatter(&df, "sepal_length", "sepal_width")
        .color_by("species")
        .title("Iris Dataset")
        .build();
    scatter.render_to_file("iris_scatter.png")?;

    // 2. Feature correlation
    let corr = viz::correlation_heatmap(&df)
        .build();
    corr.render_to_file("iris_correlation.png")?;

    // 3. Train model
    let (x_train, x_test, y_train, y_test) = df.train_test_split(0.2);
    let model = LogisticRegression::new().fit(&x_train, &y_train);

    // 4. Evaluate
    let y_pred_proba = model.predict_proba(&x_test);
    let y_pred = model.predict(&x_test);

    // ROC curve (for binary, use one-vs-rest for multiclass)
    let roc = viz::roc_curve_multiclass(&y_test, &y_pred_proba)
        .class_names(&["setosa", "versicolor", "virginica"])
        .build();
    roc.render_to_file("iris_roc.png")?;

    // Confusion matrix
    let cm = viz::confusion_matrix(&y_test, &y_pred)
        .class_names(&["setosa", "versicolor", "virginica"])
        .build();
    cm.render_to_file("iris_confusion.png")?;

    println!("Accuracy: {:.2}%", metrics::accuracy(&y_test, &y_pred) * 100.0);

    Ok(())
}
```

## Next Chapter

Continue to [Trueno-Graph Visualization](./trueno-graph.md) for graph visualization.
