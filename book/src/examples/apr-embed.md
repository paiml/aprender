# Case Study: APR Data Embedding

This example demonstrates the data embedding system for `.apr` model files, enabling bundled test data and tiny model representations.

## Overview

The embedding module provides:
- **Embedded Test Data**: Bundle sample datasets with models
- **Data Provenance**: Track complete data lineage (Toyota Way: traceability)
- **Compression Strategies**: Optimize storage for different data types
- **Tiny Model Representations**: Efficient storage for small models

## Toyota Way Principles

| Principle | Application |
|-----------|-------------|
| Traceability | DataProvenance tracks complete data lineage |
| Muda Elimination | Compression strategies minimize waste |
| Kaizen | TinyModelRepr optimizes for common patterns |

## Running the Example

```bash
cargo run --example apr_embed
```

## Embedded Test Data

Bundle sample data directly in model files:

```rust
let iris_data = EmbeddedTestData::new(
    vec![
        5.1, 3.5, 1.4, 0.2,  // Sample 1 (setosa)
        4.9, 3.0, 1.4, 0.2,  // Sample 2 (setosa)
        7.0, 3.2, 4.7, 1.4,  // Sample 3 (versicolor)
        // ...
    ],
    (6, 4),  // 6 samples, 4 features
)
.with_targets(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
.with_feature_names(vec![
    "sepal_length".into(),
    "sepal_width".into(),
    "petal_length".into(),
    "petal_width".into(),
])
.with_sample_ids(vec!["iris_001".into(), "iris_002".into(), /* ... */]);

println!("Samples: {}", iris_data.n_samples());
println!("Features: {}", iris_data.n_features());
println!("Size: {} bytes", iris_data.size_bytes());

// Access rows
let row = iris_data.get_row(0).unwrap();
let target = iris_data.get_target(0).unwrap();

// Validate integrity
iris_data.validate()?;
```

## Data Provenance

Track data lineage for reproducibility:

```rust
let provenance = DataProvenance::new("UCI Iris Dataset")
    .with_subset("stratified sample of 6 instances")
    .with_preprocessing("normalize")
    .with_preprocessing("remove_outliers")
    .with_preprocessing_steps(vec![
        "StandardScaler applied".into(),
        "PCA(n_components=4)".into(),
    ])
    .with_license("CC0 1.0 Universal")
    .with_version("1.0.0")
    .with_metadata("author", "R.A. Fisher")
    .with_metadata("year", "1936");

println!("Source: {}", provenance.source);
println!("Is Complete: {}", provenance.is_complete());
```

## Compression Strategies

Select compression based on data type:

| Strategy | Ratio | Use Case |
|----------|-------|----------|
| None | 1x | Zero latency |
| Zstd (level 3) | 2.5x | General purpose |
| Zstd (level 15) | 6x | Archive/cold |
| Delta-Zstd | 8-12x | Time series |
| Quantized (8-bit) | 4x | Neural weights |
| Quantized (4-bit) | 8x | Aggressive compression |
| Sparse | ~5x | Sparse features |

```rust
let strategies = [
    DataCompression::None,
    DataCompression::zstd(),
    DataCompression::zstd_level(15),
    DataCompression::delta_zstd(),
    DataCompression::quantized(8),
    DataCompression::quantized(4),
    DataCompression::sparse(0.001),
];

for strategy in &strategies {
    println!("{}: {:.1}x ratio", strategy.name(), strategy.estimated_ratio());
}
```

## Tiny Model Representations

Efficient storage for small models (<1 MB):

### Linear Model

```rust
let linear = TinyModelRepr::linear(
    vec![0.5, -0.3, 0.8, 0.2, -0.1],
    1.5,  // intercept
);

println!("Size: {} bytes", linear.size_bytes());  // ~24 bytes
println!("Parameters: {}", linear.n_parameters());

// Predict
let pred = linear.predict_linear(&[5.1, 3.5, 1.4, 0.2, 1.0]);
```

### Decision Stump

```rust
let stump = TinyModelRepr::stump(2, 0.5, -1.0, 1.0);
println!("Size: {} bytes", stump.size_bytes());  // 14 bytes

// Predict
let pred = stump.predict_stump(&[0.0, 0.0, 0.3, 0.0]);  // -> -1.0
```

### K-Means

```rust
let kmeans = TinyModelRepr::kmeans(vec![
    vec![5.0, 3.4, 1.5, 0.2],  // cluster 0
    vec![5.9, 2.8, 4.3, 1.3],  // cluster 1
    vec![6.6, 3.0, 5.5, 2.0],  // cluster 2
]);

// Find nearest cluster
let cluster = kmeans.predict_kmeans(&[5.1, 3.5, 1.4, 0.2]);  // -> 0
```

### Naive Bayes

```rust
let naive_bayes = TinyModelRepr::naive_bayes(
    vec![0.33, 0.33, 0.34],  // priors
    vec![
        vec![5.0, 3.4, 1.5, 0.2],  // class 0 means
        vec![5.9, 2.8, 4.3, 1.3],  // class 1 means
        vec![6.6, 3.0, 5.5, 2.0],  // class 2 means
    ],
    vec![
        vec![0.12, 0.14, 0.03, 0.01],  // class 0 variances
        vec![0.27, 0.10, 0.22, 0.04],  // class 1 variances
        vec![0.40, 0.10, 0.30, 0.07],  // class 2 variances
    ],
);
```

### KNN

```rust
let knn = TinyModelRepr::knn(
    vec![
        vec![5.1, 3.5, 1.4, 0.2],
        vec![7.0, 3.2, 4.7, 1.4],
        vec![6.3, 3.3, 6.0, 2.5],
    ],
    vec![0, 1, 2],  // labels
    1,              // k=1
);
```

## Model Validation

Detect invalid model parameters:

```rust
// Invalid: NaN coefficient
let invalid = TinyModelRepr::linear(vec![1.0, f32::NAN, 3.0], 0.0);
match invalid.validate() {
    Err(TinyModelError::InvalidCoefficient { index, value }) => {
        println!("Invalid at index {}: {}", index, value);
    }
    _ => {}
}

// Invalid: negative variance
let invalid_nb = TinyModelRepr::naive_bayes(
    vec![0.5, 0.5],
    vec![vec![1.0], vec![2.0]],
    vec![vec![0.1], vec![-0.1]],  // negative!
);
// Returns Err(TinyModelError::InvalidVariance { ... })

// Invalid: k > n_samples
let invalid_knn = TinyModelRepr::knn(
    vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    vec![0, 1],
    5,  // k=5 but only 2 samples!
);
// Returns Err(TinyModelError::InvalidK { ... })
```

## Source Code

- Example: `examples/apr_embed.rs`
- Module: `src/embed/mod.rs`
- Tiny Models: `src/embed/tiny.rs`
