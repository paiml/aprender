//! APR Data Embedding Example
//!
//! Demonstrates the data embedding system for .apr model files:
//! - Embedded test data with provenance tracking
//! - Compression strategies for different data types
//! - Tiny model representations for educational demos
//!
//! Toyota Way Principles:
//! - **Traceability**: DataProvenance tracks complete data lineage
//! - **Muda Elimination**: Compression strategies minimize waste
//! - **Kaizen**: TinyModelRepr optimizes for common small model patterns
//!
//! Run with: `cargo run --example apr_embed`

use aprender::embed::{DataCompression, DataProvenance, EmbeddedTestData, TinyModelRepr};

fn main() {
    println!("=== APR Data Embedding Demo ===\n");

    // Part 1: Embedded Test Data
    embedded_data_demo();

    // Part 2: Data Provenance
    provenance_demo();

    // Part 3: Compression Strategies
    compression_demo();

    // Part 4: Tiny Model Representations
    tiny_model_demo();

    // Part 5: Predictions with Tiny Models
    predictions_demo();

    println!("\n=== Data Embedding Demo Complete! ===");
}

fn embedded_data_demo() {
    println!("--- Part 1: Embedded Test Data ---\n");

    // Create embedded test data for Iris dataset (subset)
    let iris_data = EmbeddedTestData::new(
        vec![
            5.1, 3.5, 1.4, 0.2, // Sample 1 (setosa)
            4.9, 3.0, 1.4, 0.2, // Sample 2 (setosa)
            7.0, 3.2, 4.7, 1.4, // Sample 3 (versicolor)
            6.4, 3.2, 4.5, 1.5, // Sample 4 (versicolor)
            6.3, 3.3, 6.0, 2.5, // Sample 5 (virginica)
            5.8, 2.7, 5.1, 1.9, // Sample 6 (virginica)
        ],
        (6, 4), // 6 samples, 4 features
    )
    .with_targets(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    .with_feature_names(vec![
        "sepal_length".into(),
        "sepal_width".into(),
        "petal_length".into(),
        "petal_width".into(),
    ])
    .with_sample_ids(vec![
        "iris_001".into(),
        "iris_002".into(),
        "iris_051".into(),
        "iris_052".into(),
        "iris_101".into(),
        "iris_102".into(),
    ]);

    println!("Iris Dataset (embedded):");
    println!("  Samples: {}", iris_data.n_samples());
    println!("  Features: {}", iris_data.n_features());
    println!("  Size: {} bytes", iris_data.size_bytes());

    println!(
        "\nFeature Names: {:?}",
        iris_data
            .feature_names
            .as_ref()
            .expect("feature names were set")
    );

    println!("\nSample Data:");
    for i in 0..iris_data.n_samples() {
        let row = iris_data.get_row(i).expect("row index should be in range");
        let target = iris_data
            .get_target(i)
            .expect("target index should be in range");
        let species = match target as u8 {
            0 => "setosa",
            1 => "versicolor",
            _ => "virginica",
        };
        println!(
            "  {} [{:.1}, {:.1}, {:.1}, {:.1}] -> {}",
            iris_data.sample_ids.as_ref().expect("sample IDs were set")[i],
            row[0],
            row[1],
            row[2],
            row[3],
            species
        );
    }

    // Validate data integrity
    match iris_data.validate() {
        Ok(()) => println!("\nData validation: PASSED"),
        Err(e) => println!("\nData validation FAILED: {}", e),
    }
    println!();
}

fn provenance_demo() {
    println!("--- Part 2: Data Provenance ---\n");

    // Complete provenance for educational dataset
    let complete = DataProvenance::new("UCI Iris Dataset")
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
        .with_metadata("year", "1936")
        .with_metadata("repository", "https://archive.ics.uci.edu/ml/datasets/iris");

    println!("Complete Provenance:");
    println!("  Source: {}", complete.source);
    println!("  Subset: {:?}", complete.subset_criteria);
    println!("  Preprocessing Steps:");
    for step in &complete.preprocessing {
        println!("    - {}", step);
    }
    println!("  License: {:?}", complete.license);
    println!("  Version: {:?}", complete.version);
    println!("  Is Complete: {}", complete.is_complete());

    println!("\nMetadata:");
    for (key, value) in &complete.metadata {
        println!("    {}: {}", key, value);
    }

    // Incomplete provenance
    let incomplete = DataProvenance::new("unknown_source");
    println!("\nIncomplete Provenance:");
    println!("  Source: {}", incomplete.source);
    println!(
        "  Is Complete: {} (missing license)",
        incomplete.is_complete()
    );
    println!();
}

fn compression_demo() {
    println!("--- Part 3: Compression Strategies ---\n");

    let strategies = [
        (DataCompression::None, "Raw data, zero latency"),
        (DataCompression::zstd(), "General purpose (default level 3)"),
        (
            DataCompression::zstd_level(15),
            "High compression (level 15)",
        ),
        (DataCompression::delta_zstd(), "Time series / sorted data"),
        (DataCompression::quantized(8), "8-bit quantization"),
        (DataCompression::quantized(4), "4-bit quantization"),
        (
            DataCompression::sparse(0.001),
            "Sparse data (threshold 0.001)",
        ),
    ];

    println!("{:<25} {:>15} {:>15}", "Strategy", "Name", "Est. Ratio");
    println!("{}", "-".repeat(60));

    for (strategy, description) in &strategies {
        println!(
            "{:<25} {:>15} {:>15.1}x",
            description,
            strategy.name(),
            strategy.estimated_ratio()
        );
    }

    // Apply compression to embedded data
    println!("\nCompression Selection Guide:");
    println!("  - Time series data: delta-zstd (5-20x ratio)");
    println!("  - Neural network weights: quantized-entropy (4-8x)");
    println!("  - Sparse feature matrices: sparse (proportional to sparsity)");
    println!("  - General purpose: zstd level 3 (2-4x, fast)");
    println!("  - Archive/cold storage: zstd level 15+ (6x+, slower)");
    println!();
}

fn tiny_model_demo() {
    println!("--- Part 4: Tiny Model Representations ---\n");

    // Linear Regression
    let linear = TinyModelRepr::linear(vec![0.5, -0.3, 0.8, 0.2, -0.1], 1.5);
    println!("Linear Model:");
    println!("  Summary: {}", linear.summary());
    println!("  Type: {}", linear.model_type());
    println!("  Size: {} bytes", linear.size_bytes());
    println!("  Parameters: {}", linear.n_parameters());
    println!("  Features: {:?}", linear.n_features());
    println!("  Fits in 100 bytes: {}", linear.fits_within(100));
    println!("  Validation: {:?}", linear.validate());

    // Decision Stump
    let stump = TinyModelRepr::stump(2, 0.5, -1.0, 1.0);
    println!("\nDecision Stump:");
    println!("  Summary: {}", stump.summary());
    println!("  Size: {} bytes (extremely compact!)", stump.size_bytes());

    // Naive Bayes
    let naive_bayes = TinyModelRepr::naive_bayes(
        vec![0.33, 0.33, 0.34], // class priors
        vec![
            vec![5.0, 3.4, 1.5, 0.2], // setosa means
            vec![5.9, 2.8, 4.3, 1.3], // versicolor means
            vec![6.6, 3.0, 5.5, 2.0], // virginica means
        ],
        vec![
            vec![0.12, 0.14, 0.03, 0.01], // setosa variances
            vec![0.27, 0.10, 0.22, 0.04], // versicolor variances
            vec![0.40, 0.10, 0.30, 0.07], // virginica variances
        ],
    );
    println!("\nNaive Bayes (Iris):");
    println!("  Summary: {}", naive_bayes.summary());
    println!("  Parameters: {}", naive_bayes.n_parameters());
    println!("  Validation: {:?}", naive_bayes.validate());

    // K-Means
    let kmeans = TinyModelRepr::kmeans(vec![
        vec![5.0, 3.4, 1.5, 0.2], // centroid 0 (setosa)
        vec![5.9, 2.8, 4.3, 1.3], // centroid 1 (versicolor)
        vec![6.6, 3.0, 5.5, 2.0], // centroid 2 (virginica)
    ]);
    println!("\nK-Means (3 clusters):");
    println!("  Summary: {}", kmeans.summary());

    // Logistic Regression
    let logistic = TinyModelRepr::logistic_regression(
        vec![
            vec![0.1, 0.2, -0.3, 0.4],
            vec![-0.2, 0.1, 0.3, -0.1],
            vec![0.1, -0.3, 0.0, -0.3],
        ],
        vec![0.5, -0.2, -0.3],
    );
    println!("\nLogistic Regression (3-class):");
    println!("  Summary: {}", logistic.summary());

    // KNN
    let knn = TinyModelRepr::knn(
        vec![
            vec![5.1, 3.5, 1.4, 0.2],
            vec![7.0, 3.2, 4.7, 1.4],
            vec![6.3, 3.3, 6.0, 2.5],
        ],
        vec![0, 1, 2], // labels
        1,             // k=1
    );
    println!("\nKNN (k=1):");
    println!("  Summary: {}", knn.summary());
    println!("  Validation: {:?}", knn.validate());

    // Compressed
    let compressed = TinyModelRepr::compressed(
        DataCompression::zstd_level(10),
        vec![0x78, 0x9C, 0x63, 0x60], // sample compressed data
        1000,                         // original size
    );
    println!("\nCompressed Model:");
    println!("  Summary: {}", compressed.summary());
    println!();
}

fn predictions_demo() {
    println!("--- Part 5: Predictions with Tiny Models ---\n");

    // Linear model prediction
    let linear = TinyModelRepr::linear(vec![0.5, -0.3, 0.8, 0.2], 1.0);

    let features = [5.1, 3.5, 1.4, 0.2];
    if let Some(pred) = linear.predict_linear(&features) {
        println!("Linear Prediction:");
        println!("  Features: {:?}", features);
        println!("  Prediction: {:.4}", pred);
        println!("  (0.5*5.1 + -0.3*3.5 + 0.8*1.4 + 0.2*0.2 + 1.0)");
    }

    // Decision stump prediction
    let stump = TinyModelRepr::stump(2, 2.5, 0.0, 1.0); // petal_length < 2.5 -> setosa

    println!("\nDecision Stump Predictions:");
    let test_cases = [
        [5.1, 3.5, 1.4, 0.2], // setosa (petal_length < 2.5)
        [7.0, 3.2, 4.7, 1.4], // not setosa (petal_length >= 2.5)
    ];

    for features in &test_cases {
        if let Some(pred) = stump.predict_stump(features) {
            let label = if pred < 0.5 { "setosa" } else { "other" };
            println!(
                "  petal_length={:.1} -> prediction={:.0} ({})",
                features[2], pred, label
            );
        }
    }

    // K-Means cluster assignment
    let kmeans = TinyModelRepr::kmeans(vec![
        vec![5.0, 3.4, 1.5, 0.2], // cluster 0 (setosa-like)
        vec![5.9, 2.8, 4.3, 1.3], // cluster 1 (versicolor-like)
        vec![6.6, 3.0, 5.5, 2.0], // cluster 2 (virginica-like)
    ]);

    println!("\nK-Means Cluster Assignments:");
    let samples = [
        ([5.1, 3.5, 1.4, 0.2], "setosa"),
        ([7.0, 3.2, 4.7, 1.4], "versicolor"),
        ([6.3, 3.3, 6.0, 2.5], "virginica"),
    ];

    for (features, expected) in &samples {
        if let Some(cluster) = kmeans.predict_kmeans(features) {
            println!(
                "  {} -> cluster {} (expected: {})",
                expected, cluster, expected
            );
        }
    }

    // Model validation errors
    println!("\nModel Validation Examples:");

    let invalid_linear = TinyModelRepr::linear(vec![1.0, f32::NAN, 3.0], 0.0);
    println!("  Invalid Linear (NaN): {:?}", invalid_linear.validate());

    let invalid_knn = TinyModelRepr::knn(
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        vec![0, 1],
        5, // k > n_samples
    );
    println!("  Invalid KNN (k=5, n=2): {:?}", invalid_knn.validate());

    let invalid_nb = TinyModelRepr::naive_bayes(
        vec![0.5, 0.5],
        vec![vec![1.0], vec![2.0]],
        vec![vec![0.1], vec![-0.1]], // negative variance
    );
    println!(
        "  Invalid NaiveBayes (neg variance): {:?}",
        invalid_nb.validate()
    );
    println!();
}
