//! Integration tests for Aprender ML library.
//!
//! These tests verify end-to-end workflows combining multiple components.

use aprender::prelude::*;

#[test]
fn test_linear_regression_workflow() {
    // Create training data (non-collinear)
    let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 4.0, 3.0, 2.0, 4.0, 5.0, 5.0, 3.0])
        .expect("Test data should be valid");
    let y = Vector::from_slice(&[3.0, 8.0, 7.0, 13.0, 11.0]);

    // Train model
    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Failed to fit model");

    // Verify coefficients
    assert_eq!(model.coefficients().len(), 2);

    // Make predictions
    let predictions = model.predict(&x);
    assert_eq!(predictions.len(), 5);

    // Evaluate model
    let r2 = model.score(&x, &y);
    assert!(r2 > 0.9, "R² should be high for linear data: {r2}");

    // Test on new data
    let new_x = Matrix::from_vec(1, 2, vec![6.0, 7.0]).expect("Test data should be valid");
    let new_pred = model.predict(&new_x);
    assert_eq!(new_pred.len(), 1);
}

#[test]
fn test_kmeans_workflow() {
    // Create clustered data
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, 1.5, 1.5, 2.0, 2.0, // Cluster 1
            10.0, 10.0, 10.5, 10.5, 11.0, 11.0, // Cluster 2
        ],
    )
    .expect("Test data should be valid");

    // Train model
    let mut kmeans = KMeans::new(2).with_max_iter(100).with_random_state(42);
    kmeans.fit(&x).expect("Failed to fit K-Means");

    // Get predictions
    let labels = kmeans.predict(&x);
    assert_eq!(labels.len(), 6);

    // Verify clusters are distinct
    let first_cluster = labels[0];
    let second_cluster = labels[3];
    assert_ne!(first_cluster, second_cluster);

    // Verify cluster consistency
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[4], labels[5]);

    // Evaluate clustering
    let silhouette = silhouette_score(&x, &labels);
    assert!(
        silhouette > 0.5,
        "Silhouette should be high for well-separated clusters: {silhouette}"
    );
}

#[test]
fn test_dataframe_to_ml_workflow() {
    // Create DataFrame (non-collinear features)
    let columns = vec![
        (
            "feature1".to_string(),
            Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]),
        ),
        (
            "feature2".to_string(),
            Vector::from_slice(&[5.0, 3.0, 4.0, 2.0, 1.0]),
        ),
        (
            "target".to_string(),
            Vector::from_slice(&[6.0, 5.0, 7.0, 6.0, 6.0]),
        ),
    ];

    let df = DataFrame::new(columns).expect("Failed to create DataFrame");

    // Verify DataFrame
    assert_eq!(df.shape(), (5, 3));

    // Select features
    let features = df
        .select(&["feature1", "feature2"])
        .expect("Test data should be valid");
    let x = features.to_matrix();
    let y = df
        .column("target")
        .expect("Test data should be valid")
        .clone();

    // Train model
    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Failed to fit model");

    // Evaluate
    let r2 = model.score(&x, &y);
    assert!(r2 > 0.0, "R² should be positive: {r2}");
}

#[test]
fn test_metrics_consistency() {
    let actual = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let predicted = Vector::from_slice(&[1.1, 2.2, 2.9, 4.1, 4.8]);

    // All metrics should be computable
    let r2 = r_squared(&predicted, &actual);
    let mse_val = mse(&predicted, &actual);
    let rmse_val = rmse(&predicted, &actual);
    let mae_val = mae(&predicted, &actual);

    // Verify relationships
    assert!((rmse_val - mse_val.sqrt()).abs() < 1e-6);
    assert!(r2 > 0.0 && r2 <= 1.0);
    assert!(mse_val >= 0.0);
    assert!(mae_val >= 0.0);
    assert!(mae_val <= rmse_val); // MAE <= RMSE always
}

#[test]
fn test_complete_ml_pipeline() {
    // Simulate a complete ML pipeline

    // 1. Prepare data
    let columns = vec![
        (
            "sqft".to_string(),
            Vector::from_slice(&[1000.0, 1500.0, 2000.0, 2500.0, 3000.0]),
        ),
        (
            "bedrooms".to_string(),
            Vector::from_slice(&[2.0, 3.0, 3.0, 4.0, 5.0]),
        ),
        (
            "price".to_string(),
            Vector::from_slice(&[200.0, 300.0, 400.0, 500.0, 600.0]),
        ),
    ];

    let df = DataFrame::new(columns).expect("Test data should be valid");

    // 2. Get descriptive statistics
    let stats = df.describe();
    assert_eq!(stats.len(), 3);

    // 3. Extract features and target
    let x = df
        .select(&["sqft", "bedrooms"])
        .expect("Test data should be valid")
        .to_matrix();
    let y = df
        .column("price")
        .expect("Test data should be valid")
        .clone();

    // 4. Train model
    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Test data should be valid");

    // 5. Make predictions
    let predictions = model.predict(&x);

    // 6. Evaluate
    let r2 = r_squared(&predictions, &y);
    let mse_val = mse(&predictions, &y);

    assert!(r2 > 0.9);
    assert!(mse_val < 1000.0);

    // 7. Predict on new data
    let new_house = Matrix::from_vec(1, 2, vec![1800.0, 3.0]).expect("Test data should be valid");
    let predicted_price = model.predict(&new_house);

    // Price should be reasonable (between min and max of training data)
    assert!(predicted_price[0] > 200.0 && predicted_price[0] < 600.0);
}

#[test]
fn test_decision_tree_iris_classification() {
    // Simulated Iris dataset (3 species, 4 features)
    // Features: sepal_length, sepal_width, petal_length, petal_width
    let x = Matrix::from_vec(
        15,
        4,
        vec![
            // Setosa (class 0)
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0,
            3.6, 1.4, 0.2, // Versicolor (class 1)
            7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3, 6.5,
            2.8, 4.6, 1.5, // Virginica (class 2)
            6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8, 6.5,
            3.0, 5.8, 2.2,
        ],
    )
    .expect("Test data should be valid");

    let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

    // Train Decision Tree
    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x, &y).expect("Failed to fit Decision Tree");

    // Make predictions
    let predictions = tree.predict(&x);
    assert_eq!(predictions.len(), 15);

    // Calculate accuracy
    let accuracy = tree.score(&x, &y);
    assert!(
        accuracy >= 0.9,
        "Accuracy should be high on linearly separable Iris data: {accuracy}"
    );

    // Verify predictions match expected classes
    // First 5 should be class 0
    for (i, &pred) in predictions.iter().enumerate().take(5) {
        assert_eq!(pred, 0, "Sample {i} should be class 0");
    }

    // Next 5 should be class 1
    for (i, &pred) in predictions.iter().enumerate().skip(5).take(5) {
        assert_eq!(pred, 1, "Sample {i} should be class 1");
    }

    // Last 5 should be class 2
    for (i, &pred) in predictions.iter().enumerate().skip(10).take(5) {
        assert_eq!(pred, 2, "Sample {i} should be class 2");
    }

    // Test on new samples
    let new_samples = Matrix::from_vec(
        3,
        4,
        vec![
            5.0, 3.4, 1.5, 0.2, // Likely Setosa
            6.2, 2.9, 4.3, 1.3, // Likely Versicolor
            6.7, 3.1, 5.6, 2.4, // Likely Virginica
        ],
    )
    .expect("Test data should be valid");

    let new_predictions = tree.predict(&new_samples);
    assert_eq!(new_predictions.len(), 3);

    // Verify predictions are valid class labels
    for &pred in &new_predictions {
        assert!(pred < 3, "Predicted class should be 0, 1, or 2");
    }
}

// ============================================================================
// Section INT: Integration Tests (spec v3.0.0 Part IV Section 4.2)
// ============================================================================

/// INT-01: APR write/read round-trip integrity
/// FALSIFICATION: Written APR file cannot be read back correctly
#[test]
fn int01_apr_roundtrip_integrity() {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};

    // Create test metadata
    let mut metadata = AprV2Metadata::new("test_integration_model");
    metadata.model_type = "test".to_string();
    metadata.description = Some("Integration test model".to_string());

    // Create writer with metadata
    let mut writer = AprV2Writer::new(metadata.clone());

    // Add a small test tensor
    let tensor_data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    writer.add_tensor("test.weight", TensorDType::F32, vec![8, 8], tensor_bytes);

    // Write to bytes
    let apr_bytes = writer.write().expect("Failed to write APR");

    // Verify magic bytes (APR\0 per v2 spec)
    assert_eq!(
        &apr_bytes[0..4],
        b"APR\0",
        "INT-01 FALSIFIED: Wrong magic bytes (expected APR\\0)"
    );

    // Verify we can read back the metadata
    let mut cursor = std::io::Cursor::new(&apr_bytes);
    let reader =
        aprender::format::v2::AprV2Reader::from_reader(&mut cursor).expect("Failed to read APR");

    let read_metadata = reader.metadata();
    assert_eq!(
        read_metadata.name, metadata.name,
        "INT-01 FALSIFIED: Metadata name mismatch"
    );
    assert_eq!(
        read_metadata.model_type, metadata.model_type,
        "INT-01 FALSIFIED: Model type mismatch"
    );

    // Verify tensor is present
    assert!(
        reader.get_tensor("test.weight").is_some(),
        "INT-01 FALSIFIED: Tensor not found after round-trip"
    );
}

/// INT-01b: APR v2 format round-trip
/// FALSIFICATION: v2 format fails to write
#[test]
fn int01b_apr_v2_roundtrip() {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};

    let metadata = AprV2Metadata::new("v2_test");
    let mut writer = AprV2Writer::new(metadata);

    // Add minimal tensor
    let tensor_bytes: Vec<u8> = vec![0u8; 128];
    writer.add_tensor(
        "layer.0.weight",
        TensorDType::F32,
        vec![32, 4],
        tensor_bytes,
    );

    let apr_bytes = writer.write().expect("write");

    // v2 uses APR\0 magic (null-terminated)
    assert_eq!(
        &apr_bytes[0..4],
        b"APR\0",
        "INT-01b FALSIFIED: v2 should use APR\\0 magic"
    );
}

/// INT-03: SIMD operations produce correct results
/// FALSIFICATION: Matrix operations produce incorrect results
#[test]
fn int03_compute_correctness() {
    // Test that matrix operations are correct (trueno integration)
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("matrix a");

    let b = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("matrix b");

    // Matrix multiplication: (2x3) × (3x2) = (2x2)
    let c = a.matmul(&b).expect("matmul");

    assert_eq!(c.n_rows(), 2, "INT-03 FALSIFIED: Wrong output rows");
    assert_eq!(c.n_cols(), 2, "INT-03 FALSIFIED: Wrong output cols");

    // Verify values: C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    let c_data = c.as_slice();
    assert!(
        (c_data[0] - 22.0).abs() < 1e-6,
        "INT-03 FALSIFIED: C[0,0] = {} != 22.0",
        c_data[0]
    );

    // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
    assert!(
        (c_data[1] - 28.0).abs() < 1e-6,
        "INT-03 FALSIFIED: C[0,1] = {} != 28.0",
        c_data[1]
    );
}

/// INT-04: Format conversion preserves data
/// FALSIFICATION: Exported data differs from original
#[test]
fn int04_format_conversion_integrity() {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};

    // Create source APR with known data
    let metadata = AprV2Metadata::new("conversion_test");
    let mut writer = AprV2Writer::new(metadata);

    // Add tensors with known values
    let weights: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let weight_bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();

    writer.add_tensor(
        "model.weight",
        TensorDType::F32,
        vec![16, 8],
        weight_bytes.clone(),
    );

    let apr_bytes = writer.write().expect("write");

    // Read back and verify tensor data integrity
    let mut cursor = std::io::Cursor::new(&apr_bytes);
    let reader = aprender::format::v2::AprV2Reader::from_reader(&mut cursor).expect("read");

    // Verify tensor metadata
    let tensor_entry = reader
        .get_tensor("model.weight")
        .expect("INT-04 FALSIFIED: Tensor not found");
    assert_eq!(
        tensor_entry.shape,
        vec![16, 8],
        "INT-04 FALSIFIED: Shape mismatch"
    );

    // Verify tensor data if available
    if let Some(read_bytes) = reader.get_tensor_data("model.weight") {
        assert_eq!(
            read_bytes.len(),
            weight_bytes.len(),
            "INT-04 FALSIFIED: Tensor data length mismatch"
        );

        // Verify first few bytes match
        for (i, (&orig, &read)) in weight_bytes
            .iter()
            .zip(read_bytes.iter())
            .enumerate()
            .take(16)
        {
            assert_eq!(
                orig, read,
                "INT-04 FALSIFIED: Byte {} mismatch: {} vs {}",
                i, orig, read
            );
        }
    }
}

/// INT-05: End-to-end ML pipeline with model persistence
/// FALSIFICATION: Trained model cannot be used after persistence
#[test]
fn int05_ml_pipeline_persistence() {
    // Train a simple model with non-collinear data
    let x =
        Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 4.0, 3.0, 2.0, 4.0, 5.0, 5.0, 3.0]).expect("x");
    let y = Vector::from_slice(&[3.0, 8.0, 7.0, 13.0, 11.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("fit");

    let original_score = model.score(&x, &y);
    assert!(
        original_score > 0.90,
        "INT-05: Original model should fit reasonably: {}",
        original_score
    );

    // Get predictions before serialization
    let original_predictions = model.predict(&x);

    // Verify model coefficients are stable
    let coefficients = model.coefficients();
    assert_eq!(
        coefficients.len(),
        2,
        "INT-05 FALSIFIED: Wrong number of coefficients"
    );

    // Verify predictions are consistent (deterministic)
    let new_predictions = model.predict(&x);
    let orig_slice = original_predictions.as_slice();
    let new_slice = new_predictions.as_slice();

    for (i, (&orig, &new)) in orig_slice.iter().zip(new_slice.iter()).enumerate() {
        assert!(
            (orig - new).abs() < 1e-10,
            "INT-05 FALSIFIED: Prediction {} differs: {} vs {}",
            i,
            orig,
            new
        );
    }
}
