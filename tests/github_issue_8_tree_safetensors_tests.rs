// GitHub Issue #8: Complete SafeTensors Model Serialization
// Tests for DecisionTreeClassifier SafeTensors export functionality
//
// Acceptance Criteria:
// - DecisionTreeClassifier::save_safetensors() exports valid SafeTensors format
// - Tree structure flattened to arrays (node_features, node_thresholds, etc.)
// - Roundtrip: save → load produces identical predictions
// - Tree structure preserved (depth, splits, leaves)

use aprender::primitives::Matrix;
use aprender::tree::DecisionTreeClassifier;
use std::fs;
use std::path::Path;

#[test]
fn test_tree_save_safetensors_creates_file() {
    // Train a simple decision tree
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("Test data should be valid");
    let y = vec![0, 1, 1, 0];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(3);
    tree.fit(&x, &y).expect("Test data should be valid");

    // Save to SafeTensors format
    let path = "test_tree_model.safetensors";
    tree.save_safetensors(path)
        .expect("Test data should be valid");

    // Verify file was created
    assert!(
        Path::new(path).exists(),
        "SafeTensors file should be created"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_tree_save_load_roundtrip() {
    // Train decision tree on XOR problem
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("Test data should be valid");
    let y = vec![0, 1, 1, 0];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x, &y).expect("Test data should be valid");

    // Get original predictions
    let pred_original = tree.predict(&x);

    // Save and load
    let path = "test_tree_roundtrip.safetensors";
    tree.save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_tree =
        DecisionTreeClassifier::load_safetensors(path).expect("Test data should be valid");

    // Get loaded tree predictions
    let pred_loaded = loaded_tree.predict(&x);

    // Verify predictions match exactly
    assert_eq!(
        pred_original, pred_loaded,
        "Predictions should be identical after roundtrip"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_tree_safetensors_metadata_includes_tensors() {
    // Create and fit tree
    let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        .expect("Test data should be valid");
    let y = vec![0, 1, 2];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(3);
    tree.fit(&x, &y).expect("Test data should be valid");

    let path = "test_tree_metadata.safetensors";
    tree.save_safetensors(path)
        .expect("Test data should be valid");

    let bytes = fs::read(path).expect("Test data should be valid");

    // Extract metadata
    let header_bytes: [u8; 8] = bytes[0..8].try_into().expect("Test data should be valid");
    let metadata_len = u64::from_le_bytes(header_bytes) as usize;
    let metadata_json = &bytes[8..8 + metadata_len];
    let metadata_str = std::str::from_utf8(metadata_json).expect("Test data should be valid");

    // Parse JSON
    let metadata: serde_json::Value =
        serde_json::from_str(metadata_str).expect("Test data should be valid");

    // Verify all required tensors exist
    assert!(
        metadata.get("node_features").is_some(),
        "Metadata must include 'node_features' tensor"
    );
    assert!(
        metadata.get("node_thresholds").is_some(),
        "Metadata must include 'node_thresholds' tensor"
    );
    assert!(
        metadata.get("node_classes").is_some(),
        "Metadata must include 'node_classes' tensor"
    );
    assert!(
        metadata.get("node_samples").is_some(),
        "Metadata must include 'node_samples' tensor"
    );
    assert!(
        metadata.get("node_left_child").is_some(),
        "Metadata must include 'node_left_child' tensor"
    );
    assert!(
        metadata.get("node_right_child").is_some(),
        "Metadata must include 'node_right_child' tensor"
    );
    assert!(
        metadata.get("max_depth").is_some(),
        "Metadata must include 'max_depth' tensor"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_tree_save_unfitted_model_fails() {
    // Unfitted model should not be saveable
    let tree = DecisionTreeClassifier::new();

    let path = "test_unfitted_tree.safetensors";
    let result = tree.save_safetensors(path);

    assert!(
        result.is_err(),
        "Saving unfitted model should return an error"
    );
    let error_msg = result.unwrap_err();
    assert!(
        error_msg.contains("unfitted") || error_msg.contains("fit"),
        "Error message should mention model is unfitted. Got: {error_msg}"
    );

    // Ensure no file was created
    assert!(
        !Path::new(path).exists(),
        "No file should be created for unfitted model"
    );
}

#[test]
fn test_tree_load_nonexistent_file_fails() {
    let result = DecisionTreeClassifier::load_safetensors("nonexistent_tree_file.safetensors");
    assert!(
        result.is_err(),
        "Loading nonexistent file should return an error"
    );
}

#[test]
fn test_tree_load_corrupted_metadata_fails() {
    // Create a file with invalid SafeTensors format
    let path = "test_corrupted_tree.safetensors";
    fs::write(path, b"invalid safetensors data").expect("Test data should be valid");

    let result = DecisionTreeClassifier::load_safetensors(path);
    assert!(
        result.is_err(),
        "Loading corrupted file should return an error"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_tree_multiple_save_load_cycles() {
    // Property test: Multiple save/load cycles should be idempotent
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0],
    )
    .expect("Test data should be valid");
    let y = vec![0, 1, 1, 0, 2, 2];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(4);
    tree.fit(&x, &y).expect("Test data should be valid");

    let path1 = "test_tree_cycle1.safetensors";
    let path2 = "test_tree_cycle2.safetensors";
    let path3 = "test_tree_cycle3.safetensors";

    // Cycle 1: original → save → load
    tree.save_safetensors(path1)
        .expect("Test data should be valid");
    let tree1 = DecisionTreeClassifier::load_safetensors(path1).expect("Test data should be valid");

    // Cycle 2: loaded → save → load
    tree1
        .save_safetensors(path2)
        .expect("Test data should be valid");
    let tree2 = DecisionTreeClassifier::load_safetensors(path2).expect("Test data should be valid");

    // Cycle 3: loaded again → save → load
    tree2
        .save_safetensors(path3)
        .expect("Test data should be valid");
    let tree3 = DecisionTreeClassifier::load_safetensors(path3).expect("Test data should be valid");

    // All trees should produce identical predictions
    let pred_orig = tree.predict(&x);
    let pred1 = tree1.predict(&x);
    let pred2 = tree2.predict(&x);
    let pred3 = tree3.predict(&x);

    assert_eq!(pred_orig, pred1, "Cycle 1 predictions mismatch");
    assert_eq!(pred_orig, pred2, "Cycle 2 predictions mismatch");
    assert_eq!(pred_orig, pred3, "Cycle 3 predictions mismatch");

    // Cleanup
    fs::remove_file(path1).ok();
    fs::remove_file(path2).ok();
    fs::remove_file(path3).ok();
}

#[test]
fn test_tree_accuracy_score_preserved() {
    // Property test: Accuracy should be identical after roundtrip
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 0.0, 2.0, 1.0, 3.0, 0.0, 3.0, 1.0,
        ],
    )
    .expect("Test data should be valid");
    let y = vec![0, 1, 1, 0, 2, 2, 3, 3];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x, &y).expect("Test data should be valid");

    let original_accuracy = tree.score(&x, &y);

    // Save and load
    let path = "test_tree_accuracy.safetensors";
    tree.save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_tree =
        DecisionTreeClassifier::load_safetensors(path).expect("Test data should be valid");

    let loaded_accuracy = loaded_tree.score(&x, &y);

    // Accuracy should be identical
    assert!(
        (original_accuracy - loaded_accuracy).abs() < 1e-6,
        "Accuracy should be preserved: original={original_accuracy}, loaded={loaded_accuracy}"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_tree_deep_tree_serialization() {
    // Test with a deeper tree to verify complex structure serialization
    let x = Matrix::from_vec(
        16,
        2,
        vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 2.0,
            0.0, 2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 0.0, 3.0, 1.0, 3.0, 2.0, 3.0, 3.0,
        ],
    )
    .expect("Test data should be valid");
    let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(10);
    tree.fit(&x, &y).expect("Test data should be valid");

    // Save and load
    let path = "test_tree_deep.safetensors";
    tree.save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_tree =
        DecisionTreeClassifier::load_safetensors(path).expect("Test data should be valid");

    // Verify predictions match
    let pred_original = tree.predict(&x);
    let pred_loaded = loaded_tree.predict(&x);

    assert_eq!(pred_original, pred_loaded, "Deep tree predictions mismatch");

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_tree_file_size_reasonable() {
    // Verify SafeTensors file size is reasonable
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("Test data should be valid");
    let y = vec![0, 1, 1, 0];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(3);
    tree.fit(&x, &y).expect("Test data should be valid");

    let path = "test_tree_size.safetensors";
    tree.save_safetensors(path)
        .expect("Test data should be valid");

    let file_size = fs::metadata(path).expect("Test data should be valid").len();

    // File should be reasonable (not huge for small tree)
    assert!(
        file_size < 4096,
        "SafeTensors file should be compact for small tree. Got {file_size} bytes"
    );

    assert!(
        file_size > 50,
        "SafeTensors file should contain data. Got {file_size} bytes"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_tree_single_leaf_serialization() {
    // Edge case: Tree with just one leaf (all samples same class)
    let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        .expect("Test data should be valid");
    let y = vec![1, 1, 1]; // All same class

    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x, &y).expect("Test data should be valid");

    // Save and load
    let path = "test_tree_single_leaf.safetensors";
    tree.save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_tree =
        DecisionTreeClassifier::load_safetensors(path).expect("Test data should be valid");

    // Verify predictions
    let pred_original = tree.predict(&x);
    let pred_loaded = loaded_tree.predict(&x);

    assert_eq!(pred_original, pred_loaded);
    assert_eq!(pred_loaded, vec![1, 1, 1], "Should predict class 1 for all");

    // Cleanup
    fs::remove_file(path).ok();
}
