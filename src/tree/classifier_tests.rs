pub(crate) use super::*;
pub(crate) use crate::primitives::Matrix;

pub(super) fn simple_dataset() -> (Matrix<f32>, Vec<usize>) {
    // 2-class dataset: features [x1, x2], labels {0, 1}
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 2.0, // class 0
            1.5, 1.8, // class 0
            2.0, 2.5, // class 0
            5.0, 8.0, // class 1
            6.0, 9.0, // class 1
            7.0, 7.5, // class 1
        ],
    )
    .expect("valid matrix");
    let y = vec![0, 0, 0, 1, 1, 1];
    (x, y)
}

#[test]
fn test_default_impl() {
    let model = DecisionTreeClassifier::default();
    assert!(model.tree.is_none());
    assert!(model.max_depth.is_none());
    assert!(model.n_features.is_none());
}

#[test]
fn test_with_max_depth() {
    let model = DecisionTreeClassifier::new().with_max_depth(3);
    assert_eq!(model.max_depth, Some(3));
}

#[test]
fn test_fit_mismatched_x_y() {
    let x = Matrix::from_vec(3, 2, vec![1.0; 6]).expect("valid matrix");
    let y = vec![0, 1]; // 2 labels for 3 samples
    let mut model = DecisionTreeClassifier::new();
    let err = model.fit(&x, &y);
    assert!(err.is_err());
}

#[test]
fn test_fit_zero_samples() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("valid matrix");
    let y: Vec<usize> = vec![];
    let mut model = DecisionTreeClassifier::new();
    let err = model.fit(&x, &y);
    assert!(err.is_err());
}

#[test]
fn test_fit_and_predict() {
    let (x, y) = simple_dataset();
    let mut model = DecisionTreeClassifier::new();
    model.fit(&x, &y).expect("fit should succeed");
    let predictions = model.predict(&x);
    assert_eq!(predictions.len(), 6);
    // Should get most predictions correct on training data
    let correct = predictions
        .iter()
        .zip(y.iter())
        .filter(|(p, t)| p == t)
        .count();
    assert!(correct >= 4, "Expected at least 4 correct, got {correct}");
}

#[test]
fn test_score() {
    let (x, y) = simple_dataset();
    let mut model = DecisionTreeClassifier::new();
    model.fit(&x, &y).expect("fit should succeed");
    let accuracy = model.score(&x, &y);
    assert!(accuracy >= 0.5, "Expected accuracy >= 0.5, got {accuracy}");
}

#[test]
fn test_save_load_roundtrip() {
    let (x, y) = simple_dataset();
    let mut model = DecisionTreeClassifier::new().with_max_depth(5);
    model.fit(&x, &y).expect("fit should succeed");

    let dir = std::env::temp_dir().join("aprender_test_tree_classifier");
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join("tree_model.bin");

    model.save(&path).expect("save should succeed");
    let loaded = DecisionTreeClassifier::load(&path).expect("load should succeed");
    let orig_preds = model.predict(&x);
    let loaded_preds = loaded.predict(&x);
    assert_eq!(orig_preds, loaded_preds);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_load_nonexistent_file() {
    let result = DecisionTreeClassifier::load("/tmp/aprender_nonexistent_tree_model.bin");
    assert!(result.is_err());
}

#[test]
fn test_save_safetensors_unfitted() {
    let model = DecisionTreeClassifier::new();
    let result = model.save_safetensors("/tmp/aprender_unfitted.safetensors");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("unfitted"));
}

#[test]
fn test_save_load_safetensors_roundtrip() {
    let (x, y) = simple_dataset();
    let mut model = DecisionTreeClassifier::new().with_max_depth(3);
    model.fit(&x, &y).expect("fit should succeed");

    let dir = std::env::temp_dir().join("aprender_test_tree_safetensors");
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join("tree_model.safetensors");

    model.save_safetensors(&path).expect("save should succeed");
    let loaded = DecisionTreeClassifier::load_safetensors(&path).expect("load should succeed");

    let orig_preds = model.predict(&x);
    let loaded_preds = loaded.predict(&x);
    assert_eq!(orig_preds, loaded_preds);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_save_safetensors_no_max_depth() {
    let (x, y) = simple_dataset();
    let mut model = DecisionTreeClassifier::new(); // no max_depth
    model.fit(&x, &y).expect("fit should succeed");

    let dir = std::env::temp_dir().join("aprender_test_tree_no_depth");
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join("tree_model_nodepth.safetensors");

    model.save_safetensors(&path).expect("save should succeed");
    let loaded = DecisionTreeClassifier::load_safetensors(&path).expect("load should succeed");
    assert!(loaded.max_depth.is_none());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_load_safetensors_nonexistent() {
    let result = DecisionTreeClassifier::load_safetensors("/tmp/aprender_nonexistent.safetensors");
    assert!(result.is_err());
}

#[test]
fn test_debug_clone() {
    let model = DecisionTreeClassifier::new().with_max_depth(2);
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("DecisionTreeClassifier"));

    let cloned = model.clone();
    assert_eq!(cloned.max_depth, Some(2));
}
