use super::*;

#[test]
fn test_aligned_vec_debug_clone() {
    let vec = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]);
    let cloned = vec.clone();
    assert_eq!(vec, cloned);

    let debug = format!("{:?}", vec);
    assert!(debug.contains("AlignedVec"));
}

#[test]
fn test_model_extra_debug_clone_default() {
    let extra = ModelExtra::default();
    let cloned = extra.clone();
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("ModelExtra"));
}

#[test]
fn test_tree_data_debug_clone() {
    let tree = TreeData::default();
    let cloned = tree.clone();
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("TreeData"));
}

#[test]
fn test_layer_data_debug_clone() {
    let layer = LayerData::dense(10, 5);
    let cloned = layer.clone();
    assert_eq!(layer.input_dim, cloned.input_dim);

    let debug = format!("{:?}", layer);
    assert!(debug.contains("LayerData"));
}

#[test]
fn test_aligned_vec_push_triggers_realloc() {
    let mut vec = AlignedVec::<f32>::with_capacity(1);
    vec.push(1.0);
    vec.push(2.0);
    vec.push(3.0);
    vec.push(4.0);
    assert_eq!(vec.len(), 4);
}

#[test]
fn test_model_extra_size_bytes_all_components() {
    let tree = TreeData {
        feature_indices: vec![0, 1],
        thresholds: vec![0.5, 0.3],
        left_children: vec![1, -1],
        right_children: vec![2, -1],
        leaf_values: vec![0.0, 1.0, 0.5],
    };
    let layer = LayerData::dense(10, 5)
        .with_weights(AlignedVec::zeros(50))
        .with_biases(AlignedVec::zeros(5));
    let extra = ModelExtra::new()
        .with_tree(tree)
        .with_layers(vec![layer])
        .with_centroids(AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]))
        .with_metadata("key", vec![1, 2, 3, 4, 5]);

    assert!(extra.size_bytes() > 0);
}

#[test]
fn test_trueno_native_model_predict_linear_no_bias() {
    let params = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]);
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 3, 3, 1).with_params(params);

    // 1*1 + 2*2 + 3*3 + 0 = 14
    let pred = model.predict_linear(&[1.0, 2.0, 3.0]).unwrap();
    assert!((pred - 14.0).abs() < f32::EPSILON);
}
