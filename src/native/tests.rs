pub(crate) use super::super::*;

#[test]
fn test_aligned_vec_creation() {
    let vec = AlignedVec::<f32>::with_capacity(10);
    assert_eq!(vec.len(), 0);
    assert!(vec.capacity() >= 10);
}

#[test]
fn test_aligned_vec_from_slice() {
    let vec = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
    assert_eq!(vec.len(), 4);
    assert_eq!(vec.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_aligned_vec_zeros() {
    let vec = AlignedVec::<f32>::zeros(100);
    assert_eq!(vec.len(), 100);
    assert!(vec.as_slice().iter().all(|&x| x == 0.0));
}

#[test]
fn test_aligned_vec_push() {
    let mut vec = AlignedVec::<f32>::with_capacity(2);
    vec.push(1.0);
    vec.push(2.0);
    vec.push(3.0); // triggers reallocation

    assert_eq!(vec.len(), 3);
    assert_eq!(vec.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_aligned_vec_index() {
    let vec = AlignedVec::from_slice(&[10.0_f32, 20.0, 30.0]);
    assert_eq!(vec[0], 10.0);
    assert_eq!(vec[1], 20.0);
    assert_eq!(vec[2], 30.0);
}

#[test]
fn test_aligned_vec_get() {
    let vec = AlignedVec::from_slice(&[1.0_f32, 2.0]);
    assert_eq!(vec.get(0), Some(&1.0));
    assert_eq!(vec.get(1), Some(&2.0));
    assert_eq!(vec.get(2), None);
}

#[test]
fn test_aligned_vec_set() {
    let mut vec = AlignedVec::from_slice(&[1.0_f32, 2.0]);
    assert!(vec.set(0, 10.0));
    assert_eq!(vec[0], 10.0);
    assert!(!vec.set(5, 50.0)); // out of bounds
}

#[test]
fn test_aligned_vec_clear() {
    let mut vec = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]);
    vec.clear();
    assert!(vec.is_empty());
    assert!(vec.capacity() >= 3);
}

#[test]
fn test_aligned_vec_from_iterator() {
    let vec: AlignedVec<f32> = (0..5).map(|i| i as f32).collect();
    assert_eq!(vec.len(), 5);
    assert_eq!(vec.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_aligned_vec_eq() {
    let a = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]);
    let b = AlignedVec::from_slice(&[1.0, 2.0, 3.0]);
    let c = AlignedVec::from_slice(&[1.0, 2.0, 4.0]);

    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_trueno_native_model_creation() {
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 10, 10, 1);

    assert_eq!(model.n_params, 10);
    assert_eq!(model.n_features, 10);
    assert_eq!(model.n_outputs, 1);
}

#[test]
fn test_trueno_native_model_with_params() {
    let params = AlignedVec::from_slice(&[0.5_f32, -0.3, 0.8]);
    let bias = AlignedVec::from_slice(&[1.0_f32]);

    let model = TruenoNativeModel::new(ModelType::LinearRegression, 3, 3, 1)
        .with_params(params)
        .with_bias(bias);

    assert!(model.params.is_some());
    assert!(model.bias.is_some());
    assert!(model.is_aligned());
}

#[test]
fn test_trueno_native_model_validate() {
    let params = AlignedVec::from_slice(&[0.5_f32, -0.3, 0.8]);
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 3, 3, 1).with_params(params);

    assert!(model.validate().is_ok());
}

#[test]
fn test_trueno_native_model_validate_param_mismatch() {
    let params = AlignedVec::from_slice(&[0.5_f32, -0.3]); // only 2
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 3, 3, 1) // declared 3
        .with_params(params);

    assert!(matches!(
        model.validate(),
        Err(NativeModelError::ParamCountMismatch { .. })
    ));
}

#[test]
fn test_trueno_native_model_validate_nan() {
    let params = AlignedVec::from_slice(&[0.5_f32, f32::NAN, 0.8]);
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 3, 3, 1).with_params(params);

    assert!(matches!(
        model.validate(),
        Err(NativeModelError::InvalidParameter { index: 1, .. })
    ));
}

#[test]
fn test_trueno_native_model_predict_linear() {
    let params = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]);
    let bias = AlignedVec::from_slice(&[1.0_f32]);

    let model = TruenoNativeModel::new(ModelType::LinearRegression, 3, 3, 1)
        .with_params(params)
        .with_bias(bias);

    // 1*1 + 2*2 + 3*3 + 1 = 1 + 4 + 9 + 1 = 15
    let pred = model.predict_linear(&[1.0, 2.0, 3.0]).unwrap();
    assert!((pred - 15.0).abs() < f32::EPSILON);
}

#[test]
fn test_trueno_native_model_predict_linear_feature_mismatch() {
    let params = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]);
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 3, 3, 1).with_params(params);

    let result = model.predict_linear(&[1.0, 2.0]); // only 2 features
    assert!(matches!(
        result,
        Err(NativeModelError::FeatureMismatch {
            expected: 3,
            got: 2
        })
    ));
}

#[test]
fn test_trueno_native_model_predict_linear_missing_params() {
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 3, 3, 1);

    let result = model.predict_linear(&[1.0, 2.0, 3.0]);
    assert!(matches!(result, Err(NativeModelError::MissingParams)));
}

#[test]
fn test_model_extra() {
    let extra = ModelExtra::new()
        .with_centroids(AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]))
        .with_metadata("key", vec![1, 2, 3]);

    assert!(extra.centroids.is_some());
    assert_eq!(extra.metadata.get("key"), Some(&vec![1, 2, 3]));
    assert!(extra.size_bytes() > 0);
}

#[test]
fn test_tree_data() {
    let tree = TreeData {
        feature_indices: vec![0, 1],
        thresholds: vec![0.5, 0.3],
        left_children: vec![1, -1],
        right_children: vec![2, -1],
        leaf_values: vec![0.0, 1.0, 0.5],
    };

    assert_eq!(tree.n_nodes(), 2);
    assert!(tree.size_bytes() > 0);
}

#[test]
fn test_layer_data() {
    let layer = LayerData::dense(100, 50)
        .with_weights(AlignedVec::zeros(5000))
        .with_biases(AlignedVec::zeros(50));

    assert_eq!(layer.input_dim, 100);
    assert_eq!(layer.output_dim, 50);
    assert!(layer.size_bytes() > 0);
}

#[test]
fn test_native_model_error_display() {
    let err = NativeModelError::ParamCountMismatch {
        declared: 10,
        actual: 5,
    };
    let msg = format!("{err}");
    assert!(msg.contains("10"));
    assert!(msg.contains("5"));

    let err = NativeModelError::MissingParams;
    assert_eq!(format!("{err}"), "Missing model parameters");
}

#[test]
fn test_trueno_native_model_size_bytes() {
    let params = AlignedVec::from_slice(&[1.0_f32; 100]);
    let bias = AlignedVec::from_slice(&[1.0_f32; 10]);

    let model = TruenoNativeModel::new(ModelType::LinearRegression, 100, 100, 10)
        .with_params(params)
        .with_bias(bias);

    // params: 100 * 4 = 400, bias: 10 * 4 = 40
    assert_eq!(model.size_bytes(), 440);
}

#[test]
fn test_trueno_native_model_default() {
    let model = TruenoNativeModel::default();
    assert_eq!(model.n_params, 0);
    assert_eq!(model.n_features, 0);
    assert_eq!(model.n_outputs, 1);
}

#[test]
fn test_aligned_vec_default() {
    let vec = AlignedVec::<f32>::default();
    assert!(vec.is_empty());
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_aligned_vec_as_mut_ptr() {
    let mut vec = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]);
    let ptr = vec.as_mut_ptr();
    assert!(!ptr.is_null());
}

#[test]
fn test_aligned_vec_as_mut_slice() {
    let mut vec = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]);
    let slice = vec.as_mut_slice();
    slice[0] = 10.0;
    assert_eq!(vec[0], 10.0);
}

#[test]
fn test_aligned_vec_get_mut() {
    let mut vec = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]);
    if let Some(val) = vec.get_mut(1) {
        *val = 20.0;
    }
    assert_eq!(vec[1], 20.0);
    assert!(vec.get_mut(10).is_none());
}

#[test]
fn test_aligned_vec_index_mut() {
    let mut vec = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]);
    vec[0] = 100.0;
    assert_eq!(vec[0], 100.0);
}

#[test]
fn test_aligned_vec_is_aligned_empty() {
    let vec = AlignedVec::<f32>::with_capacity(0);
    assert!(vec.is_aligned());
}

#[test]
fn test_trueno_native_model_with_extra() {
    let extra = ModelExtra::new();
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 0, 0, 1).with_extra(extra);
    assert!(model.extra.is_some());
}

#[test]
fn test_trueno_native_model_params_ptr() {
    let params = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]);
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 3, 3, 1).with_params(params);

    let ptr = model.params_ptr();
    assert!(ptr.is_some());
    assert!(!ptr.unwrap().is_null());
}

#[test]
fn test_trueno_native_model_params_ptr_none() {
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 0, 0, 1);
    assert!(model.params_ptr().is_none());
}

#[test]
fn test_trueno_native_model_bias_ptr() {
    let bias = AlignedVec::from_slice(&[1.0_f32]);
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 0, 0, 1).with_bias(bias);

    let ptr = model.bias_ptr();
    assert!(ptr.is_some());
}

#[test]
fn test_trueno_native_model_bias_ptr_none() {
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 0, 0, 1);
    assert!(model.bias_ptr().is_none());
}

#[test]
fn test_trueno_native_model_validate_invalid_bias() {
    let params = AlignedVec::from_slice(&[1.0_f32, 2.0, 3.0]);
    let bias = AlignedVec::from_slice(&[f32::INFINITY]);
    let model = TruenoNativeModel::new(ModelType::LinearRegression, 3, 3, 1)
        .with_params(params)
        .with_bias(bias);

    let result = model.validate();
    assert!(matches!(
        result,
        Err(NativeModelError::InvalidBias { index: 0, .. })
    ));
}

#[test]
fn test_model_extra_with_tree() {
    let tree = TreeData::new();
    let extra = ModelExtra::new().with_tree(tree);
    assert!(extra.tree_data.is_some());
}

#[test]
fn test_model_extra_with_layers() {
    let layers = vec![LayerData::dense(10, 5)];
    let extra = ModelExtra::new().with_layers(layers);
    assert!(extra.layer_data.is_some());
}

#[test]
fn test_tree_data_new_and_default() {
    let tree1 = TreeData::new();
    let tree2 = TreeData::default();
    assert_eq!(tree1.n_nodes(), 0);
    assert_eq!(tree2.n_nodes(), 0);
}

#[test]
fn test_layer_type_all_variants() {
    let types = [
        LayerType::Dense,
        LayerType::ReLU,
        LayerType::Sigmoid,
        LayerType::Tanh,
        LayerType::Softmax,
        LayerType::Dropout,
        LayerType::BatchNorm,
    ];
    for lt in &types {
        let debug = format!("{:?}", lt);
        assert!(!debug.is_empty());
    }
}

#[test]
fn test_layer_type_eq() {
    assert_eq!(LayerType::Dense, LayerType::Dense);
    assert_ne!(LayerType::Dense, LayerType::ReLU);
}

#[test]
fn test_native_model_error_display_all_variants() {
    let errors = [
        NativeModelError::ParamCountMismatch {
            declared: 10,
            actual: 5,
        },
        NativeModelError::InvalidParameter {
            index: 0,
            value: f32::NAN,
        },
        NativeModelError::InvalidBias {
            index: 0,
            value: f32::INFINITY,
        },
        NativeModelError::FeatureMismatch {
            expected: 3,
            got: 2,
        },
        NativeModelError::MissingParams,
        NativeModelError::AlignmentError {
            ptr: 12345,
            required: 64,
        },
    ];

    for err in &errors {
        let msg = format!("{err}");
        assert!(!msg.is_empty());
    }
}

#[test]
fn test_native_model_error_debug_clone() {
    let err = NativeModelError::MissingParams;
    let cloned = err.clone();
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("MissingParams"));
}

#[test]
fn test_native_model_error_is_error() {
    let err = NativeModelError::MissingParams;
    let _: &dyn std::error::Error = &err;
}

#[test]
fn test_trueno_native_model_debug_clone() {
    let model = TruenoNativeModel::default();
    let cloned = model.clone();
    assert_eq!(model.n_params, cloned.n_params);

    let debug = format!("{:?}", model);
    assert!(debug.contains("TruenoNativeModel"));
}

#[path = "tests_part_02.rs"]
mod tests_part_02;
