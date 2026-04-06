// Property-based tests for Tensor
// Uses proptest to verify mathematical properties hold for all inputs

use proptest::prelude::*;
use realizar::Tensor;

// Strategy to generate valid shapes (1-3 dimensions, max 100 per dimension)
fn shape_strategy() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..=100, 1..=3)
}

// Strategy to generate tensors with random shapes and data
fn tensor_strategy() -> impl Strategy<Value = Tensor<f32>> {
    shape_strategy().prop_flat_map(|shape| {
        let size: usize = shape.iter().product();
        prop::collection::vec(-1000.0f32..1000.0, size..=size).prop_map(move |data| {
            Tensor::from_vec(shape.clone(), data).expect("Valid tensor from strategy")
        })
    })
}

proptest! {
    #[test]
    fn test_tensor_creation_preserves_size(shape in shape_strategy()) {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(shape, data).expect("test");
        prop_assert_eq!(tensor.size(), size);
    }

    #[test]
    fn test_tensor_shape_matches(shape in shape_strategy()) {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = vec![0.0; size];
        let tensor = Tensor::from_vec(shape.clone(), data).expect("test");
        prop_assert_eq!(tensor.shape(), &shape[..]);
    }

    #[test]
    fn test_tensor_ndim_matches_shape_len(tensor in tensor_strategy()) {
        prop_assert_eq!(tensor.ndim(), tensor.shape().len());
    }

    #[test]
    fn test_tensor_data_length_matches_size(tensor in tensor_strategy()) {
        prop_assert_eq!(tensor.data().len(), tensor.size());
    }

    #[test]
    fn test_tensor_clone_equality(tensor in tensor_strategy()) {
        let cloned = tensor.clone();
        prop_assert_eq!(tensor.shape(), cloned.shape());
        prop_assert_eq!(tensor.data(), cloned.data());
    }
}
