// Property-based tests for Layers
use proptest::prelude::*;
use realizar::{layers::softmax, Tensor};

const EPSILON: f32 = 1e-5;

proptest! {
    #[test]
    fn test_softmax_sums_to_one(data in prop::collection::vec(-10.0f32..10.0, 2..=100)) {
        let tensor = Tensor::from_vec(vec![data.len()], data).expect("test");
        let result = softmax(&tensor).expect("test");
        let sum: f32 = result.data().iter().sum();
        prop_assert!((sum - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_softmax_output_range(data in prop::collection::vec(-10.0f32..10.0, 2..=100)) {
        let tensor = Tensor::from_vec(vec![data.len()], data).expect("test");
        let result = softmax(&tensor).expect("test");
        for &val in result.data() {
            prop_assert!(val >= 0.0);
            prop_assert!(val <= 1.0);
        }
    }

    #[test]
    fn test_softmax_preserves_shape(shape in prop::collection::vec(2usize..=20, 1..=3)) {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let tensor = Tensor::from_vec(shape.clone(), data).expect("test");
        let result = softmax(&tensor).expect("test");
        prop_assert_eq!(result.shape(), &shape[..]);
    }
}
