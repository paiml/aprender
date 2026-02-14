use super::*;

#[test]
fn test_linear_forward_shape() {
    let layer = Linear::new(10, 5);
    let x = Tensor::ones(&[32, 10]);
    let output = layer.forward(&x);

    assert_eq!(output.shape(), &[32, 5]);
}

#[test]
fn test_linear_parameters() {
    let layer = Linear::new(10, 5);
    let params = layer.parameters();

    assert_eq!(params.len(), 2); // weight + bias
    assert_eq!(params[0].shape(), &[5, 10]); // weight
    assert_eq!(params[1].shape(), &[5]); // bias
}

#[test]
fn test_linear_without_bias() {
    let layer = Linear::without_bias(10, 5);
    let params = layer.parameters();

    assert_eq!(params.len(), 1); // weight only
    assert!(!layer.has_bias());
}

#[test]
fn test_linear_num_parameters() {
    let layer = Linear::new(10, 5);
    // weight: 10*5 = 50, bias: 5, total: 55
    assert_eq!(layer.num_parameters(), 55);
}

#[test]
fn test_linear_reproducible() {
    let layer1 = Linear::with_seed(10, 5, Some(42));
    let layer2 = Linear::with_seed(10, 5, Some(42));

    assert_eq!(layer1.weight.data(), layer2.weight.data());
}

#[test]
fn test_linear_identity_like() {
    // Create a layer with known weights to verify computation
    let mut layer = Linear::with_seed(3, 3, Some(42));

    // Set weight to identity, bias to zero (use set_weight to update cached transpose)
    let identity = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
    let zero_bias = Tensor::zeros(&[3]);

    layer.set_weight(identity.requires_grad());
    layer.set_bias(zero_bias.requires_grad());

    let x = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let output = layer.forward(&x);

    // With identity weight and zero bias, output should equal input
    assert_eq!(output.shape(), &[1, 3]);

    let out_data = output.data();
    assert!((out_data[0] - 1.0).abs() < 1e-5);
    assert!((out_data[1] - 2.0).abs() < 1e-5);
    assert!((out_data[2] - 3.0).abs() < 1e-5);
}

#[test]
fn test_linear_with_bias() {
    let mut layer = Linear::with_seed(2, 2, Some(42));

    // Set known weights (use set_weight to update cached transpose)
    layer.set_weight(Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).requires_grad());
    layer.set_bias(Tensor::new(&[10.0, 20.0], &[2]).requires_grad());

    let x = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let output = layer.forward(&x);

    // y = x @ W^T + b = [1, 2] @ [[1,0],[0,1]] + [10, 20] = [1, 2] + [10, 20] = [11, 22]
    let out_data = output.data();
    assert!((out_data[0] - 11.0).abs() < 1e-5);
    assert!((out_data[1] - 22.0).abs() < 1e-5);
}

// =========================================================================
// Property tests: weight_t cache invariant
// =========================================================================

#[test]
fn test_placeholder_is_not_ready() {
    // PROPERTY: Linear::placeholder() always creates a layer that is_ready() == false
    let layer = Linear::placeholder(64, 128);
    assert!(!layer.is_ready(), "Placeholder must not be ready");
}

#[test]
fn test_new_is_ready() {
    // PROPERTY: Linear::new() always creates a layer that is_ready() == true
    let layer = Linear::new(64, 128);
    assert!(layer.is_ready(), "Linear::new() must be ready");
}

#[test]
fn test_set_weight_makes_ready() {
    // PROPERTY: For any placeholder, set_weight() makes is_ready() == true
    let mut layer = Linear::placeholder(32, 64);
    assert!(!layer.is_ready(), "Precondition");

    let weight = Tensor::ones(&[64, 32]);
    layer.set_weight(weight);

    assert!(layer.is_ready(), "set_weight must make layer ready");
}

#[test]
fn test_is_ready_implies_forward_succeeds() {
    // PROPERTY: If is_ready() == true, forward() does not panic
    let layer = Linear::new(8, 4);
    assert!(layer.is_ready());

    let x = Tensor::ones(&[2, 8]);
    let output = layer.forward(&x); // Should not panic
    assert_eq!(output.shape(), &[2, 4]);
}

#[test]
#[should_panic(expected = "weight_t")]
fn test_not_ready_forward_panics() {
    // PROPERTY: If is_ready() == false, forward() panics
    let layer = Linear::placeholder(8, 4);
    assert!(!layer.is_ready());

    let x = Tensor::ones(&[2, 8]);
    let _ = layer.forward(&x); // Should panic
}

// =========================================================================
// Coverage: Debug impl
// =========================================================================

#[test]
fn test_linear_debug_with_bias() {
    let layer = Linear::new(10, 5);
    let debug_str = format!("{:?}", layer);
    assert!(debug_str.contains("Linear"));
    assert!(debug_str.contains("in_features"));
    assert!(debug_str.contains("out_features"));
    assert!(debug_str.contains("bias"));
    assert!(debug_str.contains("10"));
    assert!(debug_str.contains("5"));
    assert!(debug_str.contains("true"));
}

#[test]
fn test_linear_debug_without_bias() {
    let layer = Linear::without_bias(8, 4);
    let debug_str = format!("{:?}", layer);
    assert!(debug_str.contains("false"));
}

// =========================================================================
// Coverage: N-dimensional forward (ndim > 2 branch)
// =========================================================================

#[test]
fn test_linear_forward_3d_input() {
    // Tests the ndim > 2 branch in forward()
    let layer = Linear::with_seed(4, 3, Some(42));
    // 3D input: [batch=2, seq_len=3, features=4]
    let x = Tensor::ones(&[2, 3, 4]);
    let output = layer.forward(&x);
    // Output should be [2, 3, 3]
    assert_eq!(output.shape(), &[2, 3, 3]);
}

#[test]
fn test_linear_forward_4d_input() {
    // Tests the ndim > 2 branch with 4D input
    let layer = Linear::with_seed(2, 3, Some(42));
    let x = Tensor::ones(&[2, 2, 2, 2]);
    let output = layer.forward(&x);
    assert_eq!(output.shape(), &[2, 2, 2, 3]);
}

// =========================================================================
// Coverage: refresh_caches
// =========================================================================

#[test]
fn test_linear_refresh_caches() {
    let mut layer = Linear::new(4, 3);
    // Modify weight via parameters_mut, then refresh
    layer.refresh_caches();
    assert!(layer.is_ready());
    // Verify forward still works after refresh
    let x = Tensor::ones(&[1, 4]);
    let output = layer.forward(&x);
    assert_eq!(output.shape(), &[1, 3]);
}

// =========================================================================
// Coverage: parameters_mut for both bias/no-bias paths
// =========================================================================

#[test]
fn test_linear_parameters_mut_with_bias() {
    let mut layer = Linear::new(4, 3);
    let params = layer.parameters_mut();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_linear_parameters_mut_without_bias() {
    let mut layer = Linear::without_bias(4, 3);
    let params = layer.parameters_mut();
    assert_eq!(params.len(), 1);
}

// =========================================================================
// Coverage: without_bias forward (no bias branch in forward)
// =========================================================================

#[test]
fn test_linear_forward_without_bias_computation() {
    let mut layer = Linear::without_bias_with_seed(2, 2, Some(42));
    // Set known weights to verify no bias is added
    layer.set_weight(Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).requires_grad());
    let x = Tensor::new(&[3.0, 7.0], &[1, 2]);
    let output = layer.forward(&x);
    let out_data = output.data();
    // Without bias: y = x @ W^T = [3, 7] @ I = [3, 7]
    assert!((out_data[0] - 3.0).abs() < 1e-5);
    assert!((out_data[1] - 7.0).abs() < 1e-5);
}

// =========================================================================
// Coverage: accessor methods on placeholder
// =========================================================================

#[test]
fn test_placeholder_accessors() {
    let layer = Linear::placeholder(16, 8);
    assert_eq!(layer.in_features(), 16);
    assert_eq!(layer.out_features(), 8);
    assert!(!layer.has_bias());
    assert!(layer.bias().is_none());
    assert_eq!(layer.weight().shape(), &[1]); // placeholder uses 1-element tensor
}
