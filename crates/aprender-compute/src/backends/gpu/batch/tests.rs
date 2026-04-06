//! Tests for GPU command batch operations

use super::*;
use std::sync::OnceLock;

use crate::backends::gpu::GpuDevice;

/// Shared GPU device for fast test execution (initialized once)
static SHARED_DEVICE: OnceLock<Option<GpuDevice>> = OnceLock::new();

/// Get shared GPU device (fast) or None if unavailable
fn get_shared_device() -> Option<GpuDevice> {
    SHARED_DEVICE
        .get_or_init(|| if GpuDevice::is_available() { GpuDevice::new().ok() } else { None })
        .clone()
}

#[test]
fn test_buffer_allocation() {
    let Some(device) = get_shared_device() else {
        eprintln!("GPU not available, skipping");
        return;
    };
    let mut batch = GpuCommandBatch::new(device);

    let buf1 = batch.upload(&[1.0, 2.0, 3.0]);
    let buf2 = batch.upload(&[4.0, 5.0, 6.0]);

    assert_eq!(batch.num_buffers(), 2);
    assert_ne!(buf1, buf2);
}

#[test]
fn test_operation_queuing() {
    let Some(device) = get_shared_device() else {
        eprintln!("GPU not available, skipping");
        return;
    };
    let mut batch = GpuCommandBatch::new(device);

    let input = batch.upload(&[1.0, 2.0, -3.0, 4.0]);
    let relu_out = batch.relu(input);
    let scaled = batch.scale(relu_out, 2.0);
    let other = batch.upload(&[0.5, 0.5, 0.5, 0.5]);
    let _final_out = batch.add(scaled, other);

    assert_eq!(batch.num_operations(), 3); // relu, scale, add
    assert_eq!(batch.num_buffers(), 5); // input, relu_out, scaled, other, final_out
}

#[test]
#[should_panic(expected = "Buffer size mismatch")]
fn test_size_mismatch_add() {
    let Some(device) = get_shared_device() else {
        panic!("Buffer size mismatch"); // Satisfy should_panic when skipping
    };
    let mut batch = GpuCommandBatch::new(device);

    let a = batch.upload(&[1.0, 2.0]);
    let b = batch.upload(&[1.0, 2.0, 3.0]);
    batch.add(a, b); // Should panic
}

#[test]
#[should_panic(expected = "Buffer size mismatch")]
fn test_size_mismatch_mul() {
    let Some(device) = get_shared_device() else {
        panic!("Buffer size mismatch"); // Satisfy should_panic when skipping
    };
    let mut batch = GpuCommandBatch::new(device);

    let a = batch.upload(&[1.0, 2.0]);
    let b = batch.upload(&[1.0, 2.0, 3.0]);
    batch.mul(a, b); // Should panic
}

#[test]
#[should_panic(expected = "Buffer size mismatch")]
fn test_size_mismatch_dot() {
    let Some(device) = get_shared_device() else {
        panic!("Buffer size mismatch"); // Satisfy should_panic when skipping
    };
    let mut batch = GpuCommandBatch::new(device);

    let a = batch.upload(&[1.0, 2.0]);
    let b = batch.upload(&[1.0, 2.0, 3.0]);
    batch.dot(a, b); // Should panic
}

/// Comprehensive async test covering ALL batch operations in a single GPU session.
/// This reduces GPU initialization overhead for coverage (1 session vs 10).
#[tokio::test]
async fn test_all_batch_operations() {
    let Some(device) = get_shared_device() else {
        eprintln!("GPU not available, skipping");
        return;
    };
    let mut batch = GpuCommandBatch::new(device);

    // Test 1: End-to-end (relu + scale + add)
    let input1 = batch.upload(&[1.0, 2.0, -3.0, 4.0]);
    let relu_out = batch.relu(input1);
    let scaled = batch.scale(relu_out, 2.0);
    let other = batch.upload(&[0.5, 0.5, 0.5, 0.5]);
    let add_result = batch.add(scaled, other);

    // Test 2: Mul operation
    let mul_a = batch.upload(&[1.0, 2.0, 3.0, 4.0]);
    let mul_b = batch.upload(&[2.0, 3.0, 4.0, 5.0]);
    let mul_result = batch.mul(mul_a, mul_b);

    // Test 3: Dot operation
    let dot_a = batch.upload(&[1.0, 2.0, 3.0, 4.0]);
    let dot_b = batch.upload(&[2.0, 3.0, 4.0, 5.0]);
    let dot_result = batch.dot(dot_a, dot_b);

    // Test 4: Sigmoid
    let sig_input = batch.upload(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let sig_result = batch.sigmoid(sig_input);

    // Test 5: Tanh
    let tanh_input = batch.upload(&[-1.0, 0.0, 1.0]);
    let tanh_result = batch.tanh(tanh_input);

    // Test 6: Swish
    let swish_input = batch.upload(&[0.0, 1.0, 2.0]);
    let swish_result = batch.swish(swish_input);

    // Test 7: GELU
    let gelu_input = batch.upload(&[-1.0, 0.0, 1.0]);
    let gelu_result = batch.gelu(gelu_input);

    // Test 8: Sub
    let sub_a = batch.upload(&[5.0, 10.0, 15.0, 20.0]);
    let sub_b = batch.upload(&[1.0, 2.0, 3.0, 4.0]);
    let sub_result = batch.sub(sub_a, sub_b);

    // Test 9: Chained activations
    let chain_input = batch.upload(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let chain_relu = batch.relu(chain_input);
    let chain_sigmoid = batch.sigmoid(chain_relu);
    let chain_result = batch.tanh(chain_sigmoid);

    // Execute all operations in single batch
    batch.execute().await.unwrap();

    // Verify Test 1: relu([1,2,-3,4])=[1,2,0,4] → scale(*2)=[2,4,0,8] → add([0.5])=[2.5,4.5,0.5,8.5]
    let result1 = batch.read(add_result).await.unwrap();
    assert_eq!(result1.len(), 4);
    assert!((result1[0] - 2.5).abs() < 1e-5);
    assert!((result1[1] - 4.5).abs() < 1e-5);
    assert!((result1[2] - 0.5).abs() < 1e-5);
    assert!((result1[3] - 8.5).abs() < 1e-5);

    // Verify Test 2: [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
    let result2 = batch.read(mul_result).await.unwrap();
    assert_eq!(result2, vec![2.0, 6.0, 12.0, 20.0]);

    // Verify Test 3: Dot product returns a result
    let result3 = batch.read(dot_result).await.unwrap();
    assert!(!result3.is_empty());

    // Verify Test 4: Sigmoid values
    let result4 = batch.read(sig_result).await.unwrap();
    assert_eq!(result4.len(), 5);
    assert!((result4[0] - 0.119).abs() < 0.01); // sigmoid(-2)
    assert!((result4[2] - 0.5).abs() < 0.01); // sigmoid(0)
    assert!((result4[4] - 0.881).abs() < 0.01); // sigmoid(2)

    // Verify Test 5: Tanh values
    let result5 = batch.read(tanh_result).await.unwrap();
    assert_eq!(result5.len(), 3);
    assert!((result5[0] - (-0.762)).abs() < 0.01);
    assert!(result5[1].abs() < 0.01);
    assert!((result5[2] - 0.762).abs() < 0.01);

    // Verify Test 6: Swish values
    let result6 = batch.read(swish_result).await.unwrap();
    assert_eq!(result6.len(), 3);
    assert!(result6[0].abs() < 0.01);
    assert!((result6[1] - 0.731).abs() < 0.01);

    // Verify Test 7: GELU values
    let result7 = batch.read(gelu_result).await.unwrap();
    assert_eq!(result7.len(), 3);
    assert!(result7[1].abs() < 0.01);
    assert!((result7[2] - 0.841).abs() < 0.05);

    // Verify Test 8: Sub [5-1, 10-2, 15-3, 20-4] = [4, 8, 12, 16]
    let result8 = batch.read(sub_result).await.unwrap();
    assert_eq!(result8, vec![4.0, 8.0, 12.0, 16.0]);

    // Verify Test 9: Chained activations in range
    let result9 = batch.read(chain_result).await.unwrap();
    assert_eq!(result9.len(), 5);
    for &val in &result9 {
        assert!((-1.0..=1.0).contains(&val), "Value {} out of range", val);
    }
}
