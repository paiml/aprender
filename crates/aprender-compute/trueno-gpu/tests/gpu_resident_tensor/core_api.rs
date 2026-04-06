//! PHASE 1: GpuResidentTensor Core API

#[cfg(feature = "cuda")]
use trueno_gpu::driver::CudaContext;
#[cfg(feature = "cuda")]
use trueno_gpu::memory::resident::{
    reset_transfer_counters, total_d2h_transfers, total_h2d_transfers, GpuResidentTensor,
};

/// Test: Tensor created on GPU should stay on GPU without host copy
///
/// Requirement: Creating a tensor should NOT require a host copy to exist.
/// The tensor data lives exclusively on the device.
#[test]
#[cfg(feature = "cuda")]
fn test_gpu_tensor_created_on_device() {
    // Skip if CUDA not available (e.g., in CI without GPU)
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return, // Skip test
    };

    reset_transfer_counters();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];

    // Create tensor - data uploaded ONCE
    let tensor = GpuResidentTensor::from_host(&ctx, &data).expect("Upload failed");

    // Tensor should be on device
    assert!(tensor.is_device_resident());
    // Only 1 transfer (the initial upload)
    assert_eq!(tensor.host_to_device_transfers(), 1);
    assert_eq!(tensor.device_to_host_transfers(), 0);

    // Global counters also track
    assert_eq!(total_h2d_transfers(), 1);
    assert_eq!(total_d2h_transfers(), 0);
}

#[test]
#[cfg(not(feature = "cuda"))]
fn test_gpu_tensor_created_on_device() {
    // Skip test when CUDA is not available
}

/// Test: to_host() triggers exactly one D2H transfer
///
/// Requirement: Only explicit `.to_host()` should trigger device→host transfer.
#[test]
#[cfg(feature = "cuda")]
fn test_gpu_to_host_transfers() {
    // Skip if CUDA not available (e.g., in CI without GPU)
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return, // Skip test
    };

    reset_transfer_counters();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];

    // Upload data
    let mut tensor = GpuResidentTensor::from_host(&ctx, &data).expect("Upload failed");
    assert_eq!(tensor.device_to_host_transfers(), 0);

    // Download to host
    let result = tensor.to_host().expect("Download failed");
    assert_eq!(result, data);

    // Now we have 1 D2H transfer
    assert_eq!(tensor.device_to_host_transfers(), 1);
    assert_eq!(total_d2h_transfers(), 1);
}

#[test]
#[cfg(not(feature = "cuda"))]
fn test_gpu_to_host_transfers() {}

/// Test: Operations on GPU tensors should NOT transfer back to host
///
/// Requirement: matmul, softmax, etc. should keep results on GPU.
/// Only final `.to_host()` should trigger device→host transfer.
#[test]
#[cfg(feature = "cuda")]
fn test_gpu_operations_stay_on_device() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return, // Skip if no CUDA
    };

    reset_transfer_counters();

    // Create two tensors (2 H2D transfers)
    let a = GpuResidentTensor::from_host(&ctx, &vec![1.0f32; 64]).expect("Upload A");
    let b = GpuResidentTensor::from_host(&ctx, &vec![2.0f32; 64]).expect("Upload B");

    assert_eq!(total_h2d_transfers(), 2);
    assert_eq!(total_d2h_transfers(), 0);

    // Elementwise add - result stays on GPU (NO new transfers!)
    let c = a.add(&ctx, &b).expect("Add failed");

    // Check: no additional transfers occurred
    assert!(c.is_device_resident());
    assert_eq!(c.host_to_device_transfers(), 0); // Result never came from host
    assert_eq!(c.device_to_host_transfers(), 0); // Result never went to host

    // Global counters unchanged (still 2 H2D, 0 D2H)
    assert_eq!(total_h2d_transfers(), 2);
    assert_eq!(total_d2h_transfers(), 0);
}

#[test]
#[cfg(not(feature = "cuda"))]
fn test_gpu_operations_stay_on_device() {}

/// Test: Chain of operations should have ZERO intermediate transfers
///
/// Requirement: A pipeline like Q @ K^T → softmax → @ V should have:
/// - Initial upload of Q, K, V (3 transfers)
/// - Final download of output (1 transfer)
/// - ZERO intermediate transfers
#[test]
#[ignore = "TDD: Implementation pending - operation chaining not yet implemented"]
fn test_operation_chain_no_intermediate_transfers() {
    // When implemented:
    //
    // let ctx = CudaContext::new(0).expect("test");
    //
    // // Upload Q, K, V (3 H2D transfers)
    // let q = GpuResidentTensor::from_host(&ctx, &q_data).expect("test");
    // let k = GpuResidentTensor::from_host(&ctx, &k_data).expect("test");
    // let v = GpuResidentTensor::from_host(&ctx, &v_data).expect("test");
    //
    // // Chain: scores = Q @ K^T (stays on GPU)
    // let scores = q.matmul_transposed(&k).expect("test");
    //
    // // Chain: attn = softmax(scores) (stays on GPU)
    // let attn = scores.softmax(-1).expect("test");
    //
    // // Chain: output = attn @ V (stays on GPU)
    // let output = attn.matmul(&v).expect("test");
    //
    // // Verify NO intermediate transfers
    // assert_eq!(scores.device_to_host_transfers(), 0);
    // assert_eq!(attn.device_to_host_transfers(), 0);
    // assert_eq!(output.device_to_host_transfers(), 0);
    //
    // // Only transfer on explicit request
    // let result = output.to_host().expect("test");
    // // Now we have 1 D2H transfer
    // assert_eq!(output.device_to_host_transfers(), 1);

    // TODO: "TDD: Operation chaining not implemented"
}
