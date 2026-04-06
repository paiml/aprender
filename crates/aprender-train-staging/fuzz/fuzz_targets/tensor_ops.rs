#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use entrenar::autograd::{Tensor, add, mul, relu, gelu, swish};

/// Fuzz target for tensor operations
///
/// Tests that tensor operations never panic with arbitrary inputs.
/// Validates numerical stability and error handling.

#[derive(Arbitrary, Debug)]
struct TensorOpFuzzInput {
    size: u8,            // Tensor size (1..255)
    values_a: Vec<u8>,   // Raw bytes for tensor A
    values_b: Vec<u8>,   // Raw bytes for tensor B
    op_type: u8,         // Operation selector
}

fn bytes_to_f32(bytes: &[u8], size: usize) -> Vec<f32> {
    // Convert bytes to f32 values in a reasonable range
    bytes.iter()
        .take(size)
        .map(|&b| {
            // Map 0..255 to -10.0..10.0 range
            ((b as f32) / 255.0) * 20.0 - 10.0
        })
        .collect()
}

fuzz_target!(|input: TensorOpFuzzInput| {
    let size = (input.size as usize).max(1).min(64); // Limit size for performance

    // Create tensors from fuzzed input
    let values_a = bytes_to_f32(&input.values_a, size);
    let values_b = bytes_to_f32(&input.values_b, size);

    if values_a.len() < size || values_b.len() < size {
        return; // Skip if we don't have enough data
    }

    let a = Tensor::from_vec(values_a.clone(), true);
    let b = Tensor::from_vec(values_b.clone(), false);

    // Invariant 1: Element-wise operations should never panic
    match input.op_type % 8 {
        0 => {
            // Add
            let _c = add(&a, &b);
        }
        1 => {
            // Mul
            let _c = mul(&a, &b);
        }
        2 => {
            // ReLU
            let _c = relu(&a);
        }
        3 => {
            // GELU
            let _c = gelu(&a);
        }
        4 => {
            // Swish
            let _c = swish(&a);
        }
        5 => {
            // Chain: ReLU + Add
            let relu_a = relu(&a);
            let _c = add(&relu_a, &b);
        }
        6 => {
            // Chain: GELU + Mul
            let gelu_a = gelu(&a);
            let _c = mul(&gelu_a, &b);
        }
        7 => {
            // Chain: Swish + Add + Mul
            let swish_a = swish(&a);
            let add_result = add(&swish_a, &b);
            let _c = mul(&add_result, &a);
        }
        _ => unreachable!(),
    }

    // Invariant 2: Tensor creation should validate sizes
    let zeros = Tensor::zeros(size, false);
    let ones = Tensor::ones(size, false);

    assert_eq!(zeros.len(), size);
    assert_eq!(ones.len(), size);

    // Invariant 3: Operations on tensors with special values should not panic
    // Test with zeros
    let _zero_add = add(&zeros, &a);
    let _zero_mul = mul(&zeros, &a);

    // Test with ones
    let _one_add = add(&ones, &a);
    let _one_mul = mul(&ones, &a);

    // Invariant 4: Activations should handle extreme values
    let extreme_values = vec![-1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];
    let extreme_values = extreme_values.into_iter().take(size).collect();
    let extreme = Tensor::from_vec(extreme_values, false);

    let _relu_extreme = relu(&extreme);
    let _gelu_extreme = gelu(&extreme);
    let _swish_extreme = swish(&extreme);

    // Invariant 5: NaN and Inf handling
    // (We don't explicitly create NaNs, but operations shouldn't panic if they occur)
    let large_values: Vec<f32> = (0..size).map(|i| (i as f32) * 1000.0).collect();
    let large = Tensor::from_vec(large_values, false);

    let _gelu_large = gelu(&large);
    let _swish_large = swish(&large);
});
