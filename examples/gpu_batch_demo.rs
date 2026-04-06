#![allow(clippy::disallowed_methods)]
//! GPU Command Batching Demo
//!
//! Demonstrates the async GPU batch API that reduces CPU↔GPU transfers
//! by batching multiple operations together.
//!
//! **Performance Benefit**: 3x reduction in GPU transfers
//! - Traditional API: 3 operations = 6 transfers (3 up + 3 down)
//! - Batch API: 3 operations = 2 transfers (1 up + 1 down)

use trueno::backends::gpu::{GpuCommandBatch, GpuDevice};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), String> {
    println!("=== Trueno GPU Batch API Demo ===\n");

    // Check GPU availability
    if !GpuDevice::is_available() {
        println!("⚠️  GPU not available (wgpu backend not found)");
        println!("   This example requires a GPU with Vulkan/Metal/DX12 support");
        return Ok(());
    }

    println!("✅ GPU available - initializing device...\n");

    // Demo 1: Single operation (baseline)
    println!("📊 Demo 1: Single Operation (Baseline)");
    println!("   Operation: ReLU([1.0, 2.0, -3.0, 4.0])");
    demo_single_operation().await?;

    println!();

    // Demo 2: Batched operations (optimized)
    println!("📊 Demo 2: Batched Operations (Optimized)");
    println!("   Operations: ReLU → Scale(2.0) → Add([0.5, 0.5, 0.5, 0.5])");
    demo_batched_operations().await?;

    println!();

    // Demo 3: Complex pipeline
    println!("📊 Demo 3: Complex Pipeline");
    println!("   ML inference: (input * weights) → ReLU → scale → add bias");
    demo_ml_pipeline().await?;

    println!();

    // Performance comparison
    println!("📈 Performance Analysis");
    println!("   Traditional API (3 ops): 6 GPU transfers");
    println!("   Batch API (3 ops):       2 GPU transfers");
    println!("   Transfer reduction:      3x (66% fewer transfers)");
    println!("   Expected speedup:        1.5-2x for GPU-heavy workloads");

    Ok(())
}

/// Demo 1: Single operation using batch API
async fn demo_single_operation() -> Result<(), String> {
    let device = GpuDevice::new()?;
    let mut batch = GpuCommandBatch::new(device);

    // Upload input data
    let input = batch.upload(&[1.0, 2.0, -3.0, 4.0]);

    // Apply ReLU
    let output = batch.relu(input);

    // Execute batch
    batch.execute().await?;

    // Read result
    let result = batch.read(output).await?;

    println!("   Input:  [1.0, 2.0, -3.0, 4.0]");
    println!("   Output: {:?}", result);
    println!("   Expected: [1.0, 2.0, 0.0, 4.0] (negative clipped to 0)");
    assert_eq!(result, vec![1.0, 2.0, 0.0, 4.0]);
    println!("   ✅ Passed");

    Ok(())
}

/// Demo 2: Batched operations - the main benefit
async fn demo_batched_operations() -> Result<(), String> {
    let device = GpuDevice::new()?;
    let mut batch = GpuCommandBatch::new(device);

    // Queue multiple operations (no GPU execution yet!)
    let input = batch.upload(&[1.0, 2.0, -3.0, 4.0]);
    let relu_out = batch.relu(input);
    let scaled = batch.scale(relu_out, 2.0);
    let bias = batch.upload(&[0.5, 0.5, 0.5, 0.5]);
    let final_out = batch.add(scaled, bias);

    println!("   Queued 5 operations:");
    println!("   1. Upload input:  [1.0, 2.0, -3.0, 4.0]");
    println!("   2. ReLU:          max(0, input)");
    println!("   3. Scale:         relu * 2.0");
    println!("   4. Upload bias:   [0.5, 0.5, 0.5, 0.5]");
    println!("   5. Add:           scaled + bias");

    // Execute all operations in a single batch
    println!("\n   Executing batch...");
    batch.execute().await?;

    // Read final result
    let result = batch.read(final_out).await?;

    println!("\n   Computation breakdown:");
    println!("   ReLU([1, 2, -3, 4])     = [1, 2, 0, 4]");
    println!("   Scale([1, 2, 0, 4], 2)  = [2, 4, 0, 8]");
    println!("   Add([2, 4, 0, 8], 0.5)  = [2.5, 4.5, 0.5, 8.5]");
    println!("\n   Final result: {:?}", result);

    // Verify correctness
    let expected = [2.5, 4.5, 0.5, 8.5];
    for (i, (&actual, &expect)) in result.iter().zip(expected.iter()).enumerate() {
        if (actual - expect).abs() > 1e-5 {
            return Err(format!("Mismatch at index {}: expected {}, got {}", i, expect, actual));
        }
    }

    println!("   ✅ Passed");

    Ok(())
}

/// Demo 3: ML pipeline simulation
async fn demo_ml_pipeline() -> Result<(), String> {
    let device = GpuDevice::new()?;
    let mut batch = GpuCommandBatch::new(device);

    // Simulate neural network layer: y = ReLU(x * W + b)
    // Input: [1.0, 2.0, 3.0, 4.0]
    // Weights: [2.0, 2.0, 2.0, 2.0] (simplified - normally a matrix)
    // Bias: [1.0, 1.0, 1.0, 1.0]

    println!("   Simulating: y = ReLU(x * W + b)");

    let input = batch.upload(&[1.0, 2.0, 3.0, 4.0]);
    let weights = batch.upload(&[2.0, 2.0, 2.0, 2.0]);
    let bias = batch.upload(&[1.0, 1.0, 1.0, 1.0]);

    // Forward pass
    let weighted = batch.mul(input, weights); // x * W
    let added = batch.add(weighted, bias); // + b
    let activated = batch.relu(added); // ReLU(...)

    // Execute pipeline
    batch.execute().await?;

    // Read result
    let result = batch.read(activated).await?;

    println!("\n   Pipeline breakdown:");
    println!("   x * W:  [1, 2, 3, 4] * [2, 2, 2, 2] = [2, 4, 6, 8]");
    println!("   + b:    [2, 4, 6, 8] + [1, 1, 1, 1] = [3, 5, 7, 9]");
    println!("   ReLU:   [3, 5, 7, 9]                = [3, 5, 7, 9]");
    println!("\n   Final activation: {:?}", result);

    let expected = [3.0, 5.0, 7.0, 9.0];
    for (i, (&actual, &expect)) in result.iter().zip(expected.iter()).enumerate() {
        if (actual - expect).abs() > 1e-5 {
            return Err(format!("Mismatch at index {}: expected {}, got {}", i, expect, actual));
        }
    }

    println!("   ✅ Passed");

    Ok(())
}
