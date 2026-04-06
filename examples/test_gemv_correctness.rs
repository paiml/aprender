//! Quick correctness test for CoalescedGemv
use realizar::gpu::CudaScheduler;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let k = 16usize; // Small for debugging
    let n = 8usize;

    // Create simple test data: x = [1, 1, ...], A = all 1s
    // Expected: y[j] = sum over i of A[i,j] * x[i] = sum of 1.0 * 1.0 over k = k

    // x vector (1×k as row for matmul)
    let x: Vec<f32> = vec![1.0; k];

    // A matrix (k×n): A[i,j] = 1.0 for all
    let a: Vec<f32> = vec![1.0; k * n];

    // Expected output: y[j] = k (sum of k ones)
    let expected: Vec<f32> = vec![k as f32; n];

    let mut sched = CudaScheduler::new()?;
    let result = sched.matmul(&x, &a, 1, k, n)?;

    println!("Input x ({} elements): {:?}", k, x);
    println!("Matrix A ({}×{}): all 1.0s", k, n);
    println!("Expected y ({} elements): {:?}", n, expected);
    println!("Actual y   ({} elements): {:?}", result.len(), result);

    // Check correctness
    let mut max_error = 0.0f32;
    for (i, (e, a)) in expected.iter().zip(result.iter()).enumerate() {
        let error = (e - a).abs();
        if error > max_error {
            max_error = error;
        }
        if error > 0.001 {
            println!("ERROR at index {}: expected {}, got {}", i, e, a);
        }
    }

    if max_error < 0.001 {
        println!("\n✓ CORRECTNESS PASSED (max error: {})", max_error);
    } else {
        println!("\n✗ CORRECTNESS FAILED (max error: {})", max_error);
    }

    // Also test a larger size
    println!("\n--- Larger test: 4096×4096 ---");
    let k_large = 4096usize;
    let n_large = 4096usize;
    let x_large: Vec<f32> = (0..k_large).map(|i| (i % 10) as f32 * 0.1).collect();
    let a_large: Vec<f32> = (0..k_large * n_large)
        .map(|i| ((i % 7) as f32) * 0.01)
        .collect();

    // CPU reference
    let mut expected_large = vec![0.0f32; n_large];
    for j in 0..n_large {
        let mut sum = 0.0f32;
        for i in 0..k_large {
            sum += x_large[i] * a_large[i * n_large + j];
        }
        expected_large[j] = sum;
    }

    let result_large = sched.matmul(&x_large, &a_large, 1, k_large, n_large)?;

    let mut max_error_large = 0.0f32;
    let mut error_count = 0;
    for (i, (e, a)) in expected_large.iter().zip(result_large.iter()).enumerate() {
        let error = (e - a).abs();
        if error > max_error_large {
            max_error_large = error;
        }
        if error > 0.1 && error_count < 5 {
            println!("ERROR at index {}: expected {}, got {}", i, e, a);
            error_count += 1;
        }
    }

    println!("Max error: {}", max_error_large);
    if max_error_large < 0.1 {
        println!("✓ LARGE TEST PASSED");
    } else {
        println!("✗ LARGE TEST FAILED");
    }

    Ok(())
}
