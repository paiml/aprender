//! CORRECTNESS-013: Manual attention verification
//!
//! Computes attention manually using the same Q, K, V values to verify
//! what the correct output should be.
#![allow(unused_variables, clippy::needless_range_loop)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Values from debug output at position 1, layer 0
    // Q head 0 first elements (full 128 dims not shown, but we'll use what we have)
    let q_first5 = [
        0.22444972f32,
        0.1500241,
        0.16961512,
        -0.11974904,
        0.16413349,
    ];

    // K cache position 0 first elements
    let k0_first5 = [7.8360004f32, 3.5289938, -0.35749412, -0.7482981, 1.6254648];

    // K cache position 1 first elements
    let k1_first5 = [2.8995457f32, 1.7079767, 0.58705133, -0.53181976, 1.6363757];

    // V cache position 0 first elements
    let v0_first5 = [
        -0.029915452f32,
        -0.07104013,
        -0.39225078,
        0.18054092,
        -0.7353064,
    ];

    // Compute partial dot products (just first 5 elements)
    let partial_score0: f32 = q_first5.iter().zip(&k0_first5).map(|(q, k)| q * k).sum();
    let partial_score1: f32 = q_first5.iter().zip(&k1_first5).map(|(q, k)| q * k).sum();

    println!("Partial dot products (first 5 elements only):");
    println!("  Q·K0 = {:.6}", partial_score0);
    println!("  Q·K1 = {:.6}", partial_score1);

    // The full dot product (128 elements) determines softmax weights
    // We can't compute full scores, but let's see the partial contribution
    let scale = 1.0 / (128.0f32).sqrt(); // 1/11.31 ≈ 0.0884
    println!("\nScale factor: {:.6}", scale);

    // Scaled partial scores
    let scaled_partial0 = partial_score0 * scale;
    let scaled_partial1 = partial_score1 * scale;
    println!("Scaled partial scores:");
    println!("  score0 (partial) = {:.6}", scaled_partial0);
    println!("  score1 (partial) = {:.6}", scaled_partial1);

    // If we assume the full scores are proportional to these partials,
    // the softmax weights would be approximately:
    let exp0 = scaled_partial0.exp();
    let exp1 = scaled_partial1.exp();
    let sum_exp = exp0 + exp1;
    let weight0 = exp0 / sum_exp;
    let weight1 = exp1 / sum_exp;

    println!("\nApproximate softmax weights (based on partial scores):");
    println!("  weight0 = {:.6}", weight0);
    println!("  weight1 = {:.6}", weight1);

    // Compute expected output (weighted sum of V values)
    println!("\nExpected output (first 5 elements, approximate):");
    for i in 0..5 {
        let expected = weight0 * v0_first5[i] + weight1 * 0.0; // V1 unknown
        println!(
            "  out[{}] ≈ {:.6} * {:.6} + {:.6} * V1[{}]",
            i, weight0, v0_first5[i], weight1, i
        );
    }

    // What we observe:
    println!("\nObserved outputs:");
    println!("  CPU Head 0: [-0.05996498, 0.011939701, -0.024795583, 0.19075075, -0.0015141041]");
    println!("  GPU Head 0: [-0.08987898, -0.059096985, -0.4170272, 0.37128282, -0.73678464]");
    println!("  V cache[0]: {:?}", v0_first5);

    // GPU output[4] = -0.737, V0[4] = -0.735 → weight0 ≈ 1.0 on GPU!
    // CPU output[4] = -0.0015 → weight0 much smaller on CPU

    // Let me compute what weights would give the CPU output
    // Assuming V1[4] ≈ some value, we need to solve:
    // weight0 * V0[4] + weight1 * V1[4] = CPU_out[4]
    // weight0 * (-0.735) + weight1 * V1[4] = -0.0015

    println!("\n=== Analysis ===");
    println!("GPU output[4] = -0.737 ≈ V0[4] = -0.735 → GPU softmax weight0 ≈ 1.0");
    println!("CPU output[4] = -0.0015 → CPU softmax weight0 much smaller");
    println!();
    println!("This suggests GPU is computing score0 >> score1, while CPU computes score0 ≈ score1");
    println!("The bug might be in how the GPU computes or reads K values.");

    // Let me compute what scores would give weight0 ≈ 1.0
    // weight0 = exp(score0) / (exp(score0) + exp(score1))
    // For weight0 = 0.99, we need exp(score0) = 99 * exp(score1)
    // So score0 - score1 = ln(99) ≈ 4.6

    println!(
        "\nFor weight0 = 0.99: score0 - score1 ≈ {:.2}",
        (99.0f32).ln()
    );
    println!(
        "For weight0 = 0.9999: score0 - score1 ≈ {:.2}",
        (9999.0f32).ln()
    );

    Ok(())
}
