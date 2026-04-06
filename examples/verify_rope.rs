//! Verify RoPE computation for Qwen2 (NEOX style, rope_type=2)
//!
//! RoPE transforms Q and K to encode positional information.
//! - NORM style (type 0): pairs are (x[0], x[1]), (x[2], x[3]), ...
//! - NEOX style (type 2): pairs are (x[0], x[half]), (x[1], x[half+1]), ...

fn main() {
    println!("=== RoPE Verification ===\n");

    // Qwen2 config
    let hidden_dim = 896;
    let num_heads = 14;
    let head_dim = hidden_dim / num_heads; // 64
    let half_dim = head_dim / 2; // 32
    let rope_theta = 1_000_000.0f32;
    let rope_type = 2; // NEOX style

    println!("Config:");
    println!("  head_dim: {}, half_dim: {}", head_dim, half_dim);
    println!("  rope_theta: {}", rope_theta);
    println!("  rope_type: {} (NEOX)", rope_type);

    // Test vector: simple values to track transformations
    let x: Vec<f32> = (0..head_dim).map(|i| i as f32 / 10.0).collect();
    println!("\nOriginal x[0..8]: {:?}", &x[..8]);
    println!("Original x[32..40]: {:?}", &x[32..40]);

    // For NEOX style at position 1:
    // x'[i] = x[i] * cos(angle[i]) - x[i+half] * sin(angle[i])
    // x'[i+half] = x[i] * sin(angle[i]) + x[i+half] * cos(angle[i])
    let position = 1;
    let pos_f32 = position as f32;

    // Compute cos/sin for each frequency
    let mut cos_vals = vec![0.0f32; half_dim];
    let mut sin_vals = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = pos_f32 * freq;
        cos_vals[i] = angle.cos();
        sin_vals[i] = angle.sin();
    }

    println!(
        "\nPosition {}: cos_vals[0..8]: {:?}",
        position,
        &cos_vals[..8]
    );
    println!(
        "Position {}: sin_vals[0..8]: {:?}",
        position,
        &sin_vals[..8]
    );

    // Apply NEOX style rotation
    // Split x into first half and second half
    let first_half = &x[0..half_dim].to_vec();
    let second_half = &x[half_dim..].to_vec();

    let mut x_rotated = vec![0.0f32; head_dim];
    for i in 0..half_dim {
        // x'[i] = x[i] * cos - x[i+half] * sin
        x_rotated[i] = first_half[i] * cos_vals[i] - second_half[i] * sin_vals[i];
        // x'[i+half] = x[i] * sin + x[i+half] * cos
        x_rotated[i + half_dim] = first_half[i] * sin_vals[i] + second_half[i] * cos_vals[i];
    }

    println!("\nNEOX rotated x[0..8]: {:?}", &x_rotated[..8]);
    println!("NEOX rotated x[32..40]: {:?}", &x_rotated[32..40]);

    // Now verify this matches what our implementation does
    // Using the same formula from realizarquantize::apply_rope_rotation_scalar:
    // x1_new = x1 * cos - x2 * sin
    // x2_new = x1 * sin + x2 * cos
    println!("\n=== Implementation Verification ===");

    let mut x_impl: Vec<f32> = (0..head_dim).map(|i| i as f32 / 10.0).collect();
    let (x1, x2) = x_impl.split_at_mut(half_dim);

    // Apply the formula from apply_rope_rotation_scalar
    for i in 0..half_dim {
        let v1 = x1[i];
        let v2 = x2[i];
        let cos_v = cos_vals[i];
        let sin_v = sin_vals[i];
        x1[i] = v1 * cos_v - v2 * sin_v;
        x2[i] = v1 * sin_v + v2 * cos_v;
    }

    println!("Implementation x[0..8]: {:?}", &x_impl[..8]);
    println!("Implementation x[32..40]: {:?}", &x_impl[32..40]);

    // Check if they match
    let max_diff = x_rotated
        .iter()
        .zip(x_impl.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("\nMax difference: {:.10}", max_diff);

    // Now compare with NORM style (type 0) to see the difference
    println!("\n=== NORM Style Comparison ===");

    let mut x_norm: Vec<f32> = (0..head_dim).map(|i| i as f32 / 10.0).collect();
    // NORM style: adjacent pairs
    for i in 0..half_dim {
        let x0 = x_norm[2 * i];
        let x1 = x_norm[2 * i + 1];
        x_norm[2 * i] = x0 * cos_vals[i] - x1 * sin_vals[i];
        x_norm[2 * i + 1] = x0 * sin_vals[i] + x1 * cos_vals[i];
    }

    println!("NORM rotated x[0..8]: {:?}", &x_norm[..8]);
    println!("NORM rotated x[32..40]: {:?}", &x_norm[32..40]);

    println!("\n=== Key Insight ===");
    println!("NEOX style groups elements at indices [i, i+half_dim]");
    println!("NORM style groups elements at indices [2*i, 2*i+1]");
    println!("If the wrong style is used, the rotation will be completely wrong!");
}
