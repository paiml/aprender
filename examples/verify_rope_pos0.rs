//! Verify RoPE at position 0 is identity
fn main() {
    let theta = 1000000.0f32;
    let head_dim = 64usize;
    let half_dim = head_dim / 2;
    let position = 0;

    println!("RoPE at position 0 (should be identity):");
    println!(
        "  theta={}, head_dim={}, half_dim={}",
        theta, head_dim, half_dim
    );

    let pos_f32 = position as f32;
    let head_dim_f32 = head_dim as f32;

    for i in 0..4 {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
        let angle = pos_f32 * freq;
        let (sin_v, cos_v) = angle.sin_cos();
        println!(
            "  i={}: freq={:.6}, angle={:.6}, cos={:.6}, sin={:.6}",
            i, freq, angle, cos_v, sin_v
        );
    }

    // At position 0, angle = 0 * freq = 0, so cos=1, sin=0
    // This means the rotation is identity: x' = x
    println!("\nAt position 0, all angles are 0, so cos=1, sin=0");
    println!("Therefore, RoPE at position 0 is identity (no rotation)");

    // Now at position 1
    println!("\nRoPE at position 1:");
    let position = 1;
    let pos_f32 = position as f32;
    for i in 0..4 {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
        let angle = pos_f32 * freq;
        let (sin_v, cos_v) = angle.sin_cos();
        println!(
            "  i={}: freq={:.6e}, angle={:.6e}, cos={:.6}, sin={:.6}",
            i, freq, angle, cos_v, sin_v
        );
    }
}
