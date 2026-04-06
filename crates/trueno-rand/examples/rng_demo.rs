//! RNG demonstration: Philox 4x32-10 and Threefry 4x64-20.
//!
//! ```sh
//! cargo run --example rng_demo -p trueno-rand
//! ```

use trueno_rand::{Philox4x32, Rng, Threefry4x64};

fn main() {
    println!("=== trueno-rand: RNG Demo ===\n");

    // ── Philox 4x32-10 ──────────────────────────────────
    println!("--- Philox 4x32-10 ---");
    let mut rng = Philox4x32::new(42);
    let vals = rng.next_4u32();
    println!("4 u32 values: {vals:?}");

    let mut rng2 = Philox4x32::new(42);
    assert_eq!(rng2.next_4u32(), vals);
    println!("Determinism verified: same seed → same output");

    let mut rng3 = Philox4x32::new(123);
    let mut uniform = vec![0.0f32; 10000];
    rng3.fill_uniform(&mut uniform);
    let mean: f32 = uniform.iter().sum::<f32>() / uniform.len() as f32;
    println!("Uniform mean: {mean:.4} (expected: 0.5)");

    let mut rng4 = Philox4x32::new(456);
    let mut normal = vec![0.0f32; 10000];
    rng4.fill_normal(&mut normal);
    let n_mean: f64 = normal.iter().map(|&v| f64::from(v)).sum::<f64>() / normal.len() as f64;
    println!("Normal mean: {n_mean:.4} (expected: 0.0)");

    let key = [42u32, 0];
    let result = Philox4x32::generate_at(key, [0, 0, 0, 0]);
    println!("Stateless: generate_at = {result:?}");

    // ── Threefry 4x64-20 ────────────────────────────────
    println!("\n--- Threefry 4x64-20 ---");
    let mut tf = Threefry4x64::new(42);
    let tf_vals = tf.next_4u64();
    println!("4 u64 values: {tf_vals:?}");

    let mut tf2 = Threefry4x64::new(42);
    assert_eq!(tf2.next_4u64(), tf_vals);
    println!("Determinism verified: same seed → same output");

    let mut tf3 = Threefry4x64::new(789);
    let mut tf_uniform = vec![0.0f32; 10000];
    tf3.fill_uniform(&mut tf_uniform);
    let tf_mean: f32 = tf_uniform.iter().sum::<f32>() / tf_uniform.len() as f32;
    println!("Uniform mean: {tf_mean:.4} (expected: 0.5)");

    let mut tf4 = Threefry4x64::new(101);
    let mut tf_normal = vec![0.0f32; 10000];
    tf4.fill_normal(&mut tf_normal);
    let tf_n_mean: f64 =
        tf_normal.iter().map(|&v| f64::from(v)).sum::<f64>() / tf_normal.len() as f64;
    println!("Normal mean: {tf_n_mean:.4} (expected: 0.0)");

    let tf_key = [42u64, 0, 0, 0];
    let tf_result = Threefry4x64::generate_at(tf_key, [0, 0, 0, 0]);
    println!("Stateless: generate_at = {tf_result:?}");

    // ── Rng trait (dynamic dispatch) ──────────────────────
    println!("\n--- Rng trait (unified interface) ---");
    let mut philox = Philox4x32::new(555);
    let rng: &mut dyn Rng = &mut philox;
    let mut buf = vec![0.0f32; 1000];
    rng.fill_uniform(&mut buf);
    let trait_mean: f32 = buf.iter().sum::<f32>() / buf.len() as f32;
    println!("dyn Rng (Philox) uniform mean: {trait_mean:.4}");

    let mut threefry = Threefry4x64::new(555);
    let rng2: &mut dyn Rng = &mut threefry;
    rng2.fill_normal(&mut buf);
    let n_mean: f64 = buf.iter().map(|&v| f64::from(v)).sum::<f64>() / buf.len() as f64;
    println!("dyn Rng (Threefry) normal mean: {n_mean:.4}");

    println!("\n=== All RNG demos passed ===");
}
