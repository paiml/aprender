//! Contract Pipeline Demo
//!
//! Demonstrates the provable-contracts pipeline:
//!
//! 1. YAML contracts in `contracts/*.yaml` define equations with pre/postconditions
//! 2. `build.rs` reads the YAML and emits `CONTRACT_*` env vars
//! 3. `#[contract("yaml-name", equation = "eq")]` proc macro reads those env vars
//!    and injects `debug_assert!()` calls for preconditions/postconditions
//! 4. In debug builds, violations are caught at runtime
//! 5. In release builds, assertions are compiled away (zero cost)
//!
//! # Running
//!
//! ```bash
//! # Debug mode: assertions active
//! cargo run --example contract_pipeline_demo
//!
//! # Release mode: assertions compiled away, zero overhead
//! cargo run --release --example contract_pipeline_demo
//! ```

fn main() {
    println!("=== Contract Pipeline Demo ===\n");

    demo_rms_norm();
    demo_softmax();
    demo_dequantize();

    println!("\nAll contract-gated functions executed successfully.");
    println!("Pre/postconditions from YAML were enforced via debug_assert!().");
}

/// Demonstrate RMS norm with contract enforcement.
///
/// Contract: forward-pass-v1.yaml / rms_norm
/// Preconditions: !input.is_empty(), eps > 0.0, !weight.is_empty()
/// Postconditions: ret.len() == input.len(), ret.iter().all(|v| v.is_finite())
fn demo_rms_norm() {
    println!("1. RMS Norm (forward-pass-v1 / rms_norm)");

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32, 1.0, 1.0, 1.0];
    let eps = 1e-5;

    let output = realizar::gguf::ops::rms_norm(&input, &weight, eps);

    println!("   input:  {:?}", input);
    println!("   output: {:?}", output);
    println!(
        "   sum(output^2)/n = {:.6}",
        output.iter().map(|v| v * v).sum::<f32>() / output.len() as f32
    );
    assert_eq!(output.len(), input.len());
    assert!(output.iter().all(|v| v.is_finite()));
    println!("   OK: postconditions verified\n");
}

/// Demonstrate softmax with contract enforcement.
///
/// Contract: sampling-v1.yaml / softmax_inplace
/// Preconditions: !logits.is_empty(), logits.iter().all(|v| v.is_finite())
fn demo_softmax() {
    println!("2. Softmax (sampling-v1 / softmax_inplace)");

    let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
    println!("   input:  {:?}", logits);

    realizar::gguf::ops::softmax(&mut logits);

    println!("   output: {:?}", logits);
    let sum: f32 = logits.iter().sum();
    println!("   sum = {:.6} (should be ~1.0)", sum);
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(logits.iter().all(|v| *v >= 0.0 && v.is_finite()));
    println!("   OK: softmax probabilities sum to 1.0\n");
}

/// Demonstrate dequantize with contract enforcement.
///
/// Contract: quantization-v1.yaml / q4_0_dequant
/// Preconditions: !data.is_empty(), data.len() % 18 == 0
fn demo_dequantize() {
    println!("3. Dequantize Q4_0 (quantization-v1 / q4_0_dequant)");

    // Build a minimal Q4_0 block: 2 bytes scale (f16) + 16 bytes quants = 18 bytes
    let mut block = vec![0u8; 18];
    // Scale = 1.0 in f16 (0x3C00)
    let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    // Quants: all 0x88 -> nibbles (8, 8) -> values (0, 0) after subtracting 8
    for slot in &mut block[2..18] {
        *slot = 0x88;
    }

    let result = realizar::quantize::dequant::dequantize_q4_0(&block);
    match result {
        Ok(values) => {
            println!(
                "   dequantized {} values from {} bytes",
                values.len(),
                block.len()
            );
            println!("   first 8: {:?}", &values[..8.min(values.len())]);
            assert!(values.iter().all(|v| v.is_finite()));
            println!("   OK: all values finite\n");
        },
        Err(e) => {
            println!("   Error: {e}\n");
        },
    }
}
