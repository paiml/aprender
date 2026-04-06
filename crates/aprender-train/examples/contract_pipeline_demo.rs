//! # Contract Pipeline Demo
//!
//! Demonstrates the escape-proof contract enforcement pipeline for entrenar:
//!
//! ```text
//! contracts/*.yaml  ->  build.rs reads PRE/POST  ->  #[contract] proc macro
//!       |                      |                          |
//!  Lean theorem ref    ENV vars set at build     debug_assert!() injected
//! ```
//!
//! ## How it works
//!
//! 1. `contracts/matmul-v1.yaml` defines preconditions:
//!    ```yaml
//!    equations:
//!      matmul:
//!        preconditions:
//!          - "a.len() == m * k"
//!          - "b.len() == k * n"
//!          - "k > 0"
//!    ```
//!
//! 2. `build.rs` reads the YAML and emits env vars:
//!    ```text
//!    cargo:rustc-env=CONTRACT_MATMUL_V1_MATMUL_PRE_0=a.len() == m * k
//!    cargo:rustc-env=CONTRACT_MATMUL_V1_MATMUL_PRE_1=b.len() == k * n
//!    cargo:rustc-env=CONTRACT_MATMUL_V1_MATMUL_PRE_2=k > 0
//!    ```
//!
//! 3. `#[contract("matmul-v1", equation = "matmul")]` reads those env
//!    vars at compile time and injects `debug_assert!()` into the function body.
//!
//! 4. Change the YAML -> assertions change automatically at next build.
//!    Remove the YAML -> compile error (env var missing).
//!
//! ## Run
//!
//! ```bash
//! cargo run --example contract_pipeline_demo
//! ```

use entrenar::autograd::Tensor;
use ndarray::Array1;

fn main() {
    println!("=== Entrenar Contract Pipeline Demo ===\n");

    // Show build-time contract metadata
    println!("Contract binding source: {}", env!("CONTRACT_BINDING_SOURCE"));
    println!("Total bindings: {}", env!("CONTRACT_TOTAL"));
    println!("Implemented: {}", env!("CONTRACT_IMPLEMENTED"));
    println!("Partial: {}", env!("CONTRACT_PARTIAL"));
    println!("Gaps: {}\n", env!("CONTRACT_GAPS"));

    // ---- Matmul contract (matmul-v1.yaml / matmul) ----
    println!("--- Matmul contract (from YAML preconditions) ---");
    let a = Tensor::new(Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), false);
    let b = Tensor::new(Array1::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]), false);
    let c = entrenar::autograd::matmul(&a, &b, 2, 3, 2);
    println!("  matmul(A[2x3], B[3x2]) = {:?}", c.data().to_vec());
    println!("  Preconditions checked: a.len()==m*k, b.len()==k*n, k>0");
    println!("  Postconditions checked: ret.len()==m*n, all finite\n");

    // ---- Softmax contract (softmax-v1.yaml / softmax) ----
    println!("--- Softmax contract (from YAML preconditions) ---");
    let logits = Tensor::new(Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]), false);
    let probs = entrenar::autograd::softmax(&logits);
    let sum: f32 = probs.data().iter().sum();
    println!("  softmax([1,2,3,4]) = {:?}", probs.data().to_vec());
    println!("  sum = {sum:.6} (partition of unity)");
    println!("  Preconditions checked: !a.data().is_empty(), all finite\n");

    // ---- AdamW SIMD contract (optimizer-v1.yaml / adamw_update) ----
    println!("--- AdamW SIMD contract (from YAML preconditions) ---");
    let grad = vec![0.1f32; 4];
    let mut momentum = vec![0.0f32; 4];
    let mut variance = vec![0.0f32; 4];
    let mut params = vec![1.0f32; 4];
    entrenar::optim::simd_adamw_update(
        &grad,
        &mut momentum,
        &mut variance,
        &mut params,
        0.9,   // beta1
        0.999, // beta2
        0.001, // lr
        0.001, // lr_t
        0.01,  // weight_decay
        1e-8,  // epsilon
    );
    println!("  After one AdamW step: params = {:?}", &params);
    println!("  Preconditions checked: lr>0, beta1/beta2 in (0,1), eps>0, wd>=0, !grad.empty()");
    println!("  Postconditions checked: all params finite, param.len()==grad.len()\n");

    // ---- LoRA contract (lora-v1.yaml / lora_decomposition) ----
    println!("--- LoRA decomposition contract (from YAML preconditions) ---");
    let base = Tensor::new(Array1::from_vec(vec![0.1; 4 * 3]), false);
    let layer = entrenar::lora::LoRALayer::new(base, 4, 3, 2, 1.0);
    println!("  LoRA layer: d_out=4, d_in=3, rank=2, alpha=1.0");
    println!("  Preconditions checked: rank>0, rank<=min(d_out,d_in), alpha>0, base_weight.len()==d_out*d_in\n");

    // ---- Q4_0 quantization contract (quantization-v1.yaml / symmetric_4bit) ----
    println!("--- Q4_0 quantization contract (from YAML preconditions) ---");
    let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let quantized = entrenar::quant::Q4_0::quantize(&weights);
    let dequantized = quantized.dequantize();
    let max_err: f32 =
        weights.iter().zip(dequantized.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("  Q4_0 quantize-dequantize max error: {max_err:.4}");
    println!("  Compression ratio: {:.1}x", quantized.compression_ratio());
    println!("  Preconditions checked: !values.is_empty(), all values finite");
    println!("  Postconditions checked: ret.len==values.len(), scales positive\n");

    // Prevent unused variable warnings
    let _ = layer;

    println!("=== Pipeline Summary ===");
    println!("  YAML contracts:     11 files, 25+ equations with pre/postconditions");
    println!("  build.rs env vars:  Set at compile time from YAML");
    println!("  #[contract] macro:  Reads env vars, injects debug_assert!()");
    println!("  Lean theorems:      Referenced in YAML, proven in provable-contracts/lean/");
    println!("  Runtime cost:       Zero in release builds (debug_assert!)");
    println!("\n  Change YAML -> assertions change automatically.");
    println!("  Remove YAML -> compile_error!()");
}
