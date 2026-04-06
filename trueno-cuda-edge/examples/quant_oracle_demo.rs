//! Quantization Parity Oracle Demo
//!
//! Demonstrates boundary value generation and CPU/GPU parity checking
//! for quantized tensor operations.
//!
//! Run with: cargo run --example quant_oracle_demo

use trueno_cuda_edge::quant_oracle::{
    check_values_parity, BoundaryValueGenerator, ParityConfig, QuantFormat,
};

/// Display tolerance and quantization levels for each format.
fn demo_format_tolerances() {
    println!("1. Quantization Format Tolerances");
    println!("   ───────────────────────────────");

    let formats = [
        QuantFormat::Q4K,
        QuantFormat::Q5K,
        QuantFormat::Q6K,
        QuantFormat::Q8_0,
        QuantFormat::F16,
        QuantFormat::F32,
    ];

    for fmt in &formats {
        println!(
            "   {:<6} │ tolerance: {:<12} │ levels: {}",
            fmt.to_string(),
            format!("{:.2e}", fmt.tolerance()),
            fmt.levels()
        );
    }
    println!();
}

/// Format a boundary value for display.
fn format_boundary_value(v: &f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            "+Inf".to_string()
        } else {
            "-Inf".to_string()
        }
    } else if *v == 0.0 {
        if v.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        }
    } else {
        format!("{:.2e}", v)
    }
}

/// Demonstrate boundary value generation.
fn demo_boundary_generation() {
    println!("2. Boundary Value Generation");
    println!("   ──────────────────────────");

    let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);

    println!("   Universal boundaries (all formats):");
    let universal = gen.universal_boundaries();
    for (i, v) in universal.iter().enumerate() {
        print!("   {:>12}", format_boundary_value(v));
        if (i + 1) % 3 == 0 {
            println!();
        }
    }
    println!();

    println!("\n   Format-specific boundaries (Q4K, 16 levels):");
    let format_bounds = gen.format_boundaries();
    println!("   Count: {} values (16 levels x 2 signs)", format_bounds.len());
    print!("   First 8: ");
    for v in format_bounds.iter().take(8) {
        print!("{:.3} ", v);
    }
    println!("...");
    println!();
}

/// Demonstrate CPU/GPU parity checking.
fn demo_parity_checking() {
    println!("3. CPU/GPU Parity Checking");
    println!("   ────────────────────────");

    let config = ParityConfig::new(QuantFormat::Q4K);
    println!("   Format: Q4K, Tolerance: {}", config.tolerance());

    // Test 1: Identical values
    let cpu = vec![1.0, 2.0, 3.0, 4.0];
    let gpu = vec![1.0, 2.0, 3.0, 4.0];
    let report = check_values_parity(&cpu, &gpu, &config);
    println!("\n   Test 1: Identical values");
    println!("   CPU: {:?}", cpu);
    println!("   GPU: {:?}", gpu);
    println!("   Result: {}", if report.passed() { "PASS" } else { "FAIL" });

    // Test 2: Small difference (within tolerance)
    let gpu_close = vec![1.01, 2.01, 3.01, 4.01];
    let report = check_values_parity(&cpu, &gpu_close, &config);
    println!("\n   Test 2: Small difference (within 0.05 tolerance)");
    println!("   CPU: {:?}", cpu);
    println!("   GPU: {:?}", gpu_close);
    println!("   Result: {}", if report.passed() { "PASS" } else { "FAIL" });
    println!("   Max diff: {:.4}", report.max_abs_diff);

    // Test 3: Large difference (exceeds tolerance)
    let gpu_far = vec![1.0, 2.5, 3.0, 4.0];
    let report = check_values_parity(&cpu, &gpu_far, &config);
    println!("\n   Test 3: Large difference (exceeds tolerance)");
    println!("   CPU: {:?}", cpu);
    println!("   GPU: {:?}", gpu_far);
    println!("   Result: {}", if report.passed() { "PASS" } else { "FAIL" });
    println!("   Violations: {}", report.violations.len());
    for v in &report.violations {
        println!(
            "     Index {}: CPU={}, GPU={}, diff={}",
            v.index, v.cpu_value, v.gpu_value, v.abs_diff
        );
    }

    // Test 4: NaN handling
    let cpu_nan = vec![f64::NAN, 1.0];
    let gpu_nan = vec![f64::NAN, 1.0];
    let report = check_values_parity(&cpu_nan, &gpu_nan, &config);
    println!("\n   Test 4: NaN handling");
    println!("   CPU: [NaN, 1.0]");
    println!("   GPU: [NaN, 1.0]");
    println!("   Result: {}", if report.passed() { "PASS (NaN == NaN)" } else { "FAIL" });
}

fn main() {
    println!("=== Quantization Parity Oracle Demo ===\n");
    demo_format_tolerances();
    demo_boundary_generation();
    demo_parity_checking();
    println!("\n=== Demo Complete ===");
}
