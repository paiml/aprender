//! Trueno Compute Integration Example
//!
//! Demonstrates the integration of trueno 0.8.8+ compute infrastructure
//! with aprender's ML training pipeline.
//!
//! # Features Demonstrated
//!
//! - **Backend Selection**: Auto CPU/GPU dispatch based on data size
//! - **Training Guards**: NaN/Inf detection for stability (Jidoka)
//! - **Divergence Checking**: Cross-backend validation
//! - **Reproducibility**: Deterministic experiment seeding
//!
//! # Usage
//!
//! ```bash
//! cargo run --example trueno_compute_integration
//! ```

use aprender::compute::{
    select_backend, should_use_gpu, should_use_parallel, BackendCategory, DivergenceGuard,
    ExperimentSeed, TrainingGuard,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("        Trueno Compute Integration Demo (trueno 0.8.8+)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Demo 1: Backend Selection
    demo_backend_selection();

    // Demo 2: Training Guards (Jidoka)
    demo_training_guards();

    // Demo 3: Divergence Checking
    demo_divergence_checking();

    // Demo 4: Reproducible Experiments
    demo_reproducibility();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                     Demo Complete");
    println!("═══════════════════════════════════════════════════════════════════");
}

/// Demo 1: Automatic backend selection based on data size
fn demo_backend_selection() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 1: Backend Selection (Poka-Yoke)                           │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let test_sizes = [100, 1_000, 10_000, 100_000, 1_000_000];

    println!("  Size        | Backend Category  | GPU?  | Parallel?");
    println!("  {:─<55}", "");

    for size in test_sizes {
        let category = select_backend(size, false);
        let gpu = should_use_gpu(size);
        let parallel = should_use_parallel(size);

        let category_str = match category {
            BackendCategory::SimdOnly => "SIMD Only",
            BackendCategory::SimdParallel => "SIMD + Parallel",
            BackendCategory::Gpu => "GPU",
        };

        println!(
            "  {:>10}  | {:<17} | {:<5} | {}",
            format_size(size),
            category_str,
            if gpu { "Yes" } else { "No" },
            if parallel { "Yes" } else { "No" }
        );
    }

    println!("\n  With GPU available (size=1M):");
    let category = select_backend(1_000_000, true);
    println!("  → Backend: {:?}\n", category);
}

/// Demo 2: Training guards for NaN/Inf detection (Jidoka)
fn demo_training_guards() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 2: Training Guards (Jidoka - Stop on Defect)               │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let guard = TrainingGuard::new("epoch_1");

    // Valid gradients
    let valid_gradients = vec![0.01, -0.02, 0.015, -0.008];
    match guard.check_gradients(&valid_gradients) {
        Ok(()) => println!("  ✅ Valid gradients: check passed"),
        Err(e) => println!("  ❌ Valid gradients failed: {e}"),
    }

    // NaN gradient (training explosion)
    let nan_gradients = vec![0.01, f32::NAN, 0.015];
    match guard.check_gradients(&nan_gradients) {
        Ok(()) => println!("  ✅ NaN gradients: check passed (unexpected!)"),
        Err(_) => println!("  ❌ NaN gradients detected: Jidoka triggered"),
    }

    // Infinite gradient (gradient explosion)
    let inf_gradients = vec![0.01, f32::INFINITY, 0.015];
    match guard.check_gradients(&inf_gradients) {
        Ok(()) => println!("  ✅ Inf gradients: check passed (unexpected!)"),
        Err(_) => println!("  ❌ Inf gradients detected: Jidoka triggered"),
    }

    // Valid loss
    match guard.check_loss(0.5) {
        Ok(()) => println!("  ✅ Valid loss (0.5): check passed"),
        Err(e) => println!("  ❌ Valid loss failed: {e}"),
    }

    // NaN loss
    match guard.check_loss(f32::NAN) {
        Ok(()) => println!("  ✅ NaN loss: check passed (unexpected!)"),
        Err(_) => println!("  ❌ NaN loss detected: Jidoka triggered"),
    }

    println!();
}

/// Demo 3: Cross-backend divergence checking
fn demo_divergence_checking() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 3: Divergence Checking (Cross-Backend Validation)          │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    // Simulate CPU vs GPU results
    let cpu_result = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let gpu_result_close = vec![1.000001, 2.000002, 3.000001, 4.000001, 5.000002];
    let gpu_result_diverged = vec![1.1, 2.0, 3.0, 4.0, 5.0];

    let guard = DivergenceGuard::default_tolerance("cpu_vs_gpu");

    // Close results (within tolerance)
    match guard.check(&cpu_result, &gpu_result_close) {
        Ok(()) => println!("  ✅ CPU vs GPU (close): within tolerance (1e-5)"),
        Err(e) => println!("  ❌ CPU vs GPU (close) failed: {e}"),
    }

    // Diverged results (exceeds tolerance)
    match guard.check(&cpu_result, &gpu_result_diverged) {
        Ok(()) => println!("  ✅ CPU vs GPU (diverged): within tolerance (unexpected!)"),
        Err(_) => println!("  ❌ CPU vs GPU (diverged): exceeds tolerance"),
    }

    // Custom tolerance
    let relaxed_guard = DivergenceGuard::new(0.2, "relaxed_check");
    match relaxed_guard.check(&cpu_result, &gpu_result_diverged) {
        Ok(()) => println!("  ✅ Relaxed tolerance (0.2): diverged result now passes"),
        Err(e) => println!("  ❌ Relaxed tolerance failed: {e}"),
    }

    println!();
}

/// Demo 4: Reproducible experiment seeding
fn demo_reproducibility() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Demo 4: Reproducible Experiments (Deterministic Seeding)        │");
    println!("└─────────────────────────────────────────────────────────────────┘\n");

    let seed = ExperimentSeed::from_master(42);

    println!("  Master seed: {}", seed.master);
    println!("  Derived seeds:");
    println!("    Data shuffle:  {}", seed.data_shuffle);
    println!("    Weight init:   {}", seed.weight_init);
    println!("    Dropout:       {}", seed.dropout);

    // Verify determinism
    let seed2 = ExperimentSeed::from_master(42);
    let deterministic = seed.master == seed2.master
        && seed.data_shuffle == seed2.data_shuffle
        && seed.weight_init == seed2.weight_init
        && seed.dropout == seed2.dropout;

    println!(
        "\n  Determinism check: {}",
        if deterministic {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );

    // Different master = different seeds
    let seed3 = ExperimentSeed::from_master(123);
    println!("\n  Different master (123):");
    println!("    Data shuffle:  {} (different)", seed3.data_shuffle);

    println!();
}

/// Format size with K/M suffix
fn format_size(size: usize) -> String {
    if size >= 1_000_000 {
        format!("{}M", size / 1_000_000)
    } else if size >= 1_000 {
        format!("{}K", size / 1_000)
    } else {
        size.to_string()
    }
}
