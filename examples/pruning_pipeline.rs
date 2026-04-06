//! Pruning Pipeline Example
//!
//! Demonstrates the end-to-end pruning workflow using Entrenar:
//! - Configuring pruning schedules (OneShot, Gradual, Cubic)
//! - Setting up calibration for activation-weighted methods
//! - Running the prune-finetune pipeline
//!
//! # References
//! - Zhu & Gupta (2017) - To Prune or Not To Prune
//! - Sun et al. (2023) - Wanda: A Simple and Effective Pruning Approach
//!
//! Run with: cargo run --example pruning_pipeline

use entrenar::prune::{
    CalibrationConfig, PruneMethod, PruningConfig, PruningSchedule, PruningStage,
    SparsityPatternConfig,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Pruning Pipeline with Entrenar                       ║");
    println!("║         End-to-end model compression workflow                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // 1. OneShot Pruning Schedule
    // =========================================================================
    println!("📋 Schedule 1: OneShot Pruning");
    let oneshot = PruningSchedule::OneShot { step: 1000 };
    println!("   Prune at step: 1000");
    println!("   Sparsity before step 1000: {:.0}%", oneshot.sparsity_at_step(999) * 100.0);
    println!("   Sparsity at step 1000: {:.0}%", oneshot.sparsity_at_step(1000) * 100.0);
    println!("   Sparsity after step 1000: {:.0}%\n", oneshot.sparsity_at_step(1001) * 100.0);

    // =========================================================================
    // 2. Gradual Pruning Schedule
    // =========================================================================
    println!("📋 Schedule 2: Gradual Pruning");
    let gradual = PruningSchedule::Gradual {
        start_step: 1000,
        end_step: 5000,
        initial_sparsity: 0.0,
        final_sparsity: 0.5,
        frequency: 500,
    };
    println!("   Start: step 1000, End: step 5000");
    println!("   Initial sparsity: 0%, Final sparsity: 50%");
    println!("   Pruning frequency: every 500 steps");
    println!("   Sparsity progression:");
    for step in [1000, 2000, 3000, 4000, 5000] {
        println!("     Step {:>5}: {:.1}%", step, gradual.sparsity_at_step(step) * 100.0);
    }
    println!();

    // =========================================================================
    // 3. Cubic Pruning Schedule (Zhu & Gupta 2017)
    // =========================================================================
    println!("📋 Schedule 3: Cubic Pruning (Zhu & Gupta)");
    let cubic = PruningSchedule::Cubic { start_step: 0, end_step: 10000, final_sparsity: 0.7 };
    println!("   Formula: s_t = s_f * (1 - (1 - t/T)^3)");
    println!("   Final sparsity: 70%");
    println!("   Sparsity progression:");
    for step in [0, 2500, 5000, 7500, 10000] {
        println!("     Step {:>5}: {:.1}%", step, cubic.sparsity_at_step(step) * 100.0);
    }
    println!();

    // =========================================================================
    // 4. Pruning Configuration - Magnitude
    // =========================================================================
    println!("⚙️  Config 1: Magnitude Pruning (No Calibration)");
    let magnitude_config = PruningConfig::new()
        .with_method(PruneMethod::Magnitude)
        .with_target_sparsity(0.5)
        .with_pattern(SparsityPatternConfig::Unstructured)
        .with_schedule(gradual.clone());

    println!("   Method: {}", magnitude_config.method().display_name());
    println!("   Requires calibration: {}", magnitude_config.requires_calibration());
    println!("   Target sparsity: {:.0}%", magnitude_config.target_sparsity() * 100.0);
    println!("   Pattern: Unstructured\n");

    // =========================================================================
    // 5. Pruning Configuration - Wanda
    // =========================================================================
    println!("⚙️  Config 2: Wanda Pruning (Requires Calibration)");
    let wanda_config = PruningConfig::new()
        .with_method(PruneMethod::Wanda)
        .with_target_sparsity(0.5)
        .with_pattern(SparsityPatternConfig::nm_2_4())
        .with_schedule(PruningSchedule::OneShot { step: 0 });

    println!("   Method: {}", wanda_config.method().display_name());
    println!("   Requires calibration: {}", wanda_config.requires_calibration());
    println!("   Pattern: 2:4 N:M Sparsity");
    println!("   Theoretical sparsity: 50%\n");

    // =========================================================================
    // 6. Calibration Configuration
    // =========================================================================
    println!("📊 Calibration Configuration");
    let calibration_config = CalibrationConfig::new()
        .with_num_samples(128)
        .with_sequence_length(2048)
        .with_batch_size(1)
        .with_dataset("c4");

    println!("   Samples: {}", calibration_config.num_samples());
    println!("   Sequence length: {}", calibration_config.sequence_length());
    println!("   Batch size: {}", calibration_config.batch_size());
    println!("   Dataset: {}\n", calibration_config.dataset());

    // =========================================================================
    // 7. Pipeline Stages
    // =========================================================================
    println!("🔄 Pipeline Stages");
    let stages = [
        PruningStage::Idle,
        PruningStage::Calibrating,
        PruningStage::ComputingImportance,
        PruningStage::Pruning,
        PruningStage::FineTuning,
        PruningStage::Evaluating,
        PruningStage::Exporting,
        PruningStage::Complete,
    ];

    for (i, stage) in stages.iter().enumerate() {
        let status = if stage.is_active() {
            "🟢 Active"
        } else if stage.is_terminal() {
            "✅ Terminal"
        } else {
            "⚪ Waiting"
        };
        println!("   {}. {:20} {}", i + 1, stage.display_name(), status);
    }

    // =========================================================================
    // 8. Validate Configurations
    // =========================================================================
    println!("\n✓ Validating Configurations");
    match magnitude_config.validate() {
        Ok(()) => println!("   Magnitude config: Valid"),
        Err(e) => println!("   Magnitude config: Invalid - {e}"),
    }
    match wanda_config.validate() {
        Ok(()) => println!("   Wanda config: Valid"),
        Err(e) => println!("   Wanda config: Invalid - {e}"),
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Pipeline Summary                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Pruning Methods:                                            ║");
    println!("║    - Magnitude (L1/L2) - No calibration needed               ║");
    println!("║    - Wanda - Activation-weighted, needs calibration          ║");
    println!("║    - SparseGPT - Hessian-based, needs calibration            ║");
    println!("║                                                              ║");
    println!("║  Sparsity Patterns:                                          ║");
    println!("║    - Unstructured - Maximum flexibility                      ║");
    println!("║    - N:M (2:4, 4:8) - Hardware-accelerated on Ampere         ║");
    println!("║    - Block - Coarse-grained structured                       ║");
    println!("║                                                              ║");
    println!("║  Schedules:                                                  ║");
    println!("║    - OneShot - Single pruning event                          ║");
    println!("║    - Gradual - Linear interpolation                          ║");
    println!("║    - Cubic - Zhu & Gupta (2017) formula                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
