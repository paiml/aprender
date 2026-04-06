#![allow(clippy::disallowed_methods)]
//! Pipeline Verification Example
//!
//! Demonstrates the verify module for ML pipeline debugging with:
//! - Stage-by-stage ground truth comparison
//! - Multiple tolerance types (percent, stats, KL divergence)
//! - Jidoka-style stop-on-failure behavior
//! - Detailed diagnostic output for failures
//!
//! Run with: `cargo run --example pipeline_verification`

use aprender::verify::{Delta, GroundTruth, Pipeline, StageStatus, Tolerance, VerifyReport};

fn main() {
    println!("=== Pipeline Verification System ===\n");
    println!("Toyota Way: Jidoka - Built-in quality with automatic stop on defect\n");

    demo_basic_pipeline();
    demo_failure_detection();
    demo_continue_on_failure();
    demo_stats_tolerance();
    demo_ground_truth_from_data();
    demo_cosine_similarity();
    demo_kl_divergence();
    demo_whisper_pipeline();

    print_summary();
}

/// Part 1: Basic Pipeline with Percent Tolerance
fn demo_basic_pipeline() {
    println!("--- Part 1: Basic Pipeline (Percent Tolerance) ---\n");

    let pipeline = Pipeline::builder("audio-encoder")
        .stage("mel_spectrogram")
        .ground_truth_stats(-0.215, 0.448)
        .tolerance(Tolerance::percent(5.0))
        .description("Mel spectrogram extraction")
        .build_stage()
        .stage("encoder_layer_1")
        .ground_truth_stats(0.0, 1.0)
        .tolerance(Tolerance::percent(10.0))
        .description("First encoder transformer layer")
        .build_stage()
        .stage("encoder_layer_2")
        .ground_truth_stats(0.0, 1.0)
        .tolerance(Tolerance::percent(10.0))
        .description("Second encoder transformer layer")
        .build_stage()
        .build()
        .expect("Failed to build pipeline");

    println!("Pipeline: {}", pipeline.name());
    println!("Stages: {}\n", pipeline.stages().len());

    // Simulate outputs that pass verification
    let report = pipeline.verify(|stage_name| match stage_name {
        "mel_spectrogram" => Some(GroundTruth::from_stats(-0.210, 0.450)),
        "encoder_layer_1" => Some(GroundTruth::from_stats(0.02, 0.98)),
        "encoder_layer_2" => Some(GroundTruth::from_stats(-0.01, 1.02)),
        _ => None,
    });

    print_report(&report);
}

/// Part 2: Detecting Sign Flip Errors
fn demo_failure_detection() {
    println!("\n--- Part 2: Detecting Sign Flip Errors ---\n");

    let pipeline = Pipeline::builder("audio-encoder")
        .stage("mel_spectrogram")
        .ground_truth_stats(-0.215, 0.448)
        .tolerance(Tolerance::percent(5.0))
        .build_stage()
        .stage("encoder_layer_1")
        .ground_truth_stats(0.0, 1.0)
        .tolerance(Tolerance::percent(10.0))
        .build_stage()
        .stage("encoder_layer_2")
        .ground_truth_stats(0.0, 1.0)
        .tolerance(Tolerance::percent(10.0))
        .build_stage()
        .build()
        .expect("Failed to build pipeline");

    // Simulate a sign flip error in mel spectrogram
    let report = pipeline.verify(|stage_name| match stage_name {
        "mel_spectrogram" => Some(GroundTruth::from_stats(0.184, 0.448)), // SIGN FLIPPED!
        "encoder_layer_1" | "encoder_layer_2" => Some(GroundTruth::from_stats(0.0, 1.0)),
        _ => None,
    });

    print_report(&report);

    // Show diagnosis for the failure
    if let Some(failure) = report.first_failure() {
        println!("\nDiagnosis for '{}' failure:", failure.name());
        for diag in failure.diagnose() {
            println!("  - {diag}");
        }
    }
}

/// Part 3: Continue-on-Failure Mode
fn demo_continue_on_failure() {
    println!("\n--- Part 3: Continue-on-Failure Mode ---\n");

    let pipeline = Pipeline::builder("full-analysis")
        .stage("stage_a")
        .ground_truth_stats(0.0, 1.0)
        .tolerance(Tolerance::percent(5.0))
        .build_stage()
        .stage("stage_b")
        .ground_truth_stats(0.0, 1.0)
        .tolerance(Tolerance::percent(5.0))
        .build_stage()
        .stage("stage_c")
        .ground_truth_stats(0.0, 1.0)
        .tolerance(Tolerance::percent(5.0))
        .build_stage()
        .continue_on_failure() // Disable Jidoka for full analysis
        .build()
        .expect("Failed to build pipeline");

    let report = pipeline.verify(|stage_name| match stage_name {
        "stage_a" => Some(GroundTruth::from_stats(0.5, 1.0)), // FAIL
        "stage_b" => Some(GroundTruth::from_stats(0.0, 0.98)), // PASS
        "stage_c" => Some(GroundTruth::from_stats(0.3, 1.0)), // FAIL
        _ => None,
    });

    println!("With continue_on_failure(), all stages are evaluated:");
    print_report(&report);
}

/// Part 4: Stats-Based Tolerance
fn demo_stats_tolerance() {
    println!("\n--- Part 4: Stats-Based Tolerance ---\n");

    let pipeline = Pipeline::builder("precision-check")
        .stage("high_precision")
        .ground_truth_stats(0.0, 1.0)
        .tolerance(Tolerance::stats(0.01, 0.02)) // Very tight
        .build_stage()
        .stage("normal_precision")
        .ground_truth_stats(0.0, 1.0)
        .tolerance(Tolerance::stats(0.1, 0.1)) // Normal tolerance
        .build_stage()
        .build()
        .expect("Failed to build pipeline");

    let report = pipeline.verify(|stage_name| match stage_name {
        "high_precision" => Some(GroundTruth::from_stats(0.005, 1.01)),
        "normal_precision" => Some(GroundTruth::from_stats(0.05, 0.95)),
        _ => None,
    });

    print_report(&report);
}

/// Part 5: Ground Truth from Raw Data
fn demo_ground_truth_from_data() {
    println!("\n--- Part 5: Ground Truth from Raw Data ---\n");

    let reference_output = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let gt = GroundTruth::from_slice(&reference_output);

    println!("Ground truth computed from raw data:");
    println!("  Mean: {:.4}", gt.mean());
    println!("  Std:  {:.4}", gt.std());
    println!("  Min:  {:.4}", gt.min());
    println!("  Max:  {:.4}", gt.max());

    let our_output = vec![0.12, 0.19, 0.31, 0.38, 0.52, 0.58, 0.71, 0.79, 0.91, 0.98];
    let our = GroundTruth::from_slice(&our_output);

    let delta = Delta::compute(&our, &gt);
    println!("\nDelta analysis:");
    println!("  Mean delta: {:.4}", delta.mean_delta());
    println!("  Std delta:  {:.4}", delta.std_delta());
    println!("  Percent:    {:.2}%", delta.percent());
    println!("  Sign flip:  {}", delta.is_sign_flipped());
    if let Some(cos) = delta.cosine() {
        println!("  Cosine sim: {cos:.4}");
    }
}

/// Part 6: Cosine Similarity Tolerance
fn demo_cosine_similarity() {
    println!("\n--- Part 6: Cosine Similarity Tolerance ---\n");

    let vec_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let vec_b = vec![1.1, 1.9, 3.1, 3.9, 5.1];
    let vec_c = vec![-1.0, -2.0, -3.0, -4.0, -5.0];

    println!("Cosine similarity comparisons:");
    println!(
        "  vec_a vs vec_b (similar):   {:.4}",
        Delta::cosine_similarity(&vec_a, &vec_b)
    );
    println!(
        "  vec_a vs vec_c (opposite):  {:.4}",
        Delta::cosine_similarity(&vec_a, &vec_c)
    );
    println!(
        "  vec_a vs vec_a (identical): {:.4}",
        Delta::cosine_similarity(&vec_a, &vec_a)
    );
}

/// Part 7: KL Divergence for Probability Distributions
fn demo_kl_divergence() {
    println!("\n--- Part 7: KL Divergence ---\n");

    let p = vec![0.25, 0.25, 0.25, 0.25]; // Uniform
    let q = vec![0.5, 0.25, 0.125, 0.125]; // Skewed

    println!("KL divergence (distribution comparison):");
    println!("  Uniform vs Uniform: {:.4}", Delta::kl_divergence(&p, &p));
    println!("  Uniform vs Skewed:  {:.4}", Delta::kl_divergence(&p, &q));
}

/// Part 8: Real-World Whisper Pipeline Example
fn demo_whisper_pipeline() {
    println!("\n--- Part 8: Whisper Pipeline (Real-World) ---\n");

    let pipeline = Pipeline::builder("whisper-tiny")
        .stage("mel")
        .ground_truth_stats(-0.215, 0.448)
        .tolerance(Tolerance::percent(5.0))
        .description("Log-mel spectrogram (80 mel bins)")
        .build_stage()
        .stage("encoder_out")
        .ground_truth_stats(0.0, 0.8)
        .tolerance(Tolerance::percent(10.0))
        .description("Encoder final output")
        .build_stage()
        .stage("decoder_logits")
        .ground_truth_stats(0.0, 15.0)
        .tolerance(Tolerance::percent(15.0))
        .description("Decoder output logits")
        .build_stage()
        .stage("probs")
        .ground_truth_stats(0.0001, 0.01)
        .tolerance(Tolerance::percent(20.0))
        .description("Softmax probabilities")
        .build_stage()
        .build()
        .expect("Failed to build Whisper pipeline");

    let report = pipeline.verify(|stage| match stage {
        "mel" => Some(GroundTruth::from_stats(-0.220, 0.445)),
        "encoder_out" => Some(GroundTruth::from_stats(0.01, 0.78)),
        "decoder_logits" => Some(GroundTruth::from_stats(-0.5, 14.2)),
        "probs" => Some(GroundTruth::from_stats(0.00012, 0.009)),
        _ => None,
    });

    println!("Whisper-tiny pipeline verification:");
    print_report(&report);
}

fn print_summary() {
    println!("\n=== Summary ===\n");
    println!("Pipeline verification enables:");
    println!("  1. Stage-by-stage ground truth comparison");
    println!("  2. Multiple tolerance types (percent, stats, cosine, KL)");
    println!("  3. Jidoka: Stop on first failure (or continue for full analysis)");
    println!("  4. Automatic diagnosis (sign flips, distribution shifts)");
    println!("  5. Visual reporting with pass/fail/skip status");
    println!("\nUse cases:");
    println!("  - ML model porting (PyTorch -> Rust)");
    println!("  - Quantization validation");
    println!("  - CI/CD regression testing");
    println!("  - Audio/vision pipeline debugging");
    println!("\n=== Done ===");
}

/// Print a verification report with colored output
fn print_report(report: &VerifyReport) {
    println!("{}", report.summary());
    println!();

    for result in report.results() {
        let status = result.status();
        let icon = status.icon();
        let color = status.color();
        let reset = "\x1b[0m";

        print!("  {color}{icon}{reset} {}", result.name());

        if let Some(delta) = result.delta() {
            print!(" (delta: {:.2}%)", delta.percent());
        }

        if status == StageStatus::Skipped {
            print!(" [skipped due to prior failure]");
        }

        println!();
    }
}
