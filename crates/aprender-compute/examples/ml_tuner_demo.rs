#![allow(clippy::disallowed_methods)]
//! ML Tuner Demo - Learned Kernel Selection and Throughput Prediction
//!
//! This example demonstrates the ML-based tuner for ComputeBrick optimization.
//! See: docs/specifications/ml-tuner-bricks.md
//!
//! Run with default (heuristic models):
//!   cargo run --example ml_tuner_demo
//!
//! Run with RandomForest models (requires ml-tuner feature):
//!   cargo run --example ml_tuner_demo --features ml-tuner

use trueno::tuner::{BrickTuner, KernelClassifier, QuantType, ThroughputRegressor, TunerFeatures};

/// Create the standard test features for Qwen2.5-Coder-1.5B on RTX 4090.
fn make_demo_features() -> TunerFeatures {
    TunerFeatures::builder()
        .model_params_b(1.5)
        .hidden_dim(1536)
        .num_layers(28)
        .num_heads(12)
        .batch_size(4)
        .seq_len(512)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0)
        .gpu_sm_count(128)
        .cuda_graphs(true)
        .build()
}

/// Section 1: Feature vector construction and validation.
fn demo_feature_construction(features: &TunerFeatures) {
    println!("1. TunerFeatures (DIM=42 vector)");
    println!("   -----------------------------");
    println!("   Model: Qwen2.5-Coder-1.5B (Q4_K_M)");
    println!("   GPU: RTX 4090 (1000 GB/s, 128 SMs, 24GB VRAM)");
    println!("   Batch size: M=4");
    println!("   Sequence length: 512");
    println!("   CUDA graphs: enabled");
    println!();

    match features.validate() {
        Ok(()) => println!("   Feature validation: PASSED"),
        Err(e) => println!("   Feature validation: FAILED - {}", e),
    }

    let vec = features.to_vector();
    println!("   Feature vector length: {} (expected: 42)", vec.len());
    println!(
        "   Sample features: model_params_b={:.3}, batch_size_norm={:.3}, gpu_mem_bw_norm={:.3}",
        vec[0], vec[6], vec[35]
    );
    println!();
}

/// Section 2: Throughput prediction with batch size comparison.
fn demo_throughput_prediction(features: &TunerFeatures) {
    println!("2. Throughput Prediction");
    println!("   ----------------------");

    let regressor = ThroughputRegressor::new();
    let prediction = regressor.predict(features);

    println!("   Predicted throughput: {:.1} tok/s", prediction.predicted_tps);
    println!("   Confidence: {:.1}%", prediction.confidence * 100.0);
    println!("   Top contributing features:");
    for (name, importance) in prediction.top_features.iter().take(3) {
        println!("     - {}: {:.1}%", name, importance * 100.0);
    }
    println!();

    println!("   Batch size comparison:");
    for m in [1, 2, 4, 8] {
        let m_features = TunerFeatures::builder()
            .model_params_b(1.5)
            .hidden_dim(1536)
            .batch_size(m)
            .quant_type(QuantType::Q4K)
            .gpu_mem_bw_gbs(1000.0)
            .cuda_graphs(true)
            .build();
        let pred = regressor.predict(&m_features);
        println!(
            "     M={}: {:.1} tok/s (conf: {:.0}%)",
            m,
            pred.predicted_tps,
            pred.confidence * 100.0
        );
    }
    println!();
}

/// Section 3: Kernel selection by batch size.
fn demo_kernel_selection(features: &TunerFeatures) {
    println!("3. Kernel Selection");
    println!("   -----------------");

    let classifier = KernelClassifier::new();
    let recommendation = classifier.predict(features);

    println!("   Recommended kernel: {:?}", recommendation.top_kernel);
    println!("   Confidence: {:.1}%", recommendation.confidence * 100.0);
    println!("   Alternatives:");
    for (kernel, conf) in recommendation.alternatives.iter().take(3) {
        println!("     - {:?}: {:.1}%", kernel, conf * 100.0);
    }
    println!();

    println!("   Kernel selection by batch size:");
    for m in [1, 2, 4, 8] {
        let m_features = TunerFeatures::builder()
            .model_params_b(1.5)
            .batch_size(m)
            .quant_type(QuantType::Q4K)
            .cuda_graphs(m == 1)
            .build();
        let rec = classifier.predict(&m_features);
        println!("     M={}: {:?}", m, rec.top_kernel);
    }
    println!();
}

/// Section 4: Roofline model analysis.
fn demo_roofline_model() {
    println!("4. Roofline Model (Physical Limits)");
    println!("   ---------------------------------");

    let regressor = ThroughputRegressor::new();
    println!("   Theoretical max throughput (RTX 4090, M=4):");
    for (name, params_b, quant) in [
        ("0.5B Q4_K", 0.5, QuantType::Q4K),
        ("1.5B Q4_K", 1.5, QuantType::Q4K),
        ("7B Q4_K", 7.0, QuantType::Q4K),
        ("7B Q6_K", 7.0, QuantType::Q6K),
        ("32B Q4_K", 32.0, QuantType::Q4K),
    ] {
        let f = TunerFeatures::builder()
            .model_params_b(params_b)
            .batch_size(4)
            .quant_type(quant)
            .gpu_mem_bw_gbs(1000.0)
            .build();
        let pred = regressor.predict(&f);
        println!("     {}: {:.0} tok/s (roofline-clamped)", name, pred.predicted_tps);
    }
    println!();
}

/// Section 5: Full tuner recommendations.
fn demo_full_tuner(features: &TunerFeatures) {
    println!("5. Full Tuner Recommendations");
    println!("   ---------------------------");

    let tuner = BrickTuner::new();
    let full_rec = tuner.recommend(features);

    println!("   Throughput: {:.1} tok/s", full_rec.throughput.predicted_tps);
    println!("   Best kernel: {:?}", full_rec.kernel.top_kernel);
    println!("   Experiment suggestions:");
    for suggestion in full_rec.suggested_experiments.iter().take(3) {
        println!("     - {}", suggestion);
    }
    println!();
}

/// Section 6: RandomForest models (ml-tuner showcase, SHOWCASE-BRICK-001).
///
/// The SIMD `trueno` tuner produces `TunerFeatures` vectors; this section feeds them
/// directly into `aprender::RandomForestRegressor`/`RandomForestClassifier`. aprender is
/// a **dev-dependency** of `aprender-compute`, so the compute lib itself never depends on
/// the ML layer — the showcase lives entirely here at the example layer, above both.
fn demo_random_forest(features: &TunerFeatures) {
    #[cfg(feature = "ml-tuner")]
    {
        use aprender::primitives::{Matrix, Vector};
        use aprender::tree::{RandomForestClassifier, RandomForestRegressor};

        println!("6. RandomForest Models (ml-tuner showcase)");
        println!("   --------------------------------------");

        let dim = TunerFeatures::DIM;
        let n = 100usize;

        // Regressor: trueno features -> predicted throughput (tok/s).
        let mut xr = Vec::with_capacity(n * dim);
        let mut yr = Vec::with_capacity(n);
        for i in 0..n {
            let batch = 1 + (i % 8) as u32;
            let f = TunerFeatures::builder()
                .model_params_b(1.5)
                .batch_size(batch)
                .quant_type(QuantType::Q4K)
                .gpu_mem_bw_gbs(1000.0)
                .cuda_graphs(batch == 1)
                .build();
            xr.extend(f.to_vector());
            yr.push(200.0 + (batch as f32) * 80.0 + (i as f32 * 0.5));
        }
        println!("   Generated {n} regression samples");
        match Matrix::from_vec(n, dim, xr) {
            Ok(x_mat) => {
                let mut rf = RandomForestRegressor::new(100);
                match rf.fit(&x_mat, &Vector::from_vec(yr)) {
                    Ok(()) => {
                        let fx = Matrix::from_vec(1, dim, features.to_vector().to_vec())
                            .expect("single-row feature matrix");
                        let pred = rf.predict(&fx);
                        println!(
                            "   RF regressor: SUCCESS — M=4 prediction {:.1} tok/s",
                            pred.as_slice()[0]
                        );
                    }
                    Err(e) => println!("   RF regressor: FAILED - {e}"),
                }
            }
            Err(e) => println!("   RF regressor matrix: FAILED - {e}"),
        }

        // Classifier: trueno features -> kernel-type label (3=Batched, 2=Vectorized).
        let mut xc = Vec::with_capacity(n * dim);
        let mut yc: Vec<usize> = Vec::with_capacity(n);
        for i in 0..n {
            let batch = 1 + (i % 8) as u32;
            let f = TunerFeatures::builder()
                .model_params_b(1.5)
                .batch_size(batch)
                .quant_type(QuantType::Q4K)
                .build();
            xc.extend(f.to_vector());
            yc.push(if batch >= 4 { 3 } else { 2 });
        }
        match Matrix::from_vec(n, dim, xc) {
            Ok(xc_mat) => {
                let mut rf = RandomForestClassifier::new(50);
                match rf.fit(&xc_mat, &yc) {
                    Ok(()) => {
                        let preds = rf.predict(&xc_mat);
                        let correct =
                            yc.iter().zip(preds.iter()).filter(|(a, b)| **a == **b).count();
                        println!(
                            "   RF classifier: SUCCESS — train accuracy {:.1}%",
                            correct as f32 / n as f32 * 100.0
                        );
                    }
                    Err(e) => println!("   RF classifier: FAILED - {e}"),
                }
            }
            Err(e) => println!("   RF classifier matrix: FAILED - {e}"),
        }
        println!();
    }

    #[cfg(not(feature = "ml-tuner"))]
    {
        let _ = features;
        println!("6. RandomForest Models");
        println!("   --------------------");
        println!("   [Disabled - enable with: --features ml-tuner]");
        println!();
    }
}

fn main() {
    println!("=== ML Tuner Demo ===\n");
    println!("ComputeBrick kernel selection and throughput prediction");
    println!("Reference: SHOWCASE-BRICK-001, Section 12\n");

    let features = make_demo_features();

    demo_feature_construction(&features);
    demo_throughput_prediction(&features);
    demo_kernel_selection(&features);
    demo_roofline_model();
    demo_full_tuner(&features);
    demo_random_forest(&features);

    println!("=== Demo Complete ===\n");
    println!("Key takeaways:");
    println!("  - TunerFeatures: 42-dimension vector for ML models");
    println!("  - Throughput prediction with roofline clamping (v1.1.0)");
    println!("  - Kernel selection: BatchedQ4K for M>=4, VectorizedQ4K otherwise");
    println!("  - RandomForest available with --features ml-tuner");
    println!();
    println!("Next steps:");
    println!("  cargo run --example quickstart              # Basic trueno usage");
    println!("  cargo run --example performance_demo        # SIMD benchmarks");
    println!("  cargo run --features gpu --example gpu_batch_demo  # GPU operations");
}
