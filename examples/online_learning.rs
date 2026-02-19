#![allow(clippy::disallowed_methods)]
//! Online Learning and Dynamic Retraining Example
//!
//! Demonstrates incremental model updates without full retraining using:
//! - Online Linear Regression with SGD
//! - Online Logistic Regression for streaming classification
//! - Drift detection with ADWIN
//! - Curriculum learning for progressive training
//! - Knowledge distillation from teacher to student
//!
//! Run with: `cargo run --example online_learning`

use aprender::online::{
    corpus::{CorpusBuffer, CorpusBufferConfig, EvictionPolicy, Sample, SampleSource},
    curriculum::{CurriculumScheduler, DifficultyScorer, FeatureNormScorer, LinearCurriculum},
    distillation::{
        softmax_temperature, DistillationConfig, DistillationLoss, DEFAULT_TEMPERATURE,
    },
    drift::{DriftDetector, DriftDetectorFactory, ADWIN, DDM},
    orchestrator::{ObserveResult, OrchestratorBuilder},
    LearningRateDecay, OnlineLearner, OnlineLearnerConfig, OnlineLinearRegression,
    OnlineLogisticRegression,
};

fn main() {
    println!("=== Online Learning and Dynamic Retraining ===\n");

    // Part 1: Online Linear Regression
    online_linear_regression_demo();

    // Part 2: Online Logistic Regression
    online_logistic_regression_demo();

    // Part 3: Drift Detection
    drift_detection_demo();

    // Part 4: Corpus Management
    corpus_management_demo();

    // Part 5: Curriculum Learning
    curriculum_learning_demo();

    // Part 6: Knowledge Distillation
    knowledge_distillation_demo();

    // Part 7: RetrainOrchestrator
    retrain_orchestrator_demo();

    println!("\n=== Online Learning Complete! ===");
}

fn online_linear_regression_demo() {
    println!("--- Part 1: Online Linear Regression ---\n");

    // Create online linear regression model
    let config = OnlineLearnerConfig {
        learning_rate: 0.01,
        decay: LearningRateDecay::InverseSqrt,
        l2_reg: 0.001,
        ..Default::default()
    };
    let mut model = OnlineLinearRegression::with_config(2, config);

    println!("Training incrementally on streaming data (y = 2*x1 + 3*x2 + 1)...\n");

    // Simulate streaming data
    let samples = vec![
        (vec![1.0, 0.0], 3.0), // 2*1 + 3*0 + 1 = 3
        (vec![0.0, 1.0], 4.0), // 2*0 + 3*1 + 1 = 4
        (vec![1.0, 1.0], 6.0), // 2*1 + 3*1 + 1 = 6
        (vec![2.0, 1.0], 8.0), // 2*2 + 3*1 + 1 = 8
        (vec![1.0, 2.0], 9.0), // 2*1 + 3*2 + 1 = 9
    ];

    println!(
        "{:>6} {:>8} {:>8} {:>10} {:>12}",
        "Sample", "x1", "x2", "y", "Loss"
    );
    println!("{}", "-".repeat(50));

    for (i, (x, y)) in samples.iter().enumerate() {
        let loss = model
            .partial_fit(x, &[*y], None)
            .expect("partial_fit should succeed");
        println!(
            "{:>6} {:>8.1} {:>8.1} {:>10.1} {:>12.4}",
            i + 1,
            x[0],
            x[1],
            y,
            loss
        );
    }

    // Train more to converge
    for _ in 0..200 {
        for (x, y) in &samples {
            model
                .partial_fit(x, &[*y], None)
                .expect("convergence training partial_fit should succeed");
        }
    }

    println!("\nAfter convergence:");
    println!("  Weights: {:?}", model.weights());
    println!("  Bias: {:.4}", model.bias());
    println!("  Samples seen: {}", model.n_samples_seen());
    println!("  Learning rate: {:.6}", model.current_learning_rate());

    // Test predictions
    println!("\nPredictions:");
    let pred1 = model
        .predict_one(&[1.0, 1.0])
        .expect("predict_one should succeed for valid input");
    let pred2 = model
        .predict_one(&[2.0, 2.0])
        .expect("predict_one should succeed for valid input");
    println!("  f(1, 1) = {:.2} (expected: 6.0)", pred1);
    println!("  f(2, 2) = {:.2} (expected: 11.0)", pred2);
    println!();
}

fn online_logistic_regression_demo() {
    println!("--- Part 2: Online Logistic Regression ---\n");

    let config = OnlineLearnerConfig {
        learning_rate: 0.5,
        decay: LearningRateDecay::Constant,
        ..Default::default()
    };
    let mut model = OnlineLogisticRegression::with_config(2, config);

    println!("Training binary classifier incrementally...\n");

    // XOR-like pattern (simplified for online learning)
    let samples = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![1.0, 1.0], 1.0),
        (vec![0.5, 0.5], 1.0),
        (vec![0.1, 0.1], 0.0),
    ];

    // Train multiple passes
    for _ in 0..100 {
        for (x, y) in &samples {
            model
                .partial_fit(x, &[*y], None)
                .expect("logistic partial_fit should succeed");
        }
    }

    println!("Predictions after training:");
    println!("{:>8} {:>8} {:>10} {:>12}", "x1", "x2", "P(y=1)", "Class");
    println!("{}", "-".repeat(45));

    for (x, _) in &samples {
        let prob = model
            .predict_proba_one(x)
            .expect("predict_proba_one should succeed");
        let class = if prob > 0.5 { 1 } else { 0 };
        println!("{:>8.1} {:>8.1} {:>10.3} {:>12}", x[0], x[1], prob, class);
    }
    println!();
}

fn drift_detection_demo() {
    println!("--- Part 3: Drift Detection ---\n");

    // DDM (Drift Detection Method)
    println!("DDM (for sudden drift):");
    let mut ddm = DDM::new();

    // Simulate good predictions
    for _ in 0..50 {
        ddm.add_element(false); // correct prediction
    }
    println!("  After 50 correct: {:?}", ddm.detected_change());

    // Simulate sudden drift (many errors)
    for _ in 0..50 {
        ddm.add_element(true); // wrong prediction
    }
    let stats = ddm.stats();
    println!("  After 50 errors: {:?}", stats.status);
    println!("  Error rate: {:.2}%", stats.error_rate * 100.0);

    // ADWIN (Adaptive Windowing) - recommended default
    println!("\nADWIN (for gradual/sudden drift - RECOMMENDED):");
    let mut adwin = ADWIN::with_delta(0.1);

    // Low error rate period
    for _ in 0..100 {
        adwin.add_element(false);
    }
    println!("  After 100 correct predictions:");
    println!("    Status: {:?}", adwin.detected_change());
    println!("    Window size: {}", adwin.window_size());
    println!("    Mean error: {:.3}", adwin.mean());

    // Concept drift occurs
    for _ in 0..100 {
        adwin.add_element(true);
    }
    println!("\n  After 100 wrong predictions (concept drift):");
    println!("    Status: {:?}", adwin.detected_change());
    println!("    Window size: {}", adwin.window_size());
    println!("    Mean error: {:.3}", adwin.mean());

    // Factory for creating detectors
    println!("\nDrift detector factory:");
    let detector = DriftDetectorFactory::recommended();
    println!(
        "  Recommended (ADWIN) created: samples={}",
        detector.stats().n_samples
    );
    println!();
}

fn corpus_management_demo() {
    println!("--- Part 4: Corpus Management ---\n");

    // Create corpus with deduplication
    let config = CorpusBufferConfig {
        max_size: 5,
        policy: EvictionPolicy::Reservoir,
        deduplicate: true,
        seed: Some(42),
    };
    let mut buffer = CorpusBuffer::with_config(config);

    println!("Adding samples with deduplication (max_size=5):");

    // Add samples
    for i in 0..10 {
        let sample = Sample::with_source(
            vec![i as f64, (i * 2) as f64],
            vec![(i * 3) as f64],
            if i < 5 {
                SampleSource::Synthetic
            } else {
                SampleSource::Production
            },
        );
        let added = buffer.add(sample);
        println!(
            "  Sample {}: added={}, buffer_size={}",
            i,
            added,
            buffer.len()
        );
    }

    // Try duplicate
    let dup = Sample::new(vec![0.0, 0.0], vec![0.0]);
    let added = buffer.add(dup);
    println!("\nDuplicate sample: added={}", added);

    // Export dataset
    let (features, targets, n_samples, n_features) = buffer.to_dataset();
    println!("\nExported dataset:");
    println!("  Samples: {}", n_samples);
    println!("  Features: {}", n_features);
    println!(
        "  Total values: {} features, {} targets",
        features.len(),
        targets.len()
    );

    // Source breakdown
    let synthetic = buffer.samples_by_source(&SampleSource::Synthetic);
    let production = buffer.samples_by_source(&SampleSource::Production);
    println!(
        "  Synthetic: {}, Production: {}",
        synthetic.len(),
        production.len()
    );
    println!();
}

fn curriculum_learning_demo() {
    println!("--- Part 5: Curriculum Learning ---\n");

    // Linear curriculum with 5 stages
    let mut curriculum = LinearCurriculum::new(5);

    println!("Linear curriculum (5 stages):");
    println!(
        "{:>8} {:>10} {:>12} {:>10}",
        "Stage", "Progress", "Threshold", "Complete"
    );
    println!("{}", "-".repeat(45));

    for i in 0..7 {
        println!(
            "{:>8} {:>9.0}% {:>12.2} {:>10}",
            i,
            curriculum.stage() * 100.0,
            curriculum.current_threshold(),
            curriculum.is_complete()
        );
        curriculum.advance();
    }

    // Difficulty scoring
    println!("\nDifficulty scoring (feature norm):");
    let scorer = FeatureNormScorer::new();

    let samples = vec![
        vec![0.5, 0.5], // Easy: small norm
        vec![2.0, 2.0], // Medium
        vec![5.0, 5.0], // Hard: large norm
    ];

    println!("{:>12} {:>12} {:>12}", "Features", "Norm", "Difficulty");
    println!("{}", "-".repeat(40));

    for sample in &samples {
        let difficulty = scorer.score(sample, 0.0);
        println!(
            "{:>12?} {:>12.3} {:>12}",
            sample,
            difficulty,
            if difficulty < 2.0 {
                "Easy"
            } else if difficulty < 4.0 {
                "Medium"
            } else {
                "Hard"
            }
        );
    }
    println!();
}

fn knowledge_distillation_demo() {
    println!("--- Part 6: Knowledge Distillation ---\n");

    println!("Temperature scaling (T={}):", DEFAULT_TEMPERATURE);

    // Teacher's confident logits
    let teacher_logits = vec![1.0, 3.0, 0.5];

    println!("\nTeacher logits: {:?}", teacher_logits);

    // Standard softmax (T=1)
    let hard = softmax_temperature(&teacher_logits, 1.0);
    println!(
        "Hard targets (T=1): [{:.3}, {:.3}, {:.3}]",
        hard[0], hard[1], hard[2]
    );

    // Soft targets (T=3, default)
    let soft = softmax_temperature(&teacher_logits, 3.0);
    println!(
        "Soft targets (T=3): [{:.3}, {:.3}, {:.3}]",
        soft[0], soft[1], soft[2]
    );

    // Very soft (T=10)
    let very_soft = softmax_temperature(&teacher_logits, 10.0);
    println!(
        "Very soft (T=10): [{:.3}, {:.3}, {:.3}]",
        very_soft[0], very_soft[1], very_soft[2]
    );

    println!("\nNote: Higher temperature reveals 'dark knowledge' -");
    println!("the relationships between non-target classes.");

    // Distillation loss
    let config = DistillationConfig {
        temperature: DEFAULT_TEMPERATURE,
        alpha: 0.7, // 70% distillation, 30% hard labels
        learning_rate: 0.01,
        l2_reg: 0.0,
    };
    let loss = DistillationLoss::with_config(config);

    let student_logits = vec![0.5, 2.0, 0.8];
    let hard_labels = vec![0.0, 1.0, 0.0]; // True class is 1

    let distill_loss = loss
        .compute(&student_logits, &teacher_logits, &hard_labels)
        .expect("distillation loss computation should succeed");
    println!("\nDistillation loss: {:.4}", distill_loss);
    println!("  (alpha=0.7: 70% KL divergence + 30% cross-entropy)");
    println!();
}

fn retrain_orchestrator_demo() {
    println!("--- Part 7: RetrainOrchestrator ---\n");

    // Build orchestrator with custom settings
    let model = OnlineLinearRegression::new(2);
    let mut orchestrator = OrchestratorBuilder::new(model, 2)
        .min_samples(10)
        .max_buffer_size(100)
        .incremental_updates(true)
        .curriculum_learning(true)
        .curriculum_stages(3)
        .learning_rate(0.01)
        .adwin_delta(0.1)  // Sensitive drift detection
        .build();

    println!("Orchestrator created with:");
    println!("  Min samples: {}", orchestrator.config().min_samples);
    println!("  Max buffer: {}", orchestrator.config().max_buffer_size);
    println!(
        "  Curriculum learning: {}",
        orchestrator.config().curriculum_learning
    );

    // Simulate streaming predictions
    println!("\nSimulating streaming predictions:");
    println!(
        "{:>6} {:>10} {:>12} {:>10}",
        "Step", "Target", "Prediction", "Result"
    );
    println!("{}", "-".repeat(45));

    // Good predictions first
    for i in 0..5 {
        let features = vec![i as f64, (i * 2) as f64];
        let target = vec![(i * 3) as f64];
        let prediction = vec![(i * 3) as f64]; // Perfect prediction

        let result = orchestrator
            .observe(&features, &target, &prediction)
            .expect("orchestrator observe should succeed");
        println!(
            "{:>6} {:>10.1} {:>12.1} {:>10?}",
            i + 1,
            target[0],
            prediction[0],
            result
        );
    }

    // Simulate drift (bad predictions)
    for i in 5..15 {
        let features = vec![i as f64, (i * 2) as f64];
        let target = vec![1.0];
        let prediction = vec![0.0]; // Wrong prediction

        let result = orchestrator
            .observe(&features, &target, &prediction)
            .expect("orchestrator observe should succeed during drift");
        if i == 5 || i == 14 || matches!(result, ObserveResult::Retrained | ObserveResult::Warning)
        {
            println!(
                "{:>6} {:>10.1} {:>12.1} {:>10?}",
                i + 1,
                target[0],
                prediction[0],
                result
            );
        }
    }

    let stats = orchestrator.stats();
    println!("\nOrchestrator stats:");
    println!("  Samples observed: {}", stats.samples_observed);
    println!("  Retrain count: {}", stats.retrain_count);
    println!("  Buffer size: {}", stats.buffer_size);
    println!("  Drift status: {:?}", stats.drift_status);
    println!();
}
