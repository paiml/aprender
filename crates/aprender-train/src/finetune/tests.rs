//! Integration tests for finetune module

use super::*;
use crate::finetune::corpus::SampleMetadata;

#[test]
fn test_module_exports() {
    // Verify all public types are accessible
    let _device = ComputeDevice::Cpu;
    let _info = DeviceInfo::cpu_info();
    let _qa = PopperianQA::new();
    let _grade = QAGrade::F;
    let _corpus = TestGenCorpus::new();
    let _config = ReproducibilityConfig::default();
}

#[test]
fn test_end_to_end_mock_evaluation() {
    // Create mock corpus
    let corpus = TestGenCorpus::mock(10, 2, 2);
    assert_eq!(corpus.len(), 14);

    // Create evaluator
    let eval = TestEvaluator::default().without_mutation();

    // Evaluate first sample
    let sample = &corpus.train[0];
    let result = eval.evaluate(&sample.function, &sample.unit_tests);

    // Check result structure
    assert!(!result.function.is_empty());
    assert!(!result.generated_tests.is_empty());
}

#[test]
fn test_popperian_with_corpus() {
    let corpus = TestGenCorpus::mock(100, 10, 10);
    let stats = corpus.stats();

    // Create QA checklist based on corpus stats
    let mut qa = PopperianQA::new();

    // Environment is locked (we're in a test)
    qa.r4_environment_locked = true;

    // Corpus has samples
    if stats.total_samples > 0 {
        qa.v3_edge_cases_present = true;
    }

    // Score should be > 0 now
    assert!(qa.score() > 0);
}

#[test]
fn test_device_selection_for_qlora() {
    let device = ComputeDevice::auto_detect();

    // Either device should work for QLoRA
    match device {
        ComputeDevice::Cpu => {
            let info = DeviceInfo::cpu_info();
            assert!(info.memory_gb > 0.0);
        }
        ComputeDevice::Cuda { device_id } => {
            if let Some(info) = DeviceInfo::cuda_info(device_id) {
                assert!(info.sufficient_for_qlora());
            }
        }
        ComputeDevice::Wgpu { adapter_index: _ } => {
            // wgpu device detected — valid for QLoRA
        }
    }
}

#[test]
fn test_reproducibility_round_trip() {
    let config = ReproducibilityConfig::with_seed(42);
    let lock = ExperimentLock::new("test-exp")
        .with_reproducibility(config)
        .with_config_checksum("sha256:abc");

    // Serialize and deserialize
    let yaml = serde_yaml::to_string(&lock).expect("operation should succeed");
    let restored: ExperimentLock =
        serde_yaml::from_str(&yaml).expect("lock acquisition should succeed");

    assert_eq!(restored.experiment_id, "test-exp");
    assert_eq!(restored.reproducibility.seed, 42);
    assert_eq!(restored.config_checksum, "sha256:abc");
}

#[test]
fn test_corpus_format_consistency() {
    let sample = TestGenSample {
        function: "pub fn foo() -> i32 { 42 }".into(),
        unit_tests: "#[test] fn test_foo() { assert_eq!(foo(), 42); }".into(),
        property_tests: Some(
            "proptest! { #[test] fn prop(x: i32) { assert!(foo() > 0); } }".into(),
        ),
        metadata: SampleMetadata {
            crate_name: Some("example".into()),
            complexity: Some(1),
            has_generics: false,
            has_lifetimes: false,
            is_async: false,
        },
    };

    let prompt = TestGenCorpus::format_prompt(&sample);
    let target = TestGenCorpus::format_target(&sample);

    // Prompt should have system and user tags
    assert!(prompt.contains("<|im_start|>system"));
    assert!(prompt.contains("<|im_start|>user"));
    assert!(prompt.contains("<|im_start|>assistant"));

    // Target should have both test types and end tag
    assert!(target.contains("#[test]"));
    assert!(target.contains("proptest!"));
    assert!(target.ends_with("<|im_end|>"));
}

#[test]
fn test_eval_metrics_thresholds() {
    // Below minimum
    let low = EvalMetrics {
        compile_rate: 0.50,
        test_pass_rate: 0.50,
        mutation_score: 0.30,
        ..Default::default()
    };
    assert!(!low.meets_minimum());
    assert!(!low.meets_target());
    assert!(!low.meets_stretch());

    // At minimum
    let min = EvalMetrics {
        compile_rate: 0.85,
        test_pass_rate: 0.80,
        mutation_score: 0.60,
        ..Default::default()
    };
    assert!(min.meets_minimum());
    assert!(!min.meets_target());

    // At target
    let target = EvalMetrics {
        compile_rate: 0.92,
        test_pass_rate: 0.88,
        mutation_score: 0.72,
        branch_coverage_delta: 12.0,
        ..Default::default()
    };
    assert!(target.meets_target());
    assert!(!target.meets_stretch());

    // At stretch
    let stretch = EvalMetrics {
        compile_rate: 0.97,
        test_pass_rate: 0.95,
        mutation_score: 0.80,
        branch_coverage_delta: 18.0,
        ..Default::default()
    };
    assert!(stretch.meets_stretch());
}

#[test]
fn test_qa_report_format() {
    let mut qa = PopperianQA::new();
    qa.c1_parses_as_rust = true;
    qa.c2_type_checks = true;

    let report = qa.report();

    // Should have markdown structure
    assert!(report.starts_with("# Popperian"));
    assert!(report.contains("**Score:**"));
    assert!(report.contains("**Grade:**"));
    assert!(report.contains("[x] C1")); // Checked item
    assert!(report.contains("[ ] R1")); // Unchecked item
}
