//! F3: Context Lifecycle Chaos + F4: Quantization Parity Oracle

use trueno_cuda_edge::{
    lifecycle_chaos::{
        ChaosScenario, ContextLeakDetector, DestructionOrdering, LifecycleChaosConfig,
    },
    quant_oracle::{check_values_parity, BoundaryValueGenerator, ParityConfig, QuantFormat},
};

// ============================================================================
// F3: Context Lifecycle Chaos — GPU Context Management
// ============================================================================

mod lifecycle_chaos_tests {
    use super::*;

    /// Test all 8 chaos scenarios for GPU context lifecycle.
    #[test]
    fn chaos_scenario_coverage() {
        let scenarios = ChaosScenario::all();
        assert_eq!(scenarios.len(), 8);

        // Verify critical scenarios are present
        assert!(scenarios.contains(&ChaosScenario::DoubleDestroy));
        assert!(scenarios.contains(&ChaosScenario::UseAfterDestroy));
        assert!(scenarios.contains(&ChaosScenario::LeakedContext));
        assert!(scenarios.contains(&ChaosScenario::ContextExhaustion));
    }

    /// Test default chaos configuration.
    #[test]
    fn default_chaos_config() {
        let config = LifecycleChaosConfig::default();

        assert_eq!(config.scenarios.len(), 8);
        assert_eq!(config.max_contexts, 64);
        assert!(config.capture_memory_snapshots);
    }

    /// Test destruction ordering validation.
    #[test]
    fn destruction_ordering_patterns() {
        // LIFO (reverse) — correct for CUDA
        let lifo = DestructionOrdering::new(vec![2, 1, 0]);
        assert!(lifo.is_reverse());
        assert!(!lifo.is_forward());

        // FIFO (forward) — may cause issues
        let fifo = DestructionOrdering::new(vec![0, 1, 2]);
        assert!(fifo.is_forward());
        assert!(!fifo.is_reverse());

        // Random — neither
        let random = DestructionOrdering::new(vec![1, 0, 2]);
        assert!(!random.is_reverse());
        assert!(!random.is_forward());
    }

    /// Test memory leak detection with tolerance.
    #[test]
    fn leak_detection_with_tolerance() {
        let detector = ContextLeakDetector::new();

        // Within 1 MB tolerance: no leak
        let report = detector.analyze(100_000_000, 100_500_000);
        assert!(!report.has_leaks());

        // Above 1 MB tolerance: leak detected
        let report = detector.analyze(100_000_000, 102_000_000);
        assert!(report.has_leaks());
        assert!(report.total_leaked_bytes() > 0);
    }

    /// Test custom tolerance for strict leak detection.
    #[test]
    fn custom_leak_tolerance() {
        let strict = ContextLeakDetector::with_tolerance(1024); // 1 KB

        let report = strict.analyze(1000, 3000);
        assert!(report.has_leaks()); // 2000 > 1024
    }
}

// ============================================================================
// F4: Quantization Parity Oracle — SIMD/GPU Numerical Accuracy
// ============================================================================

mod quant_oracle_tests {
    use super::*;

    /// Test format-specific tolerances for trueno's quantization.
    #[test]
    fn quantization_format_tolerances() {
        // 4-bit quantization: ~5% tolerance
        assert!((QuantFormat::Q4K.tolerance() - 0.05).abs() < f64::EPSILON);

        // 5-bit quantization: ~2% tolerance
        assert!((QuantFormat::Q5K.tolerance() - 0.02).abs() < f64::EPSILON);

        // 6-bit quantization: ~1% tolerance
        assert!((QuantFormat::Q6K.tolerance() - 0.01).abs() < f64::EPSILON);

        // 8-bit quantization: ~0.5% tolerance
        assert!((QuantFormat::Q8_0.tolerance() - 0.005).abs() < f64::EPSILON);

        // F16: ~0.1% tolerance
        assert!((QuantFormat::F16.tolerance() - 0.001).abs() < f64::EPSILON);

        // F32: machine epsilon
        assert!((QuantFormat::F32.tolerance() - f64::EPSILON).abs() < f64::EPSILON);
    }

    /// Test boundary value generation for edge cases.
    #[test]
    fn boundary_value_generation() {
        let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);

        // Universal boundaries
        let universal = gen.universal_boundaries();
        assert!(universal.contains(&0.0));
        assert!(universal.iter().any(|v| v.is_nan()));
        assert!(universal.iter().any(|v| v.is_infinite()));

        // Format-specific boundaries
        let format_bounds = gen.format_boundaries();
        // Q4K has 16 levels x 2 (+-) = 32 values
        assert_eq!(format_bounds.len(), 32);

        // All boundaries
        let all = gen.all_boundaries();
        assert_eq!(all.len(), universal.len() + format_bounds.len());
    }

    /// Test parity checking for CPU/GPU comparison.
    #[test]
    fn parity_check_cpu_gpu() {
        let config = ParityConfig::new(QuantFormat::Q4K);

        // Identical values: pass
        let cpu = vec![1.0, 2.0, 3.0];
        let gpu = vec![1.0, 2.0, 3.0];
        let report = check_values_parity(&cpu, &gpu, &config);
        assert!(report.passed());

        // Small difference within tolerance: pass
        let gpu_close = vec![1.01, 2.01, 3.01];
        let report = check_values_parity(&cpu, &gpu_close, &config);
        assert!(report.passed());

        // Large difference: fail
        let gpu_far = vec![1.0, 2.5, 3.0];
        let report = check_values_parity(&cpu, &gpu_far, &config);
        assert!(!report.passed());
        assert_eq!(report.violations.len(), 1);
    }

    /// Test NaN handling in parity checks.
    #[test]
    fn parity_nan_handling() {
        let config = ParityConfig::new(QuantFormat::F32);

        // NaN vs NaN: OK (both are NaN)
        let cpu = vec![f64::NAN];
        let gpu = vec![f64::NAN];
        let report = check_values_parity(&cpu, &gpu, &config);
        assert!(report.passed());

        // NaN vs number: violation
        let gpu_num = vec![1.0];
        let report = check_values_parity(&cpu, &gpu_num, &config);
        assert!(!report.passed());
    }
}
