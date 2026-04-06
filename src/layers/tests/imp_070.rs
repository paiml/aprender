
// ============================================================================
// Phase 20: Error Recovery & Graceful Degradation (M29) - EXTREME TDD
// ============================================================================

/// IMP-070: Error Recovery Strategy
/// Target: Automatic retry with exponential backoff, GPU fallback, error classification
#[test]
#[cfg(feature = "gpu")]
fn test_imp_070_error_recovery_strategy() {
    use crate::gpu::{ErrorClassification, ErrorRecoveryStrategy, RecoveryAction};
    use std::time::Duration;

    // Test 1: Create recovery strategy with config
    let strategy = ErrorRecoveryStrategy::new()
        .with_max_retries(3)
        .with_base_delay(Duration::from_millis(100))
        .with_max_delay(Duration::from_secs(5))
        .with_jitter(0.1);

    assert_eq!(
        strategy.max_retries(),
        3,
        "IMP-070: Max retries should be 3"
    );

    // Test 2: Error classification
    let transient_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout");
    let classification = strategy.classify_error(&transient_err);
    assert_eq!(
        classification,
        ErrorClassification::Transient,
        "IMP-070: Timeout should be transient"
    );

    let fatal_err = std::io::Error::new(std::io::ErrorKind::InvalidData, "bad data");
    let classification = strategy.classify_error(&fatal_err);
    assert_eq!(
        classification,
        ErrorClassification::Fatal,
        "IMP-070: InvalidData should be fatal"
    );

    // Test 3: Recovery action for transient error
    let action = strategy.determine_action(&transient_err, 0);
    assert!(
        matches!(action, RecoveryAction::Retry { .. }),
        "IMP-070: Transient should retry"
    );

    // Test 4: Exponential backoff delay calculation
    let delay_0 = strategy.calculate_delay(0);
    let delay_1 = strategy.calculate_delay(1);
    let delay_2 = strategy.calculate_delay(2);
    assert!(delay_1 > delay_0, "IMP-070: Delay should increase");
    assert!(
        delay_2 > delay_1,
        "IMP-070: Delay should increase exponentially"
    );

    // Test 5: Max retries exceeded
    let action = strategy.determine_action(&transient_err, 4);
    assert!(
        matches!(action, RecoveryAction::Fail),
        "IMP-070: Should fail after max retries"
    );

    // Test 6: GPU fallback action
    let gpu_err = std::io::Error::other("GPU unavailable");
    let action = strategy.determine_action_with_fallback(&gpu_err, 0);
    assert!(
        matches!(action, RecoveryAction::FallbackToCpu),
        "IMP-070: GPU error should fallback to CPU"
    );
}

/// IMP-071: Graceful Degradation Modes
/// Target: GPU→CPU fallback, memory pressure response, context limiting
#[test]
#[cfg(feature = "gpu")]
fn test_imp_071_graceful_degradation() {
    use crate::gpu::{DegradationManager, DegradationMode, SystemLoad};

    // Test 1: Create degradation manager
    let mut manager = DegradationManager::new();
    assert_eq!(
        manager.current_mode(),
        DegradationMode::Normal,
        "IMP-071: Should start in Normal mode"
    );

    // Test 2: GPU unavailable triggers CPU fallback
    manager.set_gpu_available(false);
    assert_eq!(
        manager.current_mode(),
        DegradationMode::CpuFallback,
        "IMP-071: GPU unavailable should trigger CPU fallback"
    );

    // Test 3: Memory pressure reduces batch size
    manager.set_gpu_available(true);
    manager.update_memory_pressure(0.9); // 90% memory used
    let batch_size = manager.recommended_batch_size(8);
    assert!(
        batch_size < 8,
        "IMP-071: High memory pressure should reduce batch size"
    );

    // Test 4: System load affects context length
    let load = SystemLoad {
        cpu_percent: 95.0,
        memory_percent: 85.0,
        queue_depth: 100,
    };
    manager.update_system_load(load);
    let max_context = manager.recommended_max_context(4096);
    assert!(
        max_context < 4096,
        "IMP-071: High load should limit context length"
    );

    // Test 5: Quality vs latency tradeoff
    manager.set_latency_priority(true);
    assert_eq!(
        manager.current_mode(),
        DegradationMode::LowLatency,
        "IMP-071: Latency priority should set LowLatency mode"
    );

    // Test 6: Recovery to normal mode
    manager.set_gpu_available(true);
    manager.update_memory_pressure(0.3); // 30% memory used
    manager.set_latency_priority(false);
    let load = SystemLoad {
        cpu_percent: 20.0,
        memory_percent: 30.0,
        queue_depth: 5,
    };
    manager.update_system_load(load);
    assert_eq!(
        manager.current_mode(),
        DegradationMode::Normal,
        "IMP-071: Low load should restore Normal mode"
    );
}
