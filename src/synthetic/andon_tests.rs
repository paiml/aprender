use super::*;

// ============================================================================
// EXTREME TDD: AndonSeverity Tests
// ============================================================================

#[test]
fn test_andon_severity_display() {
    assert_eq!(format!("{}", AndonSeverity::Info), "INFO");
    assert_eq!(format!("{}", AndonSeverity::Warning), "WARNING");
    assert_eq!(format!("{}", AndonSeverity::Critical), "CRITICAL");
}

#[test]
fn test_andon_severity_equality() {
    assert_eq!(AndonSeverity::Info, AndonSeverity::Info);
    assert_ne!(AndonSeverity::Info, AndonSeverity::Warning);
}

#[test]
fn test_andon_severity_clone_copy() {
    let s = AndonSeverity::Critical;
    let s2 = s;
    assert_eq!(s, s2);
}

// ============================================================================
// EXTREME TDD: AndonEvent Tests
// ============================================================================

#[test]
fn test_high_rejection_rate_event() {
    let event = AndonEvent::HighRejectionRate {
        rate: 0.95,
        threshold: 0.90,
    };
    assert_eq!(event.severity(), AndonSeverity::Critical);
    assert!(event.should_halt());
    assert!(format!("{event}").contains("95.0%"));
}

#[test]
fn test_high_rejection_rate_warning() {
    // Just slightly over threshold - warning, not critical
    let event = AndonEvent::HighRejectionRate {
        rate: 0.91,
        threshold: 0.90,
    };
    assert_eq!(event.severity(), AndonSeverity::Warning);
    assert!(!event.should_halt());
}

#[test]
fn test_quality_drift_event_critical() {
    let event = AndonEvent::QualityDrift {
        current: 0.5,
        baseline: 0.8,
    };
    // 0.5 < 0.8 * 0.8 = 0.64, so critical
    assert_eq!(event.severity(), AndonSeverity::Critical);
    assert!(event.should_halt());
}

#[test]
fn test_quality_drift_event_warning() {
    let event = AndonEvent::QualityDrift {
        current: 0.7,
        baseline: 0.8,
    };
    // 0.7 >= 0.8 * 0.8 = 0.64, so warning
    assert_eq!(event.severity(), AndonSeverity::Warning);
    assert!(!event.should_halt());
}

#[test]
fn test_diversity_collapse_event() {
    let event = AndonEvent::DiversityCollapse {
        score: 0.05,
        minimum: 0.1,
    };
    assert_eq!(event.severity(), AndonSeverity::Warning);
    assert!(!event.should_halt());
    assert!(format!("{event}").contains("collapse"));
}

#[test]
fn test_generation_failure_event() {
    let event = AndonEvent::GenerationFailure {
        message: "out of memory".to_string(),
    };
    assert_eq!(event.severity(), AndonSeverity::Critical);
    assert!(event.should_halt());
    assert!(format!("{event}").contains("out of memory"));
}

#[test]
fn test_andon_event_clone() {
    let event = AndonEvent::HighRejectionRate {
        rate: 0.95,
        threshold: 0.90,
    };
    let event2 = event.clone();
    assert_eq!(event, event2);
}

// ============================================================================
// EXTREME TDD: DefaultAndon Tests
// ============================================================================

#[test]
fn test_default_andon_new() {
    let andon = DefaultAndon::new();
    assert!(!andon.is_halted());
}

#[test]
fn test_default_andon_halt_on_critical() {
    let andon = DefaultAndon::new();
    let event = AndonEvent::HighRejectionRate {
        rate: 0.99,
        threshold: 0.90,
    };
    andon.on_event(&event);
    assert!(andon.is_halted());
}

#[test]
fn test_default_andon_no_halt_on_warning() {
    let andon = DefaultAndon::new();
    let event = AndonEvent::DiversityCollapse {
        score: 0.05,
        minimum: 0.1,
    };
    andon.on_event(&event);
    assert!(!andon.is_halted());
}

#[test]
fn test_default_andon_reset() {
    let andon = DefaultAndon::new();
    let event = AndonEvent::GenerationFailure {
        message: "test".to_string(),
    };
    andon.on_event(&event);
    assert!(andon.is_halted());
    andon.reset();
    assert!(!andon.is_halted());
}

#[test]
fn test_default_andon_clone() {
    let andon1 = DefaultAndon::new();
    let andon2 = andon1.clone();
    // Both share same halted state via Arc
    let event = AndonEvent::GenerationFailure {
        message: "x".to_string(),
    };
    andon1.on_event(&event);
    assert!(andon2.is_halted());
}

// ============================================================================
// EXTREME TDD: TestAndon Tests
// ============================================================================

#[test]
fn test_test_andon_collects_events() {
    let andon = TestAndon::new();
    andon.on_high_rejection(0.95, 0.90);
    andon.on_quality_drift(0.5, 0.8);

    let events = andon.events();
    assert_eq!(events.len(), 2);
    assert_eq!(andon.count_high_rejection(), 1);
    assert_eq!(andon.count_quality_drift(), 1);
}

#[test]
fn test_test_andon_was_halted() {
    let andon = TestAndon::new();
    assert!(!andon.was_halted());

    andon.on_event(&AndonEvent::GenerationFailure {
        message: "x".to_string(),
    });
    assert!(andon.was_halted());
}

#[test]
fn test_test_andon_clear() {
    let andon = TestAndon::new();
    andon.on_high_rejection(0.95, 0.90);
    assert_eq!(andon.events().len(), 1);

    andon.clear();
    assert!(andon.events().is_empty());
    assert!(!andon.was_halted());
}

// ============================================================================
// EXTREME TDD: AndonConfig Tests
// ============================================================================

#[test]
fn test_andon_config_default() {
    let config = AndonConfig::default();
    assert!(config.enabled);
    assert!((config.rejection_threshold - 0.90).abs() < f32::EPSILON);
    assert!(config.quality_baseline.is_none());
    assert!((config.diversity_minimum - 0.1).abs() < f32::EPSILON);
}

#[test]
fn test_andon_config_builder() {
    let config = AndonConfig::new()
        .with_enabled(false)
        .with_rejection_threshold(0.85)
        .with_quality_baseline(0.7)
        .with_diversity_minimum(0.2);

    assert!(!config.enabled);
    assert!((config.rejection_threshold - 0.85).abs() < f32::EPSILON);
    assert!((config.quality_baseline.expect("baseline should be set") - 0.7).abs() < f32::EPSILON);
    assert!((config.diversity_minimum - 0.2).abs() < f32::EPSILON);
}

#[test]
fn test_andon_config_clamping() {
    let config = AndonConfig::new()
        .with_rejection_threshold(1.5)
        .with_quality_baseline(-0.5)
        .with_diversity_minimum(2.0);

    assert!((config.rejection_threshold - 1.0).abs() < f32::EPSILON);
    assert!((config.quality_baseline.expect("baseline should be set") - 0.0).abs() < f32::EPSILON);
    assert!((config.diversity_minimum - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_exceeds_rejection_threshold() {
    let config = AndonConfig::new().with_rejection_threshold(0.90);

    assert!(!config.exceeds_rejection_threshold(0.85));
    assert!(!config.exceeds_rejection_threshold(0.90));
    assert!(config.exceeds_rejection_threshold(0.91));
}

#[test]
fn test_exceeds_rejection_threshold_disabled() {
    let config = AndonConfig::new()
        .with_enabled(false)
        .with_rejection_threshold(0.90);

    assert!(!config.exceeds_rejection_threshold(0.99));
}

#[test]
fn test_has_quality_drift() {
    let config = AndonConfig::new().with_quality_baseline(0.8);

    // 10% tolerance: 0.8 * 0.9 = 0.72
    assert!(!config.has_quality_drift(0.75)); // Above threshold
    assert!(config.has_quality_drift(0.70)); // Below threshold
}

#[test]
fn test_has_quality_drift_no_baseline() {
    let config = AndonConfig::new();
    assert!(config.quality_baseline.is_none());
    assert!(!config.has_quality_drift(0.1)); // No baseline, no drift
}

#[test]
fn test_has_quality_drift_disabled() {
    let config = AndonConfig::new()
        .with_enabled(false)
        .with_quality_baseline(0.8);

    assert!(!config.has_quality_drift(0.1));
}

#[test]
fn test_has_diversity_collapse() {
    let config = AndonConfig::new().with_diversity_minimum(0.1);

    assert!(!config.has_diversity_collapse(0.15));
    assert!(!config.has_diversity_collapse(0.1));
    assert!(config.has_diversity_collapse(0.05));
}

#[test]
fn test_has_diversity_collapse_disabled() {
    let config = AndonConfig::new()
        .with_enabled(false)
        .with_diversity_minimum(0.1);

    assert!(!config.has_diversity_collapse(0.01));
}

#[test]
fn test_andon_config_clone() {
    let c1 = AndonConfig::new().with_rejection_threshold(0.85);
    let c2 = c1.clone();
    assert_eq!(c1, c2);
}

#[test]
fn test_andon_config_debug() {
    let config = AndonConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("AndonConfig"));
    assert!(debug.contains("rejection_threshold"));
}
