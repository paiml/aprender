use super::*;

#[test]
fn test_escalation_thresholds_default() {
    let thresholds = EscalationThresholds::default();
    assert!((thresholds.cv_threshold - 15.0).abs() < 0.01);
    assert!((thresholds.efficiency_threshold - 25.0).abs() < 0.01);
}

#[test]
fn test_escalation_reason_description() {
    assert!(!EscalationReason::CvExceeded.description().is_empty());
    assert!(!EscalationReason::EfficiencyLow.description().is_empty());
}

#[test]
fn test_syscall_breakdown_dominant() {
    let mut breakdown = SyscallBreakdown::new();
    breakdown.mmap_us = 100;
    breakdown.futex_us = 500;
    breakdown.read_us = 200;
    breakdown.total_us = 1000;

    assert_eq!(breakdown.dominant_syscall(), "futex");
}

#[test]
fn test_syscall_breakdown_compute() {
    let mut breakdown = SyscallBreakdown::new();
    breakdown.mmap_us = 100;
    breakdown.futex_us = 200;
    breakdown.total_us = 1000;

    assert_eq!(breakdown.compute_us(), 700);
}

#[test]
fn test_should_trace_cv() {
    let escalation = TracingEscalation::default();
    assert!(escalation.should_trace(15.1, 50.0));
    assert!(!escalation.should_trace(14.9, 50.0));
}

#[test]
fn test_should_trace_efficiency() {
    let escalation = TracingEscalation::default();
    assert!(escalation.should_trace(10.0, 24.9));
    assert!(!escalation.should_trace(10.0, 25.1));
}

#[test]
fn test_escalation_reason() {
    let escalation = TracingEscalation::default();

    assert_eq!(escalation.escalation_reason(16.0, 20.0), Some(EscalationReason::Both));
    assert_eq!(escalation.escalation_reason(16.0, 50.0), Some(EscalationReason::CvExceeded));
    assert_eq!(escalation.escalation_reason(10.0, 20.0), Some(EscalationReason::EfficiencyLow));
    assert_eq!(escalation.escalation_reason(10.0, 50.0), None);
}
