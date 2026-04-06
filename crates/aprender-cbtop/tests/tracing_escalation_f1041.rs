#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Tracing Escalation Tests (F1041-F1050)
//!
//! Popperian falsification criteria for tracing escalation per §35.2.

use cbtop::{
    EscalationReason, EscalationThresholds, OtlpSpanAttributes, SyscallBreakdown, TraceResult,
    TracingEscalation,
};

// ============================================================================
// F1041: CV threshold triggers escalation
// ============================================================================

#[test]
fn f1041_cv_above_threshold_triggers() {
    let escalation = TracingEscalation::default();
    // CV = 15.1% > 15% threshold → should trace
    assert!(escalation.should_trace(15.1, 50.0));
}

#[test]
fn f1041_cv_below_threshold_no_trigger() {
    let escalation = TracingEscalation::default();
    // CV = 14.9% < 15% threshold → should not trace
    assert!(!escalation.should_trace(14.9, 50.0));
}

#[test]
fn f1041_cv_exactly_at_threshold_no_trigger() {
    let escalation = TracingEscalation::default();
    // CV = 15.0% = threshold → should not trace (> not >=)
    assert!(!escalation.should_trace(15.0, 50.0));
}

// ============================================================================
// F1042: Efficiency threshold triggers escalation
// ============================================================================

#[test]
fn f1042_efficiency_below_threshold_triggers() {
    let escalation = TracingEscalation::default();
    // Efficiency = 24.9% < 25% threshold → should trace
    assert!(escalation.should_trace(10.0, 24.9));
}

#[test]
fn f1042_efficiency_above_threshold_no_trigger() {
    let escalation = TracingEscalation::default();
    // Efficiency = 25.1% > 25% threshold → should not trace
    assert!(!escalation.should_trace(10.0, 25.1));
}

#[test]
fn f1042_efficiency_exactly_at_threshold_no_trigger() {
    let escalation = TracingEscalation::default();
    // Efficiency = 25.0% = threshold → should not trace (< not <=)
    assert!(!escalation.should_trace(10.0, 25.0));
}

// ============================================================================
// F1043: Rate limiting prevents trace storm
// ============================================================================

#[test]
fn f1043_rate_limit_allows_up_to_max() {
    let thresholds = EscalationThresholds::default().with_rate_limit(3);
    let mut escalation = TracingEscalation::new(thresholds);

    let breakdown = SyscallBreakdown::default();

    // Should allow first 3
    for i in 0..3 {
        let result = escalation.try_trace(
            "brick",
            100,
            150,
            EscalationReason::CvExceeded,
            breakdown.clone(),
        );
        assert!(result.is_some(), "Trace {} should be allowed", i + 1);
    }
}

#[test]
fn f1043_rate_limit_blocks_excess() {
    let thresholds = EscalationThresholds::default().with_rate_limit(2);
    let mut escalation = TracingEscalation::new(thresholds);

    let breakdown = SyscallBreakdown::default();

    // First 2 allowed
    escalation.try_trace("brick", 100, 150, EscalationReason::CvExceeded, breakdown.clone());
    escalation.try_trace("brick", 100, 150, EscalationReason::CvExceeded, breakdown.clone());

    // 3rd should be blocked
    let result = escalation.try_trace("brick", 100, 150, EscalationReason::CvExceeded, breakdown);
    assert!(result.is_none());
}

#[test]
fn f1043_rate_limit_count_tracked() {
    let thresholds = EscalationThresholds::default().with_rate_limit(10);
    let mut escalation = TracingEscalation::new(thresholds);

    let breakdown = SyscallBreakdown::default();
    escalation.try_trace("brick", 100, 150, EscalationReason::CvExceeded, breakdown.clone());
    escalation.try_trace("brick", 100, 150, EscalationReason::CvExceeded, breakdown);

    assert_eq!(escalation.trace_count(), 2);
}

// ============================================================================
// F1044: Escalation reason recorded
// ============================================================================

#[test]
fn f1044_reason_cv_exceeded() {
    let escalation = TracingEscalation::default();
    let reason = escalation.escalation_reason(20.0, 50.0);
    assert_eq!(reason, Some(EscalationReason::CvExceeded));
}

#[test]
fn f1044_reason_efficiency_low() {
    let escalation = TracingEscalation::default();
    let reason = escalation.escalation_reason(10.0, 20.0);
    assert_eq!(reason, Some(EscalationReason::EfficiencyLow));
}

#[test]
fn f1044_reason_both() {
    let escalation = TracingEscalation::default();
    let reason = escalation.escalation_reason(20.0, 20.0);
    assert_eq!(reason, Some(EscalationReason::Both));
}

#[test]
fn f1044_reason_none() {
    let escalation = TracingEscalation::default();
    let reason = escalation.escalation_reason(10.0, 50.0);
    assert_eq!(reason, None);
}

#[test]
fn f1044_reason_descriptions_non_empty() {
    assert!(!EscalationReason::CvExceeded.description().is_empty());
    assert!(!EscalationReason::EfficiencyLow.description().is_empty());
    assert!(!EscalationReason::Both.description().is_empty());
    assert!(!EscalationReason::MemoryCliff.description().is_empty());
    assert!(!EscalationReason::GpuTransferOverhead.description().is_empty());
    assert!(!EscalationReason::Manual.description().is_empty());
}

// ============================================================================
// F1045: Syscall breakdown categorized
// ============================================================================

#[test]
fn f1045_mmap_categorized() {
    let mut breakdown = SyscallBreakdown::new();
    breakdown.add_syscall("mmap", 100);
    breakdown.add_syscall("munmap", 50);
    breakdown.add_syscall("mprotect", 30);
    breakdown.add_syscall("brk", 20);

    assert_eq!(breakdown.mmap_us, 200);
}

#[test]
fn f1045_futex_categorized() {
    let mut breakdown = SyscallBreakdown::new();
    breakdown.add_syscall("futex", 500);

    assert_eq!(breakdown.futex_us, 500);
}

#[test]
fn f1045_ioctl_categorized() {
    let mut breakdown = SyscallBreakdown::new();
    breakdown.add_syscall("ioctl", 300);

    assert_eq!(breakdown.ioctl_us, 300);
}

#[test]
fn f1045_read_categorized() {
    let mut breakdown = SyscallBreakdown::new();
    breakdown.add_syscall("read", 100);
    breakdown.add_syscall("pread64", 50);
    breakdown.add_syscall("readv", 25);

    assert_eq!(breakdown.read_us, 175);
}

#[test]
fn f1045_write_categorized() {
    let mut breakdown = SyscallBreakdown::new();
    breakdown.add_syscall("write", 100);
    breakdown.add_syscall("pwrite64", 50);
    breakdown.add_syscall("writev", 25);

    assert_eq!(breakdown.write_us, 175);
}

#[test]
fn f1045_other_categorized() {
    let mut breakdown = SyscallBreakdown::new();
    breakdown.add_syscall("unknown_syscall", 100);

    assert_eq!(breakdown.other_us, 100);
}

// ============================================================================
// F1046: Dominant syscall identified
// ============================================================================

#[test]
fn f1046_dominant_is_highest() {
    let mut breakdown = SyscallBreakdown::new();
    breakdown.mmap_us = 100;
    breakdown.futex_us = 500;
    breakdown.ioctl_us = 200;

    assert_eq!(breakdown.dominant_syscall(), "futex");
}

#[test]
fn f1046_dominant_with_empty_is_none() {
    let breakdown = SyscallBreakdown::new();
    assert_eq!(breakdown.dominant_syscall(), "none");
}

// ============================================================================
// F1047: Overhead percentage calculated
// ============================================================================

#[test]
fn f1047_overhead_calculation() {
    let mut breakdown = SyscallBreakdown::new();
    breakdown.mmap_us = 100;
    breakdown.futex_us = 200;
    breakdown.total_us = 1000;

    // Overhead = (100 + 200) / 1000 * 100 = 30%
    assert!((breakdown.syscall_overhead_percent() - 30.0).abs() < 0.1);
}

#[test]
fn f1047_compute_time_calculation() {
    let mut breakdown = SyscallBreakdown::new();
    breakdown.mmap_us = 100;
    breakdown.futex_us = 200;
    breakdown.total_us = 1000;

    // Compute = 1000 - 300 = 700
    assert_eq!(breakdown.compute_us(), 700);
}

// ============================================================================
// F1048: Threshold configuration works
// ============================================================================

#[test]
fn f1048_custom_cv_threshold() {
    let thresholds = EscalationThresholds::default().with_cv(20.0);
    let escalation = TracingEscalation::new(thresholds);

    // Should NOT trigger at 15% (below custom threshold)
    assert!(!escalation.should_trace(15.0, 50.0));
    // Should trigger at 21%
    assert!(escalation.should_trace(21.0, 50.0));
}

#[test]
fn f1048_custom_efficiency_threshold() {
    let thresholds = EscalationThresholds::default().with_efficiency(30.0);
    let escalation = TracingEscalation::new(thresholds);

    // Should NOT trigger at 25% (above custom threshold)
    assert!(!escalation.should_trace(10.0, 31.0));
    // Should trigger at 29%
    assert!(escalation.should_trace(10.0, 29.0));
}

// ============================================================================
// F1049: Trace result contains metrics
// ============================================================================

#[test]
fn f1049_trace_result_has_metrics() {
    let mut escalation = TracingEscalation::default();
    let mut breakdown = SyscallBreakdown::new();
    breakdown.total_us = 200;
    breakdown.mmap_us = 50;

    let result = escalation
        .try_trace("TestBrick", 100, 150, EscalationReason::CvExceeded, breakdown)
        .expect("Should create trace");

    assert_eq!(result.brick_name, "TestBrick");
    assert_eq!(result.budget_us, 100);
    assert_eq!(result.actual_us, 150);
    assert!(result.over_budget());
    assert_eq!(result.reason, EscalationReason::CvExceeded);
}

#[test]
fn f1049_efficiency_calculation() {
    let result = TraceResult {
        brick_name: "test".to_string(),
        budget_us: 100,
        actual_us: 200,
        reason: EscalationReason::CvExceeded,
        syscall_breakdown: SyscallBreakdown::default(),
        timestamp: std::time::Instant::now(),
    };

    // Efficiency = 100/200 * 100 = 50%
    assert!((result.efficiency() - 50.0).abs() < 0.1);
}

// ============================================================================
// F1050: OTLP span attributes set
// ============================================================================

#[test]
fn f1050_otlp_attributes_present() {
    let result = TraceResult {
        brick_name: "TestBrick".to_string(),
        budget_us: 100,
        actual_us: 150,
        reason: EscalationReason::CvExceeded,
        syscall_breakdown: SyscallBreakdown::default(),
        timestamp: std::time::Instant::now(),
    };

    let attrs = OtlpSpanAttributes::from_trace_result(&result);

    assert!(attrs.has_required_attributes());
    assert_eq!(attrs.attributes.get("brick.name"), Some(&"TestBrick".to_string()));
    assert_eq!(attrs.attributes.get("escalation.reason"), Some(&"cv_exceeded".to_string()));
}

#[test]
fn f1050_otlp_attributes_custom() {
    let result = TraceResult {
        brick_name: "test".to_string(),
        budget_us: 100,
        actual_us: 150,
        reason: EscalationReason::Manual,
        syscall_breakdown: SyscallBreakdown::default(),
        timestamp: std::time::Instant::now(),
    };

    let attrs =
        OtlpSpanAttributes::from_trace_result(&result).with_attribute("custom.key", "custom_value");

    assert_eq!(attrs.attributes.get("custom.key"), Some(&"custom_value".to_string()));
}

// ============================================================================
// Additional Coverage Tests
// ============================================================================

#[test]
fn test_escalation_history() {
    let mut escalation = TracingEscalation::default();
    let breakdown = SyscallBreakdown::default();

    escalation.try_trace("brick1", 100, 150, EscalationReason::CvExceeded, breakdown.clone());
    escalation.try_trace("brick2", 100, 150, EscalationReason::EfficiencyLow, breakdown);

    assert_eq!(escalation.history().len(), 2);
}

#[test]
fn test_clear_history() {
    let mut escalation = TracingEscalation::default();
    let breakdown = SyscallBreakdown::default();

    escalation.try_trace("brick", 100, 150, EscalationReason::CvExceeded, breakdown);
    assert!(!escalation.history().is_empty());

    escalation.clear_history();
    assert!(escalation.history().is_empty());
}

#[test]
fn test_gpu_transfer_threshold() {
    let escalation = TracingEscalation::default();

    assert!(escalation.should_trace_gpu_transfer(51.0));
    assert!(!escalation.should_trace_gpu_transfer(49.0));
}

#[test]
fn test_otlp_endpoint() {
    let escalation = TracingEscalation::default().with_otlp_endpoint("http://localhost:4317");

    // Just verify it compiles and doesn't panic
    assert!(escalation.thresholds().cv_threshold > 0.0);
}

#[test]
fn test_reason_otlp_values() {
    assert_eq!(EscalationReason::CvExceeded.otlp_value(), "cv_exceeded");
    assert_eq!(EscalationReason::EfficiencyLow.otlp_value(), "efficiency_low");
    assert_eq!(EscalationReason::Both.otlp_value(), "both");
    assert_eq!(EscalationReason::MemoryCliff.otlp_value(), "memory_cliff");
    assert_eq!(EscalationReason::GpuTransferOverhead.otlp_value(), "gpu_transfer_overhead");
    assert_eq!(EscalationReason::Manual.otlp_value(), "manual");
}

#[test]
fn test_syscall_breakdown_otlp_attributes() {
    let mut breakdown = SyscallBreakdown::new();
    breakdown.mmap_us = 100;
    breakdown.futex_us = 200;
    breakdown.total_us = 1000;

    let attrs = breakdown.as_otlp_attributes();

    assert_eq!(attrs.get("syscall.mmap_us"), Some(&100));
    assert_eq!(attrs.get("syscall.futex_us"), Some(&200));
    assert_eq!(attrs.get("syscall.compute_us"), Some(&700));
}
