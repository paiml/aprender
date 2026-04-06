use super::*;

#[test]
fn test_event_type_names() {
    assert_eq!(EventType::Interrupt.name(), "interrupt");
    assert_eq!(EventType::DiskIo.name(), "disk_io");
    assert_eq!(EventType::Network.name(), "network");
}

#[test]
fn test_perf_sample_spike() {
    let sample = PerformanceSample::new(0.0, 20.0, 100.0);
    assert!(sample.is_spike(15.0));
    assert!(!sample.is_spike(25.0));
}

#[test]
fn test_correlation_strength() {
    let result = CorrelationResult {
        event_type: EventType::Interrupt,
        pearson_r: 0.6,
        sample_count: 100,
        is_significant: true,
        optimal_lag: 0.0,
    };

    assert_eq!(result.strength(), "strong");
    assert!(result.has_correlation());
}

#[test]
fn test_isolation_action_names() {
    assert_eq!(IsolationAction::CpuPin.name(), "cpu_pin");
    assert_eq!(IsolationAction::None.name(), "none");
}

#[test]
fn test_system_snapshot() {
    let snapshot = SystemSnapshot::new(1.0)
        .with_irq("timer", 1000)
        .with_irq("disk", 500)
        .with_disk_io(1_000_000.0)
        .with_network(5000.0)
        .with_load_average(1.5, 1.0, 0.8);

    assert_eq!(snapshot.total_irqs(), 1500);
    assert_eq!(snapshot.disk_io_bytes_per_sec, 1_000_000.0);
}

#[test]
fn test_analyzer_add_samples() {
    let mut analyzer = CorrelationAnalyzer::new();

    analyzer.add_perf_sample(PerformanceSample::new(0.0, 5.0, 100.0));
    analyzer.add_perf_sample(PerformanceSample::new(1.0, 10.0, 120.0));

    assert_eq!(analyzer.perf_sample_count(), 2);

    analyzer.add_event_sample(EventSample::new(EventType::Interrupt, 0.0, 1000.0));
    assert_eq!(analyzer.event_sample_count(), 1);
}

#[test]
fn test_analyzer_correlation() {
    let mut analyzer = CorrelationAnalyzer::new();

    // Add correlated samples
    for i in 0..20 {
        let t = i as f64;
        analyzer.add_perf_sample(PerformanceSample::new(t, 5.0 + i as f64, 100.0));
        analyzer.add_event_sample(EventSample::new(
            EventType::Interrupt,
            t,
            100.0 + i as f64 * 10.0,
        ));
    }

    let result = analyzer.correlate_events(EventType::Interrupt).unwrap();
    assert!(result.pearson_r > 0.5);
}

#[test]
fn test_recommend_isolation() {
    let analyzer = CorrelationAnalyzer::new();
    let rec = analyzer.recommend_isolation();

    // With no data, should recommend none
    assert_eq!(rec.action, IsolationAction::None);
}
