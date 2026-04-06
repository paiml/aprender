use super::*;

#[test]
fn test_severity_names() {
    assert_eq!(AlertSeverity::Info.name(), "INFO");
    assert_eq!(AlertSeverity::Warning.name(), "WARNING");
    assert_eq!(AlertSeverity::Critical.name(), "CRITICAL");
}

#[test]
fn test_severity_colors() {
    assert_eq!(AlertSeverity::Info.color(), "#36a64f");
    assert_eq!(AlertSeverity::Critical.color(), "#ff0000");
}

#[test]
fn test_severity_parsing() {
    assert_eq!(AlertSeverity::parse("INFO"), Some(AlertSeverity::Info));
    assert_eq!(
        AlertSeverity::parse("warning"),
        Some(AlertSeverity::Warning)
    );
    assert_eq!(AlertSeverity::parse("invalid"), None);
}

#[test]
fn test_channel_names() {
    assert_eq!(AlertChannel::Console.name(), "console");
    assert_eq!(AlertChannel::slack("url").name(), "slack");
    assert_eq!(AlertChannel::pagerduty("key").name(), "pagerduty");
}

#[test]
fn test_alert_creation() {
    let alert = Alert::new("Test Alert", "This is a test", AlertSeverity::Warning)
        .with_source("cpu_temp")
        .with_value(85.0)
        .with_threshold(80.0);

    assert_eq!(alert.title, "Test Alert");
    assert_eq!(alert.severity, AlertSeverity::Warning);
    assert_eq!(alert.value, Some(85.0));
}

#[test]
fn test_alert_json() {
    let alert = Alert::new("Test", "Message", AlertSeverity::Info).with_source("test_source");

    let json = alert.to_json();
    assert!(json.contains("\"title\":\"Test\""));
    assert!(json.contains("\"severity\":\"INFO\""));
}

#[test]
fn test_slack_json() {
    let alert = Alert::new("Alert", "Body", AlertSeverity::Critical);
    let json = alert.to_slack_json();

    assert!(json.contains("\"color\":\"#ff0000\""));
    assert!(json.contains("\"title\":\"Alert\""));
}

#[test]
fn test_router_dry_run() {
    let config = AlertRouterConfig::default();
    let mut router = AlertRouter::new(config).with_dry_run(true);

    let alert = Alert::new("Test", "Message", AlertSeverity::Info);
    let results = router.send(alert);

    assert!(!results.is_empty());
    assert!(results[0].success);
}

#[test]
fn test_router_history() {
    let mut router = AlertRouter::default();

    let alert = Alert::new("Test", "Message", AlertSeverity::Info);
    router.send(alert);

    assert_eq!(router.alert_count(), 1);
    assert!(!router.history().is_empty());
}

#[test]
fn test_message_template() {
    let template = MessageTemplate::new("{severity}: {title} - {message}");
    let alert = Alert::new("CPU High", "Temperature exceeded", AlertSeverity::Critical);

    let rendered = template.render(&alert);
    assert_eq!(rendered, "CRITICAL: CPU High - Temperature exceeded");
}

#[test]
fn test_alert_from_anomaly() {
    let alert = alert_from_anomaly("cpu_temp", 95.0, 80.0, AlertSeverity::Critical);

    assert!(alert.title.contains("cpu_temp"));
    assert_eq!(alert.value, Some(95.0));
    assert_eq!(alert.threshold, Some(80.0));
}

#[test]
fn test_delivery_result() {
    let success = DeliveryResult::success("slack", 50);
    assert!(success.success);
    assert_eq!(success.duration_ms, 50);

    let failure = DeliveryResult::failure("email", "connection refused");
    assert!(!failure.success);
    assert!(failure.error.is_some());
}
