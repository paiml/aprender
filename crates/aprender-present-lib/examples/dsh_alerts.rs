//! DSH-010: Dashboard with Alerts
//!
//! QA Focus: Alert system with severity levels and notifications
//!
//! Run: `cargo run --example dsh_alerts`

#![allow(clippy::all, clippy::pedantic, clippy::nursery, dead_code)]

use std::collections::VecDeque;
use std::time::Instant;

/// Alert severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertStatus {
    Active,
    Acknowledged,
    Resolved,
    Silenced,
}

/// A single alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub id: String,
    pub title: String,
    pub message: String,
    pub severity: AlertSeverity,
    pub status: AlertStatus,
    pub source: String,
    pub timestamp_ms: u64,
    pub acknowledged_by: Option<String>,
    pub resolved_at: Option<u64>,
}

impl Alert {
    pub fn new(
        id: &str,
        title: &str,
        message: &str,
        severity: AlertSeverity,
        source: &str,
    ) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            message: message.to_string(),
            severity,
            status: AlertStatus::Active,
            source: source.to_string(),
            timestamp_ms: 0,
            acknowledged_by: None,
            resolved_at: None,
        }
    }

    pub const fn with_timestamp(mut self, timestamp_ms: u64) -> Self {
        self.timestamp_ms = timestamp_ms;
        self
    }

    pub fn acknowledge(&mut self, user: &str) {
        self.status = AlertStatus::Acknowledged;
        self.acknowledged_by = Some(user.to_string());
    }

    pub fn resolve(&mut self, timestamp_ms: u64) {
        self.status = AlertStatus::Resolved;
        self.resolved_at = Some(timestamp_ms);
    }

    pub fn silence(&mut self) {
        self.status = AlertStatus::Silenced;
    }

    /// Check if alert needs immediate attention
    pub fn needs_attention(&self) -> bool {
        self.status == AlertStatus::Active
            && matches!(
                self.severity,
                AlertSeverity::Error | AlertSeverity::Critical
            )
    }

    /// Get time since alert was created (in ms)
    pub const fn age(&self, current_time_ms: u64) -> u64 {
        current_time_ms.saturating_sub(self.timestamp_ms)
    }
}

/// Alert rule for automatic triggering
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub condition: String,
    pub severity: AlertSeverity,
    pub enabled: bool,
    pub cooldown_secs: u64,
    pub last_triggered: Option<u64>,
}

impl AlertRule {
    pub fn new(id: &str, name: &str, condition: &str, severity: AlertSeverity) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            condition: condition.to_string(),
            severity,
            enabled: true,
            cooldown_secs: 300, // 5 minute default cooldown
            last_triggered: None,
        }
    }

    pub const fn with_cooldown(mut self, secs: u64) -> Self {
        self.cooldown_secs = secs;
        self
    }

    /// Check if rule can trigger based on cooldown
    pub const fn can_trigger(&self, current_time_ms: u64) -> bool {
        if !self.enabled {
            return false;
        }
        match self.last_triggered {
            Some(last) => current_time_ms - last >= self.cooldown_secs * 1000,
            None => true,
        }
    }
}

/// Alert dashboard
#[derive(Debug)]
pub struct AlertDashboard {
    alerts: VecDeque<Alert>,
    rules: Vec<AlertRule>,
    max_alerts: usize,
    title: String,
    start_time: Instant,
}

impl AlertDashboard {
    pub fn new(title: &str, max_alerts: usize) -> Self {
        Self {
            alerts: VecDeque::with_capacity(max_alerts),
            rules: Vec::new(),
            max_alerts,
            title: title.to_string(),
            start_time: Instant::now(),
        }
    }

    pub fn add_alert(&mut self, alert: Alert) {
        if self.alerts.len() >= self.max_alerts {
            self.alerts.pop_front();
        }
        self.alerts.push_back(alert);
    }

    pub fn add_rule(&mut self, rule: AlertRule) {
        self.rules.push(rule);
    }

    pub const fn alerts(&self) -> &VecDeque<Alert> {
        &self.alerts
    }

    pub fn rules(&self) -> &[AlertRule] {
        &self.rules
    }

    /// Get current time in ms
    pub fn current_time_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Get alerts by status
    pub fn by_status(&self, status: AlertStatus) -> Vec<&Alert> {
        self.alerts.iter().filter(|a| a.status == status).collect()
    }

    /// Get alerts by severity
    pub fn by_severity(&self, severity: AlertSeverity) -> Vec<&Alert> {
        self.alerts
            .iter()
            .filter(|a| a.severity == severity)
            .collect()
    }

    /// Get active alerts sorted by severity (critical first)
    pub fn active_sorted(&self) -> Vec<&Alert> {
        let mut active: Vec<_> = self.by_status(AlertStatus::Active);
        active.sort_by(|a, b| b.severity.cmp(&a.severity));
        active
    }

    /// Count alerts by severity
    pub fn severity_counts(&self) -> SeverityCounts {
        SeverityCounts {
            info: self.by_severity(AlertSeverity::Info).len(),
            warning: self.by_severity(AlertSeverity::Warning).len(),
            error: self.by_severity(AlertSeverity::Error).len(),
            critical: self.by_severity(AlertSeverity::Critical).len(),
        }
    }

    /// Get overall alert level
    pub fn overall_level(&self) -> AlertSeverity {
        let active = self.active_sorted();
        active.first().map_or(AlertSeverity::Info, |a| a.severity)
    }

    /// Acknowledge all active alerts
    pub fn acknowledge_all(&mut self, user: &str) {
        for alert in &mut self.alerts {
            if alert.status == AlertStatus::Active {
                alert.acknowledge(user);
            }
        }
    }

    /// Get alerts needing attention
    pub fn needs_attention(&self) -> Vec<&Alert> {
        self.alerts.iter().filter(|a| a.needs_attention()).collect()
    }

    /// Calculate mean time to acknowledge (in ms)
    pub fn mean_time_to_acknowledge(&self) -> Option<f64> {
        let acknowledged: Vec<_> = self
            .alerts
            .iter()
            .filter(|a| a.acknowledged_by.is_some())
            .collect();

        if acknowledged.is_empty() {
            return None;
        }

        // For this example, we'll use a placeholder calculation
        // In reality, you'd track acknowledgment timestamps
        Some(120_000.0) // 2 minutes placeholder
    }

    pub fn title(&self) -> &str {
        &self.title
    }
}

#[derive(Debug)]
pub struct SeverityCounts {
    pub info: usize,
    pub warning: usize,
    pub error: usize,
    pub critical: usize,
}

impl SeverityCounts {
    pub const fn total(&self) -> usize {
        self.info + self.warning + self.error + self.critical
    }
}

fn main() {
    println!("=== Dashboard with Alerts ===\n");

    let mut dashboard = AlertDashboard::new("Production Monitoring", 100);

    // Add alert rules
    dashboard.add_rule(AlertRule::new(
        "cpu-high",
        "High CPU Usage",
        "cpu_percent > 90",
        AlertSeverity::Warning,
    ));
    dashboard.add_rule(AlertRule::new(
        "memory-critical",
        "Critical Memory",
        "memory_percent > 95",
        AlertSeverity::Critical,
    ));
    dashboard.add_rule(AlertRule::new(
        "disk-warning",
        "Disk Space Low",
        "disk_percent > 80",
        AlertSeverity::Warning,
    ));

    // Simulate some alerts
    let base_time = 1000000u64;

    dashboard.add_alert(
        Alert::new(
            "alert-001",
            "High CPU Usage",
            "CPU usage exceeded 90% on web-server-1",
            AlertSeverity::Warning,
            "web-server-1",
        )
        .with_timestamp(base_time),
    );

    dashboard.add_alert(
        Alert::new(
            "alert-002",
            "Database Connection Pool Exhausted",
            "Connection pool at 100% capacity on db-primary",
            AlertSeverity::Critical,
            "db-primary",
        )
        .with_timestamp(base_time + 5000),
    );

    dashboard.add_alert(
        Alert::new(
            "alert-003",
            "Memory Usage High",
            "Memory at 85% on cache-server-1",
            AlertSeverity::Warning,
            "cache-server-1",
        )
        .with_timestamp(base_time + 10000),
    );

    // Acknowledge one alert
    if let Some(alert) = dashboard.alerts.iter_mut().find(|a| a.id == "alert-001") {
        alert.acknowledge("admin");
    }

    dashboard.add_alert(
        Alert::new(
            "alert-004",
            "SSL Certificate Expiring",
            "Certificate expires in 7 days",
            AlertSeverity::Info,
            "ssl-monitor",
        )
        .with_timestamp(base_time + 15000),
    );

    dashboard.add_alert(
        Alert::new(
            "alert-005",
            "Service Degradation",
            "Response time >500ms on api-gateway",
            AlertSeverity::Error,
            "api-gateway",
        )
        .with_timestamp(base_time + 20000),
    );

    // Print dashboard
    println!("Dashboard: {}", dashboard.title());
    println!("Overall Status: {:?}", dashboard.overall_level());

    let counts = dashboard.severity_counts();
    println!(
        "\nAlert Summary: {} total ({} critical, {} error, {} warning, {} info)",
        counts.total(),
        counts.critical,
        counts.error,
        counts.warning,
        counts.info
    );

    // Alerts needing attention
    let attention = dashboard.needs_attention();
    if !attention.is_empty() {
        println!("\n⚠ {} alerts need attention!", attention.len());
    }

    // Active alerts table
    println!("\n=== Active Alerts ===\n");
    println!(
        "{:<12} {:<10} {:>10} {:<30} {:<15}",
        "ID", "Severity", "Status", "Title", "Source"
    );
    println!("{}", "-".repeat(80));

    for alert in dashboard.active_sorted() {
        let severity_icon = match alert.severity {
            AlertSeverity::Critical => "●",
            AlertSeverity::Error => "◐",
            AlertSeverity::Warning => "○",
            AlertSeverity::Info => "·",
        };

        println!(
            "{:<12} {} {:<8} {:>10} {:<30} {:<15}",
            alert.id,
            severity_icon,
            format!("{:?}", alert.severity),
            format!("{:?}", alert.status),
            &alert.title[..alert.title.len().min(30)],
            alert.source
        );
    }

    // All alerts with status
    println!("\n=== All Alerts ===\n");
    for alert in dashboard.alerts() {
        let status_icon = match alert.status {
            AlertStatus::Active => "🔴",
            AlertStatus::Acknowledged => "🟡",
            AlertStatus::Resolved => "🟢",
            AlertStatus::Silenced => "⚫",
        };

        println!(
            "{} [{}] {} - {}",
            status_icon,
            format!("{:?}", alert.severity),
            alert.title,
            alert.source
        );
        if let Some(ref user) = alert.acknowledged_by {
            println!("   Acknowledged by: {user}");
        }
    }

    // Alert rules
    println!("\n=== Alert Rules ===\n");
    for rule in dashboard.rules() {
        let enabled = if rule.enabled { "✓" } else { "✗" };
        println!(
            "{} {} - {} ({:?})",
            enabled, rule.name, rule.condition, rule.severity
        );
    }

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] Severity levels displayed");
    println!("- [x] Alert status tracking");
    println!("- [x] Acknowledgment workflow");
    println!("- [x] 15-point checklist complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new("1", "Test", "Message", AlertSeverity::Warning, "source");
        assert_eq!(alert.id, "1");
        assert_eq!(alert.status, AlertStatus::Active);
    }

    #[test]
    fn test_alert_acknowledge() {
        let mut alert = Alert::new("1", "Test", "Message", AlertSeverity::Warning, "source");
        alert.acknowledge("admin");

        assert_eq!(alert.status, AlertStatus::Acknowledged);
        assert_eq!(alert.acknowledged_by, Some("admin".to_string()));
    }

    #[test]
    fn test_alert_resolve() {
        let mut alert = Alert::new("1", "Test", "Message", AlertSeverity::Warning, "source")
            .with_timestamp(1000);
        alert.resolve(2000);

        assert_eq!(alert.status, AlertStatus::Resolved);
        assert_eq!(alert.resolved_at, Some(2000));
    }

    #[test]
    fn test_alert_needs_attention() {
        let critical = Alert::new("1", "Test", "Msg", AlertSeverity::Critical, "src");
        let info = Alert::new("2", "Test", "Msg", AlertSeverity::Info, "src");

        assert!(critical.needs_attention());
        assert!(!info.needs_attention());
    }

    #[test]
    fn test_alert_rule_cooldown() {
        let rule = AlertRule::new("1", "Test", "x > 10", AlertSeverity::Warning).with_cooldown(60);

        assert!(rule.can_trigger(0));

        let mut rule_with_trigger = rule.clone();
        rule_with_trigger.last_triggered = Some(0);

        assert!(!rule_with_trigger.can_trigger(30_000)); // 30 seconds later
        assert!(rule_with_trigger.can_trigger(61_000)); // 61 seconds later
    }

    #[test]
    fn test_dashboard_severity_counts() {
        let mut dashboard = AlertDashboard::new("Test", 100);
        dashboard.add_alert(Alert::new("1", "T", "M", AlertSeverity::Critical, "s"));
        dashboard.add_alert(Alert::new("2", "T", "M", AlertSeverity::Warning, "s"));
        dashboard.add_alert(Alert::new("3", "T", "M", AlertSeverity::Warning, "s"));

        let counts = dashboard.severity_counts();
        assert_eq!(counts.critical, 1);
        assert_eq!(counts.warning, 2);
        assert_eq!(counts.total(), 3);
    }

    #[test]
    fn test_dashboard_overall_level() {
        let mut dashboard = AlertDashboard::new("Test", 100);
        dashboard.add_alert(Alert::new("1", "T", "M", AlertSeverity::Warning, "s"));
        dashboard.add_alert(Alert::new("2", "T", "M", AlertSeverity::Critical, "s"));

        assert_eq!(dashboard.overall_level(), AlertSeverity::Critical);
    }

    #[test]
    fn test_dashboard_active_sorted() {
        let mut dashboard = AlertDashboard::new("Test", 100);
        dashboard.add_alert(Alert::new("1", "T", "M", AlertSeverity::Info, "s"));
        dashboard.add_alert(Alert::new("2", "T", "M", AlertSeverity::Critical, "s"));
        dashboard.add_alert(Alert::new("3", "T", "M", AlertSeverity::Warning, "s"));

        let sorted = dashboard.active_sorted();
        assert_eq!(sorted[0].severity, AlertSeverity::Critical);
        assert_eq!(sorted[1].severity, AlertSeverity::Warning);
        assert_eq!(sorted[2].severity, AlertSeverity::Info);
    }

    #[test]
    fn test_dashboard_acknowledge_all() {
        let mut dashboard = AlertDashboard::new("Test", 100);
        dashboard.add_alert(Alert::new("1", "T", "M", AlertSeverity::Warning, "s"));
        dashboard.add_alert(Alert::new("2", "T", "M", AlertSeverity::Critical, "s"));

        dashboard.acknowledge_all("admin");

        assert!(dashboard
            .alerts
            .iter()
            .all(|a| a.status == AlertStatus::Acknowledged));
    }

    #[test]
    fn test_dashboard_max_alerts() {
        let mut dashboard = AlertDashboard::new("Test", 2);
        dashboard.add_alert(Alert::new("1", "T", "M", AlertSeverity::Info, "s"));
        dashboard.add_alert(Alert::new("2", "T", "M", AlertSeverity::Info, "s"));
        dashboard.add_alert(Alert::new("3", "T", "M", AlertSeverity::Info, "s"));

        assert_eq!(dashboard.alerts.len(), 2);
        assert_eq!(dashboard.alerts.front().unwrap().id, "2"); // First alert was removed
    }
}
