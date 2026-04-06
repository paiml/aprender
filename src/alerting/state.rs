//! Alert state machine
//!
//! Implements Prometheus-compatible alert states:
//! - Inactive: Rule condition is false
//! - Pending: Rule condition is true, waiting for `for` duration
//! - Firing: Rule condition true for >= `for` duration
//! - Resolved: Rule was firing, now resolved

use std::time::{Duration, Instant};

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "info"),
            Severity::Warning => write!(f, "warning"),
            Severity::Critical => write!(f, "critical"),
        }
    }
}

/// Alert state (Prometheus-compatible)
#[derive(Debug, Clone, PartialEq, Default)]
pub enum AlertState {
    /// Rule condition is false
    #[default]
    Inactive,
    /// Rule condition is true, waiting for `for` duration
    Pending { since: Instant },
    /// Rule condition true for >= `for` duration
    Firing { since: Instant, notifications_sent: u32 },
    /// Rule was firing, now resolved
    Resolved { resolved_at: Instant },
}

impl AlertState {
    /// Check if alert is currently active (pending or firing)
    pub fn is_active(&self) -> bool {
        matches!(self, AlertState::Pending { .. } | AlertState::Firing { .. })
    }

    /// Check if alert is firing
    pub fn is_firing(&self) -> bool {
        matches!(self, AlertState::Firing { .. })
    }

    /// Get duration in current state
    pub fn duration(&self) -> Duration {
        match self {
            AlertState::Inactive => Duration::ZERO,
            AlertState::Pending { since } => since.elapsed(),
            AlertState::Firing { since, .. } => since.elapsed(),
            AlertState::Resolved { resolved_at } => resolved_at.elapsed(),
        }
    }

    /// Get state icon for display
    pub fn icon(&self) -> &'static str {
        match self {
            AlertState::Inactive => "○",
            AlertState::Pending { .. } => "◐",
            AlertState::Firing { .. } => "●",
            AlertState::Resolved { .. } => "◯",
        }
    }
}

/// Active alert instance
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    /// Rule name
    pub name: String,
    /// Current state
    pub state: AlertState,
    /// Severity level
    pub severity: Severity,
    /// Alert annotations
    pub annotations: AlertAnnotations,
    /// Current metric value that triggered the alert
    pub value: f64,
    /// Labels from the metric
    pub labels: std::collections::HashMap<String, String>,
}

impl ActiveAlert {
    /// Create a new active alert
    pub fn new(
        name: impl Into<String>,
        severity: Severity,
        value: f64,
        labels: std::collections::HashMap<String, String>,
    ) -> Self {
        Self {
            name: name.into(),
            state: AlertState::Pending { since: Instant::now() },
            severity,
            annotations: AlertAnnotations::default(),
            value,
            labels,
        }
    }

    /// Transition to firing state
    pub fn fire(&mut self) {
        if let AlertState::Pending { since } = self.state {
            self.state = AlertState::Firing { since, notifications_sent: 0 };
        }
    }

    /// Transition to resolved state
    pub fn resolve(&mut self) {
        self.state = AlertState::Resolved { resolved_at: Instant::now() };
    }

    /// Get duration in current state
    pub fn duration(&self) -> Duration {
        self.state.duration()
    }

    /// Get state icon
    pub fn state_icon(&self) -> &'static str {
        self.state.icon()
    }
}

/// Alert annotations (summary, description, etc.)
#[derive(Debug, Clone, Default)]
pub struct AlertAnnotations {
    /// Short summary
    pub summary: String,
    /// Detailed description
    pub description: String,
    /// Runbook URL
    pub runbook_url: Option<String>,
}

impl AlertAnnotations {
    /// Create with summary
    pub fn with_summary(summary: impl Into<String>) -> Self {
        Self { summary: summary.into(), ..Default::default() }
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set runbook URL
    pub fn runbook(mut self, url: impl Into<String>) -> Self {
        self.runbook_url = Some(url.into());
        self
    }

    /// Expand template variables in summary
    /// Supports: {{$value}}, {{$labels.name}}
    pub fn expand_summary(
        &self,
        value: f64,
        labels: &std::collections::HashMap<String, String>,
    ) -> String {
        let mut result = self.summary.clone();
        result = result.replace("{{$value}}", &format!("{:.4}", value));

        // Expand label references
        for (key, val) in labels {
            result = result.replace(&format!("{{{{$labels.{}}}}}", key), val);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_state_transitions() {
        let mut alert = ActiveAlert::new("test", Severity::Warning, 100.0, Default::default());

        assert!(matches!(alert.state, AlertState::Pending { .. }));
        assert!(alert.state.is_active());
        assert!(!alert.state.is_firing());

        alert.fire();
        assert!(matches!(alert.state, AlertState::Firing { .. }));
        assert!(alert.state.is_active());
        assert!(alert.state.is_firing());

        alert.resolve();
        assert!(matches!(alert.state, AlertState::Resolved { .. }));
        assert!(!alert.state.is_active());
    }

    #[test]
    fn test_alert_duration() {
        let alert = ActiveAlert::new("test", Severity::Warning, 100.0, Default::default());
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(alert.duration() >= std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_annotation_expansion() {
        let mut labels = std::collections::HashMap::new();
        labels.insert("instance".to_string(), "server1".to_string());

        let ann = AlertAnnotations::with_summary("Value is {{$value}} on {{$labels.instance}}");
        let expanded = ann.expand_summary(42.5, &labels);

        assert_eq!(expanded, "Value is 42.5000 on server1");
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(Severity::Info.to_string(), "info");
        assert_eq!(Severity::Warning.to_string(), "warning");
        assert_eq!(Severity::Critical.to_string(), "critical");
    }

    #[test]
    fn test_state_icons() {
        assert_eq!(AlertState::Inactive.icon(), "○");
        assert_eq!(AlertState::Pending { since: Instant::now() }.icon(), "◐");
        assert_eq!(AlertState::Firing { since: Instant::now(), notifications_sent: 0 }.icon(), "●");
        assert_eq!(AlertState::Resolved { resolved_at: Instant::now() }.icon(), "◯");
    }

    #[test]
    fn test_alert_state_default() {
        let state: AlertState = Default::default();
        assert!(matches!(state, AlertState::Inactive));
        assert!(!state.is_active());
        assert!(!state.is_firing());
    }

    #[test]
    fn test_alert_state_duration_all_variants() {
        // Inactive should have zero duration
        assert_eq!(AlertState::Inactive.duration(), Duration::ZERO);

        // Pending should have elapsed time
        let pending = AlertState::Pending { since: Instant::now() };
        std::thread::sleep(Duration::from_millis(5));
        assert!(pending.duration() >= Duration::from_millis(5));

        // Firing should have elapsed time
        let firing = AlertState::Firing { since: Instant::now(), notifications_sent: 5 };
        std::thread::sleep(Duration::from_millis(5));
        assert!(firing.duration() >= Duration::from_millis(5));

        // Resolved should have elapsed time
        let resolved = AlertState::Resolved { resolved_at: Instant::now() };
        std::thread::sleep(Duration::from_millis(5));
        assert!(resolved.duration() >= Duration::from_millis(5));
    }

    #[test]
    fn test_annotations_builder() {
        let ann = AlertAnnotations::with_summary("High CPU usage")
            .description("CPU usage exceeds 90%")
            .runbook("https://runbook.example.com/cpu");

        assert_eq!(ann.summary, "High CPU usage");
        assert_eq!(ann.description, "CPU usage exceeds 90%");
        assert_eq!(ann.runbook_url, Some("https://runbook.example.com/cpu".to_string()));
    }

    #[test]
    fn test_annotations_default() {
        let ann = AlertAnnotations::default();
        assert!(ann.summary.is_empty());
        assert!(ann.description.is_empty());
        assert!(ann.runbook_url.is_none());
    }

    #[test]
    fn test_active_alert_state_icon() {
        let mut alert = ActiveAlert::new("test", Severity::Critical, 99.0, Default::default());
        assert_eq!(alert.state_icon(), "◐"); // Pending

        alert.fire();
        assert_eq!(alert.state_icon(), "●"); // Firing

        alert.resolve();
        assert_eq!(alert.state_icon(), "◯"); // Resolved
    }

    #[test]
    fn test_fire_from_non_pending_state() {
        let mut alert = ActiveAlert::new("test", Severity::Warning, 50.0, Default::default());
        alert.fire();
        alert.fire(); // Should be a no-op when already firing
        assert!(matches!(alert.state, AlertState::Firing { .. }));
    }

    #[test]
    fn test_severity_eq_hash() {
        use std::collections::HashSet;

        let mut set: HashSet<Severity> = HashSet::new();
        set.insert(Severity::Info);
        set.insert(Severity::Warning);
        set.insert(Severity::Critical);

        assert!(set.contains(&Severity::Info));
        assert!(set.contains(&Severity::Warning));
        assert!(set.contains(&Severity::Critical));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_annotation_expand_summary_multiple_labels() {
        let mut labels = std::collections::HashMap::new();
        labels.insert("host".to_string(), "web-01".to_string());
        labels.insert("dc".to_string(), "us-east".to_string());
        labels.insert("env".to_string(), "prod".to_string());

        let ann = AlertAnnotations::with_summary(
            "Alert on {{$labels.host}} in {{$labels.dc}} ({{$labels.env}}): {{$value}}",
        );
        let expanded = ann.expand_summary(123.456, &labels);

        assert_eq!(expanded, "Alert on web-01 in us-east (prod): 123.4560");
    }

    #[test]
    fn test_active_alert_clone() {
        let mut labels = std::collections::HashMap::new();
        labels.insert("team".to_string(), "infra".to_string());

        let alert = ActiveAlert::new("test_alert", Severity::Critical, 95.0, labels);
        let cloned = alert.clone();

        assert_eq!(cloned.name, "test_alert");
        assert_eq!(cloned.severity, Severity::Critical);
        assert!((cloned.value - 95.0).abs() < f64::EPSILON);
        assert_eq!(cloned.labels.get("team"), Some(&"infra".to_string()));
    }

    #[test]
    fn test_alert_state_clone() {
        let state = AlertState::Firing { since: Instant::now(), notifications_sent: 3 };
        let cloned = state.clone();
        assert!(matches!(cloned, AlertState::Firing { notifications_sent: 3, .. }));
    }

    #[test]
    fn test_severity_copy() {
        let s1 = Severity::Warning;
        let s2 = s1; // Copy
        assert_eq!(s1, s2);
    }
}
