//! Alert types: severity, channels, messages, and delivery results.

use std::collections::HashMap;

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
}

impl AlertSeverity {
    /// Get severity name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Critical => "CRITICAL",
        }
    }

    /// Get severity color (for Slack)
    pub fn color(&self) -> &'static str {
        match self {
            Self::Info => "#36a64f",     // Green
            Self::Warning => "#ffcc00",  // Yellow
            Self::Critical => "#ff0000", // Red
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "INFO" => Some(Self::Info),
            "WARNING" | "WARN" => Some(Self::Warning),
            "CRITICAL" | "CRIT" => Some(Self::Critical),
            _ => None,
        }
    }
}

/// Alert channel types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertChannel {
    /// Slack webhook
    Slack { webhook_url: String },
    /// PagerDuty Events API
    PagerDuty { routing_key: String },
    /// Email via SMTP
    Email {
        smtp_host: String,
        to: String,
        from: String,
    },
    /// Generic HTTP webhook
    Webhook { url: String, method: String },
    /// Console output (for testing)
    Console,
}

impl AlertChannel {
    /// Get channel name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Slack { .. } => "slack",
            Self::PagerDuty { .. } => "pagerduty",
            Self::Email { .. } => "email",
            Self::Webhook { .. } => "webhook",
            Self::Console => "console",
        }
    }

    /// Create Slack channel
    pub fn slack(webhook_url: &str) -> Self {
        Self::Slack {
            webhook_url: webhook_url.to_string(),
        }
    }

    /// Create PagerDuty channel
    pub fn pagerduty(routing_key: &str) -> Self {
        Self::PagerDuty {
            routing_key: routing_key.to_string(),
        }
    }

    /// Create webhook channel
    pub fn webhook(url: &str) -> Self {
        Self::Webhook {
            url: url.to_string(),
            method: "POST".to_string(),
        }
    }
}

/// Alert message
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert ID (for deduplication)
    pub id: String,
    /// Alert title
    pub title: String,
    /// Alert message body
    pub message: String,
    /// Severity level
    pub severity: AlertSeverity,
    /// Source metric/component
    pub source: String,
    /// Metric value that triggered alert
    pub value: Option<f64>,
    /// Threshold that was exceeded
    pub threshold: Option<f64>,
    /// Creation timestamp (Unix millis)
    pub timestamp: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Alert {
    /// Create new alert
    pub fn new(title: &str, message: &str, severity: AlertSeverity) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            id: format!("{}_{}", title.replace(' ', "_").to_lowercase(), timestamp),
            title: title.to_string(),
            message: message.to_string(),
            severity,
            source: String::new(),
            value: None,
            threshold: None,
            timestamp,
            metadata: HashMap::new(),
        }
    }

    /// Set source
    pub fn with_source(mut self, source: &str) -> Self {
        self.source = source.to_string();
        self
    }

    /// Set value
    pub fn with_value(mut self, value: f64) -> Self {
        self.value = Some(value);
        self
    }

    /// Set threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Format as Slack message
    pub fn to_slack_json(&self) -> String {
        let value_str = self.value.map(|v| format!("{:.2}", v)).unwrap_or_default();
        let threshold_str = self
            .threshold
            .map(|t| format!("{:.2}", t))
            .unwrap_or_default();

        format!(
            r#"{{"attachments":[{{"color":"{}","title":"{}","text":"{}","fields":[{{"title":"Severity","value":"{}","short":true}},{{"title":"Source","value":"{}","short":true}},{{"title":"Value","value":"{}","short":true}},{{"title":"Threshold","value":"{}","short":true}}],"ts":{}}}]}}"#,
            self.severity.color(),
            self.title,
            self.message,
            self.severity.name(),
            self.source,
            value_str,
            threshold_str,
            self.timestamp / 1000
        )
    }

    /// Format as PagerDuty event
    pub fn to_pagerduty_json(&self, routing_key: &str) -> String {
        let action = match self.severity {
            AlertSeverity::Critical | AlertSeverity::Warning | AlertSeverity::Info => "trigger",
        };

        format!(
            r#"{{"routing_key":"{}","event_action":"{}","dedup_key":"{}","payload":{{"summary":"{}","source":"{}","severity":"{}","timestamp":"{}"}}}}"#,
            routing_key,
            action,
            self.id,
            self.title,
            self.source,
            self.severity.name().to_lowercase(),
            self.timestamp
        )
    }

    /// Format as generic JSON
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"id":"{}","title":"{}","message":"{}","severity":"{}","source":"{}","value":{},"threshold":{},"timestamp":{}}}"#,
            self.id,
            self.title,
            self.message,
            self.severity.name(),
            self.source,
            self.value
                .map(|v| format!("{}", v))
                .unwrap_or("null".to_string()),
            self.threshold
                .map(|t| format!("{}", t))
                .unwrap_or("null".to_string()),
            self.timestamp
        )
    }
}

/// Alert delivery result
#[derive(Debug, Clone)]
pub struct DeliveryResult {
    /// Channel name
    pub channel: String,
    /// Was delivery successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Delivery time (millis)
    pub duration_ms: u64,
}

impl DeliveryResult {
    /// Create success result
    pub fn success(channel: &str, duration_ms: u64) -> Self {
        Self {
            channel: channel.to_string(),
            success: true,
            error: None,
            duration_ms,
        }
    }

    /// Create failure result
    pub fn failure(channel: &str, error: &str) -> Self {
        Self {
            channel: channel.to_string(),
            success: false,
            error: Some(error.to_string()),
            duration_ms: 0,
        }
    }
}
