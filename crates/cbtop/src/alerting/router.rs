//! Alert routing, rate limiting, and deduplication.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::{Alert, AlertChannel, AlertSeverity, DeliveryResult};

/// Rate limiter for alerts
#[derive(Debug)]
struct RateLimiter {
    /// Max alerts per window
    max_alerts: usize,
    /// Window duration
    window: Duration,
    /// Alert timestamps by source
    timestamps: HashMap<String, Vec<Instant>>,
}

impl RateLimiter {
    fn new(max_alerts: usize, window: Duration) -> Self {
        Self { max_alerts, window, timestamps: HashMap::new() }
    }

    fn should_allow(&mut self, source: &str) -> bool {
        let now = Instant::now();
        let timestamps = self.timestamps.entry(source.to_string()).or_default();

        // Remove old timestamps
        timestamps.retain(|t| now.duration_since(*t) < self.window);

        if timestamps.len() >= self.max_alerts {
            false
        } else {
            timestamps.push(now);
            true
        }
    }
}

/// Deduplication tracker
#[derive(Debug)]
struct Deduplicator {
    /// Seen alert IDs with expiry
    seen: HashMap<String, Instant>,
    /// Dedup window
    window: Duration,
}

impl Deduplicator {
    fn new(window: Duration) -> Self {
        Self { seen: HashMap::new(), window }
    }

    fn is_duplicate(&mut self, alert_id: &str) -> bool {
        let now = Instant::now();

        // Clean old entries
        self.seen.retain(|_, expiry| now < *expiry);

        if self.seen.contains_key(alert_id) {
            true
        } else {
            self.seen.insert(alert_id.to_string(), now + self.window);
            false
        }
    }
}

/// Alert router configuration
#[derive(Debug, Clone)]
pub struct AlertRouterConfig {
    /// Channels by severity
    pub severity_routes: HashMap<AlertSeverity, Vec<AlertChannel>>,
    /// Default channels
    pub default_channels: Vec<AlertChannel>,
    /// Rate limit per minute
    pub rate_limit_per_minute: usize,
    /// Dedup window seconds
    pub dedup_window_sec: u64,
    /// Dry run mode
    pub dry_run: bool,
}

impl Default for AlertRouterConfig {
    fn default() -> Self {
        Self {
            severity_routes: HashMap::new(),
            default_channels: vec![AlertChannel::Console],
            rate_limit_per_minute: 60,
            dedup_window_sec: 300, // 5 minutes
            dry_run: false,
        }
    }
}

/// Alert router for multi-channel delivery
#[derive(Debug)]
pub struct AlertRouter {
    /// Configuration
    config: AlertRouterConfig,
    /// Rate limiter
    rate_limiter: RateLimiter,
    /// Deduplicator
    deduplicator: Deduplicator,
    /// Alert history
    history: Vec<Alert>,
    /// Max history size
    max_history: usize,
    /// Delivery results
    delivery_results: Vec<DeliveryResult>,
}

impl Default for AlertRouter {
    fn default() -> Self {
        Self::new(AlertRouterConfig::default())
    }
}

impl AlertRouter {
    /// Create new router
    pub fn new(config: AlertRouterConfig) -> Self {
        let rate_limiter = RateLimiter::new(config.rate_limit_per_minute, Duration::from_secs(60));
        let deduplicator = Deduplicator::new(Duration::from_secs(config.dedup_window_sec));

        Self {
            config,
            rate_limiter,
            deduplicator,
            history: Vec::new(),
            max_history: 1000,
            delivery_results: Vec::new(),
        }
    }

    /// Enable dry run mode
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.config.dry_run = dry_run;
        self
    }

    /// Set max history
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Add channel for severity
    pub fn add_route(&mut self, severity: AlertSeverity, channel: AlertChannel) {
        self.config.severity_routes.entry(severity).or_default().push(channel);
    }

    /// Add default channel
    pub fn add_default_channel(&mut self, channel: AlertChannel) {
        self.config.default_channels.push(channel);
    }

    /// Get channels for alert
    fn get_channels(&self, severity: AlertSeverity) -> Vec<&AlertChannel> {
        let mut channels: Vec<&AlertChannel> = self
            .config
            .severity_routes
            .get(&severity)
            .map(|c| c.iter().collect())
            .unwrap_or_default();

        if channels.is_empty() {
            channels = self.config.default_channels.iter().collect();
        }

        channels
    }

    /// Send alert to channel
    fn send_to_channel(&self, alert: &Alert, channel: &AlertChannel) -> DeliveryResult {
        if self.config.dry_run {
            return DeliveryResult::success(channel.name(), 0);
        }

        match channel {
            AlertChannel::Console => {
                println!("[{}] {} - {}", alert.severity.name(), alert.title, alert.message);
                DeliveryResult::success("console", 0)
            }
            AlertChannel::Slack { webhook_url: _ } => {
                // In a real implementation, this would make an HTTP request
                // For now, simulate success
                DeliveryResult::success("slack", 50)
            }
            AlertChannel::PagerDuty { routing_key: _ } => DeliveryResult::success("pagerduty", 100),
            AlertChannel::Email { .. } => DeliveryResult::success("email", 200),
            AlertChannel::Webhook { url: _, method: _ } => DeliveryResult::success("webhook", 30),
        }
    }

    /// Route and send alert
    pub fn send(&mut self, alert: Alert) -> Vec<DeliveryResult> {
        // Check deduplication
        if self.deduplicator.is_duplicate(&alert.id) {
            return vec![DeliveryResult::failure("all", "duplicate alert")];
        }

        // Check rate limit
        if !self.rate_limiter.should_allow(&alert.source) {
            return vec![DeliveryResult::failure("all", "rate limited")];
        }

        // Get channels and send
        let channels = self.get_channels(alert.severity);
        let results: Vec<DeliveryResult> =
            channels.iter().map(|ch| self.send_to_channel(&alert, ch)).collect();

        // Store results
        self.delivery_results.extend(results.clone());

        // Store in history
        self.history.push(alert);
        while self.history.len() > self.max_history {
            self.history.remove(0);
        }

        results
    }

    /// Get alert history
    pub fn history(&self) -> &[Alert] {
        &self.history
    }

    /// Get delivery results
    pub fn delivery_results(&self) -> &[DeliveryResult] {
        &self.delivery_results
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.delivery_results.clear();
    }

    /// Get alert count
    pub fn alert_count(&self) -> usize {
        self.history.len()
    }
}
