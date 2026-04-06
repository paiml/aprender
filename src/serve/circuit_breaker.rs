//! Cost Circuit Breaker
//!
//! Implements Toyota Way "Muda Elimination" (Waste Prevention).
//!
//! Prevents runaway API costs by tracking usage and enforcing daily budgets.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================================================================
// SERVE-CBR-001: Cost Model
// ============================================================================

/// Token pricing for a backend (per 1M tokens)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TokenPricing {
    /// Input token cost per 1M tokens (USD)
    pub input_per_million: f64,
    /// Output token cost per 1M tokens (USD)
    pub output_per_million: f64,
}

impl TokenPricing {
    /// Create new pricing
    #[must_use]
    pub const fn new(input_per_million: f64, output_per_million: f64) -> Self {
        Self { input_per_million, output_per_million }
    }

    /// Calculate cost for given token counts
    #[must_use]
    pub fn calculate(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * self.input_per_million;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.output_per_million;
        input_cost + output_cost
    }

    /// Known pricing for common models (approximate, as of late 2024)
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        const MODEL_PRICING: &[(&[&str], f64, f64)] = &[
            (&["gpt-4o"], 2.50, 10.00),
            (&["gpt-4-turbo", "gpt-4"], 10.00, 30.00),
            (&["gpt-3.5"], 0.50, 1.50),
            (&["claude-3-opus"], 15.00, 75.00),
            (&["claude-3-sonnet", "claude-3.5"], 3.00, 15.00),
            (&["claude-3-haiku"], 0.25, 1.25),
            (&["llama", "mistral"], 0.20, 0.20),
        ];
        let lower = model.to_lowercase();
        MODEL_PRICING
            .iter()
            .find(|(patterns, _, _)| patterns.iter().any(|p| lower.contains(p)))
            .map(|(_, input, output)| Self::new(*input, *output))
            .unwrap_or_else(|| Self::new(1.00, 2.00))
    }
}

impl Default for TokenPricing {
    fn default() -> Self {
        Self::new(1.00, 2.00)
    }
}

// ============================================================================
// SERVE-CBR-002: Usage Tracking
// ============================================================================

/// Usage record for a single request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub timestamp: u64,
    pub backend: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
}

/// Daily usage summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DailyUsage {
    /// Date as YYYY-MM-DD
    pub date: String,
    /// Total input tokens
    pub total_input_tokens: u64,
    /// Total output tokens
    pub total_output_tokens: u64,
    /// Total cost in USD
    pub total_cost_usd: f64,
    /// Request count
    pub request_count: u64,
    /// Usage by model
    pub by_model: HashMap<String, f64>,
}

impl DailyUsage {
    /// Create for today
    #[must_use]
    pub fn today() -> Self {
        Self { date: Self::current_date(), ..Default::default() }
    }

    /// Get current date string
    #[must_use]
    pub fn current_date() -> String {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        // Simple date calculation (not timezone aware)
        let days = now / 86400;
        let year = 1970 + (days / 365); // Approximate
        let day_of_year = days % 365;
        let month = day_of_year / 30 + 1;
        let day = day_of_year % 30 + 1;
        format!("{}-{:02}-{:02}", year, month.min(12), day.min(31))
    }

    /// Add a usage record
    pub fn add(&mut self, record: &UsageRecord) {
        self.total_input_tokens += record.input_tokens;
        self.total_output_tokens += record.output_tokens;
        self.total_cost_usd += record.cost_usd;
        self.request_count += 1;
        *self.by_model.entry(record.model.clone()).or_insert(0.0) += record.cost_usd;
    }
}

// ============================================================================
// SERVE-CBR-003: Circuit Breaker
// ============================================================================

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CircuitState {
    /// Circuit is closed, requests allowed
    #[default]
    Closed,
    /// Circuit is open, requests blocked
    Open,
    /// Half-open, testing if budget allows
    HalfOpen,
}

/// Cost circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Daily budget in USD
    pub daily_budget_usd: f64,
    /// Warning threshold (percentage of budget)
    pub warning_threshold: f64,
    /// Per-request cost limit (USD)
    pub max_request_cost_usd: f64,
    /// Auto-reset after cooldown period
    pub cooldown_seconds: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            daily_budget_usd: 10.0,    // $10/day default
            warning_threshold: 0.8,    // Warn at 80%
            max_request_cost_usd: 1.0, // Max $1 per request
            cooldown_seconds: 3600,    // 1 hour cooldown
        }
    }
}

impl CircuitBreakerConfig {
    /// Create with specific daily budget
    #[must_use]
    pub fn with_budget(daily_budget_usd: f64) -> Self {
        Self { daily_budget_usd, ..Default::default() }
    }
}

/// Cost circuit breaker
///
/// Thread-safe circuit breaker that tracks API costs and prevents overspending.
pub struct CostCircuitBreaker {
    config: CircuitBreakerConfig,
    /// Accumulated cost in millicents (for atomic operations)
    accumulated_millicents: AtomicU64,
    /// Current date for reset tracking
    current_date: RwLock<String>,
    /// Circuit state
    state: RwLock<CircuitState>,
    /// Time when circuit was opened
    opened_at: RwLock<Option<u64>>,
}

impl CostCircuitBreaker {
    /// Create a new circuit breaker
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            accumulated_millicents: AtomicU64::new(0),
            current_date: RwLock::new(DailyUsage::current_date()),
            state: RwLock::new(CircuitState::Closed),
            opened_at: RwLock::new(None),
        }
    }

    /// Create with default config
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }

    // Lock accessor helpers — single source of truth for lock patterns
    fn read_state(&self) -> CircuitState {
        *self.state.read().expect("circuit breaker state lock poisoned")
    }

    fn write_state(&self, new_state: CircuitState) {
        *self.state.write().expect("circuit breaker state lock poisoned") = new_state;
    }

    fn read_opened_at(&self) -> Option<u64> {
        *self.opened_at.read().expect("circuit breaker opened_at lock poisoned")
    }

    fn write_opened_at(&self, timestamp: Option<u64>) {
        *self.opened_at.write().expect("circuit breaker opened_at lock poisoned") = timestamp;
    }

    fn read_current_date(&self) -> String {
        self.current_date.read().expect("circuit breaker current_date lock poisoned").clone()
    }

    fn write_current_date(&self, date: String) {
        *self.current_date.write().expect("circuit breaker current_date lock poisoned") = date;
    }

    /// Check if a request with estimated cost is allowed
    pub fn check(&self, estimated_cost_usd: f64) -> Result<(), CircuitBreakerError> {
        // Reset if new day
        self.maybe_reset_daily();

        // Check per-request limit
        if estimated_cost_usd > self.config.max_request_cost_usd {
            return Err(CircuitBreakerError::RequestTooExpensive {
                estimated: estimated_cost_usd,
                limit: self.config.max_request_cost_usd,
            });
        }

        // Check circuit state
        match self.read_state() {
            CircuitState::Open => {
                // Check if cooldown has passed
                if self.cooldown_elapsed() {
                    self.write_state(CircuitState::HalfOpen);
                } else {
                    return Err(CircuitBreakerError::BudgetExceeded {
                        spent: self.accumulated_usd(),
                        budget: self.config.daily_budget_usd,
                    });
                }
            }
            CircuitState::HalfOpen | CircuitState::Closed => {}
        }

        // Check if adding this cost would exceed budget
        let current = self.accumulated_usd();
        if current + estimated_cost_usd > self.config.daily_budget_usd {
            self.write_state(CircuitState::Open);
            self.write_opened_at(Some(Self::current_timestamp()));
            return Err(CircuitBreakerError::BudgetExceeded {
                spent: current,
                budget: self.config.daily_budget_usd,
            });
        }

        Ok(())
    }

    /// Record actual cost after request completes
    pub fn record(&self, actual_cost_usd: f64) {
        let millicents = (actual_cost_usd * 100_000.0) as u64;
        self.accumulated_millicents.fetch_add(millicents, Ordering::SeqCst);

        // Check if we've hit the budget
        if self.accumulated_usd() >= self.config.daily_budget_usd {
            self.write_state(CircuitState::Open);
            self.write_opened_at(Some(Self::current_timestamp()));
        }
    }

    /// Get current accumulated cost in USD
    #[must_use]
    pub fn accumulated_usd(&self) -> f64 {
        self.accumulated_millicents.load(Ordering::SeqCst) as f64 / 100_000.0
    }

    /// Get remaining budget
    #[must_use]
    pub fn remaining_usd(&self) -> f64 {
        (self.config.daily_budget_usd - self.accumulated_usd()).max(0.0)
    }

    /// Get budget utilization percentage
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.config.daily_budget_usd == 0.0 {
            return if self.accumulated_usd() > 0.0 { 1.0 } else { 0.0 };
        }
        self.accumulated_usd() / self.config.daily_budget_usd
    }

    /// Check if at warning threshold
    #[must_use]
    pub fn is_warning(&self) -> bool {
        self.utilization() >= self.config.warning_threshold
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> CircuitState {
        self.read_state()
    }

    /// Force reset (for testing or manual override)
    pub fn reset(&self) {
        self.accumulated_millicents.store(0, Ordering::SeqCst);
        self.write_state(CircuitState::Closed);
        self.write_opened_at(None);
        self.write_current_date(DailyUsage::current_date());
    }

    fn maybe_reset_daily(&self) {
        let today = DailyUsage::current_date();
        let current = self.read_current_date();
        if current != today {
            drop(current);
            self.reset();
        }
    }

    fn cooldown_elapsed(&self) -> bool {
        if let Some(opened) = self.read_opened_at() {
            let now = Self::current_timestamp();
            now - opened >= self.config.cooldown_seconds
        } else {
            true
        }
    }

    fn current_timestamp() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO).as_secs()
    }
}

impl Default for CostCircuitBreaker {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Circuit breaker errors
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerError {
    /// Daily budget exceeded
    BudgetExceeded { spent: f64, budget: f64 },
    /// Single request too expensive
    RequestTooExpensive { estimated: f64, limit: f64 },
}

impl std::fmt::Display for CircuitBreakerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BudgetExceeded { spent, budget } => {
                write!(f, "Daily budget exceeded: ${:.2} spent of ${:.2} budget", spent, budget)
            }
            Self::RequestTooExpensive { estimated, limit } => {
                write!(f, "Request too expensive: ${:.2} estimated, ${:.2} limit", estimated, limit)
            }
        }
    }
}

impl std::error::Error for CircuitBreakerError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    // ========================================================================
    // SERVE-CBR-001: Token Pricing Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_001_pricing_calculate() {
        let pricing = TokenPricing::new(1.0, 2.0); // $1/M input, $2/M output
        let cost = pricing.calculate(1_000_000, 500_000);
        assert!((cost - 2.0).abs() < 0.001); // $1 input + $1 output
    }

    #[test]
    fn test_SERVE_CBR_001_pricing_small_amounts() {
        let pricing = TokenPricing::new(10.0, 30.0); // GPT-4 pricing
        let cost = pricing.calculate(1000, 500);
        // 1000 input = $0.01, 500 output = $0.015
        assert!((cost - 0.025).abs() < 0.001);
    }

    #[test]
    fn test_SERVE_CBR_001_pricing_for_model_gpt4() {
        let pricing = TokenPricing::for_model("gpt-4-turbo");
        assert_eq!(pricing.input_per_million, 10.0);
        assert_eq!(pricing.output_per_million, 30.0);
    }

    #[test]
    fn test_SERVE_CBR_001_pricing_for_model_claude() {
        let pricing = TokenPricing::for_model("claude-3-sonnet");
        assert_eq!(pricing.input_per_million, 3.0);
        assert_eq!(pricing.output_per_million, 15.0);
    }

    #[test]
    fn test_SERVE_CBR_001_pricing_for_model_llama() {
        let pricing = TokenPricing::for_model("llama-3.1-70b");
        assert_eq!(pricing.input_per_million, 0.20);
    }

    #[test]
    fn test_SERVE_CBR_001_pricing_default() {
        let pricing = TokenPricing::default();
        assert_eq!(pricing.input_per_million, 1.0);
        assert_eq!(pricing.output_per_million, 2.0);
    }

    // ========================================================================
    // SERVE-CBR-002: Daily Usage Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_002_daily_usage_add() {
        let mut usage = DailyUsage::today();
        let record = UsageRecord {
            timestamp: 0,
            backend: "openai".to_string(),
            model: "gpt-4".to_string(),
            input_tokens: 1000,
            output_tokens: 500,
            cost_usd: 0.025,
        };
        usage.add(&record);
        assert_eq!(usage.total_input_tokens, 1000);
        assert_eq!(usage.total_output_tokens, 500);
        assert!((usage.total_cost_usd - 0.025).abs() < 0.001);
        assert_eq!(usage.request_count, 1);
    }

    #[test]
    fn test_SERVE_CBR_002_daily_usage_by_model() {
        let mut usage = DailyUsage::today();
        usage.add(&UsageRecord {
            timestamp: 0,
            backend: "openai".to_string(),
            model: "gpt-4".to_string(),
            input_tokens: 1000,
            output_tokens: 500,
            cost_usd: 1.0,
        });
        usage.add(&UsageRecord {
            timestamp: 0,
            backend: "openai".to_string(),
            model: "gpt-3.5".to_string(),
            input_tokens: 1000,
            output_tokens: 500,
            cost_usd: 0.1,
        });
        assert_eq!(usage.by_model.get("gpt-4"), Some(&1.0));
        assert_eq!(usage.by_model.get("gpt-3.5"), Some(&0.1));
    }

    // ========================================================================
    // SERVE-CBR-003: Circuit Breaker Basic Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_003_default_config() {
        let config = CircuitBreakerConfig::default();
        assert_eq!(config.daily_budget_usd, 10.0);
        assert_eq!(config.warning_threshold, 0.8);
    }

    #[test]
    fn test_SERVE_CBR_003_check_allows_under_budget() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(10.0));
        assert!(cb.check(1.0).is_ok());
    }

    #[test]
    fn test_SERVE_CBR_003_check_blocks_over_budget() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(1.0));
        cb.record(0.9);
        let result = cb.check(0.2);
        assert!(result.is_err());
    }

    #[test]
    fn test_SERVE_CBR_003_record_accumulates() {
        let cb = CostCircuitBreaker::with_defaults();
        cb.record(1.0);
        cb.record(2.0);
        assert!((cb.accumulated_usd() - 3.0).abs() < 0.001);
    }

    // ========================================================================
    // SERVE-CBR-004: Circuit Breaker State Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_004_initial_state_closed() {
        let cb = CostCircuitBreaker::with_defaults();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_SERVE_CBR_004_opens_on_budget_exceed() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(1.0));
        cb.record(1.0);
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_SERVE_CBR_004_reset_closes_circuit() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(1.0));
        cb.record(1.0);
        assert_eq!(cb.state(), CircuitState::Open);
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!((cb.accumulated_usd()).abs() < 0.001);
    }

    // ========================================================================
    // SERVE-CBR-005: Request Limit Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_005_rejects_expensive_request() {
        let config = CircuitBreakerConfig { max_request_cost_usd: 0.5, ..Default::default() };
        let cb = CostCircuitBreaker::new(config);
        let result = cb.check(1.0);
        assert!(matches!(result, Err(CircuitBreakerError::RequestTooExpensive { .. })));
    }

    #[test]
    fn test_SERVE_CBR_005_allows_cheap_request() {
        let config = CircuitBreakerConfig { max_request_cost_usd: 1.0, ..Default::default() };
        let cb = CostCircuitBreaker::new(config);
        assert!(cb.check(0.5).is_ok());
    }

    // ========================================================================
    // SERVE-CBR-006: Utilization Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_006_utilization_percentage() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(10.0));
        cb.record(5.0);
        assert!((cb.utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_SERVE_CBR_006_remaining_budget() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(10.0));
        cb.record(3.0);
        assert!((cb.remaining_usd() - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_SERVE_CBR_006_warning_threshold() {
        let config = CircuitBreakerConfig {
            daily_budget_usd: 10.0,
            warning_threshold: 0.8,
            ..Default::default()
        };
        let cb = CostCircuitBreaker::new(config);
        cb.record(7.0);
        assert!(!cb.is_warning());
        cb.record(1.0);
        assert!(cb.is_warning());
    }

    // ========================================================================
    // SERVE-CBR-007: Error Display Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_007_budget_exceeded_display() {
        let err = CircuitBreakerError::BudgetExceeded { spent: 10.5, budget: 10.0 };
        let msg = err.to_string();
        assert!(msg.contains("10.50"));
        assert!(msg.contains("10.00"));
        assert!(msg.contains("exceeded"));
    }

    #[test]
    fn test_SERVE_CBR_007_request_expensive_display() {
        let err = CircuitBreakerError::RequestTooExpensive { estimated: 5.0, limit: 1.0 };
        let msg = err.to_string();
        assert!(msg.contains("5.00"));
        assert!(msg.contains("1.00"));
        assert!(msg.contains("expensive"));
    }

    // ========================================================================
    // SERVE-CBR-008: Cooldown and Open-State Tests
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_008_open_state_blocks_during_cooldown() {
        let config = CircuitBreakerConfig {
            daily_budget_usd: 1.0,
            max_request_cost_usd: 5.0,
            cooldown_seconds: 3600, // 1 hour cooldown
            ..Default::default()
        };
        let cb = CostCircuitBreaker::new(config);

        // Spend entire budget to open the circuit
        cb.record(1.0);
        assert_eq!(cb.state(), CircuitState::Open);

        // Now check should fail because circuit is open and cooldown hasn't elapsed
        let result = cb.check(0.01);
        assert!(result.is_err());
        assert!(matches!(result, Err(CircuitBreakerError::BudgetExceeded { .. })));
    }

    #[test]
    fn test_SERVE_CBR_008_cooldown_elapsed_with_no_opened_at() {
        // Test cooldown_elapsed when opened_at is None (returns true)
        let cb = CostCircuitBreaker::with_defaults();
        // State is Closed, opened_at is None
        assert!(cb.cooldown_elapsed());
    }

    #[test]
    fn test_SERVE_CBR_008_cooldown_elapsed_recently_opened() {
        let config = CircuitBreakerConfig {
            daily_budget_usd: 1.0,
            max_request_cost_usd: 5.0,
            cooldown_seconds: 3600,
            ..Default::default()
        };
        let cb = CostCircuitBreaker::new(config);

        // Record enough to open the circuit
        cb.record(1.0);
        assert_eq!(cb.state(), CircuitState::Open);

        // Cooldown should NOT have elapsed (just opened)
        assert!(!cb.cooldown_elapsed());
    }

    #[test]
    fn test_SERVE_CBR_008_cooldown_elapsed_with_zero_cooldown() {
        let config = CircuitBreakerConfig {
            daily_budget_usd: 1.0,
            max_request_cost_usd: 5.0,
            cooldown_seconds: 0, // Zero cooldown
            ..Default::default()
        };
        let cb = CostCircuitBreaker::new(config);

        // Open the circuit
        cb.record(1.0);
        assert_eq!(cb.state(), CircuitState::Open);

        // With zero cooldown, elapsed should be true immediately
        assert!(cb.cooldown_elapsed());
    }

    #[test]
    fn test_SERVE_CBR_008_half_open_after_cooldown() {
        let config = CircuitBreakerConfig {
            daily_budget_usd: 10.0,
            max_request_cost_usd: 5.0,
            cooldown_seconds: 0, // Zero cooldown so it immediately transitions
            ..Default::default()
        };
        let cb = CostCircuitBreaker::new(config);

        // Spend entire budget
        cb.record(10.0);
        assert_eq!(cb.state(), CircuitState::Open);

        // With zero cooldown, check should transition to HalfOpen
        // But then it checks budget and re-opens because budget is still exceeded
        let result = cb.check(0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_SERVE_CBR_008_check_transitions_open_to_halfopen_then_allows() {
        let config = CircuitBreakerConfig {
            daily_budget_usd: 10.0,
            max_request_cost_usd: 5.0,
            cooldown_seconds: 0, // Zero cooldown for instant transition
            ..Default::default()
        };
        let cb = CostCircuitBreaker::new(config);

        // Spend some (but not all) budget, then manually open circuit
        cb.record(5.0);
        cb.write_state(CircuitState::Open);
        cb.write_opened_at(Some(CostCircuitBreaker::current_timestamp()));

        // With zero cooldown, check should transition Open -> HalfOpen,
        // then allow the request since we still have budget
        let result = cb.check(1.0);
        assert!(result.is_ok());
    }

    // ========================================================================
    // SERVE-CBR-009: Budget Crossing in check()
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_009_check_opens_circuit_on_budget_cross() {
        let config = CircuitBreakerConfig {
            daily_budget_usd: 5.0,
            max_request_cost_usd: 10.0,
            ..Default::default()
        };
        let cb = CostCircuitBreaker::new(config);

        // Record 4.5 (under budget)
        cb.record(4.5);
        assert_eq!(cb.state(), CircuitState::Closed);

        // Try to add 1.0 which would exceed budget
        let result = cb.check(1.0);
        assert!(result.is_err());
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_SERVE_CBR_009_check_budget_just_under_allows() {
        let config = CircuitBreakerConfig {
            daily_budget_usd: 5.0,
            max_request_cost_usd: 10.0,
            ..Default::default()
        };
        let cb = CostCircuitBreaker::new(config);

        cb.record(4.0);
        // 4.0 + 0.5 = 4.5, under 5.0 budget
        let result = cb.check(0.5);
        assert!(result.is_ok());
    }

    // ========================================================================
    // SERVE-CBR-010: read/write accessor helpers
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_010_read_opened_at_none_initially() {
        let cb = CostCircuitBreaker::with_defaults();
        assert_eq!(cb.read_opened_at(), None);
    }

    #[test]
    fn test_SERVE_CBR_010_write_and_read_opened_at() {
        let cb = CostCircuitBreaker::with_defaults();
        let ts = CostCircuitBreaker::current_timestamp();
        cb.write_opened_at(Some(ts));
        assert_eq!(cb.read_opened_at(), Some(ts));
    }

    #[test]
    fn test_SERVE_CBR_010_write_opened_at_clears() {
        let cb = CostCircuitBreaker::with_defaults();
        cb.write_opened_at(Some(12345));
        cb.write_opened_at(None);
        assert_eq!(cb.read_opened_at(), None);
    }

    // ========================================================================
    // SERVE-CBR-011: Error trait impl
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_011_error_trait_budget_exceeded() {
        let err: Box<dyn std::error::Error> =
            Box::new(CircuitBreakerError::BudgetExceeded { spent: 10.0, budget: 5.0 });
        assert!(err.to_string().contains("exceeded"));
    }

    #[test]
    fn test_SERVE_CBR_011_error_trait_request_expensive() {
        let err: Box<dyn std::error::Error> =
            Box::new(CircuitBreakerError::RequestTooExpensive { estimated: 3.0, limit: 1.0 });
        assert!(err.to_string().contains("expensive"));
    }

    // ========================================================================
    // SERVE-CBR-012: Additional model pricing coverage
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_012_pricing_gpt4o() {
        let pricing = TokenPricing::for_model("gpt-4o-mini");
        assert_eq!(pricing.input_per_million, 2.50);
        assert_eq!(pricing.output_per_million, 10.00);
    }

    #[test]
    fn test_SERVE_CBR_012_pricing_gpt35() {
        let pricing = TokenPricing::for_model("gpt-3.5-turbo");
        assert_eq!(pricing.input_per_million, 0.50);
    }

    #[test]
    fn test_SERVE_CBR_012_pricing_claude_opus() {
        let pricing = TokenPricing::for_model("claude-3-opus-20240229");
        assert_eq!(pricing.input_per_million, 15.00);
        assert_eq!(pricing.output_per_million, 75.00);
    }

    #[test]
    fn test_SERVE_CBR_012_pricing_claude_haiku() {
        let pricing = TokenPricing::for_model("claude-3-haiku-20240307");
        assert_eq!(pricing.input_per_million, 0.25);
        assert_eq!(pricing.output_per_million, 1.25);
    }

    #[test]
    fn test_SERVE_CBR_012_pricing_claude_35() {
        let pricing = TokenPricing::for_model("claude-3.5-sonnet");
        assert_eq!(pricing.input_per_million, 3.00);
    }

    #[test]
    fn test_SERVE_CBR_012_pricing_mistral() {
        let pricing = TokenPricing::for_model("mistral-7b");
        assert_eq!(pricing.input_per_million, 0.20);
    }

    #[test]
    fn test_SERVE_CBR_012_pricing_unknown_model() {
        let pricing = TokenPricing::for_model("totally-unknown-model");
        assert_eq!(pricing.input_per_million, 1.00);
        assert_eq!(pricing.output_per_million, 2.00);
    }

    // ========================================================================
    // SERVE-CBR-013: DailyUsage current_date
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_013_current_date_format() {
        let date = DailyUsage::current_date();
        // Should be in YYYY-MM-DD format
        assert_eq!(date.len(), 10);
        assert_eq!(&date[4..5], "-");
        assert_eq!(&date[7..8], "-");
    }

    #[test]
    fn test_SERVE_CBR_013_today_has_current_date() {
        let usage = DailyUsage::today();
        let expected = DailyUsage::current_date();
        assert_eq!(usage.date, expected);
        assert_eq!(usage.total_input_tokens, 0);
        assert_eq!(usage.total_cost_usd, 0.0);
    }

    // ========================================================================
    // SERVE-CBR-014: remaining_usd edge cases
    // ========================================================================

    #[test]
    fn test_SERVE_CBR_014_remaining_usd_clamped_to_zero() {
        let cb = CostCircuitBreaker::new(CircuitBreakerConfig::with_budget(1.0));
        cb.record(2.0); // Over budget
        assert!((cb.remaining_usd()).abs() < 0.001); // Should be 0, not negative
    }
}
