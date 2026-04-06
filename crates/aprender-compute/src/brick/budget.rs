//! Token and byte performance budgets for ComputeBrick operations.

/// Performance budget expressed in token terms.
/// Aligns compute costs with LLM inference metrics.
#[derive(Debug, Clone, Copy)]
pub struct TokenBudget {
    /// Latency budget per token (microseconds)
    pub us_per_token: f64,
    /// Throughput target (tokens/second)
    pub tokens_per_sec: f64,
    /// Batch size for amortization
    pub batch_size: usize,
}

/// Performance budget for byte-oriented operations (compression, I/O).
/// Use this for trueno-zram, disk I/O, network throughput, etc.
///
/// PMAT-452: Serializable for hardware.toml export.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ByteBudget {
    /// Latency budget per page (microseconds)
    pub us_per_page: f64,
    /// Throughput target (GB/s)
    pub gb_per_sec: f64,
    /// Page size in bytes (default 4096)
    pub page_size: usize,
}

impl Default for ByteBudget {
    fn default() -> Self {
        // Default: 25 GB/s (trueno-zram ZSTD target)
        Self::from_throughput(25.0)
    }
}

impl ByteBudget {
    /// Create budget from throughput target (GB/s).
    /// 25 GB/s = 0.16µs per 4KB page
    pub fn from_throughput(gb_per_sec: f64) -> Self {
        debug_assert!(
            gb_per_sec > 0.0 && gb_per_sec.is_finite(),
            "CB-BUDGET: throughput must be positive and finite, got {}",
            gb_per_sec
        );
        let bytes_per_sec = gb_per_sec * 1e9;
        let pages_per_sec = bytes_per_sec / 4096.0;
        Self { us_per_page: 1_000_000.0 / pages_per_sec, gb_per_sec, page_size: 4096 }
    }

    /// Create budget from latency target (µs per page).
    pub fn from_latency(us_per_page: f64) -> Self {
        let pages_per_sec = 1_000_000.0 / us_per_page;
        let bytes_per_sec = pages_per_sec * 4096.0;
        Self { us_per_page, gb_per_sec: bytes_per_sec / 1e9, page_size: 4096 }
    }

    /// Set custom page size (e.g., 64KB for huge pages).
    #[must_use]
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        // Recalculate us_per_page based on new page size
        let bytes_per_sec = self.gb_per_sec * 1e9;
        let pages_per_sec = bytes_per_sec / page_size as f64;
        self.us_per_page = 1_000_000.0 / pages_per_sec;
        self.page_size = page_size;
        self
    }

    /// Convert to TokenBudget (1 token = 1 page).
    /// Useful for integrating byte workloads with token-centric monitoring.
    pub fn to_token_budget(&self) -> TokenBudget {
        TokenBudget {
            us_per_token: self.us_per_page,
            tokens_per_sec: 1_000_000.0 / self.us_per_page,
            batch_size: 1,
        }
    }

    /// Check if actual performance meets budget.
    pub fn is_met(&self, actual_us_per_page: f64) -> bool {
        actual_us_per_page <= self.us_per_page
    }

    /// Calculate budget utilization.
    pub fn utilization(&self, actual_us_per_page: f64) -> f64 {
        actual_us_per_page / self.us_per_page
    }

    /// Calculate actual throughput from latency.
    pub fn throughput_from_latency(us_per_page: f64, page_size: usize) -> f64 {
        let pages_per_sec = 1_000_000.0 / us_per_page;
        pages_per_sec * page_size as f64 / 1e9
    }
}

impl Default for TokenBudget {
    fn default() -> Self {
        // Default: 50µs/token = 20,000 tokens/sec
        Self::from_latency(50.0)
    }
}

impl TokenBudget {
    /// Create budget from latency target.
    /// 50µs/token = 20,000 tokens/sec
    pub fn from_latency(us_per_token: f64) -> Self {
        Self { us_per_token, tokens_per_sec: 1_000_000.0 / us_per_token, batch_size: 1 }
    }

    /// Create budget from throughput target.
    /// 20,000 tokens/sec = 50µs/token
    pub fn from_throughput(tokens_per_sec: f64) -> Self {
        Self { us_per_token: 1_000_000.0 / tokens_per_sec, tokens_per_sec, batch_size: 1 }
    }

    /// Set batch size for amortization.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Check if actual performance meets budget.
    pub fn is_met(&self, actual_us_per_token: f64) -> bool {
        actual_us_per_token <= self.us_per_token
    }

    /// Calculate budget utilization (0.0 = unused, 1.0 = exactly at budget, >1.0 = over budget).
    pub fn utilization(&self, actual_us_per_token: f64) -> f64 {
        actual_us_per_token / self.us_per_token
    }
}

/// Result of ComputeBrick execution with token metrics.
#[derive(Debug, Clone)]
pub struct TokenResult<T> {
    /// Computed output
    pub output: T,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Actual latency (microseconds/token)
    pub us_per_token: f64,
    /// Actual throughput (tokens/second)
    pub tokens_per_sec: f64,
    /// Did we meet the budget?
    pub budget_met: bool,
    /// Budget utilization (0.0-1.0+ where 1.0 = exactly at budget)
    pub budget_utilization: f64,
}

impl<T> TokenResult<T> {
    /// Map the output to a new type.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> TokenResult<U> {
        TokenResult {
            output: f(self.output),
            tokens_processed: self.tokens_processed,
            us_per_token: self.us_per_token,
            tokens_per_sec: self.tokens_per_sec,
            budget_met: self.budget_met,
            budget_utilization: self.budget_utilization,
        }
    }
}
