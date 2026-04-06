//! Resource scaling traits and implementations.

use super::error::{GrammarError, GrammarResult};

/// Resource scaling trait (analogous to graphics Scale<D, R>)
pub trait ResourceScale<D, R> {
    /// Scale a request from domain to range
    fn scale(&self, request: D) -> R;
    /// Get domain bounds
    fn domain(&self) -> (D, D);
    /// Get range bounds
    fn range(&self) -> (R, R);
}

/// Linear resource scaling
#[derive(Debug, Clone)]
pub struct LinearResourceScale {
    domain: (f64, f64),
    range: (f64, f64),
}

impl LinearResourceScale {
    /// Create new linear scale
    pub fn new(domain: (f64, f64), range: (f64, f64)) -> GrammarResult<Self> {
        if domain.0 >= domain.1 {
            return Err(GrammarError::InvalidScaleDomain { min: domain.0, max: domain.1 });
        }
        Ok(Self { domain, range })
    }
}

impl ResourceScale<f64, f64> for LinearResourceScale {
    fn scale(&self, request: f64) -> f64 {
        let t = (request - self.domain.0) / (self.domain.1 - self.domain.0);
        self.range.0 + t * (self.range.1 - self.range.0)
    }

    fn domain(&self) -> (f64, f64) {
        self.domain
    }

    fn range(&self) -> (f64, f64) {
        self.range
    }
}

/// Logarithmic resource scaling (for exponential resources)
#[derive(Debug, Clone)]
pub struct LogResourceScale {
    base: f64,
    domain: (f64, f64),
    range: (f64, f64),
}

impl LogResourceScale {
    /// Create new log scale
    pub fn new(base: f64, domain: (f64, f64), range: (f64, f64)) -> GrammarResult<Self> {
        if domain.0 >= domain.1 {
            return Err(GrammarError::InvalidScaleDomain { min: domain.0, max: domain.1 });
        }
        Ok(Self { base, domain, range })
    }
}

impl ResourceScale<f64, f64> for LogResourceScale {
    fn scale(&self, request: f64) -> f64 {
        let log_request = request.log(self.base);
        let log_min = self.domain.0.log(self.base);
        let log_max = self.domain.1.log(self.base);
        let t = (log_request - log_min) / (log_max - log_min);
        self.range.0 + t * (self.range.1 - self.range.0)
    }

    fn domain(&self) -> (f64, f64) {
        self.domain
    }

    fn range(&self) -> (f64, f64) {
        self.range
    }
}
