
impl InferenceConfig {
    /// Create default config for a model
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            port: 8080,
            max_batch_size: 32,
            timeout_ms: 100,
            enable_cors: true,
            metrics_path: Some("/metrics".into()),
            health_path: Some("/health".into()),
        }
    }

    /// Set port
    #[must_use]
    pub const fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set max batch size
    #[must_use]
    pub const fn with_batch_size(mut self, size: u32) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Set timeout
    #[must_use]
    pub const fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Disable CORS
    #[must_use]
    pub const fn without_cors(mut self) -> Self {
        self.enable_cors = false;
        self
    }

    /// Inference endpoint URL
    #[must_use]
    pub fn predict_url(&self) -> String {
        format!("http://localhost:{}/predict", self.port)
    }

    /// Batch inference endpoint URL
    #[must_use]
    pub fn batch_predict_url(&self) -> String {
        format!("http://localhost:{}/batch_predict", self.port)
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self::new("model.apr")
    }
}

/// Stack health status
#[derive(Debug, Clone)]
pub struct StackHealth {
    /// Component availability
    pub components: HashMap<StackComponent, ComponentHealth>,
    /// Overall status
    pub overall: HealthStatus,
    /// Last check timestamp
    pub checked_at: String,
}

impl StackHealth {
    /// Create new health status
    #[must_use]
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            overall: HealthStatus::Unknown,
            checked_at: "2025-01-01T00:00:00Z".into(),
        }
    }

    /// Set component health
    pub fn set_component(&mut self, component: StackComponent, health: ComponentHealth) {
        self.components.insert(component, health);
        self.update_overall();
    }

    /// Update overall status based on components
    fn update_overall(&mut self) {
        if self.components.is_empty() {
            self.overall = HealthStatus::Unknown;
            return;
        }

        let all_healthy = self
            .components
            .values()
            .all(|h| h.status == HealthStatus::Healthy);
        let any_unhealthy = self
            .components
            .values()
            .any(|h| h.status == HealthStatus::Unhealthy);

        self.overall = if all_healthy {
            HealthStatus::Healthy
        } else if any_unhealthy {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        };
    }

    /// Check if stack is healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.overall == HealthStatus::Healthy
    }
}

impl Default for StackHealth {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual component health
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    /// Health status
    pub status: HealthStatus,
    /// Version if available
    pub version: Option<String>,
    /// Response time in ms
    pub response_time_ms: Option<u64>,
    /// Error message if unhealthy
    pub error: Option<String>,
}

impl ComponentHealth {
    /// Create healthy status
    #[must_use]
    pub fn healthy(version: impl Into<String>) -> Self {
        Self {
            status: HealthStatus::Healthy,
            version: Some(version.into()),
            response_time_ms: None,
            error: None,
        }
    }

    /// Create unhealthy status
    #[must_use]
    pub fn unhealthy(error: impl Into<String>) -> Self {
        Self {
            status: HealthStatus::Unhealthy,
            version: None,
            response_time_ms: None,
            error: Some(error.into()),
        }
    }

    /// Create degraded status
    #[must_use]
    pub fn degraded(version: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            status: HealthStatus::Degraded,
            version: Some(version.into()),
            response_time_ms: None,
            error: Some(reason.into()),
        }
    }

    /// Set response time
    #[must_use]
    pub const fn with_response_time(mut self, ms: u64) -> Self {
        self.response_time_ms = Some(ms);
        self
    }
}

/// Health status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HealthStatus {
    /// Component is functioning normally
    Healthy,
    /// Component has issues but is operational
    Degraded,
    /// Component is not operational
    Unhealthy,
    /// Status unknown
    #[default]
    Unknown,
}

impl HealthStatus {
    /// Status name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
            Self::Unknown => "unknown",
        }
    }

    /// Is operational (healthy or degraded)?
    #[must_use]
    pub const fn is_operational(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Format compatibility information
#[derive(Debug, Clone, Copy)]
pub struct FormatCompatibility {
    /// Aprender format version
    pub apr_version: (u8, u8),
    /// Alimentar format version
    pub ald_version: (u8, u8),
    /// Are formats compatible?
    pub compatible: bool,
}

impl FormatCompatibility {
    /// Current version compatibility
    #[must_use]
    pub const fn current() -> Self {
        Self {
            apr_version: (1, 0),
            ald_version: (1, 2),
            compatible: true,
        }
    }

    /// Check if a specific APR version is compatible
    #[must_use]
    pub const fn is_apr_compatible(&self, major: u8, minor: u8) -> bool {
        major == self.apr_version.0 && minor <= self.apr_version.1
    }

    /// Check if a specific ALD version is compatible
    #[must_use]
    pub const fn is_ald_compatible(&self, major: u8, minor: u8) -> bool {
        major == self.ald_version.0 && minor <= self.ald_version.1
    }
}

impl Default for FormatCompatibility {
    fn default() -> Self {
        Self::current()
    }
}

#[cfg(test)]
mod tests;
