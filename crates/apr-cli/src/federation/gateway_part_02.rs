
impl GatewayBuilder {
    pub fn new() -> Self {
        Self {
            config: GatewayConfig::default(),
            catalog: None,
            health: None,
            circuit_breaker: None,
            router: None,
            middlewares: Vec::new(),
        }
    }

    #[must_use]
    pub fn config(mut self, config: GatewayConfig) -> Self {
        self.config = config;
        self
    }

    #[must_use]
    pub fn catalog(mut self, catalog: Arc<ModelCatalog>) -> Self {
        self.catalog = Some(catalog);
        self
    }

    #[must_use]
    pub fn health(mut self, health: Arc<HealthChecker>) -> Self {
        self.health = Some(health);
        self
    }

    #[must_use]
    pub fn circuit_breaker(mut self, cb: Arc<CircuitBreaker>) -> Self {
        self.circuit_breaker = Some(cb);
        self
    }

    #[must_use]
    pub fn router(mut self, router: Arc<Router>) -> Self {
        self.router = Some(router);
        self
    }

    #[must_use]
    pub fn middleware(mut self, middleware: impl GatewayMiddleware + 'static) -> Self {
        self.middlewares.push(Box::new(middleware));
        self
    }

    pub fn build(self) -> FederationGateway {
        let catalog = self
            .catalog
            .unwrap_or_else(|| Arc::new(ModelCatalog::new()));
        let health = self
            .health
            .unwrap_or_else(|| Arc::new(HealthChecker::default()));
        let circuit_breaker = self
            .circuit_breaker
            .unwrap_or_else(|| Arc::new(CircuitBreaker::default()));

        let router = self.router.unwrap_or_else(|| {
            Arc::new(Router::new(
                super::routing::RouterConfig::default(),
                Arc::clone(&catalog),
                Arc::clone(&health),
                Arc::clone(&circuit_breaker),
            ))
        });

        let mut gateway = FederationGateway::new(self.config, router, health, circuit_breaker);

        for middleware in self.middlewares {
            gateway.middlewares.push(middleware);
        }

        gateway
    }
}

impl Default for GatewayBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Example Middlewares
// ============================================================================

/// Logging middleware
pub struct LoggingMiddleware {
    prefix: String,
}

impl LoggingMiddleware {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }
}

impl GatewayMiddleware for LoggingMiddleware {
    fn before_route(&self, request: &mut InferenceRequest) -> FederationResult<()> {
        eprintln!(
            "[{}] Routing request {} for {:?}",
            self.prefix, request.request_id, request.capability
        );
        Ok(())
    }

    fn after_infer(
        &self,
        request: &InferenceRequest,
        response: &mut InferenceResponse,
    ) -> FederationResult<()> {
        eprintln!(
            "[{}] Request {} served by {:?} in {:?}",
            self.prefix, request.request_id, response.served_by, response.latency
        );
        Ok(())
    }

    fn on_error(&self, request: &InferenceRequest, error: &FederationError) {
        eprintln!(
            "[{}] Request {} failed: {}",
            self.prefix, request.request_id, error
        );
    }
}

/// Rate limiting middleware
pub struct RateLimitMiddleware {
    #[allow(dead_code)]
    requests_per_second: u32,
    // In production, would use a token bucket or sliding window
}

impl RateLimitMiddleware {
    pub fn new(requests_per_second: u32) -> Self {
        Self {
            requests_per_second,
        }
    }
}

impl GatewayMiddleware for RateLimitMiddleware {
    fn before_route(&self, _request: &mut InferenceRequest) -> FederationResult<()> {
        // In production, would check rate limit and return error if exceeded
        // For now, always allow
        Ok(())
    }

    fn after_infer(
        &self,
        _request: &InferenceRequest,
        _response: &mut InferenceResponse,
    ) -> FederationResult<()> {
        Ok(())
    }

    fn on_error(&self, _request: &InferenceRequest, _error: &FederationError) {}
}
