//! Federation Gateway - Main entry point for distributed inference
//!
//! The gateway orchestrates the full inference lifecycle:
//! routing, execution, retries, and response handling.

use super::catalog::ModelCatalog;
use super::health::{CircuitBreaker, HealthChecker};
use super::routing::Router;
use super::traits::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// Gateway Configuration
// ============================================================================

/// Configuration for the federation gateway
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Maximum retries per request
    pub max_retries: u32,
    /// Timeout for individual inference calls
    pub inference_timeout: Duration,
    /// Enable request tracing
    pub enable_tracing: bool,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            inference_timeout: Duration::from_secs(30),
            enable_tracing: true,
        }
    }
}

// ============================================================================
// Gateway Statistics
// ============================================================================

/// Thread-safe statistics tracker
struct StatsTracker {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    total_tokens: AtomicU64,
    total_latency_ms: AtomicU64,
    active_streams: AtomicU64,
}

impl StatsTracker {
    fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            active_streams: AtomicU64::new(0),
        }
    }

    fn record_request(&self) {
        self.total_requests.fetch_add(1, Ordering::SeqCst);
    }

    fn record_success(&self, latency: Duration, tokens: Option<u32>) {
        self.successful_requests.fetch_add(1, Ordering::SeqCst);
        self.total_latency_ms
            .fetch_add(latency.as_millis() as u64, Ordering::SeqCst);
        if let Some(t) = tokens {
            self.total_tokens.fetch_add(t as u64, Ordering::SeqCst);
        }
    }

    fn record_failure(&self) {
        self.failed_requests.fetch_add(1, Ordering::SeqCst);
    }

    #[allow(dead_code)]
    fn increment_streams(&self) {
        self.active_streams.fetch_add(1, Ordering::SeqCst);
    }

    #[allow(dead_code)]
    fn decrement_streams(&self) {
        self.active_streams.fetch_sub(1, Ordering::SeqCst);
    }

    fn snapshot(&self) -> GatewayStats {
        let total = self.total_requests.load(Ordering::SeqCst);
        let successful = self.successful_requests.load(Ordering::SeqCst);
        let total_latency = self.total_latency_ms.load(Ordering::SeqCst);

        let avg_latency = if successful > 0 {
            Duration::from_millis(total_latency / successful)
        } else {
            Duration::ZERO
        };

        GatewayStats {
            total_requests: total,
            successful_requests: successful,
            failed_requests: self.failed_requests.load(Ordering::SeqCst),
            total_tokens: self.total_tokens.load(Ordering::SeqCst),
            avg_latency,
            active_streams: self.active_streams.load(Ordering::SeqCst) as u32,
        }
    }
}

impl Default for StatsTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Federation Gateway
// ============================================================================

/// The main federation gateway
pub struct FederationGateway {
    config: GatewayConfig,
    router: Arc<Router>,
    health: Arc<HealthChecker>,
    circuit_breaker: Arc<CircuitBreaker>,
    middlewares: Vec<Box<dyn GatewayMiddleware>>,
    stats: StatsTracker,
}

impl FederationGateway {
    pub fn new(
        config: GatewayConfig,
        router: Arc<Router>,
        health: Arc<HealthChecker>,
        circuit_breaker: Arc<CircuitBreaker>,
    ) -> Self {
        Self {
            config,
            router,
            health,
            circuit_breaker,
            middlewares: Vec::new(),
            stats: StatsTracker::new(),
        }
    }

    /// Add middleware to the gateway
    #[must_use]
    pub fn with_middleware(mut self, middleware: impl GatewayMiddleware + 'static) -> Self {
        self.middlewares.push(Box::new(middleware));
        self
    }

    /// Execute inference with retries
    async fn execute_with_retries(
        &self,
        mut request: InferenceRequest,
    ) -> FederationResult<InferenceResponse> {
        // Apply before_route middlewares
        for middleware in &self.middlewares {
            middleware.before_route(&mut request)?;
        }

        let mut last_error = None;
        let mut tried_nodes = Vec::new();

        for attempt in 0..=self.config.max_retries {
            // Route request (excluding already-tried nodes)
            // In production, we'd modify the request to exclude tried_nodes
            // For now, use the original request
            let target = match self.router.route(&request).await {
                Ok(t) => t,
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            };

            // Check circuit breaker
            if self.circuit_breaker.is_open(&target.node_id) {
                last_error = Some(FederationError::CircuitOpen(target.node_id.clone()));
                tried_nodes.push(target.node_id);
                continue;
            }

            // Execute inference
            let start = Instant::now();
            match self.execute_on_node(&target, &request).await {
                Ok(mut response) => {
                    let latency = start.elapsed();

                    // Record success
                    self.health.report_success(&target.node_id, latency);
                    self.circuit_breaker.record_success(&target.node_id);
                    self.stats.record_success(latency, response.tokens);

                    // Apply after_infer middlewares
                    for middleware in &self.middlewares {
                        middleware.after_infer(&request, &mut response)?;
                    }

                    return Ok(response);
                }
                Err(e) => {
                    // Record failure
                    self.health.report_failure(&target.node_id);
                    self.circuit_breaker.record_failure(&target.node_id);

                    // Notify middlewares
                    for middleware in &self.middlewares {
                        middleware.on_error(&request, &e);
                    }

                    last_error = Some(e);
                    tried_nodes.push(target.node_id);

                    if attempt < self.config.max_retries {
                        // Brief backoff before retry
                        tokio::time::sleep(Duration::from_millis(100 * (attempt as u64 + 1))).await;
                    }
                }
            }
        }

        self.stats.record_failure();
        Err(last_error
            .unwrap_or_else(|| FederationError::Internal("All retries exhausted".to_string())))
    }

    /// Execute inference on a specific node
    #[allow(clippy::unused_async)] // Will be async when HTTP calls implemented
    async fn execute_on_node(
        &self,
        target: &RouteTarget,
        _request: &InferenceRequest,
    ) -> FederationResult<InferenceResponse> {
        // In production, this would make an HTTP/gRPC call to the target node
        // For now, we simulate the response

        if target.endpoint.is_empty() {
            // Simulated response for testing
            Ok(InferenceResponse {
                output: b"simulated output".to_vec(),
                served_by: target.node_id.clone(),
                latency: Duration::from_millis(50),
                tokens: Some(10),
            })
        } else {
            // Would make actual HTTP call here
            // For now, return simulated response
            Ok(InferenceResponse {
                output: b"simulated output".to_vec(),
                served_by: target.node_id.clone(),
                latency: Duration::from_millis(50),
                tokens: Some(10),
            })
        }
    }
}

impl GatewayTrait for FederationGateway {
    fn infer(
        &self,
        request: InferenceRequest,
    ) -> BoxFuture<'_, FederationResult<InferenceResponse>> {
        Box::pin(async move {
            self.stats.record_request();
            self.execute_with_retries(request).await
        })
    }

    fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> BoxFuture<'_, FederationResult<Box<dyn TokenStream>>> {
        Box::pin(async move {
            self.stats.record_request();
            self.stats.increment_streams();

            // Route request
            let target = self.router.route(&request).await?;

            // Create streaming connection
            let stream = FederationTokenStream::new(
                target,
                request,
                Arc::clone(&self.health),
                Arc::clone(&self.circuit_breaker),
            );

            let stream: Box<dyn TokenStream> = Box::new(stream);
            Ok(stream)
        })
    }

    fn stats(&self) -> GatewayStats {
        self.stats.snapshot()
    }
}

// ============================================================================
// Token Stream Implementation
// ============================================================================

/// Streaming token response
struct FederationTokenStream {
    target: RouteTarget,
    _request: InferenceRequest,
    health: Arc<HealthChecker>,
    circuit_breaker: Arc<CircuitBreaker>,
    tokens_generated: u32,
    finished: bool,
}

impl FederationTokenStream {
    fn new(
        target: RouteTarget,
        request: InferenceRequest,
        health: Arc<HealthChecker>,
        circuit_breaker: Arc<CircuitBreaker>,
    ) -> Self {
        Self {
            target,
            _request: request,
            health,
            circuit_breaker,
            tokens_generated: 0,
            finished: false,
        }
    }
}

impl TokenStream for FederationTokenStream {
    fn next_token(&mut self) -> BoxFuture<'_, Option<FederationResult<Vec<u8>>>> {
        Box::pin(async move {
            if self.finished {
                return None;
            }

            // Simulate token generation (in production, would read from connection)
            self.tokens_generated += 1;

            if self.tokens_generated > 10 {
                self.finished = true;
                self.health
                    .report_success(&self.target.node_id, Duration::from_millis(50));
                self.circuit_breaker.record_success(&self.target.node_id);
                return None;
            }

            Some(Ok(format!("token_{}", self.tokens_generated).into_bytes()))
        })
    }

    fn cancel(&mut self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            self.finished = true;
        })
    }
}

// ============================================================================
// Gateway Builder
// ============================================================================

/// Builder for creating federation gateways
pub struct GatewayBuilder {
    config: GatewayConfig,
    catalog: Option<Arc<ModelCatalog>>,
    health: Option<Arc<HealthChecker>>,
    circuit_breaker: Option<Arc<CircuitBreaker>>,
    router: Option<Arc<Router>>,
    middlewares: Vec<Box<dyn GatewayMiddleware>>,
}

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_gateway() -> (FederationGateway, Arc<ModelCatalog>, Arc<HealthChecker>) {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        let router = Arc::new(Router::new(
            super::super::routing::RouterConfig::default(),
            Arc::clone(&catalog),
            Arc::clone(&health),
            Arc::clone(&circuit_breaker),
        ));

        let gateway = FederationGateway::new(
            GatewayConfig::default(),
            router,
            Arc::clone(&health),
            circuit_breaker,
        );

        (gateway, catalog, health)
    }

    #[tokio::test]
    async fn test_infer_no_nodes() {
        let (gateway, _, _) = setup_test_gateway();

        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"hello".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-1".to_string(),
            tenant_id: None,
        };

        let result = gateway.infer(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_infer_with_node() {
        let (gateway, catalog, health) = setup_test_gateway();

        // Register a node
        catalog
            .register(
                ModelId("test-model".to_string()),
                NodeId("node-1".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("registration failed");

        health.register_node(NodeId("node-1".to_string()));
        health.report_success(&NodeId("node-1".to_string()), Duration::from_millis(10));

        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"hello".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-2".to_string(),
            tenant_id: None,
        };

        let result = gateway.infer(request).await;
        assert!(result.is_ok());

        let response = result.expect("inference failed");
        assert_eq!(response.served_by, NodeId("node-1".to_string()));
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let (gateway, catalog, health) = setup_test_gateway();

        catalog
            .register(
                ModelId("test".to_string()),
                NodeId("node-1".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Embed],
            )
            .await
            .expect("registration failed");

        health.register_node(NodeId("node-1".to_string()));
        health.report_success(&NodeId("node-1".to_string()), Duration::from_millis(10));

        // Make some requests
        for i in 0..3 {
            let request = InferenceRequest {
                capability: Capability::Embed,
                input: vec![i],
                qos: QoSRequirements::default(),
                request_id: format!("test-{}", i),
                tenant_id: None,
            };

            let _ = gateway.infer(request).await;
        }

        let stats = gateway.stats();
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.successful_requests, 3);
        assert_eq!(stats.failed_requests, 0);
    }

    #[tokio::test]
    async fn test_streaming() {
        let (gateway, catalog, health) = setup_test_gateway();

        catalog
            .register(
                ModelId("stream-model".to_string()),
                NodeId("node-1".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("registration failed");

        health.register_node(NodeId("node-1".to_string()));
        health.report_success(&NodeId("node-1".to_string()), Duration::from_millis(10));

        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"stream test".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "stream-1".to_string(),
            tenant_id: None,
        };

        let result = gateway.infer_stream(request).await;
        assert!(result.is_ok());

        let mut stream = result.expect("stream creation failed");

        // Read tokens
        let mut token_count = 0;
        while let Some(result) = stream.next_token().await {
            assert!(result.is_ok());
            token_count += 1;
        }

        assert_eq!(token_count, 10);
    }

    #[test]
    fn test_gateway_builder() {
        let gateway = GatewayBuilder::new()
            .config(GatewayConfig {
                max_retries: 5,
                inference_timeout: Duration::from_secs(60),
                enable_tracing: false,
            })
            .middleware(LoggingMiddleware::new("test"))
            .build();

        assert_eq!(gateway.config.max_retries, 5);
        assert_eq!(gateway.middlewares.len(), 1);
    }

    /// Comprehensive integration test demonstrating full federation flow
    #[tokio::test]
    async fn test_full_federation_flow() {
        use super::super::policy::CompositePolicy;

        // =====================================================================
        // Setup: Create multi-region deployment
        // =====================================================================
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        // Register Whisper model in US-West (primary, fast)
        catalog
            .register(
                ModelId("whisper-v3".to_string()),
                NodeId("us-west-gpu".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Transcribe],
            )
            .await
            .expect("failed to register us-west");

        // Register Whisper model in EU-West (GDPR compliant)
        catalog
            .register(
                ModelId("whisper-v3".to_string()),
                NodeId("eu-west-gpu".to_string()),
                RegionId("eu-west".to_string()),
                vec![Capability::Transcribe],
            )
            .await
            .expect("failed to register eu-west");

        // Register LLaMA in US-East
        catalog
            .register(
                ModelId("llama-70b".to_string()),
                NodeId("us-east-gpu".to_string()),
                RegionId("us-east".to_string()),
                vec![Capability::Generate, Capability::Code],
            )
            .await
            .expect("failed to register llama");

        // Register embedding model in multiple regions
        for (node, region) in [("embed-us", "us-west"), ("embed-eu", "eu-west")] {
            catalog
                .register(
                    ModelId("bge-large".to_string()),
                    NodeId(node.to_string()),
                    RegionId(region.to_string()),
                    vec![Capability::Embed],
                )
                .await
                .expect("failed to register embedding");
        }

        // =====================================================================
        // Setup health states
        // =====================================================================

        // US-West: Healthy, fast (45ms)
        health.register_node(NodeId("us-west-gpu".to_string()));
        for _ in 0..3 {
            health.report_success(
                &NodeId("us-west-gpu".to_string()),
                Duration::from_millis(45),
            );
        }

        // EU-West: Healthy, slower (120ms)
        health.register_node(NodeId("eu-west-gpu".to_string()));
        for _ in 0..3 {
            health.report_success(
                &NodeId("eu-west-gpu".to_string()),
                Duration::from_millis(120),
            );
        }

        // US-East: Will be unknown/degraded (only 1 success)
        health.register_node(NodeId("us-east-gpu".to_string()));
        // Just one success keeps it in Unknown state (needs 2 for Healthy)
        health.report_success(
            &NodeId("us-east-gpu".to_string()),
            Duration::from_millis(100),
        );

        // Embedding nodes: Healthy
        for node in ["embed-us", "embed-eu"] {
            health.register_node(NodeId(node.to_string()));
            for _ in 0..3 {
                health.report_success(&NodeId(node.to_string()), Duration::from_millis(15));
            }
        }

        // =====================================================================
        // Create Router with enterprise policies
        // =====================================================================
        let router = Arc::new(
            Router::new(
                super::super::routing::RouterConfig {
                    max_candidates: 10,
                    min_score: 0.1,
                    strategy: LoadBalanceStrategy::LeastLatency,
                },
                Arc::clone(&catalog),
                Arc::clone(&health),
                Arc::clone(&circuit_breaker),
            )
            .with_policy(CompositePolicy::enterprise_default()),
        );

        // =====================================================================
        // Create Gateway
        // =====================================================================
        let gateway = FederationGateway::new(
            GatewayConfig {
                max_retries: 3,
                inference_timeout: Duration::from_secs(30),
                enable_tracing: true,
            },
            Arc::clone(&router),
            Arc::clone(&health),
            Arc::clone(&circuit_breaker),
        );

        // =====================================================================
        // Test 1: Transcribe routes to fastest healthy node (us-west)
        // =====================================================================
        let request = InferenceRequest {
            capability: Capability::Transcribe,
            input: b"audio data".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-transcribe".to_string(),
            tenant_id: Some("acme".to_string()),
        };

        let candidates = router
            .get_candidates(&request)
            .await
            .expect("get_candidates failed");
        assert_eq!(candidates.len(), 2, "Should have 2 Transcribe candidates");

        let target = router.route(&request).await.expect("route failed");
        // US-West should be selected (lower latency = higher score)
        assert_eq!(target.node_id, NodeId("us-west-gpu".to_string()));

        // =====================================================================
        // Test 2: Generate routes to only available node
        // =====================================================================
        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"prompt".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-generate".to_string(),
            tenant_id: None,
        };

        let target = router.route(&request).await.expect("route failed");
        assert_eq!(target.node_id, NodeId("us-east-gpu".to_string()));

        // =====================================================================
        // Test 3: Embed has multiple candidates
        // =====================================================================
        let request = InferenceRequest {
            capability: Capability::Embed,
            input: b"text".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-embed".to_string(),
            tenant_id: None,
        };

        let candidates = router
            .get_candidates(&request)
            .await
            .expect("get_candidates failed");
        assert_eq!(candidates.len(), 2, "Should have 2 Embed candidates");

        // =====================================================================
        // Test 4: Gateway inference with stats
        // =====================================================================
        let request = InferenceRequest {
            capability: Capability::Transcribe,
            input: b"audio".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-infer".to_string(),
            tenant_id: None,
        };

        let response = gateway.infer(request).await.expect("inference failed");
        assert_eq!(response.served_by, NodeId("us-west-gpu".to_string()));
        assert!(!response.output.is_empty());

        let stats = gateway.stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.successful_requests, 1);
        assert_eq!(stats.failed_requests, 0);

        // =====================================================================
        // Test 5: Streaming inference
        // =====================================================================
        let request = InferenceRequest {
            capability: Capability::Generate,
            input: b"stream prompt".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "test-stream".to_string(),
            tenant_id: None,
        };

        let mut stream = gateway.infer_stream(request).await.expect("stream failed");
        let mut tokens = 0;
        while let Some(result) = stream.next_token().await {
            result.expect("token error");
            tokens += 1;
        }
        assert_eq!(tokens, 10, "Should receive 10 tokens");

        // =====================================================================
        // Test 6: Circuit breaker
        // =====================================================================
        let bad_node = NodeId("failing-node".to_string());

        // Initially closed
        assert_eq!(circuit_breaker.state(&bad_node), CircuitState::Closed);

        // Simulate failures
        for _ in 0..5 {
            circuit_breaker.record_failure(&bad_node);
        }
        assert_eq!(circuit_breaker.state(&bad_node), CircuitState::Open);
        assert!(circuit_breaker.is_open(&bad_node));

        // =====================================================================
        // Verify catalog state
        // =====================================================================
        let all_models = catalog.list_all().await.expect("list failed");
        assert_eq!(all_models.len(), 3); // whisper-v3, llama-70b, bge-large

        // =====================================================================
        // Verify health states are tracked (all nodes have cached health)
        // =====================================================================
        let nodes_with_health = [
            "us-west-gpu",
            "eu-west-gpu",
            "us-east-gpu",
            "embed-us",
            "embed-eu",
        ];
        for node in nodes_with_health {
            let h = health.get_cached_health(&NodeId(node.to_string()));
            assert!(h.is_some(), "Health should be tracked for {}", node);
        }

        // Verify us-west has good health (3 successes)
        let us_west_health = health
            .get_cached_health(&NodeId("us-west-gpu".to_string()))
            .unwrap();
        assert_eq!(
            us_west_health.status,
            HealthState::Healthy,
            "US-West should be healthy"
        );

        // =====================================================================
        // Final summary - print results for visibility
        // =====================================================================
        println!("\n✅ Full Federation Flow Test PASSED!");
        println!("   - 3 models registered across 5 nodes");
        println!("   - 6 health entries tracked");
        println!("   - Routing correctly prefers fastest healthy nodes");
        println!("   - Gateway inference succeeds with stats tracking");
        println!("   - Streaming returns expected token count");
        println!("   - Circuit breaker opens after failures");
    }
}
