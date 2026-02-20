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

include!("middleware.rs");
include!("gateway_part_03.rs");
