//! Core trait definitions for APR Federation
//!
//! These traits define the contract for federation components.
//! Implementations can be swapped for different backends (NATS, Redis, etcd, etc.)

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

// ============================================================================
// Core Types
// ============================================================================

/// Unique identifier for a model instance in the federation
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ModelId(pub String);

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a region/cluster
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RegionId(pub String);

impl std::fmt::Display for RegionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a node within a region
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct NodeId(pub String);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Model capabilities that can be queried
#[derive(Debug, Clone, PartialEq)]
pub enum Capability {
    /// Automatic speech recognition
    Transcribe,
    /// Text-to-speech
    Synthesize,
    /// Text generation (LLM)
    Generate,
    /// Code generation
    Code,
    /// Embeddings
    Embed,
    /// Image generation
    ImageGen,
    /// Custom capability
    Custom(String),
}

/// Privacy/compliance level for data routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrivacyLevel {
    /// Public data, can route anywhere
    Public = 0,
    /// Internal only, keep within org
    Internal = 1,
    /// Confidential, specific regions only
    Confidential = 2,
    /// Restricted, on-prem only
    Restricted = 3,
}

/// Quality of Service requirements
#[derive(Debug, Clone)]
pub struct QoSRequirements {
    /// Maximum acceptable latency
    pub max_latency: Option<Duration>,
    /// Minimum throughput (tokens/sec)
    pub min_throughput: Option<u32>,
    /// Privacy level required
    pub privacy: PrivacyLevel,
    /// Prefer GPU acceleration
    pub prefer_gpu: bool,
    /// Cost tier (0 = cheapest, 100 = fastest)
    pub cost_tolerance: u8,
}

impl Default for QoSRequirements {
    fn default() -> Self {
        Self {
            max_latency: None,
            min_throughput: None,
            privacy: PrivacyLevel::Internal,
            prefer_gpu: true,
            cost_tolerance: 50,
        }
    }
}

/// Inference request metadata
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Requested capability
    pub capability: Capability,
    /// Input data (opaque bytes)
    pub input: Vec<u8>,
    /// QoS requirements
    pub qos: QoSRequirements,
    /// Request ID for tracing
    pub request_id: String,
    /// User/tenant ID
    pub tenant_id: Option<String>,
}

/// Inference response
#[derive(Debug)]
pub struct InferenceResponse {
    /// Output data
    pub output: Vec<u8>,
    /// Which node handled the request
    pub served_by: NodeId,
    /// Actual latency
    pub latency: Duration,
    /// Tokens generated (if applicable)
    pub tokens: Option<u32>,
}

/// Error types for federation operations
#[derive(Debug, thiserror::Error)]
pub enum FederationError {
    #[error("No nodes available for capability: {0:?}")]
    NoCapacity(Capability),

    #[error("All nodes unhealthy for capability: {0:?}")]
    AllNodesUnhealthy(Capability),

    #[error("QoS requirements cannot be met: {0}")]
    QoSViolation(String),

    #[error("Privacy policy violation: {0}")]
    PrivacyViolation(String),

    #[error("Node unreachable: {0}")]
    NodeUnreachable(NodeId),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Circuit breaker open for node: {0}")]
    CircuitOpen(NodeId),

    #[error("Internal error: {0}")]
    Internal(String),
}

// ============================================================================
// Core Traits
// ============================================================================

/// Result type alias for federation operations
pub type FederationResult<T> = Result<T, FederationError>;

/// Boxed future for async trait methods
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Model catalog - tracks available models across the federation
pub trait ModelCatalogTrait: Send + Sync {
    /// Register a model instance
    fn register(
        &self,
        model_id: ModelId,
        node_id: NodeId,
        region_id: RegionId,
        capabilities: Vec<Capability>,
    ) -> BoxFuture<'_, FederationResult<()>>;

    /// Deregister a model instance
    fn deregister(&self, model_id: ModelId, node_id: NodeId)
        -> BoxFuture<'_, FederationResult<()>>;

    /// Find nodes with a specific capability
    fn find_by_capability(
        &self,
        capability: &Capability,
    ) -> BoxFuture<'_, FederationResult<Vec<(NodeId, RegionId)>>>;

    /// List all registered models
    fn list_all(&self) -> BoxFuture<'_, FederationResult<Vec<ModelId>>>;

    /// Get model metadata
    fn get_metadata(&self, model_id: &ModelId) -> BoxFuture<'_, FederationResult<ModelMetadata>>;
}

/// Model metadata stored in catalog
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub model_id: ModelId,
    pub name: String,
    pub version: String,
    pub capabilities: Vec<Capability>,
    pub parameters: u64,
    pub quantization: Option<String>,
}

/// Health checker - monitors node health across federation
pub trait HealthCheckerTrait: Send + Sync {
    /// Check health of a specific node
    fn check_node(&self, node_id: &NodeId) -> BoxFuture<'_, FederationResult<NodeHealth>>;

    /// Get cached health status (non-blocking)
    fn get_cached_health(&self, node_id: &NodeId) -> Option<NodeHealth>;

    /// Start background health monitoring
    fn start_monitoring(&self, interval: Duration) -> BoxFuture<'_, ()>;

    /// Stop health monitoring
    fn stop_monitoring(&self) -> BoxFuture<'_, ()>;
}

/// Node health information
#[derive(Debug, Clone)]
pub struct NodeHealth {
    pub node_id: NodeId,
    pub status: HealthState,
    pub latency_p50: Duration,
    pub latency_p99: Duration,
    pub throughput: u32,
    pub gpu_utilization: Option<f32>,
    pub queue_depth: u32,
    pub last_check: std::time::Instant,
}

/// Health state enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthState {
    /// Node is healthy and accepting requests
    Healthy,
    /// Node is degraded but functional
    Degraded,
    /// Node is unhealthy, avoid routing
    Unhealthy,
    /// Node health is unknown
    Unknown,
}

/// Router - selects the best node for a request
pub trait RouterTrait: Send + Sync {
    /// Route a request to the best available node
    fn route(&self, request: &InferenceRequest) -> BoxFuture<'_, FederationResult<RouteTarget>>;

    /// Get all possible routes for a request (for debugging/transparency)
    fn get_candidates(
        &self,
        request: &InferenceRequest,
    ) -> BoxFuture<'_, FederationResult<Vec<RouteCandidate>>>;
}

/// Selected route target
#[derive(Debug, Clone)]
pub struct RouteTarget {
    pub node_id: NodeId,
    pub region_id: RegionId,
    pub endpoint: String,
    pub estimated_latency: Duration,
    pub score: f64,
}

/// Route candidate with scoring details
#[derive(Debug, Clone)]
pub struct RouteCandidate {
    pub target: RouteTarget,
    pub scores: RouteScores,
    pub eligible: bool,
    pub rejection_reason: Option<String>,
}

/// Breakdown of routing scores
#[derive(Debug, Clone)]
pub struct RouteScores {
    pub latency_score: f64,
    pub throughput_score: f64,
    pub cost_score: f64,
    pub locality_score: f64,
    pub health_score: f64,
    pub total: f64,
}

/// Gateway - the main entry point for federation requests
pub trait GatewayTrait: Send + Sync {
    /// Execute an inference request through the federation
    fn infer(
        &self,
        request: InferenceRequest,
    ) -> BoxFuture<'_, FederationResult<InferenceResponse>>;

    /// Execute with streaming response
    fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> BoxFuture<'_, FederationResult<Box<dyn TokenStream>>>;

    /// Get gateway statistics
    fn stats(&self) -> GatewayStats;
}

/// Streaming token interface
pub trait TokenStream: Send {
    /// Get next token (None = stream complete)
    fn next_token(&mut self) -> BoxFuture<'_, Option<FederationResult<Vec<u8>>>>;

    /// Cancel the stream
    fn cancel(&mut self) -> BoxFuture<'_, ()>;
}

/// Gateway statistics
#[derive(Debug, Clone, Default)]
pub struct GatewayStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_tokens: u64,
    pub avg_latency: Duration,
    pub active_streams: u32,
}

// ============================================================================
// Middleware Traits (Tower-style composability)
// ============================================================================

/// Middleware that can wrap a gateway
pub trait GatewayMiddleware: Send + Sync {
    /// Process request before routing
    fn before_route(&self, request: &mut InferenceRequest) -> FederationResult<()>;

    /// Process response after inference
    fn after_infer(
        &self,
        request: &InferenceRequest,
        response: &mut InferenceResponse,
    ) -> FederationResult<()>;

    /// Handle errors
    fn on_error(&self, request: &InferenceRequest, error: &FederationError);
}

/// Circuit breaker for fault tolerance
pub trait CircuitBreakerTrait: Send + Sync {
    /// Check if circuit is open (should skip this node)
    fn is_open(&self, node_id: &NodeId) -> bool;

    /// Record a success
    fn record_success(&self, node_id: &NodeId);

    /// Record a failure
    fn record_failure(&self, node_id: &NodeId);

    /// Get circuit state
    fn state(&self, node_id: &NodeId) -> CircuitState;
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation
    Closed,
    /// Failing, allowing probe requests
    HalfOpen,
    /// Failing, blocking all requests
    Open,
}

// ============================================================================
// Policy Traits
// ============================================================================

/// Routing policy that influences node selection
pub trait RoutingPolicyTrait: Send + Sync {
    /// Score a candidate node (higher = better)
    fn score(&self, candidate: &RouteCandidate, request: &InferenceRequest) -> f64;

    /// Check if a candidate is eligible
    fn is_eligible(&self, candidate: &RouteCandidate, request: &InferenceRequest) -> bool;

    /// Get policy name for logging
    fn name(&self) -> &'static str;
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, Default)]
pub enum LoadBalanceStrategy {
    /// Round-robin across healthy nodes
    RoundRobin,
    /// Route to least loaded node
    LeastConnections,
    /// Route based on latency
    #[default]
    LeastLatency,
    /// Weighted random
    WeightedRandom,
    /// Consistent hashing (sticky sessions)
    ConsistentHash,
}

// ============================================================================
// Builder Pattern for Configuration
// ============================================================================

/// Builder for creating federation gateways
#[derive(Default)]
pub struct FederationBuilder {
    pub catalog: Option<Box<dyn ModelCatalogTrait>>,
    pub health_checker: Option<Box<dyn HealthCheckerTrait>>,
    pub router: Option<Box<dyn RouterTrait>>,
    pub policies: Vec<Box<dyn RoutingPolicyTrait>>,
    pub middlewares: Vec<Box<dyn GatewayMiddleware>>,
    pub load_balance: LoadBalanceStrategy,
}

impl FederationBuilder {
    pub fn new() -> Self {
        Self {
            load_balance: LoadBalanceStrategy::LeastLatency,
            ..Default::default()
        }
    }

    #[must_use]
    pub fn with_catalog(mut self, catalog: impl ModelCatalogTrait + 'static) -> Self {
        self.catalog = Some(Box::new(catalog));
        self
    }

    #[must_use]
    pub fn with_health_checker(mut self, checker: impl HealthCheckerTrait + 'static) -> Self {
        self.health_checker = Some(Box::new(checker));
        self
    }

    #[must_use]
    pub fn with_router(mut self, router: impl RouterTrait + 'static) -> Self {
        self.router = Some(Box::new(router));
        self
    }

    #[must_use]
    pub fn with_policy(mut self, policy: impl RoutingPolicyTrait + 'static) -> Self {
        self.policies.push(Box::new(policy));
        self
    }

    #[must_use]
    pub fn with_middleware(mut self, middleware: impl GatewayMiddleware + 'static) -> Self {
        self.middlewares.push(Box::new(middleware));
        self
    }

    #[must_use]
    pub fn with_load_balance(mut self, strategy: LoadBalanceStrategy) -> Self {
        self.load_balance = strategy;
        self
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "traits_tests.rs"]
mod tests;
