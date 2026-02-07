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
mod tests {
    use super::*;

    #[test]
    fn test_qos_default() {
        let qos = QoSRequirements::default();
        assert_eq!(qos.privacy, PrivacyLevel::Internal);
        assert!(qos.prefer_gpu);
        assert_eq!(qos.cost_tolerance, 50);
    }

    #[test]
    fn test_privacy_ordering() {
        assert!(PrivacyLevel::Public < PrivacyLevel::Internal);
        assert!(PrivacyLevel::Internal < PrivacyLevel::Confidential);
        assert!(PrivacyLevel::Confidential < PrivacyLevel::Restricted);
    }

    #[test]
    fn test_health_state() {
        let healthy = HealthState::Healthy;
        let degraded = HealthState::Degraded;
        assert_ne!(healthy, degraded);
    }

    #[test]
    fn test_circuit_state() {
        let closed = CircuitState::Closed;
        let open = CircuitState::Open;
        assert_ne!(closed, open);
    }

    #[test]
    fn test_federation_builder() {
        let builder =
            FederationBuilder::new().with_load_balance(LoadBalanceStrategy::LeastConnections);

        assert!(matches!(
            builder.load_balance,
            LoadBalanceStrategy::LeastConnections
        ));
    }

    #[test]
    fn test_model_id_equality() {
        let id1 = ModelId("whisper-v3".to_string());
        let id2 = ModelId("whisper-v3".to_string());
        let id3 = ModelId("llama-7b".to_string());

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_capability_variants() {
        let cap1 = Capability::Transcribe;
        let cap2 = Capability::Custom("sentiment".to_string());

        assert_ne!(cap1, cap2);
        assert_eq!(cap1, Capability::Transcribe);
    }

    // =========================================================================
    // Display trait tests
    // =========================================================================

    #[test]
    fn test_model_id_display() {
        let id = ModelId("whisper-v3".to_string());
        assert_eq!(format!("{}", id), "whisper-v3");
        assert_eq!(id.to_string(), "whisper-v3");
    }

    #[test]
    fn test_model_id_display_empty() {
        let id = ModelId(String::new());
        assert_eq!(format!("{}", id), "");
    }

    #[test]
    fn test_region_id_display() {
        let id = RegionId("us-west-2".to_string());
        assert_eq!(format!("{}", id), "us-west-2");
        assert_eq!(id.to_string(), "us-west-2");
    }

    #[test]
    fn test_region_id_display_empty() {
        let id = RegionId(String::new());
        assert_eq!(format!("{}", id), "");
    }

    #[test]
    fn test_node_id_display() {
        let id = NodeId("gpu-node-01".to_string());
        assert_eq!(format!("{}", id), "gpu-node-01");
        assert_eq!(id.to_string(), "gpu-node-01");
    }

    #[test]
    fn test_node_id_display_empty() {
        let id = NodeId(String::new());
        assert_eq!(format!("{}", id), "");
    }

    // =========================================================================
    // Hash trait tests
    // =========================================================================

    #[test]
    fn test_model_id_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ModelId("a".to_string()));
        set.insert(ModelId("b".to_string()));
        set.insert(ModelId("a".to_string())); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_region_id_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(RegionId("us-west".to_string()));
        set.insert(RegionId("eu-west".to_string()));
        set.insert(RegionId("us-west".to_string())); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_node_id_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(NodeId("node1".to_string()));
        set.insert(NodeId("node2".to_string()));
        set.insert(NodeId("node1".to_string())); // duplicate
        assert_eq!(set.len(), 2);
    }

    // =========================================================================
    // Clone/Eq trait tests
    // =========================================================================

    #[test]
    fn test_model_id_clone() {
        let id = ModelId("test".to_string());
        let cloned = id.clone();
        assert_eq!(id, cloned);
    }

    #[test]
    fn test_region_id_equality() {
        let a = RegionId("us-west".to_string());
        let b = RegionId("us-west".to_string());
        let c = RegionId("eu-west".to_string());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_node_id_equality() {
        let a = NodeId("node-1".to_string());
        let b = NodeId("node-1".to_string());
        let c = NodeId("node-2".to_string());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // =========================================================================
    // Capability exhaustive tests
    // =========================================================================

    #[test]
    fn test_all_capability_variants() {
        let caps = vec![
            Capability::Transcribe,
            Capability::Synthesize,
            Capability::Generate,
            Capability::Code,
            Capability::Embed,
            Capability::ImageGen,
            Capability::Custom("my_cap".to_string()),
        ];

        // All should be distinct from each other
        for (i, a) in caps.iter().enumerate() {
            for (j, b) in caps.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_capability_custom_equality() {
        let a = Capability::Custom("sentiment".to_string());
        let b = Capability::Custom("sentiment".to_string());
        let c = Capability::Custom("other".to_string());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_capability_debug_format() {
        let cap = Capability::Transcribe;
        let debug = format!("{:?}", cap);
        assert_eq!(debug, "Transcribe");

        let custom = Capability::Custom("test".to_string());
        let debug = format!("{:?}", custom);
        assert!(debug.contains("Custom"));
        assert!(debug.contains("test"));
    }

    // =========================================================================
    // PrivacyLevel tests
    // =========================================================================

    #[test]
    fn test_privacy_level_all_orderings() {
        let levels = [
            PrivacyLevel::Public,
            PrivacyLevel::Internal,
            PrivacyLevel::Confidential,
            PrivacyLevel::Restricted,
        ];

        // Verify strictly increasing
        for i in 0..levels.len() - 1 {
            assert!(levels[i] < levels[i + 1]);
            assert!(levels[i + 1] > levels[i]);
        }
    }

    #[test]
    fn test_privacy_level_copy() {
        let level = PrivacyLevel::Confidential;
        let copied = level;
        assert_eq!(level, copied);
    }

    #[test]
    fn test_privacy_level_eq() {
        assert_eq!(PrivacyLevel::Public, PrivacyLevel::Public);
        assert_ne!(PrivacyLevel::Public, PrivacyLevel::Internal);
    }

    // =========================================================================
    // QoSRequirements tests
    // =========================================================================

    #[test]
    fn test_qos_default_none_fields() {
        let qos = QoSRequirements::default();
        assert!(qos.max_latency.is_none());
        assert!(qos.min_throughput.is_none());
    }

    #[test]
    fn test_qos_custom_values() {
        let qos = QoSRequirements {
            max_latency: Some(Duration::from_secs(2)),
            min_throughput: Some(100),
            privacy: PrivacyLevel::Restricted,
            prefer_gpu: false,
            cost_tolerance: 10,
        };
        assert_eq!(qos.max_latency, Some(Duration::from_secs(2)));
        assert_eq!(qos.min_throughput, Some(100));
        assert_eq!(qos.privacy, PrivacyLevel::Restricted);
        assert!(!qos.prefer_gpu);
        assert_eq!(qos.cost_tolerance, 10);
    }

    // =========================================================================
    // InferenceRequest/Response tests
    // =========================================================================

    #[test]
    fn test_inference_request_construction() {
        let req = InferenceRequest {
            capability: Capability::Generate,
            input: b"hello world".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "req-123".to_string(),
            tenant_id: Some("tenant-1".to_string()),
        };
        assert_eq!(req.request_id, "req-123");
        assert_eq!(req.tenant_id, Some("tenant-1".to_string()));
        assert_eq!(req.input, b"hello world");
    }

    #[test]
    fn test_inference_request_no_tenant() {
        let req = InferenceRequest {
            capability: Capability::Embed,
            input: vec![],
            qos: QoSRequirements::default(),
            request_id: "req-456".to_string(),
            tenant_id: None,
        };
        assert!(req.tenant_id.is_none());
    }

    #[test]
    fn test_inference_request_clone() {
        let req = InferenceRequest {
            capability: Capability::Code,
            input: b"fn main()".to_vec(),
            qos: QoSRequirements::default(),
            request_id: "req-789".to_string(),
            tenant_id: None,
        };
        let cloned = req.clone();
        assert_eq!(cloned.request_id, "req-789");
        assert_eq!(cloned.input, b"fn main()");
    }

    #[test]
    fn test_inference_response_construction() {
        let resp = InferenceResponse {
            output: b"generated text".to_vec(),
            served_by: NodeId("node-42".to_string()),
            latency: Duration::from_millis(150),
            tokens: Some(25),
        };
        assert_eq!(resp.output, b"generated text");
        assert_eq!(resp.served_by, NodeId("node-42".to_string()));
        assert_eq!(resp.latency, Duration::from_millis(150));
        assert_eq!(resp.tokens, Some(25));
    }

    #[test]
    fn test_inference_response_no_tokens() {
        let resp = InferenceResponse {
            output: vec![],
            served_by: NodeId("node-1".to_string()),
            latency: Duration::from_millis(10),
            tokens: None,
        };
        assert!(resp.tokens.is_none());
    }

    // =========================================================================
    // FederationError tests
    // =========================================================================

    #[test]
    fn test_federation_error_no_capacity() {
        let err = FederationError::NoCapacity(Capability::Transcribe);
        let msg = format!("{}", err);
        assert!(msg.contains("No nodes available"));
        assert!(msg.contains("Transcribe"));
    }

    #[test]
    fn test_federation_error_all_nodes_unhealthy() {
        let err = FederationError::AllNodesUnhealthy(Capability::Generate);
        let msg = format!("{}", err);
        assert!(msg.contains("All nodes unhealthy"));
        assert!(msg.contains("Generate"));
    }

    #[test]
    fn test_federation_error_qos_violation() {
        let err = FederationError::QoSViolation("latency too high".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("QoS requirements cannot be met"));
        assert!(msg.contains("latency too high"));
    }

    #[test]
    fn test_federation_error_privacy_violation() {
        let err = FederationError::PrivacyViolation("data must stay in EU".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Privacy policy violation"));
        assert!(msg.contains("data must stay in EU"));
    }

    #[test]
    fn test_federation_error_node_unreachable() {
        let err = FederationError::NodeUnreachable(NodeId("dead-node".to_string()));
        let msg = format!("{}", err);
        assert!(msg.contains("Node unreachable"));
        assert!(msg.contains("dead-node"));
    }

    #[test]
    fn test_federation_error_timeout() {
        let err = FederationError::Timeout(Duration::from_secs(30));
        let msg = format!("{}", err);
        assert!(msg.contains("Timeout"));
        assert!(msg.contains("30"));
    }

    #[test]
    fn test_federation_error_circuit_open() {
        let err = FederationError::CircuitOpen(NodeId("overloaded".to_string()));
        let msg = format!("{}", err);
        assert!(msg.contains("Circuit breaker open"));
        assert!(msg.contains("overloaded"));
    }

    #[test]
    fn test_federation_error_internal() {
        let err = FederationError::Internal("unexpected state".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("Internal error"));
        assert!(msg.contains("unexpected state"));
    }

    #[test]
    fn test_federation_error_debug() {
        let err = FederationError::NoCapacity(Capability::Embed);
        let debug = format!("{:?}", err);
        assert!(debug.contains("NoCapacity"));
    }

    // =========================================================================
    // HealthState tests
    // =========================================================================

    #[test]
    fn test_health_state_all_variants() {
        let states = [
            HealthState::Healthy,
            HealthState::Degraded,
            HealthState::Unhealthy,
            HealthState::Unknown,
        ];
        // All distinct
        for (i, a) in states.iter().enumerate() {
            for (j, b) in states.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_health_state_copy() {
        let state = HealthState::Healthy;
        let copied = state;
        assert_eq!(state, copied);
    }

    // =========================================================================
    // CircuitState tests
    // =========================================================================

    #[test]
    fn test_circuit_state_all_variants() {
        let states = [
            CircuitState::Closed,
            CircuitState::HalfOpen,
            CircuitState::Open,
        ];
        for (i, a) in states.iter().enumerate() {
            for (j, b) in states.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_circuit_state_copy() {
        let state = CircuitState::HalfOpen;
        let copied = state;
        assert_eq!(state, copied);
    }

    // =========================================================================
    // LoadBalanceStrategy tests
    // =========================================================================

    #[test]
    fn test_load_balance_default() {
        let strategy = LoadBalanceStrategy::default();
        assert!(matches!(strategy, LoadBalanceStrategy::LeastLatency));
    }

    #[test]
    fn test_load_balance_all_variants() {
        let strategies = [
            LoadBalanceStrategy::RoundRobin,
            LoadBalanceStrategy::LeastConnections,
            LoadBalanceStrategy::LeastLatency,
            LoadBalanceStrategy::WeightedRandom,
            LoadBalanceStrategy::ConsistentHash,
        ];
        // Verify debug output
        for s in &strategies {
            let debug = format!("{:?}", s);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_load_balance_clone() {
        let strategy = LoadBalanceStrategy::WeightedRandom;
        let cloned = strategy;
        assert!(matches!(cloned, LoadBalanceStrategy::WeightedRandom));
    }

    // =========================================================================
    // RouteTarget/RouteCandidate/RouteScores tests
    // =========================================================================

    #[test]
    fn test_route_target_construction() {
        let target = RouteTarget {
            node_id: NodeId("n1".to_string()),
            region_id: RegionId("r1".to_string()),
            endpoint: "http://n1:8080".to_string(),
            estimated_latency: Duration::from_millis(50),
            score: 0.95,
        };
        assert_eq!(target.node_id, NodeId("n1".to_string()));
        assert_eq!(target.endpoint, "http://n1:8080");
        assert_eq!(target.estimated_latency, Duration::from_millis(50));
    }

    #[test]
    fn test_route_target_clone() {
        let target = RouteTarget {
            node_id: NodeId("n1".to_string()),
            region_id: RegionId("r1".to_string()),
            endpoint: "http://n1:8080".to_string(),
            estimated_latency: Duration::from_millis(50),
            score: 0.5,
        };
        let cloned = target.clone();
        assert_eq!(cloned.node_id, NodeId("n1".to_string()));
        assert_eq!(cloned.score, 0.5);
    }

    #[test]
    fn test_route_scores_construction() {
        let scores = RouteScores {
            latency_score: 0.9,
            throughput_score: 0.8,
            cost_score: 0.7,
            locality_score: 0.6,
            health_score: 1.0,
            total: 0.85,
        };
        assert_eq!(scores.latency_score, 0.9);
        assert_eq!(scores.total, 0.85);
    }

    #[test]
    fn test_route_candidate_eligible() {
        let candidate = RouteCandidate {
            target: RouteTarget {
                node_id: NodeId("n1".to_string()),
                region_id: RegionId("r1".to_string()),
                endpoint: String::new(),
                estimated_latency: Duration::from_millis(100),
                score: 0.8,
            },
            scores: RouteScores {
                latency_score: 0.9,
                throughput_score: 0.8,
                cost_score: 0.5,
                locality_score: 0.7,
                health_score: 1.0,
                total: 0.8,
            },
            eligible: true,
            rejection_reason: None,
        };
        assert!(candidate.eligible);
        assert!(candidate.rejection_reason.is_none());
    }

    #[test]
    fn test_route_candidate_rejected() {
        let candidate = RouteCandidate {
            target: RouteTarget {
                node_id: NodeId("n1".to_string()),
                region_id: RegionId("r1".to_string()),
                endpoint: String::new(),
                estimated_latency: Duration::from_millis(100),
                score: 0.0,
            },
            scores: RouteScores {
                latency_score: 0.0,
                throughput_score: 0.0,
                cost_score: 0.0,
                locality_score: 0.0,
                health_score: 0.0,
                total: 0.0,
            },
            eligible: false,
            rejection_reason: Some("Policy rejected".to_string()),
        };
        assert!(!candidate.eligible);
        assert_eq!(
            candidate.rejection_reason,
            Some("Policy rejected".to_string())
        );
    }

    // =========================================================================
    // ModelMetadata tests
    // =========================================================================

    #[test]
    fn test_model_metadata_construction() {
        let meta = ModelMetadata {
            model_id: ModelId("llama-7b".to_string()),
            name: "LLaMA 7B".to_string(),
            version: "2.0".to_string(),
            capabilities: vec![Capability::Generate, Capability::Code],
            parameters: 7_000_000_000,
            quantization: Some("Q4_K".to_string()),
        };
        assert_eq!(meta.name, "LLaMA 7B");
        assert_eq!(meta.parameters, 7_000_000_000);
        assert_eq!(meta.quantization, Some("Q4_K".to_string()));
        assert_eq!(meta.capabilities.len(), 2);
    }

    #[test]
    fn test_model_metadata_no_quantization() {
        let meta = ModelMetadata {
            model_id: ModelId("whisper".to_string()),
            name: "Whisper".to_string(),
            version: "1.0".to_string(),
            capabilities: vec![Capability::Transcribe],
            parameters: 1_500_000_000,
            quantization: None,
        };
        assert!(meta.quantization.is_none());
    }

    #[test]
    fn test_model_metadata_clone() {
        let meta = ModelMetadata {
            model_id: ModelId("test".to_string()),
            name: "Test".to_string(),
            version: "1.0".to_string(),
            capabilities: vec![Capability::Embed],
            parameters: 100,
            quantization: None,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.model_id, ModelId("test".to_string()));
    }

    // =========================================================================
    // NodeHealth tests
    // =========================================================================

    #[test]
    fn test_node_health_construction() {
        let health = NodeHealth {
            node_id: NodeId("test-node".to_string()),
            status: HealthState::Healthy,
            latency_p50: Duration::from_millis(25),
            latency_p99: Duration::from_millis(100),
            throughput: 500,
            gpu_utilization: Some(0.75),
            queue_depth: 3,
            last_check: std::time::Instant::now(),
        };
        assert_eq!(health.status, HealthState::Healthy);
        assert_eq!(health.throughput, 500);
        assert_eq!(health.gpu_utilization, Some(0.75));
        assert_eq!(health.queue_depth, 3);
    }

    #[test]
    fn test_node_health_no_gpu() {
        let health = NodeHealth {
            node_id: NodeId("cpu-node".to_string()),
            status: HealthState::Healthy,
            latency_p50: Duration::from_millis(50),
            latency_p99: Duration::from_millis(200),
            throughput: 100,
            gpu_utilization: None,
            queue_depth: 0,
            last_check: std::time::Instant::now(),
        };
        assert!(health.gpu_utilization.is_none());
    }

    // =========================================================================
    // GatewayStats tests
    // =========================================================================

    #[test]
    fn test_gateway_stats_default() {
        let stats = GatewayStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_requests, 0);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.avg_latency, Duration::ZERO);
        assert_eq!(stats.active_streams, 0);
    }

    #[test]
    fn test_gateway_stats_clone() {
        let stats = GatewayStats {
            total_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            total_tokens: 5000,
            avg_latency: Duration::from_millis(50),
            active_streams: 2,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_requests, 100);
        assert_eq!(cloned.active_streams, 2);
    }

    // =========================================================================
    // FederationBuilder tests
    // =========================================================================

    #[test]
    fn test_federation_builder_default() {
        let builder = FederationBuilder::default();
        assert!(builder.catalog.is_none());
        assert!(builder.health_checker.is_none());
        assert!(builder.router.is_none());
        assert!(builder.policies.is_empty());
        assert!(builder.middlewares.is_empty());
    }

    #[test]
    fn test_federation_builder_new_defaults() {
        let builder = FederationBuilder::new();
        assert!(matches!(
            builder.load_balance,
            LoadBalanceStrategy::LeastLatency
        ));
    }

    #[test]
    fn test_federation_builder_with_load_balance_all_strategies() {
        for strategy in [
            LoadBalanceStrategy::RoundRobin,
            LoadBalanceStrategy::LeastConnections,
            LoadBalanceStrategy::LeastLatency,
            LoadBalanceStrategy::WeightedRandom,
            LoadBalanceStrategy::ConsistentHash,
        ] {
            let builder = FederationBuilder::new().with_load_balance(strategy);
            let debug = format!("{:?}", builder.load_balance);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_federation_builder_with_policy() {
        use super::*;

        struct MockPolicy;
        impl RoutingPolicyTrait for MockPolicy {
            fn score(&self, _: &RouteCandidate, _: &InferenceRequest) -> f64 {
                1.0
            }
            fn is_eligible(&self, _: &RouteCandidate, _: &InferenceRequest) -> bool {
                true
            }
            fn name(&self) -> &'static str {
                "mock"
            }
        }

        let builder = FederationBuilder::new()
            .with_policy(MockPolicy)
            .with_policy(MockPolicy);
        assert_eq!(builder.policies.len(), 2);
    }

    #[test]
    fn test_federation_builder_with_middleware() {
        use super::*;

        struct MockMiddleware;
        impl GatewayMiddleware for MockMiddleware {
            fn before_route(&self, _: &mut InferenceRequest) -> FederationResult<()> {
                Ok(())
            }
            fn after_infer(
                &self,
                _: &InferenceRequest,
                _: &mut InferenceResponse,
            ) -> FederationResult<()> {
                Ok(())
            }
            fn on_error(&self, _: &InferenceRequest, _: &FederationError) {}
        }

        let builder = FederationBuilder::new().with_middleware(MockMiddleware);
        assert_eq!(builder.middlewares.len(), 1);
    }
}
