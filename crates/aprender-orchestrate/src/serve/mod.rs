//! Model Serving Ecosystem
//!
//! Unified interface for local and remote model serving across the ML ecosystem.
//!
//! ## Components
//!
//! - `ChatTemplateEngine` - Unified prompt templating (Llama2, Mistral, ChatML)
//! - `BackendSelector` - Intelligent backend selection with privacy tiers
//! - `CostCircuitBreaker` - Daily budget limits to prevent runaway costs
//! - `ContextManager` - Automatic token counting and truncation
//! - `StatefulFailover` - Streaming failover with context preservation
//! - `SpilloverRouter` - Hybrid cloud spillover routing
//! - `LambdaDeployer` - AWS Lambda inference deployment
//!
//! ## Toyota Way Principles
//!
//! - Standardized Work: Chat templates ensure consistent model interaction
//! - Poka-Yoke: Privacy gates prevent accidental data leakage
//! - Jidoka: Stateful failover maintains context on errors
//! - Muda Elimination: Cost circuit breakers prevent waste

pub mod backends;
#[cfg(feature = "banco")]
pub mod banco;
pub mod circuit_breaker;
pub mod context;
pub mod failover;
pub mod lambda;
pub mod router;
pub mod templates;

// Re-export key types for convenience
pub use backends::{BackendSelector, LatencyTier, PrivacyTier, ServingBackend};
pub use circuit_breaker::{CircuitBreakerConfig, CostCircuitBreaker, TokenPricing};
pub use context::{ContextManager, ContextWindow, TokenEstimator, TruncationStrategy};
pub use failover::{FailoverConfig, FailoverManager, StreamingContext};
pub use lambda::{LambdaConfig, LambdaDeployer, LambdaRuntime};
pub use router::{RejectReason, RouterConfig, RoutingDecision, SpilloverRouter};
pub use templates::{ChatMessage, ChatTemplateEngine, Role, TemplateFormat};
