//! APR Federation - Smart Model Routing & Catalog
//!
//! Enterprise-grade model federation for distributed inference.
//!
//! ## Architecture
//!
//! ```text
//!                     ┌─────────────────┐
//!                     │   Gateway       │
//!                     │  (Router)       │
//!                     └────────┬────────┘
//!                              │
//!         ┌────────────────────┼────────────────────┐
//!         ▼                    ▼                    ▼
//!   ┌──────────┐        ┌──────────┐        ┌──────────┐
//!   │ Region A │        │ Region B │        │ Region C │
//!   └──────────┘        └──────────┘        └──────────┘
//! ```
//!
//! ## Design Principles
//!
//! 1. **Capability-based routing** - Route by what models can do, not where they are
//! 2. **Policy-driven** - Latency, cost, privacy, compliance as first-class citizens
//! 3. **Zero-copy where possible** - Streaming responses without buffering
//! 4. **Graceful degradation** - Circuit breakers, fallbacks, retries

pub mod catalog;
pub mod gateway;
pub mod health;
pub mod policy;
pub mod routing;
pub mod traits;
pub mod tui;

pub use catalog::{ModelCatalog, ModelEntry};
pub use gateway::{FederationGateway, GatewayBuilder, GatewayConfig};
pub use health::{CircuitBreaker, HealthChecker, HealthConfig, HealthStatus};
pub use policy::{CompositePolicy, RoutingPolicy, SelectionCriteria};
pub use routing::{RouteDecision, Router, RouterBuilder, RouterConfig};
pub use traits::*;
pub use tui::{render_federation_dashboard, FederationApp, FederationTab};
