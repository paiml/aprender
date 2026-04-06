//! Brick implementations for cbtop
//!
//! Four-layer architecture following PROBAR-SPEC-009:
//! - Layer 1: Collectors - Data collection from hardware
//! - Layer 2: Analyzers - Business logic and derived metrics
//! - Layer 3: Panels - Rendering using presentar-terminal widgets
//! - Layer 4: Generators - Load generation

pub mod analyzers;
pub mod collectors;
pub mod generators;
pub mod panels;

// Re-exports
pub use analyzers::*;
pub use collectors::*;
pub use generators::*;
