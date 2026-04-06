//! CUDA context lifecycle chaos testing.
//!
//! GPU contexts have complex lifecycle requirements: they must be created
//! before use, destroyed after use, and destruction order can matter. This
//! module provides tools to stress-test context lifecycle handling:
//!
//! - Enumerate 8 failure scenarios (double destroy, use-after-free, leaks, etc.)
//! - Generate all destruction orderings for N contexts
//! - Detect memory and context leaks
//!
//! # Chaos Scenarios
//!
//! 1. **Double Destroy**: Destroy a context twice
//! 2. **Use After Destroy**: Use a context after destruction
//! 3. **Leaked Context**: Create without destroying
//! 4. **Reverse Destruction**: Destroy in reverse creation order
//! 5. **Random Destruction**: Destroy in arbitrary order
//! 6. **Context Exhaustion**: Create more contexts than GPU supports
//! 7. **Cross-Thread Access**: Use context from wrong thread
//! 8. **Device Reset**: Reset GPU with active contexts
//!
//! # Example
//!
//! ```
//! use trueno_cuda_edge::lifecycle_chaos::{
//!     ChaosScenario, generate_destruction_orderings, ContextLeakDetector
//! };
//!
//! // All 8 scenarios
//! assert_eq!(ChaosScenario::all().len(), 8);
//!
//! // 3! = 6 orderings for 3 contexts
//! let orderings = generate_destruction_orderings(3);
//! assert_eq!(orderings.len(), 6);
//!
//! // Leak detection with 1MB tolerance
//! let detector = ContextLeakDetector::new();
//! let report = detector.analyze(100_000_000, 100_500_000);
//! assert!(!report.has_leaks()); // within tolerance
//! ```

pub mod context;
pub mod leak_detector;
pub mod ordering;

pub use context::{ChaosScenario, LifecycleChaosConfig};
pub use leak_detector::{ContextLeakDetector, Leak, LeakReport, LEAK_TOLERANCE_BYTES};
pub use ordering::{
    generate_destruction_orderings, validate_ordering, DestructionOrdering, OrderingValidation,
};
