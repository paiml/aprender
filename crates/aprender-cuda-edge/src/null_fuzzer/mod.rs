//! Null pointer fuzzing for GPU kernel arguments.
//!
//! This module provides tools to systematically inject null pointers into
//! kernel calls and track how (or if) the null propagates through execution.
//!
//! # Components
//!
//! - [`NonNullDevicePtr`]: A guard type that rejects null (0) addresses at construction
//! - [`InjectionStrategy`]: Controls when and how null injection occurs
//! - [`NullSentinelFuzzer`]: Drives the fuzzing session
//! - [`PropagationTracker`]: Records call chains when nulls propagate
//!
//! # Example
//!
//! ```
//! use trueno_cuda_edge::null_fuzzer::{
//!     NonNullDevicePtr, InjectionStrategy, NullSentinelFuzzer, NullFuzzerConfig
//! };
//!
//! // Guard type rejects null
//! assert!(NonNullDevicePtr::<u8>::new(0).is_err());
//! assert!(NonNullDevicePtr::<u8>::new(0x1000).is_ok());
//!
//! // Configure periodic injection
//! let config = NullFuzzerConfig {
//!     strategy: InjectionStrategy::Periodic { interval: 10 },
//!     total_calls: 100,
//!     fail_fast: false,
//! };
//! let mut fuzzer = NullSentinelFuzzer::new(config);
//! assert!(fuzzer.next_call()); // call 0: inject
//! ```

pub mod guard_types;
pub mod propagation;
pub mod sentinel;

pub use guard_types::{InjectionStrategy, NonNullDevicePtr};
pub use propagation::{PropagationFrame, PropagationOutcome, PropagationPath, PropagationTracker};
pub use sentinel::{NullFuzzerConfig, NullFuzzerReport, NullSentinelFuzzer};
