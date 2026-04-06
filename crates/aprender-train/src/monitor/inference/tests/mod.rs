//! Property Tests for Inference Monitor (ENT-112)
//!
//! 200K+ proptest iterations for trace serialization, hash chain integrity, etc.
//!
//! Tests are organized into logical groups:
//! - `helpers` - Arbitrary generators for property tests
//! - `linear_path_tests` - LinearPath serialization and property tests
//! - `tree_path_tests` - TreePath serialization and property tests
//! - `decision_trace_tests` - DecisionTrace serialization and property tests
//! - `collector_tests` - RingCollector and HashChainCollector tests
//! - `counterfactual_tests` - Counterfactual explanation tests
//! - `serialization_tests` - Binary and JSON serialization tests
//! - `hash_tests` - FNV-1a hash and feature hash tests
//! - `safety_andon_tests` - Safety integrity level tests
//! - `provenance_tests` - Provenance graph tests
//! - `inference_monitor_tests` - High-level InferenceMonitor tests

mod helpers;

mod collector_tests;
mod counterfactual_tests;
mod decision_trace_tests;
mod hash_tests;
mod inference_monitor_tests;
mod linear_path_tests;
mod provenance_tests;
mod safety_andon_tests;
mod serialization_tests;
mod tree_path_tests;
