//! Mock Runtime Module for WASM Callback Testing
//!
//! Per `PROBAR-SPEC-WASM-001`, this module provides a mock runtime for testing
//! WASM callback patterns without browser APIs.
//!
//! ## Iron Lotus Philosophy
//!
//! > "Test the code, not the model"
//!
//! Property tests on models do NOT catch bugs in actual code. This mock runtime
//! allows testing the ACTUAL code's callback behavior in a controlled environment.
//!
//! ## Example
//!
//! ```rust,ignore
//! use probar::mock::{MockWasmRuntime, MockMessage, WasmCallbackTestHarness};
//!
//! let harness = WasmCallbackTestHarness::<WorkerManager>::new();
//! harness.worker.spawn("model.apr").unwrap();
//! harness.assert_state("spawning");
//!
//! // Simulate worker ready message
//! harness.runtime.receive_message(MockMessage::Ready);
//! harness.runtime.tick();
//! harness.assert_state("loading");  // Would FAIL with state sync bug!
//! ```

#[cfg(test)]
mod falsification_tests;
pub mod strategies;
pub mod test_harness;
pub mod wasm_runtime;

#[cfg(feature = "proptest")]
pub use strategies::{
    any_mock_message, error_heavy_sequence, realistic_lifecycle, valid_message_sequence,
};
pub use strategies::{edge_case_messages, error_test_messages, standard_test_messages};
pub use test_harness::{StateAssertion, TestStep, WasmCallbackTestHarness};
pub use wasm_runtime::{MockMessage, MockWasmRuntime, MockableWorker};
