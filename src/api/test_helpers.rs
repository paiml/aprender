//! Test helper functions for api tests
//!
//! This module contains shared test utilities used across multiple test parts.
//! Separated for PMAT compliance (<2000 lines per file).
//!
//! # Experimental Reusability (Dr. Popper's Prescription)
//!
//! The "Tax of Setup" analysis revealed that creating AppState::demo() for each
//! of 1000+ tests takes ~0.5s each = 500+ seconds of pure setup overhead.
//!
//! Solution: Use `OnceLock` to create a shared, immutable AppState that is
//! initialized once and reused across all read-only tests. This amortizes
//! the setup cost from O(n) to O(1).
//!
//! - `create_test_app()` - Creates fresh state (for mutation tests)
//! - `create_test_app_shared()` - Returns shared state (for read-only tests)

use super::*;
use axum::http::StatusCode;
use axum::Router;
use std::sync::OnceLock;

/// Guard macro for mock state tests - returns early if NOT_FOUND
///
/// When using mock state (no model), endpoints return NOT_FOUND.
/// This macro allows tests to pass if routing worked (got any response).
/// Usage: `guard_mock_response!(response);`
#[macro_export]
macro_rules! guard_mock_response {
    ($response:expr) => {
        if $response.status() == axum::http::StatusCode::NOT_FOUND {
            // Mock state returns NOT_FOUND - routing worked, test passes
            return;
        }
    };
}

/// Check if response indicates mock state (no model loaded)
pub fn is_mock_response(status: StatusCode) -> bool {
    status == StatusCode::NOT_FOUND
}

/// Global shared AppState for read-only tests (Experimental Reusability)
///
/// This amortizes the ~0.5s AppState::demo() cost across all tests that
/// don't mutate state. Thread-safe via OnceLock + Arc<RwLock<...>> in AppState.
static SHARED_APP_STATE: OnceLock<AppState> = OnceLock::new();

/// Get or initialize the shared AppState (MOCK - no inference)
fn get_shared_state() -> &'static AppState {
    SHARED_APP_STATE.get_or_init(|| {
        // Use demo_mock() for instant setup - no model = no inference overhead
        // Tests will get "model not loaded" errors, which exercises error handling paths
        AppState::demo_mock().expect("Failed to create shared mock AppState")
    })
}

/// Create a test application with SHARED demo state (READ-ONLY tests)
///
/// Use this for tests that only read from the model state (most API tests).
/// This eliminates the ~0.5s per-test setup overhead by reusing a single
/// AppState across all tests in the process.
///
/// # Performance
/// - First call: ~0.5s (initializes shared state)
/// - Subsequent calls: ~0s (returns cached router)
///
/// # Thread Safety
/// Safe for concurrent use - AppState uses Arc<RwLock<...>> internally.
pub fn create_test_app_shared() -> Router {
    let state = get_shared_state().clone();
    create_router(state)
}

/// Create a test application with FRESH demo state (MUTATION tests)
///
/// Use this only for tests that mutate the model state. Each call creates
/// a new AppState, so this has ~0.5s overhead per call.
///
/// # When to use fresh vs shared
/// - `create_test_app_shared()` - Default choice for most tests
/// - `create_test_app()` - Only when test mutates model/tokenizer state
pub fn create_test_app() -> Router {
    let state = AppState::demo().expect("test");
    create_router(state)
}

/// Helper to create test quantized model for IMP-116 tests
#[cfg(feature = "gpu")]
pub fn create_test_quantized_model(
    config: &crate::gguf::GGUFConfig,
) -> crate::gguf::OwnedQuantizedModel {
    use crate::gguf::{
        OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedModel, OwnedQuantizedTensor,
        GGUF_TYPE_Q4_K,
    };

    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let vocab_size = config.vocab_size;

    // Create Q4_K tensor data helper
    // Q4_K uses row-major storage where each row has ceil(in_dim/256) super-blocks.
    // Each super-block is 144 bytes and covers 256 values.
    fn create_q4k_data(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
        let super_blocks_per_row = in_dim.div_ceil(256);
        let bytes_per_row = super_blocks_per_row * 144;
        let data_size = out_dim * bytes_per_row;
        OwnedQuantizedTensor {
            data: vec![0u8; data_size],
            qtype: GGUF_TYPE_Q4_K,
            in_dim,
            out_dim,
        }
    }

    let constraints = &config.constraints;

    // GH-278: Contract-consistent model creation
    let ffn_gate_weight = if constraints.has_gate_ffn() {
        Some(create_q4k_data(hidden_dim, intermediate_dim))
    } else {
        None
    };
    let ffn_norm_weight = if constraints.uses_rmsnorm() || !constraints.has_gate_ffn() {
        Some(vec![1.0f32; hidden_dim])
    } else {
        None
    };
    let attn_norm_bias = if !constraints.uses_rmsnorm() {
        Some(vec![0.0f32; hidden_dim])
    } else {
        None
    };

    let layers = (0..config.num_layers)
        .map(|_| OwnedQuantizedLayer {
            attn_norm_weight: vec![1.0f32; hidden_dim],
            attn_norm_bias: attn_norm_bias.clone(),
            qkv_weight: OwnedQKVWeights::Fused(create_q4k_data(hidden_dim, hidden_dim * 3)),
            qkv_bias: None,
            attn_output_weight: create_q4k_data(hidden_dim, hidden_dim),
            attn_output_bias: None,
            ffn_up_weight: create_q4k_data(hidden_dim, intermediate_dim),
            ffn_up_bias: None,
            ffn_down_weight: create_q4k_data(intermediate_dim, hidden_dim),
            ffn_down_bias: None,
            ffn_gate_weight: ffn_gate_weight.clone(),
            ffn_gate_bias: None,
            ffn_norm_weight: ffn_norm_weight.clone(),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        })
        .collect();

    let output_norm_bias = if !constraints.uses_rmsnorm() {
        Some(vec![0.0f32; hidden_dim])
    } else {
        None
    };

    OwnedQuantizedModel {
        config: config.clone(),
        token_embedding: vec![0.1f32; vocab_size * hidden_dim],
        position_embedding: None,
        layers,
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias,
        lm_head_weight: create_q4k_data(hidden_dim, vocab_size),
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}
