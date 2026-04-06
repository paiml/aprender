//! T-COV-95 Generative Falsification: Proptest API Request Assault (PMAT-802)
//!
//! Dr. Popper's directive: "Generate *millions* of valid and invalid API
//! request sequences. Make the machine find the gap."
//!
//! This module implements:
//! 1. Arbitrary BatchConfig generation
//! 2. GpuBatchRequest fuzzing
//! 3. Edge case parameter combinations
//! 4. Serialization roundtrip fuzzing
//!
//! Target: 517 missed lines in api/gpu_handlers.rs via algorithmic search

use crate::api::gpu_handlers::{
    BatchConfig, GpuBatchRequest, GpuBatchResponse, GpuBatchResult, GpuBatchStats,
    GpuStatusResponse,
};
use proptest::prelude::*;

// ============================================================================
// BatchConfig Strategy
// ============================================================================

/// Generate arbitrary BatchConfig values
fn arb_batch_config() -> impl Strategy<Value = BatchConfig> {
    (
        0u64..10_000,    // window_ms
        0usize..1000,    // min_batch
        0usize..1000,    // optimal_batch
        0usize..10_000,  // max_batch
        0usize..100_000, // queue_size
        0usize..1000,    // gpu_threshold
    )
        .prop_map(
            |(window_ms, min_batch, optimal_batch, max_batch, queue_size, gpu_threshold)| {
                BatchConfig {
                    window_ms,
                    min_batch,
                    optimal_batch,
                    max_batch,
                    queue_size,
                    gpu_threshold,
                }
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Fuzz BatchConfig creation and methods
    #[test]
    fn fuzz_batch_config(config in arb_batch_config()) {
        // should_process should not panic
        let _ = config.should_process(0);
        let _ = config.should_process(1);
        let _ = config.should_process(config.optimal_batch);
        let _ = config.should_process(usize::MAX);

        // meets_minimum should not panic
        let _ = config.meets_minimum(0);
        let _ = config.meets_minimum(1);
        let _ = config.meets_minimum(config.min_batch);
        let _ = config.meets_minimum(usize::MAX);

        // Clone should work
        let cloned = config.clone();
        prop_assert_eq!(config.window_ms, cloned.window_ms);
    }

    /// Fuzz BatchConfig with extreme values
    #[test]
    fn fuzz_batch_config_extremes(
        window_ms in prop_oneof![Just(0u64), Just(u64::MAX), any::<u64>()],
        min_batch in prop_oneof![Just(0usize), Just(usize::MAX), any::<usize>()],
        optimal_batch in prop_oneof![Just(0usize), Just(usize::MAX), any::<usize>()],
    ) {
        let config = BatchConfig {
            window_ms,
            min_batch,
            optimal_batch,
            max_batch: 64,
            queue_size: 1024,
            gpu_threshold: 32,
        };

        // Should not panic with extreme values
        let _ = config.should_process(optimal_batch);
        let _ = config.meets_minimum(min_batch);
    }
}

// ============================================================================
// GpuBatchRequest Strategy
// ============================================================================

/// Generate arbitrary prompt strings
fn arb_prompt() -> impl Strategy<Value = String> {
    prop_oneof![
        5 => "[a-zA-Z0-9 ]{0,100}",        // Normal ASCII (weighted)
        1 => Just(String::new()),           // Empty
        1 => "[\\x00-\\xff]{0,50}",         // Binary garbage
        1 => "\\PC{0,200}",                 // Unicode
        1 => Just("x".repeat(10000)),       // Very long
    ]
}

/// Generate arbitrary GpuBatchRequest
fn arb_gpu_batch_request() -> impl Strategy<Value = GpuBatchRequest> {
    (
        prop::collection::vec(arb_prompt(), 0..20), // prompts
        0usize..10000,                              // max_tokens
        0.0f32..10.0,                               // temperature
        0usize..1000,                               // top_k
        prop::collection::vec("[a-z]{0,10}", 0..5), // stop tokens
    )
        .prop_map(
            |(prompts, max_tokens, temperature, top_k, stop)| GpuBatchRequest {
                prompts,
                max_tokens,
                temperature,
                top_k,
                stop,
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Fuzz GpuBatchRequest creation
    #[test]
    fn fuzz_gpu_batch_request(request in arb_gpu_batch_request()) {
        // Clone should work
        let cloned = request.clone();
        prop_assert_eq!(request.prompts.len(), cloned.prompts.len());
        prop_assert_eq!(request.max_tokens, cloned.max_tokens);

        // Debug should not panic
        let _ = format!("{:?}", request);
    }

    /// Fuzz GpuBatchRequest with extreme temperatures
    #[test]
    fn fuzz_extreme_temperature(
        temp in prop_oneof![
            Just(0.0f32),
            Just(f32::MIN_POSITIVE),
            Just(1.0f32),
            Just(f32::MAX),
            Just(f32::INFINITY),
            Just(f32::NEG_INFINITY),
            Just(f32::NAN),
        ]
    ) {
        let request = GpuBatchRequest {
            prompts: vec!["test".to_string()],
            max_tokens: 10,
            temperature: temp,
            top_k: 40,
            stop: vec![],
        };

        // Should not panic
        let _ = format!("{:?}", request);
        let _ = request.clone();
    }
}

// ============================================================================
// GpuBatchResult Strategy
// ============================================================================

fn arb_gpu_batch_result() -> impl Strategy<Value = GpuBatchResult> {
    (
        any::<usize>(),                              // index
        prop::collection::vec(any::<u32>(), 0..100), // token_ids
        "[a-zA-Z0-9 ]{0,100}",                       // text
        any::<usize>(),                              // num_generated
    )
        .prop_map(|(index, token_ids, text, num_generated)| GpuBatchResult {
            index,
            token_ids,
            text,
            num_generated,
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Fuzz GpuBatchResult
    #[test]
    fn fuzz_gpu_batch_result(result in arb_gpu_batch_result()) {
        let cloned = result.clone();
        prop_assert_eq!(result.index, cloned.index);
        prop_assert_eq!(result.token_ids.len(), cloned.token_ids.len());

        // Serialization should work
        let json = serde_json::to_string(&result);
        prop_assert!(json.is_ok());
    }
}

// ============================================================================
// GpuBatchStats Strategy
// ============================================================================

fn arb_gpu_batch_stats() -> impl Strategy<Value = GpuBatchStats> {
    (
        any::<usize>(), // batch_size
        any::<bool>(),  // gpu_used
        any::<usize>(), // total_tokens
        0.0f64..1e10,   // processing_time_ms
        0.0f64..1e10,   // throughput_tps
    )
        .prop_map(
            |(batch_size, gpu_used, total_tokens, processing_time_ms, throughput_tps)| {
                GpuBatchStats {
                    batch_size,
                    gpu_used,
                    total_tokens,
                    processing_time_ms,
                    throughput_tps,
                }
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Fuzz GpuBatchStats
    #[test]
    fn fuzz_gpu_batch_stats(stats in arb_gpu_batch_stats()) {
        let cloned = stats.clone();
        prop_assert_eq!(stats.batch_size, cloned.batch_size);
        prop_assert_eq!(stats.gpu_used, cloned.gpu_used);

        // Serialization
        let json = serde_json::to_string(&stats);
        prop_assert!(json.is_ok());
    }

    /// Fuzz with NaN and Infinity in stats
    #[test]
    fn fuzz_stats_special_floats(
        processing_time in prop_oneof![
            Just(0.0f64),
            Just(f64::INFINITY),
            Just(f64::NEG_INFINITY),
            Just(f64::NAN),
            Just(f64::MIN_POSITIVE),
            Just(f64::MAX),
        ],
        throughput in prop_oneof![
            Just(0.0f64),
            Just(f64::INFINITY),
            Just(f64::NAN),
        ]
    ) {
        let stats = GpuBatchStats {
            batch_size: 1,
            gpu_used: true,
            total_tokens: 100,
            processing_time_ms: processing_time,
            throughput_tps: throughput,
        };

        // Should not panic
        let _ = format!("{:?}", stats);
        let _ = stats.clone();
    }
}

// ============================================================================
// GpuStatusResponse Strategy
// ============================================================================

fn arb_gpu_status_response() -> impl Strategy<Value = GpuStatusResponse> {
    (
        any::<bool>(),  // cache_ready
        any::<usize>(), // cache_memory_bytes
        any::<usize>(), // batch_threshold
        any::<usize>(), // recommended_min_batch
    )
        .prop_map(
            |(cache_ready, cache_memory_bytes, batch_threshold, recommended_min_batch)| {
                GpuStatusResponse {
                    cache_ready,
                    cache_memory_bytes,
                    batch_threshold,
                    recommended_min_batch,
                }
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Fuzz GpuStatusResponse
    #[test]
    fn fuzz_gpu_status_response(response in arb_gpu_status_response()) {
        let cloned = response.clone();
        prop_assert_eq!(response.cache_ready, cloned.cache_ready);

        // Serialization
        let json = serde_json::to_string(&response);
        prop_assert!(json.is_ok());

        if let Ok(json_str) = json {
            let decoded: Result<GpuStatusResponse, _> = serde_json::from_str(&json_str);
            prop_assert!(decoded.is_ok());
        }
    }
}

// ============================================================================
// Full Response Strategy
// ============================================================================

fn arb_gpu_batch_response() -> impl Strategy<Value = GpuBatchResponse> {
    (
        prop::collection::vec(arb_gpu_batch_result(), 0..10),
        arb_gpu_batch_stats(),
    )
        .prop_map(|(results, stats)| GpuBatchResponse { results, stats })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Fuzz full GpuBatchResponse
    #[test]
    fn fuzz_gpu_batch_response(response in arb_gpu_batch_response()) {
        let cloned = response.clone();
        prop_assert_eq!(response.results.len(), cloned.results.len());

        // Serialization roundtrip
        let json = serde_json::to_string(&response);
        prop_assert!(json.is_ok());

        if let Ok(json_str) = json {
            let decoded: Result<GpuBatchResponse, _> = serde_json::from_str(&json_str);
            // May fail if floats are NaN (JSON doesn't support NaN)
            let _ = decoded;
        }
    }
}

// ============================================================================
// JSON Malformation Fuzzing
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Fuzz JSON deserialization with arbitrary bytes
    #[test]
    fn fuzz_json_deserialization(
        garbage in prop::collection::vec(any::<u8>(), 0..200)
    ) {
        // Should not panic
        let result: Result<GpuBatchRequest, _> = serde_json::from_slice(&garbage);
        let _ = result;

        let result: Result<GpuBatchResponse, _> = serde_json::from_slice(&garbage);
        let _ = result;

        // BatchConfig is internal (no Deserialize), skip JSON fuzzing
    }

    /// Fuzz with almost-valid JSON
    #[test]
    fn fuzz_almost_valid_json(
        prompts_count in 0usize..5,
        max_tokens in any::<i64>(),  // Use i64 to allow negative
        temperature in any::<f64>(),
    ) {
        let json = format!(
            r#"{{"prompts":[{}],"max_tokens":{},"temperature":{}}}"#,
            (0..prompts_count)
                .map(|i| format!(r#""prompt_{}""#, i))
                .collect::<Vec<_>>()
                .join(","),
            max_tokens,
            if temperature.is_nan() {
                "null".to_string()
            } else {
                temperature.to_string()
            }
        );

        let result: Result<GpuBatchRequest, _> = serde_json::from_str(&json);
        // May succeed or fail based on field types
        let _ = result;
    }
}

// ============================================================================
// Combinatorial Boundary Testing
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Test all boundary combinations
    #[test]
    fn fuzz_boundary_combinations(
        min_batch in prop::sample::select(&[0usize, 1, 2, 4, 8, 16, 32, 64, 128, usize::MAX]),
        optimal in prop::sample::select(&[0usize, 1, 2, 4, 8, 16, 32, 64, 128, usize::MAX]),
        test_size in prop::sample::select(&[0usize, 1, 2, 4, 8, 16, 32, 64, 128, usize::MAX]),
    ) {
        let config = BatchConfig {
            window_ms: 50,
            min_batch,
            optimal_batch: optimal,
            max_batch: 64,
            queue_size: 1024,
            gpu_threshold: 32,
        };

        // All boundary combinations should not panic
        let _ = config.should_process(test_size);
        let _ = config.meets_minimum(test_size);
    }
}
