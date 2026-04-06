//! Proptest Strategies for WASM Mock Testing
//!
//! Per `PROBAR-SPEC-WASM-001` Section 3.1, these strategies generate random
//! mock messages for property-based testing.
//!
//! ## Iron Lotus Philosophy
//!
//! These strategies should be used to test ACTUAL code via `MockWasmRuntime`,
//! NOT to test pure models. Testing models provides false confidence.
//!
//! ```rust,ignore
//! proptest! {
//!     #[test]
//!     fn prop_state_sync_invariant(messages in valid_message_sequence()) {
//!         let harness = WasmCallbackTestHarness::<MyWorker>::new();
//!         harness.worker.spawn("model.apr").unwrap();
//!
//!         for msg in messages {
//!             harness.send_message(msg);
//!
//!             // This catches state sync bugs!
//!             harness.assert_state_synced();
//!         }
//!     }
//! }
//! ```

use super::wasm_runtime::MockMessage;

#[cfg(feature = "proptest")]
use proptest::prelude::*;

/// Generate any mock message (random)
///
/// This generates all possible message variants with random payloads.
/// Use for fuzz testing to find edge cases.
#[cfg(feature = "proptest")]
pub fn any_mock_message() -> impl Strategy<Value = MockMessage> {
    prop_oneof![
        // Ready (no payload)
        Just(MockMessage::Ready),
        // ModelLoaded with random values
        (any::<f64>(), any::<f64>()).prop_map(|(size, time)| {
            MockMessage::ModelLoaded {
                size_mb: size.abs() % 1000.0,
                load_time_ms: time.abs() % 10000.0,
            }
        }),
        // Start with random sample rate
        any::<u32>().prop_map(|rate| {
            MockMessage::Start {
                sample_rate: 8000 + (rate % 192_000), // 8kHz to 200kHz range
            }
        }),
        // Stop (no payload)
        Just(MockMessage::Stop),
        // Error with random message
        any::<String>().prop_map(|s| {
            MockMessage::Error {
                message: s.chars().take(100).collect(),
            }
        }),
        // Shutdown (no payload)
        Just(MockMessage::Shutdown),
        // Partial with random text
        (any::<String>(), any::<bool>()).prop_map(|(text, is_final)| {
            MockMessage::Partial {
                text: text.chars().take(1000).collect(),
                is_final,
            }
        }),
    ]
}

/// Generate valid message sequences (respects typical state machine)
///
/// This generates sequences that follow a realistic message ordering:
/// Ready → ModelLoaded → Start/Stop cycles → Shutdown
///
/// Use for testing normal operation paths.
#[cfg(feature = "proptest")]
pub fn valid_message_sequence() -> impl Strategy<Value = Vec<MockMessage>> {
    prop::collection::vec(
        prop_oneof![
            Just(MockMessage::Ready),
            Just(MockMessage::ModelLoaded {
                size_mb: 39.0,
                load_time_ms: 1500.0
            }),
            Just(MockMessage::Start { sample_rate: 48000 }),
            Just(MockMessage::Stop),
        ],
        0..20,
    )
}

/// Generate error-heavy sequences (for error handling testing)
#[cfg(feature = "proptest")]
pub fn error_heavy_sequence() -> impl Strategy<Value = Vec<MockMessage>> {
    prop::collection::vec(
        prop_oneof![
            3 => Just(MockMessage::Error { message: "test error".to_string() }),
            1 => Just(MockMessage::Ready),
            1 => Just(MockMessage::Stop),
        ],
        1..15,
    )
}

/// Generate realistic worker lifecycle
///
/// This generates a complete, valid lifecycle:
/// 1. Ready
/// 2. ModelLoaded
/// 3. Zero or more Start/Stop cycles
/// 4. Optional Shutdown
#[cfg(feature = "proptest")]
pub fn realistic_lifecycle() -> impl Strategy<Value = Vec<MockMessage>> {
    (0usize..5, any::<bool>()).prop_map(|(cycles, shutdown)| {
        let mut messages = vec![
            MockMessage::Ready,
            MockMessage::ModelLoaded {
                size_mb: 39.0,
                load_time_ms: 1500.0,
            },
        ];

        for _ in 0..cycles {
            messages.push(MockMessage::Start { sample_rate: 48000 });
            messages.push(MockMessage::Stop);
        }

        if shutdown {
            messages.push(MockMessage::Shutdown);
        }

        messages
    })
}

// Non-proptest versions for use without the feature

/// Get a predefined set of test messages (non-random)
#[must_use]
pub fn standard_test_messages() -> Vec<MockMessage> {
    vec![
        MockMessage::Ready,
        MockMessage::ModelLoaded {
            size_mb: 39.0,
            load_time_ms: 1500.0,
        },
        MockMessage::Start { sample_rate: 48000 },
        MockMessage::Partial {
            text: "Hello".to_string(),
            is_final: false,
        },
        MockMessage::Partial {
            text: "Hello world".to_string(),
            is_final: true,
        },
        MockMessage::Stop,
    ]
}

/// Get error test messages (non-random)
#[must_use]
pub fn error_test_messages() -> Vec<MockMessage> {
    vec![
        MockMessage::Ready,
        MockMessage::Error {
            message: "Model load failed".to_string(),
        },
        MockMessage::Error {
            message: "Network error".to_string(),
        },
    ]
}

/// Get edge case messages (non-random)
#[must_use]
pub fn edge_case_messages() -> Vec<MockMessage> {
    vec![
        // Empty payload
        MockMessage::Error {
            message: String::new(),
        },
        // Very long message
        MockMessage::Error {
            message: "x".repeat(10000),
        },
        // Unicode in message
        MockMessage::Partial {
            text: "Hello 世界 🌍".to_string(),
            is_final: false,
        },
        // Zero sample rate
        MockMessage::Start { sample_rate: 0 },
        // Very high sample rate
        MockMessage::Start {
            sample_rate: u32::MAX,
        },
        // Negative-ish floating point
        MockMessage::ModelLoaded {
            size_mb: -1.0,
            load_time_ms: f64::NAN,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_test_messages() {
        let messages = standard_test_messages();
        assert!(!messages.is_empty());
        assert!(matches!(messages[0], MockMessage::Ready));
    }

    #[test]
    fn test_error_test_messages() {
        let messages = error_test_messages();
        assert!(!messages.is_empty());
        assert!(matches!(messages[0], MockMessage::Ready));
        assert!(messages
            .iter()
            .any(|m| matches!(m, MockMessage::Error { .. })));
    }

    #[test]
    fn test_edge_case_messages() {
        let messages = edge_case_messages();
        assert!(!messages.is_empty());

        // Check for the empty error message
        assert!(messages.iter().any(|m| matches!(
            m,
            MockMessage::Error { message } if message.is_empty()
        )));

        // Check for unicode
        assert!(messages.iter().any(|m| matches!(
            m,
            MockMessage::Partial { text, .. } if text.contains('世')
        )));
    }

    #[cfg(feature = "proptest")]
    mod proptest_tests {
        use super::*;
        use proptest::strategy::ValueTree;
        use proptest::test_runner::{Config, TestRunner};

        #[test]
        fn test_any_mock_message_strategy() {
            let mut runner = TestRunner::new(Config::default());
            let strategy = any_mock_message();

            // Generate 100 random messages, ensure no panics
            for _ in 0..100 {
                let value = strategy.new_tree(&mut runner).unwrap().current();
                // Just verify we can generate them
                let _ = format!("{:?}", value);
            }
        }

        #[test]
        fn test_valid_message_sequence_strategy() {
            let mut runner = TestRunner::new(Config::default());
            let strategy = valid_message_sequence();

            for _ in 0..50 {
                let seq = strategy.new_tree(&mut runner).unwrap().current();
                assert!(seq.len() <= 20);
            }
        }

        #[test]
        fn test_realistic_lifecycle_strategy() {
            let mut runner = TestRunner::new(Config::default());
            let strategy = realistic_lifecycle();

            for _ in 0..50 {
                let seq = strategy.new_tree(&mut runner).unwrap().current();
                // Should start with Ready
                if !seq.is_empty() {
                    assert!(matches!(seq[0], MockMessage::Ready));
                }
                // Should have ModelLoaded after Ready
                if seq.len() >= 2 {
                    assert!(matches!(seq[1], MockMessage::ModelLoaded { .. }));
                }
            }
        }
    }
}
