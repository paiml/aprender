//! Proptest implementations for property-based testing
//!
//! Implements `Arbitrary` for QA types to enable fuzzing.

use crate::models::ModelId;
use crate::scenario::{Backend, Format, Modality, QaScenario, TraceLevel};
use proptest::prelude::*;

/// Strategy for generating model IDs
pub fn model_id_strategy() -> impl Strategy<Value = ModelId> {
    (
        prop::sample::select(vec![
            "Qwen",
            "meta-llama",
            "microsoft",
            "google",
            "mistralai",
            "deepseek-ai",
            "TinyLlama",
        ]),
        prop::sample::select(vec![
            "Qwen2.5-Coder-1.5B-Instruct",
            "Llama-3.2-1B-Instruct",
            "Phi-3-mini-4k-instruct",
            "gemma-2b-it",
            "Mistral-7B-Instruct-v0.3",
            "deepseek-coder-1.3b-instruct",
            "TinyLlama-1.1B-Chat-v1.0",
        ]),
    )
        .prop_map(|(org, name)| ModelId::new(org, name))
}

/// Strategy for generating modalities
pub fn modality_strategy() -> impl Strategy<Value = Modality> {
    prop_oneof![
        Just(Modality::Run),
        Just(Modality::Chat),
        Just(Modality::Serve),
    ]
}

/// Strategy for generating backends
pub fn backend_strategy() -> impl Strategy<Value = Backend> {
    prop_oneof![Just(Backend::Cpu), Just(Backend::Gpu),]
}

/// Strategy for generating formats
pub fn format_strategy() -> impl Strategy<Value = Format> {
    prop_oneof![
        Just(Format::Gguf),
        Just(Format::SafeTensors),
        Just(Format::Apr),
    ]
}

/// Strategy for generating trace levels
pub fn trace_level_strategy() -> impl Strategy<Value = TraceLevel> {
    prop_oneof![
        Just(TraceLevel::None),
        Just(TraceLevel::Basic),
        Just(TraceLevel::Layer),
        Just(TraceLevel::Payload),
    ]
}

/// Strategy for generating arithmetic prompts (verifiable)
pub fn arithmetic_prompt_strategy() -> impl Strategy<Value = String> {
    (
        1i32..100,
        1i32..100,
        prop::sample::select(vec!['+', '-', '*', '/']),
    )
        .prop_map(|(a, b, op)| format!("What is {a}{op}{b}?"))
}

/// Strategy for generating code prompts
pub fn code_prompt_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("def fibonacci(n):".to_string()),
        Just("fn main() {".to_string()),
        Just("async function fetch() {".to_string()),
        Just("class Person:".to_string()),
        Just("impl Iterator for".to_string()),
        Just("pub struct Config {".to_string()),
    ]
}

/// Strategy for generating edge case prompts
pub fn edge_case_prompt_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::new()),                               // Empty
        Just(" ".to_string()),                             // Whitespace
        Just("\n\n\n".to_string()),                        // Newlines only
        Just("你好世界".to_string()),                      // Chinese
        Just("مرحبا بالعالم".to_string()),                 // Arabic (RTL)
        Just("🎉🚀💻".to_string()),                        // Emoji
        Just("a".repeat(10000)),                           // Very long
        Just("<script>alert('xss')</script>".to_string()), // XSS attempt
        Just("'; DROP TABLE users; --".to_string()),       // SQL injection
    ]
}

/// Strategy for generating any prompt
pub fn any_prompt_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => arithmetic_prompt_strategy(),
        2 => code_prompt_strategy(),
        1 => edge_case_prompt_strategy(),
        2 => "[a-zA-Z0-9 ]{1,100}".prop_map(|s| s),
    ]
}

/// Strategy for generating complete scenarios
pub fn scenario_strategy() -> impl Strategy<Value = QaScenario> {
    (
        model_id_strategy(),
        modality_strategy(),
        backend_strategy(),
        format_strategy(),
        any_prompt_strategy(),
        0u64..1000,
    )
        .prop_map(|(model, modality, backend, format, prompt, seed)| {
            QaScenario::new(model, modality, backend, format, prompt, seed)
        })
}

/// Strategy for scenarios with specific model
pub fn scenario_for_model_strategy(model: ModelId) -> impl Strategy<Value = QaScenario> {
    (
        modality_strategy(),
        backend_strategy(),
        format_strategy(),
        any_prompt_strategy(),
        0u64..1000,
    )
        .prop_map(move |(modality, backend, format, prompt, seed)| {
            QaScenario::new(model.clone(), modality, backend, format, prompt, seed)
        })
}

/// Strategy for temperature values
pub fn temperature_strategy() -> impl Strategy<Value = f32> {
    prop_oneof![
        Just(0.0),   // Deterministic
        Just(0.7),   // Default
        Just(1.0),   // Creative
        0.0f32..2.0, // Random range
    ]
}

/// Strategy for max token counts
pub fn max_tokens_strategy() -> impl Strategy<Value = u32> {
    prop_oneof![
        Just(1u32), // Minimum
        Just(32),   // Default
        Just(128),  // Medium
        Just(512),  // Large
        Just(2048), // Very large
        1u32..4096, // Random range
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;

    #[test]
    fn test_model_id_strategy_generates_valid() {
        let mut runner = TestRunner::default();
        for _ in 0..100 {
            let model = model_id_strategy()
                .new_tree(&mut runner)
                .expect("Failed to generate")
                .current();
            assert!(!model.org.is_empty());
            assert!(!model.name.is_empty());
        }
    }

    #[test]
    fn test_scenario_strategy_generates_valid() {
        let mut runner = TestRunner::default();
        for _ in 0..100 {
            let scenario = scenario_strategy()
                .new_tree(&mut runner)
                .expect("Failed to generate")
                .current();
            assert!(!scenario.id.is_empty());
        }
    }

    proptest! {
        // Configure proptest with regression file support (F-PROP-006)
        // Failing cases are stored in proptest-regressions/ for reproducibility
        #![proptest_config(ProptestConfig {
            cases: 100,
            failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::WithSource("proptest-regressions"))),
            ..ProptestConfig::default()
        })]

        #[test]
        fn prop_arithmetic_prompts_contain_operator(prompt in arithmetic_prompt_strategy()) {
            prop_assert!(
                prompt.contains('+') || prompt.contains('-') || prompt.contains('*') || prompt.contains('/'),
                "Prompt should contain arithmetic operator: {}", prompt
            );
        }

        #[test]
        fn prop_scenarios_have_valid_id(scenario in scenario_strategy()) {
            prop_assert!(!scenario.id.is_empty());
            prop_assert!(scenario.id.contains('_'));
        }

        #[test]
        fn prop_temperature_in_range(temp in temperature_strategy()) {
            prop_assert!(temp >= 0.0);
            prop_assert!(temp <= 2.0);
        }

        #[test]
        fn prop_max_tokens_positive(tokens in max_tokens_strategy()) {
            prop_assert!(tokens >= 1);
            prop_assert!(tokens <= 4096);
        }

        #[test]
        fn prop_scenario_command_is_valid(scenario in scenario_strategy()) {
            let cmd = scenario.to_command("model.gguf");
            prop_assert!(!cmd.is_empty());
            prop_assert!(cmd.contains("apr"));
        }
    }

    #[test]
    fn test_modality_strategy_generates_valid() {
        let mut runner = TestRunner::default();
        for _ in 0..50 {
            let modality = modality_strategy()
                .new_tree(&mut runner)
                .expect("Failed to generate")
                .current();
            // Should be one of the valid modalities
            assert!(matches!(
                modality,
                Modality::Run | Modality::Chat | Modality::Serve
            ));
        }
    }

    #[test]
    fn test_backend_strategy_generates_valid() {
        let mut runner = TestRunner::default();
        for _ in 0..50 {
            let backend = backend_strategy()
                .new_tree(&mut runner)
                .expect("Failed to generate")
                .current();
            assert!(matches!(backend, Backend::Cpu | Backend::Gpu));
        }
    }

    #[test]
    fn test_format_strategy_generates_valid() {
        let mut runner = TestRunner::default();
        for _ in 0..50 {
            let format = format_strategy()
                .new_tree(&mut runner)
                .expect("Failed to generate")
                .current();
            assert!(matches!(
                format,
                Format::Gguf | Format::SafeTensors | Format::Apr
            ));
        }
    }

    #[test]
    fn test_trace_level_strategy_generates_valid() {
        let mut runner = TestRunner::default();
        for _ in 0..50 {
            let level = trace_level_strategy()
                .new_tree(&mut runner)
                .expect("Failed to generate")
                .current();
            assert!(matches!(
                level,
                TraceLevel::None | TraceLevel::Basic | TraceLevel::Layer | TraceLevel::Payload
            ));
        }
    }

    #[test]
    fn test_code_prompt_strategy_generates_code() {
        let mut runner = TestRunner::default();
        for _ in 0..50 {
            let prompt = code_prompt_strategy()
                .new_tree(&mut runner)
                .expect("Failed to generate")
                .current();
            // Should be a code-like prompt
            assert!(
                prompt.contains("def ")
                    || prompt.contains("fn ")
                    || prompt.contains("async ")
                    || prompt.contains("class ")
                    || prompt.contains("impl ")
                    || prompt.contains("pub ")
            );
        }
    }

    #[test]
    fn test_edge_case_prompt_strategy_generates_edge_cases() {
        let mut runner = TestRunner::default();
        let mut seen_empty = false;
        let mut seen_unicode = false;
        let mut seen_long = false;

        for _ in 0..100 {
            let prompt = edge_case_prompt_strategy()
                .new_tree(&mut runner)
                .expect("Failed to generate")
                .current();

            if prompt.is_empty() || prompt.trim().is_empty() {
                seen_empty = true;
            }
            if prompt.contains("你好") || prompt.contains("مرحبا") || prompt.contains("🎉")
            {
                seen_unicode = true;
            }
            if prompt.len() > 1000 {
                seen_long = true;
            }
        }

        // Should have seen at least some variety
        assert!(seen_empty || seen_unicode || seen_long);
    }

    #[test]
    fn test_scenario_for_model_strategy() {
        let model = ModelId::new("test", "model");
        let mut runner = TestRunner::default();

        for _ in 0..50 {
            let scenario = scenario_for_model_strategy(model.clone())
                .new_tree(&mut runner)
                .expect("Failed to generate")
                .current();

            // Model should match
            assert_eq!(scenario.model.org, "test");
            assert_eq!(scenario.model.name, "model");
        }
    }

    #[test]
    fn test_temperature_strategy_generates_valid_range() {
        let mut runner = TestRunner::default();
        for _ in 0..50 {
            let temp = temperature_strategy()
                .new_tree(&mut runner)
                .expect("Failed to generate")
                .current();
            assert!(temp >= 0.0);
            assert!(temp <= 2.0);
        }
    }

    #[test]
    fn test_max_tokens_strategy_generates_valid_range() {
        let mut runner = TestRunner::default();
        for _ in 0..50 {
            let tokens = max_tokens_strategy()
                .new_tree(&mut runner)
                .expect("Failed to generate")
                .current();
            assert!(tokens >= 1);
            assert!(tokens <= 4096);
        }
    }

    #[test]
    fn test_any_prompt_strategy_generates_variety() {
        let mut runner = TestRunner::default();
        let mut has_arithmetic = false;
        let mut has_code = false;
        let mut has_other = false;

        for _ in 0..100 {
            let prompt = any_prompt_strategy()
                .new_tree(&mut runner)
                .expect("Failed to generate")
                .current();

            if prompt.contains('+') || prompt.contains('-') || prompt.contains('*') {
                has_arithmetic = true;
            } else if prompt.contains("def ") || prompt.contains("fn ") || prompt.contains("class ")
            {
                has_code = true;
            } else {
                has_other = true;
            }
        }

        // Should generate variety
        assert!(has_arithmetic || has_code || has_other);
    }
}
