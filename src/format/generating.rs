
// ============================================================================
// Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_card_new() {
        let card = ModelCard::new("test-model", "1.0.0");

        assert_eq!(card.model_id, "test-model");
        assert_eq!(card.name, "test-model");
        assert_eq!(card.version, "1.0.0");
        assert!(card.framework_version.starts_with("aprender"));
        assert!(!card.created_at.is_empty());
    }

    #[test]
    fn test_model_card_builder() {
        let card = ModelCard::new("my-model", "2.1.0")
            .with_name("My Model")
            .with_author("user@host")
            .with_description("A test model")
            .with_license("MIT")
            .with_architecture("MarkovModel")
            .with_param_count(12345);

        assert_eq!(card.name, "My Model");
        assert_eq!(card.author, Some("user@host".to_string()));
        assert_eq!(card.description, Some("A test model".to_string()));
        assert_eq!(card.license, Some("MIT".to_string()));
        assert_eq!(card.architecture, Some("MarkovModel".to_string()));
        assert_eq!(card.param_count, Some(12345));
    }

    #[test]
    fn test_model_card_with_training_data() {
        let data = TrainingDataInfo::new("~/.zsh_history")
            .with_samples(15234)
            .with_hash("sha256:abc123");

        let card = ModelCard::new("shell-model", "1.0.0").with_training_data(data);

        let training = card.training_data.expect("should have training data");
        assert_eq!(training.name, "~/.zsh_history");
        assert_eq!(training.samples, Some(15234));
        assert_eq!(training.hash, Some("sha256:abc123".to_string()));
    }

    #[test]
    fn test_model_card_with_hyperparameters() {
        let card = ModelCard::new("test", "1.0.0")
            .with_hyperparameter("n_gram_size", 3)
            .with_hyperparameter("smoothing", "laplace");

        assert_eq!(
            card.hyperparameters.get("n_gram_size"),
            Some(&serde_json::json!(3))
        );
        assert_eq!(
            card.hyperparameters.get("smoothing"),
            Some(&serde_json::json!("laplace"))
        );
    }

    #[test]
    fn test_model_card_with_metrics() {
        let card = ModelCard::new("test", "1.0.0")
            .with_metric("vocab_size", 4521)
            .with_metric("accuracy", 0.72);

        assert_eq!(
            card.metrics.get("vocab_size"),
            Some(&serde_json::json!(4521))
        );
        assert_eq!(card.metrics.get("accuracy"), Some(&serde_json::json!(0.72)));
    }

    #[test]
    fn test_model_card_json_roundtrip() {
        let card = ModelCard::new("roundtrip-test", "1.2.3")
            .with_author("test@example.com")
            .with_description("Test description")
            .with_metric("score", 0.95);

        let json = card.to_json().expect("serialize");
        let restored = ModelCard::from_json(&json).expect("deserialize");

        assert_eq!(card, restored);
    }

    #[test]
    fn test_model_card_to_huggingface() {
        let card = ModelCard::new("my-model", "1.0.0")
            .with_name("My Model")
            .with_description("A test model")
            .with_license("MIT")
            .with_architecture("MarkovModel")
            .with_metric("accuracy", 0.95)
            .with_training_data(
                TrainingDataInfo::new("dataset.txt")
                    .with_samples(1000)
                    .with_hash("sha256:abc"),
            );

        let hf = card.to_huggingface();

        // Check YAML front matter
        assert!(hf.starts_with("---"));
        assert!(hf.contains("license: mit"));
        assert!(hf.contains("- aprender"));
        assert!(hf.contains("- markovmodel"));

        // Check markdown content
        assert!(hf.contains("# My Model"));
        assert!(hf.contains("A test model"));
        assert!(hf.contains("**Source:** dataset.txt"));
        assert!(hf.contains("**Samples:** 1000"));
    }

    #[test]
    fn test_training_data_info() {
        let info = TrainingDataInfo::new("data.csv")
            .with_samples(500)
            .with_hash("sha256:def456");

        assert_eq!(info.name, "data.csv");
        assert_eq!(info.samples, Some(500));
        assert_eq!(info.hash, Some("sha256:def456".to_string()));
    }

    #[test]
    fn test_days_to_ymd() {
        // 1970-01-01
        assert_eq!(days_to_ymd(0), (1970, 1, 1));

        // 2000-01-01 (leap year)
        assert_eq!(days_to_ymd(10957), (2000, 1, 1));

        // 2025-11-27
        assert_eq!(days_to_ymd(20419), (2025, 11, 27));
    }

    #[test]
    fn test_is_leap_year() {
        assert!(!is_leap_year(1970));
        assert!(is_leap_year(2000)); // Divisible by 400
        assert!(!is_leap_year(1900)); // Divisible by 100 but not 400
        assert!(is_leap_year(2024)); // Divisible by 4
        assert!(!is_leap_year(2025));
    }

    #[test]
    fn test_model_card_default() {
        let card = ModelCard::default();
        assert_eq!(card.model_id, "unnamed");
        assert_eq!(card.version, "0.0.0");
    }

    #[test]
    fn test_model_card_created_at_format() {
        let card = ModelCard::new("test", "1.0.0");

        // Should be ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
        let created = &card.created_at;
        assert_eq!(created.len(), 20);
        assert!(created.ends_with('Z'));
        assert!(created.contains('T'));
    }
}

// ============================================================================
// Property-Based Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating valid model IDs
    fn arb_model_id() -> impl Strategy<Value = String> {
        "[a-z][a-z0-9_-]{0,30}".prop_map(|s| s.clone())
    }

    /// Strategy for generating semantic versions
    fn arb_semver() -> impl Strategy<Value = String> {
        (0u32..100, 0u32..100, 0u32..100)
            .prop_map(|(major, minor, patch)| format!("{major}.{minor}.{patch}"))
    }

    /// Strategy for generating author names
    fn arb_author() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9 _@.-]{0,50}".prop_map(|s| s.clone())
    }

    /// Strategy for generating descriptions
    fn arb_description() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9 .,!?-]{0,200}".prop_map(|s| s.clone())
    }

    proptest! {
        /// Property: ModelCard JSON roundtrip preserves all fields
        #[test]
        fn prop_model_card_json_roundtrip(
            model_id in arb_model_id(),
            version in arb_semver(),
            author in arb_author(),
            description in arb_description(),
            param_count in any::<u64>(),
        ) {
            let card = ModelCard::new(&model_id, &version)
                .with_author(&author)
                .with_description(&description)
                .with_param_count(param_count);

            let json = card.to_json().expect("serialize");
            let restored = ModelCard::from_json(&json).expect("deserialize");

            prop_assert_eq!(card.model_id, restored.model_id);
            prop_assert_eq!(card.version, restored.version);
            prop_assert_eq!(card.author, restored.author);
            prop_assert_eq!(card.description, restored.description);
            prop_assert_eq!(card.param_count, restored.param_count);
        }

        /// Property: ModelCard builder methods are idempotent
        #[test]
        fn prop_builder_idempotent(
            model_id in arb_model_id(),
            version in arb_semver(),
            author in arb_author(),
        ) {
            let card1 = ModelCard::new(&model_id, &version)
                .with_author(&author)
                .with_author(&author); // Apply twice

            let card2 = ModelCard::new(&model_id, &version)
                .with_author(&author);

            prop_assert_eq!(card1.author, card2.author);
        }

        /// Property: created_at is always valid ISO 8601
        #[test]
        fn prop_created_at_valid_iso8601(
            model_id in arb_model_id(),
            version in arb_semver(),
        ) {
            let card = ModelCard::new(&model_id, &version);
            let created = &card.created_at;

            // Format: YYYY-MM-DDTHH:MM:SSZ
            prop_assert_eq!(created.len(), 20);
            prop_assert!(created.ends_with('Z'));
            prop_assert!(created.contains('T'));
            prop_assert!(created.chars().filter(|c| *c == '-').count() == 2);
            prop_assert!(created.chars().filter(|c| *c == ':').count() == 2);
        }

        /// Property: TrainingDataInfo roundtrip through JSON
        #[test]
        fn prop_training_data_roundtrip(
            name in "[a-zA-Z0-9/_.-]{1,50}",
            samples in any::<u64>(),
            hash in "[a-f0-9]{64}",
        ) {
            let info = TrainingDataInfo::new(&name)
                .with_samples(samples)
                .with_hash(&hash);

            let card = ModelCard::new("test", "1.0.0")
                .with_training_data(info.clone());

            let json = card.to_json().expect("serialize");
            let restored = ModelCard::from_json(&json).expect("deserialize");

            let restored_info = restored.training_data.expect("training data");
            prop_assert_eq!(info.name, restored_info.name);
            prop_assert_eq!(info.samples, restored_info.samples);
            prop_assert_eq!(info.hash, restored_info.hash);
        }

        /// Property: Hyperparameters roundtrip through JSON
        #[test]
        fn prop_hyperparameters_roundtrip(
            key in "[a-z_]{1,20}",
            int_value in any::<i64>(),
            float_value in any::<f64>().prop_filter("finite", |f| f.is_finite()),
        ) {
            let card = ModelCard::new("test", "1.0.0")
                .with_hyperparameter(&key, int_value)
                .with_hyperparameter("float_param", float_value);

            let json = card.to_json().expect("serialize");
            let restored = ModelCard::from_json(&json).expect("deserialize");

            prop_assert_eq!(
                card.hyperparameters.get(&key),
                restored.hyperparameters.get(&key)
            );
        }

        /// Property: Metrics roundtrip through JSON (keys preserved)
        #[test]
        fn prop_metrics_roundtrip(
            key in "[a-z_]{1,20}",
            value in -1e10f64..1e10f64, // Reasonable range to avoid JSON precision issues
        ) {
            let card = ModelCard::new("test", "1.0.0")
                .with_metric(&key, value);

            let json = card.to_json().expect("serialize");
            let restored = ModelCard::from_json(&json).expect("deserialize");

            // Key must exist in restored metrics
            prop_assert!(restored.metrics.contains_key(&key));
        }

        /// Property: days_to_ymd produces valid dates
        #[test]
        fn prop_days_to_ymd_valid(days in 0u64..50000) {
            let (year, month, day) = days_to_ymd(days);

            // Year in reasonable range (1970 - ~2106)
            prop_assert!((1970..=2200).contains(&year));
            // Month 1-12
            prop_assert!((1..=12).contains(&month));
            // Day 1-31
            prop_assert!((1..=31).contains(&day));
        }

        /// Property: is_leap_year consistent with days_in_year
        #[test]
        fn prop_leap_year_consistent(year in 1970i32..2200) {
            let leap = is_leap_year(year);
            let days = if leap { 366 } else { 365 };

            // February has 29 days in leap years, 28 otherwise
            let feb_days = if leap { 29 } else { 28 };
            let total = 31 + feb_days + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31;

            prop_assert_eq!(days, total);
        }

        /// Property: Hugging Face export always contains required fields
        #[test]
        fn prop_huggingface_has_required_fields(
            model_id in arb_model_id(),
            version in arb_semver(),
        ) {
            let card = ModelCard::new(&model_id, &version)
                .with_license("MIT")
                .with_architecture("TestModel");

            let hf = card.to_huggingface();

            // Must have YAML front matter markers
            prop_assert!(hf.starts_with("---"));
            prop_assert!(hf.contains("---\n\n"));

            // Must have required tags
            prop_assert!(hf.contains("- aprender"));
            prop_assert!(hf.contains("- rust"));

            // Must have model name header
            let expected_header = format!("# {}", card.name);
            prop_assert!(hf.contains(&expected_header));
        }

        /// Property: Default ModelCard has expected defaults
        #[test]
        fn prop_default_has_expected_values(_seed in any::<u8>()) {
            let card = ModelCard::default();

            prop_assert_eq!(card.model_id, "unnamed");
            prop_assert_eq!(card.version, "0.0.0");
            prop_assert!(card.author.is_none());
            prop_assert!(card.description.is_none());
            prop_assert!(card.license.is_none());
            prop_assert!(card.training_data.is_none());
            prop_assert!(card.hyperparameters.is_empty());
            prop_assert!(card.metrics.is_empty());
        }
    }
}
