//! Integration property tests.

use super::super::*;
use proptest::prelude::*;

fn arb_model_type() -> impl Strategy<Value = ModelType> {
        prop_oneof![
            Just(ModelType::LinearRegression),
            Just(ModelType::LogisticRegression),
            Just(ModelType::DecisionTree),
            Just(ModelType::RandomForest),
            Just(ModelType::KMeans),
            Just(ModelType::NaiveBayes),
            Just(ModelType::Custom),
        ]
    }

    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(
            any::<f32>().prop_filter("finite", |f| f.is_finite()),
            10..500,
        )
    }

    fn arb_large_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(
            any::<f32>().prop_filter("finite", |f| f.is_finite()),
            1000..5000,
        )
    }

    proptest! {
        /// Property: Full metadata stack persists correctly
        #[test]
        fn prop_full_metadata_roundtrip(
            model_name in "[a-zA-Z][a-zA-Z0-9_-]{1,20}",
            description in "[a-zA-Z0-9 ]{1,50}",
            samples in 1usize..100000,
            duration_ms in 1u64..86400000,
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("full_meta.apr");

            let mut options = SaveOptions::default()
                .with_name(&model_name)
                .with_description(&description);
            options.metadata.training = Some(TrainingInfo {
                samples: Some(samples),
                duration_ms: Some(duration_ms),
                source: Some("test_data".to_string()),
            });

            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert_eq!(info.metadata.model_name.as_deref(), Some(model_name.as_str()));
            prop_assert_eq!(info.metadata.description.as_deref(), Some(description.as_str()));

            let training = info.metadata.training.expect("training");
            prop_assert_eq!(training.samples, Some(samples));
            prop_assert_eq!(training.duration_ms, Some(duration_ms));

            let loaded: Model = load(&path, ModelType::Custom).expect("load");
            prop_assert_eq!(data.len(), loaded.weights.len());
        }

        /// Property: All model types roundtrip correctly
        #[test]
        fn prop_all_model_types_roundtrip(
            model_type in arb_model_type(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("typed.apr");

            save(&model, model_type, &path, SaveOptions::default()).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert_eq!(info.model_type, model_type);

            let loaded: Model = load(&path, model_type).expect("load");
            prop_assert_eq!(data.len(), loaded.weights.len());
        }

        /// Property: Large models roundtrip correctly (stress test)
        #[test]
        fn prop_large_model_roundtrip(data in arb_large_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("large.apr");

            save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

            let loaded: Model = load(&path, ModelType::Custom).expect("load");
            prop_assert_eq!(data.len(), loaded.weights.len());

            // Verify data integrity
            for (i, (orig, loaded_val)) in data.iter().zip(loaded.weights.iter()).enumerate() {
                prop_assert_eq!(
                    orig.to_bits(),
                    loaded_val.to_bits(),
                    "Mismatch at index {}", i
                );
            }
        }

        /// Property: Distillation + License combined
        #[test]
        fn prop_distillation_with_license(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("distilled_licensed.apr");

            let distill_info = DistillationInfo {
                method: DistillMethod::Standard,
                teacher: TeacherProvenance {
                    hash: "a".repeat(64),
                    signature: None,
                    model_type: ModelType::Custom,
                    param_count: 7_000_000_000,
                    ensemble_teachers: None,
                },
                params: DistillationParams {
                    temperature: 3.0,
                    alpha: 0.7,
                    beta: None,
                    epochs: 10,
                    final_loss: Some(0.42),
                },
                layer_mapping: None,
            };

            let license = LicenseInfo {
                uuid: "12345678-1234-4123-8123-123456789abc".to_string(),
                hash: "b".repeat(64),
                expiry: Some("2025-12-31".to_string()),
                seats: Some(10),
                licensee: Some("Test Corp".to_string()),
                tier: LicenseTier::Enterprise,
            };

            let options = SaveOptions::default()
                .with_distillation_info(distill_info)
                .with_license(license);

            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");

            // Both features should be present
            prop_assert!(info.metadata.distillation_info.is_some());
            prop_assert!(info.metadata.license.is_some());
            prop_assert!(info.licensed);

            let restored_distill = info.metadata.distillation_info.expect("distillation");
            prop_assert!(matches!(restored_distill.method, DistillMethod::Standard));

            let restored_license = info.metadata.license.expect("license");
            prop_assert!(matches!(restored_license.tier, LicenseTier::Enterprise));
        }

        /// Property: Multiple saves to same path overwrites correctly
        #[test]
        fn prop_overwrite_preserves_latest(
            data1 in arb_model_data(),
            data2 in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("overwrite.apr");

            // Save first model
            let model1 = Model { weights: data1 };
            save(&model1, ModelType::Custom, &path, SaveOptions::default()).expect("save1");

            // Save second model (overwrite)
            let model2 = Model { weights: data2.clone() };
            save(&model2, ModelType::Custom, &path, SaveOptions::default()).expect("save2");

            // Load should return second model
            let loaded: Model = load(&path, ModelType::Custom).expect("load");
            prop_assert_eq!(data2.len(), loaded.weights.len());
        }

        /// Property: File size scales with data size
        #[test]
        fn prop_file_size_scales_with_data(
            small_data in proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 10..50),
            large_data in proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 500..1000)
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let dir = tempdir().expect("tempdir");

            let small_path = dir.path().join("small.apr");
            let model_small = Model { weights: small_data };
            save(&model_small, ModelType::Custom, &small_path, SaveOptions::default()).expect("save small");

            let large_path = dir.path().join("large.apr");
            let model_large = Model { weights: large_data };
            save(&model_large, ModelType::Custom, &large_path, SaveOptions::default()).expect("save large");

            let small_size = std::fs::metadata(&small_path).expect("meta").len();
            let large_size = std::fs::metadata(&large_path).expect("meta").len();

            prop_assert!(large_size > small_size, "Larger data should produce larger file");
        }
    }
