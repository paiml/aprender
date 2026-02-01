//! Metadata property tests.

use super::super::*;
use proptest::prelude::*;

fn arb_training_info() -> impl Strategy<Value = TrainingInfo> {
    (
        proptest::option::of(1usize..1_000_000),
        proptest::option::of(1u64..86_400_000), // up to 24h in ms
        proptest::option::of("[a-zA-Z0-9_/]{1,50}"),
    )
        .prop_map(|(samples, duration_ms, source)| TrainingInfo {
            samples,
            duration_ms,
            source,
        })
}

fn arb_model_name() -> impl Strategy<Value = String> {
    "[a-zA-Z][a-zA-Z0-9_-]{0,49}"
}

#[allow(clippy::disallowed_methods)] // json! macro uses unwrap internally
fn arb_hyperparams() -> impl Strategy<Value = HashMap<String, serde_json::Value>> {
    proptest::collection::hash_map(
        "[a-z_]{1,20}",
        prop_oneof![
            any::<f64>()
                .prop_filter("finite", |f| f.is_finite())
                .prop_map(|f| serde_json::json!(f)),
            any::<i32>().prop_map(|i| serde_json::json!(i)),
            "[a-z]{1,10}".prop_map(|s| serde_json::json!(s)),
        ],
        0..5,
    )
}

fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 1..50)
}

proptest! {
    /// Property: TrainingInfo serialization roundtrip
    #[test]
    fn prop_training_info_roundtrip(info in arb_training_info()) {
        let serialized = serde_json::to_string(&info).expect("serialize");
        let deserialized: TrainingInfo = serde_json::from_str(&serialized).expect("deserialize");

        prop_assert_eq!(info.samples, deserialized.samples);
        prop_assert_eq!(info.duration_ms, deserialized.duration_ms);
        prop_assert_eq!(&info.source, &deserialized.source);
    }

    /// Property: Metadata with model name persists through save/load
    #[test]
    fn prop_metadata_model_name_roundtrip(
        name in arb_model_name(),
        data in arb_model_data()
    ) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("named.apr");

        let options = SaveOptions::default().with_name(&name);
        save(&model, ModelType::Custom, &path, options).expect("save");

        let info = inspect(&path).expect("inspect");
        prop_assert_eq!(info.metadata.model_name.as_deref(), Some(name.as_str()));
    }

    /// Property: Metadata with training info persists
    #[test]
    fn prop_metadata_training_roundtrip(
        training in arb_training_info(),
        data in arb_model_data()
    ) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("trained.apr");

        let mut options = SaveOptions::default();
        options.metadata.training = Some(training.clone());
        save(&model, ModelType::Custom, &path, options).expect("save");

        let info = inspect(&path).expect("inspect");
        let restored = info.metadata.training.expect("should have training");
        prop_assert_eq!(training.samples, restored.samples);
        prop_assert_eq!(training.duration_ms, restored.duration_ms);
    }

    /// Property: Hyperparameters persist through save/load
    #[test]
    fn prop_metadata_hyperparams_roundtrip(
        hyperparams in arb_hyperparams(),
        data in arb_model_data()
    ) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("hyperparams.apr");

        let mut options = SaveOptions::default();
        options.metadata.hyperparameters = hyperparams.clone();
        save(&model, ModelType::Custom, &path, options).expect("save");

        let info = inspect(&path).expect("inspect");
        prop_assert_eq!(hyperparams.len(), info.metadata.hyperparameters.len());
        for (k, v) in &hyperparams {
            prop_assert_eq!(Some(v), info.metadata.hyperparameters.get(k));
        }
    }

    /// Property: Custom metadata persists
    #[test]
    fn prop_metadata_custom_roundtrip(
        custom in arb_hyperparams(),
        data in arb_model_data()
    ) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("custom.apr");

        let mut options = SaveOptions::default();
        options.metadata.custom = custom.clone();
        save(&model, ModelType::Custom, &path, options).expect("save");

        let info = inspect(&path).expect("inspect");
        prop_assert_eq!(custom.len(), info.metadata.custom.len());
    }

    /// Property: Aprender version is always set
    #[test]
    fn prop_metadata_version_always_set(data in arb_model_data()) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("versioned.apr");

        save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

        let info = inspect(&path).expect("inspect");
        prop_assert!(!info.metadata.aprender_version.is_empty());
        prop_assert!(info.metadata.aprender_version.contains('.'));
    }

    /// Property: Created timestamp is always set
    #[test]
    fn prop_metadata_timestamp_always_set(data in arb_model_data()) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("timestamped.apr");

        save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

        let info = inspect(&path).expect("inspect");
        prop_assert!(!info.metadata.created_at.is_empty());
    }
}
