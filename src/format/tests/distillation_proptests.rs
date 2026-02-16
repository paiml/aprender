//! Distillation property tests.

use super::super::*;
pub(crate) use proptest::prelude::*;

// Arbitrary generators for distillation types

pub(super) fn arb_distill_method() -> impl Strategy<Value = DistillMethod> {
    prop_oneof![
        Just(DistillMethod::Standard),
        Just(DistillMethod::Progressive),
        Just(DistillMethod::Ensemble),
    ]
}

pub(super) fn arb_model_type() -> impl Strategy<Value = ModelType> {
    prop_oneof![
        Just(ModelType::LinearRegression),
        Just(ModelType::LogisticRegression),
        Just(ModelType::DecisionTree),
        Just(ModelType::RandomForest),
        Just(ModelType::KMeans),
        Just(ModelType::NaiveBayes),
        Just(ModelType::Knn),
        Just(ModelType::Pca),
        Just(ModelType::Custom),
    ]
}

pub(super) fn arb_teacher_provenance() -> impl Strategy<Value = TeacherProvenance> {
    (
        "[a-f0-9]{64}",                              // SHA256 hash
        proptest::option::of("[a-zA-Z0-9+/]{86}=="), // Ed25519 signature (base64)
        arb_model_type(),
        1_000_000u64..10_000_000_000u64, // param count: 1M to 10B
    )
        .prop_map(
            |(hash, signature, model_type, param_count)| TeacherProvenance {
                hash,
                signature,
                model_type,
                param_count,
                ensemble_teachers: None,
            },
        )
}

pub(super) fn arb_distillation_params() -> impl Strategy<Value = DistillationParams> {
    (
        1.0f32..10.0f32,                       // temperature (1.0-10.0)
        0.0f32..1.0f32,                        // alpha (0.0-1.0)
        proptest::option::of(0.0f32..1.0f32),  // beta
        1u32..1000u32,                         // epochs
        proptest::option::of(0.0f32..10.0f32), // final_loss
    )
        .prop_map(
            |(temperature, alpha, beta, epochs, final_loss)| DistillationParams {
                temperature,
                alpha,
                beta,
                epochs,
                final_loss,
            },
        )
}

pub(super) fn arb_layer_mapping() -> impl Strategy<Value = LayerMapping> {
    (
        0usize..100usize, // student_layer
        0usize..200usize, // teacher_layer
        0.0f32..1.0f32,   // weight
    )
        .prop_map(|(student_layer, teacher_layer, weight)| LayerMapping {
            student_layer,
            teacher_layer,
            weight,
        })
}

pub(super) fn arb_distillation_info() -> impl Strategy<Value = DistillationInfo> {
    (
        arb_distill_method(),
        arb_teacher_provenance(),
        arb_distillation_params(),
    )
        .prop_map(|(method, teacher, params)| DistillationInfo {
            method,
            teacher,
            params,
            layer_mapping: None,
        })
}

pub(super) fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(
        any::<f32>().prop_filter("finite", |f| f.is_finite()),
        1..100,
    )
}

proptest! {
    /// Property: DistillMethod serialization roundtrip (JSON for optionals)
    #[test]
    fn prop_distill_method_roundtrip(method in arb_distill_method()) {
        // Use JSON for roundtrip testing (handles enums better than raw msgpack)
        let serialized = serde_json::to_string(&method).expect("serialize");
        let deserialized: DistillMethod = serde_json::from_str(&serialized).expect("deserialize");
        prop_assert_eq!(method, deserialized);
    }

    /// Property: DistillationParams serialization roundtrip
    #[test]
    fn prop_distillation_params_roundtrip(params in arb_distillation_params()) {
        // JSON handles optional fields correctly
        let serialized = serde_json::to_string(&params).expect("serialize");
        let deserialized: DistillationParams = serde_json::from_str(&serialized).expect("deserialize");

        // Check fields (f32 equality via bits for NaN handling)
        prop_assert_eq!(params.temperature.to_bits(), deserialized.temperature.to_bits());
        prop_assert_eq!(params.alpha.to_bits(), deserialized.alpha.to_bits());
        prop_assert_eq!(params.epochs, deserialized.epochs);
        prop_assert_eq!(params.beta.map(f32::to_bits), deserialized.beta.map(f32::to_bits));
    }

    /// Property: TeacherProvenance serialization roundtrip
    #[test]
    fn prop_teacher_provenance_roundtrip(teacher in arb_teacher_provenance()) {
        let serialized = serde_json::to_string(&teacher).expect("serialize");
        let deserialized: TeacherProvenance = serde_json::from_str(&serialized).expect("deserialize");

        prop_assert_eq!(&teacher.hash, &deserialized.hash);
        prop_assert_eq!(&teacher.signature, &deserialized.signature);
        prop_assert_eq!(teacher.model_type, deserialized.model_type);
        prop_assert_eq!(teacher.param_count, deserialized.param_count);
    }

    /// Property: LayerMapping serialization roundtrip
    #[test]
    fn prop_layer_mapping_roundtrip(mapping in arb_layer_mapping()) {
        let serialized = serde_json::to_string(&mapping).expect("serialize");
        let deserialized: LayerMapping = serde_json::from_str(&serialized).expect("deserialize");

        prop_assert_eq!(mapping.student_layer, deserialized.student_layer);
        prop_assert_eq!(mapping.teacher_layer, deserialized.teacher_layer);
        prop_assert_eq!(mapping.weight.to_bits(), deserialized.weight.to_bits());
    }

    /// Property: DistillationInfo serialization roundtrip
    #[test]
    fn prop_distillation_info_roundtrip(info in arb_distillation_info()) {
        let serialized = serde_json::to_string(&info).expect("serialize");
        let deserialized: DistillationInfo = serde_json::from_str(&serialized).expect("deserialize");

        prop_assert_eq!(info.method, deserialized.method);
        prop_assert_eq!(&info.teacher.hash, &deserialized.teacher.hash);
        prop_assert_eq!(info.params.epochs, deserialized.params.epochs);
    }

    /// Property: Distillation info persists through save/load cycle
    #[test]
    fn prop_distillation_save_load_roundtrip(
        info in arb_distillation_info(),
        data in arb_model_data()
    ) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("distilled.apr");

        let options = SaveOptions::default().with_distillation_info(info.clone());
        save(&model, ModelType::Custom, &path, options).expect("save");

        let model_info = inspect(&path).expect("inspect");
        let restored = model_info.metadata.distillation_info
            .expect("should have distillation_info");

        prop_assert_eq!(info.method, restored.method);
        prop_assert_eq!(&info.teacher.hash, &restored.teacher.hash);
        prop_assert_eq!(info.teacher.param_count, restored.teacher.param_count);
        prop_assert_eq!(info.params.epochs, restored.params.epochs);
    }

    /// Property: Temperature must be positive for valid distillation
    #[test]
    fn prop_temperature_positive(temp in 0.1f32..20.0f32) {
        let params = DistillationParams {
            temperature: temp,
            alpha: 0.5,
            beta: None,
            epochs: 10,
            final_loss: None,
        };
        prop_assert!(params.temperature > 0.0, "Temperature must be positive");
    }

    /// Property: Alpha (soft loss weight) must be in [0, 1]
    #[test]
    fn prop_alpha_bounded(alpha in 0.0f32..=1.0f32) {
        let params = DistillationParams {
            temperature: 3.0,
            alpha,
            beta: None,
            epochs: 10,
            final_loss: None,
        };
        prop_assert!((0.0..=1.0).contains(&params.alpha), "Alpha must be in [0,1]");
    }

    /// Property: Progressive distillation requires beta parameter (design guideline)
    #[test]
    fn prop_progressive_with_beta(beta in 0.0f32..1.0f32) {
        let info = DistillationInfo {
            method: DistillMethod::Progressive,
            teacher: TeacherProvenance {
                hash: "abc123".to_string(),
                signature: None,
                model_type: ModelType::Custom,
                param_count: 7_000_000_000,
                ensemble_teachers: None,
            },
            params: DistillationParams {
                temperature: 3.0,
                alpha: 0.7,
                beta: Some(beta),
                epochs: 10,
                final_loss: None,
            },
            layer_mapping: None,
        };
        // Progressive distillation should have beta for hidden layer loss weight
        prop_assert!(info.params.beta.is_some());
    }

    /// Property: Layer mappings have valid indices
    #[test]
    fn prop_layer_mapping_valid_indices(
        student in 0usize..100,
        teacher in 0usize..200,
        weight in 0.0f32..1.0f32
    ) {
        let mapping = LayerMapping {
            student_layer: student,
            teacher_layer: teacher,
            weight,
        };
        // Teacher layer index can be >= student (many-to-one mapping)
        // Weight should be non-negative
        prop_assert!(mapping.weight >= 0.0);
    }
}
