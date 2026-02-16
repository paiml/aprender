//! Signing property tests.

#![allow(unused_imports)]

use super::super::*;
pub(crate) use proptest::prelude::*;

pub(crate) use super::*;
pub(crate) use proptest::prelude::*;

/// Strategy for generating test model data
pub(super) fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-100.0f32..100.0, 1..50)
}

proptest! {
    /// Property: Signing roundtrip preserves data and verifies
    #[test]
    fn prop_signing_roundtrip(data in arb_model_data()) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data.clone() };

        // Generate signing keypair
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let verifying_key = signing_key.verifying_key();

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("test.apr");

        save_signed(&model, ModelType::Custom, &path, SaveOptions::default(), &signing_key)
            .expect("save");
        let loaded: Model = load_verified(&path, ModelType::Custom, Some(&verifying_key))
            .expect("load");

        prop_assert_eq!(loaded.weights, data);
    }

    /// Property: Wrong verification key fails
    #[test]
    fn prop_signing_wrong_key_fails(data in arb_model_data()) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };

        // Generate two different keypairs
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let wrong_key = SigningKey::generate(&mut rand::thread_rng());
        let wrong_verifying = wrong_key.verifying_key();

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("test.apr");

        save_signed(&model, ModelType::Custom, &path, SaveOptions::default(), &signing_key)
            .expect("save");
        let result: Result<Model> = load_verified(&path, ModelType::Custom, Some(&wrong_verifying));

        prop_assert!(result.is_err(), "Wrong key should fail verification");
    }

    /// Property: Signed files have SIGNED flag set
    #[test]
    fn prop_signed_flag_set(_seed in any::<u8>()) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { v: i32 }

        let model = Model { v: 1 };
        let signing_key = SigningKey::generate(&mut rand::thread_rng());

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("test.apr");

        save_signed(&model, ModelType::Custom, &path, SaveOptions::default(), &signing_key)
            .expect("save");
        let info = inspect(&path).expect("inspect");

        prop_assert!(info.signed, "SIGNED flag must be set");
    }

    /// Property: Tampering with signed file is detected
    #[test]
    fn prop_tampering_detected(data in arb_model_data()) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };

        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let verifying_key = signing_key.verifying_key();

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("test.apr");

        save_signed(&model, ModelType::Custom, &path, SaveOptions::default(), &signing_key)
            .expect("save");

        // Tamper with the file (modify a byte in the middle)
        let mut content = std::fs::read(&path).expect("read");
        if content.len() > 50 {
            content[50] ^= 0xFF; // Flip bits
            std::fs::write(&path, content).expect("write");

            let result: Result<Model> = load_verified(&path, ModelType::Custom, Some(&verifying_key));
            prop_assert!(result.is_err(), "Tampered file should fail verification");
        }
    }
}
