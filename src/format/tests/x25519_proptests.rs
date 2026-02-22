//! X25519 property tests.

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
    /// Property: X25519 roundtrip preserves data
    #[test]
    fn prop_x25519_roundtrip(data in arb_model_data()) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data.clone() };

        // Generate recipient keypair
        let recipient_secret = X25519SecretKey::random_from_rng(rand::rng());
        let recipient_public = X25519PublicKey::from(&recipient_secret);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("test.apr");

        save_for_recipient(&model, ModelType::Custom, &path, SaveOptions::default(), &recipient_public)
            .expect("save");
        let loaded: Model = load_as_recipient(&path, ModelType::Custom, &recipient_secret)
            .expect("load");

        prop_assert_eq!(loaded.weights, data);
    }

    /// Property: X25519 wrong key fails
    #[test]
    fn prop_x25519_wrong_key_fails(data in arb_model_data()) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };

        // Generate two different keypairs
        let recipient_secret = X25519SecretKey::random_from_rng(rand::rng());
        let recipient_public = X25519PublicKey::from(&recipient_secret);
        let wrong_secret = X25519SecretKey::random_from_rng(rand::rng());

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("test.apr");

        save_for_recipient(&model, ModelType::Custom, &path, SaveOptions::default(), &recipient_public)
            .expect("save");
        let result: Result<Model> = load_as_recipient(&path, ModelType::Custom, &wrong_secret);

        prop_assert!(result.is_err(), "Wrong key should fail");
    }

    /// Property: X25519 encrypted files have ENCRYPTED flag
    #[test]
    fn prop_x25519_encrypted_flag(_seed in any::<u8>()) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { v: i32 }

        let model = Model { v: 1 };
        let recipient_secret = X25519SecretKey::random_from_rng(rand::rng());
        let recipient_public = X25519PublicKey::from(&recipient_secret);

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("test.apr");

        save_for_recipient(&model, ModelType::Custom, &path, SaveOptions::default(), &recipient_public)
            .expect("save");
        let info = inspect(&path).expect("inspect");

        prop_assert!(info.encrypted, "ENCRYPTED flag must be set");
    }
}
