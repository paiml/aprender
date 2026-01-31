//! Encryption property tests.

#![allow(unused_imports)]

use super::super::*;
use proptest::prelude::*;

    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating valid passwords (8-64 chars)
    fn arb_password() -> impl Strategy<Value = String> {
        proptest::collection::vec(any::<u8>(), 8..64)
            .prop_map(|bytes| bytes.iter().map(|b| (b % 94 + 33) as char).collect())
    }

    /// Strategy for generating test model data
    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-100.0f32..100.0, 1..100)
    }

    // 3 cases for encryption tests (Argon2id has high computational cost by design)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(3))]

        /// Property: Encryption roundtrip preserves data (in-memory)
        #[test]
        fn prop_encryption_roundtrip_preserves_data(
            password in arb_password(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_encrypted(&model, ModelType::Custom, &path, SaveOptions::default(), &password)
                .expect("save");
            let loaded: Model = load_encrypted(&path, ModelType::Custom, &password)
                .expect("load");

            prop_assert_eq!(loaded.weights, data);
        }

        /// Property: Wrong password fails decryption
        #[test]
        fn prop_wrong_password_fails(
            password in arb_password(),
            wrong_password in arb_password()
        ) {
            // Skip if passwords happen to be the same
            if password == wrong_password {
                return Ok(());
            }

            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { value: i32 }

            let model = Model { value: 42 };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_encrypted(&model, ModelType::Custom, &path, SaveOptions::default(), &password)
                .expect("save");
            let result: Result<Model> = load_encrypted(&path, ModelType::Custom, &wrong_password);

            prop_assert!(result.is_err(), "Wrong password should fail");
        }

        /// Property: Encrypted files have ENCRYPTED flag set
        #[test]
        fn prop_encrypted_flag_set(password in arb_password()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { v: i32 }

            let model = Model { v: 1 };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_encrypted(&model, ModelType::Custom, &path, SaveOptions::default(), &password)
                .expect("save");
            let info = inspect(&path).expect("inspect");

            prop_assert!(info.encrypted, "ENCRYPTED flag must be set");
        }

        /// Property: load_from_bytes_encrypted roundtrip works
        #[test]
        fn prop_bytes_encrypted_roundtrip(
            password in arb_password(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_encrypted(&model, ModelType::Custom, &path, SaveOptions::default(), &password)
                .expect("save");

            let bytes = std::fs::read(&path).expect("read");
            let loaded: Model = load_from_bytes_encrypted(&bytes, ModelType::Custom, &password)
                .expect("load from bytes");

            prop_assert_eq!(loaded.weights, data);
        }
    }
