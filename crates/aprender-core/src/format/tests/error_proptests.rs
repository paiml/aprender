//! Error handling property tests.

use super::super::*;
pub(crate) use proptest::prelude::*;

pub(super) fn arb_non_magic_bytes() -> impl Strategy<Value = [u8; 4]> {
    any::<[u8; 4]>().prop_filter("not APR magic", |b| b != b"APR\x00")
}

pub(super) fn arb_invalid_model_type() -> impl Strategy<Value = u16> {
    // Valid model types are 0-16, so anything >= 17 is invalid
    17u16..=u16::MAX
}

pub(super) fn arb_invalid_compression() -> impl Strategy<Value = u8> {
    // Valid compression values are 0-3, so anything >= 4 is invalid
    4u8..=u8::MAX
}

proptest! {
    /// Property: Invalid magic bytes are rejected
    #[test]
    fn prop_invalid_magic_rejected(bad_magic in arb_non_magic_bytes()) {
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("bad_magic.apr");

        // Create a file with invalid magic
        let mut content = vec![0u8; 64];
        content[0..4].copy_from_slice(&bad_magic);
        std::fs::write(&path, &content).expect("write");

        let result = inspect(&path);
        prop_assert!(result.is_err(), "Invalid magic should be rejected");
    }

    /// Property: Truncated header is rejected
    #[test]
    fn prop_truncated_header_rejected(len in 0usize..32) {
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("truncated.apr");

        // Create a file shorter than header size
        let content = vec![0u8; len];
        std::fs::write(&path, &content).expect("write");

        let result = inspect(&path);
        prop_assert!(result.is_err(), "Truncated header should be rejected");
    }

    /// Property: Invalid model type in header is rejected
    #[test]
    fn prop_invalid_model_type_rejected(bad_type in arb_invalid_model_type()) {
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("bad_type.apr");

        // Create header with valid magic but invalid model type
        let mut content = vec![0u8; 64];
        content[0..4].copy_from_slice(b"APR\x00");
        content[4..6].copy_from_slice(&bad_type.to_le_bytes());
        std::fs::write(&path, &content).expect("write");

        let result = inspect(&path);
        prop_assert!(result.is_err(), "Invalid model type should be rejected");
    }

    /// Property: Invalid compression byte is rejected
    #[test]
    fn prop_invalid_compression_rejected(bad_comp in arb_invalid_compression()) {
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("bad_comp.apr");

        // Create header with valid magic, model type, but invalid compression
        let mut content = vec![0u8; 64];
        content[0..4].copy_from_slice(b"APR\x00");
        content[4..6].copy_from_slice(&0u16.to_le_bytes()); // Valid model type
        content[20] = bad_comp;
        std::fs::write(&path, &content).expect("write");

        let result = inspect(&path);
        prop_assert!(result.is_err(), "Invalid compression should be rejected");
    }

    /// Property: CRC mismatch is detected on load
    #[test]
    fn prop_crc_mismatch_detected(data in proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 1..50)) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data };
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("crc_test.apr");

        save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

        // Corrupt the payload (after header + metadata)
        let mut content = std::fs::read(&path).expect("read");
        if content.len() > 100 {
            content[80] ^= 0xFF; // Flip bits in payload area
            std::fs::write(&path, &content).expect("write corrupted");

            let result: Result<Model> = load(&path, ModelType::Custom);
            // Either CRC check fails or deserialization fails - both are correct
            prop_assert!(result.is_err(), "Corrupted file should fail to load");
        }
    }

    /// Property: Empty file is rejected
    #[test]
    fn prop_empty_file_rejected(_dummy in 0u8..1) {
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("empty.apr");

        std::fs::write(&path, []).expect("write empty");

        let result = inspect(&path);
        prop_assert!(result.is_err(), "Empty file should be rejected");
    }

    /// Property: Random bytes are rejected
    #[test]
    fn prop_random_bytes_rejected(random in proptest::collection::vec(any::<u8>(), 32..256)) {
        use tempfile::tempdir;

        // Skip if random bytes happen to start with APR magic (very unlikely)
        if random.len() >= 4 && &random[0..4] == b"APR\x00" {
            return Ok(());
        }

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("random.apr");

        std::fs::write(&path, &random).expect("write random");

        let result = inspect(&path);
        prop_assert!(result.is_err(), "Random bytes should be rejected");
    }

    /// Property: Format version matches constant
    #[test]
    fn prop_format_version_correct(data in proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 1..20)) {
        use tempfile::tempdir;

        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct Model { weights: Vec<f32> }

        let model = Model { weights: data.clone() };
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("versioned.apr");

        save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

        // Verify the version bytes match FORMAT_VERSION (1, 0)
        let content = std::fs::read(&path).expect("read");
        prop_assert_eq!(content[4], FORMAT_VERSION.0, "Major version mismatch");
        prop_assert_eq!(content[5], FORMAT_VERSION.1, "Minor version mismatch");

        // Verify we can load it back
        let loaded: Model = load(&path, ModelType::Custom).expect("load");
        prop_assert_eq!(data.len(), loaded.weights.len());
    }
}
