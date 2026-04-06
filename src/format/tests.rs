//! Tests for the alimentar dataset format (.ald).

use super::*;

#[test]
fn test_magic_bytes() {
    assert_eq!(&MAGIC, b"ALDF");
}

#[test]
fn test_header_roundtrip() {
    let header = Header {
        version: (1, 2),
        dataset_type: DatasetType::ImageClassification,
        metadata_size: 1024,
        payload_size: 65536,
        uncompressed_size: 131072,
        compression: Compression::ZstdL3,
        flags: flags::SIGNED | flags::ENCRYPTED,
        schema_size: 256,
        num_rows: 50000,
    };

    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), HEADER_SIZE);

    let parsed = Header::from_bytes(&bytes).expect("parse failed");
    assert_eq!(parsed, header);
}

#[test]
fn test_header_invalid_magic() {
    let mut bytes = [0u8; HEADER_SIZE];
    bytes[0..4].copy_from_slice(b"XXXX");

    let result = Header::from_bytes(&bytes);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Invalid magic"));
}

#[test]
fn test_header_flags() {
    let mut header = Header::new(DatasetType::Tabular);
    assert!(!header.is_encrypted());
    assert!(!header.is_signed());

    header.flags |= flags::ENCRYPTED;
    assert!(header.is_encrypted());
    assert!(!header.is_signed());

    header.flags |= flags::SIGNED;
    assert!(header.is_encrypted());
    assert!(header.is_signed());
}

#[test]
fn test_dataset_type_roundtrip() {
    for dt in [
        DatasetType::Tabular,
        DatasetType::ImageClassification,
        DatasetType::QuestionAnswering,
        DatasetType::Custom,
    ] {
        let value = dt.as_u16();
        let parsed = DatasetType::from_u16(value);
        assert_eq!(parsed, Some(dt));
    }
}

#[test]
fn test_compression_roundtrip() {
    for c in [
        Compression::None,
        Compression::ZstdL3,
        Compression::ZstdL19,
        Compression::Lz4,
    ] {
        let value = c.as_u8();
        let parsed = Compression::from_u8(value);
        assert_eq!(parsed, Some(c));
    }
}

#[test]
fn test_header_size_is_32() {
    let header = Header::new(DatasetType::Tabular);
    assert_eq!(header.to_bytes().len(), 32);
}

fn create_test_batch() -> arrow::array::RecordBatch {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    let name_array = StringArray::from(vec!["Alice", "Bob", "Charlie", "Diana", "Eve"]);

    arrow::array::RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(name_array)])
        .expect("create batch")
}

#[test]
fn test_save_load_roundtrip_no_compression() {
    let batch = create_test_batch();
    let batches = vec![batch];

    let options = SaveOptions {
        compression: Compression::None,
        metadata: Some(Metadata {
            name: Some("test-dataset".to_string()),
            version: Some("1.0.0".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");

    assert_eq!(loaded.header.dataset_type, DatasetType::Tabular);
    assert_eq!(loaded.header.compression, Compression::None);
    assert_eq!(loaded.header.num_rows, 5);
    assert_eq!(loaded.metadata.name, Some("test-dataset".to_string()));
    assert_eq!(loaded.batches.len(), 1);
    assert_eq!(loaded.batches[0].num_rows(), 5);
}

#[test]
fn test_save_load_roundtrip_zstd() {
    let batch = create_test_batch();
    let batches = vec![batch];

    let options = SaveOptions {
        compression: Compression::ZstdL3,
        metadata: None,
        ..Default::default()
    };

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");

    assert_eq!(loaded.header.compression, Compression::ZstdL3);
    assert_eq!(loaded.batches.len(), 1);
    assert_eq!(loaded.batches[0].num_rows(), 5);
}

#[test]
fn test_save_load_roundtrip_lz4() {
    let batch = create_test_batch();
    let batches = vec![batch];

    let options = SaveOptions {
        compression: Compression::Lz4,
        metadata: None,
        ..Default::default()
    };

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");

    assert_eq!(loaded.header.compression, Compression::Lz4);
    assert_eq!(loaded.batches.len(), 1);
    assert_eq!(loaded.batches[0].num_rows(), 5);
}

#[test]
fn test_save_empty_batches_fails() {
    let batches: Vec<arrow::array::RecordBatch> = vec![];
    let options = SaveOptions::default();

    let mut buf = Vec::new();
    let result = save(&mut buf, &batches, DatasetType::Tabular, &options);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("empty"));
}

#[test]
fn test_checksum_mismatch_detected() {
    let batch = create_test_batch();
    let batches = vec![batch];

    let mut buf = Vec::new();
    save(
        &mut buf,
        &batches,
        DatasetType::Tabular,
        &SaveOptions::default(),
    )
    .expect("save failed");

    // Corrupt a byte in the payload
    if buf.len() > 50 {
        buf[50] ^= 0xFF;
    }

    let result = load(&mut std::io::Cursor::new(&buf));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Checksum"));
}

#[test]
fn test_multiple_batches() {
    let batch1 = create_test_batch();
    let batch2 = create_test_batch();
    let batches = vec![batch1, batch2];

    let mut buf = Vec::new();
    save(
        &mut buf,
        &batches,
        DatasetType::Tabular,
        &SaveOptions::default(),
    )
    .expect("save failed");

    let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");

    assert_eq!(loaded.header.num_rows, 10);
    assert_eq!(loaded.batches.len(), 2);
}

#[cfg(feature = "format-encryption")]
#[test]
fn test_save_load_password_encrypted() {
    let batch = create_test_batch();
    let batches = vec![batch];

    let options = SaveOptions::default().with_password("test-password-123");

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    // Verify it's marked as encrypted
    let header = Header::from_bytes(&buf[..HEADER_SIZE]).expect("header parse failed");
    assert!(header.is_encrypted());

    // Load with correct password
    let load_opts = LoadOptions::default().with_password("test-password-123");
    let loaded =
        load_with_options(&mut std::io::Cursor::new(&buf), &load_opts).expect("load failed");

    assert_eq!(loaded.batches.len(), 1);
    assert_eq!(loaded.batches[0].num_rows(), 5);
}

#[cfg(feature = "format-encryption")]
#[test]
fn test_save_load_password_wrong_password() {
    let batch = create_test_batch();
    let batches = vec![batch];

    let options = SaveOptions::default().with_password("correct-password");

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    // Load with wrong password should fail
    let load_opts = LoadOptions::default().with_password("wrong-password");
    let result = load_with_options(&mut std::io::Cursor::new(&buf), &load_opts);

    assert!(result.is_err());
}

#[cfg(feature = "format-encryption")]
#[test]
fn test_save_load_recipient_encrypted() {
    use x25519_dalek::{PublicKey, StaticSecret};

    let batch = create_test_batch();
    let batches = vec![batch];

    // Generate recipient key pair
    let mut key_bytes = [0u8; 32];
    getrandom::getrandom(&mut key_bytes).expect("rng failed");
    let recipient_secret = StaticSecret::from(key_bytes);
    let recipient_public = PublicKey::from(&recipient_secret);

    let options = SaveOptions::default().with_recipient(*recipient_public.as_bytes());

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    // Verify it's marked as encrypted
    let header = Header::from_bytes(&buf[..HEADER_SIZE]).expect("header parse failed");
    assert!(header.is_encrypted());

    // Load with correct private key
    let load_opts = LoadOptions::default().with_private_key(key_bytes);
    let loaded =
        load_with_options(&mut std::io::Cursor::new(&buf), &load_opts).expect("load failed");

    assert_eq!(loaded.batches.len(), 1);
    assert_eq!(loaded.batches[0].num_rows(), 5);
}

#[cfg(feature = "format-signing")]
#[test]
fn test_save_load_signed() {
    use crate::format::signing::SigningKeyPair;

    let batch = create_test_batch();
    let batches = vec![batch];

    let key_pair = SigningKeyPair::generate().expect("keygen");
    let public_key = key_pair.public_key_bytes();

    let options = SaveOptions::default().with_signing_key(key_pair);

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    // Verify it's marked as signed
    let header = Header::from_bytes(&buf[..HEADER_SIZE]).expect("header parse failed");
    assert!(header.is_signed());

    // Load and verify signature
    let load_opts = LoadOptions::default().with_trusted_key(public_key);
    let loaded =
        load_with_options(&mut std::io::Cursor::new(&buf), &load_opts).expect("load failed");

    assert_eq!(loaded.batches.len(), 1);
    assert_eq!(loaded.signer_public_key, Some(public_key));
}

#[cfg(feature = "format-signing")]
#[test]
fn test_save_load_signed_untrusted_signer() {
    use crate::format::signing::SigningKeyPair;

    let batch = create_test_batch();
    let batches = vec![batch];

    let key_pair = SigningKeyPair::generate().expect("keygen");
    let other_key_pair = SigningKeyPair::generate().expect("keygen2");
    let other_public_key = other_key_pair.public_key_bytes();

    let options = SaveOptions::default().with_signing_key(key_pair);

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    // Load with different trusted key should fail
    let load_opts = LoadOptions::default().with_trusted_key(other_public_key);
    let result = load_with_options(&mut std::io::Cursor::new(&buf), &load_opts);

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("not in trusted keys"));
}

#[cfg(feature = "format-encryption")]
#[test]
fn test_save_load_with_license() {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::format::license::{generate_license_id, hash_licensee, LicenseBuilder};

    let batch = create_test_batch();
    let batches = vec![batch];

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let license_id = generate_license_id().expect("license id gen failed");
    let licensee_hash = hash_licensee("test@example.com");
    let license = LicenseBuilder::new(license_id, licensee_hash)
        .expires_at(now + 3600) // Valid for 1 hour
        .build();

    let options = SaveOptions::default().with_license(license);

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    // Verify it's marked as licensed
    let header = Header::from_bytes(&buf[..HEADER_SIZE]).expect("header parse failed");
    assert!(header.is_licensed());

    // Load with license verification
    let load_opts = LoadOptions::default().verify_license();
    let loaded =
        load_with_options(&mut std::io::Cursor::new(&buf), &load_opts).expect("load failed");

    assert_eq!(loaded.batches.len(), 1);
    assert!(loaded.license.is_some());
}

#[cfg(feature = "format-encryption")]
#[test]
fn test_save_load_with_expired_license() {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::format::license::{generate_license_id, hash_licensee, LicenseBuilder};

    let batch = create_test_batch();
    let batches = vec![batch];

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let license_id = generate_license_id().expect("license id gen failed");
    let licensee_hash = hash_licensee("test@example.com");
    let license = LicenseBuilder::new(license_id, licensee_hash)
        .expires_at(now - 3600) // Expired 1 hour ago
        .build();

    let options = SaveOptions::default().with_license(license);

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    // Load with license verification should fail
    let load_opts = LoadOptions::default().verify_license();
    let result = load_with_options(&mut std::io::Cursor::new(&buf), &load_opts);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("License expired"));
}

#[cfg(all(feature = "format-encryption", feature = "format-signing"))]
#[test]
fn test_save_load_encrypted_and_signed() {
    use crate::format::signing::SigningKeyPair;

    let batch = create_test_batch();
    let batches = vec![batch];

    let key_pair = SigningKeyPair::generate().expect("keygen");
    let public_key = key_pair.public_key_bytes();

    let options = SaveOptions::default()
        .with_password("secure-password")
        .with_signing_key(key_pair);

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    // Verify flags
    let header = Header::from_bytes(&buf[..HEADER_SIZE]).expect("header parse failed");
    assert!(header.is_encrypted());
    assert!(header.is_signed());

    // Load with correct credentials
    let load_opts = LoadOptions::default()
        .with_password("secure-password")
        .with_trusted_key(public_key);
    let loaded =
        load_with_options(&mut std::io::Cursor::new(&buf), &load_opts).expect("load failed");

    assert_eq!(loaded.batches.len(), 1);
    assert_eq!(loaded.signer_public_key, Some(public_key));
}

#[cfg(all(feature = "format-encryption", feature = "format-signing"))]
#[test]
fn test_save_load_full_security_suite() {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::format::{
        license::{generate_license_id, hash_licensee, LicenseBuilder},
        signing::SigningKeyPair,
    };

    let batch = create_test_batch();
    let batches = vec![batch];

    let key_pair = SigningKeyPair::generate().expect("keygen");
    let public_key = key_pair.public_key_bytes();

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let license_id = generate_license_id().expect("license id gen failed");
    let licensee_hash = hash_licensee("enterprise@company.com");
    let license = LicenseBuilder::new(license_id, licensee_hash)
        .expires_at(now + 86400)
        .seat_limit(100)
        .build();

    let options = SaveOptions::default()
        .with_password("enterprise-secret")
        .with_signing_key(key_pair)
        .with_license(license)
        .with_metadata(Metadata {
            name: Some("enterprise-dataset".to_string()),
            version: Some("2.0.0".to_string()),
            ..Default::default()
        });

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    // Verify all flags
    let header = Header::from_bytes(&buf[..HEADER_SIZE]).expect("header parse failed");
    assert!(header.is_encrypted());
    assert!(header.is_signed());
    assert!(header.is_licensed());

    // Load with full verification
    let load_opts = LoadOptions::default()
        .with_password("enterprise-secret")
        .with_trusted_key(public_key)
        .verify_license();
    let loaded =
        load_with_options(&mut std::io::Cursor::new(&buf), &load_opts).expect("load failed");

    assert_eq!(loaded.batches.len(), 1);
    assert_eq!(loaded.signer_public_key, Some(public_key));
    assert!(loaded.license.is_some());
    assert_eq!(loaded.metadata.name, Some("enterprise-dataset".to_string()));
}

#[test]
fn test_dataset_type_all_variants() {
    // Test all DatasetType variants
    let types = [
        (DatasetType::Tabular, 0x0001),
        (DatasetType::TimeSeries, 0x0002),
        (DatasetType::Graph, 0x0003),
        (DatasetType::Spatial, 0x0004),
        (DatasetType::TextCorpus, 0x0010),
        (DatasetType::TextClassification, 0x0011),
        (DatasetType::TextPairs, 0x0012),
        (DatasetType::SequenceLabeling, 0x0013),
        (DatasetType::QuestionAnswering, 0x0014),
        (DatasetType::Summarization, 0x0015),
        (DatasetType::Translation, 0x0016),
        (DatasetType::ImageClassification, 0x0020),
        (DatasetType::ObjectDetection, 0x0021),
        (DatasetType::Segmentation, 0x0022),
        (DatasetType::ImagePairs, 0x0023),
        (DatasetType::Video, 0x0024),
        (DatasetType::AudioClassification, 0x0030),
        (DatasetType::SpeechRecognition, 0x0031),
        (DatasetType::SpeakerIdentification, 0x0032),
        (DatasetType::UserItemRatings, 0x0040),
        (DatasetType::ImplicitFeedback, 0x0041),
        (DatasetType::SequentialRecs, 0x0042),
        (DatasetType::ImageText, 0x0050),
        (DatasetType::AudioText, 0x0051),
        (DatasetType::VideoText, 0x0052),
        (DatasetType::Custom, 0x00FF),
    ];

    for (dt, expected_value) in types {
        assert_eq!(dt.as_u16(), expected_value);
        assert_eq!(DatasetType::from_u16(expected_value), Some(dt));
    }
}

#[test]
fn test_dataset_type_invalid_value() {
    assert_eq!(DatasetType::from_u16(0x9999), None);
}

#[test]
fn test_compression_invalid_value() {
    assert_eq!(Compression::from_u8(0x99), None);
}

#[test]
fn test_header_is_streaming() {
    let mut header = Header::new(DatasetType::Tabular);
    assert!(!header.is_streaming());

    header.flags |= flags::STREAMING;
    assert!(header.is_streaming());
}

#[test]
fn test_header_is_licensed() {
    let mut header = Header::new(DatasetType::Tabular);
    assert!(!header.is_licensed());

    header.flags |= flags::LICENSED;
    assert!(header.is_licensed());
}

#[test]
#[cfg(feature = "provenance")]
fn test_sha256_hex_known_value() {
    // Test against known SHA-256 hash
    let hash = super::sha256_hex(b"Hello, World!");
    // SHA-256 of "Hello, World!" is known
    assert_eq!(
        hash,
        "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
    );
}

#[test]
#[cfg(feature = "provenance")]
fn test_sha256_hex_length() {
    let hash = super::sha256_hex(b"test data");
    assert_eq!(hash.len(), 64); // 256 bits = 32 bytes = 64 hex chars
}

#[test]
#[cfg(feature = "provenance")]
fn test_sha256_hex_empty() {
    let hash = super::sha256_hex(b"");
    // SHA-256 of empty string is known
    assert_eq!(
        hash,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
#[cfg(feature = "provenance")]
fn test_sha256_hex_different_inputs() {
    let hash1 = super::sha256_hex(b"input1");
    let hash2 = super::sha256_hex(b"input2");
    assert_ne!(hash1, hash2);
}

#[test]
fn test_metadata_sha256_field() {
    let mut metadata = Metadata::default();
    assert!(metadata.sha256.is_none());

    metadata.sha256 = Some("abc123".to_string());
    assert_eq!(metadata.sha256, Some("abc123".to_string()));
}

// ========== Additional edge case tests ==========

#[test]
fn test_header_from_bytes_too_short() {
    let bytes = [0u8; 10]; // Too short
    let result = Header::from_bytes(&bytes);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too short"));
}

#[test]
fn test_header_unsupported_version() {
    let mut header = Header::new(DatasetType::Tabular);
    header.version = (99, 0); // Future version
    let bytes = header.to_bytes();

    // Modify magic to be valid
    let mut valid_bytes = bytes;
    valid_bytes[0..4].copy_from_slice(&MAGIC);
    // Set unsupported version in bytes
    valid_bytes[4] = 99;
    valid_bytes[5] = 0;

    let result = Header::from_bytes(&valid_bytes);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Unsupported version"));
}

#[test]
fn test_header_unknown_dataset_type() {
    let mut bytes = [0u8; HEADER_SIZE];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 1; // version major
    bytes[5] = 0; // version minor
    bytes[6] = 0xFF; // unknown dataset type
    bytes[7] = 0xFF;

    let result = Header::from_bytes(&bytes);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Unknown dataset type"));
}

#[test]
fn test_header_unknown_compression() {
    let mut bytes = [0u8; HEADER_SIZE];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 1; // version major
    bytes[5] = 0; // version minor
    bytes[6] = 0x01; // tabular dataset type
    bytes[7] = 0x00;
    bytes[20] = 0xFF; // unknown compression

    let result = Header::from_bytes(&bytes);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Unknown compression"));
}

#[test]
fn test_header_is_trueno_native() {
    let mut header = Header::new(DatasetType::Tabular);
    assert!(!header.is_trueno_native());

    header.flags |= flags::TRUENO_NATIVE;
    assert!(header.is_trueno_native());
}

#[test]
fn test_save_options_with_compression() {
    let options = SaveOptions::default().with_compression(Compression::ZstdL19);
    assert_eq!(options.compression, Compression::ZstdL19);
}

#[test]
fn test_save_options_with_metadata() {
    let meta = Metadata {
        name: Some("test".to_string()),
        ..Default::default()
    };
    let options = SaveOptions::default().with_metadata(meta.clone());
    assert!(options.metadata.is_some());
    assert_eq!(options.metadata.unwrap().name, Some("test".to_string()));
}

#[test]
fn test_load_options_verify_license() {
    let options = LoadOptions::default().verify_license();
    assert!(options.verify_license);
}

#[test]
fn test_metadata_default() {
    let metadata = Metadata::default();
    assert!(metadata.name.is_none());
    assert!(metadata.version.is_none());
    assert!(metadata.license.is_none());
    assert!(metadata.tags.is_empty());
    assert!(metadata.description.is_none());
    assert!(metadata.citation.is_none());
    assert!(metadata.created_at.is_none());
}

#[test]
fn test_metadata_clone() {
    let metadata = Metadata {
        name: Some("test".to_string()),
        version: Some("1.0.0".to_string()),
        license: Some("MIT".to_string()),
        tags: vec!["tag1".to_string(), "tag2".to_string()],
        description: Some("desc".to_string()),
        citation: Some("cite".to_string()),
        created_at: Some("2024-01-01".to_string()),
        sha256: Some("abc123".to_string()),
    };

    let cloned = metadata.clone();
    assert_eq!(cloned.name, metadata.name);
    assert_eq!(cloned.version, metadata.version);
    assert_eq!(cloned.tags, metadata.tags);
}

#[test]
fn test_metadata_debug() {
    let metadata = Metadata::default();
    let debug = format!("{:?}", metadata);
    assert!(debug.contains("Metadata"));
}

#[test]
fn test_save_options_debug() {
    let options = SaveOptions::default();
    let debug = format!("{:?}", options);
    assert!(debug.contains("SaveOptions"));
}

#[test]
fn test_load_options_debug() {
    let options = LoadOptions::default();
    let debug = format!("{:?}", options);
    assert!(debug.contains("LoadOptions"));
}

#[test]
fn test_loaded_dataset_debug() {
    let batch = create_test_batch();
    let batches = vec![batch];

    let mut buf = Vec::new();
    save(
        &mut buf,
        &batches,
        DatasetType::Tabular,
        &SaveOptions::default(),
    )
    .expect("save failed");

    let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");
    let debug = format!("{:?}", loaded);
    assert!(debug.contains("LoadedDataset"));
}

#[test]
fn test_header_new_defaults() {
    let header = Header::new(DatasetType::ImageClassification);
    assert_eq!(header.version, (FORMAT_VERSION_MAJOR, FORMAT_VERSION_MINOR));
    assert_eq!(header.dataset_type, DatasetType::ImageClassification);
    assert_eq!(header.metadata_size, 0);
    assert_eq!(header.payload_size, 0);
    assert_eq!(header.uncompressed_size, 0);
    assert_eq!(header.compression, Compression::default());
    assert_eq!(header.flags, 0);
    assert_eq!(header.schema_size, 0);
    assert_eq!(header.num_rows, 0);
}

#[test]
fn test_compression_default() {
    let compression = Compression::default();
    assert_eq!(compression, Compression::ZstdL3);
}

#[test]
fn test_save_load_roundtrip_zstd_l19() {
    let batch = create_test_batch();
    let batches = vec![batch];

    let options = SaveOptions {
        compression: Compression::ZstdL19,
        metadata: None,
        ..Default::default()
    };

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");

    assert_eq!(loaded.header.compression, Compression::ZstdL19);
    assert_eq!(loaded.batches.len(), 1);
}

#[test]
fn test_load_file_too_small() {
    let buf = [0u8; 10]; // Too small
    let result = load(&mut std::io::Cursor::new(&buf));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too small"));
}

#[test]
fn test_header_clone() {
    let header = Header::new(DatasetType::Tabular);
    let cloned = header.clone();
    assert_eq!(cloned, header);
}

#[test]
fn test_header_debug() {
    let header = Header::new(DatasetType::Tabular);
    let debug = format!("{:?}", header);
    assert!(debug.contains("Header"));
}

#[test]
fn test_dataset_type_debug() {
    let dt = DatasetType::Tabular;
    let debug = format!("{:?}", dt);
    assert!(debug.contains("Tabular"));
}

#[test]
fn test_dataset_type_clone() {
    let dt = DatasetType::ImageClassification;
    let cloned = dt;
    assert_eq!(dt, cloned);
}

#[test]
fn test_compression_debug() {
    let c = Compression::ZstdL3;
    let debug = format!("{:?}", c);
    assert!(debug.contains("ZstdL3"));
}

#[test]
fn test_compression_clone() {
    let c = Compression::Lz4;
    let cloned = c;
    assert_eq!(c, cloned);
}

#[test]
fn test_save_options_clone() {
    let options = SaveOptions::default().with_compression(Compression::Lz4);
    let cloned = options.clone();
    assert_eq!(cloned.compression, Compression::Lz4);
}

#[test]
fn test_load_options_clone() {
    let options = LoadOptions::default().verify_license();
    let cloned = options.clone();
    assert!(cloned.verify_license);
}

#[test]
fn test_flags_constants() {
    assert_eq!(flags::ENCRYPTED, 0b0000_0001);
    assert_eq!(flags::SIGNED, 0b0000_0010);
    assert_eq!(flags::STREAMING, 0b0000_0100);
    assert_eq!(flags::LICENSED, 0b0000_1000);
    assert_eq!(flags::TRUENO_NATIVE, 0b0001_0000);
}

#[test]
fn test_header_all_flags_combined() {
    let mut header = Header::new(DatasetType::Tabular);
    header.flags = flags::ENCRYPTED
        | flags::SIGNED
        | flags::STREAMING
        | flags::LICENSED
        | flags::TRUENO_NATIVE;

    assert!(header.is_encrypted());
    assert!(header.is_signed());
    assert!(header.is_streaming());
    assert!(header.is_licensed());
    assert!(header.is_trueno_native());

    // Roundtrip to bytes and back
    let bytes = header.to_bytes();
    let parsed = Header::from_bytes(&bytes).expect("parse");
    assert_eq!(parsed.flags, header.flags);
}

#[test]
fn test_save_to_file_and_load_from_file() {
    let batch = create_test_batch();
    let batches = vec![batch];

    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("temp dir"));
    let path = temp_dir.path().join("test.ald");

    save_to_file(
        &path,
        &batches,
        DatasetType::Tabular,
        &SaveOptions::default(),
    )
    .expect("save failed");

    let loaded = load_from_file(&path).expect("load failed");
    assert_eq!(loaded.batches.len(), 1);
    assert_eq!(loaded.batches[0].num_rows(), 5);
}

#[test]
fn test_save_to_file_invalid_path() {
    let batch = create_test_batch();
    let batches = vec![batch];

    let result = save_to_file(
        "/nonexistent/path/test.ald",
        &batches,
        DatasetType::Tabular,
        &SaveOptions::default(),
    );
    assert!(result.is_err());
}

#[test]
fn test_load_from_file_not_found() {
    let result = load_from_file("/nonexistent/path/test.ald");
    assert!(result.is_err());
}

#[test]
fn test_load_from_file_with_options_not_found() {
    let result = load_from_file_with_options("/nonexistent/path/test.ald", &LoadOptions::default());
    assert!(result.is_err());
}

// === Additional edge case tests ===

#[test]
fn test_load_payload_extends_beyond_data() {
    // Create a valid header but with incorrect payload_size that exceeds data
    let batch = create_test_batch();
    let batches = vec![batch];

    let mut buf = Vec::new();
    save(
        &mut buf,
        &batches,
        DatasetType::Tabular,
        &SaveOptions::default(),
    )
    .expect("save failed");

    // Corrupt payload_size in header to be too large
    // payload_size is at offset 12-15
    buf[12] = 0xFF;
    buf[13] = 0xFF;
    buf[14] = 0xFF;
    buf[15] = 0xFF; // Set to max u32

    let result = load(&mut std::io::Cursor::new(&buf));
    // This should fail due to checksum mismatch (since we corrupted the header)
    assert!(result.is_err());
}

#[test]
fn test_all_dataset_types_save_load() {
    let all_types = [
        DatasetType::Tabular,
        DatasetType::TimeSeries,
        DatasetType::Graph,
        DatasetType::Spatial,
        DatasetType::TextCorpus,
        DatasetType::TextClassification,
        DatasetType::TextPairs,
        DatasetType::SequenceLabeling,
        DatasetType::QuestionAnswering,
        DatasetType::Summarization,
        DatasetType::Translation,
        DatasetType::ImageClassification,
        DatasetType::ObjectDetection,
        DatasetType::Segmentation,
        DatasetType::ImagePairs,
        DatasetType::Video,
        DatasetType::AudioClassification,
        DatasetType::SpeechRecognition,
        DatasetType::SpeakerIdentification,
        DatasetType::UserItemRatings,
        DatasetType::ImplicitFeedback,
        DatasetType::SequentialRecs,
        DatasetType::ImageText,
        DatasetType::AudioText,
        DatasetType::VideoText,
        DatasetType::Custom,
    ];

    let batch = create_test_batch();
    let batches = vec![batch];

    for dt in all_types {
        let mut buf = Vec::new();
        save(&mut buf, &batches, dt, &SaveOptions::default()).expect("save failed");

        let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");
        assert_eq!(loaded.header.dataset_type, dt);
    }
}

#[test]
fn test_header_eq() {
    let header1 = Header::new(DatasetType::Tabular);
    let header2 = Header::new(DatasetType::Tabular);
    let header3 = Header::new(DatasetType::Graph);

    assert_eq!(header1, header2);
    assert_ne!(header1, header3);
}

#[test]
fn test_dataset_type_eq() {
    let dt1 = DatasetType::Tabular;
    let dt2 = DatasetType::Tabular;
    let dt3 = DatasetType::Graph;

    assert_eq!(dt1, dt2);
    assert_ne!(dt1, dt3);
}

#[test]
fn test_compression_eq() {
    let c1 = Compression::Lz4;
    let c2 = Compression::Lz4;
    let c3 = Compression::ZstdL3;

    assert_eq!(c1, c2);
    assert_ne!(c1, c3);
}

#[test]
fn test_metadata_serialization() {
    let metadata = Metadata {
        name: Some("test".to_string()),
        version: Some("1.0.0".to_string()),
        license: Some("MIT".to_string()),
        tags: vec!["ml".to_string(), "nlp".to_string()],
        description: Some("A test dataset".to_string()),
        citation: Some("@article{...}".to_string()),
        created_at: Some("2024-01-01T00:00:00Z".to_string()),
        sha256: Some("abc123def456".to_string()),
    };

    // Serialize to MessagePack
    let bytes = rmp_serde::to_vec(&metadata).expect("serialize");

    // Deserialize back
    let parsed: Metadata = rmp_serde::from_slice(&bytes).expect("deserialize");

    assert_eq!(parsed.name, metadata.name);
    assert_eq!(parsed.version, metadata.version);
    assert_eq!(parsed.license, metadata.license);
    assert_eq!(parsed.tags, metadata.tags);
    assert_eq!(parsed.description, metadata.description);
    assert_eq!(parsed.sha256, metadata.sha256);
}

#[test]
fn test_header_byte_layout() {
    let header = Header {
        version: (1, 2),
        dataset_type: DatasetType::Tabular,
        metadata_size: 0x12345678,
        payload_size: 0xABCDEF01,
        uncompressed_size: 0x11223344,
        compression: Compression::ZstdL3,
        flags: 0x55,
        schema_size: 0x6677,
        num_rows: 0x123456789ABCDEF0,
    };

    let bytes = header.to_bytes();

    // Verify magic
    assert_eq!(&bytes[0..4], &MAGIC);

    // Verify version
    assert_eq!(bytes[4], 1);
    assert_eq!(bytes[5], 2);

    // Verify dataset type (little-endian)
    assert_eq!(bytes[6], 0x01); // Tabular = 0x0001
    assert_eq!(bytes[7], 0x00);

    // Verify flags
    assert_eq!(bytes[21], 0x55);
}

#[test]
fn test_save_options_default_fields() {
    let options = SaveOptions::default();

    assert_eq!(options.compression, Compression::ZstdL3);
    assert!(options.metadata.is_none());
    assert!(options.license.is_none());
}

#[test]
fn test_load_options_default_fields() {
    let options = LoadOptions::default();

    assert!(!options.verify_license);
}

#[test]
fn test_header_full_roundtrip_all_values() {
    // Test with all fields set to non-zero values
    let header = Header {
        version: (1, 2),
        dataset_type: DatasetType::VideoText,
        metadata_size: u32::MAX - 1,
        payload_size: u32::MAX - 2,
        uncompressed_size: u32::MAX - 3,
        compression: Compression::Lz4,
        flags: 0xFF,
        schema_size: u16::MAX - 1,
        num_rows: u64::MAX - 1,
    };

    let bytes = header.to_bytes();
    let parsed = Header::from_bytes(&bytes).expect("parse");

    assert_eq!(parsed.version, header.version);
    assert_eq!(parsed.dataset_type, header.dataset_type);
    assert_eq!(parsed.metadata_size, header.metadata_size);
    assert_eq!(parsed.payload_size, header.payload_size);
    assert_eq!(parsed.uncompressed_size, header.uncompressed_size);
    assert_eq!(parsed.compression, header.compression);
    assert_eq!(parsed.flags, header.flags);
    assert_eq!(parsed.schema_size, header.schema_size);
    assert_eq!(parsed.num_rows, header.num_rows);
}

#[test]
fn test_crc32_function() {
    // Test CRC32 with known values
    let data = b"hello world";
    let crc = crc32(data);
    // CRC32 should be deterministic
    assert_eq!(crc, crc32(data));

    // Different data should produce different CRC
    let other_data = b"hello worlD";
    assert_ne!(crc, crc32(other_data));
}

#[test]
fn test_metadata_with_all_fields_save_load() {
    let batch = create_test_batch();
    let batches = vec![batch];

    let options = SaveOptions::default().with_metadata(Metadata {
        name: Some("full-metadata-test".to_string()),
        version: Some("2.0.0".to_string()),
        license: Some("Apache-2.0".to_string()),
        tags: vec!["tag1".to_string(), "tag2".to_string(), "tag3".to_string()],
        description: Some("A comprehensive test with all metadata fields".to_string()),
        citation: Some("@inproceedings{test2024}".to_string()),
        created_at: Some("2024-06-15T12:00:00Z".to_string()),
        sha256: Some("0123456789abcdef".repeat(4)),
    });

    let mut buf = Vec::new();
    save(&mut buf, &batches, DatasetType::Tabular, &options).expect("save failed");

    let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");

    assert_eq!(loaded.metadata.name, Some("full-metadata-test".to_string()));
    assert_eq!(loaded.metadata.version, Some("2.0.0".to_string()));
    assert_eq!(loaded.metadata.license, Some("Apache-2.0".to_string()));
    assert_eq!(loaded.metadata.tags.len(), 3);
    assert!(loaded.metadata.description.is_some());
    assert!(loaded.metadata.citation.is_some());
    assert!(loaded.metadata.created_at.is_some());
}

#[test]
fn test_save_large_dataset() {
    // Create multiple batches to simulate a larger dataset
    let batches: Vec<_> = (0..10).map(|_| create_test_batch()).collect();

    let mut buf = Vec::new();
    save(
        &mut buf,
        &batches,
        DatasetType::Tabular,
        &SaveOptions::default(),
    )
    .expect("save failed");

    let loaded = load(&mut std::io::Cursor::new(&buf)).expect("load failed");

    // Total rows should be 10 batches * 5 rows = 50
    assert_eq!(loaded.header.num_rows, 50);
}

#[test]
fn test_dataset_type_serde() {
    use serde_json;

    let dt = DatasetType::QuestionAnswering;

    // Serialize to JSON
    let json = serde_json::to_string(&dt).expect("serialize");

    // Deserialize back
    let parsed: DatasetType = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed, dt);
}

#[test]
fn test_compression_serde() {
    use serde_json;

    let c = Compression::ZstdL19;

    // Serialize to JSON
    let json = serde_json::to_string(&c).expect("serialize");

    // Deserialize back
    let parsed: Compression = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed, c);
}
